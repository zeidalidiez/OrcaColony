from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath

from .artifacts import PackedDataset
from .multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    WorkReceipt,
    _atomic_bytes,
    _atomic_json,
    _aggregate_resource_observations,
    _campaign_payload,
    _checkpoint_numerical_profile,
    _exact_json_equal,
    _reject_duplicate_json_keys,
    _revision,
    _safe_artifact_snapshot,
    _validated_numerical_profile,
    create_http_server,
)
from .participants import ParticipantRegistry, load_participants
from .peft import (
    BURN_NDARRAY_F32_PROFILE,
    BURN_WEBGPU_F32_PROFILE,
    EXACT_CPU_FP32_PROFILE,
    INT8_FROZEN_LINEAR_PROFILE,
    LoadedLoRAManifest,
    evaluate_lora_checkpoint,
    load_lora_checkpoint,
    load_lora_manifest,
    run_lora_training,
)
from .reference import (
    CampaignConfig,
    _load_checkpoint,
    evaluate_checkpoint,
    load_campaign,
    run_training,
)


_PUBLIC_DATASET_SOURCE_FIELDS = (
    "dataset",
    "dataset_card",
    "license",
    "license_url",
    "revision",
    "selection",
)
_CAMPAIGN_STATE_FIELDS = frozenset(
    {
        "format",
        "campaign_id",
        "campaign_revision",
        "participants_revision",
        "dataset_revision",
        "worker_count",
        "lease_seconds",
        "target_steps",
        "completed_steps",
        "state",
        "current_round",
        "checkpoints",
        "last_checkpoint_metrics",
        "evaluations",
        "last_evaluation",
        "baseline_checkpoint",
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "numerical_profile",
        "publish_base_layer_bundle",
    }
)
_CAMPAIGN_CHECKPOINT_FIELDS = frozenset(
    {"step", "path", "round", "numerical_profile"}
)
_DATASET_ARTIFACT_NAMES = (
    "manifest.json",
    "tokenizer.json",
    "train.safetensors",
    "validation.safetensors",
    "DATASET-NOTICE.md",
)


def _confined_campaign_directory(
    state_dir: Path,
    relative_name: object,
    label: str,
) -> Path:
    if not isinstance(relative_name, str):
        raise ValueError(f"{label} path is invalid")
    relative = Path(relative_name)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_name
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} path is invalid")
    root = state_dir.resolve(strict=True)
    candidate = state_dir
    for part in relative.parts:
        candidate = candidate / part
        observation = os.stat(candidate, follow_symlinks=False)
        if (
            not stat.S_ISDIR(observation.st_mode)
            or candidate.is_symlink()
            or bool(
                getattr(observation, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
        ):
            raise ValueError(f"{label} path is not a managed directory")
    if not candidate.resolve(strict=True).is_relative_to(root):
        raise ValueError(f"{label} path escapes the campaign root")
    return candidate


def _campaign_directory_observation(path: Path) -> tuple[int, ...]:
    metadata = os.stat(path, follow_symlinks=False)
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


@contextmanager
def _pinned_campaign_directory(
    root: Path,
    relative: str,
    label: str,
) -> Iterator[Path]:
    candidate = _confined_campaign_directory(root, relative, label)
    relative_path = Path(relative)
    directories = [root]
    current = root
    for component in relative_path.parts:
        current = current / component
        directories.append(current)
    observations = {
        directory: _campaign_directory_observation(directory)
        for directory in directories
    }
    resolved = {directory: directory.resolve(strict=True) for directory in directories}
    try:
        with ExitStack() as leases:
            for directory in directories:
                leases.enter_context(os.scandir(directory))
            yield candidate
            for directory in directories:
                if (
                    _campaign_directory_observation(directory)
                    != observations[directory]
                    or directory.resolve(strict=True) != resolved[directory]
                ):
                    raise ValueError(f"{label} changed during validation")
    except OSError as exc:
        raise ValueError(f"{label} changed during validation") from exc


def _campaign_lock_payload(
    state: Mapping[str, object],
    lora: LoadedLoRAManifest | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "orcacolony_campaign_run_lock_v1",
        "campaign_id": state["campaign_id"],
        "campaign_revision": state["campaign_revision"],
        "participants_revision": state["participants_revision"],
        "dataset_revision": state.get("dataset_revision", "synthetic-fixture-v1"),
        "worker_count": state["worker_count"],
        "target_steps": state["target_steps"],
        "numerical_profile": state["numerical_profile"],
        "assignment_protocol_revision": 1,
        "result_protocol_revision": 2,
    }
    if lora is not None:
        payload.update(
            {
                "training_method": "frozen-base-lora",
                "lora_manifest_sha256": lora.manifest_sha256,
                "base_model_sha256": lora.config.base_model_sha256,
                "result_protocol_revision": 3,
            }
        )
    if state["publish_base_layer_bundle"]:
        payload["publish_base_layer_bundle"] = True
    return payload


class CampaignCoordinator:
    def __init__(
        self,
        campaign: CampaignConfig,
        state_dir: Path,
        participants: ParticipantRegistry,
        state: dict[str, object],
        current: GlobalStepCoordinator,
        dataset: PackedDataset | None = None,
        lora: LoadedLoRAManifest | None = None,
        rounds: Mapping[str, GlobalStepCoordinator] | None = None,
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.participants = participants
        self._state = state
        self._current = current
        self.dataset = dataset
        self.lora = lora
        self._rounds = dict(
            rounds
            if rounds is not None
            else {str(state["current_round"]): current}
        )
        self._checkpoint_snapshots: dict[int, dict[str, bytes]] = {}
        self._dataset_snapshots = (
            {
                name: dataset.artifact_bytes(name)
                for name in _DATASET_ARTIFACT_NAMES
            }
            if dataset is not None
            else {}
        )
        self._prevalidated_next_round: tuple[str, GlobalStepCoordinator] | None = None
        self._lock = threading.RLock()
        self.rounds_dir = state_dir / "rounds"
        self.checkpoints_dir = state_dir / "checkpoints"

    @classmethod
    def create(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        participants: ParticipantRegistry,
        worker_count: int,
        target_steps: int,
        lease_seconds: int = 120,
        dataset: PackedDataset | None = None,
        lora: LoadedLoRAManifest | None = None,
        publish_base_layer_bundle: bool = False,
        numerical_profile: str = EXACT_CPU_FP32_PROFILE,
    ) -> CampaignCoordinator:
        if target_steps < 1:
            raise ValueError("campaign target steps must be positive")
        if participants.campaign_id != campaign.campaign["id"]:
            raise ValueError("participant campaign does not match configuration")
        if lora is not None and lora.campaign != campaign:
            raise ValueError("LoRA manifest campaign does not match campaign run")
        if publish_base_layer_bundle and lora is None:
            raise ValueError("base layer bundle publication requires frozen-base LoRA")
        profile = _validated_numerical_profile(numerical_profile)
        if lora is None and profile == INT8_FROZEN_LINEAR_PROFILE:
            raise ValueError("int8 numerical profile requires frozen-base LoRA")
        checkpoint_profile = _checkpoint_numerical_profile(profile)
        state_dir = Path(state_dir)
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"campaign state directory is not empty: {state_dir}")
        rounds_dir = state_dir / "rounds"
        checkpoints_dir = state_dir / "checkpoints"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        current_relative = Path("rounds") / "round-00000000"
        current = GlobalStepCoordinator.create(
            campaign,
            state_dir / current_relative,
            worker_count=worker_count,
            participants=participants,
            lease_seconds=lease_seconds,
            dataset=dataset,
            lora=lora,
            publish_base_layer_bundle=publish_base_layer_bundle,
            numerical_profile=profile,
        )
        state: dict[str, object] = {
            "format": "orcacolony_campaign_state_v1",
            "campaign_id": campaign.campaign["id"],
            "campaign_revision": _revision(_campaign_payload(campaign)),
            "participants_revision": participants.revision,
            "dataset_revision": (
                dataset.revision if dataset is not None else "synthetic-fixture-v1"
            ),
            "worker_count": worker_count,
            "lease_seconds": lease_seconds,
            "target_steps": target_steps,
            "completed_steps": 0,
            "state": "campaign_running",
            "current_round": current_relative.as_posix(),
            "checkpoints": [],
            "last_checkpoint_metrics": None,
            "evaluations": [],
            "last_evaluation": None,
            "baseline_checkpoint": None,
            "training_method": "frozen-base-lora" if lora is not None else "dense",
            "lora_manifest_sha256": lora.manifest_sha256 if lora is not None else None,
            "base_model_sha256": (
                lora.config.base_model_sha256 if lora is not None else None
            ),
            "numerical_profile": profile,
            "publish_base_layer_bundle": publish_base_layer_bundle,
        }
        coordinator = cls(campaign, state_dir, participants, state, current, dataset, lora)
        if campaign.evaluation is not None:
            if dataset is None:
                raise ValueError("campaign evaluation requires dataset artifacts")
            baseline_relative = Path("checkpoints") / "step-00000000"
            if lora is not None:
                baseline = run_lora_training(
                    lora,
                    state_dir / baseline_relative,
                    target_steps=0,
                    dataset=dataset,
                    numerical_profile=checkpoint_profile,
                )
                baseline_identity = baseline.weight_checkpoint_sha256
                baseline_checkpoint = baseline.checkpoint_dir
            else:
                dense_baseline = run_training(
                    campaign,
                    state_dir / baseline_relative,
                    target_steps=0,
                    dataset=dataset,
                )
                baseline_identity = dense_baseline.model_sha256
                baseline_checkpoint = dense_baseline.checkpoint_dir
            if baseline_identity != current.status()["checkpoint_sha256"]:
                raise ValueError("baseline checkpoint does not match campaign initialization")
            state["baseline_checkpoint"] = baseline_relative.as_posix()
            coordinator._checkpoint_snapshots[0] = (
                coordinator._snapshot_versioned_checkpoint(baseline_checkpoint)
            )
            baseline_evaluation = coordinator._evaluate_versioned_checkpoint(
                0,
                baseline_checkpoint,
            )
            if baseline_evaluation is None:
                raise ValueError("baseline evaluation was not produced")
            evaluated_baseline_identity = (
                baseline_evaluation["weight_checkpoint_sha256"]
                if lora is not None
                else baseline_evaluation["checkpoint_sha256"]
            )
            if evaluated_baseline_identity != baseline_identity:
                raise ValueError("baseline evaluation checkpoint identity changed")
            if lora is not None and (
                baseline_evaluation["resume_state_sha256"]
                != current._state["resume_state_sha256"]
            ):
                raise ValueError("baseline evaluation resume-state identity changed")
        coordinator._write_state()
        coordinator._write_lock()
        coordinator._write_ledger()
        coordinator._write_evaluations()
        return coordinator

    @classmethod
    def load(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        participants: ParticipantRegistry,
        dataset: PackedDataset | None = None,
        lora: LoadedLoRAManifest | None = None,
        numerical_profile: str = EXACT_CPU_FP32_PROFILE,
    ) -> CampaignCoordinator:
        state_dir = Path(state_dir)
        state = json.loads(
            _safe_artifact_snapshot(
                state_dir,
                "campaign-state.json",
                "campaign state",
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(state, dict):
            raise ValueError("campaign state must be a JSON object")
        persisted_state_fields = frozenset(state)
        if state.get("format") != "orcacolony_campaign_state_v1":
            raise ValueError("unsupported campaign state format")
        requested_profile = _validated_numerical_profile(numerical_profile)
        numerical_profile_migrated = "numerical_profile" not in state
        if numerical_profile_migrated:
            if requested_profile != EXACT_CPU_FP32_PROFILE:
                raise ValueError("legacy campaign numerical profile is FP32")
            state["numerical_profile"] = EXACT_CPU_FP32_PROFILE
        expected_state_fields = (
            _CAMPAIGN_STATE_FIELDS - {"numerical_profile"}
            if numerical_profile_migrated
            else _CAMPAIGN_STATE_FIELDS
        )
        if persisted_state_fields != expected_state_fields:
            raise ValueError("campaign state schema is invalid")
        stored_profile = _validated_numerical_profile(state.get("numerical_profile"))
        if stored_profile != requested_profile:
            raise ValueError("campaign numerical profile does not match configuration")
        if state.get("campaign_id") != campaign.campaign["id"]:
            raise ValueError("campaign state does not match configuration")
        if state.get("campaign_revision") != _revision(_campaign_payload(campaign)):
            raise ValueError("campaign revision mismatch")
        if state.get("participants_revision") != participants.revision:
            raise ValueError("participant revision mismatch")
        campaign_worker_count = state.get("worker_count")
        if type(campaign_worker_count) is not int or campaign_worker_count < 2:
            raise ValueError("campaign worker count is invalid")
        lease_seconds = state.get("lease_seconds")
        target_steps = state.get("target_steps")
        completed_steps = state.get("completed_steps")
        if (
            type(lease_seconds) is not int
            or lease_seconds <= 0
            or type(target_steps) is not int
            or target_steps < 1
            or type(completed_steps) is not int
            or completed_steps < 0
            or completed_steps > target_steps
        ):
            raise ValueError("campaign progress integers are invalid")
        campaign_state = state.get("state")
        if campaign_state not in {"campaign_running", "campaign_complete"} or (
            campaign_state == "campaign_complete" and completed_steps != target_steps
        ) or (
            campaign_state == "campaign_running" and completed_steps >= target_steps
        ):
            raise ValueError("campaign progress state is invalid")
        expected_training_method = "frozen-base-lora" if lora is not None else "dense"
        if state.get("training_method", "dense") != expected_training_method:
            raise ValueError("campaign training method mismatch")
        if lora is not None and (
            state.get("lora_manifest_sha256") != lora.manifest_sha256
            or state.get("base_model_sha256") != lora.config.base_model_sha256
        ):
            raise ValueError("campaign LoRA identity mismatch")
        if lora is None and (
            state.get("lora_manifest_sha256") is not None
            or state.get("base_model_sha256") is not None
        ):
            raise ValueError("dense campaign contains LoRA identity")
        expected_dataset_revision = (
            dataset.revision if dataset is not None else "synthetic-fixture-v1"
        )
        if state.get("dataset_revision", "synthetic-fixture-v1") != expected_dataset_revision:
            raise ValueError("campaign dataset revision mismatch")
        if type(state.get("publish_base_layer_bundle")) is not bool:
            raise ValueError("campaign layer-bundle publication state is invalid")
        expected_baseline = (
            "checkpoints/step-00000000"
            if campaign.evaluation is not None
            else None
        )
        if state.get("baseline_checkpoint") != expected_baseline:
            raise ValueError("campaign baseline checkpoint path is invalid")
        if expected_baseline is not None:
            _confined_campaign_directory(
                state_dir,
                expected_baseline,
                "campaign baseline checkpoint",
            )
        lock_path = state_dir / "campaign-lock.json"
        expected_lock = _campaign_lock_payload(state, lora)
        expected_stored_lock = dict(expected_lock)
        if numerical_profile_migrated:
            expected_stored_lock.pop("numerical_profile")
        stored_lock = json.loads(
            _safe_artifact_snapshot(
                state_dir,
                lock_path.name,
                "campaign lock",
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not _exact_json_equal(stored_lock, expected_stored_lock):
            raise ValueError("campaign lock mismatch")
        expected_current_step = (
            completed_steps - 1
            if campaign_state == "campaign_complete"
            else completed_steps
        )
        expected_current_relative = (
            Path("rounds") / f"round-{expected_current_step:08d}"
        )
        if state.get("current_round") != expected_current_relative.as_posix():
            raise ValueError("campaign current round path is invalid")
        with _pinned_campaign_directory(
            state_dir,
            state["current_round"],
            "campaign current round",
        ) as current_path:
            current = GlobalStepCoordinator.load(
                campaign,
                current_path,
                participants=participants,
                dataset=dataset,
                lora=lora,
                numerical_profile=stored_profile,
                expected_worker_count=campaign_worker_count,
                persist_migrations=False,
                finalize_ready=False,
            )
        validated_rounds = {str(state["current_round"]): current}
        if current._state.get("lease_seconds") != lease_seconds:
            raise ValueError("campaign child lease duration differs")
        if current.has_base_layer_bundle != bool(state["publish_base_layer_bundle"]):
            raise ValueError("campaign layer-bundle publication state differs")
        checkpoints = state.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ValueError("campaign checkpoint history must be a JSON array")
        if len(checkpoints) != completed_steps:
            raise ValueError("campaign checkpoint history is incomplete")
        current_resolved = current_path.resolve()
        for expected_step, checkpoint in enumerate(checkpoints, start=1):
            expected_checkpoint_fields = _CAMPAIGN_CHECKPOINT_FIELDS
            if numerical_profile_migrated:
                expected_checkpoint_fields -= {"numerical_profile"}
            if (
                not isinstance(checkpoint, Mapping)
                or frozenset(checkpoint) != expected_checkpoint_fields
            ):
                raise ValueError("campaign checkpoint history entry schema is invalid")
            step = checkpoint.get("step")
            if type(step) is not int or step != expected_step:
                raise ValueError("campaign checkpoint history step is invalid")
            expected_round = Path("rounds") / f"round-{step - 1:08d}"
            if checkpoint.get("round") != expected_round.as_posix():
                raise ValueError("campaign checkpoint round path is invalid")
            expected_checkpoint = Path("checkpoints") / f"step-{step:08d}"
            if checkpoint.get("path") != expected_checkpoint.as_posix():
                raise ValueError("campaign checkpoint path is invalid")
            _confined_campaign_directory(
                state_dir,
                checkpoint["path"],
                "campaign checkpoint",
            )
            if numerical_profile_migrated and "numerical_profile" not in checkpoint:
                checkpoint["numerical_profile"] = stored_profile
            if checkpoint.get("numerical_profile") != stored_profile:
                raise ValueError("campaign checkpoint numerical profile mismatch")
            prior_round = _confined_campaign_directory(
                state_dir,
                expected_round.as_posix(),
                "prior campaign round",
            )
            if prior_round.resolve() == current_resolved:
                continue
            with _pinned_campaign_directory(
                state_dir,
                expected_round.as_posix(),
                "prior campaign round",
            ) as prior_round:
                prior_coordinator = GlobalStepCoordinator.load(
                    campaign,
                    prior_round,
                    participants=participants,
                    dataset=dataset,
                    lora=lora,
                    numerical_profile=stored_profile,
                    expected_worker_count=campaign_worker_count,
                    persist_migrations=False,
                    finalize_ready=False,
                )
            validated_rounds[expected_round.as_posix()] = prior_coordinator
            if prior_coordinator.has_base_layer_bundle != bool(
                state["publish_base_layer_bundle"]
            ):
                raise ValueError("prior campaign layer-bundle state differs")
        expected_last_checkpoint_metrics = None
        if completed_steps:
            last_checkpoint_round = str(checkpoints[-1]["round"])
            expected_last_checkpoint_metrics = validated_rounds[
                last_checkpoint_round
            ].status()["checkpoint_metrics"]
        if not _exact_json_equal(
            state.get("last_checkpoint_metrics"),
            expected_last_checkpoint_metrics,
        ):
            raise ValueError("campaign last checkpoint metrics differ")
        coordinator = cls(
            campaign,
            state_dir,
            participants,
            state,
            current,
            dataset,
            lora,
            validated_rounds,
        )
        baseline_relative = state.get("baseline_checkpoint")
        if baseline_relative is not None:
            with _pinned_campaign_directory(
                state_dir,
                str(baseline_relative),
                "campaign baseline checkpoint",
            ) as baseline_checkpoint:
                coordinator._checkpoint_snapshots[0] = (
                    coordinator._snapshot_versioned_checkpoint(baseline_checkpoint)
                )
        for checkpoint_entry in state["checkpoints"]:
            checkpoint_step = checkpoint_entry["step"]
            with _pinned_campaign_directory(
                state_dir,
                str(checkpoint_entry["path"]),
                "campaign checkpoint",
            ) as checkpoint_path:
                coordinator._checkpoint_snapshots[checkpoint_step] = (
                    coordinator._snapshot_versioned_checkpoint(checkpoint_path)
                )
        evaluation_profile_migrated = coordinator._validate_persisted_evaluations()
        current.finalize_if_ready()
        if current._state["state"] == "step_complete":
            current._validate_completed_checkpoint_identity()
        coordinator._prevalidate_existing_next_round()
        for validated_round in validated_rounds.values():
            validated_round.persist_validated_migrations()
        if numerical_profile_migrated or evaluation_profile_migrated:
            coordinator._write_state()
        if numerical_profile_migrated:
            coordinator._write_lock()
        coordinator._advance_if_ready()
        coordinator._write_ledger()
        coordinator._write_evaluations()
        coordinator._write_state()
        return coordinator

    @property
    def initial_model_path(self) -> Path:
        return self._current.initial_model_path

    @property
    def initial_adapter_path(self) -> Path:
        return self._current.initial_adapter_path

    @property
    def checkpoint_dir(self) -> Path:
        checkpoints = self._state["checkpoints"]
        if checkpoints:
            return self.state_dir / str(checkpoints[-1]["path"])
        return self._current.checkpoint_dir

    def oracle_gradient_path(self, assignment_id: str) -> Path:
        return self._current.oracle_gradient_path(assignment_id)

    def oracle_gradient_bytes(self, assignment_id: str) -> bytes:
        return self._current.oracle_gradient_bytes(assignment_id)

    def initial_model_bytes(self) -> bytes:
        return self._current.initial_model_bytes()

    def initial_adapter_bytes(self) -> bytes:
        return self._current.initial_adapter_bytes()

    def dataset_artifact_bytes(self, file_name: str) -> bytes:
        try:
            return self._dataset_snapshots[file_name]
        except KeyError as exc:
            raise ValueError("unknown campaign dataset artifact") from exc

    def checkpoint_artifact_bytes(self, file_name: str) -> bytes:
        checkpoints = self._state["checkpoints"]
        if not checkpoints:
            return self._current.checkpoint_artifact_bytes(file_name)
        artifacts = self.versioned_checkpoint_artifacts(checkpoints[-1]["step"])
        try:
            return artifacts[file_name]
        except KeyError as exc:
            raise ValueError("unknown checkpoint artifact") from exc

    def base_layer_bundle_artifact_path(self, file_name: str) -> Path:
        return self._current.base_layer_bundle_artifact_path(file_name)

    def base_layer_bundle_artifact_bytes(self, file_name: str) -> bytes:
        return self._current.base_layer_bundle_artifact_bytes(file_name)

    def lease(
        self,
        worker_id: str,
        worker_token: str | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if self._state["state"] == "campaign_complete":
                raise ValueError("campaign target is complete")
            return self._current.lease(
                worker_id,
                worker_token=worker_token,
                now=now,
            )

    def accept(
        self,
        submission: LeasedGradient,
        now: float | None = None,
    ) -> WorkReceipt:
        with self._lock:
            if self._state["state"] == "campaign_complete":
                raise ValueError("campaign target is complete")
            receipt = self._current.accept(submission, now=now)
            if receipt.step_complete:
                self._advance_if_ready()
            return receipt

    def _write_state(self) -> None:
        _atomic_json(self.state_dir / "campaign-state.json", self._state)

    def _lock_payload(self) -> dict[str, object]:
        return _campaign_lock_payload(self._state, self.lora)

    def _write_lock(self) -> None:
        _atomic_json(self.state_dir / "campaign-lock.json", self._lock_payload())

    def _write_evaluations(self) -> None:
        _atomic_json(
            self.state_dir / "evaluations.json",
            {
                "format": "orcacolony_campaign_evaluations_v1",
                "campaign_id": self._state["campaign_id"],
                "dataset_revision": self._state.get(
                    "dataset_revision", "synthetic-fixture-v1"
                ),
                "numerical_profile": self._state["numerical_profile"],
                "profile": (
                    dict(self.campaign.evaluation)
                    if self.campaign.evaluation is not None
                    else None
                ),
                "entries": self._state["evaluations"],
            },
        )

    def _validate_persisted_evaluations(self) -> bool:
        evaluations = self._state.get("evaluations")
        last_evaluation = self._state.get("last_evaluation")
        if not isinstance(evaluations, list) or any(
            not isinstance(entry, dict) for entry in evaluations
        ):
            raise ValueError("persisted campaign evaluations are invalid")
        if self.campaign.evaluation is None:
            if evaluations or last_evaluation is not None:
                raise ValueError("campaign has unexpected persisted evaluations")
            return False
        if not evaluations or not _exact_json_equal(last_evaluation, evaluations[-1]):
            raise ValueError("persisted last evaluation differs from evaluation history")

        baseline = self._state.get("baseline_checkpoint")
        checkpoints = self._state.get("checkpoints")
        if not isinstance(baseline, str) or not isinstance(checkpoints, list):
            raise ValueError("persisted evaluation checkpoint history is invalid")
        expected_checkpoints: list[tuple[int, str]] = [
            (0, "checkpoints/step-00000000")
        ]
        if baseline != expected_checkpoints[0][1]:
            raise ValueError("persisted baseline evaluation checkpoint differs")
        for expected_step, checkpoint in enumerate(checkpoints, start=1):
            expected_path = f"checkpoints/step-{expected_step:08d}"
            if (
                not isinstance(checkpoint, Mapping)
                or checkpoint.get("step") != expected_step
                or checkpoint.get("path") != expected_path
            ):
                raise ValueError("persisted evaluation checkpoint history differs")
            expected_checkpoints.append((expected_step, expected_path))
        if len(evaluations) != len(expected_checkpoints):
            raise ValueError("persisted evaluation history is incomplete")

        expected_profile = str(self._state["numerical_profile"])
        expected_dataset_revision = str(self._state["dataset_revision"])
        if self.dataset is None:
            raise ValueError("persisted evaluation dataset authority is unavailable")
        migrated = False
        for entry, (expected_step, relative_path) in zip(
            evaluations,
            expected_checkpoints,
            strict=True,
        ):
            evaluation_fields = {
                "format",
                "campaign_id",
                "step",
                "dataset_revision",
                "checkpoint_sha256",
                "validation_sequences",
                "loss_sum",
                "loss_weight_sum",
                "mean_loss",
                "perplexity",
                "numerical_profile",
            }
            if self.lora is not None:
                evaluation_fields.update(
                    {
                        "training_method",
                        "base_model_sha256",
                        "adapter_sha256",
                        "weight_checkpoint_sha256",
                        "resume_state_sha256",
                    }
                )
            if entry.get("format") != "orcacolony_evaluation_v1":
                raise ValueError("persisted evaluation format is invalid")
            if "numerical_profile" not in entry:
                if (
                    expected_profile != EXACT_CPU_FP32_PROFILE
                    or set(entry) != evaluation_fields - {"numerical_profile"}
                ):
                    raise ValueError(
                        "persisted evaluation predecessor schema is invalid"
                    )
                entry["numerical_profile"] = EXACT_CPU_FP32_PROFILE
                migrated = True
            elif set(entry) != evaluation_fields:
                raise ValueError("persisted evaluation schema is invalid")
            if (
                entry.get("campaign_id") != self._state["campaign_id"]
                or entry.get("dataset_revision") != expected_dataset_revision
                or type(entry.get("step")) is not int
                or entry["step"] != expected_step
            ):
                raise ValueError("persisted evaluation identity differs")
            entry_profile = entry.get("numerical_profile")
            if entry_profile != expected_profile:
                raise ValueError("evaluation numerical profile differs")
            if self.lora is not None and entry.get("training_method") != (
                "frozen-base-lora"
            ):
                raise ValueError("persisted evaluation training method differs")

            with self._owned_versioned_checkpoint(expected_step) as checkpoint_dir:
                recomputed = (
                    evaluate_lora_checkpoint(
                        self.lora,
                        checkpoint_dir,
                        self.dataset,
                    )
                    if self.lora is not None
                    else evaluate_checkpoint(
                        self.campaign,
                        checkpoint_dir,
                        self.dataset,
                    )
                )
            recomputed.setdefault("numerical_profile", expected_profile)
            if not _exact_json_equal(entry, recomputed):
                raise ValueError("persisted evaluation differs from recomputation")
        if migrated:
            self._state["last_evaluation"] = dict(evaluations[-1])
        return migrated

    def validate_evaluation_authority(self) -> None:
        with self._lock:
            if self._validate_persisted_evaluations():
                raise ValueError("evaluation authority requires migration")

    def _evaluate_versioned_checkpoint(
        self,
        step: int,
        checkpoint: Path,
    ) -> dict[str, object] | None:
        if self.campaign.evaluation is None:
            return
        if self.dataset is None:
            raise ValueError("campaign evaluation requires dataset artifacts")
        evaluations: list[dict[str, object]] = self._state[  # type: ignore[assignment]
            "evaluations"
        ]
        existing = next(
            (entry for entry in evaluations if entry["step"] == step),
            None,
        )
        if existing is None:
            expected_checkpoint = self.checkpoints_dir / f"step-{step:08d}"
            if checkpoint != expected_checkpoint:
                raise ValueError("campaign evaluation checkpoint path differs")
            with self._owned_versioned_checkpoint(step) as pinned_checkpoint:
                existing = (
                    evaluate_lora_checkpoint(
                        self.lora,
                        pinned_checkpoint,
                        self.dataset,
                    )
                    if self.lora is not None
                    else evaluate_checkpoint(
                        self.campaign,
                        pinned_checkpoint,
                        self.dataset,
                    )
                )
            existing.setdefault(
                "numerical_profile",
                self._state["numerical_profile"],
            )
            evaluations.append(existing)
            evaluations.sort(key=lambda entry: entry["step"])
        self._state["last_evaluation"] = existing
        self._write_evaluations()
        return existing

    def _snapshot_versioned_checkpoint(self, checkpoint: Path) -> dict[str, bytes]:
        expected_names = {
            "state.json",
            "optimizer.safetensors",
            "adapter.safetensors" if self.lora is not None else "model.safetensors",
        }
        return {
            name: _safe_artifact_snapshot(
                checkpoint,
                name,
                "versioned campaign checkpoint artifact",
            )
            for name in expected_names
        }

    def versioned_checkpoint_artifacts(self, step: int) -> dict[str, bytes]:
        if type(step) is not int or step < 0:
            raise ValueError("campaign checkpoint step is invalid")
        try:
            return dict(self._checkpoint_snapshots[step])
        except KeyError as exc:
            raise ValueError("campaign checkpoint snapshot is unavailable") from exc

    @contextmanager
    def _owned_versioned_checkpoint(self, step: int) -> Iterator[Path]:
        artifacts = self.versioned_checkpoint_artifacts(step)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".evaluation-step-{step:08d}-",
                dir=self.state_dir,
            )
        )
        checkpoint = temporary_root / "checkpoint"
        checkpoint.mkdir()
        try:
            for name, payload in artifacts.items():
                _atomic_bytes(checkpoint / name, payload)
            yield checkpoint
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _version_checkpoint(self, step: int) -> Path:
        destination = self.checkpoints_dir / f"step-{step:08d}"
        artifacts = self._current.completed_checkpoint_artifacts()
        expected_names = {
            "state.json",
            "optimizer.safetensors",
            "adapter.safetensors" if self.lora is not None else "model.safetensors",
        }
        if set(artifacts) != expected_names:
            raise ValueError("completed checkpoint artifact set is invalid")
        if destination.exists():
            existing = {
                name: _safe_artifact_snapshot(
                    destination,
                    name,
                    "versioned campaign checkpoint artifact",
                )
                for name in expected_names
            }
            if existing != artifacts:
                raise ValueError("versioned checkpoint does not match completed round")
            self._checkpoint_snapshots[step] = dict(artifacts)
            return destination

        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise ValueError("versioned checkpoint temporary path already exists")
        temporary.mkdir()
        try:
            for name, payload in artifacts.items():
                _atomic_bytes(temporary / name, payload)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self._checkpoint_snapshots[step] = dict(artifacts)
        return destination

    def _validate_next_round_authority(
        self,
        next_round: GlobalStepCoordinator,
        completed_step: int,
    ) -> None:
        expected_checkpoint_sha256 = (
            self._current._state["result_weight_checkpoint_sha256"]
            if self.lora is not None
            else self._current._state["model_sha256"]
        )
        expected_authority = {
            "base_step": completed_step,
            "dataset_cursor": self._current._state["result_dataset_cursor"],
            "checkpoint_sha256": expected_checkpoint_sha256,
            "resume_state_sha256": self._current._state[
                "result_resume_state_sha256"
            ],
        }
        for field, expected in expected_authority.items():
            if next_round._state[field] != expected:
                raise ValueError(
                    f"next campaign round checkpoint authority differs: {field}"
                )
        if next_round.has_base_layer_bundle != bool(
            self._state["publish_base_layer_bundle"]
        ):
            raise ValueError("next campaign layer-bundle publication state differs")
        if next_round._state["lease_seconds"] != self._state["lease_seconds"]:
            raise ValueError("next campaign child lease duration differs")

    def _prevalidate_existing_next_round(self) -> None:
        status = self._current.status()
        if status["state"] != "step_complete":
            return
        completed_step = status["step"]
        if completed_step >= self._state["target_steps"]:
            return
        next_relative = Path("rounds") / f"round-{completed_step:08d}"
        next_path = self.state_dir / next_relative
        if not next_path.exists():
            return
        with _pinned_campaign_directory(
            self.state_dir,
            next_relative.as_posix(),
            "next campaign round",
        ) as pinned_next:
            next_round = GlobalStepCoordinator.load(
                self.campaign,
                pinned_next,
                participants=self.participants,
                dataset=self.dataset,
                lora=self.lora,
                numerical_profile=str(self._state["numerical_profile"]),
                expected_worker_count=self._state["worker_count"],
                persist_migrations=False,
                finalize_ready=False,
            )
        self._validate_next_round_authority(next_round, completed_step)
        self._prevalidated_next_round = (next_relative.as_posix(), next_round)

    def _advance_if_ready(self) -> None:
        if self._current.status()["state"] != "step_complete":
            return
        completed_step = self._current.status()["step"]
        checkpoint = self._version_checkpoint(completed_step)
        checkpoints: list[dict[str, object]] = self._state["checkpoints"]  # type: ignore[assignment]
        if not any(value["step"] == completed_step for value in checkpoints):
            checkpoints.append(
                {
                    "step": completed_step,
                    "path": checkpoint.relative_to(self.state_dir).as_posix(),
                    "round": self._state["current_round"],
                    "numerical_profile": self._state["numerical_profile"],
                }
            )
        self._state["completed_steps"] = completed_step
        self._state["last_checkpoint_metrics"] = self._current.status()[
            "checkpoint_metrics"
        ]
        self._evaluate_versioned_checkpoint(completed_step, checkpoint)

        if completed_step >= self._state["target_steps"]:
            self._state["state"] = "campaign_complete"
            self._write_state()
            self._write_ledger()
            return

        next_relative = Path("rounds") / f"round-{completed_step:08d}"
        next_path = self.state_dir / next_relative
        loaded_existing_round = (next_path / "global-state.json").is_file()
        if loaded_existing_round:
            cached = self._prevalidated_next_round
            if cached is not None and cached[0] == next_relative.as_posix():
                next_round = cached[1]
            else:
                with _pinned_campaign_directory(
                    self.state_dir,
                    next_relative.as_posix(),
                    "next campaign round",
                ) as pinned_next:
                    next_round = GlobalStepCoordinator.load(
                        self.campaign,
                        pinned_next,
                        participants=self.participants,
                        dataset=self.dataset,
                        lora=self.lora,
                        numerical_profile=str(self._state["numerical_profile"]),
                        expected_worker_count=self._state["worker_count"],
                        persist_migrations=False,
                        finalize_ready=False,
                    )
        else:
            next_round = GlobalStepCoordinator.create(
                self.campaign,
                next_path,
                worker_count=self._state["worker_count"],
                participants=self.participants,
                lease_seconds=self._state["lease_seconds"],
                resume_from=checkpoint,
                dataset=self.dataset,
                lora=self.lora,
                publish_base_layer_bundle=bool(
                    self._state["publish_base_layer_bundle"]
                ),
                numerical_profile=str(self._state["numerical_profile"]),
            )
        self._validate_next_round_authority(next_round, completed_step)
        if loaded_existing_round:
            next_round.finalize_if_ready()
            if next_round._state["state"] == "step_complete":
                next_round._validate_completed_checkpoint_identity()
        self._rounds[next_relative.as_posix()] = next_round
        self._current = next_round
        self._prevalidated_next_round = None
        self._state["current_round"] = next_relative.as_posix()
        self._state["state"] = "campaign_running"
        self._write_state()
        self._write_ledger()
        if loaded_existing_round:
            next_round.persist_validated_migrations()

    def _ledger_payload(self, *, include_current: bool = False) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        completed_rounds: set[str] = set()
        for checkpoint in self._state["checkpoints"]:
            round_name = str(checkpoint["round"])
            completed_rounds.add(round_name)
            try:
                round_coordinator = self._rounds[round_name]
            except KeyError as exc:
                raise ValueError("campaign round authority is unavailable") from exc
            payload = round_coordinator.accepted_ledger_payload()
            for entry in payload["entries"]:
                entries.append({**entry, "checkpoint_step": checkpoint["step"]})
        if (
            include_current
            and str(self._state["current_round"]) not in completed_rounds
        ):
            next_step = self._state["completed_steps"] + 1
            current = self._current.accepted_ledger_payload()
            entries.extend(
                {**entry, "checkpoint_step": next_step}
                for entry in current["entries"]
            )
        entries.sort(key=lambda value: (value["checkpoint_step"], value["data_range"][0]))
        return {
            "format": "orcacolony_campaign_accepted_work_v1",
            "campaign_id": self._state["campaign_id"],
            "participants_revision": self._state["participants_revision"],
            "dataset_revision": self._state.get(
                "dataset_revision", "synthetic-fixture-v1"
            ),
            "numerical_profile": self._state["numerical_profile"],
            "entries": entries,
        }

    def _write_ledger(self) -> None:
        _atomic_json(
            self.state_dir / "accepted-work.json",
            self._ledger_payload(),
        )

    def status(self) -> dict[str, object]:
        current = self._current.public_status()
        return {
            "state": self._state["state"],
            "campaign_id": self._state["campaign_id"],
            "numerical_profile": self._state["numerical_profile"],
            "completed_steps": self._state["completed_steps"],
            "target_steps": self._state["target_steps"],
            "current_step": current["step"],
            "checkpoint_metrics": self._state["last_checkpoint_metrics"],
            "last_evaluation": self._state["last_evaluation"],
            "evaluation_gate": self.evaluation_gate(),
            "current_round": current,
        }

    def _dashboard_ledger_entries(self) -> list[dict[str, object]]:
        return [
            dict(entry)
            for entry in self._ledger_payload(include_current=True)["entries"]
        ]

    def evaluation_gate(self) -> dict[str, object] | None:
        if self.campaign.evaluation is None:
            return None
        gate = self.campaign.evaluation.get("success_gate")
        if not isinstance(gate, Mapping):
            return None
        evaluations = self._state["evaluations"]
        if not evaluations:
            return {
                "metric": "mean_loss",
                "state": "pending",
                "minimum_improvement_from_initialization": float(
                    gate["minimum_improvement_from_initialization"]
                ),
            }
        baseline = next(
            (entry for entry in evaluations if entry["step"] == 0),
            None,
        )
        latest = max(evaluations, key=lambda entry: entry["step"])
        if baseline is None:
            raise ValueError("evaluation success gate requires an initialization baseline")
        improvement = baseline["mean_loss"] - latest["mean_loss"]
        minimum = float(gate["minimum_improvement_from_initialization"])
        complete = self._state["completed_steps"] >= self._state["target_steps"]
        gate_state = "pending"
        if complete:
            gate_state = "passed" if improvement >= minimum else "failed"
        return {
            "metric": "mean_loss",
            "state": gate_state,
            "minimum_improvement_from_initialization": minimum,
            "observed_improvement_from_initialization": improvement,
            "baseline_step": baseline["step"],
            "evaluated_step": latest["step"],
        }

    def dashboard(self) -> dict[str, object]:
        with self._lock:
            status = self.status()
            entries = self._dashboard_ledger_entries()
            participant_by_id = {
                participant.contributor_id: participant
                for participant in self.participants.participants
            }
            contributor_totals: dict[str, dict[str, int]] = {}
            public_ledger: list[dict[str, object]] = []
            for entry in entries:
                contributor_id = str(entry["contributor_id"])
                participant = participant_by_id[contributor_id]
                totals = contributor_totals.setdefault(
                    contributor_id,
                    {"accepted_assignments": 0, "accepted_tokens": 0},
                )
                totals["accepted_assignments"] += 1
                totals["accepted_tokens"] += entry["loss_weight_sum"]
                public_ledger.append(
                    {
                        "contribution_id": entry["assignment_id"],
                        "checkpoint_step": entry["checkpoint_step"],
                        "accepted_tokens": entry["loss_weight_sum"],
                        "runtime_backend": entry["runtime_backend"],
                        "credit": (
                            participant.display_name
                            if (
                                participant.public_credit
                                and participant.show_contribution_totals
                            )
                            else "Anonymous"
                        ),
                    }
                )

            acknowledgements: list[dict[str, object]] = []
            anonymous_count = 0
            for contributor_id, totals in sorted(contributor_totals.items()):
                participant = participant_by_id[contributor_id]
                if participant.public_credit:
                    acknowledgement: dict[str, object] = {
                        "display_name": participant.display_name,
                    }
                    if participant.show_contribution_totals:
                        acknowledgement.update(totals)
                    acknowledgements.append(acknowledgement)
                else:
                    anonymous_count += 1

            current_round = status["current_round"]
            lora_mode = current_round["training_method"] == "frozen-base-lora"
            adapter_sha256 = (
                current_round["adapter_sha256"]
                or current_round["initial_adapter_sha256"]
            )
            weight_checkpoint_sha256 = (
                current_round["result_weight_checkpoint_sha256"]
                or current_round["checkpoint_sha256"]
            )
            resume_state_sha256 = (
                current_round["result_checkpoint_sha256"]
                or current_round["resume_state_sha256"]
            )
            assignment_states = [
                assignment["state"] for assignment in current_round["assignments"]
            ]
            source: dict[str, object] = {"dataset": "synthetic-fixture-v1"}
            dataset_packing: dict[str, object] | None = None
            if self.dataset is not None:
                manifest_source = self.dataset.manifest.get("source")
                if isinstance(manifest_source, Mapping):
                    source = {
                        field: manifest_source[field]
                        for field in _PUBLIC_DATASET_SOURCE_FIELDS
                        if field in manifest_source
                        and isinstance(
                            manifest_source[field], (str, int, float, bool)
                        )
                    }
                manifest_packing = self.dataset.manifest.get("packing")
                if isinstance(manifest_packing, Mapping):
                    dataset_packing = {
                        field: manifest_packing[field]
                        for field in (
                            "context_length",
                            "train_sequences",
                            "train_tokens",
                            "validation_sequences",
                            "validation_tokens",
                        )
                        if field in manifest_packing
                        and isinstance(manifest_packing[field], int)
                    }
            accepted_tokens = sum(
                totals["accepted_tokens"] for totals in contributor_totals.values()
            )
            return {
                "format": "orcacolony_public_dashboard_v1",
                "campaign": {
                    "id": self._state["campaign_id"],
                    "objective": self.campaign.campaign["objective"],
                    "state": self._state["state"],
                    "training_method": current_round["training_method"],
                    "numerical_profile": self._state["numerical_profile"],
                },
                "model": {
                    **asdict(self.campaign.model),
                    "parameter_count": self._current.assignments[0][
                        "parameter_count"
                    ],
                },
                "dataset": {
                    "revision": self._state["dataset_revision"],
                    "source": source,
                    "packing": dataset_packing,
                },
                "progress": {
                    "completed_steps": self._state["completed_steps"],
                    "target_steps": self._state["target_steps"],
                    "accepted_assignments": len(entries),
                    "target_assignments": self._state["target_steps"]
                    * self._state["worker_count"],
                    "accepted_tokens": accepted_tokens,
                    "open_assignments": assignment_states.count("open"),
                    "leased_assignments": assignment_states.count("leased"),
                },
                "checkpoint": {
                    "numerical_profile": self._state["numerical_profile"],
                    "sha256": (
                        resume_state_sha256
                        if lora_mode
                        else current_round["model_sha256"]
                        or current_round["checkpoint_sha256"]
                    ),
                    "base_model_sha256": (
                        current_round["base_model_sha256"] if lora_mode else None
                    ),
                    "adapter_sha256": adapter_sha256 if lora_mode else None,
                    "weight_checkpoint_sha256": (
                        weight_checkpoint_sha256 if lora_mode else None
                    ),
                    "resume_state_sha256": (
                        resume_state_sha256 if lora_mode else None
                    ),
                    "download_url": (
                        (
                            "/api/v1/checkpoint/adapter.safetensors"
                            if lora_mode
                            else "/api/v1/checkpoint/model.safetensors"
                        )
                        if self._state["completed_steps"] > 0
                        else None
                    ),
                    "parity": self._state["last_checkpoint_metrics"],
                },
                "evaluations": [dict(entry) for entry in self._state["evaluations"]],
                "evaluation_gate": self.evaluation_gate(),
                "contributors": {
                    "active_count": len(contributor_totals),
                    "anonymous_count": anonymous_count,
                    "acknowledgements": acknowledgements,
                },
                "resource_observations": _aggregate_resource_observations(
                    entries,
                    self.state_dir,
                ),
                "public_ledger": public_ledger,
            }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent multi-step OrcaColony campaign"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--dataset-artifacts", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument(
        "--numerical-profile",
        choices=(
            EXACT_CPU_FP32_PROFILE,
            BURN_NDARRAY_F32_PROFILE,
            BURN_WEBGPU_F32_PROFILE,
            INT8_FROZEN_LINEAR_PROFILE,
        ),
        default=EXACT_CPU_FP32_PROFILE,
    )
    parser.add_argument("--publish-base-layer-bundle", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-origin")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    lora = (
        load_lora_manifest(args.config, args.lora_config)
        if args.lora_config is not None
        else None
    )
    campaign = lora.campaign if lora is not None else load_campaign(args.config)
    participants = load_participants(
        args.participants,
        campaign_id=str(campaign.campaign["id"]),
    )
    dataset = (
        PackedDataset.load(args.dataset_artifacts)
        if args.dataset_artifacts is not None
        else None
    )
    if (args.state / "campaign-state.json").is_file():
        coordinator = CampaignCoordinator.load(
            campaign,
            args.state,
            participants=participants,
            dataset=dataset,
            lora=lora,
            numerical_profile=args.numerical_profile,
        )
        if coordinator.status()["target_steps"] != args.target_steps:
            raise ValueError("target steps do not match the campaign lock")
    else:
        coordinator = CampaignCoordinator.create(
            campaign,
            args.state,
            participants=participants,
            worker_count=args.workers,
            target_steps=args.target_steps,
            lease_seconds=args.lease_seconds,
            dataset=dataset,
            lora=lora,
            publish_base_layer_bundle=args.publish_base_layer_bundle,
            numerical_profile=args.numerical_profile,
        )
    server = create_http_server(
        coordinator,  # type: ignore[arg-type]
        args.browser_root,
        host=args.host,
        port=args.port,
        public_origin=args.public_origin,
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign["id"],
                "status": coordinator.status(),
                "url_template": (
                    f"http://{args.host}:{server.server_port}/"
                    "?worker=<worker-id>#token=<worker-token>"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
