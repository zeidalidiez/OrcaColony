from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from .artifacts import PackedDataset
from .multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    WorkReceipt,
    _atomic_json,
    _aggregate_resource_observations,
    _campaign_payload,
    _revision,
    create_http_server,
)
from .participants import ParticipantRegistry, load_participants
from .peft import (
    LoadedLoRAManifest,
    evaluate_lora_checkpoint,
    load_lora_manifest,
    run_lora_training,
)
from .reference import CampaignConfig, evaluate_checkpoint, load_campaign, run_training


_PUBLIC_DATASET_SOURCE_FIELDS = (
    "dataset",
    "dataset_card",
    "license",
    "license_url",
    "revision",
    "selection",
)


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
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.participants = participants
        self._state = state
        self._current = current
        self.dataset = dataset
        self.lora = lora
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
    ) -> CampaignCoordinator:
        if target_steps < 1:
            raise ValueError("campaign target steps must be positive")
        if participants.campaign_id != campaign.campaign["id"]:
            raise ValueError("participant campaign does not match configuration")
        if lora is not None and lora.campaign != campaign:
            raise ValueError("LoRA manifest campaign does not match campaign run")
        if publish_base_layer_bundle and lora is None:
            raise ValueError("base layer bundle publication requires frozen-base LoRA")
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
            coordinator._evaluate_versioned_checkpoint(0, baseline_checkpoint)
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
    ) -> CampaignCoordinator:
        state_dir = Path(state_dir)
        state = json.loads(
            (state_dir / "campaign-state.json").read_text(encoding="utf-8")
        )
        if state.get("format") != "orcacolony_campaign_state_v1":
            raise ValueError("unsupported campaign state format")
        if state.get("campaign_id") != campaign.campaign["id"]:
            raise ValueError("campaign state does not match configuration")
        if state.get("campaign_revision") != _revision(_campaign_payload(campaign)):
            raise ValueError("campaign revision mismatch")
        if state.get("participants_revision") != participants.revision:
            raise ValueError("participant revision mismatch")
        expected_training_method = "frozen-base-lora" if lora is not None else "dense"
        if state.get("training_method", "dense") != expected_training_method:
            raise ValueError("campaign training method mismatch")
        if lora is not None and (
            state.get("lora_manifest_sha256") != lora.manifest_sha256
            or state.get("base_model_sha256") != lora.config.base_model_sha256
        ):
            raise ValueError("campaign LoRA identity mismatch")
        expected_dataset_revision = (
            dataset.revision if dataset is not None else "synthetic-fixture-v1"
        )
        if state.get("dataset_revision", "synthetic-fixture-v1") != expected_dataset_revision:
            raise ValueError("campaign dataset revision mismatch")
        state.setdefault("evaluations", [])
        state.setdefault("last_evaluation", None)
        state.setdefault("baseline_checkpoint", None)
        state.setdefault("publish_base_layer_bundle", False)
        current_path = state_dir / str(state["current_round"])
        current = GlobalStepCoordinator.load(
            campaign,
            current_path,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )
        if current.has_base_layer_bundle != bool(state["publish_base_layer_bundle"]):
            raise ValueError("campaign layer-bundle publication state differs")
        checkpoints = state.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ValueError("campaign checkpoint history must be a JSON array")
        current_resolved = current_path.resolve()
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                raise ValueError("campaign checkpoint history entry must be a JSON object")
            step = checkpoint.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError("campaign checkpoint history step is invalid")
            expected_round = Path("rounds") / f"round-{step - 1:08d}"
            if checkpoint.get("round") != expected_round.as_posix():
                raise ValueError("campaign checkpoint round path is invalid")
            prior_round = (state_dir / expected_round).resolve()
            if prior_round == current_resolved:
                continue
            prior_coordinator = GlobalStepCoordinator.load(
                campaign,
                prior_round,
                participants=participants,
                dataset=dataset,
                lora=lora,
            )
            if prior_coordinator.has_base_layer_bundle != bool(
                state["publish_base_layer_bundle"]
            ):
                raise ValueError("prior campaign layer-bundle state differs")
        coordinator = cls(campaign, state_dir, participants, state, current, dataset, lora)
        lock_path = state_dir / "campaign-lock.json"
        if (
            not lock_path.is_file()
            or json.loads(lock_path.read_text(encoding="utf-8"))
            != coordinator._lock_payload()
        ):
            raise ValueError("campaign lock mismatch")
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

    def base_layer_bundle_artifact_path(self, file_name: str) -> Path:
        return self._current.base_layer_bundle_artifact_path(file_name)

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
        payload: dict[str, object] = {
            "format": "orcacolony_campaign_run_lock_v1",
            "campaign_id": self._state["campaign_id"],
            "campaign_revision": self._state["campaign_revision"],
            "participants_revision": self._state["participants_revision"],
            "dataset_revision": self._state.get(
                "dataset_revision", "synthetic-fixture-v1"
            ),
            "worker_count": self._state["worker_count"],
            "target_steps": self._state["target_steps"],
            "assignment_protocol_revision": 1,
            "result_protocol_revision": 2,
        }
        if self.lora is not None:
            payload.update(
                {
                    "training_method": "frozen-base-lora",
                    "lora_manifest_sha256": self.lora.manifest_sha256,
                    "base_model_sha256": self.lora.config.base_model_sha256,
                    "result_protocol_revision": 3,
                }
            )
        if self._state["publish_base_layer_bundle"]:
            payload["publish_base_layer_bundle"] = True
        return payload

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
                "profile": (
                    dict(self.campaign.evaluation)
                    if self.campaign.evaluation is not None
                    else None
                ),
                "entries": self._state["evaluations"],
            },
        )

    def _evaluate_versioned_checkpoint(self, step: int, checkpoint: Path) -> None:
        if self.campaign.evaluation is None:
            return
        if self.dataset is None:
            raise ValueError("campaign evaluation requires dataset artifacts")
        evaluations: list[dict[str, object]] = self._state[  # type: ignore[assignment]
            "evaluations"
        ]
        existing = next(
            (entry for entry in evaluations if int(entry["step"]) == step),
            None,
        )
        if existing is None:
            existing = (
                evaluate_lora_checkpoint(self.lora, checkpoint, self.dataset)
                if self.lora is not None
                else evaluate_checkpoint(self.campaign, checkpoint, self.dataset)
            )
            evaluations.append(existing)
            evaluations.sort(key=lambda entry: int(entry["step"]))
        self._state["last_evaluation"] = existing
        self._write_evaluations()

    def _version_checkpoint(self, step: int) -> Path:
        destination = self.checkpoints_dir / f"step-{step:08d}"
        source = self._current.checkpoint_dir
        source_state = json.loads((source / "state.json").read_text(encoding="utf-8"))
        if destination.exists():
            destination_state = json.loads(
                (destination / "state.json").read_text(encoding="utf-8")
            )
            if destination_state != source_state:
                raise ValueError("versioned checkpoint does not match completed round")
            return destination

        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
        return destination

    def _advance_if_ready(self) -> None:
        if self._current.status()["state"] != "step_complete":
            return
        completed_step = int(self._current.status()["step"])
        checkpoint = self._version_checkpoint(completed_step)
        checkpoints: list[dict[str, object]] = self._state["checkpoints"]  # type: ignore[assignment]
        if not any(int(value["step"]) == completed_step for value in checkpoints):
            checkpoints.append(
                {
                    "step": completed_step,
                    "path": checkpoint.relative_to(self.state_dir).as_posix(),
                    "round": self._state["current_round"],
                }
            )
        self._state["completed_steps"] = completed_step
        self._state["last_checkpoint_metrics"] = self._current.status()[
            "checkpoint_metrics"
        ]
        self._evaluate_versioned_checkpoint(completed_step, checkpoint)

        if completed_step >= int(self._state["target_steps"]):
            self._state["state"] = "campaign_complete"
            self._write_state()
            self._write_ledger()
            return

        next_relative = Path("rounds") / f"round-{completed_step:08d}"
        next_path = self.state_dir / next_relative
        if (next_path / "global-state.json").is_file():
            next_round = GlobalStepCoordinator.load(
                self.campaign,
                next_path,
                participants=self.participants,
                dataset=self.dataset,
                lora=self.lora,
            )
        else:
            next_round = GlobalStepCoordinator.create(
                self.campaign,
                next_path,
                worker_count=int(self._state["worker_count"]),
                participants=self.participants,
                lease_seconds=int(self._state["lease_seconds"]),
                resume_from=checkpoint,
                dataset=self.dataset,
                lora=self.lora,
                publish_base_layer_bundle=bool(
                    self._state["publish_base_layer_bundle"]
                ),
            )
        self._current = next_round
        self._state["current_round"] = next_relative.as_posix()
        self._state["state"] = "campaign_running"
        self._write_state()
        self._write_ledger()

    def _write_ledger(self) -> None:
        entries: list[dict[str, object]] = []
        for checkpoint in self._state["checkpoints"]:
            round_ledger = self.state_dir / str(checkpoint["round"]) / "accepted-work.json"
            payload = json.loads(round_ledger.read_text(encoding="utf-8"))
            for entry in payload["entries"]:
                entries.append({**entry, "checkpoint_step": checkpoint["step"]})
        entries.sort(key=lambda value: (value["checkpoint_step"], value["data_range"][0]))
        _atomic_json(
            self.state_dir / "accepted-work.json",
            {
                "format": "orcacolony_campaign_accepted_work_v1",
                "campaign_id": self._state["campaign_id"],
                "participants_revision": self._state["participants_revision"],
                "dataset_revision": self._state.get(
                    "dataset_revision", "synthetic-fixture-v1"
                ),
                "entries": entries,
            },
        )

    def status(self) -> dict[str, object]:
        current = self._current.public_status()
        return {
            "state": self._state["state"],
            "campaign_id": self._state["campaign_id"],
            "completed_steps": self._state["completed_steps"],
            "target_steps": self._state["target_steps"],
            "current_step": current["step"],
            "checkpoint_metrics": self._state["last_checkpoint_metrics"],
            "last_evaluation": self._state["last_evaluation"],
            "evaluation_gate": self.evaluation_gate(),
            "current_round": current,
        }

    def _dashboard_ledger_entries(self) -> list[dict[str, object]]:
        payload = json.loads(
            (self.state_dir / "accepted-work.json").read_text(encoding="utf-8")
        )
        entries = [dict(entry) for entry in payload["entries"]]
        completed_rounds = {
            str(checkpoint["round"]) for checkpoint in self._state["checkpoints"]
        }
        if str(self._state["current_round"]) not in completed_rounds:
            current = json.loads(
                (self._current.state_dir / "accepted-work.json").read_text(
                    encoding="utf-8"
                )
            )
            next_step = int(self._state["completed_steps"]) + 1
            entries.extend(
                {**entry, "checkpoint_step": next_step}
                for entry in current["entries"]
            )
        entries.sort(
            key=lambda value: (
                int(value["checkpoint_step"]),
                int(value["data_range"][0]),
            )
        )
        return entries

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
            (entry for entry in evaluations if int(entry["step"]) == 0),
            None,
        )
        latest = max(evaluations, key=lambda entry: int(entry["step"]))
        if baseline is None:
            raise ValueError("evaluation success gate requires an initialization baseline")
        improvement = float(baseline["mean_loss"]) - float(latest["mean_loss"])
        minimum = float(gate["minimum_improvement_from_initialization"])
        complete = int(self._state["completed_steps"]) >= int(
            self._state["target_steps"]
        )
        gate_state = "pending"
        if complete:
            gate_state = "passed" if improvement >= minimum else "failed"
        return {
            "metric": "mean_loss",
            "state": gate_state,
            "minimum_improvement_from_initialization": minimum,
            "observed_improvement_from_initialization": improvement,
            "baseline_step": int(baseline["step"]),
            "evaluated_step": int(latest["step"]),
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
                totals = contributor_totals.setdefault(
                    contributor_id,
                    {"accepted_assignments": 0, "accepted_tokens": 0},
                )
                totals["accepted_assignments"] += 1
                totals["accepted_tokens"] += int(entry["loss_weight_sum"])
                public_ledger.append(
                    {
                        "contribution_id": entry["assignment_id"],
                        "checkpoint_step": entry["checkpoint_step"],
                        "accepted_tokens": entry["loss_weight_sum"],
                        "runtime_backend": entry["runtime_backend"],
                        "credit": (
                            entry["public_credit"]["display_name"]
                            if entry["public_credit"] is not None
                            else "Anonymous"
                        ),
                    }
                )

            acknowledgements: list[dict[str, object]] = []
            anonymous_count = 0
            for contributor_id, totals in sorted(contributor_totals.items()):
                participant = participant_by_id[contributor_id]
                if participant.public_credit:
                    acknowledgements.append(
                        {"display_name": participant.display_name, **totals}
                    )
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
                    "target_assignments": int(self._state["target_steps"])
                    * int(self._state["worker_count"]),
                    "accepted_tokens": accepted_tokens,
                    "open_assignments": assignment_states.count("open"),
                    "leased_assignments": assignment_states.count("leased"),
                },
                "checkpoint": {
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
                        if int(self._state["completed_steps"]) > 0
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
