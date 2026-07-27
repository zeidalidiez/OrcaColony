from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .artifacts import PackedDataset
from .record_patch import CAMPAIGN_ID
from .record_patch_learnability import (
    _behavioral_evaluation,
    _canonical_json_bytes,
    _environment,
    _peak_rss_bytes,
    _relative_path,
    _sha256_file,
    _training_step,
    _write_checksums,
    _write_exact,
)
from .record_patch_learnability_analysis import _verify_checksums
from .reference import (
    CampaignConfig,
    TrainingResult,
    _load_checkpoint,
    _save_checkpoint,
    evaluate_checkpoint,
    load_campaign,
    validate_dataset_artifacts,
)


@dataclass(frozen=True)
class ContinuationTrainingResult:
    checkpoint_dirs: Mapping[int, Path]
    checkpoints: Mapping[int, TrainingResult]
    diagnostics: tuple[Mapping[str, object], ...]
    resume_step: int
    final_step: int
    wall_seconds: float


@dataclass(frozen=True)
class VerifiedParentRun:
    root: Path
    checkpoint_dir: Path
    evidence: Mapping[str, object]
    selected_behavioral_metrics: Mapping[str, object]


_PROTOCOL_FIELDS = frozenset(
    {
        "format",
        "id",
        "campaign",
        "dataset_revision",
        "behavioral_suite_revision",
        "evaluator_revision",
        "parent_run",
        "trajectory",
        "execution",
        "schedule",
        "selection",
        "gate",
        "resource_limits",
        "holdout_policy",
        "limitations",
    }
)
_SELECTION_POLICY = (
    "lowest public language-validation mean loss among continuation "
    "checkpoints; behavioral metrics do not select the checkpoint"
)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> Mapping[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _module_revision() -> str:
    return _sha256_file(Path(__file__).resolve())


def _protocol_object(
    value: object,
    *,
    label: str,
    fields: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"continuation protocol {label} is invalid")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"continuation protocol {label} must be a positive integer"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"continuation protocol {label} must be a nonnegative integer"
        )
    return value


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"continuation protocol {label} must be a finite number"
        )
    return float(value)


def _sha256_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"continuation protocol {label} is invalid")
    return value


def _sha256_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"continuation protocol {label} is invalid")
    _sha256_digest(value.removeprefix("sha256:"), label)
    return value


def _continuation_steps(
    values: Sequence[int],
    *,
    resume_step: int,
    label: str = "checkpoint",
) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"at least one {label} step is required")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{label} steps must be nonnegative integers")
    steps = tuple(values)
    if steps[0] != resume_step:
        raise ValueError(f"{label} steps must begin at the resume step")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError(f"{label} steps must be unique and increasing")
    return steps


def _load_continuation_protocol(
    path: str | Path,
    *,
    campaign: CampaignConfig,
    campaign_sha256: str,
    dataset_revision: str,
) -> Mapping[str, object]:
    protocol_path = Path(path).resolve()
    payload = _load_json(protocol_path, label="continuation protocol")
    if set(payload) != _PROTOCOL_FIELDS:
        raise ValueError("continuation protocol schema is invalid")
    if (
        payload.get("format")
        != "orcacolony_record_patch_continuation_protocol_v1"
    ):
        raise ValueError("unsupported continuation protocol format")
    if payload.get("id") != "record-patch-t2-centralized-continuation-v1":
        raise ValueError("continuation protocol identity is invalid")

    protocol_campaign = _protocol_object(
        payload.get("campaign"),
        label="campaign",
        fields={"id", "sha256"},
    )
    if protocol_campaign != {
        "id": CAMPAIGN_ID,
        "sha256": campaign_sha256,
    }:
        raise ValueError("continuation protocol campaign identity mismatch")
    if payload.get("dataset_revision") != dataset_revision:
        raise ValueError("continuation protocol dataset identity mismatch")
    if campaign.research is None:
        raise ValueError("continuation campaign lacks a research contract")
    behavioral_contract = _protocol_object(
        campaign.research.get("behavioral_evaluation"),
        label="campaign behavioral contract",
        fields={
            "suite_id",
            "dataset_revision",
            "evaluator_revision",
            "validation_split",
            "final_holdout_split",
        },
    )
    if (
        payload.get("behavioral_suite_revision")
        != behavioral_contract["dataset_revision"]
        or payload.get("evaluator_revision")
        != behavioral_contract["evaluator_revision"]
    ):
        raise ValueError("continuation protocol evaluator identity mismatch")

    parent_run = _protocol_object(
        payload.get("parent_run"),
        label="parent run",
        fields={
            "evidence_sha256",
            "format",
            "protocol",
            "runner_revision",
            "sha256sums_sha256",
            "source_revision",
        },
    )
    parent_protocol = _protocol_object(
        parent_run["protocol"],
        label="parent protocol",
        fields={"id", "sha256"},
    )
    if (
        parent_run["format"]
        != "orcacolony_record_patch_learnability_evidence_v1"
        or parent_protocol["id"]
        != "record-patch-t2-centralized-learnability-v1"
    ):
        raise ValueError("continuation parent run identity is invalid")
    for field in ("evidence_sha256", "sha256sums_sha256"):
        _sha256_digest(parent_run[field], f"parent {field}")
    _sha256_digest(
        parent_protocol["sha256"],
        "parent protocol revision",
    )
    _sha256_revision(
        parent_run["runner_revision"],
        "parent runner revision",
    )
    source_revision = parent_run["source_revision"]
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_revision
        )
    ):
        raise ValueError("continuation parent source revision is invalid")

    trajectory = _protocol_object(
        payload.get("trajectory"),
        label="trajectory",
        fields={
            "dataset_cursor",
            "learning_rate",
            "loss_mask",
            "model_sha256",
            "objective",
            "optimizer",
            "optimizer_sha256",
            "resume_step",
            "state_sha256",
        },
    )
    resume_step = _positive_int(
        trajectory["resume_step"],
        "resume step",
    )
    dataset_cursor = _nonnegative_int(
        trajectory["dataset_cursor"],
        "dataset cursor",
    )
    learning_rate = _finite_number(
        trajectory["learning_rate"],
        "learning rate",
    )
    for field in ("model_sha256", "optimizer_sha256", "state_sha256"):
        _sha256_digest(trajectory[field], f"trajectory {field}")
    expected_cursor = (
        resume_step * campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    if (
        dataset_cursor != expected_cursor
        or learning_rate != campaign.training.learning_rate
        or trajectory["objective"] != campaign.objective.name
        or trajectory["loss_mask"] != campaign.objective.loss_mask
        or trajectory["optimizer"] != "AdamW"
    ):
        raise ValueError("continuation trajectory differs from the campaign")

    execution = _protocol_object(
        payload.get("execution"),
        label="execution",
        fields={"donated_compute", "topology", "torch_threads"},
    )
    if execution != {
        "donated_compute": False,
        "topology": "owner-operated centralized CPU reference",
        "torch_threads": 1,
    }:
        raise ValueError("continuation execution profile is invalid")

    schedule = _protocol_object(
        payload.get("schedule"),
        label="schedule",
        fields={
            "behavioral_steps",
            "checkpoint_steps",
            "max_new_tokens",
        },
    )
    raw_checkpoints = schedule["checkpoint_steps"]
    raw_behavioral = schedule["behavioral_steps"]
    if not isinstance(raw_checkpoints, list):
        raise ValueError("continuation checkpoint schedule is invalid")
    if not isinstance(raw_behavioral, list):
        raise ValueError("continuation behavioral schedule is invalid")
    checkpoints = _continuation_steps(
        raw_checkpoints,
        resume_step=resume_step,
    )
    behavioral = _continuation_steps(
        raw_behavioral,
        resume_step=resume_step,
        label="behavioral",
    )
    if not set(behavioral).issubset(checkpoints):
        raise ValueError(
            "continuation behavioral steps must be checkpoints"
        )
    if checkpoints[-1] <= resume_step:
        raise ValueError(
            "continuation schedule must advance beyond the resume step"
        )
    max_new_tokens = _positive_int(
        schedule["max_new_tokens"],
        "generation budget",
    )
    if payload.get("selection") != _SELECTION_POLICY:
        raise ValueError("continuation selection policy is invalid")

    success_gate = (
        campaign.evaluation.get("success_gate")
        if campaign.evaluation is not None
        else None
    )
    if not isinstance(success_gate, Mapping):
        raise ValueError("campaign language success gate is invalid")
    gate = _protocol_object(
        payload.get("gate"),
        label="gate",
        fields={
            "initialization_behavioral_exact_match_count",
            "initialization_language_mean_loss",
            "minimum_behavioral_exact_match_count_improvement",
            "minimum_language_mean_loss_improvement",
        },
    )
    initial_exact_count = _nonnegative_int(
        gate["initialization_behavioral_exact_match_count"],
        "initial behavioral exact-match count",
    )
    initial_language_loss = _finite_number(
        gate["initialization_language_mean_loss"],
        "initial language mean loss",
    )
    exact_improvement = _positive_int(
        gate["minimum_behavioral_exact_match_count_improvement"],
        "behavioral improvement",
    )
    language_improvement = _finite_number(
        gate["minimum_language_mean_loss_improvement"],
        "language improvement",
    )
    if (
        initial_language_loss <= 0
        or language_improvement <= 0
        or language_improvement
        != float(success_gate["minimum_improvement_from_initialization"])
    ):
        raise ValueError("continuation language gate is invalid")

    resource_limits = _protocol_object(
        payload.get("resource_limits"),
        label="resource limits",
        fields={
            "maximum_continuation_steps",
            "maximum_peak_rss_bytes",
            "maximum_total_training_steps",
        },
    )
    maximum_continuation_steps = _positive_int(
        resource_limits["maximum_continuation_steps"],
        "maximum continuation steps",
    )
    maximum_peak_rss = _positive_int(
        resource_limits["maximum_peak_rss_bytes"],
        "maximum peak RSS",
    )
    maximum_total_steps = _positive_int(
        resource_limits["maximum_total_training_steps"],
        "maximum total training steps",
    )
    if (
        checkpoints[-1] > maximum_total_steps
        or checkpoints[-1] - resume_step > maximum_continuation_steps
        or maximum_total_steps > campaign.training.steps
    ):
        raise ValueError("continuation schedule exceeds its step limit")

    holdout_policy = _protocol_object(
        payload.get("holdout_policy"),
        label="holdout policy",
        fields={"behavioral_final_holdout", "language_final_holdout"},
    )
    if holdout_policy != {
        "behavioral_final_holdout": "must_not_open",
        "language_final_holdout": "must_not_evaluate",
    }:
        raise ValueError("continuation holdout policy is invalid")
    limitations = payload.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(item, str) or not item.strip()
            for item in limitations
        )
    ):
        raise ValueError("continuation protocol limitations are invalid")

    return {
        **dict(payload),
        "parent_run": {
            **dict(parent_run),
            "protocol": dict(parent_protocol),
        },
        "trajectory": {
            **dict(trajectory),
            "resume_step": resume_step,
            "dataset_cursor": dataset_cursor,
            "learning_rate": learning_rate,
        },
        "schedule": {
            "checkpoint_steps": checkpoints,
            "behavioral_steps": behavioral,
            "max_new_tokens": max_new_tokens,
        },
        "gate": {
            "initialization_behavioral_exact_match_count": (
                initial_exact_count
            ),
            "initialization_language_mean_loss": initial_language_loss,
            "minimum_behavioral_exact_match_count_improvement": (
                exact_improvement
            ),
            "minimum_language_mean_loss_improvement": language_improvement,
        },
        "resource_limits": {
            "maximum_continuation_steps": maximum_continuation_steps,
            "maximum_peak_rss_bytes": maximum_peak_rss,
            "maximum_total_training_steps": maximum_total_steps,
        },
        "_sha256": _sha256_file(protocol_path),
    }


def _checkpoint_record(
    evidence: Mapping[str, object],
    step: int,
) -> Mapping[str, object]:
    records = evidence.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("parent checkpoint evidence is invalid")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("step") == step
    ]
    if len(matches) != 1:
        raise ValueError("parent resume checkpoint evidence is invalid")
    return matches[0]


def _verify_parent_run(
    parent_run_dir: str | Path,
    *,
    protocol: Mapping[str, object],
    campaign_sha256: str,
    dataset_revision: str,
) -> VerifiedParentRun:
    root = Path(parent_run_dir).resolve()
    parent_contract = protocol["parent_run"]
    trajectory = protocol["trajectory"]
    gate_contract = protocol["gate"]
    if not isinstance(parent_contract, Mapping):
        raise ValueError("validated parent contract is invalid")
    if not isinstance(trajectory, Mapping):
        raise ValueError("validated trajectory contract is invalid")
    if not isinstance(gate_contract, Mapping):
        raise ValueError("validated gate contract is invalid")
    if _sha256_file(root / "SHA256SUMS") != parent_contract[
        "sha256sums_sha256"
    ]:
        raise ValueError("parent checksum-manifest digest mismatch")
    verified = _verify_checksums(root)
    if verified.get("evidence.json") != parent_contract["evidence_sha256"]:
        raise ValueError("parent evidence is absent from its checksum manifest")
    evidence_path = root / "evidence.json"
    if _sha256_file(evidence_path) != parent_contract["evidence_sha256"]:
        raise ValueError("parent evidence digest mismatch")
    evidence = _load_json(evidence_path, label="parent evidence")
    if (
        evidence.get("format") != parent_contract["format"]
        or evidence.get("campaign_id") != CAMPAIGN_ID
        or evidence.get("campaign_sha256") != campaign_sha256
        or evidence.get("dataset_revision") != dataset_revision
        or evidence.get("runner_revision")
        != parent_contract["runner_revision"]
        or evidence.get("protocol") != parent_contract["protocol"]
    ):
        raise ValueError("parent evidence identity mismatch")
    holdout = evidence.get("holdout")
    if holdout != {
        "behavioral_final_holdout_opened": False,
        "language_final_holdout_evaluated": False,
    }:
        raise ValueError("parent evidence opened a reserved holdout")
    selection = evidence.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("step") != trajectory["resume_step"]
        or selection.get("model_sha256") != trajectory["model_sha256"]
    ):
        raise ValueError("parent selected checkpoint identity mismatch")
    parent_gate = evidence.get("learnability_gate")
    if (
        not isinstance(parent_gate, Mapping)
        or parent_gate.get("passed") is not False
        or float(parent_gate.get("initial_language_mean_loss", math.nan))
        != gate_contract["initialization_language_mean_loss"]
    ):
        raise ValueError("parent learnability gate evidence is invalid")
    initial_metrics = parent_gate.get("initial_behavioral_metrics")
    if (
        not isinstance(initial_metrics, Mapping)
        or initial_metrics.get("record_exact_match_count")
        != gate_contract["initialization_behavioral_exact_match_count"]
    ):
        raise ValueError("parent initialization metrics differ from protocol")

    resume_step = int(trajectory["resume_step"])
    record = _checkpoint_record(evidence, resume_step)
    behavioral = record.get("behavioral_evaluation")
    if (
        record.get("model_sha256") != trajectory["model_sha256"]
        or record.get("checkpoint_state_sha256")
        != trajectory["state_sha256"]
        or not isinstance(behavioral, Mapping)
        or not isinstance(behavioral.get("metrics"), Mapping)
    ):
        raise ValueError("parent resume checkpoint record is invalid")
    checkpoint_dir = root / "checkpoints" / f"step-{resume_step:08d}"
    required_files = {
        f"checkpoints/step-{resume_step:08d}/model.safetensors": (
            trajectory["model_sha256"]
        ),
        f"checkpoints/step-{resume_step:08d}/optimizer.safetensors": (
            trajectory["optimizer_sha256"]
        ),
        f"checkpoints/step-{resume_step:08d}/state.json": (
            trajectory["state_sha256"]
        ),
    }
    if any(
        verified.get(relative) != digest
        for relative, digest in required_files.items()
    ):
        raise ValueError("parent resume artifacts differ from protocol")
    state = _load_json(
        checkpoint_dir / "state.json",
        label="parent checkpoint state",
    )
    if (
        state.get("step") != resume_step
        or state.get("optimizer_step") != resume_step
        or state.get("dataset_cursor") != trajectory["dataset_cursor"]
        or state.get("dataset_revision") != dataset_revision
        or not isinstance(state.get("model"), Mapping)
        or state["model"].get("sha256") != trajectory["model_sha256"]
        or not isinstance(state.get("optimizer"), Mapping)
        or state["optimizer"].get("sha256")
        != trajectory["optimizer_sha256"]
    ):
        raise ValueError("parent checkpoint trajectory mismatch")

    environment = _load_json(
        root / "environment.json",
        label="parent environment",
    )
    source = environment.get("source")
    if (
        verified.get("environment.json")
        != _sha256_file(root / "environment.json")
        or not isinstance(source, Mapping)
        or source.get("commit") != parent_contract["source_revision"]
        or source.get("dirty") is not False
    ):
        raise ValueError("parent source provenance is invalid")
    return VerifiedParentRun(
        root=root,
        checkpoint_dir=checkpoint_dir,
        evidence=evidence,
        selected_behavioral_metrics=behavioral["metrics"],
    )


def run_continuation_training(
    *,
    campaign: CampaignConfig,
    resume_from: str | Path,
    output_dir: str | Path,
    checkpoint_steps: Sequence[int],
    dataset: PackedDataset | None = None,
    maximum_peak_rss_bytes: int | None = None,
) -> ContinuationTrainingResult:
    validate_dataset_artifacts(campaign, dataset)
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("continuation training output directory is not empty")
    model, optimizer, step, dataset_cursor, loss_history = _load_checkpoint(
        campaign,
        Path(resume_from).resolve(),
    )
    steps = _continuation_steps(checkpoint_steps, resume_step=step)
    if steps[-1] > campaign.training.steps:
        raise ValueError("continuation schedule exceeds campaign step budget")
    output_root.mkdir(parents=True, exist_ok=True)
    model.train()
    resume_step = step
    diagnostics: list[Mapping[str, object]] = []
    checkpoint_dirs: dict[int, Path] = {}
    checkpoints: dict[int, TrainingResult] = {}
    started = time.perf_counter()

    for milestone in steps:
        while step < milestone:
            mean_loss, dataset_cursor, step_diagnostics = _training_step(
                campaign,
                model,
                optimizer,
                dataset,
                step=step,
                dataset_cursor=dataset_cursor,
            )
            loss_history.append(mean_loss)
            diagnostics.append(step_diagnostics)
            step += 1
            peak_rss = _peak_rss_bytes()
            if (
                maximum_peak_rss_bytes is not None
                and peak_rss is not None
                and peak_rss > maximum_peak_rss_bytes
            ):
                raise RuntimeError(
                    "continuation training exceeded its peak RSS limit"
                )
        checkpoint_dir = output_root / f"step-{step:08d}"
        checkpoint_dirs[step] = checkpoint_dir
        checkpoints[step] = _save_checkpoint(
            campaign,
            model,
            optimizer,
            checkpoint_dir,
            step,
            dataset_cursor,
            loss_history,
        )
        peak_rss = _peak_rss_bytes()
        if (
            maximum_peak_rss_bytes is not None
            and peak_rss is not None
            and peak_rss > maximum_peak_rss_bytes
        ):
            raise RuntimeError(
                "continuation checkpointing exceeded its peak RSS limit"
            )

    return ContinuationTrainingResult(
        checkpoint_dirs=checkpoint_dirs,
        checkpoints=checkpoints,
        diagnostics=tuple(diagnostics),
        resume_step=resume_step,
        final_step=step,
        wall_seconds=time.perf_counter() - started,
    )


def _continuation_markdown(evidence: Mapping[str, object]) -> bytes:
    selection = evidence["selection"]
    gate = evidence["learnability_gate"]
    resources = evidence["resources"]
    if not isinstance(selection, Mapping):
        raise ValueError("continuation selection evidence is invalid")
    if not isinstance(gate, Mapping):
        raise ValueError("continuation gate evidence is invalid")
    if not isinstance(resources, Mapping):
        raise ValueError("continuation resource evidence is invalid")
    selected_metrics = gate["selected_behavioral_metrics"]
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("selected behavioral metrics are invalid")
    disposition = "passed" if gate["passed"] is True else "did not pass"
    text = (
        "# Record Patch same-trajectory continuation\n\n"
        f"The bounded public-data gate **{disposition}**. "
        "This is a pre-volunteer qualification result, not model "
        "promotion.\n\n"
        f"- Resumed checkpoint: step `{resources['resume_step']}`\n"
        f"- Final trained checkpoint: step `{resources['final_step']}`\n"
        f"- Language-selected checkpoint: step `{selection['step']}`\n"
        f"- Initialization language loss: "
        f"`{gate['initial_language_mean_loss']}`\n"
        f"- Selected language loss: "
        f"`{gate['selected_language_mean_loss']}`\n"
        f"- Language-loss improvement: "
        f"`{gate['language_mean_loss_improvement']}`\n"
        f"- Selected exact matches: "
        f"`{selected_metrics['record_exact_match_count']}/"
        f"{selected_metrics['examples']}`\n"
        f"- Exact-match gain over initialization: "
        f"`{gate['behavioral_exact_match_count_improvement']}` examples\n"
        f"- Continuation wall time: "
        f"`{resources['continuation_training_wall_seconds']}` seconds\n"
        f"- Peak process RSS: `{resources['peak_rss_bytes']}` bytes\n\n"
        "Checkpoint selection used only public language-validation loss. "
        "Neither reserved final holdout was opened or accepted as a command "
        "argument. Public behavioral validation is diagnostic evidence only.\n\n"
        "The exact parent run and resume artifacts are bound by digest. "
        "`SHA256SUMS` covers the self-contained continuation checkpoints, "
        "evaluations, diagnostics, environment, lock, and evidence.\n"
    )
    return text.encode("utf-8")


def run_continuation_check(
    *,
    protocol_path: str | Path,
    campaign_path: str | Path,
    packed_dir: str | Path,
    public_dir: str | Path,
    parent_run_dir: str | Path,
    output_dir: str | Path,
) -> Mapping[str, object]:
    run_started = time.perf_counter()
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("continuation output directory is not empty")
    campaign_file = Path(campaign_path).resolve()
    packed_root = Path(packed_dir).resolve()
    public_root = Path(public_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    campaign = load_campaign(campaign_file)
    if campaign.campaign.get("id") != CAMPAIGN_ID:
        raise ValueError("continuation campaign identity is invalid")
    dataset = PackedDataset.load(packed_root)
    validate_dataset_artifacts(campaign, dataset)
    campaign_sha256 = _sha256_file(campaign_file)
    protocol = _load_continuation_protocol(
        protocol_path,
        campaign=campaign,
        campaign_sha256=campaign_sha256,
        dataset_revision=dataset.revision,
    )
    parent = _verify_parent_run(
        parent_run_dir,
        protocol=protocol,
        campaign_sha256=campaign_sha256,
        dataset_revision=dataset.revision,
    )
    schedule = protocol["schedule"]
    resource_limits = protocol["resource_limits"]
    trajectory = protocol["trajectory"]
    if not isinstance(schedule, Mapping):
        raise ValueError("validated continuation schedule is invalid")
    if not isinstance(resource_limits, Mapping):
        raise ValueError("validated continuation resource limits are invalid")
    if not isinstance(trajectory, Mapping):
        raise ValueError("validated continuation trajectory is invalid")
    steps = tuple(schedule["checkpoint_steps"])
    behavior_steps = tuple(schedule["behavioral_steps"])
    max_new_tokens = int(schedule["max_new_tokens"])
    maximum_peak_rss_bytes = int(
        resource_limits["maximum_peak_rss_bytes"]
    )
    torch.set_num_threads(1)
    environment = _environment(project_root)
    source = environment.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("commit") is None
        or source.get("dirty") is not False
    ):
        raise ValueError(
            "continuation requires a clean committed source revision"
        )
    if environment.get("torch_num_threads") != 1:
        raise ValueError("continuation requires exactly one torch thread")

    output_root.mkdir(parents=True, exist_ok=True)
    _write_exact(
        output_root / "environment.json",
        _canonical_json_bytes(environment),
    )
    lock = {
        "format": "orcacolony_record_patch_continuation_lock_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": campaign_sha256,
        "dataset_revision": dataset.revision,
        "runner_revision": f"sha256:{_module_revision()}",
        "protocol": {
            "id": protocol["id"],
            "sha256": protocol["_sha256"],
        },
        "parent_run": {
            "evidence_sha256": protocol["parent_run"]["evidence_sha256"],
            "sha256sums_sha256": (
                protocol["parent_run"]["sha256sums_sha256"]
            ),
            "resume_step": trajectory["resume_step"],
            "model_sha256": trajectory["model_sha256"],
            "optimizer_sha256": trajectory["optimizer_sha256"],
            "state_sha256": trajectory["state_sha256"],
        },
        "checkpoint_steps": list(steps),
        "requested_behavioral_steps": list(behavior_steps),
        "decoding": {
            "method": "greedy",
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": "first newline or EOS",
        },
        "data_access": {
            "training": "packed training split",
            "checkpoint_selection": "public language-validation slice",
            "behavioral_diagnostics": "public behavioral-validation split",
            "behavioral_final_holdout": "not opened",
            "language_final_holdout": "not evaluated",
        },
    }
    _write_exact(output_root / "run-lock.json", _canonical_json_bytes(lock))

    training = run_continuation_training(
        campaign=campaign,
        resume_from=parent.checkpoint_dir,
        output_dir=output_root / "checkpoints",
        checkpoint_steps=steps,
        dataset=dataset,
        maximum_peak_rss_bytes=maximum_peak_rss_bytes,
    )
    resume_step = training.resume_step
    resume_dir = training.checkpoint_dirs[resume_step]
    if (
        training.checkpoints[resume_step].model_sha256
        != trajectory["model_sha256"]
        or _sha256_file(resume_dir / "optimizer.safetensors")
        != trajectory["optimizer_sha256"]
        or _sha256_file(resume_dir / "state.json")
        != trajectory["state_sha256"]
    ):
        raise ValueError(
            "self-contained resume checkpoint differs from parent"
        )
    diagnostics = list(training.diagnostics)
    expected_diagnostic_steps = list(
        range(resume_step + 1, training.final_step + 1)
    )
    if [item["step"] for item in diagnostics] != expected_diagnostic_steps:
        raise ValueError("continuation diagnostics are not contiguous")
    _write_exact(
        output_root / "training-diagnostics.json",
        _canonical_json_bytes(diagnostics),
    )

    language: dict[int, Mapping[str, object]] = {}
    for step in steps:
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            campaign,
            training.checkpoint_dirs[step],
            dataset,
        )
        path = (
            output_root
            / "evaluations"
            / f"step-{step:08d}"
            / "language.json"
        )
        _write_exact(path, _canonical_json_bytes(evaluation))
        language[step] = {
            "path": path,
            "sha256": _sha256_file(path),
            "evaluation": evaluation,
            "wall_seconds": time.perf_counter() - evaluation_started,
        }
        peak_rss = _peak_rss_bytes()
        if peak_rss is not None and peak_rss > maximum_peak_rss_bytes:
            raise RuntimeError(
                "continuation language evaluation exceeded peak RSS limit"
            )
        gc.collect()

    selected_step = min(
        steps,
        key=lambda step: (
            float(language[step]["evaluation"]["mean_loss"]),
            step,
        ),
    )
    behavioral: dict[int, Mapping[str, object]] = {}
    for step in behavior_steps:
        behavioral[step] = _behavioral_evaluation(
            campaign=campaign,
            checkpoint_dir=training.checkpoint_dirs[step],
            packed_dir=packed_root,
            public_dir=public_root,
            output_dir=(
                output_root
                / "evaluations"
                / f"step-{step:08d}"
                / "behavioral"
            ),
            max_new_tokens=max_new_tokens,
        )
        peak_rss = _peak_rss_bytes()
        if peak_rss is not None and peak_rss > maximum_peak_rss_bytes:
            raise RuntimeError(
                "continuation behavioral evaluation exceeded peak RSS limit"
            )
    if selected_step not in behavioral:
        raise ValueError(
            "language-selected checkpoint lacks behavioral diagnostics"
        )

    gate_contract = protocol["gate"]
    if not isinstance(gate_contract, Mapping):
        raise ValueError("validated continuation gate is invalid")
    initial_language_loss = float(
        gate_contract["initialization_language_mean_loss"]
    )
    initial_exact_count = int(
        gate_contract["initialization_behavioral_exact_match_count"]
    )
    selected_language_loss = float(
        language[selected_step]["evaluation"]["mean_loss"]
    )
    selected_metrics = behavioral[selected_step]["metrics"]
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("selected behavioral metrics are invalid")
    language_improvement = (
        initial_language_loss - selected_language_loss
    )
    exact_count_improvement = (
        int(selected_metrics["record_exact_match_count"])
        - initial_exact_count
    )
    minimum_language_improvement = float(
        gate_contract["minimum_language_mean_loss_improvement"]
    )
    minimum_exact_improvement = int(
        gate_contract[
            "minimum_behavioral_exact_match_count_improvement"
        ]
    )
    language_passed = (
        language_improvement >= minimum_language_improvement
    )
    behavioral_passed = (
        exact_count_improvement >= minimum_exact_improvement
    )
    peak_rss_bytes = _peak_rss_bytes()
    resource_limits_passed = (
        peak_rss_bytes is None
        or peak_rss_bytes <= maximum_peak_rss_bytes
    )
    gate = {
        "policy": (
            "lowest public language-validation mean loss among continuation "
            "checkpoints, followed by at least the protocol's additional "
            "exact public behavioral-validation matches"
        ),
        "passed": (
            language_passed
            and behavioral_passed
            and resource_limits_passed
        ),
        "language_passed": language_passed,
        "behavioral_passed": behavioral_passed,
        "resource_limits_passed": resource_limits_passed,
        "minimum_language_mean_loss_improvement": (
            minimum_language_improvement
        ),
        "minimum_behavioral_exact_match_count_improvement": (
            minimum_exact_improvement
        ),
        "initial_language_mean_loss": initial_language_loss,
        "selected_language_mean_loss": selected_language_loss,
        "language_mean_loss_improvement": language_improvement,
        "initial_behavioral_exact_match_count": initial_exact_count,
        "selected_behavioral_metrics": selected_metrics,
        "behavioral_exact_match_count_improvement": (
            exact_count_improvement
        ),
        "scope": (
            "pre-volunteer public-data qualification; not a promotion "
            "decision"
        ),
    }
    language_evaluation_wall_seconds = sum(
        float(language[step]["wall_seconds"]) for step in steps
    )
    behavioral_evaluation_wall_seconds = sum(
        float(behavioral[step]["wall_seconds"])
        for step in behavior_steps
    )
    continuation_loss_weight_sum = sum(
        int(item["loss_weight_sum"]) for item in diagnostics
    )
    parent_resources = parent.evidence.get("resources")
    if not isinstance(parent_resources, Mapping):
        raise ValueError("parent resource evidence is invalid")
    resources = {
        "resume_step": resume_step,
        "final_step": training.final_step,
        "continuation_steps_completed": training.final_step - resume_step,
        "continuation_training_wall_seconds": training.wall_seconds,
        "language_evaluation_wall_seconds": (
            language_evaluation_wall_seconds
        ),
        "behavioral_evaluation_wall_seconds": (
            behavioral_evaluation_wall_seconds
        ),
        "total_wall_seconds_before_packaging": (
            time.perf_counter() - run_started
        ),
        "continuation_step_wall_seconds": sum(
            float(item["wall_seconds"]) for item in diagnostics
        ),
        "continuation_loss_weight_sum": continuation_loss_weight_sum,
        "total_trajectory_loss_weight_sum": (
            int(parent_resources["loss_weight_sum"])
            + continuation_loss_weight_sum
        ),
        "clipped_continuation_steps": sum(
            int(item["clipped"] is True) for item in diagnostics
        ),
        "peak_rss_bytes": peak_rss_bytes,
        "maximum_peak_rss_bytes": maximum_peak_rss_bytes,
        "execution": "owner-operated centralized CPU reference",
        "donated_compute": False,
    }
    checkpoint_records = [
        {
            "step": step,
            "model_sha256": training.checkpoints[step].model_sha256,
            "checkpoint_state_sha256": _sha256_file(
                training.checkpoint_dirs[step] / "state.json"
            ),
            "optimizer_sha256": _sha256_file(
                training.checkpoint_dirs[step] / "optimizer.safetensors"
            ),
            "language_evaluation": {
                "path": _relative_path(
                    language[step]["path"],
                    output_root,
                ),
                "sha256": language[step]["sha256"],
                "mean_loss": language[step]["evaluation"]["mean_loss"],
                "perplexity": language[step]["evaluation"]["perplexity"],
                "wall_seconds": language[step]["wall_seconds"],
            },
            "behavioral_evaluation": {
                "predictions_path": _relative_path(
                    behavioral[step]["predictions"],
                    output_root,
                ),
                "predictions_sha256": behavioral[step][
                    "predictions_sha256"
                ],
                "evaluation_path": _relative_path(
                    behavioral[step]["evaluation"],
                    output_root,
                ),
                "evaluation_sha256": behavioral[step][
                    "evaluation_sha256"
                ],
                "metrics": behavioral[step]["metrics"],
                "guardrails": behavioral[step]["guardrails"],
                "wall_seconds": behavioral[step]["wall_seconds"],
            },
        }
        for step in steps
    ]
    evidence = {
        "format": "orcacolony_record_patch_continuation_evidence_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": campaign_sha256,
        "dataset_revision": dataset.revision,
        "runner_revision": f"sha256:{_module_revision()}",
        "protocol": {
            "id": protocol["id"],
            "sha256": protocol["_sha256"],
        },
        "parent_run": {
            "evidence_sha256": protocol["parent_run"]["evidence_sha256"],
            "sha256sums_sha256": (
                protocol["parent_run"]["sha256sums_sha256"]
            ),
            "source_revision": protocol["parent_run"]["source_revision"],
            "resume_step": resume_step,
            "model_sha256": trajectory["model_sha256"],
            "optimizer_sha256": trajectory["optimizer_sha256"],
            "state_sha256": trajectory["state_sha256"],
            "behavioral_metrics": parent.selected_behavioral_metrics,
        },
        "checkpoint_selection_policy": _SELECTION_POLICY,
        "selection": {
            "step": selected_step,
            "model_sha256": training.checkpoints[
                selected_step
            ].model_sha256,
        },
        "learnability_gate": gate,
        "resources": resources,
        "checkpoints": checkpoint_records,
        "diagnostics": {
            "path": "training-diagnostics.json",
            "sha256": _sha256_file(
                output_root / "training-diagnostics.json"
            ),
            "steps": len(diagnostics),
            "first_step": diagnostics[0]["step"],
            "last_step": diagnostics[-1]["step"],
        },
        "environment": {
            "path": "environment.json",
            "sha256": _sha256_file(output_root / "environment.json"),
        },
        "holdout": {
            "behavioral_final_holdout_opened": False,
            "language_final_holdout_evaluated": False,
        },
        "limitations": [
            *protocol["limitations"],
            (
                "A passed continuation gate permits planning volunteer "
                "training; it does not classify or promote a capability "
                "model."
            ),
        ],
    }
    _write_exact(output_root / "evidence.json", _canonical_json_bytes(evidence))
    _write_exact(
        output_root / "CONTINUATION.md",
        _continuation_markdown(evidence),
    )
    _write_checksums(output_root)
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the exact Record Patch step-128 qualification trajectory"
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evidence = run_continuation_check(
        protocol_path=args.protocol,
        campaign_path=args.campaign,
        packed_dir=args.packed_dir,
        public_dir=args.public_dir,
        parent_run_dir=args.parent_run,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_checkpoint": evidence["selection"],
                "learnability_gate": evidence["learnability_gate"],
                "holdout": evidence["holdout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
