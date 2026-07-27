from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Mapping

from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file

from .artifacts import PackedDataset
from .campaign_run import CampaignCoordinator
from .multiworker import (
    _campaign_payload,
    _reject_duplicate_json_keys,
    _revision,
    normalize_http_origin,
)
from .participants import (
    attribution_markdown,
    build_attribution_snapshot,
    load_participants,
)
from .peft import (
    BURN_NDARRAY_F32_PROFILE,
    BURN_WEBGPU_F32_PROFILE,
    EXACT_CPU_FP32_PROFILE,
    INT8_FROZEN_LINEAR_PROFILE,
    LoadedLoRAManifest,
    evaluate_lora_final_holdout,
    load_lora_checkpoint,
    load_lora_manifest,
)
from .reference import (
    CampaignConfig,
    evaluate_final_holdout,
    load_campaign,
    tensor_sha256,
)


_DATASET_FILES = (
    "manifest.json",
    "tokenizer.json",
    "train.safetensors",
    "validation.safetensors",
    "DATASET-NOTICE.md",
)

_PUBLIC_DATASET_MANIFEST_FIELDS = {
    "files",
    "format",
    "packing",
    "source",
    "subsets",
    "tokenizer",
}
_PUBLIC_DATASET_SOURCE_FIELDS = {
    "dataset",
    "dataset_card",
    "license",
    "license_url",
    "revision",
    "selection",
    "train",
    "validation",
}


def _reject_unknown_fields(
    payload: Mapping[str, object],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unpublished fields: {', '.join(unknown)}")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_pinned_revision(value: object) -> bool:
    if not isinstance(value, str):
        return False
    is_sha256 = value.startswith("sha256:")
    digest = value.removeprefix("sha256:") if is_sha256 else value
    expected_length = 64 if is_sha256 else 40
    return len(digest) == expected_length and all(
        character in "0123456789abcdef" for character in digest
    )


def _require_nonnegative_ints(
    payload: Mapping[str, object],
    fields: set[str],
    label: str,
) -> None:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} {field} must be a nonnegative integer")


def validate_public_dataset_manifest(manifest: Mapping[str, object]) -> None:
    _reject_unknown_fields(
        manifest,
        _PUBLIC_DATASET_MANIFEST_FIELDS,
        "dataset manifest",
    )
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("dataset manifest source must be a mapping")
    _reject_unknown_fields(source, _PUBLIC_DATASET_SOURCE_FIELDS, "dataset source")
    for field in _PUBLIC_DATASET_SOURCE_FIELDS - {"train", "validation"}:
        if field in source and not isinstance(source[field], str):
            raise ValueError(f"dataset source {field} must be text")
    for subset_name in ("train", "validation"):
        subset = source.get(subset_name)
        if subset is None:
            continue
        if not isinstance(subset, Mapping):
            raise ValueError(f"dataset source {subset_name} must be a mapping")
        _reject_unknown_fields(
            subset,
            {"file", "sha256", "size"},
            f"dataset source {subset_name}",
        )
        if not isinstance(subset.get("file"), str) or not isinstance(
            subset.get("sha256"), str
        ):
            raise ValueError(f"dataset source {subset_name} file identity is invalid")
        if not _is_sha256(subset["sha256"]):
            raise ValueError(f"dataset source {subset_name} digest is invalid")
        if (
            isinstance(subset.get("size"), bool)
            or not isinstance(subset.get("size"), int)
            or subset["size"] < 0
        ):
            raise ValueError(f"dataset source {subset_name} size is invalid")

    files = manifest.get("files")
    expected_files = set(_DATASET_FILES) - {"manifest.json"}
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ValueError("dataset manifest public file set is invalid")
    if any(not _is_sha256(value) for value in files.values()):
        raise ValueError("dataset manifest contains an invalid public file digest")
    packing = manifest.get("packing")
    if not isinstance(packing, Mapping):
        raise ValueError("dataset manifest packing must be a mapping")
    _reject_unknown_fields(
        packing,
        {
            "context_length",
            "dtype",
            "stride",
            "train_sequences",
            "train_tokens",
            "validation_sequences",
            "validation_tokens",
        },
        "dataset packing",
    )
    if packing.get("dtype") != "int32":
        raise ValueError("dataset packing dtype must be int32")
    _require_nonnegative_ints(
        packing,
        {
            "context_length",
            "stride",
            "train_sequences",
            "train_tokens",
            "validation_sequences",
            "validation_tokens",
        },
        "dataset packing",
    )
    subsets = manifest.get("subsets")
    if not isinstance(subsets, Mapping):
        raise ValueError("dataset manifest subsets must be a mapping")
    _reject_unknown_fields(subsets, {"train", "validation"}, "dataset subsets")
    if set(subsets) != {"train", "validation"}:
        raise ValueError("dataset subsets must contain train and validation")
    for subset_name, subset in subsets.items():
        if not isinstance(subset, Mapping):
            raise ValueError(f"dataset subset {subset_name} must be a mapping")
        _reject_unknown_fields(
            subset,
            {
                "download_sha256",
                "downloaded_bytes",
                "stories",
                "used_bytes",
                "used_sha256",
            },
            f"dataset subset {subset_name}",
        )
        if set(subset) != {
            "download_sha256",
            "downloaded_bytes",
            "stories",
            "used_bytes",
            "used_sha256",
        }:
            raise ValueError(f"dataset subset {subset_name} is incomplete")
        if not _is_sha256(subset["download_sha256"]) or not _is_sha256(
            subset["used_sha256"]
        ):
            raise ValueError(f"dataset subset {subset_name} digest is invalid")
        _require_nonnegative_ints(
            subset,
            {"downloaded_bytes", "stories", "used_bytes"},
            f"dataset subset {subset_name}",
        )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("dataset manifest tokenizer must be a mapping")
    _reject_unknown_fields(
        tokenizer,
        {
            "format",
            "requested_vocab_size",
            "sha256",
            "special_token_ids",
            "vocab_size",
        },
        "dataset tokenizer",
    )
    if tokenizer.get("format") != "byte_level_bpe" or not _is_sha256(
        tokenizer.get("sha256")
    ):
        raise ValueError("dataset tokenizer identity is invalid")
    _require_nonnegative_ints(
        tokenizer,
        {"requested_vocab_size", "vocab_size"},
        "dataset tokenizer",
    )
    special_token_ids = tokenizer.get("special_token_ids")
    if not isinstance(special_token_ids, Mapping):
        raise ValueError("dataset tokenizer special-token ids must be a mapping")
    _reject_unknown_fields(
        special_token_ids,
        {"<bos>", "<eos>", "<pad>", "<unk>"},
        "dataset tokenizer special-token ids",
    )
    if set(special_token_ids) != {"<bos>", "<eos>", "<pad>", "<unk>"}:
        raise ValueError("dataset tokenizer special-token ids are incomplete")
    _require_nonnegative_ints(
        special_token_ids,
        {"<bos>", "<eos>", "<pad>", "<unk>"},
        "dataset tokenizer special-token ids",
    )


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload), encoding="utf-8", newline="\n")


def _copy_public_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"release input must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _validate_promotion_evidence(
    campaign: CampaignConfig,
    evidence: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    dataset_revision: str,
) -> bool:
    if campaign.research is None:
        raise ValueError("promotion evidence requires a capability research contract")
    required = {
        "format",
        "campaign_id",
        "checkpoint_sha256",
        "dataset_revision",
        "evaluation_suite",
        "primary_metric",
        "guardrails",
        "limitations",
        "artifacts",
        "reproduction",
    }
    if set(evidence) != required:
        raise ValueError("capability promotion evidence schema is invalid")
    if (
        evidence.get("format")
        != "orcacolony_capability_promotion_evidence_v1"
        or evidence.get("campaign_id") != campaign.campaign["id"]
        or evidence.get("checkpoint_sha256") != checkpoint_sha256
        or evidence.get("dataset_revision") != dataset_revision
    ):
        raise ValueError("capability promotion evidence identity is invalid")
    declared_suite = campaign.research.get("behavioral_evaluation")
    observed_suite = evidence.get("evaluation_suite")
    if not isinstance(declared_suite, Mapping) or not isinstance(
        observed_suite,
        Mapping,
    ):
        raise ValueError("capability evaluation-suite evidence is invalid")
    if set(observed_suite) != {
        "suite_id",
        "dataset_revision",
        "evaluator_revision",
        "split",
    }:
        raise ValueError("capability evaluation-suite evidence schema is invalid")
    expected_suite = {
        "suite_id": declared_suite.get("suite_id"),
        "dataset_revision": declared_suite.get("dataset_revision"),
        "evaluator_revision": declared_suite.get("evaluator_revision"),
        "split": declared_suite.get("final_holdout_split"),
    }
    if (
        dict(observed_suite) != expected_suite
        or not _is_pinned_revision(observed_suite.get("dataset_revision"))
        or not _is_pinned_revision(observed_suite.get("evaluator_revision"))
    ):
        raise ValueError("capability evaluation-suite identity differs")
    declared_metric = campaign.research.get("primary_metric")
    observed_metric = evidence.get("primary_metric")
    if not isinstance(declared_metric, Mapping) or not isinstance(
        observed_metric,
        Mapping,
    ):
        raise ValueError("capability primary-metric evidence is invalid")
    if set(observed_metric) != {"id", "value", "baseline"}:
        raise ValueError("capability primary-metric evidence schema is invalid")
    if observed_metric.get("id") != declared_metric.get("id"):
        raise ValueError("capability primary-metric evidence ID differs")
    value = observed_metric.get("value")
    declared_baseline = campaign.research.get("baseline")
    observed_baseline = observed_metric.get("baseline")
    if not isinstance(declared_baseline, Mapping) or not isinstance(
        observed_baseline,
        Mapping,
    ) or set(observed_baseline) != {"id", "revision", "value"}:
        raise ValueError("capability baseline evidence is invalid")
    if (
        observed_baseline.get("id") != declared_baseline.get("id")
        or observed_baseline.get("revision")
        != declared_baseline.get("revision")
        or not _is_pinned_revision(observed_baseline.get("revision"))
    ):
        raise ValueError("capability baseline evidence identity differs")
    baseline_value = observed_baseline.get("value")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in (value, baseline_value)
    ):
        raise ValueError("capability primary-metric values must be finite")
    threshold = float(declared_metric["success_threshold"])
    direction = declared_metric["direction"]
    primary_passed = (
        float(value) >= threshold
        if direction == "maximize"
        else float(value) <= threshold
    )
    observed_improvement = (
        float(value) - float(baseline_value)
        if direction == "maximize"
        else float(baseline_value) - float(value)
    )
    minimum_improvement = float(
        declared_metric["minimum_improvement_from_baseline"]
    )
    primary_passed = (
        primary_passed and observed_improvement >= minimum_improvement
    )

    declared_guardrails = campaign.research.get("guardrails")
    observed_guardrails = evidence.get("guardrails")
    if not isinstance(declared_guardrails, list) or not isinstance(
        observed_guardrails,
        list,
    ):
        raise ValueError("capability guardrail evidence is invalid")
    expected_ids = {
        guardrail["id"]
        for guardrail in declared_guardrails
        if isinstance(guardrail, Mapping)
    }
    observed_ids: set[object] = set()
    guardrails_passed = True
    for guardrail in observed_guardrails:
        if not isinstance(guardrail, Mapping) or set(guardrail) != {
            "id",
            "passed",
            "detail",
        }:
            raise ValueError("capability guardrail evidence schema is invalid")
        identifier = guardrail.get("id")
        if (
            not isinstance(identifier, str)
            or identifier in observed_ids
            or type(guardrail.get("passed")) is not bool
            or not isinstance(guardrail.get("detail"), str)
            or not str(guardrail["detail"]).strip()
        ):
            raise ValueError("capability guardrail evidence is invalid")
        observed_ids.add(identifier)
        guardrails_passed = guardrails_passed and bool(guardrail["passed"])
    if observed_ids != expected_ids:
        raise ValueError("capability guardrail evidence is incomplete")
    limitations = evidence.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(item, str) or not item.strip()
            for item in limitations
        )
    ):
        raise ValueError("capability promotion limitations are invalid")
    artifacts = evidence.get("artifacts")
    artifact_ids: set[str] = set()
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("capability promotion artifacts are required")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "id",
            "sha256",
            "uri",
        }:
            raise ValueError("capability promotion artifact schema is invalid")
        identifier = artifact.get("id")
        uri = artifact.get("uri")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in artifact_ids
            or not _is_sha256(artifact.get("sha256"))
            or not isinstance(uri, str)
            or not uri.strip()
            or any(character in uri for character in ("\x00", "\r", "\n"))
        ):
            raise ValueError("capability promotion artifact is invalid")
        artifact_ids.add(identifier)
    reproduction = evidence.get("reproduction")
    if not isinstance(reproduction, Mapping) or set(reproduction) != {
        "command",
        "notes",
    }:
        raise ValueError("capability promotion reproduction schema is invalid")
    command = reproduction.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(argument, str) or not argument.strip()
            for argument in command
        )
        or not isinstance(reproduction.get("notes"), str)
        or not str(reproduction["notes"]).strip()
    ):
        raise ValueError("capability promotion reproduction is invalid")
    return primary_passed and guardrails_passed


def _write_owned_checkpoint(
    artifacts: Mapping[str, bytes],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
            or type(payload) is not bytes
        ):
            raise ValueError("owned checkpoint artifact is invalid")
        (destination / name).write_bytes(payload)


def _select_checkpoint(
    coordinator: CampaignCoordinator,
    dashboard: Mapping[str, object],
) -> tuple[int, Mapping[str, object] | None]:
    evaluations = dashboard["evaluations"]
    selected_evaluation: Mapping[str, object] | None = None
    if evaluations:
        selected_evaluation = min(
            evaluations,
            key=lambda entry: (entry["mean_loss"], entry["step"]),
        )
        step = selected_evaluation["step"]
    else:
        step = dashboard["progress"]["completed_steps"]
    if type(step) is not int or step < 0:
        raise ValueError("selected checkpoint step is invalid")
    coordinator.versioned_checkpoint_artifacts(step)
    return step, selected_evaluation


def _validate_checkpoint_for_release(
    campaign: CampaignConfig,
    checkpoint: Path,
    state: Mapping[str, object],
    step: int,
    dataset_revision: str,
) -> str:
    if not isinstance(state, Mapping):
        raise ValueError("selected checkpoint state must be a mapping")
    _reject_unknown_fields(
        state,
        {
            "architecture",
            "architecture_revision",
            "campaign_id",
            "dataset_cursor",
            "dataset_revision",
            "format",
            "loss_history",
            "model",
            "optimizer",
            "optimizer_step",
            "step",
        },
        "checkpoint state",
    )
    expected = {
        "architecture": campaign.model.architecture,
        "architecture_revision": campaign.model.architecture_revision,
        "campaign_id": campaign.campaign["id"],
        "dataset_revision": dataset_revision,
        "format": "orcacolony_checkpoint_v1",
        "optimizer_step": step,
        "step": step,
    }
    for field in ("architecture_revision", "optimizer_step", "step"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"selected checkpoint {field} must be an integer")
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("selected checkpoint provenance is inconsistent")
    model = state.get("model")
    optimizer = state.get("optimizer")
    if not isinstance(model, Mapping) or not isinstance(optimizer, Mapping):
        raise ValueError("selected checkpoint artifact metadata is invalid")
    _reject_unknown_fields(model, {"file", "sha256"}, "checkpoint model")
    _reject_unknown_fields(optimizer, {"file", "sha256"}, "checkpoint optimizer")
    if model.get("file") != "model.safetensors" or optimizer.get(
        "file"
    ) != "optimizer.safetensors":
        raise ValueError("selected checkpoint artifact filenames are invalid")
    model_sha256 = _sha256_file(checkpoint / "model.safetensors")
    if model.get("sha256") != model_sha256 or optimizer.get(
        "sha256"
    ) != _sha256_file(checkpoint / "optimizer.safetensors"):
        raise ValueError("selected checkpoint artifact digest is inconsistent")
    loss_history = state.get("loss_history")
    if (
        not isinstance(loss_history, list)
        or len(loss_history) != step
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in loss_history
        )
    ):
        raise ValueError("selected checkpoint loss history is invalid")
    dataset_cursor = state.get("dataset_cursor")
    if (
        isinstance(dataset_cursor, bool)
        or not isinstance(dataset_cursor, int)
        or dataset_cursor < 0
    ):
        raise ValueError("selected checkpoint dataset cursor is invalid")
    return model_sha256


def _validate_lora_checkpoint_for_release(
    lora: LoadedLoRAManifest,
    checkpoint: Path,
    state: Mapping[str, object],
    step: int,
    dataset_revision: str,
    numerical_profile: str,
) -> dict[str, object]:
    if (
        state.get("format") != "orcacolony_lora_checkpoint_v2"
        or "numerical_profile" not in state
    ):
        raise ValueError(
            "release LoRA checkpoint numerical profile is missing; "
            "profile-bearing v2 format is required"
        )
    _, _, loaded_step, _, _ = load_lora_checkpoint(
        lora,
        checkpoint,
        expected_numerical_profile=numerical_profile,
    )
    if loaded_step != step or state.get("dataset_revision") != dataset_revision:
        raise ValueError("selected LoRA checkpoint provenance is inconsistent")
    adapter = state.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("selected LoRA adapter metadata is invalid")
    return {
        "training_method": "frozen-base-lora",
        "numerical_profile": state["numerical_profile"],
        "base_model_sha256": lora.config.base_model_sha256,
        "adapter_sha256": adapter["tensor_sha256"],
        "weight_checkpoint_sha256": state["weight_checkpoint_sha256"],
        "resume_state_sha256": state["checkpoint_sha256"],
    }


def _copy_static_site(
    browser_root: Path,
    destination: Path,
    public_coordinator_url: str | None,
    campaign_id: str,
) -> None:
    for filename in ("index.html", "index.js"):
        source = browser_root / filename
        if filename == "index.html":
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"release input must be a regular file: {source}")
            content = source.read_text(encoding="utf-8")
            coordinator_marker = '<meta name="orcacolony-coordinator" content="">'
            campaign_marker = '<meta name="orcacolony-campaign" content="">'
            if (
                content.count(coordinator_marker) != 1
                or content.count(campaign_marker) != 1
            ):
                raise ValueError("browser index does not contain release pin markers")
            coordinator = public_coordinator_url or ""
            pinned_content = content.replace(
                coordinator_marker,
                (
                    '<meta name="orcacolony-coordinator" '
                    f'content="{escape(coordinator, quote=True)}">'
                ),
            ).replace(
                campaign_marker,
                (
                    '<meta name="orcacolony-campaign" '
                    f'content="{escape(campaign_id, quote=True)}">'
                ),
            )
            (destination / filename).parent.mkdir(parents=True, exist_ok=True)
            (destination / filename).write_text(
                pinned_content,
                encoding="utf-8",
                newline="\n",
            )
        else:
            _copy_public_file(source, destination / filename)
    package_root = browser_root / "pkg"
    if not package_root.is_dir() or package_root.is_symlink():
        raise ValueError("browser root does not contain a regular pkg directory")
    package_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(package_root).parts)
    )
    if not package_files:
        raise ValueError("browser package is empty")
    for source in package_files:
        if source.is_symlink():
            raise ValueError(f"browser package may not contain symlinks: {source}")
        _copy_public_file(
            source,
            destination / "pkg" / source.relative_to(package_root),
        )


def build_release_bundle(
    campaign: CampaignConfig,
    coordinator: CampaignCoordinator,
    *,
    dataset_root: str | Path,
    browser_root: str | Path,
    project_license: str | Path,
    third_party_notice: str | Path,
    public_coordinator_url: str | None,
    output_dir: str | Path,
    promotion_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    coordinator.validate_evaluation_authority()
    dashboard = coordinator.dashboard()
    if dashboard["campaign"]["state"] != "campaign_complete":
        raise ValueError("release bundle requires a completed campaign")
    evaluation_gate = dashboard.get("evaluation_gate")
    if (
        isinstance(evaluation_gate, Mapping)
        and evaluation_gate.get("state") != "passed"
    ):
        raise ValueError("release bundle requires the declared evaluation gate to pass")

    dataset_root = Path(dataset_root).resolve()
    browser_root = Path(browser_root).resolve()
    if public_coordinator_url is not None:
        public_coordinator_url = normalize_http_origin(public_coordinator_url)
    dataset = coordinator.dataset
    if dataset is None:
        raise ValueError("release dataset authority is unavailable")
    validate_public_dataset_manifest(dataset.manifest)
    if dataset.revision != dashboard["dataset"]["revision"]:
        raise ValueError("release dataset revision does not match the campaign")

    output_dir = Path(output_dir).resolve()
    for input_root in (dataset_root, browser_root, coordinator.state_dir.resolve()):
        if output_dir.is_relative_to(input_root):
            raise ValueError("release output may not be inside an input directory")
    if output_dir.exists():
        raise ValueError(f"release output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"temporary release output already exists: {temporary}")
    temporary.mkdir()

    try:
        numerical_profile = str(dashboard["campaign"]["numerical_profile"])
        lock = coordinator._lock_payload()
        if lock.get("campaign_revision") != _revision(_campaign_payload(campaign)):
            raise ValueError("release campaign revision differs from campaign lock")
        if lock.get("numerical_profile") != numerical_profile:
            raise ValueError("release numerical profile differs from campaign lock")
        lora = coordinator.lora
        evaluations = dashboard["evaluations"]
        if not isinstance(evaluations, list) or any(
            not isinstance(entry, Mapping) for entry in evaluations
        ):
            raise ValueError("release evaluations are invalid")
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
        if lora is not None:
            evaluation_fields.update(
                {
                    "training_method",
                    "base_model_sha256",
                    "adapter_sha256",
                    "weight_checkpoint_sha256",
                    "resume_state_sha256",
                }
            )
        for evaluation in evaluations:
            if (
                evaluation.get("format") != "orcacolony_evaluation_v1"
                or set(evaluation) != evaluation_fields
            ):
                raise ValueError("release evaluation schema is invalid")
            if (
                evaluation.get("campaign_id") != dashboard["campaign"]["id"]
                or evaluation.get("dataset_revision") != dataset.revision
                or type(evaluation.get("step")) is not int
                or evaluation["step"] < 0
                or type(evaluation.get("validation_sequences")) is not int
                or evaluation["validation_sequences"] <= 0
                or type(evaluation.get("loss_weight_sum")) is not int
                or evaluation["loss_weight_sum"] <= 0
                or any(
                    type(evaluation.get(field)) is not float
                    or not math.isfinite(evaluation[field])
                    for field in ("loss_sum", "mean_loss", "perplexity")
                )
                or not _is_sha256(evaluation.get("checkpoint_sha256"))
            ):
                raise ValueError("release evaluation provenance differs")
            evaluation_profile = evaluation.get("numerical_profile")
            if evaluation_profile != numerical_profile:
                raise ValueError("release evaluation numerical profile differs")
            evaluation_step = evaluation["step"]
            evaluation_artifacts = coordinator.versioned_checkpoint_artifacts(
                evaluation_step
            )
            expected_artifact_names = {
                "state.json",
                "optimizer.safetensors",
                "adapter.safetensors" if lora is not None else "model.safetensors",
            }
            if set(evaluation_artifacts) != expected_artifact_names:
                raise ValueError("release evaluation checkpoint artifact set is invalid")
            evaluation_checkpoint = (
                temporary / ".validation" / f"step-{evaluation_step:08d}"
            )
            _write_owned_checkpoint(evaluation_artifacts, evaluation_checkpoint)
            evaluation_state = json.loads(
                evaluation_artifacts["state.json"],
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            try:
                if lora is not None:
                    if evaluation.get("training_method") != "frozen-base-lora":
                        raise ValueError("release evaluation training method differs")
                    evaluation_identities = _validate_lora_checkpoint_for_release(
                        lora,
                        evaluation_checkpoint,
                        evaluation_state,
                        evaluation_step,
                        dataset.revision,
                        numerical_profile,
                    )
                    expected_evaluation_identities = {
                        "checkpoint_sha256": evaluation_identities[
                            "resume_state_sha256"
                        ],
                        **{
                            field: evaluation_identities[field]
                            for field in (
                                "base_model_sha256",
                                "adapter_sha256",
                                "weight_checkpoint_sha256",
                                "resume_state_sha256",
                            )
                        },
                    }
                else:
                    expected_evaluation_identities = {
                        "checkpoint_sha256": _validate_checkpoint_for_release(
                            campaign,
                            evaluation_checkpoint,
                            evaluation_state,
                            evaluation_step,
                            dataset.revision,
                        )
                    }
            finally:
                shutil.rmtree(evaluation_checkpoint)
            if any(
                evaluation.get(field) != expected
                for field, expected in expected_evaluation_identities.items()
            ):
                raise ValueError(
                    "release evaluation LoRA provenance checkpoint identity differs"
                    if lora is not None
                    else "release evaluation checkpoint identity differs"
                )
        shutil.rmtree(temporary / ".validation", ignore_errors=True)

        step, selected_evaluation = _select_checkpoint(
            coordinator,
            dashboard,
        )
        selected_artifacts = coordinator.versioned_checkpoint_artifacts(step)
        expected_selected_names = {
            "state.json",
            "optimizer.safetensors",
            "adapter.safetensors" if lora is not None else "model.safetensors",
        }
        if set(selected_artifacts) != expected_selected_names:
            raise ValueError("selected checkpoint artifact set is invalid")
        checkpoint_state_bytes = selected_artifacts["state.json"]
        checkpoint_state = json.loads(
            checkpoint_state_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        published_checkpoint_dir = temporary / "checkpoint"
        _write_owned_checkpoint(selected_artifacts, published_checkpoint_dir)
        lora_identities = (
            _validate_lora_checkpoint_for_release(
                lora,
                published_checkpoint_dir,
                checkpoint_state,
                step,
                dataset.revision,
                numerical_profile,
            )
            if lora is not None
            else None
        )
        checkpoint_sha256 = (
            str(lora_identities["resume_state_sha256"])
            if lora_identities is not None
            else _validate_checkpoint_for_release(
                campaign,
                published_checkpoint_dir,
                checkpoint_state,
                step,
                dataset.revision,
            )
        )
        if selected_evaluation is not None and (
            selected_evaluation["checkpoint_sha256"] != checkpoint_sha256
        ):
            raise ValueError("selected evaluation does not match its checkpoint")
        if lora_identities is not None and selected_evaluation is not None:
            selected_lora_identities = {
                field: selected_evaluation.get(field)
                for field in (
                    "base_model_sha256",
                    "adapter_sha256",
                    "weight_checkpoint_sha256",
                    "resume_state_sha256",
                )
            }
            expected_lora_identities = {
                field: lora_identities[field] for field in selected_lora_identities
            }
            if selected_lora_identities != expected_lora_identities:
                raise ValueError(
                    "selected evaluation LoRA provenance does not match its checkpoint"
                )

        campaign_payload = _campaign_payload(campaign)
        _write_json(temporary / "campaign.json", campaign_payload)
        _write_json(temporary / "campaign-lock.json", lock)
        release_dashboard = copy.deepcopy(dashboard)
        completed_step = dashboard["progress"]["completed_steps"]
        if type(completed_step) is not int or completed_step < 0:
            raise ValueError("release completed step is invalid")
        release_dashboard["checkpoint"] = {
            "download_url": (
                "checkpoint/adapter.safetensors"
                if lora_identities is not None
                else "checkpoint/model.safetensors"
            ),
            "parity": (
                dashboard["checkpoint"]["parity"] if step == completed_step else None
            ),
            "numerical_profile": numerical_profile,
            "sha256": checkpoint_sha256,
            **(lora_identities or {}),
        }
        _write_json(temporary / "public-dashboard.json", release_dashboard)
        _write_json(
            temporary / "public-ledger.json",
            {
                "format": "orcacolony_public_ledger_v1",
                "campaign_id": dashboard["campaign"]["id"],
                "numerical_profile": numerical_profile,
                "entries": dashboard["public_ledger"],
            },
        )
        internal_ledger = coordinator._ledger_payload(include_current=True)
        internal_entries = internal_ledger.get("entries")
        if not isinstance(internal_entries, list) or any(
            not isinstance(entry, Mapping) for entry in internal_entries
        ):
            raise ValueError("release attribution ledger is invalid")
        attribution_snapshot = build_attribution_snapshot(
            coordinator.participants,
            internal_entries,
        )
        _write_json(
            temporary / "attribution-snapshot.json",
            attribution_snapshot,
        )
        (temporary / "CONTRIBUTORS.md").write_text(
            attribution_markdown(attribution_snapshot),
            encoding="utf-8",
            newline="\n",
        )
        _write_json(
            temporary / "evaluations.json",
            {
                "format": "orcacolony_campaign_evaluations_v1",
                "campaign_id": dashboard["campaign"]["id"],
                "dataset_revision": dataset.revision,
                "numerical_profile": numerical_profile,
                "profile": (
                    dict(campaign.evaluation)
                    if campaign.evaluation is not None
                    else None
                ),
                "entries": dashboard["evaluations"],
            },
        )

        if lora_identities is not None:
            _write_json(
                temporary / "lora.json",
                {
                    "format": "orcacolony_release_lora_v1",
                    "manifest_sha256": lora.manifest_sha256,
                    "config": asdict(lora.config),
                },
            )
            base_model_bytes = coordinator.initial_model_bytes()
            base_model_sha256 = tensor_sha256(load_safetensors(base_model_bytes))
            if base_model_sha256 != lora_identities["base_model_sha256"]:
                raise ValueError("release frozen-base artifact identity is inconsistent")
            (temporary / "checkpoint" / "base-model.safetensors").write_bytes(
                base_model_bytes
            )
        for filename in _DATASET_FILES:
            dataset_destination = temporary / "dataset" / filename
            dataset_destination.parent.mkdir(parents=True, exist_ok=True)
            dataset_destination.write_bytes(
                coordinator.dataset_artifact_bytes(filename)
            )
        published_dataset = PackedDataset.load(temporary / "dataset")
        if published_dataset.revision != dataset.revision:
            raise ValueError("copied release dataset revision is inconsistent")
        published_state_path = temporary / "checkpoint" / "state.json"
        published_state_bytes = published_state_path.read_bytes()
        if published_state_bytes != checkpoint_state_bytes:
            raise ValueError("copied release checkpoint state changed during export")
        published_checkpoint = json.loads(published_state_bytes)
        if lora is not None:
            copied_lora_identities = _validate_lora_checkpoint_for_release(
                lora,
                temporary / "checkpoint",
                published_checkpoint,
                step,
                dataset.revision,
                numerical_profile,
            )
            copied_base_sha256 = tensor_sha256(
                load_safetensors_file(
                    str(temporary / "checkpoint" / "base-model.safetensors")
                )
            )
            if (
                copied_lora_identities != lora_identities
                or copied_base_sha256 != lora.config.base_model_sha256
            ):
                raise ValueError("copied LoRA release checkpoint is inconsistent")
        else:
            copied_model_sha256 = _validate_checkpoint_for_release(
                campaign,
                temporary / "checkpoint",
                published_checkpoint,
                step,
                dataset.revision,
            )
            if copied_model_sha256 != checkpoint_sha256:
                raise ValueError("copied release checkpoint digest is inconsistent")
        capability_contract = (
            campaign.research is not None
            and campaign.research.get("format")
            == "orcacolony_capability_research_v1"
        )
        final_holdout_evaluation: Mapping[str, object] | None = None
        promotion_passed = False
        if capability_contract:
            final_holdout_evaluation = (
                evaluate_lora_final_holdout(
                    lora,
                    temporary / "checkpoint",
                    dataset,
                )
                if lora is not None
                else evaluate_final_holdout(
                    campaign,
                    temporary / "checkpoint",
                    dataset,
                )
            )
            if (
                final_holdout_evaluation.get("step") != step
                or final_holdout_evaluation.get("checkpoint_sha256")
                != checkpoint_sha256
                or final_holdout_evaluation.get(
                    "selection_locked_before_evaluation"
                )
                is not True
            ):
                raise ValueError(
                    "final-holdout evaluation is not bound to the selected checkpoint"
                )
            _write_json(
                temporary / "language-model-final-holdout-evaluation.json",
                final_holdout_evaluation,
            )
            if promotion_evidence is not None:
                promotion_passed = _validate_promotion_evidence(
                    campaign,
                    promotion_evidence,
                    checkpoint_sha256=checkpoint_sha256,
                    dataset_revision=dataset.revision,
                )
                _write_json(
                    temporary / "promotion-evidence.json",
                    promotion_evidence,
                )
        elif promotion_evidence is not None:
            raise ValueError(
                "systems-evidence release may not contain capability promotion evidence"
            )
        _copy_public_file(Path(project_license), temporary / "LICENSE")
        _copy_public_file(
            Path(third_party_notice),
            temporary / "THIRD_PARTY_DATA.md",
        )
        _copy_static_site(
            browser_root,
            temporary / "site",
            public_coordinator_url,
            str(dashboard["campaign"]["id"]),
        )

        (temporary / "OPERATE.md").write_text(
            "# OrcaColony v0.1 release bundle\n\n"
            "- `checkpoint/` is the selected canonical dense model or separate "
            "frozen base plus adapter, together with restart state.\n"
            "- `dataset/` is the exact redistributable packed dataset and tokenizer.\n"
            "- `site/` is the static browser worker and public campaign dashboard.\n"
            "- `public-ledger.json` contains only contributor-approved public credit.\n\n"
            "- `attribution-snapshot.json` and `CONTRIBUTORS.md` freeze release-time "
            "credit choices and accepted contribution totals.\n\n"
            + (
                "- `language-model-final-holdout-evaluation.json` records the "
                "reserved language-loss diagnostic performed after checkpoint "
                "selection. Behavioral promotion is separate and requires "
                "`promotion-evidence.json`.\n\n"
                if final_holdout_evaluation is not None
                else "- This is a systems-evidence release, not a capability-promoted "
                "model; it has no release-time language-model holdout result.\n\n"
            )
            + "Serve `site/` from an HTTPS static origin. Its mutable coordinator origin "
            "is pinned in `site/index.html`; rebuild with `--public-coordinator-url` "
            "rather than editing worker links. Run the coordinator with the exact "
            "static-site `--public-origin`, then link contributors to "
            "`?cpu-loop=<worker-id>` with the worker token in "
            "`#token=<worker-token>`. Keep participant manifests and publication "
            "credentials outside this public bundle.\n",
            encoding="utf-8",
            newline="\n",
        )

        payload_files = {
            path.relative_to(temporary).as_posix(): _sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        checkpoint_manifest: dict[str, object] = {
            "step": step,
            "numerical_profile": numerical_profile,
            "selection": (
                "lowest_mean_loss" if selected_evaluation is not None else "final"
            ),
            "evaluation": selected_evaluation,
        }
        if lora_identities is not None:
            checkpoint_manifest.update(lora_identities)
        else:
            checkpoint_manifest["model_sha256"] = checkpoint_sha256
        release_manifest: dict[str, object] = {
            "format": "orcacolony_release_bundle_v1",
            "campaign_id": dashboard["campaign"]["id"],
            "campaign_revision": _revision(campaign_payload),
            "numerical_profile": numerical_profile,
            "public_coordinator_url": public_coordinator_url,
            "participants_revision": lock["participants_revision"],
            "credit_profiles_revision": (
                coordinator.participants.credit_revision
            ),
            "attribution_snapshot_sha256": attribution_snapshot[
                "snapshot_sha256"
            ],
            "dataset_revision": dataset.revision,
            "release_classification": (
                "capability_model"
                if promotion_passed
                else (
                    "capability_candidate"
                    if capability_contract
                    else "systems_evidence_only"
                )
            ),
            "checkpoint": checkpoint_manifest,
            "language_model_final_holdout_evaluation": (
                final_holdout_evaluation
            ),
            "files": payload_files,
        }
        _write_json(temporary / "release-manifest.json", release_manifest)
        checksum_paths = [
            *sorted(payload_files),
            "release-manifest.json",
        ]
        checksums = {
            **payload_files,
            "release-manifest.json": _sha256_file(
                temporary / "release-manifest.json"
            ),
        }
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{checksums[path]}  {path}\n" for path in checksum_paths),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output_dir)
        return release_manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic public OrcaColony v0.1 release bundle"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--dataset-artifacts", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--project-license", type=Path, default=Path("LICENSE"))
    parser.add_argument(
        "--third-party-notice",
        type=Path,
        default=Path("THIRD_PARTY_DATA.md"),
    )
    parser.add_argument("--public-coordinator-url")
    parser.add_argument("--promotion-evidence", type=Path)
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
    parser.add_argument("--output", type=Path, required=True)
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
    dataset = PackedDataset.load(args.dataset_artifacts)
    coordinator = CampaignCoordinator.load(
        campaign,
        args.campaign_state,
        participants=participants,
        dataset=dataset,
        lora=lora,
        numerical_profile=args.numerical_profile,
    )
    manifest = build_release_bundle(
        campaign,
        coordinator,
        dataset_root=args.dataset_artifacts,
        browser_root=args.browser_root,
        project_license=args.project_license,
        third_party_notice=args.third_party_notice,
        public_coordinator_url=args.public_coordinator_url,
        output_dir=args.output,
        promotion_evidence=(
            json.loads(args.promotion_evidence.read_text(encoding="utf-8"))
            if args.promotion_evidence is not None
            else None
        ),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
