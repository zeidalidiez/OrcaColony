from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save as save_safetensors
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor
from torch.nn import functional as F

from .artifacts import PackedDataset
from .coordinator import _tensor_metrics
from .participants import ParticipantRegistry, load_participants
from .peft import (
    BURN_NDARRAY_F32_PROFILE,
    BURN_WEBGPU_F32_PROFILE,
    EXACT_CPU_FP32_PROFILE,
    INT8_FROZEN_LINEAR_PROFILE,
    LAYER_BUNDLE_STREAMED_FP32_PROFILE,
    NUMERICAL_PROFILES,
    LoadedLoRAManifest,
    _safe_checkpoint_artifact_path,
    adapter_named_parameters,
    apply_adapter_gradient_step,
    base_layer_bundle_artifact_contract,
    build_lora_model,
    build_profiled_lora_model,
    compute_adapter_gradients,
    create_adapter_optimizer,
    export_base_layer_bundle,
    load_adapter_state,
    load_lora_checkpoint,
    load_lora_manifest,
    lora_weight_checkpoint_sha256,
    run_lora_training,
    save_lora_checkpoint,
)
from .reference import (
    CampaignConfig,
    _create_optimizer,
    _load_checkpoint,
    _save_checkpoint,
    _sha256_file,
    build_model,
    fixture_batch,
    run_training,
    tensor_sha256,
    validate_dataset_artifacts,
)


@dataclass(frozen=True)
class LeasedGradient:
    assignment_id: str
    lease_token: str
    checkpoint_sha256: str
    loss_sum: float
    loss_weight_sum: int
    safetensors: bytes
    runtime_backend: str
    worker_telemetry: Mapping[str, object] | None = None
    coordinator_receive_seconds: float | None = None


EXACT_FP32_RUNTIME_BACKENDS = frozenset(
    {
        "python-native-cpu-f32",
        "python-native-cpu-layer-bundle-f32",
        "python-oracle-f32",
    }
)
BURN_NDARRAY_RUNTIME_BACKENDS = frozenset({"burn-ndarray-f32"})
BURN_WEBGPU_RUNTIME_BACKENDS = frozenset({"burn-webgpu-f32"})
INT8_RUNTIME_BACKENDS = frozenset(
    {
        "python-native-cpu-int8-f32-dequant",
        "python-native-cpu-layer-bundle-int8-f32-dequant",
        "python-oracle-int8-f32-dequant",
    }
)
_PROFILE_RUNTIME_BACKENDS = {
    EXACT_CPU_FP32_PROFILE: EXACT_FP32_RUNTIME_BACKENDS,
    BURN_NDARRAY_F32_PROFILE: BURN_NDARRAY_RUNTIME_BACKENDS,
    BURN_WEBGPU_F32_PROFILE: BURN_WEBGPU_RUNTIME_BACKENDS,
    INT8_FROZEN_LINEAR_PROFILE: INT8_RUNTIME_BACKENDS,
}
RUNTIME_BACKENDS = frozenset().union(*_PROFILE_RUNTIME_BACKENDS.values())
_RUNTIME_NUMERICAL_PROFILE = {
    backend: profile
    for profile, backends in _PROFILE_RUNTIME_BACKENDS.items()
    for backend in backends
}
_LAYER_BUNDLE_RUNTIME_BACKENDS = frozenset(
    {
        "python-native-cpu-layer-bundle-f32",
        "python-native-cpu-layer-bundle-int8-f32-dequant",
    }
)
_ORACLE_RUNTIME_BACKENDS = frozenset(
    {"python-oracle-f32", "python-oracle-int8-f32-dequant"}
)


def _validated_numerical_profile(value: object) -> str:
    if not isinstance(value, str) or value not in NUMERICAL_PROFILES:
        raise ValueError("numerical profile is unsupported")
    return value


def _checkpoint_numerical_profile(profile: str) -> str | None:
    return None if profile == EXACT_CPU_FP32_PROFILE else profile


def _validate_checkpoint_profile_metrics(
    profile: str,
    metrics: Mapping[str, float | int | str],
) -> None:
    strict_profile = profile in {
        EXACT_CPU_FP32_PROFILE,
        INT8_FROZEN_LINEAR_PROFILE,
    }
    minimum_cosine = 0.999999 if strict_profile else 0.999
    maximum_relative_l2 = 1e-6 if strict_profile else 0.01
    if (
        float(metrics["cosine_similarity"]) < minimum_cosine
        or float(metrics["relative_l2_error"]) > maximum_relative_l2
    ):
        raise ValueError("checkpoint is outside the numerical-profile oracle gate")

_RUNTIME_TELEMETRY_FIELDS = (
    "assignment_fetch",
    "runtime_init",
    "artifact_fetch",
    "gradient_compute",
)
_TRANSFER_TELEMETRY_FIELDS = (
    "assignment",
    "model",
    "adapter",
    "oracle_gradient",
    "result",
)
_MEMORY_TELEMETRY_FIELDS = (
    "wasm_linear",
    "process_peak_rss",
    "js_heap_used",
    "js_heap_limit",
    "device_capacity",
)
_MAX_SAFE_TELEMETRY_INTEGER = 2**53 - 1


def _expected_model_download_bytes(
    resource_profile: Mapping[str, object],
    runtime_backend: str,
) -> int:
    field = (
        "layer_bundle_download_bytes"
        if runtime_backend in _LAYER_BUNDLE_RUNTIME_BACKENDS
        else "model_download_bytes"
    )
    return int(resource_profile[field])


def _exact_mapping(
    value: object,
    fields: tuple[str, ...],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} fields are invalid")
    return value


def _validate_worker_telemetry(
    payload: Mapping[str, object] | None,
    resource_profile: Mapping[str, object],
    runtime_backend: str,
    base_layer_bundle: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if payload is None:
        if runtime_backend not in _ORACLE_RUNTIME_BACKENDS:
            raise ValueError("worker telemetry is required")
        return None
    envelope_fields = {
        "format",
        "runtime_seconds",
        "transfer_bytes",
        "memory_bytes",
    }
    has_model_artifacts = "model_artifacts" in payload
    if has_model_artifacts:
        envelope_fields.add("model_artifacts")
    if (
        set(payload) != envelope_fields
        or payload.get("format") != "orcacolony_worker_telemetry_v1"
        or (
            has_model_artifacts
            and runtime_backend not in _LAYER_BUNDLE_RUNTIME_BACKENDS
        )
    ):
        raise ValueError("worker telemetry envelope is invalid")
    runtime = _exact_mapping(
        payload.get("runtime_seconds"),
        _RUNTIME_TELEMETRY_FIELDS,
        "worker runtime telemetry",
    )
    canonical_runtime: dict[str, float] = {}
    for field in _RUNTIME_TELEMETRY_FIELDS:
        value = runtime[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or float(value) > 86_400
        ):
            raise ValueError(f"worker runtime telemetry {field} is invalid")
        canonical_runtime[field] = float(value)
    transfer = _exact_mapping(
        payload.get("transfer_bytes"),
        _TRANSFER_TELEMETRY_FIELDS,
        "worker transfer telemetry",
    )
    canonical_transfer: dict[str, int] = {}
    for field in _TRANSFER_TELEMETRY_FIELDS:
        value = transfer[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > _MAX_SAFE_TELEMETRY_INTEGER
        ):
            raise ValueError(f"worker transfer telemetry {field} is invalid")
        canonical_transfer[field] = value
    expected_transfer = {
        "model": _expected_model_download_bytes(resource_profile, runtime_backend),
        "adapter": int(resource_profile["adapter_download_bytes"]),
        "oracle_gradient": int(resource_profile["oracle_gradient_download_bytes"]),
        "result": int(resource_profile["expected_result_upload_bytes"]),
    }
    canonical_model_artifacts: list[str] | None = None
    if has_model_artifacts:
        if not isinstance(base_layer_bundle, Mapping):
            raise ValueError("worker model artifact telemetry was not assigned")
        assigned_artifacts = base_layer_bundle.get("artifacts")
        reported_artifacts = payload.get("model_artifacts")
        if not isinstance(assigned_artifacts, list) or not isinstance(
            reported_artifacts,
            list,
        ):
            raise ValueError("worker model artifact telemetry is invalid")
        assigned_sizes: dict[str, int] = {}
        assigned_order: list[str] = []
        for artifact in assigned_artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError("assigned model artifact telemetry contract is invalid")
            name = artifact.get("file")
            size = artifact.get("bytes")
            if (
                not isinstance(name, str)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or name in assigned_sizes
            ):
                raise ValueError("assigned model artifact telemetry contract is invalid")
            assigned_order.append(name)
            assigned_sizes[name] = size
        if (
            any(not isinstance(name, str) for name in reported_artifacts)
            or len(set(reported_artifacts)) != len(reported_artifacts)
            or any(name not in assigned_sizes for name in reported_artifacts)
        ):
            raise ValueError("worker model artifact telemetry is invalid")
        reported_names = set(reported_artifacts)
        canonical_model_artifacts = [
            name for name in assigned_order if name in reported_names
        ]
        if reported_artifacts != canonical_model_artifacts:
            raise ValueError("worker model artifact telemetry order differs")
        if sum(assigned_sizes[name] for name in canonical_model_artifacts) != (
            canonical_transfer["model"]
        ):
            raise ValueError("worker model artifact telemetry bytes differ")
    for field, expected in expected_transfer.items():
        if field == "model" and has_model_artifacts:
            continue
        allowed = {expected} if field == "result" else {0, expected}
        if canonical_transfer[field] not in allowed:
            raise ValueError(f"worker transfer telemetry {field} does not match assignment")
    memory = _exact_mapping(
        payload.get("memory_bytes"),
        _MEMORY_TELEMETRY_FIELDS,
        "worker memory telemetry",
    )
    canonical_memory: dict[str, int | None] = {}
    for field in _MEMORY_TELEMETRY_FIELDS:
        value = memory[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > _MAX_SAFE_TELEMETRY_INTEGER
        ):
            raise ValueError(f"worker memory telemetry {field} is invalid")
        canonical_memory[field] = value  # type: ignore[assignment]
    canonical: dict[str, object] = {
        "format": "orcacolony_worker_telemetry_v1",
        "runtime_seconds": canonical_runtime,
        "transfer_bytes": canonical_transfer,
        "memory_bytes": canonical_memory,
    }
    if canonical_model_artifacts is not None:
        canonical["model_artifacts"] = canonical_model_artifacts
    return canonical


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            # Atomic state publication can replace a temporary file while a
            # concurrent dashboard snapshot is walking the managed directory.
            continue
    return total


def _aggregate_resource_observations(
    entries: list[Mapping[str, object]],
    state_dir: Path,
) -> dict[str, object]:
    runtime = {field: 0.0 for field in _RUNTIME_TELEMETRY_FIELDS}
    transfer = {
        "assignment": 0,
        "model_download": 0,
        "adapter_download": 0,
        "oracle_gradient_download": 0,
        "result_upload": 0,
    }
    memory: dict[str, int | None] = {
        "peak_wasm_linear": None,
        "peak_process_rss": None,
        "peak_js_heap_used": None,
    }
    worker_reports = 0
    for entry in entries:
        instrumentation = entry.get("instrumentation")
        if not isinstance(instrumentation, Mapping):
            continue
        worker = instrumentation.get("worker_reported")
        measured = instrumentation.get("coordinator_measured")
        if isinstance(measured, Mapping):
            transfer["result_upload"] += int(measured["result_upload_bytes"])
        if not isinstance(worker, Mapping):
            continue
        worker_reports += 1
        worker_runtime = worker["runtime_seconds"]
        worker_transfer = worker["transfer_bytes"]
        worker_memory = worker["memory_bytes"]
        for field in _RUNTIME_TELEMETRY_FIELDS:
            runtime[field] += float(worker_runtime[field])
        transfer["assignment"] += int(worker_transfer["assignment"])
        transfer["model_download"] += int(worker_transfer["model"])
        transfer["adapter_download"] += int(worker_transfer["adapter"])
        transfer["oracle_gradient_download"] += int(
            worker_transfer["oracle_gradient"]
        )
        for source, target in (
            ("wasm_linear", "peak_wasm_linear"),
            ("process_peak_rss", "peak_process_rss"),
            ("js_heap_used", "peak_js_heap_used"),
        ):
            value = worker_memory[source]
            if value is not None:
                memory[target] = max(memory[target] or 0, int(value))
    return {
        "format": "orcacolony_resource_observations_v1",
        "accepted_assignments": len(entries),
        "worker_reports": worker_reports,
        "runtime_seconds": runtime,
        "transfer_bytes": transfer,
        "memory_bytes": memory,
        "coordinator_storage_bytes": _directory_size(state_dir),
    }


@dataclass(frozen=True)
class WorkReceipt:
    assignment_id: str
    accepted: bool
    step_complete: bool
    step: int
    model_sha256: str | None
    adapter_sha256: str | None
    weight_checkpoint_sha256: str | None
    checkpoint_sha256: str | None
    gradient_metrics: Mapping[str, float | int | str]
    checkpoint_metrics: Mapping[str, float | int | str]
    instrumentation: Mapping[str, object]


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_MAX_GRADIENT_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_ASSIGNMENT_IDENTITY_FIELDS = frozenset(
    {
        "result_file_sha256",
        "result_tensor_sha256",
        "oracle_file_sha256",
        "oracle_tensor_sha256",
        "oracle_file_size",
    }
)
_RESULT_STATE_IDENTITY_FIELDS = frozenset(
    {
        "result_dataset_cursor",
        "result_loss_history",
        "result_resume_state_sha256",
        "result_weight_checkpoint_sha256",
        "result_checkpoint_sha256",
    }
)
_GLOBAL_STATE_FIELDS = frozenset(
    {
        "format",
        "campaign_id",
        "campaign_revision",
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "initial_adapter_sha256",
        "resume_state_sha256",
        "numerical_profile",
        "base_layer_bundle",
        "participants",
        "participants_revision",
        "checkpoint_sha256",
        "dataset_revision",
        "worker_count",
        "lease_seconds",
        "state",
        "step",
        "base_step",
        "dataset_cursor",
        "loss_history",
        "has_base_checkpoint",
        "result_protocol_revision",
        "accepted_result_identity_revision",
        "assignments",
        "model_sha256",
        "adapter_sha256",
        "result_dataset_cursor",
        "result_loss_history",
        "result_resume_state_sha256",
        "result_weight_checkpoint_sha256",
        "result_checkpoint_sha256",
        "checkpoint_metrics",
    }
)
_ASSIGNMENT_FIELDS = frozenset(
    {
        "assignment_id",
        "campaign_id",
        "checkpoint_sha256",
        "training_method",
        "numerical_profile",
        "dataset_revision",
        "model",
        "global_step",
        "data_range",
        "input_ids",
        "input_shape",
        "target_ids",
        "target_shape",
        "loss_weight_sum",
        "parameter_count",
        "trainable_parameter_count",
        "adapter",
        "expected_loss_sum",
        "oracle_file",
        "oracle_file_sha256",
        "oracle_tensor_sha256",
        "oracle_file_size",
        "state",
        "attempt",
        "leased_by",
        "contributor_id",
        "lease_token",
        "lease_expires_at",
        "result_file",
        "result_file_sha256",
        "result_tensor_sha256",
        "accepted_loss_sum",
        "runtime_backend",
        "gradient_metrics",
    }
)
_LORA_ASSIGNMENT_FIELDS = frozenset(
    {
        "lora_manifest_sha256",
        "base_model_sha256",
        "adapter_sha256",
        "weight_checkpoint_sha256",
        "resume_state_sha256",
    }
)


def _assignment_fields(
    *,
    lora: bool,
    base_layer_bundle: bool,
    accepted: bool,
) -> frozenset[str]:
    fields = set(_ASSIGNMENT_FIELDS)
    if lora:
        fields.update(_LORA_ASSIGNMENT_FIELDS)
    if base_layer_bundle:
        fields.add("base_layer_bundle_manifest_sha256")
    if accepted:
        fields.add("instrumentation")
    return frozenset(fields)


def _validate_persisted_assignment_lifecycle(
    assignment: Mapping[str, object],
    participants: ParticipantRegistry,
) -> None:
    state = assignment.get("state")
    attempt = assignment.get("attempt")
    if state not in {"open", "leased", "accepted"}:
        raise ValueError("persisted assignment state is invalid")
    if type(attempt) is not int or attempt < 0:
        raise ValueError("persisted assignment attempt is invalid")
    if state == "open":
        if attempt != 0 or any(
            assignment.get(field) is not None
            for field in (
                "leased_by",
                "contributor_id",
                "lease_token",
                "lease_expires_at",
                "result_file",
                "result_file_sha256",
                "result_tensor_sha256",
                "accepted_loss_sum",
                "runtime_backend",
                "gradient_metrics",
            )
        ):
            raise ValueError("persisted open assignment lifecycle is invalid")
        return
    worker_id = assignment.get("leased_by")
    contributor_id = assignment.get("contributor_id")
    lease_token = assignment.get("lease_token")
    lease_expires_at = assignment.get("lease_expires_at")
    participant = (
        participants.participant_for_worker(worker_id)
        if isinstance(worker_id, str) and worker_id
        else None
    )
    if (
        attempt < 1
        or participant is None
        or contributor_id != participant.contributor_id
    ):
        raise ValueError("persisted assignment lease authority is invalid")
    if state == "accepted":
        if lease_token is not None or lease_expires_at is not None:
            raise ValueError("persisted accepted assignment lease is not closed")
        return
    expected_lease_token = hashlib.sha256(
        f"{assignment.get('assignment_id')}:{worker_id}:{attempt}".encode("utf-8")
    ).hexdigest()
    if (
        lease_token != expected_lease_token
        or type(lease_expires_at) not in {int, float}
        or not math.isfinite(lease_expires_at)
    ):
        raise ValueError("persisted assignment lease authority is invalid")
    if state == "leased" and any(
        assignment.get(field) is not None
        for field in (
            "result_file",
            "result_file_sha256",
            "result_tensor_sha256",
            "accepted_loss_sum",
            "runtime_backend",
            "gradient_metrics",
        )
    ):
        raise ValueError("persisted leased assignment result authority is invalid")


def _is_reparse_point(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _file_identity(metadata: os.stat_result) -> tuple[int, int]:
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    if identity[1] <= 0:
        raise ValueError("artifact filesystem identity is unavailable")
    return identity


def _file_observation(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _safe_artifact_snapshot(root: Path, file_name: str, label: str) -> bytes:
    if Path(file_name).name != file_name or not file_name:
        raise ValueError(f"{label} name is invalid")
    try:
        root_before = os.lstat(root)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or stat.S_ISLNK(root_before.st_mode)
            or _is_reparse_point(root_before)
        ):
            raise ValueError(f"{label} root is not a regular directory")
        resolved_root = root.resolve(strict=True)
        path = root / file_name
        if path.parent != root:
            raise ValueError(f"{label} escapes its root")
        with os.scandir(root) as root_entries:
            root_entry = None
            for entry in root_entries:
                if entry.name == file_name:
                    root_entry = entry
                    break
            if root_entry is None:
                raise ValueError(f"{label} is unavailable")
            entry_before = root_entry.stat(follow_symlinks=False)
            path_before = os.lstat(path)
            if (
                root_entry.is_symlink()
                or not root_entry.is_file(follow_symlinks=False)
                or not stat.S_ISREG(path_before.st_mode)
                or stat.S_ISLNK(path_before.st_mode)
                or _is_reparse_point(path_before)
                or _is_reparse_point(entry_before)
                or _file_observation(entry_before) != _file_observation(path_before)
            ):
                raise ValueError(f"{label} is not a regular file")
            if path.resolve(strict=True).parent != resolved_root:
                raise ValueError(f"{label} escapes its root")
            if (
                path_before.st_size < 0
                or path_before.st_size > _MAX_GRADIENT_ARTIFACT_BYTES
            ):
                raise ValueError(f"{label} size is invalid")

            flags = (
                os.O_RDONLY
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _is_reparse_point(opened)
                    or _file_identity(opened) != _file_identity(path_before)
                    or _file_observation(opened) != _file_observation(path_before)
                ):
                    raise ValueError(f"{label} changed while being opened")
                first = stream.read(path_before.st_size + 1)
                stream.seek(0)
                second = stream.read(path_before.st_size + 1)
                opened_after = os.fstat(stream.fileno())
                path_after = os.lstat(path)
                root_after = os.lstat(root)
                resolved_root_after = root.resolve(strict=True)
                resolved_path_parent_after = path.resolve(strict=True).parent

                if (
                    first != second
                    or len(second) != path_before.st_size
                    or opened_after.st_size != path_before.st_size
                    or _file_observation(opened_after) != _file_observation(path_before)
                    or _file_identity(opened_after) != _file_identity(path_before)
                    or _file_identity(path_after) != _file_identity(path_before)
                    or _file_observation(path_after) != _file_observation(path_before)
                    or _file_identity(root_after) != _file_identity(root_before)
                    or _file_observation(root_after) != _file_observation(root_before)
                    or not stat.S_ISREG(path_after.st_mode)
                    or stat.S_ISLNK(path_after.st_mode)
                    or _is_reparse_point(path_after)
                    or not stat.S_ISDIR(root_after.st_mode)
                    or stat.S_ISLNK(root_after.st_mode)
                    or _is_reparse_point(root_after)
                    or resolved_root_after != resolved_root
                    or resolved_path_parent_after != resolved_root
                ):
                    raise ValueError(f"{label} changed while being read")
                return second
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} is unavailable") from error


def _owned_safetensors(payload: bytes, label: str) -> dict[str, Tensor]:
    try:
        tensors = load_safetensors(payload)
    except SafetensorError as error:
        raise ValueError(f"{label} is not a valid safetensors artifact") from error
    return {
        name: tensor.detach().clone().contiguous()
        for name, tensor in tensors.items()
    }


def _validated_relative_artifact_name(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a safe plain basename")
    relative = Path(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != value
        or value in {"", ".", ".."}
    ):
        raise ValueError(f"{label} must be a safe plain basename")
    return value


def _validated_assignment_artifact_name(
    assignment: Mapping[str, object],
) -> tuple[str, str]:
    assignment_id = assignment.get("assignment_id")
    if not isinstance(assignment_id, str) or _SHA256_HEX.fullmatch(assignment_id) is None:
        raise ValueError("assignment id is invalid")
    expected_file = f"{assignment_id}.safetensors"
    if assignment.get("oracle_file") != expected_file:
        raise ValueError("oracle gradient file identity is invalid")
    result_file = assignment.get("result_file")
    if result_file is not None and result_file != expected_file:
        raise ValueError("accepted result file identity is invalid")
    return assignment_id, expected_file


def _copy_checkpoint_artifacts(
    source_root: str | Path,
    destination_root: str | Path,
    filenames: tuple[object, ...],
) -> dict[str, bytes]:
    destination_root = Path(destination_root)
    snapshots: dict[str, bytes] = {}
    for value in filenames:
        name = _validated_relative_artifact_name(
            value,
            "resume checkpoint artifact",
        )
        payload = _safe_artifact_snapshot(
            source_root,
            name,
            "resume checkpoint source artifact",
        )
        _atomic_bytes(destination_root / name, payload)
        snapshots[name] = payload
    return snapshots


def _revision(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_exact_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _campaign_payload(campaign: CampaignConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "campaign": dict(campaign.campaign),
        "model": asdict(campaign.model),
        "training": asdict(campaign.training),
        "dataset": dict(campaign.dataset) if campaign.dataset is not None else None,
    }
    if campaign.evaluation is not None:
        payload["evaluation"] = dict(campaign.evaluation)
    return payload


class GlobalStepCoordinator:
    def __init__(
        self,
        campaign: CampaignConfig,
        state_dir: Path,
        state: dict[str, object],
        dataset: PackedDataset | None = None,
        lora: LoadedLoRAManifest | None = None,
    ) -> None:
        self.campaign = campaign
        self.state_dir = state_dir
        self.initial_model_path = state_dir / "model.safetensors"
        self.initial_adapter_path = state_dir / "adapter.safetensors"
        self.base_layer_bundle_dir = state_dir / "base-layer-bundle"
        self.base_checkpoint_dir = state_dir / "base-checkpoint"
        self.oracle_dir = state_dir / "oracle-gradients"
        self.results_dir = state_dir / "results"
        self.reference_dir = state_dir / "reference-step-1"
        self.checkpoint_dir = state_dir / "checkpoint"
        self._state = state
        self.dataset = dataset
        self.lora = lora
        self._lock = threading.RLock()
        self._oracle_model_state: dict[str, Tensor] | None = None
        self._oracle_adapter_state: dict[str, Tensor] | None = None
        self._finalization_model: torch.nn.Module | None = None
        self._finalization_optimizer: torch.optim.AdamW | None = None
        self._reference_state: dict[str, Tensor] | None = None
        self._initial_model_snapshot: bytes | None = None
        self._initial_adapter_snapshot: bytes | None = None
        self._base_layer_bundle_snapshots: dict[str, bytes] = {}
        self._completed_checkpoint_snapshots: dict[str, bytes] | None = None
        self._oracle_artifact_snapshots: dict[str, bytes] = {}
        self._accepted_result_snapshots: dict[str, bytes] = {}
        self._pending_state_migration = False
        self._pending_lock_migration = False
        self.participants = ParticipantRegistry.from_payload(
            state["participants"],  # type: ignore[arg-type]
            campaign_id=str(campaign.campaign["id"]),
        )
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        campaign: CampaignConfig,
        state_dir: str | Path,
        worker_count: int,
        participants: ParticipantRegistry,
        lease_seconds: int = 60,
        resume_from: str | Path | None = None,
        dataset: PackedDataset | None = None,
        lora: LoadedLoRAManifest | None = None,
        publish_base_layer_bundle: bool = False,
        numerical_profile: str = EXACT_CPU_FP32_PROFILE,
    ) -> GlobalStepCoordinator:
        if worker_count < 2:
            raise ValueError("multi-worker proof requires at least two workers")
        if campaign.training.batch_size % worker_count:
            raise ValueError("training batch size must be divisible by worker count")
        if lease_seconds <= 0:
            raise ValueError("lease duration must be positive")
        if participants.campaign_id != campaign.campaign["id"]:
            raise ValueError("participant registry campaign mismatch")
        if lora is not None and lora.campaign != campaign:
            raise ValueError("LoRA manifest campaign does not match coordinator campaign")
        if publish_base_layer_bundle and lora is None:
            raise ValueError("base layer bundles require a frozen-base LoRA campaign")
        profile = _validated_numerical_profile(numerical_profile)
        if lora is None and profile == INT8_FROZEN_LINEAR_PROFILE:
            raise ValueError("int8 numerical profile requires a LoRA campaign")
        checkpoint_profile = _checkpoint_numerical_profile(profile)
        validate_dataset_artifacts(campaign, dataset)

        state_dir = Path(state_dir)
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"coordinator state directory is not empty: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "oracle-gradients").mkdir()
        (state_dir / "results").mkdir()

        base_step = 0
        dataset_cursor = 0
        loss_history: list[float] = []
        if lora is not None:
            if resume_from is None:
                initial_checkpoint = run_lora_training(
                    lora,
                    state_dir / "base-checkpoint",
                    target_steps=0,
                    dataset=dataset,
                    numerical_profile=checkpoint_profile,
                )
                (
                    initial_model,
                    initial_optimizer,
                    base_step,
                    dataset_cursor,
                    loss_history,
                ) = load_lora_checkpoint(
                    lora,
                    initial_checkpoint.checkpoint_dir,
                    expected_numerical_profile=profile,
                )
                resume_state_sha256 = initial_checkpoint.checkpoint_sha256
                source_weight_checkpoint_sha256 = (
                    initial_checkpoint.weight_checkpoint_sha256
                )
            else:
                source_checkpoint = Path(resume_from)
                source_state_bytes = _safe_artifact_snapshot(
                    source_checkpoint,
                    "state.json",
                    "LoRA resume checkpoint state file",
                )
                source_state = json.loads(
                    source_state_bytes,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if (
                    not isinstance(source_state, Mapping)
                    or not isinstance(source_state.get("adapter"), Mapping)
                    or not isinstance(source_state.get("optimizer"), Mapping)
                ):
                    raise ValueError("LoRA resume checkpoint state is invalid")
                resume_state_sha256 = str(source_state["checkpoint_sha256"])
                source_weight_checkpoint_sha256 = str(
                    source_state["weight_checkpoint_sha256"]
                )
                base_checkpoint_dir = state_dir / "base-checkpoint"
                base_checkpoint_dir.mkdir()
                copied_artifacts = _copy_checkpoint_artifacts(
                    source_checkpoint,
                    base_checkpoint_dir,
                    (
                        source_state["adapter"]["file"],
                        source_state["optimizer"]["file"],
                    ),
                )
                if (
                    hashlib.sha256(
                        copied_artifacts[source_state["adapter"]["file"]]
                    ).hexdigest()
                    != source_state["adapter"]["file_sha256"]
                    or hashlib.sha256(
                        copied_artifacts[source_state["optimizer"]["file"]]
                    ).hexdigest()
                    != source_state["optimizer"]["sha256"]
                ):
                    raise ValueError("LoRA resume checkpoint artifact identity mismatch")
                _atomic_bytes(base_checkpoint_dir / "state.json", source_state_bytes)
                (
                    initial_model,
                    initial_optimizer,
                    base_step,
                    dataset_cursor,
                    loss_history,
                ) = load_lora_checkpoint(
                    lora,
                    base_checkpoint_dir,
                    expected_numerical_profile=profile,
                )
                retained_state_bytes = _safe_artifact_snapshot(
                    base_checkpoint_dir,
                    "state.json",
                    "retained LoRA resume checkpoint state",
                )
                if retained_state_bytes != source_state_bytes:
                    retained_state = json.loads(
                        retained_state_bytes,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                    for section in ("adapter", "optimizer"):
                        if isinstance(retained_state.get(section), Mapping):
                            _validated_relative_artifact_name(
                                retained_state[section].get("file"),
                                "resume checkpoint artifact",
                            )
                    raise ValueError("retained LoRA checkpoint state changed after admission")
            dense_base = build_model(campaign)
            save_safetensors_file(
                {
                    name: tensor.detach().cpu().contiguous()
                    for name, tensor in sorted(dense_base.state_dict().items())
                },
                str(state_dir / "model.safetensors"),
            )
            initial_adapters = {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in adapter_named_parameters(initial_model).items()
            }
            save_safetensors_file(
                initial_adapters,
                str(state_dir / "adapter.safetensors"),
            )
            initial_adapter_sha256 = tensor_sha256(initial_adapters)
            checkpoint_sha256 = lora_weight_checkpoint_sha256(
                lora,
                initial_adapter_sha256,
                numerical_profile=checkpoint_profile,
            )
            if checkpoint_sha256 != source_weight_checkpoint_sha256:
                raise ValueError("LoRA source weight checkpoint identity mismatch")
            run_lora_training(
                lora,
                state_dir / "reference-step-1",
                target_steps=base_step + 1,
                resume_from=state_dir / "base-checkpoint",
                dataset=dataset,
                numerical_profile=checkpoint_profile,
            )
        else:
            initial_adapter_sha256 = None
            resume_state_sha256 = None
            if resume_from is None:
                initial_model = build_model(campaign)
                initial_optimizer = _create_optimizer(
                    initial_model,
                    campaign.training,
                )
                save_safetensors_file(
                    {
                        name: tensor.detach().cpu().contiguous()
                        for name, tensor in sorted(initial_model.state_dict().items())
                    },
                    str(state_dir / "model.safetensors"),
                )
            else:
                source_checkpoint = Path(resume_from)
                base_checkpoint_dir = state_dir / "base-checkpoint"
                base_checkpoint_dir.mkdir()
                source_state_bytes = _safe_artifact_snapshot(
                    source_checkpoint,
                    "state.json",
                    "dense resume checkpoint state file",
                )
                source_state = json.loads(
                    source_state_bytes,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if (
                    not isinstance(source_state, Mapping)
                    or not isinstance(source_state.get("model"), Mapping)
                    or not isinstance(source_state.get("optimizer"), Mapping)
                ):
                    raise ValueError("dense resume checkpoint state is invalid")
                resume_state_sha256 = hashlib.sha256(source_state_bytes).hexdigest()
                copied_artifacts = _copy_checkpoint_artifacts(
                    source_checkpoint,
                    base_checkpoint_dir,
                    (
                        source_state["model"]["file"],
                        source_state["optimizer"]["file"],
                    ),
                )
                if (
                    hashlib.sha256(
                        copied_artifacts[source_state["model"]["file"]]
                    ).hexdigest()
                    != source_state["model"]["sha256"]
                    or hashlib.sha256(
                        copied_artifacts[source_state["optimizer"]["file"]]
                    ).hexdigest()
                    != source_state["optimizer"]["sha256"]
                ):
                    raise ValueError("dense resume checkpoint artifact identity mismatch")
                _atomic_bytes(base_checkpoint_dir / "state.json", source_state_bytes)
                (
                    initial_model,
                    initial_optimizer,
                    base_step,
                    dataset_cursor,
                    loss_history,
                ) = _load_checkpoint(campaign, base_checkpoint_dir)
                retained_state_bytes = _safe_artifact_snapshot(
                    base_checkpoint_dir,
                    "state.json",
                    "retained dense resume checkpoint state",
                )
                if retained_state_bytes != source_state_bytes:
                    retained_state = json.loads(
                        retained_state_bytes,
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                    for section in ("model", "optimizer"):
                        if isinstance(retained_state.get(section), Mapping):
                            _validated_relative_artifact_name(
                                retained_state[section].get("file"),
                                "resume checkpoint artifact",
                            )
                    raise ValueError("retained dense checkpoint state changed after admission")
                copied_model_name = _validated_relative_artifact_name(
                    source_state["model"]["file"],
                    "base checkpoint model artifact",
                )
                _atomic_bytes(
                    state_dir / "model.safetensors",
                    copied_artifacts[copied_model_name],
                )
            checkpoint_sha256 = _sha256_file(state_dir / "model.safetensors")
            run_training(
                campaign,
                state_dir / "reference-step-1",
                target_steps=base_step + 1,
                resume_from=(
                    state_dir / "base-checkpoint" if resume_from is not None else None
                ),
                dataset=dataset,
            )

        base_layer_bundle: dict[str, object] | None = None
        if publish_base_layer_bundle:
            if lora is None:
                raise RuntimeError("base layer bundle publication lost its LoRA contract")
            exported_bundle = export_base_layer_bundle(
                campaign,
                lora.config,
                state_dir / "model.safetensors",
                _sha256_file(state_dir / "model.safetensors"),
                state_dir / "base-layer-bundle",
            )
            base_layer_bundle = base_layer_bundle_artifact_contract(
                exported_bundle.output_dir,
                exported_bundle.manifest_sha256,
                lora.config.base_model_sha256,
            )

        inputs, targets = fixture_batch(campaign, dataset_cursor, dataset)
        rows_per_assignment = campaign.training.batch_size // worker_count
        assignments: list[dict[str, object]] = []
        oracle_snapshots: dict[str, bytes] = {}
        adapter_contract: dict[str, object] | None = None
        if lora is not None:
            adapter_contract = {
                "format": lora.config.format,
                "rank": lora.config.rank,
                "alpha": lora.config.alpha,
                "dropout": lora.config.dropout,
                "targets": list(lora.config.targets),
                "tensor_order": list(initial_adapters),
                "tensor_count": len(initial_adapters),
                "value_count": sum(tensor.numel() for tensor in initial_adapters.values()),
            }
        for index in range(worker_count):
            start = index * rows_per_assignment
            end = start + rows_per_assignment
            assignment_inputs = inputs[start:end].contiguous()
            assignment_targets = targets[start:end].contiguous()
            if lora is not None:
                model = build_profiled_lora_model(
                    campaign,
                    lora.config,
                    profile,
                )
                load_adapter_state(model, initial_adapters)
                submitted = compute_adapter_gradients(
                    model,
                    assignment_inputs,
                    assignment_targets,
                )
                loss_sum_value = submitted.loss_sum
                gradients = dict(submitted.gradients)
            else:
                model = build_model(campaign)
                model.load_state_dict(initial_model.state_dict())
                loss_sum = F.cross_entropy(
                    model(assignment_inputs).reshape(-1, campaign.model.vocabulary_size),
                    assignment_targets.reshape(-1),
                    reduction="sum",
                )
                loss_sum.backward()
                loss_sum_value = float(loss_sum.detach())
                gradients = {
                    name: parameter.grad.detach().cpu().contiguous()
                    for name, parameter in sorted(model.named_parameters())
                    if parameter.grad is not None
                }
            basis = {
                "campaign_id": campaign.campaign["id"],
                "checkpoint_sha256": checkpoint_sha256,
                "training_method": (
                    "frozen-base-lora" if lora is not None else "dense"
                ),
                "numerical_profile": profile,
                "dataset_revision": (
                    dataset.revision if dataset is not None else "synthetic-fixture-v1"
                ),
                "model": {
                    "vocab_size": campaign.model.vocabulary_size,
                    "context_length": campaign.model.context_length,
                    "d_model": campaign.model.width,
                    "num_heads": campaign.model.heads,
                    "num_layers": campaign.model.layers,
                    "d_ff": campaign.model.mlp_width,
                },
                "global_step": base_step,
                "data_range": [dataset_cursor + start, dataset_cursor + end],
                "input_ids": assignment_inputs.reshape(-1).tolist(),
                "input_shape": list(assignment_inputs.shape),
                "target_ids": assignment_targets.reshape(-1).tolist(),
                "target_shape": list(assignment_targets.shape),
                "loss_weight_sum": assignment_targets.numel(),
            }
            if lora is not None:
                basis.update(
                    {
                        "lora_manifest_sha256": lora.manifest_sha256,
                        "base_model_sha256": lora.config.base_model_sha256,
                        "adapter_sha256": initial_adapter_sha256,
                        "weight_checkpoint_sha256": checkpoint_sha256,
                        "resume_state_sha256": resume_state_sha256,
                    }
                )
                if base_layer_bundle is not None:
                    basis["base_layer_bundle_manifest_sha256"] = base_layer_bundle[
                        "manifest_sha256"
                    ]
            assignment_id = hashlib.sha256(
                json.dumps(basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            oracle_file = f"{assignment_id}.safetensors"
            oracle_bytes = save_safetensors(gradients)
            _atomic_bytes(state_dir / "oracle-gradients" / oracle_file, oracle_bytes)
            oracle_snapshots[assignment_id] = oracle_bytes
            assignments.append(
                {
                    "assignment_id": assignment_id,
                    **basis,
                    "parameter_count": campaign.model.parameters,
                    "trainable_parameter_count": (
                        adapter_contract["value_count"]
                        if adapter_contract is not None
                        else campaign.model.parameters
                    ),
                    "adapter": adapter_contract,
                    "expected_loss_sum": loss_sum_value,
                    "oracle_file": oracle_file,
                    "oracle_file_sha256": hashlib.sha256(oracle_bytes).hexdigest(),
                    "oracle_tensor_sha256": tensor_sha256(gradients),
                    "oracle_file_size": len(oracle_bytes),
                    "state": "open",
                    "attempt": 0,
                    "leased_by": None,
                    "contributor_id": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "result_file": None,
                    "result_file_sha256": None,
                    "result_tensor_sha256": None,
                    "accepted_loss_sum": None,
                    "runtime_backend": None,
                    "gradient_metrics": None,
                }
            )

        state: dict[str, object] = {
            "format": "orcacolony_global_step_v1",
            "campaign_id": campaign.campaign["id"],
            "campaign_revision": _revision(_campaign_payload(campaign)),
            "training_method": (
                "frozen-base-lora" if lora is not None else "dense"
            ),
            "lora_manifest_sha256": (
                lora.manifest_sha256 if lora is not None else None
            ),
            "base_model_sha256": (
                lora.config.base_model_sha256 if lora is not None else checkpoint_sha256
            ),
            "initial_adapter_sha256": initial_adapter_sha256,
            "resume_state_sha256": resume_state_sha256,
            "numerical_profile": profile,
            "base_layer_bundle": base_layer_bundle,
            "participants": participants.as_payload(),
            "participants_revision": participants.revision,
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_revision": (
                dataset.revision if dataset is not None else "synthetic-fixture-v1"
            ),
            "worker_count": worker_count,
            "lease_seconds": lease_seconds,
            "state": "waiting_for_results",
            "step": base_step,
            "base_step": base_step,
            "dataset_cursor": dataset_cursor,
            "loss_history": loss_history,
            "has_base_checkpoint": lora is not None or resume_from is not None,
            "result_protocol_revision": 3 if lora is not None else 2,
            "accepted_result_identity_revision": 1,
            "assignments": assignments,
            "model_sha256": None,
            "adapter_sha256": None,
            "result_dataset_cursor": None,
            "result_loss_history": None,
            "result_resume_state_sha256": None,
            "result_weight_checkpoint_sha256": None,
            "result_checkpoint_sha256": None,
            "checkpoint_metrics": None,
        }
        _atomic_json(state_dir / "global-state.json", state)
        coordinator = cls(campaign, state_dir, state, dataset, lora=lora)
        coordinator._oracle_artifact_snapshots = oracle_snapshots
        coordinator._initial_model_snapshot = _safe_artifact_snapshot(
            state_dir,
            "model.safetensors",
            "global-step initial model",
        )
        coordinator._oracle_model_state = _owned_safetensors(
            coordinator._initial_model_snapshot,
            "global-step initial model",
        )
        if lora is not None:
            coordinator._initial_adapter_snapshot = _safe_artifact_snapshot(
                state_dir,
                "adapter.safetensors",
                "global-step initial adapter",
            )
            coordinator._oracle_adapter_state = {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in initial_adapters.items()
            }
        coordinator._retain_base_layer_bundle_snapshots()
        coordinator._finalization_model = initial_model
        coordinator._finalization_optimizer = initial_optimizer
        reference_name = (
            "adapter.safetensors" if lora is not None else "model.safetensors"
        )
        coordinator._reference_state = _owned_safetensors(
            _safe_artifact_snapshot(
                state_dir / "reference-step-1",
                reference_name,
                "global-step reference checkpoint",
            ),
            "global-step reference checkpoint",
        )
        coordinator._write_campaign_lock()
        coordinator._write_accepted_ledger()
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
        expected_worker_count: int | None = None,
        persist_migrations: bool = True,
        finalize_ready: bool = True,
    ) -> GlobalStepCoordinator:
        validate_dataset_artifacts(campaign, dataset)
        state_dir = Path(state_dir)
        state = json.loads(
            _safe_artifact_snapshot(
                state_dir,
                "global-state.json",
                "global-step state",
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(state, dict):
            raise ValueError("global-step state must be a JSON object")
        persisted_state_fields = frozenset(state)
        if state.get("format") != "orcacolony_global_step_v1":
            raise ValueError("unsupported global-step state format")
        obsolete_migration_fields = {
            "participants_revision",
            "training_method",
            "result_protocol_revision",
        } - persisted_state_fields
        if obsolete_migration_fields:
            if "numerical_profile" not in persisted_state_fields:
                raise ValueError("combined global-step migration is invalid")
            raise ValueError("unsupported legacy global-step migration schema")
        requested_profile = _validated_numerical_profile(numerical_profile)
        numerical_profile_migrated = "numerical_profile" not in state
        if numerical_profile_migrated:
            if requested_profile != EXACT_CPU_FP32_PROFILE:
                raise ValueError("legacy global-step numerical profile is FP32")
            if any(
                field not in state
                for field in (
                    "participants_revision",
                    "training_method",
                    "result_protocol_revision",
                    "accepted_result_identity_revision",
                )
            ):
                raise ValueError("combined global-step migration is invalid")
            expected_predecessor_fields = set(_GLOBAL_STATE_FIELDS) - {
                "numerical_profile"
            }
            if state.get("state") == "step_complete":
                expected_predecessor_fields |= {"loss_sum", "loss_weight_sum"}
            if persisted_state_fields != expected_predecessor_fields:
                raise ValueError("legacy global-step numerical-profile schema is invalid")
            predecessor_assignments = state.get("assignments")
            if not isinstance(predecessor_assignments, list):
                raise ValueError("legacy global-step assignments are invalid")
            for assignment in predecessor_assignments:
                if not isinstance(assignment, dict):
                    raise ValueError("legacy global-step assignment is invalid")
                expected_assignment_fields = set(
                    _assignment_fields(
                        lora=lora is not None,
                        base_layer_bundle=state.get("base_layer_bundle") is not None,
                        accepted=assignment.get("state") == "accepted",
                    )
                ) - {"numerical_profile"}
                if set(assignment) != expected_assignment_fields:
                    raise ValueError(
                        "legacy global-step numerical-profile assignment schema is invalid"
                    )
            state["numerical_profile"] = EXACT_CPU_FP32_PROFILE
            for assignment in state.get("assignments", []):
                if isinstance(assignment, dict):
                    assignment["numerical_profile"] = EXACT_CPU_FP32_PROFILE
        stored_profile = _validated_numerical_profile(state.get("numerical_profile"))
        if stored_profile != requested_profile:
            raise ValueError("global-step numerical profile does not match configuration")
        if state.get("campaign_id") != campaign.campaign["id"]:
            raise ValueError("global-step campaign does not match configuration")
        training_method = state.get("training_method", "dense")
        expected_training_method = (
            "frozen-base-lora" if lora is not None else "dense"
        )
        if training_method != expected_training_method:
            raise ValueError("global-step training method does not match configuration")
        if lora is not None:
            if lora.campaign != campaign:
                raise ValueError("LoRA manifest campaign does not match coordinator campaign")
            if state.get("lora_manifest_sha256") != lora.manifest_sha256:
                raise ValueError("global-step LoRA manifest digest mismatch")
        expected_dataset_revision = (
            dataset.revision if dataset is not None else "synthetic-fixture-v1"
        )
        if state.get("dataset_revision", "synthetic-fixture-v1") != expected_dataset_revision:
            raise ValueError("global-step dataset revision mismatch")
        campaign_revision = _revision(_campaign_payload(campaign))
        migrated = "participants_revision" not in state
        profile_migrated = "training_method" not in state
        if migrated or profile_migrated or numerical_profile_migrated:
            state.setdefault("base_step", 0)
            state.setdefault("dataset_cursor", 0)
            state.setdefault("loss_history", [])
            state.setdefault("has_base_checkpoint", False)
            state.setdefault("training_method", "dense")
            state.setdefault("lora_manifest_sha256", None)
            state.setdefault("base_model_sha256", state["checkpoint_sha256"])
            state.setdefault("initial_adapter_sha256", None)
            state.setdefault("resume_state_sha256", None)
            state.setdefault("adapter_sha256", None)
            state.setdefault("result_dataset_cursor", None)
            state.setdefault("result_loss_history", None)
            state.setdefault("result_resume_state_sha256", None)
            state.setdefault("result_weight_checkpoint_sha256", state.get("model_sha256"))
            state.setdefault("result_checkpoint_sha256", state.get("model_sha256"))
            if migrated:
                state["campaign_revision"] = campaign_revision
                state["participants"] = participants.as_payload()
                state["participants_revision"] = participants.revision
                for assignment in state["assignments"]:
                    worker_id = assignment.get("leased_by")
                    if worker_id is None:
                        assignment["contributor_id"] = None
                        continue
                    participant = participants.participant_for_worker(str(worker_id))
                    if participant is None:
                        raise ValueError(
                            f"existing worker is not allowlisted: {worker_id}"
                        )
                    assignment["contributor_id"] = participant.contributor_id
            elif state.get("participants_revision") != participants.revision:
                raise ValueError("participant revision mismatch")
        elif state.get("participants_revision") != participants.revision:
            raise ValueError("participant revision mismatch")
        if state.get("campaign_revision") != campaign_revision:
            raise ValueError("campaign revision mismatch")
        protocol_migrated = (
            "result_protocol_revision" not in state
            and state["state"] != "step_complete"
        )
        if protocol_migrated:
            state["result_protocol_revision"] = 2
        result_identity_migrated = "accepted_result_identity_revision" not in state
        if result_identity_migrated:
            if stored_profile != EXACT_CPU_FP32_PROFILE:
                raise ValueError(
                    "accepted result identity is missing; legacy migration is "
                    "exact-FP32 only"
                )
            expected_predecessor_state_fields = _GLOBAL_STATE_FIELDS - (
                {"accepted_result_identity_revision"}
                | _RESULT_STATE_IDENTITY_FIELDS
            )
            if state.get("state") == "step_complete":
                expected_predecessor_state_fields |= {"loss_sum", "loss_weight_sum"}
            if (
                migrated
                or profile_migrated
                or numerical_profile_migrated
                or protocol_migrated
                or persisted_state_fields != expected_predecessor_state_fields
            ):
                raise ValueError("legacy accepted-result state schema is invalid")
            assignments = state.get("assignments")
            if not isinstance(assignments, list):
                raise ValueError("legacy accepted-result assignment schema is invalid")
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    raise ValueError("legacy accepted-result assignment schema is invalid")
                expected_fields = _assignment_fields(
                    lora=lora is not None,
                    base_layer_bundle=state.get("base_layer_bundle") is not None,
                    accepted=assignment.get("state") == "accepted",
                ) - _ASSIGNMENT_IDENTITY_FIELDS
                if frozenset(assignment) != expected_fields:
                    raise ValueError("legacy accepted-result assignment schema is invalid")
            state["accepted_result_identity_revision"] = 1
            for field in _RESULT_STATE_IDENTITY_FIELDS:
                state[field] = None
            if state.get("state") == "step_complete":
                result_state_bytes = _safe_artifact_snapshot(
                    state_dir / "checkpoint",
                    "state.json",
                    "legacy completed checkpoint state",
                )
                result_state = json.loads(
                    result_state_bytes,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if not isinstance(result_state, Mapping):
                    raise ValueError("legacy completed checkpoint state is invalid")
                state["result_dataset_cursor"] = result_state.get("dataset_cursor")
                state["result_loss_history"] = result_state.get("loss_history")
                if lora is not None:
                    state["result_resume_state_sha256"] = result_state.get(
                        "checkpoint_sha256"
                    )
                    state["result_weight_checkpoint_sha256"] = result_state.get(
                        "weight_checkpoint_sha256"
                    )
                    state["result_checkpoint_sha256"] = result_state.get(
                        "checkpoint_sha256"
                    )
                else:
                    model_state = result_state.get("model")
                    if not isinstance(model_state, Mapping):
                        raise ValueError("legacy completed checkpoint model is invalid")
                    state["result_resume_state_sha256"] = hashlib.sha256(
                        result_state_bytes
                    ).hexdigest()
                    state["result_weight_checkpoint_sha256"] = None
                    state["result_checkpoint_sha256"] = model_state.get("sha256")
        elif (
            type(state["accepted_result_identity_revision"]) is not int
            or state["accepted_result_identity_revision"] != 1
        ):
            raise ValueError("unsupported accepted-result identity revision")
        if type(state.get("lease_seconds")) is not int or state["lease_seconds"] <= 0:
            raise ValueError("global-step lease duration is invalid")
        expected_result_protocol_revision = 3 if lora is not None else 2
        if (
            type(state.get("result_protocol_revision")) is not int
            or state["result_protocol_revision"] != expected_result_protocol_revision
        ):
            raise ValueError("global-step result protocol revision is invalid")
        worker_count = state.get("worker_count")
        assignments = state.get("assignments")
        if (
            type(worker_count) is not int
            or worker_count < 2
            or not isinstance(assignments, list)
            or len(assignments) != worker_count
        ):
            raise ValueError("global-step assignment coverage is incomplete")
        if expected_worker_count is not None and worker_count != expected_worker_count:
            raise ValueError("global-step worker count differs from parent campaign")
        assignment_ids = [
            assignment.get("assignment_id")
            if isinstance(assignment, Mapping)
            else None
            for assignment in assignments
        ]
        if (
            any(
                not isinstance(assignment_id, str)
                or _SHA256_HEX.fullmatch(assignment_id) is None
                for assignment_id in assignment_ids
            )
            or len(set(assignment_ids)) != worker_count
        ):
            raise ValueError("global-step assignment set identity is invalid")
        if not (
            migrated
            or profile_migrated
            or numerical_profile_migrated
            or protocol_migrated
            or result_identity_migrated
        ):
            expected_current_state_fields = set(_GLOBAL_STATE_FIELDS)
            if state.get("state") == "step_complete":
                expected_current_state_fields |= {"loss_sum", "loss_weight_sum"}
            if persisted_state_fields != expected_current_state_fields:
                raise ValueError("current global-step state schema is invalid")
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    raise ValueError("current global-step assignment schema is invalid")
                expected_fields = _assignment_fields(
                    lora=lora is not None,
                    base_layer_bundle=state.get("base_layer_bundle") is not None,
                    accepted=assignment.get("state") == "accepted",
                )
                if frozenset(assignment) != expected_fields:
                    if assignment.get("state") == "accepted" and (
                        _ASSIGNMENT_IDENTITY_FIELDS - frozenset(assignment)
                    ):
                        raise ValueError(
                            "current accepted-result assignment schema is incomplete"
                        )
                    raise ValueError("current global-step assignment schema is invalid")
        if state["state"] != "step_complete" and any(
            state.get(field) is not None
            for field in (
                "model_sha256",
                "adapter_sha256",
                "result_dataset_cursor",
                "result_loss_history",
                "result_resume_state_sha256",
                "result_weight_checkpoint_sha256",
                "result_checkpoint_sha256",
                "checkpoint_metrics",
            )
        ):
            raise ValueError("unfinished global-step result authority is not empty")
        if type(state.get("has_base_checkpoint")) is not bool or (
            lora is not None and not state["has_base_checkpoint"]
        ):
            raise ValueError("global-step base checkpoint marker is invalid")
        model_bytes = _safe_artifact_snapshot(
            state_dir,
            "model.safetensors",
            "global-step initial model",
        )
        oracle_model_state = _owned_safetensors(
            model_bytes,
            "global-step initial model",
        )
        oracle_adapter_state: dict[str, Tensor] | None = None
        if lora is None:
            if (
                hashlib.sha256(model_bytes).hexdigest()
                != state["checkpoint_sha256"]
            ):
                raise ValueError("global-step checkpoint digest mismatch")
            if state.get("has_base_checkpoint"):
                (
                    finalization_model,
                    finalization_optimizer,
                    checkpoint_step,
                    checkpoint_cursor,
                    checkpoint_loss_history,
                ) = _load_checkpoint(campaign, state_dir / "base-checkpoint")
            else:
                finalization_model = build_model(campaign)
                finalization_model.load_state_dict(oracle_model_state, strict=True)
                finalization_optimizer = _create_optimizer(
                    finalization_model,
                    campaign.training,
                )
                checkpoint_step = 0
                checkpoint_cursor = 0
                checkpoint_loss_history = []
        else:
            base_sha256 = tensor_sha256(oracle_model_state)
            if base_sha256 != lora.config.base_model_sha256:
                raise ValueError("global-step LoRA base model digest mismatch")
            adapter_bytes = _safe_artifact_snapshot(
                state_dir,
                "adapter.safetensors",
                "global-step initial adapter",
            )
            oracle_adapter_state = _owned_safetensors(
                adapter_bytes,
                "global-step initial adapter",
            )
            adapter_sha256 = tensor_sha256(oracle_adapter_state)
            if adapter_sha256 != state["initial_adapter_sha256"]:
                raise ValueError("global-step initial adapter digest mismatch")
            if (
                lora_weight_checkpoint_sha256(
                    lora,
                    adapter_sha256,
                    numerical_profile=_checkpoint_numerical_profile(stored_profile),
                )
                != state["checkpoint_sha256"]
            ):
                raise ValueError("global-step LoRA checkpoint digest mismatch")
            (
                finalization_model,
                finalization_optimizer,
                checkpoint_step,
                checkpoint_cursor,
                checkpoint_loss_history,
            ) = load_lora_checkpoint(
                lora,
                state_dir / "base-checkpoint",
                expected_numerical_profile=stored_profile,
            )
            base_checkpoint_state = json.loads(
                _safe_artifact_snapshot(
                    state_dir / "base-checkpoint",
                    "state.json",
                    "global-step base checkpoint state",
                ),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            if (
                base_checkpoint_state["weight_checkpoint_sha256"]
                != state["checkpoint_sha256"]
                or base_checkpoint_state["checkpoint_sha256"]
                != state.get("resume_state_sha256")
            ):
                raise ValueError("global-step LoRA resume-state identity mismatch")
        if lora is None:
            loaded_initial_state = {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in finalization_model.state_dict().items()
            }
            if tensor_sha256(loaded_initial_state) != tensor_sha256(oracle_model_state):
                raise ValueError("loaded dense finalization snapshot changed")
        else:
            loaded_adapter_state = {
                name: parameter.detach().cpu().clone().contiguous()
                for name, parameter in adapter_named_parameters(
                    finalization_model
                ).items()
            }
            if tensor_sha256(loaded_adapter_state) != tensor_sha256(
                oracle_adapter_state
            ):
                raise ValueError("loaded LoRA finalization snapshot changed")
        if (
            type(state.get("base_step")) is not int
            or state["base_step"] != checkpoint_step
            or type(state.get("step")) is not int
            or state["step"]
            != checkpoint_step + (1 if state.get("state") == "step_complete" else 0)
            or type(state.get("dataset_cursor")) is not int
            or not isinstance(state.get("loss_history"), list)
            or (
                state.get("state") != "step_complete"
                and (
                    state["dataset_cursor"] != checkpoint_cursor
                    or not _exact_json_equal(
                        state["loss_history"],
                        checkpoint_loss_history,
                    )
                )
            )
        ):
            raise ValueError("global-step base checkpoint progress mismatch")
        stored_base_layer_bundle = state.get("base_layer_bundle")
        if stored_base_layer_bundle is not None:
            if lora is None or not isinstance(stored_base_layer_bundle, Mapping):
                raise ValueError("global-step base layer bundle contract is invalid")
            manifest_sha256 = stored_base_layer_bundle.get("manifest_sha256")
            if not isinstance(manifest_sha256, str):
                raise ValueError("global-step base layer bundle manifest digest is invalid")
            canonical_base_layer_bundle = base_layer_bundle_artifact_contract(
                state_dir / "base-layer-bundle",
                manifest_sha256,
                lora.config.base_model_sha256,
            )
            if canonical_base_layer_bundle != stored_base_layer_bundle:
                raise ValueError("global-step base layer bundle contract differs")
        for assignment in state["assignments"]:
            if assignment.get("numerical_profile") != stored_profile:
                raise ValueError("assignment numerical profile differs")
            assignment_bundle_sha256 = assignment.get(
                "base_layer_bundle_manifest_sha256"
            )
            expected_bundle_sha256 = (
                stored_base_layer_bundle["manifest_sha256"]
                if isinstance(stored_base_layer_bundle, Mapping)
                else None
            )
            if assignment_bundle_sha256 != expected_bundle_sha256:
                raise ValueError("assignment base layer bundle identity differs")
        coordinator = cls(campaign, state_dir, state, dataset, lora=lora)
        coordinator._oracle_model_state = oracle_model_state
        coordinator._oracle_adapter_state = oracle_adapter_state
        coordinator._finalization_model = finalization_model
        coordinator._finalization_optimizer = finalization_optimizer
        coordinator._initial_model_snapshot = model_bytes
        coordinator._initial_adapter_snapshot = (
            adapter_bytes if lora is not None else None
        )
        coordinator._retain_base_layer_bundle_snapshots()
        reference_name = (
            "adapter.safetensors" if lora is not None else "model.safetensors"
        )
        coordinator._reference_state = _owned_safetensors(
            _safe_artifact_snapshot(
                state_dir / "reference-step-1",
                reference_name,
                "global-step reference checkpoint",
            ),
            "global-step reference checkpoint",
        )
        state_changed = (
            migrated
            or profile_migrated
            or numerical_profile_migrated
            or protocol_migrated
            or result_identity_migrated
        )
        lock_path = state_dir / "campaign-lock.json"
        expected_lock = coordinator._campaign_lock_payload()
        expected_stored_lock = dict(expected_lock)
        if migrated:
            expected_stored_lock.pop("campaign_revision")
            expected_stored_lock.pop("participants_revision")
        if profile_migrated:
            for field in (
                "training_method",
                "lora_manifest_sha256",
                "base_model_sha256",
                "adapter_sha256",
                "resume_state_sha256",
                *_RESULT_STATE_IDENTITY_FIELDS,
            ):
                expected_stored_lock.pop(field)
        if numerical_profile_migrated:
            expected_stored_lock.pop("numerical_profile")
        if protocol_migrated:
            expected_stored_lock.pop("result_protocol_revision")
        if result_identity_migrated:
            expected_stored_lock.pop("accepted_result_identity_revision")
            expected_stored_lock.pop("dataset_cursor")
            expected_stored_lock.pop("worker_count")
            expected_stored_lock.pop("assignment_ids")
            for field in _RESULT_STATE_IDENTITY_FIELDS:
                expected_stored_lock.pop(field)
        stored_lock = json.loads(
            _safe_artifact_snapshot(
                state_dir,
                lock_path.name,
                "global-step campaign lock",
            ),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not _exact_json_equal(stored_lock, expected_stored_lock):
            raise ValueError("campaign lock mismatch")
        for assignment_index, assignment in enumerate(coordinator.assignments):
            if not isinstance(assignment, dict):
                raise ValueError("global-step assignment entry is invalid")
            _validate_persisted_assignment_lifecycle(assignment, participants)
            _validated_assignment_artifact_name(assignment)
            if result_identity_migrated:
                if any(field in assignment for field in _ASSIGNMENT_IDENTITY_FIELDS):
                    raise ValueError(
                        "legacy accepted-result assignment schema is invalid"
                    )
                if (
                    assignment.get("state") == "accepted"
                    and stored_profile != EXACT_CPU_FP32_PROFILE
                ):
                    raise ValueError("accepted result identity is missing")
            elif any(
                field not in assignment for field in _ASSIGNMENT_IDENTITY_FIELDS
            ):
                raise ValueError(
                    "current accepted-result assignment schema is incomplete"
                )
            elif assignment.get("state") != "accepted" and (
                assignment.get("result_file_sha256") is not None
                or assignment.get("result_tensor_sha256") is not None
            ):
                raise ValueError("unaccepted result identity is not empty")
            if "dataset_revision" not in assignment:
                assignment["dataset_revision"] = expected_dataset_revision
                state_changed = True
            if "model" not in assignment:
                assignment["model"] = {
                    "vocab_size": campaign.model.vocabulary_size,
                    "context_length": campaign.model.context_length,
                    "d_model": campaign.model.width,
                    "num_heads": campaign.model.heads,
                    "num_layers": campaign.model.layers,
                    "d_ff": campaign.model.mlp_width,
                }
                state_changed = True
            if "runtime_backend" not in assignment:
                assignment["runtime_backend"] = (
                    "legacy-unknown" if assignment["state"] == "accepted" else None
                )
                state_changed = True
            runtime_backend = assignment["runtime_backend"]
            if (
                runtime_backend in _LAYER_BUNDLE_RUNTIME_BACKENDS
                and stored_base_layer_bundle is None
            ):
                raise ValueError("persisted layer-bundle runtime was not assigned")
            if runtime_backend not in {None, "legacy-unknown"} and (
                _RUNTIME_NUMERICAL_PROFILE.get(str(runtime_backend)) != stored_profile
            ):
                raise ValueError("persisted runtime numerical profile differs")
            expected_loss_sum = assignment.get("expected_loss_sum")
            accepted_loss_sum = assignment.get("accepted_loss_sum")
            if type(expected_loss_sum) is not float or not math.isfinite(
                expected_loss_sum
            ):
                raise ValueError("persisted expected loss must be a finite JSON float")
            if assignment.get("state") == "accepted":
                if type(accepted_loss_sum) is not float or not math.isfinite(
                    accepted_loss_sum
                ):
                    raise ValueError(
                        "persisted accepted loss must be a finite JSON float"
                    )
            elif accepted_loss_sum is not None:
                raise ValueError("unaccepted result loss is not empty")
            recomputed_oracle = coordinator._recomputed_assignment_oracle(
                assignment,
                assignment_index,
                training_method_migrated=profile_migrated,
                numerical_profile_migrated=numerical_profile_migrated,
            )
            _, migrated_oracle_identity = coordinator._validated_oracle_gradients(
                assignment,
                migrate_legacy_identity=result_identity_migrated,
                recomputed=recomputed_oracle,
                retain_snapshot=True,
            )
            state_changed = state_changed or migrated_oracle_identity
            if result_identity_migrated:
                assignment["result_file_sha256"] = None
                assignment["result_tensor_sha256"] = None
            instrumentation = assignment.get("instrumentation")
            if instrumentation is not None:
                if not isinstance(instrumentation, Mapping) or set(instrumentation) != {
                    "format",
                    "worker_reported",
                    "coordinator_measured",
                } or instrumentation.get("format") != (
                    "orcacolony_assignment_instrumentation_v1"
                ):
                    raise ValueError("persisted assignment instrumentation is invalid")
                public_assignment = coordinator._public_assignment(assignment)
                resource_profile = public_assignment["resource_profile"]
                canonical_worker = _validate_worker_telemetry(
                    instrumentation.get("worker_reported"),  # type: ignore[arg-type]
                    resource_profile,  # type: ignore[arg-type]
                    str(assignment["runtime_backend"]),
                    public_assignment.get("base_layer_bundle"),  # type: ignore[arg-type]
                )
                if canonical_worker != instrumentation.get("worker_reported"):
                    raise ValueError("persisted worker telemetry is not canonical")
                measured = instrumentation.get("coordinator_measured")
                measured_fields = {
                    "model_artifact_bytes",
                    "adapter_artifact_bytes",
                    "oracle_gradient_artifact_bytes",
                    "result_upload_bytes",
                    "result_receive_seconds",
                    "result_storage_bytes",
                }
                if not isinstance(measured, Mapping) or set(measured) != measured_fields:
                    raise ValueError("persisted coordinator measurements are invalid")
                result_storage_bytes = len(
                    coordinator._accepted_result_bytes(
                        assignment,
                        retain_snapshot=True,
                    )
                )
                expected_bytes = {
                    "model_artifact_bytes": _expected_model_download_bytes(
                        resource_profile,  # type: ignore[arg-type]
                        str(assignment["runtime_backend"]),
                    ),
                    "adapter_artifact_bytes": resource_profile[
                        "adapter_download_bytes"
                    ],
                    "oracle_gradient_artifact_bytes": resource_profile[
                        "oracle_gradient_download_bytes"
                    ],
                    "result_upload_bytes": result_storage_bytes,
                    "result_storage_bytes": result_storage_bytes,
                }
                if any(measured[field] != value for field, value in expected_bytes.items()):
                    raise ValueError("persisted coordinator byte measurements changed")
                receive_seconds = measured["result_receive_seconds"]
                if receive_seconds is not None and (
                    isinstance(receive_seconds, bool)
                    or not isinstance(receive_seconds, (int, float))
                    or not math.isfinite(float(receive_seconds))
                    or float(receive_seconds) < 0
                    or float(receive_seconds) > 86_400
                ):
                    raise ValueError("persisted coordinator receive duration is invalid")
            if assignment["state"] == "accepted":
                _, migrated_result_identity = (
                    coordinator._validated_accepted_gradients(
                        assignment,
                        migrate_legacy_identity=(
                            result_identity_migrated
                            and stored_profile == EXACT_CPU_FP32_PROFILE
                        ),
                        retain_snapshot=True,
                    )
                )
                state_changed = state_changed or migrated_result_identity
        coordinator._pending_state_migration = state_changed
        coordinator._pending_lock_migration = expected_stored_lock != expected_lock
        was_complete = coordinator._state["state"] == "step_complete"
        if was_complete:
            coordinator._validate_completed_checkpoint_identity()
            coordinator._capture_completed_checkpoint_snapshots()
        if finalize_ready:
            coordinator.finalize_if_ready()
        if not was_complete and coordinator._state["state"] == "step_complete":
            coordinator._validate_completed_checkpoint_identity()
            coordinator._capture_completed_checkpoint_snapshots()
        if persist_migrations:
            coordinator.persist_validated_migrations()
        return coordinator

    @property
    def assignments(self) -> list[dict[str, object]]:
        return self._state["assignments"]  # type: ignore[return-value]

    @property
    def has_base_layer_bundle(self) -> bool:
        return self._state.get("base_layer_bundle") is not None

    def _write_state(self) -> None:
        _atomic_json(self.state_dir / "global-state.json", self._state)

    def _campaign_lock_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format": "orcacolony_campaign_lock_v1",
            "campaign_id": self._state["campaign_id"],
            "campaign_revision": self._state["campaign_revision"],
            "participants_revision": self._state["participants_revision"],
            "checkpoint_sha256": self._state["checkpoint_sha256"],
            "training_method": self._state.get("training_method", "dense"),
            "lora_manifest_sha256": self._state.get("lora_manifest_sha256"),
            "base_model_sha256": self._state.get(
                "base_model_sha256", self._state["checkpoint_sha256"]
            ),
            "adapter_sha256": self._state.get("initial_adapter_sha256"),
            "resume_state_sha256": self._state.get("resume_state_sha256"),
            "result_dataset_cursor": self._state.get("result_dataset_cursor"),
            "result_loss_history": self._state.get("result_loss_history"),
            "result_resume_state_sha256": self._state.get(
                "result_resume_state_sha256"
            ),
            "result_weight_checkpoint_sha256": self._state.get(
                "result_weight_checkpoint_sha256"
            ),
            "result_checkpoint_sha256": self._state.get(
                "result_checkpoint_sha256"
            ),
            "numerical_profile": self._state["numerical_profile"],
            "dataset_revision": self._state.get(
                "dataset_revision", "synthetic-fixture-v1"
            ),
            "global_step": self._state["base_step"],
            "dataset_cursor": self._state["dataset_cursor"],
            "worker_count": self._state["worker_count"],
            "assignment_ids": [
                assignment["assignment_id"] for assignment in self.assignments
            ],
            "assignment_protocol_revision": (
                2
                if self._state.get("training_method") == "frozen-base-lora"
                else 1
            ),
            "result_protocol_revision": self._state.get(
                "result_protocol_revision", 1
            ),
            "accepted_result_identity_revision": self._state[
                "accepted_result_identity_revision"
            ],
        }
        bundle = self._state.get("base_layer_bundle")
        if isinstance(bundle, Mapping):
            payload["base_layer_bundle_manifest_sha256"] = bundle[
                "manifest_sha256"
            ]
        return payload

    def _write_campaign_lock(self) -> None:
        _atomic_json(
            self.state_dir / "campaign-lock.json",
            self._campaign_lock_payload(),
        )

    def persist_validated_migrations(self) -> None:
        with self._lock:
            if self._pending_state_migration:
                self._write_state()
                self._pending_state_migration = False
            if self._pending_lock_migration:
                self._write_campaign_lock()
                self._pending_lock_migration = False
            self._write_accepted_ledger()

    def finalize_if_ready(self) -> None:
        with self._lock:
            if self._state["state"] == "step_complete" or not self._all_accepted():
                return
            original_state = copy.deepcopy(self._state)
            self._state["state"] = "ready_to_finalize"
            try:
                self._finalize_locked()
            except Exception:
                self._state = original_state
                self._completed_checkpoint_snapshots = None
                raise

    def _validate_completed_checkpoint_identity(self) -> None:
        if self._state["state"] != "step_complete":
            return
        checkpoint_state_bytes = _safe_artifact_snapshot(
            self.checkpoint_dir,
            "state.json",
            "completed checkpoint state",
        )
        checkpoint_state = json.loads(
            checkpoint_state_bytes,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError("completed checkpoint state is invalid")
        if self.lora is not None:
            checkpoint_model, _, _, _, _ = load_lora_checkpoint(
                self.lora,
                self.checkpoint_dir,
                expected_numerical_profile=str(self._state["numerical_profile"]),
            )
            checkpoint_parameters = {
                name: parameter.detach().cpu().clone().contiguous()
                for name, parameter in adapter_named_parameters(checkpoint_model).items()
            }
            adapter = checkpoint_state.get("adapter")
            if (
                checkpoint_state.get("base_model_sha256")
                != self._state["model_sha256"]
                or not isinstance(adapter, Mapping)
                or adapter.get("tensor_sha256") != self._state["adapter_sha256"]
                or checkpoint_state.get("weight_checkpoint_sha256")
                != self._state["result_weight_checkpoint_sha256"]
                or checkpoint_state.get("checkpoint_sha256")
                != self._state["result_checkpoint_sha256"]
                or checkpoint_state.get("checkpoint_sha256")
                != self._state["result_resume_state_sha256"]
            ):
                raise ValueError("completed LoRA global-step identity mismatch")
        else:
            checkpoint_model, _, _, _, _ = _load_checkpoint(
                self.campaign,
                self.checkpoint_dir,
            )
            checkpoint_parameters = {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in checkpoint_model.state_dict().items()
            }
            model = checkpoint_state.get("model")
            if (
                not isinstance(model, Mapping)
                or model.get("sha256") != self._state["model_sha256"]
                or hashlib.sha256(checkpoint_state_bytes).hexdigest()
                != self._state["result_resume_state_sha256"]
            ):
                raise ValueError("completed global-step checkpoint identity mismatch")
        if (
            type(self._state.get("result_dataset_cursor")) is not int
            or not isinstance(self._state.get("result_loss_history"), list)
            or not _exact_json_equal(
                checkpoint_state.get("step"),
                self._state["step"],
            )
            or not _exact_json_equal(
                checkpoint_state.get("dataset_cursor"),
                self._state["result_dataset_cursor"],
            )
            or not _exact_json_equal(
                checkpoint_state.get("loss_history"),
                self._state["result_loss_history"],
            )
        ):
            raise ValueError("completed global-step progress identity mismatch")
        expected_checkpoint_metrics = _tensor_metrics(
            self._reference_state,
            checkpoint_parameters,
        )
        _validate_checkpoint_profile_metrics(
            str(self._state["numerical_profile"]),
            expected_checkpoint_metrics,
        )
        if not _exact_json_equal(
            self._state.get("checkpoint_metrics"),
            expected_checkpoint_metrics,
        ):
            raise ValueError("completed global-step checkpoint metrics changed")
        ordered_assignments = sorted(
            self.assignments,
            key=lambda assignment: assignment["data_range"][0],
        )
        expected_loss_sum = 0.0
        expected_loss_weight_sum = 0
        for assignment in ordered_assignments:
            accepted_loss_sum = assignment.get("accepted_loss_sum")
            loss_weight_sum = assignment.get("loss_weight_sum")
            if (
                assignment.get("state") != "accepted"
                or type(accepted_loss_sum) is not float
                or type(loss_weight_sum) is not int
            ):
                raise ValueError("completed global-step assignment totals are invalid")
            expected_loss_sum += accepted_loss_sum
            expected_loss_weight_sum += loss_weight_sum
        if not _exact_json_equal(
            self._state.get("loss_sum"), expected_loss_sum
        ) or not _exact_json_equal(
            self._state.get("loss_weight_sum"), expected_loss_weight_sum
        ):
            raise ValueError("completed global-step totals changed")
        if (
            _safe_artifact_snapshot(
                self.checkpoint_dir,
                "state.json",
                "completed checkpoint state",
            )
            != checkpoint_state_bytes
        ):
            raise ValueError("completed checkpoint state changed during validation")

    def accepted_ledger_payload(self) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        for assignment in sorted(
            self.assignments,
            key=lambda value: value["data_range"][0],
        ):
            if assignment["state"] != "accepted":
                continue
            contributor_id = str(assignment["contributor_id"])
            participant = next(
                value
                for value in self.participants.participants
                if value.contributor_id == contributor_id
            )
            public_credit = (
                {"display_name": participant.display_name}
                if participant.public_credit
                else None
            )
            entries.append(
                {
                    "assignment_id": assignment["assignment_id"],
                    "global_step": assignment["global_step"],
                    "data_range": assignment["data_range"],
                    "attempt": assignment["attempt"],
                    "worker_id": assignment["leased_by"],
                    "contributor_id": contributor_id,
                    "public_credit": public_credit,
                    "loss_sum": assignment["accepted_loss_sum"],
                    "loss_weight_sum": assignment["loss_weight_sum"],
                    "runtime_backend": assignment["runtime_backend"],
                    "instrumentation": assignment.get("instrumentation"),
                    "dataset_revision": assignment["dataset_revision"],
                    "training_method": assignment.get("training_method", "dense"),
                    "numerical_profile": assignment.get(
                        "numerical_profile", self._state["numerical_profile"]
                    ),
                }
            )
        return {
            "format": "orcacolony_accepted_work_v1",
            "campaign_id": self._state["campaign_id"],
            "participants_revision": self._state["participants_revision"],
            "entries": entries,
        }

    def _write_accepted_ledger(self) -> None:
        _atomic_json(
            self.state_dir / "accepted-work.json",
            self.accepted_ledger_payload(),
        )

    def _all_accepted(self) -> bool:
        return all(assignment["state"] == "accepted" for assignment in self.assignments)

    def _recomputed_assignment_oracle(
        self,
        assignment: Mapping[str, object],
        index: int,
        *,
        training_method_migrated: bool,
        numerical_profile_migrated: bool,
    ) -> tuple[float, dict[str, Tensor]]:
        worker_count = self._state.get("worker_count")
        dataset_cursor = self._state.get("dataset_cursor")
        base_step = self._state.get("base_step")
        if (
            type(worker_count) is not int
            or worker_count <= 0
            or type(dataset_cursor) is not int
            or dataset_cursor < 0
            or type(base_step) is not int
            or base_step < 0
            or self.campaign.training.batch_size % worker_count != 0
        ):
            raise ValueError("global-step assignment partition state is invalid")
        inputs, targets = fixture_batch(
            self.campaign,
            dataset_cursor,
            self.dataset,
        )
        rows_per_assignment = self.campaign.training.batch_size // worker_count
        start = index * rows_per_assignment
        end = start + rows_per_assignment
        assignment_inputs = inputs[start:end].contiguous()
        assignment_targets = targets[start:end].contiguous()
        basis: dict[str, object] = {
            "campaign_id": self.campaign.campaign["id"],
            "checkpoint_sha256": self._state["checkpoint_sha256"],
            "training_method": self._state.get("training_method", "dense"),
            "numerical_profile": self._state["numerical_profile"],
            "dataset_revision": self._state["dataset_revision"],
            "model": {
                "vocab_size": self.campaign.model.vocabulary_size,
                "context_length": self.campaign.model.context_length,
                "d_model": self.campaign.model.width,
                "num_heads": self.campaign.model.heads,
                "num_layers": self.campaign.model.layers,
                "d_ff": self.campaign.model.mlp_width,
            },
            "global_step": base_step,
            "data_range": [dataset_cursor + start, dataset_cursor + end],
            "input_ids": assignment_inputs.reshape(-1).tolist(),
            "input_shape": list(assignment_inputs.shape),
            "target_ids": assignment_targets.reshape(-1).tolist(),
            "target_shape": list(assignment_targets.shape),
            "loss_weight_sum": assignment_targets.numel(),
        }
        if self.lora is not None:
            basis.update(
                {
                    "lora_manifest_sha256": self.lora.manifest_sha256,
                    "base_model_sha256": self.lora.config.base_model_sha256,
                    "adapter_sha256": self._state["initial_adapter_sha256"],
                    "weight_checkpoint_sha256": self._state["checkpoint_sha256"],
                    "resume_state_sha256": self._state["resume_state_sha256"],
                }
            )
            bundle = self._state.get("base_layer_bundle")
            if isinstance(bundle, Mapping):
                basis["base_layer_bundle_manifest_sha256"] = bundle[
                    "manifest_sha256"
                ]
        candidate_bases = [basis]
        predecessor_basis = dict(basis)
        if training_method_migrated:
            predecessor_basis.pop("training_method")
        if numerical_profile_migrated:
            predecessor_basis.pop("numerical_profile")
        if predecessor_basis != basis:
            candidate_bases.append(predecessor_basis)
        expected_assignment_ids = {
            hashlib.sha256(
                json.dumps(
                    candidate,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for candidate in candidate_bases
        }
        migrated_fields: set[str] = set()
        if training_method_migrated:
            migrated_fields.add("training_method")
        if numerical_profile_migrated:
            migrated_fields.add("numerical_profile")
        if assignment.get("assignment_id") not in expected_assignment_ids or any(
            not _exact_json_equal(assignment.get(field), value)
            for field, value in basis.items()
            if field not in migrated_fields
        ):
            raise ValueError("assignment identity differs from its pinned inputs")

        profile = str(self._state["numerical_profile"])
        if self.lora is not None:
            if self._oracle_adapter_state is None:
                raise ValueError("authenticated oracle adapter snapshot is unavailable")
            if self._oracle_model_state is None:
                raise ValueError("authenticated oracle base-model snapshot is unavailable")
            model = build_profiled_lora_model(
                self.campaign,
                self.lora.config,
                profile,
                base_state=self._oracle_model_state,
            )
            load_adapter_state(
                model,
                self._oracle_adapter_state,
            )
            submitted = compute_adapter_gradients(
                model,
                assignment_inputs,
                assignment_targets,
            )
            return submitted.loss_sum, dict(submitted.gradients)
        if self._oracle_model_state is None:
            raise ValueError("authenticated oracle model snapshot is unavailable")
        model = build_model(self.campaign)
        model.load_state_dict(self._oracle_model_state)
        loss_sum = F.cross_entropy(
            model(assignment_inputs).reshape(
                -1,
                self.campaign.model.vocabulary_size,
            ),
            assignment_targets.reshape(-1),
            reduction="sum",
        )
        loss_sum.backward()
        gradients = {
            name: parameter.grad.detach().cpu().contiguous()
            for name, parameter in sorted(model.named_parameters())
            if parameter.grad is not None
        }
        return float(loss_sum.detach()), gradients

    def _oracle_artifact_bytes(
        self,
        assignment: Mapping[str, object],
        *,
        retain_snapshot: bool,
    ) -> bytes:
        assignment_id, expected_file = _validated_assignment_artifact_name(assignment)
        retained = self._oracle_artifact_snapshots.get(assignment_id)
        if retained is not None:
            return retained
        payload = _safe_artifact_snapshot(
            self.oracle_dir,
            expected_file,
            "oracle gradient artifact",
        )
        if retain_snapshot:
            self._oracle_artifact_snapshots[assignment_id] = payload
        return payload

    def _accepted_result_bytes(
        self,
        assignment: Mapping[str, object],
        *,
        retain_snapshot: bool,
    ) -> bytes:
        assignment_id, expected_file = _validated_assignment_artifact_name(assignment)
        if assignment.get("state") != "accepted" or assignment.get(
            "result_file"
        ) != expected_file:
            raise ValueError("accepted result file identity is invalid")
        retained = self._accepted_result_snapshots.get(assignment_id)
        if retained is not None:
            return retained
        payload = _safe_artifact_snapshot(
            self.results_dir,
            expected_file,
            "accepted result artifact",
        )
        if retain_snapshot:
            self._accepted_result_snapshots[assignment_id] = payload
        return payload

    def _validated_oracle_gradients(
        self,
        assignment: dict[str, object],
        *,
        migrate_legacy_identity: bool,
        recomputed: tuple[float, Mapping[str, Tensor]] | None = None,
        retain_snapshot: bool = False,
    ) -> tuple[dict[str, Tensor], bool]:
        oracle_bytes = self._oracle_artifact_bytes(
            assignment,
            retain_snapshot=retain_snapshot,
        )
        file_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
        gradients = _owned_safetensors(oracle_bytes, "oracle gradient artifact")
        tensor_digest = tensor_sha256(gradients)
        stored_file_sha256 = assignment.get("oracle_file_sha256")
        stored_tensor_sha256 = assignment.get("oracle_tensor_sha256")
        stored_size = assignment.get("oracle_file_size")
        legacy_identity = (
            stored_file_sha256 is None
            and stored_tensor_sha256 is None
            and stored_size is None
        )
        if not legacy_identity and (
            not isinstance(stored_file_sha256, str)
            or _SHA256_HEX.fullmatch(stored_file_sha256) is None
            or not isinstance(stored_tensor_sha256, str)
            or _SHA256_HEX.fullmatch(stored_tensor_sha256) is None
            or type(stored_size) is not int
            or stored_size < 0
        ):
            raise ValueError("oracle gradient identity is incomplete")
        if not legacy_identity and (
            stored_file_sha256 != file_sha256
            or stored_tensor_sha256 != tensor_digest
            or stored_size != len(oracle_bytes)
        ):
            raise ValueError("oracle gradient bytes or tensors changed")
        if recomputed is not None:
            recomputed_loss, recomputed_gradients = recomputed
            if (
                not _exact_json_equal(
                    assignment.get("expected_loss_sum"),
                    recomputed_loss,
                )
                or set(gradients) != set(recomputed_gradients)
                or tensor_digest != tensor_sha256(recomputed_gradients)
            ):
                raise ValueError("oracle gradient differs from independent recomputation")
        if legacy_identity:
            if not migrate_legacy_identity or recomputed is None:
                raise ValueError("oracle gradient identity is missing")
            assignment["oracle_file_sha256"] = file_sha256
            assignment["oracle_tensor_sha256"] = tensor_digest
            assignment["oracle_file_size"] = len(oracle_bytes)
        return gradients, legacy_identity

    def _validate_profile_result(
        self,
        assignment: Mapping[str, object],
        loss_sum: float,
        gradients: Mapping[str, Tensor],
        expected_gradients: Mapping[str, Tensor],
    ) -> dict[str, float | int | str]:
        expected_loss = assignment["expected_loss_sum"]
        if type(expected_loss) is not float or type(loss_sum) is not float:
            raise ValueError("loss sums must use exact JSON float values")
        gradient_metrics = _tensor_metrics(expected_gradients, gradients)
        profile = str(self._state["numerical_profile"])
        if profile in {EXACT_CPU_FP32_PROFILE, INT8_FROZEN_LINEAR_PROFILE}:
            if loss_sum != expected_loss:
                raise ValueError("loss sum is not bit-exact for numerical profile")
            if set(gradients) != set(expected_gradients) or tensor_sha256(
                gradients
            ) != tensor_sha256(expected_gradients):
                raise ValueError("gradient is not bit-exact for numerical profile")
        else:
            if abs(loss_sum - expected_loss) / abs(expected_loss) > 0.002:
                raise ValueError("loss sum is outside the numerical-profile tolerance")
            if float(gradient_metrics["cosine_similarity"]) < 0.999:
                raise ValueError(
                    "gradient cosine similarity is outside the numerical-profile tolerance"
                )
            if float(gradient_metrics["relative_l2_error"]) > 0.01:
                raise ValueError(
                    "gradient relative L2 error is outside the numerical-profile tolerance"
                )
        return gradient_metrics

    def _validated_accepted_gradients(
        self,
        assignment: dict[str, object],
        *,
        migrate_legacy_identity: bool,
        retain_snapshot: bool = False,
    ) -> tuple[dict[str, Tensor], bool]:
        result_bytes = self._accepted_result_bytes(
            assignment,
            retain_snapshot=retain_snapshot,
        )
        file_sha256 = hashlib.sha256(result_bytes).hexdigest()
        gradients = _owned_safetensors(result_bytes, "accepted gradient artifact")
        tensor_digest = tensor_sha256(gradients)
        stored_file_sha256 = assignment.get("result_file_sha256")
        stored_tensor_sha256 = assignment.get("result_tensor_sha256")
        legacy_identity = stored_file_sha256 is None and stored_tensor_sha256 is None
        if (stored_file_sha256 is None) != (stored_tensor_sha256 is None):
            raise ValueError("accepted result identity is incomplete")
        if not legacy_identity and (
            stored_file_sha256 != file_sha256
            or stored_tensor_sha256 != tensor_digest
        ):
            raise ValueError("accepted result bytes or tensors changed")
        accepted_loss_sum = assignment.get("accepted_loss_sum")
        if type(accepted_loss_sum) is not float or not math.isfinite(accepted_loss_sum):
            raise ValueError("accepted result loss is invalid")
        expected_gradients, _ = self._validated_oracle_gradients(
            assignment,
            migrate_legacy_identity=False,
        )
        metrics = self._validate_profile_result(
            assignment,
            accepted_loss_sum,
            gradients,
            expected_gradients,
        )
        if not _exact_json_equal(assignment.get("gradient_metrics"), metrics):
            raise ValueError("accepted result gradient metrics changed")
        if legacy_identity:
            if not migrate_legacy_identity:
                raise ValueError("accepted result identity is missing")
            assignment["result_file_sha256"] = file_sha256
            assignment["result_tensor_sha256"] = tensor_digest
        return gradients, legacy_identity

    def oracle_gradient_path(self, assignment_id: str) -> Path:
        assignment = self._assignment(assignment_id)
        _, expected_file = _validated_assignment_artifact_name(assignment)
        return self.oracle_dir / expected_file

    def oracle_gradient_bytes(self, assignment_id: str) -> bytes:
        assignment = self._assignment(assignment_id)
        oracle_bytes = self._oracle_artifact_bytes(
            assignment,
            retain_snapshot=False,
        )
        file_sha256 = assignment.get("oracle_file_sha256")
        tensor_digest = assignment.get("oracle_tensor_sha256")
        stored_size = assignment.get("oracle_file_size")
        if (
            not isinstance(file_sha256, str)
            or hashlib.sha256(oracle_bytes).hexdigest() != file_sha256
            or not isinstance(tensor_digest, str)
            or tensor_sha256(
                _owned_safetensors(oracle_bytes, "oracle gradient artifact")
            )
            != tensor_digest
            or stored_size != len(oracle_bytes)
        ):
            raise ValueError("oracle gradient bytes or tensors changed")
        return oracle_bytes

    def base_layer_bundle_artifact_path(self, file_name: str) -> Path:
        contract = self._state.get("base_layer_bundle")
        if not isinstance(contract, Mapping):
            raise ValueError("base layer bundle is unavailable")
        artifacts = contract.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("base layer bundle artifact contract is invalid")
        if not any(
            isinstance(artifact, Mapping) and artifact.get("file") == file_name
            for artifact in artifacts
        ):
            raise ValueError("unknown base layer bundle artifact")
        lexical_path = self.base_layer_bundle_dir / file_name
        if lexical_path.is_symlink():
            raise ValueError("base layer bundle artifact is a symlink")
        artifact_path = lexical_path.resolve(strict=True)
        if artifact_path.parent != self.base_layer_bundle_dir.resolve():
            raise ValueError("base layer bundle artifact escapes its root")
        if not artifact_path.is_file():
            raise ValueError("base layer bundle artifact is not a regular file")
        return artifact_path

    def initial_model_bytes(self) -> bytes:
        if self._initial_model_snapshot is None:
            raise ValueError("authenticated initial model snapshot is unavailable")
        return self._initial_model_snapshot

    def initial_adapter_bytes(self) -> bytes:
        if self.lora is None or self._initial_adapter_snapshot is None:
            raise ValueError("authenticated initial adapter snapshot is unavailable")
        return self._initial_adapter_snapshot

    def _retain_base_layer_bundle_snapshots(self) -> None:
        contract = self._state.get("base_layer_bundle")
        if contract is None:
            self._base_layer_bundle_snapshots = {}
            return
        if not isinstance(contract, Mapping) or not isinstance(
            contract.get("artifacts"), list
        ):
            raise ValueError("base layer bundle artifact contract is invalid")
        snapshots: dict[str, bytes] = {}
        for artifact in contract["artifacts"]:
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("file"), str)
                or type(artifact.get("bytes")) is not int
                or not isinstance(artifact.get("sha256"), str)
            ):
                raise ValueError("base layer bundle artifact entry is invalid")
            file_name = str(artifact["file"])
            payload = _safe_artifact_snapshot(
                self.base_layer_bundle_dir,
                file_name,
                "base layer bundle artifact",
            )
            if (
                len(payload) != artifact["bytes"]
                or hashlib.sha256(payload).hexdigest() != artifact["sha256"]
            ):
                raise ValueError("base layer bundle raw artifact SHA-256 mismatch")
            snapshots[file_name] = payload
        if len(snapshots) != len(contract["artifacts"]):
            raise ValueError("base layer bundle artifact names are not unique")
        self._base_layer_bundle_snapshots = snapshots

    def _capture_completed_checkpoint_snapshots(self) -> None:
        if self._state.get("state") != "step_complete":
            raise ValueError("checkpoint is not available")
        primary_name = (
            "adapter.safetensors" if self.lora is not None else "model.safetensors"
        )
        names = ("state.json", primary_name, "optimizer.safetensors")
        snapshots = {
            name: _safe_artifact_snapshot(
                self.checkpoint_dir,
                name,
                "completed checkpoint artifact",
            )
            for name in names
        }
        state_payload = json.loads(
            snapshots["state.json"],
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(state_payload, Mapping):
            raise ValueError("completed checkpoint state is invalid")
        if (
            not _exact_json_equal(state_payload.get("step"), self._state["step"])
            or not _exact_json_equal(
                state_payload.get("dataset_cursor"),
                self._state["result_dataset_cursor"],
            )
            or not _exact_json_equal(
                state_payload.get("loss_history"),
                self._state["result_loss_history"],
            )
        ):
            raise ValueError("completed checkpoint progress snapshot differs")
        optimizer = state_payload.get("optimizer")
        if (
            not isinstance(optimizer, Mapping)
            or optimizer.get("file") != "optimizer.safetensors"
            or optimizer.get("sha256")
            != hashlib.sha256(snapshots["optimizer.safetensors"]).hexdigest()
        ):
            raise ValueError("completed checkpoint optimizer identity differs")
        if self.lora is None:
            model = state_payload.get("model")
            model_digest = hashlib.sha256(snapshots[primary_name]).hexdigest()
            if (
                not isinstance(model, Mapping)
                or model.get("file") != primary_name
                or model.get("sha256") != model_digest
                or model_digest != self._state.get("model_sha256")
                or hashlib.sha256(snapshots["state.json"]).hexdigest()
                != self._state.get("result_resume_state_sha256")
            ):
                raise ValueError("completed model checkpoint identity differs")
        else:
            adapter = state_payload.get("adapter")
            adapter_file_digest = hashlib.sha256(snapshots[primary_name]).hexdigest()
            adapter_tensor_digest = tensor_sha256(
                _owned_safetensors(
                    snapshots[primary_name],
                    "completed adapter checkpoint",
                )
            )
            if (
                not isinstance(adapter, Mapping)
                or adapter.get("file") != primary_name
                or adapter.get("file_sha256") != adapter_file_digest
                or adapter.get("tensor_sha256") != adapter_tensor_digest
                or adapter_tensor_digest != self._state.get("adapter_sha256")
                or state_payload.get("weight_checkpoint_sha256")
                != self._state.get("result_weight_checkpoint_sha256")
                or state_payload.get("checkpoint_sha256")
                != self._state.get("result_checkpoint_sha256")
                or state_payload.get("checkpoint_sha256")
                != self._state.get("result_resume_state_sha256")
            ):
                raise ValueError("completed LoRA checkpoint identity differs")
        self._completed_checkpoint_snapshots = snapshots

    def completed_checkpoint_artifacts(self) -> dict[str, bytes]:
        if self._completed_checkpoint_snapshots is None:
            self._capture_completed_checkpoint_snapshots()
        return dict(self._completed_checkpoint_snapshots)

    def base_layer_bundle_artifact_bytes(self, file_name: str) -> bytes:
        _validated_relative_artifact_name(file_name, "base layer bundle artifact")
        try:
            return self._base_layer_bundle_snapshots[file_name]
        except KeyError as exc:
            raise ValueError("unknown base layer bundle artifact") from exc

    def checkpoint_artifact_bytes(self, file_name: str) -> bytes:
        if self._state.get("state") != "step_complete":
            raise ValueError("checkpoint is not available")
        try:
            return self.completed_checkpoint_artifacts()[file_name]
        except KeyError as exc:
            raise ValueError("unknown checkpoint artifact") from exc

    def _assignment(self, assignment_id: str) -> dict[str, object]:
        for assignment in self.assignments:
            if assignment["assignment_id"] == assignment_id:
                return assignment
        raise ValueError("unknown assignment")

    def _public_assignment(self, assignment: Mapping[str, object]) -> dict[str, object]:
        assignment_id = str(assignment["assignment_id"])
        is_lora = assignment.get("training_method") == "frozen-base-lora"
        stored_bundle = self._state.get("base_layer_bundle")
        public_bundle: dict[str, object] | None = None
        layer_bundle_download_bytes = 0
        if stored_bundle is not None:
            if not isinstance(stored_bundle, Mapping):
                raise ValueError("stored base layer bundle contract is invalid")
            artifacts = stored_bundle.get("artifacts")
            if not isinstance(artifacts, list):
                raise ValueError("stored base layer bundle artifact list is invalid")
            public_artifacts = [
                {
                    **artifact,
                    "url": f"/api/v1/artifacts/base-layer-bundle/{artifact['file']}",
                }
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            ]
            if len(public_artifacts) != len(artifacts):
                raise ValueError("stored base layer bundle artifact entry is invalid")
            layer_bundle_download_bytes = int(stored_bundle["download_bytes"])
            public_bundle = {
                "format": "orcacolony_assignment_base_layer_bundle_v1",
                "profile": stored_bundle["profile"],
                "manifest_sha256": stored_bundle["manifest_sha256"],
                "base_model_sha256": stored_bundle["base_model_sha256"],
                "artifacts": public_artifacts,
                "download_bytes": layer_bundle_download_bytes,
            }
        payload: dict[str, object] = {
            "format": "orcacolony_assignment_v2" if is_lora else "orcacolony_assignment_v1",
            "campaign_id": assignment["campaign_id"],
            "assignment_id": assignment_id,
            "checkpoint_sha256": assignment["checkpoint_sha256"],
            "training_method": assignment.get("training_method", "dense"),
            "numerical_profile": self._state["numerical_profile"],
            "dataset_revision": assignment["dataset_revision"],
            "model": assignment["model"],
            "global_step": assignment["global_step"],
            "data_range": assignment["data_range"],
            "input_ids": assignment["input_ids"],
            "input_shape": assignment["input_shape"],
            "target_ids": assignment["target_ids"],
            "target_shape": assignment["target_shape"],
            "loss_weight_sum": assignment["loss_weight_sum"],
            "parameter_count": assignment["parameter_count"],
            "trainable_parameter_count": assignment.get(
                "trainable_parameter_count", assignment["parameter_count"]
            ),
            "expected_loss_sum": assignment["expected_loss_sum"],
            "attempt": assignment["attempt"],
            "lease_token": assignment["lease_token"],
            "lease_expires_at": assignment["lease_expires_at"],
            "model_url": "/api/v1/artifacts/model.safetensors",
            "oracle_gradient_url": f"/api/v1/oracle/{assignment_id}.safetensors",
            "result_url": f"/api/v1/results/{assignment_id}",
            "result_protocol_revision": self._state.get(
                "result_protocol_revision", 1
            ),
            "runtime_backends": sorted(
                backend
                for backend in _PROFILE_RUNTIME_BACKENDS[
                    str(self._state["numerical_profile"])
                ]
                if backend not in _LAYER_BUNDLE_RUNTIME_BACKENDS
                or public_bundle is not None
            ),
            "telemetry_protocol_revision": 1,
            "resource_profile": {
                "format": "orcacolony_assignment_resources_v1",
                "model_download_bytes": len(self.initial_model_bytes()),
                "layer_bundle_download_bytes": layer_bundle_download_bytes,
                "adapter_download_bytes": (
                    len(self.initial_adapter_bytes()) if is_lora else 0
                ),
                "oracle_gradient_download_bytes": assignment["oracle_file_size"],
                "expected_result_upload_bytes": assignment["oracle_file_size"],
                "base_parameter_bytes_fp32": int(assignment["parameter_count"])
                * 4,
                "trainable_parameter_bytes_fp32": int(
                    assignment.get(
                        "trainable_parameter_count", assignment["parameter_count"]
                    )
                )
                * 4,
            },
        }
        if is_lora:
            payload.update(
                {
                    "lora_manifest_sha256": assignment["lora_manifest_sha256"],
                    "base_model_sha256": assignment["base_model_sha256"],
                    "adapter_sha256": assignment["adapter_sha256"],
                    "weight_checkpoint_sha256": assignment[
                        "weight_checkpoint_sha256"
                    ],
                    "resume_state_sha256": assignment["resume_state_sha256"],
                    "adapter": assignment["adapter"],
                    "adapter_url": "/api/v1/artifacts/adapter.safetensors",
                }
            )
            if public_bundle is not None:
                if (
                    assignment.get("base_layer_bundle_manifest_sha256")
                    != public_bundle["manifest_sha256"]
                ):
                    raise ValueError("assignment base layer bundle identity differs")
                payload["base_layer_bundle"] = public_bundle
        return payload

    def lease(
        self,
        worker_id: str,
        worker_token: str | None = None,
        now: float | None = None,
    ) -> dict[str, object]:
        if not worker_id:
            raise ValueError("worker identity is required")
        participant = self.participants.participant_for_worker(worker_id)
        if participant is None:
            raise ValueError("worker is not allowlisted for this campaign")
        if not self.participants.credential_is_valid(
            participant,
            worker_id,
            worker_token,
        ):
            raise ValueError("worker credential is invalid")
        now = time.time() if now is None else now
        with self._lock:
            if self._state["state"] == "step_complete":
                raise ValueError("global step is already complete")
            for assignment in self.assignments:
                if (
                    assignment["state"] == "leased"
                    and assignment["leased_by"] == worker_id
                    and float(assignment["lease_expires_at"]) > now
                ):
                    return self._public_assignment(assignment)

            available = [
                assignment
                for assignment in self.assignments
                if assignment["state"] == "open"
                or (
                    assignment["state"] == "leased"
                    and float(assignment["lease_expires_at"]) <= now
                )
            ]
            if not available:
                raise ValueError("no assignment is currently available")
            assignment = min(available, key=lambda value: value["data_range"][0])
            attempt = int(assignment["attempt"]) + 1
            lease_token = hashlib.sha256(
                f"{assignment['assignment_id']}:{worker_id}:{attempt}".encode("utf-8")
            ).hexdigest()
            assignment.update(
                {
                    "state": "leased",
                    "attempt": attempt,
                    "leased_by": worker_id,
                    "contributor_id": participant.contributor_id,
                    "lease_token": lease_token,
                    "lease_expires_at": now + int(self._state["lease_seconds"]),
                }
            )
            self._write_state()
            return self._public_assignment(assignment)

    def accept(
        self,
        submission: LeasedGradient,
        now: float | None = None,
        finalize: bool = True,
    ) -> WorkReceipt:
        now = time.time() if now is None else now
        with self._lock:
            assignment = self._assignment(submission.assignment_id)
            if assignment["state"] == "accepted":
                raise ValueError("assignment result was already accepted")
            if (
                assignment["state"] != "leased"
                or assignment["lease_token"] != submission.lease_token
                or float(assignment["lease_expires_at"]) <= now
            ):
                raise ValueError("stale lease attempt")
            if submission.checkpoint_sha256 != self._state["checkpoint_sha256"]:
                raise ValueError("checkpoint identity does not match")
            if (
                type(submission.loss_weight_sum) is not int
                or submission.loss_weight_sum != assignment["loss_weight_sum"]
            ):
                raise ValueError("loss weight does not match assignment")
            if type(submission.loss_sum) is not float or not math.isfinite(
                submission.loss_sum
            ):
                raise ValueError("loss sum must be a finite JSON float")
            if submission.runtime_backend not in RUNTIME_BACKENDS:
                raise ValueError("runtime backend is not supported")
            if (
                _RUNTIME_NUMERICAL_PROFILE[submission.runtime_backend]
                != self._state["numerical_profile"]
            ):
                raise ValueError("runtime backend numerical profile does not match assignment")
            if (
                submission.runtime_backend in _LAYER_BUNDLE_RUNTIME_BACKENDS
                and self._state.get("base_layer_bundle") is None
            ):
                raise ValueError("layer-bundle runtime was not assigned")
            if submission.coordinator_receive_seconds is not None and (
                not math.isfinite(submission.coordinator_receive_seconds)
                or submission.coordinator_receive_seconds < 0
                or submission.coordinator_receive_seconds > 86_400
            ):
                raise ValueError("coordinator receive duration is invalid")
            public_assignment = self._public_assignment(assignment)
            resource_profile = public_assignment["resource_profile"]
            worker_telemetry = _validate_worker_telemetry(
                submission.worker_telemetry,
                resource_profile,  # type: ignore[arg-type]
                submission.runtime_backend,
                public_assignment.get("base_layer_bundle"),  # type: ignore[arg-type]
            )
            gradients = _owned_safetensors(
                submission.safetensors,
                "submitted gradient artifact",
            )
            expected_gradients, _ = self._validated_oracle_gradients(
                assignment,
                migrate_legacy_identity=False,
            )
            gradient_metrics = self._validate_profile_result(
                assignment,
                submission.loss_sum,
                gradients,
                expected_gradients,
            )

            result_file = f"{submission.assignment_id}.safetensors"
            result_path = self.results_dir / result_file
            _atomic_bytes(result_path, submission.safetensors)
            persisted_result = _safe_artifact_snapshot(
                self.results_dir,
                result_file,
                "accepted result artifact",
            )
            if persisted_result != submission.safetensors:
                raise ValueError("accepted result bytes changed while being stored")
            self._accepted_result_snapshots[submission.assignment_id] = persisted_result
            instrumentation = {
                "format": "orcacolony_assignment_instrumentation_v1",
                "worker_reported": worker_telemetry,
                "coordinator_measured": {
                    "model_artifact_bytes": _expected_model_download_bytes(
                        resource_profile,  # type: ignore[arg-type]
                        submission.runtime_backend,
                    ),
                    "adapter_artifact_bytes": resource_profile[
                        "adapter_download_bytes"
                    ],
                    "oracle_gradient_artifact_bytes": resource_profile[
                        "oracle_gradient_download_bytes"
                    ],
                    "result_upload_bytes": len(submission.safetensors),
                    "result_receive_seconds": submission.coordinator_receive_seconds,
                    "result_storage_bytes": len(persisted_result),
                },
            }
            assignment.update(
                {
                    "state": "accepted",
                    "result_file": result_file,
                    "result_file_sha256": hashlib.sha256(
                        persisted_result
                    ).hexdigest(),
                    "result_tensor_sha256": tensor_sha256(gradients),
                    "accepted_loss_sum": submission.loss_sum,
                    "runtime_backend": submission.runtime_backend,
                    "gradient_metrics": gradient_metrics,
                    "instrumentation": instrumentation,
                    "lease_token": None,
                    "lease_expires_at": None,
                }
            )
            self._state["state"] = (
                "ready_to_finalize" if self._all_accepted() else "waiting_for_results"
            )
            self._write_state()
            self._write_accepted_ledger()
            if self._all_accepted() and finalize:
                self._finalize_locked()
            return self._receipt(assignment)

    def _finalize_locked(self) -> None:
        if self.lora is not None:
            self._finalize_lora_locked()
            return
        if self._finalization_model is None or self._finalization_optimizer is None:
            raise ValueError("authenticated dense finalization snapshot is unavailable")
        if self._reference_state is None:
            raise ValueError("authenticated dense reference snapshot is unavailable")
        model, optimizer = copy.deepcopy(
            (self._finalization_model, self._finalization_optimizer)
        )
        base_step = self._state["base_step"]
        dataset_cursor = self._state["dataset_cursor"]
        loss_history = list(self._state["loss_history"])
        aggregate = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in model.named_parameters()
        }
        total_loss_sum = 0.0
        total_loss_weight = 0
        for assignment in sorted(self.assignments, key=lambda value: value["data_range"][0]):
            gradients, _ = self._validated_accepted_gradients(
                assignment,
                migrate_legacy_identity=False,
            )
            for name in aggregate:
                aggregate[name].add_(gradients[name])
            total_loss_sum += float(assignment["accepted_loss_sum"])
            total_loss_weight += int(assignment["loss_weight_sum"])

        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.grad = aggregate[name].div(total_loss_weight)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.campaign.training.max_gradient_norm,
        )
        optimizer.step()
        next_step = base_step + 1
        next_cursor = (
            dataset_cursor + self.campaign.training.batch_size
        ) % self.campaign.training.dataset_sequences
        next_loss_history = [*loss_history, total_loss_sum / total_loss_weight]
        checkpoint = _save_checkpoint(
            self.campaign,
            model,
            optimizer,
            self.checkpoint_dir,
            step=next_step,
            dataset_cursor=next_cursor,
            loss_history=next_loss_history,
        )
        resume_state_sha256 = hashlib.sha256(
            _safe_artifact_snapshot(
                self.checkpoint_dir,
                "state.json",
                "completed dense checkpoint state",
            )
        ).hexdigest()
        checkpoint_metrics = _tensor_metrics(
            self._reference_state,
            {
                name: tensor.detach().cpu().clone().contiguous()
                for name, tensor in model.state_dict().items()
            },
        )
        _validate_checkpoint_profile_metrics(
            str(self._state["numerical_profile"]),
            checkpoint_metrics,
        )
        self._state.update(
            {
                "state": "step_complete",
                "step": next_step,
                "result_dataset_cursor": next_cursor,
                "result_loss_history": next_loss_history,
                "model_sha256": checkpoint.model_sha256,
                "adapter_sha256": None,
                "result_resume_state_sha256": resume_state_sha256,
                "result_checkpoint_sha256": checkpoint.model_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "loss_sum": total_loss_sum,
                "loss_weight_sum": total_loss_weight,
            }
        )
        self._capture_completed_checkpoint_snapshots()
        self._write_state()
        self._write_campaign_lock()
        _atomic_json(
            self.state_dir / "global-receipt.json",
            {
                "format": "orcacolony_global_step_receipt_v1",
                "state": "step_complete",
                "step": next_step,
                "model_sha256": checkpoint.model_sha256,
                "adapter_sha256": None,
                "checkpoint_sha256": checkpoint.model_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "assignments": [
                    {
                        "assignment_id": assignment["assignment_id"],
                        "attempt": assignment["attempt"],
                        "data_range": assignment["data_range"],
                        "gradient_metrics": assignment["gradient_metrics"],
                        "runtime_backend": assignment["runtime_backend"],
                        "instrumentation": assignment.get("instrumentation"),
                    }
                    for assignment in self.assignments
                ],
            },
        )

    def _finalize_lora_locked(self) -> None:
        if self.lora is None:
            raise RuntimeError("LoRA finalization requires a loaded manifest")
        numerical_profile = _validated_numerical_profile(
            self._state["numerical_profile"]
        )
        checkpoint_profile = _checkpoint_numerical_profile(numerical_profile)
        if self._finalization_model is None or self._finalization_optimizer is None:
            raise ValueError("authenticated LoRA finalization snapshot is unavailable")
        if self._reference_state is None:
            raise ValueError("authenticated LoRA reference snapshot is unavailable")
        model, optimizer = copy.deepcopy(
            (self._finalization_model, self._finalization_optimizer)
        )
        base_step = self._state["base_step"]
        dataset_cursor = self._state["dataset_cursor"]
        loss_history = list(self._state["loss_history"])

        aggregate = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in adapter_named_parameters(model).items()
        }
        total_loss_sum = 0.0
        total_loss_weight = 0
        for assignment in sorted(
            self.assignments,
            key=lambda value: value["data_range"][0],
        ):
            gradients, _ = self._validated_accepted_gradients(
                assignment,
                migrate_legacy_identity=False,
            )
            if set(gradients) != set(aggregate):
                raise ValueError("accepted LoRA result tensor set changed before finalization")
            for name in aggregate:
                aggregate[name].add_(gradients[name])
            total_loss_sum += float(assignment["accepted_loss_sum"])
            total_loss_weight += int(assignment["loss_weight_sum"])

        apply_adapter_gradient_step(
            model,
            optimizer,
            aggregate,
            total_loss_weight,
            self.campaign.training.max_gradient_norm,
        )
        next_step = base_step + 1
        next_cursor = (
            dataset_cursor + self.campaign.training.batch_size
        ) % self.campaign.training.dataset_sequences
        next_loss_history = [*loss_history, total_loss_sum / total_loss_weight]
        checkpoint = save_lora_checkpoint(
            self.lora,
            model,
            optimizer,
            self.checkpoint_dir,
            step=next_step,
            dataset_cursor=next_cursor,
            loss_history=next_loss_history,
            numerical_profile=checkpoint_profile,
        )
        checkpoint_metrics = _tensor_metrics(
            self._reference_state,
            {
                name: parameter.detach().cpu().clone().contiguous()
                for name, parameter in adapter_named_parameters(model).items()
            },
        )
        _validate_checkpoint_profile_metrics(numerical_profile, checkpoint_metrics)
        self._state.update(
            {
                "state": "step_complete",
                "step": next_step,
                "result_dataset_cursor": next_cursor,
                "result_loss_history": next_loss_history,
                "model_sha256": checkpoint.base_model_sha256,
                "adapter_sha256": checkpoint.adapter_sha256,
                "result_resume_state_sha256": checkpoint.checkpoint_sha256,
                "result_weight_checkpoint_sha256": (
                    checkpoint.weight_checkpoint_sha256
                ),
                "result_checkpoint_sha256": checkpoint.checkpoint_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "loss_sum": total_loss_sum,
                "loss_weight_sum": total_loss_weight,
            }
        )
        self._capture_completed_checkpoint_snapshots()
        self._write_state()
        self._write_campaign_lock()
        _atomic_json(
            self.state_dir / "global-receipt.json",
            {
                "format": "orcacolony_global_step_receipt_v2",
                "training_method": "frozen-base-lora",
                "numerical_profile": numerical_profile,
                "state": "step_complete",
                "step": next_step,
                "model_sha256": checkpoint.base_model_sha256,
                "adapter_sha256": checkpoint.adapter_sha256,
                "weight_checkpoint_sha256": checkpoint.weight_checkpoint_sha256,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "checkpoint_metrics": checkpoint_metrics,
                "assignments": [
                    {
                        "assignment_id": assignment["assignment_id"],
                        "attempt": assignment["attempt"],
                        "data_range": assignment["data_range"],
                        "gradient_metrics": assignment["gradient_metrics"],
                        "runtime_backend": assignment["runtime_backend"],
                        "instrumentation": assignment.get("instrumentation"),
                    }
                    for assignment in self.assignments
                ],
            },
        )

    def _receipt(self, assignment: Mapping[str, object]) -> WorkReceipt:
        step_complete = self._state["state"] == "step_complete"
        return WorkReceipt(
            assignment_id=str(assignment["assignment_id"]),
            accepted=True,
            step_complete=step_complete,
            step=self._state["step"],
            model_sha256=(
                str(self._state["model_sha256"]) if step_complete else None
            ),
            adapter_sha256=(
                str(self._state["adapter_sha256"])
                if step_complete and self._state.get("adapter_sha256") is not None
                else None
            ),
            weight_checkpoint_sha256=(
                str(self._state["result_weight_checkpoint_sha256"])
                if step_complete
                and self._state.get("result_weight_checkpoint_sha256") is not None
                else (
                    str(self._state["result_checkpoint_sha256"])
                    if step_complete
                    and self._state.get("training_method", "dense") == "dense"
                    else None
                )
            ),
            checkpoint_sha256=(
                str(self._state["result_checkpoint_sha256"])
                if step_complete
                and self._state.get("result_checkpoint_sha256") is not None
                else None
            ),
            gradient_metrics=assignment["gradient_metrics"],  # type: ignore[arg-type]
            checkpoint_metrics=(
                self._state["checkpoint_metrics"] if step_complete else {}
            ),  # type: ignore[arg-type]
            instrumentation=assignment.get("instrumentation", {}),  # type: ignore[arg-type]
        )

    def status(self) -> dict[str, object]:
        return {
            "state": self._state["state"],
            "campaign_id": self._state["campaign_id"],
            "checkpoint_sha256": self._state["checkpoint_sha256"],
            "training_method": self._state.get("training_method", "dense"),
            "numerical_profile": self._state["numerical_profile"],
            "base_model_sha256": self._state.get("base_model_sha256"),
            "initial_adapter_sha256": self._state.get("initial_adapter_sha256"),
            "step": self._state["step"],
            "model_sha256": self._state["model_sha256"],
            "adapter_sha256": self._state.get("adapter_sha256"),
            "resume_state_sha256": self._state.get("resume_state_sha256"),
            "result_weight_checkpoint_sha256": self._state.get(
                "result_weight_checkpoint_sha256"
            ),
            "result_checkpoint_sha256": self._state.get(
                "result_checkpoint_sha256"
            ),
            "checkpoint_metrics": self._state["checkpoint_metrics"],
            "loss_sum": self._state.get("loss_sum"),
            "loss_weight_sum": self._state.get("loss_weight_sum"),
            "resource_observations": _aggregate_resource_observations(
                [
                    assignment
                    for assignment in self.assignments
                    if assignment["state"] == "accepted"
                ],
                self.state_dir,
            ),
            "assignments": [
                {
                    "assignment_id": assignment["assignment_id"],
                    "data_range": assignment["data_range"],
                    "state": assignment["state"],
                    "attempt": assignment["attempt"],
                    "leased_by": assignment["leased_by"],
                }
                for assignment in self.assignments
            ],
        }

    def public_status(self) -> dict[str, object]:
        status = self.status()
        status["assignments"] = [
            {
                key: value
                for key, value in assignment.items()
                if key != "leased_by"
            }
            for assignment in status["assignments"]
        ]
        return status


class _GlobalStepHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: object,
        coordinator: GlobalStepCoordinator,
        directory: str,
        public_origin: str | None,
        **kwargs: object,
    ) -> None:
        self.coordinator = coordinator
        self.public_origin = public_origin
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        if (
            self.public_origin is not None
            and self.public_origin == self.headers.get("Origin")
        ):
            self.send_header("Access-Control-Allow-Origin", self.public_origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if (
            self.public_origin is None
            or self.headers.get("Origin") != self.public_origin
            or not urlsplit(self.path).path.startswith("/api/v1/")
        ):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            ", ".join(
                (
                    "Content-Type",
                    "X-Orca-Worker-Token",
                    "X-Orca-Lease-Token",
                    "X-Orca-Checkpoint-Sha256",
                    "X-Orca-Loss-Sum",
                    "X-Orca-Loss-Weight-Sum",
                    "X-Orca-Runtime-Backend",
                    "X-Orca-Worker-Telemetry",
                )
            ),
        )
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_json(
        self,
        payload: Mapping[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_artifact(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "artifact is unavailable"}, HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def _send_artifact_bytes(self, payload: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/api/v1/assignment":
            worker_id = parse_qs(parsed.query).get("worker_id", [""])[0]
            try:
                assignment = self.coordinator.lease(
                    worker_id,
                    worker_token=self.headers.get("X-Orca-Worker-Token"),
                )
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.CONFLICT)
                return
            self._send_json(assignment)
            return
        if path == "/api/v1/status":
            public_status = getattr(self.coordinator, "public_status", None)
            self._send_json(
                public_status() if public_status is not None else self.coordinator.status()
            )
            return
        if path == "/api/v1/dashboard":
            dashboard = getattr(self.coordinator, "dashboard", None)
            if dashboard is None:
                self._send_json(
                    {"error": "dashboard is unavailable"}, HTTPStatus.NOT_FOUND
                )
                return
            self._send_json(dashboard())
            return
        if path == "/api/v1/artifacts/model.safetensors":
            self._send_artifact_bytes(self.coordinator.initial_model_bytes())
            return
        if path == "/api/v1/artifacts/adapter.safetensors":
            try:
                artifact = self.coordinator.initial_adapter_bytes()
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact_bytes(artifact)
            return
        bundle_prefix = "/api/v1/artifacts/base-layer-bundle/"
        if path.startswith(bundle_prefix):
            try:
                artifact = self.coordinator.base_layer_bundle_artifact_bytes(
                    path.removeprefix(bundle_prefix)
                )
            except (FileNotFoundError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact_bytes(artifact)
            return
        if path.startswith("/api/v1/oracle/") and path.endswith(".safetensors"):
            assignment_id = path.removeprefix("/api/v1/oracle/").removesuffix(
                ".safetensors"
            )
            try:
                artifact = self.coordinator.oracle_gradient_bytes(assignment_id)
            except ValueError as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact_bytes(artifact)
            return
        if path == "/api/v1/checkpoint/model.safetensors":
            try:
                artifact = self.coordinator.checkpoint_artifact_bytes(
                    "model.safetensors"
                )
            except (FileNotFoundError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact_bytes(artifact)
            return
        if path == "/api/v1/checkpoint/adapter.safetensors":
            try:
                artifact = self.coordinator.checkpoint_artifact_bytes(
                    "adapter.safetensors"
                )
            except (FileNotFoundError, ValueError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.NOT_FOUND)
                return
            self._send_artifact_bytes(artifact)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        prefix = "/api/v1/results/"
        if not path.startswith(prefix):
            self._send_json({"error": "unknown result endpoint"}, HTTPStatus.NOT_FOUND)
            return
        assignment_id = path.removeprefix(prefix)
        try:
            maximum_length = len(self.coordinator.oracle_gradient_bytes(assignment_id)) + 1024
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > maximum_length:
                raise ValueError("gradient payload length is invalid")
            receive_started = time.perf_counter()
            result_body = self.rfile.read(content_length)
            receive_seconds = time.perf_counter() - receive_started
            if len(result_body) != content_length:
                raise ValueError("gradient payload was truncated")
            telemetry_header = self.headers.get("X-Orca-Worker-Telemetry")
            worker_telemetry = (
                json.loads(telemetry_header) if telemetry_header is not None else None
            )
            if worker_telemetry is not None and not isinstance(
                worker_telemetry, Mapping
            ):
                raise ValueError("worker telemetry must be a JSON object")
            submission = LeasedGradient(
                assignment_id=assignment_id,
                lease_token=self.headers["X-Orca-Lease-Token"],
                checkpoint_sha256=self.headers["X-Orca-Checkpoint-Sha256"],
                loss_sum=float(self.headers["X-Orca-Loss-Sum"]),
                loss_weight_sum=int(self.headers["X-Orca-Loss-Weight-Sum"]),
                safetensors=result_body,
                runtime_backend=self.headers["X-Orca-Runtime-Backend"],
                worker_telemetry=worker_telemetry,
                coordinator_receive_seconds=receive_seconds,
            )
            receipt = self.coordinator.accept(submission)
        except (KeyError, TypeError, ValueError, SafetensorError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "accepted": receipt.accepted,
                "assignment_id": receipt.assignment_id,
                "step_complete": receipt.step_complete,
                "step": receipt.step,
                "model_sha256": receipt.model_sha256,
                "adapter_sha256": receipt.adapter_sha256,
                "weight_checkpoint_sha256": receipt.weight_checkpoint_sha256,
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "gradient_metrics": receipt.gradient_metrics,
                "checkpoint_metrics": receipt.checkpoint_metrics,
                "instrumentation": receipt.instrumentation,
                "checkpoint_url": (
                    (
                        "/api/v1/checkpoint/adapter.safetensors"
                        if receipt.adapter_sha256 is not None
                        else "/api/v1/checkpoint/model.safetensors"
                    )
                    if receipt.step_complete
                    else None
                ),
            }
        )


def normalize_http_origin(origin: str) -> str:
    if any(
        character.isspace()
        or ord(character) < 32
        or character in {'"', "'", "<", ">", "\\"}
        for character in origin
    ):
        raise ValueError("public origin contains invalid characters")
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public origin must be an HTTP(S) origin without a path")
    hostname = parsed.hostname
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(hostname.split("%", 1)[0]).is_loopback
        except ValueError:
            loopback = hostname == "localhost"
        if not loopback:
            raise ValueError("public origin must use HTTPS except on loopback")
    if "%" in hostname:
        raise ValueError("public origin hostname is invalid")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            hostname = hostname.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise ValueError("public origin hostname is invalid") from error
        labels = hostname.split(".")
        if (
            len(hostname) > 253
            or all(label.isdigit() or label.startswith("0x") for label in labels)
            or any(
                not label
                or len(label) > 63
                or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                is None
                for label in labels
            )
        ):
            raise ValueError("public origin hostname is invalid")
    else:
        hostname = f"[{address.compressed}]" if address.version == 6 else str(address)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("public origin has an invalid port") from error
    default_port = 80 if parsed.scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme}://{hostname}{port_suffix}"


def create_http_server(
    coordinator: GlobalStepCoordinator,
    browser_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    public_origin: str | None = None,
) -> ThreadingHTTPServer:
    browser_root = Path(browser_root).resolve()
    if not (browser_root / "index.html").is_file():
        raise ValueError(f"browser root does not contain index.html: {browser_root}")
    if public_origin is not None:
        public_origin = normalize_http_origin(public_origin)

    def handler(*args: object, **kwargs: object) -> _GlobalStepHandler:
        return _GlobalStepHandler(
            *args,
            coordinator=coordinator,
            directory=str(browser_root),
            public_origin=public_origin,
            **kwargs,
        )

    return ThreadingHTTPServer((host, port), handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OrcaColony M2 multi-worker global-step proof"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--dataset-artifacts", type=Path)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--numerical-profile",
        choices=tuple(sorted(_PROFILE_RUNTIME_BACKENDS)),
        default=EXACT_CPU_FP32_PROFILE,
    )
    parser.add_argument("--publish-base-layer-bundle", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-origin")
    return parser


def main() -> None:
    from .reference import load_campaign

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
    state_path = args.state / "global-state.json"
    if state_path.is_file():
        coordinator = GlobalStepCoordinator.load(
            campaign,
            args.state,
            participants=participants,
            dataset=dataset,
            lora=lora,
            numerical_profile=args.numerical_profile,
        )
    else:
        coordinator = GlobalStepCoordinator.create(
            campaign,
            args.state,
            worker_count=args.workers,
            participants=participants,
            lease_seconds=args.lease_seconds,
            resume_from=args.resume_from,
            dataset=dataset,
            lora=lora,
            publish_base_layer_bundle=args.publish_base_layer_bundle,
            numerical_profile=args.numerical_profile,
        )
    server = create_http_server(
        coordinator,
        args.browser_root,
        host=args.host,
        port=args.port,
        public_origin=args.public_origin,
    )
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign["id"],
                "state": coordinator.status()["state"],
                "training_method": coordinator.status()["training_method"],
                "url_template": (
                    f"http://{args.host}:{server.server_port}/"
                    "?worker=<worker-id>#token=<worker-token>"
                ),
                "workers": args.workers,
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
