from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import (
    CampaignConfig,
    _create_optimizer,
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
    validate_dataset_artifacts,
)
from orcacolony.tile_process import (
    _await_model_readiness,
    _deserialize_tensors,
    _recv_bytes,
    _recv_json,
    _send_json,
    _serialize_tensors,
    _tile_worker_entry,
    _validate_tensor,
)
from orcacolony.tiled_model import (
    _gradient_snapshot,
    _max_abs_difference,
    _model_snapshot,
    _optimizer_tensor_snapshot,
    _prefix_activation,
    _suffix_logits,
)


_PHASES = (
    "prepared",
    "forward_accepted",
    "worker_lost",
    "replay_verified",
    "adjoint_persisted",
    "result_accepted",
    "applied",
)
_OWNED_TENSOR_FILES = (
    "tile.safetensors",
    "input.safetensors",
    "forward-output.safetensors",
    "output-adjoint.safetensors",
    "result.safetensors",
)
_TRANSACTION_IDENTITY_FIELDS = (
    "campaign_id",
    "dataset_revision",
    "checkpoint_model_sha256",
    "block_index",
    "cursor",
    "tile_sha256",
    "input_sha256",
)
_MANIFEST_FIELDS = {
    "format",
    "transaction_id",
    *_TRANSACTION_IDENTITY_FIELDS,
    "phase",
    "phase_history",
    "result_applied",
    "files",
}


@dataclass(frozen=True)
class PersistedFileEvidence:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RecoveredTileTransactionEvidence:
    format: str
    campaign_id: str
    dataset_revision: str
    start_method: str
    transaction_id: str
    checkpoint_model_sha256: str
    block_index: int
    cursor: int
    phase_history: tuple[str, ...]
    worker_model_transmissions: int
    first_worker_terminated: bool
    first_worker_exit_code: int
    replacement_worker_exit_code: int
    replay_output_bytes_identical: bool
    duplicate_result_rejected: bool
    tile_model_wire_bytes: int
    input_wire_bytes: int
    forward_output_wire_bytes: int
    output_adjoint_wire_bytes: int
    result_wire_bytes: int
    recovery_retransmitted_tensor_bytes: int
    recovery_total_tensor_bytes: int
    persisted_file_count: int
    persisted_tensor_bytes: int
    persisted_files: tuple[PersistedFileEvidence, ...]
    recovery_seconds: float
    first_worker_initialization_seconds: float
    replacement_worker_initialization_seconds: float
    first_forward_seconds: float
    replay_forward_seconds: float
    replacement_backward_seconds: float
    replacement_worker_peak_rss_bytes: int
    centralized_loss: float
    recovered_loss: float
    max_abs_raw_gradient_difference: float
    max_abs_clipped_gradient_difference: float
    max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    recovered_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    recovered_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    recovered_optimizer_sha256: str
    centralized_model_sha256: str
    recovered_model_sha256: str


@dataclass
class _TileWorkerHandle:
    process: multiprocessing.Process
    connection: Connection
    initialization_seconds: float
    ready: dict[str, object]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise RuntimeError(
                f"failed to remove incomplete transaction file: {temporary.name}"
            ) from cleanup_error
        raise


def _write_manifest(transaction_dir: Path, manifest: dict[str, Any]) -> None:
    _write_bytes_atomic(transaction_dir / "manifest.json", _canonical_json(manifest))


def _record_tensor_file(
    transaction_dir: Path,
    manifest: dict[str, Any],
    name: str,
    payload: bytes,
) -> None:
    if name not in _OWNED_TENSOR_FILES:
        raise ValueError(f"unowned transaction file: {name}")
    _write_bytes_atomic(transaction_dir / name, payload)
    files = manifest["files"]
    if not isinstance(files, dict):
        raise AssertionError("transaction file map is invalid")
    files[name] = {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}


def _transition(
    transaction_dir: Path,
    manifest: dict[str, Any],
    expected: str,
    new_phase: str,
) -> None:
    if manifest.get("phase") != expected:
        raise ValueError(
            f"transaction phase is {manifest.get('phase')!r}, expected {expected!r}"
        )
    history = manifest.get("phase_history")
    if not isinstance(history, list) or tuple(history) != _PHASES[: len(history)]:
        raise ValueError("transaction phase history is invalid")
    if new_phase != _PHASES[len(history)]:
        raise ValueError("transaction phase transition is invalid")
    history.append(new_phase)
    manifest["phase"] = new_phase
    _write_manifest(transaction_dir, manifest)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key in transaction manifest: {key}")
        payload[key] = value
    return payload


def _load_manifest(transaction_dir: Path) -> dict[str, Any]:
    payload = json.loads(
        (transaction_dir / "manifest.json").read_text("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("transaction manifest is invalid")
    return payload


def _read_owned_tensor_file(
    transaction_dir: Path,
    manifest: dict[str, Any],
    name: str,
) -> bytes:
    if name not in _OWNED_TENSOR_FILES:
        raise ValueError(f"unowned transaction file: {name}")
    files = manifest.get("files")
    if not isinstance(files, dict) or name not in files:
        raise ValueError(f"transaction file is not admitted: {name}")
    metadata = files[name]
    if (
        not isinstance(metadata, dict)
        or frozenset(metadata) != frozenset({"sha256", "size_bytes"})
        or type(metadata["sha256"]) is not str
        or type(metadata["size_bytes"]) is not int
        or metadata["size_bytes"] <= 0
    ):
        raise ValueError(f"transaction file metadata is invalid: {name}")
    path = transaction_dir / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"transaction file is not a regular owned file: {name}")
    payload = path.read_bytes()
    if len(payload) != metadata["size_bytes"]:
        raise ValueError(f"transaction file size changed: {name}")
    if _sha256_bytes(payload) != metadata["sha256"]:
        raise ValueError(f"transaction file digest changed: {name}")
    return payload


def _validate_result_ready_phase(manifest: dict[str, Any]) -> None:
    if manifest.get("result_applied") is not False:
        raise ValueError("transaction result was already applied")
    if manifest.get("phase") != "result_accepted":
        raise ValueError("transaction result is not ready to apply")
    history = manifest.get("phase_history")
    if not isinstance(history, list) or tuple(history) != _PHASES[:-1]:
        raise ValueError("transaction phase history is invalid")


def _validate_result_ready_manifest(
    transaction_dir: Path,
    manifest: dict[str, Any],
    expected_identity: dict[str, object],
) -> dict[str, bytes]:
    if frozenset(expected_identity) != frozenset(_TRANSACTION_IDENTITY_FIELDS):
        raise AssertionError("expected transaction identity is invalid")
    if frozenset(manifest) != frozenset(_MANIFEST_FIELDS):
        raise ValueError("transaction manifest schema is invalid")
    if manifest.get("format") != "orcacolony_boundary_transaction_v1":
        raise ValueError("transaction manifest format is invalid")
    _validate_result_ready_phase(manifest)
    for name, expected in expected_identity.items():
        actual = manifest.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"transaction identity is invalid: {name}")
    expected_transaction_id = hashlib.sha256(
        _canonical_json(expected_identity)
    ).hexdigest()
    if (
        type(manifest.get("transaction_id")) is not str
        or manifest["transaction_id"] != expected_transaction_id
    ):
        raise ValueError("transaction id is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or frozenset(files) != frozenset(
        _OWNED_TENSOR_FILES
    ):
        raise ValueError("transaction file map is invalid")
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise ValueError("transaction directory is invalid")
    entries = tuple(transaction_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("transaction directory contains a non-file entry")
    if {entry.name for entry in entries} != {
        "manifest.json",
        *_OWNED_TENSOR_FILES,
    }:
        raise ValueError("transaction directory contains unexpected files")
    return {
        name: _read_owned_tensor_file(transaction_dir, manifest, name)
        for name in _OWNED_TENSOR_FILES
    }


def _start_worker(
    context: multiprocessing.context.BaseContext,
    campaign: CampaignConfig,
    block_index: int,
    tile_wire: bytes,
    timeout_seconds: float,
    *,
    name: str,
    expected_tile_state_sha256: str,
) -> _TileWorkerHandle:
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_tile_worker_entry,
        args=(child_connection,),
        name=name,
        daemon=False,
    )
    started = time.perf_counter()
    process.start()
    child_connection.close()
    try:
        _send_json(
            parent_connection,
            {
                "op": "init",
                "model": asdict(campaign.model),
                "block_index": block_index,
                "seed": campaign.training.seed,
            },
        )
        _await_model_readiness(
            parent_connection,
            timeout_seconds,
            label=f"{name} model readiness",
        )
        parent_connection.send_bytes(tile_wire)
        ready, _ = _recv_json(
            parent_connection,
            timeout_seconds,
            label=f"{name} initialization",
        )
        initialization_seconds = time.perf_counter() - started
        expected = {
            "status",
            "tile_state_sha256",
            "startup_current_rss_bytes",
            "startup_peak_rss_bytes",
            "after_model_current_rss_bytes",
            "after_model_peak_rss_bytes",
        }
        if (
            frozenset(ready) != frozenset(expected)
            or ready["status"] != "ready"
            or ready["tile_state_sha256"] != expected_tile_state_sha256
        ):
            raise ValueError(f"{name} initialization acknowledgement is invalid")
        return _TileWorkerHandle(
            process=process,
            connection=parent_connection,
            initialization_seconds=initialization_seconds,
            ready=ready,
        )
    except BaseException:
        parent_connection.close()
        process.terminate()
        process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
        raise


def _terminate_worker(
    worker: _TileWorkerHandle,
    timeout_seconds: float,
) -> int:
    worker.connection.close()
    worker.process.terminate()
    worker.process.join(timeout_seconds)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join(timeout_seconds)
    if worker.process.is_alive() or worker.process.exitcode is None:
        raise RuntimeError("failed to terminate tile worker")
    return int(worker.process.exitcode)


def _stop_worker(
    worker: _TileWorkerHandle,
    timeout_seconds: float,
) -> tuple[int, dict[str, object]]:
    try:
        _send_json(worker.connection, {"op": "shutdown"})
        acknowledgement, _ = _recv_json(
            worker.connection,
            timeout_seconds,
            label="replacement shutdown",
        )
    finally:
        worker.connection.close()
        worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout_seconds)
    if worker.process.exitcode is None or worker.process.exitcode != 0:
        raise RuntimeError(f"replacement worker exited with {worker.process.exitcode}")
    if acknowledgement.get("status") != "stopped":
        raise ValueError("replacement shutdown acknowledgement is invalid")
    return int(worker.process.exitcode), acknowledgement


def _worker_forward(
    worker: _TileWorkerHandle,
    input_wire: bytes,
    timeout_seconds: float,
    *,
    assignment_id: int,
) -> tuple[bytes, dict[str, object], float]:
    _send_json(
        worker.connection,
        {"op": "forward", "assignment_id": assignment_id},
    )
    started = time.perf_counter()
    worker.connection.send_bytes(input_wire)
    acknowledgement, _ = _recv_json(
        worker.connection,
        timeout_seconds,
        label="tile forward acknowledgement",
    )
    output_wire = _recv_bytes(
        worker.connection,
        timeout_seconds,
        label="tile forward output",
    )
    elapsed = time.perf_counter() - started
    if (
        acknowledgement.get("status") != "forwarded"
        or acknowledgement.get("assignment_id") != assignment_id
    ):
        raise ValueError("tile forward acknowledgement is invalid")
    return output_wire, acknowledgement, elapsed


def _worker_backward(
    worker: _TileWorkerHandle,
    adjoint_wire: bytes,
    timeout_seconds: float,
    *,
    assignment_id: int,
) -> tuple[bytes, dict[str, object], float]:
    _send_json(
        worker.connection,
        {"op": "backward", "assignment_id": assignment_id},
    )
    started = time.perf_counter()
    worker.connection.send_bytes(adjoint_wire)
    acknowledgement, _ = _recv_json(
        worker.connection,
        timeout_seconds,
        label="tile backward acknowledgement",
    )
    if acknowledgement.get("status") == "error":
        raise RuntimeError(str(acknowledgement.get("message")))
    result_wire = _recv_bytes(
        worker.connection,
        timeout_seconds,
        label="tile backward result",
    )
    elapsed = time.perf_counter() - started
    if (
        acknowledgement.get("status") != "backwarded"
        or acknowledgement.get("assignment_id") != assignment_id
    ):
        raise ValueError("tile backward acknowledgement is invalid")
    return result_wire, acknowledgement, elapsed


def _apply_result_once(
    transaction_dir: Path,
    recovered_model: torch.nn.Module,
    recovered_optimizer: torch.optim.Optimizer,
    block_input: Tensor,
    block_index: int,
    campaign: CampaignConfig,
    *,
    expected_identity: dict[str, object],
) -> dict[str, Tensor]:
    manifest = _load_manifest(transaction_dir)
    validated_files = _validate_result_ready_manifest(
        transaction_dir,
        manifest,
        expected_identity,
    )
    result_wire = validated_files["result.safetensors"]
    result = _deserialize_tensors(result_wire)
    selected = recovered_model.blocks[block_index]
    expected_names = {f"gradient.{name}" for name, _ in selected.named_parameters()}
    expected_names.add("input_adjoint")
    if frozenset(result) != frozenset(expected_names):
        raise ValueError("recovered result tensor schema is invalid")
    validated_gradients: dict[str, Tensor] = {}
    for name, parameter in selected.named_parameters():
        validated_gradients[name] = _validate_tensor(
            result[f"gradient.{name}"],
            shape=parameter.shape,
            label=f"gradient.{name}",
        ).detach().clone()
    input_adjoint = _validate_tensor(
        result["input_adjoint"],
        shape=block_input.shape,
        label="input_adjoint",
    ).detach().clone()

    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in recovered_model.named_parameters()
    }
    try:
        for name, parameter in selected.named_parameters():
            parameter.grad = validated_gradients[name]
        block_input.backward(input_adjoint, retain_graph=True)
        recovered_raw = _gradient_snapshot(recovered_model)
        torch.nn.utils.clip_grad_norm_(
            recovered_model.parameters(),
            campaign.training.max_gradient_norm,
        )
        candidate_model = copy.deepcopy(recovered_model)
        candidate_optimizer = _create_optimizer(candidate_model, campaign.training)
        candidate_optimizer.load_state_dict(
            copy.deepcopy(recovered_optimizer.state_dict())
        )
        candidate_parameters = dict(candidate_model.named_parameters())
        for name, parameter in recovered_model.named_parameters():
            candidate = candidate_parameters[name]
            candidate.grad = (
                None if parameter.grad is None else parameter.grad.detach().clone()
            )
        candidate_optimizer.step()
        candidate_model_state = {
            name: tensor.detach().clone()
            for name, tensor in candidate_model.state_dict().items()
        }
        candidate_optimizer_state = copy.deepcopy(candidate_optimizer.state_dict())
        manifest["result_applied"] = True
        _transition(transaction_dir, manifest, "result_accepted", "applied")
    except BaseException:
        for name, parameter in recovered_model.named_parameters():
            previous = gradients_before[name]
            parameter.grad = None if previous is None else previous.detach().clone()
        raise
    recovered_model.load_state_dict(candidate_model_state)
    recovered_optimizer.load_state_dict(candidate_optimizer_state)
    return recovered_raw


def run_recovered_tile_transaction(
    campaign: CampaignConfig,
    block_index: int,
    transaction_dir: Path,
    *,
    dataset: PackedDataset | None = None,
    timeout_seconds: float = 60.0,
) -> RecoveredTileTransactionEvidence:
    validate_dataset_artifacts(campaign, dataset)
    if type(block_index) is not int or not 0 <= block_index < campaign.model.layers:
        raise ValueError("block index is outside the model")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 300.0
    ):
        raise ValueError("timeout must be finite and between zero and 300 seconds")
    timeout_seconds = float(timeout_seconds)
    transaction_dir = transaction_dir.resolve()
    if transaction_dir.exists():
        if transaction_dir.is_symlink() or any(transaction_dir.iterdir()):
            raise ValueError("transaction directory must be new or empty")
    else:
        transaction_dir.mkdir(parents=True, exist_ok=False)
    if transaction_dir.is_symlink():
        raise ValueError("transaction directory may not be a symlink")

    cursor = 0
    centralized = build_model(campaign)
    recovered = build_model(campaign)
    centralized_optimizer = _create_optimizer(centralized, campaign.training)
    recovered_optimizer = _create_optimizer(recovered, campaign.training)
    inputs, targets = fixture_batch(campaign, cursor, dataset)

    centralized.train()
    centralized_optimizer.zero_grad(set_to_none=True)
    centralized_loss_tensor = F.cross_entropy(
        centralized(inputs).reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="mean",
    )
    centralized_loss_tensor.backward()
    centralized_raw = _gradient_snapshot(centralized)
    torch.nn.utils.clip_grad_norm_(
        centralized.parameters(),
        campaign.training.max_gradient_norm,
    )
    centralized_clipped = _gradient_snapshot(centralized)
    centralized_optimizer.step()

    recovered.train()
    recovered_optimizer.zero_grad(set_to_none=True)
    block_input = _prefix_activation(recovered, inputs, block_index)
    input_wire = _serialize_tensors({"input": block_input.detach().clone()})
    tile_state = {
        name: tensor.detach().clone()
        for name, tensor in recovered.blocks[block_index].state_dict().items()
    }
    tile_wire = _serialize_tensors(tile_state)
    tile_state_sha256 = tensor_sha256(tile_state)
    model_sha256 = tensor_sha256(recovered.state_dict())
    identity = {
        "campaign_id": str(campaign.campaign["id"]),
        "dataset_revision": (
            dataset.revision if dataset is not None else "synthetic-fixture-v1"
        ),
        "checkpoint_model_sha256": model_sha256,
        "block_index": block_index,
        "cursor": cursor,
        "tile_sha256": _sha256_bytes(tile_wire),
        "input_sha256": _sha256_bytes(input_wire),
    }
    transaction_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    manifest: dict[str, Any] = {
        "format": "orcacolony_boundary_transaction_v1",
        "transaction_id": transaction_id,
        **identity,
        "phase": "prepared",
        "phase_history": ["prepared"],
        "result_applied": False,
        "files": {},
    }
    _record_tensor_file(transaction_dir, manifest, "tile.safetensors", tile_wire)
    _record_tensor_file(transaction_dir, manifest, "input.safetensors", input_wire)
    _write_manifest(transaction_dir, manifest)

    context = multiprocessing.get_context("spawn")
    first_worker: _TileWorkerHandle | None = None
    replacement: _TileWorkerHandle | None = None
    first_exit_code: int | None = None
    replacement_exit_code: int | None = None
    first_forward_elapsed = 0.0
    replay_forward_elapsed = 0.0
    backward_elapsed = 0.0
    recovery_seconds = 0.0
    replacement_shutdown: dict[str, object] | None = None
    first_initialization_seconds = 0.0
    replacement_initialization_seconds = 0.0
    error: BaseException | None = None
    try:
        first_worker = _start_worker(
            context,
            campaign,
            block_index,
            tile_wire,
            timeout_seconds,
            name="orcacolony-tile-before-loss",
            expected_tile_state_sha256=tile_state_sha256,
        )
        first_initialization_seconds = first_worker.initialization_seconds
        output_wire, _, first_forward_elapsed = _worker_forward(
            first_worker,
            input_wire,
            timeout_seconds,
            assignment_id=0,
        )
        _record_tensor_file(
            transaction_dir,
            manifest,
            "forward-output.safetensors",
            output_wire,
        )
        _transition(transaction_dir, manifest, "prepared", "forward_accepted")
        first_exit_code = _terminate_worker(first_worker, timeout_seconds)
        first_worker = None
        _transition(transaction_dir, manifest, "forward_accepted", "worker_lost")

        recovery_started = time.perf_counter()
        persisted_tile = _read_owned_tensor_file(
            transaction_dir,
            manifest,
            "tile.safetensors",
        )
        persisted_input = _read_owned_tensor_file(
            transaction_dir,
            manifest,
            "input.safetensors",
        )
        persisted_output = _read_owned_tensor_file(
            transaction_dir,
            manifest,
            "forward-output.safetensors",
        )
        replacement = _start_worker(
            context,
            campaign,
            block_index,
            persisted_tile,
            timeout_seconds,
            name="orcacolony-tile-replacement",
            expected_tile_state_sha256=tile_state_sha256,
        )
        replacement_initialization_seconds = replacement.initialization_seconds
        replay_output, _, replay_forward_elapsed = _worker_forward(
            replacement,
            persisted_input,
            timeout_seconds,
            assignment_id=0,
        )
        if replay_output != persisted_output:
            raise ValueError("replacement forward output is not byte-identical")
        _transition(transaction_dir, manifest, "worker_lost", "replay_verified")

        output_tensors = _deserialize_tensors(replay_output)
        if frozenset(output_tensors) != frozenset({"output"}):
            raise ValueError("persisted forward output schema is invalid")
        boundary_output = _validate_tensor(
            output_tensors["output"],
            shape=block_input.shape,
            label="output",
        ).requires_grad_(True)
        recovered_logits = _suffix_logits(recovered, boundary_output, block_index)
        recovered_loss_tensor = F.cross_entropy(
            recovered_logits.reshape(-1, campaign.model.vocabulary_size),
            targets.reshape(-1),
            reduction="mean",
        )
        recovered_loss_tensor.backward()
        if boundary_output.grad is None:
            raise AssertionError("recovered boundary lacks output adjoint")
        adjoint_wire = _serialize_tensors(
            {"output_adjoint": boundary_output.grad.detach().clone()}
        )
        _record_tensor_file(
            transaction_dir,
            manifest,
            "output-adjoint.safetensors",
            adjoint_wire,
        )
        _transition(
            transaction_dir,
            manifest,
            "replay_verified",
            "adjoint_persisted",
        )
        result_wire, _, backward_elapsed = _worker_backward(
            replacement,
            adjoint_wire,
            timeout_seconds,
            assignment_id=0,
        )
        _record_tensor_file(
            transaction_dir,
            manifest,
            "result.safetensors",
            result_wire,
        )
        _transition(
            transaction_dir,
            manifest,
            "adjoint_persisted",
            "result_accepted",
        )
        recovery_seconds = time.perf_counter() - recovery_started
        recovered_raw = _apply_result_once(
            transaction_dir,
            recovered,
            recovered_optimizer,
            block_input,
            block_index,
            campaign,
            expected_identity=identity,
        )
        duplicate_rejected = False
        try:
            _apply_result_once(
                transaction_dir,
                recovered,
                recovered_optimizer,
                block_input,
                block_index,
                campaign,
                expected_identity=identity,
            )
        except ValueError as exc:
            duplicate_rejected = "already applied" in str(exc)
        if not duplicate_rejected:
            raise AssertionError("duplicate result was not rejected")
        replacement_exit_code, replacement_shutdown = _stop_worker(
            replacement,
            timeout_seconds,
        )
        replacement = None
    except BaseException as exc:
        error = exc
    finally:
        if first_worker is not None:
            _terminate_worker(first_worker, timeout_seconds)
        if replacement is not None:
            _terminate_worker(replacement, timeout_seconds)
    if error is not None:
        raise error
    if (
        first_exit_code is None
        or first_exit_code == 0
        or replacement_exit_code != 0
        or replacement_shutdown is None
    ):
        raise AssertionError("worker replacement lifecycle did not complete")

    recovered_clipped = _gradient_snapshot(recovered)
    # The snapshot above is post-step parameter gradients; capture the clipped values that
    # remain attached after optimizer.step(), matching the centralized snapshot timing.
    centralized_model = _model_snapshot(centralized)
    recovered_model = _model_snapshot(recovered)
    centralized_optimizer_snapshot = _optimizer_tensor_snapshot(
        centralized,
        centralized_optimizer,
    )
    recovered_optimizer_snapshot = _optimizer_tensor_snapshot(
        recovered,
        recovered_optimizer,
    )
    final_manifest = _load_manifest(transaction_dir)
    if (
        final_manifest.get("phase") != "applied"
        or final_manifest.get("result_applied") is not True
        or tuple(final_manifest.get("phase_history", ())) != _PHASES
    ):
        raise AssertionError("final transaction manifest is invalid")
    persisted_files: list[PersistedFileEvidence] = []
    for path in sorted(transaction_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError("transaction directory contains a non-file entry")
        payload = path.read_bytes()
        persisted_files.append(
            PersistedFileEvidence(
                name=path.name,
                size_bytes=len(payload),
                sha256=_sha256_bytes(payload),
            )
        )
    if {item.name for item in persisted_files} != {
        "manifest.json",
        *_OWNED_TENSOR_FILES,
    }:
        raise ValueError("transaction directory contains unexpected files")
    persisted_tensor_bytes = sum(
        item.size_bytes for item in persisted_files if item.name != "manifest.json"
    )
    replacement_peak = int(replacement_shutdown["peak_rss_bytes"])
    return RecoveredTileTransactionEvidence(
        format="orcacolony_recovered_tile_transaction_evidence_v2",
        campaign_id=str(campaign.campaign["id"]),
        dataset_revision=str(identity["dataset_revision"]),
        start_method=context.get_start_method(),
        transaction_id=transaction_id,
        checkpoint_model_sha256=model_sha256,
        block_index=block_index,
        cursor=cursor,
        phase_history=_PHASES,
        worker_model_transmissions=2,
        first_worker_terminated=True,
        first_worker_exit_code=first_exit_code,
        replacement_worker_exit_code=replacement_exit_code,
        replay_output_bytes_identical=True,
        duplicate_result_rejected=True,
        tile_model_wire_bytes=len(tile_wire),
        input_wire_bytes=len(input_wire),
        forward_output_wire_bytes=len(persisted_output),
        output_adjoint_wire_bytes=len(adjoint_wire),
        result_wire_bytes=len(result_wire),
        recovery_retransmitted_tensor_bytes=len(tile_wire) + len(input_wire),
        recovery_total_tensor_bytes=(
            len(tile_wire)
            + len(input_wire)
            + len(replay_output)
            + len(adjoint_wire)
            + len(result_wire)
        ),
        persisted_file_count=len(persisted_files),
        persisted_tensor_bytes=persisted_tensor_bytes,
        persisted_files=tuple(persisted_files),
        recovery_seconds=recovery_seconds,
        first_worker_initialization_seconds=first_initialization_seconds,
        replacement_worker_initialization_seconds=replacement_initialization_seconds,
        first_forward_seconds=first_forward_elapsed,
        replay_forward_seconds=replay_forward_elapsed,
        replacement_backward_seconds=backward_elapsed,
        replacement_worker_peak_rss_bytes=replacement_peak,
        centralized_loss=float(centralized_loss_tensor.detach()),
        recovered_loss=float(recovered_loss_tensor.detach()),
        max_abs_raw_gradient_difference=_max_abs_difference(
            centralized_raw,
            recovered_raw,
        ),
        max_abs_clipped_gradient_difference=_max_abs_difference(
            centralized_clipped,
            recovered_clipped,
        ),
        max_abs_model_difference=_max_abs_difference(
            centralized_model,
            recovered_model,
        ),
        centralized_raw_gradient_sha256=tensor_sha256(centralized_raw),
        recovered_raw_gradient_sha256=tensor_sha256(recovered_raw),
        centralized_clipped_gradient_sha256=tensor_sha256(centralized_clipped),
        recovered_clipped_gradient_sha256=tensor_sha256(recovered_clipped),
        centralized_optimizer_sha256=tensor_sha256(
            centralized_optimizer_snapshot
        ),
        recovered_optimizer_sha256=tensor_sha256(recovered_optimizer_snapshot),
        centralized_model_sha256=tensor_sha256(centralized_model),
        recovered_model_sha256=tensor_sha256(recovered_model),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crash and recover one exact persisted tile transaction"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--transaction-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config)
    if campaign.dataset is None:
        if args.dataset is not None:
            parser.error("--dataset is only valid for data-backed campaign configs")
        dataset = None
    else:
        if args.dataset is None:
            parser.error("data-backed campaign configs require --dataset")
        dataset = PackedDataset.load(args.dataset)
    evidence = run_recovered_tile_transaction(
        campaign,
        args.block_index,
        args.transaction_dir,
        dataset=dataset,
        timeout_seconds=args.timeout_seconds,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
