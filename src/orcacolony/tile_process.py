from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import Tensor

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import (
    CampaignConfig,
    DecoderBlock,
    ModelConfig,
    _create_optimizer,
    build_model,
    configure_determinism,
    fixture_batch,
    load_campaign,
    tensor_sha256,
    validate_dataset_artifacts,
)
from orcacolony.tiled_model import (
    _gradient_snapshot,
    _max_abs_difference,
    _model_snapshot,
    _optimizer_tensor_snapshot,
    _prefix_activation,
    _suffix_logits,
)


@dataclass(frozen=True)
class ProcessTileAssignmentEvidence:
    cursor: int
    forward_input_wire_bytes: int
    forward_output_wire_bytes: int
    backward_output_adjoint_wire_bytes: int
    backward_result_wire_bytes: int
    boundary_tensor_wire_bytes: int
    control_json_wire_bytes: int
    total_application_wire_bytes: int
    centralized_loss: float
    process_tiled_loss: float
    max_abs_raw_gradient_difference: float
    max_abs_clipped_gradient_difference: float
    max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    process_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    process_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    process_optimizer_sha256: str
    centralized_model_sha256: str
    process_model_sha256: str
    forward_round_trip_seconds: float
    backward_round_trip_seconds: float
    worker_forward_seconds: float
    worker_backward_seconds: float
    worker_current_rss_bytes: int
    worker_peak_rss_bytes: int


@dataclass(frozen=True)
class PersistentTileProcessEvidence:
    format: str
    campaign_id: str
    dataset_revision: str
    start_method: str
    block_index: int
    assignment_count: int
    model_transmissions: int
    tile_forward_calls: int
    tile_state_sha256: str
    tile_model_wire_bytes: int
    initialization_round_trip_seconds: float
    initialization_control_json_wire_bytes: int
    worker_startup_current_rss_bytes: int
    worker_startup_peak_rss_bytes: int
    worker_after_model_current_rss_bytes: int
    worker_after_model_peak_rss_bytes: int
    worker_model_current_rss_delta_bytes: int
    worker_final_peak_rss_bytes: int
    cold_assignment_tensor_wire_bytes: int
    warm_assignment_tensor_wire_bytes: int
    assignments: tuple[ProcessTileAssignmentEvidence, ...]
    child_exit_code: int


def _process_memory_bytes() -> tuple[int, int]:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        if not get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    try:
        import resource
    except ImportError as exc:
        raise RuntimeError("process RSS observation is unavailable") from exc
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak = peak if sys.platform == "darwin" else peak * 1024
    current = peak
    statm = "/proc/self/statm"
    if os.path.exists(statm):
        with open(statm, encoding="ascii") as stream:
            fields = stream.read().split()
        if len(fields) >= 2:
            current = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    return current, peak


def _serialize_tensors(tensors: dict[str, Tensor]) -> bytes:
    owned = {
        name: tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    return save_safetensors(owned)


def _deserialize_tensors(payload: bytes) -> dict[str, Tensor]:
    return {name: tensor.clone() for name, tensor in load_safetensors(payload).items()}


def _send_json(connection: Connection, payload: dict[str, object]) -> int:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    connection.send_bytes(raw)
    return len(raw)


def _recv_bytes(
    connection: Connection,
    timeout_seconds: float,
    *,
    label: str,
) -> bytes:
    if not connection.poll(timeout_seconds):
        raise TimeoutError(f"timed out waiting for {label}")
    return connection.recv_bytes()


def _recv_json(
    connection: Connection,
    timeout_seconds: float,
    *,
    label: str,
) -> tuple[dict[str, object], int]:
    raw = _recv_bytes(connection, timeout_seconds, label=label)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    if payload.get("status") == "error":
        raise RuntimeError(str(payload.get("message", "tile worker failed")))
    return payload, len(raw)


def _validate_tensor(
    tensor: Tensor,
    *,
    shape: tuple[int, ...],
    label: str,
) -> Tensor:
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
        raise ValueError(f"{label} tensor contract is invalid")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} tensor is non-finite")
    return tensor.detach().clone()


def _tile_worker_entry(connection: Connection) -> None:
    try:
        startup_current, startup_peak = _process_memory_bytes()
        init_raw = connection.recv_bytes()
        init = json.loads(init_raw)
        if not isinstance(init, dict) or frozenset(init) != frozenset(
            {"op", "block_index", "model", "seed"}
        ):
            raise ValueError("tile init schema is invalid")
        if (
            init["op"] != "init"
            or type(init["block_index"]) is not int
            or type(init["seed"]) is not int
        ):
            raise ValueError("tile init semantics are invalid")
        model_payload = init["model"]
        if not isinstance(model_payload, dict):
            raise ValueError("tile model config is invalid")
        config = ModelConfig(**model_payload)
        if not 0 <= init["block_index"] < config.layers:
            raise ValueError("tile block index is outside the configured model")
        configure_determinism(init["seed"])
        tile = DecoderBlock(config)
        expected_state = tile.state_dict()
        model_wire = connection.recv_bytes()
        loaded_state = _deserialize_tensors(model_wire)
        if loaded_state.keys() != expected_state.keys():
            raise ValueError("tile model tensor names are invalid")
        for name, expected in expected_state.items():
            tensor = loaded_state[name]
            if (
                tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"tile model tensor is invalid: {name}")
        tile.load_state_dict(loaded_state, strict=True)
        tile.train()
        after_model_current, after_model_peak = _process_memory_bytes()
        tile_state_sha256 = tensor_sha256(tile.state_dict())
        forward_calls = 0
        active_assignment: int | None = None
        tile_input: Tensor | None = None
        tile_output: Tensor | None = None
        _send_json(
            connection,
            {
                "status": "ready",
                "tile_state_sha256": tile_state_sha256,
                "startup_current_rss_bytes": startup_current,
                "startup_peak_rss_bytes": startup_peak,
                "after_model_current_rss_bytes": after_model_current,
                "after_model_peak_rss_bytes": after_model_peak,
            },
        )

        while True:
            control_raw = connection.recv_bytes()
            control = json.loads(control_raw)
            if not isinstance(control, dict) or "op" not in control:
                raise ValueError("tile control schema is invalid")
            if control["op"] == "shutdown":
                current, peak = _process_memory_bytes()
                _send_json(
                    connection,
                    {
                        "status": "stopped",
                        "tile_forward_calls": forward_calls,
                        "current_rss_bytes": current,
                        "peak_rss_bytes": peak,
                    },
                )
                return
            if frozenset(control) != frozenset({"op", "assignment_id"}):
                raise ValueError("tile assignment control schema is invalid")
            assignment_id = control["assignment_id"]
            if type(assignment_id) is not int or assignment_id < 0:
                raise ValueError("tile assignment identity is invalid")

            if control["op"] == "forward":
                if active_assignment is not None:
                    raise ValueError("tile already has an active assignment")
                payload = _deserialize_tensors(connection.recv_bytes())
                if frozenset(payload) != frozenset({"input"}):
                    raise ValueError("tile forward tensor schema is invalid")
                input_tensor = payload["input"]
                if (
                    input_tensor.dtype != torch.float32
                    or input_tensor.ndim != 3
                    or input_tensor.shape[-1] != config.width
                    or input_tensor.shape[1] > config.context_length
                    or not bool(torch.isfinite(input_tensor).all())
                ):
                    raise ValueError("tile input activation is invalid")
                tile.zero_grad(set_to_none=True)
                tile_input = input_tensor.detach().clone().requires_grad_(True)
                started = time.perf_counter()
                tile_output = tile(tile_input)
                compute_seconds = time.perf_counter() - started
                forward_calls += 1
                active_assignment = assignment_id
                output_wire = _serialize_tensors({"output": tile_output})
                current, peak = _process_memory_bytes()
                _send_json(
                    connection,
                    {
                        "status": "forwarded",
                        "assignment_id": assignment_id,
                        "compute_seconds": compute_seconds,
                        "current_rss_bytes": current,
                        "peak_rss_bytes": peak,
                    },
                )
                connection.send_bytes(output_wire)
                continue

            if control["op"] == "backward":
                if (
                    active_assignment != assignment_id
                    or tile_input is None
                    or tile_output is None
                ):
                    raise ValueError("tile backward does not match active forward")
                payload = _deserialize_tensors(connection.recv_bytes())
                if frozenset(payload) != frozenset({"output_adjoint"}):
                    raise ValueError("tile backward tensor schema is invalid")
                output_adjoint = _validate_tensor(
                    payload["output_adjoint"],
                    shape=tuple(tile_output.shape),
                    label="output adjoint",
                )
                started = time.perf_counter()
                tile_output.backward(output_adjoint)
                compute_seconds = time.perf_counter() - started
                if tile_input.grad is None:
                    raise AssertionError("tile did not produce an input adjoint")
                result: dict[str, Tensor] = {
                    "input_adjoint": tile_input.grad.detach().clone()
                }
                for name, parameter in tile.named_parameters():
                    if parameter.grad is None:
                        raise AssertionError(f"tile parameter lacks gradient: {name}")
                    result[f"gradient.{name}"] = parameter.grad.detach().clone()
                result_wire = _serialize_tensors(result)
                current, peak = _process_memory_bytes()
                _send_json(
                    connection,
                    {
                        "status": "backwarded",
                        "assignment_id": assignment_id,
                        "compute_seconds": compute_seconds,
                        "current_rss_bytes": current,
                        "peak_rss_bytes": peak,
                    },
                )
                connection.send_bytes(result_wire)
                active_assignment = None
                tile_input = None
                tile_output = None
                continue
            raise ValueError("unsupported tile operation")
    except BaseException as exc:
        try:
            _send_json(
                connection,
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


def _run_parent_assignment(
    connection: Connection,
    campaign: CampaignConfig,
    dataset: PackedDataset | None,
    *,
    block_index: int,
    assignment_id: int,
    cursor: int,
    timeout_seconds: float,
) -> ProcessTileAssignmentEvidence:
    centralized = build_model(campaign)
    process_model = build_model(campaign)
    centralized_optimizer = _create_optimizer(centralized, campaign.training)
    process_optimizer = _create_optimizer(process_model, campaign.training)
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

    process_model.train()
    process_optimizer.zero_grad(set_to_none=True)
    block_input = _prefix_activation(process_model, inputs, block_index)
    input_wire = _serialize_tensors({"input": block_input})
    control_bytes = _send_json(
        connection,
        {"op": "forward", "assignment_id": assignment_id},
    )
    forward_started = time.perf_counter()
    connection.send_bytes(input_wire)
    forward_ack, ack_bytes = _recv_json(
        connection,
        timeout_seconds,
        label="tile forward acknowledgement",
    )
    output_wire = _recv_bytes(
        connection,
        timeout_seconds,
        label="tile forward output",
    )
    forward_round_trip_seconds = time.perf_counter() - forward_started
    control_bytes += ack_bytes
    if (
        frozenset(forward_ack)
        != frozenset(
            {
                "status",
                "assignment_id",
                "compute_seconds",
                "current_rss_bytes",
                "peak_rss_bytes",
            }
        )
        or forward_ack["status"] != "forwarded"
        or forward_ack["assignment_id"] != assignment_id
    ):
        raise ValueError("tile forward acknowledgement is invalid")
    output_payload = _deserialize_tensors(output_wire)
    if frozenset(output_payload) != frozenset({"output"}):
        raise ValueError("tile forward output schema is invalid")
    output = _validate_tensor(
        output_payload["output"],
        shape=tuple(block_input.shape),
        label="tile output activation",
    )

    boundary_output = output.detach().clone().requires_grad_(True)
    process_logits = _suffix_logits(process_model, boundary_output, block_index)
    process_loss_tensor = F.cross_entropy(
        process_logits.reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="mean",
    )
    process_loss_tensor.backward()
    if boundary_output.grad is None:
        raise AssertionError("coordinator suffix did not produce an output adjoint")
    adjoint_wire = _serialize_tensors(
        {"output_adjoint": boundary_output.grad.detach().clone()}
    )
    control_bytes += _send_json(
        connection,
        {"op": "backward", "assignment_id": assignment_id},
    )
    backward_started = time.perf_counter()
    connection.send_bytes(adjoint_wire)
    backward_ack, ack_bytes = _recv_json(
        connection,
        timeout_seconds,
        label="tile backward acknowledgement",
    )
    result_wire = _recv_bytes(
        connection,
        timeout_seconds,
        label="tile backward result",
    )
    backward_round_trip_seconds = time.perf_counter() - backward_started
    control_bytes += ack_bytes
    if (
        frozenset(backward_ack)
        != frozenset(
            {
                "status",
                "assignment_id",
                "compute_seconds",
                "current_rss_bytes",
                "peak_rss_bytes",
            }
        )
        or backward_ack["status"] != "backwarded"
        or backward_ack["assignment_id"] != assignment_id
    ):
        raise ValueError("tile backward acknowledgement is invalid")

    result = _deserialize_tensors(result_wire)
    expected_gradient_names = {
        f"gradient.{name}"
        for name, _ in process_model.blocks[block_index].named_parameters()
    }
    if frozenset(result) != frozenset({"input_adjoint", *expected_gradient_names}):
        raise ValueError("tile backward result schema is invalid")
    input_adjoint = _validate_tensor(
        result["input_adjoint"],
        shape=tuple(block_input.shape),
        label="input adjoint",
    )
    for name, parameter in process_model.blocks[block_index].named_parameters():
        gradient = result[f"gradient.{name}"]
        if (
            gradient.dtype != parameter.dtype
            or gradient.shape != parameter.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(f"tile block gradient is invalid: {name}")
        parameter.grad = gradient.detach().clone()
    block_input.backward(input_adjoint)
    process_raw = _gradient_snapshot(process_model)
    torch.nn.utils.clip_grad_norm_(
        process_model.parameters(),
        campaign.training.max_gradient_norm,
    )
    process_clipped = _gradient_snapshot(process_model)
    process_optimizer.step()

    centralized_model = _model_snapshot(centralized)
    process_model_snapshot = _model_snapshot(process_model)
    centralized_optimizer_snapshot = _optimizer_tensor_snapshot(
        centralized,
        centralized_optimizer,
    )
    process_optimizer_snapshot = _optimizer_tensor_snapshot(
        process_model,
        process_optimizer,
    )
    boundary_bytes = (
        len(input_wire) + len(output_wire) + len(adjoint_wire) + len(result_wire)
    )
    evidence = ProcessTileAssignmentEvidence(
        cursor=cursor,
        forward_input_wire_bytes=len(input_wire),
        forward_output_wire_bytes=len(output_wire),
        backward_output_adjoint_wire_bytes=len(adjoint_wire),
        backward_result_wire_bytes=len(result_wire),
        boundary_tensor_wire_bytes=boundary_bytes,
        control_json_wire_bytes=control_bytes,
        total_application_wire_bytes=boundary_bytes + control_bytes,
        centralized_loss=float(centralized_loss_tensor.detach()),
        process_tiled_loss=float(process_loss_tensor.detach()),
        max_abs_raw_gradient_difference=_max_abs_difference(
            centralized_raw,
            process_raw,
        ),
        max_abs_clipped_gradient_difference=_max_abs_difference(
            centralized_clipped,
            process_clipped,
        ),
        max_abs_model_difference=_max_abs_difference(
            centralized_model,
            process_model_snapshot,
        ),
        centralized_raw_gradient_sha256=tensor_sha256(centralized_raw),
        process_raw_gradient_sha256=tensor_sha256(process_raw),
        centralized_clipped_gradient_sha256=tensor_sha256(centralized_clipped),
        process_clipped_gradient_sha256=tensor_sha256(process_clipped),
        centralized_optimizer_sha256=tensor_sha256(
            centralized_optimizer_snapshot
        ),
        process_optimizer_sha256=tensor_sha256(process_optimizer_snapshot),
        centralized_model_sha256=tensor_sha256(centralized_model),
        process_model_sha256=tensor_sha256(process_model_snapshot),
        forward_round_trip_seconds=forward_round_trip_seconds,
        backward_round_trip_seconds=backward_round_trip_seconds,
        worker_forward_seconds=float(forward_ack["compute_seconds"]),
        worker_backward_seconds=float(backward_ack["compute_seconds"]),
        worker_current_rss_bytes=int(backward_ack["current_rss_bytes"]),
        worker_peak_rss_bytes=int(backward_ack["peak_rss_bytes"]),
    )
    return evidence


def run_persistent_tile_process_experiment(
    campaign: CampaignConfig,
    *,
    block_index: int,
    dataset: PackedDataset | None = None,
    timeout_seconds: float = 60.0,
) -> PersistentTileProcessEvidence:
    validate_dataset_artifacts(campaign, dataset)
    if type(block_index) is not int or not 0 <= block_index < campaign.model.layers:
        raise ValueError("block index is outside the configured model")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 300.0
    ):
        raise ValueError("timeout must be finite and between zero and 300 seconds")
    timeout_seconds = float(timeout_seconds)

    source_model = build_model(campaign)
    tile_state = {
        name: tensor.detach().clone()
        for name, tensor in source_model.blocks[block_index].state_dict().items()
    }
    tile_state_sha256 = tensor_sha256(tile_state)
    tile_model_wire = _serialize_tensors(tile_state)
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_tile_worker_entry,
        args=(child_connection,),
        name="orcacolony-tile-worker",
        daemon=False,
    )
    initialization_started = time.perf_counter()
    process.start()
    child_connection.close()
    assignments: list[ProcessTileAssignmentEvidence] = []
    init_control_bytes = 0
    shutdown_control_bytes = 0
    shutdown_ack: dict[str, object] | None = None
    error: BaseException | None = None
    try:
        init_control_bytes = _send_json(
            parent_connection,
            {
                "op": "init",
                "block_index": block_index,
                "model": asdict(campaign.model),
                "seed": campaign.training.seed,
            },
        )
        parent_connection.send_bytes(tile_model_wire)
        ready, ready_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label="tile initialization",
        )
        initialization_round_trip_seconds = (
            time.perf_counter() - initialization_started
        )
        init_control_bytes += ready_bytes
        expected_ready_fields = frozenset(
            {
                "status",
                "tile_state_sha256",
                "startup_current_rss_bytes",
                "startup_peak_rss_bytes",
                "after_model_current_rss_bytes",
                "after_model_peak_rss_bytes",
            }
        )
        if (
            frozenset(ready) != expected_ready_fields
            or ready["status"] != "ready"
            or ready["tile_state_sha256"] != tile_state_sha256
        ):
            raise ValueError("tile initialization acknowledgement is invalid")

        cursors = (
            0,
            campaign.training.batch_size % campaign.training.dataset_sequences,
        )
        for assignment_id, cursor in enumerate(cursors):
            assignment = _run_parent_assignment(
                parent_connection,
                campaign,
                dataset,
                block_index=block_index,
                assignment_id=assignment_id,
                cursor=cursor,
                timeout_seconds=timeout_seconds,
            )
            assignments.append(assignment)

        shutdown_control_bytes = _send_json(
            parent_connection,
            {"op": "shutdown"},
        )
        shutdown_ack, ack_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label="tile shutdown",
        )
        shutdown_control_bytes += ack_bytes
        if (
            frozenset(shutdown_ack)
            != frozenset(
                {
                    "status",
                    "tile_forward_calls",
                    "current_rss_bytes",
                    "peak_rss_bytes",
                }
            )
            or shutdown_ack["status"] != "stopped"
        ):
            raise ValueError("tile shutdown acknowledgement is invalid")
    except BaseException as exc:
        error = exc
    finally:
        parent_connection.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
    if error is not None:
        raise error
    if process.exitcode is None:
        raise RuntimeError("tile worker has no exit code")
    if process.exitcode != 0:
        raise RuntimeError(f"tile worker exited with code {process.exitcode}")
    if shutdown_ack is None or len(assignments) != 2:
        raise AssertionError("tile process experiment did not complete")

    cold_assignment_bytes = len(tile_model_wire) + assignments[0].boundary_tensor_wire_bytes
    warm_assignment_bytes = assignments[1].boundary_tensor_wire_bytes
    dataset_revision = (
        dataset.revision if dataset is not None else "synthetic-fixture-v1"
    )
    return PersistentTileProcessEvidence(
        format="orcacolony_persistent_tile_process_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        dataset_revision=dataset_revision,
        start_method=context.get_start_method(),
        block_index=block_index,
        assignment_count=len(assignments),
        model_transmissions=1,
        tile_forward_calls=int(shutdown_ack["tile_forward_calls"]),
        tile_state_sha256=tile_state_sha256,
        tile_model_wire_bytes=len(tile_model_wire),
        initialization_round_trip_seconds=initialization_round_trip_seconds,
        initialization_control_json_wire_bytes=(
            init_control_bytes + shutdown_control_bytes
        ),
        worker_startup_current_rss_bytes=int(
            ready["startup_current_rss_bytes"]
        ),
        worker_startup_peak_rss_bytes=int(ready["startup_peak_rss_bytes"]),
        worker_after_model_current_rss_bytes=int(
            ready["after_model_current_rss_bytes"]
        ),
        worker_after_model_peak_rss_bytes=int(
            ready["after_model_peak_rss_bytes"]
        ),
        worker_model_current_rss_delta_bytes=(
            int(ready["after_model_current_rss_bytes"])
            - int(ready["startup_current_rss_bytes"])
        ),
        worker_final_peak_rss_bytes=int(shutdown_ack["peak_rss_bytes"]),
        cold_assignment_tensor_wire_bytes=cold_assignment_bytes,
        warm_assignment_tensor_wire_bytes=warm_assignment_bytes,
        assignments=tuple(assignments),
        child_exit_code=int(process.exitcode),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent process-separated OrcaColony tile experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--block-index", type=int, required=True)
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
    evidence = run_persistent_tile_process_experiment(
        campaign,
        block_index=args.block_index,
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
