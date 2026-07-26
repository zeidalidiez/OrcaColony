from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from orcacolony.reference import (
    CampaignConfig,
    VolunteerDecoder,
    _create_optimizer,
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
)


@dataclass(frozen=True)
class TiledBlockEvidence:
    format: str
    block_index: int
    boundary_shape: tuple[int, ...]
    full_parameter_count: int
    tile_parameter_count: int
    full_model_payload_tensor_bytes: int
    tile_model_payload_tensor_bytes: int
    input_activation_tensor_bytes: int
    output_activation_tensor_bytes: int
    output_adjoint_tensor_bytes: int
    input_adjoint_tensor_bytes: int
    forward_boundary_transfer_tensor_bytes: int
    backward_boundary_transfer_tensor_bytes: int
    tile_gradient_upload_tensor_bytes: int
    cold_assignment_transfer_tensor_bytes: int
    warm_assignment_transfer_tensor_bytes: int
    full_replica_round_trip_tensor_bytes: int
    accounted_tile_tensor_bytes: int
    coordinator_selected_block_forward_calls: int
    tile_forward_calls: int
    centralized_loss: float
    tiled_loss: float
    max_abs_raw_gradient_difference: float
    max_abs_clipped_gradient_difference: float
    max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    tiled_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    tiled_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    tiled_optimizer_sha256: str
    centralized_model_sha256: str
    tiled_model_sha256: str
    centralized_step_seconds: float
    tiled_step_seconds: float
    combined_process_peak_rss_bytes: int | None


def _module_tensor_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _gradient_snapshot(model: VolunteerDecoder) -> dict[str, Tensor]:
    gradients: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"model parameter lacks gradient: {name}")
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _optimizer_tensor_snapshot(
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if state is None:
            raise AssertionError(f"model parameter lacks optimizer state: {name}")
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            value = state.get(state_name)
            if not isinstance(value, Tensor):
                raise AssertionError(
                    f"optimizer state is not a tensor: {state_name}.{name}"
                )
            tensors[f"{state_name}.{name}"] = value.detach().clone()
    return tensors


def _max_abs_difference(
    left: dict[str, Tensor],
    right: dict[str, Tensor],
) -> float:
    if left.keys() != right.keys():
        raise AssertionError("tensor mappings have different names")
    maximum = 0.0
    for name in left:
        if left[name].shape != right[name].shape:
            raise AssertionError(f"tensor mappings have different shapes: {name}")
        difference = float((left[name] - right[name]).abs().max())
        maximum = max(maximum, difference)
    return maximum


def _model_snapshot(model: VolunteerDecoder) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _peak_process_rss_bytes() -> int | None:
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
        return int(counters.PeakWorkingSetSize)
    try:
        import resource
    except ImportError:
        return None
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def _prefix_activation(
    model: VolunteerDecoder,
    token_ids: Tensor,
    block_index: int,
) -> Tensor:
    _, tokens = token_ids.shape
    if tokens > model.config.context_length:
        raise ValueError("input exceeds configured context length")
    positions = torch.arange(tokens, device=token_ids.device)
    hidden = model.token_embedding(token_ids) + model.position_embedding(positions)
    for block in model.blocks[:block_index]:
        hidden = block(hidden)
    return hidden


def _suffix_logits(
    model: VolunteerDecoder,
    hidden: Tensor,
    block_index: int,
) -> Tensor:
    for block in model.blocks[block_index + 1 :]:
        hidden = block(hidden)
    hidden = model.final_norm(hidden)
    return F.linear(hidden, model.token_embedding.weight)


def run_tiled_block_experiment(
    campaign: CampaignConfig,
    *,
    block_index: int,
) -> TiledBlockEvidence:
    if campaign.dataset is not None:
        raise ValueError("the first tiled-block experiment supports the T0 fixture only")
    if block_index < 0 or block_index >= campaign.model.layers:
        raise ValueError("block index is outside the configured model")

    centralized = build_model(campaign)
    tiled = build_model(campaign)
    tile = copy.deepcopy(tiled.blocks[block_index])
    centralized_optimizer = _create_optimizer(centralized, campaign.training)
    tiled_optimizer = _create_optimizer(tiled, campaign.training)
    inputs, targets = fixture_batch(campaign, 0)

    centralized.train()
    centralized_optimizer.zero_grad(set_to_none=True)
    centralized_started = time.perf_counter()
    centralized_loss_tensor = F.cross_entropy(
        centralized(inputs).reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="mean",
    )
    centralized_loss_tensor.backward()
    centralized_raw_gradients = _gradient_snapshot(centralized)
    torch.nn.utils.clip_grad_norm_(
        centralized.parameters(),
        campaign.training.max_gradient_norm,
    )
    centralized_clipped_gradients = _gradient_snapshot(centralized)
    centralized_optimizer.step()
    centralized_step_seconds = time.perf_counter() - centralized_started

    coordinator_selected_block_forward_calls = 0
    tile_forward_calls = 0

    def count_coordinator_forward(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        _output: Tensor,
    ) -> None:
        nonlocal coordinator_selected_block_forward_calls
        coordinator_selected_block_forward_calls += 1

    def count_tile_forward(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        _output: Tensor,
    ) -> None:
        nonlocal tile_forward_calls
        tile_forward_calls += 1

    coordinator_hook = tiled.blocks[block_index].register_forward_hook(
        count_coordinator_forward
    )
    tile_hook = tile.register_forward_hook(count_tile_forward)
    try:
        tiled.train()
        tile.train()
        tiled_optimizer.zero_grad(set_to_none=True)
        tiled_started = time.perf_counter()

        block_input = _prefix_activation(tiled, inputs, block_index)
        tile_input = block_input.detach().clone().requires_grad_(True)
        tile_output = tile(tile_input)
        boundary_output = tile_output.detach().clone().requires_grad_(True)
        tiled_logits = _suffix_logits(tiled, boundary_output, block_index)
        tiled_loss_tensor = F.cross_entropy(
            tiled_logits.reshape(-1, campaign.model.vocabulary_size),
            targets.reshape(-1),
            reduction="mean",
        )
        tiled_loss_tensor.backward()
        if boundary_output.grad is None:
            raise AssertionError("coordinator suffix did not produce an output adjoint")
        output_adjoint = boundary_output.grad.detach().clone()

        tile_output.backward(output_adjoint)
        if tile_input.grad is None:
            raise AssertionError("tile did not produce an input adjoint")
        input_adjoint = tile_input.grad.detach().clone()

        coordinator_block_parameters = dict(
            tiled.blocks[block_index].named_parameters()
        )
        tile_parameters = dict(tile.named_parameters())
        if coordinator_block_parameters.keys() != tile_parameters.keys():
            raise AssertionError("tile parameter mapping does not match coordinator block")
        for name, parameter in tile_parameters.items():
            if parameter.grad is None:
                raise AssertionError(f"tile parameter lacks gradient: {name}")
            coordinator_block_parameters[name].grad = parameter.grad.detach().clone()

        block_input.backward(input_adjoint)
        tiled_raw_gradients = _gradient_snapshot(tiled)
        torch.nn.utils.clip_grad_norm_(
            tiled.parameters(),
            campaign.training.max_gradient_norm,
        )
        tiled_clipped_gradients = _gradient_snapshot(tiled)
        tiled_optimizer.step()
        tiled_step_seconds = time.perf_counter() - tiled_started
    finally:
        coordinator_hook.remove()
        tile_hook.remove()

    centralized_model = _model_snapshot(centralized)
    tiled_model = _model_snapshot(tiled)
    centralized_optimizer_tensors = _optimizer_tensor_snapshot(
        centralized,
        centralized_optimizer,
    )
    tiled_optimizer_tensors = _optimizer_tensor_snapshot(tiled, tiled_optimizer)

    tile_model_payload_bytes = _module_tensor_bytes(tile)
    tile_gradient_upload_bytes = sum(
        _tensor_bytes(parameter.grad)
        for parameter in tile.parameters()
        if parameter.grad is not None
    )
    input_activation_bytes = _tensor_bytes(tile_input)
    output_activation_bytes = _tensor_bytes(tile_output)
    output_adjoint_bytes = _tensor_bytes(output_adjoint)
    input_adjoint_bytes = _tensor_bytes(input_adjoint)
    forward_boundary_bytes = input_activation_bytes + output_activation_bytes
    backward_boundary_bytes = output_adjoint_bytes + input_adjoint_bytes
    cold_assignment_bytes = (
        tile_model_payload_bytes
        + forward_boundary_bytes
        + backward_boundary_bytes
        + tile_gradient_upload_bytes
    )
    return TiledBlockEvidence(
        format="orcacolony_tiled_block_evidence_v1",
        block_index=block_index,
        boundary_shape=tuple(tile_input.shape),
        full_parameter_count=sum(
            parameter.numel() for parameter in centralized.parameters()
        ),
        tile_parameter_count=sum(parameter.numel() for parameter in tile.parameters()),
        full_model_payload_tensor_bytes=_module_tensor_bytes(centralized),
        tile_model_payload_tensor_bytes=tile_model_payload_bytes,
        input_activation_tensor_bytes=input_activation_bytes,
        output_activation_tensor_bytes=output_activation_bytes,
        output_adjoint_tensor_bytes=output_adjoint_bytes,
        input_adjoint_tensor_bytes=input_adjoint_bytes,
        forward_boundary_transfer_tensor_bytes=forward_boundary_bytes,
        backward_boundary_transfer_tensor_bytes=backward_boundary_bytes,
        tile_gradient_upload_tensor_bytes=tile_gradient_upload_bytes,
        cold_assignment_transfer_tensor_bytes=cold_assignment_bytes,
        warm_assignment_transfer_tensor_bytes=(
            cold_assignment_bytes - tile_model_payload_bytes
        ),
        full_replica_round_trip_tensor_bytes=2 * _module_tensor_bytes(centralized),
        accounted_tile_tensor_bytes=(
            tile_model_payload_bytes
            + tile_gradient_upload_bytes
            + input_activation_bytes
            + output_activation_bytes
            + output_adjoint_bytes
            + input_adjoint_bytes
        ),
        coordinator_selected_block_forward_calls=(
            coordinator_selected_block_forward_calls
        ),
        tile_forward_calls=tile_forward_calls,
        centralized_loss=float(centralized_loss_tensor.detach()),
        tiled_loss=float(tiled_loss_tensor.detach()),
        max_abs_raw_gradient_difference=_max_abs_difference(
            centralized_raw_gradients,
            tiled_raw_gradients,
        ),
        max_abs_clipped_gradient_difference=_max_abs_difference(
            centralized_clipped_gradients,
            tiled_clipped_gradients,
        ),
        max_abs_model_difference=_max_abs_difference(
            centralized_model,
            tiled_model,
        ),
        centralized_raw_gradient_sha256=tensor_sha256(
            centralized_raw_gradients
        ),
        tiled_raw_gradient_sha256=tensor_sha256(tiled_raw_gradients),
        centralized_clipped_gradient_sha256=tensor_sha256(
            centralized_clipped_gradients
        ),
        tiled_clipped_gradient_sha256=tensor_sha256(tiled_clipped_gradients),
        centralized_optimizer_sha256=tensor_sha256(
            centralized_optimizer_tensors
        ),
        tiled_optimizer_sha256=tensor_sha256(tiled_optimizer_tensors),
        centralized_model_sha256=tensor_sha256(centralized_model),
        tiled_model_sha256=tensor_sha256(tiled_model),
        centralized_step_seconds=centralized_step_seconds,
        tiled_step_seconds=tiled_step_seconds,
        combined_process_peak_rss_bytes=_peak_process_rss_bytes(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an exact OrcaColony boundary-tiled block experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    evidence = run_tiled_block_experiment(
        load_campaign(args.config),
        block_index=args.block_index,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
