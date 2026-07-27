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

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import (
    CampaignConfig,
    VolunteerDecoder,
    _create_optimizer,
    build_model,
    evaluation_slice,
    fixture_batch,
    load_campaign,
    objective_loss_sum,
    tensor_sha256,
    validate_dataset_artifacts,
)


@dataclass(frozen=True)
class RollingBlockEvidence:
    format: str
    steps: int
    block_sequence: tuple[int, ...]
    full_parameter_count: int
    worker_parameter_count: int
    worker_trainable_parameter_count: int
    full_payload_tensor_bytes: int
    worker_payload_tensor_bytes: int
    full_resident_tensor_bytes: int
    worker_resident_tensor_bytes: int
    selected_block_payload_tensor_bytes: int
    shared_payload_tensor_bytes: int
    mapped_gradient_bytes_per_assignment: int
    transient_worker_payload_tensor_bytes: int
    unique_coverage_payload_tensor_bytes: int
    replicated_full_payload_tensor_bytes: int
    initial_mean_loss: float
    rolling_final_mean_loss: float
    full_baseline_final_mean_loss: float
    rolling_loss_improvement: float
    full_baseline_loss_improvement: float
    rolling_model_sha256: str
    full_baseline_model_sha256: str


@dataclass(frozen=True)
class RollingBlockEvaluationPoint:
    step: int
    rolling_mean_loss: float
    full_baseline_mean_loss: float


@dataclass(frozen=True)
class DatasetRollingBlockEvidence:
    format: str
    campaign_id: str
    dataset_revision: str
    steps: int
    evaluation_interval: int
    evaluation_sequences: int
    block_sequence: tuple[int, ...]
    full_parameter_count: int
    worker_parameter_count: int
    worker_trainable_parameter_count: int
    full_payload_tensor_bytes: int
    worker_payload_tensor_bytes: int
    full_resident_tensor_bytes: int
    worker_resident_tensor_bytes: int
    selected_block_payload_tensor_bytes: int
    shared_payload_tensor_bytes: int
    mapped_gradient_bytes_per_assignment: int
    shared_state_loads: int
    block_state_loads: int
    transient_worker_payload_tensor_bytes: int
    persistent_session_payload_tensor_bytes: int
    unique_coverage_payload_tensor_bytes: int
    replicated_full_payload_tensor_bytes: int
    evaluation_history: tuple[RollingBlockEvaluationPoint, ...]
    initial_validation_mean_loss: float
    rolling_final_validation_mean_loss: float
    full_baseline_final_validation_mean_loss: float
    rolling_validation_loss_improvement: float
    full_baseline_validation_loss_improvement: float
    experiment_wall_seconds: float
    rolling_training_seconds: float
    full_baseline_training_seconds: float
    evaluation_seconds: float
    combined_process_peak_rss_bytes: int | None
    rolling_model_sha256: str
    full_baseline_model_sha256: str


@dataclass(frozen=True)
class BlockShardedEvaluationPoint:
    step: int
    sharded_mean_loss: float
    full_baseline_mean_loss: float


@dataclass(frozen=True)
class BlockShardedEvidence:
    format: str
    campaign_id: str
    dataset_revision: str
    global_steps: int
    evaluation_interval: int
    evaluation_sequences: int
    workers_per_global_step: int
    worker_assignments: int
    assignment_block_sequence: tuple[int, ...]
    block_update_counts: tuple[int, ...]
    coordinator_optimizer_steps: int
    block_optimizer_steps: tuple[int, ...]
    shared_optimizer_state_parameter_count: int
    full_parameter_count: int
    worker_parameter_count: int
    worker_trainable_parameter_count: int
    full_payload_tensor_bytes: int
    worker_payload_tensor_bytes: int
    full_resident_tensor_bytes: int
    worker_resident_tensor_bytes: int
    aggregate_worker_resident_tensor_bytes: int
    selected_block_payload_tensor_bytes: int
    shared_payload_tensor_bytes: int
    mapped_gradient_bytes_per_assignment: int
    shared_state_loads: int
    block_state_loads: int
    cold_aggregate_download_tensor_bytes: int
    warm_aggregate_download_per_step_tensor_bytes: int
    persistent_aggregate_download_tensor_bytes: int
    individual_worker_persistent_download_tensor_bytes: int
    individual_worker_unique_payload_tensor_bytes: int
    colony_unique_payload_tensor_bytes: int
    mapped_gradient_upload_tensor_bytes: int
    persistent_aggregate_round_trip_tensor_bytes: int
    replicated_full_download_tensor_bytes: int
    replicated_full_round_trip_tensor_bytes: int
    initial_shared_state_sha256: str
    final_shared_state_sha256: str
    updated_block_count: int
    evaluation_history: tuple[BlockShardedEvaluationPoint, ...]
    initial_validation_mean_loss: float
    sharded_final_validation_mean_loss: float
    full_baseline_final_validation_mean_loss: float
    sharded_validation_loss_improvement: float
    full_baseline_validation_loss_improvement: float
    experiment_wall_seconds: float
    sharded_training_seconds: float
    full_baseline_training_seconds: float
    evaluation_seconds: float
    combined_process_peak_rss_bytes: int | None
    sharded_model_sha256: str
    full_baseline_model_sha256: str


class RollingBlockWorker(nn.Module):
    """Executable shallow worker containing one mapped block from a full model."""

    def __init__(self, coordinator: VolunteerDecoder, block_index: int) -> None:
        super().__init__()
        if block_index < 0 or block_index >= len(coordinator.blocks):
            raise ValueError("block index is outside the coordinator model")
        self.config = coordinator.config
        self.objective = coordinator.objective
        self.block_index = block_index
        self.token_embedding = copy.deepcopy(coordinator.token_embedding)
        self.position_embedding = copy.deepcopy(coordinator.position_embedding)
        self.block = copy.deepcopy(coordinator.blocks[block_index])
        self.final_norm = copy.deepcopy(coordinator.final_norm)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.block.parameters():
            parameter.requires_grad_(True)

    def forward(self, token_ids: Tensor) -> Tensor:
        _, tokens = token_ids.shape
        if tokens > self.config.context_length:
            raise ValueError("input exceeds configured context length")
        positions = torch.arange(tokens, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.block(hidden)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)


class RollingBlockWorkerSession(RollingBlockWorker):
    """Persistent shallow worker that reuses shared state across block assignments."""

    def __init__(self, coordinator: VolunteerDecoder, block_index: int) -> None:
        super().__init__(coordinator, block_index)
        self._coordinator_shared_state_sha256 = _shared_state_sha256(coordinator)

    def select_block(
        self,
        coordinator: VolunteerDecoder,
        block_index: int,
    ) -> int:
        if coordinator.config != self.config:
            raise ValueError("coordinator model configuration changed during worker session")
        if _shared_state_sha256(coordinator) != self._coordinator_shared_state_sha256:
            raise ValueError("shared coordinator state changed during worker session")
        if block_index < 0 or block_index >= len(coordinator.blocks):
            raise ValueError("block index is outside the coordinator model")
        block = copy.deepcopy(coordinator.blocks[block_index])
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        self.block = block
        self.block_index = block_index
        return _tensor_bytes(block)


def _tensor_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )


def _shared_state_sha256(coordinator: VolunteerDecoder) -> str:
    shared_tensors: dict[str, Tensor] = {}
    for module_name, module in (
        ("token_embedding", coordinator.token_embedding),
        ("position_embedding", coordinator.position_embedding),
        ("final_norm", coordinator.final_norm),
    ):
        for tensor_name, tensor in module.state_dict().items():
            shared_tensors[f"{module_name}.{tensor_name}"] = tensor
    return tensor_sha256(shared_tensors)


def _resident_tensor_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*module.parameters(), *module.buffers())
    )


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
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, ValueError):
        return None


def _fixed_fixture_mean_loss(
    model: nn.Module,
    campaign: CampaignConfig,
) -> float:
    if campaign.dataset is not None:
        raise ValueError("the first rolling-block experiment supports the T0 fixture only")
    if campaign.training.dataset_sequences % campaign.training.batch_size != 0:
        raise ValueError("fixture evaluation requires complete fixed-size batches")

    loss_sum = 0.0
    loss_weight_sum = 0
    model.eval()
    with torch.no_grad():
        for cursor in range(
            0,
            campaign.training.dataset_sequences,
            campaign.training.batch_size,
        ):
            inputs, targets = fixture_batch(campaign, cursor)
            logits = model(inputs)
            loss, batch_weight_sum = objective_loss_sum(
                campaign.objective,
                logits,
                targets,
            )
            loss_sum += float(loss)
            loss_weight_sum += batch_weight_sum
    return loss_sum / loss_weight_sum


def _train_full_step(
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
    campaign: CampaignConfig,
    cursor: int,
    dataset: PackedDataset | None = None,
) -> None:
    inputs, targets = fixture_batch(campaign, cursor, dataset)
    optimizer.zero_grad(set_to_none=True)
    loss, loss_weight_sum = objective_loss_sum(
        campaign.objective,
        model(inputs),
        targets,
    )
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(loss_weight_sum)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    optimizer.step()


def _map_worker_gradient(
    worker: RollingBlockWorker,
    coordinator: VolunteerDecoder,
) -> int:
    worker_parameters = dict(worker.block.named_parameters())
    coordinator_parameters = dict(
        coordinator.blocks[worker.block_index].named_parameters()
    )
    if worker_parameters.keys() != coordinator_parameters.keys():
        raise ValueError("worker block parameters do not map to the coordinator block")

    gradient_bytes = 0
    for name, parameter in worker_parameters.items():
        if parameter.grad is None:
            raise ValueError(f"worker block gradient is missing: {name}")
        gradient = parameter.grad.detach().clone()
        coordinator_parameters[name].grad = gradient
        gradient_bytes += gradient.numel() * gradient.element_size()
    return gradient_bytes


def _held_out_mean_loss(
    model: nn.Module,
    campaign: CampaignConfig,
    dataset: PackedDataset,
) -> float:
    if campaign.evaluation is None:
        raise ValueError("campaign does not define an evaluation profile")
    validation = evaluation_slice(campaign, "validation")
    sequence_count = validation.sequence_count
    batch_size = validation.batch_size
    loss_sum = 0.0
    loss_weight_sum = 0
    model.eval()
    with torch.no_grad():
        cursor = 0
        while cursor < sequence_count:
            current_batch_size = min(batch_size, sequence_count - cursor)
            inputs, targets = dataset.validation_batch(
                cursor=cursor,
                batch_size=current_batch_size,
                sequence_limit=sequence_count,
                start_sequence=validation.start_sequence,
            )
            loss, batch_weight_sum = objective_loss_sum(
                campaign.objective,
                model(inputs),
                targets,
            )
            loss_sum += float(loss)
            loss_weight_sum += batch_weight_sum
            cursor += current_batch_size
    return loss_sum / loss_weight_sum


def run_block_sharded_experiment(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    *,
    steps: int | None = None,
    evaluation_interval: int | None = None,
) -> BlockShardedEvidence:
    """Train every block shard from one snapshot before one global optimizer step."""
    if campaign.dataset is None:
        raise ValueError("block-sharded training requires dataset artifacts")
    validate_dataset_artifacts(campaign, dataset)
    if campaign.evaluation is None:
        raise ValueError("block-sharded training requires held-out evaluation")
    if campaign.model.layers <= 1:
        raise ValueError("block-sharded training requires at least two model blocks")
    steps = campaign.training.steps if steps is None else steps
    evaluation_interval = (
        campaign.model.layers
        if evaluation_interval is None
        else evaluation_interval
    )
    if steps <= 0:
        raise ValueError("steps must be positive")
    if evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")

    experiment_started = time.perf_counter()
    coordinator = build_model(campaign)
    baseline = build_model(campaign)
    coordinator_optimizer = _create_optimizer(coordinator, campaign.training)
    baseline_optimizer = _create_optimizer(baseline, campaign.training)
    sessions = [
        RollingBlockWorkerSession(coordinator, block_index)
        for block_index in range(campaign.model.layers)
    ]

    full_payload_bytes = _tensor_bytes(coordinator)
    full_resident_bytes = _resident_tensor_bytes(coordinator)
    worker_payload_bytes = _tensor_bytes(sessions[0])
    worker_resident_bytes = _resident_tensor_bytes(sessions[0])
    selected_block_payload_bytes = _tensor_bytes(sessions[0].block)
    shared_payload_bytes = worker_payload_bytes - selected_block_payload_bytes
    for session in sessions[1:]:
        if _tensor_bytes(session) != worker_payload_bytes:
            raise ValueError("block worker payload changed across equal-size blocks")
        if _resident_tensor_bytes(session) != worker_resident_bytes:
            raise ValueError("block worker residency changed across equal-size blocks")
        if _tensor_bytes(session.block) != selected_block_payload_bytes:
            raise ValueError("selected block payload changed across equal-size blocks")

    initial_shared_state_sha256 = _shared_state_sha256(coordinator)
    initial_block_sha256 = tuple(
        tensor_sha256(block.state_dict()) for block in coordinator.blocks
    )
    mapped_gradient_bytes: int | None = None
    assignment_block_sequence: list[int] = []
    block_update_counts = [0] * campaign.model.layers
    evaluation_history: list[BlockShardedEvaluationPoint] = []
    sharded_training_seconds = 0.0
    full_baseline_training_seconds = 0.0
    evaluation_seconds = 0.0

    evaluation_started = time.perf_counter()
    initial_mean_loss = _held_out_mean_loss(coordinator, campaign, dataset)
    baseline_initial_mean_loss = _held_out_mean_loss(baseline, campaign, dataset)
    evaluation_seconds += time.perf_counter() - evaluation_started
    if initial_mean_loss != baseline_initial_mean_loss:
        raise AssertionError("sharded and baseline models must share initial evaluation")
    evaluation_history.append(
        BlockShardedEvaluationPoint(
            step=0,
            sharded_mean_loss=initial_mean_loss,
            full_baseline_mean_loss=baseline_initial_mean_loss,
        )
    )

    block_state_loads = campaign.model.layers
    for step in range(steps):
        sharded_started = time.perf_counter()
        cursor = (
            step * campaign.training.batch_size
        ) % campaign.training.dataset_sequences
        inputs, targets = fixture_batch(campaign, cursor, dataset)

        if step > 0:
            for block_index, session in enumerate(sessions):
                current_block_payload_bytes = session.select_block(
                    coordinator,
                    block_index,
                )
                if current_block_payload_bytes != selected_block_payload_bytes:
                    raise ValueError(
                        "selected block payload changed across equal-size blocks"
                    )
                block_state_loads += 1

        coordinator_optimizer.zero_grad(set_to_none=True)
        for block_index, session in enumerate(sessions):
            assignment_block_sequence.append(block_index)
            session.zero_grad(set_to_none=True)
            session.train()
            loss, loss_weight_sum = objective_loss_sum(
                campaign.objective,
                session(inputs),
                targets,
            )
            loss.backward()
            for parameter in session.block.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(loss_weight_sum)
            current_gradient_bytes = _map_worker_gradient(session, coordinator)
            if mapped_gradient_bytes is None:
                mapped_gradient_bytes = current_gradient_bytes
            elif mapped_gradient_bytes != current_gradient_bytes:
                raise ValueError("mapped gradient size changed across equal-size blocks")
            block_update_counts[block_index] += 1

        torch.nn.utils.clip_grad_norm_(
            coordinator.blocks.parameters(),
            campaign.training.max_gradient_norm,
        )
        coordinator_optimizer.step()
        sharded_training_seconds += time.perf_counter() - sharded_started

        baseline_started = time.perf_counter()
        _train_full_step(
            baseline,
            baseline_optimizer,
            campaign,
            cursor,
            dataset,
        )
        full_baseline_training_seconds += time.perf_counter() - baseline_started

        completed_step = step + 1
        if (
            completed_step % evaluation_interval == 0
            or completed_step == steps
        ):
            evaluation_started = time.perf_counter()
            evaluation_history.append(
                BlockShardedEvaluationPoint(
                    step=completed_step,
                    sharded_mean_loss=_held_out_mean_loss(
                        coordinator,
                        campaign,
                        dataset,
                    ),
                    full_baseline_mean_loss=_held_out_mean_loss(
                        baseline,
                        campaign,
                        dataset,
                    ),
                )
            )
            evaluation_seconds += time.perf_counter() - evaluation_started

    if mapped_gradient_bytes is None:
        raise AssertionError("positive steps must produce worker evidence")
    final_shared_state_sha256 = _shared_state_sha256(coordinator)
    if final_shared_state_sha256 != initial_shared_state_sha256:
        raise AssertionError("block-sharded training changed shared model state")
    final_block_sha256 = tuple(
        tensor_sha256(block.state_dict()) for block in coordinator.blocks
    )
    updated_block_count = sum(
        initial != final
        for initial, final in zip(
            initial_block_sha256,
            final_block_sha256,
            strict=True,
        )
    )
    block_optimizer_steps: list[int] = []
    for block in coordinator.blocks:
        parameter_steps: set[int] = set()
        for parameter in block.parameters():
            state = coordinator_optimizer.state.get(parameter)
            if state is None or "step" not in state:
                raise AssertionError("updated block parameter lacks AdamW step state")
            raw_step = state["step"]
            step_value = (
                int(raw_step.item())
                if isinstance(raw_step, Tensor)
                else int(raw_step)
            )
            parameter_steps.add(step_value)
        if len(parameter_steps) != 1:
            raise AssertionError("block parameters have inconsistent AdamW steps")
        block_optimizer_steps.append(parameter_steps.pop())
    if tuple(block_optimizer_steps) != tuple(block_update_counts):
        raise AssertionError("AdamW block steps do not match mapped update counts")
    shared_optimizer_state_parameter_count = sum(
        parameter in coordinator_optimizer.state
        for module in (
            coordinator.token_embedding,
            coordinator.position_embedding,
            coordinator.final_norm,
        )
        for parameter in module.parameters()
    )
    if shared_optimizer_state_parameter_count != 0:
        raise AssertionError("frozen shared parameters acquired AdamW state")
    final_evaluation = evaluation_history[-1]
    worker_parameter_count = sum(
        parameter.numel() for parameter in sessions[0].parameters()
    )
    worker_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in sessions[0].parameters()
        if parameter.requires_grad
    )
    workers_per_step = campaign.model.layers
    worker_assignments = steps * workers_per_step
    cold_aggregate_download = worker_payload_bytes * workers_per_step
    warm_aggregate_download = selected_block_payload_bytes * workers_per_step
    persistent_aggregate_download = (
        cold_aggregate_download + warm_aggregate_download * (steps - 1)
    )
    mapped_gradient_upload = mapped_gradient_bytes * worker_assignments
    replicated_full_download = full_payload_bytes * steps
    combined_process_peak_rss_bytes = _peak_process_rss_bytes()
    experiment_wall_seconds = time.perf_counter() - experiment_started
    return BlockShardedEvidence(
        format="orcacolony_block_sharded_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        dataset_revision=dataset.revision,
        global_steps=steps,
        evaluation_interval=evaluation_interval,
        evaluation_sequences=evaluation_slice(
            campaign,
            "validation",
        ).sequence_count,
        workers_per_global_step=workers_per_step,
        worker_assignments=worker_assignments,
        assignment_block_sequence=tuple(assignment_block_sequence),
        block_update_counts=tuple(block_update_counts),
        coordinator_optimizer_steps=steps,
        block_optimizer_steps=tuple(block_optimizer_steps),
        shared_optimizer_state_parameter_count=(
            shared_optimizer_state_parameter_count
        ),
        full_parameter_count=sum(
            parameter.numel() for parameter in coordinator.parameters()
        ),
        worker_parameter_count=worker_parameter_count,
        worker_trainable_parameter_count=worker_trainable_parameter_count,
        full_payload_tensor_bytes=full_payload_bytes,
        worker_payload_tensor_bytes=worker_payload_bytes,
        full_resident_tensor_bytes=full_resident_bytes,
        worker_resident_tensor_bytes=worker_resident_bytes,
        aggregate_worker_resident_tensor_bytes=(
            worker_resident_bytes * workers_per_step
        ),
        selected_block_payload_tensor_bytes=selected_block_payload_bytes,
        shared_payload_tensor_bytes=shared_payload_bytes,
        mapped_gradient_bytes_per_assignment=mapped_gradient_bytes,
        shared_state_loads=workers_per_step,
        block_state_loads=block_state_loads,
        cold_aggregate_download_tensor_bytes=cold_aggregate_download,
        warm_aggregate_download_per_step_tensor_bytes=warm_aggregate_download,
        persistent_aggregate_download_tensor_bytes=persistent_aggregate_download,
        individual_worker_persistent_download_tensor_bytes=(
            worker_payload_bytes + selected_block_payload_bytes * (steps - 1)
        ),
        individual_worker_unique_payload_tensor_bytes=worker_payload_bytes,
        colony_unique_payload_tensor_bytes=cold_aggregate_download,
        mapped_gradient_upload_tensor_bytes=mapped_gradient_upload,
        persistent_aggregate_round_trip_tensor_bytes=(
            persistent_aggregate_download + mapped_gradient_upload
        ),
        replicated_full_download_tensor_bytes=replicated_full_download,
        replicated_full_round_trip_tensor_bytes=(
            2 * replicated_full_download
        ),
        initial_shared_state_sha256=initial_shared_state_sha256,
        final_shared_state_sha256=final_shared_state_sha256,
        updated_block_count=updated_block_count,
        evaluation_history=tuple(evaluation_history),
        initial_validation_mean_loss=initial_mean_loss,
        sharded_final_validation_mean_loss=final_evaluation.sharded_mean_loss,
        full_baseline_final_validation_mean_loss=(
            final_evaluation.full_baseline_mean_loss
        ),
        sharded_validation_loss_improvement=(
            initial_mean_loss - final_evaluation.sharded_mean_loss
        ),
        full_baseline_validation_loss_improvement=(
            initial_mean_loss - final_evaluation.full_baseline_mean_loss
        ),
        experiment_wall_seconds=experiment_wall_seconds,
        sharded_training_seconds=sharded_training_seconds,
        full_baseline_training_seconds=full_baseline_training_seconds,
        evaluation_seconds=evaluation_seconds,
        combined_process_peak_rss_bytes=combined_process_peak_rss_bytes,
        sharded_model_sha256=tensor_sha256(coordinator.state_dict()),
        full_baseline_model_sha256=tensor_sha256(baseline.state_dict()),
    )


def run_dataset_rolling_block_experiment(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    *,
    steps: int | None = None,
    evaluation_interval: int | None = None,
) -> DatasetRollingBlockEvidence:
    if campaign.dataset is None:
        raise ValueError("dataset rolling-block training requires dataset artifacts")
    validate_dataset_artifacts(campaign, dataset)
    if campaign.evaluation is None:
        raise ValueError("dataset rolling-block training requires held-out evaluation")
    if campaign.model.layers <= 1:
        raise ValueError("rolling-block training requires at least two model blocks")
    steps = campaign.training.steps if steps is None else steps
    evaluation_interval = (
        campaign.model.layers
        if evaluation_interval is None
        else evaluation_interval
    )
    if steps <= 0:
        raise ValueError("steps must be positive")
    if evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")

    experiment_started = time.perf_counter()
    coordinator = build_model(campaign)
    baseline = build_model(campaign)
    coordinator_optimizer = _create_optimizer(coordinator, campaign.training)
    baseline_optimizer = _create_optimizer(baseline, campaign.training)
    session = RollingBlockWorkerSession(coordinator, 0)

    full_payload_bytes = _tensor_bytes(coordinator)
    full_resident_bytes = _resident_tensor_bytes(coordinator)
    worker_payload_bytes = _tensor_bytes(session)
    worker_resident_bytes = _resident_tensor_bytes(session)
    selected_block_payload_bytes = _tensor_bytes(session.block)
    shared_payload_bytes = worker_payload_bytes - selected_block_payload_bytes
    mapped_gradient_bytes: int | None = None
    block_sequence: list[int] = []
    evaluation_history: list[RollingBlockEvaluationPoint] = []
    rolling_training_seconds = 0.0
    full_baseline_training_seconds = 0.0
    evaluation_seconds = 0.0

    evaluation_started = time.perf_counter()
    initial_mean_loss = _held_out_mean_loss(coordinator, campaign, dataset)
    baseline_initial_mean_loss = _held_out_mean_loss(baseline, campaign, dataset)
    evaluation_seconds += time.perf_counter() - evaluation_started
    if initial_mean_loss != baseline_initial_mean_loss:
        raise AssertionError("rolling and baseline models must share initial evaluation")
    evaluation_history.append(
        RollingBlockEvaluationPoint(
            step=0,
            rolling_mean_loss=initial_mean_loss,
            full_baseline_mean_loss=baseline_initial_mean_loss,
        )
    )

    block_state_loads = 0
    for step in range(steps):
        rolling_started = time.perf_counter()
        cursor = (
            step * campaign.training.batch_size
        ) % campaign.training.dataset_sequences
        block_index = step % campaign.model.layers
        block_sequence.append(block_index)
        if step == 0:
            current_block_payload_bytes = selected_block_payload_bytes
        else:
            current_block_payload_bytes = session.select_block(
                coordinator,
                block_index,
            )
        block_state_loads += 1
        if current_block_payload_bytes != selected_block_payload_bytes:
            raise ValueError("selected block payload changed across equal-size blocks")
        if _tensor_bytes(session) != worker_payload_bytes:
            raise ValueError("rolling worker payload changed across equal-size blocks")
        if _resident_tensor_bytes(session) != worker_resident_bytes:
            raise ValueError("rolling worker residency changed across equal-size blocks")

        inputs, targets = fixture_batch(campaign, cursor, dataset)
        session.train()
        loss, loss_weight_sum = objective_loss_sum(
            campaign.objective,
            session(inputs),
            targets,
        )
        loss.backward()
        for parameter in session.block.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_weight_sum)
        torch.nn.utils.clip_grad_norm_(
            session.block.parameters(),
            campaign.training.max_gradient_norm,
        )

        coordinator_optimizer.zero_grad(set_to_none=True)
        current_gradient_bytes = _map_worker_gradient(session, coordinator)
        if mapped_gradient_bytes is None:
            mapped_gradient_bytes = current_gradient_bytes
        elif mapped_gradient_bytes != current_gradient_bytes:
            raise ValueError("mapped gradient size changed across equal-size blocks")
        coordinator_optimizer.step()
        rolling_training_seconds += time.perf_counter() - rolling_started

        baseline_started = time.perf_counter()
        _train_full_step(
            baseline,
            baseline_optimizer,
            campaign,
            cursor,
            dataset,
        )
        full_baseline_training_seconds += time.perf_counter() - baseline_started

        completed_step = step + 1
        if (
            completed_step % evaluation_interval == 0
            or completed_step == steps
        ):
            evaluation_started = time.perf_counter()
            evaluation_history.append(
                RollingBlockEvaluationPoint(
                    step=completed_step,
                    rolling_mean_loss=_held_out_mean_loss(
                        coordinator,
                        campaign,
                        dataset,
                    ),
                    full_baseline_mean_loss=_held_out_mean_loss(
                        baseline,
                        campaign,
                        dataset,
                    ),
                )
            )
            evaluation_seconds += time.perf_counter() - evaluation_started

    if mapped_gradient_bytes is None:
        raise AssertionError("positive steps must produce worker evidence")
    final_evaluation = evaluation_history[-1]
    worker_parameter_count = sum(
        parameter.numel() for parameter in session.parameters()
    )
    worker_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in session.parameters()
        if parameter.requires_grad
    )
    rolling_model_sha256 = tensor_sha256(coordinator.state_dict())
    full_baseline_model_sha256 = tensor_sha256(baseline.state_dict())
    combined_process_peak_rss_bytes = _peak_process_rss_bytes()
    experiment_wall_seconds = time.perf_counter() - experiment_started
    return DatasetRollingBlockEvidence(
        format="orcacolony_dataset_rolling_block_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        dataset_revision=dataset.revision,
        steps=steps,
        evaluation_interval=evaluation_interval,
        evaluation_sequences=evaluation_slice(
            campaign,
            "validation",
        ).sequence_count,
        block_sequence=tuple(block_sequence),
        full_parameter_count=sum(
            parameter.numel() for parameter in coordinator.parameters()
        ),
        worker_parameter_count=worker_parameter_count,
        worker_trainable_parameter_count=worker_trainable_parameter_count,
        full_payload_tensor_bytes=full_payload_bytes,
        worker_payload_tensor_bytes=worker_payload_bytes,
        full_resident_tensor_bytes=full_resident_bytes,
        worker_resident_tensor_bytes=worker_resident_bytes,
        selected_block_payload_tensor_bytes=selected_block_payload_bytes,
        shared_payload_tensor_bytes=shared_payload_bytes,
        mapped_gradient_bytes_per_assignment=mapped_gradient_bytes,
        shared_state_loads=1,
        block_state_loads=block_state_loads,
        transient_worker_payload_tensor_bytes=worker_payload_bytes * steps,
        persistent_session_payload_tensor_bytes=(
            shared_payload_bytes + selected_block_payload_bytes * steps
        ),
        unique_coverage_payload_tensor_bytes=(
            shared_payload_bytes
            + selected_block_payload_bytes * len(set(block_sequence))
        ),
        replicated_full_payload_tensor_bytes=full_payload_bytes * steps,
        evaluation_history=tuple(evaluation_history),
        initial_validation_mean_loss=initial_mean_loss,
        rolling_final_validation_mean_loss=final_evaluation.rolling_mean_loss,
        full_baseline_final_validation_mean_loss=(
            final_evaluation.full_baseline_mean_loss
        ),
        rolling_validation_loss_improvement=(
            initial_mean_loss - final_evaluation.rolling_mean_loss
        ),
        full_baseline_validation_loss_improvement=(
            initial_mean_loss - final_evaluation.full_baseline_mean_loss
        ),
        experiment_wall_seconds=experiment_wall_seconds,
        rolling_training_seconds=rolling_training_seconds,
        full_baseline_training_seconds=full_baseline_training_seconds,
        evaluation_seconds=evaluation_seconds,
        combined_process_peak_rss_bytes=combined_process_peak_rss_bytes,
        rolling_model_sha256=rolling_model_sha256,
        full_baseline_model_sha256=full_baseline_model_sha256,
    )


def run_rolling_block_experiment(
    campaign: CampaignConfig,
    *,
    steps: int | None = None,
) -> RollingBlockEvidence:
    if campaign.dataset is not None:
        raise ValueError("the first rolling-block experiment supports the T0 fixture only")
    if campaign.model.layers <= 1:
        raise ValueError("rolling-block training requires at least two model blocks")
    steps = campaign.model.layers if steps is None else steps
    if steps <= 0:
        raise ValueError("steps must be positive")

    coordinator = build_model(campaign)
    baseline = build_model(campaign)
    coordinator_optimizer = _create_optimizer(coordinator, campaign.training)
    baseline_optimizer = _create_optimizer(baseline, campaign.training)
    initial_mean_loss = _fixed_fixture_mean_loss(coordinator, campaign)

    full_payload_bytes = _tensor_bytes(coordinator)
    full_resident_bytes = _resident_tensor_bytes(coordinator)
    worker_payload_bytes: int | None = None
    worker_resident_bytes: int | None = None
    selected_block_payload_bytes: int | None = None
    mapped_gradient_bytes: int | None = None
    block_sequence: list[int] = []

    coordinator.train()
    baseline.train()
    for step in range(steps):
        cursor = (
            step * campaign.training.batch_size
        ) % campaign.training.dataset_sequences
        block_index = step % campaign.model.layers
        block_sequence.append(block_index)
        worker = RollingBlockWorker(coordinator, block_index)
        current_payload_bytes = _tensor_bytes(worker)
        current_resident_bytes = _resident_tensor_bytes(worker)
        current_block_payload_bytes = _tensor_bytes(worker.block)
        if worker_payload_bytes is None:
            worker_payload_bytes = current_payload_bytes
            worker_resident_bytes = current_resident_bytes
            selected_block_payload_bytes = current_block_payload_bytes
        elif worker_payload_bytes != current_payload_bytes:
            raise ValueError("rolling worker payload changed across equal-size blocks")
        elif worker_resident_bytes != current_resident_bytes:
            raise ValueError("rolling worker residency changed across equal-size blocks")
        elif selected_block_payload_bytes != current_block_payload_bytes:
            raise ValueError("selected block payload changed across equal-size blocks")

        inputs, targets = fixture_batch(campaign, cursor)
        worker.train()
        loss, loss_weight_sum = objective_loss_sum(
            campaign.objective,
            worker(inputs),
            targets,
        )
        loss.backward()
        for parameter in worker.block.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_weight_sum)
        torch.nn.utils.clip_grad_norm_(
            worker.block.parameters(),
            campaign.training.max_gradient_norm,
        )

        coordinator_optimizer.zero_grad(set_to_none=True)
        current_gradient_bytes = _map_worker_gradient(worker, coordinator)
        if mapped_gradient_bytes is None:
            mapped_gradient_bytes = current_gradient_bytes
        elif mapped_gradient_bytes != current_gradient_bytes:
            raise ValueError("mapped gradient size changed across equal-size blocks")
        coordinator_optimizer.step()

        _train_full_step(
            baseline,
            baseline_optimizer,
            campaign,
            cursor,
        )

    if (
        worker_payload_bytes is None
        or worker_resident_bytes is None
        or selected_block_payload_bytes is None
        or mapped_gradient_bytes is None
    ):
        raise AssertionError("positive steps must produce worker evidence")
    rolling_final_mean_loss = _fixed_fixture_mean_loss(coordinator, campaign)
    baseline_final_mean_loss = _fixed_fixture_mean_loss(baseline, campaign)
    worker_parameter_count = sum(
        parameter.numel() for parameter in RollingBlockWorker(coordinator, 0).parameters()
    )
    worker_trainable_parameter_count = sum(
        parameter.numel()
        for parameter in RollingBlockWorker(coordinator, 0).parameters()
        if parameter.requires_grad
    )
    return RollingBlockEvidence(
        format="orcacolony_rolling_block_evidence_v1",
        steps=steps,
        block_sequence=tuple(block_sequence),
        full_parameter_count=sum(
            parameter.numel() for parameter in coordinator.parameters()
        ),
        worker_parameter_count=worker_parameter_count,
        worker_trainable_parameter_count=worker_trainable_parameter_count,
        full_payload_tensor_bytes=full_payload_bytes,
        worker_payload_tensor_bytes=worker_payload_bytes,
        full_resident_tensor_bytes=full_resident_bytes,
        worker_resident_tensor_bytes=worker_resident_bytes,
        selected_block_payload_tensor_bytes=selected_block_payload_bytes,
        shared_payload_tensor_bytes=(
            worker_payload_bytes - selected_block_payload_bytes
        ),
        mapped_gradient_bytes_per_assignment=mapped_gradient_bytes,
        transient_worker_payload_tensor_bytes=worker_payload_bytes * steps,
        unique_coverage_payload_tensor_bytes=(
            worker_payload_bytes
            - selected_block_payload_bytes
            + selected_block_payload_bytes * len(set(block_sequence))
        ),
        replicated_full_payload_tensor_bytes=full_payload_bytes * steps,
        initial_mean_loss=initial_mean_loss,
        rolling_final_mean_loss=rolling_final_mean_loss,
        full_baseline_final_mean_loss=baseline_final_mean_loss,
        rolling_loss_improvement=initial_mean_loss - rolling_final_mean_loss,
        full_baseline_loss_improvement=initial_mean_loss - baseline_final_mean_loss,
        rolling_model_sha256=tensor_sha256(coordinator.state_dict()),
        full_baseline_model_sha256=tensor_sha256(baseline.state_dict()),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an OrcaColony rolling-block feasibility experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="authenticated dataset artifact directory required by data-backed configs",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--topology",
        choices=("sequential", "block-sharded"),
        default="sequential",
        help="partial-model assignment topology for data-backed experiments",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        help="held-out evaluation cadence for data-backed experiments",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config)
    if campaign.dataset is None:
        if args.topology != "sequential":
            parser.error("--topology block-sharded requires a data-backed config")
        if args.dataset is not None:
            parser.error("--dataset is only valid for data-backed campaign configs")
        if args.evaluation_interval is not None:
            parser.error(
                "--evaluation-interval is only valid for data-backed campaign configs"
            )
        evidence = run_rolling_block_experiment(
            campaign,
            steps=args.steps,
        )
    else:
        if args.dataset is None:
            parser.error("data-backed campaign configs require --dataset")
        dataset = PackedDataset.load(args.dataset)
        if args.topology == "block-sharded":
            evidence = run_block_sharded_experiment(
                campaign,
                dataset,
                steps=args.steps,
                evaluation_interval=args.evaluation_interval,
            )
        else:
            evidence = run_dataset_rolling_block_experiment(
                campaign,
                dataset,
                steps=args.steps,
                evaluation_interval=args.evaluation_interval,
            )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
