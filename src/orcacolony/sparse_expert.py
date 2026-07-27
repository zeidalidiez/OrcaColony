from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from orcacolony.reference import (
    CampaignConfig,
    DecoderBlock,
    ModelConfig,
    ObjectiveConfig,
    _create_optimizer,
    configure_determinism,
    fixture_batch,
    load_campaign,
    objective_loss_sum,
    tensor_sha256,
)
from orcacolony.tiled_model import (
    _gradient_snapshot,
    _max_abs_difference,
    _model_snapshot,
    _optimizer_tensor_snapshot,
)
from orcacolony.tile_process import _process_memory_bytes


class SparseExpert(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.up = nn.Linear(config.width, 4 * config.width)
        self.down = nn.Linear(4 * config.width, config.width)

    def forward(self, hidden: Tensor) -> Tensor:
        return hidden + self.down(F.gelu(self.up(self.norm(hidden))))


class SparseExpertDecoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        objective: ObjectiveConfig,
        expert_count: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.objective = objective
        self.expert_count = expert_count
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.shared_block = DecoderBlock(config)
        self.router = nn.Linear(config.width, expert_count, bias=False)
        self.experts = nn.ModuleList(
            SparseExpert(config) for _ in range(expert_count)
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.output_head = nn.Linear(
            config.width,
            config.vocabulary_size,
            bias=False,
        )

    def shared_hidden(self, token_ids: Tensor) -> Tensor:
        _, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[
            None, :, :
        ]
        return self.shared_block(hidden)

    def logits_for_hidden(self, hidden: Tensor) -> Tensor:
        return self.output_head(self.final_norm(hidden))


class SparseExpertWorker(nn.Module):
    def __init__(self, model: SparseExpertDecoder, expert_index: int) -> None:
        super().__init__()
        self.expert_index = expert_index
        self.objective = model.objective
        self.expert = copy.deepcopy(model.experts[expert_index])
        self.final_norm = copy.deepcopy(model.final_norm)
        self.output_head = copy.deepcopy(model.output_head)
        self.forward_calls = 0

    def forward(self, hidden: Tensor) -> Tensor:
        self.forward_calls += 1
        output = self.expert(hidden)
        return self.output_head(self.final_norm(output))


@dataclass(frozen=True)
class SparseExpertEvidence:
    format: str
    campaign_id: str
    expert_count: int
    active_expert_count: int
    routing_capacity: int
    routing_counts: tuple[int, ...]
    unconstrained_routing_counts: tuple[int, ...]
    capacity_rerouted_tokens: int
    total_routed_tokens: int
    routing_load_coefficient_of_variation: float
    worker_forward_calls: tuple[int, ...]
    router_optimizer_step: int
    shared_optimizer_step: int
    expert_optimizer_steps: tuple[int, ...]
    router_gradient_tensor_bytes: int
    full_parameter_count: int
    shared_parameter_count: int
    router_parameter_count: int
    expert_parameter_count: int
    worker_parameter_count: int
    full_payload_tensor_bytes: int
    full_input_tensor_bytes: int
    full_replica_round_trip_tensor_bytes: int
    shared_worker_cache_payload_tensor_bytes: int
    expert_payload_tensor_bytes: int
    worker_payload_tensor_bytes: int
    warm_worker_payload_tensor_bytes: int
    cold_aggregate_payload_tensor_bytes: int
    warm_aggregate_payload_tensor_bytes: int
    worker_gradient_upload_tensor_bytes: int
    aggregate_gradient_upload_tensor_bytes: int
    aggregate_input_tensor_bytes: int
    aggregate_input_adjoint_tensor_bytes: int
    cold_aggregate_round_trip_tensor_bytes: int
    warm_aggregate_round_trip_tensor_bytes: int
    centralized_loss: float
    distributed_loss: float
    max_abs_raw_gradient_difference: float
    max_abs_clipped_gradient_difference: float
    max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    distributed_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    distributed_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    distributed_optimizer_sha256: str
    centralized_model_sha256: str
    distributed_model_sha256: str
    centralized_step_seconds: float
    distributed_step_seconds: float
    combined_process_peak_rss_bytes: int | None


def _build_sparse_model(
    campaign: CampaignConfig,
    expert_count: int,
) -> SparseExpertDecoder:
    configure_determinism(campaign.training.seed)
    return SparseExpertDecoder(campaign.model, campaign.objective, expert_count)


def _tensor_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _state_tensor_bytes(module: nn.Module) -> int:
    return sum(_tensor_bytes(tensor) for tensor in module.state_dict().values())


def _gradient_tensor_bytes(module: nn.Module) -> int:
    total = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            raise AssertionError("expected worker gradient is absent")
        total += _tensor_bytes(parameter.grad)
    return total


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _balanced_top1_routes(
    logits: Tensor,
    expert_count: int,
) -> tuple[Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    flat = logits.detach().reshape(-1, expert_count)
    token_count = flat.shape[0]
    capacity = math.ceil(token_count / expert_count)
    unconstrained = flat.argmax(dim=-1)
    unconstrained_counts = tuple(
        int((unconstrained == index).sum()) for index in range(expert_count)
    )
    remaining = [capacity] * expert_count
    routes: list[int] = []
    rerouted = 0
    for token_index in range(token_count):
        ranked = torch.argsort(
            flat[token_index],
            descending=True,
            stable=True,
        ).tolist()
        selected = next(index for index in ranked if remaining[index] > 0)
        if selected != int(unconstrained[token_index]):
            rerouted += 1
        routes.append(selected)
        remaining[selected] -= 1
    route_tensor = torch.tensor(routes, dtype=torch.int64, device=logits.device)
    counts = tuple(int((route_tensor == index).sum()) for index in range(expert_count))
    if sum(counts) != token_count or any(count <= 0 for count in counts):
        raise AssertionError("capacity router failed to cover every token and expert")
    return route_tensor, counts, unconstrained_counts, rerouted, capacity


def _router_auxiliary_loss(
    logits: Tensor,
    routes: Tensor,
    expert_count: int,
    weight: float,
) -> Tensor:
    probabilities = torch.softmax(logits.reshape(-1, expert_count), dim=-1)
    probability_fraction = probabilities.mean(dim=0)
    route_fraction = F.one_hot(routes, num_classes=expert_count).float().mean(dim=0)
    return weight * expert_count * torch.sum(
        probability_fraction * route_fraction.detach()
    )


def _expert_loss(
    model: SparseExpertDecoder,
    hidden_flat: Tensor,
    targets_flat: Tensor,
    routes: Tensor,
    expert_index: int,
    total_tokens: int,
) -> Tensor:
    mask = routes == expert_index
    selected = hidden_flat[mask]
    logits = model.logits_for_hidden(model.experts[expert_index](selected))
    loss_sum, _ = objective_loss_sum(
        model.objective,
        logits,
        targets_flat[mask],
    )
    return loss_sum / total_tokens


def _optimizer_step_for_module(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
    *,
    label: str,
) -> int:
    steps: set[int] = set()
    for parameter in module.parameters():
        state = optimizer.state.get(parameter)
        if state is None or "step" not in state:
            raise AssertionError(f"{label} optimizer state is absent")
        value = state["step"]
        scalar = float(value.item()) if isinstance(value, Tensor) else float(value)
        if not math.isfinite(scalar) or scalar != int(scalar):
            raise AssertionError(f"{label} optimizer step is invalid")
        steps.add(int(scalar))
    if len(steps) != 1:
        raise AssertionError(f"{label} optimizer steps disagree")
    return next(iter(steps))


def _map_worker_gradients(
    coordinator: SparseExpertDecoder,
    worker: SparseExpertWorker,
    expert_index: int,
) -> None:
    coordinator_expert = coordinator.experts[expert_index]
    for (expected_name, parameter), (actual_name, worker_parameter) in zip(
        coordinator_expert.named_parameters(),
        worker.expert.named_parameters(),
        strict=True,
    ):
        if expected_name != actual_name or worker_parameter.grad is None:
            raise AssertionError("expert gradient mapping is incomplete")
        parameter.grad = worker_parameter.grad.detach().clone()
    for coordinator_module, worker_module, label in (
        (coordinator.final_norm, worker.final_norm, "final norm"),
        (coordinator.output_head, worker.output_head, "output head"),
    ):
        for (expected_name, parameter), (actual_name, worker_parameter) in zip(
            coordinator_module.named_parameters(),
            worker_module.named_parameters(),
            strict=True,
        ):
            if expected_name != actual_name or worker_parameter.grad is None:
                raise AssertionError(f"{label} gradient mapping is incomplete")
            contribution = worker_parameter.grad.detach()
            if parameter.grad is None:
                parameter.grad = contribution.clone()
            else:
                parameter.grad.add_(contribution)


def run_sparse_expert_experiment(
    campaign: CampaignConfig,
    *,
    expert_count: int = 4,
    router_aux_weight: float = 0.01,
) -> SparseExpertEvidence:
    if campaign.dataset is not None:
        raise ValueError("the first sparse-expert tracer requires the synthetic fixture")
    if type(expert_count) is not int or not 2 <= expert_count <= 16:
        raise ValueError("expert count must be an integer between two and sixteen")
    if (
        isinstance(router_aux_weight, bool)
        or not isinstance(router_aux_weight, (int, float))
        or not math.isfinite(float(router_aux_weight))
        or not 0.0 < float(router_aux_weight) <= 1.0
    ):
        raise ValueError("router auxiliary weight must be finite in (0, 1]")
    router_aux_weight = float(router_aux_weight)

    centralized = _build_sparse_model(campaign, expert_count)
    distributed = _build_sparse_model(campaign, expert_count)
    centralized_optimizer = _create_optimizer(centralized, campaign.training)
    distributed_optimizer = _create_optimizer(distributed, campaign.training)
    inputs, targets = fixture_batch(campaign, 0)
    total_tokens = targets.numel()

    centralized.train()
    centralized_optimizer.zero_grad(set_to_none=True)
    centralized_started = time.perf_counter()
    centralized_hidden = centralized.shared_hidden(inputs)
    centralized_flat = centralized_hidden.reshape(-1, campaign.model.width)
    centralized_router_logits = centralized.router(centralized_flat)
    routes, routing_counts, unconstrained_counts, rerouted, capacity = (
        _balanced_top1_routes(centralized_router_logits, expert_count)
    )
    centralized_aux = _router_auxiliary_loss(
        centralized_router_logits,
        routes,
        expert_count,
        router_aux_weight,
    )
    centralized_losses = [
        _expert_loss(
            centralized,
            centralized_flat,
            targets.reshape(-1),
            routes,
            expert_index,
            total_tokens,
        )
        for expert_index in range(expert_count)
    ]
    centralized_aux.backward(retain_graph=True)
    for expert_index, loss in enumerate(centralized_losses):
        loss.backward(retain_graph=expert_index < expert_count - 1)
    centralized_loss = centralized_aux + sum(centralized_losses)
    centralized_raw = _gradient_snapshot(centralized)
    torch.nn.utils.clip_grad_norm_(
        centralized.parameters(),
        campaign.training.max_gradient_norm,
    )
    centralized_clipped = _gradient_snapshot(centralized)
    centralized_optimizer.step()
    centralized_step_seconds = time.perf_counter() - centralized_started

    distributed.train()
    distributed_optimizer.zero_grad(set_to_none=True)
    distributed_started = time.perf_counter()
    distributed_hidden = distributed.shared_hidden(inputs)
    distributed_flat = distributed_hidden.reshape(-1, campaign.model.width)
    distributed_router_logits = distributed.router(distributed_flat)
    distributed_routes, distributed_counts, _, distributed_rerouted, _ = (
        _balanced_top1_routes(distributed_router_logits, expert_count)
    )
    if (
        not torch.equal(routes, distributed_routes)
        or routing_counts != distributed_counts
        or rerouted != distributed_rerouted
    ):
        raise AssertionError("centralized and distributed routing disagree")
    distributed_aux = _router_auxiliary_loss(
        distributed_router_logits,
        distributed_routes,
        expert_count,
        router_aux_weight,
    )
    distributed_aux.backward(retain_graph=True)
    workers: list[SparseExpertWorker] = []
    distributed_losses: list[Tensor] = []
    aggregate_input_tensor_bytes = 0
    aggregate_input_adjoint_tensor_bytes = 0
    for expert_index in range(expert_count):
        mask = distributed_routes == expert_index
        selected_hidden = (
            distributed_flat[mask].detach().clone().requires_grad_(True)
        )
        selected_targets = targets.reshape(-1)[mask].detach().clone()
        worker = SparseExpertWorker(distributed, expert_index)
        worker.train()
        worker.zero_grad(set_to_none=True)
        logits = worker(selected_hidden)
        worker_loss_sum, _ = objective_loss_sum(
            worker.objective,
            logits,
            selected_targets,
        )
        worker_loss = worker_loss_sum / total_tokens
        worker_loss.backward()
        if selected_hidden.grad is None:
            raise AssertionError("expert worker lacks shared-trunk input adjoint")
        _map_worker_gradients(distributed, worker, expert_index)
        scattered = torch.zeros_like(distributed_flat)
        scattered[mask] = selected_hidden.grad.detach()
        distributed_flat.backward(
            scattered,
            retain_graph=expert_index < expert_count - 1,
        )
        aggregate_input_tensor_bytes += _tensor_bytes(selected_hidden)
        aggregate_input_tensor_bytes += _tensor_bytes(selected_targets)
        aggregate_input_adjoint_tensor_bytes += _tensor_bytes(selected_hidden.grad)
        distributed_losses.append(worker_loss.detach())
        workers.append(worker)
    distributed_loss_tensor = distributed_aux.detach() + sum(distributed_losses)
    distributed_raw = _gradient_snapshot(distributed)
    torch.nn.utils.clip_grad_norm_(
        distributed.parameters(),
        campaign.training.max_gradient_norm,
    )
    distributed_clipped = _gradient_snapshot(distributed)
    distributed_optimizer.step()
    distributed_step_seconds = time.perf_counter() - distributed_started

    centralized_model = _model_snapshot(centralized)
    distributed_model = _model_snapshot(distributed)
    centralized_optimizer_snapshot = _optimizer_tensor_snapshot(
        centralized,
        centralized_optimizer,
    )
    distributed_optimizer_snapshot = _optimizer_tensor_snapshot(
        distributed,
        distributed_optimizer,
    )
    mean_count = total_tokens / expert_count
    routing_cv = math.sqrt(
        sum((count - mean_count) ** 2 for count in routing_counts) / expert_count
    ) / mean_count

    first_worker = workers[0]
    worker_payload = _state_tensor_bytes(first_worker)
    expert_payload = _state_tensor_bytes(first_worker.expert)
    worker_gradient_upload = _gradient_tensor_bytes(first_worker)
    active_expert_count = sum(count > 0 for count in routing_counts)
    shared_cache_payload = (
        _state_tensor_bytes(first_worker.final_norm)
        + _state_tensor_bytes(first_worker.output_head)
    )
    full_payload = _state_tensor_bytes(distributed)
    full_input_tensor_bytes = _tensor_bytes(inputs) + _tensor_bytes(targets)
    router_gradient_bytes = sum(
        _tensor_bytes(parameter.grad)
        for parameter in distributed.router.parameters()
        if parameter.grad is not None
    )
    shared_parameters = (
        _parameter_count(distributed.token_embedding)
        + _parameter_count(distributed.position_embedding)
        + _parameter_count(distributed.shared_block)
        + _parameter_count(distributed.router)
        + _parameter_count(distributed.final_norm)
        + _parameter_count(distributed.output_head)
    )
    router_step = _optimizer_step_for_module(
        distributed_optimizer,
        distributed.router,
        label="router",
    )
    shared_step = _optimizer_step_for_module(
        distributed_optimizer,
        distributed.shared_block,
        label="shared block",
    )
    expert_steps = tuple(
        _optimizer_step_for_module(
            distributed_optimizer,
            expert,
            label=f"expert {index}",
        )
        for index, expert in enumerate(distributed.experts)
    )

    return SparseExpertEvidence(
        format="orcacolony_sparse_expert_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        expert_count=expert_count,
        active_expert_count=active_expert_count,
        routing_capacity=capacity,
        routing_counts=routing_counts,
        unconstrained_routing_counts=unconstrained_counts,
        capacity_rerouted_tokens=rerouted,
        total_routed_tokens=total_tokens,
        routing_load_coefficient_of_variation=routing_cv,
        worker_forward_calls=tuple(worker.forward_calls for worker in workers),
        router_optimizer_step=router_step,
        shared_optimizer_step=shared_step,
        expert_optimizer_steps=expert_steps,
        router_gradient_tensor_bytes=router_gradient_bytes,
        full_parameter_count=_parameter_count(distributed),
        shared_parameter_count=shared_parameters,
        router_parameter_count=_parameter_count(distributed.router),
        expert_parameter_count=_parameter_count(distributed.experts[0]),
        worker_parameter_count=_parameter_count(first_worker),
        full_payload_tensor_bytes=full_payload,
        full_input_tensor_bytes=full_input_tensor_bytes,
        full_replica_round_trip_tensor_bytes=(
            2 * full_payload + full_input_tensor_bytes
        ),
        shared_worker_cache_payload_tensor_bytes=shared_cache_payload,
        expert_payload_tensor_bytes=expert_payload,
        worker_payload_tensor_bytes=worker_payload,
        warm_worker_payload_tensor_bytes=expert_payload,
        cold_aggregate_payload_tensor_bytes=worker_payload * active_expert_count,
        warm_aggregate_payload_tensor_bytes=expert_payload * active_expert_count,
        worker_gradient_upload_tensor_bytes=worker_gradient_upload,
        aggregate_gradient_upload_tensor_bytes=(
            worker_gradient_upload * active_expert_count
        ),
        aggregate_input_tensor_bytes=aggregate_input_tensor_bytes,
        aggregate_input_adjoint_tensor_bytes=aggregate_input_adjoint_tensor_bytes,
        cold_aggregate_round_trip_tensor_bytes=(
            worker_payload * active_expert_count
            + worker_gradient_upload * active_expert_count
            + aggregate_input_tensor_bytes
            + aggregate_input_adjoint_tensor_bytes
        ),
        warm_aggregate_round_trip_tensor_bytes=(
            expert_payload * active_expert_count
            + worker_gradient_upload * active_expert_count
            + aggregate_input_tensor_bytes
            + aggregate_input_adjoint_tensor_bytes
        ),
        centralized_loss=float(centralized_loss.detach()),
        distributed_loss=float(distributed_loss_tensor),
        max_abs_raw_gradient_difference=_max_abs_difference(
            centralized_raw,
            distributed_raw,
        ),
        max_abs_clipped_gradient_difference=_max_abs_difference(
            centralized_clipped,
            distributed_clipped,
        ),
        max_abs_model_difference=_max_abs_difference(
            centralized_model,
            distributed_model,
        ),
        centralized_raw_gradient_sha256=tensor_sha256(centralized_raw),
        distributed_raw_gradient_sha256=tensor_sha256(distributed_raw),
        centralized_clipped_gradient_sha256=tensor_sha256(centralized_clipped),
        distributed_clipped_gradient_sha256=tensor_sha256(distributed_clipped),
        centralized_optimizer_sha256=tensor_sha256(
            centralized_optimizer_snapshot
        ),
        distributed_optimizer_sha256=tensor_sha256(
            distributed_optimizer_snapshot
        ),
        centralized_model_sha256=tensor_sha256(centralized_model),
        distributed_model_sha256=tensor_sha256(distributed_model),
        centralized_step_seconds=centralized_step_seconds,
        distributed_step_seconds=distributed_step_seconds,
        combined_process_peak_rss_bytes=_process_memory_bytes()[1],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure one exact coordinator-routed sparse-expert step"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    campaign = load_campaign(args.config)
    evidence = run_sparse_expert_experiment(
        campaign,
        expert_count=args.expert_count,
        router_aux_weight=args.router_aux_weight,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
