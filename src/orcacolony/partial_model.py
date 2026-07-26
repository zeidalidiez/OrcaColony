from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from orcacolony.reference import (
    CampaignConfig,
    DecoderBlock,
    VolunteerDecoder,
    _create_optimizer,
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
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


class RollingBlockWorker(nn.Module):
    """Executable shallow worker containing one mapped block from a full model."""

    def __init__(self, coordinator: VolunteerDecoder, block_index: int) -> None:
        super().__init__()
        if block_index < 0 or block_index >= len(coordinator.blocks):
            raise ValueError("block index is outside the coordinator model")
        self.config = coordinator.config
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


def _tensor_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )


def _resident_tensor_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*module.parameters(), *module.buffers())
    )


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
            loss = F.cross_entropy(
                logits.reshape(-1, campaign.model.vocabulary_size),
                targets.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(loss)
            loss_weight_sum += targets.numel()
    return loss_sum / loss_weight_sum


def _train_full_step(
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
    campaign: CampaignConfig,
    cursor: int,
) -> None:
    inputs, targets = fixture_batch(campaign, cursor)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        model(inputs).reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="sum",
    )
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(targets.numel())
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
        loss = F.cross_entropy(
            worker(inputs).reshape(-1, campaign.model.vocabulary_size),
            targets.reshape(-1),
            reduction="sum",
        )
        loss.backward()
        for parameter in worker.block.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(targets.numel())
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
        description="Run the first OrcaColony rolling-block feasibility experiment"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evidence = run_rolling_block_experiment(
        load_campaign(args.config),
        steps=args.steps,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
