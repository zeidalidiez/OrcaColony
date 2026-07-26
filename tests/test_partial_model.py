from __future__ import annotations

import math
from pathlib import Path

from orcacolony.partial_model import (
    RollingBlockWorker,
    run_rolling_block_experiment,
)
from orcacolony.reference import build_model, load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def test_rolling_block_worker_contains_one_trainable_mapped_block() -> None:
    campaign = load_campaign(CONFIG)
    coordinator = build_model(campaign)

    worker = RollingBlockWorker(coordinator, 2)

    assert worker.block_index == 2
    assert sum(parameter.numel() for parameter in worker.parameters()) < sum(
        parameter.numel() for parameter in coordinator.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in worker.token_embedding.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in worker.position_embedding.parameters()
    )
    assert all(not parameter.requires_grad for parameter in worker.final_norm.parameters())
    assert all(parameter.requires_grad for parameter in worker.block.parameters())
    assert set(dict(worker.block.named_parameters())) == set(
        dict(coordinator.blocks[2].named_parameters())
    )


def test_rolling_block_experiment_rotates_real_gradients_across_full_model() -> None:
    campaign = load_campaign(CONFIG)

    evidence = run_rolling_block_experiment(campaign, steps=campaign.model.layers)

    assert evidence.format == "orcacolony_rolling_block_evidence_v1"
    assert evidence.block_sequence == (0, 1, 2, 3)
    assert evidence.worker_parameter_count < evidence.full_parameter_count
    assert evidence.worker_trainable_parameter_count < evidence.worker_parameter_count
    assert evidence.worker_payload_tensor_bytes < evidence.full_payload_tensor_bytes
    assert evidence.worker_resident_tensor_bytes < evidence.full_resident_tensor_bytes
    assert evidence.shared_payload_tensor_bytes == (
        evidence.worker_payload_tensor_bytes
        - evidence.selected_block_payload_tensor_bytes
    )
    assert evidence.mapped_gradient_bytes_per_assignment > 0
    assert evidence.transient_worker_payload_tensor_bytes == (
        evidence.worker_payload_tensor_bytes * campaign.model.layers
    )
    assert evidence.unique_coverage_payload_tensor_bytes == (
        evidence.full_payload_tensor_bytes
    )
    assert evidence.replicated_full_payload_tensor_bytes == (
        evidence.full_payload_tensor_bytes * campaign.model.layers
    )
    assert math.isfinite(evidence.initial_mean_loss)
    assert math.isfinite(evidence.rolling_final_mean_loss)
    assert math.isfinite(evidence.full_baseline_final_mean_loss)
    assert evidence.rolling_model_sha256 != evidence.full_baseline_model_sha256
