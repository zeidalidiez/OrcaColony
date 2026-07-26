from __future__ import annotations

import json
import math
from pathlib import Path

from orcacolony.reference import load_campaign
from orcacolony.sparse_expert import main, run_sparse_expert_experiment


CONFIG = Path(__file__).parent.parent / "campaign" / "t0-smoke.json"


def test_sparse_expert_workers_reproduce_centralized_sparse_step() -> None:
    campaign = load_campaign(CONFIG)
    evidence = run_sparse_expert_experiment(
        campaign,
        expert_count=4,
        router_aux_weight=0.01,
    )

    assert evidence.format == "orcacolony_sparse_expert_evidence_v1"
    assert evidence.expert_count == 4
    assert evidence.active_expert_count == 4
    assert evidence.total_routed_tokens == (
        campaign.training.batch_size * campaign.model.context_length
    )
    assert len(evidence.routing_counts) == 4
    assert all(count > 0 for count in evidence.routing_counts)
    assert sum(evidence.routing_counts) == evidence.total_routed_tokens
    assert math.isfinite(evidence.routing_load_coefficient_of_variation)
    assert evidence.routing_load_coefficient_of_variation >= 0.0

    assert evidence.worker_forward_calls == (1, 1, 1, 1)
    assert evidence.router_optimizer_step == 1
    assert evidence.shared_optimizer_step == 1
    assert evidence.expert_optimizer_steps == (1, 1, 1, 1)
    assert evidence.router_gradient_tensor_bytes > 0

    assert evidence.max_abs_raw_gradient_difference == 0.0
    assert evidence.max_abs_clipped_gradient_difference == 0.0
    assert evidence.max_abs_model_difference == 0.0
    assert evidence.centralized_loss == evidence.distributed_loss
    assert (
        evidence.centralized_raw_gradient_sha256
        == evidence.distributed_raw_gradient_sha256
    )
    assert (
        evidence.centralized_clipped_gradient_sha256
        == evidence.distributed_clipped_gradient_sha256
    )
    assert evidence.centralized_optimizer_sha256 == evidence.distributed_optimizer_sha256
    assert evidence.centralized_model_sha256 == evidence.distributed_model_sha256

    assert evidence.expert_parameter_count > 0
    assert evidence.worker_parameter_count > evidence.expert_parameter_count
    assert evidence.worker_parameter_count < evidence.full_parameter_count
    assert evidence.worker_payload_tensor_bytes < evidence.full_payload_tensor_bytes
    assert evidence.full_input_tensor_bytes > 0
    assert evidence.full_replica_round_trip_tensor_bytes == (
        2 * evidence.full_payload_tensor_bytes + evidence.full_input_tensor_bytes
    )
    assert evidence.warm_worker_payload_tensor_bytes == (
        evidence.expert_payload_tensor_bytes
    )
    assert evidence.cold_aggregate_payload_tensor_bytes == (
        evidence.worker_payload_tensor_bytes * evidence.active_expert_count
    )
    assert evidence.warm_aggregate_payload_tensor_bytes == (
        evidence.expert_payload_tensor_bytes * evidence.active_expert_count
    )
    assert evidence.worker_gradient_upload_tensor_bytes > 0
    assert evidence.aggregate_input_adjoint_tensor_bytes > 0
    assert evidence.aggregate_gradient_upload_tensor_bytes == (
        evidence.worker_gradient_upload_tensor_bytes * evidence.active_expert_count
    )
    assert evidence.cold_aggregate_round_trip_tensor_bytes == (
        evidence.cold_aggregate_payload_tensor_bytes
        + evidence.aggregate_gradient_upload_tensor_bytes
        + evidence.aggregate_input_tensor_bytes
        + evidence.aggregate_input_adjoint_tensor_bytes
    )
    assert evidence.warm_aggregate_round_trip_tensor_bytes == (
        evidence.warm_aggregate_payload_tensor_bytes
        + evidence.aggregate_gradient_upload_tensor_bytes
        + evidence.aggregate_input_tensor_bytes
        + evidence.aggregate_input_adjoint_tensor_bytes
    )
    assert evidence.centralized_step_seconds > 0
    assert evidence.distributed_step_seconds > 0
    assert (
        evidence.combined_process_peak_rss_bytes is None
        or evidence.combined_process_peak_rss_bytes > 0
    )


def test_sparse_expert_cli_writes_evidence(tmp_path: Path) -> None:
    output = tmp_path / "sparse-expert.json"
    main(
        [
            "--config",
            str(CONFIG),
            "--expert-count",
            "4",
            "--router-aux-weight",
            "0.01",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_sparse_expert_evidence_v1"
    assert payload["expert_count"] == 4
    assert payload["centralized_model_sha256"] == payload["distributed_model_sha256"]
