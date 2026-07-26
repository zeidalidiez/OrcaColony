from __future__ import annotations

import json
from pathlib import Path

from orcacolony.reference import load_campaign
from orcacolony.tiled_model import main, run_tiled_block_experiment


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def test_boundary_tile_reproduces_centralized_gradient_and_adamw_step() -> None:
    campaign = load_campaign(CONFIG)

    evidence = run_tiled_block_experiment(campaign, block_index=2)

    assert evidence.format == "orcacolony_tiled_block_evidence_v1"
    assert evidence.block_index == 2
    assert evidence.boundary_shape == (
        campaign.training.batch_size,
        campaign.model.context_length,
        campaign.model.width,
    )
    assert evidence.coordinator_selected_block_forward_calls == 0
    assert evidence.tile_forward_calls == 1
    assert evidence.tile_parameter_count < evidence.full_parameter_count
    assert evidence.tile_model_payload_tensor_bytes < (
        evidence.full_model_payload_tensor_bytes
    )
    assert evidence.input_activation_tensor_bytes == (
        evidence.output_activation_tensor_bytes
    )
    assert evidence.input_activation_tensor_bytes == (
        evidence.input_adjoint_tensor_bytes
    )
    assert evidence.output_activation_tensor_bytes == (
        evidence.output_adjoint_tensor_bytes
    )
    assert evidence.forward_boundary_transfer_tensor_bytes == (
        evidence.input_activation_tensor_bytes
        + evidence.output_activation_tensor_bytes
    )
    assert evidence.backward_boundary_transfer_tensor_bytes == (
        evidence.output_adjoint_tensor_bytes
        + evidence.input_adjoint_tensor_bytes
    )
    assert evidence.tile_gradient_upload_tensor_bytes == (
        evidence.tile_model_payload_tensor_bytes
    )
    assert evidence.cold_assignment_transfer_tensor_bytes == (
        evidence.tile_model_payload_tensor_bytes
        + evidence.forward_boundary_transfer_tensor_bytes
        + evidence.backward_boundary_transfer_tensor_bytes
        + evidence.tile_gradient_upload_tensor_bytes
    )
    assert evidence.warm_assignment_transfer_tensor_bytes == (
        evidence.cold_assignment_transfer_tensor_bytes
        - evidence.tile_model_payload_tensor_bytes
    )
    assert evidence.full_replica_round_trip_tensor_bytes == (
        2 * evidence.full_model_payload_tensor_bytes
    )
    assert evidence.centralized_loss == evidence.tiled_loss
    assert evidence.max_abs_raw_gradient_difference == 0.0
    assert evidence.max_abs_clipped_gradient_difference == 0.0
    assert evidence.max_abs_model_difference == 0.0
    assert evidence.centralized_raw_gradient_sha256 == (
        evidence.tiled_raw_gradient_sha256
    )
    assert evidence.centralized_clipped_gradient_sha256 == (
        evidence.tiled_clipped_gradient_sha256
    )
    assert evidence.centralized_optimizer_sha256 == evidence.tiled_optimizer_sha256
    assert evidence.centralized_model_sha256 == evidence.tiled_model_sha256
    assert evidence.centralized_step_seconds > 0
    assert evidence.tiled_step_seconds > 0
    assert (
        evidence.combined_process_peak_rss_bytes is None
        or evidence.combined_process_peak_rss_bytes > 0
    )


def test_tiled_model_cli_writes_exact_evidence(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.json"

    main(
        [
            "--config",
            str(CONFIG),
            "--block-index",
            "2",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_tiled_block_evidence_v1"
    assert payload["block_index"] == 2
    assert payload["centralized_raw_gradient_sha256"] == (
        payload["tiled_raw_gradient_sha256"]
    )
    assert payload["centralized_optimizer_sha256"] == (
        payload["tiled_optimizer_sha256"]
    )
    assert payload["centralized_model_sha256"] == payload["tiled_model_sha256"]
