from __future__ import annotations

import json
from pathlib import Path

from orcacolony.reference import load_campaign
from orcacolony.tile_process import main, run_persistent_tile_process_experiment


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def test_persistent_tile_process_reuses_one_block_for_two_exact_assignments() -> None:
    campaign = load_campaign(CONFIG)

    evidence = run_persistent_tile_process_experiment(
        campaign,
        block_index=2,
        timeout_seconds=30.0,
    )

    assert evidence.format == "orcacolony_persistent_tile_process_evidence_v1"
    assert evidence.start_method == "spawn"
    assert evidence.block_index == 2
    assert evidence.assignment_count == 2
    assert evidence.model_transmissions == 1
    assert evidence.tile_forward_calls == 2
    assert evidence.child_exit_code == 0
    assert evidence.initialization_round_trip_seconds > 0
    assert evidence.tile_model_wire_bytes > 0
    assert evidence.worker_startup_current_rss_bytes > 0
    assert evidence.worker_after_model_current_rss_bytes > 0
    assert evidence.worker_after_model_peak_rss_bytes >= (
        evidence.worker_after_model_current_rss_bytes
    )
    assert evidence.worker_final_peak_rss_bytes >= (
        evidence.worker_after_model_peak_rss_bytes
    )
    assert tuple(item.cursor for item in evidence.assignments) == (
        0,
        campaign.training.batch_size,
    )
    assert evidence.cold_assignment_tensor_wire_bytes == (
        evidence.tile_model_wire_bytes
        + evidence.assignments[0].boundary_tensor_wire_bytes
    )
    assert evidence.warm_assignment_tensor_wire_bytes == (
        evidence.assignments[1].boundary_tensor_wire_bytes
    )
    assert evidence.cold_assignment_tensor_wire_bytes == (
        evidence.warm_assignment_tensor_wire_bytes
        + evidence.tile_model_wire_bytes
    )
    for assignment in evidence.assignments:
        assert assignment.forward_input_wire_bytes > 0
        assert assignment.forward_output_wire_bytes > 0
        assert assignment.backward_output_adjoint_wire_bytes > 0
        assert assignment.backward_result_wire_bytes > 0
        assert assignment.boundary_tensor_wire_bytes == (
            assignment.forward_input_wire_bytes
            + assignment.forward_output_wire_bytes
            + assignment.backward_output_adjoint_wire_bytes
            + assignment.backward_result_wire_bytes
        )
        assert assignment.total_application_wire_bytes == (
            assignment.boundary_tensor_wire_bytes
            + assignment.control_json_wire_bytes
        )
        assert assignment.centralized_loss == assignment.process_tiled_loss
        assert assignment.max_abs_raw_gradient_difference == 0.0
        assert assignment.max_abs_clipped_gradient_difference == 0.0
        assert assignment.max_abs_model_difference == 0.0
        assert assignment.centralized_raw_gradient_sha256 == (
            assignment.process_raw_gradient_sha256
        )
        assert assignment.centralized_clipped_gradient_sha256 == (
            assignment.process_clipped_gradient_sha256
        )
        assert assignment.centralized_optimizer_sha256 == (
            assignment.process_optimizer_sha256
        )
        assert assignment.centralized_model_sha256 == assignment.process_model_sha256
        assert assignment.forward_round_trip_seconds > 0
        assert assignment.backward_round_trip_seconds > 0
        assert assignment.worker_forward_seconds > 0
        assert assignment.worker_backward_seconds > 0
        assert assignment.worker_current_rss_bytes > 0
        assert assignment.worker_peak_rss_bytes >= assignment.worker_current_rss_bytes


def test_tile_process_cli_writes_exact_evidence(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.json"

    main(
        [
            "--config",
            str(CONFIG),
            "--block-index",
            "2",
            "--timeout-seconds",
            "30",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_persistent_tile_process_evidence_v1"
    assert payload["assignment_count"] == 2
    assert payload["model_transmissions"] == 1
    assert payload["tile_forward_calls"] == 2
    assert payload["child_exit_code"] == 0
    assert all(
        item["centralized_model_sha256"] == item["process_model_sha256"]
        for item in payload["assignments"]
    )
