from __future__ import annotations

import json
from pathlib import Path

from orcacolony.full_process import main, run_persistent_full_process_control
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def test_persistent_full_process_reproduces_two_centralized_steps() -> None:
    campaign = load_campaign(CONFIG)

    evidence = run_persistent_full_process_control(
        campaign,
        timeout_seconds=30.0,
    )

    assert evidence.format == "orcacolony_persistent_full_process_evidence_v1"
    assert evidence.start_method == "spawn"
    assert evidence.assignment_count == 2
    assert evidence.model_transmissions == 1
    assert evidence.full_forward_calls == 2
    assert evidence.child_exit_code == 0
    assert evidence.initialization_round_trip_seconds > 0
    assert evidence.full_model_wire_bytes > 0
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
        evidence.full_model_wire_bytes
        + evidence.assignments[0].assignment_tensor_wire_bytes
    )
    assert evidence.warm_assignment_tensor_wire_bytes == (
        evidence.assignments[1].assignment_tensor_wire_bytes
    )
    assert evidence.cold_assignment_tensor_wire_bytes == (
        evidence.warm_assignment_tensor_wire_bytes
        + evidence.full_model_wire_bytes
    )
    for assignment in evidence.assignments:
        assert assignment.input_batch_wire_bytes > 0
        assert assignment.gradient_result_wire_bytes > 0
        assert assignment.assignment_tensor_wire_bytes == (
            assignment.input_batch_wire_bytes
            + assignment.gradient_result_wire_bytes
        )
        assert assignment.total_application_wire_bytes == (
            assignment.assignment_tensor_wire_bytes
            + assignment.control_json_wire_bytes
        )
        assert assignment.centralized_loss == assignment.process_loss
        assert assignment.max_abs_raw_gradient_difference == 0.0
        assert assignment.max_abs_clipped_gradient_difference == 0.0
        assert assignment.max_abs_model_difference == 0.0
        assert (
            assignment.centralized_raw_gradient_sha256
            == assignment.process_raw_gradient_sha256
        )
        assert (
            assignment.centralized_clipped_gradient_sha256
            == assignment.process_clipped_gradient_sha256
        )
        assert (
            assignment.centralized_optimizer_sha256
            == assignment.process_optimizer_sha256
        )
        assert (
            assignment.centralized_model_sha256
            == assignment.process_model_sha256
        )
        assert assignment.round_trip_seconds > 0
        assert assignment.worker_compute_seconds > 0
        assert assignment.worker_current_rss_bytes > 0
        assert assignment.worker_peak_rss_bytes >= assignment.worker_current_rss_bytes


def test_full_process_cli_writes_exact_evidence(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.json"

    main(
        [
            "--config",
            str(CONFIG),
            "--timeout-seconds",
            "30",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_persistent_full_process_evidence_v1"
    assert payload["assignment_count"] == 2
    assert payload["model_transmissions"] == 1
    assert payload["full_forward_calls"] == 2
    assert payload["child_exit_code"] == 0
    assert all(
        item["centralized_model_sha256"] == item["process_model_sha256"]
        for item in payload["assignments"]
    )
