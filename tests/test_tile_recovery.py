from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orcacolony.reference import load_campaign
from orcacolony.tile_recovery import main, run_recovered_tile_transaction


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
EXPECTED_FILES = {
    "manifest.json",
    "tile.safetensors",
    "input.safetensors",
    "forward-output.safetensors",
    "output-adjoint.safetensors",
    "result.safetensors",
}


def test_replacement_tile_replays_and_applies_one_exact_result(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    transaction_dir = tmp_path / "transaction"

    evidence = run_recovered_tile_transaction(
        campaign,
        block_index=2,
        transaction_dir=transaction_dir,
        timeout_seconds=30.0,
    )

    assert evidence.format == "orcacolony_recovered_tile_transaction_evidence_v1"
    assert evidence.start_method == "spawn"
    assert evidence.block_index == 2
    assert evidence.cursor == 0
    assert evidence.worker_model_transmissions == 2
    assert evidence.first_worker_terminated is True
    assert evidence.first_worker_exit_code != 0
    assert evidence.replacement_worker_exit_code == 0
    assert evidence.replay_output_bytes_identical is True
    assert evidence.duplicate_result_rejected is True
    assert evidence.phase_history == (
        "prepared",
        "forward_accepted",
        "worker_lost",
        "replay_verified",
        "adjoint_persisted",
        "result_accepted",
        "applied",
    )
    assert evidence.max_abs_raw_gradient_difference == 0.0
    assert evidence.max_abs_clipped_gradient_difference == 0.0
    assert evidence.max_abs_model_difference == 0.0
    assert (
        evidence.centralized_raw_gradient_sha256
        == evidence.recovered_raw_gradient_sha256
    )
    assert (
        evidence.centralized_clipped_gradient_sha256
        == evidence.recovered_clipped_gradient_sha256
    )
    assert (
        evidence.centralized_optimizer_sha256
        == evidence.recovered_optimizer_sha256
    )
    assert evidence.centralized_model_sha256 == evidence.recovered_model_sha256
    assert evidence.recovery_seconds > 0
    assert evidence.recovery_retransmitted_tensor_bytes == (
        evidence.tile_model_wire_bytes + evidence.input_wire_bytes
    )
    assert evidence.persisted_file_count == len(EXPECTED_FILES)
    assert evidence.persisted_tensor_bytes > 0

    assert {path.name for path in transaction_dir.iterdir()} == EXPECTED_FILES
    manifest = json.loads(
        (transaction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "orcacolony_boundary_transaction_v1"
    assert manifest["transaction_id"] == evidence.transaction_id
    assert manifest["phase"] == "applied"
    assert manifest["result_applied"] is True
    assert tuple(manifest["phase_history"]) == evidence.phase_history
    for item in evidence.persisted_files:
        path = transaction_dir / item.name
        assert path.stat().st_size == item.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item.sha256


def test_tile_recovery_cli_writes_evidence_and_transaction(
    tmp_path: Path,
) -> None:
    transaction_dir = tmp_path / "transaction"
    output_path = tmp_path / "evidence.json"

    main(
        [
            "--config",
            str(CONFIG),
            "--block-index",
            "2",
            "--transaction-dir",
            str(transaction_dir),
            "--timeout-seconds",
            "30",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_recovered_tile_transaction_evidence_v1"
    assert payload["replay_output_bytes_identical"] is True
    assert payload["duplicate_result_rejected"] is True
    assert payload["centralized_model_sha256"] == payload["recovered_model_sha256"]
    manifest = json.loads(
        (transaction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "applied"
    assert manifest["result_applied"] is True
