from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.reference import (
    CampaignConfig,
    campaign_to_mapping,
    load_campaign,
)
from orcacolony import sparse_expert_process
from orcacolony.sparse_expert_process import (
    _sha256_bytes,
    _validate_wire_identity,
    run_authenticated_sparse_process_experiment,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def _dataset_campaign(
    tmp_path: Path,
) -> tuple[CampaignConfig, PackedDataset]:
    campaign = load_campaign(CONFIG)
    corpus = (
        "A patient otter checked every rope before crossing the river.\n"
        "<|endoftext|>\n"
        "A careful raven placed each blue stone beside the old gate.\n"
        "<|endoftext|>\n"
    ) * 120
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus.encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/sparse-process-stories",
            "revision": "test-sparse-process-revision",
            "license": "cdla-sharing-1.0",
        },
        vocab_size=300,
        context_length=campaign.model.context_length,
    )
    dataset = PackedDataset.load(artifact_dir)
    campaign = replace(
        campaign,
        training=replace(campaign.training, dataset_sequences=8),
        dataset={
            "format": manifest["format"],
            "manifest_sha256": dataset.revision,
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "train_sha256": manifest["files"]["train.safetensors"],
            "validation_sha256": manifest["files"]["validation.safetensors"],
        },
        evaluation={
            "metric": "held_out_cross_entropy",
            "checkpoint_selection": "lowest_mean_loss",
            "validation_sequences": 4,
            "batch_size": 2,
        },
    )
    return campaign, dataset


def test_authenticated_sparse_process_reuses_heads_and_recovers_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)

    evidence = run_authenticated_sparse_process_experiment(
        campaign,
        dataset=dataset,
        expert_count=2,
        timeout_seconds=30.0,
    )

    assert (
        evidence.format
        == "orcacolony_authenticated_sparse_process_evidence_v1"
    )
    assert evidence.authentication_mode == (
        "coordinator-bound-sha256-safetensors-v1"
    )
    assert evidence.transport_scope == "trusted-local-spawn-pipe"
    assert evidence.start_method == "spawn"
    assert evidence.process_scheduling == (
        "sequential-workers-persistent-two-assignments"
    )
    assert evidence.assignment_state_mode == (
        "independent-one-step-controls-from-identical-initialization"
    )
    assert evidence.wire_accounting_scope == (
        "serialized-safetensors-and-json-payload-bytes-excludes-pipe-framing"
    )
    assert evidence.memory_scope == (
        "per-child-process-rss-not-concurrent-or-aggregate"
    )
    assert evidence.matched_totals_exclude_recovery is True
    assert evidence.maximum_simultaneous_worker_processes == 1
    assert evidence.expert_count == 2
    assert evidence.assignment_count == 2
    assert evidence.dataset_revision == dataset.revision
    assert evidence.frozen_head_wire_bytes > 0
    assert evidence.full_frozen_head_transmissions == 1
    assert evidence.expert_frozen_head_transmissions == 2
    assert evidence.full_trainable_state_transmissions == 2
    assert evidence.expert_trainable_state_transmissions == 4
    assert evidence.full_worker.forward_calls == 2
    assert evidence.full_worker.child_exit_code == 0
    assert tuple(worker.forward_calls for worker in evidence.expert_workers) == (
        2,
        2,
    )
    assert all(
        worker.child_exit_code == 0 for worker in evidence.expert_workers
    )
    assert evidence.full_cold_tensor_wire_bytes == (
        evidence.frozen_head_wire_bytes
        + evidence.assignments[0].full_tensor_wire_bytes
    )
    assert evidence.full_warm_tensor_wire_bytes == (
        evidence.assignments[1].full_tensor_wire_bytes
    )
    assert evidence.expert_cold_tensor_wire_bytes == (
        evidence.expert_count * evidence.frozen_head_wire_bytes
        + evidence.assignments[0].expert_aggregate_tensor_wire_bytes
    )
    assert evidence.expert_warm_tensor_wire_bytes == (
        evidence.assignments[1].expert_aggregate_tensor_wire_bytes
    )
    assert evidence.cold_tensor_wire_relative_change == (
        evidence.expert_cold_tensor_wire_bytes
        / evidence.full_cold_tensor_wire_bytes
        - 1.0
    )
    assert evidence.warm_tensor_wire_relative_change == (
        evidence.expert_warm_tensor_wire_bytes
        / evidence.full_warm_tensor_wire_bytes
        - 1.0
    )
    for assignment in evidence.assignments:
        assert assignment.full_tensor_wire_bytes == (
            assignment.full_trainable_state_wire_bytes
            + assignment.full_input_wire_bytes
            + assignment.full_gradient_result_wire_bytes
        )
        assert assignment.expert_aggregate_tensor_wire_bytes == (
            sum(assignment.expert_trainable_state_wire_bytes)
            + sum(assignment.expert_input_wire_bytes)
            + sum(assignment.expert_result_wire_bytes)
        )
        assert assignment.centralized_loss == assignment.full_process_loss
        assert assignment.centralized_loss == assignment.expert_process_loss
        assert assignment.full_max_abs_raw_gradient_difference == 0.0
        assert assignment.full_max_abs_clipped_gradient_difference == 0.0
        assert assignment.full_max_abs_model_difference == 0.0
        assert assignment.expert_max_abs_raw_gradient_difference == 0.0
        assert assignment.expert_max_abs_clipped_gradient_difference == 0.0
        assert assignment.expert_max_abs_model_difference == 0.0
        assert assignment.centralized_raw_gradient_sha256 == (
            assignment.full_process_raw_gradient_sha256
        )
        assert assignment.centralized_raw_gradient_sha256 == (
            assignment.expert_process_raw_gradient_sha256
        )
        assert assignment.centralized_clipped_gradient_sha256 == (
            assignment.full_process_clipped_gradient_sha256
        )
        assert assignment.centralized_clipped_gradient_sha256 == (
            assignment.expert_process_clipped_gradient_sha256
        )
        assert assignment.centralized_optimizer_sha256 == (
            assignment.full_process_optimizer_sha256
        )
        assert assignment.centralized_optimizer_sha256 == (
            assignment.expert_process_optimizer_sha256
        )
        assert assignment.centralized_model_sha256 == (
            assignment.full_process_model_sha256
        )
        assert assignment.centralized_model_sha256 == (
            assignment.expert_process_model_sha256
        )
    recovery = evidence.recovery
    assert recovery.assignment_accepted_before_loss is True
    assert recovery.first_worker_exit_code != 0
    assert recovery.replacement_worker_exit_code == 0
    assert recovery.replacement_result_matches_stable is True
    assert recovery.replacement_result_used_in_canonical_update is True
    assert recovery.stable_result_wire_sha256 == (
        recovery.replacement_result_wire_sha256
    )
    assert recovery.recovery_retransmitted_tensor_wire_bytes == (
        evidence.frozen_head_wire_bytes
        + evidence.expert_trainable_state_wire_bytes[0]
        + evidence.assignments[0].expert_input_wire_bytes[0]
    )
    assert recovery.replacement_result_tensor_wire_bytes == (
        evidence.assignments[0].expert_result_wire_bytes[0]
    )
    assert recovery.recovery_total_application_wire_bytes == (
        recovery.lost_worker_received_tensor_wire_bytes
        + recovery.recovery_retransmitted_tensor_wire_bytes
        + recovery.replacement_result_tensor_wire_bytes
        + recovery.recovery_control_json_wire_bytes
    )
    assert recovery.replacement_frozen_head_sha256 == (
        evidence.frozen_head_sha256
    )
    assert recovery.recovery_seconds > 0

    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(campaign_to_mapping(campaign), indent=2) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        sparse_expert_process,
        "run_authenticated_sparse_process_experiment",
        lambda *_args, **_kwargs: evidence,
    )
    sparse_expert_process.main(
        [
            "--config",
            str(campaign_path),
            "--dataset",
            str(dataset.root),
            "--expert-count",
            "2",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == evidence.format
    assert payload["recovery"]["replacement_result_matches_stable"] is True


def test_sparse_process_wire_identity_rejects_tampering() -> None:
    payload = b"authenticated tensor frame"
    _validate_wire_identity(
        payload,
        expected_sha256=_sha256_bytes(payload),
        expected_bytes=len(payload),
        label="test",
    )

    with pytest.raises(ValueError, match="wire identity mismatch"):
        _validate_wire_identity(
            payload + b"!",
            expected_sha256=_sha256_bytes(payload),
            expected_bytes=len(payload),
            label="test",
        )
