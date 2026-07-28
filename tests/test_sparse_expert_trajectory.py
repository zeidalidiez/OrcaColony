from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony import sparse_expert_trajectory
from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.reference import (
    CampaignConfig,
    campaign_to_mapping,
    load_campaign,
)
from orcacolony.sparse_expert_trajectory import (
    _apply_transaction_once,
    _canonical_json,
    _compute_transaction_candidate,
    _load_manifest,
    _reconcile_results,
    _validate_applied_checkpoint,
    run_persisted_sparse_trajectory_experiment,
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
            "dataset": "test/sparse-trajectory-stories",
            "revision": "test-sparse-trajectory-revision",
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


def test_persisted_sparse_trajectory_is_exact_and_recovers_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)
    state_dir = tmp_path / "trajectory-state"
    evidence = run_persisted_sparse_trajectory_experiment(
        campaign,
        state_dir,
        dataset=dataset,
        steps=2,
        expert_count=2,
        timeout_seconds=30.0,
        sample_interval_seconds=0.005,
    )

    assert (
        evidence.format
        == "orcacolony_persisted_sparse_trajectory_evidence_v1"
    )
    assert evidence.authentication_mode == (
        "coordinator-bound-sha256-safetensors-v1"
    )
    assert evidence.transport_scope == "trusted-local-spawn-pipe"
    assert evidence.start_method == "spawn"
    assert evidence.maximum_simultaneous_worker_processes == 1
    assert evidence.expert_count == 2
    assert evidence.step_count == 2
    assert evidence.all_steps_exact is True
    assert evidence.assignment_state_mode == (
        "sequential-adamw-trajectory-refreshed-model-routes-hidden-and-experts"
    )
    assert evidence.centralized_final_model_sha256 == (
        evidence.full_process_final_model_sha256
    )
    assert evidence.centralized_final_model_sha256 == (
        evidence.expert_process_final_model_sha256
    )
    assert evidence.centralized_final_optimizer_sha256 == (
        evidence.full_process_final_optimizer_sha256
    )
    assert evidence.centralized_final_optimizer_sha256 == (
        evidence.expert_process_final_optimizer_sha256
    )
    assert evidence.full_tensor_wire_bytes > 0
    assert evidence.expert_tensor_wire_bytes > 0
    assert evidence.full_persisted_bytes > 0
    assert evidence.expert_persisted_bytes > 0
    assert evidence.centralized_end_to_end_seconds > 0
    assert evidence.full_process_end_to_end_seconds > 0
    assert evidence.expert_process_end_to_end_seconds > 0

    assert len(evidence.full_workers) == 1
    assert evidence.full_workers[0].forward_calls == 2
    assert evidence.full_workers[0].child_exit_code == 0
    assert len(evidence.expert_workers) == 2
    assert sum(worker.forward_calls for worker in evidence.expert_workers) == 4
    assert evidence.expert_workers[0].child_exit_code != 0
    assert evidence.expert_workers[1].child_exit_code == 0
    for worker in (*evidence.full_workers, *evidence.expert_workers):
        rss = worker.external_rss
        assert rss.source == "linux-proc-status-v1"
        assert rss.sample_count > 0
        assert rss.startup_sample_count > 0
        assert rss.assignment_sample_count > 0
        assert rss.shutdown_sample_count > 0
        assert rss.lifecycle_max_current_rss_bytes > 0
        assert rss.lifecycle_max_hwm_rss_bytes > 0
        assert rss.lifecycle_max_hwm_rss_bytes >= (
            rss.startup_max_hwm_rss_bytes
        )
        assert rss.lifecycle_max_hwm_rss_bytes >= (
            rss.assignment_max_hwm_rss_bytes
        )

    replacement = evidence.worker_replacement
    assert replacement.step == 1
    assert replacement.persisted_result_index == 0
    assert replacement.persisted_result_survived_loss is True
    assert replacement.first_worker_exit_code != 0
    assert replacement.replacement_worker_exit_code == 0
    assert replacement.replacement_initialized_after_loss is True
    assert replacement.recomputed_persisted_result is False

    recovery = evidence.coordinator_recovery
    assert recovery.step == 1
    assert recovery.applied_checkpoint_published_before_loss is True
    assert recovery.manifest_applied_before_loss is False
    assert recovery.recovered_from_published_checkpoint is True
    assert recovery.recovery_start_method == "spawn"
    assert recovery.recovery_process_exit_code == 0
    assert recovery.new_process_loaded_only_persisted_state is True
    assert recovery.recomputed_from_persisted_pre_state_for_validation is True
    assert recovery.duplicate_apply_rejected is True
    assert recovery.recovery_seconds > 0

    for step in evidence.steps:
        assert step.centralized_loss == step.full_process_loss
        assert step.centralized_loss == step.expert_process_loss
        assert step.full_max_abs_raw_gradient_difference == 0.0
        assert step.expert_max_abs_raw_gradient_difference == 0.0
        assert step.full_max_abs_clipped_gradient_difference == 0.0
        assert step.expert_max_abs_clipped_gradient_difference == 0.0
        assert step.full_max_abs_model_difference == 0.0
        assert step.expert_max_abs_model_difference == 0.0
        assert step.centralized_raw_gradient_sha256 == (
            step.full_process_raw_gradient_sha256
        )
        assert step.centralized_raw_gradient_sha256 == (
            step.expert_process_raw_gradient_sha256
        )
        assert step.centralized_clipped_gradient_sha256 == (
            step.full_process_clipped_gradient_sha256
        )
        assert step.centralized_clipped_gradient_sha256 == (
            step.expert_process_clipped_gradient_sha256
        )
        assert step.centralized_optimizer_sha256 == (
            step.full_process_optimizer_sha256
        )
        assert step.centralized_optimizer_sha256 == (
            step.expert_process_optimizer_sha256
        )
        assert step.centralized_model_sha256 == (
            step.full_process_model_sha256
        )
        assert step.centralized_model_sha256 == (
            step.expert_process_model_sha256
        )
        assert step.full_transaction.phase_history == (
            "prepared",
            "results_accepted",
            "applied",
        )
        assert step.expert_transaction.phase_history == (
            "prepared",
            "results_accepted",
            "applied",
        )
        assert step.full_transaction.accepted_result_count == 1
        assert step.expert_transaction.accepted_result_count == 2
        assert step.full_transaction.duplicate_apply_rejected is True
        assert step.expert_transaction.duplicate_apply_rejected is True
        assert step.full_end_to_end_step_seconds > 0
        assert step.expert_end_to_end_step_seconds > 0
        assert step.full_coordinator_apply_seconds > 0
        assert step.expert_coordinator_apply_seconds > 0
    assert evidence.steps[1].centralized_pre_model_sha256 == (
        evidence.steps[0].centralized_model_sha256
    )
    assert evidence.steps[1].full_pre_model_sha256 == (
        evidence.steps[0].full_process_model_sha256
    )
    assert evidence.steps[1].expert_pre_model_sha256 == (
        evidence.steps[0].expert_process_model_sha256
    )
    assert evidence.steps[0].full_trainable_state_sha256 != (
        evidence.steps[1].full_trainable_state_sha256
    )
    assert evidence.steps[0].expert_trainable_state_sha256 != (
        evidence.steps[1].expert_trainable_state_sha256
    )

    final_expert_transaction = state_dir / "expert" / "step-00000001"
    with pytest.raises(ValueError, match="already applied"):
        _apply_transaction_once(campaign, final_expert_transaction)

    lagging_manifest_transaction = (
        state_dir / "full" / "step-00000000"
    )
    shutil.rmtree(lagging_manifest_transaction / "applied")
    lagging_manifest = _load_manifest(lagging_manifest_transaction)
    lagging_manifest["phase"] = "prepared"
    lagging_manifest["phase_history"] = ["prepared"]
    lagging_manifest["accepted_result_count"] = 0
    lagging_manifest["accepted_results"] = []
    lagging_manifest["result_applied"] = False
    lagging_manifest["applied_checkpoint"] = None
    (lagging_manifest_transaction / "manifest.json").write_bytes(
        _canonical_json(lagging_manifest)
    )
    reconciled_manifest = _load_manifest(lagging_manifest_transaction)
    records, _, _ = _reconcile_results(
        lagging_manifest_transaction,
        reconciled_manifest,
        write=True,
    )
    assert len(records) == 1
    assert _load_manifest(lagging_manifest_transaction)["phase"] == (
        "results_accepted"
    )
    _apply_transaction_once(campaign, lagging_manifest_transaction)

    candidate = _compute_transaction_candidate(
        campaign,
        final_expert_transaction,
        allow_applied=True,
    )
    applied_model = (
        final_expert_transaction / "applied" / "model.safetensors"
    )
    applied_original = applied_model.read_bytes()
    applied_model.write_bytes(
        applied_original[:-1] + bytes([applied_original[-1] ^ 1])
    )
    with pytest.raises(ValueError, match="digest changed"):
        _validate_applied_checkpoint(
            final_expert_transaction,
            candidate,
        )
    applied_model.write_bytes(applied_original)

    first_result = (
        state_dir
        / "expert"
        / "step-00000000"
        / "results"
        / "result-000"
        / "result.safetensors"
    )
    original = first_result.read_bytes()
    first_result.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    transaction = first_result.parents[2]
    manifest = _load_manifest(transaction)
    with pytest.raises(ValueError, match="digest changed"):
        _reconcile_results(transaction, manifest, write=False)

    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(campaign_to_mapping(campaign), indent=2) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        sparse_expert_trajectory,
        "run_persisted_sparse_trajectory_experiment",
        lambda *_args, **_kwargs: evidence,
    )
    sparse_expert_trajectory.main(
        [
            "--config",
            str(campaign_path),
            "--dataset",
            str(dataset.root),
            "--state",
            str(tmp_path / "cli-state"),
            "--steps",
            "2",
            "--expert-count",
            "2",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == evidence.format
    assert payload["all_steps_exact"] is True
    assert payload["worker_replacement"]["recomputed_persisted_result"] is False


def test_persisted_sparse_trajectory_rejects_invalid_controls(
    tmp_path: Path,
) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)

    with pytest.raises(ValueError, match="steps"):
        run_persisted_sparse_trajectory_experiment(
            campaign,
            tmp_path / "one-step",
            dataset=dataset,
            steps=1,
            expert_count=2,
        )
    with pytest.raises(ValueError, match="sample interval"):
        run_persisted_sparse_trajectory_experiment(
            campaign,
            tmp_path / "bad-sampler",
            dataset=dataset,
            steps=2,
            expert_count=2,
            sample_interval_seconds=0.5,
        )
