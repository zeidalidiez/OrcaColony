from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.partial_model import (
    RollingBlockWorker,
    RollingBlockWorkerSession,
    main,
    run_block_sharded_experiment,
    run_dataset_rolling_block_experiment,
    run_rolling_block_experiment,
)
from orcacolony.reference import (
    CampaignConfig,
    build_model,
    campaign_to_mapping,
    load_campaign,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def _evaluated_dataset_campaign(
    tmp_path: Path,
) -> tuple[CampaignConfig, PackedDataset]:
    campaign = load_campaign(CONFIG)
    corpus = (
        "A patient rabbit planted a seed and watered it every morning.\n"
        "<|endoftext|>\n"
        "A young whale followed the moonlit waves safely home.\n"
        "<|endoftext|>\n"
    ) * 120
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus.encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/rolling-block-stories",
            "revision": "test-rolling-block-revision",
            "license": "cdla-sharing-1.0",
        },
        vocab_size=300,
        context_length=campaign.model.context_length,
    )
    dataset = PackedDataset.load(artifact_dir)
    campaign = replace(
        campaign,
        training=replace(
            campaign.training,
            dataset_sequences=8,
            steps=4,
        ),
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


def test_rolling_block_worker_session_reuses_shared_state_when_block_rotates() -> None:
    campaign = load_campaign(CONFIG)
    coordinator = build_model(campaign)
    session = RollingBlockWorkerSession(coordinator, 0)
    shared_module_ids = (
        id(session.token_embedding),
        id(session.position_embedding),
        id(session.final_norm),
    )
    initial_block_id = id(session.block)

    transferred_bytes = session.select_block(coordinator, 2)

    assert session.block_index == 2
    assert id(session.block) != initial_block_id
    assert (
        id(session.token_embedding),
        id(session.position_embedding),
        id(session.final_norm),
    ) == shared_module_ids
    assert transferred_bytes > 0
    assert all(
        not parameter.requires_grad
        for module in (
            session.token_embedding,
            session.position_embedding,
            session.final_norm,
        )
        for parameter in module.parameters()
    )
    assert all(parameter.requires_grad for parameter in session.block.parameters())
    assert all(
        worker_parameter.detach().equal(coordinator_parameter.detach())
        for worker_parameter, coordinator_parameter in zip(
            session.block.parameters(),
            coordinator.blocks[2].parameters(),
            strict=True,
        )
    )


def test_rolling_block_worker_session_rejects_changed_shared_state() -> None:
    campaign = load_campaign(CONFIG)
    coordinator = build_model(campaign)
    session = RollingBlockWorkerSession(coordinator, 0)
    coordinator.token_embedding.weight.detach()[0, 0].add_(1.0)

    with pytest.raises(ValueError, match="shared coordinator state changed"):
        session.select_block(coordinator, 1)


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


def test_dataset_rolling_block_experiment_records_persistent_transfer_and_evaluation(
    tmp_path: Path,
) -> None:
    campaign, dataset = _evaluated_dataset_campaign(tmp_path)

    evidence = run_dataset_rolling_block_experiment(
        campaign,
        dataset,
        steps=4,
        evaluation_interval=2,
    )

    assert evidence.format == "orcacolony_dataset_rolling_block_evidence_v1"
    assert evidence.dataset_revision == dataset.revision
    assert evidence.block_sequence == (0, 1, 2, 3)
    assert tuple(point.step for point in evidence.evaluation_history) == (0, 2, 4)
    assert evidence.evaluation_history[0].rolling_mean_loss == (
        evidence.evaluation_history[0].full_baseline_mean_loss
    )
    assert all(
        math.isfinite(loss)
        for point in evidence.evaluation_history
        for loss in (point.rolling_mean_loss, point.full_baseline_mean_loss)
    )
    assert evidence.shared_state_loads == 1
    assert evidence.block_state_loads == evidence.steps
    assert evidence.persistent_session_payload_tensor_bytes == (
        evidence.shared_payload_tensor_bytes
        + evidence.selected_block_payload_tensor_bytes * evidence.steps
    )
    assert evidence.persistent_session_payload_tensor_bytes < (
        evidence.transient_worker_payload_tensor_bytes
    )
    assert evidence.transient_worker_payload_tensor_bytes == (
        evidence.worker_payload_tensor_bytes * evidence.steps
    )
    assert evidence.unique_coverage_payload_tensor_bytes == (
        evidence.full_payload_tensor_bytes
    )
    assert evidence.replicated_full_payload_tensor_bytes == (
        evidence.full_payload_tensor_bytes * evidence.steps
    )
    assert evidence.rolling_validation_loss_improvement == (
        evidence.initial_validation_mean_loss
        - evidence.rolling_final_validation_mean_loss
    )
    assert evidence.full_baseline_validation_loss_improvement == (
        evidence.initial_validation_mean_loss
        - evidence.full_baseline_final_validation_mean_loss
    )
    assert evidence.experiment_wall_seconds > 0
    assert evidence.rolling_training_seconds > 0
    assert evidence.full_baseline_training_seconds > 0
    assert evidence.evaluation_seconds > 0
    assert evidence.experiment_wall_seconds >= (
        evidence.rolling_training_seconds
        + evidence.full_baseline_training_seconds
        + evidence.evaluation_seconds
    )
    assert (
        evidence.combined_process_peak_rss_bytes is None
        or evidence.combined_process_peak_rss_bytes > 0
    )
    assert evidence.rolling_model_sha256 != evidence.full_baseline_model_sha256


def test_block_sharded_experiment_maps_every_block_before_one_global_step(
    tmp_path: Path,
) -> None:
    campaign, dataset = _evaluated_dataset_campaign(tmp_path)

    evidence = run_block_sharded_experiment(
        campaign,
        dataset,
        steps=2,
        evaluation_interval=1,
    )

    layers = campaign.model.layers
    assert evidence.format == "orcacolony_block_sharded_evidence_v1"
    assert evidence.dataset_revision == dataset.revision
    assert evidence.global_steps == 2
    assert evidence.worker_assignments == 2 * layers
    assert evidence.workers_per_global_step == layers
    assert evidence.assignment_block_sequence == tuple(range(layers)) * 2
    assert evidence.block_update_counts == (2,) * layers
    assert evidence.coordinator_optimizer_steps == 2
    assert evidence.block_optimizer_steps == (2,) * layers
    assert evidence.shared_optimizer_state_parameter_count == 0
    assert tuple(point.step for point in evidence.evaluation_history) == (0, 1, 2)
    assert evidence.evaluation_history[0].sharded_mean_loss == (
        evidence.evaluation_history[0].full_baseline_mean_loss
    )
    assert evidence.initial_shared_state_sha256 == evidence.final_shared_state_sha256
    assert evidence.updated_block_count == layers
    assert evidence.shared_state_loads == layers
    assert evidence.block_state_loads == 2 * layers
    assert evidence.individual_worker_unique_payload_tensor_bytes == (
        evidence.worker_payload_tensor_bytes
    )
    assert evidence.individual_worker_unique_payload_tensor_bytes < (
        evidence.full_payload_tensor_bytes
    )
    assert evidence.aggregate_worker_resident_tensor_bytes == (
        evidence.worker_resident_tensor_bytes * layers
    )
    assert evidence.cold_aggregate_download_tensor_bytes == (
        evidence.worker_payload_tensor_bytes * layers
    )
    assert evidence.colony_unique_payload_tensor_bytes == (
        evidence.cold_aggregate_download_tensor_bytes
    )
    assert evidence.warm_aggregate_download_per_step_tensor_bytes == (
        evidence.selected_block_payload_tensor_bytes * layers
    )
    assert evidence.persistent_aggregate_download_tensor_bytes == (
        evidence.cold_aggregate_download_tensor_bytes
        + evidence.warm_aggregate_download_per_step_tensor_bytes
    )
    assert evidence.individual_worker_persistent_download_tensor_bytes == (
        evidence.worker_payload_tensor_bytes
        + evidence.selected_block_payload_tensor_bytes
    )
    assert evidence.mapped_gradient_upload_tensor_bytes == (
        evidence.mapped_gradient_bytes_per_assignment * 2 * layers
    )
    assert evidence.persistent_aggregate_round_trip_tensor_bytes == (
        evidence.persistent_aggregate_download_tensor_bytes
        + evidence.mapped_gradient_upload_tensor_bytes
    )
    assert evidence.replicated_full_download_tensor_bytes == (
        evidence.full_payload_tensor_bytes * 2
    )
    assert evidence.replicated_full_round_trip_tensor_bytes == (
        2 * evidence.replicated_full_download_tensor_bytes
    )
    assert evidence.sharded_validation_loss_improvement == (
        evidence.initial_validation_mean_loss
        - evidence.sharded_final_validation_mean_loss
    )
    assert evidence.full_baseline_validation_loss_improvement == (
        evidence.initial_validation_mean_loss
        - evidence.full_baseline_final_validation_mean_loss
    )
    assert evidence.experiment_wall_seconds > 0
    assert evidence.sharded_training_seconds > 0
    assert evidence.full_baseline_training_seconds > 0
    assert evidence.evaluation_seconds > 0
    assert evidence.experiment_wall_seconds >= (
        evidence.sharded_training_seconds
        + evidence.full_baseline_training_seconds
        + evidence.evaluation_seconds
    )
    assert (
        evidence.combined_process_peak_rss_bytes is None
        or evidence.combined_process_peak_rss_bytes > 0
    )
    assert evidence.sharded_model_sha256 != evidence.full_baseline_model_sha256


def test_partial_model_cli_runs_authenticated_dataset_experiment(
    tmp_path: Path,
) -> None:
    campaign, dataset = _evaluated_dataset_campaign(tmp_path)
    config_path = tmp_path / "campaign.json"
    output_path = tmp_path / "evidence.json"
    config_path.write_text(
        json.dumps(campaign_to_mapping(campaign)),
        encoding="utf-8",
    )

    main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset.root),
            "--steps",
            "2",
            "--evaluation-interval",
            "1",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_dataset_rolling_block_evidence_v1"
    assert payload["dataset_revision"] == dataset.revision
    assert [point["step"] for point in payload["evaluation_history"]] == [0, 1, 2]


def test_partial_model_cli_selects_block_sharded_topology(tmp_path: Path) -> None:
    campaign, dataset = _evaluated_dataset_campaign(tmp_path)
    config_path = tmp_path / "campaign.json"
    output_path = tmp_path / "evidence.json"
    config_path.write_text(
        json.dumps(campaign_to_mapping(campaign)),
        encoding="utf-8",
    )

    main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset.root),
            "--topology",
            "block-sharded",
            "--steps",
            "2",
            "--evaluation-interval",
            "1",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_block_sharded_evidence_v1"
    assert payload["dataset_revision"] == dataset.revision
    assert payload["global_steps"] == 2
    assert payload["worker_assignments"] == 2 * campaign.model.layers
