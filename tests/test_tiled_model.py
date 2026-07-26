from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.reference import CampaignConfig, load_campaign
from orcacolony.tiled_model import (
    main,
    run_tiled_block_experiment,
    run_tiled_block_sweep,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def _dataset_campaign(tmp_path: Path) -> tuple[CampaignConfig, PackedDataset]:
    campaign = load_campaign(CONFIG)
    corpus = (
        "A careful fox repaired the lantern before the storm.\n"
        "<|endoftext|>\n"
        "A small robot sorted every bright shell by color.\n"
        "<|endoftext|>\n"
    ) * 120
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus.encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/tiled-stories",
            "revision": "test-tiled-revision",
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


def test_boundary_tile_uses_authenticated_dataset_batch(tmp_path: Path) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)

    evidence = run_tiled_block_experiment(
        campaign,
        block_index=1,
        dataset=dataset,
    )

    assert evidence.block_index == 1
    assert evidence.centralized_loss == evidence.tiled_loss
    assert evidence.centralized_raw_gradient_sha256 == (
        evidence.tiled_raw_gradient_sha256
    )
    assert evidence.centralized_clipped_gradient_sha256 == (
        evidence.tiled_clipped_gradient_sha256
    )
    assert evidence.centralized_optimizer_sha256 == evidence.tiled_optimizer_sha256
    assert evidence.centralized_model_sha256 == evidence.tiled_model_sha256


def test_tiled_block_sweep_binds_dataset_and_every_block(tmp_path: Path) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)

    evidence = run_tiled_block_sweep(campaign, dataset)

    block_indices = tuple(range(campaign.model.layers))
    assert evidence.format == "orcacolony_tiled_block_sweep_evidence_v1"
    assert evidence.campaign_id == campaign.campaign["id"]
    assert evidence.dataset_revision == dataset.revision
    assert evidence.block_indices == block_indices
    assert tuple(point.block_index for point in evidence.blocks) == block_indices
    assert evidence.all_raw_gradients_exact is True
    assert evidence.all_clipped_gradients_exact is True
    assert evidence.all_optimizers_exact is True
    assert evidence.all_models_exact is True
    assert all(
        point.coordinator_selected_block_forward_calls == 0
        and point.tile_forward_calls == 1
        for point in evidence.blocks
    )
    assert len({point.centralized_model_sha256 for point in evidence.blocks}) == 1
    assert evidence.total_cold_assignment_transfer_tensor_bytes == sum(
        point.cold_assignment_transfer_tensor_bytes for point in evidence.blocks
    )
    assert evidence.total_warm_assignment_transfer_tensor_bytes == sum(
        point.warm_assignment_transfer_tensor_bytes for point in evidence.blocks
    )
    assert evidence.total_replicated_full_round_trip_tensor_bytes == sum(
        point.full_replica_round_trip_tensor_bytes for point in evidence.blocks
    )


def test_tiled_model_cli_runs_authenticated_all_block_sweep(tmp_path: Path) -> None:
    campaign, dataset = _dataset_campaign(tmp_path)
    config_path = tmp_path / "campaign.json"
    output_path = tmp_path / "evidence.json"
    config_path.write_text(json.dumps(asdict(campaign)), encoding="utf-8")

    main(
        [
            "--config",
            str(config_path),
            "--dataset",
            str(dataset.root),
            "--all-blocks",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_tiled_block_sweep_evidence_v1"
    assert payload["dataset_revision"] == dataset.revision
    assert payload["block_indices"] == list(range(campaign.model.layers))
    assert len(payload["blocks"]) == campaign.model.layers
    assert payload["all_raw_gradients_exact"] is True
    assert payload["all_optimizers_exact"] is True
    assert payload["all_models_exact"] is True
