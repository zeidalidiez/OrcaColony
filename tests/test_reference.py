import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, load_file, save

from orcacolony.reference import (
    _create_optimizer,
    _save_checkpoint,
    build_model,
    campaign_from_mapping,
    campaign_revision,
    campaign_to_mapping,
    compute_fixture,
    evaluation_slice,
    export_fixture,
    load_campaign,
    run_training,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
T1_CONFIG = Path(__file__).parents[1] / "campaign" / "t1-smoke.json"


def test_campaign_mapping_round_trip_uses_wire_schema() -> None:
    campaign = load_campaign(CONFIG)

    payload = campaign_to_mapping(campaign)

    assert "objective" not in payload
    assert payload["campaign"]["objective"] == campaign.objective.name
    assert payload["campaign"]["loss_mask"] == campaign.objective.loss_mask
    assert campaign_from_mapping(payload) == campaign


def test_campaign_revision_normalizes_an_absent_dataset_to_null() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    without_dataset = campaign_from_mapping(payload)
    payload["dataset"] = None
    with_explicit_null = campaign_from_mapping(payload)

    assert campaign_revision(without_dataset) == campaign_revision(
        with_explicit_null
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("objective", "supervised_fine_tuning", "unsupported campaign objective"),
        ("loss_mask", "target_only", "unsupported campaign loss mask"),
    ),
)
def test_campaign_objective_declarations_fail_closed(
    field: str,
    value: str,
    message: str,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["campaign"][field] = value

    with pytest.raises(ValueError, match=message):
        campaign_from_mapping(payload)


def test_campaign_id_rejects_markup_or_path_characters() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["campaign"]["id"] = "unsafe/id\n# heading"

    with pytest.raises(ValueError, match="campaign id must use"):
        campaign_from_mapping(payload)


def test_capability_campaign_requires_and_separates_final_holdout() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["evaluation"] = {
        "metric": "held_out_cross_entropy",
        "checkpoint_selection": "lowest_mean_loss",
        "validation_start_sequence": 0,
        "validation_sequences": 4,
        "batch_size": 2,
        "final_holdout": {
            "start_sequence": 4,
            "sequence_count": 4,
            "batch_size": 2,
        },
    }
    payload["research"] = {
        "format": "orcacolony_capability_research_v1",
        "claim": "Improve one frozen behavioral task.",
        "baseline": {
            "id": "initialization",
            "description": "The exact initialization checkpoint.",
            "revision": "sha256:" + "1" * 64,
        },
        "primary_metric": {
            "id": "task-score",
            "description": "Frozen task evaluator score.",
            "direction": "maximize",
            "unit": "ratio",
            "success_threshold": 0.7,
            "minimum_improvement_from_baseline": 0.05,
        },
        "guardrails": [
            {
                "id": "format-validity",
                "description": "Every output remains parseable.",
            }
        ],
        "analysis_plan": ["Compare checkpoint outputs and error categories."],
        "final_holdout_policy": "release_only_after_checkpoint_selection",
        "checkpoint_selection": (
            "lowest_validation_mean_loss_before_behavioral_final_holdout"
        ),
        "behavioral_evaluation": {
            "suite_id": "frozen-task-suite",
            "dataset_revision": "sha256:" + "2" * 64,
            "evaluator_revision": "3" * 40,
            "validation_split": "validation",
            "final_holdout_split": "final_holdout",
        },
    }
    payload["publication"] = {
        "format": "orcacolony_huggingface_publication_v1",
        "model_repo_id": "OrcaColony/test-capability-model",
        "dataset_repo_id": "OrcaColony/test-capability-model-dataset",
        "model_license": "mit",
        "dataset_license": "cdla-sharing-1.0",
        "visibility_policy": "private_review_then_public",
    }

    campaign = campaign_from_mapping(payload)
    validation = evaluation_slice(campaign, "validation")
    final_holdout = evaluation_slice(campaign, "final_holdout")
    assert validation.start_sequence == 0
    assert validation.sequence_count == 4
    assert final_holdout.start_sequence == 4
    assert final_holdout.sequence_count == 4

    payload["evaluation"]["final_holdout"] = {
        "start_sequence": 3,
        "sequence_count": 4,
        "batch_size": 2,
    }
    with pytest.raises(ValueError, match="must be disjoint"):
        campaign_from_mapping(payload)


def test_t0_fixture_has_exact_model_identity_and_deterministic_gradient() -> None:
    campaign = load_campaign(CONFIG)

    first = compute_fixture(campaign)
    second = compute_fixture(campaign)

    assert first.parameter_count == 1_334_016
    assert first.loss_weight_sum == 4 * 128
    assert first.loss_sum == pytest.approx(second.loss_sum, rel=0, abs=0)
    assert first.gradient_sha256 == second.gradient_sha256


def test_checkpoint_resume_matches_an_uninterrupted_training_run(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)

    uninterrupted = run_training(
        campaign,
        output_dir=tmp_path / "uninterrupted",
        target_steps=3,
    )
    split_dir = tmp_path / "split"
    run_training(campaign, output_dir=split_dir, target_steps=1)
    resumed = run_training(
        campaign,
        output_dir=split_dir,
        target_steps=3,
        resume_from=split_dir,
    )

    assert resumed.loss_history == uninterrupted.loss_history
    assert resumed.model_sha256 == uninterrupted.model_sha256
    assert (split_dir / "model.safetensors").is_file()
    assert (split_dir / "optimizer.safetensors").is_file()
    assert (split_dir / "state.json").is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("unexpected", None, "state schema is invalid"),
        ("step", 1.0, "step must be a nonnegative integer"),
        ("optimizer_step", 1.0, "optimizer step must be a nonnegative integer"),
        ("dataset_cursor", 4.0, "dataset cursor must be a nonnegative integer"),
        ("loss_history", [1], "loss history must contain finite JSON floats"),
    ),
)
def test_dense_checkpoint_requires_exact_trajectory_json_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    campaign = load_campaign(CONFIG)
    checkpoint = tmp_path / "checkpoint"
    run_training(campaign, output_dir=checkpoint, target_steps=1)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_training(
            campaign,
            output_dir=tmp_path / "resumed",
            target_steps=2,
            resume_from=checkpoint,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("optimizer_step", 0, "optimizer step must equal"),
        ("dataset_cursor", 0, "dataset cursor differs"),
        ("loss_history", [], "loss history differs"),
    ),
)
def test_dense_checkpoint_requires_exact_trajectory_relationships(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    campaign = load_campaign(CONFIG)
    checkpoint = tmp_path / "checkpoint"
    run_training(campaign, output_dir=checkpoint, target_steps=1)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_training(
            campaign,
            output_dir=tmp_path / "resumed",
            target_steps=2,
            resume_from=checkpoint,
        )


def test_dense_checkpoint_save_rejects_missing_optimizer_state_before_writes(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    model = build_model(campaign)
    optimizer = _create_optimizer(model, campaign.training)
    output_dir = tmp_path / "checkpoint"

    with pytest.raises(ValueError, match="optimizer parameter state schema"):
        _save_checkpoint(
            campaign,
            model,
            optimizer,
            output_dir,
            step=1,
            dataset_cursor=campaign.training.batch_size,
            loss_history=[1.0],
        )

    assert not output_dir.exists()


def test_dense_checkpoint_rejects_duplicate_keys_and_artifact_path_escape(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    checkpoint = tmp_path / "checkpoint"
    run_training(campaign, output_dir=checkpoint, target_steps=1)
    state_path = checkpoint / "state.json"
    original = state_path.read_text(encoding="utf-8")
    duplicate_state = original.replace(
        '"step": 1\n',
        '"step": 1,\n  "step": 1\n',
        1,
    )
    assert duplicate_state != original
    state_path.write_text(
        duplicate_state,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate checkpoint JSON key"):
        run_training(
            campaign,
            output_dir=tmp_path / "duplicate-resume",
            target_steps=2,
            resume_from=checkpoint,
        )

    state = json.loads(original)
    state["model"]["file"] = "../model.safetensors"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact schema is invalid"):
        run_training(
            campaign,
            output_dir=tmp_path / "escaped-resume",
            target_steps=2,
            resume_from=checkpoint,
        )


@pytest.mark.parametrize("mutation", ("missing", "dtype", "nonfinite"))
def test_dense_checkpoint_requires_exact_finite_model_tensor_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    campaign = load_campaign(CONFIG)
    checkpoint = tmp_path / "checkpoint"
    run_training(campaign, output_dir=checkpoint, target_steps=1)
    model_path = checkpoint / "model.safetensors"
    tensors = load(model_path.read_bytes())
    tensor_name = next(iter(tensors))
    if mutation == "missing":
        tensors.pop(tensor_name)
        message = "tensor schema is invalid"
    elif mutation == "dtype":
        tensors[tensor_name] = tensors[tensor_name].to(torch.float64)
        message = "tensor is invalid"
    else:
        changed = tensors[tensor_name].clone()
        changed.reshape(-1)[0] = float("nan")
        tensors[tensor_name] = changed
        message = "tensor is invalid"
    payload = save(tensors)
    model_path.write_bytes(payload)
    state_path = checkpoint / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["model"]["sha256"] = hashlib.sha256(payload).hexdigest()
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_training(
            campaign,
            output_dir=tmp_path / "resumed",
            target_steps=2,
            resume_from=checkpoint,
        )


def test_t1_fixture_exports_the_dynamic_browser_model_contract(tmp_path: Path) -> None:
    campaign = load_campaign(T1_CONFIG)
    fixture = export_fixture(campaign, tmp_path / "t1-fixture")
    manifest = json.loads(
        (fixture.output_dir / "fixture.json").read_text(encoding="utf-8")
    )
    gradients = load_file(str(fixture.output_dir / "gradients.safetensors"))

    assert manifest["model"] == {
        "vocab_size": 8192,
        "context_length": 256,
        "d_model": 256,
        "num_heads": 4,
        "num_layers": 6,
        "d_ff": 1024,
    }
    assert manifest["parameter_count"] == 6_901_760
    assert sum(tensor.numel() for tensor in gradients.values()) == 6_901_760
