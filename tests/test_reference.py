import json
from pathlib import Path

import pytest
from safetensors.torch import load_file

from orcacolony.reference import (
    compute_fixture,
    export_fixture,
    load_campaign,
    run_training,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
T1_CONFIG = Path(__file__).parents[1] / "campaign" / "t1-smoke.json"


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
