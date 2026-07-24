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
