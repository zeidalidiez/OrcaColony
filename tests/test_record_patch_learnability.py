import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.record_patch_learnability import (
    _build_parser,
    _checkpoint_steps,
    _load_protocol,
    run_diagnostic_training,
)
from orcacolony.reference import load_campaign, run_training


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
RECORD_PATCH_CONFIG = (
    Path(__file__).parents[1] / "campaign" / "record-patch-t2-v1.json"
)
PROTOCOL = (
    Path(__file__).parents[1]
    / "capability"
    / "record-patch-v1"
    / "learnability-protocol.json"
)


def test_diagnostic_training_matches_reference_trajectory(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)

    diagnostic = run_diagnostic_training(
        campaign=campaign,
        output_dir=tmp_path / "diagnostic",
        checkpoint_steps=(0, 1, 2),
    )
    reference = run_training(
        campaign,
        output_dir=tmp_path / "reference",
        target_steps=2,
    )

    assert diagnostic.checkpoints[2].model_sha256 == reference.model_sha256
    assert diagnostic.checkpoints[2].loss_history == reference.loss_history
    assert len(diagnostic.diagnostics) == 2
    assert [item["step"] for item in diagnostic.diagnostics] == [1, 2]
    assert all(
        float(item["gradient_global_norm_before_clipping"]) > 0
        for item in diagnostic.diagnostics
    )
    assert all(
        float(item["update_global_norm"]) > 0
        for item in diagnostic.diagnostics
    )


@pytest.mark.parametrize(
    ("steps", "message"),
    (
        ((), "at least one"),
        ((1,), "begin at step zero"),
        ((0, 2, 1), "unique and increasing"),
        ((0, 1, 1), "unique and increasing"),
        ((0, -1), "nonnegative integers"),
    ),
)
def test_checkpoint_schedule_fails_closed(
    steps: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _checkpoint_steps(steps)


def test_learnability_cli_has_no_private_holdout_argument() -> None:
    destinations = {
        action.dest
        for action in _build_parser()._actions
    }

    assert "final_holdout" not in destinations
    assert "holdout_key" not in destinations


def test_frozen_learnability_protocol_matches_campaign() -> None:
    campaign = load_campaign(RECORD_PATCH_CONFIG)
    assert campaign.dataset is not None
    assert campaign.research is not None

    protocol = _load_protocol(
        PROTOCOL,
        campaign_sha256=hashlib.sha256(
            RECORD_PATCH_CONFIG.read_bytes()
        ).hexdigest(),
        dataset_revision=str(campaign.dataset["manifest_sha256"]),
        research=campaign.research,
        minimum_language_improvement=0.1,
    )

    assert protocol["schedule"]["checkpoint_steps"] == (
        0,
        1,
        8,
        32,
        128,
    )
    assert protocol["holdout_policy"] == {
        "behavioral_final_holdout": "must_not_open",
        "language_final_holdout": "must_not_evaluate",
    }


def test_behavioral_steps_must_be_checkpoint_milestones(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["schedule"]["behavioral_steps"] = [0, 2]
    changed = tmp_path / "protocol.json"
    changed.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    campaign = load_campaign(RECORD_PATCH_CONFIG)
    assert campaign.dataset is not None
    assert campaign.research is not None

    with pytest.raises(
        ValueError,
        match="behavioral steps must be checkpoints",
    ):
        _load_protocol(
            changed,
            campaign_sha256=hashlib.sha256(
                RECORD_PATCH_CONFIG.read_bytes()
            ).hexdigest(),
            dataset_revision=str(campaign.dataset["manifest_sha256"]),
            research=campaign.research,
            minimum_language_improvement=0.1,
        )
