import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.record_patch_continuation import (
    _build_parser,
    _continuation_steps,
    _load_continuation_protocol,
    run_continuation_training,
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
    / "continuation-protocol.json"
)


def test_continuation_matches_uninterrupted_reference_trajectory(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    parent = run_training(
        campaign,
        output_dir=tmp_path / "parent",
        target_steps=1,
    )

    continuation = run_continuation_training(
        campaign=campaign,
        resume_from=parent.checkpoint_dir,
        output_dir=tmp_path / "continuation",
        checkpoint_steps=(1, 2),
    )
    reference = run_training(
        campaign,
        output_dir=tmp_path / "reference",
        target_steps=2,
    )

    assert continuation.resume_step == 1
    assert continuation.final_step == 2
    assert (
        continuation.checkpoints[1].model_sha256
        == parent.model_sha256
    )
    assert (
        continuation.checkpoints[2].model_sha256
        == reference.model_sha256
    )
    assert (
        continuation.checkpoints[2].loss_history
        == reference.loss_history
    )
    assert [item["step"] for item in continuation.diagnostics] == [2]


@pytest.mark.parametrize(
    ("steps", "resume_step", "message"),
    (
        ((), 1, "at least one"),
        ((0, 1), 1, "begin at the resume"),
        ((1, 3, 2), 1, "unique and increasing"),
        ((1, 2, 2), 1, "unique and increasing"),
        ((1, -1), 1, "nonnegative integers"),
    ),
)
def test_continuation_schedule_fails_closed(
    steps: tuple[int, ...],
    resume_step: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _continuation_steps(steps, resume_step=resume_step)


def test_continuation_cli_has_no_private_holdout_argument() -> None:
    destinations = {
        action.dest
        for action in _build_parser()._actions
    }

    assert "parent_run" in destinations
    assert "final_holdout" not in destinations
    assert "holdout_key" not in destinations


def test_frozen_continuation_protocol_matches_campaign() -> None:
    campaign = load_campaign(RECORD_PATCH_CONFIG)
    assert campaign.dataset is not None

    protocol = _load_continuation_protocol(
        PROTOCOL,
        campaign=campaign,
        campaign_sha256=hashlib.sha256(
            RECORD_PATCH_CONFIG.read_bytes()
        ).hexdigest(),
        dataset_revision=str(campaign.dataset["manifest_sha256"]),
    )

    assert protocol["schedule"]["checkpoint_steps"] == (
        128,
        256,
        512,
    )
    assert protocol["trajectory"]["resume_step"] == 128
    assert protocol["trajectory"]["dataset_cursor"] == 512
    assert protocol["parent_run"]["evidence_sha256"] == (
        "92ec3a0d0fcf49652abbaf24274e612a9e8137395f9e0e80d1e0c8c27c7cca16"
    )
    assert protocol["holdout_policy"] == {
        "behavioral_final_holdout": "must_not_open",
        "language_final_holdout": "must_not_evaluate",
    }


def test_behavioral_steps_must_be_continuation_checkpoints(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["schedule"]["behavioral_steps"] = [128, 300]
    changed = tmp_path / "protocol.json"
    changed.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    campaign = load_campaign(RECORD_PATCH_CONFIG)
    assert campaign.dataset is not None

    with pytest.raises(
        ValueError,
        match="behavioral steps must be checkpoints",
    ):
        _load_continuation_protocol(
            changed,
            campaign=campaign,
            campaign_sha256=hashlib.sha256(
                RECORD_PATCH_CONFIG.read_bytes()
            ).hexdigest(),
            dataset_revision=str(campaign.dataset["manifest_sha256"]),
        )
