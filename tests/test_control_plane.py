import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.multiworker import GlobalStepCoordinator, LeasedGradient
from orcacolony.participants import load_participants
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def submission_for(
    coordinator: GlobalStepCoordinator,
    assignment: dict[str, object],
) -> LeasedGradient:
    return LeasedGradient(
        assignment_id=str(assignment["assignment_id"]),
        lease_token=str(assignment["lease_token"]),
        checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        loss_sum=float(assignment["expected_loss_sum"]),
        loss_weight_sum=int(assignment["loss_weight_sum"]),
        safetensors=coordinator.oracle_gradient_path(
            str(assignment["assignment_id"])
        ).read_bytes(),
        runtime_backend="python-oracle-f32",
    )


def test_allowlist_is_default_deny_and_accepted_work_keeps_credit_private(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    manifest_path = tmp_path / "participants.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "orcacolony_participants_v1",
                "campaign_id": campaign.campaign["id"],
                "participants": [
                    {
                        "contributor_id": "owner-local",
                        "worker_ids": ["worker-a"],
                        "worker_token_sha256": {
                            "worker-a": hashlib.sha256(b"token-a").hexdigest()
                        },
                        "credit": {"public": True, "display_name": "Local Owner"},
                    },
                    {
                        "contributor_id": "trusted-private",
                        "worker_ids": ["worker-b"],
                        "worker_token_sha256": {
                            "worker-b": hashlib.sha256(b"token-b").hexdigest()
                        },
                        "credit": {"public": False, "display_name": "Private Name"},
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    participants = load_participants(
        manifest_path,
        campaign_id=str(campaign.campaign["id"]),
    )
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "state",
        worker_count=2,
        participants=participants,
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        coordinator.lease("unknown-worker", worker_token="unknown", now=100)
    with pytest.raises(ValueError, match="credential"):
        coordinator.lease("worker-a", worker_token="wrong", now=100)

    first = coordinator.lease("worker-a", worker_token="token-a", now=100)
    second = coordinator.lease("worker-b", worker_token="token-b", now=100)
    coordinator.accept(submission_for(coordinator, first), now=101)
    coordinator.accept(submission_for(coordinator, second), now=101)

    lock = json.loads(
        (coordinator.state_dir / "campaign-lock.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (coordinator.state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )

    assert lock["participants_revision"] == participants.revision
    assert ledger["entries"][0]["public_credit"] == {
        "display_name": "Local Owner"
    }
    assert all(
        entry["runtime_backend"] == "python-oracle-f32"
        for entry in ledger["entries"]
    )
    assert ledger["entries"][1]["public_credit"] is None
    assert ledger["entries"][1]["contributor_id"] == "trusted-private"
    assert GlobalStepCoordinator.load(
        campaign,
        coordinator.state_dir,
        participants=participants,
    ).status()["state"] == "step_complete"
