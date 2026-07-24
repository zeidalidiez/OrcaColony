import hashlib
import json
from pathlib import Path

from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import LeasedGradient
from orcacolony.participants import ParticipantRegistry
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def participants_for(campaign_id: object) -> ParticipantRegistry:
    worker_ids = ["worker-a", "worker-b"]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "campaign-test",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(b"test-token").hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=str(campaign_id),
    )


def submission_for(
    coordinator: CampaignCoordinator,
    assignment: dict[str, object],
) -> LeasedGradient:
    assignment_id = str(assignment["assignment_id"])
    return LeasedGradient(
        assignment_id=assignment_id,
        lease_token=str(assignment["lease_token"]),
        checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        loss_sum=float(assignment["expected_loss_sum"]),
        loss_weight_sum=int(assignment["loss_weight_sum"]),
        safetensors=coordinator.oracle_gradient_path(assignment_id).read_bytes(),
        runtime_backend="python-oracle-f32",
    )


def test_campaign_advances_two_global_steps_and_versions_every_checkpoint(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
    )

    for expected_step in (1, 2):
        for worker_id in ("worker-a", "worker-b"):
            assignment = coordinator.lease(
                worker_id,
                worker_token="test-token",
                now=expected_step * 100,
            )
            coordinator.accept(
                submission_for(coordinator, assignment),
                now=expected_step * 100 + 1,
            )
        assert (state_dir / "checkpoints" / f"step-{expected_step:08d}").is_dir()

    status = coordinator.status()
    assert status["state"] == "campaign_complete"
    assert status["completed_steps"] == 2
    assert status["target_steps"] == 2
    assert status["checkpoint_metrics"]["relative_l2_error"] < 1e-6

    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert len(ledger["entries"]) == 4
    assert [entry["checkpoint_step"] for entry in ledger["entries"]] == [1, 1, 2, 2]

    recovered = CampaignCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    assert recovered.status()["state"] == "campaign_complete"
    assert recovered.status()["completed_steps"] == 2
