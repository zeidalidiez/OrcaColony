import hashlib
import json
from dataclasses import replace
from pathlib import Path

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
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
    corpus = (
        "A small fox found a red ball and shared it with a bird.\n"
        "<|endoftext|>\n"
        "A little boat crossed the pond and came safely home.\n"
        "<|endoftext|>\n"
    ) * 200
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus[: len(corpus) // 2].encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/campaign-stories",
            "revision": "test-revision",
            "license": "cdla-sharing-1.0",
            "internal_note": "must-not-be-public",
        },
        vocab_size=300,
        context_length=campaign.model.context_length,
    )
    dataset = PackedDataset.load(artifact_dir)
    campaign = replace(
        campaign,
        dataset={
            "format": manifest["format"],
            "manifest_sha256": dataset.revision,
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "train_sha256": manifest["files"]["train.safetensors"],
            "validation_sha256": manifest["files"]["validation.safetensors"],
        },
        evaluation={
            "validation_sequences": 4,
            "batch_size": 2,
            "success_gate": {
                "metric": "mean_loss",
                "minimum_improvement_from_initialization": 100.0,
            },
        },
    )
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        dataset=dataset,
    )
    assert (state_dir / "checkpoints" / "step-00000000").is_dir()

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
            if expected_step == 1 and worker_id == "worker-a":
                live_dashboard = coordinator.dashboard()
                assert live_dashboard["progress"]["accepted_assignments"] == 1
                assert live_dashboard["public_ledger"][0]["checkpoint_step"] == 1
        assert (state_dir / "checkpoints" / f"step-{expected_step:08d}").is_dir()

    status = coordinator.status()
    assert status["state"] == "campaign_complete"
    assert status["completed_steps"] == 2
    assert status["target_steps"] == 2
    assert status["evaluation_gate"]["state"] == "failed"
    assert status["checkpoint_metrics"]["relative_l2_error"] < 1e-6

    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert len(ledger["entries"]) == 4
    assert ledger["dataset_revision"] == dataset.revision
    assert [entry["checkpoint_step"] for entry in ledger["entries"]] == [1, 1, 2, 2]
    evaluations = json.loads(
        (state_dir / "evaluations.json").read_text(encoding="utf-8")
    )
    assert [entry["step"] for entry in evaluations["entries"]] == [0, 1, 2]
    assert all(entry["mean_loss"] > 0 for entry in evaluations["entries"])
    assert all(
        entry["dataset_revision"] == dataset.revision
        for entry in evaluations["entries"]
    )
    dashboard = coordinator.dashboard()
    assert dashboard["progress"]["completed_steps"] == 2
    assert dashboard["progress"]["accepted_assignments"] == 4
    assert dashboard["progress"]["accepted_tokens"] == sum(
        entry["loss_weight_sum"] for entry in ledger["entries"]
    )
    assert dashboard["contributors"] == {
        "active_count": 1,
        "anonymous_count": 1,
        "acknowledgements": [],
    }
    assert len(dashboard["public_ledger"]) == 4
    serialized_dashboard = json.dumps(dashboard)
    assert "campaign-test" not in serialized_dashboard
    assert "worker-a" not in serialized_dashboard
    assert "worker-b" not in serialized_dashboard
    assert dashboard["dataset"]["source"]["dataset"] == "test/campaign-stories"
    assert "internal_note" not in dashboard["dataset"]["source"]

    recovered = CampaignCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
        dataset=dataset,
    )
    assert recovered.status()["state"] == "campaign_complete"
    assert recovered.status()["completed_steps"] == 2
