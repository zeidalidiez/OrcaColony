import hashlib
import json
from dataclasses import replace
from pathlib import Path
import threading
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from orcacolony.multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    create_http_server,
    normalize_http_origin,
)
from orcacolony.participants import ParticipantRegistry
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def participants_for(campaign_id: object) -> ParticipantRegistry:
    worker_ids = [
        "browser-a",
        "browser-b",
        "worker-a",
        "worker-b",
        "worker-c",
    ]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "test-contributor",
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


def test_two_non_overlapping_workers_match_one_reference_global_step(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lease_seconds=60,
    )
    first = coordinator.lease("worker-a", worker_token="test-token", now=100)
    second = coordinator.lease("worker-b", worker_token="test-token", now=100)

    assert first["data_range"] == [0, 2]
    assert second["data_range"] == [2, 4]
    assert set(range(*first["data_range"])).isdisjoint(range(*second["data_range"]))

    first_submission = submission_for(coordinator, first)
    first_submission = replace(
        first_submission,
        loss_sum=first_submission.loss_sum + 0.001,
    )
    first_receipt = coordinator.accept(first_submission, now=101)
    final_receipt = coordinator.accept(submission_for(coordinator, second), now=101)

    assert first_receipt.step_complete is False
    assert final_receipt.step_complete is True
    assert final_receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    assert final_receipt.checkpoint_metrics["cosine_similarity"] > 1 - 1e-10
    assert coordinator.status()["state"] == "step_complete"
    assert coordinator.status()["loss_sum"] == pytest.approx(
        first_submission.loss_sum + float(second["expected_loss_sum"])
    )


def test_expired_attempt_is_rejected_and_accepted_work_replays_after_restart(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lease_seconds=10,
    )
    expired = coordinator.lease("worker-a", worker_token="test-token", now=100)
    replacement = coordinator.lease(
        "worker-b", worker_token="test-token", now=111
    )
    other = coordinator.lease("worker-c", worker_token="test-token", now=111)

    assert replacement["assignment_id"] == expired["assignment_id"]
    assert replacement["attempt"] == 2
    with pytest.raises(ValueError, match="stale lease"):
        coordinator.accept(submission_for(coordinator, expired), now=112)

    coordinator.accept(
        submission_for(coordinator, replacement),
        now=112,
        finalize=False,
    )
    coordinator.accept(
        submission_for(coordinator, other),
        now=112,
        finalize=False,
    )
    assert coordinator.status()["state"] == "ready_to_finalize"

    recovered = GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )

    assert recovered.status()["state"] == "step_complete"
    assert recovered.status()["checkpoint_metrics"]["relative_l2_error"] < 1e-6
    with pytest.raises(ValueError, match="already accepted"):
        recovered.accept(submission_for(coordinator, replacement), now=113)

    (recovered.checkpoint_dir / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        GlobalStepCoordinator.load(campaign, state_dir, participants=participants)


def test_http_leases_two_workers_and_closes_the_global_step(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
    )
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    (browser_root / "index.html").write_text("OrcaColony", encoding="utf-8")
    public_origin = "https://workers.example"
    server = create_http_server(
        coordinator,
        browser_root,
        port=0,
        public_origin=public_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    receipts = []
    try:
        preflight = Request(
            f"{base_url}/api/v1/assignment",
            method="OPTIONS",
            headers={
                "Origin": public_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Orca-Worker-Token",
            },
        )
        with urlopen(preflight) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == public_origin
            assert "X-Orca-Worker-Token" in response.headers[
                "Access-Control-Allow-Headers"
            ]
        for worker_id in ("browser-a", "browser-b"):
            query = urlencode({"worker_id": worker_id})
            assignment_request = Request(
                f"{base_url}/api/v1/assignment?{query}",
                headers={"X-Orca-Worker-Token": "test-token"},
            )
            with urlopen(assignment_request) as response:
                assignment = json.load(response)
            request = Request(
                f"{base_url}{assignment['result_url']}",
                data=coordinator.oracle_gradient_path(
                    assignment["assignment_id"]
                ).read_bytes(),
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Orca-Lease-Token": assignment["lease_token"],
                    "X-Orca-Checkpoint-Sha256": assignment["checkpoint_sha256"],
                    "X-Orca-Loss-Sum": str(assignment["expected_loss_sum"]),
                    "X-Orca-Loss-Weight-Sum": str(assignment["loss_weight_sum"]),
                    "X-Orca-Runtime-Backend": "python-oracle-f32",
                },
            )
            with urlopen(request) as response:
                receipts.append(json.load(response))
        with urlopen(f"{base_url}/api/v1/status") as response:
            status = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipts[0]["step_complete"] is False
    assert receipts[1]["step_complete"] is True
    assert status["state"] == "step_complete"
    assert "browser-a" not in json.dumps(status, sort_keys=True)
    assert "browser-b" not in json.dumps(status, sort_keys=True)


def test_public_origin_is_canonical_and_rejects_credentials() -> None:
    assert normalize_http_origin("https://Workers.Example:443/") == "https://workers.example"
    assert normalize_http_origin("http://[::1]:8000") == "http://[::1]:8000"
    with pytest.raises(ValueError, match="without a path"):
        normalize_http_origin("https://user:password@workers.example")
    with pytest.raises(ValueError, match="HTTPS except on loopback"):
        normalize_http_origin("http://workers.example")
    with pytest.raises(ValueError, match="invalid characters"):
        normalize_http_origin('https://workers.example" onload="alert(1)')
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://workers.example%22x")
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://[fe80::1%25eth0]")
    with pytest.raises(ValueError, match="invalid characters"):
        normalize_http_origin("https://workers.example\nevil")
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://127.1")


def test_next_global_step_resumes_model_optimizer_and_dataset_cursor(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    second = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "step-2",
        worker_count=2,
        participants=participants,
        resume_from=first.checkpoint_dir,
    )
    second_a = second.lease("worker-a", worker_token="test-token", now=200)
    second_b = second.lease("worker-b", worker_token="test-token", now=200)

    assert second_a["global_step"] == 1
    assert second_a["data_range"] == [4, 6]
    assert second_b["data_range"] == [6, 8]

    second.accept(submission_for(second, second_a), now=201)
    receipt = second.accept(submission_for(second, second_b), now=201)
    checkpoint_state = json.loads(
        (second.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )

    assert receipt.step_complete is True
    assert receipt.step == 2
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    assert checkpoint_state["step"] == 2
    assert checkpoint_state["dataset_cursor"] == 8
    assert len(checkpoint_state["loss_history"]) == 2
