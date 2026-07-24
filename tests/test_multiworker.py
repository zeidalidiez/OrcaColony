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
)
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
    )


def test_two_non_overlapping_workers_match_one_reference_global_step(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        lease_seconds=60,
    )
    first = coordinator.lease("worker-a", now=100)
    second = coordinator.lease("worker-b", now=100)

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
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        lease_seconds=10,
    )
    expired = coordinator.lease("worker-a", now=100)
    replacement = coordinator.lease("worker-b", now=111)
    other = coordinator.lease("worker-c", now=111)

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

    recovered = GlobalStepCoordinator.load(campaign, state_dir)

    assert recovered.status()["state"] == "step_complete"
    assert recovered.status()["checkpoint_metrics"]["relative_l2_error"] < 1e-6
    with pytest.raises(ValueError, match="already accepted"):
        recovered.accept(submission_for(coordinator, replacement), now=113)

    (recovered.checkpoint_dir / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        GlobalStepCoordinator.load(campaign, state_dir)


def test_http_leases_two_workers_and_closes_the_global_step(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
    )
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    (browser_root / "index.html").write_text("OrcaColony", encoding="utf-8")
    server = create_http_server(coordinator, browser_root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    receipts = []
    try:
        for worker_id in ("browser-a", "browser-b"):
            query = urlencode({"worker_id": worker_id})
            with urlopen(f"{base_url}/api/v1/assignment?{query}") as response:
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
