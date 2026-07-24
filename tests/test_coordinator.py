from pathlib import Path
import json
import threading
from urllib.request import Request, urlopen

from orcacolony.coordinator import (
    ConnectedCoordinator,
    SubmittedGradient,
    create_http_server,
)
from orcacolony.reference import load_campaign, run_training


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def test_one_submitted_gradient_matches_the_canonical_reference_step(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    coordinator = ConnectedCoordinator.create(campaign, tmp_path / "coordinator")
    assignment = coordinator.assignment

    accepted = coordinator.accept(
        SubmittedGradient(
            assignment_id=assignment["assignment_id"],
            checkpoint_sha256=assignment["checkpoint_sha256"],
            loss_sum=assignment["expected_loss_sum"],
            loss_weight_sum=assignment["loss_weight_sum"],
            safetensors=(coordinator.fixture_dir / "gradients.safetensors").read_bytes(),
        )
    )
    reference = run_training(campaign, tmp_path / "reference", target_steps=1)

    assert accepted.step == 1
    assert accepted.model_sha256 == reference.model_sha256
    assert accepted.gradient_metrics["relative_l2_error"] == 0
    assert accepted.checkpoint_metrics["relative_l2_error"] == 0
    assert (accepted.checkpoint_dir / "model.safetensors").is_file()
    assert coordinator.status()["state"] == "step_complete"


def test_http_assignment_and_result_complete_one_connected_step(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    coordinator = ConnectedCoordinator.create(campaign, tmp_path / "coordinator")
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    (browser_root / "index.html").write_text("OrcaColony", encoding="utf-8")
    server = create_http_server(coordinator, browser_root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base_url}/api/v1/assignment") as response:
            assignment = json.load(response)

        request = Request(
            f"{base_url}{assignment['result_url']}",
            data=(coordinator.fixture_dir / "gradients.safetensors").read_bytes(),
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Orca-Checkpoint-Sha256": assignment["checkpoint_sha256"],
                "X-Orca-Loss-Sum": str(assignment["expected_loss_sum"]),
                "X-Orca-Loss-Weight-Sum": str(assignment["loss_weight_sum"]),
            },
        )
        with urlopen(request) as response:
            receipt = json.load(response)
        with urlopen(f"{base_url}/api/v1/status") as response:
            status = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipt["accepted"] is True
    assert receipt["step"] == 1
    assert status["state"] == "step_complete"
    assert status["model_sha256"] == receipt["model_sha256"]
