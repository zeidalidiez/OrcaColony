import hashlib
import json
from dataclasses import replace
from pathlib import Path
import threading
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file

from orcacolony import multiworker
from orcacolony.multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    create_http_server,
    normalize_http_origin,
)
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"


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


def worker_telemetry(
    coordinator: GlobalStepCoordinator,
    assignment: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "orcacolony_worker_telemetry_v1",
        "runtime_seconds": {
            "assignment_fetch": 0.01,
            "runtime_init": 0.02,
            "artifact_fetch": 0.03,
            "gradient_compute": 0.5,
        },
        "transfer_bytes": {
            "assignment": 2048,
            "model": coordinator.initial_model_path.stat().st_size,
            "adapter": (
                coordinator.initial_adapter_path.stat().st_size
                if coordinator.lora is not None
                else 0
            ),
            "oracle_gradient": coordinator.oracle_gradient_path(
                str(assignment["assignment_id"])
            ).stat().st_size,
            "result": coordinator.oracle_gradient_path(
                str(assignment["assignment_id"])
            ).stat().st_size,
        },
        "memory_bytes": {
            "wasm_linear": 64 * 1024 * 1024,
            "process_peak_rss": None,
            "js_heap_used": 32 * 1024 * 1024,
            "js_heap_limit": 2 * 1024 * 1024 * 1024,
            "device_capacity": 8 * 1024 * 1024 * 1024,
        },
    }


def test_worker_resource_observations_are_validated_persisted_and_recovered(
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
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    resources = assignment["resource_profile"]
    assert resources["model_download_bytes"] == coordinator.initial_model_path.stat().st_size
    assert resources["adapter_download_bytes"] == 0
    assert resources["expected_result_upload_bytes"] == coordinator.oracle_gradient_path(
        str(assignment["assignment_id"])
    ).stat().st_size
    telemetry = worker_telemetry(coordinator, assignment)
    receipt = coordinator.accept(
        replace(submission_for(coordinator, assignment), worker_telemetry=telemetry),
        now=101,
        finalize=False,
    )

    assert receipt.instrumentation["worker_reported"] == telemetry
    measured = receipt.instrumentation["coordinator_measured"]
    assert measured["result_upload_bytes"] == len(
        submission_for(coordinator, assignment).safetensors
    )
    assert measured["result_storage_bytes"] == measured["result_upload_bytes"]
    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert ledger["entries"][0]["instrumentation"] == receipt.instrumentation

    recovered = GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    observations = recovered.status()["resource_observations"]
    assert observations["worker_reports"] == 1
    assert observations["runtime_seconds"]["gradient_compute"] == 0.5
    assert observations["transfer_bytes"]["result_upload"] == measured[
        "result_upload_bytes"
    ]
    assert observations["memory_bytes"]["peak_wasm_linear"] == 64 * 1024 * 1024
    assert "largest_device_capacity" not in observations["memory_bytes"]
    assert "largest_js_heap_limit" not in observations["memory_bytes"]
    assert observations["coordinator_storage_bytes"] > measured["result_storage_bytes"]

    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["assignments"][0]["instrumentation"]["worker_reported"][
        "runtime_seconds"
    ]["gradient_compute"] = -1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="worker runtime telemetry"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )


def test_burn_worker_telemetry_is_required_and_bound_to_assignment_bytes(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
    )
    assignment = coordinator.lease("worker-a", worker_token="test-token", now=100)
    burn_submission = replace(
        submission_for(coordinator, assignment),
        runtime_backend="burn-ndarray-f32",
    )
    with pytest.raises(ValueError, match="telemetry is required"):
        coordinator.accept(burn_submission, now=101)

    telemetry = worker_telemetry(coordinator, assignment)
    telemetry["transfer_bytes"]["result"] += 1
    with pytest.raises(ValueError, match="result does not match assignment"):
        coordinator.accept(
            replace(burn_submission, worker_telemetry=telemetry),
            now=101,
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
            assert "X-Orca-Worker-Telemetry" in response.headers[
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
                    "X-Orca-Worker-Telemetry": json.dumps(
                        worker_telemetry(coordinator, assignment),
                        separators=(",", ":"),
                    ),
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
    assert receipts[0]["instrumentation"]["worker_reported"]["format"] == (
        "orcacolony_worker_telemetry_v1"
    )
    assert receipts[0]["instrumentation"]["coordinator_measured"][
        "result_receive_seconds"
    ] >= 0
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


def test_lora_workers_aggregate_only_adapters_and_reload_the_checkpoint(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )

    expected_names = [
        name
        for layer in range(loaded.campaign.model.layers)
        for name in (
            f"blocks.{layer}.attention.qkv.lora_a",
            f"blocks.{layer}.attention.qkv.lora_b",
        )
    ]
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
            now=100,
        )
        assert assignment["format"] == "orcacolony_assignment_v2"
        assert assignment["training_method"] == "frozen-base-lora"
        assert assignment["base_model_sha256"] == loaded.config.base_model_sha256
        assert assignment["weight_checkpoint_sha256"] == assignment["checkpoint_sha256"]
        assert assignment["resume_state_sha256"] != assignment["checkpoint_sha256"]
        assert assignment["adapter"]["tensor_order"] == expected_names
        assert assignment["adapter"]["value_count"] == 8_192
        assert assignment["adapter_url"] == "/api/v1/artifacts/adapter.safetensors"
        receipt = coordinator.accept(submission_for(coordinator, assignment), now=101)

    assert receipt.step_complete is True
    assert receipt.model_sha256 == loaded.config.base_model_sha256
    assert receipt.adapter_sha256 is not None
    assert receipt.weight_checkpoint_sha256 is not None
    assert receipt.checkpoint_sha256 is not None
    assert receipt.weight_checkpoint_sha256 != receipt.checkpoint_sha256
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6

    checkpoint_state = json.loads(
        (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["format"] == "orcacolony_lora_checkpoint_v1"
    assert checkpoint_state["base_model_sha256"] == loaded.config.base_model_sha256
    assert checkpoint_state["adapter"]["tensor_sha256"] == receipt.adapter_sha256
    assert checkpoint_state["weight_checkpoint_sha256"] == receipt.weight_checkpoint_sha256
    assert checkpoint_state["checkpoint_sha256"] == receipt.checkpoint_sha256
    assert sorted(
        load_safetensors_file(str(coordinator.checkpoint_dir / "adapter.safetensors"))
    ) == expected_names
    assert all(
        name.startswith(("exp_avg.", "exp_avg_sq."))
        and name.split(".", 1)[1] in expected_names
        for name in load_safetensors_file(
            str(coordinator.checkpoint_dir / "optimizer.safetensors")
        )
    )

    recovered = GlobalStepCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
    )
    assert recovered.status()["state"] == "step_complete"
    assert recovered.status()["adapter_sha256"] == receipt.adapter_sha256
    assert recovered.status()["result_checkpoint_sha256"] == receipt.checkpoint_sha256


def test_next_lora_step_resumes_adapter_optimizer_and_dataset_cursor(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    second = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-2",
        worker_count=2,
        participants=participants,
        resume_from=first.checkpoint_dir,
        lora=loaded,
    )
    assert second.status()["step"] == 1
    assert second.status()["initial_adapter_sha256"] == first.status()["adapter_sha256"]
    for worker_id in ("worker-a", "worker-b"):
        assignment = second.lease(worker_id, worker_token="test-token", now=200)
        receipt = second.accept(submission_for(second, assignment), now=201)

    assert receipt.step == 2
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    checkpoint_state = json.loads(
        (second.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["step"] == 2
    assert checkpoint_state["optimizer_step"] == 2
    assert checkpoint_state["dataset_cursor"] == 8
    assert len(checkpoint_state["loss_history"]) == 2


def test_lora_resume_revalidates_artifact_paths_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    real_load = multiworker.load_lora_checkpoint

    def mutate_after_load(lora, checkpoint):
        result = real_load(lora, checkpoint)
        state_path = Path(checkpoint) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["optimizer"]["file"] = "../optimizer.safetensors"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return result

    monkeypatch.setattr(multiworker, "load_lora_checkpoint", mutate_after_load)
    with pytest.raises(ValueError, match="safe plain basename"):
        GlobalStepCoordinator.create(
            loaded.campaign,
            tmp_path / "step-2",
            worker_count=2,
            participants=participants,
            resume_from=first.checkpoint_dir,
            lora=loaded,
        )


def test_lora_http_contract_serves_assignments_artifacts_and_result_checkpoint(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
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
            assignment_request = Request(
                f"{base_url}/api/v1/assignment?{urlencode({'worker_id': worker_id})}",
                headers={"X-Orca-Worker-Token": "test-token"},
            )
            with urlopen(assignment_request) as response:
                assignment = json.load(response)
            assert assignment["model_url"] == "/api/v1/artifacts/model.safetensors"
            assert assignment["adapter_url"] == "/api/v1/artifacts/adapter.safetensors"
            with urlopen(f"{base_url}{assignment['model_url']}") as response:
                initial_base = load_safetensors(response.read())
            with urlopen(f"{base_url}{assignment['adapter_url']}") as response:
                initial_adapter = load_safetensors(response.read())
            assert multiworker.tensor_sha256(initial_base) == assignment["base_model_sha256"]
            assert multiworker.tensor_sha256(initial_adapter) == assignment["adapter_sha256"]

            result_request = Request(
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
                    "X-Orca-Worker-Telemetry": json.dumps(
                        worker_telemetry(coordinator, assignment),
                        separators=(",", ":"),
                    ),
                },
            )
            with urlopen(result_request) as response:
                receipts.append(json.load(response))

        completed = receipts[-1]
        with urlopen(f"{base_url}{completed['checkpoint_url']}") as response:
            completed_adapter = load_safetensors(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipts[0]["step_complete"] is False
    assert receipts[0]["adapter_sha256"] is None
    assert receipts[0]["checkpoint_sha256"] is None
    assert receipts[0]["checkpoint_url"] is None
    assert completed["step_complete"] is True
    assert completed["model_sha256"] == loaded.config.base_model_sha256
    assert completed["adapter_sha256"] == multiworker.tensor_sha256(completed_adapter)
    assert completed[
        "weight_checkpoint_sha256"
    ] == multiworker.lora_weight_checkpoint_sha256(
        loaded,
        completed["adapter_sha256"],
    )
    checkpoint_state = json.loads(
        (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert completed["checkpoint_sha256"] == checkpoint_state["checkpoint_sha256"]
    assert completed["checkpoint_sha256"] != completed["weight_checkpoint_sha256"]
    assert completed["checkpoint_url"] == "/api/v1/checkpoint/adapter.safetensors"


def test_dense_restart_migrates_the_pre_lora_state_and_campaign_lock(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )

    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in (
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "initial_adapter_sha256",
        "resume_state_sha256",
        "adapter_sha256",
        "result_weight_checkpoint_sha256",
        "result_checkpoint_sha256",
    ):
        state.pop(field)
    for assignment in state["assignments"]:
        for field in (
            "training_method",
            "lora_manifest_sha256",
            "base_model_sha256",
            "adapter_sha256",
            "adapter",
        ):
            assignment.pop(field, None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for field in (
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "adapter_sha256",
        "resume_state_sha256",
    ):
        lock.pop(field)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    recovered = GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    assert recovered.status()["training_method"] == "dense"
    migrated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated_state["base_model_sha256"] == migrated_state["checkpoint_sha256"]
    assert migrated_state["result_checkpoint_sha256"] is None
    migrated_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert migrated_lock["training_method"] == "dense"
    assert migrated_lock["adapter_sha256"] is None
