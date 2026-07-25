import copy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import GlobalStepCoordinator, create_http_server
from orcacolony.native_worker import (
    NativeWorkerSession,
    _prepare_cache_directory,
    _validate_assignment,
    run_assignment,
)
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"
BROWSER_ROOT = (
    Path(__file__).parents[1] / "spikes" / "burn-browser-gradient" / "www"
)


def _single_contributor_registry(
    campaign_id: str,
    worker_ids: list[str],
    token: str,
) -> ParticipantRegistry:
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "layer-bundle-security-test",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(token.encode("utf-8")).hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=campaign_id,
    )


def test_native_cache_rejects_a_symlinked_managed_directory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    victim = tmp_path / "victim"
    cache.mkdir()
    victim.mkdir()
    sentinel = victim / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    try:
        (cache / "model").symlink_to(victim, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="symlink or reparse point"):
        _prepare_cache_directory(cache, "model")

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_native_cpu_worker_reuses_content_addressed_base_and_adapter_cache(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "native-worker-test-token"
    worker_ids = ["native-a", "native-b"]
    participants = ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": loaded.campaign.campaign["id"],
            "participants": [
                {
                    "contributor_id": "native-test",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(token.encode("utf-8")).hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=str(loaded.campaign.campaign["id"]),
    )
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    coordinator_url = f"http://127.0.0.1:{server.server_port}"
    cache_dir = tmp_path / "cache"
    try:
        first = run_assignment(
            coordinator_url=coordinator_url,
            worker_id="native-a",
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=cache_dir,
        )
        second = run_assignment(
            coordinator_url=coordinator_url,
            worker_id="native-b",
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=cache_dir,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    first_transfer = first.telemetry["transfer_bytes"]
    second_transfer = second.telemetry["transfer_bytes"]
    assert first_transfer["model"] > 0
    assert first_transfer["adapter"] > 0
    assert first_transfer["oracle_gradient"] == 0
    assert second_transfer["model"] == 0
    assert second_transfer["adapter"] == 0
    assert second_transfer["oracle_gradient"] == 0
    assert first.telemetry["memory_bytes"]["process_peak_rss"] > 0
    assert second.receipt["step_complete"] is True
    assert second.receipt["checkpoint_metrics"]["relative_l2_error"] < 1e-6

    observations = coordinator.status()["resource_observations"]
    assert observations["worker_reports"] == 2
    assert observations["transfer_bytes"]["model_download"] == first_transfer["model"]
    assert observations["transfer_bytes"]["adapter_download"] == first_transfer["adapter"]
    assert observations["transfer_bytes"]["oracle_gradient_download"] == 0
    assert observations["memory_bytes"]["peak_process_rss"] > 0
    assert len(list((cache_dir / "model").glob("*.safetensors"))) == 1
    assert len(list((cache_dir / "adapter").glob("*.safetensors"))) == 1


def test_connected_layer_bundle_worker_reuses_exact_shards_across_restart(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "layer-bundle-native-worker-test-token"
    worker_ids = ["bundle-a", "bundle-b"]
    participants = ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": loaded.campaign.campaign["id"],
            "participants": [
                {
                    "contributor_id": "layer-bundle-native-test",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(token.encode("utf-8")).hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=str(loaded.campaign.campaign["id"]),
    )
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
        publish_base_layer_bundle=True,
    )
    assignment = coordinator.lease("bundle-a", worker_token=token)
    bundle_contract = assignment["base_layer_bundle"]
    assert bundle_contract["format"] == "orcacolony_assignment_base_layer_bundle_v1"
    assert bundle_contract["profile"] == "layer-bundle-streamed-fp32-v1"
    assert bundle_contract["base_model_sha256"] == loaded.config.base_model_sha256
    assert len(bundle_contract["artifacts"]) == 18
    expected_bundle_bytes = assignment["resource_profile"][
        "layer_bundle_download_bytes"
    ]
    cache_dir = tmp_path / "cache"

    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = run_assignment(
            coordinator_url=f"http://127.0.0.1:{server.server_port}",
            worker_id="bundle-a",
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=cache_dir,
            base_profile="layer-bundle",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    coordinator = GlobalStepCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
    )
    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        second = run_assignment(
            coordinator_url=f"http://127.0.0.1:{server.server_port}",
            worker_id="bundle-b",
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=cache_dir,
            base_profile="layer-bundle",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert first.telemetry["transfer_bytes"]["model"] == expected_bundle_bytes
    assert first.telemetry["transfer_bytes"]["adapter"] > 0
    assert second.telemetry["transfer_bytes"]["model"] == 0
    assert second.telemetry["transfer_bytes"]["adapter"] == 0
    assert second.receipt["step_complete"] is True
    assert second.receipt["checkpoint_metrics"]["relative_l2_error"] < 1e-6
    assert not (cache_dir / "model").exists()
    cached_bundle = cache_dir / "bundle" / bundle_contract["manifest_sha256"]
    assert len(list(cached_bundle.iterdir())) == 18


def test_layer_bundle_assignment_binds_each_artifact_url(tmp_path: Path) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "url-binding-token"
    participants = _single_contributor_registry(
        str(loaded.campaign.campaign["id"]),
        ["worker-0", "worker-1"],
        token,
    )
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
        publish_base_layer_bundle=True,
    )
    assignment = coordinator.lease("worker-0", worker_token=token)
    tampered = copy.deepcopy(assignment)
    tampered["base_layer_bundle"]["artifacts"][2]["url"] = (
        "/api/v1/artifacts/base-layer-bundle/resident.safetensors"
    )

    with pytest.raises(ValueError, match="artifact URL differs"):
        _validate_assignment(tampered, loaded, "layer-bundle")


def test_layer_bundle_publication_and_fresh_cache_reject_raw_mutation(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "raw-mutation-token"
    participants = _single_contributor_registry(
        str(loaded.campaign.campaign["id"]),
        ["worker-0", "worker-1"],
        token,
    )
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
        publish_base_layer_bundle=True,
    )
    assignment = coordinator.lease("worker-0", worker_token=token)
    artifact = assignment["base_layer_bundle"]["artifacts"][2]
    artifact_path = coordinator.base_layer_bundle_artifact_path(artifact["file"])
    mutated = bytearray(artifact_path.read_bytes())
    mutated[-1] ^= 1
    artifact_path.write_bytes(mutated)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            tmp_path / "coordinator",
            participants=participants,
            lora=loaded,
        )

    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            run_assignment(
                coordinator_url=f"http://127.0.0.1:{server.server_port}",
                worker_id="worker-0",
                worker_token=token,
                campaign_path=CONFIG,
                lora_path=LORA_CONFIG,
                cache_dir=tmp_path / "cache",
                base_profile="layer-bundle",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    cache_bundle_root = tmp_path / "cache" / "bundle"
    assert not any(
        path.name == artifact["file"]
        for path in cache_bundle_root.rglob("*.safetensors")
    )


def test_warm_layer_bundle_cache_reauthenticates_each_linear_on_use(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "warm-mutation-token"
    participants = _single_contributor_registry(
        str(loaded.campaign.campaign["id"]),
        ["warm-a", "warm-b"],
        token,
    )
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
        publish_base_layer_bundle=True,
    )
    cache_dir = tmp_path / "cache"
    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        first = run_assignment(
            coordinator_url=f"http://127.0.0.1:{server.server_port}",
            worker_id="warm-a",
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=cache_dir,
            base_profile="layer-bundle",
        )
        bundle_dir = next((cache_dir / "bundle").iterdir())
        linear = bundle_dir / "linear-00000.safetensors"
        mutated = bytearray(linear.read_bytes())
        mutated[-1] ^= 1
        linear.write_bytes(mutated)

        with pytest.raises(ValueError, match="tensor digest mismatch"):
            run_assignment(
                coordinator_url=f"http://127.0.0.1:{server.server_port}",
                worker_id="warm-b",
                worker_token=token,
                campaign_path=CONFIG,
                lora_path=LORA_CONFIG,
                cache_dir=cache_dir,
                base_profile="layer-bundle",
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert first.receipt["accepted"] is True
    assert coordinator.status()["resource_observations"]["accepted_assignments"] == 1


def test_connected_layer_bundle_t1_evidence_is_exact_and_evaluated() -> None:
    evidence_path = (
        Path(__file__).parents[1]
        / "spikes"
        / "layer-bundle-fp32"
        / "results"
        / "connected-t1.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["format"] == "orcacolony_connected_layer_bundle_proof_v1"
    assert evidence["coordinator_restart_between_assignments"] is True
    assert evidence["cache_contains_monolithic_base"] is False
    assert evidence["campaign_state"] == "campaign_complete"
    assert evidence["assignments"][0]["model_transfer_bytes"] == evidence[
        "bundle_download_bytes"
    ]
    assert evidence["assignments"][1]["model_transfer_bytes"] == 0
    assert evidence["assignments"][1]["adapter_transfer_bytes"] == 0
    assert all(
        assignment["gradient_relative_l2_error"] == 0.0
        and assignment["gradient_max_absolute_error"] == 0.0
        for assignment in evidence["assignments"]
    )
    assert evidence["checkpoint"]["relative_l2_error"] < 1e-6
    assert evidence["held_out_evaluation"]["step_1"] < evidence[
        "held_out_evaluation"
    ]["initialization"]


def test_mixed_exact_profile_t2_evidence_is_qualified_and_evaluated() -> None:
    evidence_path = (
        Path(__file__).parents[1]
        / "spikes"
        / "layer-bundle-fp32"
        / "results"
        / "mixed-t2.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    resident, bundle = evidence["profiles"]

    assert evidence["format"] == "orcacolony_mixed_exact_profiles_t2_proof_v1"
    assert evidence["mixed_exact_profiles_qualified"] is True
    assert evidence["coordinator_restart_between_assignments"] is True
    assert evidence["campaign_state"] == "campaign_complete"
    assert resident["runtime_backend"] == "python-native-cpu-f32"
    assert bundle["runtime_backend"] == "python-native-cpu-layer-bundle-f32"
    assert resident["cache_contains_monolithic_base"] is True
    assert bundle["cache_contains_monolithic_base"] is False
    assert bundle["process_peak_rss_bytes"] < resident["process_peak_rss_bytes"]
    assert all(
        profile["gradient_relative_l2_error"] == 0.0
        and profile["gradient_max_absolute_error"] == 0.0
        for profile in evidence["profiles"]
    )
    assert evidence["checkpoint"]["relative_l2_error"] < 1e-6
    assert evidence["held_out_evaluation"]["step_1"] < evidence[
        "held_out_evaluation"
    ]["initialization"]


@pytest.mark.parametrize(
    ("base_profile", "publish_base_layer_bundle"),
    (("resident", False), ("layer-bundle", True)),
)
def test_persistent_native_session_reuses_model_and_refreshes_adapter_across_steps(
    tmp_path: Path,
    base_profile: str,
    publish_base_layer_bundle: bool,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    token = "persistent-native-worker-test-token"
    worker_id = "persistent-native"
    participants = ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": loaded.campaign.campaign["id"],
            "participants": [
                {
                    "contributor_id": "persistent-native-test",
                    "worker_ids": [worker_id],
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(token.encode("utf-8")).hexdigest()
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=str(loaded.campaign.campaign["id"]),
    )
    coordinator = CampaignCoordinator.create(
        loaded.campaign,
        tmp_path / f"campaign-{base_profile}",
        participants=participants,
        worker_count=2,
        target_steps=2,
        lora=loaded,
        publish_base_layer_bundle=publish_base_layer_bundle,
    )
    server = create_http_server(coordinator, BROWSER_ROOT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        session = NativeWorkerSession(
            coordinator_url=f"http://127.0.0.1:{server.server_port}",
            worker_id=worker_id,
            worker_token=token,
            campaign_path=CONFIG,
            lora_path=LORA_CONFIG,
            cache_dir=tmp_path / f"persistent-cache-{base_profile}",
            base_profile=base_profile,
        )
        results = [session.run_assignment() for _ in range(4)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert [result.reused_model for result in results] == [False, True, True, True]
    assert [result.reused_adapter for result in results] == [False, True, False, True]
    assert session.model_build_count == 1
    assert session.adapter_load_count == 2
    assert [result.telemetry["transfer_bytes"]["model"] for result in results] == [
        results[0].telemetry["transfer_bytes"]["model"],
        0,
        0,
        0,
    ]
    assert results[0].telemetry["transfer_bytes"]["model"] > 0
    assert results[0].telemetry["transfer_bytes"]["adapter"] > 0
    assert results[1].telemetry["transfer_bytes"]["adapter"] == 0
    assert results[2].telemetry["transfer_bytes"]["adapter"] > 0
    assert results[3].telemetry["transfer_bytes"]["adapter"] == 0
    assert results[-1].receipt["step_complete"] is True
    assert results[-1].receipt["step"] == 2
    assert results[-1].receipt["checkpoint_metrics"]["relative_l2_error"] < 1e-6
    assert coordinator.status()["state"] == "campaign_complete"
    assert coordinator.dashboard()["progress"]["accepted_assignments"] == 4
