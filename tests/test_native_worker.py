import hashlib
from pathlib import Path
import threading

import pytest

from orcacolony.multiworker import GlobalStepCoordinator, create_http_server
from orcacolony.native_worker import _prepare_cache_directory, run_assignment
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"
BROWSER_ROOT = (
    Path(__file__).parents[1] / "spikes" / "burn-browser-gradient" / "www"
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
