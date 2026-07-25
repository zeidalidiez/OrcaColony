from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from orcacolony.artifacts import PackedDataset
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import create_http_server
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest


REPOSITORY = Path(__file__).resolve().parents[4]
BROWSER_ROOT = REPOSITORY / "spikes" / "burn-browser-gradient" / "www"
PROFILES = (
    (
        "t1-real-data-native-cache",
        REPOSITORY / "campaign" / "t1-tinystories-smoke.json",
        REPOSITORY / "campaign" / "t1-tinystories-lora-smoke.json",
    ),
    (
        "t2-91m-native-cache",
        REPOSITORY / "campaign" / "t2-tinystories-memory-smoke.json",
        REPOSITORY / "campaign" / "t2-tinystories-memory-lora-smoke.json",
    ),
)


def _public_worker_result(result: dict[str, object]) -> dict[str, object]:
    telemetry = result["telemetry"]
    memory = telemetry["memory_bytes"]
    receipt = {
        field: value
        for field, value in result["receipt"].items()
        if field != "instrumentation"
    }
    return {
        "assignment_id": result["assignment_id"],
        "telemetry": {
            "format": telemetry["format"],
            "runtime_seconds": telemetry["runtime_seconds"],
            "transfer_bytes": telemetry["transfer_bytes"],
            "memory_bytes": {
                "wasm_linear": memory["wasm_linear"],
                "process_peak_rss": memory["process_peak_rss"],
                "js_heap_used": memory["js_heap_used"],
            },
        },
        "receipt": receipt,
    }


def _run_worker_process(
    *,
    base_url: str,
    worker_id: str,
    token_file: Path,
    campaign_path: Path,
    lora_path: Path,
    cache_dir: Path,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "orcacolony.native_worker",
            "--coordinator",
            base_url,
            "--worker-id",
            worker_id,
            "--worker-token-file",
            str(token_file),
            "--config",
            str(campaign_path),
            "--lora-config",
            str(lora_path),
            "--cache",
            str(cache_dir),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("native worker output must be a JSON object")
    return payload


def _run_profile(
    profile_id: str,
    campaign_path: Path,
    lora_path: Path,
    dataset: PackedDataset,
    output: Path,
) -> dict[str, object]:
    loaded = load_lora_manifest(campaign_path, lora_path)
    token = secrets.token_urlsafe(32)
    worker_ids = (f"{profile_id}-a", f"{profile_id}-b")
    participants = ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": loaded.campaign.campaign["id"],
            "participants": [
                {
                    "contributor_id": "local-profile-proof",
                    "worker_ids": list(worker_ids),
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
    state_dir = output / profile_id / "state"
    cache_dir = output / profile_id / "cache"
    coordinator = CampaignCoordinator.create(
        loaded.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        lease_seconds=600,
        dataset=dataset,
        lora=loaded,
    )
    server = create_http_server(
        coordinator,  # type: ignore[arg-type]
        BROWSER_ROOT,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    token_file = output / profile_id / "worker-token.txt"
    token_file.write_text(token + "\n", encoding="utf-8", newline="\n")
    try:
        results = [
            _run_worker_process(
                base_url=base_url,
                worker_id=worker_id,
                token_file=token_file,
                campaign_path=campaign_path,
                lora_path=lora_path,
                cache_dir=cache_dir,
            )
            for worker_id in worker_ids
        ]
        dashboard = coordinator.dashboard()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        token_file.unlink(missing_ok=True)

    first, second = results
    if first["telemetry"]["transfer_bytes"]["model"] <= 0:
        raise RuntimeError(f"{profile_id} cold worker did not fetch the base")
    if second["telemetry"]["transfer_bytes"]["model"] != 0:
        raise RuntimeError(f"{profile_id} warm worker refetched the base")
    if second["telemetry"]["transfer_bytes"]["adapter"] != 0:
        raise RuntimeError(f"{profile_id} warm worker refetched the adapter")
    resources = dashboard["resource_observations"]
    if resources["worker_reports"] != 2:
        raise RuntimeError(f"{profile_id} telemetry report count mismatch")
    if any(
        field in resources["memory_bytes"]
        for field in ("largest_device_capacity", "largest_js_heap_limit")
    ):
        raise RuntimeError(f"{profile_id} public dashboard exposed hardware capacity")
    if second["receipt"]["checkpoint_metrics"]["relative_l2_error"] >= 1e-6:
        raise RuntimeError(f"{profile_id} checkpoint parity exceeded the guardrail")
    evaluations = dashboard["evaluations"]
    initialization_loss = float(evaluations[0]["mean_loss"])
    step_loss = float(evaluations[-1]["mean_loss"])
    if step_loss > initialization_loss:
        raise RuntimeError(f"{profile_id} held-out loss regressed")
    return {
        "profile_id": profile_id,
        "campaign_id": loaded.campaign.campaign["id"],
        "parameter_count": loaded.campaign.model.parameters,
        "trainable_value_count": second["receipt"]["checkpoint_metrics"]["value_count"],
        "cold_worker": _public_worker_result(first),
        "warm_worker": _public_worker_result(second),
        "resource_observations": resources,
        "checkpoint_metrics": second["receipt"]["checkpoint_metrics"],
        "evaluations": evaluations,
        "held_out_mean_loss_improvement": initialization_loss - step_loss,
        "warm_base_payload_avoidance_ratio": 1.0,
        "warm_adapter_payload_avoidance_ratio": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the P3 native cached-base T1/T2 resource profiles."
    )
    parser.add_argument(
        "--dataset-artifacts",
        type=Path,
        required=True,
        help="Frozen TinyStories artifact directory matching t1-tinystories-smoke.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / ".artifacts" / "p3-native-resource-profile-proof",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output path already exists: {args.output}")
    dataset = PackedDataset.load(args.dataset_artifacts)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise SystemExit(f"temporary output path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        profiles = [
            _run_profile(profile_id, campaign_path, lora_path, dataset, temporary)
            for profile_id, campaign_path, lora_path in PROFILES
        ]
        summary = {
            "format": "orcacolony_p3_native_resource_profile_proof_v1",
            "implementation_commit": "673017df9caa4d91f6bff96a39f56a40690c71e9",
            "dataset_revision": dataset.revision,
            "profiles": profiles,
        }
        (temporary / "proof-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(args.output / "proof-summary.json")


if __name__ == "__main__":
    main()
