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
CAMPAIGN = REPOSITORY / "campaign" / "t2-tinystories-memory-smoke.json"
LORA = REPOSITORY / "campaign" / "t2-tinystories-memory-lora-smoke.json"
BROWSER_ROOT = REPOSITORY / "spikes" / "burn-browser-gradient" / "www"
ONE_SHOT_WARM_SETUP_SECONDS = 2.9631700000027195


def _sanitized_result(result: dict[str, object]) -> dict[str, object]:
    telemetry = result["telemetry"]
    memory = telemetry["memory_bytes"]
    receipt = {
        field: value
        for field, value in result["receipt"].items()
        if field != "instrumentation"
    }
    return {
        "assignment_id": result["assignment_id"],
        "receipt": receipt,
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
        "reused_model": result["reused_model"],
        "reused_adapter": result["reused_adapter"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the P3 persistent native T2 session proof."
    )
    parser.add_argument("--dataset-artifacts", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / ".artifacts" / "p3-persistent-native-session-proof",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output path already exists: {args.output}")
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise SystemExit(f"temporary output path already exists: {temporary}")
    temporary.mkdir(parents=True)
    token_file = temporary / "worker-token.txt"
    server = None
    thread = None
    try:
        dataset = PackedDataset.load(args.dataset_artifacts)
        loaded = load_lora_manifest(CAMPAIGN, LORA)
        token = secrets.token_urlsafe(32)
        worker_id = "persistent-t2-native"
        participants = ParticipantRegistry.from_payload(
            {
                "format": "orcacolony_participants_v1",
                "campaign_id": loaded.campaign.campaign["id"],
                "participants": [
                    {
                        "contributor_id": "local-persistent-session-proof",
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
            temporary / "state",
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
        token_file.write_text(token + "\n", encoding="utf-8", newline="\n")
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
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
                str(CAMPAIGN),
                "--lora-config",
                str(LORA),
                "--cache",
                str(temporary / "cache"),
                "--assignments",
                "2",
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        session = json.loads(completed.stdout)
        dashboard = coordinator.dashboard()
        if session.get("format") != "orcacolony_native_worker_session_v1":
            raise RuntimeError("persistent worker output format mismatch")
        results = session["results"]
        if session.get("model_build_count") != 1 or session.get("adapter_load_count") != 1:
            raise RuntimeError("persistent worker rebuilt unchanged state")
        if [result["reused_model"] for result in results] != [False, True]:
            raise RuntimeError("persistent worker model reuse flags mismatch")
        if [result["reused_adapter"] for result in results] != [False, True]:
            raise RuntimeError("persistent worker adapter reuse flags mismatch")
        warm = results[1]["telemetry"]
        if warm["transfer_bytes"]["model"] != 0 or warm["transfer_bytes"]["adapter"] != 0:
            raise RuntimeError("persistent warm assignment fetched immutable state")
        if results[1]["receipt"]["checkpoint_metrics"]["relative_l2_error"] >= 1e-6:
            raise RuntimeError("persistent T2 checkpoint parity exceeded the guardrail")
        evaluations = dashboard["evaluations"]
        held_out_improvement = float(evaluations[0]["mean_loss"]) - float(
            evaluations[-1]["mean_loss"]
        )
        if held_out_improvement < 0:
            raise RuntimeError("persistent T2 held-out loss regressed")
        resources = dashboard["resource_observations"]
        if any(
            key in resources["memory_bytes"]
            for key in ("largest_device_capacity", "largest_js_heap_limit")
        ):
            raise RuntimeError("public dashboard exposed hardware capacity")
        warm_setup_seconds = float(warm["runtime_seconds"]["artifact_fetch"]) + float(
            warm["runtime_seconds"]["runtime_init"]
        )
        setup_reduction_ratio = 1.0 - (
            warm_setup_seconds / ONE_SHOT_WARM_SETUP_SECONDS
        )
        summary = {
            "format": "orcacolony_p3_persistent_native_session_proof_v1",
            "implementation_commit": "da606a03f185e3af48c34209daacb1396c4350e0",
            "dataset_revision": dataset.revision,
            "campaign_id": loaded.campaign.campaign["id"],
            "parameter_count": loaded.campaign.model.parameters,
            "model_build_count": session["model_build_count"],
            "adapter_load_count": session["adapter_load_count"],
            "cold_assignment": _sanitized_result(results[0]),
            "warm_assignment": _sanitized_result(results[1]),
            "resource_observations": resources,
            "checkpoint_metrics": results[1]["receipt"]["checkpoint_metrics"],
            "evaluations": evaluations,
            "held_out_mean_loss_improvement": held_out_improvement,
            "one_shot_warm_setup_seconds": ONE_SHOT_WARM_SETUP_SECONDS,
            "persistent_warm_setup_seconds": warm_setup_seconds,
            "warm_setup_reduction_ratio": setup_reduction_ratio,
        }
        token_file.unlink(missing_ok=True)
        (temporary / "proof-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
    print(args.output / "proof-summary.json")


if __name__ == "__main__":
    main()
