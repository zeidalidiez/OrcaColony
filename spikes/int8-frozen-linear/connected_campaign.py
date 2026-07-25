from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from safetensors.torch import load_file as load_safetensors_file

from orcacolony.artifacts import PackedDataset
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.coordinator import _tensor_metrics
from orcacolony.multiworker import create_http_server
from orcacolony.native_worker import NativeWorkerSession
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import (
    EXACT_CPU_FP32_PROFILE,
    INT8_FROZEN_LINEAR_PROFILE,
    load_lora_manifest,
    run_lora_training,
)


FORMAT = "orcacolony_p4_connected_int8_proof_v1"
WORKER_TOKEN = "p4-connected-int8-local-proof-token"
WORKERS = ("p4-int8-resident", "p4-int8-layer-bundle")


def _participants(campaign_id: str) -> ParticipantRegistry:
    token_sha256 = hashlib.sha256(WORKER_TOKEN.encode("utf-8")).hexdigest()
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "p4-local-profile-proof",
                    "worker_ids": list(WORKERS),
                    "worker_token_sha256": {
                        worker_id: token_sha256 for worker_id in WORKERS
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=campaign_id,
    )


def _server(coordinator: CampaignCoordinator, browser_root: Path, port: int = 0):
    server = create_http_server(coordinator, browser_root, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join()


def _checkpoint_provenance(state_dir: Path, history: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for entry in history:
        checkpoint_dir = state_dir / str(entry["path"])
        checkpoint_state = json.loads(
            (checkpoint_dir / "state.json").read_text(encoding="utf-8")
        )
        records.append(
            {
                "step": entry["step"],
                "numerical_profile": entry["numerical_profile"],
                "checkpoint_format": checkpoint_state["format"],
                "weight_checkpoint_sha256": checkpoint_state[
                    "weight_checkpoint_sha256"
                ],
                "resume_state_sha256": checkpoint_state["checkpoint_sha256"],
            }
        )
    return records


def run(
    *,
    campaign_path: Path,
    lora_path: Path,
    dataset_path: Path,
    browser_root: Path,
    state_dir: Path,
    exact_reference_dir: Path,
    output_path: Path,
    target_steps: int,
) -> dict[str, object]:
    if target_steps < 2:
        raise ValueError("connected restart proof requires at least two steps")
    if state_dir.exists() and any(state_dir.iterdir()):
        raise ValueError(f"state directory must be absent or empty: {state_dir}")
    if exact_reference_dir.exists() and any(exact_reference_dir.iterdir()):
        raise ValueError(
            f"exact reference directory must be absent or empty: {exact_reference_dir}"
        )

    loaded = load_lora_manifest(campaign_path, lora_path)
    dataset = PackedDataset.load(dataset_path)
    campaign_id = str(loaded.campaign.campaign["id"])
    participants = _participants(campaign_id)
    coordinator = CampaignCoordinator.create(
        loaded.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=target_steps,
        dataset=dataset,
        lora=loaded,
        publish_base_layer_bundle=True,
        numerical_profile=INT8_FROZEN_LINEAR_PROFILE,
    )
    server, thread = _server(coordinator, browser_root)
    port = server.server_port
    origin = f"http://127.0.0.1:{port}"
    resident = NativeWorkerSession(
        coordinator_url=origin,
        worker_id=WORKERS[0],
        worker_token=WORKER_TOKEN,
        campaign_path=campaign_path,
        lora_path=lora_path,
        cache_dir=state_dir.parent / f"{state_dir.name}-resident-cache",
        base_profile="resident",
        numerical_profile=INT8_FROZEN_LINEAR_PROFILE,
    )
    bundle = NativeWorkerSession(
        coordinator_url=origin,
        worker_id=WORKERS[1],
        worker_token=WORKER_TOKEN,
        campaign_path=campaign_path,
        lora_path=lora_path,
        cache_dir=state_dir.parent / f"{state_dir.name}-bundle-cache",
        base_profile="layer-bundle",
        numerical_profile=INT8_FROZEN_LINEAR_PROFILE,
    )

    resident_results = []
    bundle_results = []
    try:
        resident_results.append(resident.run_assignment())
        bundle_results.append(bundle.run_assignment())
    finally:
        _stop_server(server, thread)

    coordinator = CampaignCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        dataset=dataset,
        lora=loaded,
        numerical_profile=INT8_FROZEN_LINEAR_PROFILE,
    )
    server, thread = _server(coordinator, browser_root, port=port)
    try:
        for _ in range(1, target_steps):
            resident_results.append(resident.run_assignment())
            bundle_results.append(bundle.run_assignment())
    finally:
        _stop_server(server, thread)

    status = coordinator.status()
    if status["state"] != "campaign_complete":
        raise RuntimeError("connected int8 campaign did not complete")
    if resident.model_build_count != 1 or bundle.model_build_count != 1:
        raise RuntimeError("persistent worker rebuilt a frozen base")

    campaign_state = json.loads(
        (state_dir / "campaign-state.json").read_text(encoding="utf-8")
    )
    accepted = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    evaluations = json.loads(
        (state_dir / "evaluations.json").read_text(encoding="utf-8")
    )
    if accepted["numerical_profile"] != INT8_FROZEN_LINEAR_PROFILE:
        raise RuntimeError("accepted-work ledger lost numerical profile")
    runtimes = sorted({entry["runtime_backend"] for entry in accepted["entries"]})
    expected_runtimes = [
        "python-native-cpu-int8-f32-dequant",
        "python-native-cpu-layer-bundle-int8-f32-dequant",
    ]
    if runtimes != expected_runtimes:
        raise RuntimeError("connected proof did not exercise both int8 placements")

    mismatch_error: str | None = None
    try:
        CampaignCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=loaded,
            numerical_profile=EXACT_CPU_FP32_PROFILE,
        )
    except ValueError as exc:
        mismatch_error = str(exc)
    if mismatch_error is None:
        raise RuntimeError("campaign restart accepted the wrong numerical profile")

    exact = run_lora_training(
        loaded,
        exact_reference_dir,
        target_steps=target_steps,
        dataset=dataset,
    )
    int8_checkpoint = state_dir / str(campaign_state["checkpoints"][-1]["path"])
    fp32_separation = _tensor_metrics(
        load_safetensors_file(str(exact.checkpoint_dir / "adapter.safetensors")),
        load_safetensors_file(str(int8_checkpoint / "adapter.safetensors")),
    )

    evaluation_by_phase = {
        (
            "initialization"
            if int(record["step"]) == 0
            else f"step_{int(record['step'])}"
        ): float(record["mean_loss"])
        for record in evaluations["entries"]
    }
    initial_loss = evaluation_by_phase["initialization"]
    final_loss = evaluation_by_phase[f"step_{target_steps}"]
    receipts = [
        result.receipt for pair in zip(resident_results, bundle_results) for result in pair
    ]
    checkpoint_errors = [
        float(receipt["checkpoint_metrics"]["relative_l2_error"])
        for receipt in receipts
        if receipt["step_complete"]
    ]
    gradient_errors = [
        float(receipt["gradient_metrics"]["relative_l2_error"])
        for receipt in receipts
    ]
    proof: dict[str, object] = {
        "format": FORMAT,
        "campaign_id": campaign_id,
        "target_steps": target_steps,
        "numerical_profile": INT8_FROZEN_LINEAR_PROFILE,
        "execution_profiles": [
            {
                "runtime_backend": expected_runtimes[0],
                "placement": "authenticated-resident-base-then-int8",
                "model_build_count": resident.model_build_count,
                "adapter_load_count": resident.adapter_load_count,
                "reused_model": [result.reused_model for result in resident_results],
                "reused_adapter": [result.reused_adapter for result in resident_results],
                "model_transfer_bytes": [
                    result.telemetry["transfer_bytes"]["model"]
                    for result in resident_results
                ],
            },
            {
                "runtime_backend": expected_runtimes[1],
                "placement": "authenticated-layer-bundle-direct-int8",
                "model_build_count": bundle.model_build_count,
                "adapter_load_count": bundle.adapter_load_count,
                "reused_model": [result.reused_model for result in bundle_results],
                "reused_adapter": [result.reused_adapter for result in bundle_results],
                "model_transfer_bytes": [
                    result.telemetry["transfer_bytes"]["model"]
                    for result in bundle_results
                ],
            },
        ],
        "connected_campaign": {
            "accepted_assignments": len(accepted["entries"]),
            "runtime_backends": runtimes,
            "coordinator_restart_after_step": 1,
            "worker_gradient_relative_l2_max": max(gradient_errors),
            "checkpoint_relative_l2_max": max(checkpoint_errors),
        },
        "checkpoint_provenance": _checkpoint_provenance(
            state_dir, campaign_state["checkpoints"]
        ),
        "held_out_evaluation": {
            "initialization_mean_loss": initial_loss,
            "final_mean_loss": final_loss,
            "improvement": initial_loss - final_loss,
            "by_phase": evaluation_by_phase,
        },
        "fp32_separation": {
            "exact_profile": EXACT_CPU_FP32_PROFILE,
            **fp32_separation,
        },
        "negative_restart_gate": {
            "wrong_profile_rejected": True,
            "error": mismatch_error,
        },
        "conclusion": (
            "Homogeneous int8 is qualified for connected resident and direct-layer-"
            "bundle execution under its own oracle and checkpoint provenance; it is "
            "not interchangeable with exact FP32."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the connected homogeneous-int8 P4 qualification campaign."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--browser-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--exact-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=2)
    args = parser.parse_args()
    proof = run(
        campaign_path=args.campaign,
        lora_path=args.lora,
        dataset_path=args.dataset,
        browser_root=args.browser_root,
        state_dir=args.state,
        exact_reference_dir=args.exact_reference,
        output_path=args.output,
        target_steps=args.target_steps,
    )
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
