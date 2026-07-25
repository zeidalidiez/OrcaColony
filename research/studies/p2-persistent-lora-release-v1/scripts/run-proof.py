from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import LeasedGradient
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest


ROOT = Path(".artifacts/p2-lora-evaluated-release-proof")
CONFIG = Path("campaign/t0-smoke.json")
LORA_CONFIG = Path("campaign/t0-lora-smoke.json")
TOKEN = "evaluated-release-proof-token"


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


if ROOT.exists():
    shutil.rmtree(ROOT)
ROOT.mkdir(parents=True)
corpus = (
    "A little fox found a blue kite and brought it home to share.\n"
    "<|endoftext|>\n"
    "A kind otter helped a duckling cross the quiet river.\n"
    "<|endoftext|>\n"
) * 600
dataset_root = ROOT / "dataset"
manifest = build_dataset_artifacts(
    train_bytes=corpus.encode("utf-8"),
    validation_bytes=corpus[: len(corpus) // 2].encode("utf-8"),
    output_dir=dataset_root,
    source={
        "dataset": "local/p2-lora-evaluated-release-proof",
        "revision": "p2-lora-evaluated-release-proof-v1",
        "license": "cc0-1.0",
    },
    vocab_size=300,
    context_length=128,
)
dataset = PackedDataset.load(dataset_root)
campaign_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
campaign_payload["dataset"] = {
    "format": manifest["format"],
    "manifest_sha256": dataset.revision,
    "tokenizer_sha256": manifest["tokenizer"]["sha256"],
    "train_sha256": manifest["files"]["train.safetensors"],
    "validation_sha256": manifest["files"]["validation.safetensors"],
}
campaign_payload["evaluation"] = {
    "metric": "held_out_cross_entropy",
    "checkpoint_selection": "lowest_mean_loss",
    "validation_sequences": 8,
    "batch_size": 2,
    "success_gate": {
        "metric": "mean_loss",
        "minimum_improvement_from_initialization": 0.0001,
    },
}
campaign_path = ROOT / "campaign.json"
campaign_bytes = canonical_json(campaign_payload)
campaign_path.write_bytes(campaign_bytes)
lora_payload = json.loads(LORA_CONFIG.read_text(encoding="utf-8"))
lora_payload["base"]["campaign_file"] = campaign_path.name
lora_payload["base"]["campaign_sha256"] = hashlib.sha256(campaign_bytes).hexdigest()
lora_path = ROOT / "lora.json"
lora_path.write_bytes(canonical_json(lora_payload))
lora = load_lora_manifest(campaign_path, lora_path)
worker_ids = ["evaluated-worker-a", "evaluated-worker-b"]
token_sha256 = hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()
participant_payload = {
    "format": "orcacolony_participants_v1",
    "campaign_id": lora.campaign.campaign["id"],
    "participants": [
        {
            "contributor_id": "local-evaluated-release-proof",
            "worker_ids": worker_ids,
            "worker_token_sha256": {
                worker_id: token_sha256 for worker_id in worker_ids
            },
            "credit": {"public": False, "display_name": None},
        }
    ],
}
participants_path = ROOT / "participants.json"
participants_path.write_bytes(canonical_json(participant_payload))
participants = ParticipantRegistry.from_payload(
    participant_payload,
    campaign_id=str(lora.campaign.campaign["id"]),
)
state_dir = ROOT / "campaign-state"
coordinator = CampaignCoordinator.create(
    lora.campaign,
    state_dir,
    participants=participants,
    worker_count=2,
    target_steps=2,
    dataset=dataset,
    lora=lora,
)
for expected_step in (1, 2):
    for worker_id in worker_ids:
        assignment = coordinator.lease(
            worker_id,
            worker_token=TOKEN,
            now=expected_step * 100,
        )
        assignment_id = str(assignment["assignment_id"])
        coordinator.accept(
            LeasedGradient(
                assignment_id=assignment_id,
                lease_token=str(assignment["lease_token"]),
                checkpoint_sha256=str(assignment["checkpoint_sha256"]),
                loss_sum=float(assignment["expected_loss_sum"]),
                loss_weight_sum=int(assignment["loss_weight_sum"]),
                safetensors=coordinator.oracle_gradient_path(assignment_id).read_bytes(),
                runtime_backend="python-oracle-f32",
            ),
            now=expected_step * 100 + 1,
        )
    if expected_step == 1:
        coordinator = CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )
status = coordinator.status()
summary = {
    "status": status,
    "dashboard": coordinator.dashboard(),
    "paths": {
        "campaign": str(campaign_path),
        "lora": str(lora_path),
        "participants": str(participants_path),
        "dataset": str(dataset_root),
        "campaign_state": str(state_dir),
    },
}
(ROOT / "proof-summary.json").write_bytes(canonical_json(summary))
print(json.dumps(summary, sort_keys=True))
