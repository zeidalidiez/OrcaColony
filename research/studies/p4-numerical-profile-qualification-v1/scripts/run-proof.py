from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the P4 connected homogeneous-int8 qualification."
    )
    parser.add_argument("--dataset-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    proof_path = output / "proof-summary.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "spikes/int8-frozen-linear/connected_campaign.py"),
        "--campaign",
        str(REPO_ROOT / "campaign/t1-tinystories-smoke.json"),
        "--lora",
        str(REPO_ROOT / "campaign/t1-tinystories-lora-smoke.json"),
        "--dataset",
        str(args.dataset_artifacts.resolve()),
        "--browser-root",
        str(REPO_ROOT / "spikes/burn-browser-gradient/www"),
        "--state",
        str(output / "campaign-state"),
        "--exact-reference",
        str(output / "exact-reference"),
        "--output",
        str(proof_path),
        "--target-steps",
        "2",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("format") != "orcacolony_p4_connected_int8_proof_v1":
        raise RuntimeError("connected proof returned an unexpected format")
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
