from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from orcacolony.peft import export_base_layer_bundle, load_lora_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a pre-authenticated OrcaColony base layer bundle."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lora-config", required=True, type=Path)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--base-artifact-sha256", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_lora_manifest(args.config, args.lora_config)
    started = time.perf_counter()
    exported = export_base_layer_bundle(
        loaded.campaign,
        loaded.config,
        args.base_artifact,
        args.base_artifact_sha256,
        args.bundle,
    )
    elapsed = time.perf_counter() - started
    result = {
        "format": "orcacolony_base_layer_bundle_export_proof_v1",
        "campaign_id": loaded.campaign.campaign["id"],
        "base_model_sha256": exported.base_model_sha256,
        "source_artifact_sha256": exported.source_artifact_sha256,
        "manifest_sha256": exported.manifest_sha256,
        "linear_count": exported.linear_count,
        "artifact_bytes": exported.artifact_bytes,
        "artifact_file_count": sum(1 for path in exported.output_dir.iterdir()),
        "export_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
