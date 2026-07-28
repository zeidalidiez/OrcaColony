from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import load_campaign
from orcacolony.sparse_expert_trajectory import (
    run_content_addressed_sparse_trajectory_comparison,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare replicated and topology-local content-addressed "
            "checkpoint storage for the exact persisted sparse trajectory"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--comparison-order",
        choices=(
            "replicated-then-content-addressed",
            "content-addressed-then-replicated",
        ),
        default="replicated-then-content-addressed",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    campaign = load_campaign(args.config)
    dataset = PackedDataset.load(args.dataset)
    evidence = run_content_addressed_sparse_trajectory_comparison(
        campaign,
        args.state,
        dataset=dataset,
        steps=args.steps,
        expert_count=args.expert_count,
        router_aux_weight=args.router_aux_weight,
        timeout_seconds=args.timeout_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        comparison_order=args.comparison_order,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
