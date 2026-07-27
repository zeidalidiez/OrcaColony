from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .auxiliary_contributions import (
    load_auxiliary_contributions,
    verify_auxiliary_contribution_artifacts,
)
from .campaign_research import (
    build_campaign_evaluation_summary,
    campaign_research_revision,
    load_campaign_evaluation_evidence,
    verify_campaign_evaluation_artifacts,
)
from .reference import campaign_revision, load_campaign


def inspect_campaign_contract(config_path: str | Path) -> dict[str, object]:
    """Validate a campaign and report owner-supplied contract identities."""

    campaign = load_campaign(config_path)
    research = campaign.research
    if (
        research is None
        or research.get("format") != "orcacolony_campaign_research_v2"
    ):
        raise ValueError(
            "campaign lifecycle inspection requires a v2 campaign research "
            "contract"
        )
    evaluation = cast(
        Mapping[str, object],
        research["evaluation_contract"],
    )
    evaluator = cast(Mapping[str, object], evaluation["evaluator"])
    artifacts = cast(list[Mapping[str, object]], evaluation["artifacts"])
    metrics = cast(list[Mapping[str, object]], evaluation["metrics"])
    return {
        "format": "orcacolony_campaign_contract_inspection_v1",
        "campaign_id": campaign.campaign["id"],
        "campaign_revision": campaign_revision(campaign),
        "research_revision": campaign_research_revision(research),
        "question": research["question"],
        "usage_scenario": research["usage_scenario"],
        "evaluator": {
            "id": evaluator["id"],
            "revision": evaluator["revision"],
            "command": evaluator["command"],
        },
        "evaluation_artifacts": [
            {
                "id": artifact["id"],
                "kind": artifact["kind"],
                "revision": artifact["revision"],
                "uri": artifact["uri"],
            }
            for artifact in artifacts
        ],
        "metrics": [
            {
                "id": metric["id"],
                "label": metric["label"],
                "direction": metric["direction"],
                "unit": metric["unit"],
            }
            for metric in metrics
        ],
        "publication": (
            dict(campaign.publication)
            if campaign.publication is not None
            else None
        ),
    }


def preflight_campaign_evidence(
    config_path: str | Path,
    evidence_path: str | Path,
    *,
    release_checkpoint_sha256: str,
    evaluation_artifact_root: str | Path | None,
) -> dict[str, object]:
    """Validate supplied evidence without selecting or releasing a checkpoint."""

    campaign = load_campaign(config_path)
    research = campaign.research
    if (
        research is None
        or research.get("format") != "orcacolony_campaign_research_v2"
    ):
        raise ValueError(
            "campaign evidence preflight requires a v2 campaign research "
            "contract"
        )
    identity = campaign_revision(campaign)
    evidence = load_campaign_evaluation_evidence(evidence_path)
    summary = build_campaign_evaluation_summary(
        research,
        evidence,
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision=identity,
        release_checkpoint_sha256=release_checkpoint_sha256,
    )
    verified_artifacts = verify_campaign_evaluation_artifacts(
        evidence,
        evaluation_artifact_root,
    )
    return {
        "format": "orcacolony_campaign_evidence_preflight_v1",
        "campaign_id": campaign.campaign["id"],
        "campaign_revision": identity,
        "research_revision": campaign_research_revision(research),
        "release_checkpoint_sha256": release_checkpoint_sha256,
        "verified_bundle_artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in verified_artifacts.items()
        ],
        "summary": summary,
    }


def preflight_auxiliary_contributions(
    config_path: str | Path,
    ledger_path: str | Path,
    *,
    artifact_root: str | Path | None,
) -> dict[str, object]:
    """Validate owner-reviewed auxiliary credit without exposing private IDs."""

    campaign = load_campaign(config_path)
    identity = campaign_revision(campaign)
    ledger = load_auxiliary_contributions(
        ledger_path,
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision=identity,
    )
    bindings = verify_auxiliary_contribution_artifacts(
        ledger,
        artifact_root,
    )
    ledger_payload = ledger.as_payload()
    contributors = cast(
        list[Mapping[str, object]],
        ledger_payload["contributors"],
    )
    contribution_count = sum(
        len(cast(list[object], contributor["contributions"]))
        for contributor in contributors
    )
    return {
        "format": "orcacolony_auxiliary_contribution_preflight_v1",
        "campaign_id": campaign.campaign["id"],
        "campaign_revision": identity,
        "source_ledger_sha256": ledger.revision,
        "owner_reviewed": True,
        "contributor_count": len(contributors),
        "contribution_count": contribution_count,
        "verified_public_bundle_artifacts": [
            {"path": path, "sha256": binding["sha256"]}
            for path, binding in sorted(bindings.items())
            if binding["public"] is True
        ],
        "verified_private_bundle_artifact_count": sum(
            binding["public"] is False for binding in bindings.values()
        ),
    }


def _write_json(payload: Mapping[str, object], output: Path | None) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(serialized, end="")
        return
    if output.exists():
        raise ValueError(f"campaign lifecycle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8", newline="\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate owner-supplied campaign research, evidence, and "
            "contribution records without supplying campaign choices"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect",
        help="validate a v2 campaign and print its exact contract identities",
    )
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--output", type=Path)

    evidence = subparsers.add_parser(
        "validate-evidence",
        help="preflight owner-supplied evaluation evidence and bundled files",
    )
    evidence.add_argument("--config", type=Path, required=True)
    evidence.add_argument("--evidence", type=Path, required=True)
    evidence.add_argument("--release-checkpoint-sha256", required=True)
    evidence.add_argument("--evaluation-artifacts", type=Path)
    evidence.add_argument("--output", type=Path)

    contributions = subparsers.add_parser(
        "validate-contributions",
        help=(
            "preflight an owner-reviewed auxiliary contribution ledger and "
            "its bundled evidence"
        ),
    )
    contributions.add_argument("--config", type=Path, required=True)
    contributions.add_argument("--ledger", type=Path, required=True)
    contributions.add_argument("--artifacts", type=Path)
    contributions.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        payload = inspect_campaign_contract(args.config)
    elif args.command == "validate-evidence":
        payload = preflight_campaign_evidence(
            args.config,
            args.evidence,
            release_checkpoint_sha256=args.release_checkpoint_sha256,
            evaluation_artifact_root=args.evaluation_artifacts,
        )
    else:
        payload = preflight_auxiliary_contributions(
            args.config,
            args.ledger,
            artifact_root=args.artifacts,
        )
    _write_json(payload, args.output)


if __name__ == "__main__":
    main()
