from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.campaign_lifecycle import (
    inspect_campaign_contract,
    main,
    preflight_auxiliary_contributions,
    preflight_campaign_evidence,
)
from orcacolony.campaign_research import campaign_research_revision
from orcacolony.reference import campaign_revision, load_campaign


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "campaign" / "t0-smoke.json"


def _research() -> dict[str, object]:
    return {
        "format": "orcacolony_campaign_research_v2",
        "question": "What does the owner-supplied evaluation measure?",
        "usage_scenario": "A concrete scenario supplied by the campaign owner.",
        "evaluation_contract": {
            "evaluator": {
                "id": "owner-evaluator",
                "revision": "a" * 40,
                "command": ["python", "evaluate.py"],
            },
            "artifacts": [
                {
                    "id": "owner-inputs",
                    "kind": "dataset",
                    "revision": "sha256:" + "b" * 64,
                    "uri": "hf://datasets/OrcaColony/owner-inputs@revision",
                }
            ],
            "metrics": [
                {
                    "id": "owner-metric",
                    "label": "Owner metric",
                    "description": "An exact calculation supplied by the owner.",
                    "direction": "observe",
                    "unit": "cases",
                }
            ],
        },
        "analysis_plan": ["Inspect the owner-requested comparison."],
    }


def _config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    research = _research()
    payload["research"] = research
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, research


def _evidence(
    *,
    campaign_id: str,
    campaign_revision_value: str,
    research: dict[str, object],
    release_checkpoint_sha256: str,
    artifact_sha256: str,
) -> dict[str, object]:
    return {
        "format": "orcacolony_campaign_evaluation_evidence_v1",
        "campaign_id": campaign_id,
        "campaign_revision": campaign_revision_value,
        "research_revision": campaign_research_revision(research),
        "release_evaluation_id": "candidate",
        "evaluations": [
            {
                "id": "baseline",
                "label": "Owner baseline",
                "subject": {
                    "id": "baseline-model",
                    "label": "Owner baseline model",
                    "revision": "c" * 64,
                },
                "measurements": [
                    {"metric_id": "owner-metric", "value": 4}
                ],
                "artifacts": [
                    {
                        "id": "baseline-samples",
                        "sha256": artifact_sha256,
                        "uri": "bundle:samples.json",
                    }
                ],
            },
            {
                "id": "candidate",
                "label": "Owner release candidate",
                "subject": {
                    "id": "candidate-model",
                    "label": "Owner release candidate model",
                    "revision": release_checkpoint_sha256,
                },
                "measurements": [
                    {"metric_id": "owner-metric", "value": 7}
                ],
                "artifacts": [
                    {
                        "id": "candidate-samples",
                        "sha256": artifact_sha256,
                        "uri": "bundle:samples.json",
                    }
                ],
            },
        ],
        "comparisons": [
            {
                "id": "owner-comparison",
                "baseline_evaluation_id": "baseline",
                "candidate_evaluation_id": "candidate",
                "summary": "The comparison requested by the campaign owner.",
            }
        ],
        "findings": [
            {
                "id": "owner-finding",
                "label": "Owner finding",
                "kind": "inconclusive",
                "description": "A finding supplied with the campaign evidence.",
            }
        ],
        "limitations": ["A limitation supplied with the campaign evidence."],
        "reproduction": {
            "command": ["python", "evaluate.py"],
            "notes": "Run the owner-supplied evaluator.",
        },
    }


def test_inspection_reports_exact_owner_supplied_contract_identities(
    tmp_path: Path,
) -> None:
    config_path, research = _config(tmp_path)
    campaign = load_campaign(config_path)

    inspection = inspect_campaign_contract(config_path)

    assert inspection["campaign_revision"] == campaign_revision(campaign)
    assert inspection["research_revision"] == campaign_research_revision(
        research
    )
    assert inspection["question"] == research["question"]
    assert inspection["metrics"] == [
        {
            "id": "owner-metric",
            "label": "Owner metric",
            "direction": "observe",
            "unit": "cases",
        }
    ]
    serialized = json.dumps(inspection)
    assert "success_threshold" not in serialized
    assert "checkpoint_selection" not in serialized


def test_evidence_preflight_validates_identity_and_bundled_bytes(
    tmp_path: Path,
) -> None:
    config_path, research = _config(tmp_path)
    campaign = load_campaign(config_path)
    artifacts = tmp_path / "evaluation-artifacts"
    artifacts.mkdir()
    sample_path = artifacts / "samples.json"
    sample_path.write_text('{"case":"owner supplied"}\n', encoding="utf-8")
    artifact_sha256 = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    release_checkpoint_sha256 = "d" * 64
    evidence = _evidence(
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision_value=campaign_revision(campaign),
        research=research,
        release_checkpoint_sha256=release_checkpoint_sha256,
        artifact_sha256=artifact_sha256,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    preflight = preflight_campaign_evidence(
        config_path,
        evidence_path,
        release_checkpoint_sha256=release_checkpoint_sha256,
        evaluation_artifact_root=artifacts,
    )

    assert preflight["verified_bundle_artifacts"] == [
        {"path": "samples.json", "sha256": artifact_sha256}
    ]
    assert preflight["summary"]["release_evaluation_id"] == "candidate"
    assert preflight["summary"]["comparisons"][0]["metrics"][0][
        "absolute_change"
    ] == pytest.approx(3.0)


def test_evidence_loader_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    config_path, _ = _config(tmp_path)
    evidence_path = tmp_path / "ambiguous-evidence.json"
    evidence_path.write_text(
        '{"format":"first","format":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate campaign evaluation JSON key"):
        preflight_campaign_evidence(
            config_path,
            evidence_path,
            release_checkpoint_sha256="e" * 64,
            evaluation_artifact_root=None,
        )


def test_auxiliary_contribution_preflight_hides_private_ids(
    tmp_path: Path,
) -> None:
    config_path, _ = _config(tmp_path)
    campaign = load_campaign(config_path)
    artifacts = tmp_path / "auxiliary-artifacts"
    artifacts.mkdir()
    evidence = artifacts / "review.json"
    evidence.write_text('{"review":"complete"}\n', encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    ledger = {
        "format": "orcacolony_auxiliary_contributions_v1",
        "campaign_id": campaign.campaign["id"],
        "campaign_revision": campaign_revision(campaign),
        "owner_reviewed": True,
        "contributors": [
            {
                "contributor_id": "private-lifecycle-id",
                "credit": {
                    "visibility": "pseudonymous",
                    "display_name": "Lifecycle Helper",
                    "profile_url": None,
                    "team": None,
                    "show_contribution_details": True,
                    "show_time": False,
                    "show_hardware": False,
                    "public_disclosure_confirmed": True,
                },
                "resources": {
                    "person_time_seconds": None,
                    "compute_time_seconds": None,
                    "hardware": [],
                },
                "contributions": [
                    {
                        "id": "lifecycle-review",
                        "kind": "review",
                        "description": "Reviewed the lifecycle fixture.",
                        "status": "completed",
                        "evidence": [
                            {
                                "id": "review-record",
                                "sha256": evidence_sha256,
                                "uri": "bundle:review.json",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    ledger_path = tmp_path / "auxiliary-contributions.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    preflight = preflight_auxiliary_contributions(
        config_path,
        ledger_path,
        artifact_root=artifacts,
    )

    assert preflight["contributor_count"] == 1
    assert preflight["contribution_count"] == 1
    assert preflight["verified_public_bundle_artifacts"] == [
        {"path": "review.json", "sha256": evidence_sha256}
    ]
    assert "private-lifecycle-id" not in json.dumps(preflight)
    assert "Lifecycle Helper" not in json.dumps(preflight)

    output = tmp_path / "auxiliary-preflight.json"
    main(
        [
            "validate-contributions",
            "--config",
            str(config_path),
            "--ledger",
            str(ledger_path),
            "--artifacts",
            str(artifacts),
            "--output",
            str(output),
        ]
    )
    assert json.loads(output.read_text(encoding="utf-8")) == preflight


def test_cli_writes_inspection_without_overwriting(
    tmp_path: Path,
) -> None:
    config_path, _ = _config(tmp_path)
    output = tmp_path / "inspection.json"

    main(["inspect", "--config", str(config_path), "--output", str(output)])

    assert json.loads(output.read_text(encoding="utf-8"))["format"] == (
        "orcacolony_campaign_contract_inspection_v1"
    )
    with pytest.raises(ValueError, match="output already exists"):
        main(["inspect", "--config", str(config_path), "--output", str(output)])
