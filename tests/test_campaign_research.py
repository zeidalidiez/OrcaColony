from __future__ import annotations

import json
from pathlib import Path

import pytest

from orcacolony.campaign_research import (
    build_campaign_evaluation_summary,
    campaign_research_revision,
    validate_campaign_research_contract,
)
from orcacolony.reference import campaign_from_mapping


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"


def _research() -> dict[str, object]:
    return {
        "format": "orcacolony_campaign_research_v2",
        "question": "What changed under the campaign owner's usage evaluation?",
        "usage_scenario": (
            "The campaign owner supplies the concrete usage scenario when "
            "creating this campaign."
        ),
        "evaluation_contract": {
            "evaluator": {
                "id": "owner-evaluator",
                "revision": "a" * 40,
                "command": ["python", "evaluate.py", "--results", "results.json"],
            },
            "artifacts": [
                {
                    "id": "evaluation-inputs",
                    "kind": "dataset",
                    "revision": "sha256:" + "1" * 64,
                    "uri": "hf://datasets/OrcaColony/example@revision",
                }
            ],
            "metrics": [
                {
                    "id": "usage-score",
                    "label": "Usage score",
                    "description": "Campaign-owner-defined scenario score.",
                    "direction": "maximize",
                    "unit": "ratio",
                },
                {
                    "id": "failure-rate",
                    "label": "Failure rate",
                    "description": "Fraction of declared cases that failed.",
                    "direction": "minimize",
                    "unit": "ratio",
                },
                {
                    "id": "output-length",
                    "label": "Output length",
                    "description": "Observed output length without a preferred direction.",
                    "direction": "observe",
                    "unit": "tokens",
                },
            ],
        },
        "analysis_plan": [
            "Compare the owner-selected evaluation records and inspect artifacts."
        ],
    }


def _evidence(
    research: dict[str, object],
    *,
    release_checkpoint_sha256: str = "4" * 64,
) -> dict[str, object]:
    def evaluation(
        identifier: str,
        label: str,
        revision: str,
        values: tuple[float, float, int],
        artifact_sha256: str,
    ) -> dict[str, object]:
        return {
            "id": identifier,
            "label": label,
            "subject": {
                "id": identifier + "-model",
                "label": label + " model",
                "revision": revision,
            },
            "measurements": [
                {"metric_id": "usage-score", "value": values[0]},
                {"metric_id": "failure-rate", "value": values[1]},
                {"metric_id": "output-length", "value": values[2]},
            ],
            "artifacts": [
                {
                    "id": identifier + "-samples",
                    "sha256": artifact_sha256,
                    "uri": f"bundle:{identifier}-samples.json",
                }
            ],
        }

    return {
        "format": "orcacolony_campaign_evaluation_evidence_v1",
        "campaign_id": "owner-defined-campaign",
        "campaign_revision": "2" * 64,
        "research_revision": campaign_research_revision(research),
        "release_evaluation_id": "trained",
        "evaluations": [
            evaluation(
                "initial",
                "Initial checkpoint",
                "3" * 64,
                (0.25, 0.75, 20),
                "5" * 64,
            ),
            evaluation(
                "trained",
                "Released checkpoint",
                release_checkpoint_sha256,
                (0.50, 0.60, 24),
                "6" * 64,
            ),
        ],
        "comparisons": [
            {
                "id": "initial-to-trained",
                "baseline_evaluation_id": "initial",
                "candidate_evaluation_id": "trained",
                "summary": "Owner-requested comparison of the two evaluations.",
            }
        ],
        "findings": [
            {
                "id": "mixed-result",
                "label": "Mixed measured result",
                "kind": "mixed",
                "description": (
                    "The evidence records metric movement without assigning a "
                    "framework promotion decision."
                ),
            }
        ],
        "limitations": ["This fixture does not represent a real usage scenario."],
        "reproduction": {
            "command": ["python", "evaluate.py"],
            "notes": "Use the exact evaluator and artifacts declared by the owner.",
        },
    }


def test_campaign_research_contract_contains_owner_choices_without_gates() -> None:
    research = _research()

    validate_campaign_research_contract(research)

    serialized = json.dumps(research)
    assert "success_threshold" not in serialized
    assert "promotion" not in serialized
    assert "final_holdout" not in serialized
    assert len(campaign_research_revision(research)) == 64


def test_campaign_research_contract_rejects_framework_supplied_metric_rules() -> None:
    research = _research()
    research["evaluation_contract"]["metrics"][0]["success_threshold"] = 0.9  # type: ignore[index]

    with pytest.raises(ValueError, match="unknown fields: success_threshold"):
        validate_campaign_research_contract(research)


def test_campaign_evidence_computes_owner_requested_comparison_without_gate() -> None:
    research = _research()
    summary = build_campaign_evaluation_summary(
        research,
        _evidence(research),
        campaign_id="owner-defined-campaign",
        campaign_revision="2" * 64,
        release_checkpoint_sha256="4" * 64,
    )

    comparison = summary["comparisons"][0]
    metrics = {
        entry["metric_id"]: entry for entry in comparison["metrics"]
    }
    assert metrics["usage-score"]["absolute_change"] == pytest.approx(0.25)
    assert metrics["usage-score"]["change_in_preferred_direction"] == pytest.approx(
        0.25
    )
    assert metrics["failure-rate"]["absolute_change"] == pytest.approx(-0.15)
    assert metrics["failure-rate"]["change_in_preferred_direction"] == pytest.approx(
        0.15
    )
    assert metrics["output-length"]["absolute_change"] == pytest.approx(4.0)
    assert metrics["output-length"]["change_in_preferred_direction"] is None
    assert "decision" not in summary
    assert "passed" not in summary


def test_campaign_evidence_must_bind_the_released_checkpoint() -> None:
    research = _research()

    with pytest.raises(ValueError, match="not bound to the released checkpoint"):
        build_campaign_evaluation_summary(
            research,
            _evidence(research),
            campaign_id="owner-defined-campaign",
            campaign_revision="2" * 64,
            release_checkpoint_sha256="7" * 64,
        )


def test_campaign_evidence_must_measure_every_owner_declared_metric() -> None:
    research = _research()
    evidence = _evidence(research)
    evidence["evaluations"][0]["measurements"].pop()  # type: ignore[index]

    with pytest.raises(ValueError, match="must measure every declared metric"):
        build_campaign_evaluation_summary(
            research,
            evidence,
            campaign_id="owner-defined-campaign",
            campaign_revision="2" * 64,
            release_checkpoint_sha256="4" * 64,
        )


def test_campaign_config_accepts_owner_defined_research_without_publication() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["research"] = _research()

    campaign = campaign_from_mapping(payload)

    assert campaign.research == payload["research"]
    assert campaign.publication is None


def test_publication_contract_is_validated_independently_of_research() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["publication"] = {
        "format": "orcacolony_huggingface_publication_v1",
        "model_repo_id": "wrong-namespace/model",
        "dataset_repo_id": "OrcaColony/example-dataset",
        "model_license": "apache-2.0",
        "dataset_license": "cc0-1.0",
        "visibility_policy": "private_review_then_public",
    }

    with pytest.raises(ValueError, match="OrcaColony namespace"):
        campaign_from_mapping(payload)
