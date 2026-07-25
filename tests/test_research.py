from __future__ import annotations

import pytest

from orcacolony.research import validate_study_manifest


def _descriptor(identifier: str, label: str, description: str) -> dict[str, str]:
    return {
        "id": identifier,
        "label": label,
        "description": description,
    }


def _study_payload() -> dict[str, object]:
    return {
        "format": "orcacolony_study_v1",
        "study_id": "storage-offload-smoke-v1",
        "title": "Storage-offload contract smoke study",
        "hypothesis": (
            "Explicit local-storage placement can complete validated PEFT work "
            "within the declared resource budget."
        ),
        "status": "active",
        "use_case": {
            "claim": "Improve one frozen code-transformation task suite.",
            "baseline": _descriptor(
                "gpu-resident-baseline",
                "GPU-resident baseline",
                "The same adapter campaign with its frozen base resident in GPU memory.",
            ),
            "primary_metric": {
                **_descriptor(
                    "code-transform-pass-rate",
                    "Code transformation pass rate",
                    "Fraction of frozen final-holdout transformations that pass.",
                ),
                "direction": "maximize",
                "success_threshold": 0.70,
                "unit": "ratio",
            },
            "repeated_validation_suite": {
                **_descriptor(
                    "code-transform-validation-v1",
                    "Repeated code-transformation validation",
                    "Frozen validation suite evaluated at selected checkpoints.",
                ),
                "revision": "sha256:" + "1" * 64,
            },
            "final_holdout_suite": {
                **_descriptor(
                    "code-transform-holdout-v1",
                    "Final code-transformation holdout",
                    "Untouched final promotion suite.",
                ),
                "revision": "sha256:" + "2" * 64,
            },
            "guardrails": [
                _descriptor(
                    "format-validity",
                    "Output format validity",
                    "Generated outputs must remain parseable by the target tool.",
                )
            ],
        },
        "independent_variables": [
            _descriptor(
                "memory-placement",
                "Memory placement",
                "Compare GPU-resident and explicit local-storage-backed execution.",
            )
        ],
        "controlled_variables": [
            _descriptor(
                "adapter-config",
                "Adapter configuration",
                "Keep the exact base, adapter tensors, data, and optimizer fixed.",
            )
        ],
        "experiments": [
            {
                "experiment_id": "storage-offload-candidate",
                "role": "candidate",
                "manifest": "experiments/storage-offload-candidate.json",
            }
        ],
    }


def test_study_manifest_accepts_described_open_research_variables() -> None:
    validate_study_manifest(_study_payload())


def test_study_manifest_rejects_unknown_fields() -> None:
    payload = _study_payload()
    payload["private_note"] = "must not silently enter the public contract"

    with pytest.raises(ValueError, match="study contains unknown fields: private_note"):
        validate_study_manifest(payload)


def test_study_manifest_rejects_unsafe_experiment_paths() -> None:
    payload = _study_payload()
    payload["experiments"][0]["manifest"] = "../outside.json"  # type: ignore[index]

    with pytest.raises(ValueError, match="experiment manifest must be a safe relative"):
        validate_study_manifest(payload)


def test_study_manifest_rejects_boolean_metric_thresholds() -> None:
    payload = _study_payload()
    payload["use_case"]["primary_metric"]["success_threshold"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="success_threshold must be a finite number"):
        validate_study_manifest(payload)
