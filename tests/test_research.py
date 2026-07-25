from __future__ import annotations

import json
from pathlib import Path

import pytest

from orcacolony import research
from orcacolony.research import (
    build_result_bundle,
    validate_experiment_manifest,
    validate_study_manifest,
)


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


def _experiment_payload() -> dict[str, object]:
    return {
        "format": "orcacolony_experiment_v1",
        "study_id": "storage-offload-smoke-v1",
        "experiment_id": "storage-offload-candidate",
        "title": "Explicit local-storage candidate",
        "status": "active",
        "subject": {
            "kind": "campaign",
            "id": "code-transform-peft-storage-v1",
            "revision": "sha256:" + "3" * 64,
        },
        "artifacts": [
            {
                "id": "code",
                "kind": "git-commit",
                "revision": "a" * 40,
                "uri": "https://example.invalid/orcacolony/commit/" + "a" * 40,
            },
            {
                "id": "dataset",
                "kind": "dataset-manifest",
                "revision": "sha256:" + "4" * 64,
                "uri": "https://example.invalid/datasets/code-transform-v1",
            },
        ],
        "method": {
            "training_method": _descriptor(
                "lora-peft",
                "LoRA PEFT",
                "Train only the frozen campaign's declared adapter tensors.",
            ),
            "execution_topology": _descriptor(
                "replicated-full-model",
                "Replicated full-model execution",
                "Each worker completes its adapter-gradient assignment independently.",
            ),
            "memory_profiles": [
                _descriptor(
                    "explicit-local-storage",
                    "Explicit local-storage offload",
                    "Stream immutable base shards through a bounded RAM cache.",
                )
            ],
            "numerical_profiles": [
                _descriptor(
                    "quantized-base-fp32-adapter",
                    "Quantized base with FP32 adapters",
                    "Keep adapter gradients and coordinator accumulation in FP32.",
                )
            ],
        },
        "worker_profiles": [
            _descriptor(
                "transient-native-nvme",
                "Transient native worker with local NVMe",
                "A community worker that may complete one assignment and leave.",
            )
        ],
        "resource_budget": {
            "limits": [
                {
                    "id": "wall-time",
                    "label": "Maximum assignment wall time",
                    "value": 1800,
                    "unit": "seconds",
                },
                {
                    "id": "network-transfer",
                    "label": "Maximum assignment network transfer",
                    "value": 268435456,
                    "unit": "bytes",
                },
            ]
        },
        "reproduction": {
            "command": [
                "uv",
                "run",
                "python",
                "-m",
                "orcacolony.research",
                "record",
            ],
            "notes": "Run with the exact study, experiment, and evidence manifests.",
        },
    }


def _evidence_payload() -> dict[str, object]:
    return {
        "format": "orcacolony_experiment_evidence_v1",
        "study_id": "storage-offload-smoke-v1",
        "experiment_id": "storage-offload-candidate",
        "outcome": "rejected",
        "completed_at": "2026-07-24T18:00:00Z",
        "summary": (
            "The candidate completed correctly but missed the fixed use-case and "
            "assignment-time gates."
        ),
        "measurements": [
            {
                "id": "assignment-wall-time",
                "label": "Assignment wall time",
                "value": 2400,
                "unit": "seconds",
            }
        ],
        "evaluation": {
            "primary_metric": {
                "id": "code-transform-pass-rate",
                "value": 0.65,
            },
            "guardrails": [
                {
                    "id": "format-validity",
                    "passed": True,
                    "detail": "All generated outputs remained parseable.",
                }
            ],
        },
        "findings": [
            {
                "id": "storage-throughput-bottleneck",
                "label": "Storage throughput bottleneck",
                "description": "GPU execution remained idle while base shards loaded.",
                "kind": "negative",
            }
        ],
        "limitations": [
            "The run covered one NVMe and GPU combination and does not generalize to all storage devices."
        ],
        "artifacts": [
            {
                "id": "measurement-log",
                "kind": "json-log",
                "revision": "sha256:" + "5" * 64,
                "uri": "artifacts/measurement-log.json",
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


def test_experiment_manifest_accepts_open_execution_profiles() -> None:
    validate_experiment_manifest(_study_payload(), _experiment_payload())


def test_experiment_manifest_must_be_linked_from_the_study() -> None:
    experiment = _experiment_payload()
    experiment["experiment_id"] = "unlinked-candidate"

    with pytest.raises(ValueError, match="experiment is not referenced by the study"):
        validate_experiment_manifest(_study_payload(), experiment)


def test_result_bundle_is_deterministic_and_records_negative_findings(
    tmp_path: Path,
) -> None:
    study = _study_payload()
    experiment = _experiment_payload()
    evidence = _evidence_payload()

    first = tmp_path / "result-a"
    second = tmp_path / "result-b"
    first_result = build_result_bundle(study, experiment, evidence, first)
    second_result = build_result_bundle(study, experiment, evidence, second)

    assert first_result == second_result
    for filename in (
        "study.json",
        "experiment.json",
        "evidence.json",
        "result.json",
        "RESULT.md",
        "SHA256SUMS",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    persisted = json.loads((first / "result.json").read_text(encoding="utf-8"))
    assert persisted["decision"] == {
        "guardrails_passed": True,
        "primary_metric_passed": False,
        "use_case_passed": False,
    }
    assert persisted["outcome"] == "rejected"
    report = (first / "RESULT.md").read_text(encoding="utf-8")
    assert "Storage throughput bottleneck" in report
    assert "does not generalize to all storage devices" in report


def test_result_bundle_requires_complete_guardrail_evidence(tmp_path: Path) -> None:
    evidence = _evidence_payload()
    evidence["evaluation"]["guardrails"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="guardrail evidence must exactly match"):
        build_result_bundle(
            _study_payload(),
            _experiment_payload(),
            evidence,
            tmp_path / "result",
        )


def test_record_cli_builds_the_linked_result_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    study_root = tmp_path / "study"
    experiment_root = study_root / "experiments"
    evidence_root = study_root / "evidence"
    experiment_root.mkdir(parents=True)
    evidence_root.mkdir()
    study_path = study_root / "study.json"
    experiment_path = experiment_root / "storage-offload-candidate.json"
    evidence_path = evidence_root / "storage-offload-candidate.json"
    study_path.write_text(json.dumps(_study_payload()), encoding="utf-8")
    experiment_path.write_text(json.dumps(_experiment_payload()), encoding="utf-8")
    evidence_path.write_text(json.dumps(_evidence_payload()), encoding="utf-8")
    output = tmp_path / "result"

    research.main(
        [
            "record",
            "--study",
            str(study_path),
            "--experiment",
            str(experiment_path),
            "--evidence",
            str(evidence_path),
            "--output",
            str(output),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["format"] == "orcacolony_experiment_result_v1"
    assert printed["outcome"] == "rejected"
    assert (output / "RESULT.md").is_file()


def test_record_cli_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    study_path = tmp_path / "study.json"
    study_path.write_text(
        '{"format":"orcacolony_study_v1","format":"shadowed"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key: format"):
        research.main(
            [
                "record",
                "--study",
                str(study_path),
                "--experiment",
                str(tmp_path / "unused-experiment.json"),
                "--evidence",
                str(tmp_path / "unused-evidence.json"),
                "--output",
                str(tmp_path / "unused-result"),
            ]
        )


def test_all_committed_research_records_build(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = Path(__file__).parents[1]
    study_paths = sorted(
        (repository_root / "research" / "studies").glob("*/study.json")
    )
    assert study_paths

    built = 0
    expected = 0
    for study_path in study_paths:
        study = json.loads(study_path.read_text(encoding="utf-8"))
        for reference in study["experiments"]:
            expected += 1
            experiment_id = reference["experiment_id"]
            experiment_path = study_path.parent / reference["manifest"]
            evidence_path = study_path.parent / "evidence" / f"{experiment_id}.json"
            output = tmp_path / study["study_id"] / experiment_id
            research.main(
                [
                    "record",
                    "--study",
                    str(study_path),
                    "--experiment",
                    str(experiment_path),
                    "--evidence",
                    str(evidence_path),
                    "--output",
                    str(output),
                ]
            )
            capsys.readouterr()
            assert (output / "result.json").is_file()
            assert (output / "RESULT.md").is_file()
            built += 1

    assert expected >= 2
    assert built == expected
