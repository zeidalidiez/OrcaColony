from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path


_STUDY_STATUSES = {
    "proposed",
    "active",
    "validated",
    "rejected",
    "inconclusive",
    "promoted",
}
_ID_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*\Z")
_DESCRIPTOR_FIELDS = {"id", "label", "description"}
_ARTIFACT_FIELDS = {"id", "kind", "revision", "uri"}
_MEASUREMENT_FIELDS = {"id", "label", "value", "unit"}


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a mapping with text keys")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    required: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - required)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _require_id(value: object, label: str) -> str:
    identifier = _require_text(value, label)
    if len(identifier) > 128 or _ID_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{label} must be a lowercase research identifier")
    return identifier


def _validate_descriptor(value: object, label: str) -> str:
    descriptor = _require_mapping(value, label)
    _require_exact_fields(descriptor, _DESCRIPTOR_FIELDS, label)
    identifier = _require_id(descriptor["id"], f"{label} id")
    _require_text(descriptor["label"], f"{label} label")
    _require_text(descriptor["description"], f"{label} description")
    return identifier


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    return value


def _validate_descriptor_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> set[str]:
    entries = _require_sequence(value, label)
    if not allow_empty and not entries:
        raise ValueError(f"{label} must not be empty")
    identifiers = {
        _validate_descriptor(entry, f"{label} entry") for entry in entries
    }
    if len(identifiers) != len(entries):
        raise ValueError(f"{label} contains duplicate ids")
    return identifiers


def _validate_suite(value: object, label: str) -> None:
    suite = _require_mapping(value, label)
    _require_exact_fields(suite, _DESCRIPTOR_FIELDS | {"revision"}, label)
    _require_id(suite["id"], f"{label} id")
    _require_text(suite["label"], f"{label} label")
    _require_text(suite["description"], f"{label} description")
    _require_text(suite["revision"], f"{label} revision")


def _validate_primary_metric(value: object) -> None:
    label = "study use_case primary_metric"
    metric = _require_mapping(value, label)
    _require_exact_fields(
        metric,
        _DESCRIPTOR_FIELDS
        | {"direction", "success_threshold", "unit"},
        label,
    )
    _require_id(metric["id"], f"{label} id")
    _require_text(metric["label"], f"{label} label")
    _require_text(metric["description"], f"{label} description")
    if metric["direction"] not in {"minimize", "maximize"}:
        raise ValueError(f"{label} direction must be minimize or maximize")
    threshold = metric["success_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError(f"{label} success_threshold must be a finite number")
    _require_text(metric["unit"], f"{label} unit")


def _validate_use_case(value: object) -> None:
    label = "study use_case"
    use_case = _require_mapping(value, label)
    _require_exact_fields(
        use_case,
        {
            "claim",
            "baseline",
            "primary_metric",
            "repeated_validation_suite",
            "final_holdout_suite",
            "guardrails",
        },
        label,
    )
    _require_text(use_case["claim"], f"{label} claim")
    _validate_descriptor(use_case["baseline"], f"{label} baseline")
    _validate_primary_metric(use_case["primary_metric"])
    _validate_suite(
        use_case["repeated_validation_suite"],
        f"{label} repeated_validation_suite",
    )
    _validate_suite(use_case["final_holdout_suite"], f"{label} final_holdout_suite")
    _validate_descriptor_list(
        use_case["guardrails"],
        f"{label} guardrails",
        allow_empty=True,
    )


def _validate_relative_json_path(value: object) -> None:
    path = _require_text(value, "experiment manifest")
    parts = path.split("/")
    if (
        "\\" in path
        or path.startswith("/")
        or ":" in path
        or not path.endswith(".json")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("experiment manifest must be a safe relative JSON path")


def _validate_experiment_references(value: object) -> None:
    label = "study experiments"
    entries = _require_sequence(value, label)
    if not entries:
        raise ValueError(f"{label} must not be empty")
    identifiers: set[str] = set()
    for entry_value in entries:
        entry = _require_mapping(entry_value, f"{label} entry")
        _require_exact_fields(
            entry,
            {"experiment_id", "role", "manifest"},
            f"{label} entry",
        )
        identifier = _require_id(
            entry["experiment_id"],
            f"{label} entry experiment_id",
        )
        if identifier in identifiers:
            raise ValueError(f"{label} contains duplicate experiment ids")
        identifiers.add(identifier)
        _require_id(entry["role"], f"{label} entry role")
        _validate_relative_json_path(entry["manifest"])


def validate_study_manifest(payload: Mapping[str, object]) -> None:
    study = _require_mapping(payload, "study")
    _require_exact_fields(
        study,
        {
            "format",
            "study_id",
            "title",
            "hypothesis",
            "status",
            "use_case",
            "independent_variables",
            "controlled_variables",
            "experiments",
        },
        "study",
    )
    if study["format"] != "orcacolony_study_v1":
        raise ValueError("unsupported study format")
    _require_id(study["study_id"], "study study_id")
    _require_text(study["title"], "study title")
    _require_text(study["hypothesis"], "study hypothesis")
    if study["status"] not in _STUDY_STATUSES:
        raise ValueError("study status is invalid")
    _validate_use_case(study["use_case"])
    _validate_descriptor_list(
        study["independent_variables"],
        "study independent_variables",
    )
    _validate_descriptor_list(
        study["controlled_variables"],
        "study controlled_variables",
    )
    _validate_experiment_references(study["experiments"])


def _validate_artifact_list(value: object, label: str) -> set[str]:
    artifacts = _require_sequence(value, label)
    if not artifacts:
        raise ValueError(f"{label} must not be empty")
    identifiers: set[str] = set()
    for artifact_value in artifacts:
        artifact = _require_mapping(artifact_value, f"{label} entry")
        _require_exact_fields(artifact, _ARTIFACT_FIELDS, f"{label} entry")
        identifier = _require_id(artifact["id"], f"{label} entry id")
        if identifier in identifiers:
            raise ValueError(f"{label} contains duplicate ids")
        identifiers.add(identifier)
        _require_id(artifact["kind"], f"{label} entry kind")
        _require_text(artifact["revision"], f"{label} entry revision")
        _require_text(artifact["uri"], f"{label} entry uri")
    return identifiers


def _validate_measurement(value: object, label: str) -> str:
    measurement = _require_mapping(value, label)
    _require_exact_fields(measurement, _MEASUREMENT_FIELDS, label)
    identifier = _require_id(measurement["id"], f"{label} id")
    _require_text(measurement["label"], f"{label} label")
    number = measurement["value"]
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
    ):
        raise ValueError(f"{label} value must be a finite number")
    _require_text(measurement["unit"], f"{label} unit")
    return identifier


def _validate_measurement_list(value: object, label: str) -> set[str]:
    measurements = _require_sequence(value, label)
    if not measurements:
        raise ValueError(f"{label} must not be empty")
    identifiers = {
        _validate_measurement(entry, f"{label} entry") for entry in measurements
    }
    if len(identifiers) != len(measurements):
        raise ValueError(f"{label} contains duplicate ids")
    return identifiers


def _validate_subject(value: object) -> None:
    label = "experiment subject"
    subject = _require_mapping(value, label)
    _require_exact_fields(subject, {"kind", "id", "revision"}, label)
    _require_id(subject["kind"], f"{label} kind")
    _require_id(subject["id"], f"{label} id")
    _require_text(subject["revision"], f"{label} revision")


def _validate_method(value: object) -> None:
    label = "experiment method"
    method = _require_mapping(value, label)
    _require_exact_fields(
        method,
        {
            "training_method",
            "execution_topology",
            "memory_profiles",
            "numerical_profiles",
        },
        label,
    )
    _validate_descriptor(method["training_method"], f"{label} training_method")
    _validate_descriptor(method["execution_topology"], f"{label} execution_topology")
    _validate_descriptor_list(method["memory_profiles"], f"{label} memory_profiles")
    _validate_descriptor_list(
        method["numerical_profiles"],
        f"{label} numerical_profiles",
    )


def _validate_resource_budget(value: object) -> None:
    label = "experiment resource_budget"
    budget = _require_mapping(value, label)
    _require_exact_fields(budget, {"limits"}, label)
    _validate_measurement_list(budget["limits"], f"{label} limits")


def _validate_reproduction(value: object) -> None:
    label = "experiment reproduction"
    reproduction = _require_mapping(value, label)
    _require_exact_fields(reproduction, {"command", "notes"}, label)
    command = _require_sequence(reproduction["command"], f"{label} command")
    if not command:
        raise ValueError(f"{label} command must not be empty")
    for index, argument in enumerate(command):
        _require_text(argument, f"{label} command argument {index}")
    _require_text(reproduction["notes"], f"{label} notes")


def validate_experiment_manifest(
    study_payload: Mapping[str, object],
    experiment_payload: Mapping[str, object],
) -> None:
    validate_study_manifest(study_payload)
    experiment = _require_mapping(experiment_payload, "experiment")
    _require_exact_fields(
        experiment,
        {
            "format",
            "study_id",
            "experiment_id",
            "title",
            "status",
            "subject",
            "artifacts",
            "method",
            "worker_profiles",
            "resource_budget",
            "reproduction",
        },
        "experiment",
    )
    if experiment["format"] != "orcacolony_experiment_v1":
        raise ValueError("unsupported experiment format")
    if experiment["study_id"] != study_payload["study_id"]:
        raise ValueError("experiment study_id does not match the study")
    experiment_id = _require_id(
        experiment["experiment_id"],
        "experiment experiment_id",
    )
    referenced_ids = {
        reference["experiment_id"]
        for reference in study_payload["experiments"]  # type: ignore[union-attr]
    }
    if experiment_id not in referenced_ids:
        raise ValueError("experiment is not referenced by the study")
    _require_text(experiment["title"], "experiment title")
    if experiment["status"] not in _STUDY_STATUSES:
        raise ValueError("experiment status is invalid")
    _validate_subject(experiment["subject"])
    _validate_artifact_list(experiment["artifacts"], "experiment artifacts")
    _validate_method(experiment["method"])
    _validate_descriptor_list(
        experiment["worker_profiles"],
        "experiment worker_profiles",
    )
    _validate_resource_budget(experiment["resource_budget"])
    _validate_reproduction(experiment["reproduction"])


def _require_timestamp(value: object) -> str:
    timestamp = _require_text(value, "evidence completed_at")
    if not timestamp.endswith("Z"):
        raise ValueError("evidence completed_at must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(
            "evidence completed_at must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.utcoffset() is None:
        raise ValueError("evidence completed_at must be an ISO-8601 UTC timestamp")
    return timestamp


def _validate_findings(value: object) -> None:
    label = "evidence findings"
    findings = _require_sequence(value, label)
    if not findings:
        raise ValueError(f"{label} must not be empty")
    identifiers: set[str] = set()
    for finding_value in findings:
        finding = _require_mapping(finding_value, f"{label} entry")
        _require_exact_fields(
            finding,
            {"id", "label", "description", "kind"},
            f"{label} entry",
        )
        identifier = _require_id(finding["id"], f"{label} entry id")
        if identifier in identifiers:
            raise ValueError(f"{label} contains duplicate ids")
        identifiers.add(identifier)
        _require_text(finding["label"], f"{label} entry label")
        _require_text(finding["description"], f"{label} entry description")
        if finding["kind"] not in {"positive", "negative", "neutral"}:
            raise ValueError(f"{label} entry kind is invalid")


def _validate_limitations(value: object) -> None:
    limitations = _require_sequence(value, "evidence limitations")
    if not limitations:
        raise ValueError("evidence limitations must not be empty")
    for index, limitation in enumerate(limitations):
        _require_text(limitation, f"evidence limitation {index}")


def _validate_evidence(
    study: Mapping[str, object],
    experiment: Mapping[str, object],
    evidence_payload: Mapping[str, object],
) -> tuple[bool, bool]:
    label = "evidence"
    evidence = _require_mapping(evidence_payload, label)
    _require_exact_fields(
        evidence,
        {
            "format",
            "study_id",
            "experiment_id",
            "outcome",
            "completed_at",
            "summary",
            "measurements",
            "evaluation",
            "findings",
            "limitations",
            "artifacts",
        },
        label,
    )
    if evidence["format"] != "orcacolony_experiment_evidence_v1":
        raise ValueError("unsupported evidence format")
    if evidence["study_id"] != study["study_id"]:
        raise ValueError("evidence study_id does not match the study")
    if evidence["experiment_id"] != experiment["experiment_id"]:
        raise ValueError("evidence experiment_id does not match the experiment")
    if evidence["outcome"] not in {
        "validated",
        "rejected",
        "inconclusive",
        "promoted",
    }:
        raise ValueError("evidence outcome is invalid")
    _require_timestamp(evidence["completed_at"])
    _require_text(evidence["summary"], "evidence summary")
    _validate_measurement_list(evidence["measurements"], "evidence measurements")
    _validate_findings(evidence["findings"])
    _validate_limitations(evidence["limitations"])
    _validate_artifact_list(evidence["artifacts"], "evidence artifacts")

    evaluation = _require_mapping(evidence["evaluation"], "evidence evaluation")
    _require_exact_fields(
        evaluation,
        {"primary_metric", "guardrails"},
        "evidence evaluation",
    )
    primary = _require_mapping(
        evaluation["primary_metric"],
        "evidence primary_metric",
    )
    _require_exact_fields(primary, {"id", "value"}, "evidence primary_metric")
    use_case = _require_mapping(study["use_case"], "study use_case")
    metric = _require_mapping(use_case["primary_metric"], "study primary_metric")
    if primary["id"] != metric["id"]:
        raise ValueError("evidence primary metric does not match the study")
    primary_value = primary["value"]
    if (
        isinstance(primary_value, bool)
        or not isinstance(primary_value, (int, float))
        or not math.isfinite(float(primary_value))
    ):
        raise ValueError("evidence primary metric value must be a finite number")
    threshold = float(metric["success_threshold"])  # type: ignore[arg-type]
    primary_passed = (
        float(primary_value) >= threshold
        if metric["direction"] == "maximize"
        else float(primary_value) <= threshold
    )

    guardrails = _require_sequence(
        evaluation["guardrails"],
        "evidence guardrails",
    )
    guardrail_results: dict[str, bool] = {}
    for guardrail_value in guardrails:
        guardrail = _require_mapping(guardrail_value, "evidence guardrail")
        _require_exact_fields(
            guardrail,
            {"id", "passed", "detail"},
            "evidence guardrail",
        )
        guardrail_id = _require_id(guardrail["id"], "evidence guardrail id")
        if guardrail_id in guardrail_results:
            raise ValueError("evidence guardrails contain duplicate ids")
        if not isinstance(guardrail["passed"], bool):
            raise ValueError("evidence guardrail passed must be boolean")
        _require_text(guardrail["detail"], "evidence guardrail detail")
        guardrail_results[guardrail_id] = guardrail["passed"]
    expected_guardrail_ids = {
        guardrail["id"]
        for guardrail in use_case["guardrails"]  # type: ignore[union-attr]
    }
    if set(guardrail_results) != expected_guardrail_ids:
        raise ValueError("guardrail evidence must exactly match the study guardrails")
    guardrails_passed = all(guardrail_results.values())
    if evidence["outcome"] in {"validated", "promoted"} and not (
        primary_passed and guardrails_passed
    ):
        raise ValueError("validated or promoted evidence must pass the use-case gate")
    return primary_passed, guardrails_passed


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _repo_artifact_payload(
    artifact: Mapping[str, object],
    repository_root: Path,
) -> tuple[Path, bytes] | None:
    uri = str(artifact["uri"])
    if not uri.startswith("repo:"):
        return None
    raw_relative = uri.removeprefix("repo:")
    relative = Path(raw_relative)
    if (
        relative.is_absolute()
        or raw_relative.startswith(("/", "\\"))
        or "\\" in raw_relative
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("research repo artifact path is unsafe")
    unresolved = repository_root / relative
    if any(
        candidate.is_symlink()
        for candidate in (
            repository_root.joinpath(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        )
    ):
        raise ValueError("research repo artifact path may not contain symlinks")
    source = unresolved.resolve()
    if not source.is_relative_to(repository_root):
        raise ValueError("research repo artifact escapes the repository")
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"research repo artifact is missing: {raw_relative}")
    raw_payload = source.read_bytes()
    kind = str(artifact["kind"])
    digest_payload = raw_payload
    if kind in {"campaign-json-sha256", "lora-manifest-sha256"}:
        parsed = json.loads(
            raw_payload,
            object_pairs_hook=_reject_duplicate_keys,
        )
        digest_payload = _canonical_json(
            _require_mapping(parsed, "research repo JSON artifact")
        )
    expected = str(artifact["revision"]).removeprefix("sha256:")
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or _sha256_bytes(digest_payload) != expected
    ):
        raise ValueError(
            f"research repo artifact digest mismatch: {raw_relative}"
        )
    return relative, raw_payload


def _resolve_repo_artifacts(
    experiment_payload: Mapping[str, object],
    evidence_payload: Mapping[str, object],
    repository_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[tuple[Path, bytes]],
]:
    resolved: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    snapshots: list[tuple[Path, bytes]] = []
    for category, artifacts in (
        ("input", experiment_payload["artifacts"]),
        ("evidence", evidence_payload["artifacts"]),
    ):
        for raw_artifact in _require_sequence(
            artifacts,
            f"{category} artifacts",
        ):
            artifact = _require_mapping(
                raw_artifact,
                f"{category} artifact",
            )
            resolved_payload = _repo_artifact_payload(
                artifact,
                repository_root,
            )
            if resolved_payload is None:
                unresolved.append(
                    {
                        "id": artifact["id"],
                        "category": category,
                        "kind": artifact["kind"],
                        "uri": artifact["uri"],
                        "declared_revision": artifact["revision"],
                        "status": "not_resolved_by_recorder",
                    }
                )
                continue
            source_relative, payload = resolved_payload
            bundle_relative = (
                Path("artifacts")
                / category
                / str(artifact["id"])
                / source_relative.name
            )
            snapshots.append((bundle_relative, payload))
            resolved.append(
                {
                    "id": artifact["id"],
                    "category": category,
                    "kind": artifact["kind"],
                    "uri": artifact["uri"],
                    "declared_revision": artifact["revision"],
                    "source_path": source_relative.as_posix(),
                    "bundle_path": bundle_relative.as_posix(),
                    "bundled_sha256": _sha256_bytes(payload),
                }
            )
    resolved.sort(key=lambda item: (str(item["category"]), str(item["id"])))
    unresolved.sort(
        key=lambda item: (str(item["category"]), str(item["id"]))
    )
    snapshots.sort(key=lambda item: item[0].as_posix())
    return resolved, unresolved, snapshots


def _environment_payload(repository_root: Path) -> dict[str, object]:
    distributions: dict[str, str | None] = {}
    for distribution in (
        "orcacolony",
        "torch",
        "numpy",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
    ):
        try:
            distributions[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    lock_path = repository_root / "uv.lock"
    lock_sha256 = (
        _sha256_bytes(lock_path.read_bytes())
        if lock_path.is_file() and not lock_path.is_symlink()
        else None
    )
    return {
        "format": "orcacolony_research_environment_v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": distributions,
        "uv_lock_sha256": lock_sha256,
    }


def _result_markdown(result: Mapping[str, object]) -> str:
    use_case = _require_mapping(result["use_case"], "result use_case")
    metric = _require_mapping(use_case["primary_metric"], "result primary_metric")
    evidence = _require_mapping(result["evidence"], "result evidence")
    evaluation = _require_mapping(evidence["evaluation"], "result evaluation")
    primary = _require_mapping(
        evaluation["primary_metric"],
        "result evaluated primary metric",
    )
    decision = _require_mapping(result["decision"], "result decision")
    lines = [
        f"# {result['title']}",
        "",
        f"**Study:** `{result['study_id']}`  ",
        f"**Experiment:** `{result['experiment_id']}`  ",
        f"**Outcome:** **{str(result['outcome']).upper()}**  ",
        f"**Completed:** {result['completed_at']}",
        "",
        "## Summary",
        "",
        str(result["summary"]),
        "",
        "## Hypothesis",
        "",
        str(result["hypothesis"]),
        "",
        "## Use-case evaluation",
        "",
        str(use_case["claim"]),
        "",
        (
            f"- Primary metric: **{metric['label']}** = `{primary['value']}` "
            f"{metric['unit']}"
        ),
        f"- Primary metric gate passed: **{decision['primary_metric_passed']}**",
        f"- Guardrails passed: **{decision['guardrails_passed']}**",
        f"- Overall use-case gate passed: **{decision['use_case_passed']}**",
        "",
        "## Findings",
        "",
    ]
    for finding_value in evidence["findings"]:  # type: ignore[union-attr]
        finding = _require_mapping(finding_value, "result finding")
        lines.extend(
            [
                f"### {finding['label']} ({finding['kind']})",
                "",
                str(finding["description"]),
                "",
            ]
        )
    lines.extend(["## Limitations", ""])
    for limitation in evidence["limitations"]:  # type: ignore[union-attr]
        lines.append(f"- {limitation}")
    reproduction = _require_mapping(result["reproduction"], "result reproduction")
    lines.extend(["", "## Measurements", ""])
    for measurement_value in evidence["measurements"]:  # type: ignore[union-attr]
        measurement = _require_mapping(
            measurement_value,
            "result measurement",
        )
        lines.append(
            f"- {measurement['label']}: `{measurement['value']}` "
            f"{measurement['unit']}"
        )
    resolved = _require_sequence(
        result["resolved_repo_artifacts"],
        "result resolved artifacts",
    )
    unresolved = _require_sequence(
        result["unresolved_artifacts"],
        "result unresolved artifacts",
    )
    lines.extend(
        [
            "",
            "## Provenance and evidence files",
            "",
            "- `environment.json` records the Python, platform, dependency, and "
            "lock-file context captured by the recorder.",
        ]
    )
    for artifact_value in resolved:
        artifact = _require_mapping(artifact_value, "resolved artifact")
        lines.append(
            f"- Verified `{artifact['id']}` and bundled it at "
            f"`{artifact['bundle_path']}` (SHA-256 "
            f"`{artifact['bundled_sha256']}`)."
        )
    for artifact_value in unresolved:
        artifact = _require_mapping(artifact_value, "unresolved artifact")
        lines.append(
            f"- Not locally resolved by the recorder: `{artifact['id']}` at "
            f"`{artifact['uri']}` with declared revision "
            f"`{artifact['declared_revision']}`."
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```json",
            json.dumps(reproduction["command"], ensure_ascii=False),
            "```",
            "",
            str(reproduction["notes"]),
            "",
        ]
    )
    return "\n".join(lines)


def build_result_bundle(
    study_payload: Mapping[str, object],
    experiment_payload: Mapping[str, object],
    evidence_payload: Mapping[str, object],
    output_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    validate_study_manifest(study_payload)
    validate_experiment_manifest(study_payload, experiment_payload)
    primary_passed, guardrails_passed = _validate_evidence(
        study_payload,
        experiment_payload,
        evidence_payload,
    )
    repository = Path(
        Path.cwd() if repository_root is None else repository_root
    ).resolve()
    (
        resolved_artifacts,
        unresolved_artifacts,
        artifact_snapshots,
    ) = _resolve_repo_artifacts(
        experiment_payload,
        evidence_payload,
        repository,
    )
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"research result output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        environment = _environment_payload(repository)
        sources = {
            "study.json": _canonical_json(study_payload),
            "experiment.json": _canonical_json(experiment_payload),
            "evidence.json": _canonical_json(evidence_payload),
            "environment.json": _canonical_json(environment),
        }
        source_revisions = {
            filename.removesuffix(".json"): _sha256_bytes(payload)
            for filename, payload in sources.items()
        }
        result: dict[str, object] = {
            "format": "orcacolony_experiment_result_v1",
            "study_id": study_payload["study_id"],
            "experiment_id": experiment_payload["experiment_id"],
            "title": experiment_payload["title"],
            "study_status": study_payload["status"],
            "experiment_status": experiment_payload["status"],
            "outcome": evidence_payload["outcome"],
            "completed_at": evidence_payload["completed_at"],
            "summary": evidence_payload["summary"],
            "hypothesis": study_payload["hypothesis"],
            "use_case": study_payload["use_case"],
            "independent_variables": study_payload["independent_variables"],
            "controlled_variables": study_payload["controlled_variables"],
            "subject": experiment_payload["subject"],
            "method": experiment_payload["method"],
            "worker_profiles": experiment_payload["worker_profiles"],
            "resource_budget": experiment_payload["resource_budget"],
            "input_artifacts": experiment_payload["artifacts"],
            "evidence": {
                "measurements": evidence_payload["measurements"],
                "evaluation": evidence_payload["evaluation"],
                "findings": evidence_payload["findings"],
                "limitations": evidence_payload["limitations"],
                "artifacts": evidence_payload["artifacts"],
            },
            "reproduction": experiment_payload["reproduction"],
            "decision": {
                "primary_metric_passed": primary_passed,
                "guardrails_passed": guardrails_passed,
                "use_case_passed": primary_passed and guardrails_passed,
            },
            "source_revisions": source_revisions,
            "environment": environment,
            "resolved_repo_artifacts": resolved_artifacts,
            "unresolved_artifacts": unresolved_artifacts,
        }
        for filename, payload in sources.items():
            _write_bytes(temporary / filename, payload)
        _write_bytes(temporary / "result.json", _canonical_json(result))
        _write_bytes(
            temporary / "RESULT.md",
            _result_markdown(result).encode("utf-8"),
        )
        for relative, payload in artifact_snapshots:
            _write_bytes(temporary / relative, payload)
        checksum_files = sorted(
            path for path in temporary.rglob("*") if path.is_file()
        )
        checksums = "".join(
            f"{_sha256_bytes(path.read_bytes())}  "
            f"{path.relative_to(temporary).as_posix()}\n"
            for path in checksum_files
        )
        _write_bytes(temporary / "SHA256SUMS", checksums.encode("utf-8"))
        os.replace(temporary, output)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _load_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    return _require_mapping(payload, label)


def _validate_experiment_location(
    study_path: Path,
    study: Mapping[str, object],
    experiment_path: Path,
    experiment: Mapping[str, object],
) -> None:
    experiment_id = experiment["experiment_id"]
    reference = next(
        entry
        for entry in study["experiments"]  # type: ignore[union-attr]
        if entry["experiment_id"] == experiment_id
    )
    expected = (study_path.parent / str(reference["manifest"])).resolve()
    if expected != experiment_path.resolve():
        raise ValueError("experiment path does not match the study reference")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and record reproducible OrcaColony research results"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser(
        "record",
        help="build a deterministic result bundle from linked research manifests",
    )
    record.add_argument("--study", type=Path, required=True)
    record.add_argument("--experiment", type=Path, required=True)
    record.add_argument("--evidence", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="root used to resolve and verify repo: artifact URIs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command != "record":
        raise ValueError(f"unsupported research command: {args.command}")
    study = _load_json_mapping(args.study, "study")
    experiment = _load_json_mapping(args.experiment, "experiment")
    evidence = _load_json_mapping(args.evidence, "evidence")
    validate_study_manifest(study)
    validate_experiment_manifest(study, experiment)
    _validate_experiment_location(args.study, study, args.experiment, experiment)
    result = build_result_bundle(
        study,
        experiment,
        evidence,
        args.output,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
