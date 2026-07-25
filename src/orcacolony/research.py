from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


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
