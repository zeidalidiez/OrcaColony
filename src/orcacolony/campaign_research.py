from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


_ID_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*\Z")
_DIRECTIONS = {"maximize", "minimize", "observe"}
_FINDING_KINDS = {
    "improvement",
    "regression",
    "unchanged",
    "mixed",
    "inconclusive",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{label} must be a mapping with text keys")
    return value


def _sequence(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise ValueError(f"{label} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _exact_fields(
    payload: Mapping[str, object],
    fields: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - fields)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(payload))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")


def _text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in ("\x00", "\r"))
    ):
        raise ValueError(f"{label} must be nonempty text")
    return value


def _uri(value: object, label: str) -> str:
    uri = _text(value, label)
    if "\n" in uri:
        raise ValueError(f"{label} must be one line")
    return uri


def _identifier(value: object, label: str) -> str:
    identifier = _text(value, label)
    if len(identifier) > 128 or _ID_PATTERN.fullmatch(identifier) is None:
        raise ValueError(f"{label} must be a lowercase campaign identifier")
    return identifier


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: object, label: str) -> str:
    revision = _text(value, label)
    digest = revision.removeprefix("sha256:")
    expected_length = 64 if revision.startswith("sha256:") else 40
    if (
        len(digest) != expected_length
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            f"{label} must be an exact Git commit or sha256: digest"
        )
    return revision


def _subject_revision(value: object, label: str) -> str:
    revision = _text(value, label)
    if (
        len(revision) == 64
        and all(character in "0123456789abcdef" for character in revision)
    ):
        return revision
    return _revision(revision, label)


def _command(value: object, label: str) -> list[str]:
    command = _sequence(value, label)
    return [
        _text(argument, f"{label} argument {index}")
        for index, argument in enumerate(command)
    ]


def _finite_number(value: object, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate campaign evaluation JSON key: {key}")
        payload[key] = value
    return payload


def load_campaign_evaluation_evidence(
    path: str | Path,
) -> Mapping[str, object]:
    """Load one unambiguous campaign evaluation evidence document."""

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return _mapping(payload, "campaign evaluation evidence")


def campaign_research_revision(payload: Mapping[str, object]) -> str:
    """Return the content identity of a validated campaign research contract."""

    validate_campaign_research_contract(payload)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_artifact_contracts(value: object) -> None:
    entries = _sequence(value, "campaign research evaluation artifacts")
    identifiers: set[str] = set()
    for raw_entry in entries:
        entry = _mapping(raw_entry, "campaign research evaluation artifact")
        _exact_fields(
            entry,
            {"id", "kind", "revision", "uri"},
            "campaign research evaluation artifact",
        )
        identifier = _identifier(
            entry["id"],
            "campaign research evaluation artifact id",
        )
        if identifier in identifiers:
            raise ValueError(
                "campaign research evaluation artifact IDs must be unique"
            )
        identifiers.add(identifier)
        _identifier(
            entry["kind"],
            "campaign research evaluation artifact kind",
        )
        _revision(
            entry["revision"],
            "campaign research evaluation artifact revision",
        )
        _uri(entry["uri"], "campaign research evaluation artifact URI")


def _validate_metrics(value: object) -> dict[str, Mapping[str, object]]:
    entries = _sequence(value, "campaign research metrics")
    metrics: dict[str, Mapping[str, object]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "campaign research metric")
        _exact_fields(
            entry,
            {"id", "label", "description", "direction", "unit"},
            "campaign research metric",
        )
        identifier = _identifier(entry["id"], "campaign research metric id")
        if identifier in metrics:
            raise ValueError("campaign research metric IDs must be unique")
        _text(entry["label"], "campaign research metric label")
        _text(entry["description"], "campaign research metric description")
        if entry["direction"] not in _DIRECTIONS:
            raise ValueError(
                "campaign research metric direction must be maximize, "
                "minimize, or observe"
            )
        _text(entry["unit"], "campaign research metric unit")
        metrics[identifier] = entry
    return metrics


def validate_campaign_research_contract(
    payload: Mapping[str, object],
) -> None:
    """Validate choices supplied by a campaign owner without supplying them."""

    research = _mapping(payload, "campaign research")
    _exact_fields(
        research,
        {
            "format",
            "question",
            "usage_scenario",
            "evaluation_contract",
            "analysis_plan",
        },
        "campaign research",
    )
    if research["format"] != "orcacolony_campaign_research_v2":
        raise ValueError("unsupported campaign research contract")
    _text(research["question"], "campaign research question")
    _text(research["usage_scenario"], "campaign research usage scenario")

    evaluation = _mapping(
        research["evaluation_contract"],
        "campaign research evaluation contract",
    )
    _exact_fields(
        evaluation,
        {"evaluator", "artifacts", "metrics"},
        "campaign research evaluation contract",
    )
    evaluator = _mapping(
        evaluation["evaluator"],
        "campaign research evaluator",
    )
    _exact_fields(
        evaluator,
        {"id", "revision", "command"},
        "campaign research evaluator",
    )
    _identifier(evaluator["id"], "campaign research evaluator id")
    _revision(evaluator["revision"], "campaign research evaluator revision")
    _command(evaluator["command"], "campaign research evaluator command")
    _validate_artifact_contracts(evaluation["artifacts"])
    _validate_metrics(evaluation["metrics"])

    plan = _sequence(research["analysis_plan"], "campaign research analysis plan")
    for index, item in enumerate(plan):
        _text(item, f"campaign research analysis plan item {index}")


def _validate_evidence_artifacts(
    value: object,
    label: str,
) -> list[dict[str, object]]:
    entries = _sequence(value, label)
    artifacts: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for raw_entry in entries:
        entry = _mapping(raw_entry, f"{label} entry")
        _exact_fields(
            entry,
            {"id", "sha256", "uri"},
            f"{label} entry",
        )
        identifier = _identifier(entry["id"], f"{label} entry id")
        if identifier in identifiers:
            raise ValueError(f"{label} IDs must be unique")
        identifiers.add(identifier)
        artifacts.append(
            {
                "id": identifier,
                "sha256": _sha256(
                    entry["sha256"],
                    f"{label} entry SHA-256",
                ),
                "uri": _uri(entry["uri"], f"{label} entry URI"),
            }
        )
    return artifacts


def _validate_evaluations(
    value: object,
    metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    entries = _sequence(value, "campaign evaluation evidence evaluations")
    evaluations: dict[str, dict[str, object]] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "campaign evaluation evidence evaluation")
        _exact_fields(
            entry,
            {"id", "label", "subject", "measurements", "artifacts"},
            "campaign evaluation evidence evaluation",
        )
        identifier = _identifier(
            entry["id"],
            "campaign evaluation evidence evaluation id",
        )
        if identifier in evaluations:
            raise ValueError("campaign evaluation IDs must be unique")
        label = _text(
            entry["label"],
            "campaign evaluation evidence evaluation label",
        )
        subject = _mapping(
            entry["subject"],
            "campaign evaluation evidence subject",
        )
        _exact_fields(
            subject,
            {"id", "label", "revision"},
            "campaign evaluation evidence subject",
        )
        normalized_subject = {
            "id": _identifier(
                subject["id"],
                "campaign evaluation evidence subject id",
            ),
            "label": _text(
                subject["label"],
                "campaign evaluation evidence subject label",
            ),
            "revision": _subject_revision(
                subject["revision"],
                "campaign evaluation evidence subject revision",
            ),
        }

        raw_measurements = _sequence(
            entry["measurements"],
            "campaign evaluation evidence measurements",
        )
        measurements: dict[str, int | float] = {}
        for raw_measurement in raw_measurements:
            measurement = _mapping(
                raw_measurement,
                "campaign evaluation evidence measurement",
            )
            _exact_fields(
                measurement,
                {"metric_id", "value"},
                "campaign evaluation evidence measurement",
            )
            metric_id = _identifier(
                measurement["metric_id"],
                "campaign evaluation evidence measurement metric id",
            )
            if metric_id not in metrics:
                raise ValueError(
                    "campaign evaluation evidence contains an undeclared metric"
                )
            if metric_id in measurements:
                raise ValueError(
                    "campaign evaluation evidence contains duplicate metrics"
                )
            measurements[metric_id] = _finite_number(
                measurement["value"],
                "campaign evaluation evidence measurement value",
            )
        if set(measurements) != set(metrics):
            raise ValueError(
                "campaign evaluation evidence must measure every declared metric"
            )
        evaluations[identifier] = {
            "id": identifier,
            "label": label,
            "subject": normalized_subject,
            "measurements": measurements,
            "artifacts": _validate_evidence_artifacts(
                entry["artifacts"],
                "campaign evaluation evidence artifacts",
            ),
        }
    return evaluations


def _comparison_metrics(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    metrics: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline_values = _mapping(
        baseline["measurements"],
        "baseline measurements",
    )
    candidate_values = _mapping(
        candidate["measurements"],
        "candidate measurements",
    )
    compared: list[dict[str, object]] = []
    for metric_id, metric in metrics.items():
        before = baseline_values[metric_id]
        after = candidate_values[metric_id]
        absolute_change = float(after) - float(before)  # type: ignore[arg-type]
        direction = metric["direction"]
        preferred_direction_change: float | None
        if direction == "maximize":
            preferred_direction_change = absolute_change
        elif direction == "minimize":
            preferred_direction_change = -absolute_change
        else:
            preferred_direction_change = None
        compared.append(
            {
                "metric_id": metric_id,
                "label": metric["label"],
                "unit": metric["unit"],
                "direction": direction,
                "baseline_value": before,
                "candidate_value": after,
                "absolute_change": absolute_change,
                "change_in_preferred_direction": preferred_direction_change,
            }
        )
    return compared


def build_campaign_evaluation_summary(
    research: Mapping[str, object],
    evidence_payload: Mapping[str, object],
    *,
    campaign_id: str,
    campaign_revision: str,
    release_checkpoint_sha256: str,
) -> dict[str, object]:
    """Validate owner-supplied evidence and compute comparisons without a gate."""

    validate_campaign_research_contract(research)
    evidence = _mapping(evidence_payload, "campaign evaluation evidence")
    _exact_fields(
        evidence,
        {
            "format",
            "campaign_id",
            "campaign_revision",
            "research_revision",
            "release_evaluation_id",
            "evaluations",
            "comparisons",
            "findings",
            "limitations",
            "reproduction",
        },
        "campaign evaluation evidence",
    )
    if evidence["format"] != "orcacolony_campaign_evaluation_evidence_v1":
        raise ValueError("unsupported campaign evaluation evidence")
    if evidence["campaign_id"] != campaign_id:
        raise ValueError("campaign evaluation evidence campaign ID differs")
    _sha256(campaign_revision, "campaign revision")
    if evidence["campaign_revision"] != campaign_revision:
        raise ValueError("campaign evaluation evidence campaign revision differs")
    expected_research_revision = campaign_research_revision(research)
    if evidence["research_revision"] != expected_research_revision:
        raise ValueError("campaign evaluation evidence research revision differs")

    evaluation_contract = _mapping(
        research["evaluation_contract"],
        "campaign research evaluation contract",
    )
    metrics = _validate_metrics(evaluation_contract["metrics"])
    evaluations = _validate_evaluations(evidence["evaluations"], metrics)
    release_evaluation_id = _identifier(
        evidence["release_evaluation_id"],
        "campaign evaluation release evaluation id",
    )
    if release_evaluation_id not in evaluations:
        raise ValueError("release evaluation is absent from campaign evidence")
    release_subject = _mapping(
        evaluations[release_evaluation_id]["subject"],
        "campaign evaluation release subject",
    )
    if release_subject["revision"] != release_checkpoint_sha256:
        raise ValueError(
            "release evaluation is not bound to the released checkpoint"
        )

    raw_comparisons = _sequence(
        evidence["comparisons"],
        "campaign evaluation comparisons",
        allow_empty=True,
    )
    comparisons: list[dict[str, object]] = []
    comparison_ids: set[str] = set()
    for raw_comparison in raw_comparisons:
        comparison = _mapping(
            raw_comparison,
            "campaign evaluation comparison",
        )
        _exact_fields(
            comparison,
            {
                "id",
                "baseline_evaluation_id",
                "candidate_evaluation_id",
                "summary",
            },
            "campaign evaluation comparison",
        )
        identifier = _identifier(
            comparison["id"],
            "campaign evaluation comparison id",
        )
        if identifier in comparison_ids:
            raise ValueError("campaign evaluation comparison IDs must be unique")
        comparison_ids.add(identifier)
        baseline_id = _identifier(
            comparison["baseline_evaluation_id"],
            "campaign evaluation comparison baseline id",
        )
        candidate_id = _identifier(
            comparison["candidate_evaluation_id"],
            "campaign evaluation comparison candidate id",
        )
        if baseline_id == candidate_id:
            raise ValueError(
                "campaign evaluation comparison subjects must differ"
            )
        if baseline_id not in evaluations or candidate_id not in evaluations:
            raise ValueError(
                "campaign evaluation comparison references an unknown evaluation"
            )
        comparisons.append(
            {
                "id": identifier,
                "baseline_evaluation_id": baseline_id,
                "candidate_evaluation_id": candidate_id,
                "summary": _text(
                    comparison["summary"],
                    "campaign evaluation comparison summary",
                ),
                "metrics": _comparison_metrics(
                    evaluations[baseline_id],
                    evaluations[candidate_id],
                    metrics,
                ),
            }
        )

    raw_findings = _sequence(
        evidence["findings"],
        "campaign evaluation findings",
    )
    findings: list[dict[str, str]] = []
    finding_ids: set[str] = set()
    for raw_finding in raw_findings:
        finding = _mapping(raw_finding, "campaign evaluation finding")
        _exact_fields(
            finding,
            {"id", "label", "kind", "description"},
            "campaign evaluation finding",
        )
        identifier = _identifier(finding["id"], "campaign evaluation finding id")
        if identifier in finding_ids:
            raise ValueError("campaign evaluation finding IDs must be unique")
        finding_ids.add(identifier)
        kind = finding["kind"]
        if kind not in _FINDING_KINDS:
            raise ValueError("campaign evaluation finding kind is invalid")
        findings.append(
            {
                "id": identifier,
                "label": _text(
                    finding["label"],
                    "campaign evaluation finding label",
                ),
                "kind": str(kind),
                "description": _text(
                    finding["description"],
                    "campaign evaluation finding description",
                ),
            }
        )

    raw_limitations = _sequence(
        evidence["limitations"],
        "campaign evaluation limitations",
    )
    limitations = [
        _text(item, f"campaign evaluation limitation {index}")
        for index, item in enumerate(raw_limitations)
    ]
    reproduction = _mapping(
        evidence["reproduction"],
        "campaign evaluation reproduction",
    )
    _exact_fields(
        reproduction,
        {"command", "notes"},
        "campaign evaluation reproduction",
    )
    normalized_reproduction = {
        "command": _command(
            reproduction["command"],
            "campaign evaluation reproduction command",
        ),
        "notes": _text(
            reproduction["notes"],
            "campaign evaluation reproduction notes",
        ),
    }

    return {
        "format": "orcacolony_campaign_evaluation_summary_v1",
        "campaign_id": campaign_id,
        "campaign_revision": campaign_revision,
        "research_revision": expected_research_revision,
        "release_evaluation_id": release_evaluation_id,
        "evaluations": list(evaluations.values()),
        "comparisons": comparisons,
        "findings": findings,
        "limitations": limitations,
        "reproduction": normalized_reproduction,
    }


def campaign_evaluation_release_revision(
    research: Mapping[str, object],
    evidence_payload: Mapping[str, object],
    *,
    campaign_id: str,
    campaign_revision: str,
) -> str:
    """Return the owner-declared release subject after full evidence validation."""

    evidence = _mapping(evidence_payload, "campaign evaluation evidence")
    release_evaluation_id = _identifier(
        evidence.get("release_evaluation_id"),
        "campaign evaluation release evaluation id",
    )
    release_revision: str | None = None
    for raw_evaluation in _sequence(
        evidence.get("evaluations"),
        "campaign evaluation evidence evaluations",
    ):
        evaluation = _mapping(
            raw_evaluation,
            "campaign evaluation evidence evaluation",
        )
        if evaluation.get("id") != release_evaluation_id:
            continue
        subject = _mapping(
            evaluation.get("subject"),
            "campaign evaluation evidence subject",
        )
        release_revision = _subject_revision(
            subject.get("revision"),
            "campaign evaluation evidence subject revision",
        )
        break
    if release_revision is None:
        raise ValueError("release evaluation is absent from campaign evidence")

    build_campaign_evaluation_summary(
        research,
        evidence,
        campaign_id=campaign_id,
        campaign_revision=campaign_revision,
        release_checkpoint_sha256=release_revision,
    )
    return release_revision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_campaign_evaluation_artifacts(
    evidence_payload: Mapping[str, object],
    artifact_root: str | Path | None,
) -> dict[str, str]:
    """Verify every local ``bundle:`` artifact and return its bound digest."""

    evidence = _mapping(evidence_payload, "campaign evaluation evidence")
    bindings: list[tuple[str, str]] = []
    for raw_evaluation in _sequence(
        evidence.get("evaluations"),
        "campaign evaluation evidence evaluations",
    ):
        evaluation = _mapping(
            raw_evaluation,
            "campaign evaluation evidence evaluation",
        )
        for raw_artifact in _sequence(
            evaluation.get("artifacts"),
            "campaign evaluation evidence artifacts",
        ):
            artifact = _mapping(
                raw_artifact,
                "campaign evaluation evidence artifact",
            )
            uri = artifact.get("uri")
            if not isinstance(uri, str) or not uri.startswith("bundle:"):
                continue
            bindings.append(
                (
                    uri.removeprefix("bundle:"),
                    _sha256(
                        artifact.get("sha256"),
                        "campaign evaluation evidence artifact SHA-256",
                    ),
                )
            )

    if not bindings:
        return {}
    if artifact_root is None:
        raise ValueError(
            "bundled campaign evaluation artifacts require an artifact root"
        )
    unresolved_root = Path(artifact_root)
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise ValueError(
            "campaign evaluation artifact root must be a regular directory"
        )
    root = unresolved_root.resolve()
    verified: dict[str, str] = {}
    for raw_relative, expected_sha256 in bindings:
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or raw_relative.startswith(("/", "\\"))
            or "\\" in raw_relative
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("campaign evaluation artifact path is unsafe")
        unresolved = root / relative
        if any(
            candidate.is_symlink()
            for candidate in (
                root.joinpath(*relative.parts[:index])
                for index in range(1, len(relative.parts) + 1)
            )
        ):
            raise ValueError(
                "campaign evaluation artifact path may not contain symlinks"
            )
        source = unresolved.resolve()
        if (
            not source.is_relative_to(root)
            or not source.is_file()
            or source.is_symlink()
        ):
            raise ValueError(
                f"campaign evaluation artifact is missing: {raw_relative}"
            )
        if _sha256_file(source) != expected_sha256:
            raise ValueError(
                f"campaign evaluation artifact digest differs: {raw_relative}"
            )
        relative_name = relative.as_posix()
        previous = verified.get(relative_name)
        if previous is not None and previous != expected_sha256:
            raise ValueError(
                "campaign evaluation artifact path has conflicting digests"
            )
        verified[relative_name] = expected_sha256
    return dict(sorted(verified.items()))
