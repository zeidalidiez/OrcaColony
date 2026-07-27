from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from urllib.parse import urlsplit


_LEDGER_FORMAT = "orcacolony_auxiliary_contributions_v1"
_SNAPSHOT_FORMAT = "orcacolony_auxiliary_contribution_snapshot_v1"
_CONTRIBUTION_STATUSES = {
    "completed",
    "partial",
    "failed_informative",
}
_URI_SCHEME = re.compile(r"[a-z][a-z0-9+.-]*\Z")


@dataclass(frozen=True)
class AuxiliaryContributionLedger:
    campaign_id: str
    campaign_revision: str
    revision: str
    _canonical_payload: bytes

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        campaign_id: str,
        campaign_revision: str,
    ) -> AuxiliaryContributionLedger:
        _exact_fields(
            payload,
            {
                "format",
                "campaign_id",
                "campaign_revision",
                "owner_reviewed",
                "contributors",
            },
            "auxiliary contribution ledger",
        )
        if payload.get("format") != _LEDGER_FORMAT:
            raise ValueError("unsupported auxiliary contribution ledger format")
        expected_campaign_id = _identifier(campaign_id, "campaign id")
        if payload.get("campaign_id") != expected_campaign_id:
            raise ValueError("auxiliary contribution ledger campaign ID differs")
        expected_campaign_revision = _sha256(
            campaign_revision,
            "campaign revision",
        )
        if payload.get("campaign_revision") != expected_campaign_revision:
            raise ValueError(
                "auxiliary contribution ledger campaign revision differs"
            )
        if payload.get("owner_reviewed") is not True:
            raise ValueError(
                "auxiliary contribution ledger must be explicitly owner reviewed"
            )

        raw_contributors = _sequence(
            payload.get("contributors"),
            "auxiliary contributors",
            allow_empty=True,
        )
        contributors: list[dict[str, object]] = []
        contributor_ids: set[str] = set()
        contribution_ids: set[str] = set()
        for raw_contributor in raw_contributors:
            contributor = _mapping(
                raw_contributor,
                "auxiliary contributor",
            )
            _exact_fields(
                contributor,
                {
                    "contributor_id",
                    "credit",
                    "resources",
                    "contributions",
                },
                "auxiliary contributor",
            )
            contributor_id = _identifier(
                contributor.get("contributor_id"),
                "auxiliary contributor id",
            )
            if contributor_id in contributor_ids:
                raise ValueError("auxiliary contributor IDs must be unique")
            contributor_ids.add(contributor_id)
            credit = _normalize_credit(
                contributor.get("credit"),
                contributor_id,
            )
            resources = _normalize_resources(
                contributor.get("resources"),
                contributor_id,
                credit,
            )
            raw_contributions = _sequence(
                contributor.get("contributions"),
                f"auxiliary contributor {contributor_id} contributions",
            )
            contributions = [
                _normalize_contribution(
                    raw_contribution,
                    contributor_id,
                    contribution_ids,
                )
                for raw_contribution in raw_contributions
            ]
            contributions.sort(key=lambda item: str(item["id"]))
            contributors.append(
                {
                    "contributor_id": contributor_id,
                    "credit": credit,
                    "resources": resources,
                    "contributions": contributions,
                }
            )

        contributors.sort(key=lambda item: str(item["contributor_id"]))
        normalized: dict[str, object] = {
            "format": _LEDGER_FORMAT,
            "campaign_id": expected_campaign_id,
            "campaign_revision": expected_campaign_revision,
            "owner_reviewed": True,
            "contributors": contributors,
        }
        canonical = _canonical_json(normalized)
        return cls(
            campaign_id=expected_campaign_id,
            campaign_revision=expected_campaign_revision,
            revision=hashlib.sha256(canonical).hexdigest(),
            _canonical_payload=canonical,
        )

    def as_payload(self) -> dict[str, object]:
        payload = json.loads(self._canonical_payload)
        if not isinstance(payload, dict):
            raise AssertionError("normalized auxiliary ledger is not an object")
        return payload


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} schema is invalid")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[object]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a non-empty list")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty identifier")
    normalized = value.strip()
    if len(normalized) > 128 or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in normalized
    ):
        raise ValueError(f"{label} contains unsupported characters")
    return normalized


def _text(value: object, label: str, *, limit: int = 2048) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{label} must be non-empty single-line text")
    normalized = value.strip()
    if len(normalized) > limit:
        raise ValueError(f"{label} is too long")
    return normalized


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label, limit=256)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer or null")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _optional_https_url(value: object, label: str) -> str | None:
    normalized = _optional_text(value, label)
    if normalized is None:
        return None
    parsed = urlsplit(normalized)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} contains an invalid port") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(character.isspace() for character in normalized)
        or any(character in '<>"\'' for character in normalized)
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(
            f"{label} must be a durable HTTPS URL without credentials, "
            "query, or fragment"
        )
    return normalized


def _safe_relative_artifact_path(value: str, label: str) -> str:
    relative = Path(value)
    if (
        relative.is_absolute()
        or value.startswith(("/", "\\"))
        or "\\" in value
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} is unsafe")
    return relative.as_posix()


def _artifact_uri(value: object, label: str) -> str:
    uri = _text(value, label, limit=2048)
    if any(character.isspace() for character in uri) or any(
        character in '<>"\'' for character in uri
    ):
        raise ValueError(f"{label} contains unsupported characters")
    scheme, separator, remainder = uri.partition(":")
    if separator != ":" or _URI_SCHEME.fullmatch(scheme) is None:
        raise ValueError(f"{label} must have a lowercase URI scheme")
    if scheme in {"bundle", "repo"}:
        _safe_relative_artifact_path(remainder, f"{label} path")
        return uri
    if scheme in {"data", "file", "http", "javascript", "mailto"}:
        raise ValueError(f"{label} uses an unsupported URI scheme")
    parsed = urlsplit(uri)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{label} must not contain credentials, query, or fragment"
        )
    if scheme == "https" and parsed.hostname is None:
        raise ValueError(f"{label} must contain an HTTPS hostname")
    if scheme == "hf" and (not parsed.netloc or not parsed.path.strip("/")):
        raise ValueError(f"{label} must identify a Hugging Face artifact")
    return uri


def _normalize_credit(
    value: object,
    contributor_id: str,
) -> dict[str, object]:
    credit = _mapping(value, f"auxiliary contributor {contributor_id} credit")
    _exact_fields(
        credit,
        {
            "visibility",
            "display_name",
            "profile_url",
            "team",
            "show_contribution_details",
            "show_time",
            "show_hardware",
            "public_disclosure_confirmed",
        },
        f"auxiliary contributor {contributor_id} credit",
    )
    visibility = credit.get("visibility")
    if visibility not in {"named", "pseudonymous", "anonymous"}:
        raise ValueError(
            f"auxiliary contributor {contributor_id} visibility is invalid"
        )
    display_name = _optional_text(
        credit.get("display_name"),
        f"auxiliary contributor {contributor_id} display name",
    )
    if visibility == "anonymous" and display_name is not None:
        raise ValueError(
            f"auxiliary contributor {contributor_id} anonymous credit "
            "may not publish a name"
        )
    if visibility != "anonymous" and display_name is None:
        raise ValueError(
            f"auxiliary contributor {contributor_id} public credit needs "
            "a display name"
        )
    profile_url = _optional_https_url(
        credit.get("profile_url"),
        f"auxiliary contributor {contributor_id} profile URL",
    )
    team = _optional_text(
        credit.get("team"),
        f"auxiliary contributor {contributor_id} team",
    )
    boolean_fields: dict[str, bool] = {}
    for name in (
        "show_contribution_details",
        "show_time",
        "show_hardware",
        "public_disclosure_confirmed",
    ):
        raw = credit.get(name)
        if type(raw) is not bool:
            raise ValueError(
                f"auxiliary contributor {contributor_id} {name} "
                "must be boolean"
            )
        boolean_fields[name] = raw
    if not boolean_fields["public_disclosure_confirmed"]:
        raise ValueError(
            f"auxiliary contributor {contributor_id} public disclosure "
            "must be confirmed"
        )
    if visibility == "anonymous" and (
        profile_url is not None
        or team is not None
        or boolean_fields["show_contribution_details"]
        or boolean_fields["show_time"]
        or boolean_fields["show_hardware"]
    ):
        raise ValueError(
            f"auxiliary contributor {contributor_id} anonymous credit "
            "may not publish profile, work, time, or hardware details"
        )
    return {
        "visibility": visibility,
        "display_name": display_name,
        "profile_url": profile_url,
        "team": team,
        **boolean_fields,
    }


def _normalize_resources(
    value: object,
    contributor_id: str,
    credit: Mapping[str, object],
) -> dict[str, object]:
    resources = _mapping(
        value,
        f"auxiliary contributor {contributor_id} resources",
    )
    _exact_fields(
        resources,
        {"person_time_seconds", "compute_time_seconds", "hardware"},
        f"auxiliary contributor {contributor_id} resources",
    )
    person_time = _optional_positive_int(
        resources.get("person_time_seconds"),
        f"auxiliary contributor {contributor_id} person time",
    )
    compute_time = _optional_positive_int(
        resources.get("compute_time_seconds"),
        f"auxiliary contributor {contributor_id} compute time",
    )
    raw_hardware = _sequence(
        resources.get("hardware"),
        f"auxiliary contributor {contributor_id} hardware",
        allow_empty=True,
    )
    hardware = [
        _text(
            item,
            f"auxiliary contributor {contributor_id} hardware entry",
            limit=256,
        )
        for item in raw_hardware
    ]
    if len(set(hardware)) != len(hardware):
        raise ValueError(
            f"auxiliary contributor {contributor_id} hardware entries "
            "must be unique"
        )
    hardware.sort(key=str.casefold)
    if credit["show_time"] and person_time is None and compute_time is None:
        raise ValueError(
            f"auxiliary contributor {contributor_id} cannot publish absent time"
        )
    if credit["show_hardware"] and not hardware:
        raise ValueError(
            f"auxiliary contributor {contributor_id} cannot publish absent hardware"
        )
    return {
        "person_time_seconds": person_time,
        "compute_time_seconds": compute_time,
        "hardware": hardware,
    }


def _normalize_contribution(
    value: object,
    contributor_id: str,
    contribution_ids: set[str],
) -> dict[str, object]:
    contribution = _mapping(
        value,
        f"auxiliary contributor {contributor_id} contribution",
    )
    _exact_fields(
        contribution,
        {"id", "kind", "description", "status", "evidence"},
        f"auxiliary contributor {contributor_id} contribution",
    )
    contribution_id = _identifier(
        contribution.get("id"),
        f"auxiliary contributor {contributor_id} contribution id",
    )
    if contribution_id in contribution_ids:
        raise ValueError("auxiliary contribution IDs must be unique")
    contribution_ids.add(contribution_id)
    kind = _identifier(
        contribution.get("kind"),
        f"auxiliary contribution {contribution_id} kind",
    )
    status = contribution.get("status")
    if status not in _CONTRIBUTION_STATUSES:
        raise ValueError(
            f"auxiliary contribution {contribution_id} status is invalid"
        )
    raw_evidence = _sequence(
        contribution.get("evidence"),
        f"auxiliary contribution {contribution_id} evidence",
    )
    evidence: list[dict[str, str]] = []
    evidence_ids: set[str] = set()
    for raw_artifact in raw_evidence:
        artifact = _mapping(
            raw_artifact,
            f"auxiliary contribution {contribution_id} evidence artifact",
        )
        _exact_fields(
            artifact,
            {"id", "sha256", "uri"},
            f"auxiliary contribution {contribution_id} evidence artifact",
        )
        evidence_id = _identifier(
            artifact.get("id"),
            f"auxiliary contribution {contribution_id} evidence id",
        )
        if evidence_id in evidence_ids:
            raise ValueError(
                f"auxiliary contribution {contribution_id} evidence IDs "
                "must be unique"
            )
        evidence_ids.add(evidence_id)
        evidence.append(
            {
                "id": evidence_id,
                "sha256": _sha256(
                    artifact.get("sha256"),
                    f"auxiliary contribution {contribution_id} evidence SHA-256",
                ),
                "uri": _artifact_uri(
                    artifact.get("uri"),
                    f"auxiliary contribution {contribution_id} evidence URI",
                ),
            }
        )
    evidence.sort(key=lambda item: item["id"])
    return {
        "id": contribution_id,
        "kind": kind,
        "description": _text(
            contribution.get("description"),
            f"auxiliary contribution {contribution_id} description",
        ),
        "status": status,
        "evidence": evidence,
    }


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate auxiliary contribution JSON key: {key}")
        payload[key] = value
    return payload


def load_auxiliary_contributions(
    path: str | Path,
    *,
    campaign_id: str,
    campaign_revision: str,
) -> AuxiliaryContributionLedger:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("auxiliary contribution ledger must be a JSON object")
    return AuxiliaryContributionLedger.from_payload(
        payload,
        campaign_id=campaign_id,
        campaign_revision=campaign_revision,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_auxiliary_contribution_artifacts(
    ledger: AuxiliaryContributionLedger,
    artifact_root: str | Path | None,
) -> dict[str, dict[str, object]]:
    """Verify local evidence and identify which files may enter a public release."""

    bindings: dict[str, dict[str, object]] = {}
    payload = ledger.as_payload()
    contributors = payload["contributors"]
    if not isinstance(contributors, list):
        raise AssertionError("normalized auxiliary contributors are invalid")
    for contributor in contributors:
        if not isinstance(contributor, Mapping):
            raise AssertionError("normalized auxiliary contributor is invalid")
        credit = _mapping(contributor["credit"], "normalized auxiliary credit")
        publish = bool(credit["show_contribution_details"])
        for contribution in _sequence(
            contributor["contributions"],
            "normalized auxiliary contributions",
        ):
            normalized_contribution = _mapping(
                contribution,
                "normalized auxiliary contribution",
            )
            for artifact in _sequence(
                normalized_contribution["evidence"],
                "normalized auxiliary evidence",
            ):
                normalized_artifact = _mapping(
                    artifact,
                    "normalized auxiliary evidence artifact",
                )
                uri = str(normalized_artifact["uri"])
                if not uri.startswith("bundle:"):
                    continue
                relative = _safe_relative_artifact_path(
                    uri.removeprefix("bundle:"),
                    "auxiliary contribution artifact path",
                )
                digest = str(normalized_artifact["sha256"])
                previous = bindings.get(relative)
                if previous is not None and previous["sha256"] != digest:
                    raise ValueError(
                        "auxiliary contribution artifact path has conflicting "
                        f"digests: {relative}"
                    )
                bindings[relative] = {
                    "sha256": digest,
                    "public": publish
                    or bool(previous is not None and previous["public"]),
                }

    if not bindings:
        return {}
    if artifact_root is None:
        raise ValueError(
            "bundled auxiliary contribution artifacts require an artifact root"
        )
    unresolved_root = Path(artifact_root)
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise ValueError(
            "auxiliary contribution artifact root must be a regular directory"
        )
    root = unresolved_root.resolve()
    for relative_name, binding in sorted(bindings.items()):
        relative = Path(relative_name)
        if any(
            candidate.is_symlink()
            for candidate in (
                root.joinpath(*relative.parts[:index])
                for index in range(1, len(relative.parts) + 1)
            )
        ):
            raise ValueError(
                "auxiliary contribution artifact path may not contain symlinks"
            )
        source = (root / relative).resolve()
        if (
            not source.is_relative_to(root)
            or not source.is_file()
            or source.is_symlink()
        ):
            raise ValueError(
                f"auxiliary contribution artifact is missing: {relative_name}"
            )
        if _sha256_file(source) != binding["sha256"]:
            raise ValueError(
                "auxiliary contribution artifact digest differs: "
                f"{relative_name}"
            )
    return bindings


def copy_public_auxiliary_contribution_artifacts(
    artifact_root: str | Path | None,
    destination: str | Path,
    bindings: Mapping[str, Mapping[str, object]],
) -> None:
    public_bindings = {
        name: binding
        for name, binding in bindings.items()
        if binding.get("public") is True
    }
    if not public_bindings:
        return
    if artifact_root is None:
        raise ValueError(
            "public auxiliary contribution artifacts require an artifact root"
        )
    root = Path(artifact_root).resolve()
    destination_root = Path(destination)
    for relative_name, binding in sorted(public_bindings.items()):
        source = root / relative_name
        target = destination_root / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        if _sha256_file(target) != binding.get("sha256"):
            raise ValueError(
                "auxiliary contribution artifact changed while copying: "
                f"{relative_name}"
            )


def build_auxiliary_contribution_snapshot(
    ledger: AuxiliaryContributionLedger | None,
    *,
    campaign_id: str,
    campaign_revision: str,
    release_checkpoint_sha256: str,
    release_checkpoint_step: int,
    verified_bundle_artifacts: Mapping[
        str,
        Mapping[str, object],
    ] | None = None,
) -> dict[str, object]:
    """Build a deterministic public record without exposing private contributor IDs."""

    normalized_campaign_id = _identifier(campaign_id, "campaign id")
    normalized_campaign_revision = _sha256(
        campaign_revision,
        "campaign revision",
    )
    normalized_checkpoint = _sha256(
        release_checkpoint_sha256,
        "release checkpoint SHA-256",
    )
    if (
        isinstance(release_checkpoint_step, bool)
        or not isinstance(release_checkpoint_step, int)
        or release_checkpoint_step < 0
    ):
        raise ValueError("release checkpoint step must be a nonnegative integer")
    bindings = dict(verified_bundle_artifacts or {})
    if ledger is None:
        if bindings:
            raise ValueError(
                "auxiliary artifact bindings require a contribution ledger"
            )
        contributors: list[object] = []
        ledger_revision: str | None = None
        record_status = "not_supplied"
    else:
        if (
            ledger.campaign_id != normalized_campaign_id
            or ledger.campaign_revision != normalized_campaign_revision
        ):
            raise ValueError(
                "auxiliary contribution ledger differs from the released campaign"
            )
        payload = ledger.as_payload()
        contributors = _sequence(
            payload["contributors"],
            "normalized auxiliary contributors",
            allow_empty=True,
        )
        ledger_revision = ledger.revision
        record_status = "owner_reviewed"

    public_contributors: list[dict[str, object]] = []
    anonymous_count = 0
    anonymous_contribution_count = 0
    contribution_count = 0
    public_person_time = 0
    public_compute_time = 0
    public_hardware_contributors = 0
    for raw_contributor in contributors:
        contributor = _mapping(
            raw_contributor,
            "normalized auxiliary contributor",
        )
        credit = _mapping(
            contributor["credit"],
            "normalized auxiliary contributor credit",
        )
        resources = _mapping(
            contributor["resources"],
            "normalized auxiliary contributor resources",
        )
        raw_contributions = _sequence(
            contributor["contributions"],
            "normalized auxiliary contributor contributions",
        )
        contribution_count += len(raw_contributions)
        if credit["visibility"] == "anonymous":
            anonymous_count += 1
            anonymous_contribution_count += len(raw_contributions)
            continue

        public_entry: dict[str, object] = {
            "display_name": credit["display_name"],
            "visibility": credit["visibility"],
            "profile_url": credit["profile_url"],
            "team": credit["team"],
            "credit_profile_revision": hashlib.sha256(
                _canonical_json(credit)
            ).hexdigest(),
        }
        if credit["show_contribution_details"]:
            details: list[dict[str, object]] = []
            for raw_contribution in raw_contributions:
                contribution = _mapping(
                    raw_contribution,
                    "normalized auxiliary contribution",
                )
                public_evidence: list[dict[str, object]] = []
                for raw_artifact in _sequence(
                    contribution["evidence"],
                    "normalized auxiliary evidence",
                ):
                    artifact = _mapping(
                        raw_artifact,
                        "normalized auxiliary evidence artifact",
                    )
                    uri = str(artifact["uri"])
                    verification = "declared_external_reference"
                    if uri.startswith("bundle:"):
                        relative = uri.removeprefix("bundle:")
                        binding = bindings.get(relative)
                        if (
                            not isinstance(binding, Mapping)
                            or binding.get("sha256") != artifact["sha256"]
                            or binding.get("public") is not True
                        ):
                            raise ValueError(
                                "public auxiliary contribution artifact was "
                                f"not verified: {relative}"
                            )
                        verification = "bundled_sha256_verified"
                    public_evidence.append(
                        {
                            "id": artifact["id"],
                            "sha256": artifact["sha256"],
                            "uri": uri,
                            "verification": verification,
                        }
                    )
                details.append(
                    {
                        "id": contribution["id"],
                        "kind": contribution["kind"],
                        "description": contribution["description"],
                        "status": contribution["status"],
                        "evidence": public_evidence,
                    }
                )
            public_entry["contributions"] = details
        else:
            public_entry["contribution_details_withheld"] = True

        public_resources: dict[str, object] = {}
        if credit["show_time"]:
            person_time = resources["person_time_seconds"]
            compute_time = resources["compute_time_seconds"]
            if person_time is not None:
                public_resources["person_time_seconds"] = person_time
                public_person_time += int(person_time)
            if compute_time is not None:
                public_resources["compute_time_seconds"] = compute_time
                public_compute_time += int(compute_time)
        if credit["show_hardware"]:
            public_resources["hardware"] = list(resources["hardware"])  # type: ignore[arg-type]
            public_hardware_contributors += 1
        if public_resources:
            public_entry["resources"] = public_resources
        public_contributors.append(public_entry)

    public_contributors.sort(
        key=lambda item: str(item["display_name"]).casefold()
    )
    snapshot: dict[str, object] = {
        "format": _SNAPSHOT_FORMAT,
        "campaign_id": normalized_campaign_id,
        "campaign_revision": normalized_campaign_revision,
        "release_checkpoint_sha256": normalized_checkpoint,
        "release_checkpoint_step": release_checkpoint_step,
        "record_status": record_status,
        "source_ledger_sha256": ledger_revision,
        "public_contributors": public_contributors,
        "anonymous_contributors": {
            "count": anonymous_count,
            "contribution_count": anonymous_contribution_count,
        },
        "all_contributions": {
            "contributor_count": len(contributors),
            "contribution_count": contribution_count,
        },
        "public_resource_totals": {
            "person_time_seconds": public_person_time,
            "compute_time_seconds": public_compute_time,
            "contributors_with_public_hardware": public_hardware_contributors,
        },
        "measurement_notes": [
            "Auxiliary entries are supplied and reviewed by the campaign owner; the framework does not infer them from direct-training role labels.",
            "Each contribution carries owner-supplied evidence identities. Local bundle files are copied only after their SHA-256 digests are verified; other URIs remain declared external references.",
            "Person time, compute time, and hardware are owner-reviewed declarations rather than framework measurements; they and the work details appear only under the contributor's confirmed disclosure choices.",
            "Auxiliary work is separate from accepted direct-training assignments and tokens. A failed_informative entry does not represent accepted optimizer work.",
        ],
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(
        _canonical_json(snapshot)
    ).hexdigest()
    validate_auxiliary_contribution_snapshot(snapshot)
    return snapshot


def validate_auxiliary_contribution_snapshot(
    snapshot: Mapping[str, object],
) -> None:
    """Validate a public snapshot before copying it into another package."""

    _exact_fields(
        snapshot,
        {
            "format",
            "campaign_id",
            "campaign_revision",
            "release_checkpoint_sha256",
            "release_checkpoint_step",
            "record_status",
            "source_ledger_sha256",
            "public_contributors",
            "anonymous_contributors",
            "all_contributions",
            "public_resource_totals",
            "measurement_notes",
            "snapshot_sha256",
        },
        "auxiliary contribution snapshot",
    )
    if snapshot.get("format") != _SNAPSHOT_FORMAT:
        raise ValueError("unsupported auxiliary contribution snapshot")
    _identifier(snapshot.get("campaign_id"), "snapshot campaign id")
    _sha256(snapshot.get("campaign_revision"), "snapshot campaign revision")
    _sha256(
        snapshot.get("release_checkpoint_sha256"),
        "snapshot release checkpoint SHA-256",
    )
    _nonnegative_int(
        snapshot.get("release_checkpoint_step"),
        "snapshot release checkpoint step",
    )
    status = snapshot.get("record_status")
    if status not in {"not_supplied", "owner_reviewed"}:
        raise ValueError("auxiliary contribution snapshot status is invalid")
    source_revision = snapshot.get("source_ledger_sha256")
    if status == "owner_reviewed":
        _sha256(source_revision, "snapshot source ledger SHA-256")
    elif source_revision is not None:
        raise ValueError(
            "unsupplied auxiliary contribution snapshot has a source ledger"
        )

    public_contributors = _sequence(
        snapshot.get("public_contributors"),
        "snapshot public auxiliary contributors",
        allow_empty=True,
    )
    public_person_time = 0
    public_compute_time = 0
    public_hardware_contributors = 0
    visible_contribution_count = 0
    contribution_ids: set[str] = set()
    for raw_contributor in public_contributors:
        contributor = _mapping(
            raw_contributor,
            "snapshot public auxiliary contributor",
        )
        base_fields = {
            "display_name",
            "visibility",
            "profile_url",
            "team",
            "credit_profile_revision",
        }
        optional_fields = set(contributor) - base_fields
        if (
            not base_fields.issubset(contributor)
            or not optional_fields.issubset(
                {
                    "contributions",
                    "contribution_details_withheld",
                    "resources",
                }
            )
            or (
                ("contributions" in contributor)
                == ("contribution_details_withheld" in contributor)
            )
        ):
            raise ValueError(
                "snapshot public auxiliary contributor schema is invalid"
            )
        visibility = contributor.get("visibility")
        if visibility not in {"named", "pseudonymous"}:
            raise ValueError(
                "snapshot public auxiliary contributor visibility is invalid"
            )
        _text(
            contributor.get("display_name"),
            "snapshot public auxiliary contributor name",
            limit=256,
        )
        _optional_https_url(
            contributor.get("profile_url"),
            "snapshot public auxiliary contributor profile URL",
        )
        _optional_text(
            contributor.get("team"),
            "snapshot public auxiliary contributor team",
        )
        _sha256(
            contributor.get("credit_profile_revision"),
            "snapshot public auxiliary credit revision",
        )
        if "contributions" in contributor:
            contributions = _sequence(
                contributor["contributions"],
                "snapshot public auxiliary contributions",
            )
            visible_contribution_count += len(contributions)
            for raw_contribution in contributions:
                contribution = _mapping(
                    raw_contribution,
                    "snapshot public auxiliary contribution",
                )
                _exact_fields(
                    contribution,
                    {"id", "kind", "description", "status", "evidence"},
                    "snapshot public auxiliary contribution",
                )
                contribution_id = _identifier(
                    contribution.get("id"),
                    "snapshot public auxiliary contribution id",
                )
                if contribution_id in contribution_ids:
                    raise ValueError(
                        "snapshot auxiliary contribution IDs must be unique"
                    )
                contribution_ids.add(contribution_id)
                _identifier(
                    contribution.get("kind"),
                    "snapshot public auxiliary contribution kind",
                )
                _text(
                    contribution.get("description"),
                    "snapshot public auxiliary contribution description",
                )
                if contribution.get("status") not in _CONTRIBUTION_STATUSES:
                    raise ValueError(
                        "snapshot public auxiliary contribution status is invalid"
                    )
                evidence = _sequence(
                    contribution.get("evidence"),
                    "snapshot public auxiliary contribution evidence",
                )
                evidence_ids: set[str] = set()
                for raw_artifact in evidence:
                    artifact = _mapping(
                        raw_artifact,
                        "snapshot public auxiliary evidence artifact",
                    )
                    _exact_fields(
                        artifact,
                        {"id", "sha256", "uri", "verification"},
                        "snapshot public auxiliary evidence artifact",
                    )
                    evidence_id = _identifier(
                        artifact.get("id"),
                        "snapshot public auxiliary evidence id",
                    )
                    if evidence_id in evidence_ids:
                        raise ValueError(
                            "snapshot public auxiliary evidence IDs must be unique"
                        )
                    evidence_ids.add(evidence_id)
                    _sha256(
                        artifact.get("sha256"),
                        "snapshot public auxiliary evidence SHA-256",
                    )
                    uri = _artifact_uri(
                        artifact.get("uri"),
                        "snapshot public auxiliary evidence URI",
                    )
                    expected_verification = (
                        "bundled_sha256_verified"
                        if uri.startswith("bundle:")
                        else "declared_external_reference"
                    )
                    if artifact.get("verification") != expected_verification:
                        raise ValueError(
                            "snapshot public auxiliary evidence verification "
                            "state is invalid"
                        )
        elif contributor.get("contribution_details_withheld") is not True:
            raise ValueError(
                "snapshot withheld auxiliary details marker is invalid"
            )

        if "resources" in contributor:
            resources = _mapping(
                contributor["resources"],
                "snapshot public auxiliary resources",
            )
            if not resources or not set(resources).issubset(
                {"person_time_seconds", "compute_time_seconds", "hardware"}
            ):
                raise ValueError(
                    "snapshot public auxiliary resources schema is invalid"
                )
            if "person_time_seconds" in resources:
                person_time = _optional_positive_int(
                    resources["person_time_seconds"],
                    "snapshot public auxiliary person time",
                )
                if person_time is None:
                    raise ValueError(
                        "snapshot public auxiliary person time is null"
                    )
                public_person_time += person_time
            if "compute_time_seconds" in resources:
                compute_time = _optional_positive_int(
                    resources["compute_time_seconds"],
                    "snapshot public auxiliary compute time",
                )
                if compute_time is None:
                    raise ValueError(
                        "snapshot public auxiliary compute time is null"
                    )
                public_compute_time += compute_time
            if "hardware" in resources:
                hardware = _sequence(
                    resources["hardware"],
                    "snapshot public auxiliary hardware",
                )
                normalized_hardware = [
                    _text(
                        item,
                        "snapshot public auxiliary hardware entry",
                        limit=256,
                    )
                    for item in hardware
                ]
                if len(set(normalized_hardware)) != len(normalized_hardware):
                    raise ValueError(
                        "snapshot public auxiliary hardware entries "
                        "must be unique"
                    )
                public_hardware_contributors += 1

    anonymous = _mapping(
        snapshot.get("anonymous_contributors"),
        "snapshot anonymous auxiliary contributors",
    )
    _exact_fields(
        anonymous,
        {"count", "contribution_count"},
        "snapshot anonymous auxiliary contributors",
    )
    anonymous_count = _nonnegative_int(
        anonymous.get("count"),
        "snapshot anonymous auxiliary contributor count",
    )
    anonymous_contribution_count = _nonnegative_int(
        anonymous.get("contribution_count"),
        "snapshot anonymous auxiliary contribution count",
    )
    totals = _mapping(
        snapshot.get("all_contributions"),
        "snapshot auxiliary contribution totals",
    )
    _exact_fields(
        totals,
        {"contributor_count", "contribution_count"},
        "snapshot auxiliary contribution totals",
    )
    contributor_count = _nonnegative_int(
        totals.get("contributor_count"),
        "snapshot auxiliary contributor total",
    )
    contribution_count = _nonnegative_int(
        totals.get("contribution_count"),
        "snapshot auxiliary contribution total",
    )
    if contributor_count != len(public_contributors) + anonymous_count:
        raise ValueError(
            "snapshot auxiliary contributor total is inconsistent"
        )
    if contribution_count < (
        visible_contribution_count + anonymous_contribution_count
    ):
        raise ValueError(
            "snapshot auxiliary contribution total is inconsistent"
        )
    if status == "not_supplied" and (
        contributor_count != 0 or contribution_count != 0
    ):
        raise ValueError(
            "unsupplied auxiliary contribution snapshot contains contributions"
        )

    resource_totals = _mapping(
        snapshot.get("public_resource_totals"),
        "snapshot public auxiliary resource totals",
    )
    _exact_fields(
        resource_totals,
        {
            "person_time_seconds",
            "compute_time_seconds",
            "contributors_with_public_hardware",
        },
        "snapshot public auxiliary resource totals",
    )
    if (
        _nonnegative_int(
            resource_totals.get("person_time_seconds"),
            "snapshot public auxiliary person-time total",
        )
        != public_person_time
        or _nonnegative_int(
            resource_totals.get("compute_time_seconds"),
            "snapshot public auxiliary compute-time total",
        )
        != public_compute_time
        or _nonnegative_int(
            resource_totals.get("contributors_with_public_hardware"),
            "snapshot public auxiliary hardware contributor total",
        )
        != public_hardware_contributors
    ):
        raise ValueError(
            "snapshot public auxiliary resource totals are inconsistent"
        )
    notes = _sequence(
        snapshot.get("measurement_notes"),
        "snapshot auxiliary measurement notes",
        allow_empty=True,
    )
    for note in notes:
        _text(note, "snapshot auxiliary measurement note")

    expected_snapshot_sha256 = _sha256(
        snapshot.get("snapshot_sha256"),
        "auxiliary contribution snapshot SHA-256",
    )
    unsigned = dict(snapshot)
    unsigned.pop("snapshot_sha256")
    if hashlib.sha256(_canonical_json(unsigned)).hexdigest() != (
        expected_snapshot_sha256
    ):
        raise ValueError("auxiliary contribution snapshot SHA-256 differs")


def verify_public_auxiliary_snapshot_artifacts(
    snapshot: Mapping[str, object],
    artifact_root: str | Path,
) -> dict[str, str]:
    """Require the published artifact directory to match snapshot bindings."""

    validate_auxiliary_contribution_snapshot(snapshot)
    expected: dict[str, str] = {}
    contributors = _sequence(
        snapshot.get("public_contributors"),
        "snapshot public auxiliary contributors",
        allow_empty=True,
    )
    for raw_contributor in contributors:
        contributor = _mapping(
            raw_contributor,
            "snapshot public auxiliary contributor",
        )
        raw_contributions = contributor.get("contributions")
        if raw_contributions is None:
            continue
        for raw_contribution in _sequence(
            raw_contributions,
            "snapshot public auxiliary contributions",
        ):
            contribution = _mapping(
                raw_contribution,
                "snapshot public auxiliary contribution",
            )
            for raw_artifact in _sequence(
                contribution.get("evidence"),
                "snapshot public auxiliary contribution evidence",
            ):
                artifact = _mapping(
                    raw_artifact,
                    "snapshot public auxiliary evidence artifact",
                )
                uri = str(artifact["uri"])
                if not uri.startswith("bundle:"):
                    continue
                relative = _safe_relative_artifact_path(
                    uri.removeprefix("bundle:"),
                    "snapshot public auxiliary artifact path",
                )
                digest = str(artifact["sha256"])
                if relative in expected and expected[relative] != digest:
                    raise ValueError(
                        "snapshot public auxiliary artifact path has "
                        f"conflicting digests: {relative}"
                    )
                expected[relative] = digest

    unresolved_root = Path(artifact_root)
    if not expected:
        if unresolved_root.exists():
            if unresolved_root.is_symlink() or not unresolved_root.is_dir():
                raise ValueError(
                    "public auxiliary artifact root must be a regular directory"
                )
            if any(unresolved_root.iterdir()):
                raise ValueError(
                    "public auxiliary artifact directory contains "
                    "unreferenced entries"
                )
        return {}
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise ValueError(
            "public auxiliary artifact root must be a regular directory"
        )
    root = unresolved_root.resolve()
    actual: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(
                "public auxiliary artifact directory may not contain symlinks"
            )
        if candidate.is_file():
            actual[relative] = candidate
        elif not candidate.is_dir():
            raise ValueError(
                "public auxiliary artifact directory contains a "
                f"non-regular entry: {relative}"
            )
    if set(actual) != set(expected):
        raise ValueError(
            "public auxiliary artifact directory differs from snapshot bindings"
        )
    for relative, digest in expected.items():
        if _sha256_file(actual[relative]) != digest:
            raise ValueError(
                f"public auxiliary artifact digest differs: {relative}"
            )
    return dict(sorted(expected.items()))


def _markdown_text(value: object) -> str:
    escaped = html_escape(str(value), quote=False)
    for character in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def auxiliary_contribution_markdown(snapshot: Mapping[str, object]) -> str:
    if snapshot.get("format") != _SNAPSHOT_FORMAT:
        raise ValueError("auxiliary contribution snapshot schema is invalid")
    status = snapshot.get("record_status")
    contributors = snapshot.get("public_contributors")
    anonymous = snapshot.get("anonymous_contributors")
    totals = snapshot.get("all_contributions")
    if (
        status not in {"not_supplied", "owner_reviewed"}
        or not isinstance(contributors, list)
        or any(not isinstance(item, Mapping) for item in contributors)
        or not isinstance(anonymous, Mapping)
        or not isinstance(totals, Mapping)
    ):
        raise ValueError("auxiliary contribution snapshot schema is invalid")
    lines = ["## Auxiliary contributions", ""]
    if status == "not_supplied":
        lines.extend(
            [
                "No owner-reviewed auxiliary contribution record was supplied "
                "for this release.",
                "",
                "This does not establish that no auxiliary work occurred. The "
                "record must be completed before public publication.",
                "",
            ]
        )
        return "\n".join(lines)

    contributor_count = totals.get("contributor_count")
    contribution_count = totals.get("contribution_count")
    if contributor_count == 0:
        lines.extend(
            [
                "The owner-reviewed record contains no auxiliary contribution "
                "entries for this release.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"The owner-reviewed record contains {contribution_count} auxiliary "
            f"contribution(s) from {contributor_count} contributor(s). These "
            "entries are separate from accepted direct-training work.",
            "",
        ]
    )
    if contributors:
        for contributor in contributors:
            name = _markdown_text(contributor["display_name"])
            profile_url = contributor.get("profile_url")
            label = f"[{name}](<{profile_url}>)" if profile_url else name
            team = contributor.get("team")
            lines.append(
                f"- {label}"
                + (f"; team: {_markdown_text(team)}" if team else "")
            )
            resources = contributor.get("resources")
            if isinstance(resources, Mapping):
                resource_parts: list[str] = []
                if "person_time_seconds" in resources:
                    resource_parts.append(
                        f"person time: {resources['person_time_seconds']} seconds"
                    )
                if "compute_time_seconds" in resources:
                    resource_parts.append(
                        f"compute time: {resources['compute_time_seconds']} seconds"
                    )
                if resource_parts:
                    lines.append("  - " + "; ".join(resource_parts))
                hardware = resources.get("hardware")
                if isinstance(hardware, list):
                    for item in hardware:
                        lines.append(f"  - Hardware: {_markdown_text(item)}")
            details = contributor.get("contributions")
            if isinstance(details, list):
                for detail in details:
                    if not isinstance(detail, Mapping):
                        raise ValueError(
                            "auxiliary contribution detail is invalid"
                        )
                    lines.append(
                        "  - "
                        f"`{_markdown_text(detail['kind'])}` "
                        f"(`{_markdown_text(detail['status'])}`): "
                        f"{_markdown_text(detail['description'])}"
                    )
                    evidence = detail.get("evidence")
                    if not isinstance(evidence, list):
                        raise ValueError(
                            "auxiliary contribution evidence is invalid"
                        )
                    for artifact in evidence:
                        if not isinstance(artifact, Mapping):
                            raise ValueError(
                                "auxiliary contribution evidence is invalid"
                            )
                        artifact_id = _markdown_text(artifact["id"])
                        uri = str(artifact["uri"])
                        if uri.startswith("bundle:"):
                            target = (
                                "auxiliary-contribution-artifacts/"
                                + uri.removeprefix("bundle:")
                            )
                            evidence_label = f"[{artifact_id}](<{target}>)"
                        elif uri.startswith("https://"):
                            evidence_label = f"[{artifact_id}](<{uri}>)"
                        else:
                            evidence_label = (
                                f"{artifact_id} at `{_markdown_text(uri)}`"
                            )
                        lines.append(
                            f"    - Evidence: {evidence_label}; SHA-256 "
                            f"`{artifact['sha256']}`; "
                            f"`{artifact['verification']}`"
                        )
            elif contributor.get("contribution_details_withheld") is True:
                lines.append(
                    "  - Contribution details withheld by the contributor."
                )
    else:
        lines.append(
            "- No auxiliary contributor chose named or pseudonymous credit."
        )
    lines.extend(
        [
            "",
            "### Anonymous auxiliary contributions",
            "",
            f"{anonymous.get('count', 0)} auxiliary contributor(s) chose "
            "anonymous credit, covering "
            f"{anonymous.get('contribution_count', 0)} contribution record(s).",
            "",
            "See `auxiliary-contribution-snapshot.json` for the public record "
            "tied to this release and its evidence identities.",
            "",
        ]
    )
    return "\n".join(lines)
