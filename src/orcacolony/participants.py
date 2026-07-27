from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Participant:
    contributor_id: str
    worker_ids: tuple[str, ...]
    worker_token_sha256: tuple[tuple[str, str], ...]
    visibility: str
    display_name: str | None
    profile_url: str | None
    team: str | None
    roles: tuple[str, ...]
    show_contribution_totals: bool
    show_hardware: bool
    worker_profiles: tuple[tuple[str, str, bool], ...]
    credit_profile_revision: str

    @property
    def public_credit(self) -> bool:
        return self.visibility != "anonymous"


@dataclass(frozen=True)
class ParticipantRegistry:
    format: str
    campaign_id: str
    participants: tuple[Participant, ...]
    revision: str
    credit_revision: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        campaign_id: str,
    ) -> ParticipantRegistry:
        manifest_format = payload.get("format")
        if manifest_format not in {
            "orcacolony_participants_v1",
            "orcacolony_participants_v2",
        }:
            raise ValueError("unsupported participant manifest format")
        if payload.get("campaign_id") != campaign_id:
            raise ValueError("participant manifest campaign mismatch")
        if set(payload) != {"format", "campaign_id", "participants"}:
            raise ValueError("participant manifest schema is invalid")
        raw_participants = payload.get("participants")
        if not isinstance(raw_participants, list) or not raw_participants:
            raise ValueError("participant manifest must include at least one participant")

        participants: list[Participant] = []
        contributor_ids: set[str] = set()
        all_worker_ids: set[str] = set()
        for raw in raw_participants:
            if not isinstance(raw, dict):
                raise ValueError("participant entries must be objects")
            allowed_participant_fields = {
                "contributor_id",
                "worker_ids",
                "worker_token_sha256",
                "credit",
            }
            if manifest_format == "orcacolony_participants_v2":
                allowed_participant_fields.add("worker_profiles")
            unknown_participant_fields = sorted(
                set(raw) - allowed_participant_fields
            )
            if unknown_participant_fields:
                raise ValueError(
                    "participant entry contains unknown fields: "
                    + ", ".join(unknown_participant_fields)
                )
            contributor_id = _identifier(raw.get("contributor_id"), "contributor id")
            if contributor_id in contributor_ids:
                raise ValueError(f"duplicate contributor id: {contributor_id}")
            contributor_ids.add(contributor_id)

            raw_worker_ids = raw.get("worker_ids")
            if not isinstance(raw_worker_ids, list) or not raw_worker_ids:
                raise ValueError(f"participant {contributor_id} has no worker ids")
            worker_ids = tuple(
                sorted(_identifier(value, "worker id") for value in raw_worker_ids)
            )
            if len(set(worker_ids)) != len(worker_ids):
                raise ValueError(f"participant {contributor_id} repeats a worker id")
            overlap = all_worker_ids.intersection(worker_ids)
            if overlap:
                raise ValueError(f"worker id assigned more than once: {min(overlap)}")
            all_worker_ids.update(worker_ids)

            raw_token_hashes = raw.get("worker_token_sha256")
            if not isinstance(raw_token_hashes, dict):
                raise ValueError(
                    f"participant {contributor_id} needs worker token hashes"
                )
            if set(raw_token_hashes) != set(worker_ids):
                raise ValueError(
                    f"participant {contributor_id} token hashes must match worker ids"
                )
            worker_token_sha256: list[tuple[str, str]] = []
            for worker_id in worker_ids:
                digest = raw_token_hashes[worker_id]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(
                        f"participant {contributor_id} has an invalid worker token hash"
                    )
                worker_token_sha256.append((worker_id, digest))

            raw_credit = raw.get("credit", {})
            if not isinstance(raw_credit, dict):
                raise ValueError(f"participant {contributor_id} credit must be an object")
            if manifest_format == "orcacolony_participants_v1":
                unknown_credit = sorted(
                    set(raw_credit) - {"public", "display_name"}
                )
                if unknown_credit:
                    raise ValueError(
                        f"participant {contributor_id} credit contains unknown fields: "
                        + ", ".join(unknown_credit)
                    )
                public_credit = raw_credit.get("public", False)
                if not isinstance(public_credit, bool):
                    raise ValueError(
                        f"participant {contributor_id} public credit must be boolean"
                    )
                display_name = _optional_text(
                    raw_credit.get("display_name"),
                    f"participant {contributor_id} display name",
                )
                if public_credit and display_name is None:
                    raise ValueError(
                        f"participant {contributor_id} needs a display name for public credit"
                    )
                visibility = "named" if public_credit else "anonymous"
                profile_url = None
                team = None
                roles = ("training-compute",)
                show_contribution_totals = public_credit
                show_hardware = False
                worker_profiles: tuple[tuple[str, str, bool], ...] = ()
            else:
                allowed_credit = {
                    "visibility",
                    "display_name",
                    "profile_url",
                    "team",
                    "roles",
                    "show_contribution_totals",
                    "show_hardware",
                }
                unknown_credit = sorted(set(raw_credit) - allowed_credit)
                if unknown_credit:
                    raise ValueError(
                        f"participant {contributor_id} credit contains unknown fields: "
                        + ", ".join(unknown_credit)
                    )
                visibility = raw_credit.get("visibility")
                if visibility not in {"named", "pseudonymous", "anonymous"}:
                    raise ValueError(
                        f"participant {contributor_id} credit visibility is invalid"
                    )
                display_name = _optional_text(
                    raw_credit.get("display_name"),
                    f"participant {contributor_id} display name",
                )
                if visibility == "anonymous" and display_name is not None:
                    raise ValueError(
                        f"participant {contributor_id} anonymous credit may not publish a name"
                    )
                if visibility != "anonymous" and display_name is None:
                    raise ValueError(
                        f"participant {contributor_id} public credit needs a display name"
                    )
                profile_url = _optional_https_url(
                    raw_credit.get("profile_url"),
                    f"participant {contributor_id} profile URL",
                )
                team = _optional_text(
                    raw_credit.get("team"),
                    f"participant {contributor_id} team",
                )
                raw_roles = raw_credit.get("roles", ["training-compute"])
                if (
                    not isinstance(raw_roles, list)
                    or not raw_roles
                    or any(
                        not isinstance(role, str) or not role.strip()
                        for role in raw_roles
                    )
                ):
                    raise ValueError(
                        f"participant {contributor_id} roles must be non-empty text"
                    )
                roles = tuple(
                    sorted(
                        {
                            _identifier(role, "contributor role")
                            for role in raw_roles
                        }
                    )
                )
                show_contribution_totals = raw_credit.get(
                    "show_contribution_totals",
                    False,
                )
                show_hardware = raw_credit.get("show_hardware", False)
                if type(show_contribution_totals) is not bool:
                    raise ValueError(
                        f"participant {contributor_id} totals preference must be boolean"
                    )
                if type(show_hardware) is not bool:
                    raise ValueError(
                        f"participant {contributor_id} hardware preference must be boolean"
                    )
                if visibility == "anonymous" and (
                    profile_url is not None
                    or team is not None
                    or show_contribution_totals
                    or show_hardware
                ):
                    raise ValueError(
                        f"participant {contributor_id} anonymous credit may not publish profile fields"
                    )
                raw_worker_profiles = raw.get("worker_profiles", {})
                if not isinstance(raw_worker_profiles, dict):
                    raise ValueError(
                        f"participant {contributor_id} worker profiles must be an object"
                    )
                if not set(raw_worker_profiles).issubset(worker_ids):
                    raise ValueError(
                        f"participant {contributor_id} worker profile IDs must be allowlisted"
                    )
                parsed_worker_profiles: list[tuple[str, str, bool]] = []
                for worker_id, raw_profile in sorted(raw_worker_profiles.items()):
                    if not isinstance(raw_profile, dict) or set(raw_profile) != {
                        "hardware_class",
                        "public",
                    }:
                        raise ValueError(
                            f"participant {contributor_id} worker profile is invalid"
                        )
                    hardware_class = _optional_text(
                        raw_profile.get("hardware_class"),
                        f"participant {contributor_id} hardware class",
                    )
                    if hardware_class is None or type(raw_profile.get("public")) is not bool:
                        raise ValueError(
                            f"participant {contributor_id} worker profile is invalid"
                        )
                    parsed_worker_profiles.append(
                        (worker_id, hardware_class, raw_profile["public"])
                    )
                worker_profiles = tuple(parsed_worker_profiles)
            credit_profile_payload = {
                "visibility": visibility,
                "display_name": display_name,
                "profile_url": profile_url,
                "team": team,
                "roles": list(roles),
                "show_contribution_totals": show_contribution_totals,
                "show_hardware": show_hardware,
                "worker_profiles": [
                    {
                        "worker_id": worker_id,
                        "hardware_class": hardware_class,
                        "public": public,
                    }
                    for worker_id, hardware_class, public in worker_profiles
                ],
            }
            credit_profile_revision = hashlib.sha256(
                _canonical_json(credit_profile_payload)
            ).hexdigest()
            participants.append(
                Participant(
                    contributor_id=contributor_id,
                    worker_ids=worker_ids,
                    worker_token_sha256=tuple(worker_token_sha256),
                    visibility=visibility,
                    display_name=display_name,
                    profile_url=profile_url,
                    team=team,
                    roles=roles,
                    show_contribution_totals=show_contribution_totals,
                    show_hardware=show_hardware,
                    worker_profiles=worker_profiles,
                    credit_profile_revision=credit_profile_revision,
                )
            )

        participants.sort(key=lambda value: value.contributor_id)
        normalized = _payload(manifest_format, campaign_id, participants)
        authority = (
            normalized
            if manifest_format == "orcacolony_participants_v1"
            else _authority_payload(campaign_id, participants)
        )
        revision = hashlib.sha256(_canonical_json(authority)).hexdigest()
        credit_revision = hashlib.sha256(
            _canonical_json(_credit_payload(campaign_id, participants))
        ).hexdigest()
        return cls(
            format=manifest_format,
            campaign_id=campaign_id,
            participants=tuple(participants),
            revision=revision,
            credit_revision=credit_revision,
        )

    def as_payload(self) -> dict[str, object]:
        return _payload(self.format, self.campaign_id, self.participants)

    def participant_for_worker(self, worker_id: str) -> Participant | None:
        for participant in self.participants:
            if worker_id in participant.worker_ids:
                return participant
        return None

    def credential_is_valid(
        self,
        participant: Participant,
        worker_id: str,
        token: str | None,
    ) -> bool:
        if token is None:
            return False
        expected = dict(participant.worker_token_sha256)[worker_id]
        actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return hmac.compare_digest(actual, expected)


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > 128:
        raise ValueError(f"{label} is too long")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in normalized):
        raise ValueError(f"{label} contains unsupported characters")
    return normalized


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{label} must be non-empty")
    normalized = value.strip()
    if len(normalized) > 256:
        raise ValueError(f"{label} is too long")
    return normalized


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
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(
            character.isspace() or character in '<>"\''
            for character in normalized
        )
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials or fragment")
    return normalized


def _payload(
    manifest_format: str,
    campaign_id: str,
    participants: list[Participant] | tuple[Participant, ...],
) -> dict[str, object]:
    return {
        "format": manifest_format,
        "campaign_id": campaign_id,
        "participants": [
            {
                "contributor_id": participant.contributor_id,
                "worker_ids": list(participant.worker_ids),
                "worker_token_sha256": dict(participant.worker_token_sha256),
                "credit": (
                    {
                        "public": participant.public_credit,
                        "display_name": participant.display_name,
                    }
                    if manifest_format == "orcacolony_participants_v1"
                    else {
                        "visibility": participant.visibility,
                        "display_name": participant.display_name,
                        "profile_url": participant.profile_url,
                        "team": participant.team,
                        "roles": list(participant.roles),
                        "show_contribution_totals": (
                            participant.show_contribution_totals
                        ),
                        "show_hardware": participant.show_hardware,
                    }
                ),
                **(
                    {}
                    if manifest_format == "orcacolony_participants_v1"
                    else {
                        "worker_profiles": {
                            worker_id: {
                                "hardware_class": hardware_class,
                                "public": public,
                            }
                            for worker_id, hardware_class, public in (
                                participant.worker_profiles
                            )
                        }
                    }
                ),
            }
            for participant in participants
        ],
    }


def _authority_payload(
    campaign_id: str,
    participants: list[Participant] | tuple[Participant, ...],
) -> dict[str, object]:
    return {
        "format": "orcacolony_participant_authority_v1",
        "campaign_id": campaign_id,
        "participants": [
            {
                "contributor_id": participant.contributor_id,
                "worker_ids": list(participant.worker_ids),
                "worker_token_sha256": dict(
                    participant.worker_token_sha256
                ),
            }
            for participant in participants
        ],
    }


def _credit_payload(
    campaign_id: str,
    participants: list[Participant] | tuple[Participant, ...],
) -> dict[str, object]:
    return {
        "format": "orcacolony_credit_profiles_v1",
        "campaign_id": campaign_id,
        "profiles": [
            {
                "contributor_id": participant.contributor_id,
                "credit_profile_revision": (
                    participant.credit_profile_revision
                ),
            }
            for participant in participants
        ],
    }


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def load_participants(
    path: str | Path,
    *,
    campaign_id: str,
) -> ParticipantRegistry:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("participant manifest must be a JSON object")
    return ParticipantRegistry.from_payload(payload, campaign_id=campaign_id)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate participant JSON key: {key}")
        payload[key] = value
    return payload


def _markdown_text(value: object) -> str:
    escaped = html_escape(str(value), quote=False)
    for character in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#"):
        escaped = escaped.replace(character, "\\" + character)
    return escaped


def _entry_runtime_seconds(
    entry: Mapping[str, object],
) -> tuple[float, float]:
    instrumentation = entry.get("instrumentation")
    if not isinstance(instrumentation, Mapping):
        return 0.0, 0.0
    reported = instrumentation.get("worker_reported")
    if not isinstance(reported, Mapping):
        return 0.0, 0.0
    runtimes = reported.get("runtime_seconds")
    if not isinstance(runtimes, Mapping):
        return 0.0, 0.0
    normalized: dict[str, float] = {}
    for name, value in runtimes.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            return 0.0, 0.0
        normalized[name] = float(value)
    return (
        sum(normalized.values()),
        normalized.get("gradient_compute", 0.0),
    )


def build_attribution_snapshot(
    registry: ParticipantRegistry,
    entries: list[Mapping[str, object]],
) -> dict[str, object]:
    """Create a deterministic, privacy-filtered release-time credit snapshot."""

    participants = {
        participant.contributor_id: participant
        for participant in registry.participants
    }
    totals: dict[str, dict[str, int | float]] = {}
    hardware_totals: dict[str, dict[str, dict[str, int | float]]] = {}
    for entry in entries:
        contributor_id = entry.get("contributor_id")
        if not isinstance(contributor_id, str) or contributor_id not in participants:
            raise ValueError("attribution ledger contains an unknown contributor")
        accepted_tokens = entry.get("loss_weight_sum")
        if (
            isinstance(accepted_tokens, bool)
            or not isinstance(accepted_tokens, int)
            or accepted_tokens <= 0
        ):
            raise ValueError("attribution ledger contains an invalid token count")
        contributor_totals = totals.setdefault(
            contributor_id,
            {
                "accepted_assignments": 0,
                "accepted_tokens": 0,
                "worker_reported_total_seconds": 0.0,
                "worker_reported_gradient_seconds": 0.0,
            },
        )
        contributor_totals["accepted_assignments"] += 1
        contributor_totals["accepted_tokens"] += accepted_tokens
        total_seconds, gradient_seconds = _entry_runtime_seconds(entry)
        contributor_totals["worker_reported_total_seconds"] += total_seconds
        contributor_totals["worker_reported_gradient_seconds"] += gradient_seconds

        worker_id = entry.get("worker_id")
        if isinstance(worker_id, str):
            profile = next(
                (
                    (hardware_class, public)
                    for profile_worker_id, hardware_class, public in (
                        participants[contributor_id].worker_profiles
                    )
                    if profile_worker_id == worker_id
                ),
                None,
            )
            if profile is not None:
                hardware_class, public = profile
                if public:
                    hardware = hardware_totals.setdefault(
                        contributor_id,
                        {},
                    ).setdefault(
                        hardware_class,
                        {
                            "accepted_assignments": 0,
                            "accepted_tokens": 0,
                            "worker_reported_total_seconds": 0.0,
                            "worker_reported_gradient_seconds": 0.0,
                        },
                    )
                    hardware["accepted_assignments"] += 1
                    hardware["accepted_tokens"] += accepted_tokens
                    hardware["worker_reported_total_seconds"] += total_seconds
                    hardware["worker_reported_gradient_seconds"] += gradient_seconds

    public_contributors: list[dict[str, object]] = []
    anonymous_count = 0
    anonymous_totals = {
        "accepted_assignments": 0,
        "accepted_tokens": 0,
        "worker_reported_total_seconds": 0.0,
        "worker_reported_gradient_seconds": 0.0,
    }
    aggregate_totals = {
        "accepted_assignments": 0,
        "accepted_tokens": 0,
        "worker_reported_total_seconds": 0.0,
        "worker_reported_gradient_seconds": 0.0,
    }
    for contributor_id, contributor_totals in totals.items():
        for key, value in contributor_totals.items():
            aggregate_totals[key] += value
        participant = participants[contributor_id]
        if not participant.public_credit:
            anonymous_count += 1
            for key, value in contributor_totals.items():
                anonymous_totals[key] += value
            continue
        public_entry: dict[str, object] = {
            "display_name": participant.display_name,
            "visibility": participant.visibility,
            "profile_url": participant.profile_url,
            "team": participant.team,
            "roles": list(participant.roles),
            "credit_profile_revision": participant.credit_profile_revision,
        }
        if participant.show_contribution_totals:
            public_entry["contribution_totals"] = dict(contributor_totals)
        if participant.show_hardware:
            public_entry["hardware"] = [
                {
                    "hardware_class": hardware_class,
                    **(
                        dict(class_totals)
                        if participant.show_contribution_totals
                        else {}
                    ),
                }
                for hardware_class, class_totals in sorted(
                    hardware_totals.get(contributor_id, {}).items()
                )
            ]
        public_contributors.append(public_entry)
    public_contributors.sort(
        key=lambda entry: str(entry["display_name"]).casefold()
    )
    snapshot: dict[str, object] = {
        "format": "orcacolony_attribution_snapshot_v1",
        "campaign_id": registry.campaign_id,
        "participants_revision": registry.revision,
        "credit_profiles_revision": registry.credit_revision,
        "public_contributors": public_contributors,
        "anonymous_contributors": {
            "count": anonymous_count,
            "contribution_totals": anonymous_totals,
        },
        "all_contributions": aggregate_totals,
        "measurement_notes": [
            "Accepted assignments and tokens are coordinator-verified ledger values.",
            "Total and gradient seconds are optional worker-reported telemetry and are not independently timed by the coordinator.",
            "Hardware classes are contributor-supplied and appear only when both the worker profile and credit profile opt in.",
        ],
    }
    snapshot["snapshot_sha256"] = hashlib.sha256(
        _canonical_json(snapshot)
    ).hexdigest()
    return snapshot


def attribution_markdown(snapshot: Mapping[str, object]) -> str:
    contributors = snapshot.get("public_contributors")
    anonymous = snapshot.get("anonymous_contributors")
    if not isinstance(contributors, list) or not isinstance(anonymous, Mapping):
        raise ValueError("attribution snapshot schema is invalid")
    lines = [
        "# Community contributors",
        "",
        "Every accepted contribution represented in this release is credited according "
        "to the contributor's release-time public credit preference.",
        "",
    ]
    if contributors:
        for contributor in contributors:
            if not isinstance(contributor, Mapping):
                raise ValueError("attribution contributor entry is invalid")
            name = _markdown_text(contributor["display_name"])
            profile_url = contributor.get("profile_url")
            label = f"[{name}](<{profile_url}>)" if profile_url else name
            details: list[str] = []
            team = contributor.get("team")
            if team:
                details.append(f"team: {_markdown_text(team)}")
            roles = contributor.get("roles")
            if isinstance(roles, list) and roles:
                details.append(
                    "roles: " + ", ".join(_markdown_text(role) for role in roles)
                )
            totals = contributor.get("contribution_totals")
            if isinstance(totals, Mapping):
                details.append(
                    f"{totals['accepted_assignments']} accepted assignments, "
                    f"{totals['accepted_tokens']} accepted tokens, "
                    f"{totals['worker_reported_total_seconds']} "
                    "worker-reported seconds"
                )
            lines.append(
                f"- {label}" + (f" — {'; '.join(details)}" if details else "")
            )
            hardware = contributor.get("hardware")
            if isinstance(hardware, list):
                for item in hardware:
                    if isinstance(item, Mapping):
                        hardware_detail = (
                            f" ({item['accepted_assignments']} accepted assignments)"
                            if "accepted_assignments" in item
                            else ""
                        )
                        lines.append(
                            "  - Hardware: "
                            f"{_markdown_text(item['hardware_class'])}"
                            f"{hardware_detail}"
                        )
    else:
        lines.append("- No contributor opted into public named or pseudonymous credit.")
    lines.extend(
        [
            "",
            "## Anonymous contributions",
            "",
            f"{anonymous.get('count', 0)} contributing participant(s) chose anonymous credit.",
            "",
            "See `attribution-snapshot.json` and `public-ledger.json` for the "
            "deterministic release totals and provenance.",
            "",
        ]
    )
    return "\n".join(lines)
