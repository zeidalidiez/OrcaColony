from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Participant:
    contributor_id: str
    worker_ids: tuple[str, ...]
    worker_token_sha256: tuple[tuple[str, str], ...]
    public_credit: bool
    display_name: str | None


@dataclass(frozen=True)
class ParticipantRegistry:
    campaign_id: str
    participants: tuple[Participant, ...]
    revision: str

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        campaign_id: str,
    ) -> ParticipantRegistry:
        if payload.get("format") != "orcacolony_participants_v1":
            raise ValueError("unsupported participant manifest format")
        if payload.get("campaign_id") != campaign_id:
            raise ValueError("participant manifest campaign mismatch")
        raw_participants = payload.get("participants")
        if not isinstance(raw_participants, list) or not raw_participants:
            raise ValueError("participant manifest must include at least one participant")

        participants: list[Participant] = []
        contributor_ids: set[str] = set()
        all_worker_ids: set[str] = set()
        for raw in raw_participants:
            if not isinstance(raw, dict):
                raise ValueError("participant entries must be objects")
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
            public_credit = raw_credit.get("public", False)
            if not isinstance(public_credit, bool):
                raise ValueError(f"participant {contributor_id} public credit must be boolean")
            display_name = raw_credit.get("display_name")
            if display_name is not None:
                if not isinstance(display_name, str) or not display_name.strip():
                    raise ValueError(
                        f"participant {contributor_id} display name must be non-empty"
                    )
                display_name = display_name.strip()
            if public_credit and display_name is None:
                raise ValueError(
                    f"participant {contributor_id} needs a display name for public credit"
                )
            participants.append(
                Participant(
                    contributor_id=contributor_id,
                    worker_ids=worker_ids,
                    worker_token_sha256=tuple(worker_token_sha256),
                    public_credit=public_credit,
                    display_name=display_name,
                )
            )

        participants.sort(key=lambda value: value.contributor_id)
        normalized = _payload(campaign_id, participants)
        revision = hashlib.sha256(_canonical_json(normalized)).hexdigest()
        return cls(
            campaign_id=campaign_id,
            participants=tuple(participants),
            revision=revision,
        )

    def as_payload(self) -> dict[str, object]:
        return _payload(self.campaign_id, self.participants)

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


def _payload(
    campaign_id: str,
    participants: list[Participant] | tuple[Participant, ...],
) -> dict[str, object]:
    return {
        "format": "orcacolony_participants_v1",
        "campaign_id": campaign_id,
        "participants": [
            {
                "contributor_id": participant.contributor_id,
                "worker_ids": list(participant.worker_ids),
                "worker_token_sha256": dict(participant.worker_token_sha256),
                "credit": {
                    "public": participant.public_credit,
                    "display_name": participant.display_name,
                },
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("participant manifest must be a JSON object")
    return ParticipantRegistry.from_payload(payload, campaign_id=campaign_id)
