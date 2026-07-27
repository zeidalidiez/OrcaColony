from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.participants import (
    ParticipantRegistry,
    attribution_markdown,
    build_attribution_snapshot,
    load_participants,
)


def _registry() -> ParticipantRegistry:
    token_sha256 = hashlib.sha256(b"test-token").hexdigest()
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v2",
            "campaign_id": "credit-test",
            "participants": [
                {
                    "contributor_id": "private-public-id",
                    "worker_ids": ["worker-public"],
                    "worker_token_sha256": {
                        "worker-public": token_sha256,
                    },
                    "credit": {
                        "visibility": "pseudonymous",
                        "display_name": "Helpful Orca",
                        "profile_url": "https://huggingface.co/helpful-orca",
                        "team": "Community Lab",
                        "roles": ["training-compute", "evaluation"],
                        "show_contribution_totals": True,
                        "show_hardware": True,
                    },
                    "worker_profiles": {
                        "worker-public": {
                            "hardware_class": "consumer CPU, 32 GiB RAM",
                            "public": True,
                        }
                    },
                },
                {
                    "contributor_id": "private-anonymous-id",
                    "worker_ids": ["worker-private"],
                    "worker_token_sha256": {
                        "worker-private": token_sha256,
                    },
                    "credit": {
                        "visibility": "anonymous",
                        "display_name": None,
                        "profile_url": None,
                        "team": None,
                        "roles": ["training-compute"],
                        "show_contribution_totals": False,
                        "show_hardware": False,
                    },
                    "worker_profiles": {},
                },
            ],
        },
        campaign_id="credit-test",
    )


def _entry(
    contributor_id: str,
    worker_id: str,
    tokens: int,
    seconds: float,
) -> dict[str, object]:
    return {
        "contributor_id": contributor_id,
        "worker_id": worker_id,
        "loss_weight_sum": tokens,
        "instrumentation": {
            "worker_reported": {
                "runtime_seconds": {
                    "gradient_compute": seconds,
                }
            }
        },
    }


def test_v2_attribution_snapshot_credits_opted_in_hardware_and_hides_ids() -> None:
    snapshot = build_attribution_snapshot(
        _registry(),
        [
            _entry("private-public-id", "worker-public", 128, 2.5),
            _entry("private-anonymous-id", "worker-private", 64, 1.0),
        ],
    )
    public = snapshot["public_contributors"][0]
    assert public["display_name"] == "Helpful Orca"
    assert public["contribution_totals"]["accepted_tokens"] == 128
    assert public["hardware"] == [
        {
            "hardware_class": "consumer CPU, 32 GiB RAM",
            "accepted_assignments": 1,
            "accepted_tokens": 128,
            "worker_reported_total_seconds": 2.5,
            "worker_reported_gradient_seconds": 2.5,
        }
    ]
    assert snapshot["anonymous_contributors"]["count"] == 1
    serialized = json.dumps(snapshot)
    assert "private-public-id" not in serialized
    assert "private-anonymous-id" not in serialized
    assert "worker-public" not in serialized
    markdown = attribution_markdown(snapshot)
    assert "Helpful Orca" in markdown
    assert "1 contributing participant(s) chose anonymous credit" in markdown


def test_v2_credit_profile_rejects_non_https_profile_url() -> None:
    payload = _registry().as_payload()
    payload["participants"][0]["credit"]["profile_url"] = "http://example.test"
    with pytest.raises(ValueError, match="HTTPS URL"):
        ParticipantRegistry.from_payload(payload, campaign_id="credit-test")


def test_v2_separates_worker_authority_from_public_credit_revision() -> None:
    original = _registry()
    payload = original.as_payload()
    public = next(
        participant
        for participant in payload["participants"]
        if participant["contributor_id"] == "private-public-id"
    )
    public["credit"]["display_name"] = "Updated Alias"
    updated_credit = ParticipantRegistry.from_payload(
        payload,
        campaign_id="credit-test",
    )
    assert updated_credit.revision == original.revision
    assert updated_credit.credit_revision != original.credit_revision

    public["worker_token_sha256"]["worker-public"] = "0" * 64
    updated_authority = ParticipantRegistry.from_payload(
        payload,
        campaign_id="credit-test",
    )
    assert updated_authority.revision != original.revision


def test_participant_manifest_rejects_unknown_and_duplicate_fields(
    tmp_path: Path,
) -> None:
    payload = _registry().as_payload()
    payload["participants"][0]["worker_token"] = "must-not-be-accepted"
    with pytest.raises(ValueError, match="unknown fields"):
        ParticipantRegistry.from_payload(payload, campaign_id="credit-test")

    duplicate = tmp_path / "participants.json"
    duplicate.write_text(
        '{"format":"orcacolony_participants_v2",'
        '"format":"orcacolony_participants_v2",'
        '"campaign_id":"credit-test","participants":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate participant JSON key"):
        load_participants(duplicate, campaign_id="credit-test")
