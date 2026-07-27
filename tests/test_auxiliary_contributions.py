from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orcacolony.auxiliary_contributions import (
    AuxiliaryContributionLedger,
    auxiliary_contribution_markdown,
    build_auxiliary_contribution_snapshot,
    copy_public_auxiliary_contribution_artifacts,
    load_auxiliary_contributions,
    validate_auxiliary_contribution_snapshot,
    verify_auxiliary_contribution_artifacts,
    verify_public_auxiliary_snapshot_artifacts,
)


CAMPAIGN_ID = "auxiliary-test"
CAMPAIGN_REVISION = "a" * 64
CHECKPOINT_SHA256 = "b" * 64


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload(
    *,
    public_digest: str = "1" * 64,
    anonymous_digest: str = "2" * 64,
) -> dict[str, object]:
    return {
        "format": "orcacolony_auxiliary_contributions_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_revision": CAMPAIGN_REVISION,
        "owner_reviewed": True,
        "contributors": [
            {
                "contributor_id": "private-public-id",
                "credit": {
                    "visibility": "pseudonymous",
                    "display_name": "Helpful Orca",
                    "profile_url": "https://huggingface.co/helpful-orca",
                    "team": "Community Lab",
                    "show_contribution_details": True,
                    "show_time": True,
                    "show_hardware": True,
                    "public_disclosure_confirmed": True,
                },
                "resources": {
                    "person_time_seconds": 3600,
                    "compute_time_seconds": 7200,
                    "hardware": ["consumer GPU, 24 GiB VRAM"],
                },
                "contributions": [
                    {
                        "id": "evaluation-fixture-review",
                        "kind": "evaluation-review",
                        "description": "Reviewed evaluator fixtures and recorded discrepancies.",
                        "status": "completed",
                        "evidence": [
                            {
                                "id": "review-record",
                                "sha256": public_digest,
                                "uri": "bundle:review/review-record.json",
                            }
                        ],
                    }
                ],
            },
            {
                "contributor_id": "private-withheld-id",
                "credit": {
                    "visibility": "named",
                    "display_name": "Withheld Reviewer",
                    "profile_url": None,
                    "team": None,
                    "show_contribution_details": False,
                    "show_time": False,
                    "show_hardware": False,
                    "public_disclosure_confirmed": True,
                },
                "resources": {
                    "person_time_seconds": 1800,
                    "compute_time_seconds": None,
                    "hardware": ["private workstation detail"],
                },
                "contributions": [
                    {
                        "id": "withheld-review",
                        "kind": "review",
                        "description": "Private description that must not be released.",
                        "status": "partial",
                        "evidence": [
                            {
                                "id": "withheld-record",
                                "sha256": "3" * 64,
                                "uri": "repo:private/withheld-record.json",
                            }
                        ],
                    }
                ],
            },
            {
                "contributor_id": "private-anonymous-id",
                "credit": {
                    "visibility": "anonymous",
                    "display_name": None,
                    "profile_url": None,
                    "team": None,
                    "show_contribution_details": False,
                    "show_time": False,
                    "show_hardware": False,
                    "public_disclosure_confirmed": True,
                },
                "resources": {
                    "person_time_seconds": None,
                    "compute_time_seconds": 900,
                    "hardware": ["private anonymous hardware"],
                },
                "contributions": [
                    {
                        "id": "failed-compute-check",
                        "kind": "compute-investigation",
                        "description": "Investigated a failed compute path.",
                        "status": "failed_informative",
                        "evidence": [
                            {
                                "id": "failure-record",
                                "sha256": anonymous_digest,
                                "uri": "bundle:private/failure-record.json",
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _ledger(payload: dict[str, object] | None = None) -> AuxiliaryContributionLedger:
    return AuxiliaryContributionLedger.from_payload(
        payload or _payload(),
        campaign_id=CAMPAIGN_ID,
        campaign_revision=CAMPAIGN_REVISION,
    )


def test_snapshot_is_deterministic_privacy_filtered_and_evidence_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    public_file = root / "review" / "review-record.json"
    anonymous_file = root / "private" / "failure-record.json"
    public_file.parent.mkdir(parents=True)
    anonymous_file.parent.mkdir(parents=True)
    public_file.write_text('{"review":"complete"}\n', encoding="utf-8")
    anonymous_file.write_text('{"failure":"recorded"}\n', encoding="utf-8")
    payload = _payload(
        public_digest=_sha256(public_file.read_bytes()),
        anonymous_digest=_sha256(anonymous_file.read_bytes()),
    )
    ledger = _ledger(payload)

    bindings = verify_auxiliary_contribution_artifacts(ledger, root)
    assert bindings == {
        "private/failure-record.json": {
            "sha256": _sha256(anonymous_file.read_bytes()),
            "public": False,
        },
        "review/review-record.json": {
            "sha256": _sha256(public_file.read_bytes()),
            "public": True,
        },
    }
    snapshot = build_auxiliary_contribution_snapshot(
        ledger,
        campaign_id=CAMPAIGN_ID,
        campaign_revision=CAMPAIGN_REVISION,
        release_checkpoint_sha256=CHECKPOINT_SHA256,
        release_checkpoint_step=7,
        verified_bundle_artifacts=bindings,
    )
    repeated = build_auxiliary_contribution_snapshot(
        ledger,
        campaign_id=CAMPAIGN_ID,
        campaign_revision=CAMPAIGN_REVISION,
        release_checkpoint_sha256=CHECKPOINT_SHA256,
        release_checkpoint_step=7,
        verified_bundle_artifacts=bindings,
    )
    assert snapshot == repeated
    assert snapshot["record_status"] == "owner_reviewed"
    assert snapshot["all_contributions"] == {
        "contributor_count": 3,
        "contribution_count": 3,
    }
    assert snapshot["public_resource_totals"] == {
        "person_time_seconds": 3600,
        "compute_time_seconds": 7200,
        "contributors_with_public_hardware": 1,
    }
    assert snapshot["anonymous_contributors"] == {
        "count": 1,
        "contribution_count": 1,
    }
    helpful = next(
        contributor
        for contributor in snapshot["public_contributors"]
        if contributor["display_name"] == "Helpful Orca"
    )
    assert helpful["resources"] == {
        "person_time_seconds": 3600,
        "compute_time_seconds": 7200,
        "hardware": ["consumer GPU, 24 GiB VRAM"],
    }
    assert helpful["contributions"][0]["evidence"][0]["verification"] == (
        "bundled_sha256_verified"
    )
    withheld = next(
        contributor
        for contributor in snapshot["public_contributors"]
        if contributor["display_name"] == "Withheld Reviewer"
    )
    assert withheld["contribution_details_withheld"] is True
    assert "resources" not in withheld

    serialized = json.dumps(snapshot, sort_keys=True)
    for private_value in (
        "private-public-id",
        "private-withheld-id",
        "private-anonymous-id",
        "Private description",
        "private workstation detail",
        "private anonymous hardware",
        "failure-record",
    ):
        assert private_value not in serialized

    destination = tmp_path / "public-evidence"
    copy_public_auxiliary_contribution_artifacts(
        root,
        destination,
        bindings,
    )
    assert (
        destination / "review" / "review-record.json"
    ).read_bytes() == public_file.read_bytes()
    assert not (destination / "private").exists()
    assert verify_public_auxiliary_snapshot_artifacts(
        snapshot,
        destination,
    ) == {
        "review/review-record.json": _sha256(public_file.read_bytes())
    }

    (destination / "unreferenced.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from snapshot bindings"):
        verify_public_auxiliary_snapshot_artifacts(snapshot, destination)

    markdown = auxiliary_contribution_markdown(snapshot)
    assert "Helpful Orca" in markdown
    assert "Withheld Reviewer" in markdown
    assert "failed-compute-check" not in markdown
    assert "private-public-id" not in markdown
    assert "consumer GPU, 24 GiB VRAM" in markdown
    assert "bundled_sha256_verified" in markdown
    assert "—" not in markdown


def test_absent_and_reviewed_empty_records_remain_distinct() -> None:
    absent = build_auxiliary_contribution_snapshot(
        None,
        campaign_id=CAMPAIGN_ID,
        campaign_revision=CAMPAIGN_REVISION,
        release_checkpoint_sha256=CHECKPOINT_SHA256,
        release_checkpoint_step=0,
    )
    assert absent["record_status"] == "not_supplied"
    assert absent["source_ledger_sha256"] is None
    assert "does not establish that no auxiliary work occurred" in (
        auxiliary_contribution_markdown(absent)
    )

    payload = _payload()
    payload["contributors"] = []
    reviewed = build_auxiliary_contribution_snapshot(
        _ledger(payload),
        campaign_id=CAMPAIGN_ID,
        campaign_revision=CAMPAIGN_REVISION,
        release_checkpoint_sha256=CHECKPOINT_SHA256,
        release_checkpoint_step=0,
    )
    assert reviewed["record_status"] == "owner_reviewed"
    assert reviewed["source_ledger_sha256"] is not None
    assert "contains no auxiliary contribution entries" in (
        auxiliary_contribution_markdown(reviewed)
    )
    validate_auxiliary_contribution_snapshot(reviewed)

    tampered = json.loads(json.dumps(reviewed))
    tampered["measurement_notes"].append("Unbound mutation.")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        validate_auxiliary_contribution_snapshot(tampered)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda payload: payload.update(owner_reviewed=False),
            "explicitly owner reviewed",
        ),
        (
            lambda payload: payload["contributors"][0]["credit"].update(
                public_disclosure_confirmed=False
            ),
            "public disclosure must be confirmed",
        ),
        (
            lambda payload: payload["contributors"][2]["credit"].update(
                show_contribution_details=True
            ),
            "anonymous credit may not publish",
        ),
        (
            lambda payload: payload["contributors"][0]["contributions"][0].update(
                status="accepted-training"
            ),
            "status is invalid",
        ),
        (
            lambda payload: payload["contributors"][0]["contributions"][0].update(
                evidence=[]
            ),
            "evidence must be a non-empty list",
        ),
        (
            lambda payload: payload.update(campaign_revision="0" * 64),
            "campaign revision differs",
        ),
    ),
)
def test_ledger_rejects_ambiguous_or_unapproved_records(
    mutation: object,
    error: str,
) -> None:
    payload = json.loads(json.dumps(_payload()))
    mutation(payload)  # type: ignore[operator]
    with pytest.raises(ValueError, match=error):
        _ledger(payload)


def test_loader_rejects_duplicate_keys_and_unsafe_bundle_paths(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"format":"orcacolony_auxiliary_contributions_v1",'
        '"format":"orcacolony_auxiliary_contributions_v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate auxiliary"):
        load_auxiliary_contributions(
            duplicate,
            campaign_id=CAMPAIGN_ID,
            campaign_revision=CAMPAIGN_REVISION,
        )

    payload = _payload()
    payload["contributors"][0]["contributions"][0]["evidence"][0][
        "uri"
    ] = "bundle:../private.json"
    with pytest.raises(ValueError, match="path is unsafe"):
        _ledger(payload)


def test_bundle_verification_rejects_digest_changes_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    public = root / "review" / "review-record.json"
    anonymous = root / "private" / "failure-record.json"
    public.parent.mkdir(parents=True)
    anonymous.parent.mkdir(parents=True)
    public.write_bytes(b"public")
    anonymous.write_bytes(b"anonymous")
    payload = _payload(
        public_digest=_sha256(public.read_bytes()),
        anonymous_digest=_sha256(anonymous.read_bytes()),
    )
    ledger = _ledger(payload)
    public.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        verify_auxiliary_contribution_artifacts(ledger, root)

    public.unlink()
    public.symlink_to(anonymous)
    with pytest.raises(ValueError, match="may not contain symlinks"):
        verify_auxiliary_contribution_artifacts(ledger, root)
