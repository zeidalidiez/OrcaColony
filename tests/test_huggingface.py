from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from orcacolony.huggingface import (
    build_huggingface_packages,
    publish_huggingface_packages,
    verify_huggingface_packages,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _release(
    root: Path,
    *,
    visibility_policy: str | None = None,
    campaign_result: bool = False,
    auxiliary_status: str = "owner_reviewed",
    include_auxiliary_contributor: bool = True,
) -> None:
    campaign = {
        "campaign": {
            "id": "orcacolony-hub-test",
            "objective": "causal_lm",
            "loss_mask": "all_target_tokens",
        },
        "model": {
            "architecture": "volunteer_decoder_v1",
            "architecture_revision": 1,
            "layers": 1,
            "width": 8,
            "heads": 1,
            "mlp_width": 16,
            "vocabulary_size": 32,
            "context_length": 8,
            "positional_encoding": "learned_absolute",
            "layer_norm_epsilon": 0.00001,
            "gelu_approximation": "tanh",
            "attention_bias": True,
            "linear_bias": True,
            "tied_token_embeddings": True,
            "parameters": 1000,
        },
        "training": {
            "seed": 1,
            "batch_size": 1,
            "dataset_sequences": 1,
            "active_vocabulary_size": 32,
            "steps": 1,
            "learning_rate": 0.001,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "compute_dtype": "float32",
            "gradient_accumulation_dtype": "float32",
        },
    }
    if visibility_policy is not None:
        campaign["publication"] = {
            "format": "orcacolony_huggingface_publication_v1",
            "model_repo_id": "OrcaColony/orcacolony-hub-test",
            "dataset_repo_id": "OrcaColony/orcacolony-hub-test-dataset",
            "model_license": "mit",
            "dataset_license": "cdla-sharing-1.0",
            "visibility_policy": visibility_policy,
        }
    if campaign_result:
        campaign["research"] = {
            "format": "orcacolony_campaign_research_v2",
            "question": "What changed in the test campaign?",
            "usage_scenario": "A test-only usage scenario.",
        }
    campaign_revision = "c" * 64
    auxiliary_artifact = b'{"review":"complete"}\n'
    auxiliary_public_contributors = (
        [
            {
                "display_name": "Hub Helper",
                "visibility": "pseudonymous",
                "profile_url": None,
                "team": None,
                "credit_profile_revision": "f" * 64,
                "contributions": [
                    {
                        "id": "hub-review",
                        "kind": "release-review",
                        "description": "Reviewed the Hub fixture.",
                        "status": "completed",
                        "evidence": [
                            {
                                "id": "hub-review-record",
                                "sha256": hashlib.sha256(
                                    auxiliary_artifact
                                ).hexdigest(),
                                "uri": "bundle:review.json",
                                "verification": "bundled_sha256_verified",
                            }
                        ],
                    }
                ],
                "resources": {
                    "person_time_seconds": 60,
                    "compute_time_seconds": 120,
                    "hardware": ["test host"],
                },
            }
        ]
        if (
            auxiliary_status == "owner_reviewed"
            and include_auxiliary_contributor
        )
        else []
    )
    auxiliary_count = len(auxiliary_public_contributors)
    auxiliary_snapshot: dict[str, object] = {
        "format": "orcacolony_auxiliary_contribution_snapshot_v1",
        "campaign_id": "orcacolony-hub-test",
        "campaign_revision": campaign_revision,
        "release_checkpoint_sha256": hashlib.sha256(b"model").hexdigest(),
        "release_checkpoint_step": 1,
        "record_status": auxiliary_status,
        "source_ledger_sha256": (
            "e" * 64 if auxiliary_status == "owner_reviewed" else None
        ),
        "public_contributors": auxiliary_public_contributors,
        "anonymous_contributors": {
            "count": 0,
            "contribution_count": 0,
        },
        "all_contributions": {
            "contributor_count": auxiliary_count,
            "contribution_count": auxiliary_count,
        },
        "public_resource_totals": {
            "person_time_seconds": 60 * auxiliary_count,
            "compute_time_seconds": 120 * auxiliary_count,
            "contributors_with_public_hardware": auxiliary_count,
        },
        "measurement_notes": [],
    }
    auxiliary_snapshot_sha256 = hashlib.sha256(
        json.dumps(
            auxiliary_snapshot,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    auxiliary_snapshot["snapshot_sha256"] = auxiliary_snapshot_sha256
    text_files = {
        "campaign.json": json.dumps(campaign).encode(),
        "campaign-lock.json": json.dumps(
            {
                "format": "orcacolony_campaign_lock_v1",
                "campaign_id": "orcacolony-hub-test",
                "checkpoint_sha256": hashlib.sha256(b"initial").hexdigest(),
            }
        ).encode(),
        "evaluations.json": b"{}\n",
        "public-ledger.json": b"{}\n",
        "attribution-snapshot.json": json.dumps(
            {
                "format": "orcacolony_attribution_snapshot_v1",
                "all_contributions": {
                    "accepted_assignments": 2,
                    "accepted_tokens": 16,
                },
                "public_contributors": [],
                "anonymous_contributors": {"count": 1},
                "snapshot_sha256": "test",
            }
        ).encode(),
        "auxiliary-contribution-snapshot.json": json.dumps(
            auxiliary_snapshot
        ).encode(),
        "CONTRIBUTORS.md": (
            b"# Community contributors\n\n## Auxiliary contributions\n"
        ),
        "LICENSE": b"test license\n",
        "THIRD_PARTY_DATA.md": b"third-party data\n",
        "dataset/manifest.json": json.dumps(
            {
                "source": {
                    "dataset": "test/source",
                    "revision": "a" * 40,
                    "selection": "test fixture",
                    "license": "cdla-sharing-1.0",
                },
                "packing": {
                    "train_sequences": 1,
                    "train_tokens": 8,
                    "validation_sequences": 1,
                    "validation_tokens": 8,
                },
            }
        ).encode(),
        "dataset/tokenizer.json": b"{}\n",
        "dataset/DATASET-NOTICE.md": b"dataset notice\n",
    }
    for name, payload in text_files.items():
        _write(root / name, payload)
    if auxiliary_count:
        _write(
            root / "auxiliary-contribution-artifacts" / "review.json",
            auxiliary_artifact,
        )
    if campaign_result:
        _write(
            root / "campaign-evaluation-evidence.json",
            json.dumps(
                {
                    "format": "orcacolony_campaign_evaluation_evidence_v1",
                    "limitations": ["Test-only evidence."],
                }
            ).encode(),
        )
        _write(
            root / "campaign-evaluation-summary.json",
            json.dumps(
                {
                    "format": "orcacolony_campaign_evaluation_summary_v1",
                    "comparisons": [
                        {
                            "summary": "Compare the two owner-selected records.",
                            "metrics": [
                                {
                                    "label": "Usage score",
                                    "unit": "ratio",
                                    "baseline_value": 0.1,
                                    "candidate_value": 0.2,
                                    "absolute_change": 0.1,
                                    "direction": "maximize",
                                }
                            ],
                        }
                    ],
                    "findings": [
                        {
                            "label": "Measured change",
                            "kind": "improvement",
                            "description": "The declared score increased.",
                        }
                    ],
                    "limitations": ["Test-only evidence."],
                }
            ).encode(),
        )
        _write(
            root / "campaign-evaluation-artifacts" / "samples.json",
            b'{"sample":"test"}\n',
        )
    _write(root / "dataset/train.safetensors", b"train")
    _write(root / "dataset/validation.safetensors", b"validation")
    _write(root / "checkpoint/model.safetensors", b"model")
    _write(root / "checkpoint/optimizer.safetensors", b"optimizer")
    _write(root / "checkpoint/state.json", b"{}\n")
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": "orcacolony_release_bundle_v1",
        "campaign_id": "orcacolony-hub-test",
        "campaign_revision": campaign_revision,
        "auxiliary_contribution_record_status": auxiliary_status,
        "auxiliary_contribution_snapshot_sha256": (
            auxiliary_snapshot_sha256
        ),
        "release_classification": (
            "campaign_result" if campaign_result else "systems_evidence_only"
        ),
        "checkpoint": {
            "step": 1,
            "selection": "final",
            "model_sha256": hashlib.sha256(b"model").hexdigest(),
        },
        "files": files,
    }
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{files[name]}  {name}\n" for name in sorted(files))
        + f"{_sha256(root / 'release-manifest.json')}  release-manifest.json\n",
        encoding="utf-8",
    )


def test_huggingface_package_build_is_deterministic_and_separates_repos(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    first = tmp_path / "hub-a"
    second = tmp_path / "hub-b"
    kwargs = {
        "model_repo_id": "OrcaColony/orcacolony-hub-test",
        "dataset_repo_id": "OrcaColony/orcacolony-hub-test-dataset",
        "model_license": "mit",
        "dataset_license": "cdla-sharing-1.0",
        "source_repository": "https://github.com/zeidalidiez/OrcaColony",
        "source_revision": "a" * 40,
    }
    first_manifest = build_huggingface_packages(release, first, **kwargs)
    second_manifest = build_huggingface_packages(release, second, **kwargs)

    assert first_manifest == second_manifest
    assert first_manifest["visibility"] == "private"
    assert first_manifest["auxiliary_contribution_record_status"] == (
        "owner_reviewed"
    )
    release_auxiliary_snapshot = json.loads(
        (
            release / "auxiliary-contribution-snapshot.json"
        ).read_text(encoding="utf-8")
    )
    assert first_manifest["auxiliary_contribution_snapshot_sha256"] == (
        release_auxiliary_snapshot["snapshot_sha256"]
    )
    assert (first / "SHA256SUMS").read_bytes() == (
        second / "SHA256SUMS"
    ).read_bytes()
    assert (first / "model" / "model.safetensors").is_file()
    assert (first / "model" / "optimizer.safetensors").is_file()
    assert (first / "model" / "checkpoint-state.json").is_file()
    assert (first / "model" / "MODEL-LICENSE.md").is_file()
    assert (first / "model" / "ORCACOLONY-SOFTWARE-LICENSE").is_file()
    assert (first / "model" / "README.md").is_file()
    assert (
        first / "model" / "auxiliary-contribution-snapshot.json"
    ).is_file()
    assert (
        first
        / "model"
        / "auxiliary-contribution-artifacts"
        / "review.json"
    ).is_file()
    assert (first / "dataset" / "train.safetensors").is_file()
    assert (
        first
        / "dataset"
        / "auxiliary-contribution-artifacts"
        / "review.json"
    ).is_file()
    assert (first / "dataset" / "DATASET-LICENSE.md").is_file()
    assert (first / "dataset" / "README.md").is_file()
    assert verify_huggingface_packages(first) == first_manifest
    model_card = (first / "model" / "README.md").read_text(encoding="utf-8")
    assert "systems-evidence checkpoint" in model_card
    assert "OrcaColony/orcacolony-hub-test-dataset" in model_card
    assert "Community contributors" in model_card
    assert "1` chose anonymous credit" in model_card
    assert "owner-reviewed auxiliary record contains `1`" in model_card

    _write(first / "model" / "unmanifested.py", b"unexpected")
    with pytest.raises(ValueError, match="unmanifested or missing"):
        verify_huggingface_packages(first)


def test_huggingface_package_carries_campaign_evidence_and_artifacts(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release, campaign_result=True)

    package = tmp_path / "hub"
    manifest = build_huggingface_packages(
        release,
        package,
        model_repo_id="OrcaColony/orcacolony-hub-test",
        dataset_repo_id="OrcaColony/orcacolony-hub-test-dataset",
        model_license="mit",
        dataset_license="cdla-sharing-1.0",
        source_repository="https://github.com/zeidalidiez/OrcaColony",
        source_revision="a" * 40,
    )

    assert manifest["release_classification"] == "campaign_result"
    for repository in ("model", "dataset"):
        assert (
            package / repository / "campaign-evaluation-evidence.json"
        ).is_file()
        assert (
            package / repository / "campaign-evaluation-summary.json"
        ).is_file()
        assert (
            package
            / repository
            / "campaign-evaluation-artifacts"
            / "samples.json"
        ).is_file()
    model_card = (package / "model" / "README.md").read_text(encoding="utf-8")
    assert "does not assign a pass or promotion decision" in model_card
    assert "Usage score: `0.1` to `0.2` ratio" in model_card
    assert verify_huggingface_packages(package) == manifest


def test_huggingface_package_rejects_personal_namespace(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    with pytest.raises(ValueError, match="OrcaColony organization"):
        build_huggingface_packages(
            release,
            tmp_path / "hub",
            model_repo_id="personal/model",
            dataset_repo_id="OrcaColony/model-dataset",
            model_license="mit",
            dataset_license="cdla-sharing-1.0",
            source_repository="https://github.com/zeidalidiez/OrcaColony",
            source_revision="a" * 40,
        )


def test_huggingface_package_rejects_placeholder_license(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    with pytest.raises(ValueError, match="explicit Hugging Face license"):
        build_huggingface_packages(
            release,
            tmp_path / "hub",
            model_repo_id="OrcaColony/model",
            dataset_repo_id="OrcaColony/model-dataset",
            model_license="choose-explicitly",
            dataset_license="cdla-sharing-1.0",
            source_repository="https://github.com/zeidalidiez/OrcaColony",
            source_revision="a" * 40,
        )


@pytest.mark.parametrize("visibility", ("private", "public"))
def test_huggingface_package_allows_review_then_public_policy(
    tmp_path: Path,
    visibility: str,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(
        release,
        visibility_policy="private_review_then_public",
    )

    manifest = build_huggingface_packages(
        release,
        tmp_path / "hub",
        model_repo_id="OrcaColony/orcacolony-hub-test",
        dataset_repo_id="OrcaColony/orcacolony-hub-test-dataset",
        model_license="mit",
        dataset_license="cdla-sharing-1.0",
        source_repository="https://github.com/zeidalidiez/OrcaColony",
        source_revision="a" * 40,
        visibility=visibility,
    )

    assert manifest["visibility"] == visibility


def test_huggingface_package_rejects_visibility_outside_campaign_policy(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release, visibility_policy="private")

    with pytest.raises(ValueError, match="campaign policy"):
        build_huggingface_packages(
            release,
            tmp_path / "hub",
            model_repo_id="OrcaColony/orcacolony-hub-test",
            dataset_repo_id="OrcaColony/orcacolony-hub-test-dataset",
            model_license="mit",
            dataset_license="cdla-sharing-1.0",
            source_repository="https://github.com/zeidalidiez/OrcaColony",
            source_revision="a" * 40,
            visibility="public",
        )


def test_public_package_requires_reviewed_auxiliary_record(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(
        release,
        visibility_policy="private_review_then_public",
        auxiliary_status="not_supplied",
    )

    with pytest.raises(ValueError, match="owner-reviewed auxiliary"):
        build_huggingface_packages(
            release,
            tmp_path / "hub",
            model_repo_id="OrcaColony/orcacolony-hub-test",
            dataset_repo_id="OrcaColony/orcacolony-hub-test-dataset",
            model_license="mit",
            dataset_license="cdla-sharing-1.0",
            source_repository="https://github.com/zeidalidiez/OrcaColony",
            source_revision="a" * 40,
            visibility="public",
        )


def test_public_package_accepts_reviewed_empty_auxiliary_record(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(
        release,
        visibility_policy="private_review_then_public",
        include_auxiliary_contributor=False,
    )

    manifest = build_huggingface_packages(
        release,
        tmp_path / "hub",
        model_repo_id="OrcaColony/orcacolony-hub-test",
        dataset_repo_id="OrcaColony/orcacolony-hub-test-dataset",
        model_license="mit",
        dataset_license="cdla-sharing-1.0",
        source_repository="https://github.com/zeidalidiez/OrcaColony",
        source_revision="a" * 40,
        visibility="public",
    )

    assert manifest["visibility"] == "public"


def test_publish_refuses_existing_visibility_mismatch_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    package = tmp_path / "hub"
    build_huggingface_packages(
        release,
        package,
        model_repo_id="OrcaColony/model",
        dataset_repo_id="OrcaColony/model-dataset",
        model_license="mit",
        dataset_license="cdla-sharing-1.0",
        source_repository="https://github.com/zeidalidiez/OrcaColony",
        source_revision="a" * 40,
    )

    class FakeApi:
        def __init__(self) -> None:
            self.created: list[object] = []
            self.uploaded: list[object] = []

        def whoami(self) -> dict[str, str]:
            return {"name": "test-user"}

        def repo_exists(self, **_: object) -> bool:
            return True

        def repo_info(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(private=False, siblings=[])

        def create_repo(self, **kwargs: object) -> None:
            self.created.append(kwargs)

        def upload_folder(self, **kwargs: object) -> None:
            self.uploaded.append(kwargs)

    api = FakeApi()
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.HfApi = lambda: api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(RuntimeError, match="visibility differs"):
        publish_huggingface_packages(
            package,
            commit_message="test publish",
        )
    assert api.created == []
    assert api.uploaded == []
