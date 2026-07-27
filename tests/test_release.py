import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony import peft
from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.campaign_research import campaign_research_revision
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import LeasedGradient
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest
from orcacolony.reference import campaign_from_mapping, load_campaign
from orcacolony.release import (
    _copy_campaign_evaluation_artifacts,
    _validate_promotion_evidence,
    build_release_bundle,
    validate_public_dataset_manifest,
)


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = (
    Path(__file__).parents[1] / "campaign" / "t0-lora-smoke-cpu.json"
)


def _capability_campaign():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["evaluation"] = {
        "validation_start_sequence": 0,
        "validation_sequences": 4,
        "batch_size": 2,
        "final_holdout": {
            "start_sequence": 4,
            "sequence_count": 4,
            "batch_size": 2,
        },
    }
    payload["research"] = {
        "format": "orcacolony_capability_research_v1",
        "claim": "Improve the frozen task suite.",
        "baseline": {
            "id": "initialization",
            "description": "Exact initialization.",
            "revision": "sha256:" + "1" * 64,
        },
        "primary_metric": {
            "id": "task-score",
            "description": "Frozen task score.",
            "direction": "maximize",
            "unit": "ratio",
            "success_threshold": 0.7,
            "minimum_improvement_from_baseline": 0.05,
        },
        "guardrails": [
            {"id": "valid-output", "description": "Outputs remain valid."}
        ],
        "analysis_plan": ["Compare sample-level output changes."],
        "final_holdout_policy": "release_only_after_checkpoint_selection",
        "checkpoint_selection": (
            "lowest_validation_mean_loss_before_behavioral_final_holdout"
        ),
        "behavioral_evaluation": {
            "suite_id": "frozen-task-suite",
            "dataset_revision": "sha256:" + "2" * 64,
            "evaluator_revision": "3" * 40,
            "validation_split": "validation",
            "final_holdout_split": "final_holdout",
        },
    }
    payload["publication"] = {
        "format": "orcacolony_huggingface_publication_v1",
        "model_repo_id": "OrcaColony/test-capability-model",
        "dataset_repo_id": "OrcaColony/test-capability-model-dataset",
        "model_license": "mit",
        "dataset_license": "cdla-sharing-1.0",
        "visibility_policy": "private_review_then_public",
    }
    return campaign_from_mapping(payload)


def _promotion_evidence() -> dict[str, object]:
    return {
        "format": "orcacolony_capability_promotion_evidence_v1",
        "campaign_id": "orcacolony-t0-smoke-v1",
        "checkpoint_sha256": "4" * 64,
        "dataset_revision": "5" * 64,
        "evaluation_suite": {
            "suite_id": "frozen-task-suite",
            "dataset_revision": "sha256:" + "2" * 64,
            "evaluator_revision": "3" * 40,
            "split": "final_holdout",
        },
        "primary_metric": {
            "id": "task-score",
            "value": 0.75,
            "baseline": {
                "id": "initialization",
                "revision": "sha256:" + "1" * 64,
                "value": 0.65,
            },
        },
        "guardrails": [
            {
                "id": "valid-output",
                "passed": True,
                "detail": "All frozen outputs parsed.",
            }
        ],
        "limitations": ["One deterministic seed was evaluated."],
        "artifacts": [
            {
                "id": "sample-results",
                "sha256": "6" * 64,
                "uri": "repo:reports/evidence/sample-results.json",
            }
        ],
        "reproduction": {
            "command": ["python", "evaluate.py", "--split", "final_holdout"],
            "notes": "Run from the exact evaluator revision.",
        },
    }


def _campaign_research() -> dict[str, object]:
    return {
        "format": "orcacolony_campaign_research_v2",
        "question": "What changed under the owner-supplied usage evaluation?",
        "usage_scenario": "A test-only owner-defined usage scenario.",
        "evaluation_contract": {
            "evaluator": {
                "id": "test-evaluator",
                "revision": "7" * 40,
                "command": ["python", "evaluate.py"],
            },
            "artifacts": [
                {
                    "id": "test-inputs",
                    "kind": "dataset",
                    "revision": "sha256:" + "8" * 64,
                    "uri": "hf://datasets/OrcaColony/test-inputs@revision",
                }
            ],
            "metrics": [
                {
                    "id": "usage-score",
                    "label": "Usage score",
                    "description": "Test-only owner-defined usage score.",
                    "direction": "maximize",
                    "unit": "ratio",
                }
            ],
        },
        "analysis_plan": ["Compare the two test evaluation snapshots."],
    }


def _campaign_evidence(
    research: dict[str, object],
    *,
    campaign_id: str,
    campaign_revision: str,
    release_checkpoint_sha256: str,
    initial_artifact_sha256: str,
    released_artifact_sha256: str,
) -> dict[str, object]:
    return {
        "format": "orcacolony_campaign_evaluation_evidence_v1",
        "campaign_id": campaign_id,
        "campaign_revision": campaign_revision,
        "research_revision": campaign_research_revision(research),
        "release_evaluation_id": "released",
        "evaluations": [
            {
                "id": "initial",
                "label": "Initial checkpoint",
                "subject": {
                    "id": "initial-model",
                    "label": "Initial model",
                    "revision": "9" * 64,
                },
                "measurements": [{"metric_id": "usage-score", "value": 0.1}],
                "artifacts": [
                    {
                        "id": "initial-samples",
                        "sha256": initial_artifact_sha256,
                        "uri": "bundle:initial-samples.json",
                    }
                ],
            },
            {
                "id": "released",
                "label": "Released checkpoint",
                "subject": {
                    "id": "released-model",
                    "label": "Released model",
                    "revision": release_checkpoint_sha256,
                },
                "measurements": [{"metric_id": "usage-score", "value": 0.2}],
                "artifacts": [
                    {
                        "id": "released-samples",
                        "sha256": released_artifact_sha256,
                        "uri": "bundle:released-samples.json",
                    }
                ],
            },
        ],
        "comparisons": [
            {
                "id": "initial-to-released",
                "baseline_evaluation_id": "initial",
                "candidate_evaluation_id": "released",
                "summary": "Test comparison requested by the campaign owner.",
            }
        ],
        "findings": [
            {
                "id": "test-finding",
                "label": "Test finding",
                "kind": "improvement",
                "description": "The declared test score increased by 0.1.",
            }
        ],
        "limitations": ["This is release integration test evidence."],
        "reproduction": {
            "command": ["python", "evaluate.py"],
            "notes": "Run against the bundled test artifacts.",
        },
    }


def _participants(campaign_id: object) -> ParticipantRegistry:
    worker_ids = ["release-a", "release-b"]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "private-release-contributor",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(b"release-test-token").hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": "Hidden Name"},
                }
            ],
        },
        campaign_id=str(campaign_id),
    )


def test_promotion_evidence_requires_threshold_and_baseline_improvement() -> None:
    campaign = _capability_campaign()
    evidence = _promotion_evidence()
    assert _validate_promotion_evidence(
        campaign,
        evidence,
        checkpoint_sha256="4" * 64,
        dataset_revision="5" * 64,
    )

    evidence["primary_metric"]["baseline"]["value"] = 0.72
    assert not _validate_promotion_evidence(
        campaign,
        evidence,
        checkpoint_sha256="4" * 64,
        dataset_revision="5" * 64,
    )


def test_campaign_evaluation_artifact_bundle_fails_closed(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    sample = artifact_root / "samples.json"
    sample.write_text('{"sample":"evidence"}\n', encoding="utf-8")
    evidence = {
        "evaluations": [
            {
                "artifacts": [
                    {
                        "id": "samples",
                        "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                        "uri": "bundle:samples.json",
                    }
                ]
            }
        ]
    }
    destination = tmp_path / "release-artifacts"

    _copy_campaign_evaluation_artifacts(
        evidence,
        artifact_root,
        destination,
    )
    assert (destination / "samples.json").read_bytes() == sample.read_bytes()

    evidence["evaluations"][0]["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest differs"):
        _copy_campaign_evaluation_artifacts(
            evidence,
            artifact_root,
            tmp_path / "wrong-digest",
        )

    evidence["evaluations"][0]["artifacts"][0]["sha256"] = hashlib.sha256(
        sample.read_bytes()
    ).hexdigest()
    evidence["evaluations"][0]["artifacts"][0]["uri"] = "bundle:../samples.json"
    with pytest.raises(ValueError, match="path is unsafe"):
        _copy_campaign_evaluation_artifacts(
            evidence,
            artifact_root,
            tmp_path / "traversal",
        )

    linked = artifact_root / "linked.json"
    linked.symlink_to(sample)
    evidence["evaluations"][0]["artifacts"][0]["uri"] = "bundle:linked.json"
    with pytest.raises(ValueError, match="may not contain symlinks"):
        _copy_campaign_evaluation_artifacts(
            evidence,
            artifact_root,
            tmp_path / "symlink",
        )


@pytest.mark.parametrize(
    "revision",
    (
        "sha256:" + "1" * 40,
        "1" * 64,
    ),
)
def test_capability_contract_requires_unambiguous_pinned_revisions(
    revision: str,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    campaign = _capability_campaign()
    payload["evaluation"] = dict(campaign.evaluation)
    payload["research"] = json.loads(json.dumps(campaign.research))
    payload["publication"] = dict(campaign.publication)
    payload["research"]["baseline"]["revision"] = revision

    with pytest.raises(ValueError, match="40-character lowercase Git revision"):
        campaign_from_mapping(payload)


def _submission(
    coordinator: CampaignCoordinator,
    assignment: dict[str, object],
    *,
    runtime_backend: str = "python-oracle-f32",
) -> LeasedGradient:
    assignment_id = str(assignment["assignment_id"])
    return LeasedGradient(
        assignment_id=assignment_id,
        lease_token=str(assignment["lease_token"]),
        checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        loss_sum=float(assignment["expected_loss_sum"]),
        loss_weight_sum=int(assignment["loss_weight_sum"]),
        safetensors=coordinator.oracle_gradient_path(assignment_id).read_bytes(),
        runtime_backend=runtime_backend,
    )


def test_release_bundle_is_deterministic_complete_and_privacy_filtered(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    corpus = (
        "A curious otter crossed the creek and found its family.\n"
        "<|endoftext|>\n"
    ) * 400
    dataset_root = tmp_path / "dataset"
    artifact_manifest = build_dataset_artifacts(
        train_bytes=corpus.encode(),
        validation_bytes=corpus[: len(corpus) // 2].encode(),
        output_dir=dataset_root,
        source={
            "dataset": "test/release-stories",
            "revision": "release-test-revision",
            "license": "cdla-sharing-1.0",
        },
        vocab_size=300,
        context_length=campaign.model.context_length,
    )
    dataset = PackedDataset.load(dataset_root)
    campaign = replace(
        campaign,
        dataset={
            "format": artifact_manifest["format"],
            "manifest_sha256": dataset.revision,
            "tokenizer_sha256": artifact_manifest["tokenizer"]["sha256"],
            "train_sha256": artifact_manifest["files"]["train.safetensors"],
            "validation_sha256": artifact_manifest["files"]["validation.safetensors"],
        },
        evaluation={
            "validation_sequences": 4,
            "batch_size": 2,
            "success_gate": {
                "metric": "mean_loss",
                "minimum_improvement_from_initialization": 100.0,
            },
        },
        research=_campaign_research(),
    )
    participants = _participants(campaign.campaign["id"])
    coordinator = CampaignCoordinator.create(
        campaign,
        tmp_path / "campaign-state",
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
    )
    for worker_id in ("release-a", "release-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="release-test-token",
            now=100,
        )
        coordinator.accept(_submission(coordinator, assignment), now=101)

    browser_root = tmp_path / "browser"
    (browser_root / "pkg").mkdir(parents=True)
    (browser_root / "fixture").mkdir()
    (browser_root / "index.html").write_text(
        '<!doctype html><meta name="orcacolony-coordinator" content="">'
        '<meta name="orcacolony-campaign" content="">'
        "<title>OrcaColony</title>"
    )
    (browser_root / "index.js").write_text("console.log('worker');\n")
    (browser_root / "pkg" / "worker.js").write_text("export const ready = true;\n")
    (browser_root / "pkg" / "worker_bg.wasm").write_bytes(b"wasm-test")
    (browser_root / "fixture" / "private-fixture.bin").write_bytes(b"not-public")
    project_license = tmp_path / "LICENSE"
    project_license.write_text("MIT test license\n")
    third_party_notice = tmp_path / "THIRD_PARTY_DATA.md"
    third_party_notice.write_text("Test dataset notice\n")
    evaluation_artifacts = tmp_path / "evaluation-artifacts"
    evaluation_artifacts.mkdir()
    initial_samples = evaluation_artifacts / "initial-samples.json"
    released_samples = evaluation_artifacts / "released-samples.json"
    initial_samples.write_text('{"score":0.1}\n', encoding="utf-8")
    released_samples.write_text('{"score":0.2}\n', encoding="utf-8")
    dashboard = coordinator.dashboard()
    assert dashboard["evaluation_gate"]["state"] == "failed"
    selected_evaluation = min(
        dashboard["evaluations"],
        key=lambda entry: (entry["mean_loss"], entry["step"]),
    )
    evidence = _campaign_evidence(
        campaign.research,  # type: ignore[arg-type]
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision=str(coordinator._lock_payload()["campaign_revision"]),
        release_checkpoint_sha256=str(selected_evaluation["checkpoint_sha256"]),
        initial_artifact_sha256=hashlib.sha256(initial_samples.read_bytes()).hexdigest(),
        released_artifact_sha256=hashlib.sha256(
            released_samples.read_bytes()
        ).hexdigest(),
    )

    first = tmp_path / "release-a"
    second = tmp_path / "release-b"
    first_manifest = build_release_bundle(
        campaign,
        coordinator,
        dataset_root=dataset_root,
        browser_root=browser_root,
        project_license=project_license,
        third_party_notice=third_party_notice,
        public_coordinator_url="https://coordinator.example:443",
        output_dir=first,
        evaluation_evidence=evidence,
        evaluation_artifact_root=evaluation_artifacts,
    )
    second_manifest = build_release_bundle(
        campaign,
        coordinator,
        dataset_root=dataset_root,
        browser_root=browser_root,
        project_license=project_license,
        third_party_notice=third_party_notice,
        public_coordinator_url="https://coordinator.example:443",
        output_dir=second,
        evaluation_evidence=evidence,
        evaluation_artifact_root=evaluation_artifacts,
    )

    assert first_manifest == second_manifest
    assert (first / "SHA256SUMS").read_bytes() == (second / "SHA256SUMS").read_bytes()
    assert (first / "checkpoint" / "model.safetensors").is_file()
    assert (first / "checkpoint" / "optimizer.safetensors").is_file()
    assert (first / "checkpoint" / "state.json").is_file()
    assert (first / "dataset" / "manifest.json").is_file()
    assert (first / "dataset" / "train.safetensors").is_file()
    assert (first / "dataset" / "validation.safetensors").is_file()
    assert (first / "LICENSE").is_file()
    assert (first / "THIRD_PARTY_DATA.md").is_file()
    assert (first / "CONTRIBUTORS.md").is_file()
    assert (first / "attribution-snapshot.json").is_file()
    assert (first / "site" / "pkg" / "worker_bg.wasm").is_file()
    assert not (first / "site" / "fixture").exists()
    assert (
        '<meta name="orcacolony-coordinator" content="https://coordinator.example">'
        in (first / "site" / "index.html").read_text(encoding="utf-8")
    )
    assert (
        f'<meta name="orcacolony-campaign" content="{campaign.campaign["id"]}">'
        in (first / "site" / "index.html").read_text(encoding="utf-8")
    )

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in first.rglob("*")
        if path.is_file()
        and path.suffix in {".html", ".js", ".json", ".md", ".txt"}
        and path.name != "tokenizer.json"
    )
    assert "private-release-contributor" not in public_text
    assert "Hidden Name" not in public_text
    assert "release-a" not in public_text
    assert "release-b" not in public_text
    assert first_manifest["checkpoint"]["step"] == 1
    assert first_manifest["numerical_profile"] == peft.EXACT_CPU_FP32_PROFILE
    assert first_manifest["checkpoint"]["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )
    assert first_manifest["dataset_revision"] == dataset.revision
    assert first_manifest["release_classification"] == "campaign_result"
    assert first_manifest["language_model_final_holdout_evaluation"] is None
    assert first_manifest["campaign_evaluation"]["comparisons"][0]["metrics"][0][
        "absolute_change"
    ] == pytest.approx(0.1)
    assert (first / "campaign-evaluation-evidence.json").is_file()
    assert (first / "campaign-evaluation-summary.json").is_file()
    assert (
        first / "campaign-evaluation-artifacts" / "initial-samples.json"
    ).is_file()
    assert (
        first / "campaign-evaluation-artifacts" / "released-samples.json"
    ).is_file()
    assert first_manifest["files"]
    public_dashboard = json.loads(
        (first / "public-dashboard.json").read_text(encoding="utf-8")
    )
    assert public_dashboard["checkpoint"]["sha256"] == first_manifest["checkpoint"][
        "model_sha256"
    ]
    assert (
        public_dashboard["checkpoint"]["download_url"]
        == "checkpoint/model.safetensors"
    )
    with pytest.raises(ValueError, match="may not be inside an input directory"):
        build_release_bundle(
            campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url="https://coordinator.example",
            output_dir=browser_root / "release",
        )


def test_public_dataset_manifest_rejects_unreviewed_fields() -> None:
    with pytest.raises(ValueError, match="unpublished fields: internal_note"):
        validate_public_dataset_manifest(
            {
                "files": {},
                "format": "orcacolony_dataset_artifacts_v1",
                "packing": {},
                "source": {
                    "dataset": "test/stories",
                    "internal_note": "do not publish",
                },
                "subsets": {},
                "tokenizer": {},
            }
        )


def test_release_bundle_publishes_separate_lora_base_adapter_and_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    corpus = (
        "A thoughtful bear carried berries home for its friends.\n"
        "<|endoftext|>\n"
    ) * 400
    dataset_root = tmp_path / "dataset"
    artifact_manifest = build_dataset_artifacts(
        train_bytes=corpus.encode(),
        validation_bytes=corpus[: len(corpus) // 2].encode(),
        output_dir=dataset_root,
        source={
            "dataset": "test/lora-release-stories",
            "revision": "lora-release-test-revision",
            "license": "cdla-sharing-1.0",
        },
        vocab_size=300,
        context_length=int(campaign_payload["model"]["context_length"]),
    )
    dataset = PackedDataset.load(dataset_root)
    campaign_payload["dataset"] = {
        "format": artifact_manifest["format"],
        "manifest_sha256": dataset.revision,
        "tokenizer_sha256": artifact_manifest["tokenizer"]["sha256"],
        "train_sha256": artifact_manifest["files"]["train.safetensors"],
        "validation_sha256": artifact_manifest["files"]["validation.safetensors"],
    }
    campaign_payload["evaluation"] = {
        "metric": "held_out_cross_entropy",
        "checkpoint_selection": "lowest_mean_loss",
        "validation_sequences": 4,
        "batch_size": 2,
    }
    campaign_bytes = (
        json.dumps(
            campaign_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_bytes(campaign_bytes)
    lora_payload = json.loads(LORA_CONFIG.read_text(encoding="utf-8"))
    lora_payload["base"]["campaign_file"] = campaign_path.name
    lora_payload["base"]["campaign_sha256"] = hashlib.sha256(
        campaign_bytes
    ).hexdigest()
    lora_path = tmp_path / "lora.json"
    lora_path.write_text(json.dumps(lora_payload), encoding="utf-8")
    lora = load_lora_manifest(campaign_path, lora_path)
    participants = _participants(lora.campaign.campaign["id"])
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        tmp_path / "campaign-state",
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
        lora=lora,
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )
    for worker_id in ("release-a", "release-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="release-test-token",
            now=100,
        )
        coordinator.accept(
            _submission(
                coordinator,
                assignment,
                runtime_backend="python-oracle-int8-f32-dequant",
            ),
            now=101,
        )

    browser_root = tmp_path / "browser"
    (browser_root / "pkg").mkdir(parents=True)
    (browser_root / "index.html").write_text(
        '<!doctype html><meta name="orcacolony-coordinator" content="">'
        '<meta name="orcacolony-campaign" content="">'
    )
    (browser_root / "index.js").write_text("console.log('worker');\n")
    (browser_root / "pkg" / "worker.js").write_text("export const ready = true;\n")
    project_license = tmp_path / "LICENSE"
    project_license.write_text("MIT test license\n")
    third_party_notice = tmp_path / "THIRD_PARTY_DATA.md"
    third_party_notice.write_text("Test dataset notice\n")

    first = tmp_path / "release-first"
    second = tmp_path / "release-second"
    first_manifest = build_release_bundle(
        lora.campaign,
        coordinator,
        dataset_root=dataset_root,
        browser_root=browser_root,
        project_license=project_license,
        third_party_notice=third_party_notice,
        public_coordinator_url=None,
        output_dir=first,
    )
    second_manifest = build_release_bundle(
        lora.campaign,
        coordinator,
        dataset_root=dataset_root,
        browser_root=browser_root,
        project_license=project_license,
        third_party_notice=third_party_notice,
        public_coordinator_url=None,
        output_dir=second,
    )

    assert first_manifest == second_manifest
    train_path = dataset_root / "train.safetensors"
    train_bytes = train_path.read_bytes()
    replacement_train = dataset_root / "replacement-train.safetensors"
    replacement_train.write_bytes(b"mutated-after-campaign-admission")
    replacement_train.replace(train_path)
    try:
        retained_dataset_manifest = build_release_bundle(
            lora.campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url=None,
            output_dir=tmp_path / "release-retained-dataset",
        )
    finally:
        replacement_train.write_bytes(train_bytes)
        replacement_train.replace(train_path)
    assert retained_dataset_manifest == first_manifest
    checkpoint = first_manifest["checkpoint"]
    assert checkpoint["training_method"] == "frozen-base-lora"
    assert first_manifest["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    assert checkpoint["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    assert checkpoint["base_model_sha256"] == lora.config.base_model_sha256
    assert checkpoint["adapter_sha256"]
    assert checkpoint["weight_checkpoint_sha256"]
    assert checkpoint["resume_state_sha256"]
    assert (first / "checkpoint" / "base-model.safetensors").is_file()
    assert (first / "checkpoint" / "adapter.safetensors").is_file()
    assert (first / "checkpoint" / "optimizer.safetensors").is_file()
    assert (first / "checkpoint" / "state.json").is_file()
    assert not (first / "checkpoint" / "model.safetensors").exists()
    dashboard = json.loads(
        (first / "public-dashboard.json").read_text(encoding="utf-8")
    )
    assert dashboard["checkpoint"]["numerical_profile"] == (
        peft.INT8_FROZEN_LINEAR_PROFILE
    )
    assert json.loads((first / "public-ledger.json").read_text(encoding="utf-8"))[
        "numerical_profile"
    ] == peft.INT8_FROZEN_LINEAR_PROFILE
    assert json.loads((first / "evaluations.json").read_text(encoding="utf-8"))[
        "numerical_profile"
    ] == peft.INT8_FROZEN_LINEAR_PROFILE
    assert dashboard["checkpoint"]["download_url"] == (
        "checkpoint/adapter.safetensors"
    )
    assert dashboard["checkpoint"]["sha256"] == checkpoint[
        "resume_state_sha256"
    ]
    assert dashboard["checkpoint"]["adapter_sha256"] == checkpoint[
        "adapter_sha256"
    ]

    selected_evaluation = coordinator._state["evaluations"][-1]
    last_evaluation = coordinator._state["last_evaluation"]
    for field in (
        "base_model_sha256",
        "adapter_sha256",
        "weight_checkpoint_sha256",
        "resume_state_sha256",
    ):
        original_selected = selected_evaluation[field]
        original_last = last_evaluation[field]
        selected_evaluation[field] = "0" * 64
        last_evaluation[field] = "0" * 64
        try:
            with pytest.raises(
                ValueError,
                match="evaluation differs from recomputation",
            ):
                build_release_bundle(
                    lora.campaign,
                    coordinator,
                    dataset_root=dataset_root,
                    browser_root=browser_root,
                    project_license=project_license,
                    third_party_notice=third_party_notice,
                    public_coordinator_url=None,
                    output_dir=tmp_path / f"wrong-evaluation-{field}",
                )
        finally:
            selected_evaluation[field] = original_selected
            last_evaluation[field] = original_last

    original_selected = dict(selected_evaluation)
    original_last = dict(last_evaluation)
    for evaluation in (selected_evaluation, last_evaluation):
        evaluation["loss_sum"] = float(evaluation["loss_sum"]) + 1.0
        evaluation["mean_loss"] = float(evaluation["mean_loss"]) + 0.25
        evaluation["perplexity"] = float(evaluation["perplexity"]) + 0.5
    try:
        with pytest.raises(
            ValueError,
            match="evaluation differs from recomputation",
        ):
            build_release_bundle(
                lora.campaign,
                coordinator,
                dataset_root=dataset_root,
                browser_root=browser_root,
                project_license=project_license,
                third_party_notice=third_party_notice,
                public_coordinator_url=None,
                output_dir=tmp_path / "mutated-evaluation-metrics",
            )
    finally:
        selected_evaluation.clear()
        selected_evaluation.update(original_selected)
        last_evaluation.clear()
        last_evaluation.update(original_last)

    baseline_evaluation = coordinator._state["evaluations"][0]
    for mutation, expected_error in (
        ({"unknown_successor": True}, "schema"),
        ({"step": 0.0}, "identity"),
        ({"validation_sequences": True}, "provenance|recomputation"),
        ({"loss_sum": 1}, "provenance|recomputation"),
    ):
        original_baseline = dict(baseline_evaluation)
        baseline_evaluation.update(mutation)
        try:
            with pytest.raises(ValueError, match=expected_error):
                build_release_bundle(
                    lora.campaign,
                    coordinator,
                    dataset_root=dataset_root,
                    browser_root=browser_root,
                    project_license=project_license,
                    third_party_notice=third_party_notice,
                    public_coordinator_url=None,
                    output_dir=tmp_path / f"malformed-evaluation-{next(iter(mutation))}",
                )
        finally:
            baseline_evaluation.clear()
            baseline_evaluation.update(original_baseline)

    checkpoint_state_path = coordinator.checkpoint_dir / "state.json"
    checkpoint_state_bytes = checkpoint_state_path.read_bytes()
    checkpoint_state = json.loads(checkpoint_state_bytes)
    checkpoint_state["format"] = "orcacolony_lora_checkpoint_v1"
    checkpoint_state.pop("numerical_profile")
    checkpoint_state_path.write_text(json.dumps(checkpoint_state), encoding="utf-8")
    try:
        retained_release_dir = tmp_path / "profileless-checkpoint-release"
        build_release_bundle(
            lora.campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url=None,
            output_dir=retained_release_dir,
        )
        retained_state = json.loads(
            (retained_release_dir / "checkpoint" / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert retained_state["format"] == "orcacolony_lora_checkpoint_v2"
        assert retained_state["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    finally:
        checkpoint_state_path.write_bytes(checkpoint_state_bytes)

    wrong_lock = coordinator._lock_payload()
    wrong_lock["campaign_revision"] = "0" * 64
    with monkeypatch.context() as patcher:
        patcher.setattr(coordinator, "_lock_payload", lambda: wrong_lock)
        with pytest.raises(ValueError, match="campaign revision differs"):
            build_release_bundle(
                lora.campaign,
                coordinator,
                dataset_root=dataset_root,
                browser_root=browser_root,
                project_license=project_license,
                third_party_notice=third_party_notice,
                public_coordinator_url=None,
                output_dir=tmp_path / "wrong-campaign-revision-release",
            )

    mixed_profile_evaluations = [
        dict(evaluation) for evaluation in coordinator._state["evaluations"]
    ]
    for evaluation in mixed_profile_evaluations:
        evaluation["numerical_profile"] = peft.EXACT_CPU_FP32_PROFILE
    coordinator._state["evaluations"] = mixed_profile_evaluations
    coordinator._state["last_evaluation"] = mixed_profile_evaluations[-1]

    with pytest.raises(ValueError, match="numerical profile"):
        build_release_bundle(
            lora.campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url=None,
            output_dir=tmp_path / "mixed-profile-release",
        )

    for evaluation in mixed_profile_evaluations:
        evaluation["numerical_profile"] = None
    with pytest.raises(ValueError, match="numerical profile"):
        build_release_bundle(
            lora.campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url=None,
            output_dir=tmp_path / "null-profile-release",
        )

    for evaluation in mixed_profile_evaluations:
        evaluation.pop("numerical_profile")
    with pytest.raises(ValueError, match="evaluation .*schema"):
        build_release_bundle(
            lora.campaign,
            coordinator,
            dataset_root=dataset_root,
            browser_root=browser_root,
            project_license=project_license,
            third_party_notice=third_party_notice,
            public_coordinator_url=None,
            output_dir=tmp_path / "missing-profile-release",
        )
