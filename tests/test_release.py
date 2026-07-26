import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony import peft
from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import LeasedGradient
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest
from orcacolony.reference import load_campaign
from orcacolony.release import build_release_bundle, validate_public_dataset_manifest


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"


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
        evaluation={"validation_sequences": 4, "batch_size": 2},
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
