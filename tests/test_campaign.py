import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from orcacolony import peft
from orcacolony.artifacts import PackedDataset, build_dataset_artifacts
from orcacolony.campaign_run import CampaignCoordinator
from orcacolony.multiworker import GlobalStepCoordinator, LeasedGradient
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"


def participants_for(campaign_id: object) -> ParticipantRegistry:
    worker_ids = ["worker-a", "worker-b"]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "campaign-test",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(b"test-token").hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {"public": False, "display_name": None},
                }
            ],
        },
        campaign_id=str(campaign_id),
    )


def submission_for(
    coordinator: CampaignCoordinator,
    assignment: dict[str, object],
    *,
    runtime_backend: str = "python-oracle-f32",
) -> LeasedGradient:
    assignment_id = str(assignment["assignment_id"])
    resources = assignment["resource_profile"]
    return LeasedGradient(
        assignment_id=assignment_id,
        lease_token=str(assignment["lease_token"]),
        checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        loss_sum=float(assignment["expected_loss_sum"]),
        loss_weight_sum=int(assignment["loss_weight_sum"]),
        safetensors=coordinator.oracle_gradient_path(assignment_id).read_bytes(),
        runtime_backend=runtime_backend,
        worker_telemetry={
            "format": "orcacolony_worker_telemetry_v1",
            "runtime_seconds": {
                "assignment_fetch": 0.01,
                "runtime_init": 0.02,
                "artifact_fetch": 0.03,
                "gradient_compute": 0.5,
            },
            "transfer_bytes": {
                "assignment": 2048,
                "model": resources["model_download_bytes"],
                "adapter": resources["adapter_download_bytes"],
                "oracle_gradient": resources["oracle_gradient_download_bytes"],
                "result": resources["expected_result_upload_bytes"],
            },
            "memory_bytes": {
                "wasm_linear": 64 * 1024 * 1024,
                "process_peak_rss": None,
                "js_heap_used": 32 * 1024 * 1024,
                "js_heap_limit": 2 * 1024 * 1024 * 1024,
                "device_capacity": 8 * 1024 * 1024 * 1024,
            },
        },
    )


def test_campaign_persists_legacy_exact_profile_migration(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    CampaignCoordinator.create(
        campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
    )
    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("numerical_profile")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.pop("numerical_profile")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    recovered = CampaignCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    assert recovered.status()["numerical_profile"] == peft.EXACT_CPU_FP32_PROFILE
    migrated_state = state_path.read_bytes()
    assert json.loads(migrated_state)["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )
    assert json.loads(lock_path.read_text(encoding="utf-8"))["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )
    CampaignCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    assert state_path.read_bytes() == migrated_state


def test_campaign_advances_two_global_steps_and_versions_every_checkpoint(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    corpus = (
        "A small fox found a red ball and shared it with a bird.\n"
        "<|endoftext|>\n"
        "A little boat crossed the pond and came safely home.\n"
        "<|endoftext|>\n"
    ) * 200
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus[: len(corpus) // 2].encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/campaign-stories",
            "revision": "test-revision",
            "license": "cdla-sharing-1.0",
            "internal_note": "must-not-be-public",
        },
        vocab_size=300,
        context_length=campaign.model.context_length,
    )
    dataset = PackedDataset.load(artifact_dir)
    campaign = replace(
        campaign,
        dataset={
            "format": manifest["format"],
            "manifest_sha256": dataset.revision,
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "train_sha256": manifest["files"]["train.safetensors"],
            "validation_sha256": manifest["files"]["validation.safetensors"],
        },
        evaluation={
            "validation_sequences": 4,
            "batch_size": 2,
            "success_gate": {
                "metric": "mean_loss",
                "minimum_improvement_from_initialization": 100.0,
            },
        },
    )
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        dataset=dataset,
    )
    assert (state_dir / "checkpoints" / "step-00000000").is_dir()

    for expected_step in (1, 2):
        for worker_id in ("worker-a", "worker-b"):
            assignment = coordinator.lease(
                worker_id,
                worker_token="test-token",
                now=expected_step * 100,
            )
            coordinator.accept(
                submission_for(coordinator, assignment),
                now=expected_step * 100 + 1,
            )
            if expected_step == 1 and worker_id == "worker-a":
                live_dashboard = coordinator.dashboard()
                assert live_dashboard["progress"]["accepted_assignments"] == 1
                assert live_dashboard["public_ledger"][0]["checkpoint_step"] == 1
        assert (state_dir / "checkpoints" / f"step-{expected_step:08d}").is_dir()

    status = coordinator.status()
    assert status["state"] == "campaign_complete"
    assert status["completed_steps"] == 2
    assert status["target_steps"] == 2
    assert status["evaluation_gate"]["state"] == "failed"
    assert status["checkpoint_metrics"]["relative_l2_error"] < 1e-6

    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert len(ledger["entries"]) == 4
    assert ledger["dataset_revision"] == dataset.revision
    assert [entry["checkpoint_step"] for entry in ledger["entries"]] == [1, 1, 2, 2]
    evaluations = json.loads(
        (state_dir / "evaluations.json").read_text(encoding="utf-8")
    )
    assert [entry["step"] for entry in evaluations["entries"]] == [0, 1, 2]
    assert all(entry["mean_loss"] > 0 for entry in evaluations["entries"])
    assert all(
        entry["dataset_revision"] == dataset.revision
        for entry in evaluations["entries"]
    )
    dashboard = coordinator.dashboard()
    assert dashboard["progress"]["completed_steps"] == 2
    assert dashboard["progress"]["accepted_assignments"] == 4
    assert dashboard["progress"]["accepted_tokens"] == sum(
        entry["loss_weight_sum"] for entry in ledger["entries"]
    )
    assert dashboard["resource_observations"]["worker_reports"] == 4
    assert dashboard["resource_observations"]["runtime_seconds"][
        "gradient_compute"
    ] == 2.0
    assert dashboard["resource_observations"]["coordinator_storage_bytes"] > 0
    assert dashboard["contributors"] == {
        "active_count": 1,
        "anonymous_count": 1,
        "acknowledgements": [],
    }
    assert len(dashboard["public_ledger"]) == 4
    serialized_dashboard = json.dumps(dashboard)
    assert "campaign-test" not in serialized_dashboard
    assert "worker-a" not in serialized_dashboard
    assert "worker-b" not in serialized_dashboard
    assert dashboard["dataset"]["source"]["dataset"] == "test/campaign-stories"
    assert "internal_note" not in dashboard["dataset"]["source"]

    first_round = state_dir / "rounds" / "round-00000000"
    first_round_ledger_path = first_round / "accepted-work.json"
    first_round_ledger = json.loads(first_round_ledger_path.read_text(encoding="utf-8"))
    first_round_ledger["entries"][0]["instrumentation"]["worker_reported"][
        "runtime_seconds"
    ]["gradient_compute"] = -121.5
    first_round_ledger_path.write_text(
        json.dumps(first_round_ledger),
        encoding="utf-8",
    )

    recovered = CampaignCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
        dataset=dataset,
    )
    assert recovered.status()["state"] == "campaign_complete"
    assert recovered.status()["completed_steps"] == 2
    assert recovered.dashboard()["resource_observations"]["runtime_seconds"][
        "gradient_compute"
    ] == 2.0
    repaired_ledger = json.loads(first_round_ledger_path.read_text(encoding="utf-8"))
    assert repaired_ledger["entries"][0]["instrumentation"]["worker_reported"][
        "runtime_seconds"
    ]["gradient_compute"] == 0.5

    first_round_state_path = first_round / "global-state.json"
    first_round_state = json.loads(first_round_state_path.read_text(encoding="utf-8"))
    first_round_state["assignments"][0]["instrumentation"]["worker_reported"][
        "runtime_seconds"
    ]["gradient_compute"] = -121.5
    first_round_state_path.write_text(json.dumps(first_round_state), encoding="utf-8")
    with pytest.raises(ValueError, match="worker runtime telemetry"):
        CampaignCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
        )


def test_int8_campaign_binds_profile_across_round_restart_and_checkpoints(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        loaded.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        lora=loaded,
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
            now=100,
        )
        coordinator.accept(
            submission_for(
                coordinator,
                assignment,
                runtime_backend="python-oracle-int8-f32-dequant",
            ),
            now=101,
        )

    recovered = CampaignCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )
    assert recovered.status()["completed_steps"] == 1
    assert recovered.status()["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    for worker_id in ("worker-a", "worker-b"):
        assignment = recovered.lease(
            worker_id,
            worker_token="test-token",
            now=200,
        )
        receipt = recovered.accept(
            submission_for(
                recovered,
                assignment,
                runtime_backend="python-oracle-int8-f32-dequant",
            ),
            now=201,
        )

    assert receipt.step == 2
    assert recovered.status()["state"] == "campaign_complete"
    state = json.loads(
        (state_dir / "campaign-state.json").read_text(encoding="utf-8")
    )
    assert state["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    for checkpoint in state["checkpoints"]:
        checkpoint_state = json.loads(
            (state_dir / checkpoint["path"] / "state.json").read_text(
                encoding="utf-8"
            )
        )
        assert checkpoint_state["numerical_profile"] == (
            peft.INT8_FROZEN_LINEAR_PROFILE
        )
    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert {entry["numerical_profile"] for entry in ledger["entries"]} == {
        peft.INT8_FROZEN_LINEAR_PROFILE
    }
    with pytest.raises(ValueError, match="numerical profile"):
        CampaignCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=peft.EXACT_CPU_FP32_PROFILE,
        )


def test_lora_campaign_advances_evaluates_and_survives_between_step_restart(
    tmp_path: Path,
) -> None:
    campaign_payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    corpus = (
        "A patient rabbit planted a seed and watered it every morning.\n"
        "<|endoftext|>\n"
        "A young whale followed the moonlit waves safely home.\n"
        "<|endoftext|>\n"
    ) * 200
    artifact_dir = tmp_path / "dataset"
    manifest = build_dataset_artifacts(
        train_bytes=corpus.encode("utf-8"),
        validation_bytes=corpus[: len(corpus) // 2].encode("utf-8"),
        output_dir=artifact_dir,
        source={
            "dataset": "test/lora-campaign-stories",
            "revision": "test-lora-revision",
            "license": "cdla-sharing-1.0",
        },
        vocab_size=300,
        context_length=int(campaign_payload["model"]["context_length"]),
    )
    dataset = PackedDataset.load(artifact_dir)
    campaign_payload["dataset"] = {
        "format": manifest["format"],
        "manifest_sha256": dataset.revision,
        "tokenizer_sha256": manifest["tokenizer"]["sha256"],
        "train_sha256": manifest["files"]["train.safetensors"],
        "validation_sha256": manifest["files"]["validation.safetensors"],
    }
    campaign_payload["evaluation"] = {
        "metric": "held_out_cross_entropy",
        "checkpoint_selection": "lowest_mean_loss",
        "validation_sequences": 4,
        "batch_size": 2,
    }
    config = tmp_path / "campaign.json"
    config_bytes = (
        json.dumps(
            campaign_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    config.write_bytes(config_bytes)
    lora_payload = json.loads(LORA_CONFIG.read_text(encoding="utf-8"))
    lora_payload["base"]["campaign_file"] = config.name
    lora_payload["base"]["campaign_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    lora_config = tmp_path / "lora.json"
    lora_config.write_text(json.dumps(lora_payload), encoding="utf-8")
    lora = load_lora_manifest(config, lora_config)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        dataset=dataset,
        lora=lora,
    )

    assert coordinator.initial_adapter_path.is_file()
    assert json.loads(
        (state_dir / "checkpoints" / "step-00000000" / "state.json").read_text(
            encoding="utf-8"
        )
    )["format"] == "orcacolony_lora_checkpoint_v1"

    for expected_step in (1, 2):
        for worker_id in ("worker-a", "worker-b"):
            assignment = coordinator.lease(
                worker_id,
                worker_token="test-token",
                now=expected_step * 100,
            )
            assert assignment["training_method"] == "frozen-base-lora"
            coordinator.accept(
                submission_for(coordinator, assignment),
                now=expected_step * 100 + 1,
            )
        if expected_step == 1:
            coordinator = CampaignCoordinator.load(
                lora.campaign,
                state_dir,
                participants=participants,
                dataset=dataset,
                lora=lora,
            )

    status = coordinator.status()
    assert status["state"] == "campaign_complete"
    assert status["completed_steps"] == 2
    evaluations = json.loads(
        (state_dir / "evaluations.json").read_text(encoding="utf-8")
    )["entries"]
    assert [entry["step"] for entry in evaluations] == [0, 1, 2]
    assert all(entry["adapter_sha256"] for entry in evaluations)
    assert all(entry["weight_checkpoint_sha256"] for entry in evaluations)
    assert all(entry["resume_state_sha256"] for entry in evaluations)
    dashboard = coordinator.dashboard()
    assert dashboard["campaign"]["training_method"] == "frozen-base-lora"
    assert dashboard["checkpoint"]["download_url"] == (
        "/api/v1/checkpoint/adapter.safetensors"
    )
    assert dashboard["checkpoint"]["adapter_sha256"] == evaluations[-1][
        "adapter_sha256"
    ]


@pytest.mark.parametrize("campaign_publishes_bundle", [False, True])
def test_campaign_restart_rejects_preexisting_next_round_bundle_mode_mismatch(
    tmp_path: Path,
    campaign_publishes_bundle: bool,
) -> None:
    lora = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / f"campaign-{campaign_publishes_bundle}"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        lora=lora,
        publish_base_layer_bundle=campaign_publishes_bundle,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator._current.lease(
            worker_id,
            worker_token="test-token",
        )
        coordinator._current.accept(submission_for(coordinator, assignment))
    assert coordinator._current.status()["state"] == "step_complete"
    checkpoint = coordinator._version_checkpoint(1)
    GlobalStepCoordinator.create(
        lora.campaign,
        state_dir / "rounds" / "round-00000001",
        worker_count=2,
        participants=participants,
        resume_from=checkpoint,
        lora=lora,
        publish_base_layer_bundle=not campaign_publishes_bundle,
    )

    with pytest.raises(
        ValueError,
        match="next campaign layer-bundle publication state differs",
    ):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            lora=lora,
        )
