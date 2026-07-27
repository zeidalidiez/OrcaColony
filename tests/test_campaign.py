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
LORA_CONFIG = (
    Path(__file__).parents[1] / "campaign" / "t0-lora-smoke-cpu.json"
)


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


def participants_v2_without_public_totals(
    campaign_id: object,
) -> ParticipantRegistry:
    worker_ids = ["worker-a", "worker-b"]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v2",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "private-v2-id",
                    "worker_ids": worker_ids,
                    "worker_token_sha256": {
                        worker_id: hashlib.sha256(b"test-token").hexdigest()
                        for worker_id in worker_ids
                    },
                    "credit": {
                        "visibility": "pseudonymous",
                        "display_name": "Public Alias",
                        "profile_url": None,
                        "team": None,
                        "roles": ["training-compute"],
                        "show_contribution_totals": False,
                        "show_hardware": False,
                    },
                    "worker_profiles": {},
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


def complete_one_step_campaign(coordinator: CampaignCoordinator) -> None:
    for index, worker_id in enumerate(("worker-a", "worker-b")):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
            now=100 + index,
        )
        coordinator.accept(
            submission_for(coordinator, assignment),
            now=110 + index,
        )


def test_dashboard_honors_v2_credit_total_preference(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_v2_without_public_totals(
        campaign.campaign["id"]
    )
    coordinator = CampaignCoordinator.create(
        campaign,
        tmp_path / "campaign",
        participants=participants,
        worker_count=2,
        target_steps=1,
    )
    complete_one_step_campaign(coordinator)

    dashboard = coordinator.dashboard()
    assert dashboard["contributors"] == {
        "active_count": 1,
        "anonymous_count": 0,
        "acknowledgements": [{"display_name": "Public Alias"}],
    }
    assert all(
        entry["credit"] == "Anonymous"
        for entry in dashboard["public_ledger"]
    )
    serialized = json.dumps(dashboard)
    assert "private-v2-id" not in serialized
    assert "worker-a" not in serialized
    assert "worker-b" not in serialized

    updated_payload = participants.as_payload()
    updated_payload["participants"][0]["credit"] = {
        "visibility": "anonymous",
        "display_name": None,
        "profile_url": None,
        "team": None,
        "roles": ["training-compute"],
        "show_contribution_totals": False,
        "show_hardware": False,
    }
    updated = ParticipantRegistry.from_payload(
        updated_payload,
        campaign_id=str(campaign.campaign["id"]),
    )
    assert updated.revision == participants.revision
    assert updated.credit_revision != participants.credit_revision
    recovered = CampaignCoordinator.load(
        campaign,
        coordinator.state_dir,
        participants=updated,
    )
    assert recovered.dashboard()["contributors"] == {
        "active_count": 1,
        "anonymous_count": 1,
        "acknowledgements": [],
    }


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


def test_campaign_legacy_profile_migration_rejects_a_profiled_lock(
    tmp_path: Path,
) -> None:
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
    lock["numerical_profile"] = peft.INT8_FROZEN_LINEAR_PROFILE
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="campaign lock mismatch"):
        CampaignCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )


def test_campaign_lock_rejects_boolean_alias_for_integer_field(
    tmp_path: Path,
) -> None:
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
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["target_steps"] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="campaign lock mismatch"):
        CampaignCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra-state", "campaign state schema is invalid"),
        ("boolean-target", "campaign progress integers are invalid"),
        ("boolean-worker-count", "campaign worker count is invalid"),
        ("boolean-lease", "campaign progress integers are invalid"),
        ("child-lease-mismatch", "campaign child lease duration differs"),
        ("escaped-current-round", "campaign current round path is invalid"),
    ),
)
def test_campaign_restart_requires_exact_current_schema_types_and_round_path(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    CampaignCoordinator.create(
        loaded.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        lora=loaded,
    )
    state_path = state_dir / "campaign-state.json"
    lock_path = state_dir / "campaign-lock.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if mutation == "extra-state":
        state["unexpected"] = None
    elif mutation == "boolean-target":
        state["target_steps"] = True
        lock["target_steps"] = True
    elif mutation == "boolean-worker-count":
        state["worker_count"] = True
        lock["worker_count"] = True
    elif mutation == "boolean-lease":
        state["lease_seconds"] = True
    elif mutation == "child-lease-mismatch":
        state["lease_seconds"] += 1
    else:
        state["current_round"] = "../outside"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CampaignCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


def evaluated_lora_fixture(
    tmp_path: Path,
) -> tuple[peft.LoadedLoRAManifest, PackedDataset]:
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
    return lora, dataset


def test_lora_campaign_advances_evaluates_and_survives_between_step_restart(
    tmp_path: Path,
) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
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


def test_campaign_restart_rejects_wrong_profile_persisted_evaluation(
    tmp_path: Path,
) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
        lora=lora,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
            now=100,
        )
        coordinator.accept(submission_for(coordinator, assignment), now=101)

    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_evaluation"]["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )
    state["last_evaluation"]["numerical_profile"] = peft.INT8_FROZEN_LINEAR_PROFILE
    state["evaluations"][-1]["numerical_profile"] = peft.INT8_FROZEN_LINEAR_PROFILE
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation numerical profile differs"):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )


def test_campaign_restart_persists_legacy_exact_evaluation_profile(
    tmp_path: Path,
) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
        lora=lora,
    )
    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["evaluations"][0].pop("numerical_profile")
    state["last_evaluation"].pop("numerical_profile")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    CampaignCoordinator.load(
        lora.campaign,
        state_dir,
        participants=participants,
        dataset=dataset,
        lora=lora,
    )

    migrated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated_state["evaluations"][0]["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )
    assert migrated_state["last_evaluation"]["numerical_profile"] == (
        peft.EXACT_CPU_FP32_PROFILE
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("null-profile", "evaluation numerical profile differs"),
        ("extra-field", "evaluation predecessor schema is invalid"),
        ("unknown-format", "evaluation format is invalid"),
    ),
)
def test_campaign_restart_rejects_nonexact_evaluation_predecessor_schema(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
        lora=lora,
    )
    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    entries = (state["evaluations"][0], state["last_evaluation"])
    for entry in entries:
        if mutation == "null-profile":
            entry["numerical_profile"] = None
        elif mutation == "extra-field":
            entry.pop("numerical_profile")
            entry["unknown_identity"] = "partial-successor"
        else:
            entry.pop("numerical_profile")
            entry["format"] = "orcacolony_evaluation_unknown"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )
    assert state_path.read_bytes() == state_before


def test_dense_evaluation_exact_predecessor_migrates_once(tmp_path: Path) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "dense-campaign"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
    )
    complete_one_step_campaign(coordinator)
    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["evaluations"]
    for evaluation in state["evaluations"]:
        evaluation.pop("numerical_profile")
    state["last_evaluation"].pop("numerical_profile")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    CampaignCoordinator.load(
        lora.campaign,
        state_dir,
        participants=participants,
        dataset=dataset,
    )
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    assert {entry["numerical_profile"] for entry in migrated["evaluations"]} == {
        peft.EXACT_CPU_FP32_PROFILE
    }


def test_campaign_restart_recomputes_last_checkpoint_metrics(tmp_path: Path) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign-metrics"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
    )
    complete_one_step_campaign(coordinator)
    state_path = state_dir / "campaign-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["last_checkpoint_metrics"]["cosine_similarity"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="last checkpoint metrics differ"):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
        )

    assert state_path.read_bytes() == state_before


def test_parent_validation_precedes_child_migration_persistence(
    tmp_path: Path,
) -> None:
    lora, dataset = evaluated_lora_fixture(tmp_path)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
        dataset=dataset,
        lora=lora,
    )
    complete_one_step_campaign(coordinator)
    campaign_state_path = state_dir / "campaign-state.json"
    campaign_state = json.loads(campaign_state_path.read_text(encoding="utf-8"))

    campaign_state["checkpoints"][0]["unexpected"] = None
    campaign_state_path.write_text(json.dumps(campaign_state), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint history entry schema is invalid"):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )
    campaign_state["checkpoints"][0].pop("unexpected")
    campaign_state["evaluations"][0]["training_method"] = "dense-adamw"
    campaign_state_path.write_text(json.dumps(campaign_state), encoding="utf-8")

    round_dir = state_dir / str(campaign_state["current_round"])
    global_state_path = round_dir / "global-state.json"
    global_state = json.loads(global_state_path.read_text(encoding="utf-8"))
    global_state.pop("accepted_result_identity_revision")
    result_identity_fields = (
        "result_dataset_cursor",
        "result_loss_history",
        "result_resume_state_sha256",
        "result_weight_checkpoint_sha256",
        "result_checkpoint_sha256",
    )
    for field in result_identity_fields:
        global_state.pop(field)
    for assignment in global_state["assignments"]:
        for field in (
            "result_file_sha256",
            "result_tensor_sha256",
            "oracle_file_sha256",
            "oracle_tensor_sha256",
            "oracle_file_size",
        ):
            assignment.pop(field)
    global_state_path.write_text(json.dumps(global_state), encoding="utf-8")
    global_lock_path = round_dir / "campaign-lock.json"
    global_lock = json.loads(global_lock_path.read_text(encoding="utf-8"))
    for field in (
        "accepted_result_identity_revision",
        "dataset_cursor",
        "worker_count",
        "assignment_ids",
        *result_identity_fields,
    ):
        global_lock.pop(field)
    global_lock_path.write_text(json.dumps(global_lock), encoding="utf-8")
    global_ledger_path = round_dir / "accepted-work.json"
    campaign_before = campaign_state_path.read_bytes()
    global_state_before = global_state_path.read_bytes()
    global_lock_before = global_lock_path.read_bytes()
    global_ledger_before = global_ledger_path.read_bytes()

    with pytest.raises(ValueError, match="evaluation training method differs"):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            dataset=dataset,
            lora=lora,
        )
    assert campaign_state_path.read_bytes() == campaign_before
    assert global_state_path.read_bytes() == global_state_before
    assert global_lock_path.read_bytes() == global_lock_before
    assert global_ledger_path.read_bytes() == global_ledger_before


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


def test_campaign_versions_and_ledgers_from_retained_child_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = CampaignCoordinator.create(
        campaign,
        tmp_path / "campaign",
        participants=participants,
        worker_count=2,
        target_steps=1,
    )
    original_version = CampaignCoordinator._version_checkpoint
    admitted_model: bytes | None = None

    def mutate_sources_then_version(
        campaign_coordinator: CampaignCoordinator,
        step: int,
    ) -> Path:
        nonlocal admitted_model
        admitted_model = campaign_coordinator._current.checkpoint_artifact_bytes(
            "model.safetensors"
        )
        source_model = campaign_coordinator._current.checkpoint_dir / "model.safetensors"
        mutated = bytearray(source_model.read_bytes())
        mutated[-1] ^= 1
        source_model.write_bytes(mutated)
        (campaign_coordinator._current.state_dir / "accepted-work.json").write_text(
            '{"forged":true}',
            encoding="utf-8",
        )
        return original_version(campaign_coordinator, step)

    monkeypatch.setattr(
        CampaignCoordinator,
        "_version_checkpoint",
        mutate_sources_then_version,
    )
    complete_one_step_campaign(coordinator)

    assert admitted_model is not None
    assert (coordinator.checkpoints_dir / "step-00000001" / "model.safetensors").read_bytes() == admitted_model
    ledger = json.loads(
        (coordinator.state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert len(ledger["entries"]) == 2
    assert "forged" not in ledger
    assert all(entry["assignment_id"] for entry in ledger["entries"])


def test_preexisting_next_round_worker_mismatch_rejects_without_rewrite(
    tmp_path: Path,
) -> None:
    lora = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(lora.campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        lora.campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=2,
        lora=lora,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator._current.lease(
            worker_id,
            worker_token="test-token",
        )
        coordinator._current.accept(submission_for(coordinator, assignment))
    checkpoint = coordinator._version_checkpoint(1)
    next_dir = state_dir / "rounds" / "round-00000001"
    GlobalStepCoordinator.create(
        lora.campaign,
        next_dir,
        worker_count=4,
        participants=participants,
        resume_from=checkpoint,
        lora=lora,
    )
    before = {
        name: (next_dir / name).read_bytes()
        for name in ("global-state.json", "campaign-lock.json", "accepted-work.json")
    }
    current_dir = coordinator._current.state_dir
    current_state_path = current_dir / "global-state.json"
    current_state = json.loads(current_state_path.read_text(encoding="utf-8"))
    current_state.pop("numerical_profile")
    for assignment in current_state["assignments"]:
        assignment.pop("numerical_profile")
    current_state_path.write_text(json.dumps(current_state), encoding="utf-8")
    current_lock_path = current_dir / "campaign-lock.json"
    current_lock = json.loads(current_lock_path.read_text(encoding="utf-8"))
    current_lock.pop("numerical_profile")
    current_lock_path.write_text(json.dumps(current_lock), encoding="utf-8")
    current_ledger_path = current_dir / "accepted-work.json"
    current_before = {
        path: path.read_bytes()
        for path in (current_state_path, current_lock_path, current_ledger_path)
    }

    with pytest.raises(ValueError, match="worker count differs from parent campaign"):
        CampaignCoordinator.load(
            lora.campaign,
            state_dir,
            participants=participants,
            lora=lora,
        )
    assert all((next_dir / name).read_bytes() == payload for name, payload in before.items())
    assert all(path.read_bytes() == payload for path, payload in current_before.items())


def test_parent_validation_and_finalization_precede_child_migration_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "campaign"
    coordinator = CampaignCoordinator.create(
        campaign,
        state_dir,
        participants=participants,
        worker_count=2,
        target_steps=1,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator._current.lease(
            worker_id,
            worker_token="test-token",
        )
        coordinator._current.accept(
            submission_for(coordinator, assignment),
            finalize=False,
        )
    child_dir = coordinator._current.state_dir
    child_state_path = child_dir / "global-state.json"
    child_state = json.loads(child_state_path.read_text(encoding="utf-8"))
    child_state.pop("numerical_profile")
    for assignment in child_state["assignments"]:
        assignment.pop("numerical_profile")
    child_state_path.write_text(json.dumps(child_state), encoding="utf-8")
    child_lock_path = child_dir / "campaign-lock.json"
    child_lock = json.loads(child_lock_path.read_text(encoding="utf-8"))
    child_lock.pop("numerical_profile")
    child_lock_path.write_text(json.dumps(child_lock), encoding="utf-8")
    child_ledger_path = child_dir / "accepted-work.json"
    before = {
        child_state_path: child_state_path.read_bytes(),
        child_lock_path: child_lock_path.read_bytes(),
        child_ledger_path: child_ledger_path.read_bytes(),
    }

    def fail_finalization(self: GlobalStepCoordinator) -> None:
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(
        GlobalStepCoordinator,
        "_finalize_locked",
        fail_finalization,
    )
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        CampaignCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )
    assert all(path.read_bytes() == payload for path, payload in before.items())
