import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import threading
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save as save_safetensors

from orcacolony import multiworker, peft
from orcacolony.multiworker import (
    GlobalStepCoordinator,
    LeasedGradient,
    create_http_server,
    normalize_http_origin,
)
from orcacolony.participants import ParticipantRegistry
from orcacolony.peft import load_lora_manifest
from orcacolony.reference import load_campaign


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
LORA_CONFIG = Path(__file__).parents[1] / "campaign" / "t0-lora-smoke.json"


def participants_for(campaign_id: object) -> ParticipantRegistry:
    worker_ids = [
        "browser-a",
        "browser-b",
        "worker-a",
        "worker-b",
        "worker-c",
    ]
    return ParticipantRegistry.from_payload(
        {
            "format": "orcacolony_participants_v1",
            "campaign_id": campaign_id,
            "participants": [
                {
                    "contributor_id": "test-contributor",
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
    coordinator: GlobalStepCoordinator,
    assignment: dict[str, object],
    *,
    runtime_backend: str = "python-oracle-f32",
) -> LeasedGradient:
    return LeasedGradient(
        assignment_id=str(assignment["assignment_id"]),
        lease_token=str(assignment["lease_token"]),
        checkpoint_sha256=str(assignment["checkpoint_sha256"]),
        loss_sum=float(assignment["expected_loss_sum"]),
        loss_weight_sum=int(assignment["loss_weight_sum"]),
        safetensors=coordinator.oracle_gradient_path(
            str(assignment["assignment_id"])
        ).read_bytes(),
        runtime_backend=runtime_backend,
    )


def worker_telemetry(
    coordinator: GlobalStepCoordinator,
    assignment: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "orcacolony_worker_telemetry_v1",
        "runtime_seconds": {
            "assignment_fetch": 0.01,
            "runtime_init": 0.02,
            "artifact_fetch": 0.03,
            "gradient_compute": 0.5,
        },
        "transfer_bytes": {
            "assignment": 2048,
            "model": coordinator.initial_model_path.stat().st_size,
            "adapter": (
                coordinator.initial_adapter_path.stat().st_size
                if coordinator.lora is not None
                else 0
            ),
            "oracle_gradient": coordinator.oracle_gradient_path(
                str(assignment["assignment_id"])
            ).stat().st_size,
            "result": coordinator.oracle_gradient_path(
                str(assignment["assignment_id"])
            ).stat().st_size,
        },
        "memory_bytes": {
            "wasm_linear": 64 * 1024 * 1024,
            "process_peak_rss": None,
            "js_heap_used": 32 * 1024 * 1024,
            "js_heap_limit": 2 * 1024 * 1024 * 1024,
            "device_capacity": 8 * 1024 * 1024 * 1024,
        },
    }


def test_worker_resource_observations_are_validated_persisted_and_recovered(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    resources = assignment["resource_profile"]
    assert resources["model_download_bytes"] == coordinator.initial_model_path.stat().st_size
    assert resources["adapter_download_bytes"] == 0
    assert resources["expected_result_upload_bytes"] == coordinator.oracle_gradient_path(
        str(assignment["assignment_id"])
    ).stat().st_size
    telemetry = worker_telemetry(coordinator, assignment)
    receipt = coordinator.accept(
        replace(submission_for(coordinator, assignment), worker_telemetry=telemetry),
        now=101,
        finalize=False,
    )

    assert receipt.instrumentation["worker_reported"] == telemetry
    measured = receipt.instrumentation["coordinator_measured"]
    assert measured["result_upload_bytes"] == len(
        submission_for(coordinator, assignment).safetensors
    )
    assert measured["result_storage_bytes"] == measured["result_upload_bytes"]
    ledger = json.loads(
        (state_dir / "accepted-work.json").read_text(encoding="utf-8")
    )
    assert ledger["entries"][0]["instrumentation"] == receipt.instrumentation

    recovered = GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    observations = recovered.status()["resource_observations"]
    assert observations["worker_reports"] == 1
    assert observations["runtime_seconds"]["gradient_compute"] == 0.5
    assert observations["transfer_bytes"]["result_upload"] == measured[
        "result_upload_bytes"
    ]
    assert observations["memory_bytes"]["peak_wasm_linear"] == 64 * 1024 * 1024
    assert "largest_device_capacity" not in observations["memory_bytes"]
    assert "largest_js_heap_limit" not in observations["memory_bytes"]
    assert observations["coordinator_storage_bytes"] > measured["result_storage_bytes"]

    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["assignments"][0]["instrumentation"]["worker_reported"][
        "runtime_seconds"
    ]["gradient_compute"] = -1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="worker runtime telemetry"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )


def test_burn_worker_telemetry_is_required_and_bound_to_assignment_bytes(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        numerical_profile=peft.BURN_NDARRAY_F32_PROFILE,
    )
    assignment = coordinator.lease("worker-a", worker_token="test-token", now=100)
    burn_submission = replace(
        submission_for(coordinator, assignment),
        runtime_backend="burn-ndarray-f32",
    )
    with pytest.raises(ValueError, match="telemetry is required"):
        coordinator.accept(burn_submission, now=101)

    unassigned_bundle_submission = replace(
        submission_for(coordinator, assignment),
        runtime_backend="python-native-cpu-layer-bundle-f32",
    )
    with pytest.raises(ValueError, match="numerical profile"):
        coordinator.accept(unassigned_bundle_submission, now=101)

    telemetry = worker_telemetry(coordinator, assignment)
    telemetry["transfer_bytes"]["result"] += 1
    with pytest.raises(ValueError, match="result does not match assignment"):
        coordinator.accept(
            replace(burn_submission, worker_telemetry=telemetry),
            now=101,
        )


@pytest.mark.parametrize("mutation", ("one-ulp", "signed-zero"))
def test_exact_fp32_profile_rejects_bit_level_gradient_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease("worker-a", worker_token="test-token", now=100)
    assert assignment["runtime_backends"] == [
        "python-native-cpu-f32",
        "python-oracle-f32",
    ]
    submission = submission_for(coordinator, assignment)
    gradients = load_safetensors(submission.safetensors)
    if mutation == "one-ulp":
        name = sorted(gradients)[0]
        changed = gradients[name].clone()
        flat = changed.reshape(-1)
        flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf")))
        gradients[name] = changed
    else:
        for name in sorted(gradients):
            changed = gradients[name].clone()
            flat = changed.reshape(-1)
            zero_indices = (flat == 0).nonzero(as_tuple=False)
            if zero_indices.numel() == 0:
                continue
            flat[int(zero_indices[0])] = torch.tensor(-0.0, dtype=flat.dtype)
            assert torch.equal(changed, gradients[name])
            gradients[name] = changed
            break
        else:
            raise AssertionError("oracle gradient did not contain a signed-zero probe site")

    with pytest.raises(ValueError, match="not bit-exact"):
        coordinator.accept(
            replace(submission, safetensors=save_safetensors(gradients)),
            now=101,
        )


@pytest.mark.parametrize(
    ("numerical_profile", "runtime_backend"),
    (
        (peft.EXACT_CPU_FP32_PROFILE, "python-oracle-f32"),
        (
            peft.INT8_FROZEN_LINEAR_PROFILE,
            "python-oracle-int8-f32-dequant",
        ),
    ),
)
def test_restart_rejects_an_accepted_result_mutated_after_admission(
    tmp_path: Path,
    numerical_profile: str,
    runtime_backend: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
        numerical_profile=numerical_profile,
    )
    accepted_assignments = []
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
                runtime_backend=runtime_backend,
            ),
            now=101,
            finalize=False,
        )
        accepted_assignments.append(assignment)

    result_path = state_dir / "results" / (
        f"{accepted_assignments[0]['assignment_id']}.safetensors"
    )
    gradients = load_safetensors(result_path.read_bytes())
    for name in sorted(gradients):
        changed = gradients[name].clone()
        flat = changed.reshape(-1)
        zero_indices = (flat == 0).nonzero(as_tuple=False)
        if zero_indices.numel() == 0:
            continue
        flat[int(zero_indices[0])] = torch.tensor(-0.0, dtype=flat.dtype)
        gradients[name] = changed
        break
    else:
        raise AssertionError("accepted result did not contain a signed-zero probe site")
    result_path.write_bytes(save_safetensors(gradients))

    with pytest.raises(ValueError, match="accepted result.*changed"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=numerical_profile,
        )


@pytest.mark.parametrize(
    "identity_field",
    (
        "result_file_sha256",
        "result_tensor_sha256",
        "oracle_file_sha256",
        "oracle_tensor_sha256",
        "oracle_file_size",
    ),
)
def test_restart_rejects_removed_result_identity_from_current_state(
    tmp_path: Path,
    identity_field: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted = next(
        value for value in state["assignments"] if value["state"] == "accepted"
    )
    accepted.pop(identity_field)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="current accepted-result assignment schema is incomplete",
    ):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("accepted_result_identity_revision")
    for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
        state.pop(field)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_before_partial_migration = state_path.read_bytes()
    with pytest.raises(
        ValueError,
        match="legacy accepted-result assignment schema is invalid",
    ):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert state_path.read_bytes() == state_before_partial_migration


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra-state", "current global-step state schema is invalid"),
        ("extra-assignment", "current global-step assignment schema is invalid"),
        ("missing-assignment", "current global-step assignment schema is invalid"),
        ("boolean-lease", "lease duration is invalid"),
        ("float-protocol", "result protocol revision is invalid"),
        ("boolean-result-cursor", "unfinished global-step result authority"),
        ("integer-result-history", "unfinished global-step result authority"),
    ),
)
def test_restart_requires_exact_current_global_step_schemas(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if mutation == "extra-state":
        state["unexpected"] = None
    elif mutation == "extra-assignment":
        state["assignments"][0]["unexpected"] = None
    elif mutation == "missing-assignment":
        state["assignments"][0].pop("parameter_count")
    elif mutation == "boolean-lease":
        state["lease_seconds"] = True
        lock["lease_seconds"] = True
    elif mutation == "float-protocol":
        state["result_protocol_revision"] = 3.0
        lock["result_protocol_revision"] = 3.0
    elif mutation == "boolean-result-cursor":
        state["result_dataset_cursor"] = True
        lock["result_dataset_cursor"] = True
    else:
        state["result_loss_history"] = [1]
        lock["result_loss_history"] = [1]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


def test_legacy_result_identity_migration_rejects_partial_assignment_schema(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("accepted_result_identity_revision")
    for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
        state.pop(field)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.pop("accepted_result_identity_revision")
    lock.pop("dataset_cursor")
    lock.pop("worker_count")
    lock.pop("assignment_ids")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="legacy accepted-result assignment schema"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


@pytest.mark.parametrize(
    "mutation",
    ("extra-state-field", "extra-assignment-field", "missing-assignment-field"),
)
def test_legacy_result_identity_migration_requires_exact_predecessor_shapes(
    tmp_path: Path,
    mutation: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / mutation
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("accepted_result_identity_revision")
    for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
        state.pop(field)
    for persisted_assignment in state["assignments"]:
        for field in multiworker._ASSIGNMENT_IDENTITY_FIELDS:
            persisted_assignment.pop(field)
    if mutation == "extra-state-field":
        state["unknown_successor"] = "present"
        expected_error = "legacy accepted-result state schema"
    elif mutation == "extra-assignment-field":
        state["assignments"][0]["unknown_successor"] = "present"
        expected_error = "legacy accepted-result assignment schema"
    else:
        state["assignments"][0].pop("parameter_count")
        expected_error = "legacy accepted-result assignment schema"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.pop("accepted_result_identity_revision")
    lock.pop("dataset_cursor")
    lock.pop("worker_count")
    lock.pop("assignment_ids")
    for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
        lock.pop(field)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match=expected_error):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


def test_restart_rejects_state_and_lock_cursor_different_from_checkpoint(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["dataset_cursor"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["dataset_cursor"] = 1
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="base checkpoint progress mismatch"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


def test_restart_rejects_truncated_assignment_set_before_finalization(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    lock = json.loads(
        (state_dir / "campaign-lock.json").read_text(encoding="utf-8")
    )
    assert lock["worker_count"] == 2
    assert lock["assignment_ids"] == [
        persisted["assignment_id"] for persisted in state["assignments"]
    ]
    state["assignments"] = [
        persisted
        for persisted in state["assignments"]
        if persisted["state"] == "accepted"
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    state_before = state_path.read_bytes()

    with pytest.raises(ValueError, match="assignment coverage is incomplete"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert state_path.read_bytes() == state_before
    assert not (state_dir / "checkpoint").exists()


def test_oracle_recomputation_uses_the_authenticated_adapter_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    adapter_path = state_dir / "adapter.safetensors"
    original_bytes = adapter_path.read_bytes()
    changed = load_safetensors(original_bytes)
    name = sorted(changed)[0]
    changed[name] = changed[name].clone()
    changed[name].reshape(-1)[0] += 1.0
    changed_bytes = save_safetensors(changed)
    real_recompute = GlobalStepCoordinator._recomputed_assignment_oracle
    recomputations = 0

    def mutate_path_during_recomputation(self, *args, **kwargs):
        nonlocal recomputations
        recomputations += 1
        adapter_path.write_bytes(changed_bytes)
        try:
            return real_recompute(self, *args, **kwargs)
        finally:
            adapter_path.write_bytes(original_bytes)

    monkeypatch.setattr(
        GlobalStepCoordinator,
        "_recomputed_assignment_oracle",
        mutate_path_during_recomputation,
    )
    recovered = GlobalStepCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
    )

    assert recomputations == 2
    assert sum(
        assignment["state"] == "accepted" for assignment in recovered.assignments
    ) == 1


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("state-step", "base checkpoint progress mismatch"),
        ("assignment-step", "assignment identity differs"),
    ),
)
def test_restart_rejects_boolean_aliases_for_progress_identities(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / mutation
    GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mutation == "state-step":
        state["step"] = False
    else:
        state["assignments"][0]["global_step"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


@pytest.mark.parametrize("stored_value", (True, 1.0, "1", 2))
@pytest.mark.parametrize("location", ("state", "lock"))
def test_restart_rejects_malformed_result_identity_revision_type(
    tmp_path: Path,
    stored_value: object,
    location: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / f"coordinator-{location}-{stored_value!s}"
    GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    state_path = state_dir / "global-state.json"
    lock_path = state_dir / "campaign-lock.json"
    target_path = state_path if location == "state" else lock_path
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    payload["accepted_result_identity_revision"] = stored_value
    target_path.write_text(json.dumps(payload), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(
        ValueError,
        match=(
            "unsupported accepted-result identity revision"
            if location == "state"
            else "campaign lock mismatch"
        ),
    ):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


def test_restart_rejects_result_and_oracle_path_escape(tmp_path: Path) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted = next(
        value for value in state["assignments"] if value["state"] == "accepted"
    )
    escaped_name = "escaped.safetensors"
    original_result = state_dir / "results" / accepted["result_file"]
    (state_dir / escaped_name).write_bytes(original_result.read_bytes())
    accepted["assignment_id"] = "../escaped"
    accepted["result_file"] = f"../{escaped_name}"
    accepted["oracle_file"] = f"../{escaped_name}"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="assignment set identity is invalid"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("oracle_file", "oracle gradient file identity is invalid"),
        ("result_file", "accepted result file identity is invalid"),
    ),
)
def test_restart_rejects_mutable_assignment_artifact_name(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / field
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted = next(
        value for value in state["assignments"] if value["state"] == "accepted"
    )
    accepted[field] = "../escaped.safetensors"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


def test_restart_rejects_result_link_swap_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    result_path = state_dir / "results" / f"{assignment['assignment_id']}.safetensors"
    held_path = tmp_path / "held-result.safetensors"
    symlink_probe = tmp_path / "symlink-probe"
    try:
        symlink_probe.symlink_to(result_path)
    except OSError:
        pytest.skip("symlinks are unavailable for the link-swap regression")
    symlink_probe.unlink()

    real_open = os.open
    swapped = False

    def swapping_open(path: str | bytes | os.PathLike[str], flags: int, *args: object) -> int:
        nonlocal swapped
        if not swapped and Path(path) == result_path:
            swapped = True
            os.replace(result_path, held_path)
            result_path.symlink_to(held_path)
        return real_open(path, flags, *args)

    monkeypatch.setattr(multiworker.os, "open", swapping_open)
    with pytest.raises(ValueError, match="accepted result artifact.*(changed|unavailable)"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert swapped


def test_restart_rejects_result_root_rebind_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    results_dir = state_dir / "results"
    result_path = results_dir / f"{assignment['assignment_id']}.safetensors"
    held_dir = tmp_path / "held-results"
    symlink_probe = tmp_path / "directory-symlink-probe"
    try:
        symlink_probe.symlink_to(results_dir, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable for the root-swap regression")
    symlink_probe.unlink()

    real_open = os.open
    attempted = False

    def swapping_open(path: str | bytes | os.PathLike[str], flags: int, *args: object) -> int:
        nonlocal attempted
        if not attempted and Path(path) == result_path:
            attempted = True
            os.replace(results_dir, held_dir)
            results_dir.symlink_to(held_dir, target_is_directory=True)
        return real_open(path, flags, *args)

    monkeypatch.setattr(multiworker.os, "open", swapping_open)
    with pytest.raises(ValueError, match="accepted result artifact.*(changed|unavailable)"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )
    assert attempted


def test_restart_rejects_oracle_and_result_mutated_with_rebound_state(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    assignment = coordinator.lease(
        "worker-a",
        worker_token="test-token",
        now=100,
    )
    coordinator.accept(
        submission_for(coordinator, assignment),
        now=101,
        finalize=False,
    )
    oracle_path = coordinator.oracle_gradient_path(str(assignment["assignment_id"]))
    gradients = load_safetensors(oracle_path.read_bytes())
    name = sorted(gradients)[0]
    changed = gradients[name].clone()
    changed.reshape(-1)[0] = torch.nextafter(
        changed.reshape(-1)[0],
        torch.tensor(float("inf")),
    )
    gradients[name] = changed
    changed_bytes = save_safetensors(gradients)
    oracle_path.write_bytes(changed_bytes)
    result_path = state_dir / "results" / f"{assignment['assignment_id']}.safetensors"
    result_path.write_bytes(changed_bytes)

    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    accepted = next(
        value for value in state["assignments"] if value["state"] == "accepted"
    )
    accepted["oracle_file_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    accepted["oracle_tensor_sha256"] = multiworker.tensor_sha256(gradients)
    accepted["oracle_file_size"] = len(changed_bytes)
    accepted["result_file_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    accepted["result_tensor_sha256"] = multiworker.tensor_sha256(gradients)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="oracle gradient differs from independent"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
        )


def test_legacy_result_identity_migration_is_exact_fp32_only(tmp_path: Path) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    for profile, backend, should_migrate in (
        (peft.EXACT_CPU_FP32_PROFILE, "python-oracle-f32", True),
        (peft.BURN_NDARRAY_F32_PROFILE, "burn-ndarray-f32", False),
        (peft.BURN_WEBGPU_F32_PROFILE, "burn-webgpu-f32", False),
        (
            peft.INT8_FROZEN_LINEAR_PROFILE,
            "python-oracle-int8-f32-dequant",
            False,
        ),
    ):
        state_dir = tmp_path / profile
        coordinator = GlobalStepCoordinator.create(
            loaded.campaign,
            state_dir,
            worker_count=2,
            participants=participants,
            lora=loaded,
            numerical_profile=profile,
        )
        assignment = coordinator.lease(
            "worker-a",
            worker_token="test-token",
            now=100,
        )
        submission = submission_for(
            coordinator,
            assignment,
            runtime_backend=backend,
        )
        if backend not in multiworker._ORACLE_RUNTIME_BACKENDS:
            submission = replace(
                submission,
                worker_telemetry=worker_telemetry(coordinator, assignment),
            )
        coordinator.accept(
            submission,
            now=101,
            finalize=False,
        )
        state_path = state_dir / "global-state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.pop("accepted_result_identity_revision")
        for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
            state.pop(field)
        for persisted_assignment in state["assignments"]:
            for field in (
                "result_file_sha256",
                "result_tensor_sha256",
                "oracle_file_sha256",
                "oracle_tensor_sha256",
                "oracle_file_size",
            ):
                persisted_assignment.pop(field)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        lock_path = state_dir / "campaign-lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock.pop("accepted_result_identity_revision")
        lock.pop("dataset_cursor")
        lock.pop("worker_count")
        lock.pop("assignment_ids")
        for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
            lock.pop(field)
        lock_path.write_text(json.dumps(lock), encoding="utf-8")

        if not should_migrate:
            with pytest.raises(ValueError, match="accepted result identity is missing"):
                GlobalStepCoordinator.load(
                    loaded.campaign,
                    state_dir,
                    participants=participants,
                    lora=loaded,
                    numerical_profile=profile,
                )
            continue

        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=profile,
        )
        migrated_state = json.loads(state_path.read_text(encoding="utf-8"))
        migrated_assignment = next(
            value
            for value in migrated_state["assignments"]
            if value["state"] == "accepted"
        )
        assert migrated_state["accepted_result_identity_revision"] == 1
        assert len(migrated_assignment["result_file_sha256"]) == 64
        assert len(migrated_assignment["result_tensor_sha256"]) == 64
        assert len(migrated_assignment["oracle_file_sha256"]) == 64
        assert len(migrated_assignment["oracle_tensor_sha256"]) == 64
        assert migrated_assignment["oracle_file_size"] > 0
        assert json.loads(lock_path.read_text(encoding="utf-8"))[
            "accepted_result_identity_revision"
        ] == 1
        migrated_state_bytes = state_path.read_bytes()
        migrated_lock_bytes = lock_path.read_bytes()
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=profile,
        )
        assert state_path.read_bytes() == migrated_state_bytes
        assert lock_path.read_bytes() == migrated_lock_bytes


@pytest.mark.parametrize(
    "profile",
    (
        peft.BURN_NDARRAY_F32_PROFILE,
        peft.BURN_WEBGPU_F32_PROFILE,
        peft.INT8_FROZEN_LINEAR_PROFILE,
    ),
)
def test_approximate_result_identity_predecessor_rejects_without_acceptances(
    tmp_path: Path,
    profile: str,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / profile
    GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
        numerical_profile=profile,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("accepted_result_identity_revision")
    for field in multiworker._RESULT_STATE_IDENTITY_FIELDS:
        state.pop(field)
    for assignment in state["assignments"]:
        for field in (
            "result_file_sha256",
            "result_tensor_sha256",
            "oracle_file_sha256",
            "oracle_tensor_sha256",
            "oracle_file_size",
        ):
            assignment.pop(field)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for field in (
        "accepted_result_identity_revision",
        "dataset_cursor",
        "worker_count",
        "assignment_ids",
        *multiworker._RESULT_STATE_IDENTITY_FIELDS,
    ):
        lock.pop(field)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="exact-FP32 only"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=profile,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


def test_layer_bundle_partial_transfer_telemetry_binds_artifact_membership(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
        publish_base_layer_bundle=True,
    )
    assignment = coordinator.lease("worker-a", worker_token="test-token", now=100)
    bundle = assignment["base_layer_bundle"]
    first_artifact, second_artifact = bundle["artifacts"][2:4]
    telemetry = worker_telemetry(coordinator, assignment)
    telemetry["transfer_bytes"]["model"] = first_artifact["bytes"]
    telemetry["transfer_bytes"]["oracle_gradient"] = 0
    telemetry["model_artifacts"] = ["not-assigned.safetensors"]
    submission = replace(
        submission_for(coordinator, assignment),
        runtime_backend="python-native-cpu-layer-bundle-f32",
        worker_telemetry=telemetry,
    )
    with pytest.raises(ValueError, match="model artifact telemetry is invalid"):
        coordinator.accept(submission, now=101)

    telemetry["model_artifacts"] = [first_artifact["file"]]
    telemetry["transfer_bytes"]["model"] = first_artifact["bytes"] + 1
    with pytest.raises(ValueError, match="model artifact telemetry bytes differ"):
        coordinator.accept(
            replace(submission, worker_telemetry=telemetry),
            now=101,
        )

    telemetry["model_artifacts"] = [
        second_artifact["file"],
        first_artifact["file"],
    ]
    telemetry["transfer_bytes"]["model"] = (
        first_artifact["bytes"] + second_artifact["bytes"]
    )
    with pytest.raises(ValueError, match="model artifact telemetry order differs"):
        coordinator.accept(
            replace(submission, worker_telemetry=telemetry),
            now=101,
        )

    telemetry["model_artifacts"] = [first_artifact["file"]]
    telemetry["transfer_bytes"]["model"] = first_artifact["bytes"]
    receipt = coordinator.accept(
        replace(submission, worker_telemetry=telemetry),
        now=101,
        finalize=False,
    )
    assert receipt.instrumentation["worker_reported"]["model_artifacts"] == [
        first_artifact["file"]
    ]


def test_two_non_overlapping_workers_match_one_reference_global_step(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lease_seconds=60,
    )
    first = coordinator.lease("worker-a", worker_token="test-token", now=100)
    second = coordinator.lease("worker-b", worker_token="test-token", now=100)

    assert first["data_range"] == [0, 2]
    assert second["data_range"] == [2, 4]
    assert set(range(*first["data_range"])).isdisjoint(range(*second["data_range"]))

    first_submission = submission_for(coordinator, first)
    first_receipt = coordinator.accept(first_submission, now=101)
    final_receipt = coordinator.accept(submission_for(coordinator, second), now=101)

    assert first_receipt.step_complete is False
    assert final_receipt.step_complete is True
    assert final_receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    assert final_receipt.checkpoint_metrics["cosine_similarity"] > 1 - 1e-10
    assert coordinator.status()["state"] == "step_complete"
    assert coordinator.status()["loss_sum"] == pytest.approx(
        first_submission.loss_sum + float(second["expected_loss_sum"])
    )


def test_expired_attempt_is_rejected_and_accepted_work_replays_after_restart(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lease_seconds=10,
    )
    expired = coordinator.lease("worker-a", worker_token="test-token", now=100)
    replacement = coordinator.lease(
        "worker-b", worker_token="test-token", now=111
    )
    other = coordinator.lease("worker-c", worker_token="test-token", now=111)

    assert replacement["assignment_id"] == expired["assignment_id"]
    assert replacement["attempt"] == 2
    with pytest.raises(ValueError, match="stale lease"):
        coordinator.accept(submission_for(coordinator, expired), now=112)

    coordinator.accept(
        submission_for(coordinator, replacement),
        now=112,
        finalize=False,
    )
    coordinator.accept(
        submission_for(coordinator, other),
        now=112,
        finalize=False,
    )
    assert coordinator.status()["state"] == "ready_to_finalize"

    recovered = GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )

    assert recovered.status()["state"] == "step_complete"
    assert recovered.status()["checkpoint_metrics"]["relative_l2_error"] < 1e-6
    with pytest.raises(ValueError, match="already accepted"):
        recovered.accept(submission_for(coordinator, replacement), now=113)

    (recovered.checkpoint_dir / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        GlobalStepCoordinator.load(campaign, state_dir, participants=participants)


def test_http_leases_two_workers_and_closes_the_global_step(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
    )
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    (browser_root / "index.html").write_text("OrcaColony", encoding="utf-8")
    public_origin = "https://workers.example"
    server = create_http_server(
        coordinator,
        browser_root,
        port=0,
        public_origin=public_origin,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    receipts = []
    try:
        preflight = Request(
            f"{base_url}/api/v1/assignment",
            method="OPTIONS",
            headers={
                "Origin": public_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Orca-Worker-Token",
            },
        )
        with urlopen(preflight) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == public_origin
            assert "X-Orca-Worker-Token" in response.headers[
                "Access-Control-Allow-Headers"
            ]
            assert "X-Orca-Worker-Telemetry" in response.headers[
                "Access-Control-Allow-Headers"
            ]
        for worker_id in ("browser-a", "browser-b"):
            query = urlencode({"worker_id": worker_id})
            assignment_request = Request(
                f"{base_url}/api/v1/assignment?{query}",
                headers={"X-Orca-Worker-Token": "test-token"},
            )
            with urlopen(assignment_request) as response:
                assignment = json.load(response)
            request = Request(
                f"{base_url}{assignment['result_url']}",
                data=coordinator.oracle_gradient_path(
                    assignment["assignment_id"]
                ).read_bytes(),
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Orca-Lease-Token": assignment["lease_token"],
                    "X-Orca-Checkpoint-Sha256": assignment["checkpoint_sha256"],
                    "X-Orca-Loss-Sum": str(assignment["expected_loss_sum"]),
                    "X-Orca-Loss-Weight-Sum": str(assignment["loss_weight_sum"]),
                    "X-Orca-Runtime-Backend": "python-oracle-f32",
                    "X-Orca-Worker-Telemetry": json.dumps(
                        worker_telemetry(coordinator, assignment),
                        separators=(",", ":"),
                    ),
                },
            )
            with urlopen(request) as response:
                receipts.append(json.load(response))
        with urlopen(f"{base_url}/api/v1/status") as response:
            status = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipts[0]["step_complete"] is False
    assert receipts[1]["step_complete"] is True
    assert receipts[0]["instrumentation"]["worker_reported"]["format"] == (
        "orcacolony_worker_telemetry_v1"
    )
    assert receipts[0]["instrumentation"]["coordinator_measured"][
        "result_receive_seconds"
    ] >= 0
    assert status["state"] == "step_complete"
    assert "browser-a" not in json.dumps(status, sort_keys=True)
    assert "browser-b" not in json.dumps(status, sort_keys=True)


def test_public_origin_is_canonical_and_rejects_credentials() -> None:
    assert normalize_http_origin("https://Workers.Example:443/") == "https://workers.example"
    assert normalize_http_origin("http://[::1]:8000") == "http://[::1]:8000"
    with pytest.raises(ValueError, match="without a path"):
        normalize_http_origin("https://user:password@workers.example")
    with pytest.raises(ValueError, match="HTTPS except on loopback"):
        normalize_http_origin("http://workers.example")
    with pytest.raises(ValueError, match="invalid characters"):
        normalize_http_origin('https://workers.example" onload="alert(1)')
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://workers.example%22x")
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://[fe80::1%25eth0]")
    with pytest.raises(ValueError, match="invalid characters"):
        normalize_http_origin("https://workers.example\nevil")
    with pytest.raises(ValueError, match="hostname is invalid"):
        normalize_http_origin("https://127.1")


def test_next_global_step_resumes_model_optimizer_and_dataset_cursor(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    second = GlobalStepCoordinator.create(
        campaign,
        tmp_path / "step-2",
        worker_count=2,
        participants=participants,
        resume_from=first.checkpoint_dir,
    )
    second_a = second.lease("worker-a", worker_token="test-token", now=200)
    second_b = second.lease("worker-b", worker_token="test-token", now=200)

    assert second_a["global_step"] == 1
    assert second_a["data_range"] == [4, 6]
    assert second_b["data_range"] == [6, 8]

    second.accept(submission_for(second, second_a), now=201)
    receipt = second.accept(submission_for(second, second_b), now=201)
    checkpoint_state = json.loads(
        (second.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )

    assert receipt.step_complete is True
    assert receipt.step == 2
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    assert checkpoint_state["step"] == 2
    assert checkpoint_state["dataset_cursor"] == 8
    assert len(checkpoint_state["loss_history"]) == 2


def test_int8_profile_binds_oracle_assignment_checkpoint_and_restart(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )

    first = coordinator.lease("worker-a", worker_token="test-token", now=100)
    assert first["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    assert first["runtime_backends"] == [
        "python-native-cpu-int8-f32-dequant",
        "python-oracle-int8-f32-dequant",
    ]
    assert first["weight_checkpoint_sha256"] == peft.lora_weight_checkpoint_sha256(
        loaded,
        str(first["adapter_sha256"]),
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )
    with pytest.raises(ValueError, match="numerical profile"):
        coordinator.accept(submission_for(coordinator, first), now=101)
    coordinator.accept(
        submission_for(
            coordinator,
            first,
            runtime_backend="python-oracle-int8-f32-dequant",
        ),
        now=101,
    )
    second = coordinator.lease("worker-b", worker_token="test-token", now=100)
    receipt = coordinator.accept(
        submission_for(
            coordinator,
            second,
            runtime_backend="python-oracle-int8-f32-dequant",
        ),
        now=101,
    )

    assert receipt.step_complete is True
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    checkpoint_state = json.loads(
        (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["format"] == "orcacolony_lora_checkpoint_v2"
    assert checkpoint_state["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    global_state = json.loads(
        (state_dir / "global-state.json").read_text(encoding="utf-8")
    )
    assert global_state["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    global_receipt = json.loads(
        (state_dir / "global-receipt.json").read_text(encoding="utf-8")
    )
    assert global_receipt["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE

    recovered = GlobalStepCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
        numerical_profile=peft.INT8_FROZEN_LINEAR_PROFILE,
    )
    assert recovered.status()["numerical_profile"] == peft.INT8_FROZEN_LINEAR_PROFILE
    with pytest.raises(ValueError, match="numerical profile"):
        GlobalStepCoordinator.load(
            loaded.campaign,
            state_dir,
            participants=participants,
            lora=loaded,
            numerical_profile=peft.EXACT_CPU_FP32_PROFILE,
        )


def test_lora_workers_aggregate_only_adapters_and_reload_the_checkpoint(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        state_dir,
        worker_count=2,
        participants=participants,
        lora=loaded,
    )

    expected_names = [
        name
        for layer in range(loaded.campaign.model.layers)
        for name in (
            f"blocks.{layer}.attention.qkv.lora_a",
            f"blocks.{layer}.attention.qkv.lora_b",
        )
    ]
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
            now=100,
        )
        assert assignment["format"] == "orcacolony_assignment_v2"
        assert assignment["training_method"] == "frozen-base-lora"
        assert assignment["base_model_sha256"] == loaded.config.base_model_sha256
        assert assignment["weight_checkpoint_sha256"] == assignment["checkpoint_sha256"]
        assert assignment["resume_state_sha256"] != assignment["checkpoint_sha256"]
        assert assignment["adapter"]["tensor_order"] == expected_names
        assert assignment["adapter"]["value_count"] == 8_192
        assert assignment["adapter_url"] == "/api/v1/artifacts/adapter.safetensors"
        receipt = coordinator.accept(submission_for(coordinator, assignment), now=101)

    assert receipt.step_complete is True
    assert receipt.model_sha256 == loaded.config.base_model_sha256
    assert receipt.adapter_sha256 is not None
    assert receipt.weight_checkpoint_sha256 is not None
    assert receipt.checkpoint_sha256 is not None
    assert receipt.weight_checkpoint_sha256 != receipt.checkpoint_sha256
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6

    checkpoint_state = json.loads(
        (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["format"] == "orcacolony_lora_checkpoint_v1"
    assert checkpoint_state["base_model_sha256"] == loaded.config.base_model_sha256
    assert checkpoint_state["adapter"]["tensor_sha256"] == receipt.adapter_sha256
    assert checkpoint_state["weight_checkpoint_sha256"] == receipt.weight_checkpoint_sha256
    assert checkpoint_state["checkpoint_sha256"] == receipt.checkpoint_sha256
    assert sorted(
        load_safetensors_file(str(coordinator.checkpoint_dir / "adapter.safetensors"))
    ) == expected_names
    assert all(
        name.startswith(("exp_avg.", "exp_avg_sq."))
        and name.split(".", 1)[1] in expected_names
        for name in load_safetensors_file(
            str(coordinator.checkpoint_dir / "optimizer.safetensors")
        )
    )

    recovered = GlobalStepCoordinator.load(
        loaded.campaign,
        state_dir,
        participants=participants,
        lora=loaded,
    )
    assert recovered.status()["state"] == "step_complete"
    assert recovered.status()["adapter_sha256"] == receipt.adapter_sha256
    assert recovered.status()["result_checkpoint_sha256"] == receipt.checkpoint_sha256


def test_next_lora_step_resumes_adapter_optimizer_and_dataset_cursor(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    second = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-2",
        worker_count=2,
        participants=participants,
        resume_from=first.checkpoint_dir,
        lora=loaded,
    )
    assert second.status()["step"] == 1
    assert second.status()["initial_adapter_sha256"] == first.status()["adapter_sha256"]
    for worker_id in ("worker-a", "worker-b"):
        assignment = second.lease(worker_id, worker_token="test-token", now=200)
        receipt = second.accept(submission_for(second, assignment), now=201)

    assert receipt.step == 2
    assert receipt.checkpoint_metrics["relative_l2_error"] < 1e-6
    checkpoint_state = json.loads(
        (second.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert checkpoint_state["step"] == 2
    assert checkpoint_state["optimizer_step"] == 2
    assert checkpoint_state["dataset_cursor"] == 8
    assert len(checkpoint_state["loss_history"]) == 2


def test_lora_resume_revalidates_artifact_paths_before_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    first = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "step-1",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = first.lease(worker_id, worker_token="test-token", now=100)
        first.accept(submission_for(first, assignment), now=101)

    real_load = multiworker.load_lora_checkpoint

    def mutate_after_load(lora, checkpoint, **kwargs):
        result = real_load(lora, checkpoint, **kwargs)
        state_path = Path(checkpoint) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["optimizer"]["file"] = "../optimizer.safetensors"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return result

    monkeypatch.setattr(multiworker, "load_lora_checkpoint", mutate_after_load)
    with pytest.raises(ValueError, match="safe plain basename"):
        GlobalStepCoordinator.create(
            loaded.campaign,
            tmp_path / "step-2",
            worker_count=2,
            participants=participants,
            resume_from=first.checkpoint_dir,
            lora=loaded,
        )


def test_lora_http_contract_serves_assignments_artifacts_and_result_checkpoint(
    tmp_path: Path,
) -> None:
    loaded = load_lora_manifest(CONFIG, LORA_CONFIG)
    participants = participants_for(loaded.campaign.campaign["id"])
    coordinator = GlobalStepCoordinator.create(
        loaded.campaign,
        tmp_path / "coordinator",
        worker_count=2,
        participants=participants,
        lora=loaded,
    )
    expected_model_bytes = coordinator.initial_model_path.read_bytes()
    expected_adapter_bytes = coordinator.initial_adapter_path.read_bytes()
    mutated_model = load_safetensors(expected_model_bytes)
    first_model_name = next(iter(mutated_model))
    mutated_model[first_model_name] = mutated_model[first_model_name] + 1.0
    coordinator.initial_model_path.write_bytes(save_safetensors(mutated_model))
    mutated_adapter = load_safetensors(expected_adapter_bytes)
    first_adapter_name = next(iter(mutated_adapter))
    mutated_adapter[first_adapter_name] = mutated_adapter[first_adapter_name] + 1.0
    coordinator.initial_adapter_path.write_bytes(save_safetensors(mutated_adapter))
    browser_root = tmp_path / "browser"
    browser_root.mkdir()
    (browser_root / "index.html").write_text("OrcaColony", encoding="utf-8")
    server = create_http_server(coordinator, browser_root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    receipts = []
    try:
        for worker_id in ("browser-a", "browser-b"):
            assignment_request = Request(
                f"{base_url}/api/v1/assignment?{urlencode({'worker_id': worker_id})}",
                headers={"X-Orca-Worker-Token": "test-token"},
            )
            with urlopen(assignment_request) as response:
                assignment = json.load(response)
            assert assignment["model_url"] == "/api/v1/artifacts/model.safetensors"
            assert assignment["adapter_url"] == "/api/v1/artifacts/adapter.safetensors"
            with urlopen(f"{base_url}{assignment['model_url']}") as response:
                served_model_bytes = response.read()
                initial_base = load_safetensors(served_model_bytes)
            with urlopen(f"{base_url}{assignment['adapter_url']}") as response:
                served_adapter_bytes = response.read()
                initial_adapter = load_safetensors(served_adapter_bytes)
            assert served_model_bytes == expected_model_bytes
            assert served_adapter_bytes == expected_adapter_bytes
            assert multiworker.tensor_sha256(initial_base) == assignment["base_model_sha256"]
            assert multiworker.tensor_sha256(initial_adapter) == assignment["adapter_sha256"]

            result_request = Request(
                f"{base_url}{assignment['result_url']}",
                data=coordinator.oracle_gradient_path(
                    assignment["assignment_id"]
                ).read_bytes(),
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Orca-Lease-Token": assignment["lease_token"],
                    "X-Orca-Checkpoint-Sha256": assignment["checkpoint_sha256"],
                    "X-Orca-Loss-Sum": str(assignment["expected_loss_sum"]),
                    "X-Orca-Loss-Weight-Sum": str(assignment["loss_weight_sum"]),
                    "X-Orca-Runtime-Backend": "python-oracle-f32",
                    "X-Orca-Worker-Telemetry": json.dumps(
                        worker_telemetry(coordinator, assignment),
                        separators=(",", ":"),
                    ),
                },
            )
            with urlopen(result_request) as response:
                receipts.append(json.load(response))
            if len(receipts) == 1:
                base_adapter_path = (
                    coordinator.base_checkpoint_dir / "adapter.safetensors"
                )
                base_adapter_path.write_bytes(save_safetensors(mutated_adapter))

        completed = receipts[-1]
        expected_completed_adapter = coordinator.checkpoint_artifact_bytes(
            "adapter.safetensors"
        )
        (coordinator.checkpoint_dir / "adapter.safetensors").write_bytes(
            save_safetensors(mutated_adapter)
        )
        with urlopen(f"{base_url}{completed['checkpoint_url']}") as response:
            completed_adapter_bytes = response.read()
            completed_adapter = load_safetensors(completed_adapter_bytes)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert receipts[0]["step_complete"] is False
    assert receipts[0]["adapter_sha256"] is None
    assert receipts[0]["checkpoint_sha256"] is None
    assert receipts[0]["checkpoint_url"] is None
    assert completed["step_complete"] is True
    assert completed_adapter_bytes == expected_completed_adapter
    assert completed["model_sha256"] == loaded.config.base_model_sha256
    assert completed["adapter_sha256"] == multiworker.tensor_sha256(completed_adapter)
    assert completed[
        "weight_checkpoint_sha256"
    ] == multiworker.lora_weight_checkpoint_sha256(
        loaded,
        completed["adapter_sha256"],
    )
    checkpoint_state = json.loads(
        (coordinator.checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    assert completed["checkpoint_sha256"] == checkpoint_state["checkpoint_sha256"]
    assert completed["checkpoint_sha256"] != completed["weight_checkpoint_sha256"]
    assert completed["checkpoint_url"] == "/api/v1/checkpoint/adapter.safetensors"


def test_dense_restart_rejects_combined_pre_lora_and_profile_migration(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )

    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for field in (
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "initial_adapter_sha256",
        "resume_state_sha256",
        "adapter_sha256",
        "result_weight_checkpoint_sha256",
        "result_checkpoint_sha256",
        "numerical_profile",
    ):
        state.pop(field)
    for assignment in state["assignments"]:
        for field in (
            "training_method",
            "lora_manifest_sha256",
            "base_model_sha256",
            "adapter_sha256",
            "adapter",
            "numerical_profile",
        ):
            assignment.pop(field, None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for field in (
        "training_method",
        "lora_manifest_sha256",
        "base_model_sha256",
        "adapter_sha256",
        "resume_state_sha256",
        "numerical_profile",
    ):
        lock.pop(field)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="combined global-step migration"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


def test_global_step_legacy_profile_migration_rejects_a_profiled_lock(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "coordinator"
    GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("numerical_profile")
    for assignment in state["assignments"]:
        assignment.pop("numerical_profile")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["numerical_profile"] = peft.INT8_FROZEN_LINEAR_PROFILE
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="campaign lock mismatch"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )


def test_global_step_exact_profile_predecessor_migrates_once(tmp_path: Path) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "exact-profile-predecessor"
    GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("numerical_profile")
    for assignment in state["assignments"]:
        assignment.pop("numerical_profile")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.pop("numerical_profile")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    GlobalStepCoordinator.load(
        campaign,
        state_dir,
        participants=participants,
    )
    migrated_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated_state["numerical_profile"] == peft.EXACT_CPU_FP32_PROFILE
    assert all(
        assignment["numerical_profile"] == peft.EXACT_CPU_FP32_PROFILE
        for assignment in migrated_state["assignments"]
    )


def test_completed_dense_resume_state_binds_optimizer_and_state_bytes(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / "completed-dense"
    coordinator = GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )
    for worker_id in ("worker-a", "worker-b"):
        assignment = coordinator.lease(
            worker_id,
            worker_token="test-token",
        )
        coordinator.accept(submission_for(coordinator, assignment))
    global_state_path = state_dir / "global-state.json"
    lock_path = state_dir / "campaign-lock.json"
    original_state_bytes = global_state_path.read_bytes()
    original_lock_bytes = lock_path.read_bytes()
    for field, mutation, message in (
        ("result_dataset_cursor", 0.0, "completed global-step progress identity"),
        ("result_loss_history", [1], "completed global-step progress identity"),
        ("result_dataset_cursor", None, "current global-step state schema"),
        ("result_loss_history", None, "current global-step state schema"),
    ):
        state = json.loads(original_state_bytes)
        lock = json.loads(original_lock_bytes)
        if mutation is None:
            state.pop(field)
            lock.pop(field)
        else:
            state[field] = mutation
            lock[field] = mutation
        global_state_path.write_text(json.dumps(state), encoding="utf-8")
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            GlobalStepCoordinator.load(
                campaign,
                state_dir,
                participants=participants,
            )
        global_state_path.write_bytes(original_state_bytes)
        lock_path.write_bytes(original_lock_bytes)
    optimizer_path = coordinator.checkpoint_dir / "optimizer.safetensors"
    optimizer = load_safetensors(optimizer_path.read_bytes())
    tensor_name = next(
        name for name, tensor in optimizer.items() if tensor.dtype.is_floating_point
    )
    optimizer[tensor_name] = optimizer[tensor_name] + 1.0
    optimizer_bytes = save_safetensors(optimizer)
    optimizer_path.write_bytes(optimizer_bytes)
    checkpoint_state_path = coordinator.checkpoint_dir / "state.json"
    checkpoint_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
    checkpoint_state["optimizer"]["sha256"] = hashlib.sha256(
        optimizer_bytes
    ).hexdigest()
    checkpoint_state_path.write_text(json.dumps(checkpoint_state), encoding="utf-8")
    state_before = global_state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="completed global-step checkpoint identity"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )
    assert global_state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before


@pytest.mark.parametrize("location", ("state", "assignment"))
def test_global_step_profile_predecessor_rejects_unknown_fields_without_rewrite(
    tmp_path: Path,
    location: str,
) -> None:
    campaign = load_campaign(CONFIG)
    participants = participants_for(campaign.campaign["id"])
    state_dir = tmp_path / f"profile-predecessor-{location}"
    GlobalStepCoordinator.create(
        campaign,
        state_dir,
        worker_count=2,
        participants=participants,
    )
    state_path = state_dir / "global-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("numerical_profile")
    for assignment in state["assignments"]:
        assignment.pop("numerical_profile")
    target = state if location == "state" else state["assignments"][0]
    target["unknown_predecessor_field"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")
    lock_path = state_dir / "campaign-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.pop("numerical_profile")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    state_before = state_path.read_bytes()
    lock_before = lock_path.read_bytes()

    with pytest.raises(ValueError, match="numerical-profile.*schema"):
        GlobalStepCoordinator.load(
            campaign,
            state_dir,
            participants=participants,
        )
    assert state_path.read_bytes() == state_before
    assert lock_path.read_bytes() == lock_before
