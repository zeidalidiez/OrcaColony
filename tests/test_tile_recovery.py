from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import orcacolony.tile_recovery as tile_recovery
from orcacolony.reference import (
    _create_optimizer,
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
)
from orcacolony.tile_recovery import main, run_recovered_tile_transaction
from orcacolony.tiled_model import _prefix_activation, _suffix_logits


CONFIG = Path(__file__).parents[1] / "campaign" / "t0-smoke.json"
EXPECTED_FILES = {
    "manifest.json",
    "tile.safetensors",
    "input.safetensors",
    "forward-output.safetensors",
    "output-adjoint.safetensors",
    "result.safetensors",
}


class _StopBeforeApply(Exception):
    pass


def _prepare_result_accepted_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    transaction_dir = tmp_path / "transaction"

    def stop_before_apply(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        raise _StopBeforeApply

    original_apply = tile_recovery._apply_result_once
    monkeypatch.setattr(tile_recovery, "_apply_result_once", stop_before_apply)
    with pytest.raises(_StopBeforeApply):
        run_recovered_tile_transaction(
            load_campaign(CONFIG),
            block_index=2,
            transaction_dir=transaction_dir,
            timeout_seconds=30.0,
        )
    monkeypatch.setattr(tile_recovery, "_apply_result_once", original_apply)
    return transaction_dir


def _coordinator_before_apply(
    transaction_dir: Path,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor]:
    campaign = load_campaign(CONFIG)
    recovered = build_model(campaign)
    optimizer = _create_optimizer(recovered, campaign.training)
    inputs, targets = fixture_batch(campaign, 0)
    recovered.train()
    optimizer.zero_grad(set_to_none=True)
    block_input = _prefix_activation(recovered, inputs, 2)
    manifest = tile_recovery._load_manifest(transaction_dir)
    output_tensors = tile_recovery._deserialize_tensors(
        tile_recovery._read_owned_tensor_file(
            transaction_dir,
            manifest,
            "forward-output.safetensors",
        )
    )
    boundary_output = tile_recovery._validate_tensor(
        output_tensors["output"],
        shape=block_input.shape,
        label="output",
    ).requires_grad_(True)
    loss = F.cross_entropy(
        _suffix_logits(recovered, boundary_output, 2).reshape(
            -1,
            campaign.model.vocabulary_size,
        ),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    return recovered, optimizer, block_input


def _expected_identity(transaction_dir: Path) -> dict[str, object]:
    manifest = tile_recovery._load_manifest(transaction_dir)
    return {
        name: manifest[name]
        for name in tile_recovery._TRANSACTION_IDENTITY_FIELDS
    }


def _assert_apply_rejected_without_mutation(
    transaction_dir: Path,
    expected_identity: dict[str, object],
    match: str,
) -> None:
    recovered, optimizer, block_input = _coordinator_before_apply(transaction_dir)
    model_before = tensor_sha256(recovered.state_dict())
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in recovered.named_parameters()
    }
    with pytest.raises(ValueError, match=match):
        tile_recovery._apply_result_once(
            transaction_dir,
            recovered,
            optimizer,
            block_input,
            2,
            load_campaign(CONFIG),
            expected_identity=expected_identity,
        )
    assert tensor_sha256(recovered.state_dict()) == model_before
    assert not optimizer.state
    for name, parameter in recovered.named_parameters():
        expected = gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    transaction_dir = tmp_path / "transaction"
    transaction_dir.mkdir()
    (transaction_dir / "manifest.json").write_text(
        '{"phase":"result_accepted","phase":"applied"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        tile_recovery._load_manifest(transaction_dir)


@pytest.mark.parametrize("failure_stage", ("write", "flush", "fsync", "replace"))
def test_atomic_write_failure_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    destination = tmp_path / "manifest.json"
    temporary = tmp_path / "manifest.json.tmp"
    destination.write_bytes(b"old authority\n")
    original_open = Path.open

    class FaultingHandle:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self) -> FaultingHandle:
            return self

        def __exit__(self, *args: object) -> object:
            return self.handle.__exit__(*args)

        def write(self, payload: bytes) -> int:
            if failure_stage == "write":
                self.handle.write(payload[:1])
                raise OSError("injected atomic write failure")
            return self.handle.write(payload)

        def flush(self) -> None:
            self.handle.flush()
            if failure_stage == "flush":
                raise OSError("injected atomic flush failure")

        def fileno(self) -> int:
            return self.handle.fileno()

    def faulting_open(path: Path, *args: object, **kwargs: object) -> object:
        handle = original_open(path, *args, **kwargs)
        if path == temporary and failure_stage in {"write", "flush"}:
            return FaultingHandle(handle)
        return handle

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected atomic fsync failure")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected atomic replace failure")

    if failure_stage in {"write", "flush"}:
        monkeypatch.setattr(Path, "open", faulting_open)
    elif failure_stage == "fsync":
        monkeypatch.setattr(tile_recovery.os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(tile_recovery.os, "replace", fail_replace)

    with pytest.raises(OSError, match=f"injected atomic {failure_stage} failure"):
        tile_recovery._write_bytes_atomic(destination, b"new authority\n")

    assert destination.read_bytes() == b"old authority\n"
    assert not temporary.exists()


def test_atomic_write_cleanup_failure_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "manifest.json"
    temporary = tmp_path / "manifest.json.tmp"
    destination.write_bytes(b"old authority\n")
    original_unlink = Path.unlink

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected atomic replace failure")

    def fail_temporary_cleanup(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path == temporary:
            raise OSError("injected temporary cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tile_recovery.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)

    with pytest.raises(
        RuntimeError,
        match="failed to remove incomplete transaction file",
    ) as failure:
        tile_recovery._write_bytes_atomic(destination, b"new authority\n")

    assert isinstance(failure.value.__cause__, OSError)
    assert destination.read_bytes() == b"old authority\n"
    assert temporary.exists()


def test_malformed_phase_history_is_rejected_before_coordinator_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction_dir = _prepare_result_accepted_transaction(tmp_path, monkeypatch)
    expected_identity = _expected_identity(transaction_dir)
    manifest = tile_recovery._load_manifest(transaction_dir)
    manifest["phase_history"] = ["prepared", "worker_lost"]
    tile_recovery._write_manifest(transaction_dir, manifest)

    recovered, optimizer, block_input = _coordinator_before_apply(transaction_dir)
    model_before = tensor_sha256(recovered.state_dict())
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in recovered.named_parameters()
    }

    with pytest.raises(ValueError, match="phase history"):
        tile_recovery._apply_result_once(
            transaction_dir,
            recovered,
            optimizer,
            block_input,
            2,
            load_campaign(CONFIG),
            expected_identity=expected_identity,
        )

    assert tensor_sha256(recovered.state_dict()) == model_before
    assert not optimizer.state
    for name, parameter in recovered.named_parameters():
        expected = gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)
    rejected = tile_recovery._load_manifest(transaction_dir)
    assert rejected["phase"] == "result_accepted"
    assert rejected["result_applied"] is False


def test_failed_applied_transition_rolls_back_and_remains_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction_dir = _prepare_result_accepted_transaction(tmp_path, monkeypatch)
    expected_identity = _expected_identity(transaction_dir)
    recovered, optimizer, block_input = _coordinator_before_apply(transaction_dir)
    model_before = tensor_sha256(recovered.state_dict())
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in recovered.named_parameters()
    }
    original_replace = tile_recovery.os.replace

    def fail_applied_replace(source: object, destination: object) -> None:
        if Path(destination) == transaction_dir / "manifest.json":
            raise OSError("injected applied-state replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(tile_recovery.os, "replace", fail_applied_replace)
    with pytest.raises(OSError, match="injected applied-state replacement failure"):
        tile_recovery._apply_result_once(
            transaction_dir,
            recovered,
            optimizer,
            block_input,
            2,
            load_campaign(CONFIG),
            expected_identity=expected_identity,
        )
    monkeypatch.setattr(tile_recovery.os, "replace", original_replace)

    assert tensor_sha256(recovered.state_dict()) == model_before
    assert not optimizer.state
    for name, parameter in recovered.named_parameters():
        expected = gradients_before[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)
    retryable = tile_recovery._load_manifest(transaction_dir)
    assert retryable["phase"] == "result_accepted"
    assert retryable["result_applied"] is False
    assert not (transaction_dir / "manifest.json.tmp").exists()

    tile_recovery._apply_result_once(
        transaction_dir,
        recovered,
        optimizer,
        block_input,
        2,
        load_campaign(CONFIG),
        expected_identity=expected_identity,
    )
    applied = tile_recovery._load_manifest(transaction_dir)
    assert applied["phase"] == "applied"
    assert applied["result_applied"] is True


def test_unexpected_transaction_file_is_rejected_before_coordinator_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction_dir = _prepare_result_accepted_transaction(tmp_path, monkeypatch)
    expected_identity = _expected_identity(transaction_dir)
    (transaction_dir / "unexpected.bin").write_bytes(b"not admitted")
    recovered, optimizer, block_input = _coordinator_before_apply(transaction_dir)
    model_before = tensor_sha256(recovered.state_dict())

    with pytest.raises(ValueError, match="unexpected files"):
        tile_recovery._apply_result_once(
            transaction_dir,
            recovered,
            optimizer,
            block_input,
            2,
            load_campaign(CONFIG),
            expected_identity=expected_identity,
        )

    assert tensor_sha256(recovered.state_dict()) == model_before
    assert not optimizer.state


def test_manifest_file_and_result_corruption_are_rejected_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction_dir = _prepare_result_accepted_transaction(tmp_path, monkeypatch)
    expected_identity = _expected_identity(transaction_dir)
    baseline = {
        path.name: path.read_bytes()
        for path in transaction_dir.iterdir()
    }

    def restore() -> None:
        for path in tuple(transaction_dir.iterdir()):
            if path.name not in baseline:
                path.unlink()
        for name, payload in baseline.items():
            tile_recovery._write_bytes_atomic(transaction_dir / name, payload)

    manifest = tile_recovery._load_manifest(transaction_dir)
    manifest["campaign_id"] = "tampered-campaign"
    tile_recovery._write_manifest(transaction_dir, manifest)
    _assert_apply_rejected_without_mutation(
        transaction_dir,
        expected_identity,
        "transaction identity",
    )

    restore()
    manifest = tile_recovery._load_manifest(transaction_dir)
    manifest["unexpected"] = True
    tile_recovery._write_manifest(transaction_dir, manifest)
    _assert_apply_rejected_without_mutation(
        transaction_dir,
        expected_identity,
        "manifest schema",
    )

    restore()
    tile_path = transaction_dir / "tile.safetensors"
    tile_path.write_bytes(tile_path.read_bytes() + b"changed")
    _assert_apply_rejected_without_mutation(
        transaction_dir,
        expected_identity,
        "file size changed",
    )

    restore()
    manifest = tile_recovery._load_manifest(transaction_dir)
    result = {
        name: tensor.detach().clone()
        for name, tensor in tile_recovery._deserialize_tensors(
            (transaction_dir / "result.safetensors").read_bytes()
        ).items()
    }
    gradient_name = next(name for name in sorted(result) if name.startswith("gradient."))
    result[gradient_name].view(-1)[0] = float("nan")
    tile_recovery._record_tensor_file(
        transaction_dir,
        manifest,
        "result.safetensors",
        tile_recovery._serialize_tensors(result),
    )
    tile_recovery._write_manifest(transaction_dir, manifest)
    _assert_apply_rejected_without_mutation(
        transaction_dir,
        expected_identity,
        "tensor is non-finite",
    )


def test_recovery_worker_rejects_mismatched_tile_state_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def send_bytes(self, _payload: bytes) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeProcess:
        exitcode = -15

        def start(self) -> None:
            pass

        def terminate(self) -> None:
            pass

        def join(self, _timeout: float) -> None:
            pass

        def kill(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    class FakeContext:
        def __init__(self) -> None:
            self.parent = FakeConnection()
            self.child = FakeConnection()
            self.process = FakeProcess()

        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is True
            return self.parent, self.child

        def Process(self, **_kwargs: object) -> FakeProcess:
            return self.process

    ready = {
        "status": "ready",
        "tile_state_sha256": "wrong-state",
        "startup_current_rss_bytes": 1,
        "startup_peak_rss_bytes": 1,
        "after_model_current_rss_bytes": 1,
        "after_model_peak_rss_bytes": 1,
    }
    monkeypatch.setattr(tile_recovery, "_send_json", lambda *_args: None)
    monkeypatch.setattr(tile_recovery, "_await_model_readiness", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tile_recovery,
        "_recv_json",
        lambda *_args, **_kwargs: (ready, 0),
    )

    with pytest.raises(ValueError, match="initialization acknowledgement"):
        tile_recovery._start_worker(
            FakeContext(),
            load_campaign(CONFIG),
            2,
            b"tile",
            1.0,
            name="mismatched-state-worker",
            expected_tile_state_sha256="expected-state",
        )


def test_replacement_tile_replays_and_applies_one_exact_result(
    tmp_path: Path,
) -> None:
    campaign = load_campaign(CONFIG)
    transaction_dir = tmp_path / "transaction"

    evidence = run_recovered_tile_transaction(
        campaign,
        block_index=2,
        transaction_dir=transaction_dir,
        timeout_seconds=30.0,
    )

    assert evidence.format == "orcacolony_recovered_tile_transaction_evidence_v2"
    assert evidence.start_method == "spawn"
    assert evidence.block_index == 2
    assert evidence.cursor == 0
    assert evidence.worker_model_transmissions == 2
    assert evidence.first_worker_terminated is True
    assert evidence.first_worker_exit_code != 0
    assert evidence.replacement_worker_exit_code == 0
    assert evidence.replay_output_bytes_identical is True
    assert evidence.duplicate_result_rejected is True
    assert evidence.phase_history == (
        "prepared",
        "forward_accepted",
        "worker_lost",
        "replay_verified",
        "adjoint_persisted",
        "result_accepted",
        "applied",
    )
    assert evidence.max_abs_raw_gradient_difference == 0.0
    assert evidence.max_abs_clipped_gradient_difference == 0.0
    assert evidence.max_abs_model_difference == 0.0
    assert (
        evidence.centralized_raw_gradient_sha256
        == evidence.recovered_raw_gradient_sha256
    )
    assert (
        evidence.centralized_clipped_gradient_sha256
        == evidence.recovered_clipped_gradient_sha256
    )
    assert (
        evidence.centralized_optimizer_sha256
        == evidence.recovered_optimizer_sha256
    )
    assert evidence.centralized_model_sha256 == evidence.recovered_model_sha256
    assert evidence.recovery_seconds > 0
    assert evidence.recovery_retransmitted_tensor_bytes == (
        evidence.tile_model_wire_bytes + evidence.input_wire_bytes
    )
    assert evidence.persisted_file_count == len(EXPECTED_FILES)
    assert evidence.persisted_tensor_bytes > 0

    assert {path.name for path in transaction_dir.iterdir()} == EXPECTED_FILES
    manifest = json.loads(
        (transaction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "orcacolony_boundary_transaction_v1"
    assert manifest["transaction_id"] == evidence.transaction_id
    assert manifest["phase"] == "applied"
    assert manifest["result_applied"] is True
    assert tuple(manifest["phase_history"]) == evidence.phase_history
    persisted_by_name = {item.name: item for item in evidence.persisted_files}
    transaction_identity = {
        "campaign_id": evidence.campaign_id,
        "dataset_revision": evidence.dataset_revision,
        "checkpoint_model_sha256": evidence.checkpoint_model_sha256,
        "block_index": evidence.block_index,
        "cursor": evidence.cursor,
        "tile_sha256": persisted_by_name["tile.safetensors"].sha256,
        "input_sha256": persisted_by_name["input.safetensors"].sha256,
    }
    assert evidence.transaction_id == hashlib.sha256(
        tile_recovery._canonical_json(transaction_identity)
    ).hexdigest()
    for item in evidence.persisted_files:
        path = transaction_dir / item.name
        assert path.stat().st_size == item.size_bytes
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item.sha256


def test_tile_recovery_cli_writes_evidence_and_transaction(
    tmp_path: Path,
) -> None:
    transaction_dir = tmp_path / "transaction"
    output_path = tmp_path / "evidence.json"

    main(
        [
            "--config",
            str(CONFIG),
            "--block-index",
            "2",
            "--transaction-dir",
            str(transaction_dir),
            "--timeout-seconds",
            "30",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == "orcacolony_recovered_tile_transaction_evidence_v2"
    assert payload["replay_output_bytes_identical"] is True
    assert payload["duplicate_result_rejected"] is True
    assert payload["centralized_model_sha256"] == payload["recovered_model_sha256"]
    manifest = json.loads(
        (transaction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "applied"
    assert manifest["result_applied"] is True
