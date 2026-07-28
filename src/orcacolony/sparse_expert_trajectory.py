from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import torch
from torch import Tensor

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import (
    CampaignConfig,
    ModelConfig,
    ObjectiveConfig,
    _create_optimizer,
    campaign_revision,
    configure_determinism,
    fixture_batch,
    load_campaign,
    tensor_sha256,
    validate_dataset_artifacts,
)
from orcacolony.sparse_expert import (
    SparseExpertDecoder,
    _balanced_top1_routes,
    _build_sparse_model,
    _freeze_head,
    _head_tensor_snapshot,
    _router_auxiliary_loss,
    _trainable_gradient_snapshot,
    _trainable_optimizer_tensor_snapshot,
)
from orcacolony.sparse_expert_process import (
    _FrozenHeadExpertProcessModule,
    _expert_worker_result,
    _full_worker_result,
    _head_state,
    _load_head_state,
    _load_trainable_state,
    _reject_duplicate_json_keys,
    _sha256_bytes,
    _trainable_state,
    _validate_wire_identity,
)
from orcacolony.tile_process import (
    _deserialize_tensors,
    _send_json,
    _serialize_tensors,
)
from orcacolony.tiled_model import _max_abs_difference, _model_snapshot


_INIT_FORMAT = "orcacolony_sparse_trajectory_worker_init_v1"
_RUN_FORMAT = "orcacolony_sparse_trajectory_assignment_v1"
_TRANSACTION_FORMAT = "orcacolony_sparse_trajectory_transaction_v1"
_APPLIED_FORMAT = "orcacolony_sparse_trajectory_applied_checkpoint_v1"
_EVIDENCE_FORMAT = "orcacolony_persisted_sparse_trajectory_evidence_v1"
_TRANSPORT_SCOPE = "trusted-local-spawn-pipe"
_AUTHENTICATION_MODE = "coordinator-bound-sha256-safetensors-v1"
_RSS_SOURCE = "linux-proc-status-v1"
_PHASES = ("prepared", "results_accepted", "applied")
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "identity",
        "transaction_id",
        "phase",
        "phase_history",
        "accepted_result_count",
        "accepted_results",
        "result_applied",
        "applied_checkpoint",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "campaign_id",
        "campaign_revision",
        "dataset_revision",
        "topology",
        "step",
        "cursor",
        "expert_count",
        "router_aux_weight",
        "frozen_head_sha256",
        "pre_model_sha256",
        "pre_model_wire_sha256",
        "pre_model_wire_bytes",
        "pre_optimizer_sha256",
        "pre_optimizer_wire_sha256",
        "pre_optimizer_wire_bytes",
        "batch_wire_sha256",
        "batch_wire_bytes",
        "routes_sha256",
        "expected_result_count",
        "result_preparations",
    }
)
_RESULT_PREPARATION_FIELDS = frozenset(
    {
        "result_index",
        "expert_index",
        "trainable_state_sha256",
        "trainable_wire_sha256",
        "trainable_wire_bytes",
        "input_wire_sha256",
        "input_wire_bytes",
    }
)
_ACCEPTANCE_FIELDS = frozenset(
    {
        "format",
        "worker_kind",
        "result_index",
        "expert_index",
        "step",
        "cursor",
        "campaign_revision",
        "dataset_revision",
        "frozen_head_sha256",
        "trainable_state_sha256",
        "input_wire_sha256",
        "routes_sha256",
        "loss",
        "routing",
        "result_wire_sha256",
        "result_wire_bytes",
    }
)
_RESULT_RECORD_FIELDS = frozenset(
    {
        "result_index",
        "directory",
        "acceptance_sha256",
        "acceptance_bytes",
        "result_wire_sha256",
        "result_wire_bytes",
    }
)
_APPLIED_STATE_FIELDS = frozenset(
    {
        "format",
        "transaction_id",
        "topology",
        "step",
        "model_sha256",
        "model_wire_sha256",
        "model_wire_bytes",
        "optimizer_sha256",
        "optimizer_wire_sha256",
        "optimizer_wire_bytes",
        "loss",
        "routes_sha256",
        "routing_counts",
        "unconstrained_routing_counts",
        "capacity_rerouted_tokens",
        "routing_capacity",
    }
)


@dataclass(frozen=True)
class ExternalRssEvidence:
    source: str
    sample_interval_seconds: float
    sample_count: int
    startup_sample_count: int
    assignment_sample_count: int
    shutdown_sample_count: int
    startup_max_current_rss_bytes: int
    startup_max_hwm_rss_bytes: int
    assignment_max_current_rss_bytes: int
    assignment_max_hwm_rss_bytes: int
    shutdown_max_current_rss_bytes: int
    shutdown_max_hwm_rss_bytes: int
    lifecycle_max_current_rss_bytes: int
    lifecycle_max_hwm_rss_bytes: int


@dataclass(frozen=True)
class TrajectoryWorkerEvidence:
    worker_kind: str
    generation: int
    initialization_seconds: float
    shutdown_seconds: float
    frozen_head_transmissions: int
    trainable_state_transmissions: int
    forward_calls: int
    control_json_wire_bytes: int
    child_exit_code: int
    exit_reason: str
    frozen_head_sha256: str
    external_rss: ExternalRssEvidence


@dataclass(frozen=True)
class PersistedTransactionEvidence:
    topology: str
    step: int
    transaction_id: str
    phase_history: tuple[str, ...]
    accepted_result_count: int
    persisted_result_bytes: int
    persisted_checkpoint_bytes: int
    manifest_sha256: str
    manifest_bytes: int
    applied_model_sha256: str
    applied_model_wire_sha256: str
    applied_optimizer_sha256: str
    applied_optimizer_wire_sha256: str
    coordinator_publish_recovered: bool
    duplicate_apply_rejected: bool


@dataclass(frozen=True)
class SparseTrajectoryStepEvidence:
    step: int
    cursor: int
    routing_capacity: int
    routing_counts: tuple[int, ...]
    unconstrained_routing_counts: tuple[int, ...]
    capacity_rerouted_tokens: int
    routes_sha256: str
    routes_changed_from_previous: bool
    centralized_pre_model_sha256: str
    full_pre_model_sha256: str
    expert_pre_model_sha256: str
    centralized_pre_optimizer_sha256: str
    full_pre_optimizer_sha256: str
    expert_pre_optimizer_sha256: str
    centralized_loss: float
    full_process_loss: float
    expert_process_loss: float
    full_max_abs_raw_gradient_difference: float
    expert_max_abs_raw_gradient_difference: float
    full_max_abs_clipped_gradient_difference: float
    expert_max_abs_clipped_gradient_difference: float
    full_max_abs_model_difference: float
    expert_max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    full_process_raw_gradient_sha256: str
    expert_process_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    full_process_clipped_gradient_sha256: str
    expert_process_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    full_process_optimizer_sha256: str
    expert_process_optimizer_sha256: str
    centralized_model_sha256: str
    full_process_model_sha256: str
    expert_process_model_sha256: str
    full_trainable_state_sha256: str
    expert_trainable_state_sha256: tuple[str, ...]
    full_tensor_wire_bytes: int
    expert_tensor_wire_bytes: int
    full_control_json_wire_bytes: int
    expert_control_json_wire_bytes: int
    full_persisted_bytes: int
    expert_persisted_bytes: int
    centralized_step_seconds: float
    full_end_to_end_step_seconds: float
    expert_end_to_end_step_seconds: float
    full_worker_round_trip_seconds: float
    expert_worker_round_trip_seconds: tuple[float, ...]
    full_worker_compute_seconds: float
    expert_worker_compute_seconds: tuple[float, ...]
    full_persistence_seconds: float
    expert_persistence_seconds: float
    full_coordinator_apply_seconds: float
    expert_coordinator_apply_seconds: float
    full_transaction: PersistedTransactionEvidence
    expert_transaction: PersistedTransactionEvidence


@dataclass(frozen=True)
class WorkerReplacementEvidence:
    topology: str
    step: int
    persisted_result_index: int
    persisted_result_survived_loss: bool
    first_worker_exit_code: int
    replacement_worker_exit_code: int
    replacement_initialized_after_loss: bool
    recomputed_persisted_result: bool


@dataclass(frozen=True)
class CoordinatorRecoveryEvidence:
    topology: str
    step: int
    applied_checkpoint_published_before_loss: bool
    manifest_applied_before_loss: bool
    recovered_from_published_checkpoint: bool
    recovery_start_method: str
    recovery_process_exit_code: int
    new_process_loaded_only_persisted_state: bool
    recomputed_from_persisted_pre_state_for_validation: bool
    duplicate_apply_rejected: bool
    recovery_seconds: float


@dataclass(frozen=True)
class PersistedSparseTrajectoryEvidence:
    format: str
    campaign_id: str
    campaign_revision: str
    dataset_revision: str
    authentication_mode: str
    transport_scope: str
    start_method: str
    process_scheduling: str
    assignment_state_mode: str
    persistence_scope: str
    timing_scope: str
    memory_scope: str
    maximum_simultaneous_worker_processes: int
    expert_count: int
    step_count: int
    frozen_head_sha256: str
    frozen_head_wire_sha256: str
    frozen_head_wire_bytes: int
    centralized_end_to_end_seconds: float
    full_process_end_to_end_seconds: float
    expert_process_end_to_end_seconds: float
    full_tensor_wire_bytes: int
    expert_tensor_wire_bytes: int
    full_control_json_wire_bytes: int
    expert_control_json_wire_bytes: int
    full_persisted_bytes: int
    expert_persisted_bytes: int
    centralized_final_model_sha256: str
    full_process_final_model_sha256: str
    expert_process_final_model_sha256: str
    centralized_final_optimizer_sha256: str
    full_process_final_optimizer_sha256: str
    expert_process_final_optimizer_sha256: str
    all_steps_exact: bool
    full_workers: tuple[TrajectoryWorkerEvidence, ...]
    expert_workers: tuple[TrajectoryWorkerEvidence, ...]
    worker_replacement: WorkerReplacementEvidence
    coordinator_recovery: CoordinatorRecoveryEvidence
    steps: tuple[SparseTrajectoryStepEvidence, ...]


@dataclass(frozen=True)
class _PreparedTopologyStep:
    step: int
    cursor: int
    inputs: Tensor
    targets: Tensor
    routes: Tensor | None
    routing_counts: tuple[int, ...] | None
    unconstrained_routing_counts: tuple[int, ...] | None
    capacity_rerouted_tokens: int | None
    routing_capacity: int | None
    routes_sha256: str | None
    batch_wire: bytes
    trainable_states: tuple[dict[str, Tensor], ...]
    trainable_wires: tuple[bytes, ...]
    input_wires: tuple[bytes, ...]


@dataclass(frozen=True)
class _CollectedResult:
    acknowledgement: Mapping[str, object]
    result_wire: bytes
    control_json_wire_bytes: int
    round_trip_seconds: float


@dataclass(frozen=True)
class _TrajectorySnapshot:
    step: int
    cursor: int
    pre_model_sha256: str
    pre_optimizer_sha256: str
    loss: float
    routes_sha256: str
    routing_counts: tuple[int, ...]
    unconstrained_routing_counts: tuple[int, ...]
    capacity_rerouted_tokens: int
    routing_capacity: int
    raw_gradients: Mapping[str, Tensor]
    clipped_gradients: Mapping[str, Tensor]
    optimizer: Mapping[str, Tensor]
    model: Mapping[str, Tensor]
    step_seconds: float


@dataclass(frozen=True)
class _AppliedCandidate:
    model: SparseExpertDecoder
    optimizer: torch.optim.AdamW
    snapshot: _TrajectorySnapshot
    model_wire: bytes
    optimizer_wire: bytes
    applied_state: dict[str, object]


class _SimulatedCoordinatorLoss(RuntimeError):
    def __init__(
        self,
        candidate: _AppliedCandidate,
        persisted_bytes: int,
    ) -> None:
        super().__init__(
            "coordinator lost after applied checkpoint publication"
        )
        self.candidate = candidate
        self.persisted_bytes = persisted_bytes


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, object]:
    parsed = json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"incomplete file already exists: {temporary.name}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_owned_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"persisted file is not a regular owned file: {path.name}")
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(f"persisted file size changed: {path.name}")
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"persisted file digest changed: {path.name}")
    return payload


class _ExternalRssSampler:
    def __init__(self, process: multiprocessing.Process, interval: float) -> None:
        self.process = process
        self.interval = interval
        self.samples: list[tuple[str, int, int]] = []

    def sample(self, phase: str) -> None:
        if phase not in {"startup", "assignment", "shutdown"}:
            raise ValueError("external RSS phase is invalid")
        pid = self.process.pid
        if pid is None:
            return
        status = Path(f"/proc/{pid}/status")
        try:
            fields = status.read_text(encoding="ascii").splitlines()
        except (FileNotFoundError, ProcessLookupError):
            return
        values: dict[str, int] = {}
        for line in fields:
            if line.startswith(("VmRSS:", "VmHWM:")):
                name, raw = line.split(":", 1)
                parts = raw.split()
                if len(parts) != 2 or parts[1] != "kB":
                    raise ValueError("Linux process RSS field is invalid")
                values[name] = int(parts[0]) * 1024
        if set(values) != {"VmRSS", "VmHWM"}:
            raise ValueError("Linux process RSS fields are unavailable")
        self.samples.append((phase, values["VmRSS"], values["VmHWM"]))

    def evidence(self) -> ExternalRssEvidence:
        if not self.samples:
            raise ValueError("external child RSS sampling produced no samples")

        def count(phase: str) -> int:
            return sum(sample_phase == phase for sample_phase, _, _ in self.samples)

        def maximum(phase: str, index: int) -> int:
            values = [
                sample[index]
                for sample in self.samples
                if sample[0] == phase
            ]
            return max(values) if values else 0

        return ExternalRssEvidence(
            source=_RSS_SOURCE,
            sample_interval_seconds=self.interval,
            sample_count=len(self.samples),
            startup_sample_count=count("startup"),
            assignment_sample_count=count("assignment"),
            shutdown_sample_count=count("shutdown"),
            startup_max_current_rss_bytes=maximum("startup", 1),
            startup_max_hwm_rss_bytes=maximum("startup", 2),
            assignment_max_current_rss_bytes=maximum("assignment", 1),
            assignment_max_hwm_rss_bytes=maximum("assignment", 2),
            shutdown_max_current_rss_bytes=maximum("shutdown", 1),
            shutdown_max_hwm_rss_bytes=maximum("shutdown", 2),
            lifecycle_max_current_rss_bytes=max(
                sample[1] for sample in self.samples
            ),
            lifecycle_max_hwm_rss_bytes=max(
                sample[2] for sample in self.samples
            ),
        )


@dataclass
class _WorkerHandle:
    process: multiprocessing.Process
    connection: Connection
    worker_kind: str
    generation: int
    initialization_seconds: float
    control_json_wire_bytes: int
    frozen_head_sha256: str
    sampler: _ExternalRssSampler
    trainable_state_transmissions: int = 0
    forward_calls: int = 0


def _recv_sampled_bytes(
    worker: _WorkerHandle,
    timeout_seconds: float,
    *,
    phase: str,
    label: str,
) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    while True:
        worker.sampler.sample(phase)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {label}")
        if worker.connection.poll(min(worker.sampler.interval, remaining)):
            worker.sampler.sample(phase)
            return worker.connection.recv_bytes()


def _recv_sampled_json(
    worker: _WorkerHandle,
    timeout_seconds: float,
    *,
    phase: str,
    label: str,
) -> tuple[dict[str, object], int]:
    raw = _recv_sampled_bytes(
        worker,
        timeout_seconds,
        phase=phase,
        label=label,
    )
    return _load_json_bytes(raw, label=label), len(raw)


def _validate_init_control(control: Mapping[str, object]) -> None:
    expected = frozenset(
        {
            "format",
            "op",
            "worker_kind",
            "expert_count",
            "model",
            "objective",
            "seed",
            "router_aux_weight",
            "campaign_id",
            "campaign_revision",
            "dataset_revision",
            "frozen_head_sha256",
            "frozen_head_wire_sha256",
            "frozen_head_wire_bytes",
        }
    )
    if frozenset(control) != expected:
        raise ValueError("trajectory worker init schema is invalid")
    if (
        control["format"] != _INIT_FORMAT
        or control["op"] != "init"
        or control["worker_kind"] not in {"full", "pooled_expert"}
        or type(control["expert_count"]) is not int
        or not 2 <= int(control["expert_count"]) <= 16
        or type(control["seed"]) is not int
        or type(control["router_aux_weight"]) is not float
        or not math.isfinite(float(control["router_aux_weight"]))
        or not 0.0 < float(control["router_aux_weight"]) <= 1.0
        or not isinstance(control["campaign_id"], str)
        or not isinstance(control["campaign_revision"], str)
        or not isinstance(control["dataset_revision"], str)
        or not isinstance(control["model"], dict)
        or not isinstance(control["objective"], dict)
    ):
        raise ValueError("trajectory worker init semantics are invalid")


def _validate_run_control(
    control: Mapping[str, object],
    *,
    worker_kind: str,
    expert_count: int,
    campaign_revision_value: str,
    dataset_revision: str,
    frozen_head_sha256: str,
) -> None:
    expected = frozenset(
        {
            "format",
            "op",
            "worker_kind",
            "result_index",
            "expert_index",
            "step",
            "cursor",
            "campaign_revision",
            "dataset_revision",
            "frozen_head_sha256",
            "trainable_state_sha256",
            "trainable_wire_sha256",
            "trainable_wire_bytes",
            "input_wire_sha256",
            "input_wire_bytes",
            "routes_sha256",
            "total_tokens",
        }
    )
    if frozenset(control) != expected:
        raise ValueError("trajectory assignment schema is invalid")
    expert_index = control["expert_index"]
    routes_sha256 = control["routes_sha256"]
    if (
        control["format"] != _RUN_FORMAT
        or control["op"] != "run"
        or control["worker_kind"] != worker_kind
        or type(control["result_index"]) is not int
        or int(control["result_index"]) < 0
        or type(control["step"]) is not int
        or int(control["step"]) < 0
        or type(control["cursor"]) is not int
        or int(control["cursor"]) < 0
        or control["campaign_revision"] != campaign_revision_value
        or control["dataset_revision"] != dataset_revision
        or control["frozen_head_sha256"] != frozen_head_sha256
        or not isinstance(control["trainable_state_sha256"], str)
        or not isinstance(control["trainable_wire_sha256"], str)
        or type(control["trainable_wire_bytes"]) is not int
        or int(control["trainable_wire_bytes"]) <= 0
        or not isinstance(control["input_wire_sha256"], str)
        or type(control["input_wire_bytes"]) is not int
        or int(control["input_wire_bytes"]) <= 0
        or type(control["total_tokens"]) is not int
        or int(control["total_tokens"]) <= 0
    ):
        raise ValueError("trajectory assignment identity is invalid")
    if worker_kind == "full":
        if expert_index is not None or routes_sha256 is not None:
            raise ValueError("full trajectory assignment expert fields are invalid")
    elif (
        type(expert_index) is not int
        or not 0 <= int(expert_index) < expert_count
        or not isinstance(routes_sha256, str)
    ):
        raise ValueError("pooled-expert assignment identity is invalid")


def _validate_pooled_expert_input(
    payload: Mapping[str, Tensor],
    *,
    config: ModelConfig,
    expert_index: int,
    expert_count: int,
    total_tokens: int,
    routes_sha256: str,
) -> tuple[Tensor, Tensor]:
    if frozenset(payload) != frozenset(
        {"hidden", "targets", "positions", "routes"}
    ):
        raise ValueError("pooled-expert input schema is invalid")
    hidden = payload["hidden"]
    targets = payload["targets"]
    positions = payload["positions"]
    routes = payload["routes"]
    if (
        hidden.dtype != torch.float32
        or hidden.ndim != 2
        or hidden.shape[1] != config.width
        or not bool(torch.isfinite(hidden).all())
    ):
        raise ValueError("pooled-expert hidden tensor is invalid")
    if (
        targets.dtype != torch.int64
        or targets.ndim != 1
        or targets.shape[0] != hidden.shape[0]
        or int(targets.min()) < 0
        or int(targets.max()) >= config.vocabulary_size
    ):
        raise ValueError("pooled-expert target tensor is invalid")
    if (
        routes.dtype != torch.int64
        or routes.ndim != 1
        or routes.numel() != total_tokens
        or int(routes.min()) < 0
        or int(routes.max()) >= expert_count
        or tensor_sha256({"routes": routes}) != routes_sha256
    ):
        raise ValueError("pooled-expert routes are invalid")
    expected_positions = torch.nonzero(
        routes == expert_index,
        as_tuple=False,
    ).flatten()
    if (
        positions.dtype != torch.int64
        or positions.ndim != 1
        or not torch.equal(positions, expected_positions)
        or positions.shape[0] != hidden.shape[0]
    ):
        raise ValueError("pooled-expert selected positions are invalid")
    return hidden.detach().clone(), targets.detach().clone()


def _trajectory_worker_entry(connection: Connection) -> None:
    try:
        init_raw = connection.recv_bytes()
        init = _load_json_bytes(init_raw, label="trajectory worker init")
        _validate_init_control(init)
        worker_kind = str(init["worker_kind"])
        expert_count = int(init["expert_count"])
        config = ModelConfig(**init["model"])
        objective = ObjectiveConfig(**init["objective"])
        configure_determinism(int(init["seed"]))
        if worker_kind == "full":
            model: SparseExpertDecoder | _FrozenHeadExpertProcessModule = (
                SparseExpertDecoder(config, objective, expert_count)
            )
            _freeze_head(model)
        else:
            model = _FrozenHeadExpertProcessModule(config, objective)

        _send_json(
            connection,
            {"status": "ready_for_head", "worker_kind": worker_kind},
        )
        head_wire = connection.recv_bytes()
        _validate_wire_identity(
            head_wire,
            expected_sha256=init["frozen_head_wire_sha256"],
            expected_bytes=init["frozen_head_wire_bytes"],
            label="trajectory frozen head",
        )
        _load_head_state(model, _deserialize_tensors(head_wire))
        head_sha256 = tensor_sha256(_head_state(model))
        if head_sha256 != init["frozen_head_sha256"]:
            raise ValueError("trajectory frozen-head identity mismatch")
        model.train()
        forward_calls = 0
        _send_json(
            connection,
            {
                "status": "ready",
                "worker_kind": worker_kind,
                "frozen_head_sha256": head_sha256,
            },
        )

        while True:
            control_raw = connection.recv_bytes()
            control = _load_json_bytes(
                control_raw,
                label="trajectory worker control",
            )
            if control.get("op") == "shutdown":
                if control != {"op": "shutdown"}:
                    raise ValueError("trajectory shutdown schema is invalid")
                _send_json(
                    connection,
                    {
                        "status": "stopped",
                        "worker_kind": worker_kind,
                        "frozen_head_sha256": tensor_sha256(_head_state(model)),
                        "forward_calls": forward_calls,
                    },
                )
                return

            _validate_run_control(
                control,
                worker_kind=worker_kind,
                expert_count=expert_count,
                campaign_revision_value=str(init["campaign_revision"]),
                dataset_revision=str(init["dataset_revision"]),
                frozen_head_sha256=head_sha256,
            )
            trainable_wire = connection.recv_bytes()
            input_wire = connection.recv_bytes()
            _validate_wire_identity(
                trainable_wire,
                expected_sha256=control["trainable_wire_sha256"],
                expected_bytes=control["trainable_wire_bytes"],
                label="trajectory trainable state",
            )
            _validate_wire_identity(
                input_wire,
                expected_sha256=control["input_wire_sha256"],
                expected_bytes=control["input_wire_bytes"],
                label="trajectory assignment input",
            )
            _load_trainable_state(
                model,
                _deserialize_tensors(trainable_wire),
                expected_sha256=control["trainable_state_sha256"],
            )
            input_payload = _deserialize_tensors(input_wire)
            total_tokens = int(control["total_tokens"])

            started = time.perf_counter()
            if worker_kind == "full":
                if not isinstance(model, SparseExpertDecoder):
                    raise AssertionError("full trajectory worker type is invalid")
                if frozenset(input_payload) != frozenset(
                    {"inputs", "targets"}
                ):
                    raise ValueError("full trajectory input schema is invalid")
                inputs = input_payload["inputs"]
                targets = input_payload["targets"]
                if (
                    inputs.dtype != torch.int64
                    or targets.dtype != torch.int64
                    or inputs.shape != targets.shape
                    or inputs.ndim != 2
                    or inputs.shape[1] != config.context_length
                    or int(inputs.min()) < 0
                    or int(inputs.max()) >= config.vocabulary_size
                    or int(targets.min()) < 0
                    or int(targets.max()) >= config.vocabulary_size
                    or targets.numel() != total_tokens
                ):
                    raise ValueError("full trajectory input tensors are invalid")
                (
                    result,
                    loss,
                    routes,
                    counts,
                    unconstrained,
                    rerouted,
                    capacity,
                ) = _full_worker_result(
                    model,
                    inputs,
                    targets,
                    expert_count=expert_count,
                    router_aux_weight=float(init["router_aux_weight"]),
                )
                routes_sha256 = tensor_sha256({"routes": routes})
                routing: dict[str, object] = {
                    "routing_counts": list(counts),
                    "unconstrained_routing_counts": list(unconstrained),
                    "capacity_rerouted_tokens": rerouted,
                    "routing_capacity": capacity,
                }
            else:
                if not isinstance(model, _FrozenHeadExpertProcessModule):
                    raise AssertionError(
                        "pooled-expert trajectory worker type is invalid"
                    )
                expert_index = int(control["expert_index"])
                routes_sha256 = str(control["routes_sha256"])
                hidden, targets = _validate_pooled_expert_input(
                    input_payload,
                    config=config,
                    expert_index=expert_index,
                    expert_count=expert_count,
                    total_tokens=total_tokens,
                    routes_sha256=routes_sha256,
                )
                result, loss = _expert_worker_result(
                    model,
                    hidden,
                    targets,
                    total_tokens=total_tokens,
                )
                routing = {"selected_token_count": int(hidden.shape[0])}
            compute_seconds = time.perf_counter() - started
            forward_calls += 1
            if tensor_sha256(_head_state(model)) != head_sha256:
                raise ValueError("trajectory worker changed the frozen head")
            result_wire = _serialize_tensors(result)
            _send_json(
                connection,
                {
                    "status": "completed",
                    "worker_kind": worker_kind,
                    "result_index": control["result_index"],
                    "expert_index": control["expert_index"],
                    "step": control["step"],
                    "cursor": control["cursor"],
                    "campaign_revision": control["campaign_revision"],
                    "dataset_revision": control["dataset_revision"],
                    "frozen_head_sha256": head_sha256,
                    "trainable_state_sha256": control[
                        "trainable_state_sha256"
                    ],
                    "input_wire_sha256": control["input_wire_sha256"],
                    "routes_sha256": routes_sha256,
                    "loss": float(loss.detach()),
                    "compute_seconds": compute_seconds,
                    "routing": routing,
                    "result_wire_sha256": _sha256_bytes(result_wire),
                    "result_wire_bytes": len(result_wire),
                },
            )
            connection.send_bytes(result_wire)
    except BaseException as exc:
        try:
            _send_json(
                connection,
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


def _start_worker(
    context: multiprocessing.context.BaseContext,
    campaign: CampaignConfig,
    *,
    worker_kind: str,
    generation: int,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_wire: bytes,
    head_sha256: str,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> _WorkerHandle:
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_trajectory_worker_entry,
        args=(child_connection,),
        name=f"orcacolony-{worker_kind}-trajectory-{generation}",
        daemon=False,
    )
    started = time.perf_counter()
    process.start()
    child_connection.close()
    sampler = _ExternalRssSampler(process, sample_interval_seconds)
    worker = _WorkerHandle(
        process=process,
        connection=parent_connection,
        worker_kind=worker_kind,
        generation=generation,
        initialization_seconds=0.0,
        control_json_wire_bytes=0,
        frozen_head_sha256=head_sha256,
        sampler=sampler,
    )
    try:
        sampler.sample("startup")
        worker.control_json_wire_bytes += _send_json(
            parent_connection,
            {
                "format": _INIT_FORMAT,
                "op": "init",
                "worker_kind": worker_kind,
                "expert_count": expert_count,
                "model": asdict(campaign.model),
                "objective": asdict(campaign.objective),
                "seed": campaign.training.seed,
                "router_aux_weight": router_aux_weight,
                "campaign_id": str(campaign.campaign["id"]),
                "campaign_revision": campaign_revision_value,
                "dataset_revision": dataset_revision,
                "frozen_head_sha256": head_sha256,
                "frozen_head_wire_sha256": _sha256_bytes(head_wire),
                "frozen_head_wire_bytes": len(head_wire),
            },
        )
        readiness, readiness_bytes = _recv_sampled_json(
            worker,
            timeout_seconds,
            phase="startup",
            label="trajectory head readiness",
        )
        worker.control_json_wire_bytes += readiness_bytes
        if readiness != {
            "status": "ready_for_head",
            "worker_kind": worker_kind,
        }:
            raise ValueError("trajectory head readiness is invalid")
        parent_connection.send_bytes(head_wire)
        ready, ready_bytes = _recv_sampled_json(
            worker,
            timeout_seconds,
            phase="startup",
            label="trajectory worker initialization",
        )
        worker.control_json_wire_bytes += ready_bytes
        if ready != {
            "status": "ready",
            "worker_kind": worker_kind,
            "frozen_head_sha256": head_sha256,
        }:
            raise ValueError("trajectory worker initialization is invalid")
        worker.initialization_seconds = time.perf_counter() - started
        return worker
    except BaseException:
        parent_connection.close()
        process.terminate()
        process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
        raise


def _assignment_control(
    worker: _WorkerHandle,
    prepared: _PreparedTopologyStep,
    *,
    result_index: int,
    expert_index: int | None,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
) -> dict[str, object]:
    trainable_state = prepared.trainable_states[result_index]
    trainable_wire = prepared.trainable_wires[result_index]
    input_wire = prepared.input_wires[result_index]
    return {
        "format": _RUN_FORMAT,
        "op": "run",
        "worker_kind": worker.worker_kind,
        "result_index": result_index,
        "expert_index": expert_index,
        "step": prepared.step,
        "cursor": prepared.cursor,
        "campaign_revision": campaign_revision_value,
        "dataset_revision": dataset_revision,
        "frozen_head_sha256": head_sha256,
        "trainable_state_sha256": tensor_sha256(trainable_state),
        "trainable_wire_sha256": _sha256_bytes(trainable_wire),
        "trainable_wire_bytes": len(trainable_wire),
        "input_wire_sha256": _sha256_bytes(input_wire),
        "input_wire_bytes": len(input_wire),
        "routes_sha256": prepared.routes_sha256,
        "total_tokens": int(prepared.targets.numel()),
    }


def _validate_completion(
    acknowledgement: Mapping[str, object],
    result_wire: bytes,
    *,
    worker: _WorkerHandle,
    prepared: _PreparedTopologyStep,
    result_index: int,
    expert_index: int | None,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
) -> None:
    expected = frozenset(
        {
            "status",
            "worker_kind",
            "result_index",
            "expert_index",
            "step",
            "cursor",
            "campaign_revision",
            "dataset_revision",
            "frozen_head_sha256",
            "trainable_state_sha256",
            "input_wire_sha256",
            "routes_sha256",
            "loss",
            "compute_seconds",
            "routing",
            "result_wire_sha256",
            "result_wire_bytes",
        }
    )
    preparation_state = prepared.trainable_states[result_index]
    input_wire = prepared.input_wires[result_index]
    if (
        frozenset(acknowledgement) != expected
        or acknowledgement["status"] != "completed"
        or acknowledgement["worker_kind"] != worker.worker_kind
        or acknowledgement["result_index"] != result_index
        or acknowledgement["expert_index"] != expert_index
        or acknowledgement["step"] != prepared.step
        or acknowledgement["cursor"] != prepared.cursor
        or acknowledgement["campaign_revision"] != campaign_revision_value
        or acknowledgement["dataset_revision"] != dataset_revision
        or acknowledgement["frozen_head_sha256"] != head_sha256
        or acknowledgement["trainable_state_sha256"]
        != tensor_sha256(preparation_state)
        or acknowledgement["input_wire_sha256"] != _sha256_bytes(input_wire)
        or type(acknowledgement["loss"]) is not float
        or not math.isfinite(float(acknowledgement["loss"]))
        or type(acknowledgement["compute_seconds"]) is not float
        or float(acknowledgement["compute_seconds"]) <= 0.0
        or not isinstance(acknowledgement["routing"], dict)
        or acknowledgement["result_wire_sha256"] != _sha256_bytes(result_wire)
        or acknowledgement["result_wire_bytes"] != len(result_wire)
    ):
        raise ValueError("trajectory worker completion is invalid")
    routing = acknowledgement["routing"]
    if worker.worker_kind == "full":
        if (
            frozenset(routing)
            != frozenset(
                {
                    "routing_counts",
                    "unconstrained_routing_counts",
                    "capacity_rerouted_tokens",
                    "routing_capacity",
                }
            )
            or not isinstance(acknowledgement["routes_sha256"], str)
        ):
            raise ValueError("full trajectory routing result is invalid")
    else:
        if (
            acknowledgement["routes_sha256"] != prepared.routes_sha256
            or frozenset(routing) != frozenset({"selected_token_count"})
            or type(routing["selected_token_count"]) is not int
            or expert_index is None
            or prepared.routing_counts is None
            or routing["selected_token_count"]
            != prepared.routing_counts[expert_index]
        ):
            raise ValueError("pooled-expert routing result is invalid")


def _run_worker_assignment(
    worker: _WorkerHandle,
    prepared: _PreparedTopologyStep,
    *,
    result_index: int,
    expert_index: int | None,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
    timeout_seconds: float,
) -> _CollectedResult:
    control = _assignment_control(
        worker,
        prepared,
        result_index=result_index,
        expert_index=expert_index,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_sha256=head_sha256,
    )
    started = time.perf_counter()
    control_bytes = _send_json(worker.connection, control)
    worker.connection.send_bytes(prepared.trainable_wires[result_index])
    worker.connection.send_bytes(prepared.input_wires[result_index])
    acknowledgement, acknowledgement_bytes = _recv_sampled_json(
        worker,
        timeout_seconds,
        phase="assignment",
        label="trajectory worker completion",
    )
    control_bytes += acknowledgement_bytes
    result_wire = _recv_sampled_bytes(
        worker,
        timeout_seconds,
        phase="assignment",
        label="trajectory worker result",
    )
    elapsed = time.perf_counter() - started
    _validate_completion(
        acknowledgement,
        result_wire,
        worker=worker,
        prepared=prepared,
        result_index=result_index,
        expert_index=expert_index,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_sha256=head_sha256,
    )
    worker.control_json_wire_bytes += control_bytes
    worker.trainable_state_transmissions += 1
    worker.forward_calls += 1
    return _CollectedResult(
        acknowledgement=acknowledgement,
        result_wire=result_wire,
        control_json_wire_bytes=control_bytes,
        round_trip_seconds=elapsed,
    )


def _worker_evidence(
    worker: _WorkerHandle,
    *,
    child_exit_code: int,
    exit_reason: str,
    shutdown_seconds: float,
) -> TrajectoryWorkerEvidence:
    return TrajectoryWorkerEvidence(
        worker_kind=worker.worker_kind,
        generation=worker.generation,
        initialization_seconds=worker.initialization_seconds,
        shutdown_seconds=shutdown_seconds,
        frozen_head_transmissions=1,
        trainable_state_transmissions=worker.trainable_state_transmissions,
        forward_calls=worker.forward_calls,
        control_json_wire_bytes=worker.control_json_wire_bytes,
        child_exit_code=child_exit_code,
        exit_reason=exit_reason,
        frozen_head_sha256=worker.frozen_head_sha256,
        external_rss=worker.sampler.evidence(),
    )


def _stop_worker(
    worker: _WorkerHandle,
    timeout_seconds: float,
) -> TrajectoryWorkerEvidence:
    started = time.perf_counter()
    acknowledgement: Mapping[str, object] | None = None
    try:
        worker.sampler.sample("shutdown")
        worker.control_json_wire_bytes += _send_json(
            worker.connection,
            {"op": "shutdown"},
        )
        acknowledgement, acknowledgement_bytes = _recv_sampled_json(
            worker,
            timeout_seconds,
            phase="shutdown",
            label="trajectory worker shutdown",
        )
        worker.control_json_wire_bytes += acknowledgement_bytes
    finally:
        worker.sampler.sample("shutdown")
        worker.connection.close()
        worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout_seconds)
    shutdown_seconds = time.perf_counter() - started
    expected = {
        "status": "stopped",
        "worker_kind": worker.worker_kind,
        "frozen_head_sha256": worker.frozen_head_sha256,
        "forward_calls": worker.forward_calls,
    }
    if worker.process.exitcode != 0 or acknowledgement != expected:
        raise ValueError("trajectory worker shutdown acknowledgement is invalid")
    return _worker_evidence(
        worker,
        child_exit_code=int(worker.process.exitcode),
        exit_reason="clean-shutdown",
        shutdown_seconds=shutdown_seconds,
    )


def _terminate_worker(
    worker: _WorkerHandle,
    timeout_seconds: float,
) -> TrajectoryWorkerEvidence:
    started = time.perf_counter()
    worker.sampler.sample("shutdown")
    worker.connection.close()
    worker.process.terminate()
    worker.process.join(timeout_seconds)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join(timeout_seconds)
    if worker.process.is_alive() or worker.process.exitcode is None:
        raise RuntimeError("failed to terminate trajectory worker")
    return _worker_evidence(
        worker,
        child_exit_code=int(worker.process.exitcode),
        exit_reason="deliberate-loss-after-persisted-result",
        shutdown_seconds=time.perf_counter() - started,
    )


def _optimizer_persistence_tensors(
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    *,
    step: int,
) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not parameter.requires_grad:
            if state:
                raise ValueError(
                    f"frozen parameter acquired optimizer state: {name}"
                )
            continue
        if step == 0 and not state:
            state_tensors = {
                "step": torch.tensor(0.0, dtype=torch.float32),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
        else:
            if not isinstance(state, dict) or frozenset(state) != frozenset(
                {"step", "exp_avg", "exp_avg_sq"}
            ):
                raise ValueError(
                    f"optimizer state schema is invalid: {name}"
                )
            state_tensors = state
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            tensor = state_tensors[state_name]
            if not isinstance(tensor, Tensor):
                raise ValueError(
                    f"optimizer state is not a tensor: {state_name}.{name}"
                )
            if state_name == "step":
                if (
                    tensor.numel() != 1
                    or not bool(torch.isfinite(tensor).all())
                    or float(tensor.item()) != float(step)
                ):
                    raise ValueError(
                        f"optimizer step is invalid: {name}"
                    )
            elif (
                tensor.dtype != parameter.dtype
                or tensor.shape != parameter.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    f"optimizer moment is invalid: {state_name}.{name}"
                )
            tensors[f"{state_name}.{name}"] = tensor.detach().clone()
    return tensors


def _checkpoint_wires(
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    *,
    step: int,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    bytes,
    bytes,
]:
    model_tensors = _model_snapshot(model)
    optimizer_tensors = _optimizer_persistence_tensors(
        model,
        optimizer,
        step=step,
    )
    return (
        model_tensors,
        optimizer_tensors,
        _serialize_tensors(dict(model_tensors)),
        _serialize_tensors(optimizer_tensors),
    )


def _load_checkpoint_wires(
    campaign: CampaignConfig,
    *,
    expert_count: int,
    step: int,
    model_wire: bytes,
    optimizer_wire: bytes,
    expected_head_sha256: str,
) -> tuple[SparseExpertDecoder, torch.optim.AdamW]:
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    expected_model = model.state_dict()
    model_tensors = _deserialize_tensors(model_wire)
    if frozenset(model_tensors) != frozenset(expected_model):
        raise ValueError("persisted trajectory model schema is invalid")
    for name, expected in expected_model.items():
        tensor = model_tensors[name]
        if (
            tensor.dtype != expected.dtype
            or tensor.shape != expected.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"persisted trajectory model is invalid: {name}")
    model.load_state_dict(model_tensors, strict=True)
    if tensor_sha256(_head_tensor_snapshot(model)) != expected_head_sha256:
        raise ValueError("persisted trajectory frozen head changed")

    optimizer = _create_optimizer(model, campaign.training)
    optimizer_tensors = _deserialize_tensors(optimizer_wire)
    expected_optimizer_names = {
        f"{state_name}.{name}"
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        for state_name in ("step", "exp_avg", "exp_avg_sq")
    }
    if frozenset(optimizer_tensors) != frozenset(expected_optimizer_names):
        raise ValueError("persisted trajectory optimizer schema is invalid")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        step_tensor = optimizer_tensors[f"step.{name}"]
        exp_avg = optimizer_tensors[f"exp_avg.{name}"]
        exp_avg_sq = optimizer_tensors[f"exp_avg_sq.{name}"]
        if (
            step_tensor.numel() != 1
            or not bool(torch.isfinite(step_tensor).all())
            or float(step_tensor.item()) != float(step)
            or exp_avg.dtype != parameter.dtype
            or exp_avg.shape != parameter.shape
            or not bool(torch.isfinite(exp_avg).all())
            or exp_avg_sq.dtype != parameter.dtype
            or exp_avg_sq.shape != parameter.shape
            or not bool(torch.isfinite(exp_avg_sq).all())
        ):
            raise ValueError(
                f"persisted trajectory optimizer tensor is invalid: {name}"
            )
        optimizer.state[parameter] = {
            "step": step_tensor.detach().clone(),
            "exp_avg": exp_avg.detach().clone(),
            "exp_avg_sq": exp_avg_sq.detach().clone(),
        }
    return model, optimizer


def _prepare_full_step(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    model: SparseExpertDecoder,
    *,
    step: int,
) -> _PreparedTopologyStep:
    cursor = (
        step * campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    inputs, targets = fixture_batch(campaign, cursor, dataset)
    state = _trainable_state(model)
    input_wire = _serialize_tensors(
        {"inputs": inputs, "targets": targets}
    )
    return _PreparedTopologyStep(
        step=step,
        cursor=cursor,
        inputs=inputs.detach().clone(),
        targets=targets.detach().clone(),
        routes=None,
        routing_counts=None,
        unconstrained_routing_counts=None,
        capacity_rerouted_tokens=None,
        routing_capacity=None,
        routes_sha256=None,
        batch_wire=input_wire,
        trainable_states=(state,),
        trainable_wires=(_serialize_tensors(state),),
        input_wires=(input_wire,),
    )


def _prepare_expert_step(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    model: SparseExpertDecoder,
    *,
    step: int,
    expert_count: int,
) -> _PreparedTopologyStep:
    cursor = (
        step * campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    inputs, targets = fixture_batch(campaign, cursor, dataset)
    with torch.no_grad():
        hidden = model.shared_hidden(inputs)
        hidden_flat = hidden.reshape(-1, campaign.model.width)
        router_logits = model.router(hidden_flat)
        (
            routes,
            counts,
            unconstrained,
            rerouted,
            capacity,
        ) = _balanced_top1_routes(router_logits, expert_count)
    routes = routes.detach().clone()
    routes_sha256 = tensor_sha256({"routes": routes})
    targets_flat = targets.reshape(-1)
    states: list[dict[str, Tensor]] = []
    state_wires: list[bytes] = []
    input_wires: list[bytes] = []
    for expert_index, expert in enumerate(model.experts):
        state = {
            f"expert.{name}": parameter.detach().clone()
            for name, parameter in expert.named_parameters()
        }
        positions = torch.nonzero(
            routes == expert_index,
            as_tuple=False,
        ).flatten()
        input_wire = _serialize_tensors(
            {
                "hidden": hidden_flat[positions].detach().clone(),
                "targets": targets_flat[positions].detach().clone(),
                "positions": positions.detach().clone(),
                "routes": routes,
            }
        )
        states.append(state)
        state_wires.append(_serialize_tensors(state))
        input_wires.append(input_wire)
    batch_wire = _serialize_tensors(
        {
            "inputs": inputs,
            "targets": targets,
            "routes": routes,
        }
    )
    return _PreparedTopologyStep(
        step=step,
        cursor=cursor,
        inputs=inputs.detach().clone(),
        targets=targets.detach().clone(),
        routes=routes,
        routing_counts=counts,
        unconstrained_routing_counts=unconstrained,
        capacity_rerouted_tokens=rerouted,
        routing_capacity=capacity,
        routes_sha256=routes_sha256,
        batch_wire=batch_wire,
        trainable_states=tuple(states),
        trainable_wires=tuple(state_wires),
        input_wires=tuple(input_wires),
    )


def _result_preparations(
    prepared: _PreparedTopologyStep,
    *,
    topology: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for result_index, (
        state,
        state_wire,
        input_wire,
    ) in enumerate(
        zip(
            prepared.trainable_states,
            prepared.trainable_wires,
            prepared.input_wires,
            strict=True,
        )
    ):
        records.append(
            {
                "result_index": result_index,
                "expert_index": (
                    result_index if topology == "expert" else None
                ),
                "trainable_state_sha256": tensor_sha256(state),
                "trainable_wire_sha256": _sha256_bytes(state_wire),
                "trainable_wire_bytes": len(state_wire),
                "input_wire_sha256": _sha256_bytes(input_wire),
                "input_wire_bytes": len(input_wire),
            }
        )
    return records


def _transaction_identity(
    campaign: CampaignConfig,
    *,
    campaign_revision_value: str,
    dataset_revision: str,
    topology: str,
    expert_count: int,
    router_aux_weight: float,
    head_sha256: str,
    prepared: _PreparedTopologyStep,
    model_tensors: Mapping[str, Tensor],
    optimizer_tensors: Mapping[str, Tensor],
    model_wire: bytes,
    optimizer_wire: bytes,
) -> dict[str, object]:
    return {
        "campaign_id": str(campaign.campaign["id"]),
        "campaign_revision": campaign_revision_value,
        "dataset_revision": dataset_revision,
        "topology": topology,
        "step": prepared.step,
        "cursor": prepared.cursor,
        "expert_count": expert_count,
        "router_aux_weight": router_aux_weight,
        "frozen_head_sha256": head_sha256,
        "pre_model_sha256": tensor_sha256(model_tensors),
        "pre_model_wire_sha256": _sha256_bytes(model_wire),
        "pre_model_wire_bytes": len(model_wire),
        "pre_optimizer_sha256": tensor_sha256(optimizer_tensors),
        "pre_optimizer_wire_sha256": _sha256_bytes(optimizer_wire),
        "pre_optimizer_wire_bytes": len(optimizer_wire),
        "batch_wire_sha256": _sha256_bytes(prepared.batch_wire),
        "batch_wire_bytes": len(prepared.batch_wire),
        "routes_sha256": prepared.routes_sha256,
        "expected_result_count": len(prepared.input_wires),
        "result_preparations": _result_preparations(
            prepared,
            topology=topology,
        ),
    }


def _validate_identity(identity: object) -> dict[str, object]:
    if not isinstance(identity, dict) or frozenset(identity) != _IDENTITY_FIELDS:
        raise ValueError("trajectory transaction identity schema is invalid")
    if (
        not isinstance(identity["campaign_id"], str)
        or not isinstance(identity["campaign_revision"], str)
        or not isinstance(identity["dataset_revision"], str)
        or identity["topology"] not in {"full", "expert"}
        or type(identity["step"]) is not int
        or int(identity["step"]) < 0
        or type(identity["cursor"]) is not int
        or int(identity["cursor"]) < 0
        or type(identity["expert_count"]) is not int
        or not 2 <= int(identity["expert_count"]) <= 16
        or type(identity["router_aux_weight"]) is not float
        or not math.isfinite(float(identity["router_aux_weight"]))
        or not isinstance(identity["frozen_head_sha256"], str)
        or not isinstance(identity["pre_model_sha256"], str)
        or not isinstance(identity["pre_model_wire_sha256"], str)
        or type(identity["pre_model_wire_bytes"]) is not int
        or int(identity["pre_model_wire_bytes"]) <= 0
        or not isinstance(identity["pre_optimizer_sha256"], str)
        or not isinstance(identity["pre_optimizer_wire_sha256"], str)
        or type(identity["pre_optimizer_wire_bytes"]) is not int
        or int(identity["pre_optimizer_wire_bytes"]) <= 0
        or not isinstance(identity["batch_wire_sha256"], str)
        or type(identity["batch_wire_bytes"]) is not int
        or int(identity["batch_wire_bytes"]) <= 0
        or type(identity["expected_result_count"]) is not int
        or int(identity["expected_result_count"]) <= 0
    ):
        raise ValueError("trajectory transaction identity is invalid")
    if identity["topology"] == "full":
        if (
            identity["routes_sha256"] is not None
            or identity["expected_result_count"] != 1
        ):
            raise ValueError("full transaction routing identity is invalid")
    elif not isinstance(identity["routes_sha256"], str):
        raise ValueError("expert transaction routing identity is invalid")
    preparations = identity["result_preparations"]
    if (
        not isinstance(preparations, list)
        or len(preparations) != identity["expected_result_count"]
    ):
        raise ValueError("trajectory result preparations are invalid")
    for index, preparation in enumerate(preparations):
        if (
            not isinstance(preparation, dict)
            or frozenset(preparation) != _RESULT_PREPARATION_FIELDS
            or preparation["result_index"] != index
            or preparation["expert_index"]
            != (index if identity["topology"] == "expert" else None)
            or not isinstance(preparation["trainable_state_sha256"], str)
            or not isinstance(preparation["trainable_wire_sha256"], str)
            or type(preparation["trainable_wire_bytes"]) is not int
            or int(preparation["trainable_wire_bytes"]) <= 0
            or not isinstance(preparation["input_wire_sha256"], str)
            or type(preparation["input_wire_bytes"]) is not int
            or int(preparation["input_wire_bytes"]) <= 0
        ):
            raise ValueError(
                f"trajectory result preparation is invalid: {index}"
            )
    return identity


def _create_transaction(
    transaction_dir: Path,
    identity: dict[str, object],
    *,
    model_wire: bytes,
    optimizer_wire: bytes,
    batch_wire: bytes,
) -> str:
    _validate_identity(identity)
    if transaction_dir.exists():
        if (
            transaction_dir.is_symlink()
            or not transaction_dir.is_dir()
            or any(transaction_dir.iterdir())
        ):
            raise ValueError("trajectory transaction directory must be new or empty")
    else:
        transaction_dir.mkdir(parents=True, exist_ok=False)
    if transaction_dir.is_symlink():
        raise ValueError("trajectory transaction directory may not be a symlink")
    results_dir = transaction_dir / "results"
    results_dir.mkdir(exist_ok=False)
    _write_bytes_atomic(
        transaction_dir / "pre-model.safetensors",
        model_wire,
    )
    _write_bytes_atomic(
        transaction_dir / "pre-optimizer.safetensors",
        optimizer_wire,
    )
    _write_bytes_atomic(
        transaction_dir / "batch.safetensors",
        batch_wire,
    )
    transaction_id = _sha256_bytes(_canonical_json(identity))
    manifest = {
        "format": _TRANSACTION_FORMAT,
        "identity": identity,
        "transaction_id": transaction_id,
        "phase": "prepared",
        "phase_history": ["prepared"],
        "accepted_result_count": 0,
        "accepted_results": [],
        "result_applied": False,
        "applied_checkpoint": None,
    }
    _write_bytes_atomic(
        transaction_dir / "manifest.json",
        _canonical_json(manifest),
    )
    return transaction_id


def _load_manifest(transaction_dir: Path) -> dict[str, object]:
    manifest_path = transaction_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("trajectory transaction manifest is unavailable")
    manifest = _load_json_bytes(
        manifest_path.read_bytes(),
        label="trajectory transaction manifest",
    )
    if frozenset(manifest) != _MANIFEST_FIELDS:
        raise ValueError("trajectory transaction manifest schema is invalid")
    if manifest["format"] != _TRANSACTION_FORMAT:
        raise ValueError("trajectory transaction manifest format is invalid")
    identity = _validate_identity(manifest["identity"])
    transaction_id = _sha256_bytes(_canonical_json(identity))
    if manifest["transaction_id"] != transaction_id:
        raise ValueError("trajectory transaction id is invalid")
    phase = manifest["phase"]
    history = manifest["phase_history"]
    if (
        phase not in _PHASES
        or not isinstance(history, list)
        or not history
        or tuple(history) != _PHASES[: len(history)]
        or history[-1] != phase
    ):
        raise ValueError("trajectory transaction phase history is invalid")
    accepted = manifest["accepted_results"]
    if (
        type(manifest["accepted_result_count"]) is not int
        or not isinstance(accepted, list)
        or manifest["accepted_result_count"] != len(accepted)
        or len(accepted) > identity["expected_result_count"]
    ):
        raise ValueError("trajectory accepted-result count is invalid")
    for index, record in enumerate(accepted):
        if (
            not isinstance(record, dict)
            or frozenset(record) != _RESULT_RECORD_FIELDS
            or record["result_index"] != index
        ):
            raise ValueError("trajectory accepted-result manifest is invalid")
    applied_checkpoint = manifest["applied_checkpoint"]
    if (
        type(manifest["result_applied"]) is not bool
        or (phase == "applied") != manifest["result_applied"]
        or (phase == "applied") != isinstance(applied_checkpoint, dict)
    ):
        raise ValueError("trajectory applied-result state is invalid")
    if phase == "applied" and (
        frozenset(applied_checkpoint)
        != frozenset(
            {
                "directory",
                "state_sha256",
                "model_sha256",
                "optimizer_sha256",
            }
        )
        or applied_checkpoint["directory"] != "applied"
        or not isinstance(applied_checkpoint["state_sha256"], str)
        or not isinstance(applied_checkpoint["model_sha256"], str)
        or not isinstance(applied_checkpoint["optimizer_sha256"], str)
    ):
        raise ValueError("trajectory applied-checkpoint manifest is invalid")
    return manifest


def _write_manifest(
    transaction_dir: Path,
    manifest: dict[str, object],
) -> None:
    _write_bytes_atomic(
        transaction_dir / "manifest.json",
        _canonical_json(manifest),
    )


def _validate_transaction_root(
    transaction_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise ValueError("trajectory transaction directory is invalid")
    expected = {
        "manifest.json",
        "pre-model.safetensors",
        "pre-optimizer.safetensors",
        "batch.safetensors",
        "results",
    }
    if (transaction_dir / "applied").exists():
        expected.add("applied")
    entries = tuple(transaction_dir.iterdir())
    if {entry.name for entry in entries} != expected:
        raise ValueError("trajectory transaction contains unexpected entries")
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("trajectory transaction contains a symlink")
        if entry.name in {"results", "applied"}:
            if not entry.is_dir():
                raise ValueError("trajectory transaction directory entry is invalid")
        elif not entry.is_file():
            raise ValueError("trajectory transaction file entry is invalid")
    identity = _validate_identity(manifest["identity"])
    _read_owned_file(
        transaction_dir / "pre-model.safetensors",
        expected_bytes=int(identity["pre_model_wire_bytes"]),
        expected_sha256=str(identity["pre_model_wire_sha256"]),
    )
    _read_owned_file(
        transaction_dir / "pre-optimizer.safetensors",
        expected_bytes=int(identity["pre_optimizer_wire_bytes"]),
        expected_sha256=str(identity["pre_optimizer_wire_sha256"]),
    )
    _read_owned_file(
        transaction_dir / "batch.safetensors",
        expected_bytes=int(identity["batch_wire_bytes"]),
        expected_sha256=str(identity["batch_wire_sha256"]),
    )


def _acceptance_payload(
    result: _CollectedResult,
) -> dict[str, object]:
    acknowledgement = result.acknowledgement
    payload = {
        "format": _RUN_FORMAT,
        "worker_kind": acknowledgement["worker_kind"],
        "result_index": acknowledgement["result_index"],
        "expert_index": acknowledgement["expert_index"],
        "step": acknowledgement["step"],
        "cursor": acknowledgement["cursor"],
        "campaign_revision": acknowledgement["campaign_revision"],
        "dataset_revision": acknowledgement["dataset_revision"],
        "frozen_head_sha256": acknowledgement["frozen_head_sha256"],
        "trainable_state_sha256": acknowledgement[
            "trainable_state_sha256"
        ],
        "input_wire_sha256": acknowledgement["input_wire_sha256"],
        "routes_sha256": acknowledgement["routes_sha256"],
        "loss": acknowledgement["loss"],
        "routing": acknowledgement["routing"],
        "result_wire_sha256": _sha256_bytes(result.result_wire),
        "result_wire_bytes": len(result.result_wire),
    }
    if frozenset(payload) != _ACCEPTANCE_FIELDS:
        raise AssertionError("trajectory acceptance payload is incomplete")
    return payload


def _validate_result_directory(
    transaction_dir: Path,
    identity: Mapping[str, object],
    *,
    result_index: int,
) -> tuple[dict[str, object], dict[str, object], bytes]:
    result_dir = transaction_dir / "results" / f"result-{result_index:03d}"
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise ValueError("trajectory accepted-result directory is invalid")
    entries = tuple(result_dir.iterdir())
    if (
        {entry.name for entry in entries}
        != {"acceptance.json", "result.safetensors"}
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("trajectory accepted-result files are invalid")
    acceptance_wire = (result_dir / "acceptance.json").read_bytes()
    acceptance = _load_json_bytes(
        acceptance_wire,
        label="trajectory result acceptance",
    )
    if frozenset(acceptance) != _ACCEPTANCE_FIELDS:
        raise ValueError("trajectory result acceptance schema is invalid")
    preparations = identity["result_preparations"]
    if not isinstance(preparations, list):
        raise AssertionError("trajectory result preparations are invalid")
    preparation = preparations[result_index]
    if not isinstance(preparation, dict):
        raise AssertionError("trajectory result preparation is invalid")
    expected_worker_kind = (
        "full" if identity["topology"] == "full" else "pooled_expert"
    )
    if (
        acceptance["format"] != _RUN_FORMAT
        or acceptance["worker_kind"] != expected_worker_kind
        or acceptance["result_index"] != result_index
        or acceptance["expert_index"] != preparation["expert_index"]
        or acceptance["step"] != identity["step"]
        or acceptance["cursor"] != identity["cursor"]
        or acceptance["campaign_revision"] != identity["campaign_revision"]
        or acceptance["dataset_revision"] != identity["dataset_revision"]
        or acceptance["frozen_head_sha256"]
        != identity["frozen_head_sha256"]
        or acceptance["trainable_state_sha256"]
        != preparation["trainable_state_sha256"]
        or acceptance["input_wire_sha256"]
        != preparation["input_wire_sha256"]
        or type(acceptance["loss"]) is not float
        or not math.isfinite(float(acceptance["loss"]))
        or not isinstance(acceptance["routing"], dict)
        or type(acceptance["result_wire_bytes"]) is not int
        or int(acceptance["result_wire_bytes"]) <= 0
        or not isinstance(acceptance["result_wire_sha256"], str)
    ):
        raise ValueError("trajectory result acceptance identity is invalid")
    if identity["topology"] == "expert":
        if acceptance["routes_sha256"] != identity["routes_sha256"]:
            raise ValueError("trajectory expert result routes changed")
    elif not isinstance(acceptance["routes_sha256"], str):
        raise ValueError("trajectory full result lacks routes identity")
    result_wire = _read_owned_file(
        result_dir / "result.safetensors",
        expected_bytes=int(acceptance["result_wire_bytes"]),
        expected_sha256=str(acceptance["result_wire_sha256"]),
    )
    record = {
        "result_index": result_index,
        "directory": result_dir.name,
        "acceptance_sha256": _sha256_bytes(acceptance_wire),
        "acceptance_bytes": len(acceptance_wire),
        "result_wire_sha256": acceptance["result_wire_sha256"],
        "result_wire_bytes": acceptance["result_wire_bytes"],
    }
    return record, acceptance, result_wire


def _reconcile_results(
    transaction_dir: Path,
    manifest: dict[str, object],
    *,
    write: bool,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[bytes],
]:
    identity = _validate_identity(manifest["identity"])
    results_dir = transaction_dir / "results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError("trajectory results directory is invalid")
    entries = sorted(results_dir.iterdir(), key=lambda path: path.name)
    if any(entry.is_symlink() or not entry.is_dir() for entry in entries):
        raise ValueError("trajectory results contain an invalid entry")
    expected_names = [
        f"result-{index:03d}" for index in range(len(entries))
    ]
    if [entry.name for entry in entries] != expected_names:
        raise ValueError("trajectory accepted results are not contiguous")
    if len(entries) > int(identity["expected_result_count"]):
        raise ValueError("trajectory has too many accepted results")
    records: list[dict[str, object]] = []
    acceptances: list[dict[str, object]] = []
    result_wires: list[bytes] = []
    for index in range(len(entries)):
        record, acceptance, result_wire = _validate_result_directory(
            transaction_dir,
            identity,
            result_index=index,
        )
        records.append(record)
        acceptances.append(acceptance)
        result_wires.append(result_wire)
    persisted = manifest["accepted_results"]
    if not isinstance(persisted, list):
        raise AssertionError("trajectory accepted-result manifest is invalid")
    if persisted != records:
        if (
            not write
            or persisted != records[: len(persisted)]
            or manifest["phase"] != "prepared"
        ):
            raise ValueError("trajectory accepted-result manifest differs")
        manifest["accepted_results"] = records
        manifest["accepted_result_count"] = len(records)
        _write_manifest(transaction_dir, manifest)
    if (
        len(records) == identity["expected_result_count"]
        and manifest["phase"] == "prepared"
    ):
        if not write:
            raise ValueError("trajectory complete results were not committed")
        manifest["phase"] = "results_accepted"
        manifest["phase_history"] = ["prepared", "results_accepted"]
        _write_manifest(transaction_dir, manifest)
    return records, acceptances, result_wires


def _persist_result(
    transaction_dir: Path,
    result: _CollectedResult,
) -> int:
    manifest = _load_manifest(transaction_dir)
    _validate_transaction_root(transaction_dir, manifest)
    records, _, _ = _reconcile_results(
        transaction_dir,
        manifest,
        write=True,
    )
    identity = _validate_identity(manifest["identity"])
    if manifest["phase"] != "prepared":
        raise ValueError("trajectory transaction no longer accepts results")
    result_index = int(result.acknowledgement["result_index"])
    if result_index != len(records):
        if result_index < len(records):
            raise ValueError("trajectory result was already accepted")
        raise ValueError("trajectory result acceptance is out of order")
    if result_index >= int(identity["expected_result_count"]):
        raise ValueError("trajectory result index exceeds the transaction")
    acceptance = _acceptance_payload(result)
    stage = (
        transaction_dir
        / "results"
        / f".result-{result_index:03d}.tmp-{os.getpid()}"
    )
    target = (
        transaction_dir
        / "results"
        / f"result-{result_index:03d}"
    )
    if stage.exists() or target.exists():
        raise ValueError("trajectory result staging target already exists")
    stage.mkdir(exist_ok=False)
    try:
        _write_new_file(
            stage / "acceptance.json",
            _canonical_json(acceptance),
        )
        _write_new_file(stage / "result.safetensors", result.result_wire)
        _fsync_directory(stage)
        os.replace(stage, target)
        _fsync_directory(target.parent)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    manifest = _load_manifest(transaction_dir)
    _reconcile_results(transaction_dir, manifest, write=True)
    return len(_canonical_json(acceptance)) + len(result.result_wire)


def _validate_full_gradient_result(
    model: SparseExpertDecoder,
    result_wire: bytes,
) -> dict[str, Tensor]:
    tensors = _deserialize_tensors(result_wire)
    expected = {
        f"gradient.{name}": parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if frozenset(tensors) != frozenset(expected):
        raise ValueError("full trajectory gradient result schema is invalid")
    gradients: dict[str, Tensor] = {}
    for result_name, parameter in expected.items():
        gradient = tensors[result_name]
        if (
            gradient.dtype != parameter.dtype
            or gradient.shape != parameter.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(
                f"full trajectory gradient is invalid: {result_name}"
            )
        gradients[result_name.removeprefix("gradient.")] = (
            gradient.detach().clone()
        )
    return gradients


def _apply_full_results(
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    result_wire: bytes,
    *,
    campaign: CampaignConfig,
) -> tuple[
    float,
    str,
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    dict[str, Tensor],
    dict[str, Tensor],
]:
    gradients = _validate_full_gradient_result(model, result_wire)
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = gradients[name]
        elif parameter.grad is not None:
            raise ValueError("frozen trajectory parameter acquired a gradient")
    raw = _trainable_gradient_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    clipped = _trainable_gradient_snapshot(model)
    return (
        0.0,
        "",
        (),
        (),
        0,
        0,
        raw,
        clipped,
    )


def _apply_expert_results(
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    *,
    campaign: CampaignConfig,
    expert_count: int,
    router_aux_weight: float,
    batch: Mapping[str, Tensor],
    acceptances: list[dict[str, object]],
    result_wires: list[bytes],
    expected_routes_sha256: str,
) -> tuple[
    float,
    str,
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    dict[str, Tensor],
    dict[str, Tensor],
]:
    if frozenset(batch) != frozenset({"inputs", "targets", "routes"}):
        raise ValueError("expert trajectory persisted batch schema is invalid")
    inputs = batch["inputs"]
    targets = batch["targets"]
    persisted_routes = batch["routes"]
    if (
        inputs.dtype != torch.int64
        or targets.dtype != torch.int64
        or inputs.shape != targets.shape
        or inputs.ndim != 2
        or inputs.shape[1] != campaign.model.context_length
        or int(inputs.min()) < 0
        or int(inputs.max()) >= campaign.model.vocabulary_size
        or int(targets.min()) < 0
        or int(targets.max()) >= campaign.model.vocabulary_size
        or persisted_routes.dtype != torch.int64
        or persisted_routes.ndim != 1
        or persisted_routes.numel() != targets.numel()
        or tensor_sha256({"routes": persisted_routes})
        != expected_routes_sha256
    ):
        raise ValueError("expert trajectory persisted batch is invalid")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = model.shared_hidden(inputs)
    hidden_flat = hidden.reshape(-1, campaign.model.width)
    router_logits = model.router(hidden_flat)
    (
        routes,
        counts,
        unconstrained,
        rerouted,
        capacity,
    ) = _balanced_top1_routes(router_logits, expert_count)
    routes_sha256 = tensor_sha256({"routes": routes})
    if (
        routes_sha256 != expected_routes_sha256
        or not torch.equal(routes, persisted_routes)
    ):
        raise ValueError("expert trajectory routes changed before apply")
    auxiliary = _router_auxiliary_loss(
        router_logits,
        routes,
        expert_count,
        router_aux_weight,
    )
    auxiliary.backward(retain_graph=True)
    process_losses: list[Tensor] = []
    for expert_index, (acceptance, result_wire) in enumerate(
        zip(acceptances, result_wires, strict=True)
    ):
        tensors = _deserialize_tensors(result_wire)
        expert = model.experts[expert_index]
        expected_gradients = {
            f"gradient.expert.{name}": parameter
            for name, parameter in expert.named_parameters()
        }
        if frozenset(tensors) != frozenset(
            {"input_adjoint", *expected_gradients}
        ):
            raise ValueError(
                f"expert trajectory result schema is invalid: {expert_index}"
            )
        positions = torch.nonzero(
            routes == expert_index,
            as_tuple=False,
        ).flatten()
        selected_hidden = hidden_flat[positions].detach().clone()
        input_adjoint = tensors["input_adjoint"]
        if (
            input_adjoint.dtype != selected_hidden.dtype
            or input_adjoint.shape != selected_hidden.shape
            or not bool(torch.isfinite(input_adjoint).all())
        ):
            raise ValueError(
                f"expert trajectory input adjoint is invalid: {expert_index}"
            )
        for result_name, parameter in expected_gradients.items():
            gradient = tensors[result_name]
            if (
                gradient.dtype != parameter.dtype
                or gradient.shape != parameter.shape
                or not bool(torch.isfinite(gradient).all())
            ):
                raise ValueError(
                    f"expert trajectory gradient is invalid: {result_name}"
                )
            parameter.grad = gradient.detach().clone()
        scattered = torch.zeros_like(hidden_flat)
        scattered[positions] = input_adjoint
        hidden_flat.backward(
            scattered,
            retain_graph=expert_index < expert_count - 1,
        )
        process_losses.append(
            torch.tensor(
                float(acceptance["loss"]),
                dtype=auxiliary.dtype,
            )
        )
    loss = auxiliary.detach() + sum(process_losses)
    raw = _trainable_gradient_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    clipped = _trainable_gradient_snapshot(model)
    return (
        float(loss),
        routes_sha256,
        counts,
        unconstrained,
        rerouted,
        capacity,
        raw,
        clipped,
    )


def _compute_transaction_candidate(
    campaign: CampaignConfig,
    transaction_dir: Path,
    *,
    allow_applied: bool = False,
) -> _AppliedCandidate:
    manifest = _load_manifest(transaction_dir)
    _validate_transaction_root(transaction_dir, manifest)
    records, acceptances, result_wires = _reconcile_results(
        transaction_dir,
        manifest,
        write=False,
    )
    identity = _validate_identity(manifest["identity"])
    allowed_phases = (
        {"results_accepted", "applied"}
        if allow_applied
        else {"results_accepted"}
    )
    if (
        manifest["phase"] not in allowed_phases
        or (
            manifest["phase"] == "results_accepted"
            and manifest["result_applied"] is not False
        )
        or (
            manifest["phase"] == "applied"
            and manifest["result_applied"] is not True
        )
        or len(records) != identity["expected_result_count"]
    ):
        raise ValueError("trajectory transaction is not ready to apply")
    model_wire = _read_owned_file(
        transaction_dir / "pre-model.safetensors",
        expected_bytes=int(identity["pre_model_wire_bytes"]),
        expected_sha256=str(identity["pre_model_wire_sha256"]),
    )
    optimizer_wire = _read_owned_file(
        transaction_dir / "pre-optimizer.safetensors",
        expected_bytes=int(identity["pre_optimizer_wire_bytes"]),
        expected_sha256=str(identity["pre_optimizer_wire_sha256"]),
    )
    batch_wire = _read_owned_file(
        transaction_dir / "batch.safetensors",
        expected_bytes=int(identity["batch_wire_bytes"]),
        expected_sha256=str(identity["batch_wire_sha256"]),
    )
    model, optimizer = _load_checkpoint_wires(
        campaign,
        expert_count=int(identity["expert_count"]),
        step=int(identity["step"]),
        model_wire=model_wire,
        optimizer_wire=optimizer_wire,
        expected_head_sha256=str(identity["frozen_head_sha256"]),
    )
    pre_model = _model_snapshot(model)
    pre_optimizer = _optimizer_persistence_tensors(
        model,
        optimizer,
        step=int(identity["step"]),
    )
    if (
        tensor_sha256(pre_model) != identity["pre_model_sha256"]
        or tensor_sha256(pre_optimizer) != identity["pre_optimizer_sha256"]
    ):
        raise ValueError("trajectory persisted pre-state identity changed")
    batch = _deserialize_tensors(batch_wire)
    started = time.perf_counter()
    if identity["topology"] == "full":
        if frozenset(batch) != frozenset({"inputs", "targets"}):
            raise ValueError("full trajectory persisted batch schema is invalid")
        if (
            batch["inputs"].dtype != torch.int64
            or batch["targets"].dtype != torch.int64
            or batch["inputs"].shape != batch["targets"].shape
            or batch["inputs"].ndim != 2
            or batch["inputs"].shape[1] != campaign.model.context_length
            or int(batch["inputs"].min()) < 0
            or int(batch["inputs"].max())
            >= campaign.model.vocabulary_size
            or int(batch["targets"].min()) < 0
            or int(batch["targets"].max())
            >= campaign.model.vocabulary_size
        ):
            raise ValueError("full trajectory persisted batch is invalid")
        (
            _,
            _,
            _,
            _,
            _,
            _,
            raw,
            clipped,
        ) = _apply_full_results(
            model,
            optimizer,
            result_wires[0],
            campaign=campaign,
        )
        acceptance = acceptances[0]
        routing = acceptance["routing"]
        if (
            not isinstance(routing, dict)
            or frozenset(routing)
            != frozenset(
                {
                    "routing_counts",
                    "unconstrained_routing_counts",
                    "capacity_rerouted_tokens",
                    "routing_capacity",
                }
            )
            or not isinstance(routing["routing_counts"], list)
            or len(routing["routing_counts"]) != identity["expert_count"]
            or any(
                type(value) is not int or value <= 0
                for value in routing["routing_counts"]
            )
            or sum(routing["routing_counts"]) != batch["targets"].numel()
            or not isinstance(
                routing["unconstrained_routing_counts"],
                list,
            )
            or len(routing["unconstrained_routing_counts"])
            != identity["expert_count"]
            or any(
                type(value) is not int or value < 0
                for value in routing["unconstrained_routing_counts"]
            )
            or sum(routing["unconstrained_routing_counts"])
            != batch["targets"].numel()
            or type(routing["capacity_rerouted_tokens"]) is not int
            or routing["capacity_rerouted_tokens"] < 0
            or type(routing["routing_capacity"]) is not int
            or routing["routing_capacity"] <= 0
        ):
            raise ValueError("full trajectory persisted routing is invalid")
        loss = float(acceptance["loss"])
        routes_sha256 = str(acceptance["routes_sha256"])
        counts = tuple(int(value) for value in routing["routing_counts"])
        unconstrained = tuple(
            int(value)
            for value in routing["unconstrained_routing_counts"]
        )
        rerouted = int(routing["capacity_rerouted_tokens"])
        capacity = int(routing["routing_capacity"])
    else:
        (
            loss,
            routes_sha256,
            counts,
            unconstrained,
            rerouted,
            capacity,
            raw,
            clipped,
        ) = _apply_expert_results(
            model,
            optimizer,
            campaign=campaign,
            expert_count=int(identity["expert_count"]),
            router_aux_weight=float(identity["router_aux_weight"]),
            batch=batch,
            acceptances=acceptances,
            result_wires=result_wires,
            expected_routes_sha256=str(identity["routes_sha256"]),
        )
    optimizer.step()
    if (
        tensor_sha256(_head_tensor_snapshot(model))
        != identity["frozen_head_sha256"]
    ):
        raise ValueError("trajectory apply changed the frozen head")
    optimizer_snapshot = _trainable_optimizer_tensor_snapshot(
        model,
        optimizer,
    )
    model_snapshot = _model_snapshot(model)
    applied_model_tensors, applied_optimizer_tensors, applied_model_wire, (
        applied_optimizer_wire
    ) = _checkpoint_wires(
        model,
        optimizer,
        step=int(identity["step"]) + 1,
    )
    snapshot = _TrajectorySnapshot(
        step=int(identity["step"]),
        cursor=int(identity["cursor"]),
        pre_model_sha256=str(identity["pre_model_sha256"]),
        pre_optimizer_sha256=str(identity["pre_optimizer_sha256"]),
        loss=loss,
        routes_sha256=routes_sha256,
        routing_counts=counts,
        unconstrained_routing_counts=unconstrained,
        capacity_rerouted_tokens=rerouted,
        routing_capacity=capacity,
        raw_gradients=raw,
        clipped_gradients=clipped,
        optimizer=optimizer_snapshot,
        model=model_snapshot,
        step_seconds=time.perf_counter() - started,
    )
    state = {
        "format": _APPLIED_FORMAT,
        "transaction_id": manifest["transaction_id"],
        "topology": identity["topology"],
        "step": int(identity["step"]) + 1,
        "model_sha256": tensor_sha256(applied_model_tensors),
        "model_wire_sha256": _sha256_bytes(applied_model_wire),
        "model_wire_bytes": len(applied_model_wire),
        "optimizer_sha256": tensor_sha256(applied_optimizer_tensors),
        "optimizer_wire_sha256": _sha256_bytes(applied_optimizer_wire),
        "optimizer_wire_bytes": len(applied_optimizer_wire),
        "loss": loss,
        "routes_sha256": routes_sha256,
        "routing_counts": list(counts),
        "unconstrained_routing_counts": list(unconstrained),
        "capacity_rerouted_tokens": rerouted,
        "routing_capacity": capacity,
    }
    return _AppliedCandidate(
        model=model,
        optimizer=optimizer,
        snapshot=snapshot,
        model_wire=applied_model_wire,
        optimizer_wire=applied_optimizer_wire,
        applied_state=state,
    )


def _publish_applied_checkpoint(
    transaction_dir: Path,
    candidate: _AppliedCandidate,
) -> int:
    applied_dir = transaction_dir / "applied"
    state_wire = _canonical_json(candidate.applied_state)
    if applied_dir.exists():
        if applied_dir.is_symlink() or not applied_dir.is_dir():
            raise ValueError("trajectory applied checkpoint is invalid")
        existing_state = (applied_dir / "state.json").read_bytes()
        existing_model = (applied_dir / "model.safetensors").read_bytes()
        existing_optimizer = (
            applied_dir / "optimizer.safetensors"
        ).read_bytes()
        if (
            existing_state != state_wire
            or existing_model != candidate.model_wire
            or existing_optimizer != candidate.optimizer_wire
        ):
            raise ValueError("trajectory applied checkpoint differs on retry")
        return (
            len(existing_state)
            + len(existing_model)
            + len(existing_optimizer)
        )
    stage = transaction_dir / f".applied.tmp-{os.getpid()}"
    if stage.exists():
        raise ValueError("trajectory applied checkpoint staging exists")
    stage.mkdir(exist_ok=False)
    try:
        _write_new_file(stage / "state.json", state_wire)
        _write_new_file(stage / "model.safetensors", candidate.model_wire)
        _write_new_file(
            stage / "optimizer.safetensors",
            candidate.optimizer_wire,
        )
        _fsync_directory(stage)
        os.replace(stage, applied_dir)
        _fsync_directory(transaction_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return (
        len(state_wire)
        + len(candidate.model_wire)
        + len(candidate.optimizer_wire)
    )


def _validate_applied_checkpoint(
    transaction_dir: Path,
    candidate: _AppliedCandidate,
) -> int:
    applied_dir = transaction_dir / "applied"
    if applied_dir.is_symlink() or not applied_dir.is_dir():
        raise ValueError("trajectory applied checkpoint is unavailable")
    entries = tuple(applied_dir.iterdir())
    if (
        {entry.name for entry in entries}
        != {"state.json", "model.safetensors", "optimizer.safetensors"}
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("trajectory applied checkpoint files are invalid")
    state_wire = (applied_dir / "state.json").read_bytes()
    state = _load_json_bytes(
        state_wire,
        label="trajectory applied checkpoint state",
    )
    if (
        frozenset(state) != _APPLIED_STATE_FIELDS
        or state != candidate.applied_state
    ):
        raise ValueError("trajectory applied checkpoint state differs")
    model_wire = _read_owned_file(
        applied_dir / "model.safetensors",
        expected_bytes=int(state["model_wire_bytes"]),
        expected_sha256=str(state["model_wire_sha256"]),
    )
    optimizer_wire = _read_owned_file(
        applied_dir / "optimizer.safetensors",
        expected_bytes=int(state["optimizer_wire_bytes"]),
        expected_sha256=str(state["optimizer_wire_sha256"]),
    )
    if (
        model_wire != candidate.model_wire
        or optimizer_wire != candidate.optimizer_wire
    ):
        raise ValueError("trajectory applied checkpoint tensor bytes differ")
    return len(state_wire) + len(model_wire) + len(optimizer_wire)


def _mark_transaction_applied(
    transaction_dir: Path,
    manifest: dict[str, object],
    candidate: _AppliedCandidate,
) -> None:
    if (
        manifest["phase"] != "results_accepted"
        or manifest["result_applied"] is not False
    ):
        raise ValueError("trajectory transaction cannot enter applied phase")
    state = candidate.applied_state
    manifest["phase"] = "applied"
    manifest["phase_history"] = list(_PHASES)
    manifest["result_applied"] = True
    manifest["applied_checkpoint"] = {
        "directory": "applied",
        "state_sha256": _sha256_bytes(_canonical_json(state)),
        "model_sha256": state["model_wire_sha256"],
        "optimizer_sha256": state["optimizer_wire_sha256"],
    }
    _write_manifest(transaction_dir, manifest)


def _apply_transaction_once(
    campaign: CampaignConfig,
    transaction_dir: Path,
    *,
    simulate_loss_after_publish: bool = False,
) -> tuple[_AppliedCandidate, int]:
    manifest = _load_manifest(transaction_dir)
    _validate_transaction_root(transaction_dir, manifest)
    if manifest["phase"] == "applied":
        raise ValueError("trajectory transaction result was already applied")
    if (transaction_dir / "applied").exists():
        raise ValueError(
            "trajectory applied checkpoint requires recovery before retry"
        )
    candidate = _compute_transaction_candidate(campaign, transaction_dir)
    persisted_bytes = _publish_applied_checkpoint(
        transaction_dir,
        candidate,
    )
    if simulate_loss_after_publish:
        raise _SimulatedCoordinatorLoss(candidate, persisted_bytes)
    manifest = _load_manifest(transaction_dir)
    _mark_transaction_applied(transaction_dir, manifest, candidate)
    return candidate, persisted_bytes


def _recover_published_apply(
    campaign: CampaignConfig,
    transaction_dir: Path,
) -> tuple[_AppliedCandidate, int]:
    manifest = _load_manifest(transaction_dir)
    _validate_transaction_root(transaction_dir, manifest)
    if manifest["phase"] != "results_accepted":
        raise ValueError("trajectory published apply is not recoverable")
    if not (transaction_dir / "applied").is_dir():
        raise ValueError("trajectory published apply checkpoint is absent")
    candidate = _compute_transaction_candidate(campaign, transaction_dir)
    persisted_bytes = _validate_applied_checkpoint(
        transaction_dir,
        candidate,
    )
    manifest = _load_manifest(transaction_dir)
    _mark_transaction_applied(transaction_dir, manifest, candidate)
    return candidate, persisted_bytes


def _coordinator_recovery_entry(
    connection: Connection,
    campaign: CampaignConfig,
    transaction_dir: Path,
) -> None:
    try:
        configure_determinism(campaign.training.seed)
        candidate, persisted_bytes = _recover_published_apply(
            campaign,
            transaction_dir,
        )
        _send_json(
            connection,
            {
                "status": "recovered",
                "transaction_id": candidate.applied_state["transaction_id"],
                "model_wire_sha256": candidate.applied_state[
                    "model_wire_sha256"
                ],
                "optimizer_wire_sha256": candidate.applied_state[
                    "optimizer_wire_sha256"
                ],
                "coordinator_compute_seconds": (
                    candidate.snapshot.step_seconds
                ),
                "persisted_checkpoint_bytes": persisted_bytes,
            },
        )
    except BaseException as exc:
        try:
            _send_json(
                connection,
                {"status": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
        except BaseException:
            pass
        raise
    finally:
        connection.close()


def _recover_apply_in_new_process(
    context: multiprocessing.context.BaseContext,
    campaign: CampaignConfig,
    transaction_dir: Path,
    *,
    timeout_seconds: float,
) -> tuple[_AppliedCandidate, float, int]:
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_coordinator_recovery_entry,
        args=(child_connection, campaign, transaction_dir),
        name="orcacolony-sparse-coordinator-recovery",
        daemon=False,
    )
    process.start()
    child_connection.close()
    acknowledgement: dict[str, object] | None = None
    try:
        if not parent_connection.poll(timeout_seconds):
            raise TimeoutError(
                "timed out waiting for sparse coordinator recovery"
            )
        raw = parent_connection.recv_bytes()
        acknowledgement = _load_json_bytes(
            raw,
            label="sparse coordinator recovery",
        )
    finally:
        parent_connection.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
    if process.exitcode != 0 or acknowledgement is None:
        raise ValueError("sparse coordinator recovery process failed")
    expected = frozenset(
        {
            "status",
            "transaction_id",
            "model_wire_sha256",
            "optimizer_wire_sha256",
            "coordinator_compute_seconds",
            "persisted_checkpoint_bytes",
        }
    )
    if (
        frozenset(acknowledgement) != expected
        or acknowledgement["status"] != "recovered"
        or type(acknowledgement["coordinator_compute_seconds"]) is not float
        or float(acknowledgement["coordinator_compute_seconds"]) <= 0.0
        or type(acknowledgement["persisted_checkpoint_bytes"]) is not int
        or int(acknowledgement["persisted_checkpoint_bytes"]) <= 0
    ):
        raise ValueError("sparse coordinator recovery acknowledgement is invalid")
    candidate = _compute_transaction_candidate(
        campaign,
        transaction_dir,
        allow_applied=True,
    )
    _validate_applied_checkpoint(transaction_dir, candidate)
    if (
        acknowledgement["transaction_id"]
        != candidate.applied_state["transaction_id"]
        or acknowledgement["model_wire_sha256"]
        != candidate.applied_state["model_wire_sha256"]
        or acknowledgement["optimizer_wire_sha256"]
        != candidate.applied_state["optimizer_wire_sha256"]
    ):
        raise ValueError("sparse coordinator recovery identity differs")
    return (
        candidate,
        float(acknowledgement["coordinator_compute_seconds"]),
        int(process.exitcode),
    )


@dataclass(frozen=True)
class _TopologyStepRun:
    prepared: _PreparedTopologyStep
    snapshot: _TrajectorySnapshot
    transaction: PersistedTransactionEvidence
    tensor_wire_bytes: int
    control_json_wire_bytes: int
    persisted_bytes: int
    end_to_end_seconds: float
    worker_round_trip_seconds: tuple[float, ...]
    worker_compute_seconds: tuple[float, ...]
    persistence_seconds: float
    coordinator_apply_seconds: float
    trainable_state_sha256: tuple[str, ...]


@dataclass(frozen=True)
class _TopologyRun:
    steps: tuple[_TopologyStepRun, ...]
    workers: tuple[TrajectoryWorkerEvidence, ...]
    end_to_end_seconds: float
    tensor_wire_bytes: int
    control_json_wire_bytes: int
    persisted_bytes: int
    final_model_sha256: str
    final_optimizer_sha256: str


def _directory_file_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("persisted trajectory contains a symlink")
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _transaction_evidence(
    transaction_dir: Path,
    candidate: _AppliedCandidate,
    *,
    coordinator_publish_recovered: bool,
    duplicate_apply_rejected: bool,
) -> PersistedTransactionEvidence:
    manifest = _load_manifest(transaction_dir)
    _validate_transaction_root(transaction_dir, manifest)
    records, _, _ = _reconcile_results(
        transaction_dir,
        manifest,
        write=False,
    )
    if (
        manifest["phase"] != "applied"
        or manifest["result_applied"] is not True
        or tuple(manifest["phase_history"]) != _PHASES
    ):
        raise ValueError("trajectory transaction did not finish applied")
    result_bytes = sum(
        int(record["acceptance_bytes"])
        + int(record["result_wire_bytes"])
        for record in records
    )
    checkpoint_bytes = _validate_applied_checkpoint(
        transaction_dir,
        candidate,
    )
    manifest_wire = (transaction_dir / "manifest.json").read_bytes()
    identity = _validate_identity(manifest["identity"])
    state = candidate.applied_state
    expected_applied = {
        "directory": "applied",
        "state_sha256": _sha256_bytes(_canonical_json(state)),
        "model_sha256": state["model_wire_sha256"],
        "optimizer_sha256": state["optimizer_wire_sha256"],
    }
    if manifest["applied_checkpoint"] != expected_applied:
        raise ValueError("trajectory applied-checkpoint manifest differs")
    return PersistedTransactionEvidence(
        topology=str(identity["topology"]),
        step=int(identity["step"]),
        transaction_id=str(manifest["transaction_id"]),
        phase_history=tuple(str(value) for value in manifest["phase_history"]),
        accepted_result_count=int(manifest["accepted_result_count"]),
        persisted_result_bytes=result_bytes,
        persisted_checkpoint_bytes=checkpoint_bytes,
        manifest_sha256=_sha256_bytes(manifest_wire),
        manifest_bytes=len(manifest_wire),
        applied_model_sha256=str(state["model_sha256"]),
        applied_model_wire_sha256=str(state["model_wire_sha256"]),
        applied_optimizer_sha256=str(state["optimizer_sha256"]),
        applied_optimizer_wire_sha256=str(
            state["optimizer_wire_sha256"]
        ),
        coordinator_publish_recovered=coordinator_publish_recovered,
        duplicate_apply_rejected=duplicate_apply_rejected,
    )


def _run_centralized_step(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    *,
    step: int,
    expert_count: int,
    router_aux_weight: float,
) -> _TrajectorySnapshot:
    started = time.perf_counter()
    model_tensors, optimizer_tensors, _, _ = _checkpoint_wires(
        model,
        optimizer,
        step=step,
    )
    cursor = (
        step * campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    inputs, targets = fixture_batch(campaign, cursor, dataset)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    (
        _,
        loss,
        routes,
        counts,
        unconstrained,
        rerouted,
        capacity,
    ) = _full_worker_result(
        model,
        inputs,
        targets,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
    )
    raw = _trainable_gradient_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    clipped = _trainable_gradient_snapshot(model)
    optimizer.step()
    return _TrajectorySnapshot(
        step=step,
        cursor=cursor,
        pre_model_sha256=tensor_sha256(model_tensors),
        pre_optimizer_sha256=tensor_sha256(optimizer_tensors),
        loss=float(loss.detach()),
        routes_sha256=tensor_sha256({"routes": routes}),
        routing_counts=counts,
        unconstrained_routing_counts=unconstrained,
        capacity_rerouted_tokens=rerouted,
        routing_capacity=capacity,
        raw_gradients=raw,
        clipped_gradients=clipped,
        optimizer=_trainable_optimizer_tensor_snapshot(model, optimizer),
        model=_model_snapshot(model),
        step_seconds=time.perf_counter() - started,
    )


def _run_centralized_trajectory(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    *,
    steps: int,
    expert_count: int,
    router_aux_weight: float,
    head_sha256: str,
) -> tuple[
    tuple[_TrajectorySnapshot, ...],
    float,
    str,
    str,
]:
    started = time.perf_counter()
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    if tensor_sha256(_head_tensor_snapshot(model)) != head_sha256:
        raise AssertionError("centralized trajectory frozen head changed")
    optimizer = _create_optimizer(model, campaign.training)
    snapshots = tuple(
        _run_centralized_step(
            campaign,
            dataset,
            model,
            optimizer,
            step=step,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
        )
        for step in range(steps)
    )
    return (
        snapshots,
        time.perf_counter() - started,
        tensor_sha256(_model_snapshot(model)),
        tensor_sha256(
            _trainable_optimizer_tensor_snapshot(model, optimizer)
        ),
    )


def _create_step_transaction(
    campaign: CampaignConfig,
    model: SparseExpertDecoder,
    optimizer: torch.optim.AdamW,
    prepared: _PreparedTopologyStep,
    transaction_dir: Path,
    *,
    topology: str,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    bytes,
    bytes,
    float,
]:
    started = time.perf_counter()
    model_tensors, optimizer_tensors, model_wire, optimizer_wire = (
        _checkpoint_wires(
            model,
            optimizer,
            step=prepared.step,
        )
    )
    identity = _transaction_identity(
        campaign,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        topology=topology,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
        head_sha256=head_sha256,
        prepared=prepared,
        model_tensors=model_tensors,
        optimizer_tensors=optimizer_tensors,
        model_wire=model_wire,
        optimizer_wire=optimizer_wire,
    )
    _create_transaction(
        transaction_dir,
        identity,
        model_wire=model_wire,
        optimizer_wire=optimizer_wire,
        batch_wire=prepared.batch_wire,
    )
    return (
        model_tensors,
        optimizer_tensors,
        model_wire,
        optimizer_wire,
        time.perf_counter() - started,
    )


def _reject_duplicate_apply(
    campaign: CampaignConfig,
    transaction_dir: Path,
) -> bool:
    try:
        _apply_transaction_once(campaign, transaction_dir)
    except ValueError as exc:
        return "already applied" in str(exc)
    raise AssertionError("trajectory duplicate apply unexpectedly succeeded")


def _run_full_trajectory(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    state_root: Path,
    *,
    steps: int,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_wire: bytes,
    head_sha256: str,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> _TopologyRun:
    started = time.perf_counter()
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    optimizer = _create_optimizer(model, campaign.training)
    context = multiprocessing.get_context("spawn")
    worker: _WorkerHandle | None = None
    worker_evidence: TrajectoryWorkerEvidence | None = None
    step_runs: list[_TopologyStepRun] = []
    full_root = state_root / "full"
    full_root.mkdir(exist_ok=False)
    try:
        worker = _start_worker(
            context,
            campaign,
            worker_kind="full",
            generation=0,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=sample_interval_seconds,
        )
        for step in range(steps):
            step_started = time.perf_counter()
            prepared = _prepare_full_step(
                campaign,
                dataset,
                model,
                step=step,
            )
            transaction_dir = full_root / f"step-{step:08d}"
            (
                model_tensors,
                optimizer_tensors,
                _,
                _,
                persistence_seconds,
            ) = _create_step_transaction(
                campaign,
                model,
                optimizer,
                prepared,
                transaction_dir,
                topology="full",
                expert_count=expert_count,
                router_aux_weight=router_aux_weight,
                campaign_revision_value=campaign_revision_value,
                dataset_revision=dataset_revision,
                head_sha256=head_sha256,
            )
            result = _run_worker_assignment(
                worker,
                prepared,
                result_index=0,
                expert_index=None,
                campaign_revision_value=campaign_revision_value,
                dataset_revision=dataset_revision,
                head_sha256=head_sha256,
                timeout_seconds=timeout_seconds,
            )
            persisted_started = time.perf_counter()
            _persist_result(transaction_dir, result)
            persistence_seconds += time.perf_counter() - persisted_started
            apply_started = time.perf_counter()
            candidate, _ = _apply_transaction_once(
                campaign,
                transaction_dir,
            )
            apply_elapsed = time.perf_counter() - apply_started
            persistence_seconds += max(
                0.0,
                apply_elapsed - candidate.snapshot.step_seconds,
            )
            duplicate_rejected = _reject_duplicate_apply(
                campaign,
                transaction_dir,
            )
            transaction = _transaction_evidence(
                transaction_dir,
                candidate,
                coordinator_publish_recovered=False,
                duplicate_apply_rejected=duplicate_rejected,
            )
            model = candidate.model
            optimizer = candidate.optimizer
            step_runs.append(
                _TopologyStepRun(
                    prepared=prepared,
                    snapshot=candidate.snapshot,
                    transaction=transaction,
                    tensor_wire_bytes=(
                        len(head_wire) if step == 0 else 0
                    )
                    + len(prepared.trainable_wires[0])
                    + len(prepared.input_wires[0])
                    + len(result.result_wire),
                    control_json_wire_bytes=result.control_json_wire_bytes,
                    persisted_bytes=_directory_file_bytes(transaction_dir),
                    end_to_end_seconds=time.perf_counter() - step_started,
                    worker_round_trip_seconds=(result.round_trip_seconds,),
                    worker_compute_seconds=(
                        float(result.acknowledgement["compute_seconds"]),
                    ),
                    persistence_seconds=persistence_seconds,
                    coordinator_apply_seconds=candidate.snapshot.step_seconds,
                    trainable_state_sha256=(
                        tensor_sha256(prepared.trainable_states[0]),
                    ),
                )
            )
            if (
                tensor_sha256(model_tensors)
                != candidate.snapshot.pre_model_sha256
                or tensor_sha256(optimizer_tensors)
                != candidate.snapshot.pre_optimizer_sha256
            ):
                raise AssertionError("full trajectory pre-state identity changed")
        worker_evidence = _stop_worker(worker, timeout_seconds)
        worker = None
    finally:
        if worker is not None:
            _terminate_worker(worker, timeout_seconds)
    if worker_evidence is None:
        raise AssertionError("full trajectory worker evidence is absent")
    final_optimizer = _trainable_optimizer_tensor_snapshot(model, optimizer)
    return _TopologyRun(
        steps=tuple(step_runs),
        workers=(worker_evidence,),
        end_to_end_seconds=time.perf_counter() - started,
        tensor_wire_bytes=sum(run.tensor_wire_bytes for run in step_runs),
        control_json_wire_bytes=(
            worker_evidence.control_json_wire_bytes
        ),
        persisted_bytes=sum(run.persisted_bytes for run in step_runs),
        final_model_sha256=tensor_sha256(_model_snapshot(model)),
        final_optimizer_sha256=tensor_sha256(final_optimizer),
    )


def _run_expert_trajectory(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    state_root: Path,
    *,
    steps: int,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_wire: bytes,
    head_sha256: str,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> tuple[
    _TopologyRun,
    WorkerReplacementEvidence,
    CoordinatorRecoveryEvidence,
]:
    started = time.perf_counter()
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    optimizer = _create_optimizer(model, campaign.training)
    context = multiprocessing.get_context("spawn")
    worker: _WorkerHandle | None = None
    workers: list[TrajectoryWorkerEvidence] = []
    step_runs: list[_TopologyStepRun] = []
    expert_root = state_root / "expert"
    expert_root.mkdir(exist_ok=False)
    replacement_step = 1 if steps > 1 else 0
    coordinator_recovery_step = steps - 1
    first_loss_evidence: TrajectoryWorkerEvidence | None = None
    replacement_evidence: TrajectoryWorkerEvidence | None = None
    coordinator_recovery: CoordinatorRecoveryEvidence | None = None
    generation = 0
    try:
        worker = _start_worker(
            context,
            campaign,
            worker_kind="pooled_expert",
            generation=generation,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=sample_interval_seconds,
        )
        for step in range(steps):
            step_started = time.perf_counter()
            prepared = _prepare_expert_step(
                campaign,
                dataset,
                model,
                step=step,
                expert_count=expert_count,
            )
            transaction_dir = expert_root / f"step-{step:08d}"
            (
                model_tensors,
                optimizer_tensors,
                _,
                _,
                persistence_seconds,
            ) = _create_step_transaction(
                campaign,
                model,
                optimizer,
                prepared,
                transaction_dir,
                topology="expert",
                expert_count=expert_count,
                router_aux_weight=router_aux_weight,
                campaign_revision_value=campaign_revision_value,
                dataset_revision=dataset_revision,
                head_sha256=head_sha256,
            )
            results: list[_CollectedResult] = []
            persistence_bytes_from_results = 0
            for expert_index in range(expert_count):
                result = _run_worker_assignment(
                    worker,
                    prepared,
                    result_index=expert_index,
                    expert_index=expert_index,
                    campaign_revision_value=campaign_revision_value,
                    dataset_revision=dataset_revision,
                    head_sha256=head_sha256,
                    timeout_seconds=timeout_seconds,
                )
                results.append(result)
                persisted_started = time.perf_counter()
                persistence_bytes_from_results += _persist_result(
                    transaction_dir,
                    result,
                )
                persistence_seconds += time.perf_counter() - persisted_started
                if (
                    step == replacement_step
                    and expert_index == 0
                    and first_loss_evidence is None
                ):
                    persisted_manifest = _load_manifest(transaction_dir)
                    if persisted_manifest["accepted_result_count"] != 1:
                        raise AssertionError(
                            "expert result was not durable before worker loss"
                        )
                    first_loss_evidence = _terminate_worker(
                        worker,
                        timeout_seconds,
                    )
                    workers.append(first_loss_evidence)
                    worker = None
                    generation += 1
                    worker = _start_worker(
                        context,
                        campaign,
                        worker_kind="pooled_expert",
                        generation=generation,
                        expert_count=expert_count,
                        router_aux_weight=router_aux_weight,
                        campaign_revision_value=campaign_revision_value,
                        dataset_revision=dataset_revision,
                        head_wire=head_wire,
                        head_sha256=head_sha256,
                        timeout_seconds=timeout_seconds,
                        sample_interval_seconds=sample_interval_seconds,
                    )
            coordinator_recovered = False
            duplicate_rejected = False
            recovery_seconds = 0.0
            coordinator_apply_seconds = 0.0
            apply_started = time.perf_counter()
            if step == coordinator_recovery_step:
                replacement_evidence = _stop_worker(
                    worker,
                    timeout_seconds,
                )
                workers.append(replacement_evidence)
                worker = None
                try:
                    _apply_transaction_once(
                        campaign,
                        transaction_dir,
                        simulate_loss_after_publish=True,
                    )
                except _SimulatedCoordinatorLoss as loss:
                    first_candidate = loss.candidate
                else:
                    raise AssertionError(
                        "simulated coordinator loss did not occur"
                    )
                manifest_before_recovery = _load_manifest(transaction_dir)
                if (
                    manifest_before_recovery["phase"]
                    != "results_accepted"
                    or manifest_before_recovery["result_applied"] is not False
                    or not (transaction_dir / "applied").is_dir()
                ):
                    raise AssertionError(
                        "coordinator loss boundary was not persisted"
                    )
                recovery_started = time.perf_counter()
                (
                    candidate,
                    recovery_process_compute_seconds,
                    recovery_process_exit_code,
                ) = _recover_apply_in_new_process(
                    context,
                    campaign,
                    transaction_dir,
                    timeout_seconds=timeout_seconds,
                )
                recovery_seconds = time.perf_counter() - recovery_started
                coordinator_apply_seconds = (
                    first_candidate.snapshot.step_seconds
                    + recovery_process_compute_seconds
                    + candidate.snapshot.step_seconds
                )
                coordinator_recovered = True
                duplicate_rejected = _reject_duplicate_apply(
                    campaign,
                    transaction_dir,
                )
                coordinator_recovery = CoordinatorRecoveryEvidence(
                    topology="expert",
                    step=step,
                    applied_checkpoint_published_before_loss=True,
                    manifest_applied_before_loss=False,
                    recovered_from_published_checkpoint=True,
                    recovery_start_method=context.get_start_method(),
                    recovery_process_exit_code=recovery_process_exit_code,
                    new_process_loaded_only_persisted_state=True,
                    recomputed_from_persisted_pre_state_for_validation=True,
                    duplicate_apply_rejected=duplicate_rejected,
                    recovery_seconds=recovery_seconds,
                )
            else:
                candidate, _ = _apply_transaction_once(
                    campaign,
                    transaction_dir,
                )
                coordinator_apply_seconds = candidate.snapshot.step_seconds
                duplicate_rejected = _reject_duplicate_apply(
                    campaign,
                    transaction_dir,
                )
            apply_elapsed = time.perf_counter() - apply_started
            persistence_seconds += max(
                0.0,
                apply_elapsed - coordinator_apply_seconds,
            )
            transaction = _transaction_evidence(
                transaction_dir,
                candidate,
                coordinator_publish_recovered=coordinator_recovered,
                duplicate_apply_rejected=duplicate_rejected,
            )
            model = candidate.model
            optimizer = candidate.optimizer
            head_transmissions_this_step = (
                len(head_wire) if step == 0 else 0
            )
            if step == replacement_step:
                head_transmissions_this_step += len(head_wire)
            step_runs.append(
                _TopologyStepRun(
                    prepared=prepared,
                    snapshot=candidate.snapshot,
                    transaction=transaction,
                    tensor_wire_bytes=(
                        head_transmissions_this_step
                        + sum(len(wire) for wire in prepared.trainable_wires)
                        + sum(len(wire) for wire in prepared.input_wires)
                        + sum(len(result.result_wire) for result in results)
                    ),
                    control_json_wire_bytes=sum(
                        result.control_json_wire_bytes for result in results
                    ),
                    persisted_bytes=_directory_file_bytes(transaction_dir),
                    end_to_end_seconds=time.perf_counter() - step_started,
                    worker_round_trip_seconds=tuple(
                        result.round_trip_seconds for result in results
                    ),
                    worker_compute_seconds=tuple(
                        float(result.acknowledgement["compute_seconds"])
                        for result in results
                    ),
                    persistence_seconds=persistence_seconds,
                    coordinator_apply_seconds=coordinator_apply_seconds,
                    trainable_state_sha256=tuple(
                        tensor_sha256(state)
                        for state in prepared.trainable_states
                    ),
                )
            )
            if persistence_bytes_from_results != (
                transaction.persisted_result_bytes
            ):
                raise AssertionError(
                    "expert persisted result byte accounting changed"
                )
            if (
                tensor_sha256(model_tensors)
                != candidate.snapshot.pre_model_sha256
                or tensor_sha256(optimizer_tensors)
                != candidate.snapshot.pre_optimizer_sha256
            ):
                raise AssertionError("expert trajectory pre-state identity changed")
        if worker is not None:
            replacement_evidence = _stop_worker(worker, timeout_seconds)
            workers.append(replacement_evidence)
            worker = None
    finally:
        if worker is not None:
            workers.append(_terminate_worker(worker, timeout_seconds))
    if (
        first_loss_evidence is None
        or replacement_evidence is None
        or coordinator_recovery is None
    ):
        raise AssertionError("expert trajectory recovery evidence is incomplete")
    persisted_after_loss = (
        expert_root
        / f"step-{replacement_step:08d}"
        / "results"
        / "result-000"
    )
    if not persisted_after_loss.is_dir():
        raise AssertionError("persisted expert result was lost")
    worker_replacement = WorkerReplacementEvidence(
        topology="expert",
        step=replacement_step,
        persisted_result_index=0,
        persisted_result_survived_loss=True,
        first_worker_exit_code=first_loss_evidence.child_exit_code,
        replacement_worker_exit_code=replacement_evidence.child_exit_code,
        replacement_initialized_after_loss=True,
        recomputed_persisted_result=False,
    )
    final_optimizer = _trainable_optimizer_tensor_snapshot(model, optimizer)
    topology = _TopologyRun(
        steps=tuple(step_runs),
        workers=tuple(workers),
        end_to_end_seconds=time.perf_counter() - started,
        tensor_wire_bytes=sum(run.tensor_wire_bytes for run in step_runs),
        control_json_wire_bytes=sum(
            item.control_json_wire_bytes for item in workers
        ),
        persisted_bytes=sum(run.persisted_bytes for run in step_runs),
        final_model_sha256=tensor_sha256(_model_snapshot(model)),
        final_optimizer_sha256=tensor_sha256(final_optimizer),
    )
    return topology, worker_replacement, coordinator_recovery


def _validate_experiment_inputs(
    campaign: CampaignConfig,
    dataset: PackedDataset | None,
    state_dir: Path,
    *,
    steps: int,
    expert_count: int,
    router_aux_weight: float,
    timeout_seconds: float,
    sample_interval_seconds: float,
) -> tuple[PackedDataset, Path, float, float, float]:
    validate_dataset_artifacts(campaign, dataset)
    if campaign.dataset is None or dataset is None:
        raise ValueError(
            "persisted sparse trajectory requires dataset artifacts"
        )
    if type(steps) is not int or not 2 <= steps <= 12:
        raise ValueError("trajectory steps must be an integer between two and twelve")
    if type(expert_count) is not int or not 2 <= expert_count <= 16:
        raise ValueError("expert count must be an integer between two and sixteen")
    if (
        isinstance(router_aux_weight, bool)
        or not isinstance(router_aux_weight, (int, float))
        or not math.isfinite(float(router_aux_weight))
        or not 0.0 < float(router_aux_weight) <= 1.0
    ):
        raise ValueError("router auxiliary weight must be finite in (0, 1]")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 300.0
    ):
        raise ValueError("timeout must be finite and between zero and 300 seconds")
    if (
        isinstance(sample_interval_seconds, bool)
        or not isinstance(sample_interval_seconds, (int, float))
        or not math.isfinite(float(sample_interval_seconds))
        or not 0.001 <= float(sample_interval_seconds) <= 0.1
    ):
        raise ValueError(
            "RSS sample interval must be finite and between 0.001 and 0.1 seconds"
        )
    if not Path("/proc/self/status").is_file():
        raise RuntimeError(
            "persisted sparse trajectory requires Linux /proc RSS sampling"
        )
    state_dir = state_dir.resolve()
    if state_dir.exists():
        if (
            state_dir.is_symlink()
            or not state_dir.is_dir()
            or any(state_dir.iterdir())
        ):
            raise ValueError("trajectory state directory must be new or empty")
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
    if state_dir.is_symlink():
        raise ValueError("trajectory state directory may not be a symlink")
    return (
        dataset,
        state_dir,
        float(router_aux_weight),
        float(timeout_seconds),
        float(sample_interval_seconds),
    )


def _exact_step(
    reference: _TrajectorySnapshot,
    full: _TrajectorySnapshot,
    expert: _TrajectorySnapshot,
) -> bool:
    return (
        reference.step == full.step == expert.step
        and reference.cursor == full.cursor == expert.cursor
        and reference.pre_model_sha256
        == full.pre_model_sha256
        == expert.pre_model_sha256
        and reference.pre_optimizer_sha256
        == full.pre_optimizer_sha256
        == expert.pre_optimizer_sha256
        and reference.loss == full.loss == expert.loss
        and reference.routes_sha256
        == full.routes_sha256
        == expert.routes_sha256
        and reference.routing_counts
        == full.routing_counts
        == expert.routing_counts
        and reference.unconstrained_routing_counts
        == full.unconstrained_routing_counts
        == expert.unconstrained_routing_counts
        and reference.capacity_rerouted_tokens
        == full.capacity_rerouted_tokens
        == expert.capacity_rerouted_tokens
        and reference.routing_capacity
        == full.routing_capacity
        == expert.routing_capacity
        and _max_abs_difference(
            dict(reference.raw_gradients),
            dict(full.raw_gradients),
        )
        == 0.0
        and _max_abs_difference(
            dict(reference.raw_gradients),
            dict(expert.raw_gradients),
        )
        == 0.0
        and _max_abs_difference(
            dict(reference.clipped_gradients),
            dict(full.clipped_gradients),
        )
        == 0.0
        and _max_abs_difference(
            dict(reference.clipped_gradients),
            dict(expert.clipped_gradients),
        )
        == 0.0
        and _max_abs_difference(
            dict(reference.model),
            dict(full.model),
        )
        == 0.0
        and _max_abs_difference(
            dict(reference.model),
            dict(expert.model),
        )
        == 0.0
        and tensor_sha256(reference.raw_gradients)
        == tensor_sha256(full.raw_gradients)
        == tensor_sha256(expert.raw_gradients)
        and tensor_sha256(reference.clipped_gradients)
        == tensor_sha256(full.clipped_gradients)
        == tensor_sha256(expert.clipped_gradients)
        and tensor_sha256(reference.optimizer)
        == tensor_sha256(full.optimizer)
        == tensor_sha256(expert.optimizer)
        and tensor_sha256(reference.model)
        == tensor_sha256(full.model)
        == tensor_sha256(expert.model)
    )


def run_persisted_sparse_trajectory_experiment(
    campaign: CampaignConfig,
    state_dir: Path,
    *,
    dataset: PackedDataset | None,
    steps: int = 3,
    expert_count: int = 4,
    router_aux_weight: float = 0.01,
    timeout_seconds: float = 120.0,
    sample_interval_seconds: float = 0.01,
) -> PersistedSparseTrajectoryEvidence:
    (
        dataset,
        state_dir,
        router_aux_weight,
        timeout_seconds,
        sample_interval_seconds,
    ) = _validate_experiment_inputs(
        campaign,
        dataset,
        state_dir,
        steps=steps,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
        timeout_seconds=timeout_seconds,
        sample_interval_seconds=sample_interval_seconds,
    )
    campaign_revision_value = campaign_revision(campaign)
    dataset_revision = dataset.revision
    source = _build_sparse_model(campaign, expert_count)
    _freeze_head(source)
    head_state = _head_tensor_snapshot(source)
    head_sha256 = tensor_sha256(head_state)
    head_wire = _serialize_tensors(head_state)

    (
        centralized,
        centralized_seconds,
        centralized_final_model,
        centralized_final_optimizer,
    ) = _run_centralized_trajectory(
        campaign,
        dataset,
        steps=steps,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
        head_sha256=head_sha256,
    )
    full = _run_full_trajectory(
        campaign,
        dataset,
        state_dir,
        steps=steps,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_wire=head_wire,
        head_sha256=head_sha256,
        timeout_seconds=timeout_seconds,
        sample_interval_seconds=sample_interval_seconds,
    )
    expert, worker_replacement, coordinator_recovery = (
        _run_expert_trajectory(
            campaign,
            dataset,
            state_dir,
            steps=steps,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            sample_interval_seconds=sample_interval_seconds,
        )
    )
    if not (
        len(centralized) == len(full.steps) == len(expert.steps) == steps
    ):
        raise AssertionError("sparse trajectory step counts differ")

    evidence_steps: list[SparseTrajectoryStepEvidence] = []
    previous_routes: str | None = None
    for reference, full_run, expert_run in zip(
        centralized,
        full.steps,
        expert.steps,
        strict=True,
    ):
        full_snapshot = full_run.snapshot
        expert_snapshot = expert_run.snapshot
        if not _exact_step(reference, full_snapshot, expert_snapshot):
            raise AssertionError(
                f"sparse trajectory process result differs at step {reference.step}"
            )
        full_raw_difference = _max_abs_difference(
            dict(reference.raw_gradients),
            dict(full_snapshot.raw_gradients),
        )
        expert_raw_difference = _max_abs_difference(
            dict(reference.raw_gradients),
            dict(expert_snapshot.raw_gradients),
        )
        full_clipped_difference = _max_abs_difference(
            dict(reference.clipped_gradients),
            dict(full_snapshot.clipped_gradients),
        )
        expert_clipped_difference = _max_abs_difference(
            dict(reference.clipped_gradients),
            dict(expert_snapshot.clipped_gradients),
        )
        full_model_difference = _max_abs_difference(
            dict(reference.model),
            dict(full_snapshot.model),
        )
        expert_model_difference = _max_abs_difference(
            dict(reference.model),
            dict(expert_snapshot.model),
        )
        evidence_steps.append(
            SparseTrajectoryStepEvidence(
                step=reference.step,
                cursor=reference.cursor,
                routing_capacity=reference.routing_capacity,
                routing_counts=reference.routing_counts,
                unconstrained_routing_counts=(
                    reference.unconstrained_routing_counts
                ),
                capacity_rerouted_tokens=(
                    reference.capacity_rerouted_tokens
                ),
                routes_sha256=reference.routes_sha256,
                routes_changed_from_previous=(
                    previous_routes is not None
                    and previous_routes != reference.routes_sha256
                ),
                centralized_pre_model_sha256=reference.pre_model_sha256,
                full_pre_model_sha256=full_snapshot.pre_model_sha256,
                expert_pre_model_sha256=expert_snapshot.pre_model_sha256,
                centralized_pre_optimizer_sha256=(
                    reference.pre_optimizer_sha256
                ),
                full_pre_optimizer_sha256=(
                    full_snapshot.pre_optimizer_sha256
                ),
                expert_pre_optimizer_sha256=(
                    expert_snapshot.pre_optimizer_sha256
                ),
                centralized_loss=reference.loss,
                full_process_loss=full_snapshot.loss,
                expert_process_loss=expert_snapshot.loss,
                full_max_abs_raw_gradient_difference=full_raw_difference,
                expert_max_abs_raw_gradient_difference=(
                    expert_raw_difference
                ),
                full_max_abs_clipped_gradient_difference=(
                    full_clipped_difference
                ),
                expert_max_abs_clipped_gradient_difference=(
                    expert_clipped_difference
                ),
                full_max_abs_model_difference=full_model_difference,
                expert_max_abs_model_difference=expert_model_difference,
                centralized_raw_gradient_sha256=tensor_sha256(
                    reference.raw_gradients
                ),
                full_process_raw_gradient_sha256=tensor_sha256(
                    full_snapshot.raw_gradients
                ),
                expert_process_raw_gradient_sha256=tensor_sha256(
                    expert_snapshot.raw_gradients
                ),
                centralized_clipped_gradient_sha256=tensor_sha256(
                    reference.clipped_gradients
                ),
                full_process_clipped_gradient_sha256=tensor_sha256(
                    full_snapshot.clipped_gradients
                ),
                expert_process_clipped_gradient_sha256=tensor_sha256(
                    expert_snapshot.clipped_gradients
                ),
                centralized_optimizer_sha256=tensor_sha256(
                    reference.optimizer
                ),
                full_process_optimizer_sha256=tensor_sha256(
                    full_snapshot.optimizer
                ),
                expert_process_optimizer_sha256=tensor_sha256(
                    expert_snapshot.optimizer
                ),
                centralized_model_sha256=tensor_sha256(reference.model),
                full_process_model_sha256=tensor_sha256(
                    full_snapshot.model
                ),
                expert_process_model_sha256=tensor_sha256(
                    expert_snapshot.model
                ),
                full_trainable_state_sha256=(
                    full_run.trainable_state_sha256[0]
                ),
                expert_trainable_state_sha256=(
                    expert_run.trainable_state_sha256
                ),
                full_tensor_wire_bytes=full_run.tensor_wire_bytes,
                expert_tensor_wire_bytes=expert_run.tensor_wire_bytes,
                full_control_json_wire_bytes=(
                    full_run.control_json_wire_bytes
                ),
                expert_control_json_wire_bytes=(
                    expert_run.control_json_wire_bytes
                ),
                full_persisted_bytes=full_run.persisted_bytes,
                expert_persisted_bytes=expert_run.persisted_bytes,
                centralized_step_seconds=reference.step_seconds,
                full_end_to_end_step_seconds=(
                    full_run.end_to_end_seconds
                ),
                expert_end_to_end_step_seconds=(
                    expert_run.end_to_end_seconds
                ),
                full_worker_round_trip_seconds=(
                    full_run.worker_round_trip_seconds[0]
                ),
                expert_worker_round_trip_seconds=(
                    expert_run.worker_round_trip_seconds
                ),
                full_worker_compute_seconds=(
                    full_run.worker_compute_seconds[0]
                ),
                expert_worker_compute_seconds=(
                    expert_run.worker_compute_seconds
                ),
                full_persistence_seconds=full_run.persistence_seconds,
                expert_persistence_seconds=expert_run.persistence_seconds,
                full_coordinator_apply_seconds=(
                    full_run.coordinator_apply_seconds
                ),
                expert_coordinator_apply_seconds=(
                    expert_run.coordinator_apply_seconds
                ),
                full_transaction=full_run.transaction,
                expert_transaction=expert_run.transaction,
            )
        )
        previous_routes = reference.routes_sha256

    if not all(
        evidence.full_max_abs_raw_gradient_difference == 0.0
        and evidence.expert_max_abs_raw_gradient_difference == 0.0
        and evidence.full_max_abs_clipped_gradient_difference == 0.0
        and evidence.expert_max_abs_clipped_gradient_difference == 0.0
        and evidence.full_max_abs_model_difference == 0.0
        and evidence.expert_max_abs_model_difference == 0.0
        for evidence in evidence_steps
    ):
        raise AssertionError("sparse trajectory exactness evidence is incomplete")
    if not (
        centralized_final_model
        == full.final_model_sha256
        == expert.final_model_sha256
        and centralized_final_optimizer
        == full.final_optimizer_sha256
        == expert.final_optimizer_sha256
    ):
        raise AssertionError("sparse trajectory final state differs")
    return PersistedSparseTrajectoryEvidence(
        format=_EVIDENCE_FORMAT,
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision=campaign_revision_value,
        dataset_revision=dataset_revision,
        authentication_mode=_AUTHENTICATION_MODE,
        transport_scope=_TRANSPORT_SCOPE,
        start_method=multiprocessing.get_context("spawn").get_start_method(),
        process_scheduling=(
            "one-child-at-a-time-full-then-pooled-expert-with-one-replacement"
        ),
        assignment_state_mode=(
            "sequential-adamw-trajectory-refreshed-model-routes-hidden-and-experts"
        ),
        persistence_scope=(
            "atomic-result-directories-and-applied-checkpoint-directory-with-"
            "manifest-commit"
        ),
        timing_scope=(
            "complete-topology-lifecycle-includes-coordinator-preparation-worker-"
            "ipc-persistence-apply-recovery-and-shutdown"
        ),
        memory_scope=(
            "external-linux-proc-child-lifecycle-rss-and-high-water-by-phase"
        ),
        maximum_simultaneous_worker_processes=1,
        expert_count=expert_count,
        step_count=steps,
        frozen_head_sha256=head_sha256,
        frozen_head_wire_sha256=_sha256_bytes(head_wire),
        frozen_head_wire_bytes=len(head_wire),
        centralized_end_to_end_seconds=centralized_seconds,
        full_process_end_to_end_seconds=full.end_to_end_seconds,
        expert_process_end_to_end_seconds=expert.end_to_end_seconds,
        full_tensor_wire_bytes=full.tensor_wire_bytes,
        expert_tensor_wire_bytes=expert.tensor_wire_bytes,
        full_control_json_wire_bytes=full.control_json_wire_bytes,
        expert_control_json_wire_bytes=expert.control_json_wire_bytes,
        full_persisted_bytes=full.persisted_bytes,
        expert_persisted_bytes=expert.persisted_bytes,
        centralized_final_model_sha256=centralized_final_model,
        full_process_final_model_sha256=full.final_model_sha256,
        expert_process_final_model_sha256=expert.final_model_sha256,
        centralized_final_optimizer_sha256=centralized_final_optimizer,
        full_process_final_optimizer_sha256=full.final_optimizer_sha256,
        expert_process_final_optimizer_sha256=(
            expert.final_optimizer_sha256
        ),
        all_steps_exact=True,
        full_workers=full.workers,
        expert_workers=expert.workers,
        worker_replacement=worker_replacement,
        coordinator_recovery=coordinator_recovery,
        steps=tuple(evidence_steps),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an exact persisted multi-step sparse-expert process trajectory"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=0.01,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    campaign = load_campaign(args.config)
    dataset = PackedDataset.load(args.dataset)
    evidence = run_persisted_sparse_trajectory_experiment(
        campaign,
        args.state,
        dataset=dataset,
        steps=args.steps,
        expert_count=args.expert_count,
        router_aux_weight=args.router_aux_weight,
        timeout_seconds=args.timeout_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
