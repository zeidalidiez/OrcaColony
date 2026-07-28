from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import multiprocessing
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import torch
from torch import Tensor, nn

from orcacolony.artifacts import PackedDataset
from orcacolony.full_process import _validate_token_batch
from orcacolony.reference import (
    CampaignConfig,
    ModelConfig,
    ObjectiveConfig,
    _create_optimizer,
    campaign_revision,
    configure_determinism,
    fixture_batch,
    load_campaign,
    objective_loss_sum,
    tensor_sha256,
    validate_dataset_artifacts,
)
from orcacolony.sparse_expert import (
    SparseExpert,
    SparseExpertDecoder,
    _balanced_top1_routes,
    _build_sparse_model,
    _freeze_head,
    _head_tensor_snapshot,
    _router_auxiliary_loss,
    _trainable_gradient_snapshot,
    _trainable_optimizer_tensor_snapshot,
)
from orcacolony.tile_process import (
    _deserialize_tensors,
    _process_memory_bytes,
    _recv_bytes,
    _recv_json,
    _send_json,
    _serialize_tensors,
)
from orcacolony.tiled_model import (
    _max_abs_difference,
    _model_snapshot,
)


_AUTHENTICATION_MODE = "coordinator-bound-sha256-safetensors-v1"
_TRANSPORT_SCOPE = "trusted-local-spawn-pipe"
_RUN_CONTROL_FORMAT = "orcacolony_sparse_process_assignment_v1"
_INIT_CONTROL_FORMAT = "orcacolony_sparse_process_init_v1"


@dataclass(frozen=True)
class SparseProcessWorkerEvidence:
    worker_kind: str
    expert_index: int | None
    initialization_seconds: float
    initialization_control_json_wire_bytes: int
    shutdown_control_json_wire_bytes: int
    frozen_head_transmissions: int
    trainable_state_transmissions: int
    forward_calls: int
    worker_startup_current_rss_bytes: int
    worker_startup_peak_rss_bytes: int
    worker_after_head_current_rss_bytes: int
    worker_after_head_peak_rss_bytes: int
    worker_final_current_rss_bytes: int
    worker_final_peak_rss_bytes: int
    frozen_head_sha256: str
    child_exit_code: int


@dataclass(frozen=True)
class SparseProcessAssignmentEvidence:
    assignment_id: int
    cursor: int
    total_tokens: int
    routing_capacity: int
    routing_counts: tuple[int, ...]
    unconstrained_routing_counts: tuple[int, ...]
    capacity_rerouted_tokens: int
    routes_sha256: str
    full_trainable_state_wire_bytes: int
    full_input_wire_bytes: int
    full_gradient_result_wire_bytes: int
    full_tensor_wire_bytes: int
    full_control_json_wire_bytes: int
    full_total_application_wire_bytes: int
    expert_trainable_state_wire_bytes: tuple[int, ...]
    expert_input_wire_bytes: tuple[int, ...]
    expert_result_wire_bytes: tuple[int, ...]
    expert_aggregate_tensor_wire_bytes: int
    expert_control_json_wire_bytes: tuple[int, ...]
    expert_aggregate_control_json_wire_bytes: int
    expert_total_application_wire_bytes: int
    centralized_loss: float
    full_process_loss: float
    expert_process_loss: float
    full_max_abs_raw_gradient_difference: float
    full_max_abs_clipped_gradient_difference: float
    full_max_abs_model_difference: float
    expert_max_abs_raw_gradient_difference: float
    expert_max_abs_clipped_gradient_difference: float
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
    full_round_trip_seconds: float
    full_worker_compute_seconds: float
    expert_round_trip_seconds: tuple[float, ...]
    expert_worker_compute_seconds: tuple[float, ...]
    full_worker_peak_rss_bytes: int
    expert_worker_peak_rss_bytes: tuple[int, ...]


@dataclass(frozen=True)
class SparseProcessRecoveryEvidence:
    expert_index: int
    assignment_id: int
    assignment_accepted_before_loss: bool
    first_worker_exit_code: int
    replacement_worker_exit_code: int
    replacement_result_matches_stable: bool
    replacement_result_used_in_canonical_update: bool
    stable_result_wire_sha256: str
    replacement_result_wire_sha256: str
    lost_worker_received_tensor_wire_bytes: int
    recovery_retransmitted_tensor_wire_bytes: int
    replacement_result_tensor_wire_bytes: int
    recovery_control_json_wire_bytes: int
    recovery_total_application_wire_bytes: int
    recovery_seconds: float
    first_worker_initialization_seconds: float
    replacement_worker_initialization_seconds: float
    replacement_worker_peak_rss_bytes: int
    replacement_frozen_head_sha256: str


@dataclass(frozen=True)
class AuthenticatedSparseProcessEvidence:
    format: str
    campaign_id: str
    campaign_revision: str
    dataset_revision: str
    authentication_mode: str
    transport_scope: str
    start_method: str
    process_scheduling: str
    assignment_state_mode: str
    wire_accounting_scope: str
    memory_scope: str
    matched_totals_exclude_recovery: bool
    maximum_simultaneous_worker_processes: int
    expert_count: int
    assignment_count: int
    frozen_head_sha256: str
    frozen_head_wire_sha256: str
    frozen_head_wire_bytes: int
    full_trainable_state_sha256: str
    full_trainable_state_wire_bytes: int
    expert_trainable_state_sha256: tuple[str, ...]
    expert_trainable_state_wire_bytes: tuple[int, ...]
    full_frozen_head_transmissions: int
    expert_frozen_head_transmissions: int
    full_trainable_state_transmissions: int
    expert_trainable_state_transmissions: int
    full_cold_tensor_wire_bytes: int
    expert_cold_tensor_wire_bytes: int
    cold_tensor_wire_relative_change: float
    full_warm_tensor_wire_bytes: int
    expert_warm_tensor_wire_bytes: int
    warm_tensor_wire_relative_change: float
    full_cold_application_wire_bytes: int
    expert_cold_application_wire_bytes: int
    cold_application_wire_relative_change: float
    full_warm_application_wire_bytes: int
    expert_warm_application_wire_bytes: int
    warm_application_wire_relative_change: float
    full_shutdown_control_json_wire_bytes: int
    expert_shutdown_control_json_wire_bytes: int
    full_worker: SparseProcessWorkerEvidence
    expert_workers: tuple[SparseProcessWorkerEvidence, ...]
    recovery: SparseProcessRecoveryEvidence
    assignments: tuple[SparseProcessAssignmentEvidence, ...]


@dataclass(frozen=True)
class _PreparedAssignment:
    assignment_id: int
    cursor: int
    inputs: Tensor
    targets: Tensor
    routes: Tensor
    routing_counts: tuple[int, ...]
    unconstrained_routing_counts: tuple[int, ...]
    capacity_rerouted_tokens: int
    routing_capacity: int
    routes_sha256: str
    full_input_wire: bytes
    expert_input_wires: tuple[bytes, ...]


@dataclass(frozen=True)
class _CollectedResult:
    acknowledgement: Mapping[str, object]
    result_wire: bytes
    control_json_wire_bytes: int
    round_trip_seconds: float


@dataclass(frozen=True)
class _StepSnapshot:
    loss: float
    raw_gradients: Mapping[str, Tensor]
    clipped_gradients: Mapping[str, Tensor]
    optimizer: Mapping[str, Tensor]
    model: Mapping[str, Tensor]
    frozen_head_sha256: str


@dataclass
class _WorkerHandle:
    process: multiprocessing.Process
    connection: Connection
    worker_kind: str
    expert_index: int | None
    initialization_seconds: float
    initialization_control_json_wire_bytes: int
    ready: Mapping[str, object]


class _FrozenHeadExpertProcessModule(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        objective: ObjectiveConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.objective = objective
        self.expert = SparseExpert(config)
        self.final_norm = nn.LayerNorm(config.width)
        self.output_head = nn.Linear(
            config.width,
            config.vocabulary_size,
            bias=False,
        )
        for module in (self.final_norm, self.output_head):
            module.requires_grad_(False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output_head(self.final_norm(self.expert(hidden)))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_wire_identity(
    payload: bytes,
    *,
    expected_sha256: object,
    expected_bytes: object,
    label: str,
) -> None:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise ValueError(f"{label} wire identity is invalid")
    if len(payload) != expected_bytes or not hmac.compare_digest(
        _sha256_bytes(payload),
        expected_sha256,
    ):
        raise ValueError(f"{label} wire identity mismatch")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate process-control field: {key}")
        payload[key] = value
    return payload


def _recv_child_json(connection: Connection, *, label: str) -> dict[str, object]:
    raw = connection.recv_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _head_state(module: nn.Module) -> dict[str, Tensor]:
    final_norm = getattr(module, "final_norm", None)
    output_head = getattr(module, "output_head", None)
    if not isinstance(final_norm, nn.LayerNorm) or not isinstance(
        output_head,
        nn.Linear,
    ):
        raise TypeError("process worker lacks the frozen head modules")
    return {
        **{
            f"final_norm.{name}": tensor.detach().clone()
            for name, tensor in final_norm.state_dict().items()
        },
        **{
            f"output_head.{name}": tensor.detach().clone()
            for name, tensor in output_head.state_dict().items()
        },
    }


def _load_head_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected = _head_state(module)
    if frozenset(state) != frozenset(expected):
        raise ValueError("frozen-head tensor names are invalid")
    final_norm = getattr(module, "final_norm")
    output_head = getattr(module, "output_head")
    norm_state: dict[str, Tensor] = {}
    output_state: dict[str, Tensor] = {}
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if (
            tensor.dtype != expected_tensor.dtype
            or tensor.shape != expected_tensor.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"frozen-head tensor is invalid: {name}")
        prefix, state_name = name.split(".", 1)
        if prefix == "final_norm":
            norm_state[state_name] = tensor.detach().clone()
        else:
            output_state[state_name] = tensor.detach().clone()
    final_norm.load_state_dict(norm_state, strict=True)
    output_head.load_state_dict(output_state, strict=True)
    for frozen_module in (final_norm, output_head):
        frozen_module.requires_grad_(False)


def _trainable_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(
    module: nn.Module,
    state: Mapping[str, Tensor],
    *,
    expected_sha256: object,
) -> None:
    expected = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if frozenset(state) != frozenset(expected):
        raise ValueError("trainable-state tensor names are invalid")
    with torch.no_grad():
        for name, parameter in expected.items():
            tensor = state[name]
            if (
                tensor.dtype != parameter.dtype
                or tensor.shape != parameter.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"trainable-state tensor is invalid: {name}")
            parameter.copy_(tensor)
    if (
        not isinstance(expected_sha256, str)
        or tensor_sha256(_trainable_state(module)) != expected_sha256
    ):
        raise ValueError("trainable-state semantic identity mismatch")


def _validate_expert_input(
    payload: Mapping[str, Tensor],
    *,
    config: ModelConfig,
    expert_index: int,
    expert_count: int,
    total_tokens: int,
    expected_routes_sha256: object,
) -> tuple[Tensor, Tensor, Tensor]:
    if frozenset(payload) != frozenset(
        {"hidden", "targets", "positions", "routes"}
    ):
        raise ValueError("expert input tensor schema is invalid")
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
        raise ValueError("expert hidden tensor is invalid")
    if (
        targets.dtype != torch.int64
        or targets.ndim != 1
        or targets.shape[0] != hidden.shape[0]
        or int(targets.min()) < 0
        or int(targets.max()) >= config.vocabulary_size
    ):
        raise ValueError("expert target tensor is invalid")
    if (
        routes.dtype != torch.int64
        or routes.ndim != 1
        or routes.numel() != total_tokens
        or int(routes.min()) < 0
        or int(routes.max()) >= expert_count
    ):
        raise ValueError("expert route tensor is invalid")
    if (
        not isinstance(expected_routes_sha256, str)
        or tensor_sha256({"routes": routes}) != expected_routes_sha256
    ):
        raise ValueError("expert route identity mismatch")
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
        raise ValueError("expert selected positions are invalid")
    return (
        hidden.detach().clone(),
        targets.detach().clone(),
        positions.detach().clone(),
    )


def _validate_init_control(control: Mapping[str, object]) -> None:
    expected = frozenset(
        {
            "format",
            "op",
            "worker_kind",
            "expert_index",
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
        raise ValueError("sparse-process init schema is invalid")
    worker_kind = control["worker_kind"]
    expert_index = control["expert_index"]
    expert_count = control["expert_count"]
    if (
        control["format"] != _INIT_CONTROL_FORMAT
        or control["op"] != "init"
        or worker_kind not in {"full", "expert"}
        or type(expert_count) is not int
        or not 2 <= expert_count <= 16
        or type(control["seed"]) is not int
        or isinstance(control["router_aux_weight"], bool)
        or not isinstance(control["router_aux_weight"], (int, float))
        or not math.isfinite(float(control["router_aux_weight"]))
        or not 0.0 < float(control["router_aux_weight"]) <= 1.0
        or not isinstance(control["campaign_id"], str)
        or not isinstance(control["campaign_revision"], str)
        or not isinstance(control["dataset_revision"], str)
    ):
        raise ValueError("sparse-process init semantics are invalid")
    if worker_kind == "full":
        if expert_index is not None:
            raise ValueError("full worker cannot declare an expert index")
    elif type(expert_index) is not int or not 0 <= expert_index < expert_count:
        raise ValueError("expert worker index is invalid")
    if not isinstance(control["model"], dict) or not isinstance(
        control["objective"],
        dict,
    ):
        raise ValueError("sparse-process model contract is invalid")


def _validate_run_control(
    control: Mapping[str, object],
    *,
    worker_kind: str,
    expert_index: int | None,
    campaign_revision_value: str,
    dataset_revision: str,
    frozen_head_sha256: str,
) -> None:
    expected = frozenset(
        {
            "format",
            "op",
            "worker_kind",
            "expert_index",
            "assignment_id",
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
        raise ValueError("sparse-process assignment schema is invalid")
    if (
        control["format"] != _RUN_CONTROL_FORMAT
        or control["op"] not in {"run", "accept_then_wait"}
        or control["worker_kind"] != worker_kind
        or control["expert_index"] != expert_index
        or type(control["assignment_id"]) is not int
        or control["assignment_id"] < 0
        or type(control["cursor"]) is not int
        or control["cursor"] < 0
        or control["campaign_revision"] != campaign_revision_value
        or control["dataset_revision"] != dataset_revision
        or control["frozen_head_sha256"] != frozen_head_sha256
        or type(control["total_tokens"]) is not int
        or control["total_tokens"] <= 0
        or not isinstance(control["routes_sha256"], str)
    ):
        raise ValueError("sparse-process assignment identity is invalid")
    if worker_kind != "expert" and control["op"] == "accept_then_wait":
        raise ValueError("only an expert assignment may pause before compute")


def _full_worker_result(
    model: SparseExpertDecoder,
    inputs: Tensor,
    targets: Tensor,
    *,
    expert_count: int,
    router_aux_weight: float,
) -> tuple[dict[str, Tensor], Tensor, Tensor, tuple[int, ...], tuple[int, ...], int, int]:
    model.zero_grad(set_to_none=True)
    hidden = model.shared_hidden(inputs)
    hidden_flat = hidden.reshape(-1, model.config.width)
    router_logits = model.router(hidden_flat)
    routes, counts, unconstrained, rerouted, capacity = _balanced_top1_routes(
        router_logits,
        expert_count,
    )
    auxiliary = _router_auxiliary_loss(
        router_logits,
        routes,
        expert_count,
        router_aux_weight,
    )
    targets_flat = targets.reshape(-1)
    total_tokens = targets.numel()
    losses: list[Tensor] = []
    for expert_index in range(expert_count):
        selected = hidden_flat[routes == expert_index]
        logits = model.logits_for_hidden(model.experts[expert_index](selected))
        loss_sum, _ = objective_loss_sum(
            model.objective,
            logits,
            targets_flat[routes == expert_index],
        )
        losses.append(loss_sum / total_tokens)
    auxiliary.backward(retain_graph=True)
    for expert_index, loss in enumerate(losses):
        loss.backward(retain_graph=expert_index < expert_count - 1)
    loss = auxiliary + sum(losses)
    result = {
        f"gradient.{name}": parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    expected = sum(
        1 for parameter in model.parameters() if parameter.requires_grad
    )
    if len(result) != expected:
        raise AssertionError("full sparse worker gradient set is incomplete")
    return (
        result,
        loss,
        routes,
        counts,
        unconstrained,
        rerouted,
        capacity,
    )


def _expert_worker_result(
    model: _FrozenHeadExpertProcessModule,
    hidden: Tensor,
    targets: Tensor,
    *,
    total_tokens: int,
) -> tuple[dict[str, Tensor], Tensor]:
    model.zero_grad(set_to_none=True)
    selected_hidden = hidden.detach().clone().requires_grad_(True)
    logits = model(selected_hidden)
    loss_sum, _ = objective_loss_sum(
        model.objective,
        logits,
        targets,
    )
    loss = loss_sum / total_tokens
    loss.backward()
    if selected_hidden.grad is None:
        raise AssertionError("expert process did not produce an input adjoint")
    result: dict[str, Tensor] = {
        "input_adjoint": selected_hidden.grad.detach().clone()
    }
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise AssertionError(
                    f"expert process parameter lacks gradient: {name}"
                )
            result[f"gradient.{name}"] = parameter.grad.detach().clone()
        elif parameter.grad is not None:
            raise AssertionError(
                f"frozen expert-process head acquired gradient: {name}"
            )
    return result, loss


def _sparse_process_worker_entry(connection: Connection) -> None:
    try:
        startup_current, startup_peak = _process_memory_bytes()
        init = _recv_child_json(connection, label="sparse-process init")
        _validate_init_control(init)
        worker_kind = str(init["worker_kind"])
        expert_index = (
            int(init["expert_index"])
            if init["expert_index"] is not None
            else None
        )
        expert_count = int(init["expert_count"])
        config = ModelConfig(**init["model"])
        objective = ObjectiveConfig(**init["objective"])
        seed = int(init["seed"])
        configure_determinism(seed)
        if worker_kind == "full":
            model: SparseExpertDecoder | _FrozenHeadExpertProcessModule = (
                SparseExpertDecoder(config, objective, expert_count)
            )
            _freeze_head(model)
        else:
            model = _FrozenHeadExpertProcessModule(config, objective)

        _send_json(
            connection,
            {
                "status": "ready_for_head",
                "worker_kind": worker_kind,
                "expert_index": expert_index,
            },
        )
        head_wire = connection.recv_bytes()
        _validate_wire_identity(
            head_wire,
            expected_sha256=init["frozen_head_wire_sha256"],
            expected_bytes=init["frozen_head_wire_bytes"],
            label="frozen head",
        )
        _load_head_state(model, _deserialize_tensors(head_wire))
        frozen_head_sha256 = tensor_sha256(_head_state(model))
        if frozen_head_sha256 != init["frozen_head_sha256"]:
            raise ValueError("frozen-head semantic identity mismatch")
        after_head_current, after_head_peak = _process_memory_bytes()
        model.train()
        forward_calls = 0
        _send_json(
            connection,
            {
                "status": "ready",
                "worker_kind": worker_kind,
                "expert_index": expert_index,
                "frozen_head_sha256": frozen_head_sha256,
                "startup_current_rss_bytes": startup_current,
                "startup_peak_rss_bytes": startup_peak,
                "after_head_current_rss_bytes": after_head_current,
                "after_head_peak_rss_bytes": after_head_peak,
            },
        )

        while True:
            control = _recv_child_json(
                connection,
                label="sparse-process control",
            )
            if control.get("op") == "shutdown":
                if frozenset(control) != frozenset({"op"}):
                    raise ValueError("sparse-process shutdown schema is invalid")
                current, peak = _process_memory_bytes()
                _send_json(
                    connection,
                    {
                        "status": "stopped",
                        "worker_kind": worker_kind,
                        "expert_index": expert_index,
                        "frozen_head_sha256": tensor_sha256(
                            _head_state(model)
                        ),
                        "forward_calls": forward_calls,
                        "current_rss_bytes": current,
                        "peak_rss_bytes": peak,
                    },
                )
                return

            _validate_run_control(
                control,
                worker_kind=worker_kind,
                expert_index=expert_index,
                campaign_revision_value=str(init["campaign_revision"]),
                dataset_revision=str(init["dataset_revision"]),
                frozen_head_sha256=frozen_head_sha256,
            )
            trainable_wire = connection.recv_bytes()
            input_wire = connection.recv_bytes()
            _validate_wire_identity(
                trainable_wire,
                expected_sha256=control["trainable_wire_sha256"],
                expected_bytes=control["trainable_wire_bytes"],
                label="trainable state",
            )
            _validate_wire_identity(
                input_wire,
                expected_sha256=control["input_wire_sha256"],
                expected_bytes=control["input_wire_bytes"],
                label="assignment input",
            )
            _load_trainable_state(
                model,
                _deserialize_tensors(trainable_wire),
                expected_sha256=control["trainable_state_sha256"],
            )
            input_payload = _deserialize_tensors(input_wire)
            total_tokens = int(control["total_tokens"])
            routes_sha256 = str(control["routes_sha256"])

            if worker_kind == "full":
                if frozenset(input_payload) != frozenset(
                    {"inputs", "targets"}
                ):
                    raise ValueError("full sparse input schema is invalid")
                inputs = _validate_token_batch(
                    input_payload["inputs"],
                    config=config,
                    label="inputs",
                )
                targets = _validate_token_batch(
                    input_payload["targets"],
                    config=config,
                    label="targets",
                )
                if inputs.shape != targets.shape or targets.numel() != total_tokens:
                    raise ValueError("full sparse batch shape is invalid")
            else:
                if expert_index is None:
                    raise AssertionError("expert process lacks an index")
                hidden, targets, _ = _validate_expert_input(
                    input_payload,
                    config=config,
                    expert_index=expert_index,
                    expert_count=expert_count,
                    total_tokens=total_tokens,
                    expected_routes_sha256=routes_sha256,
                )

            if tensor_sha256(_head_state(model)) != frozen_head_sha256:
                raise ValueError("cached frozen head changed before assignment")

            if control["op"] == "accept_then_wait":
                _send_json(
                    connection,
                    {
                        "status": "accepted",
                        "worker_kind": worker_kind,
                        "expert_index": expert_index,
                        "assignment_id": control["assignment_id"],
                        "cursor": control["cursor"],
                        "campaign_revision": control["campaign_revision"],
                        "dataset_revision": control["dataset_revision"],
                        "frozen_head_sha256": frozen_head_sha256,
                        "trainable_state_sha256": control[
                            "trainable_state_sha256"
                        ],
                        "input_wire_sha256": control["input_wire_sha256"],
                        "routes_sha256": routes_sha256,
                    },
                )
                connection.recv_bytes()
                raise RuntimeError("paused loss probe unexpectedly continued")

            started = time.perf_counter()
            if worker_kind == "full":
                if not isinstance(model, SparseExpertDecoder):
                    raise AssertionError("full worker model type is invalid")
                (
                    result,
                    loss,
                    routes,
                    routing_counts,
                    unconstrained_counts,
                    rerouted,
                    capacity,
                ) = _full_worker_result(
                    model,
                    inputs,
                    targets,
                    expert_count=expert_count,
                    router_aux_weight=float(init["router_aux_weight"]),
                )
                actual_routes_sha256 = tensor_sha256({"routes": routes})
                if actual_routes_sha256 != routes_sha256:
                    raise ValueError("full worker routing identity mismatch")
                routing_payload: dict[str, object] = {
                    "routing_counts": list(routing_counts),
                    "unconstrained_routing_counts": list(unconstrained_counts),
                    "capacity_rerouted_tokens": rerouted,
                    "routing_capacity": capacity,
                }
            else:
                if not isinstance(model, _FrozenHeadExpertProcessModule):
                    raise AssertionError("expert worker model type is invalid")
                result, loss = _expert_worker_result(
                    model,
                    hidden,
                    targets,
                    total_tokens=total_tokens,
                )
                routing_payload = {
                    "selected_token_count": int(hidden.shape[0])
                }
            compute_seconds = time.perf_counter() - started
            forward_calls += 1
            if tensor_sha256(_head_state(model)) != frozen_head_sha256:
                raise ValueError("cached frozen head changed during assignment")
            result_wire = _serialize_tensors(result)
            current, peak = _process_memory_bytes()
            _send_json(
                connection,
                {
                    "status": "completed",
                    "worker_kind": worker_kind,
                    "expert_index": expert_index,
                    "assignment_id": control["assignment_id"],
                    "cursor": control["cursor"],
                    "campaign_revision": control["campaign_revision"],
                    "dataset_revision": control["dataset_revision"],
                    "frozen_head_sha256": frozen_head_sha256,
                    "trainable_state_sha256": control[
                        "trainable_state_sha256"
                    ],
                    "input_wire_sha256": control["input_wire_sha256"],
                    "routes_sha256": routes_sha256,
                    "loss": float(loss.detach()),
                    "compute_seconds": compute_seconds,
                    "current_rss_bytes": current,
                    "peak_rss_bytes": peak,
                    "result_wire_sha256": _sha256_bytes(result_wire),
                    "result_wire_bytes": len(result_wire),
                    **routing_payload,
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
    expert_index: int | None,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_wire: bytes,
    head_sha256: str,
    timeout_seconds: float,
    name: str,
) -> _WorkerHandle:
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_sparse_process_worker_entry,
        args=(child_connection,),
        name=name,
        daemon=False,
    )
    started = time.perf_counter()
    process.start()
    child_connection.close()
    control_bytes = 0
    try:
        control_bytes += _send_json(
            parent_connection,
            {
                "format": _INIT_CONTROL_FORMAT,
                "op": "init",
                "worker_kind": worker_kind,
                "expert_index": expert_index,
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
        readiness, readiness_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label=f"{name} frozen-head readiness",
        )
        control_bytes += readiness_bytes
        if readiness != {
            "status": "ready_for_head",
            "worker_kind": worker_kind,
            "expert_index": expert_index,
        }:
            raise ValueError(f"{name} readiness acknowledgement is invalid")
        parent_connection.send_bytes(head_wire)
        ready, ready_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label=f"{name} initialization",
        )
        control_bytes += ready_bytes
        expected_ready = frozenset(
            {
                "status",
                "worker_kind",
                "expert_index",
                "frozen_head_sha256",
                "startup_current_rss_bytes",
                "startup_peak_rss_bytes",
                "after_head_current_rss_bytes",
                "after_head_peak_rss_bytes",
            }
        )
        if (
            frozenset(ready) != expected_ready
            or ready["status"] != "ready"
            or ready["worker_kind"] != worker_kind
            or ready["expert_index"] != expert_index
            or ready["frozen_head_sha256"] != head_sha256
        ):
            raise ValueError(f"{name} initialization acknowledgement is invalid")
        return _WorkerHandle(
            process=process,
            connection=parent_connection,
            worker_kind=worker_kind,
            expert_index=expert_index,
            initialization_seconds=time.perf_counter() - started,
            initialization_control_json_wire_bytes=control_bytes,
            ready=ready,
        )
    except BaseException:
        parent_connection.close()
        process.terminate()
        process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
        raise


def _terminate_worker(
    worker: _WorkerHandle,
    timeout_seconds: float,
) -> int:
    worker.connection.close()
    worker.process.terminate()
    worker.process.join(timeout_seconds)
    if worker.process.is_alive():
        worker.process.kill()
        worker.process.join(timeout_seconds)
    if worker.process.is_alive() or worker.process.exitcode is None:
        raise RuntimeError("failed to terminate sparse-process worker")
    return int(worker.process.exitcode)


def _stop_worker(
    worker: _WorkerHandle,
    timeout_seconds: float,
    *,
    trainable_state_transmissions: int,
) -> SparseProcessWorkerEvidence:
    acknowledgement: Mapping[str, object] | None = None
    shutdown_control_bytes = 0
    try:
        shutdown_control_bytes += _send_json(
            worker.connection,
            {"op": "shutdown"},
        )
        acknowledgement, ack_bytes = _recv_json(
            worker.connection,
            timeout_seconds,
            label="sparse-process shutdown",
        )
        shutdown_control_bytes += ack_bytes
    finally:
        worker.connection.close()
        worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.terminate()
            worker.process.join(timeout_seconds)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout_seconds)
    if (
        worker.process.exitcode != 0
        or acknowledgement is None
        or frozenset(acknowledgement)
        != frozenset(
            {
                "status",
                "worker_kind",
                "expert_index",
                "frozen_head_sha256",
                "forward_calls",
                "current_rss_bytes",
                "peak_rss_bytes",
            }
        )
        or acknowledgement["status"] != "stopped"
        or acknowledgement["worker_kind"] != worker.worker_kind
        or acknowledgement["expert_index"] != worker.expert_index
        or acknowledgement["frozen_head_sha256"]
        != worker.ready["frozen_head_sha256"]
    ):
        raise ValueError("sparse-process shutdown acknowledgement is invalid")
    return SparseProcessWorkerEvidence(
        worker_kind=worker.worker_kind,
        expert_index=worker.expert_index,
        initialization_seconds=worker.initialization_seconds,
        initialization_control_json_wire_bytes=(
            worker.initialization_control_json_wire_bytes
        ),
        shutdown_control_json_wire_bytes=shutdown_control_bytes,
        frozen_head_transmissions=1,
        trainable_state_transmissions=trainable_state_transmissions,
        forward_calls=int(acknowledgement["forward_calls"]),
        worker_startup_current_rss_bytes=int(
            worker.ready["startup_current_rss_bytes"]
        ),
        worker_startup_peak_rss_bytes=int(
            worker.ready["startup_peak_rss_bytes"]
        ),
        worker_after_head_current_rss_bytes=int(
            worker.ready["after_head_current_rss_bytes"]
        ),
        worker_after_head_peak_rss_bytes=int(
            worker.ready["after_head_peak_rss_bytes"]
        ),
        worker_final_current_rss_bytes=int(
            acknowledgement["current_rss_bytes"]
        ),
        worker_final_peak_rss_bytes=int(acknowledgement["peak_rss_bytes"]),
        frozen_head_sha256=str(acknowledgement["frozen_head_sha256"]),
        child_exit_code=int(worker.process.exitcode),
    )


def _assignment_control(
    *,
    operation: str,
    worker: _WorkerHandle,
    prepared: _PreparedAssignment,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
    trainable_state_sha256: str,
    trainable_wire: bytes,
    input_wire: bytes,
) -> dict[str, object]:
    return {
        "format": _RUN_CONTROL_FORMAT,
        "op": operation,
        "worker_kind": worker.worker_kind,
        "expert_index": worker.expert_index,
        "assignment_id": prepared.assignment_id,
        "cursor": prepared.cursor,
        "campaign_revision": campaign_revision_value,
        "dataset_revision": dataset_revision,
        "frozen_head_sha256": head_sha256,
        "trainable_state_sha256": trainable_state_sha256,
        "trainable_wire_sha256": _sha256_bytes(trainable_wire),
        "trainable_wire_bytes": len(trainable_wire),
        "input_wire_sha256": _sha256_bytes(input_wire),
        "input_wire_bytes": len(input_wire),
        "routes_sha256": prepared.routes_sha256,
        "total_tokens": int(prepared.targets.numel()),
    }


def _validate_completed_acknowledgement(
    acknowledgement: Mapping[str, object],
    *,
    worker: _WorkerHandle,
    prepared: _PreparedAssignment,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
    trainable_state_sha256: str,
    input_wire: bytes,
    result_wire: bytes,
) -> None:
    common = {
        "status",
        "worker_kind",
        "expert_index",
        "assignment_id",
        "cursor",
        "campaign_revision",
        "dataset_revision",
        "frozen_head_sha256",
        "trainable_state_sha256",
        "input_wire_sha256",
        "routes_sha256",
        "loss",
        "compute_seconds",
        "current_rss_bytes",
        "peak_rss_bytes",
        "result_wire_sha256",
        "result_wire_bytes",
    }
    expected = common | (
        {
            "routing_counts",
            "unconstrained_routing_counts",
            "capacity_rerouted_tokens",
            "routing_capacity",
        }
        if worker.worker_kind == "full"
        else {"selected_token_count"}
    )
    if (
        frozenset(acknowledgement) != frozenset(expected)
        or acknowledgement["status"] != "completed"
        or acknowledgement["worker_kind"] != worker.worker_kind
        or acknowledgement["expert_index"] != worker.expert_index
        or acknowledgement["assignment_id"] != prepared.assignment_id
        or acknowledgement["cursor"] != prepared.cursor
        or acknowledgement["campaign_revision"] != campaign_revision_value
        or acknowledgement["dataset_revision"] != dataset_revision
        or acknowledgement["frozen_head_sha256"] != head_sha256
        or acknowledgement["trainable_state_sha256"]
        != trainable_state_sha256
        or acknowledgement["input_wire_sha256"] != _sha256_bytes(input_wire)
        or acknowledgement["routes_sha256"] != prepared.routes_sha256
        or type(acknowledgement["loss"]) is not float
        or not math.isfinite(float(acknowledgement["loss"]))
        or type(acknowledgement["compute_seconds"]) is not float
        or float(acknowledgement["compute_seconds"]) <= 0
        or acknowledgement["result_wire_bytes"] != len(result_wire)
        or acknowledgement["result_wire_sha256"] != _sha256_bytes(result_wire)
    ):
        raise ValueError("sparse-process completion acknowledgement is invalid")
    if worker.worker_kind == "full":
        if (
            tuple(acknowledgement["routing_counts"])
            != prepared.routing_counts
            or tuple(acknowledgement["unconstrained_routing_counts"])
            != prepared.unconstrained_routing_counts
            or acknowledgement["capacity_rerouted_tokens"]
            != prepared.capacity_rerouted_tokens
            or acknowledgement["routing_capacity"]
            != prepared.routing_capacity
        ):
            raise ValueError("full sparse-process routing acknowledgement is invalid")
    else:
        expert_index = worker.expert_index
        if expert_index is None or acknowledgement[
            "selected_token_count"
        ] != prepared.routing_counts[expert_index]:
            raise ValueError(
                "expert sparse-process selection acknowledgement is invalid"
            )


def _run_worker_assignment(
    worker: _WorkerHandle,
    prepared: _PreparedAssignment,
    *,
    campaign_revision_value: str,
    dataset_revision: str,
    head_sha256: str,
    trainable_state_sha256: str,
    trainable_wire: bytes,
    input_wire: bytes,
    timeout_seconds: float,
) -> _CollectedResult:
    control = _assignment_control(
        operation="run",
        worker=worker,
        prepared=prepared,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_sha256=head_sha256,
        trainable_state_sha256=trainable_state_sha256,
        trainable_wire=trainable_wire,
        input_wire=input_wire,
    )
    started = time.perf_counter()
    control_bytes = _send_json(worker.connection, control)
    worker.connection.send_bytes(trainable_wire)
    worker.connection.send_bytes(input_wire)
    acknowledgement, ack_bytes = _recv_json(
        worker.connection,
        timeout_seconds,
        label="sparse-process completion",
    )
    control_bytes += ack_bytes
    result_wire = _recv_bytes(
        worker.connection,
        timeout_seconds,
        label="sparse-process result",
    )
    elapsed = time.perf_counter() - started
    _validate_completed_acknowledgement(
        acknowledgement,
        worker=worker,
        prepared=prepared,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_sha256=head_sha256,
        trainable_state_sha256=trainable_state_sha256,
        input_wire=input_wire,
        result_wire=result_wire,
    )
    return _CollectedResult(
        acknowledgement=acknowledgement,
        result_wire=result_wire,
        control_json_wire_bytes=control_bytes,
        round_trip_seconds=elapsed,
    )


def _prepare_assignments(
    campaign: CampaignConfig,
    dataset: PackedDataset,
    *,
    expert_count: int,
) -> tuple[_PreparedAssignment, ...]:
    cursors = (
        0,
        campaign.training.batch_size % campaign.training.dataset_sequences,
    )
    prepared: list[_PreparedAssignment] = []
    for assignment_id, cursor in enumerate(cursors):
        model = _build_sparse_model(campaign, expert_count)
        _freeze_head(model)
        inputs, targets = fixture_batch(campaign, cursor, dataset)
        with torch.no_grad():
            hidden = model.shared_hidden(inputs)
            hidden_flat = hidden.reshape(-1, campaign.model.width)
            router_logits = model.router(hidden_flat)
            (
                routes,
                routing_counts,
                unconstrained_counts,
                rerouted,
                capacity,
            ) = _balanced_top1_routes(router_logits, expert_count)
        routes = routes.detach().clone()
        targets_flat = targets.reshape(-1)
        routes_sha256 = tensor_sha256({"routes": routes})
        expert_wires: list[bytes] = []
        for expert_index in range(expert_count):
            positions = torch.nonzero(
                routes == expert_index,
                as_tuple=False,
            ).flatten()
            expert_wires.append(
                _serialize_tensors(
                    {
                        "hidden": hidden_flat[positions].detach().clone(),
                        "targets": targets_flat[positions].detach().clone(),
                        "positions": positions.detach().clone(),
                        "routes": routes,
                    }
                )
            )
        prepared.append(
            _PreparedAssignment(
                assignment_id=assignment_id,
                cursor=cursor,
                inputs=inputs.detach().clone(),
                targets=targets.detach().clone(),
                routes=routes,
                routing_counts=routing_counts,
                unconstrained_routing_counts=unconstrained_counts,
                capacity_rerouted_tokens=rerouted,
                routing_capacity=capacity,
                routes_sha256=routes_sha256,
                full_input_wire=_serialize_tensors(
                    {
                        "inputs": inputs,
                        "targets": targets,
                    }
                ),
                expert_input_wires=tuple(expert_wires),
            )
        )
    return tuple(prepared)


def _run_reference_step(
    campaign: CampaignConfig,
    prepared: _PreparedAssignment,
    *,
    expert_count: int,
    router_aux_weight: float,
    expected_head_sha256: str,
) -> _StepSnapshot:
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    if tensor_sha256(_head_tensor_snapshot(model)) != expected_head_sha256:
        raise AssertionError("reference frozen head identity changed")
    optimizer = _create_optimizer(model, campaign.training)
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
        prepared.inputs,
        prepared.targets,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
    )
    if (
        not torch.equal(routes, prepared.routes)
        or counts != prepared.routing_counts
        or unconstrained != prepared.unconstrained_routing_counts
        or rerouted != prepared.capacity_rerouted_tokens
        or capacity != prepared.routing_capacity
    ):
        raise AssertionError("reference routing changed after preparation")
    raw = _trainable_gradient_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    clipped = _trainable_gradient_snapshot(model)
    optimizer.step()
    frozen_head_sha256 = tensor_sha256(_head_tensor_snapshot(model))
    if frozen_head_sha256 != expected_head_sha256:
        raise AssertionError("reference optimizer changed the frozen head")
    return _StepSnapshot(
        loss=float(loss.detach()),
        raw_gradients=raw,
        clipped_gradients=clipped,
        optimizer=_trainable_optimizer_tensor_snapshot(model, optimizer),
        model=_model_snapshot(model),
        frozen_head_sha256=frozen_head_sha256,
    )


def _apply_full_process_result(
    campaign: CampaignConfig,
    result: _CollectedResult,
    *,
    expert_count: int,
    expected_head_sha256: str,
) -> _StepSnapshot:
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    optimizer = _create_optimizer(model, campaign.training)
    tensors = _deserialize_tensors(result.result_wire)
    expected = {
        f"gradient.{name}": parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if frozenset(tensors) != frozenset(expected):
        raise ValueError("full process gradient result schema is invalid")
    optimizer.zero_grad(set_to_none=True)
    for result_name, parameter in expected.items():
        gradient = tensors[result_name]
        if (
            gradient.dtype != parameter.dtype
            or gradient.shape != parameter.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(f"full process gradient is invalid: {result_name}")
        parameter.grad = gradient.detach().clone()
    raw = _trainable_gradient_snapshot(model)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        campaign.training.max_gradient_norm,
    )
    clipped = _trainable_gradient_snapshot(model)
    optimizer.step()
    frozen_head_sha256 = tensor_sha256(_head_tensor_snapshot(model))
    if frozen_head_sha256 != expected_head_sha256:
        raise AssertionError("full process update changed the frozen head")
    return _StepSnapshot(
        loss=float(result.acknowledgement["loss"]),
        raw_gradients=raw,
        clipped_gradients=clipped,
        optimizer=_trainable_optimizer_tensor_snapshot(model, optimizer),
        model=_model_snapshot(model),
        frozen_head_sha256=frozen_head_sha256,
    )


def _apply_expert_process_results(
    campaign: CampaignConfig,
    prepared: _PreparedAssignment,
    results: tuple[_CollectedResult, ...],
    *,
    expert_count: int,
    router_aux_weight: float,
    expected_head_sha256: str,
) -> _StepSnapshot:
    if len(results) != expert_count:
        raise ValueError("expert process result count is invalid")
    model = _build_sparse_model(campaign, expert_count)
    _freeze_head(model)
    optimizer = _create_optimizer(model, campaign.training)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    hidden = model.shared_hidden(prepared.inputs)
    hidden_flat = hidden.reshape(-1, campaign.model.width)
    router_logits = model.router(hidden_flat)
    (
        routes,
        counts,
        unconstrained,
        rerouted,
        capacity,
    ) = _balanced_top1_routes(router_logits, expert_count)
    if (
        not torch.equal(routes, prepared.routes)
        or counts != prepared.routing_counts
        or unconstrained != prepared.unconstrained_routing_counts
        or rerouted != prepared.capacity_rerouted_tokens
        or capacity != prepared.routing_capacity
    ):
        raise AssertionError("expert coordinator routing changed")
    auxiliary = _router_auxiliary_loss(
        router_logits,
        routes,
        expert_count,
        router_aux_weight,
    )
    auxiliary.backward(retain_graph=True)
    process_losses: list[Tensor] = []
    for expert_index, collected in enumerate(results):
        tensors = _deserialize_tensors(collected.result_wire)
        expert = model.experts[expert_index]
        expected_gradients = {
            f"gradient.expert.{name}": parameter
            for name, parameter in expert.named_parameters()
        }
        if frozenset(tensors) != frozenset(
            {"input_adjoint", *expected_gradients}
        ):
            raise ValueError(
                f"expert {expert_index} process result schema is invalid"
            )
        positions = torch.nonzero(
            routes == expert_index,
            as_tuple=False,
        ).flatten()
        selected_hidden = hidden_flat[positions].detach().clone()
        expected_input_wire = _serialize_tensors(
            {
                "hidden": selected_hidden,
                "targets": prepared.targets.reshape(-1)[positions],
                "positions": positions,
                "routes": routes,
            }
        )
        if expected_input_wire != prepared.expert_input_wires[expert_index]:
            raise AssertionError(
                f"expert {expert_index} authenticated input changed"
            )
        input_adjoint = tensors["input_adjoint"]
        if (
            input_adjoint.dtype != selected_hidden.dtype
            or input_adjoint.shape != selected_hidden.shape
            or not bool(torch.isfinite(input_adjoint).all())
        ):
            raise ValueError(
                f"expert {expert_index} input adjoint is invalid"
            )
        for result_name, parameter in expected_gradients.items():
            gradient = tensors[result_name]
            if (
                gradient.dtype != parameter.dtype
                or gradient.shape != parameter.shape
                or not bool(torch.isfinite(gradient).all())
            ):
                raise ValueError(
                    f"expert {expert_index} gradient is invalid: {result_name}"
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
                float(collected.acknowledgement["loss"]),
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
    optimizer.step()
    frozen_head_sha256 = tensor_sha256(_head_tensor_snapshot(model))
    if frozen_head_sha256 != expected_head_sha256:
        raise AssertionError("expert process update changed the frozen head")
    return _StepSnapshot(
        loss=float(loss),
        raw_gradients=raw,
        clipped_gradients=clipped,
        optimizer=_trainable_optimizer_tensor_snapshot(model, optimizer),
        model=_model_snapshot(model),
        frozen_head_sha256=frozen_head_sha256,
    )


def _run_recovery_control(
    context: multiprocessing.context.BaseContext,
    campaign: CampaignConfig,
    prepared: _PreparedAssignment,
    stable_result: _CollectedResult,
    *,
    expert_index: int,
    expert_count: int,
    router_aux_weight: float,
    campaign_revision_value: str,
    dataset_revision: str,
    head_wire: bytes,
    head_sha256: str,
    trainable_state_sha256: str,
    trainable_wire: bytes,
    input_wire: bytes,
    timeout_seconds: float,
) -> tuple[
    _CollectedResult,
    SparseProcessRecoveryEvidence,
]:
    first: _WorkerHandle | None = None
    replacement: _WorkerHandle | None = None
    first_exit_code: int | None = None
    replacement_evidence: SparseProcessWorkerEvidence | None = None
    accepted_control_bytes = 0
    error: BaseException | None = None
    try:
        first = _start_worker(
            context,
            campaign,
            worker_kind="expert",
            expert_index=expert_index,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            name="orcacolony-sparse-expert-before-loss",
        )
        control = _assignment_control(
            operation="accept_then_wait",
            worker=first,
            prepared=prepared,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_sha256=head_sha256,
            trainable_state_sha256=trainable_state_sha256,
            trainable_wire=trainable_wire,
            input_wire=input_wire,
        )
        accepted_control_bytes += _send_json(first.connection, control)
        first.connection.send_bytes(trainable_wire)
        first.connection.send_bytes(input_wire)
        accepted, ack_bytes = _recv_json(
            first.connection,
            timeout_seconds,
            label="expert assignment acceptance before loss",
        )
        accepted_control_bytes += ack_bytes
        expected_accepted = {
            "status": "accepted",
            "worker_kind": "expert",
            "expert_index": expert_index,
            "assignment_id": prepared.assignment_id,
            "cursor": prepared.cursor,
            "campaign_revision": campaign_revision_value,
            "dataset_revision": dataset_revision,
            "frozen_head_sha256": head_sha256,
            "trainable_state_sha256": trainable_state_sha256,
            "input_wire_sha256": _sha256_bytes(input_wire),
            "routes_sha256": prepared.routes_sha256,
        }
        if accepted != expected_accepted:
            raise ValueError("lost expert assignment acceptance is invalid")
        first_initialization_seconds = first.initialization_seconds
        first_initialization_control = (
            first.initialization_control_json_wire_bytes
        )
        first_exit_code = _terminate_worker(first, timeout_seconds)
        first = None

        recovery_started = time.perf_counter()
        replacement = _start_worker(
            context,
            campaign,
            worker_kind="expert",
            expert_index=expert_index,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            name="orcacolony-sparse-expert-replacement",
        )
        replacement_initialization_seconds = replacement.initialization_seconds
        replacement_result = _run_worker_assignment(
            replacement,
            prepared,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_sha256=head_sha256,
            trainable_state_sha256=trainable_state_sha256,
            trainable_wire=trainable_wire,
            input_wire=input_wire,
            timeout_seconds=timeout_seconds,
        )
        replacement_evidence = _stop_worker(
            replacement,
            timeout_seconds,
            trainable_state_transmissions=1,
        )
        replacement = None
        recovery_seconds = time.perf_counter() - recovery_started
        stable_sha256 = _sha256_bytes(stable_result.result_wire)
        replacement_sha256 = _sha256_bytes(replacement_result.result_wire)
        if replacement_result.result_wire != stable_result.result_wire:
            raise ValueError(
                "replacement expert result is not byte-identical"
            )
        recovery = SparseProcessRecoveryEvidence(
            expert_index=expert_index,
            assignment_id=prepared.assignment_id,
            assignment_accepted_before_loss=True,
            first_worker_exit_code=int(first_exit_code),
            replacement_worker_exit_code=(
                replacement_evidence.child_exit_code
            ),
            replacement_result_matches_stable=True,
            replacement_result_used_in_canonical_update=True,
            stable_result_wire_sha256=stable_sha256,
            replacement_result_wire_sha256=replacement_sha256,
            lost_worker_received_tensor_wire_bytes=(
                len(head_wire) + len(trainable_wire) + len(input_wire)
            ),
            recovery_retransmitted_tensor_wire_bytes=(
                len(head_wire) + len(trainable_wire) + len(input_wire)
            ),
            replacement_result_tensor_wire_bytes=len(
                replacement_result.result_wire
            ),
            recovery_control_json_wire_bytes=(
                first_initialization_control
                + accepted_control_bytes
                + replacement_evidence.initialization_control_json_wire_bytes
                + replacement_result.control_json_wire_bytes
                + replacement_evidence.shutdown_control_json_wire_bytes
            ),
            recovery_total_application_wire_bytes=(
                len(head_wire)
                + len(trainable_wire)
                + len(input_wire)
                + len(head_wire)
                + len(trainable_wire)
                + len(input_wire)
                + len(replacement_result.result_wire)
                + first_initialization_control
                + accepted_control_bytes
                + replacement_evidence.initialization_control_json_wire_bytes
                + replacement_result.control_json_wire_bytes
                + replacement_evidence.shutdown_control_json_wire_bytes
            ),
            recovery_seconds=recovery_seconds,
            first_worker_initialization_seconds=(
                first_initialization_seconds
            ),
            replacement_worker_initialization_seconds=(
                replacement_initialization_seconds
            ),
            replacement_worker_peak_rss_bytes=(
                replacement_evidence.worker_final_peak_rss_bytes
            ),
            replacement_frozen_head_sha256=(
                replacement_evidence.frozen_head_sha256
            ),
        )
        return replacement_result, recovery
    except BaseException as exc:
        error = exc
    finally:
        if first is not None:
            _terminate_worker(first, timeout_seconds)
        if replacement is not None:
            _terminate_worker(replacement, timeout_seconds)
    if error is not None:
        raise error
    raise AssertionError("expert recovery control did not complete")


def _validate_experiment_inputs(
    campaign: CampaignConfig,
    dataset: PackedDataset | None,
    *,
    expert_count: int,
    router_aux_weight: float,
    timeout_seconds: float,
) -> tuple[PackedDataset, float, float]:
    validate_dataset_artifacts(campaign, dataset)
    if campaign.dataset is None or dataset is None:
        raise ValueError(
            "authenticated sparse-process control requires dataset artifacts"
        )
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
    return dataset, float(router_aux_weight), float(timeout_seconds)


def run_authenticated_sparse_process_experiment(
    campaign: CampaignConfig,
    *,
    dataset: PackedDataset | None,
    expert_count: int = 4,
    router_aux_weight: float = 0.01,
    timeout_seconds: float = 120.0,
) -> AuthenticatedSparseProcessEvidence:
    dataset, router_aux_weight, timeout_seconds = (
        _validate_experiment_inputs(
            campaign,
            dataset,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            timeout_seconds=timeout_seconds,
        )
    )
    campaign_revision_value = campaign_revision(campaign)
    dataset_revision = dataset.revision
    source = _build_sparse_model(campaign, expert_count)
    _freeze_head(source)
    head_state = _head_tensor_snapshot(source)
    head_sha256 = tensor_sha256(head_state)
    head_wire = _serialize_tensors(head_state)
    full_trainable_state = _trainable_state(source)
    full_trainable_state_sha256 = tensor_sha256(full_trainable_state)
    full_trainable_wire = _serialize_tensors(full_trainable_state)
    expert_trainable_states = tuple(
        {
            f"expert.{name}": parameter.detach().clone()
            for name, parameter in expert.named_parameters()
        }
        for expert in source.experts
    )
    expert_trainable_state_sha256 = tuple(
        tensor_sha256(state) for state in expert_trainable_states
    )
    expert_trainable_wires = tuple(
        _serialize_tensors(state) for state in expert_trainable_states
    )
    prepared = _prepare_assignments(
        campaign,
        dataset,
        expert_count=expert_count,
    )
    context = multiprocessing.get_context("spawn")

    full_results: list[_CollectedResult] = []
    full_worker_evidence: SparseProcessWorkerEvidence | None = None
    full_worker: _WorkerHandle | None = None
    try:
        full_worker = _start_worker(
            context,
            campaign,
            worker_kind="full",
            expert_index=None,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            campaign_revision_value=campaign_revision_value,
            dataset_revision=dataset_revision,
            head_wire=head_wire,
            head_sha256=head_sha256,
            timeout_seconds=timeout_seconds,
            name="orcacolony-sparse-full-worker",
        )
        for assignment in prepared:
            full_results.append(
                _run_worker_assignment(
                    full_worker,
                    assignment,
                    campaign_revision_value=campaign_revision_value,
                    dataset_revision=dataset_revision,
                    head_sha256=head_sha256,
                    trainable_state_sha256=full_trainable_state_sha256,
                    trainable_wire=full_trainable_wire,
                    input_wire=assignment.full_input_wire,
                    timeout_seconds=timeout_seconds,
                )
            )
        full_worker_evidence = _stop_worker(
            full_worker,
            timeout_seconds,
            trainable_state_transmissions=len(prepared),
        )
        full_worker = None
    finally:
        if full_worker is not None:
            _terminate_worker(full_worker, timeout_seconds)

    expert_results: list[list[_CollectedResult | None]] = [
        [None] * expert_count for _ in prepared
    ]
    expert_worker_evidence: list[SparseProcessWorkerEvidence] = []
    for expert_index in range(expert_count):
        worker: _WorkerHandle | None = None
        try:
            worker = _start_worker(
                context,
                campaign,
                worker_kind="expert",
                expert_index=expert_index,
                expert_count=expert_count,
                router_aux_weight=router_aux_weight,
                campaign_revision_value=campaign_revision_value,
                dataset_revision=dataset_revision,
                head_wire=head_wire,
                head_sha256=head_sha256,
                timeout_seconds=timeout_seconds,
                name=f"orcacolony-sparse-expert-{expert_index}",
            )
            for assignment in prepared:
                expert_results[assignment.assignment_id][expert_index] = (
                    _run_worker_assignment(
                        worker,
                        assignment,
                        campaign_revision_value=campaign_revision_value,
                        dataset_revision=dataset_revision,
                        head_sha256=head_sha256,
                        trainable_state_sha256=(
                            expert_trainable_state_sha256[expert_index]
                        ),
                        trainable_wire=expert_trainable_wires[expert_index],
                        input_wire=assignment.expert_input_wires[
                            expert_index
                        ],
                        timeout_seconds=timeout_seconds,
                    )
                )
            expert_worker_evidence.append(
                _stop_worker(
                    worker,
                    timeout_seconds,
                    trainable_state_transmissions=len(prepared),
                )
            )
            worker = None
        finally:
            if worker is not None:
                _terminate_worker(worker, timeout_seconds)

    stable_recovery_result = expert_results[0][0]
    if stable_recovery_result is None:
        raise AssertionError("stable recovery result is absent")
    replacement_result, recovery = _run_recovery_control(
        context,
        campaign,
        prepared[0],
        stable_recovery_result,
        expert_index=0,
        expert_count=expert_count,
        router_aux_weight=router_aux_weight,
        campaign_revision_value=campaign_revision_value,
        dataset_revision=dataset_revision,
        head_wire=head_wire,
        head_sha256=head_sha256,
        trainable_state_sha256=expert_trainable_state_sha256[0],
        trainable_wire=expert_trainable_wires[0],
        input_wire=prepared[0].expert_input_wires[0],
        timeout_seconds=timeout_seconds,
    )
    canonical_expert_results = [list(results) for results in expert_results]
    canonical_expert_results[0][0] = replacement_result

    assignment_evidence: list[SparseProcessAssignmentEvidence] = []
    for assignment in prepared:
        full_result = full_results[assignment.assignment_id]
        matched_experts = expert_results[assignment.assignment_id]
        canonical_experts = canonical_expert_results[
            assignment.assignment_id
        ]
        if any(result is None for result in matched_experts) or any(
            result is None for result in canonical_experts
        ):
            raise AssertionError("expert process result matrix is incomplete")
        matched_expert_tuple = tuple(
            result
            for result in matched_experts
            if result is not None
        )
        canonical_expert_tuple = tuple(
            result
            for result in canonical_experts
            if result is not None
        )
        reference = _run_reference_step(
            campaign,
            assignment,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            expected_head_sha256=head_sha256,
        )
        full_process = _apply_full_process_result(
            campaign,
            full_result,
            expert_count=expert_count,
            expected_head_sha256=head_sha256,
        )
        expert_process = _apply_expert_process_results(
            campaign,
            assignment,
            canonical_expert_tuple,
            expert_count=expert_count,
            router_aux_weight=router_aux_weight,
            expected_head_sha256=head_sha256,
        )
        full_tensor_wire_bytes = (
            len(full_trainable_wire)
            + len(assignment.full_input_wire)
            + len(full_result.result_wire)
        )
        expert_state_wire_bytes = tuple(
            len(wire) for wire in expert_trainable_wires
        )
        expert_input_wire_bytes = tuple(
            len(wire) for wire in assignment.expert_input_wires
        )
        expert_result_wire_bytes = tuple(
            len(result.result_wire) for result in matched_expert_tuple
        )
        expert_control_bytes = tuple(
            result.control_json_wire_bytes
            for result in matched_expert_tuple
        )
        expert_tensor_wire_bytes = (
            sum(expert_state_wire_bytes)
            + sum(expert_input_wire_bytes)
            + sum(expert_result_wire_bytes)
        )
        assignment_evidence.append(
            SparseProcessAssignmentEvidence(
                assignment_id=assignment.assignment_id,
                cursor=assignment.cursor,
                total_tokens=int(assignment.targets.numel()),
                routing_capacity=assignment.routing_capacity,
                routing_counts=assignment.routing_counts,
                unconstrained_routing_counts=(
                    assignment.unconstrained_routing_counts
                ),
                capacity_rerouted_tokens=(
                    assignment.capacity_rerouted_tokens
                ),
                routes_sha256=assignment.routes_sha256,
                full_trainable_state_wire_bytes=len(
                    full_trainable_wire
                ),
                full_input_wire_bytes=len(assignment.full_input_wire),
                full_gradient_result_wire_bytes=len(
                    full_result.result_wire
                ),
                full_tensor_wire_bytes=full_tensor_wire_bytes,
                full_control_json_wire_bytes=(
                    full_result.control_json_wire_bytes
                ),
                full_total_application_wire_bytes=(
                    full_tensor_wire_bytes
                    + full_result.control_json_wire_bytes
                ),
                expert_trainable_state_wire_bytes=(
                    expert_state_wire_bytes
                ),
                expert_input_wire_bytes=expert_input_wire_bytes,
                expert_result_wire_bytes=expert_result_wire_bytes,
                expert_aggregate_tensor_wire_bytes=(
                    expert_tensor_wire_bytes
                ),
                expert_control_json_wire_bytes=expert_control_bytes,
                expert_aggregate_control_json_wire_bytes=sum(
                    expert_control_bytes
                ),
                expert_total_application_wire_bytes=(
                    expert_tensor_wire_bytes + sum(expert_control_bytes)
                ),
                centralized_loss=reference.loss,
                full_process_loss=full_process.loss,
                expert_process_loss=expert_process.loss,
                full_max_abs_raw_gradient_difference=(
                    _max_abs_difference(
                        reference.raw_gradients,
                        full_process.raw_gradients,
                    )
                ),
                full_max_abs_clipped_gradient_difference=(
                    _max_abs_difference(
                        reference.clipped_gradients,
                        full_process.clipped_gradients,
                    )
                ),
                full_max_abs_model_difference=(
                    _max_abs_difference(reference.model, full_process.model)
                ),
                expert_max_abs_raw_gradient_difference=(
                    _max_abs_difference(
                        reference.raw_gradients,
                        expert_process.raw_gradients,
                    )
                ),
                expert_max_abs_clipped_gradient_difference=(
                    _max_abs_difference(
                        reference.clipped_gradients,
                        expert_process.clipped_gradients,
                    )
                ),
                expert_max_abs_model_difference=(
                    _max_abs_difference(
                        reference.model,
                        expert_process.model,
                    )
                ),
                centralized_raw_gradient_sha256=tensor_sha256(
                    reference.raw_gradients
                ),
                full_process_raw_gradient_sha256=tensor_sha256(
                    full_process.raw_gradients
                ),
                expert_process_raw_gradient_sha256=tensor_sha256(
                    expert_process.raw_gradients
                ),
                centralized_clipped_gradient_sha256=tensor_sha256(
                    reference.clipped_gradients
                ),
                full_process_clipped_gradient_sha256=tensor_sha256(
                    full_process.clipped_gradients
                ),
                expert_process_clipped_gradient_sha256=tensor_sha256(
                    expert_process.clipped_gradients
                ),
                centralized_optimizer_sha256=tensor_sha256(
                    reference.optimizer
                ),
                full_process_optimizer_sha256=tensor_sha256(
                    full_process.optimizer
                ),
                expert_process_optimizer_sha256=tensor_sha256(
                    expert_process.optimizer
                ),
                centralized_model_sha256=tensor_sha256(reference.model),
                full_process_model_sha256=tensor_sha256(
                    full_process.model
                ),
                expert_process_model_sha256=tensor_sha256(
                    expert_process.model
                ),
                full_round_trip_seconds=full_result.round_trip_seconds,
                full_worker_compute_seconds=float(
                    full_result.acknowledgement["compute_seconds"]
                ),
                expert_round_trip_seconds=tuple(
                    result.round_trip_seconds
                    for result in matched_expert_tuple
                ),
                expert_worker_compute_seconds=tuple(
                    float(result.acknowledgement["compute_seconds"])
                    for result in matched_expert_tuple
                ),
                full_worker_peak_rss_bytes=int(
                    full_result.acknowledgement["peak_rss_bytes"]
                ),
                expert_worker_peak_rss_bytes=tuple(
                    int(result.acknowledgement["peak_rss_bytes"])
                    for result in matched_expert_tuple
                ),
            )
        )

    if full_worker_evidence is None:
        raise AssertionError("full sparse worker evidence is absent")
    assignments = tuple(assignment_evidence)
    expert_workers = tuple(expert_worker_evidence)
    full_cold_tensor = (
        len(head_wire) + assignments[0].full_tensor_wire_bytes
    )
    full_warm_tensor = assignments[1].full_tensor_wire_bytes
    expert_cold_tensor = (
        expert_count * len(head_wire)
        + assignments[0].expert_aggregate_tensor_wire_bytes
    )
    expert_warm_tensor = assignments[1].expert_aggregate_tensor_wire_bytes
    full_cold_application = (
        full_cold_tensor
        + full_worker_evidence.initialization_control_json_wire_bytes
        + assignments[0].full_control_json_wire_bytes
    )
    full_warm_application = (
        full_warm_tensor + assignments[1].full_control_json_wire_bytes
    )
    expert_cold_application = (
        expert_cold_tensor
        + sum(
            worker.initialization_control_json_wire_bytes
            for worker in expert_workers
        )
        + assignments[0].expert_aggregate_control_json_wire_bytes
    )
    expert_warm_application = (
        expert_warm_tensor
        + assignments[1].expert_aggregate_control_json_wire_bytes
    )
    return AuthenticatedSparseProcessEvidence(
        format="orcacolony_authenticated_sparse_process_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        campaign_revision=campaign_revision_value,
        dataset_revision=dataset_revision,
        authentication_mode=_AUTHENTICATION_MODE,
        transport_scope=_TRANSPORT_SCOPE,
        start_method=context.get_start_method(),
        process_scheduling="sequential-workers-persistent-two-assignments",
        assignment_state_mode=(
            "independent-one-step-controls-from-identical-initialization"
        ),
        wire_accounting_scope=(
            "serialized-safetensors-and-json-payload-bytes-excludes-pipe-framing"
        ),
        memory_scope="per-child-process-rss-not-concurrent-or-aggregate",
        matched_totals_exclude_recovery=True,
        maximum_simultaneous_worker_processes=1,
        expert_count=expert_count,
        assignment_count=len(assignments),
        frozen_head_sha256=head_sha256,
        frozen_head_wire_sha256=_sha256_bytes(head_wire),
        frozen_head_wire_bytes=len(head_wire),
        full_trainable_state_sha256=full_trainable_state_sha256,
        full_trainable_state_wire_bytes=len(full_trainable_wire),
        expert_trainable_state_sha256=expert_trainable_state_sha256,
        expert_trainable_state_wire_bytes=tuple(
            len(wire) for wire in expert_trainable_wires
        ),
        full_frozen_head_transmissions=(
            full_worker_evidence.frozen_head_transmissions
        ),
        expert_frozen_head_transmissions=sum(
            worker.frozen_head_transmissions for worker in expert_workers
        ),
        full_trainable_state_transmissions=(
            full_worker_evidence.trainable_state_transmissions
        ),
        expert_trainable_state_transmissions=sum(
            worker.trainable_state_transmissions
            for worker in expert_workers
        ),
        full_cold_tensor_wire_bytes=full_cold_tensor,
        expert_cold_tensor_wire_bytes=expert_cold_tensor,
        cold_tensor_wire_relative_change=(
            expert_cold_tensor / full_cold_tensor - 1.0
        ),
        full_warm_tensor_wire_bytes=full_warm_tensor,
        expert_warm_tensor_wire_bytes=expert_warm_tensor,
        warm_tensor_wire_relative_change=(
            expert_warm_tensor / full_warm_tensor - 1.0
        ),
        full_cold_application_wire_bytes=full_cold_application,
        expert_cold_application_wire_bytes=expert_cold_application,
        cold_application_wire_relative_change=(
            expert_cold_application / full_cold_application - 1.0
        ),
        full_warm_application_wire_bytes=full_warm_application,
        expert_warm_application_wire_bytes=expert_warm_application,
        warm_application_wire_relative_change=(
            expert_warm_application / full_warm_application - 1.0
        ),
        full_shutdown_control_json_wire_bytes=(
            full_worker_evidence.shutdown_control_json_wire_bytes
        ),
        expert_shutdown_control_json_wire_bytes=sum(
            worker.shutdown_control_json_wire_bytes
            for worker in expert_workers
        ),
        full_worker=full_worker_evidence,
        expert_workers=expert_workers,
        recovery=recovery,
        assignments=assignments,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched digest-authenticated persistent sparse process workers"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--router-aux-weight", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config)
    dataset = PackedDataset.load(args.dataset)
    evidence = run_authenticated_sparse_process_experiment(
        campaign,
        dataset=dataset,
        expert_count=args.expert_count,
        router_aux_weight=args.router_aux_weight,
        timeout_seconds=args.timeout_seconds,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
