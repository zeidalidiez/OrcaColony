from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import torch
from torch import Tensor

from orcacolony.artifacts import PackedDataset
from orcacolony.reference import (
    CampaignConfig,
    ModelConfig,
    VolunteerDecoder,
    _create_optimizer,
    build_model,
    configure_determinism,
    fixture_batch,
    load_campaign,
    objective_mean_loss,
    tensor_sha256,
    validate_dataset_artifacts,
)
from orcacolony.tile_process import (
    _await_model_readiness,
    _deserialize_tensors,
    _process_memory_bytes,
    _recv_bytes,
    _recv_json,
    _send_json,
    _serialize_tensors,
)
from orcacolony.tiled_model import (
    _gradient_snapshot,
    _max_abs_difference,
    _model_snapshot,
    _optimizer_tensor_snapshot,
)


@dataclass(frozen=True)
class FullProcessAssignmentEvidence:
    cursor: int
    input_batch_wire_bytes: int
    gradient_result_wire_bytes: int
    assignment_tensor_wire_bytes: int
    control_json_wire_bytes: int
    total_application_wire_bytes: int
    centralized_loss: float
    process_loss: float
    max_abs_raw_gradient_difference: float
    max_abs_clipped_gradient_difference: float
    max_abs_model_difference: float
    centralized_raw_gradient_sha256: str
    process_raw_gradient_sha256: str
    centralized_clipped_gradient_sha256: str
    process_clipped_gradient_sha256: str
    centralized_optimizer_sha256: str
    process_optimizer_sha256: str
    centralized_model_sha256: str
    process_model_sha256: str
    round_trip_seconds: float
    worker_compute_seconds: float
    worker_current_rss_bytes: int
    worker_peak_rss_bytes: int


@dataclass(frozen=True)
class PersistentFullProcessEvidence:
    format: str
    campaign_id: str
    dataset_revision: str
    start_method: str
    assignment_count: int
    model_transmissions: int
    full_forward_calls: int
    full_state_sha256: str
    full_model_wire_bytes: int
    initialization_round_trip_seconds: float
    initialization_control_json_wire_bytes: int
    worker_startup_current_rss_bytes: int
    worker_startup_peak_rss_bytes: int
    worker_after_model_current_rss_bytes: int
    worker_after_model_peak_rss_bytes: int
    worker_model_current_rss_delta_bytes: int
    worker_final_peak_rss_bytes: int
    cold_assignment_tensor_wire_bytes: int
    warm_assignment_tensor_wire_bytes: int
    assignments: tuple[FullProcessAssignmentEvidence, ...]
    child_exit_code: int


def _validate_token_batch(
    tensor: Tensor,
    *,
    config: ModelConfig,
    label: str,
) -> Tensor:
    if (
        tensor.dtype != torch.int64
        or tensor.ndim != 2
        or tensor.shape[0] <= 0
        or tensor.shape[1] <= 0
        or tensor.shape[1] > config.context_length
    ):
        raise ValueError(f"{label} tensor contract is invalid")
    if int(tensor.min()) < 0 or int(tensor.max()) >= config.vocabulary_size:
        raise ValueError(f"{label} token is outside the configured vocabulary")
    return tensor.detach().clone()


def _full_worker_entry(connection: Connection) -> None:
    try:
        startup_current, startup_peak = _process_memory_bytes()
        init_raw = connection.recv_bytes()
        init = json.loads(init_raw)
        if not isinstance(init, dict) or frozenset(init) != frozenset(
            {"op", "model", "seed"}
        ):
            raise ValueError("full-worker init schema is invalid")
        if init["op"] != "init" or type(init["seed"]) is not int:
            raise ValueError("full-worker init semantics are invalid")
        model_payload = init["model"]
        if not isinstance(model_payload, dict):
            raise ValueError("full-worker model config is invalid")
        config = ModelConfig(**model_payload)
        configure_determinism(init["seed"])
        model = VolunteerDecoder(config)
        expected_state = model.state_dict()
        _send_json(connection, {"status": "ready_for_model"})
        model_wire = connection.recv_bytes()
        loaded_state = _deserialize_tensors(model_wire)
        if loaded_state.keys() != expected_state.keys():
            raise ValueError("full-worker model tensor names are invalid")
        for name, expected in expected_state.items():
            tensor = loaded_state[name]
            if (
                tensor.dtype != expected.dtype
                or tensor.shape != expected.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"full-worker model tensor is invalid: {name}")
        model.load_state_dict(loaded_state, strict=True)
        model.train()
        after_model_current, after_model_peak = _process_memory_bytes()
        full_state_sha256 = tensor_sha256(model.state_dict())
        forward_calls = 0
        _send_json(
            connection,
            {
                "status": "ready",
                "full_state_sha256": full_state_sha256,
                "startup_current_rss_bytes": startup_current,
                "startup_peak_rss_bytes": startup_peak,
                "after_model_current_rss_bytes": after_model_current,
                "after_model_peak_rss_bytes": after_model_peak,
            },
        )

        while True:
            control_raw = connection.recv_bytes()
            control = json.loads(control_raw)
            if not isinstance(control, dict) or "op" not in control:
                raise ValueError("full-worker control schema is invalid")
            if control["op"] == "shutdown":
                current, peak = _process_memory_bytes()
                _send_json(
                    connection,
                    {
                        "status": "stopped",
                        "full_forward_calls": forward_calls,
                        "current_rss_bytes": current,
                        "peak_rss_bytes": peak,
                    },
                )
                return
            if frozenset(control) != frozenset({"op", "assignment_id"}):
                raise ValueError("full-worker assignment control schema is invalid")
            assignment_id = control["assignment_id"]
            if (
                control["op"] != "run"
                or type(assignment_id) is not int
                or assignment_id < 0
            ):
                raise ValueError("full-worker assignment identity is invalid")
            payload = _deserialize_tensors(connection.recv_bytes())
            if frozenset(payload) != frozenset({"inputs", "targets"}):
                raise ValueError("full-worker input tensor schema is invalid")
            inputs = _validate_token_batch(
                payload["inputs"],
                config=config,
                label="inputs",
            )
            targets = _validate_token_batch(
                payload["targets"],
                config=config,
                label="targets",
            )
            if inputs.shape != targets.shape:
                raise ValueError("full-worker input and target shapes differ")

            model.zero_grad(set_to_none=True)
            started = time.perf_counter()
            logits = model(inputs)
            loss = objective_mean_loss(
                model.objective,
                logits,
                targets,
            )
            loss.backward()
            compute_seconds = time.perf_counter() - started
            forward_calls += 1
            result: dict[str, Tensor] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"full-worker parameter lacks gradient: {name}")
                result[f"gradient.{name}"] = parameter.grad.detach().clone()
            result_wire = _serialize_tensors(result)
            current, peak = _process_memory_bytes()
            _send_json(
                connection,
                {
                    "status": "completed",
                    "assignment_id": assignment_id,
                    "loss": float(loss.detach()),
                    "compute_seconds": compute_seconds,
                    "current_rss_bytes": current,
                    "peak_rss_bytes": peak,
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


def _run_parent_assignment(
    connection: Connection,
    campaign: CampaignConfig,
    dataset: PackedDataset | None,
    *,
    assignment_id: int,
    cursor: int,
    timeout_seconds: float,
) -> FullProcessAssignmentEvidence:
    centralized = build_model(campaign)
    process_model = build_model(campaign)
    centralized_optimizer = _create_optimizer(centralized, campaign.training)
    process_optimizer = _create_optimizer(process_model, campaign.training)
    inputs, targets = fixture_batch(campaign, cursor, dataset)

    centralized.train()
    centralized_optimizer.zero_grad(set_to_none=True)
    centralized_loss_tensor = objective_mean_loss(
        campaign.objective,
        centralized(inputs),
        targets,
    )
    centralized_loss_tensor.backward()
    centralized_raw = _gradient_snapshot(centralized)
    torch.nn.utils.clip_grad_norm_(
        centralized.parameters(),
        campaign.training.max_gradient_norm,
    )
    centralized_clipped = _gradient_snapshot(centralized)
    centralized_optimizer.step()

    input_wire = _serialize_tensors({"inputs": inputs, "targets": targets})
    control_bytes = _send_json(
        connection,
        {"op": "run", "assignment_id": assignment_id},
    )
    started = time.perf_counter()
    connection.send_bytes(input_wire)
    acknowledgement, ack_bytes = _recv_json(
        connection,
        timeout_seconds,
        label="full-worker completion",
    )
    gradient_wire = _recv_bytes(
        connection,
        timeout_seconds,
        label="full-worker gradients",
    )
    round_trip_seconds = time.perf_counter() - started
    control_bytes += ack_bytes
    if (
        frozenset(acknowledgement)
        != frozenset(
            {
                "status",
                "assignment_id",
                "loss",
                "compute_seconds",
                "current_rss_bytes",
                "peak_rss_bytes",
            }
        )
        or acknowledgement["status"] != "completed"
        or acknowledgement["assignment_id"] != assignment_id
        or type(acknowledgement["loss"]) is not float
        or not math.isfinite(acknowledgement["loss"])
    ):
        raise ValueError("full-worker completion acknowledgement is invalid")

    result = _deserialize_tensors(gradient_wire)
    expected_names = {
        f"gradient.{name}" for name, _ in process_model.named_parameters()
    }
    if frozenset(result) != frozenset(expected_names):
        raise ValueError("full-worker gradient tensor schema is invalid")
    process_optimizer.zero_grad(set_to_none=True)
    for name, parameter in process_model.named_parameters():
        gradient = result[f"gradient.{name}"]
        if (
            gradient.dtype != parameter.dtype
            or gradient.shape != parameter.shape
            or not bool(torch.isfinite(gradient).all())
        ):
            raise ValueError(f"full-worker gradient is invalid: {name}")
        parameter.grad = gradient.detach().clone()
    process_raw = _gradient_snapshot(process_model)
    torch.nn.utils.clip_grad_norm_(
        process_model.parameters(),
        campaign.training.max_gradient_norm,
    )
    process_clipped = _gradient_snapshot(process_model)
    process_optimizer.step()

    centralized_model = _model_snapshot(centralized)
    process_model_snapshot = _model_snapshot(process_model)
    centralized_optimizer_snapshot = _optimizer_tensor_snapshot(
        centralized,
        centralized_optimizer,
    )
    process_optimizer_snapshot = _optimizer_tensor_snapshot(
        process_model,
        process_optimizer,
    )
    tensor_wire_bytes = len(input_wire) + len(gradient_wire)
    return FullProcessAssignmentEvidence(
        cursor=cursor,
        input_batch_wire_bytes=len(input_wire),
        gradient_result_wire_bytes=len(gradient_wire),
        assignment_tensor_wire_bytes=tensor_wire_bytes,
        control_json_wire_bytes=control_bytes,
        total_application_wire_bytes=tensor_wire_bytes + control_bytes,
        centralized_loss=float(centralized_loss_tensor.detach()),
        process_loss=float(acknowledgement["loss"]),
        max_abs_raw_gradient_difference=_max_abs_difference(
            centralized_raw,
            process_raw,
        ),
        max_abs_clipped_gradient_difference=_max_abs_difference(
            centralized_clipped,
            process_clipped,
        ),
        max_abs_model_difference=_max_abs_difference(
            centralized_model,
            process_model_snapshot,
        ),
        centralized_raw_gradient_sha256=tensor_sha256(centralized_raw),
        process_raw_gradient_sha256=tensor_sha256(process_raw),
        centralized_clipped_gradient_sha256=tensor_sha256(centralized_clipped),
        process_clipped_gradient_sha256=tensor_sha256(process_clipped),
        centralized_optimizer_sha256=tensor_sha256(
            centralized_optimizer_snapshot
        ),
        process_optimizer_sha256=tensor_sha256(process_optimizer_snapshot),
        centralized_model_sha256=tensor_sha256(centralized_model),
        process_model_sha256=tensor_sha256(process_model_snapshot),
        round_trip_seconds=round_trip_seconds,
        worker_compute_seconds=float(acknowledgement["compute_seconds"]),
        worker_current_rss_bytes=int(acknowledgement["current_rss_bytes"]),
        worker_peak_rss_bytes=int(acknowledgement["peak_rss_bytes"]),
    )


def run_persistent_full_process_control(
    campaign: CampaignConfig,
    *,
    dataset: PackedDataset | None = None,
    timeout_seconds: float = 60.0,
) -> PersistentFullProcessEvidence:
    validate_dataset_artifacts(campaign, dataset)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or not 0.0 < float(timeout_seconds) <= 300.0
    ):
        raise ValueError("timeout must be finite and between zero and 300 seconds")
    timeout_seconds = float(timeout_seconds)

    source_model = build_model(campaign)
    full_state = {
        name: tensor.detach().clone()
        for name, tensor in source_model.state_dict().items()
    }
    full_state_sha256 = tensor_sha256(full_state)
    full_model_wire = _serialize_tensors(full_state)
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=_full_worker_entry,
        args=(child_connection,),
        name="orcacolony-full-worker",
        daemon=False,
    )
    initialization_started = time.perf_counter()
    process.start()
    child_connection.close()
    assignments: list[FullProcessAssignmentEvidence] = []
    init_control_bytes = 0
    shutdown_control_bytes = 0
    ready: dict[str, object] | None = None
    shutdown_ack: dict[str, object] | None = None
    initialization_round_trip_seconds = 0.0
    error: BaseException | None = None
    try:
        init_control_bytes = _send_json(
            parent_connection,
            {
                "op": "init",
                "model": asdict(campaign.model),
                "seed": campaign.training.seed,
            },
        )
        init_control_bytes += _await_model_readiness(
            parent_connection,
            timeout_seconds,
            label="full-worker model readiness",
        )
        parent_connection.send_bytes(full_model_wire)
        ready, ready_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label="full-worker initialization",
        )
        initialization_round_trip_seconds = (
            time.perf_counter() - initialization_started
        )
        init_control_bytes += ready_bytes
        if (
            frozenset(ready)
            != frozenset(
                {
                    "status",
                    "full_state_sha256",
                    "startup_current_rss_bytes",
                    "startup_peak_rss_bytes",
                    "after_model_current_rss_bytes",
                    "after_model_peak_rss_bytes",
                }
            )
            or ready["status"] != "ready"
            or ready["full_state_sha256"] != full_state_sha256
        ):
            raise ValueError("full-worker initialization acknowledgement is invalid")

        cursors = (
            0,
            campaign.training.batch_size % campaign.training.dataset_sequences,
        )
        for assignment_id, cursor in enumerate(cursors):
            assignments.append(
                _run_parent_assignment(
                    parent_connection,
                    campaign,
                    dataset,
                    assignment_id=assignment_id,
                    cursor=cursor,
                    timeout_seconds=timeout_seconds,
                )
            )

        shutdown_control_bytes = _send_json(
            parent_connection,
            {"op": "shutdown"},
        )
        shutdown_ack, ack_bytes = _recv_json(
            parent_connection,
            timeout_seconds,
            label="full-worker shutdown",
        )
        shutdown_control_bytes += ack_bytes
        if (
            frozenset(shutdown_ack)
            != frozenset(
                {
                    "status",
                    "full_forward_calls",
                    "current_rss_bytes",
                    "peak_rss_bytes",
                }
            )
            or shutdown_ack["status"] != "stopped"
        ):
            raise ValueError("full-worker shutdown acknowledgement is invalid")
    except BaseException as exc:
        error = exc
    finally:
        parent_connection.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout_seconds)
    if error is not None:
        raise error
    if process.exitcode is None:
        raise RuntimeError("full worker has no exit code")
    if process.exitcode != 0:
        raise RuntimeError(f"full worker exited with code {process.exitcode}")
    if ready is None or shutdown_ack is None or len(assignments) != 2:
        raise AssertionError("full-process control did not complete")

    cold_assignment_bytes = (
        len(full_model_wire) + assignments[0].assignment_tensor_wire_bytes
    )
    warm_assignment_bytes = assignments[1].assignment_tensor_wire_bytes
    dataset_revision = (
        dataset.revision if dataset is not None else "synthetic-fixture-v1"
    )
    return PersistentFullProcessEvidence(
        format="orcacolony_persistent_full_process_evidence_v1",
        campaign_id=str(campaign.campaign["id"]),
        dataset_revision=dataset_revision,
        start_method=context.get_start_method(),
        assignment_count=len(assignments),
        model_transmissions=1,
        full_forward_calls=int(shutdown_ack["full_forward_calls"]),
        full_state_sha256=full_state_sha256,
        full_model_wire_bytes=len(full_model_wire),
        initialization_round_trip_seconds=initialization_round_trip_seconds,
        initialization_control_json_wire_bytes=(
            init_control_bytes + shutdown_control_bytes
        ),
        worker_startup_current_rss_bytes=int(
            ready["startup_current_rss_bytes"]
        ),
        worker_startup_peak_rss_bytes=int(ready["startup_peak_rss_bytes"]),
        worker_after_model_current_rss_bytes=int(
            ready["after_model_current_rss_bytes"]
        ),
        worker_after_model_peak_rss_bytes=int(
            ready["after_model_peak_rss_bytes"]
        ),
        worker_model_current_rss_delta_bytes=(
            int(ready["after_model_current_rss_bytes"])
            - int(ready["startup_current_rss_bytes"])
        ),
        worker_final_peak_rss_bytes=int(shutdown_ack["peak_rss_bytes"]),
        cold_assignment_tensor_wire_bytes=cold_assignment_bytes,
        warm_assignment_tensor_wire_bytes=warm_assignment_bytes,
        assignments=tuple(assignments),
        child_exit_code=int(process.exitcode),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a matched persistent full-model process control"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    campaign = load_campaign(args.config)
    if campaign.dataset is None:
        if args.dataset is not None:
            parser.error("--dataset is only valid for data-backed campaign configs")
        dataset = None
    else:
        if args.dataset is None:
            parser.error("data-backed campaign configs require --dataset")
        dataset = PackedDataset.load(args.dataset)
    evidence = run_persistent_full_process_control(
        campaign,
        dataset=dataset,
        timeout_seconds=args.timeout_seconds,
    )
    payload = json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
