from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from orcacolony.artifacts import PackedDataset
from orcacolony.coordinator import _tensor_metrics
from orcacolony import peft
from orcacolony.peft import (
    INT8_FROZEN_LINEAR_PROFILE,
    adapter_named_parameters,
    apply_adapter_gradient_step,
    build_int8_lora_model,
    build_lora_model,
    compute_adapter_gradients,
    create_adapter_optimizer,
    load_lora_checkpoint,
    load_lora_manifest,
    save_lora_checkpoint,
)
from orcacolony.reference import fixture_batch, tensor_sha256, validate_dataset_artifacts


FP32_PROFILE = "exact-cpu-fp32-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare deterministic FP32 and homogeneous int8 LoRA trajectories."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-step", type=int, default=10)
    parser.add_argument("--evaluation-steps", default="0,1,2,5,10,20")
    return parser.parse_args()


def tensor_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += storage.nbytes()
    return total


def adapter_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in adapter_named_parameters(model).items()
    }


def optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
) -> dict[str, Tensor]:
    snapshot: dict[str, Tensor] = {}
    for name, parameter in adapter_named_parameters(model).items():
        state = optimizer.state.get(parameter)
        if not state:
            snapshot[f"step.{name}"] = torch.zeros((), dtype=torch.float32)
            snapshot[f"exp_avg.{name}"] = torch.zeros_like(parameter).detach().cpu()
            snapshot[f"exp_avg_sq.{name}"] = torch.zeros_like(parameter).detach().cpu()
            continue
        snapshot[f"step.{name}"] = torch.as_tensor(
            state["step"], dtype=torch.float32
        ).detach().cpu().reshape(())
        snapshot[f"exp_avg.{name}"] = state["exp_avg"].detach().cpu().contiguous()
        snapshot[f"exp_avg_sq.{name}"] = (
            state["exp_avg_sq"].detach().cpu().contiguous()
        )
    return snapshot


def evaluate_model(
    model: nn.Module,
    loaded: peft.LoadedLoRAManifest,
    dataset: PackedDataset,
) -> dict[str, float | int]:
    evaluation = loaded.campaign.evaluation
    if evaluation is None:
        raise ValueError("trajectory campaign requires an evaluation profile")
    sequence_count = int(evaluation["validation_sequences"])
    batch_size = int(evaluation["batch_size"])
    loss_sum = 0.0
    loss_weight_sum = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        cursor = 0
        while cursor < sequence_count:
            current_batch_size = min(batch_size, sequence_count - cursor)
            inputs, targets = dataset.validation_batch(
                cursor=cursor,
                batch_size=current_batch_size,
                sequence_limit=sequence_count,
            )
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, loaded.campaign.model.vocabulary_size),
                targets.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(loss)
            loss_weight_sum += targets.numel()
            cursor += current_batch_size
    model.train(was_training)
    mean_loss = loss_sum / loss_weight_sum
    return {
        "validation_sequences": sequence_count,
        "loss_sum": loss_sum,
        "loss_weight_sum": loss_weight_sum,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def profiled_checkpoint_sha256(
    numerical_profile: str,
    base_model_sha256: str,
    adapter_sha256: str,
) -> str:
    payload = {
        "adapter_sha256": adapter_sha256,
        "base_model_sha256": base_model_sha256,
        "numerical_profile": numerical_profile,
    }
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_sharded_gradients(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
) -> tuple[float, int, dict[str, Tensor], str]:
    loss_sum = 0.0
    loss_weight_sum = 0
    aggregated: dict[str, Tensor] | None = None
    for row in range(inputs.shape[0]):
        result = compute_adapter_gradients(
            model,
            inputs[row : row + 1],
            targets[row : row + 1],
        )
        loss_sum += result.loss_sum
        loss_weight_sum += result.loss_weight_sum
        if aggregated is None:
            aggregated = {
                name: tensor.detach().cpu().clone()
                for name, tensor in result.gradients.items()
            }
        else:
            for name, tensor in result.gradients.items():
                aggregated[name].add_(tensor.detach().cpu())
    if aggregated is None:
        raise ValueError("homogeneous int8 trajectory requires at least one shard")
    return loss_sum, loss_weight_sum, aggregated, tensor_sha256(aggregated)


def main() -> None:
    args = parse_args()
    loaded = load_lora_manifest(args.campaign, args.lora)
    dataset = PackedDataset.load(args.dataset)
    validate_dataset_artifacts(loaded.campaign, dataset)
    target_steps = loaded.campaign.training.steps
    if args.restart_step <= 0 or args.restart_step >= target_steps:
        raise ValueError("restart step must be inside the trajectory")
    evaluation_steps = sorted({int(value) for value in args.evaluation_steps.split(",")})
    if not evaluation_steps or evaluation_steps[0] != 0 or evaluation_steps[-1] != target_steps:
        raise ValueError("evaluation steps must include zero and the campaign target")
    if any(step < 0 or step > target_steps for step in evaluation_steps):
        raise ValueError("evaluation step is outside the campaign trajectory")
    if args.work_dir.exists():
        raise ValueError(f"trajectory work directory already exists: {args.work_dir}")
    args.work_dir.mkdir(parents=True)

    fp32_model = build_lora_model(loaded.campaign, loaded.config)
    int8_model = build_int8_lora_model(loaded.campaign, loaded.config)
    int8_central_model = build_int8_lora_model(loaded.campaign, loaded.config)
    initial_adapter_sha256 = tensor_sha256(adapter_state(fp32_model))
    if any(
        tensor_sha256(adapter_state(model)) != initial_adapter_sha256
        for model in (int8_model, int8_central_model)
    ):
        raise ValueError("FP32 and int8 adapters do not share initialization")
    fp32_optimizer = create_adapter_optimizer(fp32_model, loaded.campaign.training)
    int8_optimizer = create_adapter_optimizer(int8_model, loaded.campaign.training)
    int8_central_optimizer = create_adapter_optimizer(
        int8_central_model, loaded.campaign.training
    )
    quantized_linear_count = sum(
        isinstance(module, peft.Int8FrozenLinear) for module in int8_model.modules()
    )

    evaluations: list[dict[str, object]] = []
    steps: list[dict[str, object]] = []
    fp32_loss_history: list[float] = []
    int8_loss_history: list[float] = []
    int8_central_loss_history: list[float] = []
    resumed_model: nn.Module | None = None
    resumed_optimizer: torch.optim.AdamW | None = None
    restart_checks: list[dict[str, object]] = []
    runtime_seconds = {
        "fp32_central_full_batch": 0.0,
        "int8_homogeneous_two_shard": 0.0,
        "int8_central_full_batch": 0.0,
    }

    def record_evaluation(step: int) -> None:
        started = time.perf_counter()
        fp32_evaluation = evaluate_model(fp32_model, loaded, dataset)
        runtime_seconds["fp32_central_full_batch"] += time.perf_counter() - started
        started = time.perf_counter()
        int8_evaluation = evaluate_model(int8_model, loaded, dataset)
        runtime_seconds["int8_homogeneous_two_shard"] += (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        int8_central_evaluation = evaluate_model(
            int8_central_model, loaded, dataset
        )
        runtime_seconds["int8_central_full_batch"] += time.perf_counter() - started
        if resumed_model is not None:
            resumed_evaluation = evaluate_model(resumed_model, loaded, dataset)
            if resumed_evaluation != int8_evaluation:
                raise ValueError("restarted int8 evaluation differs")
        evaluations.append(
            {
                "step": step,
                "fp32": fp32_evaluation,
                "int8": int8_evaluation,
                "int8_central": int8_central_evaluation,
                "homogeneous_minus_central_int8_mean_loss": (
                    float(int8_evaluation["mean_loss"])
                    - float(int8_central_evaluation["mean_loss"])
                ),
                "int8_minus_fp32_mean_loss": (
                    float(int8_evaluation["mean_loss"])
                    - float(fp32_evaluation["mean_loss"])
                ),
            }
        )

    record_evaluation(0)
    dataset_cursor = 0
    for step in range(1, target_steps + 1):
        inputs, targets = fixture_batch(loaded.campaign, dataset_cursor, dataset)

        started = time.perf_counter()
        fp32_gradient = compute_adapter_gradients(fp32_model, inputs, targets)
        apply_adapter_gradient_step(
            fp32_model,
            fp32_optimizer,
            fp32_gradient.gradients,
            fp32_gradient.loss_weight_sum,
            loaded.campaign.training.max_gradient_norm,
        )
        runtime_seconds["fp32_central_full_batch"] += time.perf_counter() - started

        started = time.perf_counter()
        int8_central_gradient = compute_adapter_gradients(
            int8_central_model, inputs, targets
        )
        apply_adapter_gradient_step(
            int8_central_model,
            int8_central_optimizer,
            int8_central_gradient.gradients,
            int8_central_gradient.loss_weight_sum,
            loaded.campaign.training.max_gradient_norm,
        )
        runtime_seconds["int8_central_full_batch"] += time.perf_counter() - started

        started = time.perf_counter()
        (
            int8_loss_sum,
            int8_loss_weight_sum,
            int8_gradients,
            int8_gradient_sha256,
        ) = compute_sharded_gradients(int8_model, inputs, targets)
        apply_adapter_gradient_step(
            int8_model,
            int8_optimizer,
            int8_gradients,
            int8_loss_weight_sum,
            loaded.campaign.training.max_gradient_norm,
        )
        runtime_seconds["int8_homogeneous_two_shard"] += (
            time.perf_counter() - started
        )

        fp32_loss_history.append(fp32_gradient.loss_sum / fp32_gradient.loss_weight_sum)
        int8_loss_history.append(int8_loss_sum / int8_loss_weight_sum)
        int8_central_loss_history.append(
            int8_central_gradient.loss_sum / int8_central_gradient.loss_weight_sum
        )
        dataset_cursor = (
            dataset_cursor + loaded.campaign.training.batch_size
        ) % loaded.campaign.training.dataset_sequences

        restart_gradient_sha256: str | None = None
        if resumed_model is not None and resumed_optimizer is not None:
            (
                resumed_loss_sum,
                resumed_loss_weight_sum,
                resumed_gradients,
                restart_gradient_sha256,
            ) = compute_sharded_gradients(resumed_model, inputs, targets)
            if (
                restart_gradient_sha256 != int8_gradient_sha256
                or resumed_loss_sum != int8_loss_sum
            ):
                raise ValueError("restarted int8 gradient differs")
            apply_adapter_gradient_step(
                resumed_model,
                resumed_optimizer,
                resumed_gradients,
                resumed_loss_weight_sum,
                loaded.campaign.training.max_gradient_norm,
            )
            adapter_replay = _tensor_metrics(
                adapter_state(int8_model), adapter_state(resumed_model)
            )
            optimizer_replay = _tensor_metrics(
                optimizer_state(int8_model, int8_optimizer),
                optimizer_state(resumed_model, resumed_optimizer),
            )
            if (
                adapter_replay["max_absolute_error"] != 0.0
                or optimizer_replay["max_absolute_error"] != 0.0
            ):
                raise ValueError("restarted int8 state differs")
            restart_checks.append(
                {
                    "step": step,
                    "gradient_sha256": restart_gradient_sha256,
                    "adapter_metrics": adapter_replay,
                    "optimizer_metrics": optimizer_replay,
                }
            )

        steps.append(
            {
                "step": step,
                "dataset_cursor_after": dataset_cursor,
                "fp32_loss_mean": fp32_loss_history[-1],
                "int8_loss_mean": int8_loss_history[-1],
                "int8_central_loss_mean": int8_central_loss_history[-1],
                "relative_loss_error": abs(
                    int8_loss_sum - fp32_gradient.loss_sum
                )
                / abs(fp32_gradient.loss_sum),
                "fp32_gradient_sha256": fp32_gradient.gradient_sha256,
                "int8_gradient_sha256": int8_gradient_sha256,
                "int8_central_gradient_sha256": (
                    int8_central_gradient.gradient_sha256
                ),
                "gradient_metrics": _tensor_metrics(
                    fp32_gradient.gradients, int8_gradients
                ),
                "homogeneous_gradient_metrics": _tensor_metrics(
                    int8_central_gradient.gradients, int8_gradients
                ),
                "adapter_metrics": _tensor_metrics(
                    adapter_state(fp32_model), adapter_state(int8_model)
                ),
                "homogeneous_adapter_metrics": _tensor_metrics(
                    adapter_state(int8_central_model), adapter_state(int8_model)
                ),
                "restart_gradient_sha256": restart_gradient_sha256,
            }
        )

        if step == args.restart_step:
            checkpoint = save_lora_checkpoint(
                loaded,
                int8_model,
                int8_optimizer,
                args.work_dir / f"step-{step:08d}",
                step=step,
                dataset_cursor=dataset_cursor,
                loss_history=int8_loss_history,
            )
            (
                resumed_model,
                resumed_optimizer,
                resumed_step,
                resumed_cursor,
                resumed_history,
            ) = load_lora_checkpoint(loaded, checkpoint.checkpoint_dir)
            peft._quantize_frozen_linears(resumed_model)
            if (
                resumed_step != step
                or resumed_cursor != dataset_cursor
                or resumed_history != int8_loss_history
            ):
                raise ValueError("int8 checkpoint restart metadata differs")
            if _tensor_metrics(
                adapter_state(int8_model), adapter_state(resumed_model)
            )["max_absolute_error"] != 0.0:
                raise ValueError("int8 checkpoint restart adapter differs")
            if _tensor_metrics(
                optimizer_state(int8_model, int8_optimizer),
                optimizer_state(resumed_model, resumed_optimizer),
            )["max_absolute_error"] != 0.0:
                raise ValueError("int8 checkpoint restart optimizer differs")

        if step in evaluation_steps:
            record_evaluation(step)

    fp32_adapter = adapter_state(fp32_model)
    int8_adapter = adapter_state(int8_model)
    int8_central_adapter = adapter_state(int8_central_model)
    fp32_adapter_sha256 = tensor_sha256(fp32_adapter)
    int8_adapter_sha256 = tensor_sha256(int8_adapter)
    int8_central_adapter_sha256 = tensor_sha256(int8_central_adapter)
    final_homogeneous_metrics = _tensor_metrics(
        int8_central_adapter, int8_adapter
    )
    minimum_improvement = float(
        loaded.campaign.evaluation["success_gate"][
            "minimum_improvement_from_initialization"
        ]
    )
    int8_held_out_improvement = (
        float(evaluations[0]["int8"]["mean_loss"])
        - float(evaluations[-1]["int8"]["mean_loss"])
    )
    homogeneous_candidate = (
        final_homogeneous_metrics["relative_l2_error"] < 1e-5
        and all(
            step_result["homogeneous_gradient_metrics"]["relative_l2_error"]
            < 1e-5
            for step_result in steps
        )
        and len(restart_checks) == target_steps - args.restart_step
        and int8_held_out_improvement >= minimum_improvement
    )
    result = {
        "format": "orcacolony_int8_homogeneous_trajectory_v1",
        "campaign_id": loaded.campaign.campaign["id"],
        "campaign_steps": target_steps,
        "dataset_revision": dataset.revision,
        "base_model_sha256": loaded.config.base_model_sha256,
        "initial_adapter_sha256": initial_adapter_sha256,
        "profiles": {
            "fp32": FP32_PROFILE,
            "int8": INT8_FROZEN_LINEAR_PROFILE,
        },
        "profiled_final_checkpoint_sha256": {
            "fp32": profiled_checkpoint_sha256(
                FP32_PROFILE,
                loaded.config.base_model_sha256,
                fp32_adapter_sha256,
            ),
            "int8": profiled_checkpoint_sha256(
                INT8_FROZEN_LINEAR_PROFILE,
                loaded.config.base_model_sha256,
                int8_adapter_sha256,
            ),
            "int8_central": profiled_checkpoint_sha256(
                INT8_FROZEN_LINEAR_PROFILE,
                loaded.config.base_model_sha256,
                int8_central_adapter_sha256,
            ),
        },
        "adapter_value_count": sum(tensor.numel() for tensor in fp32_adapter.values()),
        "quantized_linear_count": quantized_linear_count,
        "resident_tensor_bytes": {
            "fp32": tensor_bytes(fp32_model),
            "int8": tensor_bytes(int8_model),
        },
        "runtime_seconds": runtime_seconds,
        "restart_step": args.restart_step,
        "restart_checks": restart_checks,
        "steps": steps,
        "evaluations": evaluations,
        "final_adapter_sha256": {
            "fp32": fp32_adapter_sha256,
            "int8": int8_adapter_sha256,
            "int8_central": int8_central_adapter_sha256,
        },
        "final_adapter_metrics": _tensor_metrics(fp32_adapter, int8_adapter),
        "final_homogeneous_metrics": final_homogeneous_metrics,
        "homogeneous_profile_candidate": homogeneous_candidate,
        "candidate_criteria": {
            "maximum_homogeneous_gradient_relative_l2": 1e-5,
            "maximum_final_homogeneous_adapter_relative_l2": 1e-5,
            "minimum_held_out_improvement": minimum_improvement,
            "required_exact_restart_checks": target_steps - args.restart_step,
        },
        "profiled_checkpoint_identity_fields": [
            "numerical_profile",
            "base_model_sha256",
            "adapter_sha256",
        ],
        "held_out_improvement": {
            "fp32": (
                float(evaluations[0]["fp32"]["mean_loss"])
                - float(evaluations[-1]["fp32"]["mean_loss"])
            ),
            "int8": int8_held_out_improvement,
            "int8_central": (
                float(evaluations[0]["int8_central"]["mean_loss"])
                - float(evaluations[-1]["int8_central"]["mean_loss"])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
