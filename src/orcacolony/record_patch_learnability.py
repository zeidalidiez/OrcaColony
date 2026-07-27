from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from tokenizers import Tokenizer

from .artifacts import PackedDataset
from .record_patch import (
    CAMPAIGN_ID,
    _greedy_completion,
    _prediction_bytes,
    evaluate_predictions,
    load_behavioral_split,
)
from .reference import (
    CampaignConfig,
    TrainingResult,
    _create_optimizer,
    _load_checkpoint,
    _save_checkpoint,
    build_model,
    evaluate_checkpoint,
    fixture_batch,
    load_campaign,
    objective_loss_sum,
    validate_dataset_artifacts,
)


@dataclass(frozen=True)
class DiagnosticTrainingResult:
    checkpoint_dirs: Mapping[int, Path]
    checkpoints: Mapping[int, TrainingResult]
    diagnostics: tuple[Mapping[str, object], ...]
    wall_seconds: float


_PROTOCOL_FIELDS = frozenset(
    {
        "format",
        "id",
        "campaign",
        "dataset_revision",
        "behavioral_suite_revision",
        "evaluator_revision",
        "execution",
        "schedule",
        "selection",
        "gate",
        "resource_limits",
        "holdout_policy",
        "limitations",
    }
)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _module_revision() -> str:
    return _sha256_file(Path(__file__).resolve())


def _checkpoint_steps(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("at least one checkpoint step is required")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("checkpoint steps must be nonnegative integers")
    steps = tuple(values)
    if steps[0] != 0:
        raise ValueError("checkpoint schedule must begin at step zero")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("checkpoint steps must be unique and increasing")
    return steps


def _protocol_object(
    value: object,
    *,
    label: str,
    fields: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"learnability protocol {label} is invalid")
    return value


def _positive_protocol_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"learnability protocol {label} must be a positive integer"
        )
    return value


def _load_protocol(
    path: str | Path,
    *,
    campaign_sha256: str,
    dataset_revision: str,
    research: Mapping[str, object],
    minimum_language_improvement: float,
) -> Mapping[str, object]:
    protocol_path = Path(path).resolve()
    payload = json.loads(
        protocol_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping) or set(payload) != _PROTOCOL_FIELDS:
        raise ValueError("learnability protocol schema is invalid")
    if payload.get("format") != "orcacolony_record_patch_learnability_protocol_v1":
        raise ValueError("unsupported learnability protocol format")
    if payload.get("id") != "record-patch-t2-centralized-learnability-v1":
        raise ValueError("learnability protocol identity is invalid")
    campaign = _protocol_object(
        payload.get("campaign"),
        label="campaign",
        fields={"id", "sha256"},
    )
    if campaign != {
        "id": CAMPAIGN_ID,
        "sha256": campaign_sha256,
    }:
        raise ValueError("learnability protocol campaign identity mismatch")
    if payload.get("dataset_revision") != dataset_revision:
        raise ValueError("learnability protocol dataset identity mismatch")
    behavioral_contract = _protocol_object(
        research.get("behavioral_evaluation"),
        label="campaign behavioral contract",
        fields={
            "suite_id",
            "dataset_revision",
            "evaluator_revision",
            "validation_split",
            "final_holdout_split",
        },
    )
    if (
        payload.get("behavioral_suite_revision")
        != behavioral_contract["dataset_revision"]
        or payload.get("evaluator_revision")
        != behavioral_contract["evaluator_revision"]
    ):
        raise ValueError("learnability protocol evaluator identity mismatch")
    execution = _protocol_object(
        payload.get("execution"),
        label="execution",
        fields={"donated_compute", "topology", "torch_threads"},
    )
    if execution != {
        "donated_compute": False,
        "topology": "owner-operated centralized CPU reference",
        "torch_threads": 1,
    }:
        raise ValueError("learnability protocol execution profile is invalid")
    schedule = _protocol_object(
        payload.get("schedule"),
        label="schedule",
        fields={
            "behavioral_steps",
            "checkpoint_steps",
            "max_new_tokens",
        },
    )
    raw_checkpoints = schedule["checkpoint_steps"]
    raw_behavioral = schedule["behavioral_steps"]
    if not isinstance(raw_checkpoints, list):
        raise ValueError("learnability protocol checkpoint schedule is invalid")
    if not isinstance(raw_behavioral, list):
        raise ValueError("learnability protocol behavioral schedule is invalid")
    checkpoints = _checkpoint_steps(raw_checkpoints)
    behavioral = _checkpoint_steps(raw_behavioral)
    if not set(behavioral).issubset(checkpoints):
        raise ValueError(
            "learnability protocol behavioral steps must be checkpoints"
        )
    max_new_tokens = _positive_protocol_int(
        schedule["max_new_tokens"],
        "generation budget",
    )
    if payload.get("selection") != (
        "lowest public language-validation mean loss; behavioral metrics "
        "do not select the checkpoint"
    ):
        raise ValueError("learnability protocol selection policy is invalid")
    gate = _protocol_object(
        payload.get("gate"),
        label="gate",
        fields={
            "minimum_behavioral_exact_match_count_improvement",
            "minimum_language_mean_loss_improvement",
        },
    )
    exact_improvement = _positive_protocol_int(
        gate["minimum_behavioral_exact_match_count_improvement"],
        "behavioral improvement",
    )
    language_improvement = gate["minimum_language_mean_loss_improvement"]
    if (
        isinstance(language_improvement, bool)
        or not isinstance(language_improvement, (int, float))
        or not math.isfinite(float(language_improvement))
        or float(language_improvement) <= 0
    ):
        raise ValueError("learnability protocol language gate is invalid")
    if float(language_improvement) != minimum_language_improvement:
        raise ValueError(
            "learnability protocol language gate differs from campaign"
        )
    resource_limits = _protocol_object(
        payload.get("resource_limits"),
        label="resource limits",
        fields={"maximum_peak_rss_bytes", "maximum_training_steps"},
    )
    maximum_steps = _positive_protocol_int(
        resource_limits["maximum_training_steps"],
        "maximum training steps",
    )
    maximum_rss = _positive_protocol_int(
        resource_limits["maximum_peak_rss_bytes"],
        "maximum peak RSS",
    )
    if checkpoints[-1] > maximum_steps:
        raise ValueError("learnability schedule exceeds its step limit")
    holdout_policy = _protocol_object(
        payload.get("holdout_policy"),
        label="holdout policy",
        fields={"behavioral_final_holdout", "language_final_holdout"},
    )
    if holdout_policy != {
        "behavioral_final_holdout": "must_not_open",
        "language_final_holdout": "must_not_evaluate",
    }:
        raise ValueError("learnability protocol holdout policy is invalid")
    limitations = payload.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(item, str) or not item.strip()
            for item in limitations
        )
    ):
        raise ValueError("learnability protocol limitations are invalid")
    return {
        **dict(payload),
        "schedule": {
            "checkpoint_steps": checkpoints,
            "behavioral_steps": behavioral,
            "max_new_tokens": max_new_tokens,
        },
        "gate": {
            "minimum_behavioral_exact_match_count_improvement": (
                exact_improvement
            ),
            "minimum_language_mean_loss_improvement": float(
                language_improvement
            ),
        },
        "resource_limits": {
            "maximum_peak_rss_bytes": maximum_rss,
            "maximum_training_steps": maximum_steps,
        },
        "_sha256": _sha256_file(protocol_path),
    }


def _tensor_global_norm(tensors: Sequence[torch.Tensor]) -> float:
    squared = 0.0
    for tensor in tensors:
        squared += float(
            torch.sum(
                tensor.detach() * tensor.detach(),
                dtype=torch.float64,
            )
        )
    return math.sqrt(squared)


def _training_step(
    campaign: CampaignConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    dataset: PackedDataset | None,
    *,
    step: int,
    dataset_cursor: int,
) -> tuple[float, int, Mapping[str, object]]:
    started = time.perf_counter()
    inputs, targets = fixture_batch(campaign, dataset_cursor, dataset)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss_sum, loss_weight_sum = objective_loss_sum(
        campaign.objective,
        logits,
        targets,
    )
    loss_sum.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(loss_weight_sum)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise ValueError("training step produced no gradients")
    gradient_max_abs = max(
        float(torch.max(torch.abs(gradient.detach())))
        for gradient in gradients
    )
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            campaign.training.max_gradient_norm,
        )
    )
    parameters = list(model.parameters())
    parameter_norm = _tensor_global_norm(parameters)
    parameters_before = [
        parameter.detach().clone()
        for parameter in parameters
    ]
    optimizer.step()
    update_squared = 0.0
    update_max_abs = 0.0
    for parameter, previous in zip(parameters, parameters_before, strict=True):
        update = parameter.detach() - previous
        update_squared += float(
            torch.sum(update * update, dtype=torch.float64)
        )
        update_max_abs = max(
            update_max_abs,
            float(torch.max(torch.abs(update))),
        )
    update_norm = math.sqrt(update_squared)
    mean_loss = float(loss_sum.detach()) / loss_weight_sum
    next_cursor = (
        dataset_cursor + campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    diagnostics = {
        "step": step + 1,
        "dataset_cursor_start": dataset_cursor,
        "dataset_cursor_end": next_cursor,
        "loss_sum": float(loss_sum.detach()),
        "loss_weight_sum": loss_weight_sum,
        "training_mean_loss": mean_loss,
        "gradient_global_norm_before_clipping": gradient_norm,
        "gradient_max_abs_before_clipping": gradient_max_abs,
        "max_gradient_norm": campaign.training.max_gradient_norm,
        "clipped": gradient_norm > campaign.training.max_gradient_norm,
        "parameter_global_norm_before_update": parameter_norm,
        "update_global_norm": update_norm,
        "relative_update_global_norm": (
            update_norm / parameter_norm if parameter_norm else None
        ),
        "update_max_abs": update_max_abs,
        "wall_seconds": time.perf_counter() - started,
    }
    if not all(
        math.isfinite(float(value))
        for key, value in diagnostics.items()
        if key
        not in {
            "step",
            "dataset_cursor_start",
            "dataset_cursor_end",
            "loss_weight_sum",
            "clipped",
        }
        and value is not None
    ):
        raise ValueError("training diagnostics contain a non-finite value")
    return mean_loss, next_cursor, diagnostics


def run_diagnostic_training(
    *,
    campaign: CampaignConfig,
    output_dir: str | Path,
    checkpoint_steps: Sequence[int],
    dataset: PackedDataset | None = None,
    maximum_peak_rss_bytes: int | None = None,
) -> DiagnosticTrainingResult:
    validate_dataset_artifacts(campaign, dataset)
    steps = _checkpoint_steps(checkpoint_steps)
    if steps[-1] > campaign.training.steps:
        raise ValueError("checkpoint schedule exceeds the campaign step budget")
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("diagnostic training output directory is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    model = build_model(campaign)
    optimizer = _create_optimizer(model, campaign.training)
    model.train()
    step = 0
    dataset_cursor = 0
    loss_history: list[float] = []
    diagnostics: list[Mapping[str, object]] = []
    checkpoint_dirs: dict[int, Path] = {}
    checkpoints: dict[int, TrainingResult] = {}
    started = time.perf_counter()

    for milestone in steps:
        while step < milestone:
            mean_loss, dataset_cursor, step_diagnostics = _training_step(
                campaign,
                model,
                optimizer,
                dataset,
                step=step,
                dataset_cursor=dataset_cursor,
            )
            loss_history.append(mean_loss)
            diagnostics.append(step_diagnostics)
            step += 1
            peak_rss = _peak_rss_bytes()
            if (
                maximum_peak_rss_bytes is not None
                and peak_rss is not None
                and peak_rss > maximum_peak_rss_bytes
            ):
                raise RuntimeError(
                    "diagnostic training exceeded its peak RSS limit"
                )
        checkpoint_dir = output_root / f"step-{step:08d}"
        checkpoint_dirs[step] = checkpoint_dir
        checkpoints[step] = _save_checkpoint(
            campaign,
            model,
            optimizer,
            checkpoint_dir,
            step,
            dataset_cursor,
            loss_history,
        )

    return DiagnosticTrainingResult(
        checkpoint_dirs=checkpoint_dirs,
        checkpoints=checkpoints,
        diagnostics=tuple(diagnostics),
        wall_seconds=time.perf_counter() - started,
    )


def _git_context(root: Path) -> Mapping[str, object]:
    def command(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def _behavioral_evaluation(
    *,
    campaign: CampaignConfig,
    checkpoint_dir: Path,
    packed_dir: Path,
    public_dir: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> Mapping[str, object]:
    started = time.perf_counter()
    model, optimizer, step, _, _ = _load_checkpoint(
        campaign,
        checkpoint_dir,
    )
    del optimizer
    model.eval()
    tokenizer = Tokenizer.from_file(str(packed_dir / "tokenizer.json"))
    _, examples = load_behavioral_split(
        public_dir=public_dir,
        split="behavioral_validation",
    )
    predictions = [
        {
            "id": example["id"],
            "output": _greedy_completion(
                model,
                tokenizer,
                str(example["prompt"]),
                max_new_tokens=max_new_tokens,
            ),
        }
        for example in examples
    ]
    predictions_path = output_dir / "predictions.jsonl"
    _write_exact(predictions_path, _prediction_bytes(predictions))
    evaluation = evaluate_predictions(
        public_dir=public_dir,
        split="behavioral_validation",
        predictions_path=predictions_path,
    )
    behavioral_contract = (
        campaign.research.get("behavioral_evaluation")
        if campaign.research is not None
        else None
    )
    if (
        not isinstance(behavioral_contract, Mapping)
        or evaluation["behavioral_suite_revision"]
        != behavioral_contract.get("dataset_revision")
        or evaluation["evaluator_revision"]
        != behavioral_contract.get("evaluator_revision")
    ):
        raise ValueError(
            "behavioral evaluation differs from the frozen campaign"
        )
    evaluation_path = output_dir / "evaluation.json"
    _write_exact(evaluation_path, _canonical_json_bytes(evaluation))
    del model
    gc.collect()
    return {
        "step": step,
        "predictions": predictions_path,
        "predictions_sha256": _sha256_file(predictions_path),
        "evaluation": evaluation_path,
        "evaluation_sha256": _sha256_file(evaluation_path),
        "metrics": evaluation["metrics"],
        "guardrails": evaluation["guardrails"],
        "wall_seconds": time.perf_counter() - started,
    }


def _environment(root: Path) -> Mapping[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_cuda_available": torch.cuda.is_available(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "source": _git_context(root),
    }


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_checksums(output_root: Path) -> None:
    paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not path.name.endswith(".tmp")
    )
    payload = "".join(
        f"{_sha256_file(path)}  {_relative_path(path, output_root)}\n"
        for path in paths
    ).encode("utf-8")
    _write_exact(output_root / "SHA256SUMS", payload)


def _learnability_markdown(evidence: Mapping[str, object]) -> bytes:
    selection = evidence["selection"]
    gate = evidence["learnability_gate"]
    resources = evidence["resources"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection evidence is invalid")
    if not isinstance(gate, Mapping):
        raise ValueError("learnability gate evidence is invalid")
    if not isinstance(resources, Mapping):
        raise ValueError("resource evidence is invalid")
    selected_metrics = gate["selected_behavioral_metrics"]
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("selected behavioral metrics are invalid")
    disposition = (
        "passed" if gate["passed"] is True else "did not pass"
    )
    text = (
        "# Record Patch bounded learnability check\n\n"
        f"The bounded public-data gate **{disposition}**. "
        "This is a pre-volunteer qualification result, not model promotion.\n\n"
        f"- Selected checkpoint: step `{selection['step']}`\n"
        f"- Initialization language loss: "
        f"`{gate['initial_language_mean_loss']}`\n"
        f"- Selected language loss: "
        f"`{gate['selected_language_mean_loss']}`\n"
        f"- Language-loss improvement: "
        f"`{gate['language_mean_loss_improvement']}`\n"
        f"- Selected exact matches: "
        f"`{selected_metrics['record_exact_match_count']}/"
        f"{selected_metrics['examples']}`\n"
        f"- Exact-match gain over initialization: "
        f"`{gate['behavioral_exact_match_count_improvement']}` examples\n"
        f"- Training wall time: `{resources['training_wall_seconds']}` seconds\n"
        f"- Peak process RSS: `{resources['peak_rss_bytes']}` bytes\n\n"
        "Checkpoint selection used only the public language-validation loss. "
        "The behavioral final holdout was not opened or accepted as a command "
        "argument. Public behavioral validation is diagnostic evidence only.\n\n"
        "The full machine-readable record includes every training-step "
        "gradient norm, clipping decision, parameter update norm, checkpoint "
        "identity, language evaluation, public prediction file, and evaluator "
        "result. `SHA256SUMS` covers the retained run files.\n"
    )
    return text.encode("utf-8")


def run_learnability_check(
    *,
    protocol_path: str | Path,
    campaign_path: str | Path,
    packed_dir: str | Path,
    public_dir: str | Path,
    output_dir: str | Path,
) -> Mapping[str, object]:
    run_started = time.perf_counter()
    output_root = Path(output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("learnability output directory is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_file = Path(campaign_path).resolve()
    packed_root = Path(packed_dir).resolve()
    public_root = Path(public_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    campaign = load_campaign(campaign_file)
    if campaign.campaign.get("id") != CAMPAIGN_ID:
        raise ValueError("learnability campaign identity is invalid")
    if campaign.research is None:
        raise ValueError("learnability campaign lacks a research contract")
    dataset = PackedDataset.load(packed_root)
    validate_dataset_artifacts(campaign, dataset)
    success_gate = (
        campaign.evaluation.get("success_gate")
        if campaign.evaluation is not None
        else None
    )
    if not isinstance(success_gate, Mapping):
        raise ValueError("campaign language success gate is invalid")
    minimum_language_improvement = float(
        success_gate["minimum_improvement_from_initialization"]
    )
    campaign_sha256 = _sha256_file(campaign_file)
    protocol = _load_protocol(
        protocol_path,
        campaign_sha256=campaign_sha256,
        dataset_revision=dataset.revision,
        research=campaign.research,
        minimum_language_improvement=minimum_language_improvement,
    )
    schedule = protocol["schedule"]
    resource_limits = protocol["resource_limits"]
    if not isinstance(schedule, Mapping):
        raise ValueError("validated learnability schedule is invalid")
    if not isinstance(resource_limits, Mapping):
        raise ValueError("validated learnability resource limits are invalid")
    steps = tuple(schedule["checkpoint_steps"])
    behavior = tuple(schedule["behavioral_steps"])
    max_new_tokens = int(schedule["max_new_tokens"])
    maximum_peak_rss_bytes = int(
        resource_limits["maximum_peak_rss_bytes"]
    )
    environment = _environment(project_root)
    _write_exact(
        output_root / "environment.json",
        _canonical_json_bytes(environment),
    )
    lock = {
        "format": "orcacolony_record_patch_learnability_lock_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": campaign_sha256,
        "dataset_revision": dataset.revision,
        "runner_revision": f"sha256:{_module_revision()}",
        "protocol": {
            "id": protocol["id"],
            "sha256": protocol["_sha256"],
        },
        "checkpoint_steps": list(steps),
        "requested_behavioral_steps": list(behavior),
        "decoding": {
            "method": "greedy",
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "stop": "first newline or EOS",
        },
        "data_access": {
            "training": "packed training split",
            "checkpoint_selection": "public language-validation slice",
            "behavioral_diagnostics": "public behavioral-validation split",
            "behavioral_final_holdout": "not opened",
        },
    }
    _write_exact(output_root / "run-lock.json", _canonical_json_bytes(lock))

    training = run_diagnostic_training(
        campaign=campaign,
        output_dir=output_root / "checkpoints",
        checkpoint_steps=steps,
        dataset=dataset,
        maximum_peak_rss_bytes=maximum_peak_rss_bytes,
    )
    _write_exact(
        output_root / "training-diagnostics.json",
        _canonical_json_bytes(list(training.diagnostics)),
    )

    language: dict[int, Mapping[str, object]] = {}
    for step in steps:
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            campaign,
            training.checkpoint_dirs[step],
            dataset,
        )
        path = output_root / "evaluations" / f"step-{step:08d}" / "language.json"
        _write_exact(path, _canonical_json_bytes(evaluation))
        language[step] = {
            "path": path,
            "sha256": _sha256_file(path),
            "evaluation": evaluation,
            "wall_seconds": time.perf_counter() - evaluation_started,
        }
        peak_rss = _peak_rss_bytes()
        if peak_rss is not None and peak_rss > maximum_peak_rss_bytes:
            raise RuntimeError(
                "language evaluation exceeded the peak RSS limit"
            )
        gc.collect()

    selected_step = min(
        steps,
        key=lambda step: (
            float(language[step]["evaluation"]["mean_loss"]),
            step,
        ),
    )
    evaluated_behavior_steps = tuple(
        sorted({0, selected_step, *behavior})
    )
    behavioral: dict[int, Mapping[str, object]] = {}
    for step in evaluated_behavior_steps:
        behavioral[step] = _behavioral_evaluation(
            campaign=campaign,
            checkpoint_dir=training.checkpoint_dirs[step],
            packed_dir=packed_root,
            public_dir=public_root,
            output_dir=(
                output_root
                / "evaluations"
                / f"step-{step:08d}"
                / "behavioral"
            ),
            max_new_tokens=max_new_tokens,
        )
        peak_rss = _peak_rss_bytes()
        if peak_rss is not None and peak_rss > maximum_peak_rss_bytes:
            raise RuntimeError(
                "behavioral evaluation exceeded the peak RSS limit"
            )

    baseline_contract = campaign.research.get("baseline")
    if not isinstance(baseline_contract, Mapping):
        raise ValueError("campaign baseline contract is invalid")
    initialization_revision = (
        f"sha256:{training.checkpoints[0].model_sha256}"
    )
    if baseline_contract.get("revision") != initialization_revision:
        raise ValueError("step-zero checkpoint differs from frozen baseline")
    initial_language_loss = float(language[0]["evaluation"]["mean_loss"])
    selected_language_loss = float(
        language[selected_step]["evaluation"]["mean_loss"]
    )
    initial_metrics = behavioral[0]["metrics"]
    selected_metrics = behavioral[selected_step]["metrics"]
    if not isinstance(initial_metrics, Mapping):
        raise ValueError("initial behavioral metrics are invalid")
    if not isinstance(selected_metrics, Mapping):
        raise ValueError("selected behavioral metrics are invalid")
    language_improvement = (
        initial_language_loss - selected_language_loss
    )
    exact_count_improvement = (
        int(selected_metrics["record_exact_match_count"])
        - int(initial_metrics["record_exact_match_count"])
    )
    language_passed = (
        selected_step > 0
        and language_improvement >= minimum_language_improvement
    )
    protocol_gate = protocol["gate"]
    if not isinstance(protocol_gate, Mapping):
        raise ValueError("validated learnability gate is invalid")
    minimum_exact_count_improvement = int(
        protocol_gate[
            "minimum_behavioral_exact_match_count_improvement"
        ]
    )
    behavioral_passed = (
        exact_count_improvement >= minimum_exact_count_improvement
    )
    peak_rss_bytes = _peak_rss_bytes()
    resource_limits_passed = (
        peak_rss_bytes is None
        or peak_rss_bytes <= maximum_peak_rss_bytes
    )
    gate = {
        "policy": (
            "lowest public language-validation mean loss, followed by at "
            "least the protocol's additional exact public "
            "behavioral-validation matches"
        ),
        "passed": (
            language_passed
            and behavioral_passed
            and resource_limits_passed
        ),
        "language_passed": language_passed,
        "behavioral_passed": behavioral_passed,
        "resource_limits_passed": resource_limits_passed,
        "minimum_language_mean_loss_improvement": (
            minimum_language_improvement
        ),
        "minimum_behavioral_exact_match_count_improvement": (
            minimum_exact_count_improvement
        ),
        "initial_language_mean_loss": initial_language_loss,
        "selected_language_mean_loss": selected_language_loss,
        "language_mean_loss_improvement": language_improvement,
        "initial_behavioral_metrics": initial_metrics,
        "selected_behavioral_metrics": selected_metrics,
        "behavioral_exact_match_count_improvement": (
            exact_count_improvement
        ),
        "scope": (
            "pre-volunteer public-data qualification; not a promotion "
            "decision"
        ),
    }
    diagnostics = list(training.diagnostics)
    language_evaluation_wall_seconds = sum(
        float(language[step]["wall_seconds"]) for step in steps
    )
    behavioral_evaluation_wall_seconds = sum(
        float(behavioral[step]["wall_seconds"])
        for step in evaluated_behavior_steps
    )
    resources = {
        "training_wall_seconds": training.wall_seconds,
        "language_evaluation_wall_seconds": (
            language_evaluation_wall_seconds
        ),
        "behavioral_evaluation_wall_seconds": (
            behavioral_evaluation_wall_seconds
        ),
        "total_wall_seconds_before_packaging": (
            time.perf_counter() - run_started
        ),
        "total_step_wall_seconds": sum(
            float(item["wall_seconds"]) for item in diagnostics
        ),
        "steps_completed": steps[-1],
        "loss_weight_sum": sum(
            int(item["loss_weight_sum"]) for item in diagnostics
        ),
        "clipped_steps": sum(
            int(item["clipped"] is True) for item in diagnostics
        ),
        "peak_rss_bytes": peak_rss_bytes,
        "maximum_peak_rss_bytes": maximum_peak_rss_bytes,
        "execution": "owner-operated centralized CPU reference",
        "donated_compute": False,
    }
    checkpoint_records = [
        {
            "step": step,
            "model_sha256": training.checkpoints[step].model_sha256,
            "checkpoint_state_sha256": _sha256_file(
                training.checkpoint_dirs[step] / "state.json"
            ),
            "language_evaluation": {
                "path": _relative_path(
                    language[step]["path"],
                    output_root,
                ),
                "sha256": language[step]["sha256"],
                "mean_loss": language[step]["evaluation"]["mean_loss"],
                "perplexity": language[step]["evaluation"]["perplexity"],
                "wall_seconds": language[step]["wall_seconds"],
            },
            "behavioral_evaluation": (
                {
                    "predictions_path": _relative_path(
                        behavioral[step]["predictions"],
                        output_root,
                    ),
                    "predictions_sha256": behavioral[step][
                        "predictions_sha256"
                    ],
                    "evaluation_path": _relative_path(
                        behavioral[step]["evaluation"],
                        output_root,
                    ),
                    "evaluation_sha256": behavioral[step][
                        "evaluation_sha256"
                    ],
                    "metrics": behavioral[step]["metrics"],
                    "guardrails": behavioral[step]["guardrails"],
                    "wall_seconds": behavioral[step]["wall_seconds"],
                }
                if step in behavioral
                else None
            ),
        }
        for step in steps
    ]
    evidence = {
        "format": "orcacolony_record_patch_learnability_evidence_v1",
        "campaign_id": CAMPAIGN_ID,
        "campaign_sha256": campaign_sha256,
        "dataset_revision": dataset.revision,
        "runner_revision": f"sha256:{_module_revision()}",
        "protocol": {
            "id": protocol["id"],
            "sha256": protocol["_sha256"],
        },
        "initialization_revision": initialization_revision,
        "checkpoint_selection_policy": (
            "lowest public language-validation mean loss; behavioral "
            "metrics do not select the checkpoint"
        ),
        "selection": {
            "step": selected_step,
            "model_sha256": training.checkpoints[
                selected_step
            ].model_sha256,
        },
        "learnability_gate": gate,
        "resources": resources,
        "checkpoints": checkpoint_records,
        "diagnostics": {
            "path": "training-diagnostics.json",
            "sha256": _sha256_file(
                output_root / "training-diagnostics.json"
            ),
            "steps": len(diagnostics),
        },
        "environment": {
            "path": "environment.json",
            "sha256": _sha256_file(output_root / "environment.json"),
        },
        "holdout": {
            "behavioral_final_holdout_opened": False,
            "language_final_holdout_evaluated": False,
        },
        "limitations": [
            *protocol["limitations"],
            (
                "A passed learnability gate permits planning volunteer "
                "training; it does not classify a capability model."
            ),
        ],
    }
    _write_exact(output_root / "evidence.json", _canonical_json_bytes(evidence))
    _write_exact(
        output_root / "LEARNABILITY.md",
        _learnability_markdown(evidence),
    )
    _write_checksums(output_root)
    return evidence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded centralized Record Patch learnability check"
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evidence = run_learnability_check(
        protocol_path=args.protocol,
        campaign_path=args.campaign,
        packed_dir=args.packed_dir,
        public_dir=args.public_dir,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_checkpoint": evidence["selection"],
                "learnability_gate": evidence["learnability_gate"],
                "holdout": evidence["holdout"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
