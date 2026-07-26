from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save as save_safetensors
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor, nn
from torch.nn import functional as F

from .artifacts import PackedDataset


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


_CHECKPOINT_STATE_FIELDS = frozenset(
    {
        "format",
        "campaign_id",
        "architecture",
        "architecture_revision",
        "step",
        "optimizer_step",
        "dataset_revision",
        "dataset_cursor",
        "loss_history",
        "model",
        "optimizer",
    }
)
_CHECKPOINT_ARTIFACT_FIELDS = frozenset({"file", "sha256"})


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    architecture_revision: int
    layers: int
    width: int
    heads: int
    mlp_width: int
    vocabulary_size: int
    context_length: int
    positional_encoding: str
    layer_norm_epsilon: float
    gelu_approximation: str
    attention_bias: bool
    linear_bias: bool
    tied_token_embeddings: bool
    parameters: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    dataset_sequences: int
    active_vocabulary_size: int
    steps: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_gradient_norm: float
    compute_dtype: str
    gradient_accumulation_dtype: str


@dataclass(frozen=True)
class CampaignConfig:
    campaign: Mapping[str, object]
    model: ModelConfig
    training: TrainingConfig
    dataset: Mapping[str, object] | None = None
    evaluation: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FixtureResult:
    parameter_count: int
    loss_sum: float
    loss_weight_sum: int
    gradient_sha256: str


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_dir: Path
    steps_completed: int
    loss_history: tuple[float, ...]
    model_sha256: str


@dataclass(frozen=True)
class FixtureExportResult:
    output_dir: Path
    loss_sum: float
    loss_weight_sum: int
    gradient_sha256: str


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.width % config.heads != 0:
            raise ValueError("model width must be divisible by attention heads")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(
            config.width,
            3 * config.width,
            bias=config.attention_bias,
        )
        self.output = nn.Linear(
            config.width,
            config.width,
            bias=config.linear_bias,
        )
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, tokens, width = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(value: Tensor) -> Tensor:
            return value.view(batch, tokens, self.heads, self.head_width).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(self.causal_mask[:tokens, :tokens], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, width)
        return self.output(attended)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, config.mlp_width, bias=config.linear_bias),
            nn.GELU(approximate=config.gelu_approximation),
            nn.Linear(config.mlp_width, config.width, bias=config.linear_bias),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class VolunteerDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture != "volunteer_decoder_v1":
            raise ValueError(f"unsupported architecture: {config.architecture}")
        if config.positional_encoding != "learned_absolute":
            raise ValueError("volunteer_decoder_v1 requires learned absolute positions")
        if not config.tied_token_embeddings:
            raise ValueError("volunteer_decoder_v1 requires tied token embeddings")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width, eps=config.layer_norm_epsilon)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor) -> Tensor:
        _, tokens = token_ids.shape
        if tokens > self.config.context_length:
            raise ValueError("input exceeds configured context length")
        positions = torch.arange(tokens, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight)


def campaign_from_mapping(payload: Mapping[str, object]) -> CampaignConfig:
    return CampaignConfig(
        campaign=cast(Mapping[str, object], payload["campaign"]),
        model=ModelConfig(**cast(Mapping[str, Any], payload["model"])),
        training=TrainingConfig(**cast(Mapping[str, Any], payload["training"])),
        dataset=cast(Mapping[str, object] | None, payload.get("dataset")),
        evaluation=cast(Mapping[str, object] | None, payload.get("evaluation")),
    )


def load_campaign(path: str | Path) -> CampaignConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("campaign configuration must be a JSON object")
    return campaign_from_mapping(payload)


def configure_determinism(seed: int) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def build_model(campaign: CampaignConfig) -> VolunteerDecoder:
    configure_determinism(campaign.training.seed)
    model = VolunteerDecoder(campaign.model)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != campaign.model.parameters:
        raise ValueError(
            f"model parameter count mismatch: config={campaign.model.parameters}, actual={actual}"
        )
    return model


def validate_dataset_artifacts(
    campaign: CampaignConfig,
    dataset: PackedDataset | None,
) -> None:
    if campaign.dataset is None:
        if dataset is not None:
            raise ValueError("campaign does not declare external dataset artifacts")
        if campaign.evaluation is not None:
            raise ValueError("campaign evaluation requires external dataset artifacts")
        return
    if dataset is None:
        raise ValueError("campaign requires external dataset artifacts")
    expected = campaign.dataset
    checks = {
        "manifest_sha256": dataset.revision,
        "tokenizer_sha256": dataset.manifest["tokenizer"]["sha256"],
        "train_sha256": dataset.manifest["files"]["train.safetensors"],
        "validation_sha256": dataset.manifest["files"]["validation.safetensors"],
    }
    for key, actual in checks.items():
        if expected.get(key) != actual:
            raise ValueError(f"campaign dataset identity mismatch: {key}")
    if int(dataset.manifest["packing"]["context_length"]) != campaign.model.context_length:
        raise ValueError("campaign and dataset context lengths do not match")
    if int(dataset.manifest["tokenizer"]["vocab_size"]) > campaign.model.vocabulary_size:
        raise ValueError("tokenizer vocabulary exceeds the campaign model vocabulary")
    if campaign.training.dataset_sequences > dataset.train_inputs.shape[0]:
        raise ValueError("campaign requests more sequences than the packed dataset contains")
    if int(dataset.train_targets.max()) >= campaign.model.vocabulary_size:
        raise ValueError("packed dataset token ID exceeds the model vocabulary")
    if campaign.evaluation is not None:
        if campaign.evaluation.get("metric", "held_out_cross_entropy") != (
            "held_out_cross_entropy"
        ):
            raise ValueError("unsupported campaign evaluation metric")
        if campaign.evaluation.get(
            "checkpoint_selection", "lowest_mean_loss"
        ) != "lowest_mean_loss":
            raise ValueError("unsupported checkpoint selection rule")
        validation_sequences = int(campaign.evaluation["validation_sequences"])
        evaluation_batch_size = int(campaign.evaluation["batch_size"])
        if validation_sequences <= 0 or evaluation_batch_size <= 0:
            raise ValueError("evaluation dimensions must be positive")
        if evaluation_batch_size > validation_sequences:
            raise ValueError("evaluation batch size exceeds validation sequence count")
        success_gate = campaign.evaluation.get("success_gate")
        if success_gate is not None:
            if not isinstance(success_gate, Mapping):
                raise ValueError("evaluation success gate must be a mapping")
            if success_gate.get("metric") != "mean_loss":
                raise ValueError("unsupported evaluation success-gate metric")
            minimum_value = success_gate.get(
                "minimum_improvement_from_initialization"
            )
            if isinstance(minimum_value, bool) or not isinstance(
                minimum_value, (int, float)
            ):
                raise ValueError("evaluation success-gate improvement must be numeric")
            minimum_improvement = float(minimum_value)
            if not math.isfinite(minimum_improvement) or minimum_improvement <= 0:
                raise ValueError("evaluation success-gate improvement must be positive")
        if validation_sequences > dataset.validation_inputs.shape[0]:
            raise ValueError(
                "campaign evaluation exceeds the packed validation dataset"
            )


def fixture_batch(
    campaign: CampaignConfig,
    cursor: int = 0,
    dataset: PackedDataset | None = None,
) -> tuple[Tensor, Tensor]:
    model = campaign.model
    training = campaign.training
    if campaign.dataset is not None:
        if dataset is None:
            raise ValueError("campaign requires external dataset artifacts")
        return dataset.batch(
            cursor=cursor,
            batch_size=training.batch_size,
            sequence_limit=training.dataset_sequences,
        )
    positions = torch.arange(model.context_length + 1, dtype=torch.long)
    rows = []
    for offset in range(training.batch_size):
        sequence_id = (cursor + offset) % training.dataset_sequences
        tokens = (
            sequence_id * 7 + positions
        ) % training.active_vocabulary_size
        rows.append(tokens + 1)
    batch = torch.stack(rows)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


def tensor_sha256(tensors: Mapping[str, Tensor]) -> str:
    canonical = {
        name: tensors[name].detach().cpu().contiguous()
        for name in sorted(tensors)
    }
    return hashlib.sha256(save_safetensors(canonical)).hexdigest()


def compute_fixture(
    campaign: CampaignConfig,
    dataset: PackedDataset | None = None,
) -> FixtureResult:
    validate_dataset_artifacts(campaign, dataset)
    model = build_model(campaign)
    inputs, targets = fixture_batch(campaign, dataset=dataset)
    logits = model(inputs)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="sum",
    )
    loss_sum.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return FixtureResult(
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        loss_sum=float(loss_sum.detach()),
        loss_weight_sum=targets.numel(),
        gradient_sha256=tensor_sha256(gradients),
    )


def _create_optimizer(
    model: VolunteerDecoder,
    training: TrainingConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon,
        weight_decay=training.weight_decay,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(
    campaign: CampaignConfig,
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
    output_dir: Path,
    step: int,
    dataset_cursor: int,
    loss_history: list[float],
) -> TrainingResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"
    optimizer_path = output_dir / "optimizer.safetensors"
    state_path = output_dir / "state.json"
    model_tmp = output_dir / "model.safetensors.tmp"
    optimizer_tmp = output_dir / "optimizer.safetensors.tmp"
    state_tmp = output_dir / "state.json.tmp"

    model_tensors = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    save_safetensors_file(model_tensors, str(model_tmp))
    os.replace(model_tmp, model_path)

    optimizer_tensors: dict[str, Tensor] = {}
    optimizer_step = step
    for name, parameter in model.named_parameters():
        parameter_state = optimizer.state[parameter]
        optimizer_tensors[f"exp_avg.{name}"] = (
            parameter_state.get("exp_avg", torch.zeros_like(parameter))
            .detach()
            .cpu()
            .contiguous()
        )
        optimizer_tensors[f"exp_avg_sq.{name}"] = (
            parameter_state.get("exp_avg_sq", torch.zeros_like(parameter))
            .detach()
            .cpu()
            .contiguous()
        )
        optimizer_step = int(parameter_state.get("step", torch.tensor(step)).item())
    save_safetensors_file(optimizer_tensors, str(optimizer_tmp))
    os.replace(optimizer_tmp, optimizer_path)

    state = {
        "format": "orcacolony_checkpoint_v1",
        "campaign_id": campaign.campaign["id"],
        "architecture": campaign.model.architecture,
        "architecture_revision": campaign.model.architecture_revision,
        "step": step,
        "optimizer_step": optimizer_step,
        "dataset_cursor": dataset_cursor,
        "dataset_revision": (
            campaign.dataset["manifest_sha256"]
            if campaign.dataset is not None
            else "synthetic-fixture-v1"
        ),
        "loss_history": loss_history,
        "model": {
            "file": model_path.name,
            "sha256": _sha256_file(model_path),
        },
        "optimizer": {
            "file": optimizer_path.name,
            "sha256": _sha256_file(optimizer_path),
        },
    }
    state_tmp.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(state_tmp, state_path)
    return TrainingResult(
        checkpoint_dir=output_dir,
        steps_completed=step,
        loss_history=tuple(loss_history),
        model_sha256=state["model"]["sha256"],
    )


def _load_checkpoint(
    campaign: CampaignConfig,
    checkpoint_dir: Path,
) -> tuple[VolunteerDecoder, torch.optim.AdamW, int, int, list[float]]:
    state = json.loads(
        (checkpoint_dir / "state.json").read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(state, dict) or frozenset(state) != _CHECKPOINT_STATE_FIELDS:
        raise ValueError("checkpoint state schema is invalid")
    if state["format"] != "orcacolony_checkpoint_v1":
        raise ValueError("unsupported checkpoint format")
    if state["campaign_id"] != campaign.campaign["id"]:
        raise ValueError("checkpoint campaign does not match configuration")
    if (
        type(state["architecture_revision"]) is not int
        or state["architecture_revision"] != campaign.model.architecture_revision
    ):
        raise ValueError("checkpoint architecture revision does not match configuration")
    expected_dataset_revision = (
        campaign.dataset["manifest_sha256"]
        if campaign.dataset is not None
        else "synthetic-fixture-v1"
    )
    if state.get("dataset_revision", "synthetic-fixture-v1") != expected_dataset_revision:
        raise ValueError("checkpoint dataset revision does not match configuration")
    if type(state.get("step")) is not int or state["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if type(state.get("optimizer_step")) is not int or state["optimizer_step"] < 0:
        raise ValueError("checkpoint optimizer step must be a nonnegative integer")
    if type(state.get("dataset_cursor")) is not int or state["dataset_cursor"] < 0:
        raise ValueError("checkpoint dataset cursor must be a nonnegative integer")
    loss_history = state.get("loss_history")
    if not isinstance(loss_history, list) or any(
        type(loss) is not float or not math.isfinite(loss) for loss in loss_history
    ):
        raise ValueError("checkpoint loss history must contain finite JSON floats")

    model_artifact = state.get("model")
    optimizer_artifact = state.get("optimizer")
    if (
        not isinstance(model_artifact, dict)
        or frozenset(model_artifact) != _CHECKPOINT_ARTIFACT_FIELDS
        or model_artifact.get("file") != "model.safetensors"
        or not isinstance(optimizer_artifact, dict)
        or frozenset(optimizer_artifact) != _CHECKPOINT_ARTIFACT_FIELDS
        or optimizer_artifact.get("file") != "optimizer.safetensors"
    ):
        raise ValueError("checkpoint artifact schema is invalid")
    model_path = checkpoint_dir / model_artifact["file"]
    optimizer_path = checkpoint_dir / optimizer_artifact["file"]
    checkpoint_root = checkpoint_dir.resolve(strict=True)
    for artifact_path in (model_path, optimizer_path):
        metadata = artifact_path.lstat()
        if (
            artifact_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or int(getattr(metadata, "st_file_attributes", 0)) & 0x400
            or not artifact_path.resolve(strict=True).is_relative_to(checkpoint_root)
        ):
            raise ValueError("checkpoint artifact path is unsafe")
    if _sha256_file(model_path) != model_artifact["sha256"]:
        raise ValueError("model checkpoint digest mismatch")
    if _sha256_file(optimizer_path) != optimizer_artifact["sha256"]:
        raise ValueError("optimizer checkpoint digest mismatch")

    model = build_model(campaign)
    model.load_state_dict(load_safetensors_file(str(model_path)))
    optimizer = _create_optimizer(model, campaign.training)
    optimizer_tensors = load_safetensors_file(str(optimizer_path))
    expected_optimizer_tensors = {
        f"{prefix}.{name}"
        for name, _ in model.named_parameters()
        for prefix in ("exp_avg", "exp_avg_sq")
    }
    if set(optimizer_tensors) != expected_optimizer_tensors:
        raise ValueError("optimizer checkpoint tensor schema is invalid")
    optimizer_step = float(state["optimizer_step"])
    for name, parameter in model.named_parameters():
        for prefix in ("exp_avg", "exp_avg_sq"):
            tensor = optimizer_tensors[f"{prefix}.{name}"]
            if (
                tensor.dtype != parameter.dtype
                or tensor.shape != parameter.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError("optimizer checkpoint tensor is invalid")
        optimizer.state[parameter] = {
            "step": torch.tensor(optimizer_step),
            "exp_avg": optimizer_tensors[f"exp_avg.{name}"].clone(),
            "exp_avg_sq": optimizer_tensors[f"exp_avg_sq.{name}"].clone(),
        }
    return (
        model,
        optimizer,
        state["step"],
        state["dataset_cursor"],
        list(loss_history),
    )


def evaluate_checkpoint(
    campaign: CampaignConfig,
    checkpoint_dir: str | Path,
    dataset: PackedDataset,
) -> dict[str, object]:
    validate_dataset_artifacts(campaign, dataset)
    if campaign.evaluation is None:
        raise ValueError("campaign does not define an evaluation profile")
    checkpoint_dir = Path(checkpoint_dir)
    model, _, step, _, _ = _load_checkpoint(campaign, checkpoint_dir)
    sequence_count = int(campaign.evaluation["validation_sequences"])
    batch_size = int(campaign.evaluation["batch_size"])
    loss_sum = 0.0
    loss_weight_sum = 0
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
            batch_loss = F.cross_entropy(
                logits.reshape(-1, campaign.model.vocabulary_size),
                targets.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(batch_loss)
            loss_weight_sum += targets.numel()
            cursor += current_batch_size
    mean_loss = loss_sum / loss_weight_sum
    checkpoint_state = json.loads(
        (checkpoint_dir / "state.json").read_text(encoding="utf-8")
    )
    return {
        "format": "orcacolony_evaluation_v1",
        "campaign_id": campaign.campaign["id"],
        "step": step,
        "dataset_revision": dataset.revision,
        "checkpoint_sha256": checkpoint_state["model"]["sha256"],
        "validation_sequences": sequence_count,
        "loss_sum": loss_sum,
        "loss_weight_sum": loss_weight_sum,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def run_training(
    campaign: CampaignConfig,
    output_dir: str | Path,
    target_steps: int | None = None,
    resume_from: str | Path | None = None,
    dataset: PackedDataset | None = None,
) -> TrainingResult:
    validate_dataset_artifacts(campaign, dataset)
    output_dir = Path(output_dir)
    target_steps = campaign.training.steps if target_steps is None else target_steps
    if resume_from is None:
        model = build_model(campaign)
        optimizer = _create_optimizer(model, campaign.training)
        step = 0
        dataset_cursor = 0
        loss_history: list[float] = []
    else:
        model, optimizer, step, dataset_cursor, loss_history = _load_checkpoint(
            campaign,
            Path(resume_from),
        )
    if target_steps < step or (target_steps == step and resume_from is not None):
        raise ValueError("target_steps must be greater than the checkpoint step")

    model.train()
    while step < target_steps:
        inputs, targets = fixture_batch(campaign, dataset_cursor, dataset)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, campaign.model.vocabulary_size),
            targets.reshape(-1),
            reduction="sum",
        )
        loss_weight_sum = targets.numel()
        loss_sum.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_weight_sum)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            campaign.training.max_gradient_norm,
        )
        optimizer.step()

        loss_history.append(float(loss_sum.detach()) / loss_weight_sum)
        dataset_cursor = (
            dataset_cursor + campaign.training.batch_size
        ) % campaign.training.dataset_sequences
        step += 1

    return _save_checkpoint(
        campaign,
        model,
        optimizer,
        output_dir,
        step,
        dataset_cursor,
        loss_history,
    )


def export_fixture(
    campaign: CampaignConfig,
    output_dir: str | Path,
    dataset: PackedDataset | None = None,
) -> FixtureExportResult:
    validate_dataset_artifacts(campaign, dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(campaign)
    inputs, targets = fixture_batch(campaign, dataset=dataset)
    logits = model(inputs)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, campaign.model.vocabulary_size),
        targets.reshape(-1),
        reduction="sum",
    )
    loss_sum.backward()

    model_path = output_dir / "model.safetensors"
    batch_path = output_dir / "batch.safetensors"
    gradient_path = output_dir / "gradients.safetensors"
    save_safetensors_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in sorted(model.state_dict().items())
        },
        str(model_path),
    )
    save_safetensors_file(
        {
            "input_ids": inputs.detach().clone().contiguous(),
            "target_ids": targets.detach().clone().contiguous(),
        },
        str(batch_path),
    )
    gradients = {
        name: parameter.grad.detach().cpu().contiguous()
        for name, parameter in sorted(model.named_parameters())
        if parameter.grad is not None
    }
    save_safetensors_file(gradients, str(gradient_path))

    files = {
        path.name: _sha256_file(path)
        for path in (model_path, batch_path, gradient_path)
    }
    manifest = {
        "format": "orcacolony_reference_fixture_v1",
        "campaign_id": campaign.campaign["id"],
        "architecture": campaign.model.architecture,
        "architecture_revision": campaign.model.architecture_revision,
        "model": {
            "vocab_size": campaign.model.vocabulary_size,
            "context_length": campaign.model.context_length,
            "d_model": campaign.model.width,
            "num_heads": campaign.model.heads,
            "num_layers": campaign.model.layers,
            "d_ff": campaign.model.mlp_width,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "objective": campaign.campaign["objective"],
        "loss_mask": campaign.campaign["loss_mask"],
        "loss_reduction": "sum",
        "dataset_revision": (
            dataset.revision if dataset is not None else "synthetic-fixture-v1"
        ),
        "loss_sum": float(loss_sum.detach()),
        "loss_weight_sum": targets.numel(),
        "compute_dtype": campaign.training.compute_dtype,
        "input_ids": inputs.reshape(-1).tolist(),
        "input_shape": list(inputs.shape),
        "target_ids": targets.reshape(-1).tolist(),
        "target_shape": list(targets.shape),
        "tensor_order": sorted(gradients),
        "files": files,
    }
    (output_dir / "fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FixtureExportResult(
        output_dir=output_dir,
        loss_sum=float(loss_sum.detach()),
        loss_weight_sum=targets.numel(),
        gradient_sha256=files[gradient_path.name],
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OrcaColony single-process reference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run the deterministic T0 training loop")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int)
    train.add_argument("--resume", type=Path)
    train.add_argument("--dataset-artifacts", type=Path)

    fixture = subparsers.add_parser(
        "fixture",
        help="export the language-neutral browser parity fixture",
    )
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--dataset-artifacts", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    campaign = load_campaign(args.config)
    dataset = (
        PackedDataset.load(args.dataset_artifacts)
        if args.dataset_artifacts is not None
        else None
    )
    if args.command == "train":
        result = run_training(
            campaign,
            output_dir=args.output,
            target_steps=args.steps,
            resume_from=args.resume,
            dataset=dataset,
        )
        summary = {
            "campaign_id": campaign.campaign["id"],
            "checkpoint_dir": str(result.checkpoint_dir),
            "steps_completed": result.steps_completed,
            "parameter_count": campaign.model.parameters,
            "initial_loss": result.loss_history[0],
            "final_loss": result.loss_history[-1],
            "loss_history": result.loss_history,
            "model_sha256": result.model_sha256,
        }
    else:
        fixture = export_fixture(campaign, args.output, dataset=dataset)
        summary = {
            "campaign_id": campaign.campaign["id"],
            "fixture_dir": str(fixture.output_dir),
            "loss_sum": fixture.loss_sum,
            "loss_weight_sum": fixture.loss_weight_sum,
            "gradient_sha256": fixture.gradient_sha256,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
