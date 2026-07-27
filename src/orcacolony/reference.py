from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor, nn
from torch.nn import functional as F

from .artifacts import PackedDataset
from .campaign_research import validate_campaign_research_contract


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
class ObjectiveConfig:
    name: str
    loss_mask: str


@dataclass(frozen=True)
class EvaluationSlice:
    name: str
    start_sequence: int
    sequence_count: int
    batch_size: int


@dataclass(frozen=True)
class CampaignConfig:
    campaign: Mapping[str, object]
    objective: ObjectiveConfig
    model: ModelConfig
    training: TrainingConfig
    dataset: Mapping[str, object] | None = None
    evaluation: Mapping[str, object] | None = None
    research: Mapping[str, object] | None = None
    publication: Mapping[str, object] | None = None


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
    def __init__(
        self,
        config: ModelConfig,
        objective: ObjectiveConfig | None = None,
    ) -> None:
        super().__init__()
        if config.architecture != "volunteer_decoder_v1":
            raise ValueError(f"unsupported architecture: {config.architecture}")
        if config.positional_encoding != "learned_absolute":
            raise ValueError("volunteer_decoder_v1 requires learned absolute positions")
        if not config.tied_token_embeddings:
            raise ValueError("volunteer_decoder_v1 requires tied token embeddings")

        self.config = config
        self.objective = objective or ObjectiveConfig(
            name="causal_lm",
            loss_mask="all_target_tokens",
        )
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


def _objective_from_mapping(payload: Mapping[str, object]) -> ObjectiveConfig:
    allowed = {"id", "objective", "loss_mask"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "campaign metadata contains unknown fields: " + ", ".join(unknown)
        )
    campaign_id = payload.get("id")
    if (
        not isinstance(campaign_id, str)
        or not campaign_id
        or len(campaign_id) > 128
        or any(
            character
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789._-"
            )
            for character in campaign_id
        )
    ):
        raise ValueError(
            "campaign id must use 1-128 letters, digits, dots, underscores, or hyphens"
        )
    objective = payload.get("objective")
    loss_mask = payload.get("loss_mask")
    if objective != "causal_lm":
        raise ValueError(f"unsupported campaign objective: {objective!r}")
    if loss_mask != "all_target_tokens":
        raise ValueError(f"unsupported campaign loss mask: {loss_mask!r}")
    return ObjectiveConfig(name=objective, loss_mask=loss_mask)


def _objective_loss(
    objective: ObjectiveConfig,
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str,
) -> tuple[Tensor, int]:
    if objective.name != "causal_lm":
        raise ValueError(f"unsupported campaign objective: {objective.name!r}")
    if objective.loss_mask != "all_target_tokens":
        raise ValueError(f"unsupported campaign loss mask: {objective.loss_mask!r}")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unsupported objective loss reduction: {reduction!r}")
    if logits.ndim < 2 or targets.shape != logits.shape[:-1]:
        raise ValueError("objective logits and targets have incompatible shapes")
    if targets.dtype != torch.long:
        raise ValueError("objective targets must be int64 token ids")
    weight_sum = targets.numel()
    if weight_sum <= 0:
        raise ValueError("objective batch must contain at least one target token")
    return (
        F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction=reduction,
        ),
        weight_sum,
    )


def objective_loss_sum(
    objective: ObjectiveConfig,
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, int]:
    """Execute the declared objective with coordinator-compatible summed loss.

    Only causal LM over every target token is implemented today. Campaign
    loading rejects any other declaration, so future SFT or masked objectives
    cannot silently fall back to this loss.
    """

    return _objective_loss(
        objective,
        logits,
        targets,
        reduction="sum",
    )


def objective_mean_loss(
    objective: ObjectiveConfig,
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Execute the declared objective with PyTorch's exact mean reduction."""

    loss, _ = _objective_loss(
        objective,
        logits,
        targets,
        reduction="mean",
    )
    return loss


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _required_revision(value: object, label: str) -> str:
    revision = _required_text(value, label)
    is_sha256 = revision.startswith("sha256:")
    raw_digest = revision.removeprefix("sha256:") if is_sha256 else revision
    expected_length = 64 if is_sha256 else 40
    if (
        len(raw_digest) != expected_length
        or any(character not in "0123456789abcdef" for character in raw_digest)
    ):
        raise ValueError(
            f"{label} must be a 40-character lowercase Git revision or "
            "sha256: followed by 64 lowercase hexadecimal characters"
        )
    return revision


def _required_huggingface_repo_id(value: object, label: str) -> str:
    repo_id = _required_text(value, label)
    prefix = "OrcaColony/"
    name = repo_id.removeprefix(prefix)
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )
    if (
        not repo_id.startswith(prefix)
        or "/" in name
        or not 1 <= len(name) <= 96
        or not name[0].isalnum()
        or not name[-1].isalnum()
        or "--" in name
        or ".." in name
        or any(character not in allowed for character in name)
    ):
        raise ValueError(
            f"{label} must use one valid repository in the OrcaColony namespace"
        )
    return repo_id


def _required_license_id(value: object, label: str) -> str:
    license_id = _required_text(value, label)
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789.+-"
    )
    if (
        len(license_id) > 64
        or any(character not in allowed for character in license_id)
        or license_id.casefold()
        in {"choose-explicitly", "other", "replace-me", "tbd", "unknown"}
        or license_id.casefold().startswith("replace-")
    ):
        raise ValueError(f"{label} must be an explicit license identifier")
    return license_id


def _validate_publication_contract(
    publication: Mapping[str, object] | None,
) -> None:
    if publication is None:
        return
    if publication.get("format") != "orcacolony_huggingface_publication_v1":
        raise ValueError("unsupported campaign publication contract")
    required_publication = {
        "format",
        "model_repo_id",
        "dataset_repo_id",
        "model_license",
        "dataset_license",
        "visibility_policy",
    }
    if set(publication) != required_publication:
        raise ValueError("campaign publication contract is invalid")
    for field in ("model_repo_id", "dataset_repo_id"):
        _required_huggingface_repo_id(
            publication.get(field),
            f"publication {field}",
        )
    for field in ("model_license", "dataset_license"):
        _required_license_id(
            publication.get(field),
            f"publication {field}",
        )
    if publication.get("visibility_policy") not in {
        "private",
        "public",
        "private_review_then_public",
    }:
        raise ValueError("publication visibility policy is invalid")


def _validate_legacy_capability_contract(campaign: CampaignConfig) -> None:
    research = campaign.research
    if research is None:
        return
    if research.get("format") != "orcacolony_capability_research_v1":
        raise ValueError("unsupported campaign research contract")
    allowed = {
        "format",
        "claim",
        "baseline",
        "primary_metric",
        "guardrails",
        "analysis_plan",
        "final_holdout_policy",
        "behavioral_evaluation",
        "checkpoint_selection",
    }
    unknown = sorted(set(research) - allowed)
    if unknown:
        raise ValueError(
            "capability research contains unknown fields: " + ", ".join(unknown)
        )
    if set(research) != allowed:
        raise ValueError("capability research contract is incomplete")
    _required_text(research.get("claim"), "capability claim")
    for key in ("baseline", "primary_metric"):
        value = research.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"capability {key} must be an object")
    baseline = cast(Mapping[str, object], research["baseline"])
    if set(baseline) != {"id", "description", "revision"}:
        raise ValueError("capability baseline contract is invalid")
    for field in ("id", "description", "revision"):
        _required_text(baseline.get(field), f"capability baseline {field}")
    _required_revision(
        baseline.get("revision"),
        "capability baseline revision",
    )
    primary_metric = cast(Mapping[str, object], research["primary_metric"])
    if set(primary_metric) != {
        "id",
        "description",
        "direction",
        "unit",
        "success_threshold",
        "minimum_improvement_from_baseline",
    }:
        raise ValueError("capability primary metric contract is invalid")
    for field in ("id", "description", "unit"):
        _required_text(
            primary_metric.get(field),
            f"capability primary metric {field}",
        )
    if primary_metric.get("direction") not in {"minimize", "maximize"}:
        raise ValueError("capability primary metric direction is invalid")
    threshold = primary_metric.get("success_threshold")
    minimum_improvement = primary_metric.get(
        "minimum_improvement_from_baseline"
    )
    for value, label in (
        (threshold, "threshold"),
        (minimum_improvement, "minimum improvement"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"capability primary metric {label} must be finite"
            )
    if float(minimum_improvement) <= 0:
        raise ValueError(
            "capability primary metric minimum improvement must be positive"
        )
    guardrails = research.get("guardrails")
    if not isinstance(guardrails, list) or not guardrails:
        raise ValueError("capability research requires guardrails")
    guardrail_ids: set[str] = set()
    for raw_guardrail in guardrails:
        if not isinstance(raw_guardrail, Mapping) or set(raw_guardrail) != {
            "id",
            "description",
        }:
            raise ValueError("capability guardrail contract is invalid")
        identifier = _required_text(
            raw_guardrail.get("id"),
            "capability guardrail id",
        )
        if identifier in guardrail_ids:
            raise ValueError("capability guardrail IDs must be unique")
        guardrail_ids.add(identifier)
        _required_text(
            raw_guardrail.get("description"),
            "capability guardrail description",
        )
    analysis_plan = research.get("analysis_plan")
    if (
        not isinstance(analysis_plan, list)
        or not analysis_plan
        or any(
            not isinstance(item, str) or not item.strip()
            for item in analysis_plan
        )
    ):
        raise ValueError("capability research requires an analysis plan")
    if research.get("final_holdout_policy") != (
        "release_only_after_checkpoint_selection"
    ):
        raise ValueError("capability final-holdout policy is invalid")
    if research.get("checkpoint_selection") != (
        "lowest_validation_mean_loss_before_behavioral_final_holdout"
    ):
        raise ValueError("capability checkpoint-selection policy is invalid")
    behavioral_evaluation = research.get("behavioral_evaluation")
    if not isinstance(behavioral_evaluation, Mapping) or set(
        behavioral_evaluation
    ) != {
        "suite_id",
        "dataset_revision",
        "evaluator_revision",
        "validation_split",
        "final_holdout_split",
    }:
        raise ValueError("capability behavioral-evaluation contract is invalid")
    for field in ("suite_id", "validation_split", "final_holdout_split"):
        _required_text(
            behavioral_evaluation.get(field),
            f"capability behavioral evaluation {field}",
        )
    for field in ("dataset_revision", "evaluator_revision"):
        _required_revision(
            behavioral_evaluation.get(field),
            f"capability behavioral evaluation {field}",
        )
    if behavioral_evaluation.get("validation_split") == (
        behavioral_evaluation.get("final_holdout_split")
    ):
        raise ValueError(
            "behavioral validation and final-holdout splits must differ"
        )
    if campaign.evaluation is None:
        raise ValueError("capability research requires evaluation")
    validation = evaluation_slice(campaign, "validation")
    final_holdout = evaluation_slice(campaign, "final_holdout")
    if max(validation.start_sequence, final_holdout.start_sequence) < min(
        validation.start_sequence + validation.sequence_count,
        final_holdout.start_sequence + final_holdout.sequence_count,
    ):
        raise ValueError(
            "campaign validation and final holdout ranges must be disjoint"
        )

    if campaign.publication is None:
        raise ValueError("capability research requires publication settings")


def _validate_campaign_contract(campaign: CampaignConfig) -> None:
    _validate_publication_contract(campaign.publication)
    research = campaign.research
    if research is None:
        return
    research_format = research.get("format")
    if research_format == "orcacolony_capability_research_v1":
        _validate_legacy_capability_contract(campaign)
    elif research_format == "orcacolony_campaign_research_v2":
        validate_campaign_research_contract(research)
    else:
        raise ValueError("unsupported campaign research contract")


def campaign_from_mapping(payload: Mapping[str, object]) -> CampaignConfig:
    allowed_top_level = {
        "campaign",
        "model",
        "training",
        "dataset",
        "evaluation",
        "research",
        "publication",
    }
    unknown = sorted(set(payload) - allowed_top_level)
    if unknown:
        raise ValueError(
            "campaign configuration contains unknown fields: " + ", ".join(unknown)
        )
    for required in ("campaign", "model", "training"):
        if required not in payload:
            raise ValueError(f"campaign configuration is missing {required}")
    campaign_payload = payload["campaign"]
    model_payload = payload["model"]
    training_payload = payload["training"]
    if not isinstance(campaign_payload, Mapping):
        raise ValueError("campaign metadata must be a JSON object")
    if not isinstance(model_payload, Mapping):
        raise ValueError("campaign model must be a JSON object")
    if not isinstance(training_payload, Mapping):
        raise ValueError("campaign training must be a JSON object")
    dataset_payload = payload.get("dataset")
    evaluation_payload = payload.get("evaluation")
    research_payload = payload.get("research")
    publication_payload = payload.get("publication")
    for name, value in (
        ("dataset", dataset_payload),
        ("evaluation", evaluation_payload),
        ("research", research_payload),
        ("publication", publication_payload),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"campaign {name} must be a JSON object")
    objective = _objective_from_mapping(campaign_payload)
    config = CampaignConfig(
        campaign=dict(campaign_payload),
        objective=objective,
        model=ModelConfig(**cast(Mapping[str, Any], model_payload)),
        training=TrainingConfig(**cast(Mapping[str, Any], training_payload)),
        dataset=cast(Mapping[str, object] | None, dataset_payload),
        evaluation=cast(Mapping[str, object] | None, evaluation_payload),
        research=cast(Mapping[str, object] | None, research_payload),
        publication=cast(Mapping[str, object] | None, publication_payload),
    )
    _validate_campaign_contract(config)
    return config


def campaign_to_mapping(campaign: CampaignConfig) -> dict[str, object]:
    """Return the canonical JSON-shaped representation of a campaign."""

    payload: dict[str, object] = {
        "campaign": dict(campaign.campaign),
        "model": asdict(campaign.model),
        "training": asdict(campaign.training),
    }
    for name, value in (
        ("dataset", campaign.dataset),
        ("evaluation", campaign.evaluation),
        ("research", campaign.research),
        ("publication", campaign.publication),
    ):
        if value is not None:
            payload[name] = dict(value)
    return payload


def load_campaign(path: str | Path) -> CampaignConfig:
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("campaign configuration must be a JSON object")
    return campaign_from_mapping(payload)


def configure_determinism(seed: int) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def build_model(campaign: CampaignConfig) -> VolunteerDecoder:
    configure_determinism(campaign.training.seed)
    model = VolunteerDecoder(campaign.model, campaign.objective)
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
        validation = evaluation_slice(campaign, "validation")
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
        if (
            validation.start_sequence + validation.sequence_count
            > dataset.validation_inputs.shape[0]
        ):
            raise ValueError(
                "campaign evaluation exceeds the packed validation dataset"
            )
        final_holdout = (
            evaluation_slice(campaign, "final_holdout")
            if campaign.evaluation.get("final_holdout") is not None
            else None
        )
        if final_holdout is not None:
            if (
                final_holdout.start_sequence + final_holdout.sequence_count
                > dataset.validation_inputs.shape[0]
            ):
                raise ValueError(
                    "campaign final holdout exceeds the packed validation dataset"
                )
            validation_range = range(
                validation.start_sequence,
                validation.start_sequence + validation.sequence_count,
            )
            final_range = range(
                final_holdout.start_sequence,
                final_holdout.start_sequence + final_holdout.sequence_count,
            )
            if max(validation_range.start, final_range.start) < min(
                validation_range.stop,
                final_range.stop,
            ):
                raise ValueError(
                    "campaign validation and final holdout ranges must be disjoint"
                )
        if (
            campaign.research is not None
            and campaign.research.get("format")
            == "orcacolony_capability_research_v1"
            and final_holdout is None
        ):
            raise ValueError(
                "capability research requires a disjoint final holdout"
            )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def evaluation_slice(
    campaign: CampaignConfig,
    name: str,
) -> EvaluationSlice:
    if campaign.evaluation is None:
        raise ValueError("campaign does not define an evaluation profile")
    if name == "validation":
        start = _nonnegative_int(
            campaign.evaluation.get("validation_start_sequence", 0),
            "validation start sequence",
        )
        sequence_count = _positive_int(
            campaign.evaluation.get("validation_sequences"),
            "validation sequence count",
        )
        batch_size = _positive_int(
            campaign.evaluation.get("batch_size"),
            "validation batch size",
        )
    elif name == "final_holdout":
        raw = campaign.evaluation.get("final_holdout")
        if not isinstance(raw, Mapping):
            raise ValueError("campaign does not define a final holdout")
        unknown = sorted(
            set(raw) - {"start_sequence", "sequence_count", "batch_size"}
        )
        if unknown:
            raise ValueError(
                "final holdout contains unknown fields: " + ", ".join(unknown)
            )
        if set(raw) != {"start_sequence", "sequence_count", "batch_size"}:
            raise ValueError("final holdout is incomplete")
        start = _nonnegative_int(
            raw.get("start_sequence"),
            "final holdout start sequence",
        )
        sequence_count = _positive_int(
            raw.get("sequence_count"),
            "final holdout sequence count",
        )
        batch_size = _positive_int(
            raw.get("batch_size"),
            "final holdout batch size",
        )
    else:
        raise ValueError(f"unsupported evaluation slice: {name}")
    if batch_size > sequence_count:
        raise ValueError(f"{name} batch size exceeds its sequence count")
    return EvaluationSlice(
        name=name,
        start_sequence=start,
        sequence_count=sequence_count,
        batch_size=batch_size,
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
    loss_sum, loss_weight_sum = objective_loss_sum(
        campaign.objective,
        logits,
        targets,
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
        loss_weight_sum=loss_weight_sum,
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


def _validate_checkpoint_trajectory(
    campaign: CampaignConfig,
    *,
    step: object,
    optimizer_step: object,
    dataset_cursor: object,
    loss_history: object,
) -> tuple[int, int, list[float]]:
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("checkpoint optimizer step must be a nonnegative integer")
    if optimizer_step != step:
        raise ValueError("checkpoint optimizer step must equal the training step")
    expected_cursor = (
        step * campaign.training.batch_size
    ) % campaign.training.dataset_sequences
    if type(dataset_cursor) is not int or dataset_cursor < 0:
        raise ValueError("checkpoint dataset cursor must be a nonnegative integer")
    if dataset_cursor != expected_cursor:
        raise ValueError("checkpoint dataset cursor differs from its trajectory")
    if (
        not isinstance(loss_history, list)
        or any(type(loss) is not float or not math.isfinite(loss) for loss in loss_history)
    ):
        raise ValueError("checkpoint loss history must contain finite JSON floats")
    if len(loss_history) != step:
        raise ValueError("checkpoint loss history differs from its trajectory")
    return step, dataset_cursor, list(loss_history)


def _optimizer_checkpoint_tensors(
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
    step: int,
) -> dict[str, Tensor]:
    named_parameters = list(model.named_parameters())
    expected_parameters = [parameter for _, parameter in named_parameters]
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if (
        len(optimizer_parameters) != len(expected_parameters)
        or len({id(parameter) for parameter in optimizer_parameters})
        != len(optimizer_parameters)
        or {id(parameter) for parameter in optimizer_parameters}
        != {id(parameter) for parameter in expected_parameters}
    ):
        raise ValueError("optimizer parameter ownership is invalid")

    tensors: dict[str, Tensor] = {}
    for name, parameter in named_parameters:
        parameter_state = optimizer.state.get(parameter)
        if step == 0 and not parameter_state:
            exp_avg = torch.zeros_like(parameter)
            exp_avg_sq = torch.zeros_like(parameter)
        else:
            if not isinstance(parameter_state, dict) or set(parameter_state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError("optimizer parameter state schema is invalid")
            raw_step = parameter_state["step"]
            if not isinstance(raw_step, Tensor) or raw_step.numel() != 1:
                raise ValueError("optimizer parameter step is invalid")
            step_value = float(raw_step.detach().cpu().item())
            if not math.isfinite(step_value) or not step_value.is_integer() or int(
                step_value
            ) != step:
                raise ValueError("optimizer parameter steps are inconsistent")
            exp_avg = parameter_state["exp_avg"]
            exp_avg_sq = parameter_state["exp_avg_sq"]
        for prefix, tensor in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != parameter.dtype
                or tensor.shape != parameter.shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError("optimizer checkpoint tensor is invalid")
            tensors[f"{prefix}.{name}"] = tensor.detach().cpu().clone().contiguous()
        if step == 0 and (
            bool(torch.count_nonzero(exp_avg))
            or bool(torch.count_nonzero(exp_avg_sq))
        ):
            raise ValueError("step-zero optimizer moments must be zero")
    return tensors


def _save_checkpoint(
    campaign: CampaignConfig,
    model: VolunteerDecoder,
    optimizer: torch.optim.AdamW,
    output_dir: Path,
    step: int,
    dataset_cursor: int,
    loss_history: list[float],
) -> TrainingResult:
    step, dataset_cursor, loss_history = _validate_checkpoint_trajectory(
        campaign,
        step=step,
        optimizer_step=step,
        dataset_cursor=dataset_cursor,
        loss_history=loss_history,
    )
    optimizer_tensors = _optimizer_checkpoint_tensors(model, optimizer, step)
    model_tensors = {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in model_tensors.values()
    ):
        raise ValueError("model checkpoint tensor is invalid")
    model_bytes = save_safetensors(model_tensors)
    optimizer_bytes = save_safetensors(optimizer_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"
    optimizer_path = output_dir / "optimizer.safetensors"
    state_path = output_dir / "state.json"
    model_tmp = output_dir / "model.safetensors.tmp"
    optimizer_tmp = output_dir / "optimizer.safetensors.tmp"
    state_tmp = output_dir / "state.json.tmp"

    model_tmp.write_bytes(model_bytes)
    os.replace(model_tmp, model_path)

    optimizer_tmp.write_bytes(optimizer_bytes)
    os.replace(optimizer_tmp, optimizer_path)

    state = {
        "format": "orcacolony_checkpoint_v1",
        "campaign_id": campaign.campaign["id"],
        "architecture": campaign.model.architecture,
        "architecture_revision": campaign.model.architecture_revision,
        "step": step,
        "optimizer_step": step,
        "dataset_cursor": dataset_cursor,
        "dataset_revision": (
            campaign.dataset["manifest_sha256"]
            if campaign.dataset is not None
            else "synthetic-fixture-v1"
        ),
        "loss_history": list(loss_history),
        "model": {
            "file": model_path.name,
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        },
        "optimizer": {
            "file": optimizer_path.name,
            "sha256": hashlib.sha256(optimizer_bytes).hexdigest(),
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


def _artifact_identity(metadata: os.stat_result) -> tuple[int, int]:
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    if identity[1] <= 0:
        raise ValueError("checkpoint artifact filesystem identity is unavailable")
    return identity


def _artifact_observation(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _owned_checkpoint_artifact_bytes(
    root: Path,
    file_name: str,
    label: str,
    *,
    max_bytes: int,
) -> bytes:
    try:
        root_before = os.lstat(root)
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or stat.S_ISLNK(root_before.st_mode)
            or int(getattr(root_before, "st_file_attributes", 0)) & reparse_flag
        ):
            raise ValueError(f"{label} root is unsafe")
        resolved_root = root.resolve(strict=True)
        path = root / file_name
        if path.parent != root:
            raise ValueError(f"{label} escapes its root")
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or int(getattr(before, "st_file_attributes", 0)) & reparse_flag
            or before.st_size > max_bytes
        ):
            raise ValueError(f"{label} is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                _artifact_identity(opened) != _artifact_identity(before)
                or _artifact_observation(opened) != _artifact_observation(before)
            ):
                raise ValueError(f"{label} changed while being opened")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(max_bytes + 1)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.lstat(path)
        root_after = os.lstat(root)
        if len(payload) > max_bytes:
            raise ValueError(f"{label} exceeds its size limit")
        if (
            _artifact_identity(after_fd) != _artifact_identity(before)
            or _artifact_observation(after_fd) != _artifact_observation(before)
            or _artifact_identity(after_path) != _artifact_identity(before)
            or _artifact_observation(after_path) != _artifact_observation(before)
            or _artifact_identity(root_after) != _artifact_identity(root_before)
            or _artifact_observation(root_after) != _artifact_observation(root_before)
            or path.resolve(strict=True).parent != resolved_root
            or root.resolve(strict=True) != resolved_root
        ):
            raise ValueError(f"{label} changed while being acquired")
        return payload
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ValueError(f"{label} could not be acquired safely") from exc


def _load_checkpoint(
    campaign: CampaignConfig,
    checkpoint_dir: Path,
) -> tuple[VolunteerDecoder, torch.optim.AdamW, int, int, list[float]]:
    model = build_model(campaign)
    expected_model_tensors = model.state_dict()
    model_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in expected_model_tensors.values()
    )
    state = json.loads(
        _owned_checkpoint_artifact_bytes(
            checkpoint_dir,
            "state.json",
            "checkpoint state",
            max_bytes=1024 * 1024,
        ),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(state, dict) or frozenset(state) != _CHECKPOINT_STATE_FIELDS:
        raise ValueError("checkpoint state schema is invalid")
    if state["format"] != "orcacolony_checkpoint_v1":
        raise ValueError("unsupported checkpoint format")
    if state["campaign_id"] != campaign.campaign["id"]:
        raise ValueError("checkpoint campaign does not match configuration")
    if (
        state.get("architecture") != campaign.model.architecture
        or type(state["architecture_revision"]) is not int
        or state["architecture_revision"] != campaign.model.architecture_revision
    ):
        raise ValueError("checkpoint architecture does not match configuration")
    expected_dataset_revision = (
        campaign.dataset["manifest_sha256"]
        if campaign.dataset is not None
        else "synthetic-fixture-v1"
    )
    if state.get("dataset_revision", "synthetic-fixture-v1") != expected_dataset_revision:
        raise ValueError("checkpoint dataset revision does not match configuration")
    step, dataset_cursor, loss_history = _validate_checkpoint_trajectory(
        campaign,
        step=state.get("step"),
        optimizer_step=state.get("optimizer_step"),
        dataset_cursor=state.get("dataset_cursor"),
        loss_history=state.get("loss_history"),
    )

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
    model_bytes = _owned_checkpoint_artifact_bytes(
        checkpoint_dir,
        "model.safetensors",
        "model checkpoint",
        max_bytes=model_tensor_bytes + 1024 * 1024,
    )
    optimizer_bytes = _owned_checkpoint_artifact_bytes(
        checkpoint_dir,
        "optimizer.safetensors",
        "optimizer checkpoint",
        max_bytes=(2 * model_tensor_bytes) + 1024 * 1024,
    )
    if hashlib.sha256(model_bytes).hexdigest() != model_artifact["sha256"]:
        raise ValueError("model checkpoint digest mismatch")
    if hashlib.sha256(optimizer_bytes).hexdigest() != optimizer_artifact["sha256"]:
        raise ValueError("optimizer checkpoint digest mismatch")

    model_tensors = load_safetensors(model_bytes)
    if set(model_tensors) != set(expected_model_tensors):
        raise ValueError("model checkpoint tensor schema is invalid")
    for name, expected in expected_model_tensors.items():
        tensor = model_tensors[name]
        if (
            tensor.dtype != expected.dtype
            or tensor.shape != expected.shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("model checkpoint tensor is invalid")
    model.load_state_dict(
        {name: tensor.clone() for name, tensor in model_tensors.items()},
        strict=True,
    )
    optimizer = _create_optimizer(model, campaign.training)
    optimizer_tensors = load_safetensors(optimizer_bytes)
    expected_optimizer_tensors = {
        f"{prefix}.{name}"
        for name, _ in model.named_parameters()
        for prefix in ("exp_avg", "exp_avg_sq")
    }
    if set(optimizer_tensors) != expected_optimizer_tensors:
        raise ValueError("optimizer checkpoint tensor schema is invalid")
    optimizer_step = float(step)
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
        step,
        dataset_cursor,
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
    validation = evaluation_slice(campaign, "validation")
    sequence_count = validation.sequence_count
    batch_size = validation.batch_size
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
                start_sequence=validation.start_sequence,
            )
            logits = model(inputs)
            batch_loss, batch_weight_sum = objective_loss_sum(
                campaign.objective,
                logits,
                targets,
            )
            loss_sum += float(batch_loss)
            loss_weight_sum += batch_weight_sum
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


def evaluate_final_holdout(
    campaign: CampaignConfig,
    checkpoint_dir: str | Path,
    dataset: PackedDataset,
) -> dict[str, object]:
    """Evaluate the selected dense checkpoint on the reserved promotion slice.

    This function is intentionally separate from repeated checkpoint evaluation.
    The release builder calls it only after `_select_checkpoint` has fixed the
    checkpoint using validation evidence.
    """

    validate_dataset_artifacts(campaign, dataset)
    final_holdout = evaluation_slice(campaign, "final_holdout")
    checkpoint = Path(checkpoint_dir)
    model, _, step, _, _ = _load_checkpoint(campaign, checkpoint)
    loss_sum = 0.0
    loss_weight_sum = 0
    model.eval()
    with torch.no_grad():
        cursor = 0
        while cursor < final_holdout.sequence_count:
            current_batch_size = min(
                final_holdout.batch_size,
                final_holdout.sequence_count - cursor,
            )
            inputs, targets = dataset.validation_batch(
                cursor=cursor,
                batch_size=current_batch_size,
                sequence_limit=final_holdout.sequence_count,
                start_sequence=final_holdout.start_sequence,
            )
            batch_loss, batch_weight_sum = objective_loss_sum(
                campaign.objective,
                model(inputs),
                targets,
            )
            loss_sum += float(batch_loss)
            loss_weight_sum += batch_weight_sum
            cursor += current_batch_size
    mean_loss = loss_sum / loss_weight_sum
    checkpoint_state = json.loads(
        (checkpoint / "state.json").read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return {
        "format": "orcacolony_final_holdout_evaluation_v1",
        "campaign_id": campaign.campaign["id"],
        "objective": campaign.objective.name,
        "loss_mask": campaign.objective.loss_mask,
        "step": step,
        "dataset_revision": dataset.revision,
        "checkpoint_sha256": checkpoint_state["model"]["sha256"],
        "start_sequence": final_holdout.start_sequence,
        "sequence_count": final_holdout.sequence_count,
        "loss_sum": loss_sum,
        "loss_weight_sum": loss_weight_sum,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "selection_locked_before_evaluation": True,
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
        loss_sum, loss_weight_sum = objective_loss_sum(
            campaign.objective,
            logits,
            targets,
        )
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
    loss_sum, _ = objective_loss_sum(
        campaign.objective,
        logits,
        targets,
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
