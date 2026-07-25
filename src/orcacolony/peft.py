from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor, nn
from torch.nn import functional as F

from .reference import (
    CampaignConfig,
    TrainingConfig,
    VolunteerDecoder,
    build_model,
    fixture_batch,
    load_campaign,
    tensor_sha256,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_PATTERN = re.compile(
    r"(?:[a-z_][a-z0-9_]*|[0-9]+)(?:\.(?:[a-z_][a-z0-9_]*|[0-9]+))*\Z"
)


@dataclass(frozen=True)
class LoRAConfig:
    format: str
    base_model_sha256: str
    rank: int
    alpha: float
    dropout: float
    adapter_seed: int
    initialization_std: float
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.format != "orcacolony_lora_v1":
            raise ValueError("unsupported LoRA configuration format")
        if _SHA256_PATTERN.fullmatch(self.base_model_sha256) is None:
            raise ValueError("base_model_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(float(self.alpha))
            or float(self.alpha) <= 0
        ):
            raise ValueError("LoRA alpha must be a positive finite number")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
        ):
            raise ValueError("LoRA dropout must be a finite number")
        if float(self.dropout) != 0.0:
            raise ValueError("orcacolony_lora_v1 requires zero adapter dropout")
        if (
            isinstance(self.adapter_seed, bool)
            or not isinstance(self.adapter_seed, int)
            or self.adapter_seed < 0
        ):
            raise ValueError("adapter_seed must be a non-negative integer")
        if (
            isinstance(self.initialization_std, bool)
            or not isinstance(self.initialization_std, (int, float))
            or not math.isfinite(float(self.initialization_std))
            or float(self.initialization_std) <= 0
        ):
            raise ValueError("initialization_std must be a positive finite number")
        if not self.targets:
            raise ValueError("LoRA targets must not be empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("LoRA targets must be unique")
        for target in self.targets:
            if not isinstance(target, str) or _TARGET_PATTERN.fullmatch(target) is None:
                raise ValueError(f"invalid LoRA target: {target!r}")


@dataclass(frozen=True)
class AdapterGradientResult:
    loss_sum: float
    loss_weight_sum: int
    gradients: Mapping[str, Tensor]
    gradient_sha256: str


@dataclass(frozen=True)
class LoadedLoRAManifest:
    campaign: CampaignConfig
    config: LoRAConfig
    campaign_path: Path
    manifest_path: Path
    campaign_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class LoRAFixtureExportResult:
    output_dir: Path
    gradient_sha256: str
    updated_adapter_sha256: str


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        config: LoRAConfig,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.base = base
        self.scaling = float(config.alpha) / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(
                (config.rank, base.in_features),
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                (base.out_features, config.rank),
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )
        with torch.no_grad():
            self.lora_a.copy_(
                torch.randn(
                    self.lora_a.shape,
                    generator=generator,
                    dtype=self.lora_a.dtype,
                    device=self.lora_a.device,
                )
                * float(config.initialization_std)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        adapter_hidden = F.linear(inputs, self.lora_a)
        adapter_output = F.linear(adapter_hidden, self.lora_b)
        return self.base(inputs) + adapter_output * self.scaling


def _resolve_child(module: nn.Module, component: str) -> nn.Module:
    if component.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
        index = int(component)
        if index >= len(module):
            raise ValueError(f"LoRA target index is out of range: {component}")
        return module[index]
    child = getattr(module, component, None)
    if not isinstance(child, nn.Module):
        raise ValueError(f"LoRA target component is not a module: {component}")
    return child


def _resolve_parent(model: nn.Module, target: str) -> tuple[nn.Module, str]:
    components = target.split(".")
    parent = model
    for component in components[:-1]:
        parent = _resolve_child(parent, component)
    return parent, components[-1]


def build_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
) -> VolunteerDecoder:
    model = build_model(campaign)
    actual_base_sha256 = tensor_sha256(model.state_dict())
    if actual_base_sha256 != config.base_model_sha256:
        raise ValueError(
            "LoRA base model digest mismatch: "
            f"expected={config.base_model_sha256}, actual={actual_base_sha256}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.adapter_seed)
    for target in config.targets:
        parent, child_name = _resolve_parent(model, target)
        child = getattr(parent, child_name, None)
        if not isinstance(child, nn.Linear):
            raise ValueError(f"LoRA target is not a linear layer: {target}")
        setattr(parent, child_name, LoRALinear(child, config, generator))

    adapters = adapter_named_parameters(model)
    if len(adapters) != 2 * len(config.targets):
        raise ValueError("LoRA trainable tensor set does not match the target manifest")
    return model


def adapter_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    adapters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }
    adapters = dict(sorted(adapters.items()))
    if not adapters:
        raise ValueError("model does not contain LoRA adapter parameters")
    if any(not parameter.requires_grad for parameter in adapters.values()):
        raise ValueError("all LoRA adapter parameters must be trainable")
    unexpected_trainable = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in adapters
    )
    if unexpected_trainable:
        raise ValueError(
            f"non-adapter parameters remain trainable: {unexpected_trainable}"
        )
    return adapters


def compute_adapter_gradients(
    model: VolunteerDecoder,
    inputs: Tensor,
    targets: Tensor,
) -> AdapterGradientResult:
    adapters = adapter_named_parameters(model)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="sum",
    )
    loss_sum.backward()

    gradients: dict[str, Tensor] = {}
    for name, parameter in adapters.items():
        if parameter.grad is None:
            raise ValueError(f"adapter gradient is missing: {name}")
        gradient = parameter.grad.detach().to(torch.float32).cpu().contiguous()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"adapter gradient contains non-finite values: {name}")
        gradients[name] = gradient
    unexpected_base_gradients = sorted(
        name
        for name, parameter in model.named_parameters()
        if name not in adapters and parameter.grad is not None
    )
    if unexpected_base_gradients:
        raise ValueError(
            f"frozen base parameters received gradients: {unexpected_base_gradients}"
        )
    return AdapterGradientResult(
        loss_sum=float(loss_sum.detach()),
        loss_weight_sum=targets.numel(),
        gradients=gradients,
        gradient_sha256=tensor_sha256(gradients),
    )


def create_adapter_optimizer(
    model: nn.Module,
    training: TrainingConfig,
) -> torch.optim.AdamW:
    adapters = adapter_named_parameters(model)
    return torch.optim.AdamW(
        list(adapters.values()),
        lr=training.learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon,
        weight_decay=training.weight_decay,
    )


def apply_adapter_gradient_step(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: Mapping[str, Tensor],
    loss_weight_sum: int,
    max_gradient_norm: float,
) -> float:
    if (
        isinstance(loss_weight_sum, bool)
        or not isinstance(loss_weight_sum, int)
        or loss_weight_sum <= 0
    ):
        raise ValueError("loss_weight_sum must be a positive integer")
    if (
        isinstance(max_gradient_norm, bool)
        or not isinstance(max_gradient_norm, (int, float))
        or not math.isfinite(float(max_gradient_norm))
        or float(max_gradient_norm) <= 0
    ):
        raise ValueError("max_gradient_norm must be a positive finite number")

    adapters = adapter_named_parameters(model)
    if sorted(gradients) != list(adapters):
        missing = sorted(set(adapters) - set(gradients))
        unexpected = sorted(set(gradients) - set(adapters))
        raise ValueError(
            "adapter gradient names differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != {id(parameter) for parameter in adapters.values()}:
        raise ValueError("optimizer parameter set does not exactly match the adapters")

    optimizer.zero_grad(set_to_none=True)
    for name, parameter in adapters.items():
        gradient = gradients[name]
        if gradient.dtype != torch.float32:
            raise ValueError(f"adapter gradient must be float32: {name}")
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"adapter gradient shape differs for {name}: "
                f"expected={tuple(parameter.shape)}, actual={tuple(gradient.shape)}"
            )
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"adapter gradient contains non-finite values: {name}")
        parameter.grad = gradient.to(parameter.device, parameter.dtype).clone()
        parameter.grad.div_(loss_weight_sum)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        list(adapters.values()),
        float(max_gradient_norm),
    )
    optimizer.step()
    return float(gradient_norm)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - expected)
    missing = sorted(expected - set(payload))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_lora_manifest(
    campaign_path: str | Path,
    manifest_path: str | Path,
) -> LoadedLoRAManifest:
    campaign_path = Path(campaign_path)
    manifest_path = Path(manifest_path)
    campaign_payload = json.loads(
        campaign_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    manifest_payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    campaign_mapping = _require_mapping(campaign_payload, "campaign configuration")
    manifest = _require_mapping(manifest_payload, "LoRA manifest")
    _require_exact_fields(manifest, {"format", "base", "adapter"}, "LoRA manifest")
    if manifest["format"] != "orcacolony_lora_manifest_v1":
        raise ValueError("unsupported LoRA manifest format")

    base = _require_mapping(manifest["base"], "LoRA manifest base")
    _require_exact_fields(
        base,
        {"campaign_file", "campaign_sha256", "model_sha256"},
        "LoRA manifest base",
    )
    campaign_file = base["campaign_file"]
    if (
        not isinstance(campaign_file, str)
        or not campaign_file.endswith(".json")
        or "/" in campaign_file
        or "\\" in campaign_file
        or campaign_file in {".", ".."}
    ):
        raise ValueError("LoRA base campaign_file must be a safe JSON filename")
    expected_campaign_path = (manifest_path.parent / campaign_file).resolve()
    if expected_campaign_path != campaign_path.resolve():
        raise ValueError("LoRA manifest does not reference the supplied campaign path")
    for field in ("campaign_sha256", "model_sha256"):
        value = base[field]
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"LoRA base {field} must be a lowercase SHA-256 digest")
    campaign_sha256 = _sha256_bytes(_canonical_json(campaign_mapping))
    if campaign_sha256 != base["campaign_sha256"]:
        raise ValueError("LoRA base campaign digest mismatch")

    adapter = _require_mapping(manifest["adapter"], "LoRA manifest adapter")
    _require_exact_fields(
        adapter,
        {
            "rank",
            "alpha",
            "dropout",
            "seed",
            "initialization_std",
            "targets",
        },
        "LoRA manifest adapter",
    )
    targets = adapter["targets"]
    if not isinstance(targets, list):
        raise ValueError("LoRA manifest adapter targets must be a JSON array")
    config = LoRAConfig(
        format="orcacolony_lora_v1",
        base_model_sha256=base["model_sha256"],  # type: ignore[arg-type]
        rank=adapter["rank"],  # type: ignore[arg-type]
        alpha=adapter["alpha"],  # type: ignore[arg-type]
        dropout=adapter["dropout"],  # type: ignore[arg-type]
        adapter_seed=adapter["seed"],  # type: ignore[arg-type]
        initialization_std=adapter["initialization_std"],  # type: ignore[arg-type]
        targets=tuple(targets),  # type: ignore[arg-type]
    )
    campaign = load_campaign(campaign_path)
    build_lora_model(campaign, config)
    return LoadedLoRAManifest(
        campaign=campaign,
        config=config,
        campaign_path=campaign_path.resolve(),
        manifest_path=manifest_path.resolve(),
        campaign_sha256=campaign_sha256,
        manifest_sha256=_sha256_bytes(_canonical_json(manifest)),
    )


def _cpu_tensor_mapping(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: tensors[name].detach().cpu().contiguous()
        for name in sorted(tensors)
    }


def _adapter_state(model: nn.Module) -> dict[str, Tensor]:
    return _cpu_tensor_mapping(adapter_named_parameters(model))


def _base_parameter_state(model: nn.Module) -> dict[str, Tensor]:
    adapter_names = set(adapter_named_parameters(model))
    return {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in sorted(model.named_parameters())
        if name not in adapter_names
    }


def export_lora_fixture(
    loaded: LoadedLoRAManifest,
    output_dir: str | Path,
) -> LoRAFixtureExportResult:
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"LoRA fixture output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        campaign = loaded.campaign
        config = loaded.config
        dense_base = build_model(campaign)
        worker = build_lora_model(campaign, config)
        inputs, targets = fixture_batch(campaign)
        initial_adapter = _adapter_state(worker)
        submitted = compute_adapter_gradients(worker, inputs, targets)

        coordinator = build_lora_model(campaign, config)
        coordinator_base_before = _base_parameter_state(coordinator)
        coordinator_optimizer = create_adapter_optimizer(coordinator, campaign.training)
        gradient_norm = apply_adapter_gradient_step(
            coordinator,
            coordinator_optimizer,
            submitted.gradients,
            submitted.loss_weight_sum,
            campaign.training.max_gradient_norm,
        )
        updated_adapter = _adapter_state(coordinator)
        if any(
            not torch.equal(parameter, coordinator_base_before[name])
            for name, parameter in _base_parameter_state(coordinator).items()
        ):
            raise ValueError("coordinator adapter step changed a frozen base parameter")

        reference = build_lora_model(campaign, config)
        reference_optimizer = create_adapter_optimizer(reference, campaign.training)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss_sum = F.cross_entropy(
            reference(inputs).reshape(-1, campaign.model.vocabulary_size),
            targets.reshape(-1),
            reduction="sum",
        )
        (reference_loss_sum / targets.numel()).backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter_named_parameters(reference).values()),
            campaign.training.max_gradient_norm,
        )
        reference_optimizer.step()
        reference_adapter = _adapter_state(reference)
        if any(
            not torch.equal(updated_adapter[name], reference_adapter[name])
            for name in updated_adapter
        ):
            raise ValueError("submitted adapter gradient step differs from the reference")

        artifact_tensors = {
            "base.safetensors": _cpu_tensor_mapping(dense_base.state_dict()),
            "adapter.safetensors": initial_adapter,
            "batch.safetensors": {
                "input_ids": inputs.detach().cpu().contiguous(),
                "target_ids": targets.detach().cpu().contiguous(),
            },
            "gradients.safetensors": _cpu_tensor_mapping(submitted.gradients),
            "updated-adapter.safetensors": updated_adapter,
        }
        for filename, tensors in artifact_tensors.items():
            save_safetensors_file(tensors, str(temporary / filename))
        file_hashes = {
            filename: _sha256_file(temporary / filename)
            for filename in sorted(artifact_tensors)
        }
        adapter_names = sorted(initial_adapter)
        fixture: dict[str, object] = {
            "format": "orcacolony_lora_fixture_v1",
            "campaign_id": campaign.campaign["id"],
            "input_shape": list(inputs.shape),
            "input_ids": inputs.reshape(-1).tolist(),
            "target_ids": targets.reshape(-1).tolist(),
            "model": {
                "vocab_size": campaign.model.vocabulary_size,
                "context_length": campaign.model.context_length,
                "d_model": campaign.model.width,
                "num_heads": campaign.model.heads,
                "num_layers": campaign.model.layers,
                "d_ff": campaign.model.mlp_width,
            },
            "source": {
                "campaign_file": loaded.campaign_path.name,
                "campaign_sha256": loaded.campaign_sha256,
                "lora_manifest_file": loaded.manifest_path.name,
                "lora_manifest_sha256": loaded.manifest_sha256,
            },
            "base": {
                "architecture": campaign.model.architecture,
                "architecture_revision": campaign.model.architecture_revision,
                "model_sha256": config.base_model_sha256,
                "parameter_count": sum(
                    parameter.numel() for parameter in dense_base.parameters()
                ),
            },
            "adapter": {
                "format": config.format,
                "rank": config.rank,
                "alpha": config.alpha,
                "dropout": config.dropout,
                "seed": config.adapter_seed,
                "initialization_std": config.initialization_std,
                "targets": list(config.targets),
                "tensor_order": adapter_names,
                "tensor_count": len(adapter_names),
                "value_count": sum(tensor.numel() for tensor in initial_adapter.values()),
                "initial_sha256": tensor_sha256(initial_adapter),
                "updated_sha256": tensor_sha256(updated_adapter),
            },
            "loss_sum": submitted.loss_sum,
            "loss_weight_sum": submitted.loss_weight_sum,
            "gradient": {
                "sha256": submitted.gradient_sha256,
                "tensor_order": sorted(submitted.gradients),
            },
            "gradient_contract": {
                "loss_reduction": "sum",
                "accumulation_dtype": "float32",
                "tensor_set": "complete_adapter_manifest",
                "normalization_owner": "coordinator",
            },
            "one_step_update": {
                "optimizer": "adamw",
                "loss_normalization": "divide_by_total_loss_weight_once",
                "gradient_clipping": "global_norm_once_after_normalization",
                "max_gradient_norm": campaign.training.max_gradient_norm,
                "gradient_norm_before_clipping": gradient_norm,
                "matches_mean_loss_reference": True,
                "frozen_base_unchanged": True,
            },
            "files": file_hashes,
        }
        fixture_path = temporary / "fixture.json"
        fixture_path.write_bytes(
            (json.dumps(fixture, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        checksum_paths = sorted(path for path in temporary.iterdir() if path.is_file())
        (temporary / "SHA256SUMS").write_bytes(
            "".join(
                f"{_sha256_file(path)}  {path.name}\n" for path in checksum_paths
            ).encode("utf-8")
        )
        os.replace(temporary, output)
        return LoRAFixtureExportResult(
            output_dir=output,
            gradient_sha256=submitted.gradient_sha256,
            updated_adapter_sha256=tensor_sha256(updated_adapter),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded OrcaColony PEFT reference proofs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser(
        "export-fixture",
        help="export a deterministic frozen-base LoRA gradient and update fixture",
    )
    fixture.add_argument("--campaign", type=Path, required=True)
    fixture.add_argument("--lora", type=Path, required=True)
    fixture.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command != "export-fixture":
        raise ValueError(f"unsupported PEFT command: {args.command}")
    loaded = load_lora_manifest(args.campaign, args.lora)
    result = export_lora_fixture(loaded, args.output)
    print(
        json.dumps(
            {
                "format": "orcacolony_lora_fixture_export_v1",
                "output_dir": str(result.output_dir),
                "gradient_sha256": result.gradient_sha256,
                "updated_adapter_sha256": result.updated_adapter_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
