from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .reference import (
    CampaignConfig,
    TrainingConfig,
    VolunteerDecoder,
    build_model,
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
        if self.dropout != 0.0:
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
