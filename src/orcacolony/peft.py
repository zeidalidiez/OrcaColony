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
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch import Tensor, nn
from torch.nn import functional as F

from .artifacts import PackedDataset
from .reference import (
    CampaignConfig,
    CausalSelfAttention,
    TrainingConfig,
    VolunteerDecoder,
    build_model,
    campaign_from_mapping,
    fixture_batch,
    tensor_sha256,
    validate_dataset_artifacts,
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


@dataclass(frozen=True)
class LoRATrainingResult:
    checkpoint_dir: Path
    steps_completed: int
    loss_history: tuple[float, ...]
    base_model_sha256: str
    adapter_sha256: str
    weight_checkpoint_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class BaseLayerBundleExportResult:
    output_dir: Path
    manifest_path: Path
    manifest_sha256: str
    base_model_sha256: str
    source_artifact_sha256: str
    linear_count: int
    artifact_bytes: int


@dataclass(frozen=True)
class _LayerBundleLinearDescriptor:
    module_path: str
    artifact_path: Path
    artifact_sha256: str
    tensor_sha256: str
    artifact_bytes: int
    weight_shape: tuple[int, int]
    has_bias: bool


@dataclass(frozen=True)
class _LoadedBaseLayerBundle:
    root: Path
    manifest_sha256: str
    base_model_sha256: str
    resident_path: Path
    resident_sha256: str
    resident_tensor_sha256: str
    resident_bytes: int
    resident_shapes: Mapping[str, tuple[int, ...]]
    linears: Mapping[str, _LayerBundleLinearDescriptor]


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


EXACT_CPU_FP32_PROFILE = "exact-cpu-fp32-v1"
BURN_NDARRAY_F32_PROFILE = "burn-ndarray-f32-v1"
BURN_WEBGPU_F32_PROFILE = "burn-webgpu-f32-v1"
INT8_FROZEN_LINEAR_PROFILE = "int8-per-output-symmetric-f32-dequant-v1"
NUMERICAL_PROFILES = frozenset(
    {
        EXACT_CPU_FP32_PROFILE,
        BURN_NDARRAY_F32_PROFILE,
        BURN_WEBGPU_F32_PROFILE,
        INT8_FROZEN_LINEAR_PROFILE,
    }
)


class _Int8FrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, qweight: Tensor, scales: Tensor, bias: Tensor | None) -> Tensor:
        ctx.save_for_backward(qweight, scales)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            weight = qweight.to(dtype=torch.float32) * scales[:, None].float()
            return F.linear(inputs.float(), weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        qweight, scales = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            weight = qweight.to(dtype=torch.float32) * scales[:, None].float()
            return grad_output.float().matmul(weight), None, None, None


class Int8FrozenLinear(nn.Module):
    """Frozen per-output symmetric int8 weight with FP32 scales and bias."""

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("int8 frozen linear source must not be trainable")
        weight = source.weight.detach().to(dtype=torch.float32)
        scales = (
            weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            / 127.0
        )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer(
            "qweight",
            torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scales", scales)
        self.register_buffer(
            "bias",
            None
            if source.bias is None
            else source.bias.detach().to(dtype=torch.float32).clone(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != torch.float32:
            raise ValueError("int8 frozen-linear profile requires FP32 activations")
        return _Int8FrozenLinearFunction.apply(
            inputs, self.qweight, self.scales, self.bias
        )


def _quantize_frozen_linears(module: nn.Module) -> int:
    count = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            if not isinstance(child.base, nn.Linear):
                raise ValueError("LoRA base is not an FP32 linear layer")
            child.base = Int8FrozenLinear(child.base)
            count += 1
        elif isinstance(child, nn.Linear):
            setattr(module, child_name, Int8FrozenLinear(child))
            count += 1
        else:
            count += _quantize_frozen_linears(child)
    return count


STREAMED_FP32_FROZEN_LINEAR_PROFILE = "streamed-fp32-frozen-linear-v1"


class _StreamedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, module: "StreamedFrozenLinear") -> Tensor:
        ctx.module = module
        weight, bias = module.load_tensors()
        with torch.autocast(device_type="cpu", enabled=False):
            return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        weight, _ = ctx.module.load_tensors()
        with torch.autocast(device_type="cpu", enabled=False):
            return grad_output.float().matmul(weight), None


class StreamedFrozenLinear(nn.Module):
    """Path-only exact-FP32 frozen linear with authenticated reloads."""

    def __init__(self, source: nn.Linear, artifact_path: Path) -> None:
        super().__init__()
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("streamed frozen linear source must not be trainable")
        tensors = {"weight": source.weight.detach().cpu().contiguous()}
        if source.bias is not None:
            tensors["bias"] = source.bias.detach().cpu().contiguous()
        save_safetensors_file(tensors, artifact_path)
        self.artifact_path = artifact_path
        self.expected_tensor_sha256 = tensor_sha256(tensors)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.has_bias = source.bias is not None
        self.artifact_bytes = artifact_path.stat().st_size
        self.read_bytes = 0
        self.read_count = 0

    def load_tensors(self) -> tuple[Tensor, Tensor | None]:
        mapped = load_safetensors_file(self.artifact_path, device="cpu")
        tensors = {
            name: tensor.clone().contiguous() for name, tensor in mapped.items()
        }
        del mapped
        expected_names = {"weight", "bias"} if self.has_bias else {"weight"}
        if set(tensors) != expected_names:
            raise ValueError("streamed linear tensor names differ")
        weight = tensors["weight"]
        bias = tensors.get("bias")
        if weight.dtype != torch.float32 or weight.shape != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError("streamed linear weight contract differs")
        if bias is not None and (
            bias.dtype != torch.float32 or bias.shape != (self.out_features,)
        ):
            raise ValueError("streamed linear bias contract differs")
        if tensor_sha256(tensors) != self.expected_tensor_sha256:
            raise ValueError("streamed linear tensor digest mismatch")
        if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError("streamed linear tensors must be finite")
        self.read_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        self.read_count += 1
        return weight, bias

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != torch.float32 or inputs.device.type != "cpu":
            raise ValueError("streamed frozen-linear profile requires CPU FP32 activations")
        return _StreamedFrozenLinearFunction.apply(inputs, self)


def _stream_frozen_linears(
    module: nn.Module,
    storage_dir: Path,
    counter: list[int],
) -> int:
    count = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            if not isinstance(child.base, nn.Linear):
                raise ValueError("LoRA base is not an FP32 linear layer")
            artifact_path = storage_dir / f"linear-{counter[0]:05d}.safetensors"
            counter[0] += 1
            child.base = StreamedFrozenLinear(child.base, artifact_path)
            count += 1
        elif isinstance(child, nn.Linear):
            artifact_path = storage_dir / f"linear-{counter[0]:05d}.safetensors"
            counter[0] += 1
            setattr(module, child_name, StreamedFrozenLinear(child, artifact_path))
            count += 1
        else:
            count += _stream_frozen_linears(child, storage_dir, counter)
    return count


DIRECT_STREAMED_FP32_PROFILE = "direct-streamed-fp32-v1"


def _direct_tensor_snapshot(artifact_path: Path, key: str) -> Tensor:
    with safe_open(artifact_path, framework="pt", device="cpu") as reader:
        if key not in reader.keys():
            raise ValueError(f"base artifact is missing tensor: {key}")
        return reader.get_tensor(key).clone().contiguous()


def _direct_linear_snapshot(
    artifact_path: Path,
    weight_key: str,
    bias_key: str | None,
) -> dict[str, Tensor]:
    tensors = {"weight": _direct_tensor_snapshot(artifact_path, weight_key)}
    if bias_key is not None:
        tensors["bias"] = _direct_tensor_snapshot(artifact_path, bias_key)
    return tensors


class _DirectStreamedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, module: "DirectStreamedFrozenLinear") -> Tensor:
        weight, bias = module.load_tensors()
        ctx.module = module
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        weight, _ = ctx.module.load_tensors()
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            return grad_output.float().matmul(weight), None


class DirectStreamedFrozenLinear(nn.Module):
    """Path-only FP32 frozen linear backed by one authenticated base artifact."""

    def __init__(
        self,
        artifact_path: Path,
        weight_key: str,
        bias_key: str | None,
        expected_tensors: Mapping[str, Tensor],
    ) -> None:
        super().__init__()
        weight = expected_tensors["weight"]
        if weight.ndim != 2 or weight.dtype != torch.float32:
            raise ValueError("direct streamed weight must be a rank-2 FP32 tensor")
        bias = expected_tensors.get("bias")
        if bias_key is None:
            if bias is not None:
                raise ValueError("unexpected direct streamed bias")
        elif bias is None or bias.shape != (weight.shape[0],):
            raise ValueError("direct streamed bias shape differs")
        self.artifact_path = artifact_path.resolve(strict=True)
        self.weight_key = weight_key
        self.bias_key = bias_key
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])
        self.expected_tensor_sha256 = tensor_sha256(expected_tensors)
        self.read_bytes = 0
        self.read_count = 0

    def load_tensors(self) -> tuple[Tensor, Tensor | None]:
        tensors = _direct_linear_snapshot(
            self.artifact_path,
            self.weight_key,
            self.bias_key,
        )
        weight = tensors["weight"]
        bias = tensors.get("bias")
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError("direct streamed weight shape differs")
        if weight.dtype != torch.float32:
            raise ValueError("direct streamed weight must remain FP32")
        if self.bias_key is not None:
            if bias is None or bias.shape != (self.out_features,):
                raise ValueError("direct streamed bias shape differs")
            if bias.dtype != torch.float32:
                raise ValueError("direct streamed bias must remain FP32")
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()):
            raise ValueError("direct streamed tensors contain non-finite values")
        if tensor_sha256(tensors) != self.expected_tensor_sha256:
            raise ValueError("direct streamed tensor digest mismatch")
        self.read_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        self.read_count += 1
        return weight, bias

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise ValueError("direct streamed profile requires CPU FP32 activations")
        return _DirectStreamedFrozenLinearFunction.apply(inputs, self)


def _replace_with_direct_streamed_linears(
    module: nn.Module,
    artifact_path: Path,
    prefix: str = "",
) -> tuple[int, set[str]]:
    count = 0
    consumed: set[str] = set()
    for child_name, child in list(module.named_children()):
        child_path = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            source = child.base
            if not isinstance(source, nn.Linear):
                raise ValueError("direct streamed LoRA base is not linear")
            weight_key = f"{child_path}.weight"
            bias_key = f"{child_path}.bias" if source.bias is not None else None
            expected = _direct_linear_snapshot(artifact_path, weight_key, bias_key)
            child.base = DirectStreamedFrozenLinear(
                artifact_path,
                weight_key,
                bias_key,
                expected,
            )
            consumed.add(weight_key)
            if bias_key is not None:
                consumed.add(bias_key)
            count += 1
        elif isinstance(child, nn.Linear):
            if any(parameter.requires_grad for parameter in child.parameters()):
                raise ValueError("direct streaming requires frozen linear parameters")
            weight_key = f"{child_path}.weight"
            bias_key = f"{child_path}.bias" if child.bias is not None else None
            expected = _direct_linear_snapshot(artifact_path, weight_key, bias_key)
            setattr(
                module,
                child_name,
                DirectStreamedFrozenLinear(
                    artifact_path,
                    weight_key,
                    bias_key,
                    expected,
                ),
            )
            consumed.add(weight_key)
            if bias_key is not None:
                consumed.add(bias_key)
            count += 1
        else:
            nested_count, nested_consumed = _replace_with_direct_streamed_linears(
                child,
                artifact_path,
                child_path,
            )
            count += nested_count
            consumed.update(nested_consumed)
    return count, consumed


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


def quantize_lora_frozen_base(model: VolunteerDecoder) -> VolunteerDecoder:
    """Convert an already authenticated FP32 LoRA base to the int8 profile."""

    if _quantize_frozen_linears(model) <= 0:
        raise ValueError("int8 frozen-linear profile did not replace any layers")
    if any(isinstance(module, nn.Linear) for module in model.modules()):
        raise ValueError("int8 frozen-linear profile retained an FP32 linear")
    adapter_named_parameters(model)
    return model


def build_int8_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
) -> VolunteerDecoder:
    """Build the explicit offline int8-frozen-linear / FP32-adapter profile.

    This profile intentionally has a distinct numerical trajectory and is not a
    connected ``python-native-cpu-f32`` backend. The current builder first
    authenticates and constructs the complete FP32 base, then converts frozen
    linears; it reduces steady tensor storage, not peak startup residency.
    """

    return quantize_lora_frozen_base(build_lora_model(campaign, config))


def _validated_lora_numerical_profile(value: object) -> str:
    if not isinstance(value, str) or value not in NUMERICAL_PROFILES:
        raise ValueError("LoRA numerical profile is unsupported")
    return value


def build_profiled_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
    numerical_profile: str,
) -> VolunteerDecoder:
    profile = _validated_lora_numerical_profile(numerical_profile)
    if profile == INT8_FROZEN_LINEAR_PROFILE:
        return build_int8_lora_model(campaign, config)
    return build_lora_model(campaign, config)


def build_streamed_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
    storage_dir: str | Path,
) -> VolunteerDecoder:
    """Build the explicit offline exact-FP32 streamed-linear profile.

    The builder authenticates and constructs the complete FP32 model before it
    exports and removes frozen linears. It proves steady retained tensor
    reduction, not lower peak startup residency. Every forward/backward reload
    revalidates names, shapes, dtypes, finite values, and tensor identity.
    """

    model = build_lora_model(campaign, config)
    storage = Path(storage_dir)
    if storage.exists():
        raise FileExistsError(f"streamed linear storage already exists: {storage}")
    if not storage.parent.is_dir():
        raise FileNotFoundError(
            f"streamed linear storage parent does not exist: {storage.parent}"
        )
    storage.mkdir()
    try:
        if _stream_frozen_linears(model, storage, [0]) <= 0:
            raise ValueError("streamed frozen-linear profile did not replace any layers")
        adapter_named_parameters(model)
    except BaseException:
        shutil.rmtree(storage, ignore_errors=True)
        raise
    return model


def build_direct_streamed_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
    base_artifact_path: str | Path,
    base_artifact_sha256: str,
    adapter_state: Mapping[str, Tensor],
) -> VolunteerDecoder:
    """Construct an exact-FP32 streamed model directly from authenticated artifacts.

    ``base_artifact_sha256`` is the worker-facing raw-file identity supplied by
    an authenticated artifact manifest. It is intentionally distinct from the
    canonical tensor-set identity in ``config.base_model_sha256``. The complete
    artifact is hashed before and after construction; every streamed linear
    also retains and rechecks its own tensor-set identity on every load.
    """

    if _SHA256_PATTERN.fullmatch(base_artifact_sha256) is None:
        raise ValueError("base artifact SHA-256 must be a lowercase digest")
    artifact_path = Path(base_artifact_path).resolve(strict=True)
    if not artifact_path.is_file():
        raise ValueError("base artifact path must be a regular file")
    if _sha256_file(artifact_path) != base_artifact_sha256:
        raise ValueError("base artifact SHA-256 mismatch")

    with torch.device("meta"):
        model = VolunteerDecoder(campaign.model)
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

    direct_count, consumed = _replace_with_direct_streamed_linears(
        model,
        artifact_path,
    )
    if direct_count <= 0:
        raise ValueError("direct streamed profile did not replace any linears")
    model.to_empty(device="cpu")

    with safe_open(artifact_path, framework="pt", device="cpu") as reader:
        artifact_names = set(reader.keys())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_a") or name.endswith(".lora_b"):
                continue
            tensor = _direct_tensor_snapshot(artifact_path, name)
            if tensor.shape != parameter.shape:
                raise ValueError(f"base artifact tensor shape differs: {name}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"base artifact tensor is not FP32: {name}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"base artifact tensor is non-finite: {name}")
            parameter.copy_(tensor)
            consumed.add(name)

    context_length = campaign.model.context_length
    causal_mask = torch.triu(
        torch.ones(context_length, context_length, dtype=torch.bool),
        diagonal=1,
    )
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.causal_mask = causal_mask.clone()

    if consumed != artifact_names:
        missing = sorted(artifact_names - consumed)
        unexpected = sorted(consumed - artifact_names)
        raise ValueError(
            "base artifact tensor partition differs: "
            f"unconsumed={missing}, missing={unexpected}"
        )
    load_adapter_state(model, adapter_state)
    adapter_named_parameters(model)
    if _sha256_file(artifact_path) != base_artifact_sha256:
        raise ValueError("base artifact changed during direct construction")
    return model


LAYER_BUNDLE_STREAMED_FP32_PROFILE = "layer-bundle-streamed-fp32-v1"
LAYER_BUNDLE_INT8_PROFILE = (
    "layer-bundle-int8-per-output-symmetric-f32-dequant-v1"
)
_LAYER_BUNDLE_ARTIFACT_PATTERN = re.compile(
    r"(?:resident|linear-[0-9]{5})\.safetensors\Z"
)


def _layer_bundle_shape(
    value: object,
    label: str,
    *,
    rank: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    shape: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"{label} dimensions must be positive integers")
        shape.append(dimension)
    if rank is not None and len(shape) != rank:
        raise ValueError(f"{label} must have rank {rank}")
    return tuple(shape)


def _layer_bundle_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _layer_bundle_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _layer_bundle_artifact_path(
    root: Path,
    filename: object,
    expected_bytes: int,
    label: str,
) -> Path:
    if (
        not isinstance(filename, str)
        or _LAYER_BUNDLE_ARTIFACT_PATTERN.fullmatch(filename) is None
        or Path(filename).name != filename
    ):
        raise ValueError(f"{label} must be a safe bundle artifact filename")
    lexical_path = root / filename
    if lexical_path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    artifact_path = lexical_path.resolve(strict=True)
    if artifact_path.parent != root or not artifact_path.is_file():
        raise ValueError(f"{label} must be a regular file inside the bundle")
    if artifact_path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte size differs")
    return artifact_path


def _base_layer_layout(
    campaign: CampaignConfig,
) -> tuple[dict[str, tuple[int, ...]], list[tuple[str, tuple[int, int], bool]]]:
    with torch.device("meta"):
        model = VolunteerDecoder(campaign.model)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != campaign.model.parameters:
        raise ValueError("layer-bundle model parameter count differs")
    state_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in model.state_dict().items()
    }
    linears = [
        (
            name,
            (int(module.out_features), int(module.in_features)),
            module.bias is not None,
        )
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    return state_shapes, linears


def export_base_layer_bundle(
    campaign: CampaignConfig,
    config: LoRAConfig,
    base_artifact_path: str | Path,
    base_artifact_sha256: str,
    output_dir: str | Path,
) -> BaseLayerBundleExportResult:
    """Export one authenticated resident shard plus one shard per frozen linear.

    This is an offline publication step. It scans the complete source artifact,
    verifies its canonical tensor identity against ``base_model_sha256``, and
    emits a manifest whose digest can be authenticated before worker startup.
    """

    expected_source_sha256 = _layer_bundle_digest(
        base_artifact_sha256,
        "base artifact SHA-256",
    )
    lexical_source = Path(base_artifact_path)
    if lexical_source.is_symlink():
        raise ValueError("base artifact must not be a symlink")
    source_path = lexical_source.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("base artifact must be a regular file")
    if _sha256_file(source_path) != expected_source_sha256:
        raise ValueError("base artifact SHA-256 mismatch")

    mapped = load_safetensors_file(source_path, device="cpu")
    source_tensors = {
        name: tensor.clone().contiguous()
        for name, tensor in mapped.items()
    }
    del mapped
    if _sha256_file(source_path) != expected_source_sha256:
        raise ValueError("base artifact changed during bundle export")
    if any(tensor.dtype != torch.float32 for tensor in source_tensors.values()):
        raise ValueError("base artifact tensors must all be FP32")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in source_tensors.values()):
        raise ValueError("base artifact tensors must all be finite")
    if tensor_sha256(source_tensors) != config.base_model_sha256:
        raise ValueError("base artifact canonical tensor identity mismatch")

    expected_shapes, linear_layout = _base_layer_layout(campaign)
    if set(source_tensors) != set(expected_shapes):
        raise ValueError("base artifact tensor names differ from the campaign model")
    for name, expected_shape in expected_shapes.items():
        if tuple(source_tensors[name].shape) != expected_shape:
            raise ValueError(f"base artifact tensor shape differs: {name}")

    requested_output = Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"base layer bundle already exists: {requested_output}")
    parent = requested_output.parent.resolve(strict=True)
    if not parent.is_dir() or requested_output.name in {"", ".", ".."}:
        raise ValueError("base layer bundle output must have an existing parent")
    output = parent / requested_output.name
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=parent)
    )
    try:
        consumed: set[str] = set()
        linear_entries: list[dict[str, object]] = []
        for index, (module_path, weight_shape, has_bias) in enumerate(linear_layout):
            weight_name = f"{module_path}.weight"
            bias_name = f"{module_path}.bias"
            tensors = {"weight": source_tensors[weight_name]}
            consumed.add(weight_name)
            if has_bias:
                tensors["bias"] = source_tensors[bias_name]
                consumed.add(bias_name)
            filename = f"linear-{index:05d}.safetensors"
            artifact_path = temporary / filename
            save_safetensors_file(tensors, str(artifact_path))
            linear_entries.append(
                {
                    "module": module_path,
                    "file": filename,
                    "file_sha256": _sha256_file(artifact_path),
                    "tensor_sha256": tensor_sha256(tensors),
                    "bytes": artifact_path.stat().st_size,
                    "weight_shape": list(weight_shape),
                    "bias": has_bias,
                }
            )

        resident_tensors = {
            name: source_tensors[name]
            for name in sorted(set(source_tensors) - consumed)
        }
        if not resident_tensors:
            raise ValueError("base layer bundle has no resident tensors")
        resident_path = temporary / "resident.safetensors"
        save_safetensors_file(resident_tensors, str(resident_path))
        manifest: dict[str, object] = {
            "format": "orcacolony_base_layer_bundle_v1",
            "base_model_sha256": config.base_model_sha256,
            "source_artifact_sha256": expected_source_sha256,
            "resident": {
                "file": resident_path.name,
                "file_sha256": _sha256_file(resident_path),
                "tensor_sha256": tensor_sha256(resident_tensors),
                "bytes": resident_path.stat().st_size,
                "tensors": {
                    name: {"shape": list(tensor.shape)}
                    for name, tensor in sorted(resident_tensors.items())
                },
            },
            "linears": linear_entries,
        }
        manifest_bytes = _canonical_json(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        artifact_bytes = sum(
            path.stat().st_size
            for path in temporary.iterdir()
            if path.suffix == ".safetensors"
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return BaseLayerBundleExportResult(
        output_dir=output,
        manifest_path=output / "manifest.json",
        manifest_sha256=manifest_sha256,
        base_model_sha256=config.base_model_sha256,
        source_artifact_sha256=expected_source_sha256,
        linear_count=len(linear_entries),
        artifact_bytes=artifact_bytes,
    )


def _load_base_layer_bundle(
    bundle_dir: str | Path,
    expected_manifest_sha256: str,
    expected_base_model_sha256: str,
) -> _LoadedBaseLayerBundle:
    manifest_sha256 = _layer_bundle_digest(
        expected_manifest_sha256,
        "base layer bundle manifest SHA-256",
    )
    base_model_sha256 = _layer_bundle_digest(
        expected_base_model_sha256,
        "expected base model SHA-256",
    )
    lexical_root = Path(bundle_dir)
    if lexical_root.is_symlink():
        raise ValueError("base layer bundle root must not be a symlink")
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("base layer bundle root must be a directory")
    lexical_manifest = root / "manifest.json"
    if lexical_manifest.is_symlink():
        raise ValueError("base layer bundle manifest must not be a symlink")
    manifest_path = lexical_manifest.resolve(strict=True)
    if manifest_path.parent != root or not manifest_path.is_file():
        raise ValueError("base layer bundle manifest must be inside the bundle")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != manifest_sha256:
        raise ValueError("base layer bundle manifest SHA-256 mismatch")
    manifest_payload = json.loads(
        manifest_bytes.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    manifest = _require_mapping(manifest_payload, "base layer bundle manifest")
    _require_exact_fields(
        manifest,
        {
            "format",
            "base_model_sha256",
            "source_artifact_sha256",
            "resident",
            "linears",
        },
        "base layer bundle manifest",
    )
    if _canonical_json(manifest) != manifest_bytes:
        raise ValueError("base layer bundle manifest is not canonical JSON")
    if manifest["format"] != "orcacolony_base_layer_bundle_v1":
        raise ValueError("unsupported base layer bundle format")
    declared_base = _layer_bundle_digest(
        manifest["base_model_sha256"],
        "base layer bundle base_model_sha256",
    )
    if declared_base != base_model_sha256:
        raise ValueError("base layer bundle does not match base_model_sha256")
    _layer_bundle_digest(
        manifest["source_artifact_sha256"],
        "base layer bundle source_artifact_sha256",
    )

    resident = _require_mapping(manifest["resident"], "base layer bundle resident")
    _require_exact_fields(
        resident,
        {"file", "file_sha256", "tensor_sha256", "bytes", "tensors"},
        "base layer bundle resident",
    )
    resident_bytes = _layer_bundle_positive_int(
        resident["bytes"],
        "base layer bundle resident bytes",
    )
    resident_path = _layer_bundle_artifact_path(
        root,
        resident["file"],
        resident_bytes,
        "base layer bundle resident artifact",
    )
    resident_shapes_payload = _require_mapping(
        resident["tensors"],
        "base layer bundle resident tensors",
    )
    if not resident_shapes_payload:
        raise ValueError("base layer bundle resident tensors must not be empty")
    resident_shapes: dict[str, tuple[int, ...]] = {}
    for name, contract_payload in resident_shapes_payload.items():
        if _TARGET_PATTERN.fullmatch(name) is None:
            raise ValueError(f"invalid base layer bundle resident tensor name: {name!r}")
        contract = _require_mapping(
            contract_payload,
            f"base layer bundle resident tensor {name}",
        )
        _require_exact_fields(
            contract,
            {"shape"},
            f"base layer bundle resident tensor {name}",
        )
        resident_shapes[name] = _layer_bundle_shape(
            contract["shape"],
            f"base layer bundle resident tensor shape: {name}",
        )

    linears_payload = manifest["linears"]
    if not isinstance(linears_payload, list) or not linears_payload:
        raise ValueError("base layer bundle linears must be a non-empty JSON array")
    linears: dict[str, _LayerBundleLinearDescriptor] = {}
    artifact_paths = {resident_path}
    for index, entry_payload in enumerate(linears_payload):
        entry = _require_mapping(
            entry_payload,
            f"base layer bundle linear {index}",
        )
        _require_exact_fields(
            entry,
            {
                "module",
                "file",
                "file_sha256",
                "tensor_sha256",
                "bytes",
                "weight_shape",
                "bias",
            },
            f"base layer bundle linear {index}",
        )
        module_path = entry["module"]
        if not isinstance(module_path, str) or _TARGET_PATTERN.fullmatch(module_path) is None:
            raise ValueError("base layer bundle linear module path is invalid")
        if module_path in linears:
            raise ValueError("base layer bundle linear module paths must be unique")
        artifact_bytes = _layer_bundle_positive_int(
            entry["bytes"],
            f"base layer bundle linear bytes: {module_path}",
        )
        artifact_path = _layer_bundle_artifact_path(
            root,
            entry["file"],
            artifact_bytes,
            f"base layer bundle linear artifact: {module_path}",
        )
        if artifact_path in artifact_paths:
            raise ValueError("base layer bundle artifact files must be unique")
        artifact_paths.add(artifact_path)
        has_bias = entry["bias"]
        if not isinstance(has_bias, bool):
            raise ValueError("base layer bundle linear bias must be boolean")
        weight_shape = _layer_bundle_shape(
            entry["weight_shape"],
            f"base layer bundle linear weight shape: {module_path}",
            rank=2,
        )
        linears[module_path] = _LayerBundleLinearDescriptor(
            module_path=module_path,
            artifact_path=artifact_path,
            artifact_sha256=_layer_bundle_digest(
                entry["file_sha256"],
                f"base layer bundle linear file SHA-256: {module_path}",
            ),
            tensor_sha256=_layer_bundle_digest(
                entry["tensor_sha256"],
                f"base layer bundle linear tensor SHA-256: {module_path}",
            ),
            artifact_bytes=artifact_bytes,
            weight_shape=(weight_shape[0], weight_shape[1]),
            has_bias=has_bias,
        )
    expected_artifact_names = {
        "manifest.json",
        resident_path.name,
        *(descriptor.artifact_path.name for descriptor in linears.values()),
    }
    if {entry.name for entry in root.iterdir()} != expected_artifact_names:
        raise ValueError("base layer bundle artifact set differs from the manifest")
    return _LoadedBaseLayerBundle(
        root=root,
        manifest_sha256=manifest_sha256,
        base_model_sha256=declared_base,
        resident_path=resident_path,
        resident_sha256=_layer_bundle_digest(
            resident["file_sha256"],
            "base layer bundle resident file SHA-256",
        ),
        resident_tensor_sha256=_layer_bundle_digest(
            resident["tensor_sha256"],
            "base layer bundle resident tensor SHA-256",
        ),
        resident_bytes=resident_bytes,
        resident_shapes=resident_shapes,
        linears=linears,
    )


def base_layer_bundle_artifact_contract(
    bundle_dir: str | Path,
    bundle_manifest_sha256: str,
    base_model_sha256: str,
    *,
    verify_artifacts: bool = True,
) -> dict[str, object]:
    """Return the exact transport contract for an authenticated layer bundle.

    Coordinators use the default full raw-file verification before publication.
    Workers may set ``verify_artifacts=False`` after fresh-download SHA-256 checks;
    that path still authenticates the canonical manifest, exact membership, names,
    sizes, and base identity without rescanning every warm cached shard.
    """

    bundle = _load_base_layer_bundle(
        bundle_dir,
        bundle_manifest_sha256,
        base_model_sha256,
    )
    artifacts = [
        {
            "file": "manifest.json",
            "sha256": bundle.manifest_sha256,
            "bytes": (bundle.root / "manifest.json").stat().st_size,
        },
        {
            "file": bundle.resident_path.name,
            "sha256": bundle.resident_sha256,
            "bytes": bundle.resident_bytes,
        },
        *[
            {
                "file": descriptor.artifact_path.name,
                "sha256": descriptor.artifact_sha256,
                "bytes": descriptor.artifact_bytes,
            }
            for descriptor in sorted(
                bundle.linears.values(),
                key=lambda value: value.artifact_path.name,
            )
        ],
    ]
    if verify_artifacts:
        for artifact in artifacts:
            path = bundle.root / str(artifact["file"])
            if _sha256_file(path) != artifact["sha256"]:
                raise ValueError(
                    f"base layer bundle raw artifact SHA-256 mismatch: {path.name}"
                )
    return {
        "format": "orcacolony_base_layer_bundle_artifacts_v1",
        "profile": LAYER_BUNDLE_STREAMED_FP32_PROFILE,
        "manifest_sha256": bundle.manifest_sha256,
        "base_model_sha256": bundle.base_model_sha256,
        "artifacts": artifacts,
        "download_bytes": sum(int(artifact["bytes"]) for artifact in artifacts),
    }


class _LayerBundleStreamedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        module: "LayerBundleStreamedFrozenLinear",
    ) -> Tensor:
        weight, bias = module.load_tensors()
        ctx.module = module
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), weight, bias)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        weight, _ = ctx.module.load_tensors()
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            return grad_output.float().matmul(weight), None


class LayerBundleStreamedFrozenLinear(nn.Module):
    """Exact-FP32 frozen linear backed by one authenticated bundle shard."""

    def __init__(self, descriptor: _LayerBundleLinearDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.artifact_path = descriptor.artifact_path
        self.expected_artifact_sha256 = descriptor.artifact_sha256
        self.expected_tensor_sha256 = descriptor.tensor_sha256
        self.artifact_bytes = descriptor.artifact_bytes
        self.out_features, self.in_features = descriptor.weight_shape
        self.has_bias = descriptor.has_bias
        self.read_bytes = 0
        self.read_count = 0

    def load_tensors(self) -> tuple[Tensor, Tensor | None]:
        if self.artifact_path.stat().st_size != self.artifact_bytes:
            raise ValueError("layer-bundle linear artifact byte size differs")
        mapped = load_safetensors_file(self.artifact_path, device="cpu")
        tensors = {
            name: tensor.clone().contiguous()
            for name, tensor in mapped.items()
        }
        del mapped
        expected_names = {"weight", "bias"} if self.has_bias else {"weight"}
        if set(tensors) != expected_names:
            raise ValueError("layer-bundle linear tensor names differ")
        weight = tensors["weight"]
        bias = tensors.get("bias")
        if weight.dtype != torch.float32 or weight.shape != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError("layer-bundle linear weight contract differs")
        if bias is not None and (
            bias.dtype != torch.float32 or bias.shape != (self.out_features,)
        ):
            raise ValueError("layer-bundle linear bias contract differs")
        if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()):
            raise ValueError("layer-bundle linear tensors must be finite")
        if tensor_sha256(tensors) != self.expected_tensor_sha256:
            raise ValueError("layer-bundle linear tensor digest mismatch")
        self.read_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors.values()
        )
        self.read_count += 1
        return weight, bias

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise ValueError("layer-bundle streamed profile requires CPU FP32 activations")
        return _LayerBundleStreamedFrozenLinearFunction.apply(inputs, self)


class LayerBundleInt8FrozenLinear(Int8FrozenLinear):
    """Resident int8 linear quantized from one authenticated bundle shard."""

    def __init__(self, source: LayerBundleStreamedFrozenLinear) -> None:
        nn.Module.__init__(self)
        weight, bias = source.load_tensors()
        scales = (
            weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
            / 127.0
        )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.artifact_path = source.artifact_path
        self.expected_artifact_sha256 = source.expected_artifact_sha256
        self.expected_tensor_sha256 = source.expected_tensor_sha256
        self.artifact_bytes = source.artifact_bytes
        self.artifact_open_count = source.read_count
        self.artifact_read_bytes = source.read_bytes
        self.register_buffer(
            "qweight",
            torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8),
        )
        self.register_buffer("scales", scales)
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().to(dtype=torch.float32).clone(),
        )


def _quantize_layer_bundle_streamed_linears(module: nn.Module) -> int:
    count = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            if not isinstance(child.base, LayerBundleStreamedFrozenLinear):
                raise ValueError("layer-bundle int8 LoRA base is not streamed")
            child.base = LayerBundleInt8FrozenLinear(child.base)
            count += 1
        elif isinstance(child, LayerBundleStreamedFrozenLinear):
            setattr(module, child_name, LayerBundleInt8FrozenLinear(child))
            count += 1
        else:
            count += _quantize_layer_bundle_streamed_linears(child)
    return count


def _replace_with_layer_bundle_linears(
    module: nn.Module,
    descriptors: Mapping[str, _LayerBundleLinearDescriptor],
    prefix: str = "",
) -> tuple[int, set[str]]:
    count = 0
    consumed: set[str] = set()
    for child_name, child in list(module.named_children()):
        child_path = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            source = child.base
            if not isinstance(source, nn.Linear):
                raise ValueError("layer-bundle LoRA base is not linear")
            descriptor = descriptors.get(child_path)
            if descriptor is None:
                raise ValueError(f"base layer bundle is missing linear: {child_path}")
            if descriptor.weight_shape != tuple(source.weight.shape):
                raise ValueError(f"base layer bundle linear shape differs: {child_path}")
            if descriptor.has_bias != (source.bias is not None):
                raise ValueError(f"base layer bundle linear bias differs: {child_path}")
            child.base = LayerBundleStreamedFrozenLinear(descriptor)
            consumed.add(child_path)
            count += 1
        elif isinstance(child, nn.Linear):
            if any(parameter.requires_grad for parameter in child.parameters()):
                raise ValueError("layer-bundle streaming requires frozen linears")
            descriptor = descriptors.get(child_path)
            if descriptor is None:
                raise ValueError(f"base layer bundle is missing linear: {child_path}")
            if descriptor.weight_shape != tuple(child.weight.shape):
                raise ValueError(f"base layer bundle linear shape differs: {child_path}")
            if descriptor.has_bias != (child.bias is not None):
                raise ValueError(f"base layer bundle linear bias differs: {child_path}")
            setattr(
                module,
                child_name,
                LayerBundleStreamedFrozenLinear(descriptor),
            )
            consumed.add(child_path)
            count += 1
        else:
            nested_count, nested_consumed = _replace_with_layer_bundle_linears(
                child,
                descriptors,
                child_path,
            )
            count += nested_count
            consumed.update(nested_consumed)
    return count, consumed


def build_layer_bundle_streamed_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
    bundle_dir: str | Path,
    bundle_manifest_sha256: str,
    adapter_state: Mapping[str, Tensor],
) -> VolunteerDecoder:
    """Construct a meta/empty LoRA model from pre-authenticated layer shards."""

    bundle = _load_base_layer_bundle(
        bundle_dir,
        bundle_manifest_sha256,
        config.base_model_sha256,
    )
    with torch.device("meta"):
        model = VolunteerDecoder(campaign.model)
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        if actual_parameters != campaign.model.parameters:
            raise ValueError("layer-bundle model parameter count differs")
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

    linear_count, consumed_linears = _replace_with_layer_bundle_linears(
        model,
        bundle.linears,
    )
    if linear_count <= 0 or consumed_linears != set(bundle.linears):
        raise ValueError("base layer bundle linear partition differs")
    model.to_empty(device="cpu")

    mapped_resident = load_safetensors_file(bundle.resident_path, device="cpu")
    resident_tensors = {
        name: tensor.clone().contiguous()
        for name, tensor in mapped_resident.items()
    }
    del mapped_resident
    if bundle.resident_path.stat().st_size != bundle.resident_bytes:
        raise ValueError("base layer bundle resident artifact byte size differs")
    if set(resident_tensors) != set(bundle.resident_shapes):
        raise ValueError("base layer bundle resident tensor names differ")
    if tensor_sha256(resident_tensors) != bundle.resident_tensor_sha256:
        raise ValueError("base layer bundle resident tensor digest mismatch")

    adapter_suffixes = (".lora_a", ".lora_b")
    resident_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.endswith(adapter_suffixes)
    }
    if set(resident_parameters) != set(resident_tensors):
        raise ValueError("base layer bundle resident parameter partition differs")
    with torch.no_grad():
        for name, parameter in resident_parameters.items():
            tensor = resident_tensors[name]
            if tensor.dtype != torch.float32:
                raise ValueError(f"base layer bundle resident tensor is not FP32: {name}")
            if tensor.shape != parameter.shape or tensor.shape != bundle.resident_shapes[name]:
                raise ValueError(f"base layer bundle resident tensor shape differs: {name}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"base layer bundle resident tensor is non-finite: {name}")
            parameter.copy_(tensor)

    context_length = campaign.model.context_length
    causal_mask = torch.triu(
        torch.ones(context_length, context_length, dtype=torch.bool),
        diagonal=1,
    )
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.causal_mask = causal_mask.clone()
    load_adapter_state(model, adapter_state)
    adapter_named_parameters(model)
    return model


def build_layer_bundle_int8_lora_model(
    campaign: CampaignConfig,
    config: LoRAConfig,
    bundle_dir: str | Path,
    bundle_manifest_sha256: str,
    adapter_state: Mapping[str, Tensor],
) -> VolunteerDecoder:
    """Construct resident int8 linears one authenticated bundle shard at a time."""

    model = build_layer_bundle_streamed_lora_model(
        campaign,
        config,
        bundle_dir,
        bundle_manifest_sha256,
        adapter_state,
    )
    expected_count = sum(
        isinstance(module, LayerBundleStreamedFrozenLinear)
        for module in model.modules()
    )
    quantized_count = _quantize_layer_bundle_streamed_linears(model)
    if expected_count <= 0 or quantized_count != expected_count:
        raise ValueError("layer-bundle int8 linear partition differs")
    if any(
        isinstance(module, (nn.Linear, LayerBundleStreamedFrozenLinear))
        for module in model.modules()
    ):
        raise ValueError("layer-bundle int8 model retained an FP32 linear")
    adapter_named_parameters(model)
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


def _safe_checkpoint_artifact_path(
    root: str | Path,
    filename: object,
    label: str,
) -> Path:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or bool(Path(filename).drive)
        or Path(filename).name != filename
    ):
        raise ValueError(f"{label} must be a safe plain basename")
    resolved_root = Path(root).resolve()
    resolved_path = (resolved_root / filename).resolve()
    if resolved_path.parent != resolved_root:
        raise ValueError(f"{label} must resolve under the checkpoint root")
    return resolved_path


def _nonnegative_integral_step(value: object, label: str) -> int:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{label} must be a non-negative integral optimizer step")
        value = value.detach().cpu().item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
    ):
        raise ValueError(f"{label} must be a non-negative integral optimizer step")
    return int(value)


def _validated_optimizer_tensors(
    adapters: Mapping[str, nn.Parameter],
    optimizer: torch.optim.AdamW,
    checkpoint_step: object,
) -> tuple[dict[str, Tensor], int]:
    declared_step = _nonnegative_integral_step(
        checkpoint_step,
        "LoRA checkpoint optimizer step",
    )
    parameter_states = optimizer.state
    expected_parameters = set(adapters.values())
    actual_parameters = set(parameter_states)
    if not actual_parameters and declared_step == 0:
        return (
            {
                f"{prefix}{name}": torch.zeros_like(parameter).detach().cpu().contiguous()
                for name, parameter in adapters.items()
                for prefix in ("exp_avg.", "exp_avg_sq.")
            },
            0,
        )
    if actual_parameters != expected_parameters:
        raise ValueError("LoRA optimizer state does not exactly match the adapters")

    optimizer_tensors: dict[str, Tensor] = {}
    parameter_steps: list[int] = []
    for name, parameter in adapters.items():
        parameter_state = parameter_states[parameter]
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"LoRA optimizer state fields differ for adapter: {name}")
        parameter_steps.append(
            _nonnegative_integral_step(
                parameter_state["step"],
                f"LoRA optimizer parameter step for {name}",
            )
        )
        for state_name in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state[state_name]
            if not isinstance(tensor, Tensor):
                raise ValueError(f"LoRA optimizer {state_name} must be a tensor: {name}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"LoRA optimizer {state_name} must be float32: {name}")
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"LoRA optimizer {state_name} shape differs for {name}: "
                    f"expected={tuple(parameter.shape)}, actual={tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"LoRA optimizer {state_name} contains non-finite values: {name}"
                )
            optimizer_tensors[f"{state_name}.{name}"] = (
                tensor.detach().cpu().contiguous()
            )
    if len(set(parameter_steps)) != 1:
        raise ValueError("LoRA optimizer steps do not agree across adapter parameters")
    optimizer_step = parameter_steps[0]
    if optimizer_step != declared_step:
        raise ValueError("LoRA optimizer step does not match the checkpoint training step")
    return optimizer_tensors, optimizer_step


def load_lora_manifest(
    campaign_path: str | Path,
    manifest_path: str | Path,
    *,
    verify_base_model: bool = True,
) -> LoadedLoRAManifest:
    """Load the LoRA contract, optionally deferring deterministic-base rebuild.

    ``verify_base_model=False`` is only for startup paths that separately bind
    an authenticated artifact manifest to ``base_model_sha256``. The default
    preserves the resident deterministic-base verification used elsewhere.
    """

    if not isinstance(verify_base_model, bool):
        raise ValueError("verify_base_model must be boolean")
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
    campaign = campaign_from_mapping(campaign_mapping)
    if verify_base_model:
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


def lora_weight_checkpoint_sha256(
    loaded: LoadedLoRAManifest,
    adapter_sha256: str,
    *,
    numerical_profile: str | None = None,
) -> str:
    if _SHA256_PATTERN.fullmatch(adapter_sha256) is None:
        raise ValueError("adapter_sha256 must be a lowercase SHA-256 digest")
    if numerical_profile is not None:
        profile = _validated_lora_numerical_profile(numerical_profile)
        return _sha256_bytes(
            _canonical_json(
                {
                    "format": "orcacolony_lora_checkpoint_identity_v2",
                    "lora_manifest_sha256": loaded.manifest_sha256,
                    "base_model_sha256": loaded.config.base_model_sha256,
                    "adapter_sha256": adapter_sha256,
                    "numerical_profile": profile,
                }
            )
        )
    return _sha256_bytes(
        _canonical_json(
            {
                "format": "orcacolony_lora_checkpoint_identity_v1",
                "lora_manifest_sha256": loaded.manifest_sha256,
                "base_model_sha256": loaded.config.base_model_sha256,
                "adapter_sha256": adapter_sha256,
            }
        )
    )


def _validated_lora_trajectory(
    loaded: LoadedLoRAManifest,
    step: int,
    dataset_cursor: object,
    loss_history: object,
) -> tuple[int, list[float]]:
    if isinstance(dataset_cursor, bool) or not isinstance(dataset_cursor, int):
        raise ValueError("LoRA checkpoint dataset cursor must be an integer")
    expected_cursor = (
        step * loaded.campaign.training.batch_size
    ) % loaded.campaign.training.dataset_sequences
    if dataset_cursor != expected_cursor:
        raise ValueError(
            "LoRA checkpoint dataset cursor does not match the training step: "
            f"expected={expected_cursor}, actual={dataset_cursor}"
        )
    if not isinstance(loss_history, list):
        raise ValueError("LoRA checkpoint loss history must be a JSON array")
    if len(loss_history) != step:
        raise ValueError(
            "LoRA checkpoint loss history length does not match the training step"
        )
    normalized_history: list[float] = []
    for index, loss in enumerate(loss_history):
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError(
                "LoRA checkpoint loss history values must be finite numbers: "
                f"index={index}"
            )
        normalized_loss = float(loss)
        if not math.isfinite(normalized_loss):
            raise ValueError(
                "LoRA checkpoint loss history values must be finite: "
                f"index={index}"
            )
        normalized_history.append(normalized_loss)
    return dataset_cursor, normalized_history


def lora_resume_state_sha256(
    loaded: LoadedLoRAManifest,
    *,
    weight_checkpoint_sha256: str,
    adapter_file_sha256: str,
    optimizer_file_sha256: str,
    step: int,
    optimizer_step: int,
    dataset_cursor: int,
    dataset_revision: str,
    loss_history: list[float],
    numerical_profile: str | None = None,
) -> str:
    for label, digest in (
        ("weight checkpoint", weight_checkpoint_sha256),
        ("adapter file", adapter_file_sha256),
        ("optimizer file", optimizer_file_sha256),
    ):
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"LoRA {label} digest must be lowercase SHA-256")
    payload: dict[str, object] = {
        "format": (
            "orcacolony_lora_resume_state_identity_v2"
            if numerical_profile is not None
            else "orcacolony_lora_resume_state_identity_v1"
        ),
        "campaign_id": loaded.campaign.campaign["id"],
        "campaign_sha256": loaded.campaign_sha256,
        "lora_manifest_sha256": loaded.manifest_sha256,
        "base_model_sha256": loaded.config.base_model_sha256,
        "weight_checkpoint_sha256": weight_checkpoint_sha256,
        "adapter_file_sha256": adapter_file_sha256,
        "optimizer_file_sha256": optimizer_file_sha256,
        "step": step,
        "optimizer_step": optimizer_step,
        "dataset_cursor": dataset_cursor,
        "dataset_revision": dataset_revision,
        "loss_history": loss_history,
    }
    if numerical_profile is not None:
        payload["numerical_profile"] = _validated_lora_numerical_profile(
            numerical_profile
        )
    return _sha256_bytes(_canonical_json(payload))


def load_adapter_state(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
) -> None:
    adapters = adapter_named_parameters(model)
    if sorted(tensors) != list(adapters):
        missing = sorted(set(adapters) - set(tensors))
        unexpected = sorted(set(tensors) - set(adapters))
        raise ValueError(
            "adapter checkpoint names differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    validated: dict[str, Tensor] = {}
    for name, parameter in adapters.items():
        tensor = tensors[name]
        if tensor.dtype != torch.float32:
            raise ValueError(f"adapter checkpoint must be float32: {name}")
        if tensor.shape != parameter.shape:
            raise ValueError(
                f"adapter checkpoint shape differs for {name}: "
                f"expected={tuple(parameter.shape)}, actual={tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"adapter checkpoint contains non-finite values: {name}")
        validated[name] = tensor.to(parameter.device, parameter.dtype)
    snapshots = {
        name: parameter.detach().clone()
        for name, parameter in adapters.items()
    }
    with torch.no_grad():
        try:
            for name, parameter in adapters.items():
                parameter.copy_(validated[name])
        except BaseException:
            for name, parameter in adapters.items():
                parameter.copy_(snapshots[name])
            raise


def save_lora_checkpoint(
    loaded: LoadedLoRAManifest,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    output_dir: str | Path,
    *,
    step: int,
    dataset_cursor: int,
    loss_history: list[float],
    numerical_profile: str | None = None,
) -> LoRATrainingResult:
    adapters = adapter_named_parameters(model)
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameters != {id(parameter) for parameter in adapters.values()}:
        raise ValueError("LoRA checkpoint optimizer does not exactly own the adapters")
    checkpoint_step = _nonnegative_integral_step(
        step,
        "LoRA checkpoint training step",
    )
    optimizer_tensors, optimizer_step = _validated_optimizer_tensors(
        adapters,
        optimizer,
        checkpoint_step,
    )
    normalized_cursor, normalized_history = _validated_lora_trajectory(
        loaded,
        checkpoint_step,
        dataset_cursor,
        loss_history,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter_path = output / "adapter.safetensors"
    optimizer_path = output / "optimizer.safetensors"
    state_path = output / "state.json"
    adapter_tmp = output / "adapter.safetensors.tmp"
    optimizer_tmp = output / "optimizer.safetensors.tmp"
    state_tmp = output / "state.json.tmp"

    adapter_tensors = _adapter_state(model)
    adapter_sha256 = tensor_sha256(adapter_tensors)
    save_safetensors_file(adapter_tensors, str(adapter_tmp))
    os.replace(adapter_tmp, adapter_path)

    save_safetensors_file(optimizer_tensors, str(optimizer_tmp))
    os.replace(optimizer_tmp, optimizer_path)

    profile = (
        _validated_lora_numerical_profile(numerical_profile)
        if numerical_profile is not None
        else None
    )
    weight_checkpoint_sha256 = lora_weight_checkpoint_sha256(
        loaded,
        adapter_sha256,
        numerical_profile=profile,
    )
    adapter_file_sha256 = _sha256_file(adapter_path)
    optimizer_file_sha256 = _sha256_file(optimizer_path)
    dataset_revision = (
        loaded.campaign.dataset["manifest_sha256"]
        if loaded.campaign.dataset is not None
        else "synthetic-fixture-v1"
    )
    checkpoint_sha256 = lora_resume_state_sha256(
        loaded,
        weight_checkpoint_sha256=weight_checkpoint_sha256,
        adapter_file_sha256=adapter_file_sha256,
        optimizer_file_sha256=optimizer_file_sha256,
        step=checkpoint_step,
        optimizer_step=optimizer_step,
        dataset_cursor=normalized_cursor,
        dataset_revision=dataset_revision,
        loss_history=normalized_history,
        numerical_profile=profile,
    )
    state: dict[str, object] = {
        "format": (
            "orcacolony_lora_checkpoint_v2"
            if profile is not None
            else "orcacolony_lora_checkpoint_v1"
        ),
        "campaign_id": loaded.campaign.campaign["id"],
        "campaign_sha256": loaded.campaign_sha256,
        "lora_manifest_sha256": loaded.manifest_sha256,
        "base_model_sha256": loaded.config.base_model_sha256,
        "weight_checkpoint_sha256": weight_checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "step": checkpoint_step,
        "optimizer_step": optimizer_step,
        "dataset_cursor": normalized_cursor,
        "dataset_revision": dataset_revision,
        "loss_history": normalized_history,
        "adapter": {
            "file": adapter_path.name,
            "file_sha256": adapter_file_sha256,
            "tensor_sha256": adapter_sha256,
            "tensor_order": list(adapters),
        },
        "optimizer": {
            "file": optimizer_path.name,
            "sha256": optimizer_file_sha256,
        },
    }
    if profile is not None:
        state["numerical_profile"] = profile
    state_tmp.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(state_tmp, state_path)
    return LoRATrainingResult(
        checkpoint_dir=output,
        steps_completed=checkpoint_step,
        loss_history=tuple(normalized_history),
        base_model_sha256=loaded.config.base_model_sha256,
        adapter_sha256=adapter_sha256,
        weight_checkpoint_sha256=weight_checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )


def load_lora_checkpoint(
    loaded: LoadedLoRAManifest,
    checkpoint_dir: str | Path,
    *,
    expected_numerical_profile: str | None = None,
) -> tuple[VolunteerDecoder, torch.optim.AdamW, int, int, list[float]]:
    checkpoint = Path(checkpoint_dir).resolve()
    state_path = _safe_checkpoint_artifact_path(
        checkpoint,
        "state.json",
        "LoRA checkpoint state file",
    )
    state = _require_mapping(
        json.loads(
            state_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        ),
        "LoRA checkpoint state",
    )
    checkpoint_format = state.get("format")
    state_fields = {
        "format",
        "campaign_id",
        "campaign_sha256",
        "lora_manifest_sha256",
        "base_model_sha256",
        "weight_checkpoint_sha256",
        "checkpoint_sha256",
        "step",
        "optimizer_step",
        "dataset_cursor",
        "dataset_revision",
        "loss_history",
        "adapter",
        "optimizer",
    }
    if checkpoint_format == "orcacolony_lora_checkpoint_v1":
        numerical_profile = EXACT_CPU_FP32_PROFILE
        identity_profile = None
    elif checkpoint_format == "orcacolony_lora_checkpoint_v2":
        state_fields.add("numerical_profile")
        numerical_profile = _validated_lora_numerical_profile(
            state.get("numerical_profile")
        )
        identity_profile = numerical_profile
    else:
        raise ValueError("unsupported LoRA checkpoint format")
    _require_exact_fields(state, state_fields, "LoRA checkpoint state")
    if expected_numerical_profile is not None and numerical_profile != (
        _validated_lora_numerical_profile(expected_numerical_profile)
    ):
        raise ValueError("LoRA checkpoint numerical profile mismatch")
    if state.get("campaign_id") != loaded.campaign.campaign["id"]:
        raise ValueError("LoRA checkpoint campaign does not match configuration")
    if state.get("campaign_sha256") != loaded.campaign_sha256:
        raise ValueError("LoRA checkpoint campaign digest mismatch")
    if state.get("lora_manifest_sha256") != loaded.manifest_sha256:
        raise ValueError("LoRA checkpoint manifest digest mismatch")
    if state.get("base_model_sha256") != loaded.config.base_model_sha256:
        raise ValueError("LoRA checkpoint base model digest mismatch")
    expected_dataset_revision = (
        loaded.campaign.dataset["manifest_sha256"]
        if loaded.campaign.dataset is not None
        else "synthetic-fixture-v1"
    )
    if state.get("dataset_revision") != expected_dataset_revision:
        raise ValueError("LoRA checkpoint dataset revision mismatch")

    checkpoint_step = _nonnegative_integral_step(
        state.get("step"),
        "LoRA checkpoint training step",
    )
    optimizer_step = _nonnegative_integral_step(
        state.get("optimizer_step"),
        "LoRA checkpoint optimizer step",
    )
    if optimizer_step != checkpoint_step:
        raise ValueError("LoRA checkpoint optimizer step does not match training step")
    dataset_cursor, loss_history = _validated_lora_trajectory(
        loaded,
        checkpoint_step,
        state.get("dataset_cursor"),
        state.get("loss_history"),
    )

    adapter_state = _require_mapping(state.get("adapter"), "LoRA checkpoint adapter")
    optimizer_state = _require_mapping(
        state.get("optimizer"),
        "LoRA checkpoint optimizer",
    )
    _require_exact_fields(
        adapter_state,
        {"file", "file_sha256", "tensor_sha256", "tensor_order"},
        "LoRA checkpoint adapter",
    )
    _require_exact_fields(
        optimizer_state,
        {"file", "sha256"},
        "LoRA checkpoint optimizer",
    )
    adapter_path = _safe_checkpoint_artifact_path(
        checkpoint,
        adapter_state.get("file"),
        "LoRA adapter checkpoint file",
    )
    optimizer_path = _safe_checkpoint_artifact_path(
        checkpoint,
        optimizer_state.get("file"),
        "LoRA optimizer checkpoint file",
    )
    if _sha256_file(adapter_path) != adapter_state.get("file_sha256"):
        raise ValueError("LoRA adapter checkpoint file digest mismatch")
    if _sha256_file(optimizer_path) != optimizer_state.get("sha256"):
        raise ValueError("LoRA optimizer checkpoint digest mismatch")
    adapter_tensors = load_safetensors_file(str(adapter_path))
    adapter_sha256 = tensor_sha256(adapter_tensors)
    if adapter_sha256 != adapter_state.get("tensor_sha256"):
        raise ValueError("LoRA adapter tensor digest mismatch")
    weight_checkpoint_sha256 = lora_weight_checkpoint_sha256(
        loaded,
        adapter_sha256,
        numerical_profile=identity_profile,
    )
    if weight_checkpoint_sha256 != state.get("weight_checkpoint_sha256"):
        raise ValueError("LoRA weight checkpoint identity mismatch")

    model = build_profiled_lora_model(
        loaded.campaign,
        loaded.config,
        numerical_profile,
    )
    if adapter_state.get("tensor_order") != list(adapter_named_parameters(model)):
        raise ValueError("LoRA adapter tensor order does not match the manifest")
    load_adapter_state(model, adapter_tensors)
    adapters = adapter_named_parameters(model)
    optimizer = create_adapter_optimizer(model, loaded.campaign.training)
    optimizer_tensors = load_safetensors_file(str(optimizer_path))
    expected_optimizer_names = {
        prefix + name
        for name in adapters
        for prefix in ("exp_avg.", "exp_avg_sq.")
    }
    if set(optimizer_tensors) != expected_optimizer_names:
        raise ValueError("LoRA optimizer tensor set does not match the adapters")
    for name, parameter in adapters.items():
        parameter_state: dict[str, Tensor] = {
            "step": torch.tensor(float(optimizer_step), dtype=torch.float32),
        }
        for state_name in ("exp_avg", "exp_avg_sq"):
            tensor = optimizer_tensors[f"{state_name}.{name}"]
            if tensor.dtype != torch.float32:
                raise ValueError(f"LoRA optimizer {state_name} must be float32: {name}")
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"LoRA optimizer {state_name} shape differs for {name}: "
                    f"expected={tuple(parameter.shape)}, actual={tuple(tensor.shape)}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"LoRA optimizer {state_name} contains non-finite values: {name}"
                )
            parameter_state[state_name] = tensor.to(parameter.device).clone()
        optimizer.state[parameter] = parameter_state
    checkpoint_sha256 = lora_resume_state_sha256(
        loaded,
        weight_checkpoint_sha256=weight_checkpoint_sha256,
        adapter_file_sha256=str(adapter_state.get("file_sha256")),
        optimizer_file_sha256=str(optimizer_state.get("sha256")),
        step=checkpoint_step,
        optimizer_step=optimizer_step,
        dataset_cursor=dataset_cursor,
        dataset_revision=expected_dataset_revision,
        loss_history=loss_history,
        numerical_profile=identity_profile,
    )
    if checkpoint_sha256 != state.get("checkpoint_sha256"):
        raise ValueError("LoRA checkpoint identity mismatch")
    return (
        model,
        optimizer,
        checkpoint_step,
        dataset_cursor,
        loss_history,
    )


def evaluate_lora_checkpoint(
    loaded: LoadedLoRAManifest,
    checkpoint_dir: str | Path,
    dataset: PackedDataset,
) -> dict[str, object]:
    validate_dataset_artifacts(loaded.campaign, dataset)
    if loaded.campaign.evaluation is None:
        raise ValueError("campaign does not define an evaluation profile")
    checkpoint = Path(checkpoint_dir)
    model, _, step, _, _ = load_lora_checkpoint(loaded, checkpoint)
    sequence_count = int(loaded.campaign.evaluation["validation_sequences"])
    batch_size = int(loaded.campaign.evaluation["batch_size"])
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
                logits.reshape(-1, loaded.campaign.model.vocabulary_size),
                targets.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(batch_loss)
            loss_weight_sum += targets.numel()
            cursor += current_batch_size
    checkpoint_state = _require_mapping(
        json.loads((checkpoint / "state.json").read_text(encoding="utf-8")),
        "LoRA checkpoint state",
    )
    mean_loss = loss_sum / loss_weight_sum
    return {
        "format": "orcacolony_evaluation_v1",
        "campaign_id": loaded.campaign.campaign["id"],
        "training_method": "frozen-base-lora",
        "numerical_profile": checkpoint_state.get(
            "numerical_profile", EXACT_CPU_FP32_PROFILE
        ),
        "step": step,
        "dataset_revision": dataset.revision,
        "base_model_sha256": checkpoint_state["base_model_sha256"],
        "adapter_sha256": checkpoint_state["adapter"]["tensor_sha256"],
        "weight_checkpoint_sha256": checkpoint_state["weight_checkpoint_sha256"],
        "resume_state_sha256": checkpoint_state["checkpoint_sha256"],
        "checkpoint_sha256": checkpoint_state["checkpoint_sha256"],
        "validation_sequences": sequence_count,
        "loss_sum": loss_sum,
        "loss_weight_sum": loss_weight_sum,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def run_lora_training(
    loaded: LoadedLoRAManifest,
    output_dir: str | Path,
    *,
    target_steps: int,
    resume_from: str | Path | None = None,
    dataset: PackedDataset | None = None,
    numerical_profile: str | None = None,
) -> LoRATrainingResult:
    validate_dataset_artifacts(loaded.campaign, dataset)
    profile = (
        _validated_lora_numerical_profile(numerical_profile)
        if numerical_profile is not None
        else EXACT_CPU_FP32_PROFILE
    )
    if resume_from is None:
        model = build_profiled_lora_model(
            loaded.campaign,
            loaded.config,
            profile,
        )
        optimizer = create_adapter_optimizer(model, loaded.campaign.training)
        step = 0
        dataset_cursor = 0
        loss_history: list[float] = []
    else:
        model, optimizer, step, dataset_cursor, loss_history = load_lora_checkpoint(
            loaded,
            resume_from,
            expected_numerical_profile=profile,
        )
    if target_steps < step or (target_steps == step and resume_from is not None):
        raise ValueError("target_steps must be greater than the LoRA checkpoint step")

    while step < target_steps:
        inputs, targets = fixture_batch(
            loaded.campaign,
            dataset_cursor,
            dataset,
        )
        submitted = compute_adapter_gradients(model, inputs, targets)
        apply_adapter_gradient_step(
            model,
            optimizer,
            submitted.gradients,
            submitted.loss_weight_sum,
            loaded.campaign.training.max_gradient_norm,
        )
        loss_history.append(submitted.loss_sum / submitted.loss_weight_sum)
        dataset_cursor = (
            dataset_cursor + loaded.campaign.training.batch_size
        ) % loaded.campaign.training.dataset_sequences
        step += 1
    return save_lora_checkpoint(
        loaded,
        model,
        optimizer,
        output_dir,
        step=step,
        dataset_cursor=dataset_cursor,
        loss_history=loss_history,
        numerical_profile=numerical_profile,
    )


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
