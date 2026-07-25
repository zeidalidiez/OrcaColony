from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from orcacolony import peft
from orcacolony.peft import compute_adapter_gradients, load_lora_manifest


class StreamedFrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, module):
        ctx.module = module
        weight, bias = module.load_tensors()
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        weight, _ = ctx.module.load_tensors()
        return grad_output.matmul(weight), None


class StreamedFrozenLinear(nn.Module):
    def __init__(self, source: nn.Linear, path: Path) -> None:
        super().__init__()
        if any(parameter.requires_grad for parameter in source.parameters()):
            raise ValueError("streamed frozen linear source must not be trainable")
        tensors = {"weight": source.weight.detach().cpu().contiguous()}
        if source.bias is not None:
            tensors["bias"] = source.bias.detach().cpu().contiguous()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, path)
        self.path = path
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.read_bytes = 0
        self.read_count = 0
        self.artifact_bytes = path.stat().st_size

    def load_tensors(self):
        mapped = load_file(self.path, device="cpu")
        tensors = {
            name: tensor.clone().contiguous() for name, tensor in mapped.items()
        }
        del mapped
        self.read_bytes += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        self.read_count += 1
        return tensors["weight"], tensors.get("bias")

    def forward(self, inputs):
        if inputs.dtype != torch.float32:
            raise ValueError("streamed frozen-linear profile requires FP32 activations")
        return StreamedFrozenLinearFunction.apply(inputs, self)


def stream_frozen_linears(module: nn.Module, root: Path, prefix: str = "model") -> int:
    count = 0
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}"
        if isinstance(child, peft.LoRALinear):
            if not isinstance(child.base, nn.Linear):
                raise ValueError("unexpected LoRA base")
            child.base = StreamedFrozenLinear(
                child.base, root / f"{child_prefix}.base.safetensors"
            )
            count += 1
        elif isinstance(child, nn.Linear):
            setattr(
                module,
                child_name,
                StreamedFrozenLinear(child, root / f"{child_prefix}.safetensors"),
            )
            count += 1
        else:
            count += stream_frozen_linears(child, root, child_prefix)
    return count


def tensor_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += tensor.untyped_storage().nbytes()
    return total


def gradient_metrics(reference, candidate) -> dict[str, float]:
    names = list(reference)
    ref = torch.cat([reference[name].reshape(-1).double() for name in names])
    got = torch.cat([candidate[name].reshape(-1).double() for name in names])
    delta = got - ref
    return {
        "cosine_similarity": float(F.cosine_similarity(ref, got, dim=0)),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref)
        ),
        "max_absolute_error": float(delta.abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FP32-resident and exact-FP32 streamed frozen linears."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.storage.exists():
        raise SystemExit(f"storage path exists: {args.storage}")

    loaded = load_lora_manifest(args.campaign, args.lora)
    campaign = loaded.campaign
    tokens = campaign.training.batch_size * campaign.model.context_length
    inputs = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        campaign.training.active_vocabulary_size
    ).reshape(campaign.training.batch_size, campaign.model.context_length)
    targets = (inputs + 1).remainder(campaign.training.active_vocabulary_size)

    fp32_model = peft.build_lora_model(campaign, loaded.config)
    fp32_resident = tensor_bytes(fp32_model)
    started = time.perf_counter()
    fp32 = compute_adapter_gradients(fp32_model, inputs, targets)
    fp32_seconds = time.perf_counter() - started

    streamed_model = peft.build_lora_model(campaign, loaded.config)
    linear_count = stream_frozen_linears(streamed_model, args.storage)
    streamed_resident = tensor_bytes(streamed_model)
    started = time.perf_counter()
    streamed = compute_adapter_gradients(streamed_model, inputs, targets)
    streamed_seconds = time.perf_counter() - started
    modules = [
        module
        for module in streamed_model.modules()
        if isinstance(module, StreamedFrozenLinear)
    ]
    result = {
        "format": "orcacolony_streamed_fp32_frozen_linear_spike_v1",
        "campaign_id": campaign.campaign["id"],
        "parameter_count": campaign.model.parameters,
        "streamed_linear_count": linear_count,
        "fp32_resident_tensor_bytes": fp32_resident,
        "streamed_resident_tensor_bytes": streamed_resident,
        "resident_tensor_reduction_ratio": 1.0 - streamed_resident / fp32_resident,
        "streamed_artifact_bytes": sum(module.artifact_bytes for module in modules),
        "streamed_read_bytes": sum(module.read_bytes for module in modules),
        "streamed_read_count": sum(module.read_count for module in modules),
        "fp32_gradient_seconds": fp32_seconds,
        "streamed_gradient_seconds": streamed_seconds,
        "runtime_ratio": streamed_seconds / fp32_seconds,
        "fp32_loss_sum": fp32.loss_sum,
        "streamed_loss_sum": streamed.loss_sum,
        "gradient_metrics": gradient_metrics(fp32.gradients, streamed.gradients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
