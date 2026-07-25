from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from orcacolony import peft
from orcacolony.peft import compute_adapter_gradients, load_lora_manifest


class Int8FrozenLinearFunction(torch.autograd.Function):
    """Rebuild FP32 weights in forward/backward instead of saving them in autograd."""

    @staticmethod
    def forward(ctx, inputs, qweight, scales, bias):
        ctx.save_for_backward(qweight, scales)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            weight = qweight.to(dtype=torch.float32) * scales[:, None].float()
            return F.linear(inputs.float(), weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            weight = qweight.to(dtype=torch.float32) * scales[:, None].float()
            return grad_output.float().matmul(weight), None, None, None


class Int8FrozenLinear(nn.Module):
    """Per-output-channel symmetric int8 weight with an FP32 scale and bias."""

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().to(dtype=torch.float32)
        scales = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
        qweight = torch.round(weight / scales[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().to(dtype=torch.float32).clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return Int8FrozenLinearFunction.apply(inputs, self.qweight, self.scales, self.bias)


def quantize_frozen_linears(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, peft.LoRALinear):
            if not isinstance(child.base, nn.Linear):
                raise TypeError(f"unexpected LoRA base: {type(child.base)!r}")
            child.base = Int8FrozenLinear(child.base)
            count += 1
        elif isinstance(child, nn.Linear):
            if any(parameter.requires_grad for parameter in child.parameters()):
                raise ValueError(f"refusing to quantize trainable linear {name}")
            setattr(module, name, Int8FrozenLinear(child))
            count += 1
        else:
            count += quantize_frozen_linears(child)
    return count


def tensor_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += tensor.untyped_storage().nbytes()
    return total


def gradient_metrics(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, float]:
    reference_flat = torch.cat(
        [reference[name].reshape(-1).double() for name in reference]
    )
    candidate_flat = torch.cat(
        [candidate[name].reshape(-1).double() for name in reference]
    )
    delta = candidate_flat - reference_flat
    return {
        "cosine_similarity": float(
            F.cosine_similarity(reference_flat, candidate_flat, dim=0)
        ),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_flat)
        ),
        "max_absolute_error": float(delta.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FP32 and per-row int8 frozen-linear LoRA gradients."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = load_lora_manifest(args.campaign, args.lora)
    campaign = loaded.campaign
    tokens = campaign.training.batch_size * campaign.model.context_length
    input_ids = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        campaign.training.active_vocabulary_size
    ).reshape(campaign.training.batch_size, campaign.model.context_length)
    target_ids = (input_ids + 1).remainder(
        campaign.training.active_vocabulary_size
    )

    fp32_model = peft.build_lora_model(loaded.campaign, loaded.config)
    fp32_bytes = tensor_bytes(fp32_model)
    fp32 = compute_adapter_gradients(fp32_model, input_ids, target_ids)

    int8_model = peft.build_lora_model(loaded.campaign, loaded.config)
    quantized_linear_count = quantize_frozen_linears(int8_model)
    int8_bytes = tensor_bytes(int8_model)
    int8 = compute_adapter_gradients(int8_model, input_ids, target_ids)

    result = {
        "format": "orcacolony_int8_frozen_linear_spike_v1",
        "campaign_id": campaign.campaign["id"],
        "parameter_count": campaign.model.parameters,
        "adapter_value_count": sum(
            tensor.numel()
            for tensor in peft.adapter_named_parameters(fp32_model).values()
        ),
        "quantized_linear_count": quantized_linear_count,
        "fp32_resident_tensor_bytes": fp32_bytes,
        "int8_resident_tensor_bytes": int8_bytes,
        "resident_tensor_reduction_ratio": 1.0 - int8_bytes / fp32_bytes,
        "fp32_loss_sum": fp32.loss_sum,
        "int8_loss_sum": int8.loss_sum,
        "relative_loss_sum_error": abs(int8.loss_sum - fp32.loss_sum)
        / abs(fp32.loss_sum),
        "gradient_metrics": gradient_metrics(fp32.gradients, int8.gradients),
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
