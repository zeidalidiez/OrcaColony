from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from orcacolony.peft import (
    DirectStreamedFrozenLinear,
    adapter_named_parameters,
    build_direct_streamed_lora_model,
    build_lora_model,
    compute_adapter_gradients,
    load_adapter_state,
    load_lora_manifest,
)
from orcacolony.reference import tensor_sha256


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory() -> tuple[int, int]:
    if os.name != "nt":
        raise RuntimeError("this bounded startup proof currently measures Windows RSS")
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def resident_tensor_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += tensor.untyped_storage().nbytes()
    return total


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure isolated full-resident or direct-streamed T2 startup."
    )
    parser.add_argument("--mode", choices=("resident", "direct"), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = load_lora_manifest(args.campaign, args.lora)
    adapter_state = load_file(args.adapter, device="cpu")
    initial_rss, _ = process_memory()
    base_file_sha256 = file_sha256(args.base)
    started = time.perf_counter()
    if args.mode == "resident":
        model = build_lora_model(loaded.campaign, loaded.config)
        load_adapter_state(model, adapter_state)
    else:
        model = build_direct_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            args.base,
            base_file_sha256,
            adapter_state,
        )
    build_seconds = time.perf_counter() - started
    rss_after_build, _ = process_memory()

    tokens = loaded.campaign.training.batch_size * loaded.campaign.model.context_length
    inputs = (torch.arange(tokens, dtype=torch.long) * 17 + 3).remainder(
        loaded.campaign.training.active_vocabulary_size
    ).reshape(
        loaded.campaign.training.batch_size,
        loaded.campaign.model.context_length,
    )
    targets = (inputs + 1).remainder(
        loaded.campaign.training.active_vocabulary_size
    )
    gradient_started = time.perf_counter()
    result = compute_adapter_gradients(model, inputs, targets)
    gradient_seconds = time.perf_counter() - gradient_started
    direct_linears = [
        module
        for module in model.modules()
        if isinstance(module, DirectStreamedFrozenLinear)
    ]
    rss_after_gradient, peak_rss = process_memory()
    payload = {
        "format": "orcacolony_direct_streamed_startup_proof_v1",
        "mode": args.mode,
        "campaign_id": loaded.campaign.campaign["id"],
        "parameter_count": loaded.campaign.model.parameters,
        "base_file_sha256": base_file_sha256,
        "base_file_bytes": args.base.stat().st_size,
        "initial_rss_bytes": initial_rss,
        "build_seconds": build_seconds,
        "rss_after_build_bytes": rss_after_build,
        "resident_tensor_bytes": resident_tensor_bytes(model),
        "gradient_seconds": gradient_seconds,
        "rss_after_gradient_bytes": rss_after_gradient,
        "peak_rss_bytes": peak_rss,
        "loss_sum": result.loss_sum,
        "loss_weight_sum": result.loss_weight_sum,
        "gradient_sha256": tensor_sha256(result.gradients),
        "adapter_value_count": sum(
            tensor.numel() for tensor in adapter_named_parameters(model).values()
        ),
        "direct_linear_count": len(direct_linears),
        "streamed_read_bytes": sum(module.read_bytes for module in direct_linears),
        "streamed_read_count": sum(module.read_count for module in direct_linears),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
