from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from orcacolony.peft import (
    LayerBundleInt8FrozenLinear,
    adapter_named_parameters,
    build_int8_lora_model,
    build_layer_bundle_int8_lora_model,
    compute_adapter_gradients,
    load_adapter_state,
    load_lora_manifest,
)
from orcacolony.reference import tensor_sha256


def process_memory() -> tuple[int, int]:
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

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    process = get_current_process()
    if not get_process_memory_info(
        process, ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def tensor_bytes(model: torch.nn.Module) -> int:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare convert-after-resident and direct layer-bundle int8 startup."
    )
    parser.add_argument("--mode", choices=("converted", "bundle"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_lora_manifest(
        args.config,
        args.lora_config,
        verify_base_model=False,
    )
    adapter_state = {
        name: tensor.clone().contiguous()
        for name, tensor in load_file(str(args.adapter), device="cpu").items()
    }
    if tensor_sha256(adapter_state) != args.adapter_sha256:
        raise ValueError("adapter tensor identity differs")
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
    gc.collect()
    initial_rss, initial_peak = process_memory()

    started = time.perf_counter()
    if args.mode == "converted":
        model = build_int8_lora_model(loaded.campaign, loaded.config)
        load_adapter_state(model, adapter_state)
    else:
        if args.bundle is None or args.bundle_manifest_sha256 is None:
            raise ValueError("bundle mode requires bundle path and manifest identity")
        model = build_layer_bundle_int8_lora_model(
            loaded.campaign,
            loaded.config,
            args.bundle,
            args.bundle_manifest_sha256,
            adapter_state,
        )
    build_seconds = time.perf_counter() - started
    rss_after_build, peak_after_build = process_memory()
    retained = tensor_bytes(model)
    direct_linears = [
        module
        for module in model.modules()
        if isinstance(module, LayerBundleInt8FrozenLinear)
    ]
    expected_linear_count = loaded.campaign.model.layers * 4
    if args.mode == "bundle" and len(direct_linears) != expected_linear_count:
        raise ValueError("direct int8 linear count differs")

    started = time.perf_counter()
    gradient = compute_adapter_gradients(model, inputs, targets)
    gradient_seconds = time.perf_counter() - started
    rss_after_gradient, peak_rss = process_memory()
    result = {
        "format": "orcacolony_direct_int8_startup_proof_v1",
        "mode": args.mode,
        "campaign_id": loaded.campaign.campaign["id"],
        "parameter_count": loaded.campaign.model.parameters,
        "base_model_sha256": loaded.config.base_model_sha256,
        "adapter_sha256": args.adapter_sha256,
        "adapter_parameter_count": len(adapter_named_parameters(model)),
        "retained_tensor_bytes": retained,
        "initial_rss_bytes": initial_rss,
        "initial_peak_rss_bytes": initial_peak,
        "rss_after_build_bytes": rss_after_build,
        "peak_after_build_bytes": peak_after_build,
        "rss_after_gradient_bytes": rss_after_gradient,
        "peak_rss_bytes": peak_rss,
        "build_seconds": build_seconds,
        "gradient_seconds": gradient_seconds,
        "loss_sum": gradient.loss_sum,
        "loss_weight_sum": gradient.loss_weight_sum,
        "gradient_sha256": gradient.gradient_sha256,
        "bundle_manifest_sha256": args.bundle_manifest_sha256,
        "bundle_artifact_open_count": sum(
            module.artifact_open_count for module in direct_linears
        ),
        "bundle_artifact_read_bytes": sum(
            module.artifact_read_bytes for module in direct_linears
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("this proof currently records Windows working-set counters")
    main()
