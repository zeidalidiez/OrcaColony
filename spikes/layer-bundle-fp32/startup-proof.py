from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from orcacolony.peft import (
    DirectStreamedFrozenLinear,
    LayerBundleStreamedFrozenLinear,
    adapter_named_parameters,
    build_direct_streamed_lora_model,
    build_layer_bundle_streamed_lora_model,
    build_lora_model,
    compute_adapter_gradients,
    load_lora_manifest,
)
from orcacolony.reference import tensor_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure isolated resident/direct/layer-bundle FP32 startup."
    )
    parser.add_argument("--mode", choices=("resident", "direct", "bundle"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lora-config", required=True, type=Path)
    parser.add_argument("--base-artifact", required=True, type=Path)
    parser.add_argument("--base-artifact-sha256", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-manifest-sha256")
    parser.add_argument("--expected-gradient-sha256")
    parser.add_argument("--expected-loss-sum", type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def retained_tensor_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in [*model.parameters(), *model.buffers()]:
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += storage.nbytes()
    return total


def main() -> None:
    args = parse_args()
    if args.mode == "bundle" and (
        args.bundle is None or args.bundle_manifest_sha256 is None
    ):
        raise ValueError("bundle mode requires --bundle and --bundle-manifest-sha256")
    if args.mode != "bundle" and (
        args.bundle is not None or args.bundle_manifest_sha256 is not None
    ):
        raise ValueError("bundle arguments are only valid in bundle mode")
    if sha256_file(args.adapter) != args.adapter_sha256:
        raise ValueError("adapter artifact SHA-256 mismatch")

    loaded = load_lora_manifest(
        args.config,
        args.lora_config,
        verify_base_model=False,
    )
    adapter_state = {
        name: tensor.clone().contiguous()
        for name, tensor in load_file(args.adapter, device="cpu").items()
    }
    if tensor_sha256(adapter_state) != args.adapter_sha256:
        raise ValueError("adapter canonical tensor identity mismatch")

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
    initial_rss, _ = process_memory()
    build_started = time.perf_counter()
    if args.mode == "resident":
        model = build_lora_model(loaded.campaign, loaded.config)
        if tensor_sha256(adapter_state) != args.adapter_sha256:
            raise ValueError("adapter state changed before resident load")
        from orcacolony.peft import load_adapter_state

        load_adapter_state(model, adapter_state)
    elif args.mode == "direct":
        model = build_direct_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            args.base_artifact,
            args.base_artifact_sha256,
            adapter_state,
        )
    else:
        assert args.bundle is not None
        assert args.bundle_manifest_sha256 is not None
        model = build_layer_bundle_streamed_lora_model(
            loaded.campaign,
            loaded.config,
            args.bundle,
            args.bundle_manifest_sha256,
            adapter_state,
        )
    build_seconds = time.perf_counter() - build_started
    rss_after_build, peak_after_build = process_memory()

    direct_linears = [
        module
        for module in model.modules()
        if isinstance(module, DirectStreamedFrozenLinear)
    ]
    bundle_linears = [
        module
        for module in model.modules()
        if isinstance(module, LayerBundleStreamedFrozenLinear)
    ]
    startup_streamed_read_count = sum(
        module.read_count for module in [*direct_linears, *bundle_linears]
    )
    startup_streamed_read_bytes = sum(
        module.read_bytes for module in [*direct_linears, *bundle_linears]
    )
    if args.mode in {"direct", "bundle"}:
        selected = direct_linears if args.mode == "direct" else bundle_linears
        expected_linear_count = loaded.campaign.model.layers * 4
        if len(selected) != expected_linear_count:
            raise ValueError("streamed profile did not replace every model linear")
        if any(isinstance(module, torch.nn.Linear) for module in model.modules()):
            raise ValueError("streamed T2 profile retained an original linear")
        if startup_streamed_read_count != 0 or startup_streamed_read_bytes != 0:
            raise ValueError("streamed T2 profile read a linear during startup")

    retained_bytes = retained_tensor_bytes(model)
    gradient_started = time.perf_counter()
    gradients = compute_adapter_gradients(model, inputs, targets)
    gradient_seconds = time.perf_counter() - gradient_started
    rss_after_gradient, final_peak_rss = process_memory()
    streamed_linears = [*direct_linears, *bundle_linears]

    if (
        args.expected_gradient_sha256 is not None
        and gradients.gradient_sha256 != args.expected_gradient_sha256
    ):
        raise ValueError("adapter gradient SHA-256 differs from the expected oracle")
    if (
        args.expected_loss_sum is not None
        and gradients.loss_sum != args.expected_loss_sum
    ):
        raise ValueError("loss sum differs from the expected oracle")

    bundle_artifact_bytes = None
    bundle_file_count = None
    bundle_resident_file_bytes = None
    if args.mode == "bundle":
        assert args.bundle is not None
        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        bundle_artifact_bytes = sum(
            int(path.stat().st_size)
            for path in args.bundle.iterdir()
            if path.suffix == ".safetensors"
        )
        bundle_file_count = sum(1 for path in args.bundle.iterdir())
        bundle_resident_file_bytes = int(manifest["resident"]["bytes"])

    result = {
        "format": "orcacolony_layer_bundle_startup_proof_v1",
        "mode": args.mode,
        "campaign_id": loaded.campaign.campaign["id"],
        "parameter_count": loaded.campaign.model.parameters,
        "base_model_sha256": loaded.config.base_model_sha256,
        "base_artifact_sha256": args.base_artifact_sha256,
        "base_artifact_bytes": args.base_artifact.stat().st_size,
        "adapter_sha256": args.adapter_sha256,
        "adapter_value_count": sum(tensor.numel() for tensor in adapter_state.values()),
        "bundle_manifest_sha256": args.bundle_manifest_sha256,
        "bundle_artifact_bytes": bundle_artifact_bytes,
        "bundle_file_count": bundle_file_count,
        "bundle_resident_file_bytes": bundle_resident_file_bytes,
        "streamed_linear_count": len(streamed_linears),
        "startup_streamed_read_count": startup_streamed_read_count,
        "startup_streamed_read_bytes": startup_streamed_read_bytes,
        "retained_tensor_bytes": retained_bytes,
        "initial_rss_bytes": initial_rss,
        "rss_after_build_bytes": rss_after_build,
        "peak_after_build_bytes": peak_after_build,
        "rss_after_gradient_bytes": rss_after_gradient,
        "peak_rss_bytes": final_peak_rss,
        "build_seconds": build_seconds,
        "gradient_seconds": gradient_seconds,
        "streamed_read_count": sum(module.read_count for module in streamed_linears),
        "streamed_read_bytes": sum(module.read_bytes for module in streamed_linears),
        "loss_sum": gradients.loss_sum,
        "loss_weight_sum": gradients.loss_weight_sum,
        "gradient_sha256": gradients.gradient_sha256,
        "adapter_parameter_count": len(adapter_named_parameters(model)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
