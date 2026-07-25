from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save as save_safetensors

from .multiworker import normalize_http_origin
from .peft import (
    LoadedLoRAManifest,
    adapter_named_parameters,
    build_lora_model,
    compute_adapter_gradients,
    load_adapter_state,
    load_lora_manifest,
    lora_weight_checkpoint_sha256,
)
from .reference import tensor_sha256


_MAX_JSON_BYTES = 64 * 1024 * 1024
_USER_AGENT = "OrcaColony/0.1 native-cpu-worker"


@dataclass(frozen=True)
class NativeWorkerResult:
    assignment_id: str
    receipt: Mapping[str, object]
    telemetry: Mapping[str, object]


def _origin(value: str) -> tuple[str, str, int]:
    parsed = urlsplit(value)
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, parsed.hostname or "", parsed.port or default_port


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    def __init__(self, coordinator_origin: str) -> None:
        super().__init__()
        self.coordinator_origin = coordinator_origin

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if _origin(newurl) != _origin(self.coordinator_origin):
            raise ValueError("coordinator redirect crossed the pinned origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _coordinator_url(origin: str, path: object) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("coordinator artifact URL must be an absolute path")
    url = urljoin(f"{origin}/", path.removeprefix("/"))
    if _origin(url) != _origin(origin):
        raise ValueError("coordinator artifact URL crossed the pinned origin")
    return url


def _read_response(
    opener: Any,
    request: Request,
    *,
    maximum_bytes: int,
) -> bytes:
    with opener.open(request, timeout=180) as response:
        if _origin(response.geturl()) != _origin(request.full_url):
            raise ValueError("coordinator response crossed the pinned origin")
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("coordinator response exceeds the declared byte limit")
    return payload


def _json_response(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or attributes & reparse_point
    ):
        raise ValueError("native worker cache contains a symlink or reparse point")
    return int(metadata.st_dev), int(metadata.st_ino)


def _prepare_cache_directory(cache_dir: Path, kind: str) -> tuple[Path, tuple[int, int]]:
    if kind not in {"model", "adapter"}:
        raise ValueError("native worker cache kind is invalid")
    absolute = Path(os.path.abspath(cache_dir))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            current.mkdir()
        except FileExistsError:
            pass
        _directory_identity(current)
    target_dir = absolute / kind
    try:
        target_dir.mkdir()
    except FileExistsError:
        pass
    return target_dir, _directory_identity(target_dir)


def _validate_tensor_cache(path: Path, digest: str, expected_bytes: int) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or attributes & reparse_point
        or metadata.st_size != expected_bytes
    ):
        return False
    try:
        return tensor_sha256(load_safetensors_file(str(path))) == digest
    except Exception:
        return False


def _cached_artifact(
    *,
    opener: Any,
    origin: str,
    cache_dir: Path,
    kind: str,
    digest: object,
    artifact_url: object,
    expected_bytes: object,
) -> tuple[Path, int]:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{kind} tensor digest is invalid")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError(f"{kind} artifact byte count is invalid")
    target_dir, directory_identity = _prepare_cache_directory(cache_dir, kind)
    target = target_dir / f"{digest}.safetensors"
    if _validate_tensor_cache(target, digest, expected_bytes):
        if _directory_identity(target_dir) != directory_identity:
            raise ValueError("native worker cache directory changed during validation")
        return target, 0
    target.unlink(missing_ok=True)
    request = Request(
        _coordinator_url(origin, artifact_url),
        headers={"User-Agent": _USER_AGENT},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_dir,
        prefix=f".{digest}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    downloaded = 0
    try:
        with os.fdopen(descriptor, "wb") as output, opener.open(
            request,
            timeout=180,
        ) as response:
            if _origin(response.geturl()) != _origin(origin):
                raise ValueError("artifact response crossed the pinned origin")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise ValueError(f"{kind} artifact length does not match assignment")
            while chunk := response.read(min(1024 * 1024, expected_bytes - downloaded + 1)):
                downloaded += len(chunk)
                if downloaded > expected_bytes:
                    raise ValueError(f"{kind} artifact exceeds the assigned byte count")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if downloaded != expected_bytes:
            raise ValueError(f"{kind} artifact length does not match assignment")
        if not _validate_tensor_cache(temporary, digest, expected_bytes):
            raise ValueError(f"{kind} artifact tensor digest mismatch")
        if _directory_identity(target_dir) != directory_identity:
            raise ValueError("native worker cache directory changed during download")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, downloaded


def _load_frozen_base(
    loaded: LoadedLoRAManifest,
    model: torch.nn.Module,
    base_path: Path,
) -> None:
    base_tensors = load_safetensors_file(str(base_path))
    state = model.state_dict()
    targets = set(loaded.config.targets)
    for name, tensor in base_tensors.items():
        mapped = name
        parent, separator, child = name.rpartition(".")
        if separator and parent in targets and child in {"weight", "bias"}:
            mapped = f"{parent}.base.{child}"
        if mapped not in state or state[mapped].shape != tensor.shape:
            raise ValueError(f"frozen base tensor does not match native model: {name}")
        state[mapped] = tensor
    model.load_state_dict(state, strict=True)


def _peak_process_rss_bytes() -> int | None:
    if os.name == "nt":
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
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, ValueError):
        return None


def _device_capacity_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    if os.name != "nt" and hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            return None
    return None


def _validate_assignment(
    assignment: Mapping[str, object],
    loaded: LoadedLoRAManifest,
) -> None:
    if assignment.get("campaign_id") != loaded.campaign.campaign["id"]:
        raise ValueError("assignment campaign does not match native worker configuration")
    if assignment.get("training_method") != "frozen-base-lora":
        raise ValueError("native worker currently requires frozen-base LoRA assignments")
    if assignment.get("lora_manifest_sha256") != loaded.manifest_sha256:
        raise ValueError("assignment LoRA manifest digest mismatch")
    if assignment.get("base_model_sha256") != loaded.config.base_model_sha256:
        raise ValueError("assignment frozen-base digest mismatch")
    adapter = assignment.get("adapter")
    if not isinstance(adapter, Mapping):
        raise ValueError("assignment adapter contract is missing")
    if (
        adapter.get("rank") != loaded.config.rank
        or float(adapter.get("alpha", 0)) != float(loaded.config.alpha)
        or tuple(adapter.get("targets", ())) != loaded.config.targets
    ):
        raise ValueError("assignment adapter contract does not match configuration")
    resources = assignment.get("resource_profile")
    if not isinstance(resources, Mapping) or resources.get("format") != (
        "orcacolony_assignment_resources_v1"
    ):
        raise ValueError("assignment resource profile is invalid")


def run_assignment(
    *,
    coordinator_url: str,
    worker_id: str,
    worker_token: str,
    campaign_path: str | Path,
    lora_path: str | Path,
    cache_dir: str | Path,
) -> NativeWorkerResult:
    origin = normalize_http_origin(coordinator_url)
    if not worker_id or not worker_token:
        raise ValueError("worker ID and token must not be empty")
    loaded = load_lora_manifest(campaign_path, lora_path)
    opener = build_opener(_SameOriginRedirectHandler(origin))

    assignment_started = time.perf_counter()
    assignment_request = Request(
        f"{origin}/api/v1/assignment?{urlencode({'worker_id': worker_id})}",
        headers={
            "User-Agent": _USER_AGENT,
            "X-Orca-Worker-Token": worker_token,
        },
    )
    assignment_body = _read_response(
        opener,
        assignment_request,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    assignment_seconds = time.perf_counter() - assignment_started
    assignment = _json_response(assignment_body, "assignment response")
    _validate_assignment(assignment, loaded)
    resources = assignment["resource_profile"]

    artifact_started = time.perf_counter()
    base_path, model_network_bytes = _cached_artifact(
        opener=opener,
        origin=origin,
        cache_dir=Path(cache_dir),
        kind="model",
        digest=assignment["base_model_sha256"],
        artifact_url=assignment["model_url"],
        expected_bytes=resources["model_download_bytes"],
    )
    adapter_path, adapter_network_bytes = _cached_artifact(
        opener=opener,
        origin=origin,
        cache_dir=Path(cache_dir),
        kind="adapter",
        digest=assignment["adapter_sha256"],
        artifact_url=assignment["adapter_url"],
        expected_bytes=resources["adapter_download_bytes"],
    )
    artifact_seconds = time.perf_counter() - artifact_started

    runtime_started = time.perf_counter()
    model = build_lora_model(loaded.campaign, loaded.config)
    _load_frozen_base(loaded, model, base_path)
    adapter_tensors = load_safetensors_file(str(adapter_path))
    load_adapter_state(model, adapter_tensors)
    expected_weight_identity = lora_weight_checkpoint_sha256(
        loaded,
        tensor_sha256(adapter_tensors),
    )
    if assignment.get("weight_checkpoint_sha256") != expected_weight_identity:
        raise ValueError("assignment worker-weight identity mismatch")
    runtime_seconds = time.perf_counter() - runtime_started

    shape = assignment.get("input_shape")
    input_ids = assignment.get("input_ids")
    target_ids = assignment.get("target_ids")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(input_ids, list)
        or not isinstance(target_ids, list)
        or len(input_ids) != shape[0] * shape[1]
        or len(target_ids) != shape[0] * shape[1]
    ):
        raise ValueError("assignment token tensors are invalid")
    inputs = torch.tensor(input_ids, dtype=torch.long).reshape(shape)
    targets = torch.tensor(target_ids, dtype=torch.long).reshape(shape)
    gradient_started = time.perf_counter()
    gradient_result = compute_adapter_gradients(model, inputs, targets)
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_bytes = save_safetensors(dict(gradient_result.gradients))
    if len(gradient_bytes) != resources["expected_result_upload_bytes"]:
        raise ValueError("native gradient byte count does not match assignment")

    telemetry: dict[str, object] = {
        "format": "orcacolony_worker_telemetry_v1",
        "runtime_seconds": {
            "assignment_fetch": assignment_seconds,
            "runtime_init": runtime_seconds,
            "artifact_fetch": artifact_seconds,
            "gradient_compute": gradient_seconds,
        },
        "transfer_bytes": {
            "assignment": len(assignment_body),
            "model": model_network_bytes,
            "adapter": adapter_network_bytes,
            "oracle_gradient": 0,
            "result": len(gradient_bytes),
        },
        "memory_bytes": {
            "wasm_linear": None,
            "process_peak_rss": _peak_process_rss_bytes(),
            "js_heap_used": None,
            "js_heap_limit": None,
            "device_capacity": _device_capacity_bytes(),
        },
    }
    result_request = Request(
        _coordinator_url(origin, assignment["result_url"]),
        data=gradient_bytes,
        method="POST",
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/octet-stream",
            "X-Orca-Lease-Token": str(assignment["lease_token"]),
            "X-Orca-Checkpoint-Sha256": str(assignment["checkpoint_sha256"]),
            "X-Orca-Loss-Sum": str(gradient_result.loss_sum),
            "X-Orca-Loss-Weight-Sum": str(gradient_result.loss_weight_sum),
            "X-Orca-Runtime-Backend": "python-native-cpu-f32",
            "X-Orca-Worker-Telemetry": json.dumps(
                telemetry,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    )
    receipt_body = _read_response(
        opener,
        result_request,
        maximum_bytes=1024 * 1024,
    )
    receipt = _json_response(receipt_body, "result receipt")
    if receipt.get("accepted") is not True:
        raise ValueError("coordinator did not accept native worker result")
    return NativeWorkerResult(
        assignment_id=str(assignment["assignment_id"]),
        receipt=receipt,
        telemetry=telemetry,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one authenticated cached-base native CPU LoRA assignment"
    )
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-token-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lora-config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    token = args.worker_token_file.read_text(encoding="utf-8").strip()
    result = run_assignment(
        coordinator_url=args.coordinator,
        worker_id=args.worker_id,
        worker_token=token,
        campaign_path=args.config,
        lora_path=args.lora_config,
        cache_dir=args.cache,
    )
    print(
        json.dumps(
            {
                "assignment_id": result.assignment_id,
                "receipt": result.receipt,
                "telemetry": result.telemetry,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
