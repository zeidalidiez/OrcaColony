from __future__ import annotations

import json
import multiprocessing
import sys
import time

import torch

import orcacolony.tile_process as tile_process
from orcacolony.reference import load_campaign


def stall_without_reading(connection: object) -> None:
    time.sleep(5.0)
    connection.close()  # type: ignore[attr-defined]


def stall_before_forward_input(connection: object) -> None:
    init = json.loads(connection.recv_bytes())  # type: ignore[attr-defined]
    connection.send_bytes(b'{"status":"ready_for_model"}')  # type: ignore[attr-defined]
    model_wire = connection.recv_bytes()  # type: ignore[attr-defined]
    state = tile_process._deserialize_tensors(model_wire)
    current, peak = tile_process._process_memory_bytes()
    tile_process._send_json(
        connection,  # type: ignore[arg-type]
        {
            "status": "ready",
            "tile_state_sha256": tile_process.tensor_sha256(state),
            "startup_current_rss_bytes": current,
            "startup_peak_rss_bytes": peak,
            "after_model_current_rss_bytes": current,
            "after_model_peak_rss_bytes": peak,
        },
    )
    control = json.loads(connection.recv_bytes())  # type: ignore[attr-defined]
    assert init["op"] == "init" and control["op"] == "forward"
    time.sleep(5.0)
    connection.close()  # type: ignore[attr-defined]


def oversized_prefix_activation(*_args: object) -> torch.Tensor:
    return torch.zeros(
        (2, 4096, 128),
        dtype=torch.float32,
        requires_grad=True,
    )


if __name__ == "__main__":
    phase = sys.argv[1]
    tile_process._tile_worker_entry = (
        stall_without_reading
        if phase == "initialization"
        else stall_before_forward_input
    )
    if phase == "forward":
        tile_process._prefix_activation = oversized_prefix_activation
    campaign = load_campaign("campaign/t0-smoke.json")
    started = time.perf_counter()
    try:
        tile_process.run_persistent_tile_process_experiment(
            campaign,
            block_index=2,
            timeout_seconds=0.05,
        )
    except BaseException as exc:
        print(f"exception={type(exc).__name__}", flush=True)
        print(f"elapsed_seconds={time.perf_counter() - started:.6f}", flush=True)
        print(f"active_children={len(multiprocessing.active_children())}", flush=True)
