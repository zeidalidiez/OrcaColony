from __future__ import annotations

import multiprocessing
import time

import orcacolony.full_process as full_process
from orcacolony.reference import load_campaign


def stall_without_reading(connection: object) -> None:
    time.sleep(5.0)
    connection.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    full_process._full_worker_entry = stall_without_reading
    campaign = load_campaign("campaign/t0-smoke.json")
    started = time.perf_counter()
    try:
        full_process.run_persistent_full_process_control(
            campaign,
            timeout_seconds=0.05,
        )
    except BaseException as exc:
        print(f"exception={type(exc).__name__}", flush=True)
        print(f"elapsed_seconds={time.perf_counter() - started:.6f}", flush=True)
        print(f"active_children={len(multiprocessing.active_children())}", flush=True)
