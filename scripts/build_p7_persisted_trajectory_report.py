#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED_AT = "2026-07-28T08:05:51Z"
IMPLEMENTATION_REVISION = "390b66bf2c5d67c726be4ad9ee77e0d3611f56c4"
PRIMARY_EVIDENCE = "reports/evidence/p7-persisted-trajectory-t1.json"
REPEAT_EVIDENCE = "reports/evidence/p7-persisted-trajectory-t1-repeat.json"
STUDY_EVIDENCE = (
    "research/studies/p7-persisted-trajectory-t1-v1/"
    "evidence/t1-persisted-sparse-trajectory.json"
)
OUTPUT = "reports/artifacts/p7-persisted-trajectory-t1-report.json"
EXPECTED_SHA256 = {
    PRIMARY_EVIDENCE: (
        "17292d431208f5d9078c6bacc3ffad0ce6db000355a2ce8047c60ab88756c123"
    ),
    REPEAT_EVIDENCE: (
        "d3220db012020b17cd8e4cc72cd413b608bc3b1355d145f6a0b732e9689e912c"
    ),
}

# Supporting chart map for final-context QA.
CHART_MAP = {
    "section": "The exact path is slower in both runs",
    "question": "How does complete three-step elapsed time compare by topology?",
    "family": "comparison",
    "type": "grouped vertical bar",
    "fields": {
        "x": "topology",
        "y": "complete_seconds",
        "color": "run_label",
    },
    "takeaway": (
        "The pooled expert executor is slower than the matched full process in "
        "both independent runs."
    ),
    "palette_policy": "hard two-root cap for primary versus repeat",
    "delivery": OUTPUT,
}


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json(path: str) -> dict[str, object]:
    payload = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source must be an object: {path}")
    return payload


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _query_rows(
    connection: sqlite3.Connection,
    query_path: str,
) -> list[dict[str, Any]]:
    cursor = connection.execute((ROOT / query_path).read_text(encoding="utf-8"))
    columns = tuple(description[0] for description in cursor.description)
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _query_source(
    *,
    source_id: str,
    label: str,
    path: str,
    description: str,
    metric_definitions: list[str],
    filters: list[str],
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "description": description,
            "engine": "SQLite",
            "executed_at": GENERATED_AT,
            "language": "sql",
            "sql": (ROOT / path).read_text(encoding="utf-8").rstrip(),
            "tables_used": ["report_source_documents"],
            "filters": filters,
            "metric_definitions": metric_definitions,
        },
    }


def _deterministic_projection(payload: dict[str, object]) -> dict[str, object]:
    projected = deepcopy(payload)
    for key in (
        "centralized_end_to_end_seconds",
        "expert_control_json_wire_bytes",
        "expert_process_end_to_end_seconds",
        "full_control_json_wire_bytes",
        "full_process_end_to_end_seconds",
    ):
        projected.pop(key, None)

    recovery = projected.get("coordinator_recovery")
    if isinstance(recovery, dict):
        recovery.pop("recovery_seconds", None)

    for worker_key in ("expert_workers", "full_workers"):
        workers = projected.get(worker_key)
        if not isinstance(workers, list):
            continue
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            for key in (
                "control_json_wire_bytes",
                "external_rss",
                "initialization_seconds",
                "shutdown_seconds",
            ):
                worker.pop(key, None)

    steps = projected.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in tuple(step):
                if key.endswith("_seconds") or key.endswith(
                    "_control_json_wire_bytes"
                ):
                    step.pop(key)
    return projected


def _build_sources() -> list[dict[str, object]]:
    evidence_filter = [
        f"source_path IN ({PRIMARY_EVIDENCE}, {REPEAT_EVIDENCE})"
    ]
    return [
        _query_source(
            source_id="headline_query",
            label="Persisted-trajectory headline query",
            path="reports/queries/p7-persisted-trajectory-headline.sql",
            description=(
                "Calculate exact-update coverage and primary-run traffic, "
                "persistence, elapsed-time, child-RSS, and recovery headlines."
            ),
            filters=evidence_filter,
            metric_definitions=[
                (
                    "Exact update rate is the fraction of all six run-steps where "
                    "centralized, full-process, and pooled-expert raw gradients, "
                    "clipped gradients, AdamW state, and model state all match."
                ),
                (
                    "Tensor traffic reduction = 1 - pooled-expert serialized "
                    "tensor bytes / matched full-process serialized tensor bytes."
                ),
                (
                    "Persisted-byte reduction = 1 - pooled-expert transaction "
                    "bytes / matched full-process transaction bytes."
                ),
                (
                    "Child high-water RSS reduction = 1 - maximum pooled-expert "
                    "child VmHWM / matched full child VmHWM."
                ),
            ],
        ),
        _query_source(
            source_id="timing_query",
            label="Persisted-trajectory timing query",
            path="reports/queries/p7-persisted-trajectory-timing.sql",
            description=(
                "Return complete elapsed time for centralized, matched full-process, "
                "and pooled-expert execution in both independent runs."
            ),
            filters=evidence_filter,
            metric_definitions=[
                (
                    "Complete process-path elapsed time includes coordinator "
                    "preparation, IPC, durable transaction writes, apply, recovery, "
                    "and shutdown. Centralized time has no process or persistence work."
                ),
                (
                    "Change versus full = topology elapsed seconds / matched "
                    "full-process elapsed seconds - 1 within the same run."
                ),
            ],
        ),
        _query_source(
            source_id="exactness_query",
            label="Persisted-trajectory exactness query",
            path="reports/queries/p7-persisted-trajectory-exactness.sql",
            description=(
                "Compare all six run-step rows for routing, batch loss, tensor "
                "differences, and gradient, optimizer, and model identities."
            ),
            filters=evidence_filter,
            metric_definitions=[
                (
                    "State hashes match only when centralized, full-process, and "
                    "pooled-expert raw gradient, clipped gradient, AdamW, and "
                    "complete model SHA-256 values all agree."
                ),
                (
                    "Training-batch loss is recorded on a different batch at each "
                    "step and is not a model-quality or generalization metric."
                ),
            ],
        ),
        _query_source(
            source_id="traffic_query",
            label="Persisted-trajectory traffic and storage query",
            path="reports/queries/p7-persisted-trajectory-traffic.sql",
            description=(
                "Compare primary-run serialized tensor traffic, canonical JSON "
                "traffic, and durable transaction bytes by process topology."
            ),
            filters=[f"source_path = {PRIMARY_EVIDENCE}"],
            metric_definitions=[
                (
                    "Tensor wire bytes include safetensors payloads sent and "
                    "received across the local process boundary for all three steps."
                ),
                (
                    "Control JSON wire bytes include canonical application controls "
                    "and vary slightly when timing values have different text lengths."
                ),
                (
                    "Persisted bytes include pre-state, batch, accepted results, "
                    "manifests, and applied checkpoints written for all three steps."
                ),
            ],
        ),
        _query_source(
            source_id="memory_query",
            label="Persisted-trajectory child-memory query",
            path="reports/queries/p7-persisted-trajectory-memory.sql",
            description=(
                "Aggregate externally sampled current and high-water RSS across "
                "each child generation for both topologies and both runs."
            ),
            filters=evidence_filter,
            metric_definitions=[
                (
                    "Maximum current RSS and VmHWM are sampled externally from "
                    "Linux /proc from child spawn through shutdown."
                ),
                (
                    "The values are per-child lifecycle maxima. They exclude the "
                    "coordinator and cannot be summed across sequential children."
                ),
            ],
        ),
        _query_source(
            source_id="recovery_query",
            label="Persisted-trajectory recovery query",
            path="reports/queries/p7-persisted-trajectory-recovery.sql",
            description=(
                "Report the durable worker-loss result and fresh-process "
                "coordinator recovery outcome for both independent runs."
            ),
            filters=evidence_filter,
            metric_definitions=[
                (
                    "Worker recovery passes when accepted result zero remains "
                    "durable, is not recomputed, and the replacement exits cleanly."
                ),
                (
                    "Coordinator recovery begins after the applied checkpoint "
                    "directory is published and includes fresh-process load, exact "
                    "recomputation for validation, manifest commit, and exit."
                ),
            ],
        ),
        {
            "id": "primary_evidence",
            "label": "Primary persisted-trajectory evidence",
            "path": PRIMARY_EVIDENCE,
        },
        {
            "id": "repeat_evidence",
            "label": "Repeat persisted-trajectory evidence",
            "path": REPEAT_EVIDENCE,
        },
        {
            "id": "study_evidence",
            "label": "P7 persisted-trajectory study evidence",
            "path": STUDY_EVIDENCE,
        },
        {
            "id": "implementation",
            "label": "Persisted sparse-trajectory implementation commit",
            "href": (
                "https://github.com/zeidalidiez/OrcaColony/commit/"
                f"{IMPLEMENTATION_REVISION}"
            ),
        },
        {
            "id": "t1_campaign",
            "label": "Frozen T1 systems campaign configuration",
            "path": "campaign/t1-tinystories-system-proof.json",
        },
    ]


def _build_artifact() -> dict[str, object]:
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"evidence digest mismatch for {path}: {actual} != {expected}"
            )

    primary = _load_json(PRIMARY_EVIDENCE)
    repeat = _load_json(REPEAT_EVIDENCE)
    study = _load_json(STUDY_EVIDENCE)
    if primary["campaign_revision"] != repeat["campaign_revision"]:
        raise ValueError("evidence campaign revisions differ")
    if primary["dataset_revision"] != repeat["dataset_revision"]:
        raise ValueError("evidence dataset revisions differ")
    if _deterministic_projection(primary) != _deterministic_projection(repeat):
        raise ValueError("primary and repeat semantic evidence differ")
    if study["outcome"] != "validated":
        raise ValueError("study evidence is not validated")

    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE report_source_documents "
            "(source_path TEXT PRIMARY KEY, document TEXT NOT NULL)"
        )
        for path in (PRIMARY_EVIDENCE, REPEAT_EVIDENCE):
            connection.execute(
                "INSERT INTO report_source_documents VALUES (?, ?)",
                (path, (ROOT / path).read_text(encoding="utf-8")),
            )
        headline = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-headline.sql",
        )
        timing = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-timing.sql",
        )
        exactness = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-exactness.sql",
        )
        traffic = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-traffic.sql",
        )
        memory = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-memory.sql",
        )
        recovery = _query_rows(
            connection,
            "reports/queries/p7-persisted-trajectory-recovery.sql",
        )
    finally:
        connection.close()

    if len(headline) != 1 or len(timing) != 6 or len(exactness) != 6:
        raise ValueError("headline, timing, or exactness query row count is invalid")
    if len(traffic) != 2 or len(memory) != 4 or len(recovery) != 2:
        raise ValueError("traffic, memory, or recovery query row count is invalid")
    if float(headline[0]["exact_update_rate"]) != 1.0:
        raise ValueError("trajectory evidence is not exact")
    if any(int(row["state_hashes_match"]) != 1 for row in exactness):
        raise ValueError("trajectory state hashes do not match")
    if any(
        not (
            int(row["persisted_result_survived_loss"]) == 1
            and int(row["recomputed_persisted_result"]) == 0
            and int(row["fresh_process_only_persisted_state"]) == 1
            and int(row["recovered_from_checkpoint"]) == 1
            and int(row["duplicate_apply_rejected"]) == 1
            and int(row["recovery_process_exit_code"]) == 0
        )
        for row in recovery
    ):
        raise ValueError("recovery evidence does not satisfy the report claims")

    title = "Exact persisted sparse trajectories with measured recovery costs"
    manifest: dict[str, object] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Technical findings from the exact P7 T1 persisted multi-step "
            "sparse-trajectory control."
        ),
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "exact_update_card",
                "description": (
                    "Centralized, full-process, and pooled-expert agreement across "
                    "three steps in each of two independent runs."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Exact run-step comparisons",
                        "field": "exact_update_rate",
                        "format": "percent",
                    },
                    {
                        "label": "Compared run-steps",
                        "field": "compared_run_steps",
                        "format": "number",
                    },
                    {
                        "label": "Independent runs",
                        "field": "independent_runs",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "tensor_traffic_card",
                "description": (
                    "Three-step pooled-expert serialized tensor traffic versus the "
                    "matched full-process path."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Expert tensor traffic reduction",
                        "field": "tensor_traffic_reduction",
                        "format": "percent",
                    },
                    {
                        "label": "Persisted-byte reduction",
                        "field": "persisted_byte_reduction",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "elapsed_card",
                "description": (
                    "Primary-run complete elapsed time for the pooled-expert path "
                    "relative to the matched full-process path."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Expert elapsed change",
                        "field": "expert_elapsed_change",
                        "format": "percent",
                        "signed": True,
                    },
                    {
                        "label": "Matched full",
                        "field": "full_complete_seconds",
                        "format": "number",
                        "unit": "seconds",
                    },
                    {
                        "label": "Pooled expert",
                        "field": "expert_complete_seconds",
                        "format": "number",
                        "unit": "seconds",
                    },
                ],
            },
            {
                "id": "child_hwm_card",
                "description": (
                    "Primary-run maximum child VmHWM only; coordinator memory and "
                    "concurrent colony memory are outside scope."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Expert child HWM reduction",
                        "field": "child_hwm_reduction",
                        "format": "percent",
                    },
                    {
                        "label": "Matched full child",
                        "field": "full_child_hwm_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    {
                        "label": "Maximum expert child",
                        "field": "expert_child_hwm_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "complete_timing_chart",
                "title": "Complete topology elapsed time",
                "subtitle": (
                    "Two independent three-step runs; the pooled expert executor "
                    "is slower in both."
                ),
                "type": "bar",
                "dataset": "timing",
                "sourceId": "timing_query",
                "valueFormat": "number",
                "unit": "seconds",
                "xAxisTitle": "Execution topology",
                "yAxisTitle": "Elapsed seconds",
                "encodings": {
                    "x": {
                        "field": "topology",
                        "type": "nominal",
                        "label": "Execution topology",
                    },
                    "y": {
                        "field": "complete_seconds",
                        "type": "quantitative",
                        "label": "Elapsed seconds",
                        "format": "number",
                        "unit": "seconds",
                    },
                    "color": {
                        "field": "run_label",
                        "type": "nominal",
                        "label": "Run",
                    },
                    "tooltip": [
                        {
                            "field": "change_vs_full",
                            "type": "quantitative",
                            "label": "Change versus matched full",
                            "format": "percent",
                        }
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "exactness_table",
                "title": "Sequential exactness and routing audit",
                "subtitle": (
                    "Three steps in each run; loss is a per-batch diagnostic, "
                    "not a capability metric."
                ),
                "dataset": "exactness",
                "sourceId": "exactness_query",
                "density": "dense",
                "defaultSort": {"field": "step", "direction": "asc"},
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {
                        "field": "step",
                        "label": "Step",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "cursor",
                        "label": "Cursor",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "routing_counts",
                        "label": "Routing counts",
                        "type": "text",
                    },
                    {
                        "field": "rerouted_tokens",
                        "label": "Rerouted",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "routes_changed",
                        "label": "Routes changed",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "training_batch_loss",
                        "label": "Batch loss",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "max_raw_gradient_difference",
                        "label": "Raw grad max diff",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "max_clipped_gradient_difference",
                        "label": "Clipped grad max diff",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "max_model_difference",
                        "label": "Model max diff",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "state_hashes_match",
                        "label": "State hashes match",
                        "format": "number",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "recovery_table",
                "title": "Worker and coordinator recovery audit",
                "subtitle": (
                    "Worker loss follows durable result acceptance; coordinator "
                    "loss follows checkpoint publication but precedes manifest apply."
                ),
                "dataset": "recovery",
                "sourceId": "recovery_query",
                "density": "dense",
                "defaultSort": {
                    "field": "recovery_seconds",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {
                        "field": "worker_loss_step",
                        "label": "Worker-loss step",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "first_worker_exit_code",
                        "label": "Lost-worker exit",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "persisted_result_survived_loss",
                        "label": "Result survived",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "recomputed_persisted_result",
                        "label": "Result recomputed",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "replacement_worker_exit_code",
                        "label": "Replacement exit",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "coordinator_loss_step",
                        "label": "Coordinator-loss step",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "fresh_process_only_persisted_state",
                        "label": "Fresh process",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "recomputed_for_validation",
                        "label": "Apply revalidated",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "recovered_from_checkpoint",
                        "label": "Recovered",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "duplicate_apply_rejected",
                        "label": "Duplicate rejected",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "recovery_process_exit_code",
                        "label": "Recovery exit",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "recovery_seconds",
                        "label": "Recovery time",
                        "format": "number",
                        "unit": "seconds",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "traffic_table",
                "title": "Three-step traffic and persistence accounting",
                "subtitle": (
                    "Primary run; private pipe framing and long-run checkpoint "
                    "compaction are outside scope."
                ),
                "dataset": "traffic",
                "sourceId": "traffic_query",
                "density": "spacious",
                "defaultSort": {
                    "field": "tensor_wire_bytes",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "topology", "label": "Topology", "type": "text"},
                    {
                        "field": "tensor_wire_bytes",
                        "label": "Tensor traffic",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "control_json_wire_bytes",
                        "label": "Control JSON",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "persisted_bytes",
                        "label": "Persisted",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "tensor_change_vs_full",
                        "label": "Tensor change vs full",
                        "format": "percent",
                        "align": "right",
                    },
                    {
                        "field": "persisted_change_vs_full",
                        "label": "Persisted change vs full",
                        "format": "percent",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "memory_table",
                "title": "Externally sampled child RSS",
                "subtitle": (
                    "Linux child-process lifecycle maxima; coordinator and "
                    "aggregate colony memory are excluded."
                ),
                "dataset": "memory",
                "sourceId": "memory_query",
                "density": "dense",
                "defaultSort": {
                    "field": "max_hwm_rss_bytes",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {"field": "topology", "label": "Topology", "type": "text"},
                    {
                        "field": "worker_generations",
                        "label": "Generations",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "rss_samples",
                        "label": "RSS samples",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "max_current_rss_bytes",
                        "label": "Max current RSS",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "max_hwm_rss_bytes",
                        "label": "Max VmHWM",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "hwm_reduction_vs_full",
                        "label": "HWM reduction vs full",
                        "format": "percent",
                        "align": "right",
                    },
                ],
            },
        ],
        "sources": _build_sources(),
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Technical summary\n\n"
                    "Two independent one-thread runs advanced the same three-step "
                    "trajectory through centralized, matched full-process, and "
                    "pooled-expert paths. All **six run-step comparisons** matched "
                    "exactly through raw gradients, clipping, AdamW state, complete "
                    "model state, and loss. A durable expert result survived worker "
                    "loss without recomputation, and a fresh coordinator recovered "
                    "a published checkpoint in **5.92 and 6.27 seconds**. The pooled "
                    "path reduced tensor traffic by **45.68%** and maximum observed "
                    "child VmHWM by **32.89% to 34.32%**, but persisted bytes fell "
                    "only **7.12%** and complete elapsed time was **67.40% to 69.07% "
                    "higher** than the matched full process. This is an exact local "
                    "systems control, not an efficiency win or model-quality result."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "exact_update_card",
                    "tensor_traffic_card",
                    "elapsed_card",
                    "child_hwm_card",
                ],
            },
            {
                "id": "timing_finding",
                "type": "markdown",
                "sourceId": "timing_query",
                "body": (
                    "## The exact path is slower in both runs\n\n"
                    "The pooled expert executor took **21.68 seconds** in the "
                    "primary run and **19.28 seconds** in the repeat. The matched "
                    "full process took **12.83 and 11.52 seconds**. These complete "
                    "times include coordinator preparation, local IPC, transaction "
                    "writes, apply, recovery, and shutdown for each process path. "
                    "The centralized path is shown as a lower-bound control because "
                    "it performs no process transport or persistence."
                ),
            },
            {
                "id": "timing_chart",
                "type": "chart",
                "chartId": "complete_timing_chart",
            },
            {
                "id": "scope_definitions",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## What the control measures\n\n"
                    "**Exact update** means identical raw gradients, clipped "
                    "gradients, AdamW tensors, and complete model tensors for one "
                    "run-step comparison. **Tensor traffic** counts serialized "
                    "safetensors payloads crossing the local process boundary. "
                    "**Persisted bytes** count pre-state, batch, accepted results, "
                    "manifests, and applied checkpoints written for the three "
                    "transactions. **Child VmHWM** is the largest externally sampled "
                    "Linux process high-water mark for one child generation. This "
                    "randomly initialized sparse tracer uses the frozen T1 tokens "
                    "and dimensions but is not a practical campaign model. No "
                    "campaign training or donated work occurred, so no contributor "
                    "credit was created."
                ),
            },
            {
                "id": "exactness_finding",
                "type": "markdown",
                "sourceId": "exactness_query",
                "body": (
                    "## Sequential state and refreshed routing repeat exactly\n\n"
                    "Every maximum tensor difference is **0.0**, and every "
                    "centralized, full-process, and pooled-expert state hash matches. "
                    "Steps one and two start from the prior applied model and "
                    "optimizer identities. Route identities change on both later "
                    "batches, while capacity enforcement still assigns "
                    "`[128,128,128,128]` tokens and reroutes 34, 44, and 42 of 512 "
                    "tokens. The recorded losses decline across different batches, "
                    "but that sequence is not a capability or generalization metric."
                ),
            },
            {
                "id": "exactness_table_block",
                "type": "table",
                "tableId": "exactness_table",
            },
            {
                "id": "recovery_finding",
                "type": "markdown",
                "sourceId": "recovery_query",
                "body": (
                    "## Durable boundaries prevent recompute and double apply\n\n"
                    "At step one, expert result zero was already durable when worker "
                    "generation zero exited with code **-15**. Its replacement "
                    "completed the remaining assignments and did not recompute result "
                    "zero. At step two, the applied checkpoint directory was "
                    "published before the manifest recorded `applied`. A new spawned "
                    "coordinator loaded only persisted state, recomputed the expected "
                    "apply for validation, committed the manifest, rejected duplicate "
                    "application, and exited with code **0** in both runs."
                ),
            },
            {
                "id": "recovery_table_block",
                "type": "table",
                "tableId": "recovery_table",
            },
            {
                "id": "traffic_finding",
                "type": "markdown",
                "sourceId": "traffic_query",
                "body": (
                    "## Tensor savings do not translate into equal disk savings\n\n"
                    "Across three steps, the pooled expert executor moved "
                    "**70,522,032 tensor bytes** versus **129,825,192** for the "
                    "matched full process, a **45.68% reduction**. Durable transaction "
                    "material was **441,492,728 bytes** versus **475,339,837**, only "
                    "**7.12% lower**. Repeated pre-state and applied checkpoints "
                    "dominate this short control, so sparse process traffic alone "
                    "does not solve persistence cost."
                ),
            },
            {
                "id": "traffic_table_block",
                "type": "table",
                "tableId": "traffic_table",
            },
            {
                "id": "memory_finding",
                "type": "markdown",
                "sourceId": "memory_query",
                "body": (
                    "## Per-child high-water RSS is lower, with a narrow scope\n\n"
                    "The matched full child reached **596,279,296 and 607,838,208 "
                    "bytes** of VmHWM. Maximum pooled-expert child VmHWM was "
                    "**400,171,008 and 399,204,352 bytes**, a **32.89% to 34.32% "
                    "reduction**. The parent sampled from spawn through shutdown, "
                    "which improves on lifecycle-boundary readings. These are still "
                    "child-only Linux observations. They exclude coordinator memory "
                    "and do not establish concurrent colony memory."
                ),
            },
            {
                "id": "memory_table_block",
                "type": "table",
                "tableId": "memory_table",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "implementation",
                "body": (
                    "## Experimental design and transaction boundary\n\n"
                    "Implementation commit "
                    f"`{IMPLEMENTATION_REVISION}` advances real coordinator-owned "
                    "AdamW state for three batches. A matched persistent full worker "
                    "and one persistent pooled expert executor each receive one "
                    "frozen head per generation and refreshed trainable state per "
                    "step. Every transaction binds the campaign, dataset, head, "
                    "pre-state, batch, routes, results, and applied state through "
                    "canonical JSON and safetensors SHA-256 identities. Results and "
                    "applied checkpoints publish through fsynced atomic directories. "
                    "This protects a trusted local protocol against corruption and "
                    "partial publication; it does not authenticate a hostile peer."
                ),
            },
            {
                "id": "reproduction",
                "type": "markdown",
                "body": (
                    "## Reproduce the evidence\n\n"
                    "Use implementation commit "
                    f"`{IMPLEMENTATION_REVISION}` on Linux. Build the exact dataset, "
                    "then run the trajectory twice with separate state directories:\n\n"
                    "```bash\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.artifacts "
                    "--output .artifacts/p7-trajectory-dataset\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_trajectory "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-trajectory-dataset "
                    "--state .artifacts/p7-trajectory-primary-state "
                    "--steps 3 --expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 --sample-interval-seconds 0.01 "
                    "--output .artifacts/p7-trajectory-primary.json\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_trajectory "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-trajectory-dataset "
                    "--state .artifacts/p7-trajectory-repeat-state "
                    "--steps 3 --expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 --sample-interval-seconds 0.01 "
                    "--output .artifacts/p7-trajectory-repeat.json\n"
                    "```\n\n"
                    "The dataset manifest must hash to `99e5642b...`. The committed "
                    "evidence files hash to `17292d43...` and `d3220db0...`. The six "
                    "committed SQLite queries reproduce every report dataset. Compare "
                    "semantic fields exactly and treat timings, RSS, sample counts, "
                    "and JSON lengths containing timing text as environmental."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Limitations and robustness boundary\n\n"
                    "This two-run, three-step CPU control does not evaluate model "
                    "quality, long-run optimizer behavior, learned expert "
                    "specialization, checkpoint compaction, aggregate memory, "
                    "concurrent throughput, or network transport. The pooled executor "
                    "is not the four-process expert-affine topology from Report 014. "
                    "Recovery recomputes the expected update for validation, which "
                    "adds cost. Linux `/proc` is required. The trusted local pipe and "
                    "digest checks are not sandboxing, remote identity, or proof of "
                    "volunteer computation."
                ),
            },
            {
                "id": "next_step",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Next: reduce durable state cost before remote execution\n\n"
                    "The next bounded method slice should keep these exact transaction "
                    "and recovery gates while testing checkpoint deduplication or "
                    "incremental state storage. It should measure whether the "
                    "persisted-byte total and recovery time fall without weakening "
                    "at-most-once apply. Concurrent or remote workers should remain a "
                    "later step because this control still shows substantial local "
                    "elapsed and persistence overhead."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Questions for the next method slice\n\n"
                    "- Can immutable checkpoint tensors be content-addressed once "
                    "without weakening exact recovery?\n"
                    "- Can recovery validate a published checkpoint without "
                    "recomputing the full expected update?\n"
                    "- How does retained state grow over dozens of steps after safe "
                    "transaction compaction?\n"
                    "- Does the child-memory advantage survive concurrent expert "
                    "workers when aggregate coordinator-plus-child RSS is measured?\n"
                    "- Which integrity and identity boundary is required before any "
                    "untrusted remote worker is admitted?"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "timing": timing,
                "exactness": exactness,
                "traffic": traffic,
                "memory": memory,
                "recovery": recovery,
            },
        },
    }


def main() -> None:
    artifact = _build_artifact()
    output = ROOT / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
