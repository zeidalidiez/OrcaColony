#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED_AT = "2026-07-30T03:28:09Z"
IMPLEMENTATION_REVISION = "2e0dcd6e82551330f87dedde8376539db3cd9899"
MERGED_EXECUTION_REVISION = "c43c2cb5bb339e037ab48701f38a07d104b86de4"
PRIMARY_EVIDENCE = "reports/evidence/p7-content-addressed-state-t1.json"
REPEAT_EVIDENCE = "reports/evidence/p7-content-addressed-state-t1-repeat.json"
STUDY_EVIDENCE = (
    "research/studies/p7-content-addressed-state-t1-v1/"
    "evidence/t1-content-addressed-state.json"
)
OUTPUT = "reports/artifacts/p7-content-addressed-state-t1-report.json"
EXPECTED_SHA256 = {
    PRIMARY_EVIDENCE: (
        "8a75ba9f057ae04e034851c033630f71ef699e8ef84581a1264fc0d2dec1f481"
    ),
    REPEAT_EVIDENCE: (
        "047e2e67a1a157aca4f33cc63df043fca9bba777a1ccdb8c8c8582d70a7d1e5b"
    ),
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
            "filters": [
                (
                    "source_path IN "
                    f"({PRIMARY_EVIDENCE}, {REPEAT_EVIDENCE})"
                )
            ],
            "metric_definitions": metric_definitions,
        },
    }


def _trajectory_projection(payload: dict[str, object]) -> dict[str, object]:
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


def _comparison_projection(payload: dict[str, object]) -> dict[str, object]:
    projected = deepcopy(payload)
    projected.pop("comparison_order", None)
    for layout in ("replicated", "content_addressed"):
        nested = projected.get(layout)
        if not isinstance(nested, dict):
            raise ValueError(f"comparison layout is missing: {layout}")
        projected[layout] = _trajectory_projection(nested)
    return projected


def _build_sources() -> list[dict[str, object]]:
    return [
        _query_source(
            source_id="headline_query",
            label="Content-addressed state headline query",
            path="reports/queries/p7-content-addressed-state-headline.sql",
            description=(
                "Calculate storage reductions, exact comparison coverage, "
                "repeat identity, and checkpoint-reference reuse."
            ),
            metric_definitions=[
                (
                    "Persisted-byte reduction equals replicated regular-file "
                    "payload bytes minus content-addressed regular-file payload "
                    "bytes within one topology root."
                ),
                (
                    "Reduction fraction equals persisted-byte reduction divided "
                    "by replicated persisted bytes."
                ),
                (
                    "An exact layout-run-step matches centralized raw and clipped "
                    "gradients, AdamW state, complete model state, and loss."
                ),
            ],
        ),
        _query_source(
            source_id="storage_query",
            label="Content-addressed state storage query",
            path="reports/queries/p7-content-addressed-state-storage.sql",
            description=(
                "Return replicated and content-addressed persisted bytes for the "
                "full-process and pooled-expert topology controls."
            ),
            metric_definitions=[
                (
                    "Persisted bytes sum regular-file payload lengths under the "
                    "topology root, including its shared blob directory and "
                    "excluding directory metadata and block allocation."
                ),
                (
                    "Repeat matches is one when the opposite-order run reproduces "
                    "the same persisted-byte value."
                ),
            ],
        ),
        _query_source(
            source_id="timing_query",
            label="Content-addressed state timing query",
            path="reports/queries/p7-content-addressed-state-timing.sql",
            description=(
                "Compare full, expert, and fresh-coordinator recovery elapsed "
                "times in two runs with opposite execution orders."
            ),
            metric_definitions=[
                (
                    "Candidate change equals content-addressed seconds divided by "
                    "replicated seconds minus one within the same run."
                ),
                (
                    "Complete topology time includes coordinator preparation, "
                    "worker IPC, persistence, apply, recovery, and shutdown."
                ),
            ],
        ),
        _query_source(
            source_id="exactness_query",
            label="Content-addressed state exactness query",
            path="reports/queries/p7-content-addressed-state-exactness.sql",
            description=(
                "Audit all twelve run, layout, and step rows for exact gradient, "
                "optimizer, model, and transaction identities."
            ),
            metric_definitions=[
                (
                    "Exact is one only when centralized, full-process, and "
                    "pooled-expert raw gradients, clipped gradients, AdamW state, "
                    "and complete model state match with zero maximum difference."
                )
            ],
        ),
        _query_source(
            source_id="recovery_query",
            label="Content-addressed state recovery query",
            path="reports/queries/p7-content-addressed-state-recovery.sql",
            description=(
                "Audit durable worker-result reuse and fresh-process coordinator "
                "recovery for both layouts in both runs."
            ),
            metric_definitions=[
                (
                    "Worker recovery passes when the durable accepted result "
                    "survives loss and is not recomputed."
                ),
                (
                    "Coordinator recovery passes when a new process loads only "
                    "persisted state, validates the published checkpoint, commits "
                    "the manifest, rejects duplicate apply, and exits cleanly."
                ),
            ],
        ),
        {
            "id": "primary_evidence",
            "label": "Primary replicated-first evidence",
            "path": PRIMARY_EVIDENCE,
        },
        {
            "id": "repeat_evidence",
            "label": "Repeat content-addressed-first evidence",
            "path": REPEAT_EVIDENCE,
        },
        {
            "id": "study_evidence",
            "label": "P7 content-addressed state study evidence",
            "path": STUDY_EVIDENCE,
        },
        {
            "id": "implementation",
            "label": "Content-addressed state implementation commit",
            "href": (
                "https://github.com/zeidalidiez/OrcaColony/commit/"
                f"{IMPLEMENTATION_REVISION}"
            ),
        },
        {
            "id": "merged_execution_revision",
            "label": "Merged revision used for execution",
            "href": (
                "https://github.com/zeidalidiez/OrcaColony/commit/"
                f"{MERGED_EXECUTION_REVISION}"
            ),
        },
        {
            "id": "t1_campaign",
            "label": "Frozen T1 systems configuration",
            "path": "campaign/t1-tinystories-system-proof.json",
        },
    ]


def _load_report_datasets() -> dict[str, list[dict[str, Any]]]:
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
        return {
            "headline": _query_rows(
                connection,
                "reports/queries/p7-content-addressed-state-headline.sql",
            ),
            "storage": _query_rows(
                connection,
                "reports/queries/p7-content-addressed-state-storage.sql",
            ),
            "timing": _query_rows(
                connection,
                "reports/queries/p7-content-addressed-state-timing.sql",
            ),
            "exactness": _query_rows(
                connection,
                "reports/queries/p7-content-addressed-state-exactness.sql",
            ),
            "recovery": _query_rows(
                connection,
                "reports/queries/p7-content-addressed-state-recovery.sql",
            ),
        }
    finally:
        connection.close()


def _validate_report_datasets(
    datasets: dict[str, list[dict[str, Any]]],
) -> None:
    if len(datasets["headline"]) != 1:
        raise ValueError("headline query row count is invalid")
    if len(datasets["storage"]) != 4:
        raise ValueError("storage query row count is invalid")
    if len(datasets["timing"]) != 6:
        raise ValueError("timing query row count is invalid")
    if len(datasets["exactness"]) != 12:
        raise ValueError("exactness query row count is invalid")
    if len(datasets["recovery"]) != 4:
        raise ValueError("recovery query row count is invalid")

    headline = datasets["headline"][0]
    if int(headline["exact_layout_run_steps"]) != 12:
        raise ValueError("not every layout, run, and step comparison is exact")
    if int(headline["compared_layout_run_steps"]) != 12:
        raise ValueError("unexpected comparison count")
    if int(headline["storage_repeated_exactly"]) != 1:
        raise ValueError("storage totals did not repeat")
    if int(headline["identity_guardrails_passed"]) != 1:
        raise ValueError("identity guardrails did not pass")
    if any(int(row["repeat_matches"]) != 1 for row in datasets["storage"]):
        raise ValueError("a storage measurement did not repeat")
    if any(int(row["exact"]) != 1 for row in datasets["exactness"]):
        raise ValueError("an exactness audit row failed")

    for row in datasets["recovery"]:
        if not (
            int(row["persisted_result_survived_loss"]) == 1
            and int(row["recomputed_persisted_result"]) == 0
            and int(row["fresh_process_only_persisted_state"]) == 1
            and int(row["recovered_from_checkpoint"]) == 1
            and int(row["duplicate_apply_rejected"]) == 1
            and int(row["recovery_process_exit_code"]) == 0
        ):
            raise ValueError("a recovery audit row failed")

    changes: dict[str, list[float]] = {}
    for row in datasets["timing"]:
        changes.setdefault(str(row["metric"]), []).append(
            float(row["candidate_change_fraction"])
        )
    if set(changes) != {
        "Full elapsed",
        "Expert elapsed",
        "Recovery elapsed",
    }:
        raise ValueError("timing metric set is invalid")
    if any(
        len(values) != 2 or values[0] * values[1] >= 0.0
        for values in changes.values()
    ):
        raise ValueError("timing directions did not flip as reported")


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
    if primary.get("format") != (
        "orcacolony_content_addressed_sparse_trajectory_comparison_v1"
    ):
        raise ValueError("primary evidence format is invalid")
    if repeat.get("format") != primary.get("format"):
        raise ValueError("repeat evidence format differs")
    if primary.get("comparison_order") != (
        "replicated-then-content-addressed"
    ):
        raise ValueError("primary comparison order is invalid")
    if repeat.get("comparison_order") != (
        "content-addressed-then-replicated"
    ):
        raise ValueError("repeat comparison order is invalid")
    if _comparison_projection(primary) != _comparison_projection(repeat):
        raise ValueError("primary and repeat deterministic evidence differ")
    if study.get("outcome") != "validated":
        raise ValueError("study evidence is not validated")

    datasets = _load_report_datasets()
    _validate_report_datasets(datasets)

    title = "Content-addressed checkpoints remove repeated state safely"
    manifest: dict[str, object] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Technical findings from the paired P7 T1 replicated and "
            "content-addressed checkpoint-storage control."
        ),
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "full_storage_card",
                "description": (
                    "Three-step full-process topology, including the topology-local "
                    "blob store."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Content-addressed bytes",
                        "field": "full_content_addressed_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    {
                        "label": "Reduction",
                        "field": "full_reduction_fraction",
                        "format": "percent",
                    },
                    {
                        "label": "Bytes removed",
                        "field": "full_reduction_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                ],
            },
            {
                "id": "expert_storage_card",
                "description": (
                    "Three-step pooled-expert topology, including the "
                    "topology-local blob store."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Content-addressed bytes",
                        "field": "expert_content_addressed_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    {
                        "label": "Reduction",
                        "field": "expert_reduction_fraction",
                        "format": "percent",
                    },
                    {
                        "label": "Bytes removed",
                        "field": "expert_reduction_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                ],
            },
            {
                "id": "exactness_card",
                "description": (
                    "Centralized, full-process, and pooled-expert identity checks "
                    "for both layouts and both run orders."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Exact layout/run/steps",
                        "field": "exact_layout_run_steps",
                        "format": "number",
                    },
                    {
                        "label": "Compared layout/run/steps",
                        "field": "compared_layout_run_steps",
                        "format": "number",
                    },
                    {
                        "label": "Storage repeated",
                        "field": "storage_repeated_exactly",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "reuse_card",
                "description": (
                    "Reference and unique-blob counts within each independent "
                    "topology-local store."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Checkpoint references",
                        "field": "checkpoint_references_per_topology",
                        "format": "number",
                    },
                    {
                        "label": "Unique blobs",
                        "field": "unique_blobs_per_topology",
                        "format": "number",
                    },
                    {
                        "label": "Reused references",
                        "field": "reused_references_per_topology",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "storage_chart",
                "title": "Persisted file payload by topology and layout",
                "subtitle": (
                    "The opposite-order repeat reproduced every byte total exactly."
                ),
                "type": "bar",
                "dataset": "storage",
                "sourceId": "storage_query",
                "valueFormat": "compact",
                "unit": "bytes",
                "xAxisTitle": "Execution topology",
                "yAxisTitle": "Persisted regular-file payload bytes",
                "encodings": {
                    "x": {
                        "field": "topology",
                        "type": "nominal",
                        "label": "Execution topology",
                    },
                    "y": {
                        "field": "persisted_bytes",
                        "type": "quantitative",
                        "label": "Persisted bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    "color": {
                        "field": "layout",
                        "type": "nominal",
                        "label": "Storage layout",
                    },
                    "tooltip": [
                        {
                            "field": "reduction_fraction",
                            "type": "quantitative",
                            "label": "Reduction versus replicated",
                            "format": "percent",
                        },
                        {
                            "field": "repeat_matches",
                            "type": "quantitative",
                            "label": "Repeat matches",
                            "format": "number",
                        },
                    ],
                },
            },
            {
                "id": "timing_chart",
                "title": "Content-addressed elapsed-time change by run",
                "subtitle": (
                    "All three directions reverse when execution order is reversed."
                ),
                "type": "bar",
                "dataset": "timing",
                "sourceId": "timing_query",
                "valueFormat": "percent",
                "xAxisTitle": "Measured lifecycle",
                "yAxisTitle": "Change versus replicated layout",
                "encodings": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Measured lifecycle",
                    },
                    "y": {
                        "field": "candidate_change_fraction",
                        "type": "quantitative",
                        "label": "Candidate change",
                        "format": "percent",
                    },
                    "color": {
                        "field": "run_label",
                        "type": "nominal",
                        "label": "Run order",
                    },
                    "tooltip": [
                        {
                            "field": "baseline_seconds",
                            "type": "quantitative",
                            "label": "Replicated seconds",
                            "format": "number",
                            "unit": "seconds",
                        },
                        {
                            "field": "candidate_seconds",
                            "type": "quantitative",
                            "label": "Content-addressed seconds",
                            "format": "number",
                            "unit": "seconds",
                        },
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "storage_table",
                "title": "Exact storage accounting",
                "subtitle": (
                    "Regular-file payload bytes under each independent topology "
                    "root; directory metadata and block allocation are excluded."
                ),
                "dataset": "storage",
                "sourceId": "storage_query",
                "density": "spacious",
                "defaultSort": {
                    "field": "persisted_bytes",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "topology", "label": "Topology", "type": "text"},
                    {"field": "layout", "label": "Layout", "type": "text"},
                    {
                        "field": "persisted_bytes",
                        "label": "Persisted bytes",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "reduction_bytes",
                        "label": "Bytes removed",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "reduction_fraction",
                        "label": "Reduction",
                        "format": "percent",
                        "align": "right",
                    },
                    {
                        "field": "repeat_matches",
                        "label": "Repeat matches",
                        "format": "number",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "exactness_table",
                "title": "Layout, run, and step exactness audit",
                "subtitle": (
                    "Twelve rows compare centralized, full-process, and "
                    "pooled-expert gradient and state identities."
                ),
                "dataset": "exactness",
                "sourceId": "exactness_query",
                "density": "dense",
                "defaultSort": {
                    "field": "step",
                    "direction": "asc",
                },
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {"field": "layout", "label": "Layout", "type": "text"},
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
                        "field": "exact",
                        "label": "Exact",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "transaction_ids_match",
                        "label": "Transaction IDs match",
                        "format": "number",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "recovery_table",
                "title": "Worker and coordinator recovery audit",
                "subtitle": (
                    "Both layouts pass the same durable-result, fresh-process, "
                    "and at-most-once gates in both runs."
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
                    {"field": "layout", "label": "Layout", "type": "text"},
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
                        "field": "fresh_process_only_persisted_state",
                        "label": "Fresh-process load",
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
                        "field": "recovery_seconds",
                        "label": "Recovery time",
                        "format": "number",
                        "unit": "seconds",
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
                    "The topology-local content-addressed layout removed exactly "
                    "**138,198,293 bytes** of repeated checkpoint payload from each "
                    "three-step topology. Persisted payload fell **29.07%** for the "
                    "full-process control and **31.30%** for the pooled-expert "
                    "control. The opposite-order repeat reproduced every storage "
                    "total and non-environmental identity. All **12 layout, run, "
                    "and step rows** remained exact, and worker and coordinator "
                    "recovery passed under both layouts. Full, expert, and recovery "
                    "timing each changed direction when execution order was "
                    "reversed, so the evidence supports no timing improvement or "
                    "regression. This closes the repeated-checkpoint storage target."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "full_storage_card",
                    "expert_storage_card",
                    "exactness_card",
                    "reuse_card",
                ],
            },
            {
                "id": "storage_finding",
                "type": "markdown",
                "sourceId": "storage_query",
                "body": (
                    "## Duplicate checkpoint payload fell by 138.2 MB per topology\n\n"
                    "The full-process root fell from **475,339,837** to "
                    "**337,141,544 bytes**. The pooled-expert root fell from "
                    "**441,492,728** to **303,294,435 bytes**. The absolute saving "
                    "is equal because both paths stop writing the same duplicated "
                    "model and optimizer checkpoint payloads into transaction "
                    "directories. Each independent store has **8 unique blobs**, "
                    "**12 checkpoint references**, and **4 reused references**."
                ),
            },
            {
                "id": "storage_chart_block",
                "type": "chart",
                "chartId": "storage_chart",
            },
            {
                "id": "storage_table_block",
                "type": "table",
                "tableId": "storage_table",
            },
            {
                "id": "timing_finding",
                "type": "markdown",
                "sourceId": "timing_query",
                "body": (
                    "## Timing did not move in a repeatable direction\n\n"
                    "With replicated storage first, content addressing changed "
                    "full elapsed time by **-12.28%**, expert elapsed time by "
                    "**+1.51%**, and recovery time by **+19.77%**. With content "
                    "addressing first, the same comparisons were **+26.09%**, "
                    "**-3.85%**, and **-8.22%**. The candidate and baseline ranges "
                    "overlap for every measure. Two local observations are enough "
                    "to reject a stable direction here, but not enough to estimate "
                    "a performance effect."
                ),
            },
            {
                "id": "timing_chart_block",
                "type": "chart",
                "chartId": "timing_chart",
            },
            {
                "id": "scope_definitions",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Scope, data, and metric definitions\n\n"
                    "This is the frozen three-step T1 **systems control** from the "
                    "preceding trajectory study. It is not a practical training "
                    "campaign. **Persisted bytes** are the sum of regular-file "
                    "payload lengths under one topology root, including that root's "
                    "shared blob directory. The metric excludes directory metadata, "
                    "filesystem block allocation, and temporary run roots after "
                    "measurement. **Exact** means centralized, full-process, and "
                    "pooled-expert raw gradients, clipped gradients, AdamW tensors, "
                    "complete model tensors, loss, and transaction identities agree "
                    "for the recorded step. No model-quality metric, campaign "
                    "decision, donated computation, or contributor credit is part "
                    "of this control."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "implementation",
                "body": (
                    "## Experimental design kept identity and failure gates fixed\n\n"
                    f"Implementation commit `{IMPLEMENTATION_REVISION}` adds a "
                    "separate candidate layout while leaving the replicated "
                    "trajectory unchanged. Full-process and pooled-expert blob "
                    "stores remain topology-local so neither control shares files "
                    "with the other. Canonical references bind immutable "
                    "safetensors blobs by SHA-256 and byte length. Closed-directory, "
                    "canonical-JSON, blob-membership, digest, physical-byte, "
                    "duplicate-apply, worker-loss, and fresh-coordinator checks fail "
                    "closed. The primary run executed replicated storage first; the "
                    "repeat reversed that order. Both used one compute thread and "
                    "at most one child process."
                ),
            },
            {
                "id": "exactness_finding",
                "type": "markdown",
                "sourceId": "exactness_query",
                "body": (
                    "## Exact state advancement survives the storage change\n\n"
                    "Every one of the **12 layout, run, and step rows** has zero "
                    "maximum tensor difference and matching gradient, optimizer, "
                    "model, and transaction identities. The candidate final model "
                    "and optimizer hashes match the replicated control in both "
                    "topologies and both runs."
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
                    "## Recovery and at-most-once behavior remain intact\n\n"
                    "Under both layouts and both run orders, an accepted expert "
                    "result survived worker loss without recomputation. A fresh "
                    "coordinator loaded only persisted state, validated the "
                    "published applied checkpoint, completed the lagging manifest "
                    "transition, rejected duplicate application, and exited with "
                    "code zero. Candidate recovery took **6.70 to 7.10 seconds**, "
                    "but the paired timing direction is not stable."
                ),
            },
            {
                "id": "recovery_table_block",
                "type": "table",
                "tableId": "recovery_table",
            },
            {
                "id": "reproduction",
                "type": "markdown",
                "body": (
                    "## Reproduce the evidence\n\n"
                    f"Use merged revision `{MERGED_EXECUTION_REVISION}`, whose tree "
                    "matches the implementation commit, on Linux. Build the frozen "
                    "dataset once, then use fresh state roots and opposite orders:\n\n"
                    "```bash\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.artifacts "
                    "--output .artifacts/p7-content-store-dataset\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_content_store "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-content-store-dataset "
                    "--state .artifacts/p7-content-store-primary-state "
                    "--steps 3 --expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 --sample-interval-seconds 0.01 "
                    "--comparison-order replicated-then-content-addressed "
                    "--output .artifacts/p7-content-store-primary.json\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_content_store "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-content-store-dataset "
                    "--state .artifacts/p7-content-store-repeat-state "
                    "--steps 3 --expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 --sample-interval-seconds 0.01 "
                    "--comparison-order content-addressed-then-replicated "
                    "--output .artifacts/p7-content-store-repeat.json\n"
                    "```\n\n"
                    "The dataset manifest SHA-256 is "
                    "`99e5642bec2a9fa0b7f6175ed5f4821bf4f9aa2c08ec1038f12bfdfb302bb4af`. "
                    "The committed primary and repeat evidence SHA-256 values are "
                    "`8a75ba9f057ae04e034851c033630f71ef699e8ef84581a1264fc0d2dec1f481` "
                    "and "
                    "`047e2e67a1a157aca4f33cc63df043fca9bba777a1ccdb8c8c8582d70a7d1e5b`. "
                    "Temporary state is about 1.5 GB per run and may be removed "
                    "after the evidence JSON is secured."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Limitations, uncertainty, and robustness\n\n"
                    "The deterministic result covers two local CPU runs, two "
                    "storage layouts, two process topologies, and three optimizer "
                    "steps. It does not measure long-run retention, garbage "
                    "collection, interrupted-blob cleanup, filesystem block use, "
                    "network behavior, concurrent coordinators, or hostile peers. "
                    "Timing is an environmental host observation and is explicitly "
                    "inconclusive. Recovery still recomputes the expected update for "
                    "validation. The sparse tracer is not a practical model, and "
                    "this record establishes no training benefit or model "
                    "capability."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Close the storage target and return to the project decision\n\n"
                    "The content-addressed layout should remain the qualified "
                    "experimental control. The measured storage problem is solved "
                    "for this bounded trajectory, exactness is preserved, and the "
                    "timing evidence gives no reason for another storage mechanism "
                    "iteration. Do not open another checkpoint-storage branch now. "
                    "The next substantive target should come from one of the actual "
                    "project gates: an owner-defined practical campaign contract, "
                    "or operator-supplied inputs for a remote trusted deployment. "
                    "Neither is selected by this systems report."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- Which practical usage scenario, evaluator, model, data, and "
                    "decision criteria will the owner choose for the first campaign?\n"
                    "- Which hosting, HTTPS, participant, and operational inputs are "
                    "available if remote trusted deployment becomes the next target?\n"
                    "- Does a future campaign produce evidence that checkpoint "
                    "garbage collection or cheaper recovery validation is worth "
                    "prioritizing?\n"
                    "- What long-run retention policy should be tested only after a "
                    "real campaign establishes its checkpoint and audit needs?"
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
            "datasets": datasets,
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
