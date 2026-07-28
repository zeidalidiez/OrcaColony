#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GENERATED_AT = "2026-07-28T06:30:41Z"
IMPLEMENTATION_REVISION = "56204c7dceba49a487c153948dbb6a1fa3d2e54e"
PRIMARY_EVIDENCE = "reports/evidence/p7-authenticated-process-t1.json"
REPEAT_EVIDENCE = "reports/evidence/p7-authenticated-process-t1-repeat.json"
STUDY_EVIDENCE = (
    "research/studies/p7-authenticated-process-t1-v1/"
    "evidence/t1-four-expert-authenticated-process.json"
)
OUTPUT = "reports/artifacts/p7-authenticated-process-t1-report.json"
EXPECTED_SHA256 = {
    PRIMARY_EVIDENCE: (
        "ed8bfe6676f922d5ddace31d3eed4b74faa75fb4da3282d5e32e2b07c2164c3b"
    ),
    REPEAT_EVIDENCE: (
        "24e7f45b436cb011362061dc7deb336c87cf0f01a24b672a922670f35bcc08a9"
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
            "reports/queries/p7-authenticated-process-headline.sql",
        )
        traffic = _query_rows(
            connection,
            "reports/queries/p7-authenticated-process-traffic.sql",
        )
        exactness = _query_rows(
            connection,
            "reports/queries/p7-authenticated-process-exactness.sql",
        )
        memory = _query_rows(
            connection,
            "reports/queries/p7-authenticated-process-memory.sql",
        )
        recovery = _query_rows(
            connection,
            "reports/queries/p7-authenticated-process-recovery.sql",
        )
    finally:
        connection.close()

    if len(headline) != 1 or len(traffic) != 4:
        raise ValueError("headline or traffic query returned an invalid row count")
    if len(exactness) != 4 or len(memory) != 4 or len(recovery) != 2:
        raise ValueError("audit query returned an invalid row count")
    if float(headline[0]["max_state_difference"]) != 0.0:
        raise ValueError("process evidence is not exact")
    if int(headline[0]["recovery_result_exact"]) != 1:
        raise ValueError("replacement result is not exact")
    if any(int(row["state_hashes_match"]) != 1 for row in exactness):
        raise ValueError("process state hashes do not match")

    title = "Persistent sparse experts preserve exact updates but not peak memory"
    sources = [
        _query_source(
            source_id="headline_query",
            label="Authenticated-process headline query",
            path="reports/queries/p7-authenticated-process-headline.sql",
            description=(
                "Calculate the warm and cold payload changes, maximum state "
                "difference, and recovery headline from primary process evidence."
            ),
            filters=[f"source_path = {PRIMARY_EVIDENCE}"],
            metric_definitions=[
                (
                    "Warm tensor reduction = 1 - four-expert warm safetensors "
                    "bytes / matched full warm safetensors bytes."
                ),
                (
                    "Application payload = serialized safetensors payload bytes "
                    "+ canonical JSON payload bytes; private pipe framing is excluded."
                ),
                (
                    "Maximum state difference is the largest full-process or "
                    "expert-process raw-gradient, clipped-gradient, or complete-model "
                    "maximum absolute difference across both assignments."
                ),
            ],
        ),
        _query_source(
            source_id="traffic_query",
            label="Authenticated-process traffic query",
            path="reports/queries/p7-authenticated-process-traffic.sql",
            description=(
                "Produce matched cold and warm safetensors, JSON control, and "
                "total application payload bytes for full and expert processes."
            ),
            filters=[f"source_path = {PRIMARY_EVIDENCE}"],
            metric_definitions=[
                (
                    "Cold includes one frozen-head transmission to the full worker "
                    "and one to each of four expert workers."
                ),
                (
                    "Warm is the second assignment after each stable process "
                    "retains the same immutable head."
                ),
                (
                    "Relative change = expert application bytes / matched full "
                    "application bytes - 1 within the same cache state."
                ),
            ],
        ),
        _query_source(
            source_id="exactness_query",
            label="Authenticated-process exactness query",
            path="reports/queries/p7-authenticated-process-exactness.sql",
            description=(
                "Compare both assignments and both runs for routing, loss, "
                "maximum differences, and gradient, optimizer, and model hashes."
            ),
            filters=[
                f"source_path IN ({PRIMARY_EVIDENCE}, {REPEAT_EVIDENCE})"
            ],
            metric_definitions=[
                (
                    "State hashes match only when centralized, full-process, "
                    "and expert-process raw gradient, clipped gradient, AdamW, "
                    "and complete model SHA-256 values all agree."
                )
            ],
        ),
        _query_source(
            source_id="memory_query",
            label="Authenticated-process memory query",
            path="reports/queries/p7-authenticated-process-memory.sql",
            description=(
                "Report per-child current RSS ranges and process high-water RSS "
                "for the matched full child and four sequential expert children."
            ),
            filters=[
                f"source_path IN ({PRIMARY_EVIDENCE}, {REPEAT_EVIDENCE})"
            ],
            metric_definitions=[
                (
                    "Current RSS is sampled at worker entry, after the frozen head "
                    "loads, and at shutdown."
                ),
                (
                    "Peak RSS is the Linux process high-water mark and already "
                    "includes interpreter and import startup before worker entry."
                ),
            ],
        ),
        _query_source(
            source_id="recovery_query",
            label="Authenticated-process recovery query",
            path="reports/queries/p7-authenticated-process-recovery.sql",
            description=(
                "Report the deliberate accepted-assignment loss, replacement "
                "result identity, separate recovery traffic, and post-loss timing."
            ),
            filters=[
                f"source_path IN ({PRIMARY_EVIDENCE}, {REPEAT_EVIDENCE})"
            ],
            metric_definitions=[
                (
                    "Recovery total application bytes = lost-worker tensor inputs "
                    "+ replacement tensor inputs + replacement result safetensors "
                    "+ all loss-control JSON payload bytes."
                ),
                (
                    "Recovery seconds begin after the first worker is terminated "
                    "and include replacement initialization, compute, and shutdown."
                ),
            ],
        ),
        {
            "id": "primary_evidence",
            "label": "Primary authenticated-process evidence",
            "path": PRIMARY_EVIDENCE,
        },
        {
            "id": "repeat_evidence",
            "label": "Repeat authenticated-process evidence",
            "path": REPEAT_EVIDENCE,
        },
        {
            "id": "study_evidence",
            "label": "P7 authenticated-process study evidence",
            "path": STUDY_EVIDENCE,
        },
        {
            "id": "implementation",
            "label": "Authenticated sparse-process implementation commit",
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

    manifest: dict[str, object] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Technical findings from the exact P7 T1 persistent sparse-process control."
        ),
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "warm_reduction_card",
                "description": (
                    "Four expert processes versus the matched full process after "
                    "both retain the same immutable head."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Warm safetensors reduction",
                        "field": "warm_tensor_reduction",
                        "format": "percent",
                    },
                    {
                        "label": "Warm application reduction",
                        "field": "warm_application_reduction",
                        "format": "percent",
                    },
                    {
                        "label": "Cold application change",
                        "field": "cold_application_change",
                        "format": "percent",
                        "signed": True,
                    },
                ],
            },
            {
                "id": "exact_difference_card",
                "description": (
                    "Largest centralized-to-process difference across raw gradients, "
                    "clipped gradients, and complete model tensors."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Maximum state difference",
                        "field": "max_state_difference",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "recovery_payload_card",
                "description": (
                    "Separate payload moved by the deliberate lost-assignment and "
                    "replacement control, excluded from matched traffic totals."
                ),
                "dataset": "headline",
                "sourceId": "headline_query",
                "metrics": [
                    {
                        "label": "Loss-control application payload",
                        "field": "recovery_total_application_wire_bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    {
                        "label": "Replacement result exact",
                        "field": "recovery_result_exact",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "application_payload_chart",
                "title": "Matched cold and warm application payload bytes",
                "subtitle": (
                    "Primary run; the repeat changed only one JSON byte in two "
                    "matched totals."
                ),
                "type": "bar",
                "dataset": "traffic_comparison",
                "sourceId": "traffic_query",
                "valueFormat": "compact",
                "unit": "bytes",
                "xAxisTitle": "Cache state",
                "yAxisTitle": "Application payload bytes per assignment",
                "encodings": {
                    "x": {
                        "field": "cache_state",
                        "type": "nominal",
                        "label": "Cache state",
                    },
                    "y": {
                        "field": "application_wire_bytes",
                        "type": "quantitative",
                        "label": "Application payload bytes",
                        "format": "compact",
                        "unit": "bytes",
                    },
                    "color": {
                        "field": "topology",
                        "type": "nominal",
                        "label": "Topology",
                    },
                    "tooltip": [
                        {
                            "field": "tensor_wire_bytes",
                            "type": "quantitative",
                            "label": "Safetensors payload",
                            "format": "number",
                            "unit": "bytes",
                        },
                        {
                            "field": "control_json_wire_bytes",
                            "type": "quantitative",
                            "label": "JSON control payload",
                            "format": "number",
                            "unit": "bytes",
                        },
                        {
                            "field": "relative_change",
                            "type": "quantitative",
                            "label": "Expert change versus full",
                            "format": "percent",
                        },
                    ],
                },
            }
        ],
        "tables": [
            {
                "id": "traffic_audit_table",
                "title": "Application payload accounting",
                "subtitle": (
                    "Primary-run safetensors and canonical JSON payload bytes; "
                    "private pipe framing excluded."
                ),
                "dataset": "traffic_comparison",
                "sourceId": "traffic_query",
                "density": "spacious",
                "defaultSort": {
                    "field": "application_wire_bytes",
                    "direction": "desc",
                },
                "columns": [
                    {
                        "field": "cache_state",
                        "label": "Cache state",
                        "type": "text",
                    },
                    {
                        "field": "topology",
                        "label": "Topology",
                        "type": "text",
                    },
                    {
                        "field": "tensor_wire_bytes",
                        "label": "Safetensors",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "control_json_wire_bytes",
                        "label": "JSON control",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "application_wire_bytes",
                        "label": "Application total",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "relative_change",
                        "label": "Expert change versus full",
                        "format": "percent",
                        "movement": True,
                        "align": "right",
                    },
                ],
            },
            {
                "id": "exactness_audit_table",
                "title": "Exactness and routing audit",
                "subtitle": "Two assignments in each independent one-thread run.",
                "dataset": "exactness",
                "sourceId": "exactness_query",
                "density": "dense",
                "defaultSort": {
                    "field": "assignment_id",
                    "direction": "asc",
                },
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {
                        "field": "assignment_id",
                        "label": "Assignment",
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
                        "field": "centralized_loss",
                        "label": "Loss",
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
                "id": "memory_audit_table",
                "title": "Per-child RSS observations",
                "subtitle": (
                    "Current RSS lifecycle samples and Linux process high-water "
                    "marks; expert rows are ranges across four sequential children."
                ),
                "dataset": "memory",
                "sourceId": "memory_query",
                "density": "dense",
                "defaultSort": {
                    "field": "peak_rss_max",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "run_label", "label": "Run", "type": "text"},
                    {"field": "topology", "label": "Topology", "type": "text"},
                    {
                        "field": "worker_count",
                        "label": "Workers",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "startup_current_rss_min",
                        "label": "Startup current min",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "startup_current_rss_max",
                        "label": "Startup current max",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "after_head_current_rss_min",
                        "label": "After head min",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "after_head_current_rss_max",
                        "label": "After head max",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "final_current_rss_min",
                        "label": "Final current min",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "final_current_rss_max",
                        "label": "Final current max",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "peak_rss_min",
                        "label": "High-water min",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "peak_rss_max",
                        "label": "High-water max",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                ],
            },
            {
                "id": "recovery_audit_table",
                "title": "Accepted-assignment recovery audit",
                "subtitle": (
                    "The first child is deliberately terminated after acceptance; "
                    "replacement timing begins after termination."
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
                        "field": "first_worker_exit_code",
                        "label": "First exit",
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
                        "field": "replacement_result_matches_stable",
                        "label": "Result exact",
                        "format": "number",
                        "align": "right",
                    },
                    {
                        "field": "lost_worker_tensor_wire_bytes",
                        "label": "Lost-worker tensors",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "retransmitted_tensor_wire_bytes",
                        "label": "Retransmitted tensors",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "replacement_result_wire_bytes",
                        "label": "Replacement result",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "recovery_control_json_wire_bytes",
                        "label": "Recovery JSON",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "recovery_total_application_wire_bytes",
                        "label": "Recovery total",
                        "format": "number",
                        "unit": "bytes",
                        "align": "right",
                    },
                    {
                        "field": "replacement_initialization_seconds",
                        "label": "Replacement init",
                        "format": "number",
                        "unit": "seconds",
                        "align": "right",
                    },
                    {
                        "field": "recovery_seconds",
                        "label": "Post-loss recovery",
                        "format": "number",
                        "unit": "seconds",
                        "align": "right",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": f"# {title}",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Technical summary\n\n"
                    "Two one-thread runs on the exact frozen T1 manifest produced "
                    "the same routing, safetensors payload sizes, losses, gradients, "
                    "AdamW tensors, model tensors, cache counts, and replacement "
                    "result identity. Four warm expert processes moved "
                    "**17,913,408 safetensors bytes**, 55.75% below the matched "
                    "full process. Including canonical JSON payloads changed that "
                    "reduction only to 55.73%. Cold expert fan-out remained 5.36% "
                    "above full. The expert children ended with lower current RSS, "
                    "but their process high-water mark was higher and already "
                    "dominated by interpreter/import startup. The topology is exact "
                    "and warm-transfer-efficient in trusted local processes, but it "
                    "has not established peak-memory savings, end-to-end speed, "
                    "remote authentication, or model quality."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "warm_reduction_card",
                    "exact_difference_card",
                    "recovery_payload_card",
                ],
            },
            {
                "id": "scope_definitions",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## What these measurements cover\n\n"
                    "**Cold** is the first assignment and includes one frozen-head "
                    "transmission to the full child or one to each of four expert "
                    "children. **Warm** is the second assignment after each stable "
                    "process retains that head; every trainable tensor is still "
                    "refreshed. **Application payload** is safetensors plus canonical "
                    "JSON bytes and excludes private multiprocessing framing. The "
                    "two assignments are isolated one-step controls from identical "
                    "initialization, not a sequential training trajectory. This "
                    "experimental sparse tracer consumes the frozen T1 data and "
                    "model dimensions but is not the historical six-layer T1 model. "
                    "No campaign training or donated volunteer work occurred, so "
                    "this control creates no contributor-credit entry."
                ),
            },
            {
                "id": "traffic_finding",
                "type": "markdown",
                "sourceId": "traffic_query",
                "body": (
                    "## Warm process traffic keeps the advantage\n\n"
                    "The process boundary did not erase Report 013's warm result. "
                    "The primary warm assignment moved **17,920,210 application "
                    "bytes** through four expert children versus **40,479,906 "
                    "bytes** through the full child. Cold expert setup moved "
                    "**51,489,365 bytes** versus **48,872,200 bytes**. The repeat "
                    "changed one cold expert JSON byte and one warm full JSON byte "
                    "because timing values had different text lengths; all "
                    "safetensors byte totals repeated exactly."
                ),
            },
            {
                "id": "traffic_chart",
                "type": "chart",
                "chartId": "application_payload_chart",
            },
            {
                "id": "traffic_table",
                "type": "table",
                "tableId": "traffic_audit_table",
            },
            {
                "id": "exactness_finding",
                "type": "markdown",
                "sourceId": "exactness_query",
                "body": (
                    "## Both process paths reconstruct the same update\n\n"
                    "All **four run-assignment pairs** had 0.0 maximum raw-gradient, "
                    "clipped-gradient, and complete-model differences. Centralized, "
                    "full-process, and expert-process gradient, AdamW, and model "
                    "SHA-256 values matched. Capacity routing forced "
                    "`[128,128,128,128]` counts and rerouted 34 and 41 of 512 tokens; "
                    "the equal load is a routing policy result, not learned expert "
                    "specialization."
                ),
            },
            {
                "id": "exactness_table",
                "type": "table",
                "tableId": "exactness_audit_table",
            },
            {
                "id": "recovery_finding",
                "type": "markdown",
                "sourceId": "recovery_query",
                "body": (
                    "## A lost accepted assignment recomputes byte-for-byte\n\n"
                    "Expert zero authenticated assignment zero and was deliberately "
                    "terminated with exit code -15 before compute. Its replacement "
                    "exited cleanly and returned the same **2,236,032-byte** result "
                    "with SHA-256 `413d4c51...`; that replacement result was used in "
                    "canonical reconstruction. Post-loss recovery took 3.37 to "
                    "3.84 seconds. The separate loss control moved **23,508,639 "
                    "application bytes**, including the discarded first attempt, "
                    "replacement inputs, replacement result, and JSON controls."
                ),
            },
            {
                "id": "recovery_table",
                "type": "table",
                "tableId": "recovery_audit_table",
            },
            {
                "id": "memory_finding",
                "type": "markdown",
                "sourceId": "memory_query",
                "body": (
                    "## Current RSS falls, but peak memory remains unproven\n\n"
                    "At shutdown, expert current RSS was **373.4 to 376.0 MB** "
                    "versus **570.2 to 570.5 MB** for the matched full child, a "
                    "34.05% to 34.54% reduction. That is a useful steady-state "
                    "signal. It is not a peak result: the expert process high-water "
                    "mark was already **591.5 to 595.4 MB** before model work and "
                    "ended 3.75% to 4.36% above full. A later harness needs an "
                    "external lifecycle sampler that separates interpreter/import "
                    "startup from model execution."
                ),
            },
            {
                "id": "memory_table",
                "type": "table",
                "tableId": "memory_audit_table",
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "implementation",
                "body": (
                    "## Experimental design and trust boundary\n\n"
                    "Implementation commit "
                    f"`{IMPLEMENTATION_REVISION}` builds one matched full child and "
                    "four expert-affine children with the `spawn` start method. "
                    "Stable children run sequentially and retain one immutable "
                    "8,390,904-byte final norm/output head across two assignments. "
                    "Every assignment binds the campaign revision, frozen dataset "
                    "manifest, head, trainable state, input, routes, and result "
                    "frame with SHA-256 and exact byte counts. The coordinator owns "
                    "shared-trunk and router work, gradient reduction, one global "
                    "clip, and AdamW. This is corruption-resistant local framing "
                    "inside a trusted spawned-pipe scope, not cryptographic "
                    "authentication of a hostile remote peer."
                ),
            },
            {
                "id": "reproduction",
                "type": "markdown",
                "body": (
                    "## Reproduce the evidence\n\n"
                    "Use implementation commit "
                    f"`{IMPLEMENTATION_REVISION}`. Build the exact data artifact, "
                    "then run the control twice:\n\n"
                    "```bash\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.artifacts "
                    "--output .artifacts/p7-sparse-process-dataset\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_process "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-sparse-process-dataset "
                    "--expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 "
                    "--output .artifacts/p7-sparse-process-primary.json\n"
                    "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
                    "uv run python -m orcacolony.sparse_expert_process "
                    "--config campaign/t1-tinystories-system-proof.json "
                    "--dataset .artifacts/p7-sparse-process-dataset "
                    "--expert-count 4 --router-aux-weight 0.01 "
                    "--timeout-seconds 120 "
                    "--output .artifacts/p7-sparse-process-repeat.json\n"
                    "```\n\n"
                    "The data build must reproduce manifest SHA-256 "
                    "`99e5642b...`. Committed evidence SHA-256 values are "
                    "`ed8bfe66...` and `24e7f45b...`. The five committed SQLite "
                    "queries reproduce the report datasets from those JSON files. "
                    "Compare deterministic fields exactly and keep timing, RSS, "
                    "and JSON lengths that contain timing text as observations."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## What this result does not establish\n\n"
                    "This experiment does not train or evaluate a useful model. "
                    "It does not execute a refreshed multi-step trajectory, persist "
                    "a recovery transaction, survive coordinator loss, run workers "
                    "concurrently, measure aggregate RSS, include network transport, "
                    "sandbox a worker, or prove donated computation. Worker timers "
                    "exclude coordinator shared-trunk, routing, reduction, clipping, "
                    "and optimizer work, so they are not end-to-end speed results. "
                    "The first loss child pauses cooperatively after acceptance; "
                    "sudden remote failure and hostile acknowledgements remain "
                    "outside scope."
                ),
            },
            {
                "id": "next_step",
                "type": "markdown",
                "sourceId": "study_evidence",
                "body": (
                    "## Next: refresh state through a persisted multi-step control\n\n"
                    "The next bounded P7 slice should advance several sequential "
                    "coordinator AdamW steps, refresh shared hidden rows, routes, "
                    "and expert trainable state after each checkpoint, and persist "
                    "accepted expert results so a replacement or coordinator restart "
                    "cannot double-apply work. The same harness should sample child "
                    "RSS externally across the full lifecycle and report complete "
                    "coordinator-plus-worker elapsed time. Remote volunteer transport "
                    "should wait until those local state and recovery semantics are "
                    "exact."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Questions for the next slice\n\n"
                    "- Does the 55.75% warm tensor reduction persist when routes and "
                    "expert weights change after every optimizer step?\n"
                    "- How many warm assignments repay the 5.34% cold tensor penalty "
                    "under real worker availability?\n"
                    "- What is the true model-compute peak after interpreter startup "
                    "is separated from lifecycle sampling?\n"
                    "- Can an accepted expert result survive coordinator restart and "
                    "still be applied at most once?\n"
                    "- Which integrity mechanism is sufficient before an untrusted "
                    "public worker is allowed to contribute?"
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
                "traffic_comparison": traffic,
                "exactness": exactness,
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
