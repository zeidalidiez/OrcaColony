WITH runs AS (
  SELECT
    CASE source_path
      WHEN 'reports/evidence/p7-persisted-trajectory-t1.json' THEN 0
      ELSE 1
    END AS run_order,
    CASE source_path
      WHEN 'reports/evidence/p7-persisted-trajectory-t1.json' THEN 'Primary'
      ELSE 'Repeat'
    END AS run_label,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-persisted-trajectory-t1.json',
    'reports/evidence/p7-persisted-trajectory-t1-repeat.json'
  )
),
full_memory AS (
  SELECT
    run_order,
    run_label,
    max(
      json_extract(value, '$.external_rss.lifecycle_max_current_rss_bytes')
    ) AS full_max_current_rss_bytes,
    max(
      json_extract(value, '$.external_rss.lifecycle_max_hwm_rss_bytes')
    ) AS full_max_hwm_rss_bytes
  FROM runs, json_each(runs.doc, '$.full_workers')
  GROUP BY run_order, run_label
),
expert_memory AS (
  SELECT
    run_order,
    run_label,
    count(*) AS expert_worker_generations,
    sum(json_extract(value, '$.external_rss.sample_count'))
      AS expert_rss_samples,
    max(
      json_extract(value, '$.external_rss.lifecycle_max_current_rss_bytes')
    ) AS expert_max_current_rss_bytes,
    max(
      json_extract(value, '$.external_rss.lifecycle_max_hwm_rss_bytes')
    ) AS expert_max_hwm_rss_bytes
  FROM runs, json_each(runs.doc, '$.expert_workers')
  GROUP BY run_order, run_label
),
memory AS (
  SELECT
    runs.run_order,
    runs.run_label,
    0 AS topology_order,
    'Matched full process' AS topology,
    json_array_length(runs.doc, '$.full_workers') AS worker_generations,
    (
      SELECT sum(json_extract(value, '$.external_rss.sample_count'))
      FROM json_each(runs.doc, '$.full_workers')
    ) AS rss_samples,
    full_memory.full_max_current_rss_bytes AS max_current_rss_bytes,
    full_memory.full_max_hwm_rss_bytes AS max_hwm_rss_bytes,
    0.0 AS hwm_reduction_vs_full
  FROM runs
  JOIN full_memory USING (run_order, run_label)

  UNION ALL

  SELECT
    runs.run_order,
    runs.run_label,
    1,
    'Pooled expert executor',
    expert_memory.expert_worker_generations,
    expert_memory.expert_rss_samples,
    expert_memory.expert_max_current_rss_bytes,
    expert_memory.expert_max_hwm_rss_bytes,
    1.0 - 1.0 * expert_memory.expert_max_hwm_rss_bytes
      / full_memory.full_max_hwm_rss_bytes
  FROM runs
  JOIN full_memory USING (run_order, run_label)
  JOIN expert_memory USING (run_order, run_label)
)
SELECT
  run_label,
  topology,
  worker_generations,
  rss_samples,
  max_current_rss_bytes,
  max_hwm_rss_bytes,
  hwm_reduction_vs_full
FROM memory
ORDER BY run_order, topology_order;
