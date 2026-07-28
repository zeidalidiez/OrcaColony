WITH runs AS (
  SELECT
    CASE source_path
      WHEN 'reports/evidence/p7-authenticated-process-t1.json'
        THEN 0
      ELSE 1
    END AS run_order,
    CASE source_path
      WHEN 'reports/evidence/p7-authenticated-process-t1.json'
        THEN 'Primary'
      ELSE 'Repeat'
    END AS run_label,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-authenticated-process-t1.json',
    'reports/evidence/p7-authenticated-process-t1-repeat.json'
  )
),
memory AS (
  SELECT
    run_order,
    run_label,
    0 AS topology_order,
    'Matched full process' AS topology,
    1 AS worker_count,
    json_extract(
      doc,
      '$.full_worker.worker_startup_current_rss_bytes'
    ) AS startup_current_rss_min,
    json_extract(
      doc,
      '$.full_worker.worker_startup_current_rss_bytes'
    ) AS startup_current_rss_max,
    json_extract(
      doc,
      '$.full_worker.worker_after_head_current_rss_bytes'
    ) AS after_head_current_rss_min,
    json_extract(
      doc,
      '$.full_worker.worker_after_head_current_rss_bytes'
    ) AS after_head_current_rss_max,
    json_extract(
      doc,
      '$.full_worker.worker_final_current_rss_bytes'
    ) AS final_current_rss_min,
    json_extract(
      doc,
      '$.full_worker.worker_final_current_rss_bytes'
    ) AS final_current_rss_max,
    json_extract(
      doc,
      '$.full_worker.worker_final_peak_rss_bytes'
    ) AS peak_rss_min,
    json_extract(
      doc,
      '$.full_worker.worker_final_peak_rss_bytes'
    ) AS peak_rss_max
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    1,
    'Expert process',
    json_array_length(doc, '$.expert_workers'),
    min(json_extract(value, '$.worker_startup_current_rss_bytes')),
    max(json_extract(value, '$.worker_startup_current_rss_bytes')),
    min(json_extract(value, '$.worker_after_head_current_rss_bytes')),
    max(json_extract(value, '$.worker_after_head_current_rss_bytes')),
    min(json_extract(value, '$.worker_final_current_rss_bytes')),
    max(json_extract(value, '$.worker_final_current_rss_bytes')),
    min(json_extract(value, '$.worker_final_peak_rss_bytes')),
    max(json_extract(value, '$.worker_final_peak_rss_bytes'))
  FROM runs, json_each(runs.doc, '$.expert_workers')
  GROUP BY run_order, run_label, doc
)
SELECT
  run_label,
  topology,
  worker_count,
  startup_current_rss_min,
  startup_current_rss_max,
  after_head_current_rss_min,
  after_head_current_rss_max,
  final_current_rss_min,
  final_current_rss_max,
  peak_rss_min,
  peak_rss_max
FROM memory
ORDER BY run_order, topology_order;
