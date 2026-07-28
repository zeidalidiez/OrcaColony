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
timing AS (
  SELECT
    run_order,
    run_label,
    0 AS topology_order,
    'Centralized' AS topology,
    json_extract(doc, '$.centralized_end_to_end_seconds') AS complete_seconds,
    1.0 * json_extract(doc, '$.centralized_end_to_end_seconds')
      / json_extract(doc, '$.full_process_end_to_end_seconds') - 1.0
      AS change_vs_full
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    1,
    'Matched full process',
    json_extract(doc, '$.full_process_end_to_end_seconds'),
    0.0
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    2,
    'Pooled expert executor',
    json_extract(doc, '$.expert_process_end_to_end_seconds'),
    1.0 * json_extract(doc, '$.expert_process_end_to_end_seconds')
      / json_extract(doc, '$.full_process_end_to_end_seconds') - 1.0
  FROM runs
)
SELECT
  run_label,
  topology,
  complete_seconds,
  change_vs_full
FROM timing
ORDER BY topology_order, run_order;
