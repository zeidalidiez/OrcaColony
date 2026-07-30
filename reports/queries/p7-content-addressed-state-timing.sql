WITH runs AS (
  SELECT
    CASE source_path
      WHEN 'reports/evidence/p7-content-addressed-state-t1.json' THEN 0
      ELSE 1
    END AS run_order,
    CASE source_path
      WHEN 'reports/evidence/p7-content-addressed-state-t1.json'
        THEN 'Primary: replicated first'
      ELSE 'Repeat: content addressed first'
    END AS run_label,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-content-addressed-state-t1.json',
    'reports/evidence/p7-content-addressed-state-t1-repeat.json'
  )
),
timing_rows AS (
  SELECT
    run_order,
    run_label,
    0 AS metric_order,
    'Full elapsed' AS metric,
    json_extract(doc, '$.replicated.full_process_end_to_end_seconds')
      AS baseline_seconds,
    json_extract(doc, '$.content_addressed.full_process_end_to_end_seconds')
      AS candidate_seconds
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    1,
    'Expert elapsed',
    json_extract(doc, '$.replicated.expert_process_end_to_end_seconds'),
    json_extract(doc, '$.content_addressed.expert_process_end_to_end_seconds')
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    2,
    'Recovery elapsed',
    json_extract(doc, '$.replicated.coordinator_recovery.recovery_seconds'),
    json_extract(
      doc,
      '$.content_addressed.coordinator_recovery.recovery_seconds'
    )
  FROM runs
)
SELECT
  run_label,
  metric,
  baseline_seconds,
  candidate_seconds,
  1.0 * candidate_seconds / baseline_seconds - 1.0
    AS candidate_change_fraction
FROM timing_rows
ORDER BY metric_order, run_order;
