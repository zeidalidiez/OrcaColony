WITH runs AS (
  SELECT
    CASE source_path
      WHEN 'reports/evidence/p7-content-addressed-state-t1.json' THEN 0
      ELSE 1
    END AS run_order,
    CASE source_path
      WHEN 'reports/evidence/p7-content-addressed-state-t1.json' THEN 'Primary'
      ELSE 'Repeat'
    END AS run_label,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-content-addressed-state-t1.json',
    'reports/evidence/p7-content-addressed-state-t1-repeat.json'
  )
),
layouts AS (
  SELECT
    run_order,
    run_label,
    0 AS layout_order,
    'Replicated' AS layout,
    json_extract(doc, '$.replicated') AS layout_doc
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    1,
    'Content addressed',
    json_extract(doc, '$.content_addressed')
  FROM runs
)
SELECT
  run_label,
  layout,
  json_extract(
    layout_doc,
    '$.worker_replacement.persisted_result_survived_loss'
  ) AS persisted_result_survived_loss,
  json_extract(
    layout_doc,
    '$.worker_replacement.recomputed_persisted_result'
  ) AS recomputed_persisted_result,
  json_extract(
    layout_doc,
    '$.coordinator_recovery.new_process_loaded_only_persisted_state'
  ) AS fresh_process_only_persisted_state,
  json_extract(
    layout_doc,
    '$.coordinator_recovery.recovered_from_published_checkpoint'
  ) AS recovered_from_checkpoint,
  json_extract(
    layout_doc,
    '$.coordinator_recovery.duplicate_apply_rejected'
  ) AS duplicate_apply_rejected,
  json_extract(
    layout_doc,
    '$.coordinator_recovery.recovery_process_exit_code'
  ) AS recovery_process_exit_code,
  json_extract(layout_doc, '$.coordinator_recovery.recovery_seconds')
    AS recovery_seconds
FROM layouts
ORDER BY run_order, layout_order;
