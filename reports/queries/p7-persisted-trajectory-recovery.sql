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
)
SELECT
  run_label,
  json_extract(doc, '$.worker_replacement.step') AS worker_loss_step,
  json_extract(doc, '$.worker_replacement.first_worker_exit_code')
    AS first_worker_exit_code,
  json_extract(doc, '$.worker_replacement.persisted_result_index')
    AS persisted_result_index,
  json_extract(doc, '$.worker_replacement.persisted_result_survived_loss')
    AS persisted_result_survived_loss,
  json_extract(doc, '$.worker_replacement.recomputed_persisted_result')
    AS recomputed_persisted_result,
  json_extract(doc, '$.worker_replacement.replacement_worker_exit_code')
    AS replacement_worker_exit_code,
  json_extract(doc, '$.coordinator_recovery.step')
    AS coordinator_loss_step,
  json_extract(
    doc,
    '$.coordinator_recovery.applied_checkpoint_published_before_loss'
  ) AS checkpoint_published_before_loss,
  json_extract(doc, '$.coordinator_recovery.manifest_applied_before_loss')
    AS manifest_applied_before_loss,
  json_extract(
    doc,
    '$.coordinator_recovery.new_process_loaded_only_persisted_state'
  ) AS fresh_process_only_persisted_state,
  json_extract(
    doc,
    '$.coordinator_recovery.recomputed_from_persisted_pre_state_for_validation'
  ) AS recomputed_for_validation,
  json_extract(
    doc,
    '$.coordinator_recovery.recovered_from_published_checkpoint'
  ) AS recovered_from_checkpoint,
  json_extract(doc, '$.coordinator_recovery.duplicate_apply_rejected')
    AS duplicate_apply_rejected,
  json_extract(doc, '$.coordinator_recovery.recovery_process_exit_code')
    AS recovery_process_exit_code,
  json_extract(doc, '$.coordinator_recovery.recovery_seconds')
    AS recovery_seconds
FROM runs
ORDER BY run_order;
