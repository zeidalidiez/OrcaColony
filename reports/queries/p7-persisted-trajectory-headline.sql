WITH runs AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-persisted-trajectory-t1.json',
    'reports/evidence/p7-persisted-trajectory-t1-repeat.json'
  )
),
evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-persisted-trajectory-t1.json'
),
step_checks AS (
  SELECT
    CASE
      WHEN json_extract(value, '$.centralized_raw_gradient_sha256')
        = json_extract(value, '$.full_process_raw_gradient_sha256')
      AND json_extract(value, '$.centralized_raw_gradient_sha256')
        = json_extract(value, '$.expert_process_raw_gradient_sha256')
      AND json_extract(value, '$.centralized_clipped_gradient_sha256')
        = json_extract(value, '$.full_process_clipped_gradient_sha256')
      AND json_extract(value, '$.centralized_clipped_gradient_sha256')
        = json_extract(value, '$.expert_process_clipped_gradient_sha256')
      AND json_extract(value, '$.centralized_optimizer_sha256')
        = json_extract(value, '$.full_process_optimizer_sha256')
      AND json_extract(value, '$.centralized_optimizer_sha256')
        = json_extract(value, '$.expert_process_optimizer_sha256')
      AND json_extract(value, '$.centralized_model_sha256')
        = json_extract(value, '$.full_process_model_sha256')
      AND json_extract(value, '$.centralized_model_sha256')
        = json_extract(value, '$.expert_process_model_sha256')
      THEN 1.0
      ELSE 0.0
    END AS exact_update
  FROM runs, json_each(doc, '$.steps')
),
memory AS (
  SELECT
    (
      SELECT max(
        json_extract(value, '$.external_rss.lifecycle_max_hwm_rss_bytes')
      )
      FROM json_each(doc, '$.full_workers')
    ) AS full_child_hwm_bytes,
    (
      SELECT max(
        json_extract(value, '$.external_rss.lifecycle_max_hwm_rss_bytes')
      )
      FROM json_each(doc, '$.expert_workers')
    ) AS expert_child_hwm_bytes
  FROM evidence
)
SELECT
  (SELECT avg(exact_update) FROM step_checks) AS exact_update_rate,
  (SELECT count(*) FROM step_checks) AS compared_run_steps,
  2 AS independent_runs,
  1.0
    - 1.0 * json_extract(doc, '$.expert_tensor_wire_bytes')
      / json_extract(doc, '$.full_tensor_wire_bytes')
    AS tensor_traffic_reduction,
  1.0
    - 1.0 * json_extract(doc, '$.expert_persisted_bytes')
      / json_extract(doc, '$.full_persisted_bytes')
    AS persisted_byte_reduction,
  1.0 * json_extract(doc, '$.expert_process_end_to_end_seconds')
      / json_extract(doc, '$.full_process_end_to_end_seconds')
    - 1.0
    AS expert_elapsed_change,
  json_extract(doc, '$.full_process_end_to_end_seconds')
    AS full_complete_seconds,
  json_extract(doc, '$.expert_process_end_to_end_seconds')
    AS expert_complete_seconds,
  1.0 - 1.0 * expert_child_hwm_bytes / full_child_hwm_bytes
    AS child_hwm_reduction,
  full_child_hwm_bytes,
  expert_child_hwm_bytes,
  json_extract(doc, '$.coordinator_recovery.recovery_seconds')
    AS coordinator_recovery_seconds
FROM evidence, memory;
