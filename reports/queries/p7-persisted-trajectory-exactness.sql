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
steps AS (
  SELECT
    run_order,
    run_label,
    json_extract(value, '$.step') AS step,
    json_extract(value, '$.cursor') AS cursor,
    json_extract(value, '$.routing_counts') AS routing_counts,
    json_extract(value, '$.capacity_rerouted_tokens') AS rerouted_tokens,
    json_extract(value, '$.routes_changed_from_previous') AS routes_changed,
    json_extract(value, '$.centralized_loss') AS training_batch_loss,
    max(
      json_extract(value, '$.full_max_abs_raw_gradient_difference'),
      json_extract(value, '$.expert_max_abs_raw_gradient_difference')
    ) AS max_raw_gradient_difference,
    max(
      json_extract(value, '$.full_max_abs_clipped_gradient_difference'),
      json_extract(value, '$.expert_max_abs_clipped_gradient_difference')
    ) AS max_clipped_gradient_difference,
    max(
      json_extract(value, '$.full_max_abs_model_difference'),
      json_extract(value, '$.expert_max_abs_model_difference')
    ) AS max_model_difference,
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
      THEN 1
      ELSE 0
    END AS state_hashes_match
  FROM runs, json_each(runs.doc, '$.steps')
)
SELECT
  run_label,
  step,
  cursor,
  routing_counts,
  rerouted_tokens,
  routes_changed,
  training_batch_loss,
  max_raw_gradient_difference,
  max_clipped_gradient_difference,
  max_model_difference,
  state_hashes_match
FROM steps
ORDER BY run_order, step;
