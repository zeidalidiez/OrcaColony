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
    doc,
    0 AS layout_order,
    'Replicated' AS layout,
    '$.replicated' AS layout_path
  FROM runs

  UNION ALL

  SELECT
    run_order,
    run_label,
    doc,
    1,
    'Content addressed',
    '$.content_addressed'
  FROM runs
),
step_rows AS (
  SELECT
    run_order,
    run_label,
    layout_order,
    layout,
    doc,
    step.value AS step_doc
  FROM layouts
  CROSS JOIN json_each(json_extract(doc, layout_path || '.steps')) AS step
)
SELECT
  run_label,
  layout,
  json_extract(step_doc, '$.step') AS step,
  json_extract(step_doc, '$.cursor') AS cursor,
  CASE
    WHEN json_extract(step_doc, '$.centralized_raw_gradient_sha256')
      = json_extract(step_doc, '$.full_process_raw_gradient_sha256')
    AND json_extract(step_doc, '$.centralized_raw_gradient_sha256')
      = json_extract(step_doc, '$.expert_process_raw_gradient_sha256')
    AND json_extract(step_doc, '$.centralized_clipped_gradient_sha256')
      = json_extract(step_doc, '$.full_process_clipped_gradient_sha256')
    AND json_extract(step_doc, '$.centralized_clipped_gradient_sha256')
      = json_extract(step_doc, '$.expert_process_clipped_gradient_sha256')
    AND json_extract(step_doc, '$.centralized_optimizer_sha256')
      = json_extract(step_doc, '$.full_process_optimizer_sha256')
    AND json_extract(step_doc, '$.centralized_optimizer_sha256')
      = json_extract(step_doc, '$.expert_process_optimizer_sha256')
    AND json_extract(step_doc, '$.centralized_model_sha256')
      = json_extract(step_doc, '$.full_process_model_sha256')
    AND json_extract(step_doc, '$.centralized_model_sha256')
      = json_extract(step_doc, '$.expert_process_model_sha256')
    AND json_extract(step_doc, '$.centralized_loss')
      = json_extract(step_doc, '$.full_process_loss')
    AND json_extract(step_doc, '$.centralized_loss')
      = json_extract(step_doc, '$.expert_process_loss')
    AND json_extract(step_doc, '$.full_max_abs_raw_gradient_difference') = 0.0
    AND json_extract(step_doc, '$.expert_max_abs_raw_gradient_difference') = 0.0
    AND json_extract(
      step_doc,
      '$.full_max_abs_clipped_gradient_difference'
    ) = 0.0
    AND json_extract(
      step_doc,
      '$.expert_max_abs_clipped_gradient_difference'
    ) = 0.0
    AND json_extract(step_doc, '$.full_max_abs_model_difference') = 0.0
    AND json_extract(step_doc, '$.expert_max_abs_model_difference') = 0.0
    THEN 1
    ELSE 0
  END AS exact,
  json_extract(doc, '$.transaction_ids_match') AS transaction_ids_match
FROM step_rows
ORDER BY run_order, layout_order, step;
