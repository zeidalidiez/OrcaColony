WITH evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-authenticated-process-t1.json'
)
SELECT
  1.0
    - 1.0 * json_extract(doc, '$.expert_warm_tensor_wire_bytes')
      / json_extract(doc, '$.full_warm_tensor_wire_bytes')
    AS warm_tensor_reduction,
  1.0
    - 1.0 * json_extract(doc, '$.expert_warm_application_wire_bytes')
      / json_extract(doc, '$.full_warm_application_wire_bytes')
    AS warm_application_reduction,
  1.0 * json_extract(doc, '$.expert_cold_tensor_wire_bytes')
      / json_extract(doc, '$.full_cold_tensor_wire_bytes')
    - 1.0
    AS cold_tensor_change,
  1.0 * json_extract(doc, '$.expert_cold_application_wire_bytes')
      / json_extract(doc, '$.full_cold_application_wire_bytes')
    - 1.0
    AS cold_application_change,
  (
    SELECT max(
      max(
        json_extract(value, '$.full_max_abs_raw_gradient_difference'),
        json_extract(value, '$.full_max_abs_clipped_gradient_difference'),
        json_extract(value, '$.full_max_abs_model_difference'),
        json_extract(value, '$.expert_max_abs_raw_gradient_difference'),
        json_extract(value, '$.expert_max_abs_clipped_gradient_difference'),
        json_extract(value, '$.expert_max_abs_model_difference')
      )
    )
    FROM json_each(doc, '$.assignments')
  ) AS max_state_difference,
  json_extract(
    doc,
    '$.recovery.replacement_result_matches_stable'
  ) AS recovery_result_exact,
  json_extract(
    doc,
    '$.recovery.recovery_total_application_wire_bytes'
  ) AS recovery_total_application_wire_bytes
FROM evidence;
