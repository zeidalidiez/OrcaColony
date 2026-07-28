WITH evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-sparse-expert-cached-head-t0.json'
)
SELECT
  1.0
    - 1.0 * json_extract(doc, '$.warm_aggregate_round_trip_tensor_bytes')
      / json_extract(doc, '$.full_warm_round_trip_tensor_bytes')
    AS warm_reduction,
  1.0 * json_extract(doc, '$.cold_aggregate_round_trip_tensor_bytes')
      / json_extract(doc, '$.full_cold_round_trip_tensor_bytes')
    - 1.0
    AS cold_change,
  max(
    json_extract(doc, '$.max_abs_raw_gradient_difference'),
    json_extract(doc, '$.max_abs_clipped_gradient_difference'),
    json_extract(doc, '$.max_abs_model_difference')
  ) AS max_state_difference,
  json_extract(doc, '$.frozen_head_gradient_tensor_bytes')
    AS head_gradient_bytes,
  json_extract(doc, '$.frozen_head_optimizer_state_parameter_count')
    AS head_optimizer_parameters
FROM evidence;
