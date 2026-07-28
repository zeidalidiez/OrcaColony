WITH evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-sparse-expert-cached-head-t0.json'
),
traffic AS (
  SELECT
    'Cold' AS cache_state,
    'Matched full' AS topology,
    json_extract(doc, '$.full_payload_tensor_bytes')
      AS model_download_bytes,
    json_extract(doc, '$.full_gradient_upload_tensor_bytes')
      AS gradient_upload_bytes,
    json_extract(doc, '$.full_input_tensor_bytes')
      AS input_boundary_bytes,
    json_extract(doc, '$.full_cold_round_trip_tensor_bytes')
      AS round_trip_bytes,
    0.0 AS relative_change
  FROM evidence

  UNION ALL

  SELECT
    'Cold',
    'Four experts',
    json_extract(doc, '$.cold_aggregate_payload_tensor_bytes'),
    json_extract(doc, '$.aggregate_gradient_upload_tensor_bytes'),
    json_extract(doc, '$.aggregate_input_tensor_bytes')
      + json_extract(doc, '$.aggregate_input_adjoint_tensor_bytes'),
    json_extract(doc, '$.cold_aggregate_round_trip_tensor_bytes'),
    json_extract(doc, '$.cold_aggregate_round_trip_relative_change')
  FROM evidence

  UNION ALL

  SELECT
    'Warm',
    'Matched full',
    json_extract(doc, '$.full_warm_payload_tensor_bytes'),
    json_extract(doc, '$.full_gradient_upload_tensor_bytes'),
    json_extract(doc, '$.full_input_tensor_bytes'),
    json_extract(doc, '$.full_warm_round_trip_tensor_bytes'),
    0.0
  FROM evidence

  UNION ALL

  SELECT
    'Warm',
    'Four experts',
    json_extract(doc, '$.warm_aggregate_payload_tensor_bytes'),
    json_extract(doc, '$.aggregate_gradient_upload_tensor_bytes'),
    json_extract(doc, '$.aggregate_input_tensor_bytes')
      + json_extract(doc, '$.aggregate_input_adjoint_tensor_bytes'),
    json_extract(doc, '$.warm_aggregate_round_trip_tensor_bytes'),
    json_extract(doc, '$.warm_aggregate_round_trip_relative_change')
  FROM evidence
)
SELECT
  cache_state,
  topology,
  model_download_bytes,
  gradient_upload_bytes,
  input_boundary_bytes,
  round_trip_bytes,
  relative_change
FROM traffic
ORDER BY
  CASE cache_state WHEN 'Cold' THEN 0 ELSE 1 END,
  CASE topology WHEN 'Matched full' THEN 0 ELSE 1 END;
