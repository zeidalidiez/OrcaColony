WITH evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-authenticated-process-t1.json'
),
traffic AS (
  SELECT
    0 AS cache_order,
    'Cold' AS cache_state,
    0 AS topology_order,
    'Matched full process' AS topology,
    json_extract(doc, '$.full_cold_tensor_wire_bytes')
      AS tensor_wire_bytes,
    json_extract(doc, '$.full_cold_application_wire_bytes')
      - json_extract(doc, '$.full_cold_tensor_wire_bytes')
      AS control_json_wire_bytes,
    json_extract(doc, '$.full_cold_application_wire_bytes')
      AS application_wire_bytes,
    0.0 AS relative_change
  FROM evidence

  UNION ALL

  SELECT
    0,
    'Cold',
    1,
    'Four expert processes',
    json_extract(doc, '$.expert_cold_tensor_wire_bytes'),
    json_extract(doc, '$.expert_cold_application_wire_bytes')
      - json_extract(doc, '$.expert_cold_tensor_wire_bytes'),
    json_extract(doc, '$.expert_cold_application_wire_bytes'),
    json_extract(doc, '$.cold_application_wire_relative_change')
  FROM evidence

  UNION ALL

  SELECT
    1,
    'Warm',
    0,
    'Matched full process',
    json_extract(doc, '$.full_warm_tensor_wire_bytes'),
    json_extract(doc, '$.full_warm_application_wire_bytes')
      - json_extract(doc, '$.full_warm_tensor_wire_bytes'),
    json_extract(doc, '$.full_warm_application_wire_bytes'),
    0.0
  FROM evidence

  UNION ALL

  SELECT
    1,
    'Warm',
    1,
    'Four expert processes',
    json_extract(doc, '$.expert_warm_tensor_wire_bytes'),
    json_extract(doc, '$.expert_warm_application_wire_bytes')
      - json_extract(doc, '$.expert_warm_tensor_wire_bytes'),
    json_extract(doc, '$.expert_warm_application_wire_bytes'),
    json_extract(doc, '$.warm_application_wire_relative_change')
  FROM evidence
)
SELECT
  cache_state,
  topology,
  tensor_wire_bytes,
  control_json_wire_bytes,
  application_wire_bytes,
  relative_change
FROM traffic
ORDER BY cache_order, topology_order;
