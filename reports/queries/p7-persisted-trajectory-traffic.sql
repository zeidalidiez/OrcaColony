WITH evidence AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-persisted-trajectory-t1.json'
),
accounting AS (
  SELECT
    0 AS topology_order,
    'Matched full process' AS topology,
    json_extract(doc, '$.full_tensor_wire_bytes') AS tensor_wire_bytes,
    json_extract(doc, '$.full_control_json_wire_bytes')
      AS control_json_wire_bytes,
    json_extract(doc, '$.full_persisted_bytes') AS persisted_bytes,
    0.0 AS tensor_change_vs_full,
    0.0 AS persisted_change_vs_full
  FROM evidence

  UNION ALL

  SELECT
    1,
    'Pooled expert executor',
    json_extract(doc, '$.expert_tensor_wire_bytes'),
    json_extract(doc, '$.expert_control_json_wire_bytes'),
    json_extract(doc, '$.expert_persisted_bytes'),
    1.0 * json_extract(doc, '$.expert_tensor_wire_bytes')
      / json_extract(doc, '$.full_tensor_wire_bytes') - 1.0,
    1.0 * json_extract(doc, '$.expert_persisted_bytes')
      / json_extract(doc, '$.full_persisted_bytes') - 1.0
  FROM evidence
)
SELECT
  topology,
  tensor_wire_bytes,
  control_json_wire_bytes,
  persisted_bytes,
  tensor_change_vs_full,
  persisted_change_vs_full
FROM accounting
ORDER BY topology_order;
