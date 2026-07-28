WITH runs AS (
  SELECT
    CASE source_path
      WHEN 'reports/evidence/p7-authenticated-process-t1.json'
        THEN 0
      ELSE 1
    END AS run_order,
    CASE source_path
      WHEN 'reports/evidence/p7-authenticated-process-t1.json'
        THEN 'Primary'
      ELSE 'Repeat'
    END AS run_label,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-authenticated-process-t1.json',
    'reports/evidence/p7-authenticated-process-t1-repeat.json'
  )
)
SELECT
  run_label,
  json_extract(
    doc,
    '$.recovery.first_worker_exit_code'
  ) AS first_worker_exit_code,
  json_extract(
    doc,
    '$.recovery.replacement_worker_exit_code'
  ) AS replacement_worker_exit_code,
  json_extract(
    doc,
    '$.recovery.replacement_result_matches_stable'
  ) AS replacement_result_matches_stable,
  json_extract(
    doc,
    '$.recovery.lost_worker_received_tensor_wire_bytes'
  ) AS lost_worker_tensor_wire_bytes,
  json_extract(
    doc,
    '$.recovery.recovery_retransmitted_tensor_wire_bytes'
  ) AS retransmitted_tensor_wire_bytes,
  json_extract(
    doc,
    '$.recovery.replacement_result_tensor_wire_bytes'
  ) AS replacement_result_wire_bytes,
  json_extract(
    doc,
    '$.recovery.recovery_control_json_wire_bytes'
  ) AS recovery_control_json_wire_bytes,
  json_extract(
    doc,
    '$.recovery.recovery_total_application_wire_bytes'
  ) AS recovery_total_application_wire_bytes,
  json_extract(
    doc,
    '$.recovery.replacement_worker_initialization_seconds'
  ) AS replacement_initialization_seconds,
  json_extract(
    doc,
    '$.recovery.recovery_seconds'
  ) AS recovery_seconds
FROM runs
ORDER BY run_order;
