WITH runs AS (
  SELECT
    source_path,
    json(document) AS doc
  FROM report_source_documents
  WHERE source_path IN (
    'reports/evidence/p7-content-addressed-state-t1.json',
    'reports/evidence/p7-content-addressed-state-t1-repeat.json'
  )
),
run_layouts AS (
  SELECT source_path, doc, '$.replicated' AS layout_path
  FROM runs
  UNION ALL
  SELECT source_path, doc, '$.content_addressed'
  FROM runs
),
primary_run AS (
  SELECT doc
  FROM runs
  WHERE source_path = 'reports/evidence/p7-content-addressed-state-t1.json'
),
repeat_run AS (
  SELECT doc
  FROM runs
  WHERE source_path = 'reports/evidence/p7-content-addressed-state-t1-repeat.json'
)
SELECT
  (
    SELECT sum(
      CASE
        WHEN json_extract(doc, layout_path || '.all_steps_exact') = 1
        THEN json_array_length(json_extract(doc, layout_path || '.steps'))
        ELSE 0
      END
    )
    FROM run_layouts
  ) AS exact_layout_run_steps,
  (
    SELECT sum(json_array_length(json_extract(doc, layout_path || '.steps')))
    FROM run_layouts
  ) AS compared_layout_run_steps,
  2 AS independent_runs,
  json_extract(p.doc, '$.full_storage.replicated_persisted_bytes')
    AS full_replicated_bytes,
  json_extract(p.doc, '$.full_storage.content_addressed_persisted_bytes')
    AS full_content_addressed_bytes,
  json_extract(p.doc, '$.full_storage.persisted_byte_reduction')
    AS full_reduction_bytes,
  json_extract(p.doc, '$.full_storage.persisted_byte_reduction_fraction')
    AS full_reduction_fraction,
  json_extract(p.doc, '$.expert_storage.replicated_persisted_bytes')
    AS expert_replicated_bytes,
  json_extract(p.doc, '$.expert_storage.content_addressed_persisted_bytes')
    AS expert_content_addressed_bytes,
  json_extract(p.doc, '$.expert_storage.persisted_byte_reduction')
    AS expert_reduction_bytes,
  json_extract(p.doc, '$.expert_storage.persisted_byte_reduction_fraction')
    AS expert_reduction_fraction,
  json_extract(p.doc, '$.full_storage.replicated_persisted_bytes')
    + json_extract(p.doc, '$.expert_storage.replicated_persisted_bytes')
    AS combined_replicated_bytes,
  json_extract(p.doc, '$.full_storage.content_addressed_persisted_bytes')
    + json_extract(p.doc, '$.expert_storage.content_addressed_persisted_bytes')
    AS combined_content_addressed_bytes,
  json_extract(p.doc, '$.full_storage.persisted_byte_reduction')
    + json_extract(p.doc, '$.expert_storage.persisted_byte_reduction')
    AS combined_reduction_bytes,
  1.0 - 1.0 * (
    json_extract(p.doc, '$.full_storage.content_addressed_persisted_bytes')
      + json_extract(
        p.doc,
        '$.expert_storage.content_addressed_persisted_bytes'
      )
  ) / (
    json_extract(p.doc, '$.full_storage.replicated_persisted_bytes')
      + json_extract(p.doc, '$.expert_storage.replicated_persisted_bytes')
  ) AS combined_reduction_fraction,
  json_extract(p.doc, '$.full_storage.content_addressed_blob_count')
    AS unique_blobs_per_topology,
  json_extract(p.doc, '$.full_storage.checkpoint_reference_count')
    AS checkpoint_references_per_topology,
  json_extract(p.doc, '$.full_storage.reused_checkpoint_reference_count')
    AS reused_references_per_topology,
  CASE
    WHEN json_extract(p.doc, '$.full_storage')
      = json_extract(r.doc, '$.full_storage')
    AND json_extract(p.doc, '$.expert_storage')
      = json_extract(r.doc, '$.expert_storage')
    THEN 1
    ELSE 0
  END AS storage_repeated_exactly,
  CASE
    WHEN json_extract(p.doc, '$.all_steps_exact') = 1
    AND json_extract(r.doc, '$.all_steps_exact') = 1
    AND json_extract(p.doc, '$.transaction_ids_match') = 1
    AND json_extract(r.doc, '$.transaction_ids_match') = 1
    THEN 1
    ELSE 0
  END AS identity_guardrails_passed
FROM primary_run p
CROSS JOIN repeat_run r;
