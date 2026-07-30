WITH primary_run AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-content-addressed-state-t1.json'
),
repeat_run AS (
  SELECT json(document) AS doc
  FROM report_source_documents
  WHERE source_path = 'reports/evidence/p7-content-addressed-state-t1-repeat.json'
),
storage_rows AS (
  SELECT
    0 AS topology_order,
    0 AS layout_order,
    'Full process' AS topology,
    'Replicated' AS layout,
    json_extract(p.doc, '$.full_storage.replicated_persisted_bytes')
      AS persisted_bytes,
    0 AS reduction_bytes,
    0.0 AS reduction_fraction,
    CASE
      WHEN json_extract(p.doc, '$.full_storage.replicated_persisted_bytes')
        = json_extract(r.doc, '$.full_storage.replicated_persisted_bytes')
      THEN 1 ELSE 0
    END AS repeat_matches
  FROM primary_run p CROSS JOIN repeat_run r

  UNION ALL

  SELECT
    0,
    1,
    'Full process',
    'Content addressed',
    json_extract(p.doc, '$.full_storage.content_addressed_persisted_bytes'),
    json_extract(p.doc, '$.full_storage.persisted_byte_reduction'),
    json_extract(p.doc, '$.full_storage.persisted_byte_reduction_fraction'),
    CASE
      WHEN json_extract(
        p.doc,
        '$.full_storage.content_addressed_persisted_bytes'
      ) = json_extract(
        r.doc,
        '$.full_storage.content_addressed_persisted_bytes'
      )
      THEN 1 ELSE 0
    END
  FROM primary_run p CROSS JOIN repeat_run r

  UNION ALL

  SELECT
    1,
    0,
    'Pooled expert',
    'Replicated',
    json_extract(p.doc, '$.expert_storage.replicated_persisted_bytes'),
    0,
    0.0,
    CASE
      WHEN json_extract(p.doc, '$.expert_storage.replicated_persisted_bytes')
        = json_extract(r.doc, '$.expert_storage.replicated_persisted_bytes')
      THEN 1 ELSE 0
    END
  FROM primary_run p CROSS JOIN repeat_run r

  UNION ALL

  SELECT
    1,
    1,
    'Pooled expert',
    'Content addressed',
    json_extract(p.doc, '$.expert_storage.content_addressed_persisted_bytes'),
    json_extract(p.doc, '$.expert_storage.persisted_byte_reduction'),
    json_extract(p.doc, '$.expert_storage.persisted_byte_reduction_fraction'),
    CASE
      WHEN json_extract(
        p.doc,
        '$.expert_storage.content_addressed_persisted_bytes'
      ) = json_extract(
        r.doc,
        '$.expert_storage.content_addressed_persisted_bytes'
      )
      THEN 1 ELSE 0
    END
  FROM primary_run p CROSS JOIN repeat_run r
)
SELECT
  topology,
  layout,
  persisted_bytes,
  reduction_bytes,
  reduction_fraction,
  repeat_matches
FROM storage_rows
ORDER BY topology_order, layout_order;
