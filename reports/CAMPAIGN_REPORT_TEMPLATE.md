# Campaign research report template

Use this template after a campaign owner has defined and run a campaign. Do not
fill in a usage scenario, metric, threshold, conclusion, or next experiment on
the owner's behalf.

## Agent findings

Identify the campaign, exact released checkpoint, source revision, data
revision, evaluator revision, and evidence revision.

State that the section contains the reporting agent's findings. Summarize what
the supplied evidence shows, including improvements, regressions, unchanged
measurements, mixed results, and uncertainty. Do not convert an observed metric
change into a broader model claim.

## Owner-defined campaign

- Research question:
- Concrete usage scenario:
- Training objective and recipe:
- Evaluator and exact revision:
- Evaluation input artifacts and exact revisions:
- Declared metrics, units, and preferred directions:
- Owner-supplied decision criteria, if any:
- Evaluated model or checkpoint revisions:

If the owner did not define an item, mark it absent. Do not invent it.

## Evaluation records and comparisons

For each evaluated subject:

- identify the exact model or checkpoint;
- report every declared metric;
- link the sample-level or aggregate evidence artifacts;
- record decoding, prompting, tool, and environment settings that affect the
  result.

For each owner-requested comparison:

- show the two exact evaluations;
- show baseline value, candidate value, raw change, and declared direction;
- distinguish measured movement from the report author's interpretation;
- include representative improvements, regressions, and failures when the
  supplied artifacts support them.

## Training and systems observations

When available, report training loss, gradient norms, clipping, update norms,
parameter or adapter movement, optimizer anomalies, steps, tokens, numerical
profile, execution topology, wall time, accepted worker time, and aggregate
resource observations.

These are diagnostics. They do not replace the owner-defined usage evaluation.

## Data, provenance, and reproduction

Link or bundle:

- campaign, dataset, tokenizer, checkpoint, optimizer, and evaluator manifests;
- sample-level outputs and score files;
- exact reproduction commands and captured environment;
- source Git commit and Hugging Face commit revisions;
- `SHA256SUMS`;
- unresolved external artifacts, clearly labeled as unresolved;
- limitations, reviewer notes, and open questions.

A hash establishes identity. It does not establish that a metric is meaningful
or an interpretation is correct.

## Contributor credit

Link `CONTRIBUTORS.md` and `attribution-snapshot.json`. Credit accepted training
work according to each contributor's named, pseudonymous, anonymous, totals,
profile, and hardware preferences.

Do not imply that direct-training credit covers data work, evaluation design,
review, hosting, or other auxiliary contributions unless those contributions
were separately recorded and approved for publication.

## Owner decision

Leave the campaign decision to the campaign owner. Record it only after it has
been supplied:

- what the owner concludes from the evidence;
- whether any model will be published or tested further;
- what, if anything, the owner wants the next campaign to change.
