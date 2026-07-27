# Task benchmark training report template

Use one copy of this template for every experimental, candidate, or promoted
task model. The implementation may store `capability_candidate` or
`capability_model` as a release classification; those labels mean performance
on the declared use case only. They do not imply general intelligence. Do not
replace failed or inconclusive results with a success narrative.

## Claim and disposition

- Campaign and selected checkpoint:
- Release classification:
- Narrow use case and falsifiable task claim:
- Frozen baseline ID and revision:
- Frozen behavioral suite, dataset revision, evaluator revision, and split:
- Primary metric, absolute threshold, minimum baseline improvement, and result:
- Guardrail results:

State the answer first. Explain whether the evidence supports the claim and what
it does not establish.

## What training changed

Compare initialization, the frozen baseline, intermediate checkpoints, and the
selected checkpoint:

- language-loss and behavioral-metric trajectories;
- sample-level output deltas, with stable non-personal example IDs;
- improvements and regressions by frozen data bucket;
- an explicit error taxonomy with counts and representative examples;
- gradient norms, clipping, update norms, adapter or parameter movement, and
  optimizer anomalies;
- memorization, contamination, duplication, and nearest-training-example checks;
- unrelated-task, safety, formatting, and forgetting guardrails;
- ablations and repeated-seed uncertainty where the claim depends on them.

Never infer a causal training explanation from a metric movement alone. Mark
observed associations, controlled comparisons, and hypotheses separately.

## Data and compute

Record exact training/evaluation dataset revisions, selection and transformation
rules, tokenizer revision, model configuration, objective/loss mask, steps,
tokens, numerical profile, execution topology, wall time, accepted worker time,
and public contributor-approved hardware classes.

Link `CONTRIBUTORS.md` and `attribution-snapshot.json`. Credit only accepted work
and follow each contributor's named, pseudonymous, anonymous, totals, profile,
and hardware preferences.

## Evidence and reproduction

The Hugging Face benchmark, model card, evaluator, and exact score files are the
public verification surface. The human-readable report is the agent's
interpretation of that record. Link or bundle:

- campaign, dataset, tokenizer, checkpoint, optimizer, and evaluation manifests;
- sample-level behavioral results and the frozen evaluator source;
- the exact reproduction command and captured `environment.json`;
- `SHA256SUMS`, source Git commit, and any unresolved external artifact;
- model and dataset Hugging Face repository IDs and exact Hub commit revisions;
- known limitations, failed checks, reviewer notes, and open questions.

A hash proves artifact identity, not scientific honesty. The report must leave
enough evidence for an interested reader to rerun the evaluator, inspect examples,
and challenge the interpretation.
