# OrcaColony human-readable reports

This directory is the human-readable findings layer for OrcaColony. It is separate from `research/`, which remains the canonical machine-readable study, experiment, evidence, and reproduction-record system.

Open [`index.html`](index.html) in a browser to browse published reports.

## Reporting boundary

These pages are agent-authored research findings. They state what was run, what
was observed, how the agent interprets it, and what should be tested next. They
are not an independent certification of model quality.

For a task benchmark, the public verification record belongs with the
Hugging Face dataset, evaluator, model revision, and score files. For a
community campaign, accepted-work records and contributor-approved credit
belong with that campaign's release. Local report checksums make the findings
auditable, but do not turn a narrow task score into evidence of general
intelligence.

## Two parallel tracks

1. **Method engineering (P5–P7)** continues through bounded, runnable experiments. It does not wait for time-consuming human campaign review.
2. **Practical campaigns** remain human-directed. The campaign owner chooses the use case, data, examples, rubric, checkpoint interpretation, and promotion decision.

Automation may collect measurements, verify revisions, render comparisons, and preserve reproduction commands. It must not silently decide that a model is useful, promote a checkpoint, or replace human interpretation of examples.

## Report requirements

Every important report should answer, in plain language:

- What question was tested?
- What changed relative to the baseline?
- What improved, and by how much?
- What regressed or remained unresolved?
- Which examples or measurements support the conclusion?
- What does the evidence not prove?
- What should the next iteration change?

Reports are self-contained HTML with no remote scripts. Supporting measured JSON may live under `evidence/`. Reports should link the exact implementation commit and reproduction command when available.

## Practical campaign review points

Human-directed campaigns add reports for:

- data and split review;
- baseline outputs and error taxonomy;
- checkpoint comparisons;
- final holdout results;
- approved findings and next-iteration recommendations.

P5–P7 systems reports may be published before those campaign reviews are complete, but must label systems evidence separately from practical model-quality evidence.

Start each practical model report from
[`CAPABILITY_REPORT_TEMPLATE.md`](CAPABILITY_REPORT_TEMPLATE.md). It requires
baseline/checkpoint output deltas, bucketed errors, optimizer diagnostics,
memorization and forgetting checks, contributor-approved attribution, exact
artifacts, environment capture, and Hub commit revisions.

The existing code uses the term `capability` for its promotion records. In
reports, that term is limited to the declared use case and benchmark. A Record
Patch result supports only a claim about applying Record Patch operations.

The first practical task record is
[`record-patch-t2-baseline.html`](record-patch-t2-baseline.html). It retains the
zero-exact-match initialization result, sample outputs, withheld-holdout
boundary, and byte-identical reproduction under the supported Python 3.11
runtime before any training.

The first training-effect record is
[`record-patch-t2-learnability-v1.html`](record-patch-t2-learnability-v1.html).
It preserves the failed 128-step public learnability gate, all checkpoint
outputs, prompt and answer diagnostics, optimizer measurements, train-to-public
similarity checks, resource use, and the decision to keep volunteer training
blocked while a same-trajectory continuation is tested.
