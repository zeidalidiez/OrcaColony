# OrcaColony human-readable reports

This directory is the human-readable findings layer for OrcaColony. It is separate from `research/`, which remains the canonical machine-readable study, experiment, evidence, and reproduction-record system.

Open [`index.html`](index.html) in a browser to browse published reports.

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
