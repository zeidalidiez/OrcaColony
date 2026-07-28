# OrcaColony human-readable reports

This directory is the human-readable findings layer for OrcaColony. It is separate from `research/`, which remains the canonical machine-readable study, experiment, evidence, and reproduction-record system.

Open [`index.html`](index.html) in a browser to browse published reports.

## Reporting boundary

These pages are agent-authored research findings. They state what was run, what
was observed, and how the reporting agent interprets the supplied evidence.
They are not an independent certification of model quality and must not fill in
campaign choices that the owner did not make.

For a campaign result, the public research record belongs with the exact
Hugging Face model and data revisions, the campaign-owner-defined evaluator,
score files, bundled evidence, limitations, accepted-work records, and
contributor-approved direct and auxiliary credit. Auxiliary work remains
separate from accepted optimizer work and links its own evidence identities.
Local report checksums make the findings auditable, but do not make the
reporting agent the owner of the campaign.

## Two parallel tracks

1. **Method engineering (P5–P7)** continues through bounded, runnable experiments. It does not wait for time-consuming human campaign review.
2. **Practical campaigns** remain owner-directed. The campaign owner chooses the use case, data, examples, rubric, checkpoint interpretation, decision criteria, and next campaign.

Automation may collect measurements, verify revisions, render owner-requested
comparisons, and preserve reproduction commands. It must not silently choose a
task, metric, threshold, model decision, or next experiment.

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

The current P7 systems finding is
[Report 013](p7-frozen-cached-head-t0.html). Its canonical report artifact,
source SQL, primary and repeat evidence, and linked machine-readable study are
committed beside the report. It is a frozen-head tensor-accounting result, not
a practical model evaluation.

## Practical campaign review points

Human-directed campaigns add reports for:

- data and split review;
- baseline outputs and error taxonomy;
- checkpoint comparisons;
- final holdout results;
- approved findings and next-iteration recommendations.

P5–P7 systems reports may be published before those campaign reviews are complete, but must label systems evidence separately from practical model-quality evidence.

Start each new campaign report from
[`CAMPAIGN_REPORT_TEMPLATE.md`](CAMPAIGN_REPORT_TEMPLATE.md). It requires the
reporting agent to identify owner-defined fields, measured comparisons,
training diagnostics, contributor-approved attribution, exact artifacts,
environment capture, and Hub commit revisions without filling in missing
campaign decisions.

The older
[`CAPABILITY_REPORT_TEMPLATE.md`](CAPABILITY_REPORT_TEMPLATE.md) filename is
retained only for historical Record Patch links.

The retained Record Patch initialization record is
[`record-patch-t2-baseline.html`](record-patch-t2-baseline.html). It retains the
zero-exact-match initialization result, sample outputs, withheld-holdout
boundary, and byte-identical reproduction under the supported Python 3.11
runtime before any training.

The first Record Patch training-effect record is
[`record-patch-t2-learnability-v1.html`](record-patch-t2-learnability-v1.html).
It preserves the failed 128-step experiment criterion, all checkpoint
outputs, prompt and answer diagnostics, optimizer measurements, train-to-public
similarity checks, resource use, and the subsequent same-trajectory
continuation.

The continuation findings are
[`record-patch-t2-continuation-v1.html`](record-patch-t2-continuation-v1.html).
They record the rise to `30/32` strict canonical JSON alongside `0/32` exact
and semantic record matches and distinguish the experiment's usage metric from
language diagnostics. They do not select the project's next campaign.
