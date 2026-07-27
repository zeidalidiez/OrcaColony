---
title: README
colorFrom: gray
colorTo: blue
sdk: static
short_description: Volunteer training and reproducible small-model research.
pinned: false
---

# OrcaColony

OrcaColony studies how donated compute and careful experiments can improve
language models that are small enough to train and test outside frontier labs.

The project combines:

- a volunteer training framework with bounded work units, coordinator-owned
  optimization, restartable campaigns, and contributor-controlled credit;
- systems research on browser workers, CPU and GPU execution, PEFT, memory
  placement, partial-model methods, recovery, and sparse experts;
- owner-defined training campaigns with versioned usage evaluations,
  sample-level evidence, reproducible model and data releases, and
  contributor-controlled credit.

## Current status

The training system has passed local correctness, restart, recovery, and
resource studies. The campaign framework accepts a research question, usage
scenario, evaluator, inputs, metrics, comparisons, and analysis plan supplied by
the campaign owner. It preserves exact revisions, model evaluations, evidence
files, limitations, and contributor credit without assigning a mandatory
success or promotion state.

Record Patch v1 is retained as a historical owner-operated prototype. Its
17.5M-parameter model reached 30 of 32 strict canonical JSON outputs after 512
updates but remained at 0 of 32 exact and semantic record matches. No donated
compute was used. Its thresholds, holdouts, and proposed follow-up do not define
the next OrcaColony campaign.

Negative, unchanged, and inconclusive results can be released with the same
provenance and credit records as positive results. Training loss remains a
diagnostic, not proof that a model became more useful.

## Releases and credit

Each model release can include the exact campaign, data revision, tokenizer,
checkpoint, optimizer state, owner-defined evaluation records, bundled evidence
artifacts, source revision, checksums, limitations, and linked agent findings.

People who donate accepted training work choose named, pseudonymous, or
anonymous credit. Releases record contributor-approved totals and hardware
details without publishing worker credentials or private identifiers.

- [Source code](https://github.com/zeidalidiez/OrcaColony)
- [Progress report](https://github.com/zeidalidiez/OrcaColony/blob/main/PROGRESS_REPORT.md)
- [Research records](https://github.com/zeidalidiez/OrcaColony/tree/main/research)
- [Human-readable reports](https://github.com/zeidalidiez/OrcaColony/tree/main/reports)
- [Record Patch learnability report](https://github.com/zeidalidiez/OrcaColony/blob/main/reports/record-patch-t2-learnability-v1.html)
- [Record Patch continuation findings](https://github.com/zeidalidiez/OrcaColony/blob/main/reports/record-patch-t2-continuation-v1.html)
