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
- capability studies with fixed baselines, separate validation and final
  holdouts, sample-level evidence, and reproducible model and dataset releases.

## Current status

The training system has passed local correctness, restart, recovery, and
resource studies. The first capability study is Record Patch v1, a frozen
17.5M-parameter task with generated CC0 data, a public validation suite, a
separately keyed final holdout, an exact evaluator, and an Apache-2.0 model
release target.

Its random initialization solved 0 of 32 public cases. Python 3.11 reproduced
the initialization, every prediction, and the evaluation exactly. The first
bounded centralized learning check reduced public language loss from 9.1207
to 1.5636, but the selected checkpoint still solved 0 of 32 public behavioral
cases. All 32 generated answers used repeated JSON object keys and failed
strict parsing. The predeclared learning gate therefore failed.

Neither reserved final holdout was opened, and no donated compute was used.
The exact step-128 model, optimizer state, and data cursor were retained for a
same-trajectory continuation. Public model and dataset repositories remain
blocked until a predeclared check demonstrates behavioral improvement.

Negative and inconclusive results stay in the record. Training loss is treated
as a diagnostic, not proof that a model became more useful.

## Releases and credit

Each model release will include the exact campaign, data revision, tokenizer,
checkpoint, optimizer state, evaluations, source revision, checksums,
limitations, and linked report.

People who donate accepted training work choose named, pseudonymous, or
anonymous credit. Releases record contributor-approved totals and hardware
details without publishing worker credentials or private identifiers.

- [Source code](https://github.com/zeidalidiez/OrcaColony)
- [Progress report](https://github.com/zeidalidiez/OrcaColony/blob/main/PROGRESS_REPORT.md)
- [Research records](https://github.com/zeidalidiez/OrcaColony/tree/main/research)
- [Human-readable reports](https://github.com/zeidalidiez/OrcaColony/tree/main/reports)
- [Record Patch learnability report](https://github.com/zeidalidiez/OrcaColony/blob/main/reports/record-patch-t2-learnability-v1.html)
