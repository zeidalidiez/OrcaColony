# Record Patch v1

This file preserves one completed owner-operated prototype and the exact choices
used to produce its retained evidence. Its task, metrics, thresholds, splits,
holdout policy, and proposed follow-up are not OrcaColony framework defaults and
do not select a future campaign.

## Question

Can the true 17,538,816-parameter T2 model learn to apply a short ordered patch
to a flat JSON record and return the exact canonical result?

This is a deliberately narrow structured-text task. It is useful as a proxy for
configuration edits, metadata updates, and tool arguments. It does not establish
general code editing, broad instruction following, arithmetic reasoning, or
safe autonomous tool use.

## Input and output

An input has one flat JSON record followed by one or more patch operations:

```text
record_patch_v1
record {"active":true,"color":"blue","owner":"mira"}
patch
SET color "red"
DELETE owner
SET priority 3
result
```

The only accepted output is the canonical one-line JSON result:

```json
{"active":true,"color":"red","priority":3}
```

Values are JSON strings, integers, booleans, or null. Nested objects and arrays
are outside v1.

## Operation semantics

- `SET key value` creates or replaces `key`.
- `DELETE key` removes `key`. Deleting an absent key is a no-op.
- `RENAME source target` moves the source value to `target` and removes
  `source`. An absent source is a no-op. If `target` exists, the source value
  replaces it.
- Operations run from top to bottom.
- Output keys are sorted lexicographically with no optional whitespace.
- A JSON parser that accepts duplicate object keys is not valid for this task.

The evaluator recomputes every expected result from the retained starting record
and operation list. It does not trust a stored target by itself.

## Behavioral buckets

The validation and final-holdout suites are balanced across:

1. set an existing field;
2. set a new field;
3. delete an existing field;
4. rename to a new field;
5. overwrite one field twice;
6. delete and recreate one field;
7. rename over an existing field;
8. apply a mixed four-operation sequence.

The primary metric is strict exact match after removing leading and trailing
whitespace. Component metrics retain JSON validity, semantic equality,
canonical serialization, single-operation accuracy, and per-bucket results.

## Historical experiment criteria

- Primary exact match must be at least `0.70`.
- It must improve by at least `0.20` over the exact T2 initialization.
- Valid JSON rate must be at least `0.95`.
- Canonical JSON rate must be at least `0.90`.
- Exact match across the four single-operation buckets must be at least `0.80`.

These values were hypotheses recorded for this experiment, not framework
requirements or evidence. The measured runs did not meet them.

## Data and license

All records, values, and operations are generated locally by
`src/orcacolony/record_patch.py`. The task contains no scraped text, personal
data, external corpus, or teacher-model output. The generated dataset is
released under CC0 1.0.

The training, language-validation, and behavioral-validation keys are public.
The final-holdout key and examples are stored under ignored `.artifacts/` until
checkpoint selection. The committed suite lock records their count and SHA-256
without revealing them.

Training and language-loss evaluation use the currently supported
`causal_lm`/`all_target_tokens` objective over complete prompt-and-answer
transcripts. Prompt tokens carry loss. This is not target-only supervised
fine-tuning, and reports must retain that limitation.

## Model and release destination

- Architecture: true T2, 8 layers, width 384, 6 heads, MLP width 1536.
- Vocabulary and context: byte-level BPE up to 8,192 entries, 512 tokens.
- Exact parameter count: 17,538,816.
- Model repository: `OrcaColony/record-patch-t2-v1`.
- Dataset repository: `OrcaColony/record-patch-v1`.
- Model-weights license: Apache-2.0.
- Dataset license: CC0-1.0.
- Visibility: private review before any separate public release.

No Hugging Face model or dataset repository has been created from this
prototype. A future publication is an explicit owner decision and may preserve
the negative result honestly; publication does not imply that the model is
useful.

## Bounded learnability protocol

[`learnability-protocol.json`](learnability-protocol.json) fixes the first
centralized check before its result is known. It evaluates checkpoints at steps
`0`, `1`, `8`, `32`, and `128`. Checkpoint selection uses only the lowest public
language-validation mean loss. Public behavioral validation is recorded at
every milestone but cannot select the checkpoint.

The pre-volunteer gate requires both:

- at least `0.1` lower language-validation mean loss than initialization; and
- at least one additional exact match on the 32 public behavioral cases at the
  language-selected checkpoint.

The run is limited to one CPU thread, 128 updates, and 3 GiB peak process RSS.
It records every step's training loss, pre-clipping gradient norm, clipping
decision, parameter-update norm, checkpoint digest, evaluation, timing, and
environment. This is owner-operated qualification work, not donated compute.
Passing permits planning a volunteer run; it is not capability promotion.

## Commands

Freeze the task, private holdout, packed causal dataset, campaign, and exact
initialization identity:

```bash
uv run python -m orcacolony.record_patch freeze \
  --public-dir capability/record-patch-v1 \
  --private-dir .artifacts/record-patch-v1 \
  --campaign campaign/record-patch-t2-v1.json
```

Run the initialization baseline on behavioral validation only:

```bash
uv run python -m orcacolony.record_patch baseline \
  --campaign campaign/record-patch-t2-v1.json \
  --packed-dir .artifacts/record-patch-v1/packed \
  --public-dir capability/record-patch-v1 \
  --output .artifacts/record-patch-t2-baseline
```

The final holdout is intentionally not an argument to that command.

Run the predeclared bounded learnability check:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
uv run python -m orcacolony.record_patch_learnability \
  --protocol capability/record-patch-v1/learnability-protocol.json \
  --campaign campaign/record-patch-t2-v1.json \
  --packed-dir .artifacts/record-patch-v1/packed \
  --public-dir capability/record-patch-v1 \
  --output .artifacts/record-patch-t2-learnability-v1
```

The learnability command has no final-holdout or holdout-key argument.

The first measured run scored `0/32` exact and `2/32` valid JSON. It is retained
in [`../../reports/record-patch-t2-baseline.html`](../../reports/record-patch-t2-baseline.html).
Python 3.11.15 reproduced the initialization identity, all 32 predictions, and
the complete evaluation byte for byte from the earlier Python 3.14.4 run. Both
runs used CPU-only PyTorch 2.13.0. A fresh Python 3.11 freeze also reproduced
the public suite, private holdout digest, source corpora, packed dataset,
campaign bytes, and initialization identity. The final holdout remained
unopened by evaluation.

## Measured qualification result

The 128-step check did not pass. Public language-validation mean loss improved
from `9.120742341162453` to `1.56358497243532`, so step 128 was selected without
using behavior. It remained `0/32` exact, `0/32` semantic, and `0/32` strict
valid JSON. Every output had become object-shaped, but all 32 repeated at least
one key.

The run covered 512 of 4,618 packed training sequences, about `0.111` epochs.
It clipped 114 of 128 updates. Public answer-token loss improved from `9.1174`
to `2.0935`, but teacher-forced answer-token accuracy was only `40.7%` and no
complete teacher-forced answer was correct. None of the 32 public prompts or
targets occurred exactly in training.

This was a negative owner-operated result. Under the prototype's protocol, no
donated compute followed it. The continuation control was frozen in
[`continuation-protocol.json`](continuation-protocol.json). It binds the exact
step-128 model, AdamW state, data cursor, objective, learning rate, parent
evidence, and decoding policy. It evaluates total trajectory steps `128`,
`256`, and `512`, selects only by public language loss, and applies the same
language-plus-exact-behavior gate. Step 512 covers about `0.443` packed-data
epochs. This control changes exposure only. A target-only objective or
different learning rate remains a separate later experiment. See
[`../../reports/record-patch-t2-learnability-v1.html`](../../reports/record-patch-t2-learnability-v1.html).

Run the committed continuation from the retained parent run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
uv run python -m orcacolony.record_patch_continuation \
  --protocol capability/record-patch-v1/continuation-protocol.json \
  --campaign campaign/record-patch-t2-v1.json \
  --packed-dir .artifacts/record-patch-v1/packed \
  --public-dir capability/record-patch-v1 \
  --parent-run .artifacts/record-patch-t2-learnability-v1 \
  --output .artifacts/record-patch-t2-continuation-v1
```

The continuation command has no final-holdout or holdout-key argument. It
verifies the full parent checksum manifest before loading the resume checkpoint
and writes self-contained checkpoints and checksums for the new run.

## Measured continuation result

The continuation did not pass the task gate. Public language mean loss fell to
`1.3693936844946633` at step 256 and `1.2297416774319931` at step 512, so step
512 was selected without using task behavior. Strict canonical JSON rose from
`0/32` at step 128 to `28/32` at step 256 and `30/32` at step 512. Exact and
semantic record matches remained `0/32` at every checkpoint.

The selected checkpoint reproduced only 6 expected key-value pairs across the
public records and retained only 3 unchanged key-value pairs. Teacher-forced
answer-token accuracy reached `45.0%`, but no complete teacher-forced answer
was correct. Prompt loss continued to improve while answer loss was worse at
step 512 than step 256. The measured pattern is improved transcript and JSON
form without learned record transformation.

This evaluation measures only the Record Patch scenario. It does not claim to
measure general intelligence. The original analysis proposed an answer-token
loss comparison, but that proposal is not the project's next campaign unless a
campaign owner explicitly chooses it. Both reserved holdouts remained closed.

## Public experiment and campaign records

The human-readable page should state the agent's findings and separate
observations from hypotheses. If this experiment is published on Hugging Face,
its record should contain the exact model and data revisions, runnable
evaluator, decoding settings, score files, sample outputs, and limitations. A
separate benchmark product is not implied.

When a community campaign occurs, its release must separately include
contributor-approved credit, accepted-work totals, and opted-in hardware
details. The current owner-operated qualification run is not a community
campaign and claims no community contribution.
