# Record Patch v1

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

## Frozen promotion gates

- Primary exact match must be at least `0.70`.
- It must improve by at least `0.20` over the exact T2 initialization.
- Valid JSON rate must be at least `0.95`.
- Canonical JSON rate must be at least `0.90`.
- Exact match across the four single-operation buckets must be at least `0.80`.

These values are hypotheses, not evidence. Failure or an inconclusive result
will be published as such.

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

No Hugging Face model or dataset repository should be created from this task
until the frozen artifacts and baseline have passed review. Creating a
destination is not evidence that a useful model exists.

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

## Required evidence after training

The capability report must publish sample-level outputs, bucketed regressions,
language-loss and behavioral trajectories, gradient and update diagnostics,
duplicate and nearest-training-record checks, contributor-approved credit,
environment identity, exact files and checksums, and both Hugging Face commit
revisions. It must distinguish observations from causal explanations.
