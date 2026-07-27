# Campaign research framework

OrcaColony supplies the machinery for a campaign. It does not supply the
campaign's research question.

When a campaign is created, its owner decides:

- what model or checkpoint will be trained;
- the concrete usage scenario being investigated;
- the training data, objective, recipe, and stopping point;
- the evaluator and every evaluation input;
- which metrics to record and whether higher, lower, or neither direction is
  preferred;
- which model evaluations should be compared;
- any decision criteria the owner wants to apply;
- what the evidence supports and what should be tried next;
- the Hugging Face repository names, licenses, and visibility policy.

The framework validates and preserves those choices. It must not insert a task,
threshold, benchmark, holdout policy, checkpoint-selection rule, promotion
gate, or follow-up experiment that the campaign owner did not declare.

## Campaign contract

New research campaigns use `orcacolony_campaign_research_v2` in the campaign's
`research` field. Start from
[`../campaign/campaign-research-v2.example.json`](../campaign/campaign-research-v2.example.json).

The contract contains:

- `question` and `usage_scenario`, written by the campaign owner;
- a versioned evaluator and exact reproduction command;
- versioned evaluation inputs such as datasets, prompt sets, rubrics, or test
  harnesses;
- one or more numeric metrics with owner-selected `maximize`, `minimize`, or
  `observe` direction;
- an owner-written analysis plan.

The contract intentionally has no required threshold or success state.
Campaign-specific decision rules may live in the campaign's reviewed protocol
and report. They are evidence interpretation, not permission for the framework
to train, publish, or credit a result.

The older `orcacolony_capability_research_v1` format is retained only so the
Record Patch prototype and its exact historical artifacts remain loadable. It
must not be copied into a new campaign.

Validate a completed draft and print the exact campaign and research revisions
without creating or changing a campaign:

```bash
uv run python -m orcacolony.campaign_lifecycle inspect \
  --config campaign/<campaign>.json
```

The output repeats the owner-supplied evaluator, inputs, metrics, publication
settings, question, and usage scenario so they can be reviewed together with
the two content identities. It does not fill in missing choices.

## Evaluation evidence

After the campaign owner has run the evaluations they chose, use
`orcacolony_campaign_evaluation_evidence_v1`. Start from
[`../campaign/campaign-evaluation-evidence-v1.example.json`](../campaign/campaign-evaluation-evidence-v1.example.json).

Each evaluation record binds:

- a reader-facing ID and label;
- the exact evaluated subject and revision;
- one finite value for every metric declared by the campaign owner;
- one or more sample-level or aggregate evidence artifacts with SHA-256
  identities.

Comparisons name any two evaluation records. The framework calculates the raw
metric change and, when the owner declared `maximize` or `minimize`, the signed
change in that preferred direction. It does not label the campaign successful,
approve a model, or choose the next experiment.

The evaluation identified by `release_evaluation_id` must bind the exact
released checkpoint. This prevents a model card from presenting measurements
from different weights.

Preflight the completed evidence and its local `bundle:` files before building
an operational release:

```bash
uv run python -m orcacolony.campaign_lifecycle validate-evidence \
  --config campaign/<campaign>.json \
  --evidence <campaign-evaluation.json> \
  --evaluation-artifacts <evaluation-artifact-directory> \
  --release-checkpoint-sha256 <owner-selected-checkpoint-sha256>
```

This checks campaign and research identities, every declared measurement,
comparison, finding, limitation, release-checkpoint binding, and bundled file
digest. JSON with duplicate keys is rejected.

Artifact URIs beginning with `bundle:` are paths below the directory passed to
`orcacolony.release --evaluation-artifacts`. The release builder verifies their
hashes and copies them into `campaign-evaluation-artifacts/`. Other durable URIs
remain references and are labeled as such rather than being treated as locally
verified files.

## Release and publication

A completed campaign can be released whether its measurements improved,
regressed, or remained inconclusive. A failed optional training diagnostic is
recorded in the dashboard and release; it does not block packaging.

For a v2 campaign, the release builder does not select the lowest built-in
language-loss diagnostic. It uses the checkpoint named by
`release_evaluation_id` when that revision identifies exactly one built-in
evaluated checkpoint. Otherwise the owner must pass
`--release-checkpoint-step`. In both cases, supplied evaluation evidence must
bind the checkpoint bytes that are actually packaged. Legacy systems studies
retain their frozen historical selection behavior.

When evaluation evidence is supplied, the release contains:

- `campaign-evaluation-evidence.json`, the owner-supplied record;
- `campaign-evaluation-summary.json`, the validated evaluations and computed
  comparisons;
- `campaign-evaluation-artifacts/`, for every verified `bundle:` artifact;
- the exact model, optimizer state, campaign, data, tokenizer, source, and
  checksums;
- `attribution-snapshot.json` and `CONTRIBUTORS.md`, generated from accepted
  work and each contributor's public-credit choices.

The Hugging Face builder copies the evaluation record and bundled artifacts
into both the model and dataset packages. The model card reports the evidence
without turning metric movement into a framework decision.

Example release arguments:

```bash
uv run python -m orcacolony.release \
  --config campaign/<campaign>.json \
  --participants <private-participants.json> \
  --dataset-artifacts <dataset-directory> \
  --campaign-state <completed-campaign-state> \
  --browser-root spikes/burn-browser-gradient/www \
  --evaluation-evidence <campaign-evaluation.json> \
  --evaluation-artifacts <evaluation-artifact-directory> \
  --release-checkpoint-step <step-selected-by-campaign-owner> \
  --output .artifacts/<campaign>-release
```

`--release-checkpoint-step` is optional when the release evaluation identifies
exactly one built-in evaluated checkpoint.

Publication remains an explicit second step. See
[`../HUGGINGFACE.md`](../HUGGINGFACE.md) for deterministic package review and
authenticated upload.
