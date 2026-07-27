# Hugging Face publication

OrcaColony's public Hub namespace is
[`OrcaColony`](https://huggingface.co/OrcaColony).

## Organization card

The organization overview is maintained from
`huggingface/organization-card/README.md`. Hugging Face displays it through the
public static Space named `OrcaColony/README`. This special Space is the
organization card only; it is not a demo and does not contain model or dataset
artifacts. The last published public revision and file digest are recorded in
`huggingface/organization-card/publication.json`; when the retained source has
newer edits, that record continues to describe the remote card until an
explicit update is published.

## Repository layout

Use separate repositories for each reviewed campaign release:

- Model: `OrcaColony/<campaign-name>`
- Dataset: `OrcaColony/<campaign-name>-dataset`
- Optional demo/evaluator Space: `OrcaColony/<campaign-name>-demo`

Models and data do **not** live inside a Space. A Space is a separately deployed
demo or evaluation application that links to versioned model and dataset
repositories. Do not create one until a model has a useful interaction or tester
workflow to demonstrate.

Repository names, licenses, and visibility are chosen by the campaign owner
when that campaign is defined. A name present in a campaign file is only a
destination declaration. It does not create a repository or claim a result.

The dataset repository is the data package for that campaign. Depending on what
the owner declared, it may contain training data, evaluation inputs, evaluator
artifacts, sample-level outputs, or score files. The framework does not create a
separate benchmark product unless the campaign owner explicitly asks for one.

Improved, regressed, unchanged, and inconclusive checkpoints can all be
published as research results. Publication requires internally consistent
artifacts and reviewed cards. It does not require a framework-assigned pass
state.

## Public evidence boundary

For each published campaign result, the Hub repositories should make these
records easy to inspect:

1. The owner-defined campaign record: question, usage scenario, data, training
   recipe, evaluator, metrics, analysis plan, and exact revisions.
2. The model result: exact model commit, evaluation measurements, requested
   comparisons, sample or aggregate evidence, reproduction command,
   limitations, and the owner's conclusion when supplied.
3. The contribution record: accepted work, aggregate compute, and
   contributor-approved names, pseudonyms, anonymous entries, totals, and
   hardware details.

Agent-authored reports may state the agent's findings and interpretation. They
must link the underlying files and must not fill in campaign choices or an owner
decision that was never supplied.

## Authentication

Never paste a Hugging Face token into chat, a command committed to shell history,
an issue, a report, or a repository file. OrcaColony's builder needs no
credentials. Authentication is required only for the explicit `publish` command.

After the project environment is installed, use Hugging Face's browser login:

```bash
uv run hf auth login
uv run hf auth whoami
```

The authenticated personal account must have permission to create and update the
target repositories in the `OrcaColony` organization (`contributor`, `write`, or
`admin`, depending on the organization's access policy). Browser login stores a
refreshable credential in Hugging Face's user cache outside this repository. If
browser login is unavailable, create a personal User Access Token with the
narrowest repository write access possible and paste it only into the
interactive `hf auth login` prompt. Never send that token to another person or
agent.

For CI or a hosted Space, use a scoped secret named `HF_TOKEN`; never store its
value in Git. A read token is sufficient for downloading private artifacts. A
write token is required only for creating or updating repositories.

## Preflight campaign evidence

Before building a v2 operational release, validate the owner-supplied campaign
and evidence contracts:

```bash
uv run python -m orcacolony.campaign_lifecycle inspect \
  --config campaign/<campaign>.json

uv run python -m orcacolony.campaign_lifecycle validate-evidence \
  --config campaign/<campaign>.json \
  --evidence <campaign-evaluation.json> \
  --evaluation-artifacts <evaluation-artifact-directory> \
  --release-checkpoint-sha256 <owner-selected-checkpoint-sha256>
```

These commands expose exact content revisions and verify the evidence without
creating a Hub repository or choosing a campaign metric, checkpoint, or
publication outcome. During release, the evidence-selected checkpoint is used
when it identifies exactly one built-in evaluation. Otherwise the campaign
owner supplies `--release-checkpoint-step`.

## Build locally first

The operational release must exist before Hub packaging. Then build two
deterministic, network-free repository directories:

```bash
uv run python -m orcacolony.huggingface build \
  --release .artifacts/<release> \
  --output .artifacts/<release>-huggingface \
  --model-repo-id OrcaColony/<campaign-name> \
  --dataset-repo-id OrcaColony/<campaign-name>-dataset \
  --model-license <chosen-model-license> \
  --dataset-license <source-compatible-dataset-license> \
  --visibility private \
  --source-revision <git-commit>

uv run python -m orcacolony.huggingface verify \
  --package .artifacts/<release>-huggingface
```

The package contains:

- a model card with training/data/checkpoint provenance, the evidence actually
  supplied, limitations, and visible contributor acknowledgment;
- a dataset card and exact packed dataset/tokenizer provenance;
- custom OrcaColony load and generation metadata;
- the selected restart state and optimizer artifact for trajectory
  reproducibility, in addition to generation weights;
- recorded training diagnostics;
- `campaign-evaluation-evidence.json`,
  `campaign-evaluation-summary.json`, and verified bundled evaluation
  artifacts when the campaign owner supplied them;
- `CONTRIBUTORS.md` and `attribution-snapshot.json`;
- a deterministic publication manifest and checksums.

The model-license value is intentionally required. The repository's MIT software
license does not silently decide the license for trained weights. The dataset
license must exactly match the frozen source manifest, and its compatibility with
the transformations must still be reviewed. The generated model repository
labels the framework license separately from `MODEL-LICENSE.md`; the dataset
repository carries `DATASET-LICENSE.md`. For the current TinyStories artifacts,
that policy must be reviewed against the recorded CDLA-Sharing-1.0 notice before
publication.

## Publish explicitly

After reviewing every generated file:

```bash
uv run python -m orcacolony.huggingface publish \
  --package .artifacts/<release>-huggingface \
  --commit-message "Publish private <campaign-name> review release" \
  --result .artifacts/<release>-huggingface-private-publish-result.json \
  --confirm-upload
```

`--confirm-upload` is mandatory because this command creates or mutates both Hub
repositories. The command uses the locally authenticated user; it accepts no
token argument and prints both resulting Hub commit revisions for the permanent
release record. The required `--result` path stores the same authenticated user,
visibility, repository IDs, and exact Hub commits outside the reviewed upload
package. Before either upload, the publisher checks both existing repositories: a
visibility mismatch or a stale remote file absent from the reviewed package
fails closed. Use a new repository name or explicitly reconcile the old
repository instead of letting a supposedly exact release inherit old files.

The build-time `--visibility` choice is recorded in the publication manifest and
the uploader must follow it; publication cannot silently override the campaign's
`visibility_policy`. A campaign may choose `private`, `public`, or
`private_review_then_public`. Use `private_review_then_public` for the ordinary
OrcaColony flow:

1. Build, verify, inspect, and publish the private package.
2. Complete license, attribution, model-card, evidence, and independent tester
   review without changing the release inputs.
3. Build a new package from the exact same operational release and source
   revision with `--visibility public`; verify and inspect it again.
4. Explicitly change both existing Hub repositories from private to public in
   their Hugging Face settings.
5. Publish the public package with a new result path, such as
   `.artifacts/<release>-huggingface-public-publish-result.json`.

Step 4 is deliberately not automated. Until both repository settings match the
reviewed public package, the publisher fails before uploading either repository.
Keep both the private and public publication-result records; they record which
authenticated account performed each transition and which Hub commits resulted.
