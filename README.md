# OrcaColony

**Repository version:** [`0.1.1`](VERSION)

OrcaColony is a self-hostable framework and reproducible research vehicle for
community model training. It lets independent contributors complete bounded
work against a canonical training state, then preserves the accepted work,
model trajectory, evidence, and contributor-approved credit.

The research goal is to find practical ways to make non-frontier-size models
more useful while keeping every claim tied to a concrete, reproducible usage
evaluation.

## Scope boundary

OrcaColony supplies the campaign machinery. It does not decide what a campaign
should train or what success means.

When a practical campaign is created, its owner supplies:

- the model, data, objective, training recipe, and stopping point;
- the concrete usage scenario and research question;
- the evaluator, evaluation inputs, metrics, and comparisons;
- any thresholds, guardrails, holdouts, or checkpoint-selection rule;
- the interpretation, publication settings, and next-step decisions.

The framework validates and locks those choices, distributes work, reconciles
accepted results, records evidence, generates contributor credit, and packages
the completed result. It does not insert a default task, benchmark, metric,
threshold, checkpoint choice, or follow-up experiment.

## Current status

| Area | State | Where to verify it |
| --- | --- | --- |
| Training framework | Deterministic reference, browser, multi-worker, restart, retry, release, and attribution paths are implemented | [Specification](SPEC.md), [detailed guide](IMPLEMENTATION_GUIDE.md) |
| Campaign research | Owner-defined v2 contracts, evidence preflight, checkpoint binding, and Hugging Face packaging are implemented | [Campaign framework](research/CAMPAIGN_FRAMEWORK.md) |
| Systems research | PEFT, local placement, partial-model, tiled recovery, and sparse-expert methods have bounded studies with explicit limitations | [Research index](research/README.md), [reports](reports/index.html) |
| Contributor credit | Accepted direct-training work and contributor disclosure choices are supported; a separate auxiliary-work ledger remains open | [Campaign formats](campaign/README.md), [current roadmap](PROGRESS_REPORT.md) |
| Practical campaign | Intentionally undefined until a campaign owner supplies the complete contract | [Current roadmap](PROGRESS_REPORT.md) |
| Public model and data release | Organization page exists; no owner-defined campaign model or dataset package has been published yet | [Hugging Face publication](HUGGINGFACE.md) |
| Public volunteer deployment | Trusted local paths are proven; untrusted public compute integrity and operator-owned hosting remain incomplete | [Current roadmap](PROGRESS_REPORT.md) |

Historical Record Patch experiments remain available as narrow negative
training-effect evidence. They do not define the next campaign or the
framework's default metric. See
[the task record](capability/record-patch-v1/TASK.md) and
[the published findings](reports/record-patch-t2-continuation-v1.html).

## Start here

| Need | Document |
| --- | --- |
| Current build position, blockers, and next bounded work | [PROGRESS_REPORT.md](PROGRESS_REPORT.md) |
| Campaign-owner and framework responsibilities | [research/CAMPAIGN_FRAMEWORK.md](research/CAMPAIGN_FRAMEWORK.md) |
| Full architecture and contracts | [SPEC.md](SPEC.md) |
| Detailed commands and measured systems proofs | [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) |
| Machine-readable studies and evidence | [research/README.md](research/README.md) |
| Human-readable findings and limitations | [reports/index.html](reports/index.html) |
| Campaign configuration and contributor-credit formats | [campaign/README.md](campaign/README.md) |
| Hugging Face repository, authentication, review, and publication flow | [HUGGINGFACE.md](HUGGINGFACE.md) |
| Version policy | [VERSIONING.md](VERSIONING.md) |

## Local development

OrcaColony requires Python 3.11 and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pytest
```

Run the smallest deterministic reference paths with:

```bash
uv run python -m orcacolony.reference fixture \
  --config campaign/t0-smoke.json \
  --output .artifacts/fixture

uv run python -m orcacolony.reference train \
  --config campaign/t0-smoke.json \
  --output .artifacts/m0
```

Generated artifacts belong under `.artifacts/`. The detailed browser,
multi-worker, persistent-campaign, native-worker, placement, and release
commands are retained in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

## Campaign contract and evidence preflight

A campaign owner begins with the neutral examples under `campaign/`, supplies
the campaign-specific values, and reviews the resulting contract before any
training is authorized.

Validate the exact campaign and research identities:

```bash
uv run python -m orcacolony.campaign_lifecycle inspect \
  --config campaign/<campaign>.json
```

After the owner-selected evaluations have been run, validate the evidence and
all local `bundle:` files:

```bash
uv run python -m orcacolony.campaign_lifecycle validate-evidence \
  --config campaign/<campaign>.json \
  --evidence <campaign-evaluation.json> \
  --evaluation-artifacts <evaluation-artifact-directory> \
  --release-checkpoint-sha256 <owner-selected-checkpoint-sha256>
```

These commands validate supplied choices. They do not create a metric,
threshold, release decision, or campaign plan.

## Evidence and credit

The public record for a completed campaign is intended to include:

- exact model, data, source, evaluator, and environment revisions;
- the owner-defined usage scenario, metrics, requested comparisons, findings,
  and limitations;
- sample-level or aggregate evidence with digest verification;
- reproducible commands and deterministic package checksums;
- accepted-work totals and each contributor's approved name, pseudonym,
  anonymity, totals, roles, and hardware disclosures;
- exact Hugging Face model and dataset commits after publication.

Agent-authored reports may explain findings, but the proof remains in the
versioned model, data, evaluator outputs, campaign records, and contribution
records.

## Versioning

The current version is stored in [`VERSION`](VERSION). Every project commit
must increase it and keep `pyproject.toml`, `src/orcacolony/__init__.py`, and
`uv.lock` synchronized.

Check local work before committing:

```bash
python scripts/check_version.py --working-tree --base-ref HEAD
```

GitHub checks every new commit in a pull request and every push to `main`.
See [VERSIONING.md](VERSIONING.md) for the exact policy.

## License and data notices

The framework source is licensed under the [MIT License](LICENSE). Dataset and
model licenses are campaign-specific and must be declared explicitly.
Third-party data provenance and notices are recorded in
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
