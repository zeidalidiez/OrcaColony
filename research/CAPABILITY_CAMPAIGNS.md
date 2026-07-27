# Historical capability-campaign contract

This document preserves the
`orcacolony_capability_research_v1` contract used by the Record Patch prototype.
Its mandatory baseline, thresholds, holdouts, guardrails, checkpoint selection,
and promotion classification are historical experiment choices. They are not
the framework for new campaigns.

New campaigns use
[`CAMPAIGN_FRAMEWORK.md`](CAMPAIGN_FRAMEWORK.md), where the campaign owner
supplies the usage scenario, evaluator, metrics, comparisons, and decisions.
The framework validates and publishes those choices without adding a task or a
pass/fail gate.

Systems evidence answers whether a training mechanism is correct, recoverable,
or resource-feasible. A capability campaign answers whether a selected model is
better at one frozen use case and what caused that change. These are separate
release dispositions.

A campaign declaring `orcacolony_capability_research_v1` must freeze:

- one falsifiable capability claim;
- a versioned baseline, an absolute success threshold, and a positive minimum
  improvement from that baseline;
- the behavioral suite's exact dataset and evaluator revisions, with distinct
  validation and final-holdout split names;
- a language-loss validation range used for checkpoint selection;
- a disjoint language-loss holdout range used only after selection;
- all guardrails and their pass/fail evaluators;
- a training-effect analysis plan;
- model and dataset licenses and exact `OrcaColony/...` Hub destinations.

The built-in language-loss evaluator remains a training diagnostic; its
post-selection holdout is not behavioral proof. The current v1 selection policy
is explicitly `lowest_validation_mean_loss_before_behavioral_final_holdout`.
A model is classified as a `capability_candidate` when it has the reserved
language-loss result but no passing behavioral promotion evidence. It becomes a
`capability_model` only when `orcacolony_capability_promotion_evidence_v1`
matches the selected checkpoint, training dataset, frozen behavioral suite,
baseline, primary metric, evidence artifacts, reproduction command, and every
declared guardrail.

The capability-specific part of a campaign has this shape (ordinary `campaign`,
`model`, `training`, and dataset identity fields are omitted here):

```json
{
  "evaluation": {
    "metric": "held_out_cross_entropy",
    "checkpoint_selection": "lowest_mean_loss",
    "validation_start_sequence": 0,
    "validation_sequences": 128,
    "batch_size": 8,
    "final_holdout": {
      "start_sequence": 128,
      "sequence_count": 128,
      "batch_size": 8
    }
  },
  "research": {
    "format": "orcacolony_capability_research_v1",
    "claim": "A concrete falsifiable behavior claim.",
    "baseline": {
      "id": "frozen-baseline",
      "description": "Exact comparison model and prompting policy.",
      "revision": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "primary_metric": {
      "id": "task-score",
      "description": "Frozen sample-level task score.",
      "direction": "maximize",
      "unit": "ratio",
      "success_threshold": 0.7,
      "minimum_improvement_from_baseline": 0.05
    },
    "guardrails": [
      {
        "id": "format-validity",
        "description": "Every scored output remains parseable."
      }
    ],
    "analysis_plan": [
      "Compare sample-level outputs, error buckets, and update diagnostics."
    ],
    "final_holdout_policy": "release_only_after_checkpoint_selection",
    "checkpoint_selection": "lowest_validation_mean_loss_before_behavioral_final_holdout",
    "behavioral_evaluation": {
      "suite_id": "frozen-task-suite",
      "dataset_revision": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      "evaluator_revision": "3333333333333333333333333333333333333333",
      "validation_split": "validation",
      "final_holdout_split": "final_holdout"
    }
  },
  "publication": {
    "format": "orcacolony_huggingface_publication_v1",
    "model_repo_id": "OrcaColony/example-capability-model",
    "dataset_repo_id": "OrcaColony/example-capability-model-dataset",
    "model_license": "REPLACE-WITH-MODEL-LICENSE",
    "dataset_license": "REPLACE-WITH-DATASET-LICENSE",
    "visibility_policy": "private_review_then_public"
  }
}
```

The two `REPLACE-...` values are explanatory placeholders and are intentionally
invalid. Replace them with reviewed Hugging Face license identifiers before
loading the campaign. The visibility policy is also part of the frozen campaign:
`private` and `public` allow only that state, while
`private_review_then_public` allows a reviewed private package followed by a
separately built and explicitly approved public package. The publisher never
changes an existing repository's visibility.

Promotion evidence has this shape:

```json
{
  "format": "orcacolony_capability_promotion_evidence_v1",
  "campaign_id": "exact-campaign-id",
  "checkpoint_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
  "dataset_revision": "5555555555555555555555555555555555555555555555555555555555555555",
  "evaluation_suite": {
    "suite_id": "declared-behavioral-suite-id",
    "dataset_revision": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
    "evaluator_revision": "3333333333333333333333333333333333333333",
    "split": "declared-final-holdout-split"
  },
  "primary_metric": {
    "id": "declared-behavioral-metric-id",
    "value": 0.0,
    "baseline": {
      "id": "declared-baseline-id",
      "revision": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
      "value": 0.0
    }
  },
  "guardrails": [
    {
      "id": "declared-guardrail-id",
      "passed": true,
      "detail": "Evidence-backed explanation."
    }
  ],
  "limitations": [
    "At least one concrete limitation is required."
  ],
  "artifacts": [
    {
      "id": "sample-level-results",
      "sha256": "6666666666666666666666666666666666666666666666666666666666666666",
      "uri": "durable-artifact-location"
    }
  ],
  "reproduction": {
    "command": [
      "python",
      "path/to/frozen-evaluator.py"
    ],
    "notes": "Exact setup and interpretation notes."
  }
}
```

The release validator establishes identity, completeness, threshold,
baseline-improvement, and guardrail consistency. It does not magically prove
that an arbitrary external evaluator was honest. Reports must therefore publish
the sample-level artifact, evaluator source, exact command, limitations, and
enough data for an interested reader to rerun or independently audit the claim.

Reports for capability experiments should answer:

1. What behavior changed relative to initialization and the frozen baseline?
2. Which examples/data buckets improved or regressed?
3. How large and stable was the effect across seeds or deterministic repeats?
4. What happened to gradient/update norms, outputs, errors, memorization, and
   unrelated guardrails?
5. What data, compute, wall time, and volunteer hardware produced the result?
6. Which exact artifacts and commands let another person verify the claim?

Use [`../reports/CAPABILITY_REPORT_TEMPLATE.md`](../reports/CAPABILITY_REPORT_TEMPLATE.md)
for the durable human-readable report.
