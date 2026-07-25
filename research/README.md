# OrcaColony research records

OrcaColony studies compare bounded experiments against one fixed hypothesis and use-case evaluation contract. The stable v0.1 campaign path remains the correctness baseline; a research record does not promote an execution method by itself.

## Layout

Each study owns its linked experiment and evidence manifests:

```text
research/studies/<study-id>/
  study.json
  experiments/
    <experiment-id>.json
  evidence/
    <experiment-id>.json
```

The schema deliberately keeps training methods, execution topologies, memory profiles, numerical profiles, worker profiles, variables, and experiment roles as open identified descriptors with human-readable labels and descriptions. Workflow states and result dispositions remain constrained so tools can distinguish proposed, active, validated, rejected, inconclusive, and promoted work.

## Recording a result

Run the committed P1 contract fixture from the repository root:

```bash
uv run python -m orcacolony.research record \
  --study research/studies/p1-research-contract-smoke-v1/study.json \
  --experiment research/studies/p1-research-contract-smoke-v1/experiments/deterministic-result-bundle.json \
  --evidence research/studies/p1-research-contract-smoke-v1/evidence/deterministic-result-bundle.json \
  --output .artifacts/p1-research-contract-result
```

The command validates every field, rejects duplicate JSON keys, verifies that the study references the exact experiment path, evaluates the declared primary metric and guardrails, and atomically writes:

- `study.json`, `experiment.json`, and `evidence.json` as canonical source records.
- `result.json` as the machine-readable conclusion.
- `RESULT.md` as the human-readable report.
- `SHA256SUMS` for every generated result file.

The output path must not already exist. Remove or choose another ignored `.artifacts/` path before repeating the command.

## Committed studies

- `p1-research-contract-smoke-v1` validates the research-record machinery itself. It makes no training-method claim.
- `p2-lora-numerical-v1` validates the first frozen-base LoRA numerical slice against an independent one-step reference. Its evidence explicitly does not claim browser/native parity, campaign recovery, offload performance, or useful model adaptation.

Every committed study, linked experiment, and conventionally named evidence file is rebuilt by the repository test suite. To render the P2 result directly, substitute the three `p2-lora-numerical-v1` manifest paths in the command above and choose a fresh ignored output directory.

## Interpretation

A `validated` or `promoted` result must pass the study's primary use-case threshold and every guardrail. A `rejected` or `inconclusive` result remains publishable and must retain its findings and limitations. Promotion into the supported framework still requires the end-to-end campaign, restart, retry, provenance, evaluation, and release gates in [`SPEC.md`](../../SPEC.md).
