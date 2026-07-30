# OrcaColony research records

This directory contains the established systems-study recorder. It compares
bounded experiments against a study author's fixed hypothesis and measurement
contract. It does not define the task or evaluation for a future training
campaign.

New owner-defined training campaigns use
[`CAMPAIGN_FRAMEWORK.md`](CAMPAIGN_FRAMEWORK.md). The stable v0.1 campaign path
remains the correctness baseline; a research record does not admit an execution
method into the supported framework by itself.

## Layout

Each study owns its linked experiment and evidence manifests:

```text
research/studies/<study-id>/
  study.json
  experiments/
    <experiment-id>.json
  evidence/
    <experiment-id>.json
  scripts/
    <reproduction-script>.py
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
- `environment.json` with Python, platform, dependency, and `uv.lock` identity.
- `result.json` as the machine-readable conclusion.
- `RESULT.md` as the human-readable report, including measurements, limitations,
  reproduction, and artifact-resolution status.
- byte snapshots of every digest-verified `repo:` artifact under `artifacts/`.
- `SHA256SUMS` for every generated result file.

The output path must not already exist. Remove or choose another ignored `.artifacts/` path before repeating the command.

`repo:` artifact references are resolved relative to `--repository-root`, checked
against their declared digest, rejected if they traverse a symlink or escape the
repository, and copied into the result. Other URI schemes remain declared
references; the recorder labels them `not_resolved_by_recorder` instead of
pretending that their bytes were verified.

## Committed studies

- `p1-research-contract-smoke-v1` validates the research-record machinery itself. It makes no training-method claim.
- `p2-lora-numerical-v1` validates the first frozen-base LoRA numerical slice against an independent one-step reference. Its evidence explicitly does not claim browser/native parity, campaign recovery, offload performance, or useful model adaptation.
- `p2-browser-lora-parity-v1` compares CPU/WASM and WebGPU against the same Python adapter-gradient oracle. It records the WebGPU cold-run cost as a negative finding and does not claim coordinator integration or offload.
- `p2-connected-browser-lora-v1` validates two authenticated CPU/WASM adapter assignments, coordinator-owned aggregation and AdamW, separate worker-weight and full resume-state identities, restart recovery, and accepted-checkpoint parity against the independent Python step. It does not claim heterogeneous-network performance, frozen-base offload, public untrusted participation, or useful adaptation quality.
- `p2-persistent-lora-release-v1` validates two persistent adapter steps, coordinator reload after step 1, per-step held-out evaluation, a positive fixed-use-case gate, lowest-loss checkpoint selection, and a deterministic release with separate base, adapter, worker-weight, and complete resume-state identities. Its evaluated workers are local Python oracles; the connected browser study remains the real Burn worker evidence.
- `p3-native-resource-profile-v1` validates privacy-filtered runtime/payload/memory/storage observations with real connected Burn CPU/WASM workers, then qualifies digest-validated cached-base native CPU workers on frozen TinyStories at 6.9M and 91.5M parameters. Warm assignments avoid all base and adapter fetch payloads while retaining numerical and held-out-loss guardrails. It records one-shot cache validation/runtime initialization as the dominant warm T2 cost and does not claim packet-level network measurement, quantized placement, mapped/NVMe offload, or useful model quality.
- `p3-persistent-native-session-v1` validates bounded in-process model/adapter reuse at T2. One authenticated process completes both shards with one model build and one adapter load, removes 99.9986% of one-shot warm setup time, and retains identical checkpoint and held-out behavior. It also records failure-atomic adapter refresh and the lack of participant-diversity and memory-placement gains.
- `p4-numerical-profile-qualification-v1` closes the local P3 placement matrix and P4 admission contract by validating connected homogeneous int8 across resident conversion and direct authenticated layer bundles. Four real T1 assignments cross a coordinator restart with exact profile-oracle gradients, profile-bound v2 checkpoints, positive frozen holdout movement, zero warm model payload, and fail-closed separation from bit-exact CPU FP32 and separately named Burn profiles.
- `p7-frozen-cached-head-t0-v1` validates an exact one-step sparse-expert
  decomposition with the final norm and untied output head frozen and retained
  by both matched cache models. Four warm experts used 53.17% fewer accounted
  tensor bytes than the full control, while cold fan-out remained 7.34% more
  expensive. It does not claim process cache reuse, measured transport,
  reliability, memory savings, training quality, or production readiness.
- `p7-authenticated-process-t1-v1` validates exact trusted-local persistent
  full and expert processes across two independent assignments, including one
  replacement after an accepted assignment is lost. Warm expert tensor traffic
  falls 55.75%, while cold fan-out and process high-water RSS remain negative
  findings. It does not claim a sequential training trajectory, persisted
  coordinator recovery, remote authentication, or model quality.
- `p7-persisted-trajectory-t1-v1` validates two exact three-step trajectories
  with durable accepted results, worker replacement without recomputation, and
  fresh-process coordinator recovery after checkpoint publication. Pooled
  expert tensor traffic and child VmHWM are lower, but elapsed time is
  67.40%-69.07% higher and persisted bytes fall only 7.12%. It does not claim
  practical training benefit, aggregate memory, concurrent throughput, remote
  trust, or donated computation.
- `p7-content-addressed-state-t1-v1` compares the replicated trajectory with
  topology-local immutable checkpoint blobs in two opposite execution orders.
  Persisted file payload falls 29.07% for the full-process control and 31.30%
  for the pooled-expert control while all 12 layout, run, and step rows remain
  exact and both recovery paths pass. Timing direction is inconclusive. The
  result closes this bounded storage target and does not claim model quality,
  donated computation, or a practical campaign result.

Reproduce the persistent release study's source evidence with:

```bash
uv run python research/studies/p2-persistent-lora-release-v1/scripts/run-proof.py
uv run python -m orcacolony.release \
  --config .artifacts/p2-lora-evaluated-release-proof/campaign.json \
  --lora-config .artifacts/p2-lora-evaluated-release-proof/lora.json \
  --participants .artifacts/p2-lora-evaluated-release-proof/participants.json \
  --dataset-artifacts .artifacts/p2-lora-evaluated-release-proof/dataset \
  --campaign-state .artifacts/p2-lora-evaluated-release-proof/campaign-state \
  --browser-root spikes/burn-browser-gradient/www \
  --project-license LICENSE \
  --third-party-notice THIRD_PARTY_DATA.md \
  --public-coordinator-url https://coordinator.example \
  --output .artifacts/p2-lora-evaluated-release-proof/release
(cd .artifacts/p2-lora-evaluated-release-proof/release && sha256sum -c SHA256SUMS)
```

Reproduce the native T1/T2 resource-profile evidence with isolated worker processes:

```bash
uv run python -m orcacolony.artifacts \
  --output .artifacts/p3-native-resource-dataset
uv run python research/studies/p3-native-resource-profile-v1/scripts/run-proof.py \
  --dataset-artifacts .artifacts/p3-native-resource-dataset \
  --output .artifacts/p3-native-resource-profile-proof
```

Reproduce the persistent T2 session comparison with:

```bash
uv run python -m orcacolony.artifacts \
  --output .artifacts/p3-persistent-native-dataset
uv run python research/studies/p3-persistent-native-session-v1/scripts/run-proof.py \
  --dataset-artifacts .artifacts/p3-persistent-native-dataset \
  --output .artifacts/p3-persistent-native-session-proof
```

Reproduce the connected homogeneous-int8 qualification with:

```bash
uv run python -m orcacolony.artifacts \
  --output .artifacts/p4-numerical-profile-dataset
uv run python research/studies/p4-numerical-profile-qualification-v1/scripts/run-proof.py \
  --dataset-artifacts .artifacts/p4-numerical-profile-dataset \
  --output .artifacts/p4-numerical-profile-proof
```

Validate and bundle the committed P7 frozen cached-head evidence with:

```bash
uv run python -m orcacolony.research record \
  --study research/studies/p7-frozen-cached-head-t0-v1/study.json \
  --experiment research/studies/p7-frozen-cached-head-t0-v1/experiments/t0-four-expert-frozen-cached-head.json \
  --evidence research/studies/p7-frozen-cached-head-t0-v1/evidence/t0-four-expert-frozen-cached-head.json \
  --output .artifacts/p7-frozen-cached-head-result
```

Its raw primary and repeat commands are pinned to implementation commit
`14dd341b57cc1ab79518037e7c3d1e5cde712332` in the experiment manifest.

Validate and bundle the committed persisted-trajectory evidence with:

```bash
uv run python -m orcacolony.research record \
  --study research/studies/p7-persisted-trajectory-t1-v1/study.json \
  --experiment research/studies/p7-persisted-trajectory-t1-v1/experiments/t1-persisted-sparse-trajectory.json \
  --evidence research/studies/p7-persisted-trajectory-t1-v1/evidence/t1-persisted-sparse-trajectory.json \
  --output .artifacts/p7-persisted-trajectory-result
```

The experiment manifest pins the one-thread primary and repeat commands,
implementation commit, evidence hashes, frozen dataset revision, 875 MB
temporary-state expectation, and Linux `/proc` requirement.

Validate and bundle the committed content-addressed state evidence with:

```bash
uv run python -m orcacolony.research record \
  --study research/studies/p7-content-addressed-state-t1-v1/study.json \
  --experiment research/studies/p7-content-addressed-state-t1-v1/experiments/t1-content-addressed-state.json \
  --evidence research/studies/p7-content-addressed-state-t1-v1/evidence/t1-content-addressed-state.json \
  --output .artifacts/p7-content-addressed-state-result
```

The experiment manifest pins the opposite-order one-thread commands,
implementation and merged execution revisions, evidence hashes, frozen dataset
revision, 1.5 GB temporary comparison-root expectation, and Linux `/proc`
requirement. The raw state roots are reproducible temporary material and may be
removed after the evidence JSON is secured.

Every committed study, linked experiment, and conventionally named evidence file is rebuilt by the repository test suite. To render any result directly, substitute that study's three manifest paths in the command above and choose a fresh ignored output directory.

## Interpretation

For the historical `orcacolony_study_v1` format, a `validated` or `promoted`
systems result must pass the study's declared threshold and guardrails. A
`rejected` or `inconclusive` result remains publishable and must retain its
findings and limitations. This study status is separate from the neutral v2
campaign evaluation record and from a campaign owner's model-publication
decision.
