# OrcaColony Progress Report

**Last updated:** 2026-07-24

**Current phase:** P2 — proving PEFT adapter-gradient semantics on top of the completed research-record contract and v0.1 baseline

**Canonical specification:** [`SPEC.md`](SPEC.md)

## Purpose and update rule

This is the living project handoff and priority report. It answers four questions without requiring a separate status review:

1. What are we trying to build?
2. What has been proven or shipped?
3. What remains incomplete?
4. What is the next bounded priority?

Update this file in every project commit made as part of the active roadmap, before that commit is pushed. Keep it decision-oriented rather than copying the Git history. Record meaningful changes to the build position, priorities, blockers, evidence, or next target. Negative experimental findings are progress and belong here when they change the plan.

## Project direction

OrcaColony is both:

1. A reusable, self-hostable framework for community model-training campaigns.
2. A reproducible research vehicle for testing how heterogeneous, unreliable, ordinary machines can make useful contributions to models that may exceed an individual contributor's GPU VRAM or total local memory.

The participation model is community accumulation, not a permanent cluster of weak computers. A contributor may arrive, complete one or several bounded pieces of accepted work, receive attribution according to their preference, and leave. The coordinator reconciles many such contributions into a canonical campaign result.

The project will preserve a stable correctness baseline while testing additional execution methods through bounded studies. Methods graduate into supported framework capabilities only after they pass declared correctness, reliability, resource, provenance, and use-case evaluation gates.

## Overall build position

### Proven stable foundation

- Deterministic Python reference training and resumable safetensors checkpoints.
- Burn browser forward/backward execution through WebGPU and CPU/WASM.
- Complete named-gradient export and tolerance-based browser/Python parity.
- Connected coordinator-owned AdamW updates.
- Multi-worker assignments, leases, retries, stale-result rejection, and at-most-once accepted work.
- Crash-safe canonical checkpoint advancement and persistent multi-step campaigns.
- Owner-approved, default-deny participation and contributor-controlled public attribution.
- Frozen TinyStories data/tokenizer provenance and real-data T1 campaigns.
- Initialization and checkpoint evaluation with machine-readable success gates.
- Public dashboard, exact-origin browser deployment support, and deterministic privacy-filtered release bundles.
- A bounded 12-step, two-worker local T1 system proof that passed its declared held-out-loss gate and matched the centralized reference.

### Current boundary

The local v0.1 system path is proven. A several-day campaign with distinct remote, owner-approved participants remains an operator deployment milestone rather than a local implementation blocker.

The current architecture uses replicated full-model data parallelism: every direct-gradient worker can execute its complete assignment independently against one canonical checkpoint. The complete model may eventually be placed across that worker's GPU VRAM, system RAM, and local storage; it does not have to remain entirely in GPU VRAM. Cross-worker partial-model methods are a post-v0.1 research track rather than part of the proven baseline.

## Research principles

- Prefer the smallest model that meets the campaign's declared use case; parameter count is not a status goal.
- Do not restrict campaigns to tiny models merely because contributors lack datacenter GPUs.
- Preserve transient, bounded community participation rather than assuming permanent workers.
- Keep the current replicated-model path as the numerical and operational oracle.
- Separate training method from execution topology and memory placement.
- Compare methods through reproducible studies rather than unsupported claims.
- Require every campaign to declare a concrete use case and a fixed evaluation contract.
- Evaluate checkpoints throughout a campaign on frozen validation data; reserve a final holdout for promotion.
- Record failed and inconclusive experiments as first-class findings.
- Promote one measured capability at a time; do not productionize every research idea.

## Priority order

### P0 — Align the specification with the research-first direction

**Status:** Complete

- Clarify replicated full-model data parallelism and why it remains the correctness baseline.
- Clarify that a complete-local worker may use VRAM, system RAM, memory mapping, and local storage offload.
- Define campaigns, studies, execution topologies, numerical profiles, and experimental graduation.
- Require a concrete use-case evaluation contract for every campaign.
- Record future partial-model and asynchronous reconciliation tracks without pretending they are already implemented.

### P1 — Add the research study and experiment contract

**Status:** Complete

- Machine-readable study manifest tying comparable campaigns to one hypothesis and use case.
- Experiment status: proposed, active, validated, rejected, inconclusive, or promoted.
- Reproducibility fields for model/data/code revisions, worker profiles, execution topology, numerical profile, resource use, and evaluation results.
- A standard result report that includes negative findings and limitations.

Completed within P1:

- Added the fail-closed `orcacolony_study_v1` validator.
- Kept research variables and experiment roles open through identified, display-ready descriptors instead of a narrow fixed method menu.
- Locked the hypothesis, use-case baseline, primary metric and threshold, repeated validation suite, final holdout, guardrails, controlled variables, and safe experiment references.
- Added linked `orcacolony_experiment_v1` and `orcacolony_experiment_evidence_v1` contracts for exact subjects, artifact revisions, method/topology/placement descriptions, worker profiles, budgets, reproduction commands, measurements, evaluation, findings, and limitations.
- Added deterministic, atomic experiment-result bundles containing canonical source manifests, source hashes, a machine-readable decision, a human-readable `RESULT.md`, and `SHA256SUMS`.
- Enforced that validated or promoted evidence passes the declared primary metric and every guardrail while preserving rejected and inconclusive findings as publishable outcomes.
- Added the real `python -m orcacolony.research record` command with duplicate-key rejection and exact study-to-experiment path binding.
- Committed a self-contained contract study pinned to the exact CLI implementation revision and proved its real result bundle and checksums through the command line.

### P2 — Prove PEFT with complete adapter-gradient semantics

**Status:** In progress

- Freeze an exact base checkpoint.
- Define one LoRA configuration and exact trainable-tensor manifest.
- Preserve summed masked loss and complete unnormalized gradients for adapter tensors.
- Keep the canonical adapter optimizer at the coordinator.
- Prove restart, retry, evaluation, and release for an adapter campaign.

Completed within the first P2 numerical slice:

- Added an isolated Python LoRA runtime without changing the stable dense model path.
- Pinned the deterministic T0 base identity to SHA-256 `d31269889b0154bb962bf976361ddc81d47498f1d3af9244eaf24d4ad9e1d060`.
- Targeted the combined attention QKV projection in all four layers with rank 4, alpha 8, zero dropout, and a dedicated adapter seed.
- Froze every base parameter and exposed exactly 8 named trainable adapter tensors containing 8,192 values.
- Preserved byte-for-byte equal initial logits because the LoRA B matrices initialize to zero.
- Exported deterministic complete FP32 adapter gradients for summed loss; the fixture gradient SHA-256 is `7ce16dfd740fd5a249257de6ab442943577b86917f9ae77604c5097ce1a5b8e2`.
- Proved that assigning the submitted unnormalized gradients, dividing by total loss weight, clipping once, and applying coordinator-owned AdamW produces the same adapter tensors as an independent mean-loss reference step while every base tensor remains unchanged.
- Added a fail-closed `orcacolony_lora_manifest_v1` that pins the canonical campaign content, base-model digest, exact QKV targets, rank, alpha, dropout, seed, and initialization.
- Added the real `python -m orcacolony.peft export-fixture` command. It atomically exports the dense base, initial adapters, deterministic batch, complete adapter gradients, updated adapters, a self-describing proof manifest, and portable `SHA256SUMS`.
- Produced two byte-identical real fixture exports and verified every checksum through `sha256sum -c`.
- Recorded the updated adapter SHA-256 as `93b8701db49573d20bdb59e50c9dd4a0d74eb5e4f5778005d4f21271f715d8c8`; the global norm before the one allowed clipping operation was `0.038248103111982346`.

### P3 — Qualify local memory tiers

**Status:** Planned

Test increasingly flexible placement while retaining one-worker assignment independence:

1. Full GPU residency.
2. GPU plus system-RAM offload.
3. Quantized frozen-base placement.
4. Explicit local-storage/NVMe out-of-core execution with bounded caches and prefetching.
5. Remote shard streaming only if computation per transferred byte can justify it.

The native worker is the first target for explicit offload. Browser support remains measurement-driven.

### P4 — Qualify numerical and mixed execution profiles

**Status:** Planned

- Certify each precision, quantization, runtime, and offload profile against a canonical fixture.
- Separate numerical semantics from placement-only differences.
- Prove mixed-profile aggregation before admitting profiles to the same campaign.
- Record runtime provenance and revoke profiles that exhibit systematic numerical or operational failure.

### P5 — Explore rolling partial-model training

**Status:** Planned research

- Give capability-sized subnetworks or parameter slices to workers.
- Rotate coverage across a larger coordinator-owned global model.
- Reconcile updates from one frozen checkpoint round.
- Compare against replicated full-model training on the same use-case evaluation.
- Determine whether the quality and convergence tradeoff is acceptable for transformers.

### P6 — Explore exact asynchronous tiled computation

**Status:** Planned research

- Divide one representative transformer layer into retryable matrix or tensor tasks.
- Persist dependency state at the coordinator.
- Reassign failed tiles without requiring permanent peer pipelines.
- Reconstruct exact forward and backward results.
- Measure worker memory, network transfer, cache reuse, latency, and compute-to-transfer ratio before expanding to a complete model.

### P7 — Explore sparse experts and other advanced topologies

**Status:** Deferred research

- Expert-sharded or sparse models.
- Router and shared-trunk training.
- Expert availability, replication, reconciliation, and release semantics.
- Model-parallel systems for unreliable peers only after simpler tracks establish the relevant evidence.

## Immediate next bounded target

Pin this exact exporter revision into a P2 study, record the verified fixture as a reproducible research result, and then begin browser adapter-gradient parity against the same artifacts. Campaign restart and a useful public base remain later P2 work.

## Remaining major work

- PEFT numerical and campaign vertical slice.
- Native worker with explicit RAM and storage offload.
- Profile certification and mixed-profile proof.
- Rolling-submodel feasibility study.
- Exact tiled-computation feasibility study.
- Later sparse-expert investigation.
- Operator-owned remote trusted campaign when deployment inputs are available.

## Current blockers and hold-ups

- No blocker prevents the specification and research-framework work.
- Remote trusted deployment still requires operator-owned HTTPS hosting choices and approved participants, but that does not block local research milestones.
- Larger-model execution methods are hypotheses until measured; none should be described as supported merely because a paper or prototype demonstrates the general idea.

## Change record

### 2026-07-24

- Established this living progress report.
- Recorded the completed local v0.1 foundation.
- Adopted the research-first direction and community micro-contribution framing.
- Ordered the initial research tracks from specification alignment through PEFT, local offload, partial models, tiled computation, and sparse experts.
- Completed the P0 specification alignment while preserving the v0.1 numerical, trust, provenance, and release contracts.
- Defined campaigns versus studies; separated training method, execution topology, memory placement, and numerical profile.
- Added the fixed use-case evaluation contract, native hierarchical-memory direction, partial-model research semantics, profile qualification sequence, and experimental graduation rules.
- Added the first runnable P1 code: a fail-closed study-manifest validator with open, display-ready research descriptors and safe experiment references.
- Added linked experiment and evidence validation plus deterministic JSON/Markdown result bundles that retain negative findings and limitations.
- Added a tested command-line path that loads exact linked manifests and writes the complete research result bundle.
- Completed P1 with a committed study fixture, two byte-identical real CLI bundles, successful checksum verification, and explicit proof limitations.
- Added the first P2 numerical vertical: deterministic frozen-base LoRA gradients and an exact coordinator-compatible adapter update matching an independent reference.
- Added the exact LoRA manifest and deterministic artifact exporter, fixed checksum output to use portable LF records, and proved two real byte-identical exports with all hashes valid.
