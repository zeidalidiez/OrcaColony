# OrcaColony Progress Report

**Last updated:** 2026-07-25

**Current phase:** P3 — measuring and qualifying native local-memory profiles while preserving the P2 adapter and dense baselines

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

**Status:** Complete

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
- Published the `p2-lora-numerical-v1` study, experiment, and evidence manifests pinned to exporter commit `5aff4392d3faf0eac740d43c28f57150a7929153`.
- Generated and checksum-verified the P2 research result with the exact-match metric and all three guardrails passing. Its limitations explicitly exclude browser/native parity, campaign restart/retry, offload, and useful adaptation quality.
- Added a regression gate that rebuilds every committed research record, so invalid or unlinked studies fail the normal test suite.
- Extended subsequent LoRA fixtures with the exact model dimensions, input shape, input IDs, and target IDs required by a standalone browser parity run; the earlier validated P2 study remains pinned to its original exporter commit and hashes.
- Added separate Burn LoRA entry points that freeze the complete base, load the eight rank-4 QKV adapters, execute the same forward graph, and export only the exact adapter-gradient manifest. The established dense entry points remain intact.
- Proved real CPU/WASM browser parity over all 8 adapter tensors and 8,192 values: cosine `0.999999999999924`, relative L2 `3.851581853662727e-7`, maximum absolute error `1.3113021850585938e-6`, and `0.978 s` elapsed.
- Proved real WebGPU browser parity over the same adapter set: exact Python summed loss, cosine `0.9999999999998102`, relative L2 `6.144610293037611e-7`, maximum absolute error `1.1175870895385742e-6`, and `21.650 s` elapsed.
- Rebuilt and reran the original dense CPU/WASM path after the LoRA change; all 52 gradients retained their prior relative L2 result `2.788216012272494e-7`.
- Published the two-experiment `p2-browser-lora-parity-v1` study pinned to browser implementation commit `7aab3ccf536234a45bf0e21753e044d70af66fcd`; both result bundles and every checksum verified.
- Preserved the 21.650-second WebGPU cold-run cost as a negative finding rather than hiding it behind the passing numerical gate.
- Added explicit frozen-base LoRA mode to the restartable global-step coordinator. Assignments now bind the immutable base, initial adapter, worker-facing weight identity, full resume-state identity, exact eight-tensor trainable manifest, and LoRA manifest revision while dense assignments retain their existing protocol.
- Added adapter-only aggregation, one coordinator-owned AdamW state, separate adapter checkpoints, and strict base/adapter/optimizer reload validation. Two-worker updates matched independent centralized adapter checkpoints below `1e-6` relative L2 across both a first step and a resumed second step; only the eight adapter tensors receive optimizer state.
- Closed independent-review findings with basename-confined checkpoint artifacts, exact finite FP32 optimizer-moment validation, consistent integral optimizer-step checks, and negative traversal/corruption tests. The full resume-state identity now binds the exact adapter and optimizer artifacts, training and optimizer steps, dataset cursor and revision, and finite loss history; malformed trajectory metadata is rejected before save and on load. Dense pre-LoRA coordinator state and lock files migrate explicitly instead of failing after the protocol extension.
- Completed the connected HTTP/browser contract for adapter assignments: the coordinator serves the frozen base and current adapter independently, accepts only adapter gradients, publishes separate adapter-weight and resume-state identities, and serves the resulting adapter checkpoint.
- Proved the real authenticated two-browser CPU/WASM path. Both disjoint assignments returned exactly 8 tensors and 8,192 values and were accepted once; their gradient relative L2 errors were `4.80808764225882e-7` and `4.599249278948762e-7`. The canonical adapter survived coordinator restart and matched an independent centralized Python step with cosine `0.9999999999985897`, relative L2 `1.6795715402185043e-6`, and maximum absolute error `7.257913239300251e-7`.
- Published `p2-connected-browser-lora-v1`, pinned to implementation commit `cab9b6c49e4b9fad666a9f42396a28541ffc84ce`, with the two accepted browser assignments, separate weight and full resume-state identities, restart evidence, independent Python comparison, exact artifact hashes, and explicit one-host/synthetic-fixture limitations.
- Extended the persistent campaign runner through frozen-base LoRA initialization, versioned adapter checkpoints, per-checkpoint held-out evaluation, restart between global steps, retry recovery, explicit dashboard identities, and the `--lora-config` CLI path while preserving dense campaign behavior. A real two-step Burn CPU/WASM run accepted four browser assignments and 1,024 tokens across a coordinator restart; its final adapter SHA-256 was `91aea4b661edaab1c79dc78d3019d48c512ab02cfe56a92cc795a48bab0b982b`, its full resume-state identity was `cf4b3444f732f28daecf912263991ceaa6282c75bd07ae108d95b136aee1793d`, and final adapter parity remained within `7.3883157399442305e-6` relative L2 of the independent Python trajectory.
- Extended deterministic release bundles to validate and publish frozen-base LoRA state without overloading identities: `base-model.safetensors`, `adapter.safetensors`, optimizer moments, and complete resume metadata remain separate, while checkpoint selection still follows the campaign evaluation profile. A real two-step evaluated release proof improved held-out mean loss from `8.373976469039917` to `8.350127935409546`, passed its declared `0.0001` minimum-improvement gate by `0.023848533630371094`, selected step 2, built through the public CLI, and passed every generated `SHA256SUMS` entry.
- Published `p2-persistent-lora-release-v1`, pinned to implementation commit `b571d692152a0fe51892cdb4da1c91f4dc66d1e0`, with an executable proof script, exact generated-artifact hashes, per-step evaluation, restart and parity guardrails, deterministic release evidence, and explicit one-host/Python-oracle/local-corpus limitations.

### P3 — Qualify local memory tiers

**Status:** In progress

Test increasingly flexible placement while retaining one-worker assignment independence:

1. Full GPU residency.
2. GPU plus system-RAM offload.
3. Quantized frozen-base placement.
4. Explicit local-storage/NVMe out-of-core execution with bounded caches and prefetching.
5. Remote shard streaming only if computation per transferred byte can justify it.

The native worker is the first target for explicit offload. Browser support remains measurement-driven.

Completed within the first P3 measurement and native-baseline slice:

- Added a strict `orcacolony_worker_telemetry_v1` contract for assignment fetch, runtime initialization, artifact fetch, gradient compute, worker-observed payload bytes, optional runtime-specific memory observations, and device capacity. Non-oracle runtime profiles must report it; malformed, non-finite, out-of-range, or assignment-inconsistent reports fail closed.
- Added coordinator-owned artifact sizes, result-upload bytes, receive duration, persisted-result bytes, and current state-directory storage measurements to accepted receipts. Instrumentation survives restart validation and is aggregated without exposing worker identity in public campaign dashboards.
- Added visible dashboard cards for aggregate gradient compute, worker API payload transfer, peak observed worker memory, and coordinator storage.
- Proved the exact staged protocol with two real authenticated Burn CPU/WASM assignments. They reported `0.682199999988079` seconds of aggregate gradient compute, `10,681,648` frozen-base download bytes, `46,661,632` bytes of peak WASM linear memory, and retained `1.6795715402185043e-6` checkpoint relative-L2 parity.
- Selected a content-addressed cached-base native CPU profile as the first native baseline because the browser proof transferred the immutable 5,340,824-byte base once per assignment while computing each bounded gradient in well under one second. This is a measured baseline decision, not arbitrary commitment to CPU residency as the final placement architecture.
- Added an authenticated same-origin native worker that streams base and adapter artifacts to digest-named safetensors cache entries, revalidates tensor identities, loads only declared LoRA state, computes complete adapter gradients, reports process peak RSS, and submits through the existing coordinator contract.
- Proved two native assignments with one shared cache: the first fetched the 5,340,824-byte base and 33,520-byte adapter; the second fetched zero model and adapter bytes. Aggregate native gradient compute was `0.055746099998941645` seconds, process peak RSS was `316,760,064` bytes, and checkpoint relative L2 was `3.038902086339097e-7`.
- Qualified the same native cache contract against frozen TinyStories at T1 (6,901,760 parameters) and T2 (91,544,064 parameters). Both warm second processes fetched zero base and adapter payload bytes, both one-step held-out losses improved, and checkpoint relative L2 stayed at `1.2317732405376926e-7` and `2.1866353039878446e-7` respectively.
- Measured the T2 366,190,504-byte base, `1,746,419,712`-byte peak native process RSS, `378,093,711` bytes of coordinator storage, and `1.653274500000407` seconds of warm gradient compute. Warm cache revalidation plus runtime initialization took `2.9631700000027195` seconds, or 1.79 times compute.
- Published `p3-native-resource-profile-v1`, pinned to implementation commit `673017df9caa4d91f6bff96a39f56a40690c71e9`, with the real connected Burn dashboard, an automated isolated-process T1/T2 proof, exact artifact hashes, numerical and held-out guardrails, privacy filtering, and explicit one-host/payload-not-wire/random-base limitations.
- Selected a persistent native process/model session as the next throughput profile because one-shot warm setup dominates T2 compute. Quantized frozen-base placement remains the next memory profile; an idealized T2 FP32-to-int8 base conversion can reduce the observed process peak by no more than 15.73% before runtime overheads.
- Added bounded persistent native sessions through `--assignments N`. One authenticated process now retains the validated base/model, reuses unchanged adapter state, and fetches and loads only a new checkpoint-specific adapter after campaign advancement. Adapter refresh validates and converts the complete tensor set before copying any parameter, so a malformed refresh cannot leave a mixed in-memory adapter. A two-step test completed four assignments with one model build and two adapter loads.
- Proved the persistent session on the 91.5M-parameter T2 profile. The reused second assignment reduced base-cache validation from `1.013742800001637` seconds to `0.0000092999980552122` seconds and runtime initialization from `1.9494272000010824` seconds to `0.00012460000289138407` seconds while preserving zero model/adapter payload, `2.1866353039878446e-7` checkpoint relative L2, and the same held-out improvement.
- Published `p3-persistent-native-session-v1`, pinned to implementation commit `da606a03f185e3af48c34209daacb1396c4350e0`. Its isolated reproduction measured `0.00004199999966658652` seconds of total warm setup, a `0.9999858259905214` reduction from the one-shot baseline, while recording unchanged FP32 resident memory, single-worker shard filling, and retry-fault-injection limits.
- Ran and preserved the `int8-frozen-linear` comparison across T0/T1/T2. Per-row symmetric int8 frozen linears reduced unique resident model-tensor bytes by 43.08%, 50.18%, and 69.23%; T2 fell from `367,552,512` to `113,080,320` bytes.
- The int8 verdict is partial rather than connected acceptance: T2 adapter-gradient relative L2 was `0.030381955632016452` despite `0.9995400063256481` cosine and `0.00002760711142521869` relative loss-sum error. That drift is roughly 3,038 times the current FP32 bound, so the quantized profile remains offline until P4 defines and validates its own oracle/trajectory.
- Added the explicit offline `int8-per-output-symmetric-f32-dequant-v1` model builder with FP32-only activations and adapters. It converts an already constructed FP32 model, proving steady tensor storage but not lower peak startup RSS or larger-than-RAM loading.
- Ran and preserved the exact-FP32 `streamed-fp32-linear` comparison across T0/T1/T2. T2 retained tensor bytes fell from `367,552,512` to `27,482,112` (92.52%) while loss and every adapter-gradient value remained bit-identical.
- The T2 streamed gradient issued 95 authenticated layer reads totaling `673,053,696` bytes, 1.979 times the 340,077,504-byte linear artifact set, and took 1.307 times the full-resident runtime on a warm local-storage run after defensively copying each mapped tensor before validation/use.
- Added the explicit offline `streamed-fp32-frozen-linear-v1` builder. Every forward/backward reload validates exact tensor names, shapes, FP32 dtype, finite values, and tensor identity. Like the int8 builder, this first version converts a fully constructed model and therefore does not yet lower startup peak RSS.
- Added `direct-streamed-fp32-v1`: a restartable meta/empty builder that authenticates one raw base artifact before/after construction, partitions every base tensor exactly, materializes only non-linear/adapter tensors, and streams 48 T2 frozen linears directly from that artifact. It never invokes the resident FP32 builder.
- In isolated T2 processes, direct streaming preserved exact loss and gradient SHA-256 while reducing retained tensors by 92.52%, RSS after build by 49.67%, and RSS after gradient by 45.88%. Peak RSS fell only 1.94%; build and gradient runtime rose to 1.713x and 2.305x. The single-file profile is therefore insufficient for connected offload.
- Added deterministic `orcacolony_base_layer_bundle_v1` publication and `layer-bundle-streamed-fp32-v1` construction. The manifest binds the canonical base identity to exact resident/linear shard membership, raw transport hashes, semantic tensor hashes, shapes, sizes, modules, and bias contracts. Meta/empty startup opens no linear shard; each later forward/backward load validates an owned FP32 snapshot.
- Corrected the direct-profile resource interpretation: the earlier proof's default LoRA-manifest parse built a complete resident model before direct construction. With parser-only contract loading and the direct builder still performing its own complete artifact checks, direct final T2 peak fell from `1,377,845,248` to `740,708,352` bytes. The earlier measurement remains valid for that process path but did not isolate the direct constructor.
- In the new isolated T2 comparison, resident/direct/bundle returned the same `4687.0` loss and exact gradient SHA-256. The bundle retained `27,482,112` tensor bytes, opened zero linears at startup, and reduced final peak RSS by 46.35% versus resident. It matched corrected direct memory but reduced build time by 62.53% (`3.548609` to `1.329829` seconds) by removing complete-container startup work.
- Added connected layer-bundle publication and consumption. Coordinator state, locks, and assignment IDs bind the manifest; assignments carry the base identity plus ordered raw hashes, byte counts, and exact same-origin URLs. The new `python-native-cpu-layer-bundle-f32` provenance is accepted only when a bundle was actually assigned, under the unchanged strict gradient/loss/checkpoint contract.
- Fresh native downloads hash every artifact before atomic cache publication. Warm restart hashes the small manifest, validates exact membership/metadata, authenticates the resident shard, and checks every linear's semantic tensor digest on use without rescanning the complete base. Mutation, URL, membership, coordinator-reload, restart, persistent-session, and resident-compatibility tests fail closed or retain exact parity as appropriate.
- Completed a two-process T1 TinyStories campaign across a deliberate coordinator restart. The cold worker transferred the 27,621,509-byte bundle; the warm worker transferred zero model and adapter bytes; neither cached a monolithic base. Both gradients were exact, process peaks were 371,679,232 and 372,981,760 bytes, and aggregate checkpoint relative L2 was `1.2317732405376926e-7`.
- Evaluated the fixed use case at initialization and after the accepted connected checkpoint. Held-out mean loss over 16 declared TinyStories validation sequences improved from `9.041835904121399` to `9.041222333908081`.
- Qualified mixed exact placement at T2 with one resident worker and one layer-bundle worker across a coordinator restart. Both gradients were bit-exact, aggregate checkpoint relative L2 was `2.1866353039878446e-7`, and held-out mean loss improved from `9.171425819396973` to `9.168077945709229`.
- The connected T2 bundle worker reduced process peak RSS from `1,746,604,032` to `556,244,992` bytes (68.15%) versus resident execution. The tradeoff was a 2.238x gradient runtime and 1.116x cold fetch-plus-initialization time with essentially unchanged model payload size. Resident and bundle are therefore qualified as placements within one exact FP32 profile; approximate profiles remain separate.

### P4 — Qualify numerical and mixed execution profiles

**Status:** In progress

- Certify each precision, quantization, runtime, and offload profile against a canonical fixture.
- Separate numerical semantics from placement-only differences.
- Prove mixed-profile aggregation before admitting profiles to the same campaign.
- Record runtime provenance and revoke profiles that exhibit systematic numerical or operational failure.
- Fixed a 20-step T1 TinyStories FP32/int8 trajectory with evaluations at steps `0, 1, 2, 5, 10, 20`. Two-shard homogeneous int8 stayed within `3.1241858279774175e-7` gradient relative L2 and `1.5512369314698598e-7` final-adapter relative L2 of centralized int8.
- Proved exact int8 checkpoint restart for ten subsequent gradients, adapters, and AdamW states. Int8 held-out mean loss improved by `0.10614669322967529`, versus `0.10496234893798828` FP32, while retaining 50.18% fewer model-tensor bytes at T1.
- Preserved the numerical boundary: int8 reached 5.12% maximum per-step gradient drift and 2.64% final-adapter drift versus FP32. It is a homogeneous-profile candidate with a distinct profile-bound identity, not an exact-profile worker or mixed-aggregation participant.
- Added direct authenticated int8 construction from layer bundles. Each FP32 shard is validated, quantized, and discarded before the next opens; the resulting buffers/loss/gradients match convert-after-resident int8 exactly without complete FP32 startup residency.
- At T2 the direct builder retained the same `113,080,320` tensor bytes while reducing peak RSS through build from `1,380,302,848` to `458,956,800` bytes (66.75%) and final process peak to `845,414,400` bytes (38.75% lower). T1 final peak was 2.71% higher, so promotion remains scale/workload qualified rather than universal.

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

Bind int8 numerical-profile identity into assignment/checkpoint provenance and run a connected homogeneous int8 campaign against the fixed profile-specific oracle using direct bundle construction. Keep exact-FP32 acceptance and mixed aggregation unchanged.

## Remaining major work

- Connected homogeneous int8 assignment/oracle/checkpoint campaign proof.
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

### 2026-07-25

- Added deterministic authenticated base layer bundles and meta/empty construction with exact partition, restart, short-sequence, autocast, owned-snapshot, mutation, and no-linear-startup-read coverage.
- Proved exact T2 layer-bundle gradients while reducing resident-relative final process peak by 46.35% and direct-relative build time by 62.53%.
- Corrected the earlier direct-startup interpretation by isolating and removing a redundant resident validation build from artifact-backed startup.
- Advanced the immediate target to a real connected layer-bundle worker rather than stopping at the offline profile.
- Bound layer-bundle identities and exact shard URLs into authenticated coordinator state/assignments, added fresh-download and warm-use cache authentication, and retained strict FP32 result acceptance under a distinct runtime provenance.
- Completed a real T1 TinyStories campaign across a coordinator restart with exact gradients, zero warm model/adapter payload bytes, no monolithic worker cache, sub-`1.24e-7` checkpoint parity, and positive held-out loss movement.
- Qualified resident and layer-bundle workers together at T2: exact per-worker gradients, sub-`2.19e-7` aggregate checkpoint parity, positive held-out movement, and 68.15% lower bundle-worker process peak RSS with the measured runtime/payload tradeoff preserved.
- Started P4 with a fixed 20-step T1 int8 trajectory: positive held-out movement, exact restart, sub-`3.13e-7` homogeneous aggregation drift, and explicit separation from the 2.64%-diverged FP32 adapter trajectory.
- Added direct authenticated int8 bundle construction with exact converted-int8 parity and a 66.75% T2 build-peak reduction, while preserving the T1 final-peak counterexample and connected-profile provenance gap.
- Closed a delayed layer-bundle review finding: LoRA manifests now construct `CampaignConfig` from the same in-memory campaign payload whose digest was authenticated, eliminating a second-read semantic mutation race in both resident-verification and deferred-base paths.
- Closed the connected-layer-bundle restart/correctness review findings: partial cache repairs now report exact ordered downloaded-artifact membership and byte totals, pre-existing next rounds must match the campaign's bundle-publication mode before state advancement, adapter copies roll back on mid-install faults, and persistent native sessions quarantine any model whose refresh raises.

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
- Published and rebuilt the validated `p2-lora-numerical-v1` research record while retaining explicit limits on what the proof establishes.
- Proved frozen-base LoRA adapter-gradient parity in real CPU/WASM and WebGPU browser runs while retaining the original dense browser result.
- Published and checksum-verified the CPU/WASM and WebGPU browser parity experiments, including the slower WebGPU cold-run finding and explicit coordinator/offload limitations.
- Added the explicit adapter-only global-step coordinator contract, coordinator-owned adapter checkpoint and optimizer persistence, strict reload identities, and a two-step resume proof against independent centralized LoRA updates.
- Remediated checkpoint traversal and optimizer-corruption findings from an independent exact-tree review, preserved dense-state migration, and completed the LoRA HTTP/browser artifact and receipt contract.
- Added validated runtime, transfer, memory, and storage observations to worker assignments, accepted receipts, restart state, campaign aggregation, and the public dashboard.
- Proved the measurement protocol with two real authenticated Burn CPU/WASM assignments and used the repeated immutable-base transfer finding to select the first native baseline.
- Added and exercised a streaming, content-addressed cached-base native CPU LoRA worker; a warm second assignment eliminated base and adapter fetch payloads while retaining tighter-than-required adapter parity.
- Qualified cached-base native execution on frozen TinyStories at 6.9M and 91.5M parameters, with positive one-step held-out movement and sub-`2.19e-7` checkpoint relative L2.
- Published the P3 resource-profile study and used its measured T2 setup/compute ratio to select persistent process/model reuse before quantization and mapped offload.
- Added bounded persistent native sessions and proved T2 in-memory model/adapter reuse removes essentially all warm validation/initialization overhead without changing the canonical checkpoint.
- Published the persistent-session study with failure-atomic refresh evidence and retained its throughput-only, one-worker, FP32-resident limitations.
- Proved int8 frozen-linear storage reduction across T0/T1/T2, preserved the partial numerical result, and refused to admit its 1.7%–3.0% adapter-gradient drift under the FP32 worker identity.
- Proved exact-FP32 streamed linears retain only 7.48% of T2 model tensor bytes with bit-identical gradients, while preserving the startup-RSS and warm-page-cache limitations that still block a larger-than-RAM claim.
- Proved direct meta/empty construction halves steady T2 process RSS without materializing resident FP32 linears, but rejected it for connected use because complete-base scanning reduced peak RSS by only 1.94% and made gradients 2.305x slower.
