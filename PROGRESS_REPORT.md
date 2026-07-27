# OrcaColony Progress Report

**Last updated:** 2026-07-26

**Current phase:** Parallel P7 systems research and Record Patch pre-training qualification

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

P5–P7 method engineering and human-directed practical campaigns are parallel tracks. Mechanism work continues through measured executable slices without waiting for time-consuming manual evaluation. The campaign owner remains responsible for practical goals, data selection, example review, evaluation rubrics, checkpoint interpretation, and promotion decisions; automation preserves evidence rather than replacing those judgments.

## Overall build position

### Proven stable foundation

- Deterministic Python reference training and resumable safetensors checkpoints.
- Burn browser forward/backward execution through WebGPU and CPU/WASM.
- Complete named-gradient export and tolerance-based browser/Python parity.
- Connected coordinator-owned AdamW updates.
- Multi-worker assignments, leases, retries, stale-result rejection, and at-most-once accepted work.
- Crash-safe canonical checkpoint advancement and persistent multi-step campaigns.
- Owner-approved, default-deny participation and privacy-filtered v1 attribution.
- Frozen TinyStories data/tokenizer provenance and real-data T1 campaigns.
- Initialization and checkpoint evaluation with machine-readable success gates.
- Public dashboard, exact-origin browser deployment support, and deterministic privacy-filtered release bundles.
- A bounded 12-step, two-worker local T1 system proof that passed its declared held-out-loss gate and matched the centralized reference.

### Current boundary

The local numerical, coordinator, restart, and operational-release preflight is
proven. Capability/release infrastructure now fails closed on unsupported
objectives, separates behavioral promotion from language-loss diagnostics,
freezes baseline-improvement and evaluation-suite identities, snapshots
contributor-selected credit, verifies committed research artifacts, captures
environment context, and builds separate deterministic Hugging Face model/data
packages. Record Patch v1 now supplies the first frozen true-T2 task, legal
synthetic recipe, private final holdout, exact evaluator, campaign, and measured
initialization outputs. Python 3.11.15 reproduced the initialization identity,
all 32 predictions, and the evaluation JSON byte for byte from the earlier
Python 3.14 run. The complete v0.1 public-research acceptance path is still not
proven. The first centralized learnability check completed and failed its
behavioral gate despite a large language-loss improvement. It now needs a
passing same-trajectory continuation or a measured recipe revision, followed
by a reviewed release and clean load/generation test, actual model/data Hub
commit revisions, and a several-day campaign with distinct remote
owner-approved participants.

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
- Publish important findings as self-contained human-readable HTML under `reports/`; retain `research/` unchanged as the existing machine-readable study-record system.

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

**Status:** Complete

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
- Closed P3 with measured resident FP32, persistent resident reuse, exact local-storage layer bundles, and quantized frozen-linear placement. Direct authenticated T2 bundle-int8 construction reduced build peak RSS by 66.75% versus resident conversion; native GPU-plus-RAM and remote streaming remain explicitly unclaimed because the CPU-only host could not qualify them and transfer economics did not justify inventing support.

### P4 — Qualify numerical and mixed execution profiles

**Status:** Complete

- Certify each precision, quantization, runtime, and offload profile against a canonical fixture.
- Separate numerical semantics from placement-only differences.
- Prove mixed-profile aggregation before admitting profiles to the same campaign.
- Record runtime provenance and revoke profiles that exhibit systematic numerical or operational failure.
- Fixed a 20-step T1 TinyStories FP32/int8 trajectory with evaluations at steps `0, 1, 2, 5, 10, 20`. Two-shard homogeneous int8 stayed within `3.1241858279774175e-7` gradient relative L2 and `1.5512369314698598e-7` final-adapter relative L2 of centralized int8.
- Proved exact int8 checkpoint restart for ten subsequent gradients, adapters, and AdamW states. Int8 held-out mean loss improved by `0.10614669322967529`, versus `0.10496234893798828` FP32, while retaining 50.18% fewer model-tensor bytes at T1.
- Preserved the numerical boundary: int8 reached 5.12% maximum per-step gradient drift and 2.64% final-adapter drift versus FP32. It is a homogeneous-profile candidate with a distinct profile-bound identity, not an exact-profile worker or mixed-aggregation participant.
- Added direct authenticated int8 construction from layer bundles. Each FP32 shard is validated, quantized, and discarded before the next opens; the resulting buffers/loss/gradients match convert-after-resident int8 exactly without complete FP32 startup residency.
- At T2 the direct builder retained the same `113,080,320` tensor bytes while reducing peak RSS through build from `1,380,302,848` to `458,956,800` bytes (66.75%) and final process peak to `845,414,400` bytes (38.75% lower). T1 final peak was 2.71% higher, so promotion remains scale/workload qualified rather than universal.
- Added explicit `exact-cpu-fp32-v1`, `burn-ndarray-f32-v1`, `burn-webgpu-f32-v1`, and `int8-per-output-symmetric-f32-dequant-v1` identities. Assignment backends are filtered by profile; cross-profile submissions fail before telemetry or gradient acceptance, and exact CPU FP32 rejects one-ULP and signed-zero bit-pattern changes rather than relying on value equality.
- Added profile-authenticated LoRA checkpoint v2 while retaining exact, profileless v1 only as an immediate-predecessor restart migration. Release never migrates and requires explicit profile-bearing v2 state. Numerical identity now binds worker-facing weight and complete resume-state digests, campaign state/locks/history, assignments, accepted ledgers, evaluations, dashboards, CLI restart, and release provenance; wrong-profile restart and metadata tampering fail closed.
- Completed a real two-step connected T1 int8 campaign with one resident-converted and one direct-layer-bundle worker per step. All four gradients were bit-exact against the int8 oracle; maximum aggregate checkpoint relative L2 was `6.12852298785371e-8`; both persistent workers built once and transferred zero warm model bytes across a coordinator restart; held-out mean loss improved by `0.001623988151550293`.
- Preserved the arithmetic boundary: the connected final int8 adapter differed from exact FP32 by `0.011018934557552882` relative L2, so only placement profiles sharing the same numerical identity may mix. Cross-numerical-profile aggregation is prohibited rather than tolerance-widened.
- Published validated study `p4-numerical-profile-qualification-v1`, pinned to implementation commit `8af895ce8a0ac5925d4a612afb94f5dc0a42ecad`, with connected proof, 20-step trajectory, T2 direct-startup evidence, fixed TinyStories evaluation, negative findings, and CPU-only/local-host limitations.
- Closed the P3/P4 persisted-authority path with exact current and immediate-predecessor state, lock, assignment, evaluation, and checkpoint schemas; duplicate-key rejection; exact JSON types; exact-FP32-only migration; complete assignment-set authentication; and delayed migration persistence after parent, child, checkpoint, and finalization validation.
- Split input trajectory authority from completed-result trajectory authority. Input cursor/history continue to reconstruct every assignment and oracle, while separately authenticated result cursor/history, weight identity, and resume-state identity bind completed checkpoints and the next campaign round.
- Coordinator admission now retains owned model, adapter, layer-bundle, checkpoint, dataset, evaluation, ledger, and release bytes. Finalization, campaign versioning, dashboard aggregation, HTTP serving, and release publication no longer consume mutable admitted source paths.
- The P3/P4 closeout is published, synchronized, and backed by a 200-test gate. Repeated immutable reviews timed out or reviewed superseded trees, so no independent `passed=true` claim is made; further review-contract polishing is stopped unless a defect blocks a real campaign or P5 experiment.

### P5 — Explore rolling partial-model training

**Status:** Completed research — shallow block training not promoted

- Give capability-sized subnetworks or parameter slices to workers.
- Rotate coverage across a larger coordinator-owned global model.
- Reconcile updates from one frozen checkpoint round.
- Compare against replicated full-model training on the same use-case evaluation.
- Determine whether the quality and convergence tradeoff is acceptable for transformers.
- First bounded experiment: keep a complete canonical model at the coordinator, give a worker an executable submodel containing shared input/output components plus one selected transformer block, rotate the selected block across rounds, merge only explicitly mapped trainable gradients, and compare memory, transfer, coverage, and fixed-use-case loss movement with replicated full-model training. A negative result is publishable evidence rather than a reason to expand the protocol first.
- The first T0 rolling-block run is executable and measured. Four assignments selected blocks `0, 1, 2, 3`; each worker retained `2,973,184` tensor bytes versus `5,401,600` for the full model (44.96% lower), received `2,956,800` payload tensor bytes versus `5,336,064` (44.59% lower), and returned `793,088` gradient bytes for `198,272` trainable parameters. Fixed-fixture mean loss improved by `0.6197383403778076`, 38.02% of the `1.6302005052566528` improvement from four replicated full-model updates.
- The same run exposed the central tradeoff early: shared embedding/output state dominates each worker, and one persistent worker rotating through all four blocks eventually receives `5,336,064` unique tensor bytes—the complete full-model payload. This prototype proved lower peak residency and real mapped updates, not a way for one rotating worker to avoid complete-model exposure.
- The authenticated T1 experiment now provides that next evidence. Across twelve assignments and two complete six-block rotations, worker tensor residency was `11,877,376` bytes versus `28,000,256` for the full model (57.58% lower). Cold worker payload was 57.21% lower, and retaining shared state once reduced twelve-step model download from `141,742,080` transient-worker bytes to `46,561,280` persistent-session bytes: 67.15% below transient workers and 85.95% below twelve replicated full-model downloads. Each assignment separately returned `3,159,040` mapped gradient bytes.
- Held-out TinyStories mean loss improved monotonically from `9.041835904121399` to `8.740711331367493`, but the full-model control reached `7.6404712200164795`. Rolling improvement was `0.30112457275390625`, only 21.49% of the control's `1.4013646841049194`. An independent repeat matched every deterministic field exactly, including all seven evaluation pairs and both final model hashes; isolated-process wall time was 26.972–27.927 seconds and combined-process peak RSS was 700,350,464–705,372,160 bytes.
- The schedule-normalized block-sharded control assigned all six blocks from one checkpoint and batch before one coordinator AdamW step. All six live optimizer trajectories reached step 12, frozen shared tensors acquired no optimizer state, and every deterministic field repeated exactly. Held-out improvement rose to `0.5503915548324585`, 82.78% above sequential rotation but only 39.28% of the full control; only `0.007922887802124023` additional improvement—1.44% of the sharded total—occurred from step 8 to 12, showing a shallow-objective plateau.
- Block affinity kept each worker's unique tensor-position exposure at `11,811,840` bytes, 57.21% below full. Warm colony download per step was 31.34% below a full replica, but the colony paid `70,871,040` cold bytes (2.57 full-model equivalents), `279,367,680` persistent download bytes (only 15.67% below full), `506,818,560` round-trip bytes (23.51% below full), and `71,264,256` aggregate resident tensor bytes across six workers. Direct shallow block training is therefore preserved as a useful negative result and is not promoted.
- P6 and P7 implementation do not wait on the slower practical-campaign review loop. The first owner-directed campaign target is noisy bug reports to validated triage JSON; its data, rubric, baseline examples, checkpoints, and final interpretation remain manual approval points.

### Cross-cutting worker reliability — hidden assignment canaries

**Status:** Planned for public untrusted participation; not a P5 prerequisite

- Occasionally issue an ordinary-looking audit assignment whose expected gradient/loss is retained privately by the coordinator.
- Never expose the expected answer or a distinct public canary flag, and never aggregate audit-only results into the optimizer.
- Use sampled audit outcomes, duplicate work, and ordinary protocol failures to build bounded worker reliability/reputation evidence once per-assignment oracle computation no longer scales.
- These are compute-integrity checks, not planted training-data canaries: no synthetic secret examples are inserted into the corpus, learned by the model, or later searched for in model output.
- The current trusted-participant pilot already verifies every submitted assignment against a coordinator-known oracle, so adding a reputation subsystem now would duplicate existing admission rather than advance the larger-model research.

### P6 — Explore exact recoverable local tiled computation

**Status:** Completed bounded local feasibility — network/asynchronous operation remains open

- Divide one representative transformer layer into retryable matrix or tensor tasks.
- Persist dependency state at the coordinator.
- Reassign failed tiles without requiring permanent peer pipelines.
- Reconstruct exact forward and backward results.
- Measure worker memory, network transfer, cache reuse, latency, and compute-to-transfer ratio before expanding to a complete model.
- The first exact tracer replaced block 2 inside the real full-model graph. The coordinator produced the true prefix activation, the tile returned the block output, the coordinator suffix produced the true output adjoint, and the tile returned both exact block gradients and the input adjoint needed for prefix backpropagation. The coordinator's selected block executed zero times while the tile executed once.
- Raw gradients, globally clipped gradients, complete AdamW tensor state, and the post-step model all matched the centralized CPU FP32 reference exactly with maximum absolute differences of `0.0`. Two isolated CLI runs repeated every deterministic field and digest.
- The tile retained `198,272` parameters, 14.86% of the `1,334,016`-parameter T0 model. Its `793,088` model bytes were 85.14% below full; those weights plus `793,088` gradient bytes and four `262,144`-byte activation/adjoint boundaries produced a `2,634,752`-byte cold round trip (75.31% below replicated full) and `1,841,664` warm bytes (82.74% below full). Boundary traffic alone was `1,048,576` bytes, 19.65% of one full-model payload.
- This does not yet establish worker peak RSS, coordinator memory reduction, connected latency, asynchronous retryability, or a model larger than coordinator memory. Exactness restores a live forward/backward dependency chain whose volunteer reliability cost must be measured explicitly.
- The authenticated T1 sweep substituted every block position `0..5` independently. First, middle, and final tiles all matched centralized raw/clipped gradients, complete AdamW state, and post-step model identity with `0.0` maximum differences; two complete sweeps repeated every deterministic nested field under dataset revision `99e5642bec2a9fa0b7f6175ed5f4821bf4f9aa2c08ec1038f12bfdfb302bb4af`.
- Each T1 tile retained `789,760` parameters, 11.44% of the `6,901,760`-parameter model, and its `3,159,040` model bytes were 88.56% below full. From T0 to T1 the full payload grew 5.17x while four `524,288`-byte boundaries made `2,097,152` activation/adjoint bytes—only 2x the T0 boundary and 7.60% of a T1 full-model payload. Per-block cold/warm accounted round trips were `8,415,232`/`5,256,192` bytes, 84.76%/90.48% below full; all six positions totaled `50,491,392`/`31,537,152` bytes versus `331,284,480` for six full round trips.
- T1 timing still overlaps locally—centralized `0.358`–`0.417` seconds versus tiled `0.360`–`0.391`—and combined process peak RSS of `1,059,287,040`–`1,090,740,224` bytes remains contaminated by repeated full-model/optimizer allocation. Process-separated worker memory and transport are now the next evidence boundary.
- A persistent trusted-local `spawn` child now executes T1 block 2 across two different same-checkpoint cursors with one model transmission. Safetensors/JSON frames validate exact schemas, shapes, dtypes, finiteness, assignment order, and tile-state identity; finite polling and forced termination bound the child lifecycle. Both independent runs reproduced every deterministic field, raw/clipped gradient, AdamW state, and updated model exactly.
- Serialized cold/warm tile tensor payloads were `8,417,672`/`5,257,632` bytes, 84.75%/90.48% below a replicated-full round trip. Safetensors overhead was only `2,440` cold bytes (0.03%), and all JSON controls totaled `1,487`–`1,488` bytes after adding the model-readiness handshake; private multiprocessing frame headers are excluded explicitly. Local forward/backward IPC round trips measured `13.56`–`17.23`/`25.04`–`28.16` ms.
- Cold spawn-to-ready took `2.88`–`2.93` seconds. Isolated worker RSS was `199,143,424`–`199,733,248` bytes before model load, `289,431,552`–`290,197,504` after load, and `330,760,192`–`331,751,424` final peak. Runtime/autograd overhead therefore dominates the `3,160,040`-byte tile; no resident-memory saving may be claimed until a full-model child is measured under the same harness.
- The launcher is trusted-local research only: no sandbox, Job Object, restricted token, environment scrubbing, persisted boundary transaction, crash/reassignment proof, network transport, or refreshed-weight trajectory. Report 006 makes those boundaries explicit.
- A matched persistent full-model child now uses the same trusted-local spawn lifecycle, exact T1 checkpoint, two cursors, deterministic settings, safetensors/JSON framing, and coordinator-owned clipping/AdamW. Both full-process runs repeated every deterministic field and exact gradient, optimizer, and model identity.
- The full worker peaked at `665,174,016`–`666,329,088` bytes versus `330,760,192`–`331,751,424` for the tile. The tile therefore reduced isolated worker peak RSS by 50.21%–50.27%, saving `334,413,824`–`334,577,664` bytes. Current RSS after model load was 24.25%–24.55% lower.
- Matched cold/warm tensor payload was `55,237,296`/`27,623,160` bytes for the full worker and `8,417,672`/`5,257,632` for the tile, a tile reduction of 84.76%/80.97%. Report 007 qualifies worker-memory feasibility but makes no end-to-end speed claim because coordinator prefix/suffix and optimizer work are outside the tile IPC timers.
- A coordinator-owned boundary transaction now persists tile weights, prefix activation, first forward output, output adjoint, backward result, identity, phase history, file digests/sizes, and a durable apply bit. The seven phases are `prepared → forward_accepted → worker_lost → replay_verified → adjoint_persisted → result_accepted → applied`.
- Before coordinator mutation, apply validates the exact manifest schema and runtime identity, recomputes the transaction ID, requires complete phase history and a closed five-file map, verifies every owned file digest/size, and validates every result tensor. The first and replacement worker acknowledgements are bound to the expected canonical tile state.
- The first worker was deliberately terminated after forward (`exit=-15`). A replacement loaded only persisted tile/input bytes, reproduced the serialized output byte-for-byte, returned exact gradients/input adjoint, and exited cleanly. Centralized and recovered raw gradients, clipped gradients, AdamW state, and model bytes were identical in two T1 runs; a second apply attempt was rejected before mutation.
- AdamW is staged on shadow model/optimizer state. Partial write, flush, `fsync`, and replacement failures clean their incomplete temporary file; a real applied-state replacement failure also restores coordinator gradients and retries byte-exactly from the unchanged `result_accepted` transaction. Only successful publication installs the staged state. Evidence v2 binds the pre-update checkpoint and lets readers recompute the transaction ID from committed fields.
- Recovery took `3.11`–`3.15` seconds, dominated by `2.79`–`2.82` seconds of replacement initialization (`89.45%`–`89.67%`). Recovery retransmitted `3,684,408` bytes before replay and persisted `8,417,672` tensor bytes plus a `1,340`-byte manifest. Report 008 closes bounded P6 feasibility as qualified but conditional.
- P6 does **not** establish coordinator crash recovery, externally authenticated manifests, sudden-power-loss directory durability, hostile payload safety, network transport, refreshed-weight multi-step execution, coordinator-memory reduction, or production readiness. The coordinator and its prefix autograd graph survive the tested worker loss; death after applied-state publication but before live candidate installation remains outside scope because this tracer does not persist model/optimizer checkpoints. If incomplete-file removal itself fails, retry stops fail-closed and requires intervention.

### P7 — Explore sparse experts and other advanced topologies

**Status:** Active research — exact independent T0 experts measured

- Expert-sharded or sparse models.
- Router and shared-trunk training.
- Expert availability, replication, reconciliation, and release semantics.
- Model-parallel systems for unreliable peers only after simpler tracks establish the relevant evidence.
- A capacity-bounded top-1 router and four independent token-wise MLP experts now reproduce centralized sparse raw/clipped gradients, router/shared/expert AdamW state, loss, and updated model bytes exactly. The worker objective includes the complete token loss and returns the expert/head gradients plus shared-trunk input adjoints, so it does not require P6's second live output-adjoint round trip.
- Byte-canonical aggregation required an explicit untied output head: tying input embedding and output projection made independently returned head gradients accumulate in a different FP32 order. This deliberate extra shared state is part of the P7 payload result, not hidden implementation detail.
- One cold expert worker carries `2,626,048` bytes versus `7,167,488` full bytes, a 63.36% reduction; the `2,098,176`-byte shared norm/head cache is 79.90% of that cold worker payload. With it cached, expert-only payload is `527,872` bytes, 92.64% below full. But four cold workers duplicate `10,504,192` model bytes and produce a `21,536,768`-byte round trip, 46.55%/50.15% above full. Warm model bytes fall 70.54% and warm round trip only 8.36% below full because shared-head gradients still dominate uploads. The full comparison includes its `8,192` input bytes.
- Unconstrained routing counts `[108, 147, 152, 105]` had 16.87% load CV. Capacity 128 rerouted 44/512 tokens (8.59%) to force `[128, 128, 128, 128]`; perfect reported balance is therefore policy intervention, not learned-router evidence. Report 009 keeps the one-step synthetic/no-process/no-quality limitations explicit.

## Immediate next bounded target

Run two explicitly separate bounded tracks:

1. **Capability track:** use the reproduced Python 3.11 Record Patch baseline
   and the failed 128-step public qualification as the fixed starting evidence.
   Freeze a continuation schedule, resume the exact step-128 model, AdamW state,
   data cursor, objective, learning rate, and decoding policy, and evaluate only
   later public milestones. Do not open either reserved holdout or request
   donated compute until exact behavioral learning appears. If same-trajectory
   exposure still fails, test target-only loss and learning-rate changes as
   separate controls.
2. **P7 systems track:** freeze and persistently cache the untied final
   norm/output head in both centralized and distributed sparse controls. Train
   only router, shared trunk, and experts, preserve byte-exact canonical AdamW,
   and recompute cold/warm result traffic without duplicated head gradients.
   Advance to authenticated T1/process workers only if this one-variable control
   produces a meaningful aggregate advantage rather than an
   individual-worker-only saving.

## Remaining major work

- P7 cached-head, authenticated-data, process-memory, retry, and quality qualification.
- Complete the first useful specialization campaign at the specification's
  17,538,816-parameter T2 tier. Record Patch v1 now freezes its task and starting
  measurement and retains an initial negative learnability result plus a
  step-128 trained checkpoint, but no passing learnability gate, volunteer work,
  or promotion evidence exists. The historical 91,544,064-parameter files and
  P3/P4 evidence retain their existing `t2` identifiers for provenance but are
  treated as a legacy 91M memory-stress tier, not as the current T2 capability
  tier.
- Extend the new Record Patch training-effect analyzer with forgetting
  guardrails, ablations, and repeated-seed uncertainty where relevant. The
  first trained comparison now includes checkpoint output deltas, error
  taxonomy, data buckets, gradient/update diagnostics, prompt/answer loss, and
  exact plus nearest-training-record checks.
- Promotion records currently validate artifact IDs, hashes, and locations, but
  the capability release path does not yet fetch or snapshot arbitrary external
  evaluator/sample-result URIs. The first real campaign must put those artifacts
  in a digest-verified `repo:` research bundle or extend the publisher to bundle
  them; a link and a claimed hash alone are not durable proof. Holdout separation
  is likewise enforced by the release workflow, not by a cryptographic
  prohibition against an operator inspecting it early.
- Exercise contributor credit-profile v2 with real volunteer choices and review
  the generated release snapshot. The implementation separates worker authority
  from reloadable credit choices and accounts for accepted tokens,
  worker-reported time, roles, and opted-in hardware classes.
- Accepted direct-training work is covered by the generated attribution
  snapshot. A separate evidence ledger for auxiliary work such as data curation,
  evaluator construction, review, hosting, or failed-but-helpful compute attempts
  does not yet exist and must not be implied by the current role labels.
- Migrate remaining generated/external research references into durable,
  digest-verifiable locations. Committed `repo:` artifacts now verify and bundle;
  other URI schemes are explicitly reported as unresolved rather than implied
  proof.
- Build a real deterministic Hugging Face package, choose source-compatible
  model/data licenses, run the custom dense/LoRA load and generation smoke test,
  review private repositories, and record both Hub commit revisions. Record Patch
  reserves `OrcaColony/record-patch-t2-v1` and
  `OrcaColony/record-patch-v1`; neither repository exists and no model/data
  package has been uploaded.
- Hidden-canary or replicated-work compute integrity before untrusted public
  participation; the diagnostic oracle-gradient endpoint is not a production
  proof that donated computation occurred.
- Operator-owned remote trusted campaign when deployment inputs are available.

## Current blockers and hold-ups

- No blocker prevents the specification and research-framework work.
- Volunteer training remains blocked. The first bounded centralized check
  passed its language-loss condition but failed at `0/32` public exact matches
  and `0/32` strict valid JSON. No donated compute should be requested until a
  predeclared continuation or recipe control shows exact behavioral learning on
  public data.
- Remote trusted deployment still requires operator-owned HTTPS hosting choices and approved participants, but that does not block local research milestones.
- Native GPU and GPU-plus-system-RAM placement were not qualified on the CPU-only PyTorch host; browser WebGPU remains a separate measured numerical/runtime profile rather than evidence for a native offload path.
- Larger-model execution methods are hypotheses until measured; none should be described as supported merely because a paper or prototype demonstrates the general idea.

## Change record

### 2026-07-26

- Added a separate post-run analysis command that verifies every retained run
  checksum, preserves all public sample outputs, splits teacher-forced prompt
  and answer metrics, classifies output failures, summarizes gradient and
  update trajectories, and recomputes exact and nearest training-record
  comparisons without accepting a private-holdout input.
- Completed the predeclared 128-step centralized check under clean source
  revision `96d2af5bcde7a872b0521c9ca5ab95c6f760d673`. Step 128 lowered
  public language mean loss from `9.120742341162453` to
  `1.56358497243532`, but scored `0/32` exact, `0/32` semantic, and `0/32`
  strict valid JSON. All 32 outputs became object-shaped with duplicate keys,
  so the behavioral gate failed and volunteer training remains blocked.
- Preserved the negative result in Report 011 with all public checkpoint
  outputs, 128 step diagnostics, prompt/answer token metrics, exact and nearest
  training comparisons, compute measurements, checksums, credit boundary, and
  next control. The run covered about `0.111` packed-data epochs, clipped 114
  updates, used 1,631,789,056 bytes peak RSS, and opened neither reserved
  holdout.
- Froze the first Record Patch centralized learnability protocol before the
  measured run. It fixes checkpoints `0`, `1`, `8`, `32`, and `128`, selects
  only by public language-validation loss, requires at least `0.1` language-loss
  improvement plus one additional exact public behavioral match, caps execution
  at one CPU thread and 3 GiB peak RSS, and excludes both reserved holdouts.
- Added a separate learnability runner without modifying the hash-pinned frozen
  evaluator. It records every step's loss, pre-clipping gradient norm, clipping
  decision, update norm, checkpoint identity, public language and behavioral
  results, timings, environment, and checksums. Its diagnostic trajectory is
  regression-tested against the established centralized reference.
- Audited the progress report, specification, campaign files, research records,
  release path, and current implementation as a research vehicle rather than
  treating systems correctness as model-capability evidence.
- Corrected the roadmap boundary between the specification's approximately
  17.5M-parameter T2 and the immutable historical 91,544,064-parameter `t2`
  memory-stress campaign identifiers.
- Made the campaign objective executable and fail-closed. Every current training,
  evaluation, LoRA, partial-model, tiled-process, recovery, and sparse-expert path
  now routes its token loss through the validated objective contract; unsupported
  objectives and masks reject during campaign loading.
- Added the capability-campaign contract with an exact baseline, absolute metric
  threshold, positive minimum baseline improvement, frozen behavioral
  data/evaluator revisions, distinct behavioral splits, disjoint language-loss
  validation/holdout ranges, guardrails, analysis plan, checkpoint-selection
  policy, licenses, a private-review/public-release visibility policy, and
  `OrcaColony/...` Hub destinations.
- Separated the post-selection language-loss holdout diagnostic from behavioral
  promotion evidence. Promotion now binds the selected checkpoint, training
  dataset, behavioral suite, exact baseline, threshold and improvement, every
  guardrail, limitations, evidence hashes, and a reproduction command.
- Added v2 contributor attribution with named, pseudonymous, and anonymous
  choices; optional HTTPS profile/team/roles; independent totals/hardware
  preferences; worker-reported total/gradient time; public hardware classes; a
  deterministic credit revision; release-time snapshots; and generated
  `CONTRIBUTORS.md`. V2 worker authority remains locked while credit choices may
  be refreshed safely on coordinator reload; v1 recovery behavior remains
  unchanged.
- Fixed the public dashboard to honor the v2 totals preference rather than
  linking a public alias to per-assignment token counts without consent.
- Made research recording resolve, digest-check, and snapshot committed `repo:`
  evidence, label other artifact schemes unresolved, capture
  `environment.json`, and surface measurements/provenance in `RESULT.md`. Corrected
  the committed P4 connected-proof digest to the repository's LF bytes.
- Added a capability-report template requiring sample-level output deltas,
  bucketed errors, optimizer diagnostics, memorization/forgetting checks,
  negative findings, contributor credit, environment identity, exact artifacts,
  and Hub revisions.
- Added deterministic, network-free Hugging Face model and dataset package
  builders, exact closed-file/checksum verification, custom dense/LoRA loading
  and generation, campaign-locked visibility policy, explicit licenses, and an
  authenticated two-repository publisher that accepts no token argument.
- Made Hub packaging private by default, added visible model/dataset-card
  contributor totals, retained optimizer/restart evidence, separated the
  framework software license from the selected model-weights license, required
  the dataset publication license to match its frozen source manifest, and made
  publication refuse visibility mismatches or stale remote files. Successful
  publication now requires a separate local JSON record of both Hub commits.
- Published the public `OrcaColony/README` organization card from its retained
  repository source and recorded exact Hub revision
  `7a27bbb8069fb463b0da67bd38f4c7763bb400ba` plus the public file digest. The
  card now records the frozen Record Patch task, supported-runtime baseline,
  learnability gate, negative-result policy, and contributor-credit policy
  without implying that a capability model has already been released.
- Selected Record Patch v1 as the first narrow capability task. It applies
  ordered `SET`, `DELETE`, and `RENAME` operations to flat JSON records and
  requires exact canonical output. The task is project-generated, contains no
  scraped or teacher-produced text, and is licensed CC0-1.0.
- Added a deterministic hash-stream generator, transcript builder, byte-level
  tokenizer/packer, strict prompt-and-oracle validation, sample-level evaluator,
  exact per-bucket metrics, Wilson interval, guardrails, oracle reproduction,
  and initialization-baseline CLI.
- Froze 32,768 training examples, 1,024 language-validation examples, 32 public
  behavioral-validation examples, and 128 separately keyed final-holdout
  examples. The public suite lock is
  `91acf17f01bc1c59d6aeb1bb75322b0021b164e4fa750fa3ad90eed77531d087`;
  the final examples remain under ignored local artifacts and only their count
  and digest are public.
- Added the true 17,538,816-parameter T2 campaign with exact packed-data hashes,
  disjoint 71/72-sequence language-loss slices, a 2,048-step proposed budget,
  fixed behavioral thresholds and guardrails, Apache-2.0 model weights,
  CC0-1.0 data, contributor-credit intake, and private-review-first
  `OrcaColony/record-patch-t2-v1` and `OrcaColony/record-patch-v1` destinations.
- Ran the deterministic initialization baseline on all 32 public cases with
  greedy decoding. It scored `0/32` exact, `2/32` valid JSON, `2/32` canonical
  JSON, and `0/16` across the four single-operation buckets. The final holdout
  was not opened. Report 010 retains every output and reproduces evaluation JSON
  SHA-256 `e05c2c67db6b151678124777ecf3d232b6cc3fc1bbf079d6a6190cbba4a516f6`.
- Preserved the original Python 3.14.4 baseline environment, then repeated it
  with Python 3.11.15 and CPU-only PyTorch 2.13.0. The initialization identity,
  all 32 predictions, and evaluation JSON were byte-identical. The final holdout
  remained unopened. A full fresh freeze with the original withheld key also
  reproduced the public suite, private holdout digest
  `f140f35e9d2e5e9292dc0f347c5d6c95658caf7bcad20e8ec81509b9e98ba496`,
  packed manifest
  `4e71f26b3e06b15360eef333ced96a16791dc2bc9ad9c29ff11d8b728106baae`,
  and campaign
  `039fea26e0eb5f86146e152bc2bfaffe96bb78db0b8318cc21982ef0b872bfe5`.
  A centralized learnability check now gates volunteer training.
- Lightweight validation passed source/test compilation, diff whitespace checks,
  deterministic participant and Hub-package smokes, reconstruction of all nine
  committed research records, a byte-stable Record Patch re-freeze, public
  checksum verification, packed campaign validation, final-holdout separation,
  oracle scoring, exact baseline recomputation, and local-link checks. The
  temporary static report preview returned HTTP 200 and the expected page title,
  although the shared browser snapshot failed. Python 3.11 Record Patch tests
  passed, and the baseline reproduced exactly.
- Made the official CPU PyTorch index the default development source and removed
  the unused CUDA dependency graph from `uv.lock`. Preserved the earlier T0 LoRA
  manifest for historical evidence, added a backend-specific CPU fixture for
  current tests, and documented that seed plus Torch version alone is not a
  portable initialized-weight identity across wheel builds.
- Added an explicit campaign wire-format serializer after the objective contract
  became an internal dataclass field, and corrected the participant-credit test
  to select contributors by stable ID rather than sorted position. The complete
  Python 3.11 CPU suite now passes: `259 passed`.
- No campaign training job, GPU job, model/data repository creation, or
  model/data upload ran. The test suite did execute its bounded CPU gradient and
  optimizer checks.

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
- Completed P3 by qualifying persistent resident, exact authenticated layer-bundle, and quantized frozen-linear placements while preserving measured negative or unsupported GPU-plus-RAM and remote-streaming boundaries.
- Completed P4 with explicit runtime/numerical identities, bit-exact exact-CPU admission, profile-specific int8 oracles, profile-bound v2 checkpoint/restart/evaluation/release provenance, v1 FP32 compatibility, and fail-closed cross-profile admission.
- Ran the connected T1 mixed-placement int8 proof across a coordinator restart: four exact profile gradients, `6.12852298785371e-8` maximum checkpoint relative L2, zero warm model payload, positive held-out movement, and deliberate 1.1019% separation from exact FP32.
- Published the validated `p4-numerical-profile-qualification-v1` research record and paused before P5 for roadmap discussion.
- Extended deterministic release envelopes so dense and LoRA bundles, public dashboards, ledgers, evaluations, and top-level manifests all publish the campaign-locked numerical profile; the release test now exercises a real int8-profile checkpoint.
- Replaced strict-profile tensor value equality with canonical tensor-byte identity after an adversarial signed-zero mutation proved numerically equal tensors were not necessarily bit-exact.
- Closed the P4 immutable-review blockers: legacy campaign and global-step profile migration now requires the exact recognized legacy lock; accepted result files bind both complete-file and canonical-tensor digests and are revalidated against their profile oracle before restart or finalization consumption. A lock-bound result-identity revision prevents removed digests from masquerading as legacy state; type-exact JSON comparison rejects boolean and floating-point aliases, duplicate keys, and hybrid predecessor locks. Migration also requires the exact immediate-predecessor state and assignment key sets with no combined older migration, so unknown, missing, or partially revisioned fields cannot be relabeled. Result and oracle artifacts use confined digest-derived names and owned handle-checked snapshots with reparse, replacement, and in-place-change detection; acquisition retains the directory enumeration handle and file descriptor through post-read root/path identity and metadata checks. Oracle raw/tensor identities are recorded, while reload requires exactly the campaign worker count, binds the count and ordered assignment IDs in the current round lock, independently reconstructs every indexed shard with type-exact basis comparison, and recomputes its expected loss and gradients from owned model/adapter snapshots, the pinned campaign, checkpoint-bound dataset cursor, and shard before any recognized pre-revision exact-FP32 result may migrate. A truncated assignment list cannot make `all()` authorize partial-batch finalization. Integer step/cursor and type-exact loss history must agree with the authenticated base checkpoint, and the current campaign lock also binds the cursor.
- Persisted evaluations now bind the campaign, dataset, integer step, numerical profile, and authenticated checkpoint identity. Missing profiles migrate only when the field is genuinely absent from the exact recognized legacy exact-FP32 record; explicit nulls, extra/partial fields, unknown formats, and approximate-profile records fail closed. Release construction independently rejects missing, null, or mixed campaign, evaluation, and selected-checkpoint profiles and directly compares the selected LoRA evaluation's base-model, adapter, weight-checkpoint, and resumable-state identities with the authenticated selected checkpoint.
- Closed the final delayed immutable-review findings without transferring approval from superseded trees. Current global-step, assignment, campaign, and checkpoint records now require exact field sets and exact JSON scalar types; dense and LoRA checkpoint trajectories reject integral floats, booleans, duplicate keys, unknown fields, unsafe artifact names, and malformed optimizer tensor sets. Approximate-profile result-identity predecessors reject even when no assignment has yet been accepted. Campaign restart binds canonical round/checkpoint paths under retained directory handles, validates every parent record before persisting any child migration or ledger rewrite, and rejects wrong LoRA evaluation methods. Finalization uses owned in-memory model, adapter, optimizer, and reference snapshots rather than reopening admitted paths, while HTTP responses serve owned or freshly digest-revalidated bytes. Release accepts only explicit profile-bearing LoRA v2 checkpoints and binds the campaign-lock revision directly to the released campaign payload.
- Closed the remaining stale-tree findings only where they reproduced on the latest code. Packed datasets now authenticate each exact manifest member once, parse safetensors from retained bytes, clone their tensors, and provide the same admitted bytes to campaign/release publication. Dense checkpoints validate exact step/optimizer/cursor/loss-history relationships, complete per-parameter AdamW state, common optimizer steps, FP32 finite model tensors, and step-zero moments before writing. Campaign reload and release recompute every persisted evaluation from retained dataset and versioned-checkpoint snapshots, so finite self-consistent metric mutations cannot alter checkpoint selection or the success gate. Persisted and submitted assignment losses also require exact finite JSON floats rather than accepting integer or boolean aliases.
- The resulting closeout gate passed `190` tests plus source/test compilation, lock verification, diff validation, all five campaign/release/native-worker/PEFT/research CLI entry-point checks, and a bounded added-line credential/dangerous-execution scan.
- Processed the delayed immutable review of superseded tree `88151e1b2eebf3c9ff91f4428e80ac2d248bf0f4` as a tree-specific non-approval and reproduced its remaining concrete claims against current code. Obsolete participant, training-method, and result-protocol migration shapes now reject before mutation instead of bypassing current exact schemas; only the explicitly recognized exact numerical-profile and accepted-result predecessors remain migratable. Persisted assignment states now bind exact attempt and lease lifecycle types and participant authority. Completed global-step loss totals and checkpoint metrics are independently recomputed from authenticated assignments, reference tensors, and checkpoint tensors, and campaign restart requires its last-checkpoint metrics to match that validated child authority exactly.
- The post-review follow-up gate passed `200` tests plus source/test compilation, lock verification, diff validation, all five CLI entry-point checks, and the bounded added-line credential/dangerous-execution scan.
- Began P5 with a runnable T0 rolling-block worker rather than more P4 contract work. One complete four-block coverage cycle mapped real gradients into a coordinator-owned full model, reduced per-worker tensor residency by 44.96%, and improved fixed-fixture loss, while exposing that complete persistent-worker coverage still transfers the full model's unique tensor bytes and achieves only 38.02% of the four-step full-model baseline improvement.
- Recorded hidden coordinator-known audit assignments as a future sampled reputation mechanism for public untrusted workers, distinct from planted training-data canaries and explicitly not a prerequisite for P5 or the trusted-participant pilot.
- The first P5 vertical-slice gate passed `202` tests plus compilation, lock verification, diff validation, the new CLI help path, and a reproducible real four-step experiment.
- Added the separate root `reports/` human findings layer without changing `research/`. Report 001 turns the measured P5 T0 result into a self-contained HTML decision record with exact evidence, positive and negative findings, limitations, reproduction commands, and the next experiment. Method engineering and human-directed practical campaign review now proceed in parallel.
- Added persistent rolling-worker sessions, authenticated external-dataset execution, held-out trajectory capture, download accounting, timing partitions, and combined-process peak-RSS evidence. Two independent twelve-step T1 runs reproduced every deterministic result exactly while exposing the sequential topology's 21.49% quality-progress ratio.
- Published Report 002 with both raw evidence files. It preserves the 57.58% worker-residency and 85.95% repeated-download improvements alongside the full-coverage transfer limit, widening quality gap, non-equivalent update schedule, and the block-sharded next experiment.
- The P5 T1 gate passed `206` tests plus source/test compilation, lock verification, diff validation, CLI help, exact Report 001 reproduction, two exact deterministic T1 repetitions, report-claim recomputation, local-link validation, and browser inspection.
- Added a schedule-normalized block-sharded topology: six persistent block-affine workers load one shared checkpoint/batch, map every block gradient, and advance one coordinator AdamW step only after complete coverage. Live optimizer-state checks bind all block steps and prove frozen shared parameters retain no optimizer state.
- Repeated the twelve-step T1 run exactly. Schedule normalization improved loss progress 82.78% over sequential rotation but reached only 39.28% of the full control and plateaued after step 8; per-worker exposure stayed low while cold colony transfer and aggregate residency reached 2.57x and 2.55x the full-model values.
- Published Report 003 and closed P5 without promotion. The negative result advances P6 toward exact boundary-preserving tiled execution instead of further scaling the shallow surrogate objective.
- The P5 conclusion gate passed `208` tests plus source/test compilation, lock and CLI verification, two exact deterministic block-sharded runs, report-claim recomputation, local-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Began P6 with a real exact boundary substitution rather than another shallow objective. Block 2 executes only on the tile while true prefix/suffix activations and adjoints reconstruct byte-identical full-model gradients, clipping, AdamW state, and model update.
- Repeated the T0 boundary tracer exactly and published Report 004. The tile uses 14.86% of model parameters and the accounted cold/warm round trips are 75.31%/82.74% below a replicated-full round trip, while autograd intermediates, process-separated worker RSS, coordinator residency, network latency, and retry state remain explicit open gates.
- The first P6 gate passed `210` tests, including warning-strict tiled tests, plus source/test compilation, lock and CLI verification, two exact deterministic runs, report-claim recomputation, local-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Generalized exact boundary substitution to authenticated packed data and added a dataset-bound all-block sweep envelope. Every T1 position independently reproduces the same centralized gradient, optimizer, and model identities while the envelope binds complete block coverage to the exact campaign and manifest.
- Repeated both six-block sweeps exactly and published Report 005. T1 reduces tile parameter share to 11.44%, cold/warm accounted round trips to 15.24%/9.52% of full, and boundary traffic to 7.60% of one full payload; process separation is now the explicit next gate.
- The authenticated all-block P6 gate passed `213` tests, including warning-strict tiled tests, plus source/test compilation, lock and CLI verification, exact nested sweep repetition, report-claim recomputation, local-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Added a bounded persistent tile child protocol with exact safetensors tensor frames, JSON controls, deterministic child configuration, finite lifecycle, one cold model transfer, and two warm-compatible same-checkpoint assignments. The first process run exposed and fixed a `4.66e-9` gradient drift caused by missing child thread/determinism configuration.
- Repeated the authenticated T1 process run exactly and published Report 006. Serialized overhead is negligible and local IPC is tens of milliseconds, but cold startup is about three seconds and isolated worker peak RSS is about 332 MB, so a matched full-process control is now required before any memory-savings claim.
- The process-separated P6 gate passed `215` tests, including warning-strict child-process tests, plus source/test compilation, lock and CLI verification, exact evidence-copy and report-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Added and repeated a matched persistent full-model process control with the same campaign, dataset, cursors, launcher, framing, and coordinator optimizer authority as the tile worker. The control reproduced centralized training exactly.
- Published Report 007: exact process tiling cuts isolated T1 worker peak RSS 50.21%–50.27% and cold/warm tensor payload 84.76%/80.97% versus the matched full worker. P6 now advances to persisted crash/retry rather than additional memory profiling.
- The matched full-process control gate passed `217` tests, including warning-strict process tests, plus source/test compilation, lock and CLI verification, exact control evidence and report-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Added a fail-closed persisted boundary transaction and deliberate tile-worker crash/replacement path. Replay is byte-identical, one exact result is applied, duplicate application is rejected durably, and all centralized optimizer/model identities remain exact.
- Repeated the authenticated T1 recovery and published Report 008. Recovery costs about 3.2 seconds and 3.68 MB of pre-replay retransmission; P6 closes as qualified but conditional, and the autonomous method track advances to P7 sparse experts.
- The final P6 recovery gate passed `219` tests, including warning-strict crash/recovery tests, plus source/test compilation, lock and CLI verification, live persisted-file digest/size checks, report-link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Began P7 with a capacity-bounded top-1 router, shared transformer trunk, four independent expert workers, and an explicit untied output head. Two T0 runs repeated every deterministic route, gradient, optimizer, model, and byte field exactly.
- Published Report 009: individual cold/warm expert payload is 63.36%/92.64% below full, but duplicated cold colony round trip is 50.15% above full and warm round trip only 8.36% below. The next control freezes and caches the shared head before any T1/process scale-up.
- The first P7 gate passed `221` tests, including warning-strict sparse-expert tests, plus source/test compilation, lock and CLI verification, two exact deterministic runs, fair full-input round-trip accounting, report-claim/link validation, browser inspection, unchanged `research/`, diff validation, and the added-line security scan.
- Accepted a valid `passed=false` immutable review of the original Report 008 recovery commit. It proved that malformed persisted phase history was rejected only after gradient/model/optimizer mutation, leaving the transaction retryable for a second application; it also found that the original report's `88.1%`–`90.0%` initialization claim did not match its committed evidence-supported `89.8%`–`90.0%` range.
- Remediated current `main` additively: exact manifest/runtime identity and duplicate-key validation, transaction-ID recomputation, closed owned-file admission, all-file digest/size checks, complete finite result validation, tile-state acknowledgement binding, and shadow AdamW staging all precede live coordinator mutation. A high-level injected applied-state write failure restored gradients and retried successfully from the unchanged persisted transaction.
- Regenerated authenticated T1 recovery evidence as schema v2 with a pre-update checkpoint identity sufficient to recompute each transaction ID. Primary/repeat deterministic fields and all prior training identities remained exact; Report 008 now uses the final-source environmental ranges and explicitly retains the coordinator-crash publication/install gap.
- The recovery-authority remediation gate passed `230` tests, including `15` warning-strict process/recovery tests, duplicate-key and corruption matrices, failed-write retry, exact transaction reconstruction, compilation, lock/CLI checks, evidence/report/link validation, browser inspection, diff validation, and the added-line security scan.
- A second valid `passed=false` immutable review found that the high-level write mock did not exercise the atomic writer: partial write, flush, `fsync`, or replacement failure could leave `manifest.json.tmp`, causing the exact closed-file validator to reject the otherwise unchanged retryable transaction.
- Added four-stage atomic-writer fault injection and a real applied-state `os.replace` failure. Ordinary failures now remove the incomplete temporary file, preserve the prior manifest, restore coordinator gradients, and complete a byte-exact retry; inability to remove residue escalates explicitly and remains a disclosed manual-intervention boundary.
- Regenerated both authenticated T1 recovery records from the corrected final source. Every deterministic v2 field and prior training identity remained exact, both transaction IDs recomputed, and Report 008 now uses the fresh environmental ranges.
- The atomic-publication remediation gate passed `235` tests and `20` warning-strict process/recovery tests, plus source/test compilation, lock and CLI verification, exact transaction reconstruction, evidence/report/link validation, browser inspection, diff validation, and the added-line security scan.

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
