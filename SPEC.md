# OrcaColony — Open Volunteer Model Training Framework

**Technical specification:** v0.7 (approved v0.1 execution profile and post-v0.1 research charter)
**Date:** July 24, 2026
**Status:** Approved v0.1 baseline and post-v0.1 research direction
**Primary deployment:** One campaign operated by the project owner
**Repository goal:** A reusable campaign framework and reproducible research vehicle for volunteer-compatible model training

---

## 0. Approved v0.1 execution profile

This section records the implementation decisions that govern v0.1. Where an illustrative example elsewhere differs, this section and the executable campaign configuration take precedence.

- The public project and repository name is **OrcaColony**.
- This document remains the north-star architecture. Version 0.1 ends with Milestone 4; later milestones form the continuing post-v0.1 implementation and research program.
- The approved v0.1 numerical, trust, provenance, and replicated-model contracts remain the stable correctness baseline. Post-v0.1 studies may test PEFT, additional local memory tiers, partial-model work, and other execution topologies without retroactively changing what v0.1 proved.
- Participation means many transient community contributions accumulating toward one goal. It does not assume that a fixed collection of low-end machines remains online as a permanent training cluster.
- Work advances one runnable milestone at a time, beginning with Milestone 0. Tests and documentation stay limited to behavior that protects that path.
- Milestone 0 uses Python 3.11 and PyTorch as the independent single-process oracle. Canonical model and optimizer tensors use safetensors with JSON metadata.
- The correctness transport is FP32. Workers export gradients of a summed masked loss; the coordinator normalizes once by total loss weight and clips only the aggregate. Lossy transport is a later, separately qualified optimization.
- Floating-point parity is tolerance-based across devices. Identities, tensor order, shapes, serialization, and hashes are exact.
- Browser feasibility is proven with one backend before connected control-plane work. Rust/Burn is the first candidate, TensorFlow.js is the bounded fallback, and custom autodiff is not the starting point.
- One browser backend may complete the first connected slice. WebGPU and CPU-browser paths are both required before Milestone 4.
- Initial participants are manually selected people trusted by the project owner. Local feasibility work does not require generalized accounts. Before broader distribution, direct-training admission uses a minimal owner-approved, default-deny allowlist. Open untrusted submission is deferred until viability justifies result-path hardening.
- Anonymous public attribution is independent of operational admission.
- Version 0.1 uses model, tokenizer, dataset, and shard artifacts that may legally be redistributed to workers. Compression, tokenization, or worker-held decryption is not treated as data confidentiality; sensitive-data execution is a separate future architecture.
- GitHub Pages is the single-threaded WASM compatibility profile. Browser work is explicitly initiated, visibly stoppable, and restartable after interruption rather than resumable mid-backward.

---

## Project summary

> OrcaColony is a self-hostable, Folding@home-style framework and research vehicle for community model training. A campaign host freezes the model, data, learning objective, concrete use case, evaluation contract, and release destination. Contributors may arrive briefly, complete one or several bounded pieces of accepted work, and leave; the framework reconciles those contributions into one canonical campaign result and records attribution according to contributor preference. The project preserves replicated full-model data parallelism as its correctness baseline while experimentally evaluating PEFT, local RAM and storage offload, partial-model reconciliation, exact tiled computation, and later sparse or model-parallel methods. It prefers the smallest model that meets the declared use case but does not impose the memory limits of one weak GPU as the project's permanent ceiling. Every released model or research finding publishes the exact artifacts, configuration, evaluation, provenance, and limitations needed to reproduce it.

## 1. Executive decision and v0.1 baseline

Version 0.1 uses **replicated full-model data parallelism** and will not partition one model execution across volunteer machines.

The complete model is the v0.1 unit of direct training. Every direct-gradient worker independently executes the same complete, deliberately small model against the same canonical checkpoint but processes a different data assignment. Many people contribute concurrently; they have replicated checkpoint copies rather than each owning a different layer. Faster machines process more assigned batches and slower machines process fewer.

"Complete" does not inherently mean that every tensor must remain simultaneously in GPU VRAM. A complete-local native worker may eventually address the base through GPU VRAM, unified memory, system RAM, memory-mapped files, and explicitly managed local storage. Its defining property is that it can finish its assignment without another live worker computing a required stage. The currently proven v0.1 browser profiles use simpler resident execution; additional memory tiers require separate qualification.

This makes the first system much closer to Folding@home:

1. A host announces one active campaign.
2. A visitor contributes a bounded amount of time and compute.
3. The coordinator assigns a deterministic work unit appropriate for that machine.
4. The worker computes a result against the current campaign checkpoint.
5. Accepted work is incorporated into the next checkpoint and credited publicly.
6. The worker requests another unit or leaves.

No layer slicing, remote autograd, WAN pipeline parallelism, or tensor-shard routing is required in v0.1. This is a reliability and correctness baseline, not a permanent prohibition. Post-v0.1 studies may test asynchronously scheduled partial-model or tiled work when they preserve exact provenance, bounded retries, and a declared reconciliation rule.

---

## 2. Product definition

The project is a self-hosted, Folding@home-style framework and research platform for collaboratively training open models with volunteer browsers and optional native workers.

A campaign site presents one common objective:

> **Today we are training: _[model name]_**
>
> **Goal: _[specific capability]_**
>
> **Training data: _[published dataset/corpus revision]_**

The project has three public outputs:

1. **Framework repository.** Anyone can fork it, define a campaign, and deploy an independent instance.
2. **Campaign instance.** The project owner chooses and operates one specific model-training campaign, including its data, checkpoints, evaluation, contribution ledger, and Hugging Face publication.
3. **Research record.** Comparable campaigns can form a study that tests one volunteer-training hypothesis and publishes reproducible positive, negative, or inconclusive findings for others to rerun or extend.

A campaign is not a loose pool in which participants choose what to train. Every accepted worker advances one named campaign, one canonical model, and one immutable data revision.

Before a worker submits its first work unit, the site asks how that contributor wants to be credited. A contributor may use a real name, a pseudonym, an optional profile link, a team affiliation, or remain anonymous. The selected public credit identity is separate from login and operational account data.

---

## 3. What the campaign owner chooses

The campaign owner controls:

- Campaign goal and description.
- Concrete intended use case and the claim the campaign will test.
- Model architecture and parameter count.
- Random initialization or exact base-checkpoint revision.
- Tokenizer and tokenizer revision.
- Training mode.
- Execution topology, numerical profile, and admitted local memory profiles.
- Corpus or structured training dataset.
- Source licenses and redistribution policy.
- Filtering, deduplication, normalization, and formatting rules.
- Train, validation, and test splits.
- Objective and loss masking.
- Optimizer and learning-rate schedule.
- Global batch target.
- Token or example budget.
- Checkpoint cadence.
- Evaluation suite and success thresholds.
- Repeated validation cadence, final holdout, baseline, and guardrails for the declared use case.
- Contribution-credit rules and contributor-attribution options.
- Canonical public corpus or dataset page and source-manifest links.
- Hugging Face organization and repository.

These choices are compiled into an immutable campaign lockfile. Workers cannot substitute another model, dataset, tokenizer, or objective and still receive credit.

---

## 4. What “training it at something” means

There is no single generic act called “pointing a model at a corpus.” The campaign must choose a learning objective that matches the desired capability.

### 4.1 Training from scratch on raw text

The model begins with random weights. It receives token sequences and learns to predict the next token.

Example:

```text
Input:  "The TCP connection entered the"
Target: "ESTABLISHED"
```

For every token position, the model predicts a probability distribution over the vocabulary. Cross-entropy loss measures how wrong it was. Backpropagation computes how each parameter should change to reduce that loss.

This teaches the model the statistical patterns present in the selected corpus. It can learn language, syntax, domain terminology, common structures, and some factual associations. It is compute- and data-intensive, even for small models.

Use this mode when the campaign goal is to prove the distributed training system or create a small base model from a tightly scoped corpus.

### 4.2 Continued pretraining on a specialist corpus

The model begins from an existing openly licensed base checkpoint and continues next-token training on a domain corpus.

Examples of domain corpora:

- Permissively licensed source code and documentation.
- Public technical standards and manuals.
- Open scientific papers in one discipline.
- Public-domain historical documents.
- A project’s own licensed documentation and issue history.

Continued pretraining primarily teaches domain language, patterns, and knowledge. It does not automatically teach the model to follow a particular instruction or produce a strict output format.

Use this mode when the target is “understand this domain better.”

### 4.3 Supervised fine-tuning for a concrete function

The training dataset contains input and desired-output pairs.

```json
{
  "input": "Convert this API response into the project schema: ...",
  "target": {"status": "ok", "items": [...]}
}
```

The loss is normally applied to the desired output tokens. This teaches behavior more directly than raw corpus training.

Use this mode when the target is something testable, such as:

- Structured extraction.
- Classification.
- Code transformation.
- Query generation.
- Tool-call selection.
- Domain question answering.
- Summarization in a fixed format.

### 4.4 Distillation and synthetic training data

Access to substantial AI inference is particularly valuable here.

A larger teacher model can generate candidate examples, labels, explanations, transformations, or rankings. Those outputs are then filtered, deduplicated, automatically verified where possible, and converted into a training dataset for the smaller volunteer-trained model.

A practical pipeline is:

```text
Real source material or task generator
            ↓
Teacher-model generation
            ↓
Schema checks / unit tests / factual checks / filtering
            ↓
Deduplication and quality scoring
            ↓
Frozen SFT dataset
            ↓
Volunteer training of the small model
```

Synthetic data is not accepted blindly. The campaign must publish the generation prompts, teacher identifiers where permitted, filtering rules, and validation method.

Use this mode when the goal is a narrow useful model and direct collection of enough human-authored examples is impractical.

### 4.5 Preference optimization and reinforcement learning

Preference pairs, reward models, and verifiable reinforcement learning can be added later. They require more infrastructure and make contributor accounting harder. They are not required for v0.1.

---

## 5. How one distributed training step works

Version 0.1 uses centrally coordinated gradient work units. The coordinator holds the canonical optimizer state; workers do not need to hold Adam momentum or variance tensors.

At global step `s`, the coordinator publishes checkpoint `θ_s`.

### 5.1 Assignment

A worker receives:

- Campaign and campaign-revision ID.
- Canonical checkpoint hash.
- Exact tokenized dataset shard and range.
- Objective definition.
- Loss-mask definition.
- Numerical mode.
- Required number of examples or loss-bearing tokens.
- Work-unit deadline.

The assignment size is selected from measured worker throughput. A CPU may receive one small accumulation unit; a consumer GPU may receive many.

### 5.2 Local computation

The worker:

1. Loads the complete small model.
2. Loads its assigned data.
3. Runs the forward pass.
4. Calculates the campaign-defined loss.
5. Runs backward propagation.
6. Accumulates gradients over the assigned examples.
7. Uploads the accumulated gradient, token count, losses, hashes, and runtime metadata.

The worker does **not** apply the canonical AdamW or SGD update in the initial protocol.

### 5.3 Aggregation

For worker `i`, the frozen objective defines a loss numerator `N_i`, loss-weight denominator `D_i`, and gradient sum `S_i`:

```text
N_i = Σ_j weight_ij × token_loss_ij
D_i = Σ_j weight_ij
S_i = ∂N_i / ∂θ
```

The coordinator rejects `D_i = 0`, verifies the denominator from the assignment, and computes:

```text
G = (Σ S_i) / (Σ D_i)
mean_loss = (Σ N_i) / (Σ D_i)
```

Global-norm clipping is applied once to `G`, never independently to worker gradients. When the campaign’s exact global loss-weight target has been accepted, the coordinator applies one optimizer step:

```text
θ_(s+1) = OptimizerStep(θ_s, G)
```

It then publishes checkpoint `s+1` and begins issuing new work. Up to declared floating-point tolerances, this is ordinary data-parallel training with a global batch assembled from many volunteer microbatches.

### 5.4 Why weak machines still help

A weak worker does not have to keep pace with a faster worker. It receives less data and contributes fewer accepted tokens to the global batch. It never becomes a required stage in a pipeline.

For the v0.1 replicated profile, the worker must be able to complete the model execution within its qualified local resources and lease. Those resources need not eventually be limited to GPU VRAM: native profiles may combine VRAM, system RAM, memory mapping, and explicit local-storage offload. Merely relying on uncontrolled operating-system swap is not a qualified execution plan.

Post-v0.1 research also tests work types that do not require a worker to hold the complete global model, including rolling subnetworks and exact tiled computation. In those modes, a weak worker contributes a smaller parameter or computation task that the coordinator reconciles. Such results are not assumed to be equivalent to v0.1 gradients; each mode must declare and validate its own mathematics.

### 5.5 Communication tradeoff

Every submitted gradient has approximately the size of the trainable model state before compression, regardless of how many examples produced it. To prevent a weak worker from uploading a large gradient for almost no useful computation, the scheduler enforces a minimum work-to-upload ratio.

For the smallest campaigns this is acceptable. Later versions can add:

- FP16, INT8, sparse, or low-rank gradient compression.
- Error-feedback compression.
- Longer local accumulation.
- Local-SGD or parameter-delta mode.
- Hivemind-style decentralized averaging among native peers.

Correctness comes before aggressive communication optimization.

---

## 6. How all contributors work toward the same goal

Every result is bound to all of the following:

```text
campaign revision
model architecture revision
checkpoint hash
optimizer-step number
tokenizer revision
dataset revision
exact assigned data range
objective revision
runtime protocol revision
```

The coordinator rejects work that does not match the active campaign state.

Workers are not independently choosing documents or taking unrelated optimizer steps. They are collectively constructing the global batch for one canonical optimizer step.

A campaign state machine is:

```text
DRAFT
  ↓ owner freezes model, data, objective, and evaluation
READY
  ↓ campaign begins
COLLECTING_STEP_s
  ↓ accepted global-batch target reached
AGGREGATING_STEP_s
  ↓ optimizer update and validation succeed
PUBLISHING_CHECKPOINT_s+1
  ↓
COLLECTING_STEP_s+1
  ...
  ↓ token/example budget or stop condition reached
EVALUATING
  ↓
COMPLETED or FAILED
```

---

## 7. Corpus and dataset control

### 7.1 A raw corpus is still a versioned dataset

The host does not merely provide a list of URLs. The campaign data pipeline produces a frozen, reproducible artifact:

1. Record each source and license.
2. Download a declared source revision or snapshot.
3. Normalize encoding and document structure.
4. Remove or redact prohibited material.
5. Deduplicate documents and near-duplicates.
6. Split documents before tokenization to avoid leakage.
7. Tokenize with the campaign tokenizer.
8. Pack sequences according to declared rules.
9. Hash every output shard.
10. Publish a dataset manifest and lockfile.

### 7.2 Required splits

- **Training split:** sent to contributors and used for gradient computation.
- **Validation split:** evaluated periodically to track generalization and overfitting.
- **Test split:** used only for final or milestone evaluation. It can be withheld until release if the campaign wants a cleaner measurement.

### 7.3 Dataset lockfile

Example:

```json
{
  "dataset_id": "open-code-transform-v1",
  "revision": "sha256:...",
  "dataset_page": "https://huggingface.co/datasets/example-org/open-code-transform-v1",
  "source_manifest_url": "https://huggingface.co/datasets/example-org/open-code-transform-v1/blob/main/SOURCES.md",
  "tokenizer": {
    "id": "campaign-tokenizer-v1",
    "revision": "sha256:..."
  },
  "sources": [
    {
      "name": "source-a",
      "url": "https://example.org/source-a",
      "revision": "...",
      "license": "Apache-2.0",
      "license_url": "https://example.org/source-a/LICENSE",
      "weight": 0.6
    },
    {
      "name": "source-b",
      "url": "https://example.org/source-b",
      "revision": "...",
      "license": "MIT",
      "license_url": "https://example.org/source-b/LICENSE",
      "weight": 0.4
    }
  ],
  "preprocessing_commit": "...",
  "train_shards": [
    {"path": "train/00000.arrow", "sha256": "...", "loss_tokens": 1234567}
  ],
  "validation_shards": [],
  "test_commitment": "sha256:..."
}
```

### 7.4 Data assignment

The coordinator leases deterministic shard ranges. It records which ranges have been attempted, accepted, rejected, or retried. The same example is not silently counted multiple times unless the campaign explicitly allows multiple epochs.

### 7.5 Public corpus and dataset disclosure

For v0.1 direct training, every model, tokenizer, dataset shard, and required derivative sent to a worker must be legally redistributable to that worker. A dataset that cannot cross the worker boundary is not eligible for the browser campaign profile.

The campaign site and every Hugging Face model release must clearly identify the data used for training. The public presentation includes:

- Corpus or dataset name.
- A direct link to the canonical dataset page, preferably a versioned Hugging Face dataset repository.
- Exact dataset revision, commit, or cryptographic digest.
- A direct link to the source manifest.
- Source URLs and source-specific licenses.
- Preprocessing-code commit.
- Train, validation, and test split sizes.
- Loss-bearing token or example counts.
- Synthetic-data generation and filtering disclosures, when applicable.

The model card must not use vague descriptions such as “trained on public data” when the framework has enough information to provide exact sources. If the full corpus cannot be redistributed, the repository still publishes the source list, acquisition instructions, preprocessing code, hashes, and legal basis for use.

---

## 8. Determining whether the model improved at the target

Training loss alone is insufficient. Every campaign begins with a concrete use-case claim and a frozen evaluation contract. A systems-proof campaign may define its use case as validating distributed correctness, but it must not turn lower language-model loss alone into a broader usefulness claim.

### 8.1 Required use-case evaluation contract

Before training begins, the campaign lock declares:

- **Use-case claim:** the behavior or capability the campaign is intended to improve.
- **Baseline:** the initialization or base checkpoint and any relevant comparison model.
- **Repeated validation suite:** frozen examples used to evaluate initialization and selected checkpoints throughout the campaign.
- **Primary metric and threshold:** the result that would demonstrate useful improvement for the stated use case.
- **Guardrails:** behaviors, formats, or general capabilities that must not regress beyond declared limits.
- **Final holdout:** an untouched suite used for promotion or release rather than repeated checkpoint selection.
- **Evaluation cadence:** which checkpoints are evaluated and how failures affect continuation or release.

The repeated validation suite answers whether training is moving toward the fixed use case. The final holdout reduces the risk of selecting and tuning repeatedly against the only test. A research study compares methods using the same use-case contract wherever the hypothesis permits.

### 8.2 Baseline

Before volunteer training starts, run the full evaluation suite against:

- The initialization checkpoint.
- Any relevant existing comparison model.
- A trivial or rule-based baseline where useful.

### 8.3 Held-out evaluation

Checkpoints are evaluated on examples excluded from training. Metrics depend on the goal:

| Goal | Suitable measures |
|---|---|
| Next-token domain modeling | Held-out cross-entropy and perplexity |
| Classification | Accuracy, F1, calibration |
| Structured extraction | Exact match, field-level F1, schema validity |
| Code transformation | Compilation rate, unit-test pass rate, semantic checks |
| Tool selection | Correct tool and argument accuracy |
| Question answering | Exact match, rubric score, citation or retrieval checks |
| Summarization | Task-specific rubric plus factuality checks |

### 8.4 Success gates

A campaign should declare thresholds before training, for example:

```text
Primary: unit-test pass rate rises from 31% to at least 55%
Guardrail: general code benchmark falls by no more than 3 percentage points
Format: valid JSON on at least 99% of held-out inputs
Efficiency: target task completes within the deployment latency budget
```

### 8.5 Checkpoint selection

The final release is not necessarily the checkpoint with the lowest training loss. The campaign selects the best checkpoint using the declared held-out metric and guardrails.

---

## 9. Deployment architecture

```text
                   GitHub Pages or static host
                campaign UI + WASM/WebGPU worker
                              |
                              | HTTPS / WebSocket
                              v
                    Campaign coordinator API
      auth | campaign state | leases | accounting | progress
             /                |                 \
            /                 |                  \
   Artifact storage     Gradient aggregator      Evaluation service
 model/data/checkpoints   canonical optimizer      held-out benchmarks
            \                 |                  /
             \                |                 /
                     Checkpoint publisher
                              |
                              v
                         Hugging Face
```

### 9.1 Static campaign site

The site displays:

- “Today we are training…”
- Goal and model size.
- Exact data and licensing information.
- Current checkpoint and accepted-token progress.
- Evaluation curves.
- Active contributor count.
- Browser contribution controls.
- Native-worker download.
- Public contribution ledger.
- Links to source code and released artifacts.

### 9.2 Coordinator

Responsibilities:

- Campaign state machine.
- Worker registration and capability records.
- Work-unit creation and leasing.
- Reservation of global-batch capacity.
- Validation and acceptance status.
- Token and contribution accounting.
- Checkpoint advancement.
- Authentication and rate limits.

### 9.3 Artifact storage

Stores immutable objects:

- Model checkpoints.
- Checkpoint deltas.
- Tokenized dataset shards.
- Gradient uploads.
- Validation reports.
- Campaign manifests.
- Research-study manifests and experiment reports.
- Content-addressed model or tensor shards used by qualified offload profiles.
- Contribution ledgers.

Object storage should be content-addressed where practical.

### 9.4 Aggregator and canonical optimizer

The aggregator can run on CPU. It streams accepted gradient tensors, accumulates in FP32, applies optimizer state shard-by-shard, and writes a new checkpoint. The host does not need a training GPU merely to coordinate the campaign.

### 9.5 Evaluation service

The initial evaluator can run on the host, a trusted volunteer, rented inference, or a separate scheduled service. Evaluation work is recorded separately from direct training work.

---

## 10. Browser viability

### 10.1 GitHub Pages role

GitHub Pages can host the static application and browser worker bundle. It cannot host the mutable coordinator, database, artifact store, aggregation process, or secret publication credentials.

### 10.2 Browser execution modes

- **WebGPU:** preferred for compatible GPUs.
- **WASM CPU:** compatibility path for machines without WebGPU.
- **Native worker:** preferred for long sessions, broader hardware access, and better performance.

The same campaign protocol should support all three.

### 10.3 Browser limitations

- Background tabs can be throttled, frozen, or discarded.
- GPU features and memory limits vary by browser and driver.
- Long-lived peer-to-peer networking is less reliable than ordinary outbound HTTPS.
- GitHub Pages cannot supply every cross-origin isolation header needed by all multithreaded-WASM designs.

Therefore browser work units must be short, restartable or reissuable, and safe to abandon. Mid-backward persistence is not required in v0.1.

### 10.4 Recommended runtime strategy

Do not compile Hivemind or Prime directly into the browser. Build a small browser runtime around a portable tensor/autodiff engine and the project’s own protocol.

The first browser-runtime spike uses Rust/Burn and falls back to TensorFlow.js if the exact graph cannot export complete gradients in the browser. A purpose-built transformer/autodiff implementation is considered only after a framework-backed spike fails.

Before connected control-plane work, the runtime must load the exact T0 checkpoint, run a complete forward and backward pass, export every named FP32 gradient without applying an optimizer, serialize canonical bytes, and match the independent reference within a frozen tolerance profile. The spike records peak memory, compute time, readback time, serialization time, and repeated-run cleanup behavior.

The first supported model architecture remains intentionally narrow.

### 10.5 Native memory hierarchy and out-of-core execution

Native workers may qualify progressively larger local memory profiles:

1. Complete GPU residency.
2. GPU plus system-RAM offload.
3. Quantized frozen-base placement.
4. Memory-mapped or explicitly cached local-storage/NVMe execution.
5. Remote shard streaming only when measured reuse makes its transfer cost acceptable.

A complete-local worker does not need every tensor simultaneously resident in RAM or VRAM. It must have independently addressable access to every tensor required by the assignment and finish without another live worker providing a mandatory computation stage. Storage-backed profiles use explicit shard layouts, bounded caches, integrity checks, prefetching, and eviction; uncontrolled pagefile or swap thrashing is not an admitted memory strategy.

PEFT is the first target for advanced placement because the large base can remain immutable while trainable adapter state, gradients, and canonical optimizer state stay small. Full dense out-of-core training remains a separate, more I/O-intensive profile. Every profile records peak VRAM and RAM, local-storage footprint and I/O, transfer volume, useful compute time, and completion rate.

---

## 11. Hivemind and Prime usage

### 11.1 Hivemind

Hivemind is a strong reference for:

- Collaborative batch accounting.
- Decentralized peer discovery.
- Parameter and gradient averaging.
- Compression.
- State synchronization.
- Native-worker collaboration over the internet.

Its normal optimizer model is particularly relevant once every worker can hold the complete small model. Hivemind can be used directly by a native Python worker or adapted behind a gateway.

The browser worker will not run the existing Python/PyTorch package. The project may port selected algorithms and protocol ideas, or connect browser workers to a service that participates in a Hivemind cohort.

### 11.2 Prime / DiLoCo

Prime is not required for v0.1. Its useful ideas become relevant when:

- Models are large enough that frequent full-gradient uploads are expensive.
- Stronger contributors can perform many local steps.
- A stable cohort can synchronize less often.

Prime remains a future local-SGD/DiLoCo backend, not the foundation of the browser campaign.

---

## 12. Concrete model-size progression

Model size is the only direct-training "slice" in the approved v0.1 profile. A campaign chooses one complete model, and every v0.1 direct-gradient worker independently executes that same model. The project does not begin by partitioning layers or tensors across machines.

The progression uses one narrow GPT-style decoder family so that each rung changes as little as possible:

- Decoder-only causal transformer.
- Pre-normalization, causal multi-head attention, and a 4x GELU MLP.
- Tied token-embedding and output weights.
- No dropout in deterministic reference runs.
- A campaign-owned BPE tokenizer.
- The same checkpoint, corpus revision, objective, and evaluation for every worker in a campaign.

For a controlled size comparison, keep the tokenizer, context length, corpus, objective, and optimizer fixed and change only the model configuration. Context length or vocabulary size can be increased in a separate experiment after parameter scaling is understood.

Post-v0.1 studies treat four axes separately rather than calling all scaling "a larger model":

- **Model scale:** total and active parameters, architecture, and context.
- **Training method:** dense, PEFT, distillation, local updates, or another declared algorithm.
- **Execution topology:** replicated full model, rolling submodel, exact tiled task graph, sparse experts, or another declared topology.
- **Placement and numerical profile:** VRAM, RAM, local storage, quantization, compute dtype, and accumulation dtype.

Changing one axis does not silently authorize changes to the others. A study freezes the remaining variables where practical so that its finding has a clear interpretation.

### 12.1 Recommended ladder

| Tier | Approx. parameters | Layers / width / heads / MLP | Vocabulary / context target | Purpose | Rough from-scratch token planning range |
|---|---:|---|---|---|---:|
| **T0 — smoke** | **1.3M** | 4 / 128 / 2 / 512 | 4K / 128 | Numerical and browser implementation fixture; not a public model claim | 5M–25M |
| **T1 — first real campaign** | **6.9M** | 6 / 256 / 4 / 1024 | 8K / 256 | First complete volunteer-trained model and public systems proof | 100M–175M |
| **T2 — public alpha** | **17.5M** | 8 / 384 / 6 / 1536 | 8K / 512 | First model expected to show a more useful, measurable narrow capability | 250M–450M |
| **T3 — serious small model** | **46.7M** | 12 / 512 / 8 / 2048 | 16K / 512–1024 | Main browser-GPU and consumer-native specialization target | 700M–1.2B |
| **T4 — browser/native boundary** | **111M** | 12 / 768 / 12 / 3072 | 32K / 1024 | Scale test after compression and checkpoint distribution are proven | 1.7B–2.8B |
| **T5 — native-first whole model** | **337M** | 24 / 1024 / 16 / 4096 | 32K / 1024 | Last planned whole-model rung before reassessing the architecture | 5B–8B |

The token ranges are planning references for causal pretraining, not mandatory budgets. They are intentionally broad and roughly follow the principle that training tokens should rise with parameter count. A supervised specialization campaign instead uses the number of verified examples and epochs justified by its held-out task metric.

The parameter-state floor also rises predictably. Before activations and runtime overhead, FP32 weights plus gradients require about eight bytes per parameter; an FP16 gradient upload requires about two bytes per parameter. This puts T1 at roughly 55 MB of local parameter state and a 14 MB gradient upload, T3 at roughly 374 MB and 93 MB, and T5 at roughly 2.7 GB and 674 MB. Real memory usage is higher because activations and temporary buffers are not included.

### 12.2 The actual starting model

T0 exists only to make the implementation correct. The **first actual distributed training target should be T1: approximately 6.9 million parameters**.

That size is recommended because it is:

- Small enough for ordinary CPU machines and browser WebGPU implementations.
- Large enough to show obvious learning on a deliberately constrained corpus.
- Cheap enough to repeat after finding protocol or numerical defects.
- Small enough that uncompressed checkpoints and gradients remain operationally manageable.
- Large enough to produce a model artifact that contributors can inspect and run themselves.

A strong first corpus is a campaign-owned, tightly constrained synthetic text corpus generated with the project owner's available inference capacity, then filtered and published with exact prompts, source material, licenses, and hashes. A TinyStories-style corpus is a useful pattern: prior work demonstrated that models below 10 million parameters can learn coherent behavior when the data distribution is intentionally simple and well controlled. The campaign does not need a broad general-web corpus to prove the system.

The first public evaluation stays small:

- Held-out cross-entropy or perplexity.
- A frozen set of human-readable generation prompts.
- One simple quality rubric or deterministic task check.
- Comparison against the same model trained by the single-process reference implementation.

It does not need a broad benchmark suite.

### 12.3 Promotion gates

The project moves up exactly one rung only when all of the following are true:

1. **Reference parity:** distributed gradients and optimizer steps match the single-process implementation within the declared numerical tolerance.
2. **Completed campaign:** the current tier has finished an end-to-end campaign, published a checkpoint, and produced a valid contribution ledger.
3. **Stable learning:** training and held-out loss follow a sane curve without unexplained divergence, and the declared task metric improves where the campaign claims specialization.
4. **Hardware fit:** the next tier has been benchmarked on the weakest hardware class the campaign intends to admit, with practical memory headroom rather than a one-off out-of-memory near miss.
5. **Useful compute-to-transfer ratio:** median work-unit compute time is at least five times median result-upload time, or compression has already fixed the imbalance.
6. **Acceptable churn:** expired and rejected work do not consume enough capacity to make the larger campaign wasteful.
7. **A reason to scale:** evaluation indicates under-capacity, underfitting, or a capability benefit likely to come from a larger model. "The next model is bigger" is not by itself a reason.
8. **One major variable at a time:** the first comparison at the next rung reuses the prior corpus, objective, tokenizer, and context where practical.
9. **Fixed use-case evidence:** the next rung is evaluated against the campaign's predeclared use-case contract, including its guardrails and final promotion holdout.

Failure at a gate means running another campaign at the current size, fixing the measured bottleneck, or opening a bounded study of a different training method, placement profile, or execution topology. It does not promote a wider redesign automatically.

### 12.4 Expected participation by tier

- **T0–T1:** browser CPU, browser WebGPU, and native workers are all first-class direct-training participants.
- **T2:** direct CPU participation remains a target, but admission is based on the benchmark rather than a promise that every laptop will be fast.
- **T3:** browser GPUs and native consumer GPUs become the main throughput source; sufficiently capable CPUs can still contribute smaller work units.
- **T4:** browser GPU and native workers are expected to dominate. CPU-only direct training remains available only when the compute-to-transfer admission rule passes.
- **T5:** native workers are the primary path. Browser participation is experimental unless real measurements show otherwise.

A campaign should remain at a smaller tier when moving upward would exclude most of the community without producing a clear capability gain.

### 12.5 Sizing rationale

The first rung is intentionally small rather than toy-only. The TinyStories study showed that carefully constrained synthetic data can support coherent behavior in models below 10 million parameters. Compute-optimal scaling research also supports increasing training tokens as model size increases; the table uses that relationship only as a rough planning aid, not a universal recipe for small or specialized models.

References:

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

## 13. Worker admission and assignment

### 13.1 Capability benchmark

The worker reports measured, not merely advertised, capability:

- A bounded allocation-and-operation fit probe; browsers do not reliably expose free VRAM.
- Available execution tiers and configured quotas for VRAM, system RAM, and local model storage.
- Sequential and representative shard I/O, cache behavior, and prefetch capability for storage-backed native profiles.
- Forward/backward throughput.
- Supported numeric types.
- Gradient serialization throughput.
- Download and upload throughput.
- Browser lifecycle behavior.
- Requested utilization and session duration.

### 13.2 Admission rule

A direct-training worker is admitted only when:

- Its declared work type and required state fit within the qualified memory and storage profile without uncontrolled swapping.
- A replicated-profile worker can independently complete the full assignment; a partial-model worker receives a self-contained bounded task under that mode's declared reconciliation rule.
- The runtime passes numerical parity tests.
- The estimated work unit can finish before its lease expires.
- The compute-to-transfer ratio exceeds the campaign minimum.

Machines that fail direct-training admission can perform auxiliary work, reported separately:

- Dataset checks.
- Evaluation tasks.
- Synthetic-data validation.
- Tokenization checks.
- Deduplication or checksum tasks.

### 13.3 Dynamic work size

Work size is chosen to target a bounded duration, such as several minutes rather than hours. The exact duration is a campaign setting.

A fast GPU receives more microbatches per lease. A slow CPU receives fewer. Both start from the same checkpoint and return the same result type.

---

## 14. Work-unit protocol

Example assignment:

```json
{
  "protocol": 1,
  "campaign_id": "code-transform-small-v1",
  "campaign_revision": "sha256:...",
  "global_step": 1842,
  "checkpoint": {
    "uri": "...",
    "sha256": "..."
  },
  "dataset": {
    "revision": "sha256:...",
    "shard": "train-00017.arrow",
    "start_example": 1200,
    "end_example": 1232
  },
  "objective": "supervised_causal_lm",
  "loss_mask": "target_tokens_only",
  "gradient_format": "fp32_le_v1",
  "loss_token_target": 16384,
  "lease_expires_at": "...",
  "result_upload": "presigned-object-url"
}
```

Example result manifest:

```json
{
  "work_id": "...",
  "checkpoint_sha256": "...",
  "dataset_revision": "...",
  "loss_tokens": 16384,
  "loss_sum": 44466.176,
  "loss_weight_sum": 16384,
  "gradient_sha256": "...",
  "runtime": "browser-webgpu-v0.1.3",
  "device_class": "webgpu",
  "elapsed_seconds": 312.4,
  "status": "completed"
}
```

Large tensor data is uploaded separately from the JSON control message.

---

## 15. Validation and trust

Version 0.1 begins with manually selected, non-adversarial participants and validates for faults rather than attempting Byzantine robustness. Local feasibility work does not require a generalized account system. Before distribution expands beyond the first trusted participants, direct-training admission becomes default-deny through a minimal owner-approved allowlist. Open untrusted submission and stronger result-path hardening follow only after viability is demonstrated.

Checks include:

- Work-unit and checkpoint hashes.
- Dataset-range identity.
- Tensor shapes and chunk completeness.
- NaN and infinity rejection.
- Gradient norm bounds.
- Loss plausibility.
- Runtime-version admission.
- Random duplicated assignments.
- Trusted re-execution of a small sample.
- Numerical comparison across CPU and WebGPU implementations.

Credit is granted only after acceptance into a canonical optimizer step.

---

## 16. Contribution accounting and attribution

Direct-training contribution is measured primarily in accepted loss-bearing tokens, because every direct worker holds the same model for that campaign step.

Public metrics:

- Accepted loss-bearing tokens.
- Accepted examples.
- Accepted work units.
- Estimated training FLOPs.
- Active contribution time.
- Rejected or expired work units.
- Checkpoint range contributed to.
- Runtime and hardware class, optionally hidden or generalized.

Auxiliary contribution is displayed in separate categories and is not presented as direct-training tokens.

### 16.1 Contributor credit profile

Before the first work unit begins, the site asks the contributor how they want to appear in public acknowledgments. Participation does not require disclosure of a legal name.

Supported choices:

1. **Named credit:** a real name or other personally chosen public name.
2. **Pseudonymous credit:** a handle that is not presented as a legal identity.
3. **Anonymous credit:** no public identifier; the contribution is included only in aggregate totals.
4. **Team affiliation:** an optional team, organization, school, community, or household shown alongside named or pseudonymous credit.
5. **Profile link:** an optional HTTPS link to GitHub, Hugging Face, a personal site, or another public profile.

A credit profile contains only public-facing fields:

```json
{
  "visibility": "pseudonymous",
  "display_name": "bobby3060",
  "profile_url": "https://github.com/example",
  "team": "Seattle Home Compute",
  "show_contribution_totals": true,
  "show_hardware_class": false
}
```

Authentication identifiers, email addresses, IP addresses, and raw hardware fingerprints are never written to the public ledger or model repository.

A contributor can update their public credit profile. Each model release records a release-time attribution snapshot so that generated acknowledgments remain reproducible. The user interface must explain that already-published immutable release artifacts may continue to contain the earlier opted-in attribution.

### 16.2 Public acknowledgment requirements

The campaign site maintains a live contributors page. Every Hugging Face model release includes a visible thank-you section in the model card and a complete generated `CONTRIBUTORS.md` file.

The model card should use language such as:

```markdown
## Community contributors

This model was trained with compute donated by 427 contributors.
Thank you to every person who contributed accepted training work.

[View the complete contributor acknowledgments and contribution ledger](./CONTRIBUTORS.md)
```

`CONTRIBUTORS.md` lists every named or pseudonymous contributor who opted into public credit. The default order is alphabetical, not a competitive ranking. Each entry may include the selected profile link, team, and contribution totals if the contributor opted to display them.

Anonymous contributors are thanked explicitly as a group, for example:

```text
Also thanked: 83 contributors who chose to remain anonymous.
```

A leaderboard may exist on the live campaign site, but release acknowledgments remain complete and inclusive rather than showing only top contributors.

### 16.3 Public ledger record

A public ledger record contains:

```text
opaque contributor ID
public credit-profile revision or anonymous marker
campaign revision
work-unit ID
checkpoint hash
dataset-range hash
accepted token count
result hash
validation status
canonical optimizer step
software version
completion timestamp
```

The released model includes a Merkle root or digest of the final ledger. Public acknowledgment files are generated from the accepted ledger and the release-time credit-profile snapshot, not maintained by hand.

---

## 17. Campaign configuration

Example `campaign.yaml`:

```yaml
campaign:
  id: code-transform-small-v1
  title: "Today we're training: Code Transform Small"
  description: >-
    A small open model specialized in deterministic source-code transformations.
  owner: example-org
  model_license: Apache-2.0
  use_case:
    claim: >-
      Improve deterministic source-code transformation success on the frozen
      project task suite without unacceptable general-language regression.
    baseline: initialization

model:
  architecture: volunteer_decoder_v1
  architecture_revision: 1
  layers: 12
  width: 512
  heads: 8
  mlp_width: 2048
  vocabulary_size: 16384
  context_length: 1024
  positional_encoding: learned_absolute
  layer_norm_epsilon: 0.00001
  gelu_approximation: tanh
  attention_bias: true
  linear_bias: true
  tied_token_embeddings: true
  parameters: 46742528
  initialization:
    type: pretrained
    checkpoint: hf://example-org/open-small-base@<commit>
  tokenizer:
    source: hf://example-org/open-small-tokenizer@<commit>

objective:
  type: supervised_causal_lm
  loss_mask: target_tokens_only

training_data:
  dataset_lock: datasets/code-transform-v1.lock.json
  dataset_page: https://huggingface.co/datasets/example-org/code-transform-v1
  source_manifest_url: >-
    https://huggingface.co/datasets/example-org/code-transform-v1/blob/main/SOURCES.md
  epochs: 3
  shuffle_seed: 20260723

optimization:
  mode: centralized_gradient_accumulation
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.1
  global_loss_tokens_per_step: 1048576
  gradient_accumulation_dtype: fp32
  worker_gradient_format: fp32_le_v1
  max_gradient_norm: 1.0

execution:
  training_method: dense
  topology: replicated_full_model
  numerical_profile: fp32_reference_v1
  admitted_memory_profiles:
    - gpu_resident
    - cpu_resident
  experimental: false

workers:
  browser_webgpu: true
  browser_wasm_cpu: true
  native: true
  target_work_seconds: 300
  minimum_compute_to_upload_ratio: 5.0
  lease_seconds: 900

contributor_credit:
  ask_before_first_work_unit: true
  default_visibility: anonymous
  allow_named: true
  allow_pseudonymous: true
  allow_anonymous: true
  allow_profile_url: true
  allow_team: true
  show_totals_opt_in: true
  show_hardware_opt_in: true
  acknowledgments_order: alphabetical

stopping:
  max_loss_tokens: 5000000000
  early_stop_metric: validation_task_score
  early_stop_patience: 10

evaluation:
  baseline_checkpoint: initialization
  every_steps: 100
  repeated_validation_suites:
    - eval/code_transform_unit_tests.yaml
    - eval/format_validity.yaml
  final_holdout_suite: eval/code_transform_final_holdout.yaml
  primary_metric: code_transform_pass_rate
  success_threshold: 0.70
  guardrails:
    - metric: heldout_language_loss
      max_regression_percent: 5

publishing:
  huggingface_repo: example-org/code-transform-small
  huggingface_dataset_repo: example-org/code-transform-v1
  publish_intermediate: true
  publish_ledger: true
  publish_dataset_manifest: true
  publish_source_manifest: true
  model_card_link_corpus: true
  model_card_thank_contributors: true
  contributors_file: CONTRIBUTORS.md
```

---

## 18. Hugging Face model page and release package

The Hugging Face repository is the permanent public record of what the campaign trained, how it trained it, and who contributed.

### 18.1 Required model-card summary

The generated `README.md` model card visibly states:

- The campaign goal.
- Whether the model was trained from scratch, continued-pretrained, or supervised-fine-tuned.
- The exact initialization or base-model revision.
- The corpus or dataset name as a direct link.
- The exact corpus or dataset revision.
- Total accepted training tokens or examples.
- Training objective and major optimizer settings.
- Primary held-out evaluation results and guardrails.
- Intended uses and limitations.
- Model and dataset licenses.
- Total contributor count and a direct acknowledgment link.

The corpus link belongs near the top of the model card rather than being buried in a provenance appendix.

### 18.2 Required acknowledgment section

Every release model card includes a visible `Community contributors` section thanking all contributors. It links to `CONTRIBUTORS.md`, which contains every opted-in public name or pseudonym and explicitly counts anonymous contributors.

A release must not omit a contributor merely because their contribution was small. Any accepted direct-training work included in the released checkpoint is eligible for acknowledgment according to that contributor's chosen credit profile.

### 18.3 Required release files

```text
README.md
CONTRIBUTORS.md
LICENSE
config.json
tokenizer files
model-*.safetensors

training/
  campaign.yaml
  campaign.lock.json
  dataset-manifest.json
  source-manifest.md
  contribution-ledger.parquet
  contribution-ledger.digest
  attribution-snapshot.json
  checkpoint-lineage.json
  evaluations.json
  software-lock.json
  reproduce.md
```

When the corpus is published on Hugging Face, the model card links directly to that dataset repository and exact commit. When it is hosted elsewhere, the card links to the canonical dataset page and includes a mirrored source manifest inside the model repository.

---

## 19. Repository layout

The stable repository path should remain small enough that a new contributor can understand the complete executable campaign without navigating every experiment. Research records are separated from promoted runtime code and may not silently change the default campaign contract.

```text
OrcaColony/
  README.md
  SPEC.md
  PROGRESS_REPORT.md
  LICENSE

  campaign/
    schema.json
    example.yaml

  web/
    app/
    runtime/
      webgpu/
      wasm/

  server/
    api/
    scheduler/
    aggregator/
    ledger/

  training/
    reference/
    data/
    evaluation/
    publishing/

  research/
    README.md
    studies/
    experiments/

  tests/
    reference/
    protocol/
    browser-smoke/

  deploy/
    github-pages/
    docker-compose.yml
```

Native workers, additional runtime implementations, and partial-model systems are added only when an active experiment or promoted milestone requires them. A general plugin abstraction is introduced only after at least two proven execution methods establish the interface it must support.

---

## 20. Scope discipline: actionable implementation and bounded research

The project must be built as a sequence of narrow, working vertical slices and falsifiable experiments. Documentation and tests support an executable milestone or record a completed study; they must not become substitutes for running the method being evaluated.

For each milestone, work proceeds in this order:

1. Implement the smallest end-to-end path that satisfies the milestone.
2. Demonstrate that path manually with real inputs and outputs.
3. Add only the tests needed to protect its important behavior.
4. Update only the instructions and specification details needed to run or reproduce it.

### 20.1 No unbounded speculative implementation

- A subsystem is not implemented unless the active milestone or a confirmed defect requires it.
- Future ideas are recorded in the ordered research backlog rather than implemented preemptively.
- A prioritized idea may receive the smallest runnable spike needed to accept or reject its hypothesis; a spike is not automatically a supported framework feature.
- Abstractions are introduced after a second real use case appears, not in anticipation of one.
- Coding agents must not broaden a task beyond the named milestone, acceptance criterion, and files without explicit project-owner approval.
- A technically interesting improvement is not automatically in scope.

### 20.2 Bounded testing

Testing should concentrate on numerical correctness, protocol integrity, and regressions that could corrupt accepted training work.

The initial test set should consist of:

- A deterministic single-process reference loss and gradient check.
- Checkpoint and gradient serialization round trips.
- One multi-worker aggregation equivalence test.
- One browser smoke test for each currently supported execution path.
- Focused regression tests for defects actually encountered.

Do not build exhaustive browser matrices, broad fuzzing suites, long-duration load tests, speculative property tests, duplicate integration suites, or performance laboratories before the corresponding code path is actionable and used. Testing is expanded when observed failures, release risk, or the next milestone justifies it.

### 20.3 Bounded documentation

For v0.1, the canonical human-readable documentation is:

- `README.md` for setup, local execution, and campaign operation.
- `SPEC.md` for product behavior and architecture decisions.
- `PROGRESS_REPORT.md` for the current build position, remaining work, blockers, and priority order.
- The campaign schema and example configuration.
- Release provenance generated for published models.

Post-v0.1 research adds one bounded record per active study or experiment. Do not create parallel architecture documents, exhaustive API references, tutorial collections, or design catalogs without an executable method or concrete comparison to record. Code comments should explain non-obvious numerical or protocol invariants, not restate the code.

### 20.4 Milestone change control

Every implementation task should name the milestone and acceptance criterion it advances. Work discovered outside that boundary is placed in the backlog unless it blocks the current path. A milestone is complete when its vertical slice works, its high-risk behavior has targeted tests, and another developer has enough instructions to run it.

### 20.5 Research experiment discipline

Every active experiment declares before implementation:

1. The hypothesis and why it could improve volunteer participation or model usefulness.
2. The fixed campaign use case and baseline it will compare against.
3. The smallest runnable artifact that can falsify the hypothesis.
4. Correctness, resource, reliability, and use-case evaluation gates.
5. The variables held constant and the one major variable being tested.
6. The evidence and artifacts that will be published.
7. A terminal disposition: validated, rejected, inconclusive, or promoted.

Negative and inconclusive results are retained when reproducible. A successful spike proves only its declared hypothesis; promotion into the stable framework requires the graduation rules in Section 25.

---

## 21. Implementation milestones

### Milestone 0: single-process reference

- T0 smoke configuration: approximately 1.3M parameters.
- One supported tiny decoder architecture.
- Deterministic dataset iterator.
- Reference forward/backward implementation.
- Canonical optimizer and checkpoint format.
- Reproducible loss curve on one machine.

### Milestone 1: browser gradient and connected-worker proof

#### Milestone 1a: local browser-runtime gate

- Load the complete T0 model in a browser using one candidate backend.
- Compute a complete FP32 gradient and export every named tensor.
- Serialize the gradient canonically.
- Match the reference implementation within the frozen tolerance profile.
- Measure compute, memory, readback, serialization, cancellation, and cleanup behavior.

#### Milestone 1b: one connected worker

- Serve one fixed assignment through a minimal local ingest endpoint.
- Upload and validate one gradient artifact.
- Apply one canonical optimizer step and publish the resulting local checkpoint.
- Do not add accounts, dashboards, or generalized scheduling to this proof.

### Milestone 2: multi-worker global step and recovery

- Use a local scheduler harness to issue exact, non-overlapping data ranges to several workers.
- Aggregate their gradients and close the global batch without unexplained overshoot.
- Apply one canonical optimizer step and match a reference global-batch step within tolerance.
- Prove expiry, retry, stale-result rejection, at-most-one contributing attempt per assignment, and crash-safe step replay.

### Milestone 3: campaign control plane and second browser path

- Campaign lockfile.
- Work leases and reservations.
- Checkpoint versioning.
- Contributor accounts, credit preferences, minimal owner-approved allowlisting, and accepted-work ledger.
- Live contributor acknowledgments and progress dashboard.
- Failure and retry handling.
- Complete the remaining WebGPU or CPU-browser path before Milestone 4.

### Milestone 4: trusted public correctness campaign and v0.1 release

- T1 first real campaign: approximately 6.9M parameters.
- One small from-scratch language-model campaign using redistributable artifacts.
- Several days of browser and CPU contribution from manually trusted or owner-allowlisted participants.
- Public checkpoints and contribution ledger.
- Comparison with a centralized reference run using the frozen parity profile.
- Completion of this milestone is the v0.1 release boundary.

### Milestone 5: useful specialization campaign

- Use T2 (approximately 17.5M parameters) by default, or T3 only when T2 promotion gates pass and evaluation justifies the increase.
- Select an openly licensed compatible checkpoint or train the chosen small architecture from scratch.
- Build a task-specific corpus or SFT dataset.
- Use available AI inference for candidate-data generation where useful.
- Declare one concrete use-case claim; freeze its repeated validation suite, final holdout, baseline, guardrails, and success criteria before training.
- Run a public volunteer campaign.
- Publish the best checkpoint, linked corpus, contributor acknowledgments, and complete provenance on Hugging Face.

### Milestone 6: reproducible research-study contract

- Add machine-readable study and experiment manifests.
- Tie comparable campaigns to one hypothesis and one fixed use-case evaluation contract.
- Record code, model, dataset, topology, numerical profile, worker profile, resource, reliability, and evaluation evidence.
- Publish validated, rejected, and inconclusive outcomes with reproduction instructions.
- Keep the stable v0.1 campaign path unchanged while the research harness is introduced.

### Milestone 7: PEFT campaign vertical slice

- Freeze one legally redistributable base and one exact LoRA configuration.
- Declare the complete trainable-adapter tensor manifest.
- Preserve summed-loss gradients, coordinator normalization, one global clip, canonical optimizer ownership, restart, provenance, evaluation, and release semantics over the adapter state.
- Prove the base remains immutable and adapter updates match a single-process reference within tolerance.
- Run a bounded multi-worker PEFT campaign against a fixed use case.

### Milestone 8: local memory tiers and mixed-profile qualification

- Qualify GPU-resident, system-RAM-offloaded, quantized-base, and explicit local-storage profiles one at a time.
- Implement bounded caches, content-addressed shards, prefetching, eviction, and storage quotas for native workers where needed.
- Measure peak memory, storage I/O, network transfer, throughput, completion, and evaluation behavior.
- Prove mixed-profile aggregation against a homogeneous reference before allowing profiles to share a campaign.
- Advance model size only when Section 12 gates and the campaign's use-case evidence justify it.

### Milestone 9: rolling partial-model study

- Extract capability-sized subnetworks or parameter slices from a larger canonical model.
- Rotate coverage across frozen checkpoint rounds and reconcile overlapping updates deterministically.
- Preserve leases, retries, provenance, attribution, and checkpoint lineage.
- Compare convergence and the fixed use-case result with replicated full-model training.
- Promote the method only if a global model larger than the admitted worker slices delivers a justified benefit.

### Milestone 10: exact asynchronous tiled-computation study

- Divide one representative transformer layer into persistent forward and backward tensor tasks.
- Allow transient workers to lease, complete, or abandon tiles without becoming permanent pipeline stages.
- Reconstruct the exact reference result through deterministic reductions and retry failed work.
- Measure cache reuse and computation per transferred byte before expanding to a complete-model task graph.

### Milestone 11: sparse experts and advanced topologies

- Investigate expert-sharded or other sparse architectures only after the simpler memory and partial-model studies establish the relevant bottlenecks.
- Define router, shared-state, availability, replication, reconciliation, evaluation, and release semantics before implementation.
- Treat live model-parallel peer systems as a measured research option, not the default assumption for community participation.

---

## 22. Version 0.1 acceptance criteria

Version 0.1 is complete when:

1. A campaign owner can define and freeze one common model, dataset, objective, and evaluation suite.
2. The campaign site can be deployed as a static frontend, including GitHub Pages.
3. A browser can load the complete supported model and compute a valid gradient.
4. A CPU-only browser can complete the T0 correctness work unit; admission to a T1 campaign is based on measured fit and completion time.
5. Multiple heterogeneous workers can contribute different batches to one canonical optimizer step.
6. No worker chooses arbitrary training data or advances an independent model.
7. Worker dropout does not corrupt the canonical checkpoint.
8. Before their first work unit, contributors can choose named, pseudonymous, or anonymous public credit and an optional profile link or team.
9. Accepted contributions are tracked by opaque contributor ID, chosen public attribution, and checkpoint.
10. Fixed one-step and fixed-K-step distributed fixtures satisfy the frozen numerical parity profile; campaign loss curves are an additional health signal.
11. The final model card directly links the exact training corpus or dataset revision and source manifest.
12. The Hugging Face release visibly thanks all contributors, lists every opted-in contributor in `CONTRIBUTORS.md`, and counts anonymous contributors.
13. The final or best checkpoint, campaign manifest, dataset manifest, evaluations, code revision, attribution snapshot, and contribution ledger are published.

---

## 23. Recommended first two campaigns

### Campaign 1: system proof

- T0 is used privately as the implementation fixture; the public campaign uses the T1 approximately 6.9M-parameter decoder.
- Tiny decoder trained from scratch.
- Small, clearly licensed text corpus.
- Causal next-token objective.
- Main goal: prove the framework, not produce a competitive general model.
- Success measure: distributed training matches the reference loss curve and survives worker churn.

### Campaign 2: useful narrow model

- T2 approximately 17.5M parameters by default; use T3 approximately 46.7M only after T2 demonstrates a capacity limit.
- Small openly licensed compatible base checkpoint, or the selected architecture trained from scratch before specialization.
- One concrete, automatically measurable function.
- SFT dataset built from real examples plus carefully verified synthetic examples.
- Main goal: beat the base checkpoint on a frozen held-out task suite.
- Success measure: declared task metric improves without violating guardrails.

This sequence separates infrastructure validation from claims of model usefulness.

---

## 24. Key architecture decisions

1. **The campaign owner chooses the corpus and objective.** Contributors do not train random things.
2. **Replicated full-model data parallelism is the v0.1 correctness baseline.** Every v0.1 direct-gradient worker independently executes the complete assignment, but post-v0.1 research may qualify local offload and partial-model work.
3. **Workers compute gradients, not canonical optimizer steps, in v0.1.** This removes local Adam-state requirements and makes heterogeneous contributions additive.
4. **The coordinator constructs one global batch from accepted work.** All work advances one checkpoint.
5. **GitHub Pages hosts the static client, not the mutable training service.**
6. **Hivemind informs native collaboration and later averaging.** It is not compiled directly into the browser.
7. **Prime is deferred.** DiLoCo is a scaling option, not an initial dependency.
8. **Raw corpus training and task training are separate campaign modes.** The chosen data must match the claimed capability.
9. **Every campaign declares a concrete use case.** Improvement is established through its frozen repeated validation and final holdout, not training loss alone.
10. **Every released model includes reproducible campaign and contribution provenance.**
11. **Contributors control how they are credited.** Named, pseudonymous, linked, team-based, and anonymous attribution are supported.
12. **The model page directly names and links the training corpus.** Dataset sources, revisions, licenses, and preprocessing are visible.
13. **Every accepted contributor is thanked.** The model card links to a complete generated acknowledgment file rather than recognizing only top contributors.
14. **Working software and runnable experiments precede broad process.** Tests and documentation stay bounded to the active milestone or study and expand only in response to demonstrated needs.
15. **Community participation is incremental and transient.** The design does not assume that contributors dedicate permanent workers or remain online together.
16. **GPU VRAM is not the permanent model-size boundary.** Qualified native profiles may use system RAM and explicit local-storage offload; uncontrolled swapping is not a profile.
17. **Partial-model contribution is a first-class research question.** Rolling subnetworks and exact tiled tasks are evaluated as distinct methods with explicit reconciliation semantics.
18. **Training method, execution topology, placement, and numerical profile are separate axes.** A change in one does not silently authorize changes in the others.
19. **The framework publishes findings as well as models.** Reproducible negative and inconclusive results are retained so the community can avoid repeating failed approaches.

---

## 25. Research program and experimental graduation

### 25.1 Community contribution model

OrcaColony is not premised on maintaining a permanent cluster of weak computers. It accumulates useful, attributable pieces of work from community members who may participate briefly and independently. A campaign may admit several contribution roles:

- Complete gradients from replicated-model workers.
- Complete adapter gradients from PEFT workers.
- Partial parameter updates from a declared rolling-submodel method.
- Exact tensor or layer computations from a persistent task graph.
- Expert updates from a future sparse topology.
- Separately accounted evaluation, data validation, synthetic-data, or verification tasks.

The coordinator must identify the role and mathematics of every accepted result. Different result types are never combined merely because they came from the same checkpoint.

### 25.2 Campaigns, studies, and profiles

- A **campaign** is one locked training run with one canonical state, data revision, use-case evaluation contract, training method, execution topology, and admitted profile set.
- A **study** compares two or more campaigns or bounded experiments against one hypothesis and shared use-case contract.
- A **training method** defines what is optimized, such as dense parameters, LoRA adapters, or rolling subnetworks.
- An **execution topology** defines how one accepted contribution is computed and reconciled, such as replicated full model, partial submodel, tiled task graph, or sparse experts.
- A **memory profile** defines where tensors may live, such as GPU-resident, RAM-offloaded, or local-storage-backed.
- A **numerical profile** defines representations and arithmetic that may affect the result, including quantization, compute dtype, accumulation dtype, kernels, and tolerance.

Memory placement that is numerically equivalent may share a campaign after qualification. Materially different base quantizations or training mathematics remain separate campaign profiles until mixed aggregation is proven.

### 25.3 Ordered execution tracks

The research program evaluates the following tracks in dependency order:

1. **Replicated dense baseline.** Preserve the proven full-gradient path as the numerical and operational oracle.
2. **Replicated PEFT.** Freeze the complete base and train a declared adapter tensor set while retaining coordinator-owned aggregation and optimization.
3. **Local hierarchical memory.** Qualify RAM and explicit local-storage offload so a worker may execute a base larger than its GPU VRAM and, where practical, larger than its system RAM.
4. **Mixed numerical profiles.** Certify and then deliberately combine compatible runtime, precision, quantization, and placement profiles.
5. **Rolling partial models.** Train capability-sized subnetworks or parameter slices from a global model larger than an individual worker assignment and reconcile coverage across checkpoint rounds.
6. **Exact tiled task graph.** Split model operations into persistent retryable computations so a worker holds only one tile or block while the coordinator respects forward/backward dependencies.
7. **Sparse experts and advanced model parallelism.** Explore larger total capacity only after simpler methods reveal a measured need.

The complete-local path is not discarded when partial-model research begins. It remains the fallback execution route, reference implementation, and comparison baseline.

### 25.4 Asynchronous partial-model reconciliation

"Asynchronous" means that workers may claim and finish independent available tasks without remaining online together. It does not mean that the mathematical dependency graph disappears.

For a rolling-submodel round:

1. Freeze global checkpoint `s`.
2. Derive capability-sized submodel assignments with declared parameter coverage.
3. Let transient workers complete those assignments independently.
4. Retry expired slices and reject stale revisions.
5. Reconcile accepted overlapping updates using the method's frozen rule.
6. Advance to checkpoint `s+1` only after the round's coverage and evaluation gates pass.

These updates are not presumed equal to slices of the full-model gradient. The study must compare convergence and use-case results with the replicated baseline.

For an exact tiled task graph, the coordinator persists intermediate state and unlocks tasks only when their inputs are available. Workers may be interchangeable and retryable, but forward layers, backward signals, and deterministic reductions still impose dependency barriers. The first spike stops at one representative transformer layer unless computation per transferred byte and failure recovery justify expansion.

### 25.5 Numerical and mixed-profile qualification

New profiles graduate in four steps:

1. **Canonical fixture:** run one exact checkpoint and batch through the independent reference.
2. **Single-profile qualification:** compare loss, complete declared gradient set, one-step update, fixed-K-step replay, resource use, and use-case evaluation.
3. **Mixed-profile proof:** aggregate controlled contributions from candidate profiles and compare with the homogeneous reference campaign.
4. **Operational admission:** record profile provenance per result, monitor systematic drift or failure, and revoke profiles that no longer satisfy their gate.

Placement-only differences such as GPU residency versus numerically equivalent RAM offload may qualify together. Aggressive base quantization, different adapter definitions, or changed local-update algorithms are separate semantics until evidence proves compatibility.

### 25.6 Evidence, status, and promotion

Every study records:

- Hypothesis and fixed use-case contract.
- Baseline and variables held constant.
- Exact code, model, dataset, tokenizer, and runtime revisions.
- Training method, topology, memory profile, and numerical profile.
- Worker capability classes and participation pattern.
- Correctness, memory, storage, network, throughput, churn, and evaluation evidence.
- Known limitations and confounders.
- Reproduction instructions and immutable artifact hashes.
- Final status: validated, rejected, inconclusive, or promoted.

Promotion into the stable framework requires a runnable end-to-end campaign, restart and retry safety, exact provenance, a supported release path, and improvement or non-regression under the declared use-case gates. A method that only runs once remains experimental. A method that saves memory but makes accepted work impractically slow is recorded but not admitted by default.

`PROGRESS_REPORT.md` is updated in every roadmap commit to record the current build position, remaining major work, blockers, priority order, and immediate bounded target. Git history remains the detailed change log; the progress report remains the concise operational handoff.

---

## 26. Summary

OrcaColony first proved replicated full-model data parallelism: many transient contributors independently execute bounded work against replicated copies of one checkpoint, and the coordinator reconciles complete gradients into one canonical optimizer path. T0 and T1 remain the correctness foundation. A complete-local native worker may later place tensors across GPU VRAM, system RAM, and explicit local storage; complete does not mean entirely GPU-resident.

The project now also serves as a reproducible research vehicle. It will test PEFT, hierarchical local memory, mixed profiles, rolling subnetworks, exact tiled computation, and later sparse topologies through bounded studies. The aim is to let ordinary community members contribute useful pieces toward models that may exceed their individual hardware, without pretending that every experiment is ordinary data-parallel SGD or requiring contributors to form a permanent synchronized cluster.

Every campaign freezes one concrete use-case claim, repeated validation suite, final holdout, baseline, and guardrails. Parameter count is not a status target: the project prefers the smallest model that meets the use case, while allowing measured methods to move beyond the memory ceiling of one weak GPU. Models and research findings publish exact artifacts, campaign or study configuration, evaluation, contribution provenance, limitations, and contributor-controlled attribution.
