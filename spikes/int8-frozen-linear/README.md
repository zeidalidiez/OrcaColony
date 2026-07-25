# P3 spike: int8 frozen linear with FP32 LoRA

## Question

**Given** the persistent T2 worker's complete FP32 CPU residency, **when** every frozen `nn.Linear` weight is stored as per-output-channel symmetric int8 with FP32 scales/biases and dequantized inside a custom autograd function, **then** how much model tensor storage is removed and how far do FP32 adapter gradients move?

This began as an offline feasibility spike. P4 now promotes the measured design as a connected **homogeneous** numerical profile without loosening or impersonating exact FP32 acceptance.

## Approach

The spike compares two models built from the same campaign and LoRA manifest:

1. the existing frozen-FP32-base/FP32-adapter model;
2. a copy whose frozen linear weights are replaced by `Int8FrozenLinear` buffers.

`Int8FrozenLinearFunction` recreates each FP32 weight in both forward and backward. It saves only int8 weights and per-row scales in the autograd context, rather than retaining all dequantized weights through backward. Token/position embeddings and normalization parameters remain FP32. The script reports unique parameter/buffer storage, summed-loss drift, and flattened adapter-gradient cosine/relative-L2/max-absolute error.

PyTorch dynamic quantized linear was not used because this profile needs an explicit backward path to hidden activations while the LoRA branch remains trainable.

## Reproduce

From the repository root:

```bash
uv run python spikes/int8-frozen-linear/spike.py \
  --campaign campaign/t0-smoke.json \
  --lora campaign/t0-lora-smoke.json \
  --output spikes/int8-frozen-linear/results/t0.json
uv run python spikes/int8-frozen-linear/spike.py \
  --campaign campaign/t1-tinystories-smoke.json \
  --lora campaign/t1-tinystories-lora-smoke.json \
  --output spikes/int8-frozen-linear/results/t1.json
uv run python spikes/int8-frozen-linear/spike.py \
  --campaign campaign/t2-tinystories-memory-smoke.json \
  --lora campaign/t2-tinystories-memory-lora-smoke.json \
  --output spikes/int8-frozen-linear/results/t2.json
```

The inputs are deterministic synthetic token batches so the three result files are directly reproducible. They are not held-out TinyStories quality measurements.

## Results

| Profile | Parameters | Quantized linears | FP32 tensors | Int8 tensors | Tensor reduction | Gradient cosine | Gradient relative L2 | Relative loss-sum error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 1,334,016 | 16 | 5.183 MiB | 2.950 MiB | 43.08% | 0.9998568 | 1.709% | 0.00562% |
| T1 | 6,901,760 | 24 | 26.797 MiB | 13.350 MiB | 50.18% | 0.9997198 | 2.367% | 0.00206% |
| T2 | 91,544,064 | 48 | 350.525 MiB | 107.842 MiB | 69.23% | 0.9995400 | 3.038% | 0.00276% |

The T2 resident-tensor result is materially better than the earlier idealized base-only estimate because all large frozen linear matrices become int8 while embeddings and small non-linear state remain FP32. It is a tensor-storage result, not process RSS: allocator, Python/PyTorch runtime, transient dequantization, and input/activation memory are excluded.

The numerical result is not compatible with the existing exact FP32 worker contract. Adapter-gradient relative L2 is 1.71%–3.04%, or roughly 1,709–3,038 times the current `1e-5` FP32 acceptance bound. High cosine alone is not permission to mix these gradients into the FP32 trajectory.

## P4 homogeneous T1 trajectory

The follow-up fixes a 20-step TinyStories campaign, evaluates steps `0, 1, 2, 5, 10, 20`, compares exact FP32 with int8, aggregates the int8 candidate from two one-sequence shards, and restarts the int8 model/AdamW state at step 10:

```bash
uv run python spikes/int8-frozen-linear/trajectory.py \
  --campaign campaign/t1-tinystories-int8-trajectory.json \
  --lora campaign/t1-tinystories-int8-trajectory-lora.json \
  --dataset .artifacts/p3-t1-profile/dataset \
  --work-dir .artifacts/int8-t1-trajectory-work \
  --output spikes/int8-frozen-linear/results/t1-trajectory.json
```

| Step | FP32 held-out mean loss | Two-shard int8 held-out mean loss | Int8 − FP32 |
|---:|---:|---:|---:|
| 0 | 9.0418359041 | 9.0421773195 | +0.0003414154 |
| 1 | 9.0412223339 | 9.0415543318 | +0.0003319979 |
| 2 | 9.0402287245 | 9.0405533314 | +0.0003246069 |
| 5 | 9.0347447395 | 9.0349780321 | +0.0002332926 |
| 10 | 9.0154596567 | 9.0153789520 | −0.0000807047 |
| 20 | 8.9368735552 | 8.9360306263 | −0.0008429289 |

The candidate passed the campaign's positive-improvement gate: FP32 improved `0.1049623489`, while two-shard int8 improved `0.1061466932`. This is one short random-base TinyStories trajectory, not evidence that int8 is generally better.

Two-shard int8 aggregation stayed close to the centralized int8 control: maximum per-step gradient relative L2 was `3.1241858279774175e-7`, and final adapter relative L2 was `1.5512369314698598e-7`. Ten post-restart gradients, adapters, and optimizer states were exact. By contrast, int8 intentionally followed a different trajectory from FP32: maximum per-step gradient relative L2 reached `0.051245135154007214`, and final adapter relative L2 was `0.02636555504582058`.

Int8 retained `13,998,080` tensor bytes versus `28,098,560` FP32 bytes (50.18% less). Total measured two-shard int8 compute/evaluation was `10.68649519997416` seconds versus `10.177064399962546` seconds FP32 (1.050x) in the preserved run. Those timings serialize both shards and exclude HTTP, process startup, and a direct quantized artifact loader.

The evidence therefore promotes int8 to a **homogeneous-profile candidate** with its own oracle and profile-bound checkpoint identity. It does not admit int8 gradients into exact-FP32 aggregation.

## Direct authenticated int8 construction

`build_layer_bundle_int8_lora_model(...)` now creates the meta/empty exact-FP32 bundle structure, loads and authenticates one frozen-linear shard at a time, quantizes that owned FP32 snapshot immediately, and discards it before opening the next shard. The final model retains the same int8 buffers as `build_int8_lora_model(...)` without ever constructing all resident FP32 linears.

Reproduce the isolated converted/direct comparison with [`startup-proof.py`](startup-proof.py). The preserved T1/T2 result files use the same base, adapter, and deterministic full-context batch in separate Windows processes.

| Measurement | T1 converted | T1 direct bundle | T2 converted | T2 direct bundle |
|---|---:|---:|---:|---:|
| Retained tensor bytes | 13,998,080 | 13,998,080 | 113,080,320 | 113,080,320 |
| Peak RSS through build | 362,385,408 | 316,235,776 | 1,380,302,848 | 458,956,800 |
| Final process peak RSS | 442,548,224 | 454,524,928 | 1,380,302,848 | 845,414,400 |
| Build seconds | 1.350311 | 1.575045 | 3.289739 | 3.117886 |
| Gradient seconds | 0.265296 | 0.086320 | 2.675096 | 0.737205 |
| Authenticated linear opens | 0 | 24 | 0 | 48 |

Converted and direct modes returned identical loss and gradient SHA-256 at both scales. At T2, direct construction reduced peak RSS through build by **66.75%**, current post-build RSS by **30.82%**, and final process peak by **38.75%**. It read each linear shard once (`340,070,400` logical tensor bytes) and retained no FP32 linear weight. T1's final peak was 2.71% higher despite a lower build peak, so the result is scale/workload dependent. Timings are raw one-run observations with warm local storage and should not be treated as a stable speed ranking.

## Connected P4 qualification

[`connected_campaign.py`](connected_campaign.py) runs two real authenticated T1 native workers for two optimizer steps under `int8-per-output-symmetric-f32-dequant-v1`: one loads the complete authenticated resident base before quantizing it, while the other constructs int8 directly from the authenticated layer bundle. The coordinator is restarted after step 1 while both persistent worker sessions retain their validated models.

```bash
uv run python spikes/int8-frozen-linear/connected_campaign.py \
  --campaign campaign/t1-tinystories-smoke.json \
  --lora campaign/t1-tinystories-lora-smoke.json \
  --dataset .artifacts/p3-t1-profile/dataset \
  --browser-root spikes/burn-browser-gradient/www \
  --state .artifacts/p4-int8-connected-state \
  --exact-reference .artifacts/p4-int8-exact-reference \
  --output spikes/int8-frozen-linear/results/connected-t1.json \
  --target-steps 2
```

The preserved result records four accepted assignments, exact per-worker parity against the int8 oracle, maximum canonical-checkpoint relative L2 `6.12852298785371e-8`, one model build per worker, zero warm model transfer, and v2 checkpoint/resume identities authenticated with the numerical profile. Frozen held-out mean loss improved from `9.042177319526672` to `9.040553331375122`. The final connected int8 adapter remained deliberately distinct from exact FP32 at `0.011018934557552882` relative L2, and loading the campaign as exact FP32 failed closed.

Exact CPU FP32 is now a separate bit-exact worker-acceptance profile. Burn NdArray and WebGPU have separate runtime/numerical identities and retain their measured tolerance gates rather than being advertised as exact CPU FP32. Workers from different numerical profiles cannot enter one campaign; resident and bundle **placements** may mix only when their numerical identity is the same.

## Verdict: QUALIFIED FOR HOMOGENEOUS T1 CAMPAIGNS

### What worked

- Per-row int8 frozen linears retained an explicit differentiable path to hidden activations and FP32 LoRA parameters.
- Unique resident model tensor storage fell by 43.08% at T0, 50.18% at T1, and 69.23% at T2.
- Loss-sum drift stayed below 0.006% in all three deterministic comparisons.
- Adapter-gradient cosine remained above 0.99954.
- Connected resident and direct-bundle workers matched the profile-specific int8 oracle exactly and survived a coordinator restart with profile-bound checkpoints.

### What did not

- Adapter gradients did not preserve the current FP32 numerical contract: relative L2 reached 3.038% at T2.
- The original one-gradient storage comparison alone did not prove connected acceptance, trajectory behavior, held-out quality, process RSS, or throughput; those claims come from the separate trajectory, startup, and connected records.
- The connected qualification is two T1 steps on one Windows host. It does not establish long-horizon convergence, heterogeneous WAN performance, or a T2 connected trajectory.
- Int8 remains numerically incompatible with exact FP32 and is rejected from exact-FP32 campaigns.

### Surprises

- The storage reduction improved with model scale because T2 is dominated by frozen linear matrices; FP32 embeddings and normalization state are a larger fraction of T0/T1.
- Very small loss drift coexisted with materially larger adapter-gradient drift. Loss parity is therefore not a sufficient admission test for quantized training workers.

### Recommendation for the real build

1. Keep `int8-per-output-symmetric-f32-dequant-v1` profile-bound and homogeneous; never relabel it as `python-native-cpu-f32`.
2. Preserve exact CPU FP32 as the bit-exact baseline and preserve separate Burn runtime profiles with their own measured gates.
3. Reuse resident and direct-bundle int8 placements within one campaign only while base, adapter, numerical-profile, worker-weight, and resumable-state identities all match.
4. Treat longer T1/T2 trajectories and heterogeneous remote-host measurements as new experiments, not as reasons to weaken this bounded qualification.
