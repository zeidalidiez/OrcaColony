# P3 spike: int8 frozen linear with FP32 LoRA

## Question

**Given** the persistent T2 worker's complete FP32 CPU residency, **when** every frozen `nn.Linear` weight is stored as per-output-channel symmetric int8 with FP32 scales/biases and dequantized inside a custom autograd function, **then** how much model tensor storage is removed and how far do FP32 adapter gradients move?

This is an offline feasibility spike. It does **not** add a connected runtime backend or loosen coordinator acceptance.

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

## Verdict: PARTIAL

### What worked

- Per-row int8 frozen linears retained an explicit differentiable path to hidden activations and FP32 LoRA parameters.
- Unique resident model tensor storage fell by 43.08% at T0, 50.18% at T1, and 69.23% at T2.
- Loss-sum drift stayed below 0.006% in all three deterministic comparisons.
- Adapter-gradient cosine remained above 0.99954.

### What did not

- Adapter gradients did not preserve the current FP32 numerical contract: relative L2 reached 3.038% at T2.
- This spike does not prove connected coordinator acceptance, mixed-profile aggregation, campaign convergence, held-out quality, process RSS, or throughput.
- The spike constructs an FP32 model before conversion, so it does not solve peak startup residency or larger-than-RAM loading.

### Surprises

- The storage reduction improved with model scale because T2 is dominated by frozen linear matrices; FP32 embeddings and normalization state are a larger fraction of T0/T1.
- Very small loss drift coexisted with materially larger adapter-gradient drift. Loss parity is therefore not a sufficient admission test for quantized training workers.

### Recommendation for the real build

1. Reuse the custom int8 frozen-linear/autograd design as an explicit **offline numerical profile**, not as `python-native-cpu-f32` and not under the current FP32 tolerance.
2. In P4, define a quantized oracle and fixed quantized trajectory, then test homogeneous quantized campaigns before any mixed-profile aggregation.
3. Keep P3's next connected placement target exact-FP32 and pursue mapped/layer-streamed storage separately; that can address peak residency without silently changing the optimization objective.
4. A production int8 loader must construct quantized modules directly from the frozen artifact rather than building the complete FP32 model first.
