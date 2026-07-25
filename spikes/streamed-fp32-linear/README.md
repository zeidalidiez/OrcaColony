# P3 spike: exact-FP32 streamed frozen linears

## Question

**Given** the persistent native worker's complete FP32 model residency, **when** frozen linear weights are moved into per-layer safetensors and reloaded in both forward and backward, **then** can OrcaColony reduce retained model tensors without changing loss or adapter gradients, and what storage-I/O/runtime cost appears?

## Approach

The spike deliberately keeps numerical semantics unchanged:

- Build the authenticated FP32 LoRA fixture.
- Write each frozen `nn.Linear` weight and bias to a separate safetensors file.
- Replace the in-memory frozen module with a path-only `StreamedFrozenLinear`.
- Load the exact FP32 tensors during forward.
- Load them again during backward instead of retaining the forward weight in the autograd graph.
- Keep embeddings, norms, position state, and FP32 LoRA parameters resident.
- Compare summed loss and the complete ordered adapter-gradient vector against the unchanged FP32 model.

This is a placement experiment, not a new precision profile.

## Reproduce

Each storage path must not already exist:

```bash
uv run python spikes/streamed-fp32-linear/spike.py \
  --campaign campaign/t0-smoke.json \
  --lora campaign/t0-lora-smoke.json \
  --storage .artifacts/streamed-fp32-t0 \
  --output .artifacts/streamed-fp32-t0.json

uv run python spikes/streamed-fp32-linear/spike.py \
  --campaign campaign/t1-tinystories-smoke.json \
  --lora campaign/t1-tinystories-lora-smoke.json \
  --storage .artifacts/streamed-fp32-t1 \
  --output .artifacts/streamed-fp32-t1.json

uv run python spikes/streamed-fp32-linear/spike.py \
  --campaign campaign/t2-tinystories-memory-smoke.json \
  --lora campaign/t2-tinystories-memory-lora-smoke.json \
  --storage .artifacts/streamed-fp32-t2 \
  --output .artifacts/streamed-fp32-t2.json
```

## Results

| Profile | FP32 resident tensors | Streamed resident tensors | Reduction | Storage reads / gradient | Runtime ratio | Gradient relative L2 | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 1.33M | 5,434,368 B | 2,270,208 B | 58.22% | 6,130,176 B | 2.548x | 0 | 0 |
| T1 6.90M | 28,098,560 B | 9,168,896 B | 67.37% | 37,069,824 B | 1.613x | 0 | 0 |
| T2 91.54M | 367,552,512 B | 27,482,112 B | 92.52% | 673,053,696 B | 1.307x | 0 | 0 |

At T2, retained tensors fell from 350.53 MiB to 26.21 MiB. One gradient read 1.979 times the 340,077,504-byte streamed artifact set: 48 forward reads and 47 backward reads. Despite that 673 MB of application-level storage reads and a defensive copy of every mapped tensor before validation/use, measured gradient runtime increased by 30.66% on this host's warm local storage/page cache.

The committed result files preserve one measured timing run. Tensor counts, artifact/read bytes, losses, and gradient metrics reproduce exactly; wall-clock fields are expected to vary by host and run.

## Verdict: PROCEED, WITH A DIRECT-STARTUP REQUIREMENT

The core placement idea works better than the int8 candidate for P3:

- Adapter gradients were bit-identical at every scale.
- Loss sums were identical.
- T2 retained model-tensor bytes fell by 92.52%.
- The measured T2 runtime penalty was bounded at 30.66% in this warm-cache run.
- The connected FP32 numerical identity can remain unchanged.

However, this spike first builds the complete FP32 model and only then writes/replaces its linears. It therefore proves **steady retained tensor reduction**, not lower peak startup RSS or ability to start a model larger than RAM. Production work must construct the streamed model directly from the authenticated base artifact without first materializing every linear.

## What did not work yet

- No direct-from-artifact/meta-device construction.
- No process current/peak RSS measurement for the streamed model.
- No cold-cache, slow-disk, remote-object-store, or concurrent-worker I/O measurement.
- One safetensors file per linear is a simple spike layout, not a finalized artifact format.
- Reads are synchronous and sequential; there is no prefetch, read coalescing, or bounded layer cache.

## Surprises

- Relative overhead decreased with scale: 2.548x at T0, 1.613x at T1, and 1.307x at T2. Python/file-open and authenticated-copy overhead dominates tiny models, while matrix compute hides more I/O at T2.
- Exact backward needs one fewer load than twice the linear count because the first frozen path has no input gradient dependency; T2 issued 95 reads for 48 linears.
- Exact FP32 streaming retained fewer model tensor bytes than the int8 linear profile at T2 (27,482,112 versus 113,080,320) because streamed linears retain no weights at all.

## Recommendation

1. Promote the path-only streamed linear and reload-on-backward behavior into an explicit offline profile.
2. Keep it outside the connected worker until artifact identity, retry, and I/O telemetry are integrated.
3. Build the next slice directly from the frozen-base safetensors via meta/empty construction so startup does not materialize all FP32 linears.
4. Measure current and peak process RSS plus cold/warm storage behavior before claiming larger-than-RAM support.
5. Treat local warm-cache timing as a lower-bound I/O result, not a volunteer-network result.
