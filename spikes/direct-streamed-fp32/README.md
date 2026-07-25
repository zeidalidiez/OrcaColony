# P3 spike: direct-from-artifact streamed FP32 startup

## Question

**Given** the convert-after-build streamed profile's exact gradients and 92.52% retained-tensor reduction, **when** the model is instead created on PyTorch's `meta` device, every frozen linear is replaced with a path-only module, and only embeddings/norms/adapters are materialized on CPU, **then** does isolated T2 startup peak RSS fall enough to support the first connected offload worker?

## Approach

`build_direct_streamed_lora_model(...)`:

1. verifies a caller-supplied raw SHA-256 for the complete base safetensors artifact;
2. constructs `VolunteerDecoder` and LoRA wrappers on `meta`, never calling the resident FP32 builder;
3. replaces all 48 frozen T2 linears with `DirectStreamedFrozenLinear` modules backed by the single base artifact;
4. materializes only non-linear resident tensors and the adapter on CPU;
5. verifies exact artifact tensor partition and atomically loads the adapter;
6. hashes the complete base artifact again after construction; and
7. clones and revalidates every streamed linear snapshot before forward/backward use.

The startup proof runs resident and direct modes in separate Python processes and records Windows working-set RSS, peak working-set RSS, retained tensor bytes, gradient identity, runtime, and streamed reads.

## Reproduction

Supply a T2 base plus initial adapter emitted by a completed/local campaign:

```bash
uv run python spikes/direct-streamed-fp32/startup-proof.py \
  --mode resident \
  --campaign campaign/t2-tinystories-memory-smoke.json \
  --lora campaign/t2-tinystories-memory-lora-smoke.json \
  --base /path/to/model.safetensors \
  --adapter /path/to/adapter.safetensors \
  --output .artifacts/direct-resident-t2.json

uv run python spikes/direct-streamed-fp32/startup-proof.py \
  --mode direct \
  --campaign campaign/t2-tinystories-memory-smoke.json \
  --lora campaign/t2-tinystories-memory-lora-smoke.json \
  --base /path/to/model.safetensors \
  --adapter /path/to/adapter.safetensors \
  --output .artifacts/direct-streamed-t2.json
```

## Result

| Metric | Full resident | Direct streamed | Change |
|---|---:|---:|---:|
| Retained tensor bytes | 367,552,512 | 27,482,112 | -92.52% |
| RSS after build | 674,168,832 | 339,329,024 | -49.67% |
| RSS after gradient | 746,471,424 | 404,025,344 | -45.88% |
| Peak RSS | 1,405,042,688 | 1,377,845,248 | -1.94% |
| Build runtime | 2.067770 s | 3.541670 s | 1.713x |
| Gradient runtime | 3.230996 s | 7.446539 s | 2.305x |
| Storage reads / gradient | 0 | 673,053,696 B | +673,053,696 B |
| Gradient SHA-256 | `227b7637…cdf3b11d` | `227b7637…cdf3b11d` | exact |
| Loss sum | 4687.0 | 4687.0 | exact |

The committed RSS/timing values preserve one isolated host run and are expected to vary. Artifact/gradient identities, loss, tensor counts, and logical read bytes reproduce exactly.

## Verdict

**PARTIAL — keep the direct builder, but it is not sufficient for a connected offload worker yet.**

### What worked

- The direct builder does not call/materialize the resident FP32 model.
- Retained tensor bytes fell by 92.52%.
- Post-build current RSS fell by 49.67% and post-gradient RSS by 45.88%.
- Complete adapter gradients and loss were exact across isolated processes.
- Rebuilding from the same base/adapter artifacts reproduced the trajectory.

### What did not

- Peak RSS fell by only 1.94%, nowhere near the retained-tensor reduction.
- Build time rose 71.28%.
- Gradient time rose 130.47%, worse than the per-layer exported spike.
- The single-file path repeatedly opens the complete safetensors container and performs full-artifact pre/post hashes; scanning and page-faulting the base keeps startup peak high.

### Surprises

- Current working-set RSS can fall substantially after construction even when process peak barely changes. Retained tensor counts alone are not evidence that a model can start above RAM.
- A path-only model backed by one monolithic safetensors file was slower than the per-layer bundle despite avoiding export, because container/open/hash behavior dominates.

## Limitations

- One Windows host and one warm local-storage/page-cache run; timing/RSS values are observations, not cross-host guarantees.
- `base_file_sha256` is a distinct raw artifact identity supplied by the caller; it is not yet bound into the authenticated campaign assignment beside `base_model_sha256`.
- The raw artifact is hashed before and after construction, but publication is not yet handle-bound against a same-user replace/restore race.
- The proof does not control OS working-set trimming or cold-cache behavior.
- This remains offline and is not a registered connected backend.

## Recommendation

Do not connect this profile yet. The next profile should consume a pre-authenticated per-layer bundle/manifest so startup never rescans or opens the monolithic base for each layer. The manifest must bind every layer digest and the raw bundle identity to `base_model_sha256`, then isolated T2 peak RSS must materially fall before connected assignment work begins.
