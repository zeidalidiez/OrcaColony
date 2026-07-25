# Authenticated FP32 layer-bundle startup proof

This spike tests the next resource-placement step after the monolithic direct-streamed profile: publish one small resident shard and one safetensors shard per frozen linear, then construct the LoRA model without scanning or opening every linear during startup.

The profile remains exact CPU FP32. It does not alter the connected worker tolerance or register a new coordinator backend by itself.

## Contract

`export_base_layer_bundle(...)` is an offline publication step. It:

1. checks the source artifact's raw SHA-256 before and after acquiring owned tensor snapshots;
2. requires an exact finite FP32 tensor set with the campaign model's names and shapes;
3. recomputes the canonical tensor-set identity and requires it to equal `base_model_sha256`;
4. writes one deterministic resident shard plus one deterministic shard for every frozen linear into a fresh directory; and
5. emits canonical `orcacolony_base_layer_bundle_v1` JSON containing `base_model_sha256`, source raw-file identity, exact artifact membership, raw-file hashes, tensor hashes, shapes, byte sizes, module paths, and bias contracts.

The SHA-256 of that canonical manifest is the authenticated bundle identity supplied to `build_layer_bundle_streamed_lora_model(...)`. The builder binds it back to the LoRA contract's `base_model_sha256`, validates exact bundle membership and paths, builds on `meta`, replaces all frozen linears before `to_empty`, loads only the resident shard, restores the original 2-D causal masks, and failure-atomically loads the adapter. No linear shard is opened during construction.

Every forward/backward linear reload clones its small safetensors mapping into owned contiguous memory before checking exact names, FP32 dtype, shape, finiteness, and tensor SHA-256. The manifest's raw-file hashes are transport/cache identities for the forthcoming connected downloader; lazy compute authenticates semantic tensor content without adding a second full-file read to every layer operation.

`load_lora_manifest(..., verify_base_model=False)` deliberately skips the parser's usual deterministic resident rebuild. It is safe only in a startup path that immediately verifies an independently authenticated base artifact or bundle manifest bound to the same `base_model_sha256`; the default remains `True` everywhere else.

## Reproduction

The source T2 base and adapter are local proof artifacts and are not committed. With those artifacts present:

```bash
BASE=.artifacts/p3-t2-persistent/cache/model/47a536cd24b50e7a3bd7a36dc224e2e31774ab0c1c0738df0256e6f579fc15e5.safetensors
ADAPTER=.artifacts/p3-t2-persistent/cache/adapter/2737368f5af28772d29525421193a0cbae1687026cb6b1396db2f8020101f0f6.safetensors
BUNDLE=.artifacts/p3-layer-bundle-t2
BASE_SHA=47a536cd24b50e7a3bd7a36dc224e2e31774ab0c1c0738df0256e6f579fc15e5
ADAPTER_SHA=2737368f5af28772d29525421193a0cbae1687026cb6b1396db2f8020101f0f6
MANIFEST_SHA=605832705b40f5f76d964df6b7efb4995cf123af252271d27852739ecaf5c6ad
GRADIENT_SHA=227b763759a9a63da9eae0ca98af6166a2bccd0dd08212aa668a3b09cdf3b11d

uv run python spikes/layer-bundle-fp32/export-bundle.py \
  --config campaign/t2-tinystories-memory-smoke.json \
  --lora-config campaign/t2-tinystories-memory-lora-smoke.json \
  --base-artifact "$BASE" --base-artifact-sha256 "$BASE_SHA" \
  --bundle "$BUNDLE" --output .artifacts/p3-layer-bundle-export-t2.json

for MODE in resident direct bundle; do
  EXTRA=()
  if [ "$MODE" = bundle ]; then
    EXTRA=(--bundle "$BUNDLE" --bundle-manifest-sha256 "$MANIFEST_SHA")
  fi
  uv run python spikes/layer-bundle-fp32/startup-proof.py \
    --mode "$MODE" \
    --config campaign/t2-tinystories-memory-smoke.json \
    --lora-config campaign/t2-tinystories-memory-lora-smoke.json \
    --base-artifact "$BASE" --base-artifact-sha256 "$BASE_SHA" \
    --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" \
    --expected-gradient-sha256 "$GRADIENT_SHA" --expected-loss-sum 4687 \
    "${EXTRA[@]}" --output ".artifacts/p3-layer-bundle-${MODE}-t2.json"
done
```

The committed [`results`](results) are the measured Windows runs used below.

## T2 result

All three isolated processes used the same 91,544,064-parameter campaign, adapter, deterministic batch, and lightweight LoRA-manifest parse. Every mode returned loss sum `4687.0` and gradient SHA-256 `227b763759a9a63da9eae0ca98af6166a2bccd0dd08212aa668a3b09cdf3b11d`.

| Measurement | Resident | Corrected direct | Layer bundle |
|---|---:|---:|---:|
| Retained tensor bytes | 367,552,512 | 27,482,112 | 27,482,112 |
| Current RSS after build | 649,510,912 | 377,868,288 | 378,261,504 |
| Peak RSS through build | 1,379,958,784 | 379,973,632 | 378,261,504 |
| Current RSS after gradient | 715,304,960 | 407,752,704 | 412,426,240 |
| Final process peak RSS | 1,379,958,784 | 740,708,352 | 740,360,192 |
| Build seconds | 2.740844 | 3.548609 | 1.329829 |
| Gradient seconds | 2.296598 | 2.734846 | 2.563094 |
| Linear reads during startup | 0 | 0 | 0 |
| Streamed gradient reads | 0 | 95 / 673,053,696 B | 95 / 673,053,696 B |

Relative to resident construction, the layer bundle reduced retained tensors by **92.52%**, current post-build RSS by **41.76%**, peak RSS through build by **72.59%**, current post-gradient RSS by **42.34%**, and final process peak RSS by **46.35%**. Its gradient was 1.116x resident in this warm run.

Relative to the corrected direct profile, the bundle had effectively the same process memory and exact numerical behavior, but reduced build time by **62.53%** (`3.548609 s` to `1.329829 s`) because startup no longer hashes, snapshots, validates, and partitions the 366,190,504-byte monolithic container. The 50 bundle entries occupy 366,187,968 bytes, effectively the same storage as the source rather than compression.

## Correction to the earlier direct result

The earlier direct-startup script called the default `load_lora_manifest(...)`, which built and validated a full resident base before the measured direct constructor. That redundant pre-build raised lifetime peak RSS to `1,377,845,248` bytes and confounded the conclusion that direct construction itself preserved resident-scale peak memory.

With the parser-only path and the direct builder still performing its own complete raw/canonical artifact checks, corrected direct final peak was `740,708,352` bytes—a **46.24% reduction** from the earlier observation. The layer bundle does not materially improve corrected direct peak (`0.047%` here); its measured advantage is removing complete-container startup work and providing independently downloadable authenticated shards.

## Decision and limits

The result is sufficient to proceed to a **connected layer-bundle worker vertical slice**:

- exact gradients and loss remained unchanged;
- final process peak was materially below full residency;
- retained/current memory remained bounded by resident non-linears plus transient layer/activation work;
- no linear artifact was opened during startup; and
- each layer now has a manifest-bound transport identity suitable for content-addressed fetch/cache.

It does not yet prove a public untrusted worker, larger-than-RAM execution, cold physical I/O, or cross-platform RSS. This was one Windows run with warm local storage. Logical tensor reads are not physical disk or wire bytes. The exporter still performs complete source scans offline. The standalone loader uses path checks plus owned snapshot/digest validation, not an immutable filesystem snapshot or retained-handle lease; a same-user concurrent writer is detected when content is consumed, but hostile mutable-directory integrity is not claimed. The connected slice must bind the expected manifest SHA and `base_model_sha256` in authenticated assignment state, validate every downloaded shard against its raw digest before cache publication, preserve exact membership, and keep the existing strict FP32 result acceptance unchanged.
