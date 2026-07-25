# Authenticated FP32 layer-bundle startup proof

This spike tests the next resource-placement step after the monolithic direct-streamed profile: publish one small resident shard and one safetensors shard per frozen linear, then construct the LoRA model without scanning or opening every linear during startup.

The profile remains exact CPU FP32. The connected implementation registers a distinct provenance backend without weakening the coordinator's numerical acceptance.

## Contract

`export_base_layer_bundle(...)` is an offline publication step. It:

1. checks the source artifact's raw SHA-256 before and after acquiring owned tensor snapshots;
2. requires an exact finite FP32 tensor set with the campaign model's names and shapes;
3. recomputes the canonical tensor-set identity and requires it to equal `base_model_sha256`;
4. writes one deterministic resident shard plus one deterministic shard for every frozen linear into a fresh directory; and
5. emits canonical `orcacolony_base_layer_bundle_v1` JSON containing `base_model_sha256`, source raw-file identity, exact artifact membership, raw-file hashes, tensor hashes, shapes, byte sizes, module paths, and bias contracts.

The SHA-256 of that canonical manifest is the authenticated bundle identity supplied to `build_layer_bundle_streamed_lora_model(...)`. The builder binds it back to the LoRA contract's `base_model_sha256`, validates exact bundle membership and paths, builds on `meta`, replaces all frozen linears before `to_empty`, loads only the resident shard, restores the original 2-D causal masks, and failure-atomically loads the adapter. No linear shard is opened during construction.

Every forward/backward linear reload clones its small safetensors mapping into owned contiguous memory before checking exact names, FP32 dtype, shape, finiteness, and tensor SHA-256. The manifest's raw-file hashes are transport/cache identities for the connected downloader; lazy compute authenticates semantic tensor content without adding a second full-file read to every layer operation.

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

## Connected T1 TinyStories result

The connected vertical slice is now exercised end to end through the authenticated HTTP coordinator and two separate native-worker processes. The coordinator was deliberately stopped after the first accepted assignment and loaded from durable state before the second worker ran against the same cache.

Enable publication and consumption with:

```bash
uv run python -m orcacolony.campaign_run \
  --config campaign/t1-tinystories-smoke.json \
  --lora-config campaign/t1-tinystories-lora-smoke.json \
  --participants <participants.json> \
  --dataset-artifacts .artifacts/p3-t1-profile/dataset \
  --state <campaign-state> --browser-root spikes/burn-browser-gradient/www \
  --workers 2 --target-steps 1 --publish-base-layer-bundle

uv run python -m orcacolony.native_worker \
  --coordinator <coordinator-origin> --worker-id <worker-id> \
  --worker-token-file <token-file> \
  --config campaign/t1-tinystories-smoke.json \
  --lora-config campaign/t1-tinystories-lora-smoke.json \
  --cache <shared-cache> --base-profile layer-bundle
```

The committed [`connected-t1.json`](results/connected-t1.json) records:

| Measurement | First process | Restarted coordinator + warm-cache process |
|---|---:|---:|
| Bundle/model transfer | 27,621,509 B | 0 B |
| Adapter transfer | 99,424 B | 0 B |
| Process peak RSS | 371,679,232 B | 372,981,760 B |
| Artifact fetch | 0.722035 s | 0.014692 s |
| Runtime initialization | 1.456090 s | 1.315322 s |
| Gradient relative L2 / max error | 0 / 0 | 0 / 0 |

The cache contained one manifest, one resident shard, and 24 linear shards—no monolithic model artifact. The bundle was only **0.0267%** larger than the original T1 monolith. Strict aggregation produced checkpoint relative L2 `1.2317732405376926e-7` and maximum absolute error `6.246045813895762e-8`. Held-out TinyStories mean loss improved from `9.041835904121399` at initialization to `9.041222333908081` after the fixed one-step campaign; both the initialization and step-one checkpoint were evaluated over 16 declared validation sequences.

The assignment binds the manifest SHA, `base_model_sha256`, ordered artifact names, raw SHA-256 values, exact byte counts, and exact same-origin URLs. Fresh downloads hash bytes before atomic cache publication. Warm startup hashes the small manifest, validates exact directory membership and file metadata, loads/authenticates the resident shard, and defers each linear's semantic tensor digest to first use. A mutated fresh publication, a changed URL, an unexpected cache member, or a semantically changed warm shard fails closed. `python-native-cpu-layer-bundle-f32` results are accepted only when that bundle was actually assigned; gradient, loss, and checkpoint tolerances are unchanged.

## Decision and limits

The connected slice establishes:

- exact gradients and loss remained unchanged;
- final process peak was materially below full residency;
- retained/current memory remained bounded by resident non-linears plus transient layer/activation work;
- no monolithic base was downloaded or cached;
- coordinator restart and warm-cache reuse preserved exact results with zero repeated model or adapter payload bytes; and
- the fixed TinyStories use-case evaluation improved after the accepted checkpoint.

It does not yet prove a public untrusted worker, larger-than-RAM execution, cold physical I/O, or cross-platform RSS. These were Windows runs with local storage. Logical tensor reads are not physical disk or packet-level wire bytes. The exporter still performs complete source scans offline, and the coordinator currently retains both the monolith and a same-size bundle. Warm cache reuse intentionally avoids rehashing every raw shard at startup; semantic digest checks fail on use rather than automatically deleting and refetching a corrupt shard. The path checks plus owned snapshots are not an immutable filesystem snapshot or retained-handle lease, so hostile same-user concurrent mutation is not claimed safe beyond fail-closed content verification.
