# OrcaColony

OrcaColony is a volunteer-compute framework and reproducible research vehicle for community model training. Its proven local systems path accumulates independent, bounded contributions against one canonical checkpoint; active studies evaluate PEFT, local offload, partial-model work, tiled recovery, and sparse experts. Capability-research infrastructure now proceeds as a parallel track: every promoted model must have a fixed use case, versioned behavioral baseline and suite, positive baseline improvement, separately reserved language-loss and behavioral holdouts, training-effect analysis, contributor attribution snapshot, and reproducible Hugging Face model and dataset packages. The Milestone 0 through Milestone 3 reference, browser, multi-worker, persistent campaign, trusted-participant, live dashboard, evaluation, and operational-release paths described in [SPEC.md](SPEC.md) are runnable. A bounded local T1 preflight exercises the Milestone 4 system profile through 12 real-data optimizer steps. The first T2 task, evaluator, and supported-runtime baseline are frozen, but a training-effect result, public Hub publication, and remote trusted-participant campaign remain incomplete.

See [PROGRESS_REPORT.md](PROGRESS_REPORT.md) for the current build position, completed work, blockers, priority order, and immediate bounded target. Browse [reports/index.html](reports/index.html) for human-readable findings, comparisons, limitations, and next-iteration decisions.

## Development

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pytest
uv run python -m orcacolony.reference fixture --config campaign/t0-smoke.json --output .artifacts/fixture
uv run python -m orcacolony.reference train --config campaign/t0-smoke.json --output .artifacts/m0
```

The fixture command exports the model, deterministic token batch, summed gradients, and exact hashes needed by the browser parity spike. The training command writes a JSON summary and a resumable safetensors checkpoint under `.artifacts/m0`.

## Reproducible research records

[`research/README.md`](research/README.md) defines the linked study, experiment, evidence, and result-bundle flow used for post-v0.1 research. Run the committed contract proof with:

```bash
uv run python -m orcacolony.research record \
  --study research/studies/p1-research-contract-smoke-v1/study.json \
  --experiment research/studies/p1-research-contract-smoke-v1/experiments/deterministic-result-bundle.json \
  --evidence research/studies/p1-research-contract-smoke-v1/evidence/deterministic-result-bundle.json \
  --output .artifacts/p1-research-contract-result
```

The command fails closed on malformed or unlinked inputs and atomically produces canonical source manifests, captured environment provenance, digest-verified snapshots of committed `repo:` evidence, explicit unresolved-artifact warnings, `result.json`, a readable `RESULT.md`, and `SHA256SUMS`. Validated and promoted outcomes must pass the study's fixed use-case metric and every guardrail; rejected and inconclusive records retain their findings and limitations.

Machine-readable research records remain under `research/`. The separate root [`reports/`](reports/) directory publishes self-contained HTML for human interpretation. P5–P7 method engineering continues through bounded executable studies while slower owner-directed data, example, checkpoint, and practical-quality review proceeds in parallel.

Capability-model promotion uses the stricter contract in
[`research/CAPABILITY_CAMPAIGNS.md`](research/CAPABILITY_CAMPAIGNS.md). Hub
repository layout, credential handling, deterministic packaging, and explicit
publication are documented in [`HUGGINGFACE.md`](HUGGINGFACE.md).

## First capability task

[`capability/record-patch-v1/TASK.md`](capability/record-patch-v1/TASK.md)
defines the first true-T2 task. A 17,538,816-parameter decoder must apply ordered
`SET`, `DELETE`, and `RENAME` operations to flat JSON records and emit the exact
canonical result. The task has deterministic CC0 data, a public 32-example
behavioral-validation suite, a separately keyed 128-example final holdout, an
exact sample-level evaluator, fixed thresholds, contributor intake, and
private-first Hugging Face destinations.

The measured random initialization scored `0/32` exact and `2/32` valid JSON.
Read [`reports/record-patch-t2-baseline.html`](reports/record-patch-t2-baseline.html)
for the complete baseline, outputs, hashes, and limitations. Python 3.11.15
reproduced the initialization identity, all 32 predictions, and the complete
evaluation byte for byte from the earlier Python 3.14 run. The final holdout was
not opened. The first centralized learnability schedule and pass/fail rule are
fixed in
[`capability/record-patch-v1/learnability-protocol.json`](capability/record-patch-v1/learnability-protocol.json).
It stops at 128 one-thread CPU updates, selects by public language loss, requires
an exact behavioral improvement, records gradient and update diagnostics, and
has no private-holdout input.

That first check did not pass. Step 128 lowered public language loss from
`9.1207` to `1.5636`, but the language-selected checkpoint remained `0/32`
exact and `0/32` strict valid JSON because all 32 outputs repeated object keys.
It covered only `0.111` packed-data epochs, so the next control will continue
the same exact checkpoint and optimizer before changing the objective or
learning rate. Volunteer training remains blocked. Read
[`reports/record-patch-t2-learnability-v1.html`](reports/record-patch-t2-learnability-v1.html)
for every public output, optimizer diagnostics, overlap analysis, and the
negative disposition.

## PEFT numerical proof

The first P2 slice freezes the exact T0 base and applies rank-4 LoRA adapters to every combined attention QKV projection. Export its deterministic base, initial adapter, batch, complete summed-loss adapter gradients, and independently verified one-step adapter update with:

```bash
uv run python -m orcacolony.peft export-fixture \
  --campaign campaign/t0-smoke.json \
  --lora campaign/t0-lora-smoke-cpu.json \
  --output .artifacts/t0-lora-fixture-v1

cd .artifacts/t0-lora-fixture-v1
sha256sum -c SHA256SUMS
```

The manifest pins the base campaign and model hashes, exact adapter targets and initialization, named trainable tensor order, summed-loss gradient contract, coordinator normalization and clipping rules, and updated-adapter hash. This command is the isolated Python numerical proof; the connected browser and native profiles below extend that contract without implying that T0 itself is a useful adaptation target.

## Browser feasibility proof

The bounded Burn spike in [`spikes/burn-browser-gradient`](spikes/burn-browser-gradient) loads that exact fixture, performs browser forward/backward through either WebGPU or CPU/WASM, exports every gradient, and compares it with the Python oracle. Its measured results are recorded with the spike.

Build it, then run the M1b coordinator from a fresh state directory:

```bash
bash spikes/burn-browser-gradient/build-browser.sh
uv run python -m orcacolony.coordinator \
  --config campaign/t0-smoke.json \
  --state .artifacts/m1b \
  --browser-root spikes/burn-browser-gradient/www
```

Open the URL printed by the coordinator. The browser receives one checkpoint-bound assignment, uploads all gradients, and the coordinator validates them, applies the canonical optimizer step, and writes a new checkpoint plus `receipt.json`.

## Multi-worker global step

Run the M2 coordinator from a fresh state directory:

```bash
uv run python -m orcacolony.multiworker \
  --config campaign/t0-smoke.json \
  --participants campaign/t0-local-participants.json \
  --state .artifacts/m2 \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2
```

Open `http://localhost:8000/?worker=browser-a#token=local-browser-a` and then `http://localhost:8000/?worker=browser-b#token=local-browser-b`. The coordinator denies worker IDs absent from the exact participant manifest and verifies their token against its stored SHA-256 digest. The browser keeps the token in the URL fragment, which is not sent in the HTTP request URL. The coordinator leases non-overlapping ranges, persists each accepted result, aggregates both summed gradients, normalizes and clips once, and publishes one crash-replayable canonical checkpoint. `campaign-lock.json` freezes the campaign, participant, checkpoint, and protocol revisions; `accepted-work.json` records accepted contributions without publishing names unless their contributor opted in.

Use `?cpu=browser-a#token=local-browser-a` to force the Burn NdArray CPU/WASM worker. A verified mixed WebGPU/CPU global step recorded each result backend in the accepted-work ledger and matched the Python checkpoint with cosine similarity `0.9999999999871374` and relative L2 error `5.071985986524343e-6`.

Continue from that canonical model and optimizer state by starting the next coordinator with `--resume-from`:

```bash
uv run python -m orcacolony.multiworker \
  --config campaign/t0-smoke.json \
  --participants campaign/t0-local-participants.json \
  --state .artifacts/m2-step-2 \
  --resume-from .artifacts/m2/checkpoint \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2
```

The verified two-step browser run advanced the checkpoint to step 2 and dataset cursor 8 while preserving AdamW state. Its step-2 model reached cosine similarity `0.999999999999976` and relative L2 error `2.185048989144713e-7` against the resumed Python reference.

Run the explicit frozen-base LoRA profile through the same connected global-step path with:

```bash
uv run python -m orcacolony.multiworker \
  --config campaign/t0-smoke.json \
  --lora-config campaign/t0-lora-smoke-cpu.json \
  --participants campaign/t0-local-participants.json \
  --state .artifacts/p2-connected-lora \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2
```

The coordinator serves the immutable dense base and current adapter separately, binds every assignment to both weight identities, the full optimizer-backed resume-state identity, and the exact eight-tensor trainable manifest, accepts no base gradients, and publishes only the next canonical adapter plus its optimizer state. Use the same `?cpu=browser-a#token=local-browser-a` and `?cpu=browser-b#token=local-browser-b` URLs for the CPU/WASM path. In the connected proof, both browser assignments were accepted, the adapter checkpoint survived coordinator restart, and its 8,192 values matched an independent centralized Python step with cosine `0.9999999999985897`, relative L2 `1.6795715402185043e-6`, and maximum absolute error `7.257913239300251e-7`.

This establishes connected adapter-only coordination, not reduced frozen-base execution memory. Persistent LoRA evaluation and release now use the same campaign path below; dense mode remains unchanged.

## Persistent multi-step campaign

The campaign runner automatically promotes each completed global step, versions its checkpoint, opens the next round with the preserved optimizer and dataset cursor, and rebuilds a campaign-wide accepted-work ledger:

```bash
uv run python -m orcacolony.campaign_run \
  --config campaign/t0-smoke.json \
  --lora-config campaign/t0-lora-smoke-cpu.json \
  --participants campaign/t0-local-participants.json \
  --state .artifacts/campaign \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2 \
  --numerical-profile burn-ndarray-f32-v1 \
  --target-steps 2
```

Omit `--lora-config` for dense training. In LoRA mode the runner evaluates versioned adapters, resumes the exact coordinator AdamW trajectory, and exposes distinct base, adapter, worker-weight, and complete resume-state identities. The deterministic release CLI accepts the same `--lora-config` and publishes `base-model.safetensors`, `adapter.safetensors`, optimizer moments, and resume metadata separately; see `p2-persistent-lora-release-v1` in [`research/README.md`](research/README.md) for the complete reproducible command.

Open `http://localhost:8000/?cpu-loop=browser-a#token=local-browser-a` to keep one CPU/WASM worker requesting assignments until the target is complete; use `?loop=<worker-id>` for the automatic WebGPU path. The verified continuous browser processed both shards across two rounds without reloads, produced `checkpoints/step-00000001` and `checkpoints/step-00000002`, retained four attribution-ledger entries, survived a coordinator restart, and finished step 2 with cosine similarity `0.9999999999999722` and relative L2 error `2.3485458071128626e-7` against the resumed Python reference.

The worker page also shows live optimizer-step and accepted-token progress, frozen model and dataset provenance, checkpoint evaluation history, opt-in contributor acknowledgements, and a privacy-filtered public work ledger. It polls `/api/v1/dashboard` without a worker credential. For a frontend hosted on a separate static origin, build the release site with an operator-pinned coordinator URL and pass that static site's exact origin as `--public-origin` to the coordinator. Remote worker execution refuses a coordinator supplied only through a mutable query parameter. Keep participant manifests and credentials off the static host; credentials remain in URL fragments and are sent only in worker API headers.

`campaign/t0-local-participants.json` contains development-only worker identities. Replace it with an owner-approved manifest before exposing a coordinator to another participant; unknown workers remain denied by default.

## Measured browser and native resource profiles

P3 assignments and accepted receipts carry a strict resource-observation contract. Workers report assignment/runtime/artifact/gradient durations, worker-observed response payload bytes, and runtime-specific memory observations; the coordinator independently records declared artifact sizes, exact result-upload and persisted-result bytes, receive duration, and current campaign storage. Reports are validated on acceptance and restart, aggregated without worker identity or hardware-capacity fields, and rendered on the public dashboard. Payload bytes measure what crossed the worker API boundary; they do not claim packet-level wire size or distinguish a browser HTTP-cache hit.

The first measured two-worker CPU/WASM LoRA proof downloaded the immutable 5,340,824-byte base once per assignment. That repeated transfer selected a content-addressed cached-base native CPU worker as the first native baseline. Run one authenticated native assignment with a token stored outside the command line:

```bash
uv run python -m orcacolony.native_worker \
  --coordinator http://127.0.0.1:8000 \
  --worker-id native-a \
  --worker-token-file .artifacts/native-a.token \
  --config campaign/t0-smoke.json \
  --lora-config campaign/t0-lora-smoke-cpu.json \
  --cache .artifacts/native-worker-cache \
  --assignments 2
```

The worker pins the coordinator origin, rejects cross-origin redirects, streams the frozen base and adapter into digest-named safetensors cache entries, validates tensor identities before reuse, computes only the declared adapter gradients, and submits through the same lease and telemetry contract. The reproducible [`p3-native-resource-profile-v1`](research/studies/p3-native-resource-profile-v1) study extends the proof to frozen TinyStories T1 and a historical 91,544,064-parameter memory-stress profile whose immutable campaign ID uses `t2`. That legacy label is retained for provenance; new capability work reserves T2 for the specification's approximately 17.5M-parameter tier. At both tested scales the warm second process fetched zero base and adapter payload bytes. The 91M profile retained `2.1866353039878446e-7` checkpoint relative L2, improved held-out mean loss by `0.0033478736877441406`, and measured `1,746,419,712` bytes peak process RSS.

`--assignments N` is a bounded persistent session: it validates and builds the base/model once, reuses the in-memory adapter while its digest is unchanged, and fetches and loads only a new adapter after the coordinator advances a global step. Adapter refresh validates and converts the complete tensor set before mutating the model, so malformed state cannot leave a mixed adapter behind. The two-step integration test builds one model across four assignments and loads exactly two checkpoint-specific adapters.

The one-shot T2 baseline spent `1.013742800001637 s` revalidating its 366,190,504-byte cached base and `1.9494272000010824 s` rebuilding/loading the runtime before `1.653274500000407 s` of gradient compute. The reproducible [`p3-persistent-native-session-v1`](research/studies/p3-persistent-native-session-v1) proof reduced reused-assignment setup to `0.00004199999966658652 s`—a 99.9986% reduction—while retaining `2.1866353039878446e-7` checkpoint relative L2. Complete FP32 CPU residency remains the baseline; authenticated layer-bundle storage and int8 frozen-linear residency are separately qualified placements rather than silent substitutions.

### Connected homogeneous int8 frozen-linear profile

`build_int8_lora_model(...)` exposes the explicit numerical profile `int8-per-output-symmetric-f32-dequant-v1`. It stores frozen linear weights as per-output-channel symmetric int8 buffers, reconstructs FP32 weights inside a custom forward/backward function, and leaves LoRA parameters FP32 and trainable. The coordinator selects an int8 oracle and profile-authenticated v2 checkpoint identity; exact CPU FP32 remains bit-exact, while Burn NdArray and WebGPU use separate named numerical/runtime profiles.

The reproducible [`int8-frozen-linear` spike](spikes/int8-frozen-linear) reduced unique resident model-tensor bytes by 43.08% at T0, 50.18% at T1, and 69.23% at T2. T2 adapter-gradient cosine remained `0.9995400`, but relative L2 reached `0.03038196`—about 3,038 times the connected FP32 bound.

The P4 follow-up fixed a 20-step T1 TinyStories trajectory and a profile-specific int8 control. Two-shard int8 aggregation stayed within `3.1242e-7` gradient relative L2 of centralized int8, final adapters stayed within `1.5513e-7`, and ten post-restart gradients/adapter/optimizer states were exact. Int8 held-out mean loss improved by `0.1061466932` versus `0.1049623489` FP32 while retaining 50.18% fewer model-tensor bytes. The int8 and FP32 adapters still diverged by 2.6366%, so they cannot share a campaign.

`build_layer_bundle_int8_lora_model(...)` authenticates and quantizes one layer-bundle shard at a time without resident FP32 construction. It reproduced converted-int8 loss and gradient SHA-256 exactly at T1/T2. At T2 it retained the same `113,080,320` tensor bytes while reducing build peak RSS from `1,380,302,848` to `458,956,800` bytes and final peak from `1,380,302,848` to `845,414,400`. The connected T1 qualification mixed one resident-converted and one direct-bundle int8 worker across two optimizer steps and a coordinator restart: all four worker gradients were exact against the int8 oracle, maximum checkpoint relative L2 was `6.12852298785371e-8`, warm model transfer was zero, and held-out loss improved by `0.001623988151550293`. The final adapter remained 1.1019% from exact FP32, and cross-profile restart was rejected.

### Offline exact-FP32 streamed-linear profile

`build_streamed_lora_model(..., storage_dir)` exposes `streamed-fp32-frozen-linear-v1`. It exports each frozen linear to an exclusive per-layer safetensors directory, removes those tensors from the resident model, and reloads authenticated FP32 weights in both forward and backward. Every reload checks exact names, shapes, dtypes, finite values, and tensor SHA-256; CPU FP32 semantics remain distinct from ambient autocast.

The [`streamed-fp32-linear` spike](spikes/streamed-fp32-linear) retained only `27,482,112` model-tensor bytes at T2 versus `367,552,512` for full residency, a 92.52% reduction. Complete adapter gradients and loss were bit-identical. The T2 gradient read `673,053,696` application-level storage bytes and took 1.307x the full-resident runtime on one warm local-storage run, including a defensive copy before validation/use. This builder still constructs the full FP32 model before exporting its linears, so the next slice must construct directly from the authenticated base artifact before claiming lower startup RSS or larger-than-RAM execution.

### Direct streamed startup profile

`build_direct_streamed_lora_model(...)` creates the decoder and LoRA wrappers on PyTorch's `meta` device, replaces all frozen linears with path-only modules backed by one authenticated base safetensors file, materializes only non-linear tensors/adapters on CPU, and supports deterministic restart from base plus adapter artifacts. It never calls the resident FP32 model builder.

The original [`direct-streamed-fp32` startup proof](spikes/direct-streamed-fp32) preserved the exact T2 loss and gradient SHA-256 while reducing retained tensors by 92.52%, RSS after build by 49.67%, and RSS after gradient by 45.88%. Its apparent 1.94% peak-RSS reduction was later traced to the proof calling the default LoRA-manifest loader, which built a complete resident base before the direct constructor. That observation remains published, but it does not isolate direct construction.

### Authenticated layer-bundle startup profile

`export_base_layer_bundle(...)` performs the complete raw/canonical base verification once as an offline publication step, then writes one resident safetensors shard, one shard per frozen linear, and a canonical manifest bound to `base_model_sha256`. `build_layer_bundle_streamed_lora_model(...)` authenticates that small manifest, constructs on `meta`, opens only the resident shard at startup, and checks an owned exact-FP32 tensor snapshot against its manifest digest on every lazy linear load. The ordinary LoRA loader still verifies the deterministic base by default; only authenticated artifact-backed startup explicitly defers that redundant resident build.

The [`layer-bundle-fp32` proof](spikes/layer-bundle-fp32) retained the exact T2 loss and gradient SHA-256, opened zero linear shards during startup, reduced retained tensor bytes by 92.52% and final process peak RSS by 46.35% versus resident construction, and reduced build time by 62.53% versus a corrected direct run. Corrected direct and bundle peak RSS were effectively equal (`740,708,352` versus `740,360,192` bytes); the bundle's gain is independently authenticated/fetchable layers and removal of complete-container startup work, not further process-memory reduction over direct construction.

The profile now runs through the connected native worker with `--publish-base-layer-bundle` on the coordinator and `--base-profile layer-bundle` on the worker. Authenticated assignments bind the base identity, manifest, ordered raw hashes/byte counts, and exact same-origin shard URLs; fresh downloads are hashed before atomic cache publication, and warm shards are reauthenticated semantically when used. A two-process T1 TinyStories campaign crossed a coordinator restart with exact gradients, transferred `27,621,509` model bytes on the cold worker and `0` on the warm worker, cached no monolithic base, matched the centralized checkpoint at relative L2 `1.2317732405376926e-7`, and improved declared held-out mean loss from `9.041835904121399` to `9.041222333908081`.

A mixed T2 campaign then aggregated one resident and one layer-bundle worker under the same exact CPU FP32 contract. Both worker gradients were bit-exact; the aggregate checkpoint matched at relative L2 `2.1866353039878446e-7`, and held-out loss improved from `9.171425819396973` to `9.168077945709229`. The bundle worker reduced process peak RSS from `1,746,604,032` to `556,244,992` bytes (68.15%) at a 2.238x gradient-runtime cost and essentially unchanged cold model payload. This qualifies mixed resident/bundle placement, not mixed approximate numerical profiles.

## T1 scale proof

`campaign/t1-smoke.json` raises the same dynamic browser runtime to the planned T1 shape: 6 layers, width 256, 4 heads, 8,192 tokens, context 256, and exactly 6,901,760 parameters. Build its synthetic parity fixture with:

```bash
ORCACOLONY_CONFIG=campaign/t1-smoke.json \
ORCACOLONY_FIXTURE_DIR=.artifacts/t1-browser-fixture \
bash spikes/burn-browser-gradient/build-browser.sh
```

The verified browser results covered all 76 tensors and 6,901,760 gradient values. CPU/WASM completed one batch-1 pass in `2.285 s` with cosine similarity `0.9999999999994084`; WebGPU completed it in `9.466 s` with cosine similarity `0.9999999999986344`. A continuous CPU/WASM worker then completed a two-shard, two-step T1 campaign with preserved AdamW state, two versioned checkpoints, and four ledger entries. Its final checkpoint matched the resumed Python reference with cosine similarity `0.9999999999999996` and relative L2 error `2.9341757103884218e-8`.

This proves the T1 runtime and coordination scale using deterministic synthetic tokens. It is not yet the Milestone 4 language-model campaign; the frozen redistributable corpus, tokenizer, evaluation set, and multi-day trusted run remain separate promotion requirements.

## Frozen TinyStories artifacts

Build a reproducible, license-noticed TinyStories subset and an 8,192-entry byte-level BPE tokenizer from the pinned upstream revision:

```bash
uv run python -m orcacolony.artifacts \
  --output .artifacts/tinystories-t1-v1
```

The default build downloads 16 MiB of training data and 2 MiB of validation data, trims both to complete stories, and packs shifted input/target sequences of length 256 into safetensors. Two independent builds were byte-identical. The measured artifact contains 18,544 training stories, 2,502 validation stories, 15,666 training sequences, and 1,934 validation sequences. Its tokenizer SHA-256 is `7cb0fc243e7fa2bcfb9e1087ece80f3cbffc642d2c53f3213edaab218d0139bb`.

Source revision, complete-source hashes, subset hashes, tokenizer identity, packing metadata, file hashes, attribution, transformations, and license URL are frozen in the generated manifest and notice. See [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

Run the T1 browser campaign directly against those frozen artifacts:

```bash
uv run python -m orcacolony.campaign_run \
  --config campaign/t1-tinystories-smoke.json \
  --participants campaign/t1-tinystories-local-participants.json \
  --dataset-artifacts .artifacts/tinystories-t1-v1 \
  --state .artifacts/t1-tinystories-campaign \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2 \
  --numerical-profile burn-ndarray-f32-v1 \
  --target-steps 2
```

The verified continuous CPU/WASM run consumed decoded TinyStories sequences rather than synthetic fixture tokens, completed four browser assignments and two optimizer steps, and carried dataset manifest revision `99e5642bec2a9fa0b7f6175ed5f4821bf4f9aa2c08ec1038f12bfdfb302bb4af` through every assignment, campaign lock, checkpoint, evaluation, and ledger entry. Mean training loss moved from `9.075937747955322` to `8.720811367034912`; the final browser-applied checkpoint matched the centralized reference with cosine similarity `0.9999999999999996` and relative L2 error `2.8234941806709188e-8`.

The frozen held-out profile evaluates the initialization checkpoint and every completed checkpoint on the first 16 packed validation sequences. In the verified run, held-out mean cross-entropy moved from `9.041835904121399` at initialization to `8.723058700561523` after step 1 and `8.514237880706787` after step 2. Results and checkpoint hashes are persisted in `evaluations.json`, and the campaign survives restart without duplicating evaluations.

Export a completed campaign as a deterministic, privacy-filtered release directory:

```bash
uv run python -m orcacolony.release \
  --config campaign/t1-tinystories-smoke.json \
  --participants campaign/t1-tinystories-local-participants.json \
  --dataset-artifacts .artifacts/tinystories-t1-v2 \
  --campaign-state .artifacts/t1-tinystories-campaign \
  --browser-root spikes/burn-browser-gradient/www \
  --public-coordinator-url https://coordinator.example \
  --output .artifacts/t1-tinystories-release
```

The bundle selects the checkpoint with the lowest frozen held-out loss and includes its model, optimizer and restart metadata; exact packed data and tokenizer; evaluation records; a public dashboard snapshot and privacy-filtered contribution ledger; the fixture-free static browser site; a canonical release manifest; and `SHA256SUMS`. Participant manifests and credentials are deliberately excluded.

## Bounded T1 system proof

`campaign/t1-tinystories-system-proof.json` extends the same frozen T1 campaign profile to 12 optimizer steps and declares a pre-run success gate requiring held-out mean-loss improvement of at least `0.5` from initialization. A completed local run used two concurrently active, separately attributed CPU/WASM browser identities and produced:

- 12 canonical optimizer steps from 24 accepted browser assignments and 6,144 accepted tokens;
- balanced public acknowledgements of 12 assignments and 3,072 tokens for each browser identity;
- 13 immutable evaluations covering initialization and steps 1 through 12;
- held-out mean loss `9.041835904121399 → 7.640471279621124`, an improvement of `1.4013646245002747`, passing the declared gate;
- a final 6,901,760-value checkpoint with relative L2 error `4.431320086521597e-9`, maximum absolute error `2.679882982192794e-8`, and cosine similarity `1.0` against the resumed centralized reference; and
- a successful completed-campaign restart plus a deterministic public release bundle whose complete `SHA256SUMS` verified.

The exported site was also served from a separate static origin with a pinned campaign and coordinator origin; a real CPU/WASM browser completed a cross-origin assignment/result cycle through exact-origin CORS. Remote non-loopback origins require HTTPS.

This is the bounded local preflight for Milestone 4. The planned several-day campaign with distinct remote, owner-approved participants still requires an operator-selected HTTPS deployment and those participant approvals.
