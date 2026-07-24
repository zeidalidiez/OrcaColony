# OrcaColony

OrcaColony is a volunteer-compute framework for training one small open model at a time. The Milestone 0 through early Milestone 3 reference, browser, connected-worker, multi-worker, checkpoint-continuation, trusted-participant, and attribution proofs described in [SPEC.md](SPEC.md) are runnable.

## Development

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pytest
uv run python -m orcacolony.reference fixture --config campaign/t0-smoke.json --output .artifacts/fixture
uv run python -m orcacolony.reference train --config campaign/t0-smoke.json --output .artifacts/m0
```

The fixture command exports the model, deterministic token batch, summed gradients, and exact hashes needed by the browser parity spike. The training command writes a JSON summary and a resumable safetensors checkpoint under `.artifacts/m0`.

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

## Persistent multi-step campaign

The campaign runner automatically promotes each completed global step, versions its checkpoint, opens the next round with the preserved optimizer and dataset cursor, and rebuilds a campaign-wide accepted-work ledger:

```bash
uv run python -m orcacolony.campaign_run \
  --config campaign/t0-smoke.json \
  --participants campaign/t0-local-participants.json \
  --state .artifacts/campaign \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2 \
  --target-steps 2
```

Open `http://localhost:8000/?cpu-loop=browser-a#token=local-browser-a` to keep one CPU/WASM worker requesting assignments until the target is complete; use `?loop=<worker-id>` for the automatic WebGPU path. The verified continuous browser processed both shards across two rounds without reloads, produced `checkpoints/step-00000001` and `checkpoints/step-00000002`, retained four attribution-ledger entries, survived a coordinator restart, and finished step 2 with cosine similarity `0.9999999999999722` and relative L2 error `2.3485458071128626e-7` against the resumed Python reference.

`campaign/t0-local-participants.json` contains development-only worker identities. Replace it with an owner-approved manifest before exposing a coordinator to another participant; unknown workers remain denied by default.

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
  --target-steps 2
```

The verified continuous CPU/WASM run consumed decoded TinyStories sequences rather than synthetic fixture tokens, completed four browser assignments and two optimizer steps, and carried dataset manifest revision `99e5642bec2a9fa0b7f6175ed5f4821bf4f9aa2c08ec1038f12bfdfb302bb4af` through every assignment, campaign lock, checkpoint, evaluation, and ledger entry. Mean training loss moved from `9.075937747955322` to `8.720811367034912`; the final browser-applied checkpoint matched the centralized reference with cosine similarity `0.9999999999999996` and relative L2 error `2.8234941806709188e-8`.

The frozen held-out profile evaluates the initialization checkpoint and every completed checkpoint on the first 16 packed validation sequences. In the verified run, held-out mean cross-entropy moved from `9.041835904121399` at initialization to `8.723058700561523` after step 1 and `8.514237880706787` after step 2. Results and checkpoint hashes are persisted in `evaluations.json`, and the campaign survives restart without duplicating evaluations.

This remains a two-step correctness smoke, not the multi-day Milestone 4 training run.
