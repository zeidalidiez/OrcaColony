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

`campaign/t0-local-participants.json` contains development-only worker identities. Replace it with an owner-approved manifest before exposing a coordinator to another participant; unknown workers remain denied by default.
