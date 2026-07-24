# OrcaColony

OrcaColony is a volunteer-compute framework for training one small open model at a time. The Milestone 0 through Milestone 2 reference, browser, connected-worker, and multi-worker global-step proofs described in [SPEC.md](SPEC.md) are runnable.

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

The bounded Burn/WebGPU spike in [`spikes/burn-browser-gradient`](spikes/burn-browser-gradient) loads that exact fixture, performs browser forward/backward, exports every gradient, and compares it with the Python oracle. Its measured M1a result is recorded with the spike.

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
  --state .artifacts/m2 \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2
```

Open `http://localhost:8000/?worker=browser-a` and then `http://localhost:8000/?worker=browser-b`. The coordinator leases non-overlapping ranges, persists each accepted result, aggregates both summed gradients, normalizes and clips once, and publishes one crash-replayable canonical checkpoint.

Continue from that canonical model and optimizer state by starting the next coordinator with `--resume-from`:

```bash
uv run python -m orcacolony.multiworker \
  --config campaign/t0-smoke.json \
  --state .artifacts/m2-step-2 \
  --resume-from .artifacts/m2/checkpoint \
  --browser-root spikes/burn-browser-gradient/www \
  --workers 2
```

The verified two-step browser run advanced the checkpoint to step 2 and dataset cursor 8 while preserving AdamW state. Its step-2 model reached cosine similarity `0.999999999999976` and relative L2 error `2.185048989144713e-7` against the resumed Python reference.
