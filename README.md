# OrcaColony

OrcaColony is a volunteer-compute framework for training one small open model at a time. The project is currently implementing the Milestone 0 single-process reference described in [SPEC.md](SPEC.md).

## Development

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run pytest
uv run python -m orcacolony.reference fixture --config campaign/t0-smoke.json --output .artifacts/fixture
uv run python -m orcacolony.reference train --config campaign/t0-smoke.json --output .artifacts/m0
```

The fixture command exports the model, deterministic token batch, summed gradients, and exact hashes needed by the browser parity spike. The training command writes a JSON summary and a resumable safetensors checkpoint under `.artifacts/m0`.
