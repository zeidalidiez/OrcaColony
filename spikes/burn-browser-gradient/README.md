# Burn browser-gradient worker

This browser worker loads a Python T0 fixture into Burn, runs one summed-loss backward pass on browser WebGPU or CPU/WASM, and returns either every dense FP32 gradient or the complete frozen-base LoRA adapter-gradient set as a safetensors payload.

Build the generated browser bundle from the repository root:

```bash
bash spikes/burn-browser-gradient/build-browser.sh
python -m http.server 8000 --directory spikes/burn-browser-gradient/www
```

Then open <http://localhost:8000>. Add `?backend=cpu` to force the CPU/WASM backend. Generated fixture and `wasm-pack` output stay untracked.

## Verified result

On July 24, 2026, the full T0 pass ran in Headless Chrome 150 on Windows WebGPU without cross-origin isolation:

- 1,334,016 FP32 gradient values across all 52 parameter tensors
- summed loss: `4267.70556640625` in both Burn and the Python oracle
- gradient cosine similarity: `0.9999999999997954`
- relative L2 error: `5.799612307812865e-7`
- maximum absolute error: `0.0001220703125`
- elapsed browser time, including initialization and readback: `7.23 s`

This clears M1a for the Burn-first path; the TensorFlow.js fallback is not needed at this point.

The same full batch ran through `Autodiff<NdArray<f32, i32>>` in browser WASM:

- summed loss: `4267.705078125`
- gradient cosine similarity: `0.9999999999999276`
- relative L2 error: `2.788216012272494e-7`
- maximum absolute error: `0.0000762939453125`
- elapsed browser time: `0.914 s`

The CPU path is selected automatically when WebGPU is unavailable. Connected workers can force it with `?cpu=<worker-id>#token=<worker-token>`; result protocol v2 records `burn-ndarray-f32` or `burn-webgpu-f32` in the accepted-work ledger.

## Connected-worker proof

Run the M1b coordinator instead of the static server to exercise assignment download, complete-gradient upload, validation, and one centrally applied optimizer step:

```bash
uv run python -m orcacolony.coordinator \
  --config campaign/t0-smoke.json \
  --state .artifacts/m1b \
  --browser-root spikes/burn-browser-gradient/www
```

The verified browser round trip accepted all 52 tensors and produced a step-1 checkpoint with cosine similarity `0.9999999999820228` and relative L2 error `5.996203529187336e-6` against the canonical Python step.

The same client also completes the M2 two-worker proof through `python -m orcacolony.multiworker`. Two browser workers processed disjoint ranges `[0, 2]` and `[2, 4]`; their centrally aggregated checkpoint reached cosine similarity `0.9999999999820688` and relative L2 error `5.988475132799176e-6` against the full-batch Python step.

Passing `--resume-from <previous-state>/checkpoint` continues the canonical model, AdamW state, step number, dataset cursor, and loss history. A verified second browser round used ranges `[4, 6]` and `[6, 8]` and matched the resumed Python step with relative L2 error `2.185048989144713e-7`.

`python -m orcacolony.campaign_run` promotes and versions completed checkpoints and opens subsequent rounds automatically. `?loop=<worker-id>` keeps a WebGPU worker active for the whole campaign; `?cpu-loop=<worker-id>` does the same with CPU/WASM. A verified continuous CPU/WASM worker completed four assignments across two rounds without page reloads, produced step-1 and step-2 checkpoint directories plus one campaign ledger, and matched the final Python checkpoint with relative L2 error `2.3485458071128626e-7`.

## Dynamic T1 scale proof

The worker model dimensions now come from the exact assignment rather than compile-time T0 constants. With `campaign/t1-smoke.json`, the unchanged WASM module loaded a 6,901,760-parameter, 6-layer, width-256 model and exported all 76 gradient tensors. CPU/WASM reached relative L2 error `6.297321030850067e-7` in `2.285 s`; WebGPU reached `1.4272976248000147e-6` in `9.466 s`. A subsequent two-step connected T1 campaign finished with checkpoint relative L2 error `2.9341757103884218e-8`.

## Frozen-base LoRA parity proof

Build the P2 local parity fixture with:

```bash
ORCACOLONY_LORA_CONFIG=campaign/t0-lora-smoke.json \
ORCACOLONY_FIXTURE_DIR=.artifacts/browser-lora-fixture \
bash spikes/burn-browser-gradient/build-browser.sh
```

The Burn module loads the exact dense base with gradients disabled, loads the eight adapter tensors separately, applies rank-4 QKV LoRA in all four layers, and exports only the manifest-declared adapter gradients. The existing dense entry points and 52-tensor export remain separate and unchanged.

The real local browser proof produced:

| Backend | Loss | Adapter tensors / values | Cosine | Relative L2 | Maximum absolute error | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| CPU/WASM | `4267.705078125` | 8 / 8,192 | `0.999999999999924` | `3.851581853662727e-7` | `1.3113021850585938e-6` | `0.978 s` |
| WebGPU | `4267.70556640625` | 8 / 8,192 | `0.9999999999998102` | `6.144610293037611e-7` | `1.1175870895385742e-6` | `21.650 s` |

Both passed the provisional parity gate against Python gradient SHA-256 `7ce16dfd740fd5a249257de6ab442943577b86917f9ae77604c5097ce1a5b8e2`. A fresh dense CPU/WASM regression still exported all 52 tensors and reproduced its prior relative L2 error `2.788216012272494e-7`. LoRA mode is deliberately local-only until adapter checkpoint, lease, aggregation, restart, evaluation, and release semantics are integrated into the coordinator.
