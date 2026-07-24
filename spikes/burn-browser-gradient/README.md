# Burn browser-gradient spike

This bounded M1a spike loads the Python M0 T0 fixture into Burn, runs one summed-loss backward pass on browser WebGPU, and returns every FP32 gradient as a safetensors payload.

Build the generated browser bundle from the repository root:

```bash
bash spikes/burn-browser-gradient/build-browser.sh
python -m http.server 8000 --directory spikes/burn-browser-gradient/www
```

Then open <http://localhost:8000>. Generated fixture and `wasm-pack` output stay untracked.

## Verified result

On July 24, 2026, the full T0 pass ran in Headless Chrome 150 on Windows WebGPU without cross-origin isolation:

- 1,334,016 FP32 gradient values across all 52 parameter tensors
- summed loss: `4267.70556640625` in both Burn and the Python oracle
- gradient cosine similarity: `0.9999999999997954`
- relative L2 error: `5.799612307812865e-7`
- maximum absolute error: `0.0001220703125`
- elapsed browser time, including initialization and readback: `7.23 s`

This clears M1a for the Burn-first path; the TensorFlow.js fallback is not needed at this point.
