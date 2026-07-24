#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd -W)"
spike="$root/spikes/burn-browser-gradient"
fixture="$root/.artifacts/browser-fixture"

cd "$root"
uv run python -m orcacolony.reference fixture \
  --config campaign/t0-smoke.json \
  --output "$fixture"

wasm-pack build "$spike" \
  --target web \
  --release \
  --out-dir "$spike/www/pkg"

mkdir -p "$spike/www/fixture"
cp "$fixture/fixture.json" "$spike/www/fixture/"
cp "$fixture/model.safetensors" "$spike/www/fixture/"
cp "$fixture/gradients.safetensors" "$spike/www/fixture/"

printf 'Browser spike built at %s\n' "$spike/www"
