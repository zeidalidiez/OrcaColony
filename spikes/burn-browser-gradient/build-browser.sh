#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
case "$(uname -s)" in
  CYGWIN* | MINGW* | MSYS*) root="$(cygpath -m "$root")" ;;
esac
spike="$root/spikes/burn-browser-gradient"
config="${ORCACOLONY_CONFIG:-$root/campaign/t0-smoke.json}"
fixture="${ORCACOLONY_FIXTURE_DIR:-$root/.artifacts/browser-fixture}"

cd "$root"
uv run python -m orcacolony.reference fixture \
  --config "$config" \
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
