#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
case "$(uname -s)" in
  CYGWIN* | MINGW* | MSYS*) root="$(cygpath -m "$root")" ;;
esac
spike="$root/spikes/burn-browser-gradient"
config="${ORCACOLONY_CONFIG:-$root/campaign/t0-smoke.json}"
fixture="${ORCACOLONY_FIXTURE_DIR:-$root/.artifacts/browser-fixture}"
lora_config="${ORCACOLONY_LORA_CONFIG:-}"
dataset_args=()
if [[ -n "${ORCACOLONY_DATASET_ARTIFACTS:-}" ]]; then
  dataset_args=(--dataset-artifacts "$ORCACOLONY_DATASET_ARTIFACTS")
fi

cd "$root"
if [[ -n "$lora_config" ]]; then
  uv run python -m orcacolony.peft export-fixture \
    --campaign "$config" \
    --lora "$lora_config" \
    --output "$fixture"
else
  uv run python -m orcacolony.reference fixture \
    --config "$config" \
    --output "$fixture" \
    "${dataset_args[@]}"
fi

wasm-pack build "$spike" \
  --target web \
  --release \
  --out-dir "$spike/www/pkg"

rm -rf "$spike/www/fixture"
mkdir -p "$spike/www/fixture"
cp "$fixture/fixture.json" "$spike/www/fixture/"
cp "$fixture/gradients.safetensors" "$spike/www/fixture/"
if [[ -n "$lora_config" ]]; then
  cp "$fixture/base.safetensors" "$spike/www/fixture/"
  cp "$fixture/adapter.safetensors" "$spike/www/fixture/"
else
  cp "$fixture/model.safetensors" "$spike/www/fixture/"
fi

printf 'Browser spike built at %s\n' "$spike/www"
