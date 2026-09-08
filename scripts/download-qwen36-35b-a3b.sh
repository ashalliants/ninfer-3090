#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_35b_a3b.ninfer"

# Revision 560f227e, not c8b8c1c0. The older pin predates the DFlash bundle: its
# artifact-manifest.json has no "dflash" source at all, so an artifact fetched with it cannot run
# --spec dflash and makes ninfer_qwen3_6_35b_a3b_dflash_load_plan_test skip with "this artifact
# carries no DFlash bundle". 560f227e adds it (from z-lab/Qwen3.6-35B-A3B-DFlash) for 0.38 GB more.
# If you repin this, check the manifest still lists a dflash source.
mkdir -p -- "$model_dir"
printf '%s\n' 'Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model...'
if ! curl -L -C - --fail --output "$model" \
  'https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/560f227e5a7104756d1a108201a8aa75654ea688/qwen3_6_35b_a3b.ninfer'; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
printf 'Model ready: %s\n' "$model"
