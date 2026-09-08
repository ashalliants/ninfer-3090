#!/usr/bin/env bash
set -euo pipefail

# The Qwen3.6 27B groupwise-int artifact, target_key qwen3_6_27b. This is a different model family
# from qwen3_8_27b, which is why having the latter does not satisfy the former: four real-model
# tests -- ninfer_qwen3_6_27b_prefix_real_test, _score_real_test, _load_plan_test and the Qwen3.6
# 27B half of the engine suite -- skip without it.
#
# Pinned at faaa0c14 (17,495,365,888 bytes,
# sha256 7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b).

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_27b.ninfer"

mkdir -p -- "$model_dir"
printf '%s\n' 'Downloading the Qwen3.6-27B model (16.3 GB)...'
if ! curl -L -C - --fail --output "$model" \
  'https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/faaa0c140d0a92743872256a8b78a954b3984018/qwen3_6_27b.ninfer'; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
printf 'Model ready: %s\n' "$model"
printf 'Point the tests at it with:  export NINFER_QWEN3_6_27B_WEIGHTS=%s\n' "$model"
