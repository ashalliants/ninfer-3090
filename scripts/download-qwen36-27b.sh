#!/usr/bin/env bash
set -euo pipefail

# The Qwen3.6 27B groupwise-int artifact, target_key qwen3_6_27b. This is a different model family
# from qwen3_8_27b, which is why having the latter does not satisfy the former: four real-model
# tests -- ninfer_qwen3_6_27b_prefix_real_test, _score_real_test, _load_plan_test and the Qwen3.6
# 27B half of the engine suite -- skip without it.
revision='faaa0c140d0a92743872256a8b78a954b3984018'
expected_size=17495365888
expected_sha256='7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_27b.ninfer"

# Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
# Resuming straight onto the final path appends at the current length without checking what wrote
# those bytes, so a leftover partial from a different revision would be spliced into this one and
# produce a plausibly sized, wholly corrupt file. See download-qwen36-35b-a3b.sh, where changing
# the pin made that a live hazard rather than a hypothetical one.
part="$model.$revision.part"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }

mkdir -p -- "$model_dir"

if [ -f "$model" ]; then
  if [ "$(file_size "$model")" = "$expected_size" ]; then
    printf 'Model already present: %s\n' "$model"
    exit 0
  fi
  printf '%s\n' "Existing $model is not revision $revision; fetching the pinned one." >&2
fi

printf '%s\n' 'Downloading the Qwen3.6-27B model (16.3 GiB)...'
if ! curl -L -C - --fail --output "$part" \
  "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/$revision/qwen3_6_27b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi

actual_size="$(file_size "$part")"
if [ "$actual_size" != "$expected_size" ]; then
  printf 'Expected %s bytes, got %s. Delete %s and run this script again.\n' \
    "$expected_size" "$actual_size" "$part" >&2
  exit 1
fi

# Set NINFER_SKIP_SHA256=1 to skip this: it costs a full re-read of 16.3 GiB. The size check above
# already rejects a truncated or spliced file, so this one is here for silent corruption.
if [ "${NINFER_SKIP_SHA256:-0}" != '1' ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256="$(sha256sum -- "$part" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    actual_sha256="$(shasum -a 256 -- "$part" | cut -d' ' -f1)"
  else
    actual_sha256=''
    printf '%s\n' 'No sha256sum or shasum found; skipping the checksum.' >&2
  fi
  if [ -n "$actual_sha256" ] && [ "$actual_sha256" != "$expected_sha256" ]; then
    printf 'Checksum mismatch (expected %s, got %s). Delete %s and run this script again.\n' \
      "$expected_sha256" "$actual_sha256" "$part" >&2
    exit 1
  fi
fi

mv -f -- "$part" "$model"
printf 'Model ready: %s\n' "$model"
printf 'Point the tests at it with:  export NINFER_QWEN3_6_27B_WEIGHTS=%s\n' "$model"
