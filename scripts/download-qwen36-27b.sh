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

# Verifies a file against expected_size and, unless NINFER_SKIP_SHA256=1, expected_sha256.
# Used both for an existing "$model" (so a same-sized-but-corrupt file is not accepted forever
# just because it happened to pass once, or was replaced out from under this script) and for a
# freshly downloaded "$part" -- one check that cannot drift out of sync with itself. A missing
# sha256sum/shasum fails closed rather than silently promoting an unverified file: the whole
# reason this artifact is checksummed is to catch a same-sized-but-corrupt file, and silently
# skipping that would defeat it, not just once, but for every future run on that host.
verify() {
  [ "$(file_size "$1")" = "$expected_size" ] || return 1
  [ "${NINFER_SKIP_SHA256:-0}" = '1' ] && return 0
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum -- "$1" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 -- "$1" | cut -d' ' -f1)"
  else
    printf 'No sha256sum or shasum found; cannot verify %s. Set NINFER_SKIP_SHA256=1 to accept it unverified.\n' "$1" >&2
    return 1
  fi
  [ "$actual" = "$expected_sha256" ]
}

mkdir -p -- "$model_dir"

if [ -f "$model" ]; then
  if verify "$model"; then
    printf 'Model already present: %s\n' "$model"
    exit 0
  fi
  printf '%s\n' "Existing $model did not verify against revision $revision; fetching the pinned one." >&2
fi

printf '%s\n' 'Downloading the Qwen3.6-27B model (16.3 GiB)...'
if ! curl -L -C - --fail --output "$part" \
  "https://huggingface.co/neroued/Qwen3.6-27B-NInfer/resolve/$revision/qwen3_6_27b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi

if ! verify "$part"; then
  printf 'Downloaded file at %s failed verification against revision %s. Delete it and run this script again.\n' \
    "$part" "$revision" >&2
  exit 1
fi

mv -f -- "$part" "$model"
printf 'Model ready: %s\n' "$model"
printf 'Point the tests at it with:  export NINFER_QWEN3_6_27B_WEIGHTS=%s\n' "$model"
