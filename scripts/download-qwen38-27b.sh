#!/usr/bin/env bash
set -euo pipefail

# The Qwen3.8-27B dense artifact -- the default 27B for every benchmark in this repository, and the
# one docs/config-calculator.html's "27b" rows are measured against.
#
# This script pinned the revision in its URL but verified nothing it received, and resumed straight
# onto the final path: a truncated or corrupt 17 GB download was accepted silently, and a leftover
# partial from any other fetch would be appended to rather than replaced. The size and hash below
# are what HuggingFace reports for this revision (X-Linked-Size and X-Linked-ETag on the resolve
# URL), which is also what the local artifact every published measurement was taken against hashes
# to. Structure matches download-qwen36-27b.sh deliberately, so the two cannot drift.
revision='18dfc887423fa5aabf3cb56fac41490e462b3fab'
expected_size=18210531328
expected_sha256='eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_8_27b.ninfer"

# Staged under a revision-scoped name so that curl -C - can only ever resume the same artifact.
# Resuming straight onto the final path appends at the current length without checking what wrote
# those bytes, so a leftover partial from a different revision would be spliced into this one and
# produce a plausibly sized, wholly corrupt file.
part="$model.$revision.part"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }

# Verifies a file against expected_size and, unless NINFER_SKIP_SHA256=1, expected_sha256. Used
# both for an existing "$model" and for a freshly downloaded "$part", so the two cannot disagree.
# A missing sha256sum/shasum fails closed rather than silently promoting an unverified file.
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

printf '%s\n' 'Downloading the Qwen3.8-27B model (17.0 GiB)...'
if ! curl -L -C - --fail --output "$part" \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/$revision/qwen3_8_27b.ninfer"; then
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
printf 'Point the tests at it with:  export NINFER_QWEN3_8_27B_WEIGHTS=%s\n' "$model"
