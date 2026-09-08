#!/usr/bin/env bash
set -euo pipefail

# Revision 560f227e, not c8b8c1c0. The older pin predates the DFlash bundle: its
# artifact-manifest.json has no "dflash" source at all, so an artifact fetched with it cannot run
# --spec dflash and makes ninfer_qwen3_6_35b_a3b_dflash_load_plan_test skip with "this artifact
# carries no DFlash bundle". 560f227e adds it (from z-lab/Qwen3.6-35B-A3B-DFlash) for 0.38 GB more.
# If you repin this, check the manifest still lists a dflash source, and update the size and
# checksum below along with it.
revision='560f227e5a7104756d1a108201a8aa75654ea688'
expected_size=22783246080
expected_sha256='1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2'

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_6_35b_a3b.ninfer"

# curl -C - resumes by appending at the current file length, without checking what wrote those
# bytes. Because the pinned revision changed, a partial download of the *previous* artifact sits at
# exactly this path on any machine that ran the older script, and resuming onto it would splice the
# tail of one artifact onto the head of another: a file of entirely plausible size that is corrupt
# throughout. Staging under a name that carries the revision means a resume can only ever continue
# the same artifact, and the checks below are what promote it to the final name.
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

printf '%s\n' 'Downloading the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)...'
if ! curl -L -C - --fail --output "$part" \
  "https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/resolve/$revision/qwen3_6_35b_a3b.ninfer"; then
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
