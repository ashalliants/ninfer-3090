#!/usr/bin/env bash
# Verifies the Basalt-learnings options on a Linux GPU box: runs tools/bench/basalt_gpu_verify.py one
# step at a time, then writes the summary.
#
#   bash tools/bench/basalt_gpu_verify.sh --build-dir build --dry-run
#   bash tools/bench/basalt_gpu_verify.sh --build-dir build --flash-next /models/qwen4exp-iq2xs-mtp.ninfer \
#       --qwen27b /models/qwen3_8_27b.ninfer --kld-base-27b /kld/qwen38-27b-q8_0-512.kld
#
# Every argument goes to basalt_gpu_verify.py unchanged (`--help` lists them). NINFER_PYTHON picks the
# interpreter (default python3).
#
# No GPU lock here. The shared-GPU lock (CLAUDE.md, tools/gpu-lock.ps1) is a Windows file-share lock
# for the shared desktop, and no Linux bench script in this repository takes one: the Linux GPU box
# is a rented machine used by one session at a time (single-tenant). Do not run this on a Linux
# machine other jobs share without adding a lock around each `--action run` below. GPU tests run
# through plain `ctest -L gpu` for the same reason.
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${NINFER_PYTHON:-python3}"
core="$here/basalt_gpu_verify.py"

for arg in "$@"; do
  case "$arg" in
    --dry-run|--help|-h) exec "$python" "$core" "$@" ;;
  esac
done

if ! listing="$("$python" "$core" "$@" --action steps)"; then
  printf 'basalt_gpu_verify.py --action steps failed; run it with --dry-run to see why.\n' >&2
  exit 1
fi
mapfile -t steps <<< "$listing"

failed=()
for line in "${steps[@]}"; do
  [[ -n "$line" ]] || continue
  id="${line%%$'\t'*}"
  "$python" "$core" "$@" --action run --step "$id" || failed+=("$id")
done

"$python" "$core" "$@" --action summary
if (( ${#failed[@]} )); then
  printf 'Steps that failed: %s\n' "${failed[*]}"
  exit 1
fi
