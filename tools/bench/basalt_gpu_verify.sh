#!/usr/bin/env bash
# Verifies the Basalt-learnings options on a Linux GPU box: runs tools/bench/basalt_gpu_verify.py one
# step at a time, then writes the summary.
#
#   bash tools/bench/basalt_gpu_verify.sh --build-dir build --dry-run
#   bash tools/bench/basalt_gpu_verify.sh --build-dir build --flash-next /models/qwen4exp-iq2xs-mtp.ninfer \
#       --qwen27b /models/qwen3_8_27b.ninfer --kld-base-27b /kld/qwen38-27b-q8_0-512.kld
#   bash tools/bench/basalt_gpu_verify.sh <the same arguments> --resume   # continue a run
#
# Every argument goes to basalt_gpu_verify.py unchanged (`--help` lists them). NINFER_PYTHON picks the
# interpreter (default python3). The results directory is resolved once, before the first step, and
# passed to every call, so a run that crosses midnight stays in one directory.
#
# No GPU lock here. The shared-GPU lock (CLAUDE.md, tools/gpu-lock.ps1) is a Windows file-share lock
# for the shared desktop, and no Linux bench script in this repository takes one: the Linux GPU box
# is a rented machine used by one session at a time (single-tenant). Do not run this on a Linux
# machine other jobs share without adding a lock around each `--action run` below. GPU tests run
# through plain `ctest -L gpu` for the same reason.
#
# Under WSL the card is the shared Windows 3090, whose lock a Linux process cannot take (share modes
# do not carry over to WSL), so this refuses to run there; the dry run still works. NINFER_PROC_VERSION
# names another file to read instead of /proc/version (for the self-test).
set -uo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${NINFER_PYTHON:-python3}"
core="$here/basalt_gpu_verify.py"

for arg in "$@"; do
  case "$arg" in
    --dry-run|--help|-h) exec "$python" "$core" "$@" ;;
  esac
done

if grep -qi microsoft "${NINFER_PROC_VERSION:-/proc/version}" 2>/dev/null; then
  printf '%s\n' \
    'basalt_gpu_verify.sh: this is WSL. The RTX 3090 here is the shared Windows card, and its GPU lock' \
    '(%TEMP%\ninfer-gpu.lock, CLAUDE.md "GPU lock") can only be held from Windows. Run the Windows' \
    'script instead, from PowerShell: pwsh tools/bench/basalt_gpu_verify.ps1 <the same arguments>' >&2
  exit 2
fi

if ! out="$("$python" "$core" "$@" --action out)"; then
  printf 'basalt_gpu_verify.py --action out failed; run it with --dry-run to see why.\n' >&2
  exit 1
fi
set -- "$@" --out "$out"

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
