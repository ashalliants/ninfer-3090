#!/usr/bin/env bash
# Linux counterpart of benchmark-qwen38-3090.bat: the Qwen3.8-27B cohort sweep on one RTX 3090.
set -euo pipefail

# ======================== EDITABLE SETTINGS ========================
MAX_CONTEXT="${NINFER_BENCH_MAX_CONTEXT:-131072}"
OUTPUT_TOKENS="${NINFER_BENCH_OUTPUT_TOKENS:-1024}"
PREFILL_PROMPT_CHARACTERS="${NINFER_BENCH_PREFILL_CHARS:-28000}"
COHORTS="${NINFER_BENCH_COHORTS:-1,2,4,8}"
KV_DTYPE="${NINFER_BENCH_KV_DTYPE:-rk8v4}"
START_DELAY_SECONDS="${NINFER_BENCH_START_DELAY:-10}"
# ==================================================================

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${NINFER_BENCH_MODEL:-${NINFER_MODEL_DIR:-$repo/..}/qwen3_8_27b.ninfer}"
server="${NINFER_BENCH_SERVER:-$repo/build-linux/apps/ninfer-serve}"

if [[ ! -x "$server" ]]; then
  printf 'ERROR: Server not found: %s\n' "$server" >&2
  printf 'Build it first:  ./scripts/build.sh\n' >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'ERROR: Model not found: %s\n' "$model" >&2
  printf 'Download it first:  ./scripts/download-qwen38-27b.sh\n' >&2
  exit 1
fi
command -v uv >/dev/null || { printf 'ERROR: uv is not available in PATH.\n' >&2; exit 1; }

printf '\nRTX 3090 Qwen3.8 benchmark\n'
printf '  Shared context : %s tokens (C1 full; C8 capped at 8K per request)\n' "$MAX_CONTEXT"
printf '  Decode output  : %s tokens\n' "$OUTPUT_TOKENS"
printf '  Cohorts        : %s\n' "$COHORTS"
printf '  KV cache       : %s\n' "$KV_DTYPE"
printf '  Results        : %s/benchmark_results/linux_3090_*\n' "$repo"
if [[ "${KV_DTYPE,,}" == 'int8' && "$MAX_CONTEXT" -gt 65536 ]]; then
  printf 'WARNING: This high-context INT8 profile is not the recommended 3090 benchmark setting.\n'
fi
printf '\nStarting in %s seconds. Press Ctrl+C to cancel.\n' "$START_DELAY_SECONDS"
sleep "$START_DELAY_SECONDS"

cd -- "$repo"
NINFER_BENCH_SERVER="$server" \
NINFER_BENCH_MODEL="$model" \
NINFER_BENCH_MAX_CONTEXT="$MAX_CONTEXT" \
NINFER_BENCH_OUTPUT_TOKENS="$OUTPUT_TOKENS" \
NINFER_BENCH_PREFILL_CHARS="$PREFILL_PROMPT_CHARACTERS" \
NINFER_BENCH_COHORTS="$COHORTS" \
NINFER_BENCH_KV_DTYPE="$KV_DTYPE" \
  uv run tools/bench/run_qwen38_windows_3090_benchmarks.py

printf '\nBENCHMARK COMPLETE. Open the results directory printed above.\n'
