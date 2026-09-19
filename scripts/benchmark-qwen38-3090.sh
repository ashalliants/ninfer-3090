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
# Hands wide prefill GEMMs to cuBLAS: about 1.73x prefill for +0.156% perplexity (4.343155 ->
# 4.349944 on the 1M corpus). Set to 0 to measure the default-quality engine instead. The chunk
# follows it, because the route only amortises its weight-sized dequantise over a call's tokens,
# and at this sweep's usual 512 it is a loss.
# This sweep stays on MTP3. DFlash2 at K=7 is much faster single-stream -- 172.3 tok/s against
# MTP3's 124.3 through the CLI -- but measured through the serve path at the cohort levels this
# sweep runs, it wins only at C1: aggregate tok/s C1/C2/C4 of 97.7/147.5/187.8 against MTP3's
# 91.6/153.3/223.1, and at C8 it fails to start because the draft model's extra weights leave too
# little for the runtime reservation. Batching already amortises the weight sweep that speculation
# exploits, so the extra columns become pure cost as concurrency rises.
# Set NINFER_BENCH_SPEC=dflash2 to measure it anyway, and expect a win only at C1.
SPEC="${NINFER_BENCH_SPEC:-mtp}"
DRAFT_TOKENS="${NINFER_BENCH_DRAFT_TOKENS:-}"
PREFILL_CUBLAS="${NINFER_BENCH_PREFILL_CUBLAS:-1}"
PREFILL_CHUNK="${NINFER_BENCH_PREFILL_CHUNK:-}"
# ==================================================================

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$SPEC" == 'dflash2' ]]; then default_model='qwen3_8_27b_dflash2.ninfer'; else default_model='qwen3_8_27b.ninfer'; fi
model="${NINFER_BENCH_MODEL:-${NINFER_MODEL_DIR:-$repo/..}/$default_model}"
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
printf '  Speculation    : %s%s\n' "$SPEC" "${DRAFT_TOKENS:+ K=$DRAFT_TOKENS}"
if [[ "$PREFILL_CUBLAS" != '0' ]]; then
  printf '  Prefill route  : cuBLAS, +0.156%% perplexity (NINFER_BENCH_PREFILL_CUBLAS=0 for the default engine)\n'
else
  printf '  Prefill route  : default integer-activation\n'
fi
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
NINFER_BENCH_SPEC="$SPEC" \
${DRAFT_TOKENS:+NINFER_BENCH_DRAFT_TOKENS="$DRAFT_TOKENS"} \
NINFER_BENCH_PREFILL_CUBLAS="$PREFILL_CUBLAS" \
${PREFILL_CHUNK:+NINFER_BENCH_PREFILL_CHUNK="$PREFILL_CHUNK"} \
  uv run tools/bench/run_qwen38_windows_3090_benchmarks.py

printf '\nBENCHMARK COMPLETE. Open the results directory printed above.\n'
