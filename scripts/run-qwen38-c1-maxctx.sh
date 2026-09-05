#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Qwen3.8-27B on one RTX 3090, tuned for context and prefix reuse rather than for caution.
#
# run-qwen38-c1.sh serves 65,536 tokens of INT8 and leaves 2.85 GiB of the card unused. This
# profile spends it: rk8v4 KV, MTP3 speculation plus the draft head, and the tuned context cache.
#
# WHAT rk8v4 BUYS. The same KV that holds 171,648 INT8 tokens holds 226,560 rk8v4 tokens - +33%
# context for +0.082% perplexity. It is opt-in precisely because INT8 is the quality default.
#
# CONTEXT CACHE. A checkpoint is a KV prefix plus a StateImage, and on this model the StateImage
# is 147 MiB flat regardless of prefix length - 48 GDN layers of 128x128x48 FP32 recurrent state
# plus conv. That is 2.4x the 35B-A3B's 61.4 MiB, so the slots are correspondingly expensive:
# --host-state-slots 32 pins 4.59 GiB of host memory. It is host memory, not device, and it is
# what takes prefix reuse from 8.4% to 98.3% on a multi-preamble workload.
#
# --auto-prefix-grid lets two callers whose prompts merely start alike share a cached prefix with
# no client hint. A grid point is only materialised once two independent callers have both asked
# for it, so it cannot waste a slot speculatively.
#
# MEASURED, with a Windows desktop running (a headless box has roughly 1.5 GiB more to spend):
#
#   lanes  KV      context   runtime    free after startup
#   ---------------------------------------------------------
#   1      int8     65,536   2.73 GiB   2.85 GiB   <- what run-qwen38-c1.sh does
#   1      rk8v4   131,072   3.93 GiB   1.68 GiB
#   1      rk8v4   163,840   4.77 GiB   854.3 MiB
#   2      rk8v4   131,072   4.31 GiB   1.38 GiB   <- default here
#
# Text only: --vision costs context, and run-qwen38-vision.sh already covers image work at 32K.
# Rungs if startup refuses: 163840 / 131072 / 114688 / 98304 / 65536.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

MODEL="${NINFER_MODEL:-${NINFER_MODEL_DIR:-/mnt/c/Ninefer-3090/models}/qwen3_8_27b.ninfer}"
CONTEXT="${NINFER_CONTEXT:-131072}"
CONCURRENCY="${NINFER_CONCURRENCY:-2}"
KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"
KV_DTYPE="${NINFER_KV_DTYPE:-rk8v4}"
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8080}"

SPEC="${NINFER_SPEC:-mtp}"
case "$SPEC" in
  mtp)  spec_args=(--spec mtp --draft-tokens "${NINFER_DRAFT_TOKENS:-3}" --lm-head-draft) ;;
  none) spec_args=() ;;
  *) printf 'NINFER_SPEC must be mtp or none, got %s\n' "$SPEC" >&2; exit 2 ;;
esac

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="${NINFER_SERVER:-$root/build-linux/apps/ninfer-serve}"
[[ -x "$server" ]] || server="$root/ninfer-serve"

if [[ ! -x "$server" ]]; then
  printf 'Missing ninfer-serve (looked for %s)\n' "$server" >&2
  printf 'Build it first:  ./scripts/build.sh\n' >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Missing model: %s\n' "$MODEL" >&2
  printf 'Download it first:  ./download-qwen38.sh\n' >&2
  exit 1
fi

if [[ "$SPEC" == 'mtp' ]]; then spec_label='MTP3 + draft head'; else spec_label='no speculation'; fi
printf 'Qwen3.8-27B  |  C%s  |  context %s  |  KV pool %s  |  %s  |  %s  |  text only\n' \
  "$CONCURRENCY" "$CONTEXT" "$KV_CAPACITY" "$KV_DTYPE" "$spec_label"
printf 'Cache: 8 shared / 8 private / 32 host states  |  automatic prefix grid on\n'
printf 'API: http://%s:%s/v1\n\n' "$HOST" "$PORT"

exec "$server" "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --max-concurrency "$CONCURRENCY" \
  --max-context "$CONTEXT" \
  --kv-capacity "$KV_CAPACITY" \
  --kv-dtype "$KV_DTYPE" \
  "${spec_args[@]}" \
  --prefill-chunk 1024 \
  --max-pending-requests 16 --pending-timeout-ms 600000 \
  --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 \
  --auto-prefix-grid
