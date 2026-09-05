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
#   lanes  KV      context   vision   runtime    free after startup
#   ------------------------------------------------------------------
#   1      int8     65,536   off      2.73 GiB   2.85 GiB   <- what run-qwen38-c1.sh does
#   1      rk8v4    49,152   overlay  1.82 GiB   3.33 GiB
#   1      rk8v4    98,304   overlay  3.09 GiB   2.07 GiB
#   1      rk8v4   131,072   off      3.93 GiB   1.68 GiB
#   1      rk8v4   131,072   overlay  3.94 GiB   1.59 GiB
#   2      rk8v4   131,072   overlay  4.32 GiB   1.26 GiB
#   1      rk8v4   163,840   overlay  4.78 GiB   763.2 MiB
#
# Vision is on: overlay residency costs about 10 MiB of runtime reservation, so there is no reason
# to trade it away. run-qwen38-vision.sh remains for the plain 32K image profile.
#
# WHY NOT 262,144 LIKE THE 35B-A3B. The runtime reservation is dead linear in context - seven
# points from 49,152 to 163,840 fit
#
#     runtime_bytes = 0.553 GiB + 27,719 x context        (worst residual 3.9 MiB)
#
# at 27.07 KiB/token, and a second lane adds a flat 0.38 GiB. A headless 3090 has about 7.06 GiB
# for the reservation, so 262,144 needs 7.32 GiB at one lane and 7.70 GiB at two - it does not fit
# either way. The zero-margin ceilings are roughly 252,000 tokens at C1 and 237,000 at C2.
#
# That is not a tuning failure, it is the model: the 27B spends 16 full-attention layers x 4
# kv_heads x 256 head_dim per token against the 35B-A3B's 10 x 2 x 256, which is 3.2x the KV per
# token - 27.07 KiB against roughly 7.8. The 35B-A3B reaches 262,144 because its KV is cheap.
#
# The default below is 212,992: 6.43 GiB predicted at two lanes, leaving +0.63 GiB. 196,608 is the
# more cautious rung at +1.05 GiB. Both are extrapolated rather than measured - this machine cannot
# start either - so treat the first headless start as the confirmation and drop a rung if it
# refuses. Rungs: 229376 / 212992 / 196608 / 163840 / 131072 / 114688 / 98304 / 65536.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

MODEL="${NINFER_MODEL:-${NINFER_MODEL_DIR:-/mnt/c/Ninefer-3090/models}/qwen3_8_27b.ninfer}"
CONTEXT="${NINFER_CONTEXT:-212992}"
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

# Vision stays on. Overlay residency keeps the tower host-pinned and streams each image through a
# borrowed device window, so it costs essentially nothing resident: measured at 131,072 the runtime
# reservation goes 3.93 -> 3.94 GiB with it enabled, about 10 MiB.
VISION="${NINFER_VISION:-on}"
case "$VISION" in
  on)  vision_args=(--vision --vision-residency "${NINFER_VISION_RESIDENCY:-overlay}") ;;
  off) vision_args=() ;;
  *) printf 'NINFER_VISION must be on or off, got %s\n' "$VISION" >&2; exit 2 ;;
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
if [[ "$VISION" == 'on' ]]; then vision_label='vision (overlay)'; else vision_label='text only'; fi
printf 'Qwen3.8-27B  |  C%s  |  context %s  |  KV pool %s  |  %s  |  %s  |  %s\n' \
  "$CONCURRENCY" "$CONTEXT" "$KV_CAPACITY" "$KV_DTYPE" "$spec_label" "$vision_label"
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
  "${vision_args[@]}" \
  --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 \
  --auto-prefix-grid
