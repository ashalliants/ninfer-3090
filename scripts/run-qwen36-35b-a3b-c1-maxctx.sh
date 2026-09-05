#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Qwen3.6-35B-A3B on one RTX 3090 -- single user. Linux/WSL2 counterpart of
# run-qwen36-35b-a3b-c1-maxctx.bat; see that file for the full context ladder, the speculation
# memory accounting and the decode measurements. The essentials are repeated here.
#
# Maximum context on this machine is not a fixed number: the engine sizes its runtime reservation
# from whatever VRAM is free after the weights land, and under WSL the Windows desktop is still
# holding part of the card. The default below is sized with margin. If it fails to start, drop one
# rung: 114688 / 98304 / 90112 / 81920.
#
# CONTEXT CACHE. A checkpoint is a KV prefix plus a StateImage, and on this model the StateImage is
# 61.4 MiB *flat* regardless of prefix length -- 30 GDN layers of 128x128x32 FP32 recurrent state
# plus conv. Unlike KV pages, which several checkpoints of one conversation share, it cannot be
# shared between two frontiers at all: the recurrent state at token N is a function of every token
# before it. That fixed cost is why the shipped catalog is small.
#
# Measured, eight distinct ~1430-token preambles round-robined, reuse after the first round:
#
#   --max-shared-prefixes    reuse    what happens
#   ------------------------------------------------------------------------------------
#   4  (the old default)      8.4%    only one preamble ever stays cached; constant thrash
#   8                        98.3%    all eight stay; prefill 0.317 s -> 0.044 s
#   16                       98.3%    no better than 8 once the catalog fits the working set
#
# Sizing rule: --max-shared-prefixes should match the number of distinct preambles in play, and
# --host-state-slots roughly twice (shared + private). Raising them costs pinned HOST memory and
# nothing on the device -- measured on this profile, both settings resolve the full 114,688 tokens
# with byte-identical 492.4 MiB free after startup; the only change is host state pinned
# 982.6 MiB -> 1.92 GiB.
#
# --auto-prefix-grid offers shared candidates on a content-independent token grid, so two requests
# that merely start alike -- the same pasted document in two different chats, the same few-shot
# preamble inside one user message -- converge on the same frontier without any client hint. A grid
# point is only ever materialised once two independent callers have both asked for it, so it cannot
# waste a slot speculatively. Measured on a prompt with no structural boundary at all: 0% -> 82.6%
# reuse, prefill -71%, TTFT -64%, and the cold requests before it warms are unchanged.
#
# LONG CONVERSATIONS do not need any of the above -- they run on the private turn-closure path,
# which is what --max-private-continuations sizes. Measured with a 194-turn coding-agent loop
# (tool calls, file pastes, whole history resent each turn) growing to 64,668 tokens: exactly one
# cache miss, on turn 1. The limit on turn count is --max-context, not the cache.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

# Override with NINFER_MODEL=... to point at a copy on the Linux filesystem, which loads faster
# than reading 20.6 GiB across the 9p mount.
MODEL="${NINFER_MODEL:-/mnt/c/Ninefer-3090/models/qwen3_6_35b_a3b.ninfer}"

# Rungs: 81920 / 90112 / 98304 / 114688 / 131072.
CONTEXT="${NINFER_CONTEXT:-114688}"

# One lane by default. --kv-capacity is the shared pool and --max-context is the per-request cap,
# so two lanes do not need twice the memory unless you also want twice the per-request context.
# Measured on a 3090 with the desktop running (a headless box has ~1.5 GiB more to spend):
#
#   NINFER_CONCURRENCY  NINFER_CONTEXT  NINFER_KV_CAPACITY  runtime    free after startup
#   ---------------------------------------------------------------------------------------
#   1 (default)             114,688            114,688      1.22 GiB        492.4 MiB
#   2                        57,344            114,688      1.44 GiB        356.4 MiB
#   2                        65,536            131,072      1.57 GiB        201.3 MiB
#
# So for two agents at 64K each:  NINFER_CONCURRENCY=2 NINFER_CONTEXT=65536 \
#                                 NINFER_KV_CAPACITY=131072 ./run-qwen36-35b-a3b-c1-maxctx.sh
CONCURRENCY="${NINFER_CONCURRENCY:-1}"
KV_CAPACITY="${NINFER_KV_CAPACITY:-$CONTEXT}"

# 0.0.0.0 exposes an unauthenticated OpenAI-compatible endpoint to your whole LAN. Under WSL2 that
# is the WSL virtual network rather than the host LAN unless you have set up port forwarding.
HOST="${NINFER_HOST:-127.0.0.1}"
PORT="${NINFER_PORT:-8080}"

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="${NINFER_SERVER:-$root/build-linux/apps/ninfer-serve}"
[[ -x "$server" ]] || server="$root/ninfer-serve"

if [[ ! -x "$server" ]]; then
  printf 'Missing ninfer-serve (looked for %s)\n' "$server" >&2
  printf 'Build it first:  cmake --build "%s/build-linux"\n' "$root" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Missing model: %s\n' "$MODEL" >&2
  exit 1
fi

printf 'Qwen3.6-35B-A3B  |  C%s  |  context %s  |  KV pool %s  |  rk8v4  |  MTP3 + draft head  |  Vision (overlay)\n' \
  "$CONCURRENCY" "$CONTEXT" "$KV_CAPACITY"
printf 'Cache: 8 shared / 8 private / 32 host states  |  automatic prefix grid on\n'
printf 'API: http://%s:%s/v1\n\n' "$HOST" "$PORT"

exec "$server" "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --max-concurrency "$CONCURRENCY" \
  --max-context "$CONTEXT" \
  --kv-capacity "$KV_CAPACITY" \
  --kv-dtype rk8v4 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prefill-chunk 512 \
  --max-pending-requests 16 --pending-timeout-ms 600000 \
  --vision --vision-residency overlay \
  --max-private-continuations 8 --max-shared-prefixes 8 --host-state-slots 32 --host-kv-mib 8192 \
  --auto-prefix-grid
