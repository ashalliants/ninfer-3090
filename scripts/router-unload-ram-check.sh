#!/usr/bin/env bash
# Checks what a router model switch leaves behind, on a real GPU:
#   1. the router's RAM with no model loaded returns to about where it started, and
#   2. a model with a context store saves its conversation when it is unloaded, and reuses it when
#      it loads again.
# It starts its own router from a models preset, sends requests, switches models, and stops the
# router (by its PID) at the end. Run it with the GPU free and the lock held, e.g.
#   flock ~/.cache/ninfer/gpu.lock scripts/router-unload-ram-check.sh \
#     --serve build-v100/apps/ninfer-serve --preset ~/ninfer-router/models.ini \
#     --first Qwen3.6-35B-A3B --second Inkling-2-Mini
# --first should have `context-store` set in the preset for check 2 (it is skipped otherwise).
# Options: --port N (default 18089), --slack-gib N (RAM allowed above the start, default 1.0),
#          --prompt-words N (size of the conversation to build, default 3000).
set -euo pipefail

SERVE="" PRESET="" FIRST="" SECOND="" PORT=18089 SLACK=1.0 WORDS=3000
while [ $# -gt 0 ]; do
  case "$1" in
    --serve) SERVE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --first) FIRST="$2"; shift 2 ;;
    --second) SECOND="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --slack-gib) SLACK="$2"; shift 2 ;;
    --prompt-words) WORDS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -x "$SERVE" ] && [ -f "$PRESET" ] && [ -n "$FIRST" ] && [ -n "$SECOND" ] || {
  sed -n '2,14p' "$0"; exit 2; }

BASE="http://127.0.0.1:$PORT"
LOG="$(mktemp -t router-unload-check.XXXXXX.log)"
"$SERVE" --models-preset "$PRESET" --models-max 1 --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
stop() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap stop EXIT
echo "router pid $PID, log $LOG"

for _ in $(seq 120); do
  curl -fsS "$BASE/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "router exited"; tail -20 "$LOG"; exit 1; }
  sleep 1
done

rss_gib() { awk '/^VmRSS/ {printf "%.2f", $2 / 1048576}' "/proc/$PID/status"; }
PROMPT="$(python3 -c "import sys; print(' '.join(f'note {i} about the hotel booking' for i in range(int(sys.argv[1]) // 6)))" "$WORDS")"
ask() { # model, prompt -> prints the response JSON
  python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 8, "temperature": 0,
      "messages": [{"role": "user", "content": sys.argv[2] + "\nReply with OK."}]}))' "$1" "$2" |
    curl -fsS -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
cached() { python3 -c 'import json,sys; u=json.load(sys.stdin).get("usage",{}); print((u.get("prompt_tokens_details") or {}).get("cached_tokens",0), u.get("prompt_tokens",0))'; }

START=$(rss_gib); echo "RAM at start, nothing loaded: $START GiB"
read -r C1 P1 < <(ask "$FIRST" "$PROMPT" | cached); echo "$FIRST: $P1 prompt tokens, $C1 cached"
LOADED=$(rss_gib); echo "RAM with $FIRST loaded: $LOADED GiB"
ask "$SECOND" "hello" >/dev/null; echo "switched to $SECOND"
grep -E "unloaded '$FIRST'" "$LOG" | tail -1 || echo "(no unload line for $FIRST)"
curl -fsS "$BASE/models/unload" -H 'Content-Type: application/json' -d "{\"model\": \"$SECOND\"}" >/dev/null
for _ in $(seq 120); do grep -q "unloaded '$SECOND'" "$LOG" && break; sleep 1; done
grep -E "unloaded '$SECOND'" "$LOG" | tail -1
EMPTY=$(rss_gib); echo "RAM after both unloaded: $EMPTY GiB"

fail=0
if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' "$EMPTY" "$START" "$SLACK"; then
  echo "PASS: RAM returned to within $SLACK GiB of the start ($START -> $EMPTY GiB)"
else
  echo "FAIL: RAM after unloading is $EMPTY GiB, more than $SLACK GiB above the start ($START GiB)"; fail=1
fi

if grep -q "unloaded '$FIRST' | context store:" "$LOG"; then
  if grep "unloaded '$FIRST' | context store:" "$LOG" | tail -1 | grep -qE "context store: [1-9][0-9]* saved"; then
    echo "PASS: $FIRST saved its conversation when unloaded"
  else
    echo "FAIL: $FIRST saved nothing when unloaded"; fail=1
  fi
  read -r C2 P2 < <(ask "$FIRST" "$PROMPT" | cached)
  echo "$FIRST again: $P2 prompt tokens, $C2 cached"
  if [ "$C2" -gt 0 ]; then echo "PASS: the reload reused the saved conversation"; else echo "FAIL: the reload recomputed the prompt"; fail=1; fi
else
  echo "SKIP: $FIRST has no context store (set context-store in the preset)"
fi
exit "$fail"
