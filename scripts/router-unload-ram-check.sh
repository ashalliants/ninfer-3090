#!/usr/bin/env bash
# Checks what a router model switch leaves behind, on a real GPU:
#   1. the router's RAM with no model loaded does not grow from one model switch to the next: a
#      second round of switches must end within --slack-gib of the first round's end (what the
#      first round keeps for good, against the empty start, is reported but not judged), and
#   2. a model with a context store saves its conversation when it is unloaded, and reuses it when
#      it loads again.
# It starts its own router from a models preset, sends requests, switches models, and stops the
# router (by its PID) at the end. Run it with the GPU free and the lock held, e.g.
#   flock ~/.cache/ninfer/gpu.lock scripts/router-unload-ram-check.sh \
#     --serve build-v100/apps/ninfer-serve --preset ~/ninfer-router/models.ini \
#     --first Qwen3.6-35B-A3B --second Inkling-2-Mini
#   3. each unload line says how much heap is still in use once the model is gone (the memory no
#      trim can return), and the router's swap is reported beside its RAM;
#   4. with --tight-ram (Qwen3.8-Flash-Next as --first): a start whose RAM ledger cannot spare the
#      context cache's 512 MiB minimum still starts, with the context store off for that run and
#      one warning, instead of failing after loading its weights. The router is restarted with
#      --first's ram-headroom-mib raised so that about 0.25 GiB is left for the cache, computed from
#      the "RAM ledger:" line of --first's first load.
# --first should have `context-store` set in the preset for checks 2 and 4 (skipped otherwise).
# Options: --port N (default 18089), --slack-gib N (RAM growth allowed per round, default 1.0),
#          --prompt-words N (size of the conversation to build, default 3000), --tight-ram.
set -euo pipefail

SERVE="" PRESET="" FIRST="" SECOND="" PORT=18089 SLACK=1.0 WORDS=3000 TIGHT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --serve) SERVE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --first) FIRST="$2"; shift 2 ;;
    --second) SECOND="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --slack-gib) SLACK="$2"; shift 2 ;;
    --prompt-words) WORDS="$2"; shift 2 ;;
    --tight-ram) TIGHT=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -x "$SERVE" ] && [ -f "$PRESET" ] && [ -n "$FIRST" ] && [ -n "$SECOND" ] || {
  sed -n '2,22p' "$0"; exit 2; }

BASE="http://127.0.0.1:$PORT"
PID=""
stop() { [ -n "$PID" ] || return 0; kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; PID=""; }
trap stop EXIT
start_router() { # preset [may-fail]: start a router on it and wait until its startup loads are done
  # With may-fail, a router that exits during startup returns 1 instead of ending the check.
  LOG="$(mktemp -t router-unload-check.XXXXXX.log)"
  "$SERVE" --models-preset "$1" --models-max 1 --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
  PID=$!
  echo "router pid $PID, log $LOG"
  # /health answers before the router has loaded its preset's startup models: wait for the
  # "listening on" line, which the router logs once those loads are done.
  for _ in $(seq 1800); do
    grep -q "listening on http" "$LOG" && return 0
    if ! kill -0 "$PID" 2>/dev/null; then
      wait "$PID" 2>/dev/null || true; PID=""
      [ -n "${2:-}" ] && return 1
      echo "router exited"; tail -20 "$LOG"; exit 1
    fi
    sleep 1
  done
  echo "FAIL: the router never finished starting"; exit 1
}
start_router "$PRESET"

rss_gib() { awk '/^VmRSS/ {printf "%.2f", $2 / 1048576}' "/proc/$PID/status"; }
swap_gib() { awk '/^VmSwap/ {printf "%.2f", $2 / 1048576}' "/proc/$PID/status"; }
PROMPT="$(python3 -c "import sys; print(' '.join(f'note {i} about the hotel booking' for i in range(int(sys.argv[1]) // 6)))" "$WORDS")"
ask() { # model, prompt -> prints the response JSON
  python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 8, "temperature": 0,
      "messages": [{"role": "user", "content": sys.argv[2] + "\nReply with OK."}]}))' "$1" "$2" |
    curl -fsS -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
cached() { python3 -c 'import json,sys; u=json.load(sys.stdin).get("usage",{}); print((u.get("prompt_tokens_details") or {}).get("cached_tokens",0), u.get("prompt_tokens",0))'; }

# Model states from GET /models, "id=state" per line (unloaded, loading, loaded, sleeping,
# unloading). Read into a variable and matched with here-strings: a `states | grep -q` pipeline
# fails under pipefail when grep exits early, which read as "no match".
states() { curl -fsS "$BASE/models" | python3 -c 'import json,sys
for m in json.load(sys.stdin)["data"]: print(m["id"] + "=" + m["status"]["value"])'; }
resident() { local st; st="$(states)"; grep -v "=unloaded$" <<<"$st" || true; }
unload_and_wait() { # model: unload it and wait until GET /models says it is unloaded
  curl -fsS "$BASE/models/unload" -H 'Content-Type: application/json' -d "{\"model\": \"$1\"}" >/dev/null
  local st
  for _ in $(seq 300); do
    st="$(states)"
    grep -qFx "$1=unloaded" <<<"$st" && return 0
    sleep 1
  done
  echo "FAIL: $1 did not unload"; grep "^$1=" <<<"$st"; exit 1
}
log_since() { tail -n +"$(($1 + 1))" "$LOG"; } # log lines after line $1

# The baseline is the router with nothing loaded: unload whatever startup left resident, then
# refuse to measure while anything is still there.
for m in $(resident | cut -d= -f1); do
  echo "unloading $m (resident after startup) before the baseline"
  unload_and_wait "$m"
done
if [ -n "$(resident)" ]; then
  echo "FAIL: a model is still loaded, so there is no empty baseline to measure:"; resident; exit 1
fi
sleep 2
START=$(rss_gib); echo "RAM at start, nothing loaded (checked via GET /models): $START GiB"
read -r C1 P1 < <(ask "$FIRST" "$PROMPT" | cached); echo "$FIRST: $P1 prompt tokens, $C1 cached"
LOADED=$(rss_gib); echo "RAM with $FIRST loaded: $LOADED GiB"
SWITCH=$(wc -l <"$LOG") # only log lines after this belong to the switch
ask "$SECOND" "hello" >/dev/null; echo "switched to $SECOND"
FIRST_UNLOAD="$(log_since "$SWITCH" | grep -F "unloaded '$FIRST'" | tail -1 || true)"
echo "${FIRST_UNLOAD:-(no unload line for $FIRST)}"
SECOND_AT=$(wc -l <"$LOG")
unload_and_wait "$SECOND"
log_since "$SECOND_AT" | grep -F "unloaded '$SECOND'" | tail -1 || true
if [ -n "$(resident)" ]; then echo "FAIL: models still loaded after the unloads:"; resident; exit 1; fi
EMPTY=$(rss_gib); echo "RAM after both unloaded (checked via GET /models): $EMPTY GiB (swap $(swap_gib) GiB)"

echo "kept by the first round of switches: $(python3 -c 'import sys; print(f"{float(sys.argv[1]) - float(sys.argv[2]):.2f}")' "$EMPTY" "$START") GiB above the empty start"
fail=0

if grep -qF "of heap still in use" <<<"$FIRST_UNLOAD"; then
  echo "PASS: the unload line says how much heap is still in use"
else
  echo "FAIL: the unload line has no heap-in-use figure"; fail=1
fi

if grep -qF "| context store:" <<<"$FIRST_UNLOAD"; then
  if grep -qE "context store: [1-9][0-9]* saved" <<<"$FIRST_UNLOAD"; then
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

# Check 1: a second round of switches (back to --first above, now to --second and unloaded) must
# not leave more behind than the first did: growth here is what builds up over a day of switching.
ask "$SECOND" "hello" >/dev/null
unload_and_wait "$SECOND"
if [ -n "$(resident)" ]; then echo "FAIL: models still loaded after the second round:"; resident; exit 1; fi
EMPTY2=$(rss_gib); echo "RAM after the second round (checked via GET /models): $EMPTY2 GiB (swap $(swap_gib) GiB)"
if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) + float(sys.argv[3]) else 1)' "$EMPTY2" "$EMPTY" "$SLACK"; then
  echo "PASS: RAM did not grow by more than $SLACK GiB over a second round of switches ($EMPTY -> $EMPTY2 GiB)"
else
  echo "FAIL: RAM grew from $EMPTY to $EMPTY2 GiB over a second round of switches (more than $SLACK GiB)"; fail=1
fi

# Check 4: a start whose RAM ledger cannot spare the context cache falls back instead of failing.
if [ "$TIGHT" = 1 ]; then
  LEDGER="$(grep -F "RAM ledger: available" "$LOG" | head -1 || true)"
  if [ -z "$LEDGER" ] || ! grep -qF "| context store:" <<<"$FIRST_UNLOAD"; then
    echo "SKIP: --tight-ram needs a Qwen3.8-Flash-Next --first with a context store (no RAM ledger line or store)"
  else
    # What the cache could have had: the ledger's available memory less every term planned before it.
    # "caches" (the n-gram RAM blocks) is left out: it is sized last, from what the cache leaves, so
    # it shrinks to make room instead of squeezing the cache.
    HEADROOM_MIB="$(python3 -c 'import re,sys
t=sys.argv[1]
def g(pat): return float(re.search(pat, t).group(1))
avail=min(g(r"available ([0-9.]+)"), g(r"commit ([0-9.]+)"))
floor=sum(g(p) for p in (r"reserve ([0-9.]+)", r"other pins ([0-9.]+)", r"later pins ([0-9.]+)",
                         r"staging ([0-9.]+)", r"growth ([0-9.]+)", r"GiB of ([0-9.]+) needed"))
spare=avail-floor
print(int((g(r"reserve ([0-9.]+)") + spare - 0.25) * 1024))' "$LEDGER")"
    TIGHT_PRESET="$(mktemp -t router-unload-check.XXXXXX.ini)"
    write_tight_preset() { python3 -c 'import sys
lines=open(sys.argv[1]).read().splitlines(); out=[]; inside=False
for line in lines:
    if line.startswith("["):
        if inside: out.append("ram-headroom-mib = " + sys.argv[3])
        inside = line.strip() == "[" + sys.argv[2] + "]"
    if inside and line.split("=")[0].strip() == "ram-headroom-mib": continue
    out.append(line)
if inside: out.append("ram-headroom-mib = " + sys.argv[3])
open(sys.argv[4], "w").write("\n".join(out) + "\n")' "$PRESET" "$FIRST" "$HEADROOM_MIB" "$TIGHT_PRESET"; }
    write_tight_preset
    stop
    echo "tight start: ram-headroom-mib $HEADROOM_MIB for $FIRST (about 0.25 GiB left for its context cache)"
    if ! start_router "$TIGHT_PRESET" may-fail; then
      # Part of "caches" is fixed and part is the n-gram share, sized after the context cache, so
      # the estimate can overshoot. The refusal says how far short the experts are: back off by
      # that plus 0.1 GiB and start once more.
      SHORT="$(grep -m1 -oE '[0-9.]+ GiB short' "$LOG" | cut -d' ' -f1)"
      [ -n "$SHORT" ] || { echo "FAIL: the tight start failed for another reason:"; grep -E "ERROR" "$LOG" | tail -3; exit 1; }
      HEADROOM_MIB="$(python3 -c 'import sys; print(int(sys.argv[1]) - int((float(sys.argv[2]) + 0.1) * 1024))' "$HEADROOM_MIB" "$SHORT")"
      echo "tight start: experts $SHORT GiB short; retrying with ram-headroom-mib $HEADROOM_MIB"
      write_tight_preset
      start_router "$TIGHT_PRESET"
    fi
    TIGHT_AT=$(wc -l <"$LOG")
    if ask "$FIRST" "hello" >/dev/null; then
      if log_since "$TIGHT_AT" | grep -qF "context store: off for this run" || grep -qF "context store: off for this run" "$LOG"; then
        echo "PASS: the tight start served a request with the context store off for this run"
        grep -F "context store: off for this run" "$LOG" | head -1 | cut -c1-240
      elif grep -qF "context cache: off" "$LOG"; then
        echo "FAIL: the cache was off but the store was not reported off"; fail=1
      else
        echo "INCONCLUSIVE: the cache still fitted (available RAM moved between starts); $(grep -F "context cache:" "$LOG" | head -1)"
      fi
    else
      echo "FAIL: the tight start did not serve a request:"; grep -E "ERROR|failed to load" "$LOG" | tail -3; fail=1
    fi
    rm -f "$TIGHT_PRESET"
  fi
fi
exit "$fail"
