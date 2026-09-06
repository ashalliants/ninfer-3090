#!/usr/bin/env bash
# Runs ON the rented box, piped in over ssh. Nothing is edited there; this is a local file.
#
# Two questions:
#   1. How much context can we actually hold at a given concurrency, in each mode?
#   2. Under that load, does it still generate at a usable rate, or did the crossings cripple it?
#
# Aggregate tok/s under concurrent load is the number that matters -- single-stream decode is the
# worst case for an offload design, because the per-forward-pass crossing cost is paid by one
# token instead of being shared across a batch.
set -uo pipefail

MODEL=/root/models/qwen3_6_35b_a3b.ninfer
SERVE=/root/build/apps/ninfer-serve
PORT=18080

start_server() {   # devices concurrency context
  timeout 900 "$SERVE" "$MODEL" --devices "$1" --max-context "$3" --kv-capacity auto \
    --kv-dtype int8 --max-concurrency "$2" --host 127.0.0.1 --port "$PORT" \
    > /root/s.log 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    grep -qE "listening on|startup failed|FATAL|^usage:" /root/s.log && break
    sleep 2
  done
  grep -q "listening on" /root/s.log
}

stop_server() {
  # Kill the server itself, not just the `timeout` wrapper around it. Killing the wrapper leaves
  # ninfer-serve alive holding its VRAM, and the next probe then fails to fit 2 GiB of weights on a
  # 24 GiB card for no visible reason.
  kill "${SERVER_PID:-0}" 2>/dev/null
  pkill -f "$SERVE" 2>/dev/null
  wait 2>/dev/null
  for _ in $(seq 1 30); do
    pgrep -f "$SERVE" >/dev/null || break
    sleep 1
  done
  sleep 2
}

capacity_line() { grep -E "capacity \| KV" /root/s.log | head -1 | sed 's/.*capacity | //'; }
fail_line()     { grep -m1 -E "startup failed|FATAL|^error:" /root/s.log | cut -c1-110; }

probe() {          # devices concurrency context
  if start_server "$@"; then
    printf "  devices=%-4s C=%-3s ctx=%-7s -> %s\n" "$1" "$2" "$3" "$(capacity_line)"
    stop_server
    return 0
  fi
  printf "  devices=%-4s C=%-3s ctx=%-7s -> FAILED: %s\n" "$1" "$2" "$3" "$(fail_line)"
  stop_server
  return 1
}

# One request generating $2 tokens; prints the token count when it completes.
fire() {
  curl -s --max-time 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.6-35b-a3b\",\"max_tokens\":$2,\"temperature\":0,
         \"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
    | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
    print(d.get('usage', {}).get('completion_tokens', 0))
except Exception:
    print(0)
"
}

bench() {          # devices concurrency context tokens_each
  local devices=$1 conc=$2 ctx=$3 tokens=$4
  if ! start_server "$devices" "$conc" "$ctx"; then
    printf "  devices=%-4s C=%-3s -> FAILED: %s\n" "$devices" "$conc" "$(fail_line)"
    stop_server
    return 1
  fi
  local cap; cap=$(capacity_line)

  # Warm once so the first request's graph/allocation work is not charged to the measurement.
  fire "Say OK." 4 >/dev/null

  local start end total=0
  start=$(date +%s.%N)
  local pids=() outs=()
  for i in $(seq 1 "$conc"); do
    outs+=("/root/r$i.txt")
    fire "Write a detailed paragraph about the number $i and its mathematical properties." \
      "$tokens" > "/root/r$i.txt" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  end=$(date +%s.%N)

  for f in "${outs[@]}"; do total=$(( total + $(cat "$f" 2>/dev/null || echo 0) )); done
  local elapsed; elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")
  local rate;    rate=$(python3 -c "print(f'{$total / max($end - $start, 0.001):.1f}')")
  printf "  devices=%-4s C=%-3s ctx=%-7s | %s\n" "$devices" "$conc" "$ctx" "$cap"
  printf "        %s tokens in %ss = %s tok/s aggregate\n" "$total" "$elapsed" "$rate"
  stop_server
}

case "${1:-all}" in
ceiling)
  echo "=== context ceiling, single GPU ==="
  for c in "1 262144" "1 131072" "2 131072" "4 65536"; do probe 0 $c; done
  # 8 is kMaximumConcurrency; higher is refused at argument parsing, not for want of memory.
  echo "=== context ceiling, expert offload ==="
  for c in "8 262144" "8 131072" "8 65536" "4 262144" "2 262144"; do probe 0,1 $c; done
  ;;
throughput)
  echo "=== aggregate throughput ==="
  bench 0   1 32768 80
  bench 0,1 1 32768 80
  bench 0,1 4 32768 80
  bench 0,1 8 32768 80
  ;;
*)
  bash "$0" ceiling
  bash "$0" throughput
  ;;
esac
