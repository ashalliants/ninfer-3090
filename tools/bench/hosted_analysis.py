#!/usr/bin/env python3
"""Summarise `run_hosted_workload.py` runs: hit split, lost tokens by cause, TTFT, lanes, store.

    python tools/bench/hosted_analysis.py runs/c4-h8 runs/c8-h8 [--markdown] [--json out.json]

Per run it reports:

- **Hit split** per request. `private` reused at least the conversation's own last reusable point
  (minus `--tolerance` tokens); `shared` reused at least the product prompt but not the private
  point; `partial` reused something below both; `none` reused under 64 tokens. First requests of
  a conversation can only be `shared` or `none`.
- **Lost tokens**: `ideal - cached` summed over requests, where `ideal` is what a cache that kept
  every completed reusable point forever would have reused (estimated from server usage: previous
  prompt + reply, or prompt only with thinking on). Each request's loss is given one cause:

  | cause | requests | meaning |
  |---|---|---|
  | `diverged` | edit, branch | the reusable point is an older turn's endpoint, superseded by later turns |
  | `shared_miss` | first | the product prompt was not reused although an earlier request completed it |
  | `evicted_capacity` | append, fanout | own endpoint not reused, and an LRU of the run's Host + Device KV bytes would also have dropped it |
  | `evicted_host_capacity` | append, fanout | that LRU would hold it, an LRU of the Host bytes alone would not (it needs Device retention) |
  | `evicted_policy` | append, fanout | own endpoint not reused, but even the Host-only LRU would still hold it |
  | `store_miss` | resume | after a restart with a context store |

  Separately, `sibling_ceiling` sums tokens a concurrent sibling was prefilling at the time
  (fan-out children past the first): what single-flight at the divergence could add.
  The LRU is a ceiling model, not the engine: it holds one image per conversation of
  `tokens * --kv-bytes-per-token + --state-bytes`, in `host_context_bytes + kv_capacity_tokens *
  kv-bytes-per-token` of room, evicting least recently used.
- **TTFT** (client, retries after 429 included) and server queue wait / binding / restore
  (from the joined request log), p50/p95/p99.
- **Lanes**: mean `running / max_concurrency` and mean `waiting`, from `/v1/load` polls.
- **Counters** (from `/metrics` before and after): checkpoint vs root selections, pressure spill
  pages, preemptions, hydrations, store bytes written per hour, Host peak.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

MIB = 1024 * 1024


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metric(after: dict[str, float], before: dict[str, float], name: str) -> float:
    return sum(v - before.get(k, 0.0) for k, v in after.items()
               if k == name or k.startswith(name + "{"))


def lru_ceiling(records: list[dict[str, Any]], capacity: float, kv_bytes: float,
                state_bytes: float) -> set[int]:
    """Request ids whose private source an ideal LRU of `capacity` bytes would have dropped."""
    events = []
    for r in records:
        events.append((r["submit_s"], 0, r))
        if r["error"] is None:
            events.append((r["done_s"], 1, r))
    events.sort(key=lambda e: (e[0], e[1]))
    cache: OrderedDict[str, float] = OrderedDict()
    used = 0.0
    dropped: set[int] = set()
    for _, kind, r in events:
        key = r["conversation"]
        if kind == 0:
            if r["kind"] in ("append", "fanout", "resume") and r["private_ideal"] > 0:
                if key in cache:
                    cache.move_to_end(key)
                else:
                    dropped.add(r["request"])
            continue
        if r["kind"] == "fanout":
            continue  # children do not move the parent's endpoint
        size = (r["prompt_tokens"] + r["output_tokens"]) * kv_bytes + state_bytes
        used -= cache.pop(key, 0.0)
        cache[key] = size
        used += size
        while used > capacity and len(cache) > 1:
            _, freed = cache.popitem(last=False)
            used -= freed
    return dropped


def analyse(run: Path, tolerance: int, kv_bytes: float, state_bytes: float,
            warmup_s: float = 0.0) -> dict[str, Any]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (run / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line]
    server = json.loads((run / "server_records.json").read_text(encoding="utf-8"))
    loads = [json.loads(line) for line in
             (run / "load.jsonl").read_text(encoding="utf-8").splitlines() if line]
    loads = [entry for entry in loads if "requests" in entry]
    capacity = loads[0]["capacity"] if loads else {}
    lru_bytes = capacity.get("host_context_bytes", 0) + \
        capacity.get("kv_capacity_tokens", 0) * kv_bytes
    dropped = lru_ceiling(records, lru_bytes, kv_bytes, state_bytes)
    host_bytes = capacity.get("host_context_bytes", 0)
    dropped_host = lru_ceiling(records, host_bytes, kv_bytes, state_bytes)

    # The LRU model sees every request; the statistics only those submitted after warm-up.
    measured = [r for r in records if r["submit_s"] >= warmup_s]
    ok = [r for r in measured if r["error"] is None]
    split: dict[str, int] = defaultdict(int)
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lost: dict[str, float] = defaultdict(float)
    sibling = 0.0
    for r in ok:
        cached, ideal, shared = r["cached_tokens"], r["private_ideal"], r["shared_tokens"]
        private_point = ideal if r["kind"] != "first" else None
        if private_point and private_point > shared and cached >= private_point - tolerance:
            cls = "private"
        elif shared and cached >= shared - tolerance:
            cls = "shared"
        elif cached >= 64:
            cls = "partial"
        else:
            cls = "none"
        r["class"] = cls
        split[cls] += 1
        stats = by_kind[r["kind"]]
        stats["requests"] += 1
        stats["prompt"] += r["prompt_tokens"]
        stats["cached"] += cached
        stats["ideal"] += max(ideal, cached)
        stats[cls] += 1
        loss = max(0, ideal - cached)
        if loss > tolerance:
            if r["kind"] in ("edit", "branch"):
                cause = "diverged"
            elif r["kind"] == "first":
                cause = "shared_miss"
            elif r["kind"] == "resume":
                cause = "store_miss"
            elif r["request"] in dropped:
                cause = "evicted_capacity"
            elif r["request"] in dropped_host:
                cause = "evicted_host_capacity"
            else:
                cause = "evicted_policy"
            lost[cause] += loss
            lost[cause + "_requests"] += 1
            r["cause"] = cause
        extra = r["inflight_ideal"] - max(ideal, cached)
        if r["kind"] == "fanout" and extra > tolerance:
            sibling += extra

    ttft = [r["client_ttft_s"] for r in ok if r["client_ttft_s"] is not None]
    joined = [server[r["response_id"]] for r in ok if r.get("response_id") in server]
    queue = [s["engine_timing"]["queue_wait_seconds"] for s in joined]
    binding = [s["first_output_timing"]["initial_binding_seconds"] for s in joined
               if s.get("first_output_timing")]
    restore_s, restore_bytes = [], []
    fallback: dict[str, int] = defaultdict(int)
    source_wait = []
    for s in joined:
        fallback[s["generation"]["admission"]["fallback_reason"]] += 1
        source_wait.append(s["generation"]["admission"]["source_wait_seconds"])
        timing = s.get("first_output_timing") or {}
        transfers = timing.get("context_transfers") or {}
        h2d_bytes = sum(transfers.get(res, {}).get("h2d", {}).get("bytes", 0)
                        for res in ("state", "main_kv", "backend_kv"))
        if h2d_bytes:
            restore_bytes.append(h2d_bytes)
            restore_s.append(sum(transfers.get(res, {}).get("h2d", {}).get("seconds", 0.0)
                                 for res in ("state", "main_kv", "backend_kv")))
    concurrency = capacity.get("max_concurrency") or 1
    running = [entry["requests"]["running"] / concurrency for entry in loads]
    waiting = [entry["requests"]["waiting"] for entry in loads]
    prefilling = [entry["requests"]["prefilling"] for entry in loads]
    before, after = summary["metrics_before"], summary["metrics_after"]
    wall = summary["wall_s"]
    m = {name: metric(after, before, name) for name in (
        "ninfer_checkpoint_selections_total", "ninfer_root_selections_total",
        "ninfer_prompt_tokens_total", "ninfer_prompt_tokens_cached_total",
        "ninfer_prefill_tokens_total", "ninfer_prefill_seconds_total",
        "ninfer_decode_tokens_total", "ninfer_generation_tokens_total",
        "ninfer_decode_seconds_total", "ninfer_preemptions_total",
        "ninfer_replayed_tokens_total", "ninfer_context_pressure_spill_pages_total",
        "ninfer_context_store_writes_total", "ninfer_context_store_bytes_written_total",
        "ninfer_context_store_bytes_reused_total", "ninfer_context_store_dropped_total",
        "ninfer_context_store_hydrations_total", "ninfer_context_store_hydrated_tokens_total",
        "ninfer_context_store_hydration_failures_total",
        "ninfer_context_store_hydration_seconds_total", "ninfer_context_store_evicted_total",
        "ninfer_waiting_expired_requests_total")}
    transfers = {k: v - before.get(k, 0.0) for k, v in after.items()
                 if k.startswith("ninfer_context_transfer_bytes_total{")}
    total_prompt = sum(r["prompt_tokens"] for r in ok)
    total_cached = sum(r["cached_tokens"] for r in ok)
    total_ideal = sum(max(r["private_ideal"], r["cached_tokens"]) for r in ok)
    return {
        "run": run.name, "warmup_s": warmup_s,
        "preset": summary["preset"], "thinking": summary["thinking"], "wall_s": wall,
        "capacity": capacity,
        "requests": len(measured), "errors": len(measured) - len(ok),
        "rejections_429": sum(r.get("rejections", 0) for r in records),
        "split": dict(split),
        "tokens": {"prompt": total_prompt, "cached": total_cached, "ideal": total_ideal,
                   "hit_rate": total_cached / total_prompt if total_prompt else None,
                   "ideal_hit_rate": total_ideal / total_prompt if total_prompt else None},
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "lost_tokens": dict(lost), "sibling_ceiling_tokens": sibling,
        "lru_ceiling_bytes": lru_bytes,
        "ttft_s": {"n": len(ttft), "p50": pct(ttft, .5), "p95": pct(ttft, .95),
                   "p99": pct(ttft, .99), "max": max(ttft) if ttft else None},
        "ttft_by_class": {cls: {"n": len(v), "p50": pct(v, .5), "p95": pct(v, .95)}
                          for cls in ("private", "shared", "partial", "none")
                          for v in [[r["client_ttft_s"] for r in ok
                                     if r.get("class") == cls and r["client_ttft_s"]]]},
        "queue_s": {"n": len(queue), "p50": pct(queue, .5), "p95": pct(queue, .95),
                    "p99": pct(queue, .99)},
        "binding_s": {"p50": pct(binding, .5), "p95": pct(binding, .95)},
        "restore": {"n": len(restore_s), "seconds_p50": pct(restore_s, .5),
                    "seconds_p95": pct(restore_s, .95),
                    "mib_p50": (pct(restore_bytes, .5) or 0) / MIB},
        "source_wait_s_total": sum(source_wait),
        "fallback_reasons": dict(fallback),
        "joined": len(joined),
        "lanes": {"utilisation": statistics.fmean(running) if running else None,
                  "all_busy_fraction": (sum(1 for x in running if x >= 1) / len(running)
                                        if running else None),
                  "waiting_mean": statistics.fmean(waiting) if waiting else None,
                  "waiting_max": max(waiting) if waiting else None,
                  "prefilling_mean": statistics.fmean(prefilling) if prefilling else None},
        "throughput": {"decode_tok_s_wall": m["ninfer_decode_tokens_total"] / wall,
                       "generation_tok_s_wall": m["ninfer_generation_tokens_total"] / wall,
                       "prefill_tok_s_wall": m["ninfer_prefill_tokens_total"] / wall,
                       "prefill_tok_s_exec": (m["ninfer_prefill_tokens_total"]
                                              / m["ninfer_prefill_seconds_total"]
                                              if m["ninfer_prefill_seconds_total"] else None)},
        "counters": m,
        "store_mib_written_per_hour": m["ninfer_context_store_bytes_written_total"] / MIB
        / (wall / 3600.0),
        "host_peak_mib": after.get("ninfer_host_context_peak_bytes", 0) / MIB,
        # Gauges as the server reported them at the end (start-up restore is set before the run).
        "store_gauges": {k: after.get(k) for k in (
            "ninfer_context_store_images", "ninfer_context_store_used_bytes",
            "ninfer_context_store_restored_sessions", "ninfer_context_store_restored_bytes",
            "ninfer_context_store_restore_seconds")},
        "transfer_seconds": metric(after, before, "ninfer_context_transfer_seconds_total"),
        "transfers_mib": {k: v / MIB for k, v in transfers.items() if v},
    }


def fmt(value: Any, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def markdown(rows: list[dict[str, Any]]) -> str:
    out = ["| run | req | hit (cached/prompt) | ideal | private / shared / partial / none |"
           " TTFT p50 / p95 / p99 s | queue p50 / p95 s | lanes busy | decode tok/s |",
           "|---|---:|---:|---:|---|---|---|---:|---:|"]
    for r in rows:
        s = r["split"]
        out.append(
            f"| {r['run']} | {r['requests'] - r['errors']} | {fmt(r['tokens']['hit_rate'], '.3f')}"
            f" | {fmt(r['tokens']['ideal_hit_rate'], '.3f')} | {s.get('private', 0)} / "
            f"{s.get('shared', 0)} / {s.get('partial', 0)} / {s.get('none', 0)} | "
            f"{fmt(r['ttft_s']['p50'])} / {fmt(r['ttft_s']['p95'])} / {fmt(r['ttft_s']['p99'])} | "
            f"{fmt(r['queue_s']['p50'])} / {fmt(r['queue_s']['p95'])} | "
            f"{fmt(r['lanes']['utilisation'])} | {fmt(r['throughput']['decode_tok_s_wall'], '.1f')} |")
    out += ["", "| run | lost: diverged | shared_miss | evicted_capacity | evicted_host_capacity | evicted_policy |"
            " store_miss | sibling ceiling | spill pages | preemptions | 429s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lost = r["lost_tokens"]

        def cell(cause: str) -> str:
            return f"{int(lost.get(cause, 0)):,} ({int(lost.get(cause + '_requests', 0))})"
        out.append(f"| {r['run']} | {cell('diverged')} | {cell('shared_miss')} | "
                   f"{cell('evicted_capacity')} | {cell('evicted_host_capacity')} | {cell('evicted_policy')} | {cell('store_miss')} | "
                   f"{int(r['sibling_ceiling_tokens']):,} | "
                   f"{int(r['counters']['ninfer_context_pressure_spill_pages_total'])} | "
                   f"{int(r['counters']['ninfer_preemptions_total'])} | {r['rejections_429']} |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--tolerance", type=int, default=256,
                        help="tokens below the ideal still counted as a hit (default 256)")
    parser.add_argument("--kv-bytes-per-token", type=float, default=18_000.0,
                        help="LRU model: KV bytes per token (27B rk4v4 ~18 KB, serving.md)")
    parser.add_argument("--state-bytes", type=float, default=74.5 * MIB,
                        help="LRU model: recurrent state per image (27B FP16 state 74.5 MiB)")
    parser.add_argument("--warmup-s", type=float, default=0.0,
                        help="leave requests submitted before this out of the statistics "
                             "(counters, lanes and throughput still cover the whole run)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    rows = [analyse(run, args.tolerance, args.kv_bytes_per_token, args.state_bytes,
                    args.warmup_s)
            for run in args.runs]
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if args.markdown:
        print(markdown(rows))
    else:
        print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
