#!/usr/bin/env python3
"""Hosted multi-tenant rotation against a running ninfer-serve.

T tenants share S system prompts; each visit appends one tool-result-sized user turn to its own
conversation (restarting at --max-prompt-tokens). W client workers (= lanes) take the next tenant
in a fixed rotation order, so at most W conversations are in flight and the rest must come back
from the Host tier, the store, or be re-prefilled. Greedy, thinking off.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.ninfer_serve.client import NInferServeClient  # noqa: E402
from tools.ninfer_serve.openai_chat import chat_request  # noqa: E402

WORDS = (
    "anchor bearing cable damper engine flange gasket housing impeller journal keeper lever "
    "manifold nozzle orifice piston quill rotor spindle tappet union valve washer yoke bracket "
    "coupling detent eccentric fulcrum governor hinge idler jib knuckle linkage mandrel nut "
    "outrigger pawl ratchet sprocket trunnion"
).split()


def filler(rng: random.Random, words: int) -> str:
    out = []
    for start in range(0, words, 16):
        out.append(" ".join(rng.choice(WORDS) for _ in range(min(16, words - start))) + ".")
    return "\n".join(out)


@dataclass
class Tenant:
    index: int
    rng: random.Random
    system: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    visits: int = 0
    busy: bool = False

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}]


@dataclass
class Sample:
    tenant: int
    visit: int
    started: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    ttft: float | None
    total: float
    error: str | None


def scrape(base: str) -> dict[str, float]:
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^(ninfer_[a-z0-9_]+(?:\{[^}]*\})?) ([-+0-9.eE]+)$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18200")
    ap.add_argument("--tenants", type=int, default=24)
    ap.add_argument("--systems", type=int, default=3)
    ap.add_argument("--system-words", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--requests", type=int, default=80)
    ap.add_argument("--turn-words-min", type=int, default=800)
    ap.add_argument("--turn-words-max", type=int, default=2400)
    ap.add_argument("--max-prompt-tokens", type=int, default=30000)
    ap.add_argument("--reply-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--system-seed", type=int, default=None, help="seed of the shared system prompts (default --seed)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    client = NInferServeClient(args.base_url, 1800.0)
    model = client.discover_model()
    srng = random.Random(args.seed if args.system_seed is None else args.system_seed)
    systems = [f"You are support agent family {s}. Tools manual follows.\n" + filler(srng, args.system_words)
               for s in range(args.systems)]
    tenants = [Tenant(i, random.Random(args.seed * 1000 + i), systems[i % args.systems]) for i in range(args.tenants)]
    for t in tenants:
        t.reset()
    order = list(range(args.tenants))
    random.Random(args.seed + 1).shuffle(order)
    lock = threading.Lock()
    cursor = [0]
    issued = [0]
    samples: list[Sample] = []
    t0 = time.perf_counter()
    before = scrape(args.base_url)

    def next_tenant() -> Tenant | None:
        with lock:
            if issued[0] >= args.requests:
                return None
            for _ in range(len(order)):
                t = tenants[order[cursor[0] % len(order)]]
                cursor[0] += 1
                if not t.busy:
                    t.busy = True
                    issued[0] += 1
                    return t
            return None

    def worker() -> None:
        while True:
            t = next_tenant()
            if t is None:
                return
            words = t.rng.randint(args.turn_words_min, args.turn_words_max)
            t.messages.append({"role": "user", "content": f"Tenant {t.index} visit {t.visits}. Tool result:\n" + filler(t.rng, words)})
            request = chat_request(model, t.messages, args.reply_tokens)
            started = time.perf_counter() - t0
            try:
                ex = client.prepare(request)
                sent: list[int] = []
                res = ex.execute(on_sent=sent.append)
            except Exception as exc:  # a failed request is a failed sample, never a missing one
                s = Sample(t.index, t.visits, started, 0, 0, 0, None, 0.0, f"{type(exc).__name__}: {exc}")
                with lock:
                    samples.append(s)
                    t.visits += 1
                    t.messages.pop()
                    t.busy = False
                print(f"t={t.index} v={s.visit} err={s.error}", file=sys.stderr, flush=True)
                continue
            first = next((e.received_ns for e in res.events if e.kind == "model_output"), None)
            text = "".join(e.output for e in res.events if e.kind == "model_output")
            usage: dict[str, Any] = {}
            for e in res.events:
                if e.payload and isinstance(e.payload.get("usage"), dict):
                    usage = e.payload["usage"]
            start = sent[0] if sent else res.http.sent_ns
            end = res.http.ended_ns or time.perf_counter_ns()
            err = res.protocol_error or res.error_code or res.http.error
            if res.http.status not in (None, 200) and err is None:
                err = f"http {res.http.status}"
            s = Sample(t.index, t.visits, started, int(usage.get("prompt_tokens") or 0),
                       int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                       int(usage.get("completion_tokens") or 0),
                       (first - start) / 1e9 if first and start else None, (end - start) / 1e9 if start else 0.0, err)
            with lock:
                samples.append(s)
                t.visits += 1
                if err is None:
                    t.messages.append({"role": "assistant", "content": text})
                    if s.prompt_tokens >= args.max_prompt_tokens:
                        t.reset()
                else:
                    t.messages.pop()
                t.busy = False
            print(f"t={t.index} v={s.visit} prompt={s.prompt_tokens} cached={s.cached_tokens} ttft={s.ttft} err={err}", file=sys.stderr, flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    after = scrape(args.base_url)
    ok = [s for s in samples if s.error is None]
    ttfts = [s.ttft for s in ok if s.ttft is not None]
    prompt = sum(s.prompt_tokens for s in ok)
    cached = sum(s.cached_tokens for s in ok)
    decode_time = sum(max(1e-9, s.total - (s.ttft or 0)) for s in ok)
    comp = sum(s.completion_tokens for s in ok)
    delta = {k.replace("ninfer_", ""): after.get(k, 0) - before.get(k, 0)
             for k in after if "context_store" in k or k in ("ninfer_prompt_tokens_total",)}
    summary = {
        "requests": len(samples), "errors": len(samples) - len(ok), "wall_s": wall,
        "hit_rate": cached / prompt if prompt else 0.0, "prompt_tokens": prompt,
        "reprefilled_tokens": prompt - cached,
        "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95), "ttft_p99": pct(ttfts, 0.99),
        "ttft_mean": statistics.fmean(ttfts) if ttfts else None,
        "decode_tok_s_per_stream": comp / decode_time if decode_time else None,
        "store_bytes_written": delta.get("context_store_bytes_written_total"),
        "store_bytes_written_per_hour": (delta.get("context_store_bytes_written_total") or 0) * 3600 / wall,
        "store": delta,
        "read_rate": after.get("ninfer_context_store_read_bytes_per_second"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "args": {k: str(v) for k, v in vars(args).items()},
                   "samples": [asdict(s) for s in samples]}, f, indent=1)
    print(json.dumps(summary, indent=1))
    if len(samples) != args.requests:
        print(f"incomplete run: {len(samples)} samples recorded, {args.requests} requested", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
