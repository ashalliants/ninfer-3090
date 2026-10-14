#!/usr/bin/env python3
"""Seed conversations of several sizes into a context store, then continue each after a restart.

seed:     one turn per conversation (sizes in words), saved with the reply to --state.
continue: one short follow-up per saved conversation; prints TTFT and cached tokens per size.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.ninfer_serve.client import NInferServeClient  # noqa: E402
from tools.ninfer_serve.openai_chat import chat_request  # noqa: E402

WORDS = ("anchor bearing cable damper engine flange gasket housing impeller journal keeper lever "
         "manifold nozzle orifice piston quill rotor spindle tappet union valve washer yoke").split()


def run(client, model, messages, reply_tokens):
    ex = client.prepare(chat_request(model, messages, reply_tokens))
    sent: list[int] = []
    res = ex.execute(on_sent=sent.append)
    first = next((e.received_ns for e in res.events if e.kind == "model_output"), None)
    text = "".join(e.output for e in res.events if e.kind == "model_output")
    usage = {}
    for e in res.events:
        if e.payload and isinstance(e.payload.get("usage"), dict):
            usage = e.payload["usage"]
    start = sent[0] if sent else res.http.sent_ns
    err = res.protocol_error or res.error_code or res.http.error
    return text, usage, (first - start) / 1e9 if first and start else None, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seed", "continue"])
    ap.add_argument("--base-url", default="http://127.0.0.1:18200")
    ap.add_argument("--sizes", default="1100,1900,3000,4600,7600,12000")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    client = NInferServeClient(args.base_url, 1800.0)
    model = client.discover_model()
    if args.mode == "seed":
        convs = []
        for i, words in enumerate(int(w) for w in args.sizes.split(",")):
            rng = random.Random(1000 + i)
            body = " ".join(rng.choice(WORDS) for _ in range(words))
            messages = [{"role": "system", "content": f"You are reviewer {i}."},
                        {"role": "user", "content": f"Log {i}:\n{body}\nSummarise it in one line."}]
            text, usage, ttft, err = run(client, model, messages, 32)
            if err or ttft is None:
                # No first output is as invalid as an error, as in continue mode.
                print("seed error", err or "no first output")
                return 1
            messages.append({"role": "assistant", "content": text})
            convs.append({"words": words, "messages": messages, "prompt": usage.get("prompt_tokens")})
            print(f"seed words={words} prompt={usage.get('prompt_tokens')} ttft={ttft:.3f}")
            # Idle long enough for nothing; the shutdown flush writes every conversation.
        args.state.write_text(json.dumps(convs))
        return 0
    convs = json.loads(args.state.read_text())
    rows = []
    for c in convs:
        messages = c["messages"] + [{"role": "user", "content": "Now list the three most frequent words."}]
        text, usage, ttft, err = run(client, model, messages, 32)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        rows.append({"words": c["words"], "prompt": usage.get("prompt_tokens"), "cached": cached,
                     "ttft": ttft, "error": err})
        print(json.dumps(rows[-1]))
        time.sleep(0.5)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    # The same predicate sum_restart.py uses: no first output is as invalid as an error.
    failed = [r for r in rows if r["error"] or r["ttft"] is None]
    if failed:
        print(f"continue error: {len(failed)} of {len(rows)} continuations failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
