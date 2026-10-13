"""Measurement client for the Flash-Next context store / NVMe tier work (scratch, not product code).

Sends OpenAI chat completions to a running ninfer-serve, streaming, and prints one JSON line per
request: label, prompt tokens, cached tokens, TTFT (client), total, completion tokens, decode tok/s.

usage: client.py PORT PLAN.json
PLAN: a list of {"label", "messages": [...], "max_tokens", "repeat"} (repeat default 1).
"""
import json
import sys
import time
import urllib.request


def run(port, label, messages, max_tokens):
    body = json.dumps({
        "model": "Qwen3.8-Flash-Next",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    end = time.perf_counter()
    usage = usage or {}
    completion = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = (first - start) if first else None
    decode = (completion - 1) / (end - first) if first and completion > 1 and end > first else None
    out = {"label": label, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
           "ttft_s": round(ttft, 4) if ttft else None, "total_s": round(end - start, 4),
           "completion_tokens": completion, "decode_tok_s": round(decode, 2) if decode else None,
           "text": "".join(text)[:120]}
    print(json.dumps(out), flush=True)
    return out


def main():
    port = int(sys.argv[1])
    with open(sys.argv[2], encoding="utf-8") as f:
        plan = json.load(f)
    for step in plan:
        for _ in range(step.get("repeat", 1)):
            run(port, step["label"], step["messages"], step.get("max_tokens", 16))


if __name__ == "__main__":
    main()
