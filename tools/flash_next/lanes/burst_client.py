"""Burst client for the lanes gate: the spike's eight conversations (client.py on origin/spike/qwen4exp-lanes)
all submitted at t = 0 (an agent burst, in index order), streamed, to a local ninfer-serve. Prints one JSON line
per request (seconds to first token, to completion, completion tokens) and a summary line.

usage: python burst_client.py <port> <out json> <repo root> [seq]
(seq: one request at a time, as the spike's client did, for lone-request speed)
"""
import json
import os
import sys
import threading
import time
import urllib.request

port, out_path, repo = sys.argv[1], sys.argv[2], sys.argv[3]


def read(rel, limit=None):
    with open(os.path.join(repo, rel), encoding="utf-8") as f:
        text = f.read()
    return text[:limit] if limit else text


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file in the repository",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the repository for a regex",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
                    "required": ["pattern"]}}},
]

tool_history = [
    {"role": "system", "content": "You are a coding agent working in a C++/CUDA repository. Use the tools to "
                                  "investigate before answering. Call tools with JSON arguments."},
    {"role": "user", "content": "The expert cache hit rate dropped after the last merge. Find out why and propose a fix."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "search", "arguments": json.dumps({"pattern": "kDecodePromotion", "glob": "*.cpp"})}}]},
    {"role": "tool", "tool_call_id": "c1", "content":
     "src/models/qwen4_exp/program/program.cpp:181:constexpr std::size_t kDecodePromotionsPerLayer = 1;\n"
     "src/models/qwen4_exp/program/program.cpp:183:constexpr std::uint64_t kDecodePromotionInterval = 4;\n"
     "src/models/qwen4_exp/program/program.cpp:2360:        if (stats.promotions < residency_->frames()) { return kDecodePromotionsPerLayer * tokens; }"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {
        "name": "read_file", "arguments": json.dumps({"path": "src/models/qwen4_exp/program/expert_residency.cpp"})}}]},
    {"role": "tool", "tool_call_id": "c2", "content": read("src/models/qwen4_exp/program/expert_residency.cpp", 9000)},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "type": "function", "function": {
        "name": "run_shell", "arguments": json.dumps({"command": "git log --oneline -8 -- src/models/qwen4_exp"})}}]},
    {"role": "tool", "tool_call_id": "c3", "content": "04901788 Merge pull request #277\n564bc3ef Merge pull request #276\n"
     "2d861905 docs: idle cost\nadc45e76 Merge fork/master\n9a1b2c3d perf: restore displaced experts after prefill\n"
     "77e0f1aa fix: lease give-back order\n5c4d3e2f feat: wide prefill calls\n1f2e3d4c feat: expert cache warm start"},
]

CONVERSATIONS = [
    ("codegen", {"messages": [{"role": "user", "content":
        "Write a production-quality Python module implementing an LRU cache with per-entry TTL, thread safety, "
        "a decorator for memoizing functions, and statistics (hits, misses, evictions). Then write pytest unit "
        "tests covering expiry, eviction order, concurrency and the decorator. Explain the design briefly."}]}),
    ("review", {"messages": [{"role": "user", "content":
        "Review this C++ file for correctness bugs, undefined behaviour and performance problems. Be specific "
        "and cite line contents.\n\n```cpp\n" + read("src/models/qwen4_exp/program/expert_cache/expert_cache.cpp") +
        "\n```"}]}),
    ("story", {"messages": [{"role": "user", "content":
        "Write a 1,500-word literary short story about a lighthouse keeper on a remote Scottish island who "
        "starts receiving letters from a version of herself forty years in the future. Rich imagery, dialogue, "
        "and a quiet, ambiguous ending."}]}),
    ("tools", {"messages": tool_history, "tools": TOOLS, "tool_choice": "auto"}),
    ("summary", {"messages": [{"role": "user", "content":
        "Summarize the following documentation for an operator who must deploy this server tomorrow. Produce a "
        "checklist, the key commands, and the three most likely failure modes with fixes.\n\n" +
        read("docs/serving.md", 40000)}]}),
    ("math", {"messages": [{"role": "user", "content":
        "A factory has three machines. Machine A produces 120 parts per hour with a 2% defect rate, B produces 90 "
        "per hour with 1.5%, C produces 150 per hour with 3%. Parts are mixed and inspected; an inspector catches "
        "95% of defects and falsely rejects 1% of good parts. (1) What fraction of rejected parts are actually "
        "defective? (2) If a shipped part is defective, what is the probability it came from C? (3) The plant "
        "wants under 0.05% defective shipped parts: what minimum catch rate is needed? Show all working."}]}),
    ("sql", {"messages": [{"role": "user", "content":
        "Design a PostgreSQL schema for a hotel booking system (properties, room types, rooms, rate plans, "
        "bookings, guests, payments, housekeeping tasks). Include indexes, constraints that prevent double "
        "booking, and five realistic reporting queries with explanations. Then write a migration that adds "
        "multi-currency pricing without downtime."}]}),
    ("chinese", {"messages": [{"role": "user", "content":
        "请用中文详细解释混合专家模型（Mixture of Experts）的工作原理，包括路由器、负载均衡损失、专家并行、以及在单张消费级显卡上"
        "用主机内存卸载专家时的缓存策略。最后给出一个简短的对比表。"}]}),
]



def stream(body, t0, out):
    body = dict(body)
    body.update({"stream": True, "stream_options": {"include_usage": True}})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first, usage, finish = None, {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first = time.time() - t0
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    out["first"], out["done"], out["usage"], out["finish"] = first, time.time() - t0, usage, finish


with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
    MODEL = json.loads(r.read())["data"][0]["id"]


def prepared(body):
    body = dict(body)
    body.update({"model": MODEL, "max_tokens": 1024, "temperature": 0})
    return body


# Warm-up (not timed): one short request, so the first measured request is not the session's first.
warm = {}
stream({**prepared({"messages": [{"role": "user", "content": "Say hello in five words."}]}), "max_tokens": 16},
       time.time(), warm)

results = [dict(i=i, name=name) for i, (name, _) in enumerate(CONVERSATIONS)]
t0 = time.time()
threads = []
sequential = len(sys.argv) > 4 and sys.argv[4] == "seq"
for i, (name, body) in enumerate(CONVERSATIONS):
    if sequential:
        t0 = time.time()
        stream(prepared(body), t0, results[i])
        continue
    th = threading.Thread(target=stream, args=(prepared(body), t0, results[i]))
    th.start()
    threads.append(th)
    time.sleep(0.05)  # submission order = index order (the queue is FIFO)
for th in threads:
    th.join()
makespan = max(r["done"] for r in results)
tokens = sum(r["usage"].get("completion_tokens", 0) for r in results)
for r in results:
    print(json.dumps(r), flush=True)
summary = {"makespan": round(makespan, 2), "completion_tokens": tokens,
           "aggregate_tok_s_over_makespan": round(tokens / makespan, 2),
           "mean_completion": round(sum(r["done"] for r in results) / len(results), 2),
           "mean_ttft": round(sum(r["first"] for r in results) / len(results), 2)}
print("SUMMARY " + json.dumps(summary), flush=True)
with open(out_path, "w") as f:
    json.dump({"results": results, "summary": summary}, f, indent=1)
