"""Spike client: sends eight different conversations to a local ninfer-serve, one at a time, and
moves the route trace written during each request to trace_<i>.bin (NINFER_ROUTE_TRACE).

usage: python client.py <port> <trace path> <out dir> <repo root>
"""
import json
import os
import sys
import time
import urllib.request

port, trace, out_dir, repo = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


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


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30) as r:
    MODEL = json.loads(r.read())["data"][0]["id"]
os.makedirs(out_dir, exist_ok=True)
for i, (name, body) in enumerate(CONVERSATIONS):
    body = dict(body)
    body.update({"model": MODEL, "max_tokens": 1024, "temperature": 0, "stream": False})
    if os.path.exists(trace):
        os.remove(trace)
    t0 = time.time()
    resp = post(body)
    dt = time.time() - t0
    if os.path.exists(trace):
        os.replace(trace, os.path.join(out_dir, f"trace_{i}_{name}.bin"))
    usage = resp.get("usage", {})
    print(json.dumps({"i": i, "name": name, "seconds": round(dt, 2), "usage": usage,
                      "finish": resp["choices"][0].get("finish_reason")}), flush=True)
    with open(os.path.join(out_dir, f"resp_{i}_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=1)
