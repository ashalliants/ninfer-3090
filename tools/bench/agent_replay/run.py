"""Drive deterministic copy-heavy agent sessions through a running ninfer-serve."""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.bench.agent_replay.metrics import SourceIndex, flatten_counters, summarize
from tools.bench.agent_replay.workload import (
    SYSTEM_PROMPT,
    TOOLS,
    Corpus,
    SessionBuilder,
    TurnPlan,
    derived_seed,
    est_tokens,
)
from tools.ninfer_serve.client import NInferServeClient
from tools.ninfer_serve.openai_chat import chat_request

SCHEMA = "ninfer_agent_replay"
SCHEMA_VERSION = 1
# Template framing per message (role markers, tool_response wrapper) in prompt-token estimates.
MESSAGE_OVERHEAD_TOKENS = 16


@dataclass
class RunConfig:
    base_url: str
    api_key: str | None
    repo: Path
    commit: str
    seed: int
    sessions: int
    concurrency: int
    min_context: int
    max_context: int
    obs_tokens: tuple[int, int]
    max_copy_tokens: int
    max_turns: int
    timeout_seconds: float
    request_log: Path | None
    out: Path


@dataclass
class Reply:
    response_id: str | None
    content: str
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, Any]
    timings: dict[str, Any]
    client_ttft_s: float | None
    client_total_s: float
    error: str | None


def send(client: NInferServeClient, model: str, messages: list[dict[str, Any]],
         max_tokens: int) -> Reply:
    request = chat_request(model, messages, max_tokens, tools=TOOLS)
    sent: list[int] = []
    result = client.prepare(request).execute(on_sent=sent.append)
    response_id: str | None = None
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    first_ns: int | None = None
    for event in result.events:
        payload = event.payload or {}
        response_id = response_id or event.response_id
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        if isinstance(payload.get("timings"), dict):
            timings = payload["timings"]
        for choice in payload.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(int(call.get("index", len(calls))),
                                        {"id": None, "name": "", "arguments": ""})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
        if event.kind == "model_output" and first_ns is None:
            first_ns = event.received_ns
    start = sent[0] if sent else result.http.sent_ns
    end = result.http.ended_ns or time.perf_counter_ns()
    error = result.protocol_error or result.error_code or result.http.error
    if result.http.status not in (None, 200) and error is None:
        error = f"http {result.http.status}: {result.error_message}"
    return Reply(
        response_id=response_id,
        content="".join(content),
        tool_calls=[calls[i] for i in sorted(calls)],
        finish_reason=finish_reason,
        usage=usage,
        timings=timings,
        client_ttft_s=(first_ns - start) / 1e9 if first_ns and start else None,
        client_total_s=(end - start) / 1e9 if start else 0.0,
        error=error,
    )


def output_text(reply: Reply) -> str:
    """What the model generated, with tool-call argument values unescaped."""
    parts = [reply.content]
    for call in reply.tool_calls:
        try:
            arguments = json.loads(call["arguments"])
        except (json.JSONDecodeError, TypeError):
            parts.append(call["arguments"])
            continue
        if isinstance(arguments, dict):
            parts.extend(v if isinstance(v, str) else json.dumps(v) for v in arguments.values())
    return "\n".join(p for p in parts if p)


def tool_ack(call: dict[str, Any]) -> str:
    try:
        path = json.loads(call["arguments"]).get("file_path", "the file")
    except (json.JSONDecodeError, AttributeError, TypeError):
        path = "the file"
    if call["name"] in ("Write", "Edit"):
        return f"The file {path} has been updated successfully."
    return "OK."


class SessionRunner:
    def __init__(self, config: RunConfig, corpus: Corpus, model: str, index: int,
                 emit: Any) -> None:
        self.config = config
        self.index = index
        self.emit = emit
        self.client = NInferServeClient(config.base_url, config.timeout_seconds, config.api_key)
        self.model = model
        self.builder = SessionBuilder(corpus, config.seed, index,
                                      obs_tokens=config.obs_tokens,
                                      max_copy_tokens=config.max_copy_tokens)
        rng = random.Random(derived_seed("target", config.seed, index))
        self.target = rng.randint(config.min_context, config.max_context)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.sources = SourceIndex()
        self.sources.add(SYSTEM_PROMPT)
        self.turn_index: list[dict[str, Any]] = []
        self.stop_reason = "max_turns"

    def _append(self, message: dict[str, Any], *texts: str) -> None:
        self.messages.append(message)
        for text in texts:
            self.sources.add(text)

    def _stage(self, plan: TurnPlan, pending: list[dict[str, Any]], replied_text: bool) -> None:
        for call in pending:
            ack = tool_ack(call)
            self._append({"role": "tool", "tool_call_id": call["id"], "content": ack}, ack)
        if replied_text and plan.instruction is None:
            self._append({"role": "user", "content": "Continue."}, "Continue.")
        obs = plan.observation
        call_id = f"call_s{self.index}_t{plan.turn}"
        arguments = json.dumps(obs.arguments, ensure_ascii=False)
        self._append({"role": "assistant", "content": "",
                      "tool_calls": [{"id": call_id, "type": "function",
                                      "function": {"name": obs.tool, "arguments": arguments}}]},
                     arguments)
        self._append({"role": "tool", "tool_call_id": call_id, "content": obs.text}, obs.text)
        if plan.instruction is not None:
            self._append({"role": "user", "content": plan.instruction}, plan.instruction)

    def run(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        task, plan = self.builder.opening()
        self._append({"role": "user", "content": task}, task)
        pending: list[dict[str, Any]] = []
        replied_text = False
        context = est_tokens(len(SYSTEM_PROMPT) + len(task)) + 1500  # tool schema
        for turn in range(self.config.max_turns):
            if turn > 0:
                plan = self.builder.next_turn(turn)
            staged = len(plan.observation.text) + len(plan.instruction or "")
            projected = context + est_tokens(staged) + MESSAGE_OVERHEAD_TOKENS * 4
            if projected + plan.max_tokens > self.target:
                self.stop_reason = "target_context"
                break
            self._stage(plan, pending, replied_text)
            reply = send(self.client, self.model, self.messages, plan.max_tokens)
            sample = self._sample(plan, reply)
            samples.append(sample)
            self.emit(sample)
            if reply.error is not None:
                self.stop_reason = "error"
                break
            text = output_text(reply)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                assistant["tool_calls"] = [
                    {"id": c["id"] or f"call_model_s{self.index}_t{turn}_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for i, c in enumerate(reply.tool_calls)
                ]
            self._append(assistant, text)
            pending = assistant.get("tool_calls", [])
            pending = [{"id": c["id"], "name": c["function"]["name"],
                        "arguments": c["function"]["arguments"]} for c in pending]
            replied_text = not pending
            context = sample["prompt_tokens"] + sample["output_tokens"]
        return samples

    def _sample(self, plan: TurnPlan, reply: Reply) -> dict[str, Any]:
        covered, total = self.sources.overlap(output_text(reply)) if reply.error is None else (0, 0)
        details = reply.usage.get("prompt_tokens_details") or {}
        counters: dict[str, Any] = {}
        flatten_counters("timings", {k: v for k, v in reply.timings.items()
                                     if isinstance(v, int)}, counters)
        self.turn_index.append({"turn": plan.turn, "message_count": len(self.messages),
                                "response_id": reply.response_id, "category": plan.category})
        sample = {
            "session": self.index,
            "turn": plan.turn,
            "category": plan.category,
            "copy": plan.copy,
            "response_id": reply.response_id,
            "error": reply.error,
            "finish_reason": reply.finish_reason,
            "max_tokens": plan.max_tokens,
            "prompt_tokens": int(reply.usage.get("prompt_tokens") or 0),
            "cached_tokens": int(details.get("cached_tokens") or 0),
            "output_tokens": int(reply.usage.get("completion_tokens") or 0),
            "tool_calls": [c["name"] for c in reply.tool_calls],
            "output_chars": len(output_text(reply)),
            "overlap_covered": covered,
            "overlap_units": total,
            "copy_overlap": covered / total if total else None,
            "client_ttft_s": reply.client_ttft_s,
            "client_total_s": reply.client_total_s,
            "server_timings": reply.timings,
            "observation": {"tool": plan.observation.tool, "arguments": plan.observation.arguments,
                            "chars": len(plan.observation.text)},
            "notes": plan.notes,
            "counters": counters,
        }
        apply_timing(sample, None)
        return sample


def apply_timing(sample: dict[str, Any], done: dict[str, Any] | None) -> None:
    """TTFT and decode time from the best source: request log, usage timings, then client."""
    ttft = total = decode = None
    source = None
    if done is not None:
        seconds = done.get("timings_seconds") or {}
        if seconds.get("total") is not None and seconds.get("ttft") is not None:
            ttft, total, source = seconds["ttft"], seconds["total"], "request_log"
            decode = total - ttft
    if source is None and sample["server_timings"].get("predicted_ms") is not None:
        timings = sample["server_timings"]
        # `prompt_ms` is prompt wall time from the initial binding attempt, so it omits
        # queue/preparation; TTFT stays the client's first-output latency and `prompt_ms` is
        # reported separately.
        ttft = sample["client_ttft_s"]
        decode = timings["predicted_ms"] / 1000.0
        if timings.get("prompt_ms") is not None:
            sample["prompt_s"] = timings["prompt_ms"] / 1000.0
        source = "usage_timings"
    if source is None and sample["client_ttft_s"] is not None:
        ttft = sample["client_ttft_s"]
        decode = sample["client_total_s"] - ttft
        source = "client"
    sample["ttft_s"] = ttft
    sample["decode_s"] = decode if decode and decode > 0 else None
    sample["decode_tps"] = (sample["output_tokens"] / decode
                            if decode and decode > 0 and sample["output_tokens"] else None)
    sample["timing_source"] = source


def read_request_log(path: Path, offset: int) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    with path.open("rb") as log:
        log.seek(offset)
        for raw in log:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "request_done":
                continue
            response_id = (record.get("request") or {}).get("response_id")
            if response_id:
                done[response_id] = record
    return done


def merge_request_log(samples: list[dict[str, Any]], done: dict[str, dict[str, Any]],
                      token_ids_path: Path) -> int:
    merged = 0
    with token_ids_path.open("w", encoding="utf-8") as ids:
        for sample in samples:
            record = done.get(sample.get("response_id") or "")
            if record is None:
                continue
            merged += 1
            counters: dict[str, Any] = {}
            speculative = {k: v for k, v in (record.get("speculative") or {}).items()
                           if k != "draft_window"}
            flatten_counters("speculative", speculative, counters)
            sample["counters"].update(counters)
            sample["speculative_backend"] = (record.get("speculative") or {}).get("backend")
            sample["draft_window"] = (record.get("speculative") or {}).get("draft_window")
            result = record.get("result") or {}
            sample["prefix_reuse_path"] = result.get("prefix_reuse_path")
            sample["computed_prefill_tokens"] = result.get("computed_prefill_tokens")
            sample["request_log_timings_s"] = record.get("timings_seconds")
            apply_timing(sample, record)
            ids.write(json.dumps({"session": sample["session"], "turn": sample["turn"],
                                  "response_id": sample["response_id"],
                                  "generated_token_ids": result.get("generated_token_ids") or []})
                      + "\n")
    return merged


def run(config: RunConfig) -> int:
    for name in ("sessions", "max_turns", "concurrency"):
        if getattr(config, name) < 1:
            print(f"error: --{name.replace('_', '-')} must be at least 1", file=sys.stderr)
            return 2
    corpus = Corpus(config.repo, config.commit)
    probe = NInferServeClient(config.base_url, config.timeout_seconds, config.api_key)
    model = probe.discover_model()
    config.out.mkdir(parents=True, exist_ok=True)
    log_offset = config.request_log.stat().st_size if config.request_log else 0
    lock = threading.Lock()
    samples: list[dict[str, Any]] = []
    started = time.time()

    def emit(sample: dict[str, Any]) -> None:
        with lock:
            samples.append(sample)
            print(f"session={sample['session']} turn={sample['turn']} {sample['category']:<13} "
                  f"prompt={sample['prompt_tokens']} cached={sample['cached_tokens']} "
                  f"out={sample['output_tokens']} finish={sample['finish_reason']} "
                  f"tps={_fmt(sample.get('decode_tps'), '.1f')} "
                  f"overlap={_fmt(sample.get('copy_overlap'), '.2f')} error={sample['error']}",
                  file=sys.stderr, flush=True)

    runners = [SessionRunner(config, corpus, model, i, emit) for i in range(config.sessions)]
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        list(pool.map(lambda r: r.run(), runners))
    wall = time.time() - started

    merged = None
    if config.request_log is not None:
        done = read_request_log(config.request_log, log_offset)
        merged = merge_request_log(samples, done, config.out / "output_token_ids.jsonl")

    samples.sort(key=lambda s: (s["session"], s["turn"]))
    with (config.out / "turns.jsonl").open("w", encoding="utf-8") as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with (config.out / "transcripts.jsonl").open("w", encoding="utf-8") as out:
        for runner in runners:
            out.write(json.dumps({"session": runner.index, "tools": TOOLS,
                                  "messages": runner.messages, "turns": runner.turn_index},
                                 ensure_ascii=False) + "\n")
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "corpus_commit": corpus.commit,
        "config": {
            "base_url": config.base_url, "seed": config.seed, "sessions": config.sessions,
            "concurrency": config.concurrency, "min_context": config.min_context,
            "max_context": config.max_context, "obs_tokens": list(config.obs_tokens),
            "max_copy_tokens": config.max_copy_tokens, "max_turns": config.max_turns,
            "request_log": str(config.request_log) if config.request_log else None,
        },
        "wall_seconds": wall,
        "request_log_merged": merged,
        "sessions": [
            {"session": r.index, "target_context": r.target, "turns": len(r.turn_index),
             "stop_reason": r.stop_reason,
             "final_prompt_tokens": max((s["prompt_tokens"] for s in samples
                                         if s["session"] == r.index), default=0)}
            for r in runners
        ],
        **summarize(samples),
    }
    (config.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("sessions", "request_log_merged", "wall_seconds")},
                     indent=2))
    print_table(summary)
    if not samples:
        print("error: no turn was measured (every session stopped before its first request; "
              "raise --max-context or check --min-context)", file=sys.stderr)
        return 1
    return 0 if all(s["error"] is None for s in samples) else 1


def _fmt(value: Any, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def print_table(summary: dict[str, Any]) -> None:
    rows = [(name, group) for name, group in summary["by_category"].items()]
    rows += [("copy", summary["copy"]), ("non_copy", summary["non_copy"]), ("all", summary["all"])]
    print(f"{'category':<14}{'turns':>6}{'out tok':>9}{'decode t/s':>11}{'ttft p50':>10}"
          f"{'overlap':>9}{'accept':>8}{'tok/rnd':>8}")
    for name, group in rows:
        if not group.get("turns"):
            continue
        print(f"{name:<14}{group['turns']:>6}{_fmt(group.get('output_tokens_sum'), 'd'):>9}"
              f"{_fmt(group.get('decode_tps'), '.1f'):>11}{_fmt(group.get('ttft_s_p50'), '.2f'):>10}"
              f"{_fmt(group.get('copy_overlap'), '.2f'):>9}"
              f"{_fmt(group.get('spec_acceptance'), '.2f'):>8}"
              f"{_fmt(group.get('spec_tokens_per_round'), '.2f'):>8}")
