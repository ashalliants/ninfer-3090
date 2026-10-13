#!/usr/bin/env python3
"""Drive a hosted multi-tenant chat workload through one running ninfer-serve.

Models what a provider serving many people sees, as opposed to one user's agent loop
(`run_agent_rotation.py`, `agent_replay`):

- N tenants, each holding one conversation at a time. Every conversation starts with one of a few
  shared "product" prompts (system text plus a tool catalogue, identical for every tenant on that
  product), then a private opening document, then grows turn by turn with appended tool output.
- A conversation that passes `--max-prompt` tokens ends and its tenant opens a new one.
- Occasionally a turn rewrites the latest tool output (`edit`, the client retried with other
  input) or goes back several turns and continues from there (`branch`, deep history edit).
- Bursty presets add fan-out: a tenant opens a new document and fires several sub-agent requests
  at once, all sharing the same new prefix (history + document), which tests sibling reuse.
- Arrivals: `poisson` gives each tenant an exponential think time between its reply and its next
  turn; `bursty` releases a random group of tenants together every burst period.
- One request in flight per conversation (a client waits for its reply), except fan-out children.

Every request records the server's prompt/cached tokens and the client TTFT, plus what an ideal
cache could have reused (see `hosted_analysis.py`, which classifies hits and losses). With
`--request-log` (the server's `--request-log-jsonl` file) each request is joined to its server
record for queue wait, binding/restore time, computed prefill and admission detail. `/v1/load` is
polled once a second for lane utilisation and `/metrics` is scraped before and after the run.

    python tools/bench/run_hosted_workload.py --preset hosted-32 --duration-s 900 \
        --request-log server-requests.jsonl --out runs/c8-h8

`--save-state` writes every open conversation at the end; `--resume-state FILE --resume-once`
sends exactly one append turn per saved conversation (used to measure a restart with
`--context-store`). Content is synthetic word filler from a seeded RNG, so runs are repeatable up to
the model's free-running replies (temperature 0).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest

WORDS = (
    "anchor bearing cable damper engine flange gasket housing impeller journal keeper lever "
    "manifold nozzle orifice piston quill rotor spindle tappet union valve washer yoke bracket "
    "coupling detent eccentric fulcrum governor hinge idler jib knuckle linkage mandrel nut "
    "outrigger pawl ratchet sprocket trunnion ledger invoice tenant region cluster replica shard "
    "deploy rollback quota latency budget incident alert metric window"
).split()
# Filler density measured on the Qwen3.8 tokenizer: about 1.35 tokens per word.
TOKENS_PER_WORD = 1.35


@dataclass(frozen=True)
class Preset:
    """One named workload shape. Every field is documented in tools/bench/README.md."""

    tenants: int
    products: tuple[int, ...]          # shared system+tools prompt size per product, tokens
    opening: tuple[int, int]           # private opening document, tokens (log-uniform)
    turn: tuple[int, int]              # appended tool output per turn, tokens (log-uniform)
    max_prompt: int                    # a conversation ends past this many prompt tokens
    think_s: float                     # mean think time between a reply and the next turn
    reply_tokens: int                  # max output tokens with thinking off
    edit_p: float                      # probability a turn rewrites the latest tool output
    branch_p: float                    # probability a turn goes back 2-6 turns
    arrivals: str = "poisson"          # poisson | bursty
    burst_period_s: float = 60.0
    burst_size: int = 8
    fanout_p: float = 0.0              # probability a turn is a sub-agent fan-out
    fanout_width: int = 3
    fanout_doc: tuple[int, int] = (4000, 16000)


PRESETS: dict[str, Preset] = {
    "smoke": Preset(tenants=4, products=(1500, 3000), opening=(1000, 4000), turn=(300, 1500),
                    max_prompt=16000, think_s=2.0, reply_tokens=64, edit_p=0.1, branch_p=0.1),
    "hosted-16": Preset(tenants=16, products=(2000, 6000, 12000), opening=(2000, 40000),
                        turn=(300, 4000), max_prompt=100000, think_s=20.0, reply_tokens=160,
                        edit_p=0.05, branch_p=0.05),
    "hosted-32": Preset(tenants=32, products=(2000, 6000, 12000), opening=(2000, 40000),
                        turn=(300, 4000), max_prompt=100000, think_s=40.0, reply_tokens=160,
                        edit_p=0.05, branch_p=0.05),
    "hosted-64": Preset(tenants=64, products=(2000, 6000, 12000), opening=(2000, 40000),
                        turn=(300, 4000), max_prompt=100000, think_s=80.0, reply_tokens=160,
                        edit_p=0.05, branch_p=0.05),
    "bursty-32": Preset(tenants=32, products=(2000, 6000, 12000), opening=(2000, 40000),
                        turn=(300, 4000), max_prompt=100000, think_s=40.0, reply_tokens=160,
                        edit_p=0.05, branch_p=0.05, arrivals="bursty", burst_period_s=45.0,
                        burst_size=8, fanout_p=0.15, fanout_width=3),
}

THINKING_MAX_TOKENS = 640
THINKING_BUDGET = 384


def log_uniform(rng: random.Random, bounds: tuple[int, int]) -> int:
    low, high = bounds
    return int(math.exp(rng.uniform(math.log(low), math.log(high))))


def filler(rng: random.Random, tokens: int) -> str:
    words = max(1, int(tokens / TOKENS_PER_WORD))
    lines = []
    for start in range(0, words, 16):
        lines.append(" ".join(rng.choice(WORDS) for _ in range(min(16, words - start))) + ".")
    return "\n".join(lines)


def product_prompt(index: int, tokens: int) -> str:
    rng = random.Random(f"product-{index}")
    tools = "\n".join(
        f"- tool `{rng.choice(WORDS)}_{rng.choice(WORDS)}_{n}`: {filler(rng, 60)}"
        for n in range(max(1, tokens // 400)))
    head = (f"You are the assistant for product {index}. Answer tersely, in at most three "
            "sentences, and never repeat the tool output back.\nAvailable tools:\n")
    return head + tools + "\nReference manual:\n" + filler(rng, tokens // 2)


@dataclass
class Turn:
    """Messages one turn added and the token offset where they start in the prompt."""

    messages: list[dict[str, Any]]
    start: int = 0       # tokens before this turn's messages (estimated from server usage)
    endpoint: int = 0    # prompt + reply tokens once answered (0 while in flight or failed)


@dataclass
class Conversation:
    tenant: int
    serial: int
    product: int
    turns: list[Turn] = field(default_factory=list)
    prompt_tokens: int = 0

    @property
    def key(self) -> str:
        return f"t{self.tenant}c{self.serial}"

    def messages(self, system: str) -> list[dict[str, Any]]:
        out = [{"role": "system", "content": system}]
        for turn in self.turns:
            out.extend(turn.messages)
        return out


@dataclass
class Record:
    request: int
    tenant: int
    conversation: str
    product: int
    kind: str                   # first | append | edit | branch | fanout | resume
    turn: int
    submit_s: float             # seconds since run start
    done_s: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    client_ttft_s: float | None = None
    total_s: float = 0.0
    response_id: str | None = None
    error: str | None = None
    # What an ideal cache could have reused, in tokens (estimates from usage, see analysis).
    shared_tokens: int = 0      # this product's shared prefix
    private_ideal: int = 0      # deepest completed prefix of this conversation lineage
    inflight_ideal: int = 0     # deepest prefix including requests still in flight (siblings)
    source_age_s: float | None = None   # time since the private source was produced
    fanout_group: str | None = None
    rejections: int = 0         # 429 server_overloaded answers before admission


class LoadPoller(threading.Thread):
    def __init__(self, base_url: str, api_key: str | None, out: Path) -> None:
        super().__init__(daemon=True)
        self.url = base_url.rstrip("/") + "/v1/load"
        self.api_key = api_key
        self.out = out
        self.stop = threading.Event()

    def run(self) -> None:
        with self.out.open("w", encoding="utf-8") as log:
            while not self.stop.wait(1.0):
                try:
                    load = fetch_json(self.url, self.api_key)
                except Exception as error:  # noqa: BLE001 -- a missed poll is recorded, not fatal
                    load = {"error": str(error)}
                load["t"] = time.time()
                log.write(json.dumps(load) + "\n")
                log.flush()


def fetch_text(url: str, api_key: str | None) -> str:
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8")


def fetch_json(url: str, api_key: str | None) -> dict[str, Any]:
    return json.loads(fetch_text(url, api_key))


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


class Workload:
    def __init__(self, args: argparse.Namespace, preset: Preset) -> None:
        self.args = args
        self.preset = preset
        self.client_args = (args.base_url, args.timeout_seconds, args.api_key)
        self.model = NInferServeClient(*self.client_args).discover_model()
        self.products = [product_prompt(i, size) for i, size in enumerate(preset.products)]
        self.lock = threading.Lock()
        self.records: list[Record] = []
        self.started = time.time()
        self.deadline = self.started + args.duration_s
        self.next_request = 0
        self.serials: dict[int, int] = {}
        # Shared-prefix bookkeeping: product -> (first submit time, first completion time).
        self.product_seen: dict[int, float] = {}
        self.product_done: dict[int, float] = {}
        self.shared_tokens: dict[int, int] = {}
        self.endpoints_done: dict[str, float] = {}
        self.log = open(args.out / "requests.jsonl", "w", encoding="utf-8")

    def now(self) -> float:
        return time.time() - self.started

    def new_conversation(self, tenant: int, rng: random.Random) -> Conversation:
        serial = self.serials.get(tenant, 0)
        self.serials[tenant] = serial + 1
        product = tenant % len(self.products)
        conv = Conversation(tenant, serial, product)
        opening = filler(rng, log_uniform(rng, self.preset.opening))
        conv.turns.append(Turn([{"role": "user", "content":
                                 f"Tenant {tenant} conversation {serial}. Here is the case file; "
                                 f"wait for tool output before answering.\n{opening}"}]))
        return conv

    def tool_turn(self, rng: random.Random, label: str) -> list[dict[str, Any]]:
        body = filler(rng, log_uniform(rng, self.preset.turn))
        return [{"role": "user", "content": f"Tool output ({label}):\n{body}\n"
                 "Summarise the fault in one sentence."}]

    # ----- one request --------------------------------------------------------------------

    def payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        thinking = self.args.thinking == "on"
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "enable_thinking": thinking,
            "max_completion_tokens": THINKING_MAX_TOKENS if thinking else self.preset.reply_tokens,
        }
        if thinking:
            payload["thinking_budget"] = THINKING_BUDGET
        return payload

    def send(self, record: Record, messages: list[dict[str, Any]]) -> str:
        client = NInferServeClient(*self.client_args)
        request = ProtocolRequest(protocol="openai_chat", path="/v1/chat/completions",
                                  payload=self.payload(messages), stream=True)
        while True:
            sent: list[int] = []
            result = client.prepare(request).execute(on_sent=sent.append)
            # 429/529 server_overloaded: the ingress bound is full. A gateway would hold the
            # request; do the same and count it, with the TTFT clock left running.
            if result.http.status in (429, 529) and self.now() < self.args.duration_s + 600:
                record.rejections += 1
                time.sleep(1.0)
                continue
            break
        content: list[str] = []
        usage: dict[str, Any] = {}
        first_ns = None
        for event in result.events:
            payload = event.payload or {}
            record.response_id = record.response_id or event.response_id
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
            if event.kind == "model_output" and first_ns is None:
                first_ns = event.received_ns
        start = sent[0] if sent else result.http.sent_ns
        end = result.http.ended_ns or time.perf_counter_ns()
        error = result.protocol_error or result.error_code or result.http.error
        if result.http.status not in (None, 200) and error is None:
            error = f"http {result.http.status}"
        record.error = error
        record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        record.cached_tokens = int((usage.get("prompt_tokens_details") or {})
                                   .get("cached_tokens") or 0)
        record.output_tokens = int(usage.get("completion_tokens") or 0)
        record.client_ttft_s = (first_ns - start) / 1e9 if first_ns and start else None
        record.total_s = (end - start) / 1e9 if start else 0.0
        if record.rejections and first_ns:
            # TTFT as the tenant saw it: from its first attempt, retries included.
            record.client_ttft_s = (time.time() - self.started - record.submit_s) - \
                (end - first_ns) / 1e9
        record.done_s = self.now()
        return "".join(content)

    def make_record(self, conv: Conversation, kind: str, private_ideal: int,
                    source_time: float | None, group: str | None = None) -> Record:
        with self.lock:
            index = self.next_request
            self.next_request += 1
            now = self.now()
            shared = self.shared_tokens.get(conv.product, 0)
            # A shared prefix is reusable once any earlier request of the product completed;
            # in flight counts toward the single-flight ceiling only.
            done = self.product_done.get(conv.product)
            seen = self.product_seen.get(conv.product)
            shared_done = shared if done is not None else 0
            shared_inflight = shared if seen is not None else 0
            self.product_seen.setdefault(conv.product, now)
        record = Record(request=index, tenant=conv.tenant, conversation=conv.key,
                        product=conv.product, kind=kind, turn=len(conv.turns) - 1,
                        submit_s=now, shared_tokens=shared,
                        private_ideal=max(private_ideal, shared_done),
                        inflight_ideal=max(private_ideal, shared_inflight),
                        source_age_s=(now - source_time) if source_time is not None else None,
                        fanout_group=group)
        return record

    def finish(self, record: Record) -> None:
        with self.lock:
            self.records.append(record)
            if record.error is None:
                self.product_done.setdefault(record.product, record.done_s)
            self.log.write(json.dumps(asdict(record)) + "\n")
            self.log.flush()
        print(f"[{record.done_s:7.1f}s] req={record.request} {record.conversation} "
              f"{record.kind:<6} prompt={record.prompt_tokens} cached={record.cached_tokens} "
              f"ideal={record.private_ideal} ttft={_fmt(record.client_ttft_s)} "
              f"err={record.error}", file=sys.stderr, flush=True)

    def calibrate_shared(self) -> None:
        """Token count of each product prompt, from the server's own count (no generation)."""
        for index, system in enumerate(self.products):
            body = json.dumps({"model": self.model, "system": system, "max_tokens": 1,
                               "messages": [{"role": "user", "content": "x"}]}).encode()
            request = urllib.request.Request(
                self.args.base_url.rstrip("/") + "/v1/messages/count_tokens", data=body,
                headers={"Content-Type": "application/json"})
            if self.args.api_key:
                request.add_header("x-api-key", self.args.api_key)
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(json.loads(response.read())["input_tokens"])
            # Minus the trailing user turn and generation prompt (about 10 tokens).
            self.shared_tokens[index] = max(0, total - 10)

    # ----- one tenant ---------------------------------------------------------------------

    def answer(self, conv: Conversation, record: Record, messages: list[dict[str, Any]]) -> bool:
        reply = self.send(record, messages)
        self.finish(record)
        if record.error is not None:
            return False
        turn = conv.turns[-1]
        if self.args.thinking == "on":
            # The template drops closed reasoning from history: the reusable point is the input
            # boundary before the reply, not the endpoint.
            turn.endpoint = record.prompt_tokens
        else:
            turn.endpoint = record.prompt_tokens + record.output_tokens
        # An assistant turn's `start` is the reusable point of the exchange it closes.
        conv.turns.append(Turn([{"role": "assistant", "content": reply}], start=turn.endpoint))
        conv.prompt_tokens = record.prompt_tokens
        with self.lock:
            self.endpoints_done[conv.key] = record.done_s
        return True

    def step(self, conv: Conversation, rng: random.Random) -> Conversation | None:
        """Run one turn of `conv`; return the conversation to continue, None to open a new one."""
        system = self.products[conv.product]
        last_done = self.endpoints_done.get(conv.key)
        if conv.turns[-1].messages[0]["role"] != "assistant":
            record = self.make_record(conv, "first", 0, None)
            ok = self.answer(conv, record, conv.messages(system))
            return conv if ok else None
        if conv.prompt_tokens >= self.preset.max_prompt:
            return None
        roll = rng.random()
        if roll < self.preset.fanout_p:
            self.fanout(conv, rng, system)
            return conv
        roll -= self.preset.fanout_p
        # Turns alternate user/assistant and end with an assistant turn here.
        if roll < self.preset.edit_p and len(conv.turns) >= 4:
            del conv.turns[-2:]          # replace the latest tool output (and its reply)
            kind = "edit"
        elif roll < self.preset.edit_p + self.preset.branch_p and len(conv.turns) >= 8:
            back = rng.randint(2, min(6, len(conv.turns) // 2 - 1))
            del conv.turns[-2 * back:]   # go back `back` exchanges and continue from there
            kind = "branch"
        else:
            kind = "append"
        ideal = conv.turns[-1].start
        conv.turns.append(Turn(self.tool_turn(rng, f"turn {len(conv.turns)}"), start=ideal))
        record = self.make_record(conv, kind, ideal, last_done)
        if not self.answer(conv, record, conv.messages(system)):
            conv.turns.pop()
            return None
        return conv

    def fanout(self, conv: Conversation, rng: random.Random, system: str) -> None:
        """Several sub-agents at once over one new shared document; the parent does not grow."""
        doc = filler(rng, log_uniform(rng, self.preset.fanout_doc))
        base = conv.turns[-1].start
        group = f"{conv.key}f{len(conv.turns)}"
        doc_tokens = int(len(doc.split()) * TOKENS_PER_WORD)
        threads = []
        for child in range(self.preset.fanout_width):
            messages = conv.messages(system) + [{"role": "user", "content":
                                                f"Sub-task document:\n{doc}\nSub-agent {child}: "
                                                f"check item {rng.randint(1, 999)} only."}]
            record = self.make_record(conv, "fanout", base, self.endpoints_done.get(conv.key),
                                      group)
            # Siblings share history + document: with ideal single-flight only the first one
            # prefills the document.
            record.inflight_ideal = base + (doc_tokens if child else 0)

            def run(record: Record = record, messages: list[dict[str, Any]] = messages) -> None:
                self.send(record, messages)
                self.finish(record)

            thread = threading.Thread(target=run)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    def tenant_loop(self, tenant: int, gate: threading.Event | None) -> None:
        rng = random.Random(self.args.seed * 1000 + tenant)
        # Spread the first arrivals over one think time.
        if not self.wait(rng.uniform(0, self.preset.think_s), gate):
            return
        conv = self.new_conversation(tenant, rng)
        while self.now() < self.args.duration_s:
            result = self.step(conv, rng)
            conv = result if result is not None else self.new_conversation(tenant, rng)
            with self.lock:
                self.open_conversations[tenant] = conv
            if not self.wait(rng.expovariate(1.0 / self.preset.think_s), gate):
                return

    def wait(self, seconds: float, gate: threading.Event | None) -> bool:
        if gate is not None:
            # Bursty: wait for this tenant's group to be released.
            while not gate.wait(1.0):
                if self.now() >= self.args.duration_s:
                    return False
            gate.clear()
            return self.now() < self.args.duration_s
        end = time.time() + seconds
        while time.time() < end:
            if self.now() >= self.args.duration_s:
                return False
            time.sleep(min(1.0, end - time.time()))
        return True

    def run(self) -> None:
        self.open_conversations: dict[int, Conversation] = {}
        threads = []
        gates: list[threading.Event] | None = None
        if self.preset.arrivals == "bursty":
            gates = [threading.Event() for _ in range(self.preset.tenants)]
        for tenant in range(self.preset.tenants):
            gate = gates[tenant] if gates else None
            thread = threading.Thread(target=self.tenant_loop, args=(tenant, gate), daemon=True)
            thread.start()
            threads.append(thread)
        if gates:
            rng = random.Random(self.args.seed)
            while self.now() < self.args.duration_s:
                for index in rng.sample(range(self.preset.tenants), self.preset.burst_size):
                    gates[index].set()
                time.sleep(self.preset.burst_period_s)
        for thread in threads:
            thread.join()

    def resume_once(self, state: list[dict[str, Any]]) -> None:
        """One append turn per saved conversation, all submitted at a steady trickle."""
        threads = []
        for saved in state:
            conv = Conversation(saved["tenant"], saved["serial"], saved["product"],
                                [Turn(**t) for t in saved["turns"]], saved["prompt_tokens"])
            rng = random.Random(f"resume-{conv.key}")

            def run(conv: Conversation = conv, rng: random.Random = rng) -> None:
                system = self.products[conv.product]
                ideal = conv.turns[-1].start
                conv.turns.append(Turn(self.tool_turn(rng, "after restart"), start=ideal))
                record = self.make_record(conv, "resume", ideal, None)
                self.answer(conv, record, conv.messages(system))

            thread = threading.Thread(target=run)
            thread.start()
            threads.append(thread)
            time.sleep(self.args.resume_gap_s)
        for thread in threads:
            thread.join()


def _fmt(value: float | None) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="hosted-32")
    parser.add_argument("--tenants", type=int, help="override the preset's tenant count")
    parser.add_argument("--think-s", type=float, help="override the preset's mean think time")
    parser.add_argument("--thinking", choices=("on", "off"), default="off")
    parser.add_argument("--duration-s", type=float, default=900.0,
                        help="stop issuing new turns after this long (in-flight ones finish)")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--request-log", type=Path,
                        help="the server's --request-log-jsonl file, joined by response id")
    parser.add_argument("--save-state", action="store_true",
                        help="write open conversations to OUT/state.json at the end")
    parser.add_argument("--resume-state", type=Path, help="state.json from an earlier run")
    parser.add_argument("--resume-once", action="store_true",
                        help="with --resume-state: one append turn per saved conversation")
    parser.add_argument("--resume-gap-s", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    overrides = {}
    if args.tenants:
        overrides["tenants"] = args.tenants
    if args.think_s:
        overrides["think_s"] = args.think_s
    if overrides:
        preset = Preset(**{**asdict(preset), **overrides})
    args.out.mkdir(parents=True, exist_ok=True)
    log_offset = args.request_log.stat().st_size if args.request_log else 0
    metrics_url = args.base_url.rstrip("/") + "/metrics"
    before = parse_metrics(fetch_text(metrics_url, args.api_key))
    workload = Workload(args, preset)
    workload.calibrate_shared()
    poller = LoadPoller(args.base_url, args.api_key, args.out / "load.jsonl")
    poller.start()
    started = time.time()
    if args.resume_state:
        if not args.resume_once:
            print("error: --resume-state needs --resume-once", file=sys.stderr)
            return 2
        workload.resume_once(json.loads(args.resume_state.read_text(encoding="utf-8")))
    else:
        workload.run()
    wall = time.time() - started
    poller.stop.set()
    poller.join()
    after = parse_metrics(fetch_text(metrics_url, args.api_key))
    workload.log.close()
    if args.save_state and not args.resume_state:
        state = [{"tenant": c.tenant, "serial": c.serial, "product": c.product,
                  "prompt_tokens": c.prompt_tokens, "turns": [asdict(t) for t in c.turns]}
                 for c in workload.open_conversations.values()
                 if c.turns and c.turns[-1].messages[0]["role"] == "assistant"]
        (args.out / "state.json").write_text(json.dumps(state), encoding="utf-8")
    server_records: dict[str, Any] = {}
    if args.request_log:
        with args.request_log.open("rb") as log:
            log.seek(log_offset)
            for raw in log:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "request_done":
                    rid = (record.get("request") or {}).get("response_id")
                    if rid:
                        record["result"].pop("generated_token_ids", None)
                        server_records[rid] = record
    (args.out / "server_records.json").write_text(json.dumps(server_records), encoding="utf-8")
    summary = {
        "preset": args.preset, "preset_fields": asdict(preset), "thinking": args.thinking,
        "duration_s": args.duration_s, "wall_s": wall, "seed": args.seed,
        "shared_tokens": workload.shared_tokens, "requests": len(workload.records),
        "errors": sum(1 for r in workload.records if r.error),
        "joined_server_records": sum(1 for r in workload.records
                                     if r.response_id in server_records),
        "metrics_before": before, "metrics_after": after,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("requests", "errors", "joined_server_records",
                                              "wall_s")}))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
