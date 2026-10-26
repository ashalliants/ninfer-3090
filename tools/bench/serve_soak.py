#!/usr/bin/env python3
# Adapted from Basalt 8d73a39 tools/serve_soak.py (MIT, third_party/basalt/LICENSE).
"""Randomised soak of a running ninfer-serve: mixed traffic for minutes or hours, flagging what breaks.

Adapted in shape from Basalt's tools/serve_soak.py (8d73a39, MIT, third_party/basalt/LICENSE), rebuilt on
tools/streaming_http so requests are timed and cancelled the way the other bench tools do.

The traffic is a seeded random mix of

  chat      OpenAI /v1/chat/completions and Anthropic /v1/messages streams
  cancel    a stream dropped after a few events, or a few hundred ms after it was sent
  near      a prompt that fills the context except for the reply (expect 200)
  over      a prompt longer than the context (expect HTTP 400 context_length_exceeded)
  oversize  a body above --max-request-mib (expect HTTP 413; a reset while uploading is accepted and counted)
  image     base64 data-URL images from --images DIR (a server without --vision answers 400
            vision_disabled, which stops the image requests and is not a failure)

with 1..--max-concurrency requests in flight. Every --sample-seconds it reads /v1/load, /metrics and /slots
and the server's resident memory (--pid) and keeps them in the report.

It exits non-zero when any of these happened (exit codes: 0 clean, 1 findings, 2 could not start):

  - a response status or body that the request kind does not expect, a stream that ended without its
    terminator, an error event inside a stream, or a transport error;
  - a stream with no SSE event for --stall-seconds (counted from the send until the first event, then from
    the last event); the stalled request is cancelled so the run can finish;
  - resident memory that grew by more than --rss-growth-mib over the last half of the run;
  - a server that stopped answering its sampling endpoints (3 rounds in a row).

The soak sends a normal Host header, no Origin and no Sec-Fetch-*, and the key (if any) as a Bearer token, so
it passes the server's Host/Origin checks. The card is busy for the whole run: hold the GPU lock for the
server's lifetime (CLAUDE.md "GPU lock"). See tools/bench/README.md.

Only the Python standard library and tools/streaming_http are used (psutil is used for the server's memory
on Windows when it is installed).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.streaming_http.client import HttpClientError, StreamingHttpClient  # noqa: E402
from tools.streaming_http.sse import SseDecodeError, SseDecoder  # noqa: E402

SCHEMA_VERSION = 1
NS = 1_000_000_000
KINDS = ("chat", "cancel", "near", "over", "oversize", "image")
# Kinds that cost a lot (a context-sized prefill, a 384 MiB body) are rationed per run.
RATIONED = ("near", "over", "oversize")
WEIGHTS = {"chat": 0.70, "cancel": 0.13, "near": 0.03, "over": 0.04, "oversize": 0.01, "image": 0.09}
EXIT_CLEAN, EXIT_FINDINGS, EXIT_USAGE = 0, 1, 2
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
               ".gif": "image/gif"}
MAX_IMAGE_BYTES = 4 << 20

TOPICS = (
    "Write a Python module with an LRU cache, a rate limiter and a retry decorator, with docstrings.",
    "Explain how the immune system learns from vaccines, for an educated reader.",
    "Write a long narrative about a lighthouse keeper during a winter storm.",
    "Produce a JSON array of 20 customer records with id, name, email and a list of orders. JSON only.",
    "Work through several definite integrals step by step, explaining each substitution.",
    "Compare B-trees and LSM trees for a write-heavy workload, with a table of trade-offs.",
)
FILLER = ("The committee reviewed the quarterly figures and noted that several regional offices had exceeded "
          "their targets, while others reported delays caused by supply chain disruptions, staffing changes "
          "and unusually severe weather. ")


# ----------------------------------------------------------------------------------------------
# Server memory


def make_rss_reader(pid: int | None, proc_root: str = "/proc") -> tuple[Callable[[], float | None] | None, str]:
    """A function returning the process's resident MiB, and a note when there is none.

    Linux reads VmRSS from <proc_root>/<pid>/status. Elsewhere psutil is used when it can be imported;
    otherwise there is no reader and the note says why (nothing is installed or required).
    """
    if not pid:
        return None, "no --pid: server memory is not sampled"
    status = Path(proc_root) / str(pid) / "status"
    if status.is_file():
        def read_proc() -> float | None:
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0  # kB -> MiB
            except (OSError, ValueError, IndexError):
                pass
            return None
        return read_proc, ""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None, (f"server memory is not sampled: {status} does not exist (not Linux) and psutil is not "
                      "installed")
    try:
        process = psutil.Process(pid)
    except Exception as error:  # noqa: BLE001 - psutil.NoSuchProcess and friends
        return None, f"server memory is not sampled: psutil cannot open pid {pid}: {error}"

    def read_psutil() -> float | None:
        try:
            return process.memory_info().rss / (1 << 20)
        except Exception:  # noqa: BLE001
            return None
    return read_psutil, ""


def rss_growth_mib(series: Sequence[float]) -> float | None:
    """Growth over the last half of the samples: the median of the half's last fifth minus its first fifth.

    Medians keep one noisy sample from deciding the verdict. None when there are too few samples to say.
    """
    if len(series) < 4:
        return None
    half = list(series[len(series) // 2:])
    k = max(1, len(half) // 5)
    return statistics.median(half[-k:]) - statistics.median(half[:k])


# ----------------------------------------------------------------------------------------------
# Wire helpers


def parse_error(body: bytes) -> tuple[str | None, str | None, str]:
    """(code, type, message) of an OpenAI- or Anthropic-shaped error body; None where absent."""
    try:
        obj = json.loads(body)
    except ValueError:
        return None, None, body[:200].decode("utf-8", errors="replace")
    error = obj.get("error") if isinstance(obj, dict) else None
    if not isinstance(error, dict):
        return None, None, str(obj)[:200]
    return error.get("code"), error.get("type"), str(error.get("message", ""))[:200]


def parse_metrics(text: str) -> dict[str, float]:
    """The unlabelled ninfer_* series of a Prometheus text body (labelled ones are many and large)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("ninfer_") or "{" in line:
            continue
        name, _, value = line.partition(" ")
        try:
            out[name] = float(value.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def chat_messages(protocol: str, text: str, image: tuple[str, bytes] | None) -> list[dict[str, Any]]:
    if image is None:
        return [{"role": "user", "content": text}]
    media_type, data = image
    b64 = base64.b64encode(data).decode("ascii")
    if protocol == "openai":
        parts = [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                 {"type": "text", "text": text}]
    else:
        parts = [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                 {"type": "text", "text": text}]
    return [{"role": "user", "content": parts}]


# ----------------------------------------------------------------------------------------------
# One request


@dataclass
class Request:
    id: int
    kind: str
    protocol: str                      # "openai" | "anthropic"
    expect: str                        # what a pass looks like, for the report
    sent_ns: int | None = None
    last_event_ns: int | None = None
    status: int | None = None
    streaming: bool = False
    events: int = 0
    terminated: bool = False
    error: str | None = None
    code: str | None = None
    cancel_after_events: int | None = None
    cancelled_by_us: bool = False
    stalled: bool = False
    done: bool = False
    outcome: str = ""                  # "ok" | "ok_reset" | "skipped_no_vision" | "cancelled" | "failed"
    failure: str | None = None
    exchange: Any = None
    started_s: float = 0.0
    seconds: float = 0.0

    def cancel(self) -> None:
        self.cancelled_by_us = True
        if self.exchange is not None:
            self.exchange.cancel()

    def on_event(self, event: Any) -> None:
        self.events += 1
        self.last_event_ns = event.received_ns
        if self.protocol == "openai":
            if event.data == "[DONE]":
                self.terminated = True
                return
            try:
                payload = json.loads(event.data)
            except ValueError:
                self.error = self.error or f"non-JSON stream event: {event.data[:80]!r}"
                return
            if isinstance(payload, dict) and payload.get("error"):
                self.error = self.error or f"error event: {str(payload['error'])[:200]}"
        else:
            if event.event == "error":
                self.error = self.error or f"error event: {event.data[:200]}"
            elif event.event == "message_stop":
                self.terminated = True
        if self.cancel_after_events is not None and self.events >= self.cancel_after_events:
            self.cancel()


class Soak:
    """State shared by the traffic threads and the sampler."""

    def __init__(self, args: argparse.Namespace, rss_reader: Callable[[], float | None] | None,
                 rss_note: str) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.client = StreamingHttpClient(args.url, args.timeout_seconds)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.requests: list[Request] = []
        self.samples: list[dict[str, Any]] = []
        self.stalls: list[dict[str, Any]] = []
        self.model = "ninfer"
        self.context = args.context or 8192
        self.tokens_per_char = 0.25
        self.vision_enabled = True
        self.images: list[tuple[str, bytes]] = []
        self.rss_reader = rss_reader
        self.rss_note = rss_note
        self.sample_failures = 0
        self.server_lost = False
        self.t0 = time.monotonic()

    # -- plain GETs --------------------------------------------------------------------------

    def headers(self, protocol: str = "openai", accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.args.api_key:
            headers["Authorization"] = f"Bearer {self.args.api_key}"
        if protocol == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def get(self, path: str) -> tuple[int | None, bytes]:
        try:
            result = self.client.prepare("GET", path, b"", self.headers()).execute()
        except (OSError, HttpClientError) as error:
            return None, str(error).encode()
        return result.status, result.body

    def get_json(self, path: str) -> Any:
        status, body = self.get(path)
        if status != 200:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return None

    # -- setup --------------------------------------------------------------------------------

    def preflight(self) -> str | None:
        """None when the server answers; otherwise why the soak cannot start."""
        status, body = self.get("/v1/models")
        if status is None:
            return f"cannot reach {self.args.url}: {body.decode(errors='replace')}"
        if status != 200:
            return f"GET /v1/models returned HTTP {status}: {body[:200].decode(errors='replace')}"
        try:
            self.model = json.loads(body)["data"][0]["id"]
        except (ValueError, KeyError, IndexError, TypeError):
            pass  # keep the default; the first request will say if it is wrong
        slots = self.get_json("/slots")
        if not self.args.context and isinstance(slots, list):
            contexts = [slot.get("n_ctx") for slot in slots if isinstance(slot, dict)]
            contexts = [value for value in contexts if isinstance(value, int) and value > 0]
            if contexts:
                self.context = max(contexts)
        self.calibrate()
        return None

    def count_tokens(self, text: str) -> int | None:
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": text}]}).encode()
        headers = self.headers("anthropic")
        headers["Content-Type"] = "application/json"
        try:
            result = self.client.prepare("POST", "/v1/messages/count_tokens", body, headers).execute()
        except (OSError, HttpClientError):
            return None
        if result.status != 200:
            return None
        try:
            value = json.loads(result.body)["input_tokens"]
        except (ValueError, KeyError, TypeError):
            return None
        return value if isinstance(value, int) and value > 0 else None

    def calibrate(self) -> None:
        sample = FILLER * 20
        tokens = self.count_tokens(sample)
        if tokens:
            self.tokens_per_char = tokens / len(sample)

    def prompt_of_tokens(self, target: int) -> str:
        """About `target` prompt tokens of filler; measured with count_tokens and corrected once."""
        chars = max(len(FILLER), int(target / self.tokens_per_char))
        text = (FILLER * (chars // len(FILLER) + 1))[:chars]
        measured = self.count_tokens(text)
        if measured and abs(measured - target) > max(8, target // 100):
            chars = max(len(FILLER), int(chars * target / measured))
            text = (FILLER * (chars // len(FILLER) + 1))[:chars]
        return text

    # -- requests -----------------------------------------------------------------------------

    def build(self, req: Request) -> tuple[str, bytes, bool]:
        """(path, body, wants_stream) for the request, drawing its randomness from the soak's RNG."""
        rng = self.rng
        kind = req.kind
        text = rng.choice(TOPICS)
        if rng.random() < 0.5:
            text = f"[{rng.getrandbits(40):010x}] {text}"  # unique prefix: no cache hit for this one
        max_tokens = rng.choice((32, 128, 400, 1200))
        image = None
        if kind == "cancel":
            max_tokens = 1500
            if rng.random() < 0.7:
                req.cancel_after_events = rng.randint(1, 40)
            else:
                delay = rng.uniform(0.05, 1.5)
                timer = threading.Timer(delay, req.cancel)
                timer.daemon = True
                timer.start()
        elif kind == "near":
            max_tokens = 128
            text = self.prompt_of_tokens(max(64, self.context - max_tokens - 256)) + "\n\nSummarize the passage."
        elif kind == "over":
            text = self.prompt_of_tokens(self.context + 2000) + "\n\nSummarize."
        elif kind == "image":
            image = rng.choice(self.images)
            text = "Describe this image in a few sentences."
            max_tokens = 200
        if kind == "oversize":
            padding = (self.args.max_request_mib + 1) << 20
            head = json.dumps({"model": self.model, "max_tokens": 1, "stream": True,
                               "messages": [{"role": "user", "content": ""}]}).encode()
            cut = head.rindex(b'""') + 1
            return "/v1/chat/completions", head[:cut] + b"a" * padding + head[cut:], True
        if req.protocol == "openai":
            body = {"model": self.model, "messages": chat_messages("openai", text, image),
                    "max_tokens": max_tokens, "temperature": 0.7, "stream": True,
                    "stream_options": {"include_usage": True}}
            return "/v1/chat/completions", json.dumps(body).encode(), True
        body = {"model": self.model, "messages": chat_messages("anthropic", text, image),
                "max_tokens": max_tokens, "temperature": 0.7, "stream": True}
        return "/v1/messages", json.dumps(body).encode(), True

    def run_request(self, req: Request) -> None:
        started = time.monotonic()
        req.started_s = started - self.t0
        result = None
        try:
            path, body, _ = self.build(req)
            headers = self.headers(req.protocol, "text/event-stream")
            headers["Content-Type"] = "application/json"
            req.exchange = self.client.prepare("POST", path, body, headers)
            decoder = SseDecoder()

            def on_sent(ns: int) -> None:
                req.sent_ns = req.last_event_ns = ns

            def on_headers(head: Any) -> None:
                req.status = head.status
                req.streaming = head.status == 200 and "text/event-stream" in head.headers.get("content-type", "")

            def on_chunk(block: bytes, ns: int) -> None:
                if not req.streaming:
                    return
                try:
                    for event in decoder.feed(block, ns):
                        req.on_event(event)
                except SseDecodeError as error:
                    req.error = req.error or str(error)

            if req.cancelled_by_us:  # the timer fired before the exchange existed
                req.exchange.cancel()
            result = req.exchange.execute(on_sent=on_sent, on_headers=on_headers, on_chunk=on_chunk)
            if req.streaming and result.error is None and not req.cancelled_by_us:
                for event in decoder.finish(time.perf_counter_ns()):
                    req.on_event(event)
        except (OSError, HttpClientError, SseDecodeError) as error:
            req.error = req.error or f"{type(error).__name__}: {error}"
        except Exception as error:  # noqa: BLE001 - a bug here must not kill the soak thread silently
            req.error = req.error or f"internal {type(error).__name__}: {error}"
        finally:
            req.seconds = time.monotonic() - started
            self.finish(req, result)

    def finish(self, req: Request, result: Any) -> None:
        body = result.body if result is not None else b""
        transport_error = result.error if result is not None else None
        code, error_type, message = parse_error(body) if req.status not in (None, 200) else (None, None, "")
        req.code = code
        outcome, failure = "ok", None
        if req.stalled:
            outcome, failure = "failed", "stalled"
        elif req.cancelled_by_us and req.kind == "cancel":
            outcome = "cancelled"
            if req.status not in (None, 200):
                outcome, failure = "failed", f"HTTP {req.status} {code or error_type or ''} {message}".strip()
            elif req.error:
                outcome, failure = "failed", req.error
        elif req.kind == "oversize":
            if req.status == 413:
                outcome = "ok"
            elif req.status is None and (transport_error or req.error) and not req.stalled:
                outcome = "ok_reset"  # the server may close the socket before it has read the body
            else:
                outcome, failure = "failed", f"expected HTTP 413, got {req.status} {code or ''} {message}".strip()
        elif req.kind == "over":
            right_code = code == "context_length_exceeded" if req.protocol == "openai" \
                else error_type == "invalid_request_error"
            if req.status == 400 and right_code:
                outcome = "ok"
            else:
                outcome, failure = "failed", (f"expected HTTP 400 context_length_exceeded, got {req.status} "
                                              f"{code or error_type or ''} {message}").strip()
        elif req.kind == "image" and req.status == 400 and code == "vision_disabled":
            outcome = "skipped_no_vision"
            self.vision_enabled = False
        else:
            if req.status != 200:
                shown = req.status if req.status is not None else "no response"
                detail = transport_error or req.error or f"{code or error_type or ''} {message}".strip()
                outcome, failure = "failed", f"expected HTTP 200, got {shown}: {detail}".strip()
            elif transport_error and not req.cancelled_by_us:
                outcome, failure = "failed", f"transport error: {transport_error}"
            elif req.error:
                outcome, failure = "failed", req.error
            elif not req.terminated and not req.cancelled_by_us:
                outcome, failure = "failed", f"stream ended after {req.events} events without its terminator"
        req.outcome, req.failure = outcome, failure
        req.done = True

    # -- traffic ------------------------------------------------------------------------------

    def launch(self, kind: str) -> Request:
        with self.lock:
            req = Request(id=len(self.requests) + 1, kind=kind,
                          protocol="openai" if self.rng.random() < 0.6 else "anthropic",
                          expect={"over": "400 context_length_exceeded", "oversize": "413"}.get(kind, "200"))
            if kind == "oversize":
                req.protocol = "openai"
            self.requests.append(req)
        threading.Thread(target=self.run_request, args=(req,), daemon=True).start()
        return req

    def pick(self, kinds: Sequence[str], used: dict[str, int], limits: dict[str, int]) -> str:
        pool = [k for k in kinds if not (k in RATIONED and used[k] >= limits[k])
                and not (k == "image" and (not self.images or not self.vision_enabled))]
        if not pool:
            return "chat" if "chat" in kinds else kinds[0]
        return self.rng.choices(pool, [WEIGHTS[k] for k in pool])[0]

    def traffic(self) -> None:
        args = self.args
        kinds = [k for k in KINDS if k in args.kinds]
        minutes = max(args.minutes, 0.0)
        limits = {k: int(minutes // 10) * 2 + 1 for k in RATIONED}
        used = {k: 0 for k in KINDS}
        # One of every enabled kind first, so a short run still covers each path.
        first = [k for k in kinds if not (k == "image" and not self.images)]
        self.rng.shuffle(first)
        target = self.rng.randint(1, args.max_concurrency)
        end = self.t0 + minutes * 60.0
        launched = 0
        while time.monotonic() < end and not self.stop.is_set():
            if args.max_requests and launched >= args.max_requests:
                break
            with self.lock:
                live = sum(1 for r in self.requests if not r.done)
            if self.rng.random() < 0.05:
                target = self.rng.randint(1, args.max_concurrency)
            if live < target:
                kind = first.pop() if first else self.pick(kinds, used, limits)
                if kind == "image" and (not self.images or not self.vision_enabled):
                    kind = "chat" if "chat" in kinds else kind
                used[kind] += 1
                self.launch(kind)
                launched += 1
            time.sleep(0.02)

    # -- sampling and stall detection -----------------------------------------------------------

    def check_stalls(self) -> None:
        now = time.perf_counter_ns()
        limit_ns = int(self.args.stall_seconds * NS)
        with self.lock:
            live = [r for r in self.requests if not r.done and not r.stalled and not r.cancelled_by_us
                    and r.last_event_ns is not None]
        for req in live:
            quiet = now - req.last_event_ns
            if quiet > limit_ns:
                req.stalled = True
                entry = {"id": req.id, "kind": req.kind, "protocol": req.protocol,
                         "phase": "before first event" if req.events == 0 else "mid-stream",
                         "events": req.events, "quiet_seconds": round(quiet / NS, 2)}
                with self.lock:
                    self.stalls.append(entry)
                print(f"STALL request {req.id} ({req.kind}, {req.protocol}): no event for "
                      f"{entry['quiet_seconds']} s, {entry['phase']}; cancelling", flush=True)
                req.cancel()

    def sample_once(self) -> None:
        sample: dict[str, Any] = {"t": round(time.monotonic() - self.t0, 2)}
        reached = 0
        load = self.get_json("/v1/load")
        if isinstance(load, dict):
            reached += 1
            requests = load.get("requests") or {}
            sample["queue"] = len(requests.get("queue") or [])
            sample["queue_total"] = requests.get("queue_total")
        status, body = self.get("/metrics")
        if status == 200:
            reached += 1
            sample["metrics"] = parse_metrics(body.decode("utf-8", errors="replace"))
        slots = self.get_json("/slots")
        if isinstance(slots, list):
            reached += 1
            sample["slots"] = len(slots)
            sample["slots_busy"] = sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing"))
            sample["ctx_used"] = sum(s.get("n_ctx_used") or 0 for s in slots if isinstance(s, dict))
        if self.rss_reader is not None:
            sample["rss_mib"] = self.rss_reader()
        with self.lock:
            live = [r for r in self.requests if not r.done]
            sample["in_flight"] = len(live)
            sample["requests"] = len(self.requests)
            sample["failed"] = sum(1 for r in self.requests if r.done and r.outcome == "failed")
            self.samples.append(sample)
        self.sample_failures = self.sample_failures + 1 if reached == 0 else 0
        if self.sample_failures >= 3 and not self.server_lost:
            self.server_lost = True
            print("server stopped answering /v1/load, /metrics and /slots; ending the run", flush=True)
            self.stop.set()
        print(json.dumps({k: v for k, v in sample.items() if k != "metrics"}), flush=True)

    def sampler(self) -> None:
        while not self.stop.wait(self.args.sample_seconds):
            self.check_stalls()
            self.sample_once()

    # -- the run ------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        args = self.args
        self.t0 = time.monotonic()
        started_wall = time.time()
        sampler = threading.Thread(target=self.sampler, daemon=True)
        sampler.start()
        self.traffic()
        # Drain: no new requests; the rest finish, or are cut after the grace period.
        deadline = time.monotonic() + args.drain_seconds
        while time.monotonic() < deadline and not self.server_lost:
            with self.lock:
                if all(r.done for r in self.requests):
                    break
            time.sleep(0.05)
        with self.lock:
            leftover = [r for r in self.requests if not r.done]
        for req in leftover:
            if not req.stalled:
                req.stalled = True
                with self.lock:
                    self.stalls.append({"id": req.id, "kind": req.kind, "protocol": req.protocol,
                                        "phase": "unfinished at the end of the drain", "events": req.events,
                                        "quiet_seconds": None})
            req.cancel()
        for _ in range(100):
            with self.lock:
                if all(r.done for r in self.requests):
                    break
            time.sleep(0.05)
        self.stop.set()
        sampler.join(timeout=args.sample_seconds + 5)
        self.sample_once()
        return self.report(started_wall)

    def report(self, started_wall: float) -> dict[str, Any]:
        args = self.args
        by_kind: dict[str, dict[str, int]] = {}
        failures = []
        for req in self.requests:
            bucket = by_kind.setdefault(req.kind, {})
            bucket[req.outcome or "unfinished"] = bucket.get(req.outcome or "unfinished", 0) + 1
            if req.outcome == "failed":
                failures.append({"id": req.id, "kind": req.kind, "protocol": req.protocol, "status": req.status,
                                 "code": req.code, "events": req.events, "expected": req.expect,
                                 "at_seconds": round(req.started_s, 1), "reason": req.failure})
        series = [s["rss_mib"] for s in self.samples if s.get("rss_mib") is not None]
        growth = rss_growth_mib(series)
        rss: dict[str, Any] = {"sampled": bool(series), "samples": len(series), "limit_mib": args.rss_growth_mib}
        if self.rss_note:
            rss["note"] = self.rss_note
        elif self.rss_reader is not None and growth is None:
            rss["note"] = "too few samples (need 4) for a growth verdict; run longer"
        if series:
            rss.update(first_mib=round(series[0], 1), last_mib=round(series[-1], 1),
                       max_mib=round(max(series), 1))
        rss["growth_last_half_mib"] = None if growth is None else round(growth, 1)
        rss["flagged"] = growth is not None and growth > args.rss_growth_mib
        findings = []
        if failures:
            findings.append(f"{len(failures)} request(s) with an unexpected result")
        if self.stalls:
            findings.append(f"{len(self.stalls)} stalled stream(s)")
        if rss["flagged"]:
            findings.append(f"server RSS grew {rss['growth_last_half_mib']} MiB over the last half "
                            f"(limit {args.rss_growth_mib})")
        if self.server_lost:
            findings.append("the server stopped answering")
        return {
            "schema_version": SCHEMA_VERSION,
            "url": args.url, "model": self.model, "context": self.context, "seed": args.seed,
            "minutes": args.minutes, "started_unix": round(started_wall, 1),
            "duration_seconds": round(time.monotonic() - self.t0, 1),
            "settings": {"kinds": list(args.kinds), "max_concurrency": args.max_concurrency,
                         "stall_seconds": args.stall_seconds, "rss_growth_mib": args.rss_growth_mib,
                         "sample_seconds": args.sample_seconds, "max_request_mib": args.max_request_mib,
                         "images": len(self.images)},
            "requests": {"total": len(self.requests), "by_kind": by_kind},
            "failures": failures, "stalls": self.stalls, "rss": rss,
            "samples": self.samples,
            "findings": findings, "exit_code": EXIT_FINDINGS if findings else EXIT_CLEAN,
        }


# ----------------------------------------------------------------------------------------------
# CLI


def load_images(directory: str) -> list[tuple[str, bytes]]:
    images = []
    for path in sorted(Path(directory).iterdir()):
        media_type = IMAGE_TYPES.get(path.suffix.lower())
        if media_type and path.is_file() and path.stat().st_size <= MAX_IMAGE_BYTES:
            images.append((media_type, path.read_bytes()))
    return images


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--minutes", type=float, required=True, help="how long to send new requests")
    p.add_argument("--api-key", default=None, help="sent as 'Authorization: Bearer'")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--images", default=None, metavar="DIR", help="directory of png/jpg/webp/gif files")
    p.add_argument("--out", required=True, metavar="report.json")
    p.add_argument("--pid", type=int, default=None, help="server process id, for its resident memory")
    p.add_argument("--rss-growth-mib", type=float, default=512.0,
                   help="fail when RSS grows by more than this over the last half of the run")
    p.add_argument("--stall-seconds", type=float, default=60.0, help="a stream with no event this long is stalled")
    p.add_argument("--sample-seconds", type=float, default=2.0)
    p.add_argument("--max-concurrency", type=int, default=4, help="most requests in flight at once")
    p.add_argument("--max-request-mib", type=int, default=384,
                   help="the server's --max-request-mib; an 'oversize' body is one MiB above it")
    p.add_argument("--context", type=int, default=0, help="context tokens (default: n_ctx from /slots)")
    p.add_argument("--kinds", default=",".join(KINDS),
                   help="comma list of request kinds to send: " + ", ".join(KINDS))
    p.add_argument("--max-requests", type=int, default=0, help="stop launching after this many (0: no limit)")
    p.add_argument("--drain-seconds", type=float, default=None,
                   help="how long to let in-flight requests finish after the time is up (default: stall + 30)")
    p.add_argument("--timeout-seconds", type=float, default=None,
                   help="socket timeout per read (default: stall + 60, so the stall check fires first)")
    args = p.parse_args(argv)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KINDS]
    if unknown or not kinds:
        p.error(f"--kinds: unknown {unknown or 'empty list'}; choose from {', '.join(KINDS)}")
    args.kinds = kinds
    if args.minutes < 0 or args.stall_seconds <= 0 or args.sample_seconds <= 0 or args.max_concurrency < 1:
        p.error("--minutes must be >= 0 and --stall-seconds, --sample-seconds, --max-concurrency positive")
    if args.drain_seconds is None:
        args.drain_seconds = args.stall_seconds + 30.0
    if args.timeout_seconds is None:
        args.timeout_seconds = args.stall_seconds + 60.0
    args.timeout_seconds = float(args.timeout_seconds)
    return args


def main(argv: Sequence[str] | None = None,
         rss_reader: Callable[[], float | None] | None = None) -> int:
    """Run the soak; `rss_reader` replaces the --pid reader (the self-test injects a growing one)."""
    args = parse_args(argv)
    note = ""
    if rss_reader is None:
        rss_reader, note = make_rss_reader(args.pid)
        if note:
            print(f"note: {note}", file=sys.stderr)
    soak = Soak(args, rss_reader, note)
    if args.images:
        try:
            soak.images = load_images(args.images)
        except OSError as error:
            print(f"error: --images: {error}", file=sys.stderr)
            return EXIT_USAGE
        if not soak.images:
            print(f"error: --images {args.images} holds no png/jpg/webp/gif of at most 4 MiB", file=sys.stderr)
            return EXIT_USAGE
    problem = soak.preflight()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return EXIT_USAGE
    print(f"soaking {args.url} model={soak.model} context={soak.context} for {args.minutes} min "
          f"seed={args.seed} kinds={','.join(args.kinds)}", flush=True)
    report = soak.run()
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("== requests by kind and outcome:")
    for kind, outcomes in sorted(report["requests"]["by_kind"].items()):
        print(f"   {kind:9s} {outcomes}")
    for failure in report["failures"][:20]:
        print(f"   FAILED #{failure['id']} {failure['kind']}/{failure['protocol']}: {failure['reason']}")
    rss = report["rss"]
    print(f"rss: {rss.get('note') or ''}growth over the last half {rss['growth_last_half_mib']} MiB "
          f"(limit {rss['limit_mib']})")
    if report["findings"]:
        print("SOAK FAILED: " + "; ".join(report["findings"]))
    else:
        print("soak clean")
    print(f"report: {args.out}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
