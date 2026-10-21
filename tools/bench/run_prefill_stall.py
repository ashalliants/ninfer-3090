#!/usr/bin/env python3
"""What does a long uncached prompt do to a stream that is decoding next to it?

Against a running ninfer-serve, this starts a streaming chat completion, lets it settle, submits
one long uncached prompt (non-streaming, max_tokens 1) and records the stream's token timestamps
around it. Optionally it also sends a short request part-way through the long prompt, and cancels
the long prompt and resubmits it. It answers the questions behind the prefill scheduling flags
(--prefill-stall-budget-ms, --prefill-decode-share, --yield-long-prefill-ms,
--prefill-anchor-tokens; see docs/maintainer/decode-during-long-prefill.md):

  stream      inter-event gaps before, during and after the long prompt, and events/s
  long        how long the prompt took and the prompt tokens the server reports for it
  short       the short request's time to first event (with --short-delay-seconds)
  retry       how long the same prompt took after a cancel (with --cancel-long-after-seconds)

The server must be started with enough lanes for the scenario: the stream and the long prompt
run at once, so --max-concurrency 2 or more; with --max-concurrency 1 the long prompt queues behind
the stream and the short request shows the head-of-line wait instead. Run each flag setting against
a fresh server so the prompt cache starts cold, and take the baseline (no flags) the same way.

Only the Python standard library and tools/streaming_http are used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.streaming_http.client import StreamingHttpClient  # noqa: E402
from tools.streaming_http.sse import SseDecoder  # noqa: E402

NS = 1_000_000_000
SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------------------------
# Pure analysis


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of `values` (0 <= fraction <= 1); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return float(ordered[rank - 1])


def summarize_gaps(stamps_ns: Sequence[int], start_ns: int, end_ns: int,
                   overlapping: bool = False) -> dict[str, Any]:
    """Gaps between consecutive events in [start_ns, end_ns], in milliseconds.

    By default a gap counts when both its events lie in the window. With `overlapping` it counts,
    whole, when any part of it does: that is how the window of the long prompt picks up the stall
    that begins at the last event before the prompt and ends at the first one after it. Using
    "inside" for the windows before and after and "overlapping" for the one between assigns every
    gap to exactly one of them. `open_gap_ms` is the time from the last event to the window's end,
    a lower bound on the stall when the stream was still blocked at the end.
    """
    stamps = sorted(stamps_ns)
    gaps_ms: list[float] = []
    for earlier, later in zip(stamps, stamps[1:]):
        if overlapping:
            counted = later >= start_ns and earlier <= end_ns
        else:
            counted = earlier >= start_ns and later <= end_ns
        if counted:
            gaps_ms.append((later - earlier) / 1e6)
    inside = [stamp for stamp in stamps if start_ns <= stamp <= end_ns]
    before = [stamp for stamp in stamps if stamp < start_ns]
    last = inside[-1] if inside else (before[-1] if before else None)
    open_gap_ms = max(0.0, (end_ns - last) / 1e6) if last is not None else 0.0
    seconds = max(0.0, (end_ns - start_ns) / NS)
    return {
        "window_seconds": seconds,
        "events": len(inside),
        "events_per_second": (len(inside) / seconds) if seconds > 0 else 0.0,
        "gap_count": len(gaps_ms),
        "gap_p50_ms": percentile(gaps_ms, 0.50),
        "gap_p90_ms": percentile(gaps_ms, 0.90),
        "gap_p99_ms": percentile(gaps_ms, 0.99),
        "gap_max_ms": max(gaps_ms) if gaps_ms else 0.0,
        "gap_mean_ms": statistics.fmean(gaps_ms) if gaps_ms else 0.0,
        "open_gap_ms": open_gap_ms,
    }


_SYLLABLES = (
    "ka", "lo", "mi", "ne", "ru", "sa", "te", "vo", "xi", "za", "bre", "dul", "fen", "gor",
    "hal", "jin", "kes", "lum", "nor", "pax", "qui", "ryn", "sol", "tur", "vek", "wyn", "zor",
)


def build_prompt(target_tokens: int, words_per_token: float, seed: int) -> str:
    """Text of about `target_tokens` tokens that no other prompt shares a prefix with.

    Words are random syllable pairs drawn from a seeded generator, so a given seed always gives the
    same prompt (a resubmission is an exact repeat) and a different seed shares nothing with it.
    `words_per_token` is a guess at the tokenizer; the server reports the real count in `usage`.
    """
    rng = random.Random(seed)
    words = max(1, int(target_tokens * words_per_token))
    body = " ".join(rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES) for _ in range(words))
    return f"[run {seed}] Read the following and reply with the single word OK.\n\n{body}\n"


# ----------------------------------------------------------------------------------------------
# Requests


@dataclasses.dataclass
class StreamRecord:
    stamps_ns: list[int] = dataclasses.field(default_factory=list)
    sent_ns: int | None = None
    ended_ns: int | None = None
    status: int | None = None
    error: str | None = None

    def first_event_seconds(self) -> float | None:
        if self.sent_ns is None or not self.stamps_ns:
            return None
        return (self.stamps_ns[0] - self.sent_ns) / NS


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _body(model: str, prompt: str, max_tokens: int, stream: bool, extra: dict[str, Any]) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def _delta_has_text(event_data: str) -> bool:
    if not event_data or event_data == "[DONE]":
        return False
    try:
        parsed = json.loads(event_data)
    except ValueError:
        return False
    for choice in parsed.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
    return False


class Prober:
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float,
                 extra_body: dict[str, Any]) -> None:
        self.client = StreamingHttpClient(base_url, timeout)
        self.model = model
        self.api_key = api_key
        self.extra = extra_body

    def start_stream(self, prompt: str, max_tokens: int) -> tuple[Any, StreamRecord, threading.Thread]:
        """Start a streaming completion on a thread; returns (exchange, record, thread)."""
        record = StreamRecord()
        exchange = self.client.prepare(
            "POST", "/v1/chat/completions", _body(self.model, prompt, max_tokens, True, self.extra),
            _headers(self.api_key) | {"Accept": "text/event-stream"},
        )
        decoder = SseDecoder()

        def on_sent(now: int) -> None:
            record.sent_ns = now

        def on_headers(head: Any) -> None:
            record.status = head.status

        def on_chunk(block: bytes, received_ns: int) -> None:
            if record.status != 200:
                return
            for event in decoder.feed(block, received_ns):
                if _delta_has_text(event.data):
                    record.stamps_ns.append(event.received_ns)

        def run() -> None:
            result = exchange.execute(on_sent=on_sent, on_headers=on_headers, on_chunk=on_chunk)
            record.ended_ns = result.ended_ns
            # The scenario ends the stream by cancelling it; the socket error that causes is expected.
            record.error = None if result.cancel_requested else result.error
            if result.status not in (None, 200) and record.error is None:
                record.error = f"HTTP {result.status}: {result.body[:200]!r}"

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return exchange, record, thread

    def start_blocking(self, prompt: str, max_tokens: int) -> tuple[Any, dict[str, Any], threading.Thread]:
        """A non-streaming completion on a thread; the dict is filled in when it finishes."""
        outcome: dict[str, Any] = {}
        exchange = self.client.prepare(
            "POST", "/v1/chat/completions", _body(self.model, prompt, max_tokens, False, self.extra),
            _headers(self.api_key),
        )

        def run() -> None:
            outcome["sent_ns"] = time.perf_counter_ns()
            result = exchange.execute()
            outcome["ended_ns"] = result.ended_ns
            outcome["status"] = result.status
            outcome["error"] = result.error
            outcome["cancelled"] = result.cancelled
            if result.status == 200:
                try:
                    usage = json.loads(result.body).get("usage") or {}
                    outcome["prompt_tokens"] = usage.get("prompt_tokens")
                    outcome["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    )
                except ValueError:
                    outcome["error"] = "response body is not JSON"
            elif result.error is None:
                outcome["error"] = f"HTTP {result.status}: {result.body[:200]!r}"

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return exchange, outcome, thread


def _seconds(outcome: dict[str, Any]) -> float | None:
    if outcome.get("sent_ns") is None or outcome.get("ended_ns") is None:
        return None
    return (outcome["ended_ns"] - outcome["sent_ns"]) / NS


def _long_report(outcome: dict[str, Any]) -> dict[str, Any]:
    seconds = _seconds(outcome)
    tokens = outcome.get("prompt_tokens")
    cached = outcome.get("cached_tokens") or 0
    report: dict[str, Any] = {
        "seconds": seconds,
        "status": outcome.get("status"),
        "error": outcome.get("error"),
        "cancelled": bool(outcome.get("cancelled")),
        "prompt_tokens": tokens,
        "cached_tokens": cached,
    }
    if seconds and tokens and outcome.get("status") == 200:
        report["prefill_tokens_per_second"] = (tokens - cached) / seconds
    return report


# ----------------------------------------------------------------------------------------------
# Scenario


def run_scenario(args: argparse.Namespace) -> dict[str, Any]:
    prober = Prober(args.base_url, args.model, args.api_key, args.timeout,
                    json.loads(args.extra_body) if args.extra_body else {})
    long_prompt = build_prompt(args.long_tokens, args.words_per_token, args.seed)

    stream_exchange, stream, stream_thread = prober.start_stream(
        "Write a very long, detailed story about a lighthouse keeper. Do not stop.",
        args.stream_tokens,
    )
    deadline = time.monotonic() + args.timeout
    while len(stream.stamps_ns) < args.settle_events and stream_thread.is_alive():
        if time.monotonic() > deadline:
            raise RuntimeError("the stream never produced the settle events")
        time.sleep(0.05)
    if len(stream.stamps_ns) < args.settle_events:
        raise RuntimeError(f"the stream ended early: {stream.error}")
    time.sleep(args.baseline_seconds)

    long_start_ns = time.perf_counter_ns()
    long_exchange, long_outcome, long_thread = prober.start_blocking(long_prompt, 1)

    short_report: dict[str, Any] | None = None
    cancel_report: dict[str, Any] | None = None
    short_thread = None
    short_record = None
    short_started = False
    cancelled_at_ns = None
    while long_thread.is_alive():
        elapsed = (time.perf_counter_ns() - long_start_ns) / NS
        if args.short_delay_seconds is not None and not short_started and elapsed >= args.short_delay_seconds:
            short_started = True
            _, short_record, short_thread = prober.start_stream(
                "Reply with the single word hello.", args.short_tokens
            )
        if args.cancel_long_after_seconds is not None and cancelled_at_ns is None \
                and elapsed >= args.cancel_long_after_seconds:
            cancelled_at_ns = long_exchange.cancel()
        time.sleep(0.02)
    long_end_ns = long_outcome.get("ended_ns") or time.perf_counter_ns()

    retry_report = None
    if cancelled_at_ns is not None:
        cancel_report = _long_report(long_outcome)
        _, retry_outcome, retry_thread = prober.start_blocking(long_prompt, 1)
        retry_thread.join(args.timeout)
        retry_report = _long_report(retry_outcome)
        long_end_ns = retry_outcome.get("ended_ns") or long_end_ns

    if short_thread is not None:
        short_thread.join(args.timeout)
        first = short_record.first_event_seconds() if short_record else None
        short_report = {
            "started_after_seconds": args.short_delay_seconds,
            "time_to_first_event_seconds": first,
            "error": short_record.error if short_record else None,
        }

    time.sleep(args.after_seconds)
    after_end_ns = time.perf_counter_ns()
    stream_exchange.cancel()
    stream_thread.join(args.timeout)

    stamps = stream.stamps_ns
    settle_end = stamps[args.settle_events - 1]
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": {key: getattr(args, key) for key in (
            "base_url", "model", "long_tokens", "words_per_token", "seed", "stream_tokens",
            "settle_events", "baseline_seconds", "after_seconds", "short_delay_seconds",
            "short_tokens", "cancel_long_after_seconds", "extra_body")},
        "stream": {
            "total_events": len(stamps),
            "error": stream.error,
            "baseline": summarize_gaps(stamps, settle_end, long_start_ns),
            "during_long_prompt": summarize_gaps(stamps, long_start_ns, long_end_ns, overlapping=True),
            "after": summarize_gaps(stamps, long_end_ns, after_end_ns),
        },
        "long": _long_report(long_outcome),
    }
    if cancel_report is not None:
        report["long"] = cancel_report
        report["retry"] = retry_report
    if short_report is not None:
        report["short"] = short_report
    return report


def print_summary(report: dict[str, Any]) -> None:
    stream = report["stream"]
    print("stream inter-event gaps (ms)        p50      p90      p99      max   events/s")
    for name in ("baseline", "during_long_prompt", "after"):
        row = stream[name]
        print(f"  {name:<30}{row['gap_p50_ms']:8.1f}{row['gap_p90_ms']:9.1f}{row['gap_p99_ms']:9.1f}"
              f"{row['gap_max_ms']:9.1f}{row['events_per_second']:10.2f}")
    long = report["long"]
    print(f"long prompt: {long.get('prompt_tokens')} tokens in {long.get('seconds')} s "
          f"({long.get('prefill_tokens_per_second', 0):.0f} tok/s, cached {long.get('cached_tokens')})"
          f"{' [cancelled]' if long.get('cancelled') else ''}")
    if "retry" in report:
        retry = report["retry"]
        print(f"retry:       {retry.get('prompt_tokens')} tokens in {retry.get('seconds')} s "
              f"({retry.get('prefill_tokens_per_second', 0):.0f} tok/s, cached {retry.get('cached_tokens')})")
    if "short" in report:
        short = report["short"]
        print(f"short request: first event after {short['time_to_first_event_seconds']} s "
              f"(sent {short['started_after_seconds']} s into the long prompt)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True, help="model id (see GET /v1/models)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--long-tokens", type=int, default=100_000,
                        help="approximate prompt tokens of the long request (the server reports the real count)")
    parser.add_argument("--words-per-token", type=float, default=0.5,
                        help="how many generated words make a token; adjust until usage matches --long-tokens")
    parser.add_argument("--seed", type=int, default=None,
                        help="prompt seed; a fresh random seed (printed) keeps the prompt uncached")
    parser.add_argument("--stream-tokens", type=int, default=4000, help="max_tokens of the neighbouring stream")
    parser.add_argument("--settle-events", type=int, default=40,
                        help="events the stream must produce before the baseline window starts")
    parser.add_argument("--baseline-seconds", type=float, default=5.0)
    parser.add_argument("--after-seconds", type=float, default=5.0)
    parser.add_argument("--short-delay-seconds", type=float, default=None,
                        help="send a short streaming request this long after the long prompt (yield / head-of-line probe)")
    parser.add_argument("--short-tokens", type=int, default=16)
    parser.add_argument("--cancel-long-after-seconds", type=float, default=None,
                        help="cancel the long prompt this long after it starts, then resubmit it (anchor probe)")
    parser.add_argument("--extra-body", default="",
                        help="JSON object merged into every request body, e.g. '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(1 << 31)
    if args.extra_body:
        try:
            if not isinstance(json.loads(args.extra_body), dict):
                parser.error("--extra-body must be a JSON object")
        except ValueError:
            parser.error("--extra-body is not valid JSON")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"seed {args.seed}", file=sys.stderr)
    report = run_scenario(args)
    print_summary(report)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
