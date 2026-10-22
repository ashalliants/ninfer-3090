"""Self-test of run_prefill_stall.py against a fake server. Run: python -I tools/bench/test_prefill_stall.py

The fake server streams a token every 20 ms and, while it "prefills" a long prompt, stalls every
stream for 0.5 s -- the shape the tool exists to measure -- so the expected numbers are known
without a GPU: baseline gaps near 20 ms, a stall gap of about 0.5 s while the long prompt runs.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from tools.bench import run_prefill_stall as probe  # noqa: E402

TOKEN_INTERVAL = 0.02
STALL = 0.5
LONG_PROMPT_CHARS = 2000


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.stall_until = 0.0
        self.long_requests = 0


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-ninfer"
    state = State()

    def log_message(self, *args):  # silence
        pass

    def _wait_out_stall(self):
        while True:
            with self.state.lock:
                remaining = self.state.stall_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.01))

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(body["max_tokens"]):
                    self._wait_out_stall()
                    event = {"choices": [{"delta": {"content": "x"}}]}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(TOKEN_INTERVAL)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        # A long, non-streaming request: every stream stalls while it prefills.
        if len(prompt) > LONG_PROMPT_CHARS:
            with self.state.lock:
                self.state.long_requests += 1
                self.state.stall_until = time.monotonic() + STALL
            time.sleep(STALL)
        payload = json.dumps({"usage": {"prompt_tokens": len(prompt.split()) * 2,
                                        "prompt_tokens_details": {"cached_tokens": 0}}}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def fake_server():
    Handler.state = State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(extra):
    server = fake_server()
    try:
        args = probe.parse_args([
            "--base-url", f"http://127.0.0.1:{server.server_port}", "--model", "fake",
            "--long-tokens", "2000", "--words-per-token", "1.0", "--seed", "7",
            "--stream-tokens", "400", "--settle-events", "10", "--baseline-seconds", "0.3",
            "--after-seconds", "0.3", "--timeout", "30", *extra])
        return probe.run_scenario(args), Handler.state
    finally:
        server.shutdown()


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def test_percentile():
    check(probe.percentile([], 0.5) == 0.0, "empty percentile")
    values = list(range(1, 101))
    check(probe.percentile(values, 0.50) == 50 and probe.percentile(values, 0.99) == 99
          and probe.percentile(values, 1.0) == 100 and probe.percentile(values, 0.0) == 1,
          "nearest-rank percentiles")


def test_gap_window():
    ms = 1_000_000
    # Events every 10 ms, a 300 ms hole from 100 to 400, then steady again.
    stamps = [i * 10 * ms for i in range(11)] + [400 * ms + i * 10 * ms for i in range(10)]
    whole = probe.summarize_gaps(stamps, 0, 500 * ms)
    check(whole["gap_max_ms"] == 300.0 and whole["gap_p50_ms"] == 10.0, "whole-stream gaps")
    # A window that opens inside the hole counts the whole 300 ms gap that overlaps it, if asked to.
    inside = probe.summarize_gaps(stamps, 200 * ms, 350 * ms, overlapping=True)
    check(inside["gap_max_ms"] == 300.0 and inside["events"] == 0, "a gap overlapping the window")
    check(inside["open_gap_ms"] == 250.0, "open gap is time since the last event")
    # By default a gap that is not wholly inside the window belongs to the neighbouring one.
    check(probe.summarize_gaps(stamps, 200 * ms, 350 * ms)["gap_count"] == 0,
          "a gap straddling a window edge was counted inside it")
    before_hole = probe.summarize_gaps(stamps, 0, 250 * ms)
    after_hole = probe.summarize_gaps(stamps, 250 * ms, 500 * ms)
    check(before_hole["gap_max_ms"] == 10.0 and after_hole["gap_max_ms"] == 10.0,
          "the stall was assigned to a neighbouring window as well as its own")
    quiet = probe.summarize_gaps(stamps, 0, 90 * ms)
    check(quiet["gap_max_ms"] == 10.0 and quiet["events"] == 10, "a window with no hole")
    check(probe.summarize_gaps([], 0, ms)["events"] == 0, "no events")


def test_prompt_is_deterministic_and_uncached_across_seeds():
    a = probe.build_prompt(1000, 0.5, 1)
    check(a == probe.build_prompt(1000, 0.5, 1), "same seed, same prompt")
    check(a != probe.build_prompt(1000, 0.5, 2), "different seeds, different prompts")
    check(len(a.split()) > 450, "prompt is about the requested size")
    check(a.split("\n", 1)[0] != probe.build_prompt(1000, 0.5, 2).split("\n", 1)[0],
          "different seeds differ from the very first line, so no prefix is shared")


def test_scenario_sees_the_stall():
    report, state = run([])
    stream = report["stream"]
    check(stream["error"] is None, f"stream error {stream['error']}")
    check(stream["baseline"]["gap_p50_ms"] < 80, f"baseline p50 {stream['baseline']['gap_p50_ms']}")
    check(stream["baseline"]["gap_max_ms"] < 250, "baseline has a stall in it")
    during = stream["during_long_prompt"]
    check(during["gap_max_ms"] >= STALL * 1000 * 0.8, f"stall not seen: {during['gap_max_ms']}")
    check(report["long"]["status"] == 200 and report["long"]["prompt_tokens"] > 0
          and report["long"]["seconds"] >= STALL * 0.9, f"long request report {report['long']}")
    check(report["long"]["prefill_tokens_per_second"] > 0, "prefill rate is reported")
    check(state.long_requests == 1, "the long prompt was sent once")
    check("short" not in report and "retry" not in report, "no optional sections unasked")
    json.dumps(report)


def test_short_request_waits_out_the_stall():
    report, _ = run(["--short-delay-seconds", "0.1"])
    first = report["short"]["time_to_first_event_seconds"]
    check(first is not None and first >= STALL * 0.6, f"short request did not wait: {first}")


def test_cancel_and_retry():
    report, state = run(["--cancel-long-after-seconds", "0.1"])
    check(report["long"]["cancelled"], "the first long request was not cancelled")
    check("retry" in report and report["retry"]["status"] == 200, f"no retry: {report.get('retry')}")
    check(state.long_requests == 2, "the prompt was resubmitted once")


def test_cancel_after_completion_is_not_a_retry():
    # The fake long request takes STALL seconds; a cancel scheduled after that lands on a finished
    # request, and the run must fail instead of "retrying" a prompt that completed normally.
    try:
        run(["--cancel-long-after-seconds", str(STALL + 0.4)])
    except RuntimeError as error:
        check("before the cancel took effect" in str(error), f"unexpected error: {error}")
        return
    check(False, "a cancel after completion produced a report")


def test_windows_start_at_the_wire_send_timestamp():
    report, _ = run([])
    during = report["stream"]["during_long_prompt"]
    check(during["gap_max_ms"] >= STALL * 1000 * 0.6, "the stall is in the during window")
    check(report["long"]["seconds"] is not None and report["long"]["seconds"] >= STALL * 0.9,
          "the long duration is measured from the send")


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
    print("OK prefill stall probe")
