"""Self-test of serve_soak.py against a fake server. Run: python -I tools/bench/test_serve_soak.py

The fake speaks just enough of ninfer-serve (/v1/models, /slots, /v1/load, /metrics, count_tokens, OpenAI
and Anthropic streams, 400 context_length_exceeded, 413, vision_disabled) and can be switched into the
ways a server goes wrong: a stream that goes quiet, an unexpected 400 and 500, a growing resident memory
(injected as a reader), and a server that is gone. Each must make the soak exit non-zero, and a clean
server must exit 0. No GPU, a few seconds.
"""
import json
import os
import struct
import sys
import tempfile
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from tools.bench import serve_soak as soak  # noqa: E402

CONTEXT = 2000
MAX_REQUEST_MIB = 1


class State:
    def __init__(self, mode="clean", vision=False):
        self.mode = mode
        self.vision = vision
        self.lock = threading.Lock()
        self.lane = threading.Lock()   # the one lane of mode "lane": streams run one at a time
        self.waiting = []              # when each queued request arrived
        self.busy = False
        self.first_post: float | None = None
        self.seen = []            # (method, path, headers dict)
        self.posts = {"chat": 0, "messages": 0, "over": 0, "oversize": 0, "image": 0}


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-ninfer"
    state = State()

    def log_message(self, format, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self):
        with self.state.lock:
            self.state.seen.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))

    def do_GET(self):
        self._record()
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": "fake-model"}]})
        if self.path == "/slots":
            busy = self.state.busy if self.state.mode in ("lane", "idle_queue", "idle_queue_head") else True
            return self._json(200, [{"id": 0, "n_ctx": CONTEXT, "n_ctx_used": 10, "is_processing": busy}])
        if self.path == "/v1/load":
            now = time.monotonic()
            with self.state.lock:
                if self.state.mode == "lane":
                    queue = [{"id": "r", "waited_s": now - t} for t in self.state.waiting]
                elif self.state.mode.startswith("idle_queue") and self.state.first_post is not None:
                    # Two requests "queued" behind a lane that is idle and never takes them.
                    head = 100.0 if self.state.mode == "idle_queue_head" else now - self.state.first_post
                    queue = [{"id": "a", "waited_s": head}, {"id": "b", "waited_s": head}]
                else:
                    queue = []
            return self._json(200, {"requests": {"queue": queue, "queue_total": len(queue)}})
        if self.path == "/metrics":
            body = b"# TYPE ninfer_requests_total counter\nninfer_requests_total 3\nninfer_x{a=\"b\"} 1\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self._json(404, {"error": {"message": "no"}})

    def do_POST(self):
        self._record()
        length = int(self.headers["Content-Length"])
        if length > MAX_REQUEST_MIB << 20:
            with self.state.lock:
                self.state.posts["oversize"] += 1
            self.close_connection = True
            return self._json(413, {"error": {"code": "request_too_large", "message": "too big"}})
        body = json.loads(self.rfile.read(length))
        messages = body["messages"]
        text = json.dumps(messages)
        if self.path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": max(1, len(text) // 4)})
        anthropic = self.path == "/v1/messages"
        key = "messages" if anthropic else "chat"
        with self.state.lock:
            self.state.posts[key] += 1
        if "image" in text or "data:image" in text:
            with self.state.lock:
                self.state.posts["image"] += 1
            if not self.state.vision:
                err = {"error": {"code": "vision_disabled", "message": "Vision is disabled"}}
                return self._json(400, err if not anthropic else {"type": "error", "error": {
                    "type": "invalid_request_error", "message": "Vision is disabled"}})
        if len(text) // 4 > CONTEXT and self.state.mode != "accept_over":
            with self.state.lock:
                self.state.posts["over"] += 1
            if anthropic:
                return self._json(400, {"type": "error", "error": {
                    "type": "invalid_request_error", "message": "prompt is too long"}})
            return self._json(400, {"error": {"code": "context_length_exceeded", "message": "too long"}})
        mode = self.state.mode
        if mode.startswith("idle_queue"):
            with self.state.lock:
                self.state.first_post = self.state.first_post or time.monotonic()
            time.sleep(8)           # never admitted; the soak ends the request
            return
        if mode == "lane":
            with self.state.lock:
                self.state.waiting.append(time.monotonic())
            self.state.lane.acquire()
            with self.state.lock:
                self.state.waiting.pop(0)
                self.state.busy = True
            try:
                return self._stream(body, anthropic, mode)
            finally:
                with self.state.lock:
                    self.state.busy = False
                self.state.lane.release()
        if mode == "bad400":
            return self._json(400, {"error": {"code": "invalid_request", "message": "nope"}})
        if mode == "boom500":
            return self._json(500, {"error": {"code": "internal_error", "message": "boom"}})
        self._stream(body, anthropic, mode)

    def _stream(self, body, anthropic, mode):
        delay = 0.03 if mode == "lane" else 0.01
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i in range(min(body["max_tokens"], 12)):
                if anthropic:
                    event = {"type": "content_block_delta", "delta": {"text": "x"}}
                    self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(event)}\n\n".encode())
                else:
                    event = {"choices": [{"delta": {"content": "x"}}]}
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                if mode == "stall" and i == 1:
                    time.sleep(6)         # far beyond --stall-seconds; the soak cancels it
                time.sleep(delay)
            if anthropic:
                self.wfile.write(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
            else:
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def fake_server(mode="clean", vision=False):
    Handler.state = State(mode, vision)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Handler.state


def png():
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * 4 for _ in range(4))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def run_soak(server, extra=(), rss_reader=None, minutes="0.01"):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.json")
        argv = ["--url", f"http://127.0.0.1:{server.server_port}", "--minutes", minutes, "--out", out,
                "--seed", "3", "--sample-seconds", "0.1", "--context", str(CONTEXT), "--drain-seconds", "5",
                "--max-request-mib", str(MAX_REQUEST_MIB), *extra]
        sink = StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            code = soak.main(argv, rss_reader=rss_reader)
        with open(out) as handle:
            report = json.load(handle)
    return code, report, sink.getvalue()


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def test_a_clean_server_passes_and_every_kind_runs():
    server, state = fake_server("clean", vision=True)
    try:
        with tempfile.TemporaryDirectory() as images:
            with open(os.path.join(images, "a.png"), "wb") as handle:
                handle.write(png())
            code, report, output = run_soak(server, ["--images", images, "--api-key", "sekrit", "--max-concurrency", "3"])
    finally:
        server.shutdown()
    check(code == 0, f"a clean server exited {code}: {report['findings']} {report['failures']}\n{output}")
    check(report["findings"] == [] and report["failures"] == [] and report["stalls"] == [], "findings on clean")
    kinds = report["requests"]["by_kind"]
    for kind in soak.KINDS:
        check(kind in kinds, f"kind {kind} never ran: {sorted(kinds)}")
    check(set(kinds["over"]) == {"ok"}, f"over-long prompts: {kinds['over']}")
    check(set(kinds["oversize"]) <= {"ok", "ok_reset"}, f"oversize: {kinds['oversize']}")
    check(state.posts["oversize"] >= 1 and state.posts["over"] >= 1, f"server saw {state.posts}")
    check(set(kinds["near"]) == {"ok"}, f"near-limit prompts must fit: {kinds['near']}")
    check(set(kinds["image"]) == {"ok"}, f"images against a vision server: {kinds['image']}")
    check(report["samples"] and "queue_total" in report["samples"][0] and report["samples"][0]["slots"] == 1,
          "samples carry /v1/load and /slots")
    check(report["samples"][0]["metrics"] == {"ninfer_requests_total": 3.0}, "unlabelled ninfer_ metrics")
    check(report["context"] == CONTEXT and report["model"] == "fake-model", "context and model discovered")
    # Host/Origin/auth hygiene: a normal Host, the key as Bearer, nothing a browser would add.
    posts = [h for method, _, h in state.seen if method == "POST"]
    check(posts and all(h.get("authorization") == "Bearer sekrit" for h in posts), "Bearer key on every POST")
    check(all("origin" not in h and not any(k.startswith("sec-fetch") for k in h) for _, _, h in state.seen),
          "Origin or Sec-Fetch-* sent")
    check(all(h.get("host", "").startswith("127.0.0.1:") for _, _, h in state.seen), "Host header")
    json.dumps(report)


def test_images_are_skipped_when_the_server_has_no_vision():
    server, state = fake_server("clean", vision=False)
    try:
        with tempfile.TemporaryDirectory() as images:
            with open(os.path.join(images, "a.png"), "wb") as handle:
                handle.write(png())
            code, report, output = run_soak(server, ["--images", images, "--kinds", "chat,image", "--max-requests", "2"])
    finally:
        server.shutdown()
    check(code == 0, f"vision_disabled must not fail the run: {report['failures']}\n{output}")
    check(report["requests"]["by_kind"]["image"].get("skipped_no_vision") == 1,
          f"image outcome {report['requests']['by_kind']['image']}")
    check(state.posts["image"] == 1, "images were retried after vision_disabled")


def test_a_stalled_stream_is_flagged():
    server, _ = fake_server("stall")
    try:
        started = time.monotonic()
        code, report, output = run_soak(server, ["--kinds", "chat", "--stall-seconds", "0.6",
                                                  "--max-concurrency", "1"])
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
    check(code == 1, f"stall exit code {code}\n{output}")
    check(report["stalls"] and report["stalls"][0]["phase"] == "mid-stream", f"stalls {report['stalls']}")
    check(any(f["reason"] == "stalled" for f in report["failures"]), "the stalled request is a failure")
    check(elapsed < 6, f"the soak waited out the stalled stream ({elapsed:.1f} s) instead of cancelling it")


def test_queued_requests_on_one_lane_are_not_stalls():
    # One lane, four clients: each waits behind the others for longer than --stall-seconds, which is
    # queueing and not silence. The server reports the queue, so none of it counts.
    server, state = fake_server("lane")
    try:
        code, report, output = run_soak(server, ["--kinds", "chat", "--stall-seconds", "0.3",
                                                  "--max-concurrency", "4"], minutes="0.02")
    finally:
        server.shutdown()
    check(code == 0, f"queued requests were flagged: {report['stalls']} {report['failures'][:2]}\n{output}")
    check(report["settings"]["lanes"] == 1 and report["requests"]["total"] >= 4, f"{report['requests']}")
    check(any(s.get("queue_total") for s in report["samples"]), "the queue was never sampled")
    check(report["duration_seconds"] > 0.6, "run too short to prove anything")


def test_the_default_concurrency_is_the_lane_count():
    server, _ = fake_server("clean")
    try:
        code, report, output = run_soak(server, ["--kinds", "chat", "--max-requests", "1"], minutes="1")
    finally:
        server.shutdown()
    check(report["settings"]["lanes"] == 1 and report["settings"]["max_concurrency"] == 1,
          f"settings {report['settings']}")


def test_a_queue_behind_an_idle_lane_is_flagged():
    for mode in ("idle_queue", "idle_queue_head"):
        server, _ = fake_server(mode)
        try:
            code, report, output = run_soak(server, ["--kinds", "chat", "--stall-seconds", "0.5",
                                                      "--max-concurrency", "2"])
        finally:
            server.shutdown()
        check(code == 1 and any(s["kind"] == "queue" for s in report["stalls"]),
              f"{mode}: an idle lane with a queue was not flagged: {code} {report['stalls']}\n{output}")


def test_an_unexpected_400_is_flagged():
    server, _ = fake_server("bad400")
    try:
        code, report, output = run_soak(server, ["--kinds", "chat", "--max-requests", "2"], minutes="1")
    finally:
        server.shutdown()
    check(code == 1, f"unexpected 400 exit code {code}\n{output}")
    check(report["failures"] and "expected HTTP 200, got 400" in report["failures"][0]["reason"],
          f"failure text {report['failures'][:1]}")


def test_an_unexpected_500_is_flagged():
    server, _ = fake_server("boom500")
    try:
        code, report, output = run_soak(server, ["--kinds", "chat,cancel", "--max-requests", "3"], minutes="1")
    finally:
        server.shutdown()
    check(code == 1, f"500 exit code {code}\n{output}")
    check(any(f["status"] == 500 for f in report["failures"]), f"no 500 in {report['failures'][:2]}")


def test_a_wrong_status_for_an_expected_error_is_flagged():
    # The over-long prompt must be a 400 context_length_exceeded; a server that streams it is wrong.
    server, _ = fake_server("accept_over")
    try:
        code, report, output = run_soak(server, ["--kinds", "over", "--max-requests", "2"], minutes="1")
    finally:
        server.shutdown()
    check(code == 1 and any("expected HTTP 400 context_length_exceeded" in f["reason"]
                            for f in report["failures"]), f"over-long accepted: {code} {report['failures'][:1]}")


def test_growing_rss_is_flagged_and_flat_rss_is_not():
    for growing, expected in ((True, 1), (False, 0)):
        server, _ = fake_server("clean")
        reading = {"n": 0}

        def reader(growing=growing, reading=reading):
            reading["n"] += 1
            return 1000.0 + (reading["n"] * 40.0 if growing else (reading["n"] % 2))
        try:
            code, report, output = run_soak(server, ["--kinds", "chat", "--rss-growth-mib", "100"],
                                            rss_reader=reader)
        finally:
            server.shutdown()
        check(report["rss"]["samples"] >= 6, f"too few rss samples: {report['rss']}")
        check(code == expected and report["rss"]["flagged"] == bool(growing),
              f"growing={growing}: exit {code}, rss {report['rss']}\n{output}")


def test_rss_growth_arithmetic_and_proc_reader():
    check(soak.rss_growth_mib([1, 2, 3]) is None, "too few samples give no verdict")
    check(soak.rss_growth_mib([5000] * 4 + [1000] * 10) == 0, "warm-up in the first half is not growth")
    check(soak.rss_growth_mib([1000] * 8 + [1000, 1000, 1700, 1700, 1700, 1700, 1700, 1700]) == 700,
          "a step in the last half")
    check(soak.rss_growth_mib([1000] * 14 + [9000] + [1000] * 5) == 0,
          "one noisy sample must not decide")
    with tempfile.TemporaryDirectory() as proc:
        os.makedirs(os.path.join(proc, "4242"))
        with open(os.path.join(proc, "4242", "status"), "w") as handle:
            handle.write("Name:\tninfer-serve\nVmPeak:\t 9999 kB\nVmRSS:\t  2097152 kB\nThreads:\t4\n")
        reader, note = soak.make_rss_reader(4242, proc)
        check(reader is not None and reader() == 2048.0 and not note, "VmRSS parsed as MiB")
        reader, note = soak.make_rss_reader(None, proc)
        check(reader is None and "no --pid" in note, "no pid, no reader")


def test_a_gone_server_is_a_usage_error_and_a_dying_one_a_finding():
    server, _ = fake_server("clean")
    port = server.server_port
    server.shutdown()
    server.server_close()
    with tempfile.TemporaryDirectory() as tmp:
        sink = StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            code = soak.main(["--url", f"http://127.0.0.1:{port}", "--minutes", "0.01",
                              "--out", os.path.join(tmp, "r.json")])
        check(code == 2 and "cannot reach" in sink.getvalue(), f"unreachable server: {code} {sink.getvalue()}")
    server, _ = fake_server("clean")
    killer = threading.Timer(0.6, lambda: None)
    try:
        killer = threading.Timer(0.6, lambda: (server.shutdown(), server.server_close()))
        killer.start()
        code, report, output = run_soak(server, ["--kinds", "chat"], minutes="0.1")
    finally:
        killer.cancel()
    check(code == 1 and "the server stopped answering" in report["findings"], f"dying server: {code} {output}")


def test_the_oversize_body_is_streamed_in_small_chunks():
    args = soak.parse_args(["--url", "http://x", "--minutes", "1", "--out", "o", "--max-request-mib", "3"])
    chunks, length = soak.Soak(args, None, "").oversize_body()
    sizes = [len(chunk) for chunk in chunks]
    check(sum(sizes) == length and length > 4 << 20, f"declared {length}, produced {sum(sizes)}")
    check(max(sizes) <= 1 << 20, "a chunk larger than 1 MiB: the body is being held whole")


def test_exit_two_for_a_bad_out_path_and_for_a_harness_bug():
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        code = soak.main(["--url", "http://127.0.0.1:1", "--minutes", "0.01", "--out", "/nonexistent-dir/r.json"])
    check(code == 2 and "not writable" in sink.getvalue(), f"bad --out: {code} {sink.getvalue()}")
    server, _ = fake_server("clean")
    original = soak.Soak.preflight
    try:
        def explode(self):
            raise ZeroDivisionError("harness bug")
        soak.Soak.preflight = explode
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(sink), redirect_stderr(sink):
            code = soak.main(["--url", f"http://127.0.0.1:{server.server_port}", "--minutes", "0.01",
                              "--out", os.path.join(tmp, "r.json")])
    finally:
        soak.Soak.preflight = original
        server.shutdown()
    check(code == 2 and "harness itself failed" in sink.getvalue(), f"harness bug: {code}")


def test_an_interrupt_writes_a_partial_report():
    server, _ = fake_server("stall")
    original = soak.Soak.traffic
    try:
        def interrupted(self):
            original(self)
            raise KeyboardInterrupt
        soak.Soak.traffic = interrupted
        code, report, output = run_soak(server, ["--kinds", "chat", "--stall-seconds", "30"], minutes="0.005")
    finally:
        soak.Soak.traffic = original
        server.shutdown()
    check(report["interrupted"] and code == 2 and report["failures"] == [] and report["stalls"] == [],
          f"interrupt: {code} {report['failures']} {report['stalls']}")


def test_a_sampler_bug_is_recorded_and_fails_the_run():
    server, _ = fake_server("clean")
    original = soak.Soak.check_queue
    try:
        def broken(self, sample):
            raise RuntimeError("boom")
        soak.Soak.check_queue = broken
        code, report, output = run_soak(server, ["--kinds", "chat"])
    finally:
        soak.Soak.check_queue = original
        server.shutdown()
    check(code == 1 and report["harness_errors"] and len(report["samples"]) >= 3,
          f"sampler bug: {code} {report['harness_errors'][:1]} samples={len(report['samples'])}")


def test_argument_checks():
    for bad in (["--kinds", "chat,bogus"], ["--minutes", "-1"]):
        sink = StringIO()
        with redirect_stderr(sink):
            try:
                soak.parse_args(["--url", "http://x", "--minutes", "1", "--out", "o", *bad])
            except SystemExit as exit_:
                check(exit_.code == 2, f"argparse exit {exit_.code}")
                continue
        check(False, f"{bad} was accepted")


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            began = time.monotonic()
            function()
            print(f"  {name}: {time.monotonic() - began:.1f} s")
    print("OK serve soak")
