"""Server sessions for the Flash-Next context store / NVMe tier measurements (scratch).

usage: bench.py SESSIONS.json OUT_PREFIX
SESSIONS: list of {"name", "exe", "args": [...], "plan": "decode+hits" | "turns" | "resume" | ...}
Each session starts ninfer-serve in its own process group, waits for /health, runs its plan
through client.run, stops it with CTRL_BREAK (graceful: the context store's shutdown write runs)
and keeps its log in OUT_PREFIX-<name>-<k>.log. Results go to OUT_PREFIX.jsonl.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402

MODEL = r"C:\ninfer-fork\ninfer-3090\models\qwen4exp-iq2xs-mtp-vision.ninfer"
PORT = 8197
BASE = ["--host", "127.0.0.1", "--port", str(PORT), "--max-concurrency", "1", "--max-context", "32768",
        "--kv-capacity", "32768", "--kv-dtype", "int8", "--prefill-chunk", "1024", "--max-pending-requests", "16",
        "--pending-timeout-ms", "600000", "--spec", "mtp", "--draft-tokens", "3", "--vision", "--no-expert-state",
        "--log-stats-interval-ms", "0"]
DOCS = r"C:\ninfer-fork\ninfer-3090\.claude\worktrees\linked-baking-piglet\.claude\worktrees\fn-store\docs"


def corpus():
    text = []
    for root, _, files in os.walk(DOCS):
        for name in sorted(files):
            if name.endswith(".md"):
                with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
                    text.append(f.read())
    return "\n\n".join(text)


TEXT = corpus()


def doc(start, chars):
    return TEXT[start:start + chars]


def user(content):
    return [{"role": "user", "content": content}]


PROMPT8 = user("Summarize the following notes in three sentences.\n\n" + doc(0, 26000))
PROMPT30 = user("List the five most important facts in these notes.\n\n" + doc(200000, 98000))
SHORT = user("Write a long, detailed story about a lighthouse keeper and a storm.")


def plan(kind):
    if kind == "decode+hits":
        return [("decode-warm", SHORT, 64, 1), ("decode", SHORT, 512, 3),
                ("cold30", PROMPT30, 16, 1), ("hit30", PROMPT30, 16, 5),
                ("cold8", PROMPT8, 16, 1), ("hit8", PROMPT8, 16, 5)]
    if kind == "conversation":  # restart: build a conversation, the next turn comes after the restart
        return [("cold30", PROMPT30, 64, 1), ("cold8", PROMPT8, 64, 1)]
    if kind == "after-restart":
        return [("again30", PROMPT30, 16, 1), ("again8", PROMPT8, 16, 1), ("hit30", PROMPT30, 16, 3),
                ("hit8", PROMPT8, 16, 3)]
    raise ValueError(kind)


def wait_ready(proc, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server not ready")


def main():
    sessions = json.load(open(sys.argv[1], encoding="utf-8"))
    prefix = sys.argv[2]
    out = open(prefix + ".jsonl", "a", encoding="utf-8")
    for k, s in enumerate(sessions):
        log_path = f"{prefix}-{k:02d}-{s['name']}.log"
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen([s["exe"], MODEL] + BASE + s.get("args", []), stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            try:
                ready = wait_ready(proc)
                out.write(json.dumps({"session": k, "name": s["name"], "event": "ready", "seconds": round(ready, 1)}) + "\n")
                out.flush()
                print(f"[{s['name']}] ready in {ready:.0f} s (pid {proc.pid})", flush=True)
                for label, messages, max_tokens, repeat in plan(s["plan"]):
                    for _ in range(repeat):
                        r = client.run(PORT, label, messages, max_tokens)
                        r.update({"session": k, "name": s["name"]})
                        out.write(json.dumps(r) + "\n")
                        out.flush()
            finally:
                stop = time.time()
                if proc.poll() is None:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    try:
                        proc.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        proc.kill()  # our own child, by handle
                        proc.wait()
                out.write(json.dumps({"session": k, "name": s["name"], "event": "stopped",
                                      "seconds": round(time.time() - stop, 1), "code": proc.returncode}) + "\n")
                out.flush()
                print(f"[{s['name']}] stopped in {time.time() - stop:.0f} s, code {proc.returncode}", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
