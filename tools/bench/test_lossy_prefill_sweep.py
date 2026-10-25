"""Self-test of run_lossy_prefill_sweep.py. Run: python -I tools/bench/test_lossy_prefill_sweep.py

No server and no GPU: the summariser against fixture results (the acceptance numbers' edges), grid
expansion, command construction, the eval config, and the Edit / divergence / speed extractors.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_lossy_prefill_sweep as sweep  # noqa: E402


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def row(point, workload, metrics, ok=True, error=None):
    return {"point": point, "params": {}, "workload": workload, "ok": ok, "error": error, "metrics": metrics}


BASE = [
    row("baseline", "needle", {"accuracy": {"32k": 1.0, "64k": 1.0, "100k": 1.0}}),
    row("baseline", "agent_replay", {"edit_rate": 0.90, "tool_call_validity": 1.0}),
    row("baseline", "speed", {"total_s": 100.0, "neighbour_gap_max_ms": 900}),
    row("baseline", "greedy_agreement", {"outputs": [[1, 2, 3, 4]] * 2}),
]


def point_rows(point, needle=1.0, edit=0.90, total=39.0):
    return [
        row(point, "needle", {"accuracy": {"32k": needle, "64k": needle, "100k": needle}}),
        row(point, "agent_replay", {"edit_rate": edit, "tool_call_validity": 0.99}),
        row(point, "speed", {"total_s": total, "scout_ms": 4000, "main_ms": 30000, "kept_tokens": 20000,
                             "uncached_tokens": 100000, "neighbour_gap_max_ms": 300}),
        row(point, "greedy_agreement", {"outputs": [[1, 2, 3, 9], [1, 2, 3, 4]]}),
    ]


def checks(rows, point):
    return {c["name"]: c for c in sweep.evaluate_point(rows, point)["checks"]}


def test_needle_threshold():
    rows = BASE + point_rows("good", needle=0.96) + point_rows("bad", needle=0.94)
    check(all(c["pass"] for n, c in checks(rows, "good").items() if n.startswith("needle")), "0.96x passes")
    check(all(c["pass"] is False for n, c in checks(rows, "bad").items() if n.startswith("needle")), "0.94x fails")
    check(abs(checks(rows, "good")["needle_32k"]["value"] - 0.96) < 1e-9, "ratio value")


def test_edit_threshold():
    rows = BASE + point_rows("a", edit=0.875) + point_rows("b", edit=0.88) + point_rows("c", edit=0.95)
    check(checks(rows, "a")["edit"]["pass"] is False, "-2.5 points fails")
    check(checks(rows, "b")["edit"]["pass"] is True, "-2.0 points is within 2 points")
    check(checks(rows, "c")["edit"]["pass"] is True, "an improvement passes")


def test_time_threshold():
    rows = BASE + point_rows("fast", total=39.0) + point_rows("slow", total=41.0)
    check(checks(rows, "fast")["time"]["pass"] is True, "0.39x passes")
    check(checks(rows, "slow")["time"]["pass"] is False, "0.41x fails")


def test_verdicts_and_missing():
    rows = BASE + point_rows("ok") + point_rows("bad", total=60.0) + [row("part", "speed", {"total_s": 30.0})]
    check(sweep.evaluate_point(rows, "ok")["verdict"] == "PASS", "all pass")
    check(sweep.evaluate_point(rows, "bad")["verdict"] == "FAIL", "one fail")
    check(sweep.evaluate_point(rows, "part")["verdict"] == "INCOMPLETE", "unmeasured is not a pass")
    rows.append(row("down", "server", {}, ok=False, error="server exited with 1"))
    check(sweep.evaluate_point(rows, "down")["verdict"] == "FAIL", "a failed start fails the point")


def test_summary_md_and_from_json():
    rows = BASE + point_rows("good", needle=0.96) + point_rows("bad", needle=0.94, edit=0.875, total=39.0)
    rows.append(row("down", "server", {}, ok=False, error="server exited with 1"))
    text = sweep.render_summary(rows)
    check("| good |" in text and "| bad |" in text, "rows present")
    good = next(l for l in text.splitlines() if l.startswith("| good |"))
    bad = next(l for l in text.splitlines() if l.startswith("| bad |"))
    check("0.960 pass" in good and "FAIL" not in good, good)
    check("0.940 FAIL" in bad and "-2.5 FAIL" in bad and "0.39 pass" in bad, bad)
    check("`down` / server: server exited with 1" in text, "failures listed")
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.jsonl"
        results.write_text("".join(json.dumps(r) + "\n" for r in rows))
        check(sweep.main(["--from-json", str(results), "--out", d]) == 0, "from-json exits 0")
        check((Path(d) / "summary.md").read_text() == text, "summary.md written")


def test_grid():
    grid = sweep.parse_grid(["filter_layer=7,15", "keep=0.1,0.2", "block=64", "strategy=select,summary"])
    points = sweep.expand_grid(grid)
    ids = [p["id"] for p in points]
    check(len(points) == 5 and ids[-1] == "summary", f"4 select points + one summary: {ids}")
    check("fl7_k0.1_b64" in ids, ids)
    check(sweep.expand_grid(sweep.parse_grid(["keep=0.2"]))[0]["id"] == "k0.2", "select is the default")
    for bad in (["nope=1"], ["keep="], ["strategy=x"], ["scout_a4=2"], ["keep"]):
        try:
            sweep.parse_grid(bad)
        except ValueError:
            continue
        check(False, f"{bad} should be refused")


def test_commands():
    select = sweep.expand_grid(sweep.parse_grid(["filter_layer=11", "keep=0.3", "block=128", "scout_a4=1"]))[0]
    cmd = sweep.server_command("ninfer-serve m.ninfer --kv-dtype rk4v4", 18130, Path("/tmp/rl.jsonl"), select["params"])
    check(cmd[:4] == ["ninfer-serve", "m.ninfer", "--kv-dtype", "rk4v4"], cmd)
    joined = " ".join(cmd)
    for fragment in ("--port 18130", "--request-log-jsonl /tmp/rl.jsonl", "--lossy-prefill-min-tokens 30000",
                     "--lossy-prefill-filter-layer 11", "--lossy-prefill-keep 0.3", "--lossy-prefill-block 128"):
        check(fragment in joined, f"{fragment} in {joined}")
    check(cmd[-1] == "--lossy-prefill-scout-a4" and "--lossy-prefill-strategy" not in cmd, "a4 switch, no strategy flag for select")
    summary = sweep.server_command("ninfer-serve m", 1, Path("l"), {"strategy": "summary"})
    check(summary[-2:] == ["--lossy-prefill-strategy", "summary"], summary)
    base = sweep.server_command("ninfer-serve m", 1, Path("l"), None, ["--max-concurrency", "2"])
    check(not any("lossy" in a for a in base) and "--max-concurrency" in base, f"baseline has no lossy flags: {base}")
    replay = sweep.agent_replay_command(port=1, out=Path("o"), request_log=Path("l"), sessions=4, seed=None)
    check(replay[1:4] == ["-m", "tools.bench.agent_replay", "run"] and "--min-context" in replay
          and replay[replay.index("--min-context") + 1] == "30000" and replay[replay.index("--max-context") + 1] == "120000", replay)
    stall = sweep.stall_command(base_url="http://x", model="m", output=Path("o.json"))
    check(stall[stall.index("--long-tokens") + 1] == "108000", stall)


def test_needle_config():
    config = sweep.needle_config(base_url="http://127.0.0.1:1", model="m", corpus="/c", tokenizer_path="/t", runs_dir="/r")
    jobs = config["suites"]["needle"]["jobs"]
    check([j["id"] for j in jobs] == ["needle_32k", "needle_64k", "needle_100k"], jobs)
    extra = jobs[2]["backend_args"]["dataset_args"]["extra_params"]
    check(extra["context_lengths_min"] == extra["context_lengths_max"] == 100000
          and extra["document_depth_percent_intervals"] == 5 and extra["tokenizer_path"] == "/t", extra)
    check(jobs[0]["backend_args"]["dataset_args"]["local_path"] == "/c", "corpus path from the argument")
    check("neroued" not in json.dumps(config), "no hard-coded home paths")
    summary = {"results": [{"job_id": "needle_32k", "status": "completed", "primary_metric": "accuracy", "metrics": {"accuracy": 0.8}},
                           {"job_id": "needle_64k", "status": "completed", "primary_metric": "accuracy", "metrics": {"accuracy": 0.6}}]}
    check(sweep.parse_needle_summary(summary) == {"32k": 0.8, "64k": 0.6}, "needle summary parsed")
    summary["results"][0]["status"] = "failed"
    try:
        sweep.parse_needle_summary(summary)
        check(False, "a failed job is an error")
    except RuntimeError:
        pass


def call(name, args):
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args) if not isinstance(args, str) else args}}


def test_edit_metrics():
    tools = [{"type": "function", "function": {"name": "Edit", "parameters": {"required": ["file_path", "old_string", "new_string"]}}},
             {"type": "function", "function": {"name": "Read", "parameters": {"required": ["file_path"]}}}]
    files = {"a.py": "x = 1\nreturn x\n", "b.py": "y = 2\n"}
    messages = [{"role": "system"}, {"role": "user"},
                {"role": "assistant", "tool_calls": [call("Edit", {"file_path": "/repo/a.py", "old_string": "x = 1\nreturn x\n", "new_string": "z"})]},   # 2: hit
                {"role": "assistant", "tool_calls": [call("Edit", {"file_path": "/repo/b.py", "old_string": "y = 3\n", "new_string": "z"})]},            # 3: miss
                {"role": "assistant", "tool_calls": [call("Edit", "{not json")]},                                                                         # 4: invalid
                {"role": "assistant", "tool_calls": [call("Read", {"file_path": "/repo/a.py"})]},                                                        # 5: valid, not an edit
                {"role": "assistant", "content": "no call"}]                                                                                              # 6: edit turn, no call
    turns = [{"turn": 1, "message_count": 2, "category": "edit_function"},
             {"turn": 2, "message_count": 3, "category": "edit_function"},
             {"turn": 3, "message_count": 4, "category": "edit_function"},
             {"turn": 4, "message_count": 5, "category": "explore"},
             {"turn": 5, "message_count": 6, "category": "edit_function"}]
    result = sweep.edit_metrics([{"tools": tools, "messages": messages, "turns": turns}], files.get)
    check(result["edit_turns"] == 4 and result["edit_rate"] == 0.25, result)
    check(result["tool_calls"] == 4 and abs(result["tool_call_validity"] - 0.75) < 1e-9, result)


def test_divergence_and_speed():
    check(sweep.first_divergence([1, 2, 3], [1, 2, 4]) == 2, "differs at 2")
    check(sweep.first_divergence([1, 2, 3], [1, 2]) == 2 and sweep.first_divergence([1], [1]) == 1, "prefix / equal")
    a = sweep.agreement_summary([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 9], [1, 2, 3, 4]])
    check(a["identical"] == 1 and a["first_divergence_median"] == 3.5 and a["first_divergence_min"] == 3, a)
    report = {"long": {"prompt_tokens": 108000, "seconds": 41.5},
              "stream": {"during_long_prompt": {"gap_max_ms": 310.0, "gap_p99_ms": 200.0}}}
    m = sweep.speed_metrics(report, {"scout_ms": 3000.0, "main_ms": 9000.0, "kept_tokens": 21000, "uncached_tokens": 107000, "outcome": "applied"})
    check(m["total_s"] == 41.5 and m["scout_ms"] == 3000.0 and m["neighbour_gap_max_ms"] == 310.0, m)
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "rl.jsonl"
        log.write_text(json.dumps({"event": "request_done", "lossy_prefill": None}) + "\n"
                       + json.dumps({"event": "request_done", "lossy_prefill": {"uncached_tokens": 5, "scout_ms": 1}}) + "\n"
                       + json.dumps({"event": "request_done", "lossy_prefill": {"uncached_tokens": 107000, "scout_ms": 3000}}) + "\n")
        check(sweep.long_request_lossy_record(log, 0)["scout_ms"] == 3000, "biggest lossy record")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
for t in TESTS:
    t()
print(f"OK: {len(TESTS)} tests")
