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
    row("baseline", "needle", {"accuracy": {"32k": 1.0, "64k": 1.0, "100k": 1.0},
                               "samples": {"32k": 22, "64k": 22, "100k": 22}}),
    row("baseline", "agent_replay", {"edit_rate": 0.90, "tool_call_validity": 1.0}),
    row("baseline", "speed", {"total_s": 100.0, "neighbour_gap_max_ms": 900}),
    row("baseline", "greedy_agreement", {"outputs": [[1, 2, 3, 4]] * 2}),
]


LOSSY = {"requests": 6, "none": 1, "completed": 5}


def point_rows(point, needle=1.0, edit=0.90, total=39.0, lossy=LOSSY, samples=22):
    rows = [
        row(point, "needle", {"accuracy": {"32k": needle, "64k": needle, "100k": needle},
                              "samples": {"32k": samples, "64k": samples, "100k": samples}}),
        row(point, "agent_replay", {"edit_rate": edit, "tool_call_validity": 0.99}),
        row(point, "speed", {"total_s": total, "scout_ms": 4000, "main_ms": 30000, "kept_tokens": 20000,
                             "uncached_tokens": 100000, "neighbour_gap_max_ms": 300}),
        row(point, "greedy_agreement", {"outputs": [[1, 2, 3, 9], [1, 2, 3, 4]]}),
    ]
    if lossy is not None:
        for r in rows:
            r["metrics"]["lossy_outcomes"] = lossy
    rows[1]["metrics"]["edit_turns"] = 52
    return rows


def checks(rows, point):
    return {c["name"]: c for c in sweep.evaluate_point(rows, point)["checks"]}


def test_needle_threshold():
    rows = BASE + point_rows("good", needle=0.96) + point_rows("bad", needle=0.94)
    check(all(c["pass"] for n, c in checks(rows, "good").items() if n.startswith("needle")), "0.96x passes")
    check(all(c["pass"] is False for n, c in checks(rows, "bad").items() if n.startswith("needle")), "0.94x fails")
    check(abs(checks(rows, "good")["needle_32k"]["value"] - 0.96) < 1e-9, "ratio value")
    # 22 samples: one miss against a perfect baseline passes, two fail.
    rows = BASE + point_rows("one_miss", needle=21 / 22) + point_rows("two_miss", needle=20 / 22)
    check(checks(rows, "one_miss")["needle_64k"]["pass"] is True, "21/22 passes")
    check(checks(rows, "two_miss")["needle_64k"]["pass"] is False, "20/22 fails")
    # Too few samples on either side: not measured, never a pass or a fail.
    rows = BASE + point_rows("smoke", needle=1.0, samples=5)
    c = checks(rows, "smoke")["needle_100k"]
    check(c["pass"] is None and "5 / baseline 22 samples" in c["detail"], c)
    check(sweep.evaluate_point(rows, "smoke")["verdict"] == "INCOMPLETE", "a smoke needle run decides nothing")


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
          and extra["document_depth_percent_intervals"] == 11 and extra["tokenizer_path"] == "/t", extra)
    subsets = jobs[2]["backend_args"]["subset_list"]
    check(subsets == ["english", "chinese"], subsets)
    check(len(subsets) * extra["document_depth_percent_intervals"] >= sweep.NEEDLE_MIN_SAMPLES,
          "the config yields enough samples per length for the 0.95x gate")
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


def test_speed_report_validation():
    good = {"long": {"status": 200, "prompt_tokens": 108000, "cached_tokens": 0, "seconds": 40.0}}
    sweep.check_speed_report(good, 131072)
    bad_reports = [
        {"long": {"status": 500, "error": "boom", "prompt_tokens": None, "seconds": 0.3}},
        {"long": {**good["long"], "cached_tokens": 4096}},
        {"long": {**good["long"], "prompt_tokens": 60000}},
        {"long": {**good["long"], "prompt_tokens": 140000}},
        {},
    ]
    for report in bad_reports:
        try:
            sweep.check_speed_report(report, 131072)
        except RuntimeError:
            continue
        check(False, f"should be refused: {report}")
    check(sweep.speed_metrics(good, None)["cached_tokens"] == 0, "cached tokens recorded")


def test_resume_repairs_failures():
    rows = BASE + [row("p", "server", {}, ok=False, error="exit 1"), row("p", "needle", {}, ok=False, error="x")]
    rows += point_rows("p")  # later ok rows for every workload
    ev = sweep.evaluate_point(rows, "p")
    check(ev["errors"] == [] and ev["verdict"] == "PASS", f"superseded failures do not count: {ev['errors']}")
    only_server = BASE + [row("q", "needle", {}, ok=False, error="x"), row("q", "server", {}, ok=False, error="exit 1"),
                          row("q", "speed", {"total_s": 30.0}, ok=True)]
    errors = sweep.evaluate_point(only_server, "q")["errors"]
    check([e["workload"] for e in errors] == ["needle"], "a later ok row clears the server failure, not another workload's")
    later_fail = BASE + point_rows("r") + [row("r", "speed", {}, ok=False, error="late")]
    check(sweep.evaluate_point(later_fail, "r")["verdict"] == "FAIL", "a failure after the ok row stands")


def test_lossy_evidence():
    nothing = {"requests": 6, "none": 6}
    rows = BASE + point_rows("noop", lossy=nothing) + point_rows("unknown", lossy=None) + point_rows("ran")
    check(sweep.evaluate_point(rows, "noop")["verdict"] == "INCOMPLETE", "zero completed lossy is not a pass")
    check(sweep.evaluate_point(rows, "unknown")["verdict"] == "INCOMPLETE", "no outcome counts is not a pass")
    check(sweep.evaluate_point(rows, "ran")["verdict"] == "PASS", "completed lossy requests pass")
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "rl.jsonl"
        recs = [{"event": "request_done", "lossy_prefill": None},
                {"event": "request_done", "lossy_prefill": {"outcome": "completed"}},
                {"event": "request_done", "lossy_prefill": {"outcome": "completed"}},
                {"event": "request_done", "lossy_prefill": {"outcome": "skipped"}},
                {"event": "request_started"}]
        log.write_text("".join(json.dumps(r) + "\n" for r in recs))
        counts = sweep.lossy_outcome_counts(log)
        check(counts == {"requests": 4, "none": 1, "completed": 2, "skipped": 1}, counts)
        summary = [{"event": "request_done", "lossy_prefill": {"outcome": "completed", "strategy": "summary",
                                                                "summary_probe": "cold"}},
                   {"event": "request_done", "lossy_prefill": {"outcome": "skipped", "strategy": "summary",
                                                                "summary_probe": "timeout"}},
                   {"event": "request_done", "lossy_prefill": {"outcome": "completed", "strategy": "summary",
                                                                "summary_probe": None}}]
        log.write_text("".join(json.dumps(r) + "\n" for r in summary))
        counts = sweep.lossy_outcome_counts(log)
        check(counts == {"requests": 3, "none": 0, "completed": 2, "skipped": 1, "summary_probe_cold": 1,
                         "summary_probe_timeout": 1}, counts)
    timed = BASE + point_rows("sum", lossy={"requests": 5, "none": 0, "completed": 3, "skipped": 2,
                                           "summary_probe_timeout": 2})
    ev = sweep.evaluate_point(timed, "sum")
    ran_checks = [c for c in ev["checks"] if c["name"].startswith("lossy_ran_")]
    check(ran_checks and all("2 summary probe timeout(s)" in c["detail"] for c in ran_checks), ran_checks)
    check("3 (t2)" in sweep.lossy_cell(timed, "sum"), sweep.lossy_cell(timed, "sum"))
    text = sweep.render_summary(rows)
    ran = next(l for l in text.splitlines() if l.startswith("| ran |"))
    check("5/5/-/5/5" in ran and "| 52 |" in ran, ran)


def test_agent_command_passes_repo_and_failure_inputs():
    cmd = sweep.agent_replay_command(port=1, out=Path("o"), request_log=Path("l"), sessions=26, seed=7,
                                     repo=Path("/r"), commit="abc")
    check(cmd[cmd.index("--repo") + 1] == "/r" and cmd[cmd.index("--commit") + 1] == "abc"
          and cmd[cmd.index("--sessions") + 1] == "26", cmd)
    args = sweep.parse_args(["--from-json", "x.jsonl"])
    check(args.agent_sessions == 26 and args.greedy_chars == 330_000, "defaults sized for the gate / ~100K tokens")


def test_greedy_generate_needs_the_log_record():
    import http.server
    import threading
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "rl.jsonl"
        log.write_text(json.dumps({"event": "request_done", "request": {"response_id": "old"}, "result": {"generated_token_ids": [9]}}) + "\n")
        start = log.stat().st_size
        known = {"logged": True}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                rid = "r1" if known["logged"] else "r2"
                if known["logged"]:
                    with log.open("a") as f:
                        f.write(json.dumps({"event": "request_done", "request": {"response_id": "r1"},
                                            "result": {"generated_token_ids": [1, 2, 3]}}) + "\n")
                body = json.dumps({"id": rid, "choices": [{"message": {"content": "hi"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            ids, cursor = sweep.greedy_generate(url, "m", "p", 8, log, start, wait_seconds=5)
            check(ids == [1, 2, 3] and cursor == log.stat().st_size, (ids, cursor))
            known["logged"] = False
            try:
                sweep.greedy_generate(url, "m", "p", 8, log, cursor, wait_seconds=1)
                check(False, "a missing record must fail, not fall back to characters")
            except RuntimeError:
                pass
        finally:
            server.shutdown()


def test_kept_blocks_and_jaccard():
    # Blocks of 64 kept whole; a partial run covers no block; the last block is clipped to the span.
    check(sweep.kept_blocks([[0, 128], [130, 200], [256, 300]], 300, 64) == {0, 1, 4}, "whole blocks only")
    check(sweep.kept_blocks([[10, 70]], 300, 64) == set(), "a run inside two blocks keeps neither")
    # scored_only drops the forced head (first 64) and recent (last 100 -> blocks ending past 200).
    check(sweep.kept_blocks([[0, 300]], 300, 64, True, 64, 100) == {1, 2}, "head and recent left out")
    check(sweep.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5 and sweep.jaccard(set(), set()) is None, "jaccard")


def jaccard_rows(a4_runs, int8_runs, outcome="completed", uncached=1000, head="0", recent="0"):
    def kept(runs):
        return [{"prompt": i, "outcome": outcome, "uncached_tokens": uncached, "kept_runs": r}
                for i, r in enumerate(runs)]
    common = {"block": "64", "strategy": "select", "head": head, "recent": recent}
    return [
        {"point": "b64_a1", "params": dict(common, scout_a4="1"), "workload": "jaccard", "ok": True,
         "error": None, "metrics": {"kept": kept(a4_runs), "lossy_outcomes": LOSSY}},
        {"point": "b64_a0", "params": dict(common, scout_a4="0"), "workload": "jaccard", "ok": True,
         "error": None, "metrics": {"kept": kept(int8_runs), "lossy_outcomes": LOSSY}},
    ]


def test_jaccard_gate():
    same = [[[0, 640]], [[0, 64], [640, 1000]]]
    rows = BASE + jaccard_rows(same, same)
    check(sweep.scout_a4_partner(rows, "b64_a1") == "b64_a0", "the partner is the scout_a4=0 twin")
    check(sweep.scout_a4_partner(rows, "b64_a0") is None, "an int8 point has no partner")
    gate = checks(rows, "b64_a1")["jaccard"]
    check(gate["pass"] is True and gate["value"] == 1.0, gate)
    check("jaccard" not in checks(rows, "b64_a0"), "only the a4 point carries the gate")
    # 9 of 10 blocks shared -> 0.9 passes; 8 of 10 -> 0.8 fails (the minimum over prompts).
    tenth = [[[0, 640]], [[0, 640]]]
    nine = [[[0, 576], [640, 704]], [[0, 640]]]
    eight = [[[0, 512], [640, 768]], [[0, 640]]]
    check(abs(checks(BASE + jaccard_rows(nine, tenth), "b64_a1")["jaccard"]["value"] - 9 / 11) < 1e-9,
          "9 shared of 11 distinct blocks")
    check(checks(BASE + jaccard_rows(nine, tenth), "b64_a1")["jaccard"]["pass"] is False, "9/11 < 0.9 fails")
    check(checks(BASE + jaccard_rows(eight, tenth), "b64_a1")["jaccard"]["pass"] is False, "0.67 fails")
    nineteen = [[[0, 1216]], [[0, 1216]]]
    near = [[[0, 1152], [1216, 1280]], [[0, 1216]]]
    near_rows = BASE + jaccard_rows(near, nineteen, uncached=2000)
    value = checks(near_rows, "b64_a1")["jaccard"]["value"]
    check(abs(value - 18 / 20) < 1e-9 and checks(near_rows, "b64_a1")["jaccard"]["pass"] is True,
          ("18/20 = 0.9 passes", value))
    # Ruling R11: the gate is over the score-chosen blocks. Both keep blocks 0..19 (forced head of
    # 1280 tokens) and differ in one chosen block: all kept blocks 20/22 = 0.909 would pass, but
    # the score-chosen sets {20} and {21} share nothing, so the gate fails.
    forced_a4 = [[[0, 1280], [1280, 1344]]]
    forced_8 = [[[0, 1280], [1344, 1408]]]
    forced = BASE + jaccard_rows(forced_a4, forced_8, uncached=2000, head="1280")
    gate = checks(forced, "b64_a1")["jaccard"]
    check(gate["pass"] is False and gate["value"] == 0.0 and "all kept blocks 0.909" in gate["detail"], gate)
    overlap = sweep.jaccard_metrics(forced, "b64_a1")
    check(abs(overlap["jaccard_min"] - 20 / 22) < 1e-9 and overlap["scored_min"] == 0.0, overlap)
    # With the default head/recent the forced blocks are left out of the gate too.
    defaults = BASE + jaccard_rows(nine, tenth)
    for r in defaults:
        if r["workload"] == "jaccard":
            r["params"].pop("head"), r["params"].pop("recent")
    overlap = sweep.jaccard_metrics(defaults, "b64_a1")
    check(overlap["scored_min"] is None and overlap["unmeasured"] == [0, 1],
          ("with recent 4096 > the span no block is score-chosen: nothing measured", overlap))
    gate = checks(defaults, "b64_a1")["jaccard"]
    check(gate["pass"] is None and "not measured" in gate["detail"], ("not measured, never a vacuous pass", gate))
    check(sweep.evaluate_point(defaults, "b64_a1")["verdict"] != "PASS", "an unmeasured gate cannot PASS")
    # A prompt that did not complete lossy prefill on one side fails rather than being skipped.
    broken = jaccard_rows(same, same)
    broken[1]["metrics"]["kept"][1]["outcome"] = "skipped"
    gate = checks(BASE + broken, "b64_a1")["jaccard"]
    check(gate["pass"] is False and "unmatched" in gate["detail"], gate)
    # Without the partner's rows the gate is not measured, and the point is INCOMPLETE, not PASS.
    alone = BASE + jaccard_rows(same, same)[:1]
    check(checks(alone, "b64_a1")["jaccard"]["pass"] is None, "no partner, not measured")
    text = sweep.render_summary(BASE + jaccard_rows(same, same))
    check("## Int4 scout" in text and "| b64_a1 | b64_a0 | 2 | 1.000 |" in text, text)


def test_jaccard_record_needs_the_detail_flag():
    rec = sweep.jaccard_record(3, {"outcome": "completed", "uncached_tokens": 10, "kept_tokens": 4,
                                   "kept_runs": [[0, 4]]})
    check(rec["prompt"] == 3 and rec["kept_runs"] == [[0, 4]], rec)
    check(sweep.jaccard_record(1, None)["outcome"] is None, "a request without lossy prefill")
    try:
        sweep.jaccard_record(1, {"outcome": "completed"})
        check(False, "a record without kept_runs must fail")
    except RuntimeError:
        pass
    args = sweep.parse_args(["--from-json", "x.jsonl", "--workloads", "jaccard"])
    check(args.workload_names == ["jaccard"] and "jaccard" not in sweep.WORKLOADS, "jaccard is opt-in")


def test_preflight():
    # The record must show the path completed; anything else names what it showed.
    ok = {"event": "request_done", "lossy_prefill": {"outcome": "completed", "uncached_tokens": 60000,
                                                       "kept_tokens": 12000}}
    check(sweep.check_preflight(ok)["kept_tokens"] == 12000, "a completed preflight passes")
    for record, needle in (({"event": "request_done"}, "did not take the lossy path"),
                           (None, "did not take the lossy path"),
                           ({"lossy_prefill": {"outcome": "skipped", "summary_probe": "timeout"}},
                            "'skipped'")):
        try:
            sweep.check_preflight(record)
            check(False, f"preflight accepted {record}")
        except RuntimeError as exc:
            check(needle in str(exc), str(exc))
    # A conversation, long enough for the summary strategy's protected head and tail, nonce first.
    messages = sweep.preflight_messages("x" * 3000, "n1", turns=30)
    check(len(messages) == 61 and messages[0]["content"].startswith("[preflight n1]")
          and messages[-1]["role"] == "user", messages[:2])
    check(sum(len(m["content"]) for m in messages if m["role"] == "user") > 3000, "all the text is sent")

    # run_point: a failed preflight aborts the point before any workload; the baseline has none.
    with tempfile.TemporaryDirectory() as d:
        args = sweep.parse_args(["--server-cmd", "serve", "--model", "m", "--out", d,
                                 "--workloads", "agent_replay,speed"])
        s = sweep.Sweep(args)
        calls = []

        class Proc:
            pass

        s.start_server = lambda point, params, extra, tag: (calls.append(("start", tag)) or (Proc(), Path(d) / "rl"))
        real_stop = sweep.stop_server
        sweep.stop_server = lambda proc: calls.append(("stop",))
        s.run_workload = lambda point, params, name, log: calls.append(("run", name))
        outcomes = iter([RuntimeError("preflight: lossy_prefill.outcome is 'skipped'")])

        def failing(log):
            raise next(outcomes)

        s.preflight = failing
        s.run_point("p1", {"keep": "0.2"}, ["agent_replay", "speed"])
        check(calls == [("start", "quality"), ("stop",)], calls)
        rows = sweep.load_rows([s.results])
        check(rows[-1]["workload"] == "preflight" and not rows[-1]["ok"]
              and "quality server" in rows[-1]["error"], rows[-1])
        check(sweep.evaluate_point(rows, "p1")["verdict"] == "FAIL", "an aborted point fails")
        calls.clear()
        s.preflight = lambda log: {"outcome": "completed", "kept_tokens": 1}
        s.run_point("p2", {"keep": "0.2"}, ["agent_replay", "speed"])
        check(calls == [("start", "quality"), ("run", "agent_replay"), ("stop",),
                        ("start", "speed"), ("run", "speed"), ("stop",)], calls)
        check(sum(1 for r in sweep.load_rows([s.results]) if r["point"] == "p2" and r["workload"] == "preflight"
                  and r["ok"]) == 2, "one preflight per lossy server")
        calls.clear()
        s.preflight = failing  # never called for the baseline
        s.run_point(sweep.BASELINE, None, ["agent_replay"])
        check(calls == [("start", "quality"), ("run", "agent_replay"), ("stop",)], calls)
        sweep.stop_server = real_stop


def test_needle_answer_judge():
    ref = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    q = "\nWhat is the best thing to do in San Francisco?\n"
    hit = sweep.needle_answer_hit
    check(hit("Eat a sandwich and sit in Dolores Park on a sunny day.", ref, q), "short correct answer")
    check(hit("The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
              ref, q), "full sentence")
    check(not hit("Visit the Golden Gate Bridge.", ref, q), "wrong answer")
    check(not hit("Eat a sandwich.", ref, q), "partial answer")
    check(not hit("", ref, q), "empty answer")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        for length in sweep.NEEDLE_LENGTHS:
            d = run / "backends" / f"needle_{length}" / "reviews" / "m"
            d.mkdir(parents=True)
            lines = []
            for i, pred in enumerate(["Eat a sandwich and sit in Dolores Park on a sunny day.", "No idea."]):
                lines.append(json.dumps({"target": ref, "messages": [
                    {"role": "system", "content": "short"},
                    {"role": "user", "content": f"<text>...</text>\n<question>{q}</question>"}],
                    "sample_score": {"score": {"prediction": pred, "value": {f"C D{i}": 0}}}}))
            (d / "needle_haystack_english.jsonl").write_text("\n".join(lines) + "\n")
        r = sweep.rescore_needle_run(run)
        check(r["accuracy"] == {"32k": 0.5, "64k": 0.5, "100k": 0.5}, r)
        check(r["samples"] == {"32k": 2, "64k": 2, "100k": 2} and r["rule_accuracy"]["32k"] == 0.0, r)
        check(sweep.rescore_needle_run(run / "missing") is None, "no reviews -> None")


def test_agent_edit_turns_and_gate_source():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "obs"}]
    transcripts = [{"session": 0, "tools": [], "messages": msgs + [{"role": "assistant", "content": "x"}],
                    "turns": [{"turn": 0, "message_count": 2, "category": "explore", "response_id": "a"},
                              {"turn": 1, "message_count": 4, "category": "edit_function", "response_id": "b"},
                              {"turn": 2, "message_count": 4, "category": "edit_function", "response_id": "c"}]}]
    samples = [{"response_id": "a", "prompt_tokens": 50000, "max_tokens": 10},
               {"response_id": "b", "prompt_tokens": 40000, "max_tokens": 700},
               {"response_id": "c", "prompt_tokens": 20000, "max_tokens": 700}]
    got = sweep.agent_edit_turns(transcripts, samples)
    check(len(got) == 1 and got[0]["turn"] == 1 and got[0]["messages"] == msgs and got[0]["max_tokens"] == 700, got)
    # The Edit gate reads agent_edit when the point and the baseline have it, else agent_replay.
    rows = BASE + point_rows("p", edit=0.90)
    check(checks(rows, "p")["edit"]["pass"] is True, "replay rows gate when no agent_edit")
    rows += [row("baseline", "agent_edit", {"edit_rate": 0.9, "edit_turns": 30}),
             row("p", "agent_edit", {"edit_rate": 0.8, "edit_turns": 30, "lossy_outcomes": LOSSY})]
    c = checks(rows, "p")["edit"]
    check(c["pass"] is False and abs(c["value"] + 10.0) < 1e-9 and "cold" in c["detail"], c)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
for t in TESTS:
    t()
print(f"OK: {len(TESTS)} tests")
