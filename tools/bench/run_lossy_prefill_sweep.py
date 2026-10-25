#!/usr/bin/env python3
"""Quality and speed sweep for lossy long-prompt prefill (docs/maintainer/lossy-long-prefill.md).

For each grid point this starts a fresh `ninfer-serve` with the lossy flags, runs the chosen
workloads against it, appends one row per workload to `results.jsonl`, and finally writes
`summary.md`: every acceptance check against the fixed numbers below, marked pass or FAIL, against
the baseline point (the same server with lossy prefill off). Needs a GPU, an artifact and, for the
needle workload, the eval/ virtualenv with the Needle-in-a-Haystack corpus; run it under the GPU
lock on a shared card (CLAUDE.md). The unit test (test_lossy_prefill_sweep.py) needs none of that.

    python tools/bench/run_lossy_prefill_sweep.py \\
        --server-cmd "build/apps/ninfer-serve models/qwen3_5_27b.ninfer --kv-dtype rk4v4 \\
                      --max-context 131072 --max-concurrency 2 --prefill-cublas" \\
        --model qwen3.5-27b --tokenizer-path /path/to/hf-tokenizer-dir --needle-corpus /path/to/corpus \\
        --grid filter_layer=7,11,15,23,31 keep=0.1,0.2,0.3,0.5 block=32,64,128 strategy=select \\
        --workloads needle,agent_replay,greedy_agreement,speed --out profiles/lossy-sweep

Pieces: `--workloads` and `--only REGEX` (on point ids) run a subset; `--resume` skips (point,
workload) pairs already ok in results.jsonl; `--from-json results.jsonl [more.jsonl]` only
re-summarises (no server). A server that fails to start for a point is recorded (workload "server")
and the sweep goes on. The baseline runs first unless `--skip-baseline`; then give the earlier run's
results to `--baseline-from`.

Acceptance (fixed, docs and the task's Global Constraints; not to be changed after seeing results):
needle accuracy at 32K/64K/100K >= 95% of baseline; agent-replay Edit exact-match (old_string in the
file) no more than 2 points under baseline; 100K cold prefill <= 40% of the baseline's time.
Greedy agreement (first divergence from the baseline's greedy output) replaces the spec's
teacher-forced KL in v1 and has no pass/fail threshold; it is reported only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- acceptance numbers (fixed) -------------------------------------------------------------

NEEDLE_MIN_RATIO = 0.95
EDIT_MAX_DROP_POINTS = 2.0
TIME_MAX_RATIO = 0.40

NEEDLE_LENGTHS = {"32k": 32768, "64k": 65536, "100k": 100000}
NEEDLE_DEPTHS = 5  # 0, 25, 50, 75, 100 percent
SPEED_LONG_TOKENS = 108_000
SPEED_LAYER_SLICES = 8
GREEDY_PROMPTS = 20

WORKLOADS = ("needle", "agent_replay", "greedy_agreement", "speed")
BASELINE = "baseline"

# ---- grid ------------------------------------------------------------------------------------

# grid key -> (server flag, short id prefix). `scout_a4` is a switch: 1 adds the flag, 0 omits it.
KNOBS: dict[str, tuple[str, str]] = {
    "filter_layer": ("--lossy-prefill-filter-layer", "fl"),
    "keep": ("--lossy-prefill-keep", "k"),
    "block": ("--lossy-prefill-block", "b"),
    "recent": ("--lossy-prefill-recent", "r"),
    "head": ("--lossy-prefill-head", "h"),
    "query_rows": ("--lossy-prefill-query-rows", "q"),
    "pool_window": ("--lossy-prefill-pool-window", "pw"),
    "skip": ("--lossy-prefill-skip", "sk"),
    "scout_a4": ("--lossy-prefill-scout-a4", "a"),
}
GRID_KEYS = tuple(KNOBS) + ("strategy", "min_tokens")
STRATEGIES = ("select", "summary")
DEFAULT_MIN_TOKENS = "30000"


def parse_grid(items: Sequence[str]) -> dict[str, list[str]]:
    grid: dict[str, list[str]] = {}
    for item in items:
        key, sep, values = item.partition("=")
        if not sep or key not in GRID_KEYS or not values:
            raise ValueError(f"--grid entries are KEY=V1,V2,...; KEY one of {', '.join(GRID_KEYS)}: {item!r}")
        parsed = [v for v in values.split(",") if v]
        if key == "strategy" and any(v not in STRATEGIES for v in parsed):
            raise ValueError(f"strategy must be one of {STRATEGIES}: {item!r}")
        if key == "scout_a4" and any(v not in ("0", "1") for v in parsed):
            raise ValueError(f"scout_a4 must be 0 or 1: {item!r}")
        grid[key] = parsed
    return grid


def point_id(params: dict[str, str]) -> str:
    if params.get("strategy") == "summary":
        return "summary"
    parts = [f"{KNOBS[k][1]}{params[k]}" for k in KNOBS if k in params]
    if params.get("min_tokens") not in (None, DEFAULT_MIN_TOKENS):
        parts.append(f"m{params['min_tokens']}")
    return "_".join(parts) or "default"


def expand_grid(grid: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid, as [{"id", "params"}], select points first.

    Knobs not named in the grid keep the server's defaults (no flag). The summary strategy ignores
    the selection knobs, so it expands to one point (`summary`) whatever the knobs are.
    """
    strategies = grid.get("strategy", ["select"])
    knob_keys = [k for k in KNOBS if k in grid]
    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    for strategy in strategies:
        if strategy == "summary":
            params = {"strategy": "summary"}
            if "min_tokens" in grid:
                params["min_tokens"] = grid["min_tokens"][0]
            combos: list[dict[str, str]] = [params]
        else:
            combos = []
            for values in itertools.product(*(grid[k] for k in knob_keys)):
                params = dict(zip(knob_keys, values))
                params["strategy"] = "select"
                if "min_tokens" in grid:
                    params["min_tokens"] = grid["min_tokens"][0]
                combos.append(params)
        for params in combos:
            pid = point_id(params)
            if pid not in seen:
                seen.add(pid)
                points.append({"id": pid, "params": params})
    return points


def lossy_flags(params: dict[str, str]) -> list[str]:
    """Server flags for one grid point. `--lossy-prefill-strategy` only for summary (it is Task 16's
    flag; select runs must work on a server that does not have it)."""
    flags = ["--lossy-prefill-min-tokens", params.get("min_tokens", DEFAULT_MIN_TOKENS)]
    for key, (flag, _) in KNOBS.items():
        if key not in params:
            continue
        if key == "scout_a4":
            if params[key] == "1":
                flags.append(flag)
        else:
            flags += [flag, params[key]]
    if params.get("strategy") == "summary":
        flags += ["--lossy-prefill-strategy", "summary"]
    return flags


def server_command(base_cmd: str, port: int, request_log: Path, params: dict[str, str] | None,
                   extra: Sequence[str] = ()) -> list[str]:
    """`params` None is the baseline: lossy prefill off (no lossy flags)."""
    cmd = shlex.split(base_cmd) + ["--port", str(port), "--request-log-jsonl", str(request_log)]
    cmd += list(extra)
    if params is not None:
        cmd += lossy_flags(params)
    return cmd


# ---- workload command construction -----------------------------------------------------------

def needle_config(*, base_url: str, model: str, corpus: str, tokenizer_path: str, runs_dir: str,
                  lengths: dict[str, int] = NEEDLE_LENGTHS, depths: int = NEEDLE_DEPTHS,
                  limit: int | None = None) -> dict[str, Any]:
    """An eval/ config (JSON, which is YAML) after eval/configs/qwen3_6_35b_needle_haystack.yaml:
    one evalscope needle_haystack job per context length, `depths` document depths 0-100%."""
    jobs = []
    for name, tokens in lengths.items():
        job: dict[str, Any] = {
            "id": f"needle_{name}",
            "backend": "evalscope",
            "dataset": "needle_haystack",
            "target": "lossy_api",
            "max_concurrency": 1,
            "generation": {"temperature": 0, "max_tokens": 512, "seed": 42,
                           "extra_body": {"enable_thinking": False}},
            "backend_args": {
                "subset_list": ["english"],
                "judge_strategy": "rule",
                "dataset_args": {"local_path": corpus, "extra_params": {
                    "context_lengths_min": tokens, "context_lengths_max": tokens,
                    "context_lengths_num_intervals": 1,
                    "document_depth_percent_min": 0, "document_depth_percent_max": 100,
                    "document_depth_percent_intervals": depths,
                    "tokenizer_path": tokenizer_path, "show_score": True}},
            },
        }
        if limit is not None:
            job["limit"] = limit
        jobs.append(job)
    return {
        "schema_version": 1,
        "targets": {"lossy_api": {
            "protocol": "openai_chat", "base_url": f"{base_url}/v1", "model": model,
            "max_concurrency": 1,
            "request": {"timeout_seconds": 86400, "retries": 2, "retry_interval_seconds": 10}}},
        "suites": {"needle": {"jobs": jobs}},
        "runtime": {"max_parallel_jobs": 1, "runs_dir": runs_dir,
                    "progress": {"enabled": False, "refresh_seconds": 5, "heartbeat_seconds": 60},
                    "samples": {"retention": "all"}},
    }


def needle_command(eval_python: str, config_path: Path) -> list[str]:
    return [eval_python, "-m", "ninfer_eval", "run", "--config", str(config_path), "--suite", "needle"]


def agent_replay_command(*, port: int, out: Path, request_log: Path, sessions: int, seed: int | None,
                         min_context: int = 30000, max_context: int = 120000) -> list[str]:
    cmd = [sys.executable, "-m", "tools.bench.agent_replay", "run", "--port", str(port),
           "--sessions", str(sessions), "--concurrency", "1",
           "--min-context", str(min_context), "--max-context", str(max_context),
           "--request-log", str(request_log), "--out", str(out)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


def stall_command(*, base_url: str, model: str, output: Path, long_tokens: int = SPEED_LONG_TOKENS,
                  seed: int = 20261025, words_per_token: float | None = None) -> list[str]:
    cmd = [sys.executable, str(REPO_ROOT / "tools/bench/run_prefill_stall.py"),
           "--base-url", base_url, "--model", model, "--long-tokens", str(long_tokens),
           "--seed", str(seed), "--output", str(output)]
    if words_per_token is not None:
        cmd += ["--words-per-token", str(words_per_token)]
    return cmd


# ---- metric extraction -----------------------------------------------------------------------

def parse_needle_summary(summary: dict[str, Any]) -> dict[str, float]:
    """ninfer_eval summary.json -> {"32k": accuracy, ...}; a failed job is an error."""
    out: dict[str, float] = {}
    for result in summary.get("results", []):
        job = str(result.get("job_id", ""))
        if not job.startswith("needle_"):
            continue
        if result.get("status") not in ("completed", "succeeded", "ok", "success"):
            raise RuntimeError(f"{job}: status {result.get('status')}: {result.get('error')}")
        score = (result.get("metrics") or {}).get(result.get("primary_metric"))
        if not isinstance(score, (int, float)):
            raise RuntimeError(f"{job}: no primary metric score")
        out[job.removeprefix("needle_")] = float(score)
    if not out:
        raise RuntimeError("needle run produced no needle_* results")
    return out


def edit_metrics(transcripts: Sequence[dict[str, Any]],
                 read_file: Callable[[str], str | None]) -> dict[str, Any]:
    """Tool-call validity and the Edit `old_string`-in-file rate from agent_replay's transcripts.jsonl.

    `turns[i].message_count` is the message count when the model was asked, so the model's reply is
    `messages[message_count]` (the scripted observations before it are assistant messages too and
    are not counted). Validity: over all model tool calls, the name is one of the session's tools and
    the arguments are a JSON object holding every required property. Edit rate: over the
    `edit_function` turns, the share whose reply made an Edit whose `old_string` occurs verbatim in
    the file it names (`read_file` gets the path after the /repo prefix; None means unreadable). A
    turn that made no valid Edit counts as a miss.
    """
    calls = valid_calls = 0
    edit_turns = edit_hits = 0
    for session in transcripts:
        required = {t["function"]["name"]: set(t["function"]["parameters"].get("required", []))
                    for t in session.get("tools", [])}
        messages = session.get("messages", [])
        for turn in session.get("turns", []):
            index = turn.get("message_count")
            reply = messages[index] if isinstance(index, int) and index < len(messages) else None
            tool_calls = (reply or {}).get("tool_calls") or []
            hit = False
            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name")
                try:
                    args = json.loads(function.get("arguments") or "")
                except (TypeError, ValueError):
                    args = None
                ok = isinstance(args, dict) and name in required and required[name] <= args.keys()
                calls += 1
                valid_calls += bool(ok)
                if ok and name == "Edit" and turn.get("category") == "edit_function":
                    path = str(args["file_path"]).removeprefix("/repo/")
                    text = read_file(path)
                    if text is not None and isinstance(args["old_string"], str) \
                            and args["old_string"] and args["old_string"] in text:
                        hit = True
            if turn.get("category") == "edit_function":
                edit_turns += 1
                edit_hits += hit
    return {
        "tool_calls": calls,
        "tool_call_validity": valid_calls / calls if calls else None,
        "edit_turns": edit_turns,
        "edit_rate": edit_hits / edit_turns if edit_turns else None,
    }


def first_divergence(reference: Sequence[Any], candidate: Sequence[Any]) -> int:
    """Index of the first position where the two sequences differ; the shorter length when one is a
    prefix of the other; their common length when equal."""
    for i, (a, b) in enumerate(zip(reference, candidate)):
        if a != b:
            return i
    return min(len(reference), len(candidate))


def agreement_summary(reference: Sequence[Sequence[Any]], candidate: Sequence[Sequence[Any]]) -> dict[str, Any]:
    divs = [first_divergence(r, c) for r, c in zip(reference, candidate)]
    identical = sum(1 for r, c in zip(reference, candidate) if list(r) == list(c))
    fractions = [d / len(r) for d, r in zip(divs, reference) if len(r)]
    return {
        "prompts": len(divs),
        "identical": identical,
        "first_divergence_median": statistics.median(divs) if divs else None,
        "first_divergence_min": min(divs) if divs else None,
        "agreed_fraction_mean": statistics.fmean(fractions) if fractions else None,
    }


def speed_metrics(stall_report: dict[str, Any], lossy_record: dict[str, Any] | None) -> dict[str, Any]:
    """From run_prefill_stall's report and, for a lossy run, the long request's `lossy_prefill`
    request-log object (scout_ms, main_ms, kept/uncached tokens, outcome)."""
    long = stall_report.get("long") or {}
    during = (stall_report.get("stream") or {}).get("during_long_prompt") or {}
    out: dict[str, Any] = {
        "prompt_tokens": long.get("prompt_tokens"),
        "total_s": long.get("seconds"),
        "neighbour_gap_max_ms": during.get("gap_max_ms"),
        "neighbour_gap_p99_ms": during.get("gap_p99_ms"),
    }
    if lossy_record:
        for key in ("scout_ms", "main_ms", "kept_tokens", "uncached_tokens", "filter_layer", "outcome"):
            out[key] = lossy_record.get(key)
    return out


def long_request_lossy_record(request_log: Path, offset: int) -> dict[str, Any] | None:
    """The `lossy_prefill` object with the most uncached tokens among the request_done records
    written after `offset` (the long prompt; the stream's short request never takes the path)."""
    best: dict[str, Any] | None = None
    if not request_log.exists():
        return None
    with request_log.open("rb") as log:
        log.seek(offset)
        for raw in log:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            lossy = record.get("lossy_prefill") if record.get("event") == "request_done" else None
            if lossy and (best is None or lossy.get("uncached_tokens", 0) > best.get("uncached_tokens", 0)):
                best = lossy
    return best


# ---- the summariser --------------------------------------------------------------------------

def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _metric(rows: list[dict[str, Any]], point: str, workload: str) -> dict[str, Any] | None:
    found = None
    for row in rows:  # the last ok row wins (a rerun replaces an earlier one)
        if row["point"] == point and row["workload"] == workload and row.get("ok"):
            found = row.get("metrics") or {}
    return found


def check(name: str, value: float | None, passed: bool | None, detail: str) -> dict[str, Any]:
    return {"name": name, "value": value, "pass": passed, "detail": detail}


def evaluate_point(rows: list[dict[str, Any]], point: str) -> dict[str, Any]:
    """Every acceptance check for one point against the baseline rows. `pass` is True, False or None
    (not measured: the workload did not run or failed, or the baseline lacks the number)."""
    checks: list[dict[str, Any]] = []
    needle, base_needle = _metric(rows, point, "needle"), _metric(rows, BASELINE, "needle")
    for length in NEEDLE_LENGTHS:
        value = base = None
        if needle and length in (needle.get("accuracy") or {}):
            value = needle["accuracy"][length]
        if base_needle and length in (base_needle.get("accuracy") or {}):
            base = base_needle["accuracy"][length]
        if value is None or not base:
            checks.append(check(f"needle_{length}", None, None, "not measured"))
            continue
        ratio = value / base
        checks.append(check(f"needle_{length}", ratio, ratio >= NEEDLE_MIN_RATIO - 1e-9,
                            f"{value:.3f} / baseline {base:.3f} (need >= {NEEDLE_MIN_RATIO:.2f}x)"))
    agent, base_agent = _metric(rows, point, "agent_replay"), _metric(rows, BASELINE, "agent_replay")
    if agent and base_agent and agent.get("edit_rate") is not None and base_agent.get("edit_rate") is not None:
        delta = (agent["edit_rate"] - base_agent["edit_rate"]) * 100.0
        checks.append(check("edit", delta, delta >= -EDIT_MAX_DROP_POINTS - 1e-9,
                            f"{delta:+.1f} points vs baseline (need >= -{EDIT_MAX_DROP_POINTS:.0f})"))
    else:
        checks.append(check("edit", None, None, "not measured"))
    speed, base_speed = _metric(rows, point, "speed"), _metric(rows, BASELINE, "speed")
    if speed and base_speed and speed.get("total_s") and base_speed.get("total_s"):
        ratio = speed["total_s"] / base_speed["total_s"]
        checks.append(check("time", ratio, ratio <= TIME_MAX_RATIO + 1e-9,
                            f"{speed['total_s']:.1f} s / baseline {base_speed['total_s']:.1f} s "
                            f"(need <= {TIME_MAX_RATIO:.2f}x)"))
    else:
        checks.append(check("time", None, None, "not measured"))
    failed = [r for r in rows if r["point"] == point and not r.get("ok")]
    if any(c["pass"] is False for c in checks) or failed:
        verdict = "FAIL"
    elif all(c["pass"] for c in checks):
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"
    return {"point": point, "checks": checks, "verdict": verdict, "errors": failed}


def _mark(c: dict[str, Any], fmt: str) -> str:
    if c["pass"] is None:
        return "-"
    return f"{format(c['value'], fmt)} {'pass' if c['pass'] else 'FAIL'}"


def points_in(rows: list[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    for row in rows:
        if row["point"] != BASELINE and row["point"] not in order:
            order.append(row["point"])
    return order


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lossy prefill sweep",
        "",
        "Acceptance (fixed): needle >= 0.95x baseline at 32K/64K/100K; Edit within 2 points of "
        "baseline; 100K cold prefill <= 0.40x baseline time. Greedy agreement and the speed "
        "columns are reported, not gated.",
        "",
        "| point | needle 32k | needle 64k | needle 100k | Edit (pts) | time (x) | scout ms | main ms "
        "| kept/uncached | neighbour gap ms | tool-call valid | greedy median div | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    evals = []
    for point in points_in(rows):
        ev = evaluate_point(rows, point)
        evals.append(ev)
        by = {c["name"]: c for c in ev["checks"]}
        speed = _metric(rows, point, "speed") or {}
        agent = _metric(rows, point, "agent_replay") or {}
        greedy = _metric(rows, point, "greedy_agreement") or {}
        base_greedy = _metric(rows, BASELINE, "greedy_agreement") or {}
        agreement = None
        if greedy.get("outputs") and base_greedy.get("outputs"):
            agreement = agreement_summary(base_greedy["outputs"], greedy["outputs"])

        def num(value: Any, fmt: str) -> str:
            return format(value, fmt) if isinstance(value, (int, float)) else "-"

        kept = (f"{speed['kept_tokens']}/{speed['uncached_tokens']}"
                if speed.get("kept_tokens") is not None else "-")
        lines.append(
            f"| {point} | {_mark(by['needle_32k'], '.3f')} | {_mark(by['needle_64k'], '.3f')} "
            f"| {_mark(by['needle_100k'], '.3f')} | {_mark(by['edit'], '+.1f')} | {_mark(by['time'], '.2f')} "
            f"| {num(speed.get('scout_ms'), '.0f')} | {num(speed.get('main_ms'), '.0f')} | {kept} "
            f"| {num(speed.get('neighbour_gap_max_ms'), '.0f')} | {num(agent.get('tool_call_validity'), '.3f')} "
            f"| {num(agreement['first_divergence_median'], '.0f') if agreement else '-'} | {ev['verdict']} |")
    base_speed = _metric(rows, BASELINE, "speed") or {}
    lines += ["", "Baseline: " + (
        f"100K-class prefill {base_speed['total_s']:.1f} s, neighbour worst gap "
        f"{base_speed.get('neighbour_gap_max_ms') or 0:.0f} ms." if base_speed.get("total_s")
        else "no speed measurement; the time check cannot be evaluated.")]
    errors = [e for ev in evals for e in ev["errors"]] + \
             [r for r in rows if r["point"] == BASELINE and not r.get("ok")]
    if errors:
        lines += ["", "## Failures", ""]
        lines += [f"- `{e['point']}` / {e['workload']}: {e.get('error')}" for e in errors]
    notes = [f"- `{ev['point']}` {c['name']}: {c['detail']}" for ev in evals for c in ev["checks"]
             if c["pass"] is False]
    if notes:
        lines += ["", "## Failed checks", ""] + notes
    lines.append("")
    return "\n".join(lines)


# ---- running ---------------------------------------------------------------------------------

class Sweep:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = args.out
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.results = self.out / "results.jsonl"

    def record(self, point: str, params: dict[str, str] | None, workload: str, ok: bool,
               metrics: dict[str, Any] | None = None, error: str | None = None) -> None:
        row = {"point": point, "params": params or {}, "workload": workload, "ok": ok,
               "error": error, "metrics": metrics or {}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.results.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{point}] {workload}: {'ok' if ok else 'FAILED ' + str(error)}", file=sys.stderr, flush=True)

    def done(self) -> set[tuple[str, str]]:
        if not (self.args.resume and self.results.exists()):
            return set()
        return {(r["point"], r["workload"]) for r in load_rows([self.results]) if r.get("ok")}

    def start_server(self, point: str, params: dict[str, str] | None, extra: Sequence[str],
                     tag: str) -> tuple[subprocess.Popen, Path]:
        directory = self.out / point
        directory.mkdir(parents=True, exist_ok=True)
        request_log = directory / f"{tag}-request-log.jsonl"
        request_log.touch()
        cmd = server_command(self.args.server_cmd, self.args.port, request_log, params, extra)
        (directory / f"{tag}-server-cmd.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        log = open(directory / f"{tag}-server.log", "wb")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        deadline = time.monotonic() + self.args.start_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode} (see {directory / (tag + '-server.log')})")
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as response:
                    if response.status == 200:
                        return proc, request_log
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        stop_server(proc)
        raise RuntimeError(f"server not ready after {self.args.start_timeout:.0f} s")

    def run_point(self, point: str, params: dict[str, str] | None, workloads: Sequence[str]) -> None:
        done = self.done()
        todo = [w for w in workloads if (point, w) not in done]
        quality = [w for w in todo if w != "speed"]
        groups = []
        if quality:
            groups.append(("quality", quality, []))
        if "speed" in todo:
            # The stall probe needs a lane for the stream next to the long prompt; the slices flag is
            # the scheduling setting the neighbour-gap number is defined for.
            groups.append(("speed", ["speed"], ["--max-concurrency", "2",
                                                "--prefill-layer-slices", str(SPEED_LAYER_SLICES)]))
        for tag, names, extra in groups:
            try:
                proc, request_log = self.start_server(point, params, extra, tag)
            except Exception as exc:  # noqa: BLE001 - a failed start is a result, not a crash
                self.record(point, params, "server", False, error=str(exc))
                continue
            try:
                for name in names:
                    self.run_workload(point, params, name, request_log)
            finally:
                stop_server(proc)

    def run_workload(self, point: str, params: dict[str, str] | None, name: str, request_log: Path) -> None:
        try:
            metrics = getattr(self, f"workload_{name}")(point, request_log)
        except Exception as exc:  # noqa: BLE001
            self.record(point, params, name, False, error=f"{type(exc).__name__}: {exc}")
        else:
            self.record(point, params, name, True, metrics)

    # -- workloads --

    def workload_needle(self, point: str, request_log: Path) -> dict[str, Any]:
        args = self.args
        directory = self.out / point
        runs = directory / "needle-runs"
        config = directory / "needle-config.yaml"
        config.write_text(json.dumps(needle_config(
            base_url=self.base_url, model=args.model, corpus=str(args.needle_corpus),
            tokenizer_path=str(args.tokenizer_path), runs_dir=str(runs), limit=args.needle_limit),
            indent=2), encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "eval"))
        proc = subprocess.run(needle_command(args.eval_python, config), cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True)
        (directory / "needle.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        summaries = sorted(runs.glob("*/summary.json"))
        if not summaries:
            raise RuntimeError(f"ninfer_eval exit {proc.returncode}; no summary (see {directory / 'needle.log'})")
        return {"accuracy": parse_needle_summary(json.loads(summaries[-1].read_text(encoding="utf-8"))),
                "run": str(summaries[-1].parent)}

    def workload_agent_replay(self, point: str, request_log: Path) -> dict[str, Any]:
        args = self.args
        out = self.out / point / "agent-replay"
        cmd = agent_replay_command(port=args.port, out=out, request_log=request_log,
                                   sessions=args.agent_sessions, seed=args.agent_seed)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        (self.out / point / "agent-replay.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        transcripts_path, summary_path = out / "transcripts.jsonl", out / "summary.json"
        if not transcripts_path.exists() or not summary_path.exists():
            raise RuntimeError(f"agent_replay exit {proc.returncode}; no output (see agent-replay.log)")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        commit = summary.get("corpus_commit") or "HEAD"
        repo = str(args.agent_repo)

        def read_file(path: str) -> str | None:
            shown = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
            return shown.stdout.decode("utf-8", "replace") if shown.returncode == 0 else None

        transcripts = [json.loads(line) for line in transcripts_path.read_text(encoding="utf-8").splitlines() if line]
        metrics = edit_metrics(transcripts, read_file)
        metrics.update(complete=summary.get("complete"), replay_exit=proc.returncode,
                       turns=(summary.get("all") or {}).get("turns"))
        return metrics

    def workload_greedy_agreement(self, point: str, request_log: Path) -> dict[str, Any]:
        outputs = []
        for index in range(self.args.greedy_prompts):
            prompt = greedy_prompt(self.args.agent_repo, index, self.args.greedy_chars)
            outputs.append(greedy_generate(self.base_url, self.args.model, prompt,
                                           self.args.greedy_max_tokens, request_log))
        return {"outputs": outputs, "prompts": len(outputs)}

    def workload_speed(self, point: str, request_log: Path) -> dict[str, Any]:
        out = self.out / point / "stall.json"
        offset = request_log.stat().st_size
        cmd = stall_command(base_url=self.base_url, model=self.args.model, output=out,
                            long_tokens=self.args.speed_tokens, words_per_token=self.args.words_per_token)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        (self.out / point / "stall.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if not out.exists():
            raise RuntimeError(f"run_prefill_stall exit {proc.returncode}: {proc.stderr[-300:]}")
        report = json.loads(out.read_text(encoding="utf-8"))
        lossy = long_request_lossy_record(request_log, offset)
        return speed_metrics(report, lossy)


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def greedy_prompt(repo: Path, index: int, chars: int) -> str:
    """A long, deterministic prompt per index: tracked source files in a seeded order, then a question."""
    listing = subprocess.run(["git", "-C", str(repo), "ls-files", "*.py", "*.cpp", "*.h", "*.md", "*.cu"],
                             capture_output=True, text=True, check=True).stdout.split()
    random.Random(20261025 + index).shuffle(listing)
    parts: list[str] = []
    size = 0
    for name in listing:
        try:
            text = (repo / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        parts.append(f"=== {name} ===\n{text}\n")
        size += len(parts[-1])
        if size >= chars:
            break
    return "".join(parts) + "\nList the first three file names above, in order, and say what each is for."


def greedy_generate(base_url: str, model: str, prompt: str, max_tokens: int,
                    request_log: Path | None) -> list[Any]:
    """Greedy completion; returns committed token ids from the request log when it has them, else the
    characters of the text (the divergence index is then a character index)."""
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                       "messages": [{"role": "user", "content": prompt}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.load(response)
    text = reply["choices"][0]["message"].get("content") or ""
    if request_log is not None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for line in reversed(request_log.read_text(encoding="utf-8").splitlines()[-50:]):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if (record.get("request") or {}).get("response_id") == reply.get("id"):
                    ids = (record.get("result") or {}).get("generated_token_ids")
                    if isinstance(ids, list):
                        return ids
            time.sleep(0.5)
    return list(text)


# ---- CLI -------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--server-cmd", help="ninfer-serve command line (artifact, --kv-dtype rk4v4, "
                        "--max-context ...); the harness adds --port, --request-log-jsonl and the lossy flags")
    parser.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2",
                        help=f"keys: {', '.join(GRID_KEYS)}; omitted knobs keep the server default; "
                        "strategy=select,summary (summary ignores the knobs and runs once)")
    parser.add_argument("--workloads", default=",".join(WORKLOADS), help="comma list of "
                        + ", ".join(WORKLOADS))
    parser.add_argument("--only", default=None, help="regex: run only the grid points whose id matches")
    parser.add_argument("--out", type=Path, default=Path("profiles/lossy-sweep"))
    parser.add_argument("--from-json", nargs="+", type=Path, default=None, metavar="RESULTS.jsonl",
                        help="do not run anything: summarise these results files")
    parser.add_argument("--baseline-from", nargs="+", type=Path, default=[], metavar="RESULTS.jsonl",
                        help="results files whose baseline rows join this run's summary")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip (point, workload) pairs already ok in --out/results.jsonl")
    parser.add_argument("--port", type=int, default=18130)
    parser.add_argument("--model", default=None, help="served model id (needle config and requests)")
    parser.add_argument("--start-timeout", type=float, default=1200.0)
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="HF tokenizer dir (needle)")
    parser.add_argument("--needle-corpus", type=Path, default=None,
                        help="Needle-in-a-Haystack corpus dir (evalscope dataset_args.local_path)")
    parser.add_argument("--needle-limit", type=int, default=None, help="cap samples per length (smoke)")
    parser.add_argument("--eval-python", default=str(REPO_ROOT / "eval/.venv/bin/python"))
    parser.add_argument("--agent-sessions", type=int, default=4)
    parser.add_argument("--agent-seed", type=int, default=None)
    parser.add_argument("--agent-repo", type=Path, default=REPO_ROOT,
                        help="git repository the replay reads (and the greedy prompts are cut from)")
    parser.add_argument("--greedy-prompts", type=int, default=GREEDY_PROMPTS)
    parser.add_argument("--greedy-chars", type=int, default=250_000, help="prompt size, characters (~60K tokens)")
    parser.add_argument("--greedy-max-tokens", type=int, default=96)
    parser.add_argument("--speed-tokens", type=int, default=SPEED_LONG_TOKENS)
    parser.add_argument("--words-per-token", type=float, default=None)
    args = parser.parse_args(argv)
    names = [w for w in args.workloads.split(",") if w]
    if any(w not in WORKLOADS for w in names):
        parser.error(f"--workloads: choose from {', '.join(WORKLOADS)}")
    args.workload_names = names
    if args.from_json is None:
        if not args.server_cmd or not args.model:
            parser.error("--server-cmd and --model are required unless --from-json")
        if "needle" in names and not (args.needle_corpus and args.tokenizer_path):
            parser.error("the needle workload needs --needle-corpus and --tokenizer-path")
        try:
            args.grid_spec = parse_grid(args.grid)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.from_json is not None:
        rows = load_rows(args.from_json)
        text = render_summary(rows)
        print(text)
        if args.out and args.out.exists():
            (args.out / "summary.md").write_text(text, encoding="utf-8")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    sweep = Sweep(args)
    points = expand_grid(args.grid_spec) if args.grid else []
    if args.only:
        points = [p for p in points if re.search(args.only, p["id"])]
    if not args.skip_baseline:
        sweep.run_point(BASELINE, None, args.workload_names)
    for point in points:
        sweep.run_point(point["id"], point["params"], args.workload_names)
    rows = load_rows([sweep.results] + list(args.baseline_from)) if sweep.results.exists() else []
    text = render_summary(rows)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
