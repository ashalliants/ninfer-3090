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
and the sweep goes on. Before its workloads every lossy server gets a preflight: one long cold
conversation whose request_done must log lossy_prefill.outcome "completed"; otherwise the point is
recorded as failed (workload "preflight", with what the record showed) and skipped, so a trigger
that cannot fire never passes as a measurement of the lossy path. The baseline runs first unless `--skip-baseline`; then give the earlier run's
results to `--baseline-from`.

Acceptance (fixed, docs and the task's Global Constraints; not to be changed after seeing results):
needle accuracy at 32K/64K/100K >= 95% of baseline; agent-replay Edit exact-match (old_string in the
file) no more than 2 points under baseline; 100K cold prefill <= 40% of the baseline's time.
Greedy agreement (first divergence from the baseline's greedy output) replaces the spec's
teacher-forced KL in v1 and has no pass/fail threshold; it is reported only.

Int4 scout gate (`--workloads jaccard`, opt-in, not in the default list): a `scout_a4=1` point and
its partner (the same knobs with scout_a4 off) each send the same long prompts; the kept-block
Jaccard between the two runs' `lossy_prefill.kept_runs` (logged under --request-log-lossy-detail,
which the harness adds for this workload), over the score-chosen blocks only (the forced head and
recent blocks left out; ruling R11), must be >= 0.9 on every prompt. The all-blocks Jaccard is shown
for reference. The rest of that gate --
quality unchanged within noise, scout GEMMs >= 1.4x -- is the a4 point's own quality rows against
its partner's, and the GPU test ninfer_q4_a4_linear_test's PERF lines.
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
# Samples per length: NEEDLE_DEPTHS document depths (0, 10, ..., 100 percent) x NEEDLE_SUBSETS
# (evalscope's needle_haystack makes one sample per subset, length and depth) = 22. Against a perfect
# baseline the 0.95x gate then allows one miss (21/22 = 0.955) and fails two (20/22 = 0.909); with
# the earlier 5 samples one miss was already 0.80x, so the gate could not tell 95% from 80%.
NEEDLE_DEPTHS = 11
NEEDLE_SUBSETS = ("english", "chinese")
# A needle check with fewer samples on either side is reported as not measured (a --needle-limit
# smoke run, a truncated job), never as pass or fail.
NEEDLE_MIN_SAMPLES = 20
SPEED_LONG_TOKENS = 108_000
SPEED_LAYER_SLICES = 8
GREEDY_PROMPTS = 20
# Preflight (every lossy server, before its workloads): one long cold conversation that must log
# lossy_prefill.outcome == "completed", else the point is aborted. ~60K tokens at 3.3 chars/token, in
# PREFLIGHT_TURNS user turns so the summary strategy has history to summarise too.
PREFLIGHT_CHARS = 200_000
PREFLIGHT_TURNS = 30

WORKLOADS = ("needle", "agent_replay", "agent_edit", "greedy_agreement", "speed")
# agent_edit: the baseline agent_replay's edit_function turns whose prompt has at least this many
# tokens, each sent cold on its own (a server with --no-prefix-reuse), so lossy prefill filters
# every one of them. agent_replay itself grows each session from a few thousand tokens and every
# later turn reuses the conversation's private record, which lossy prefill never filters, so its
# Edit rate does not exercise the feature.
AGENT_EDIT_MIN_PROMPT = 34_000
# Run only when named in --workloads.
OPTIONAL_WORKLOADS = ("jaccard",)
ALL_WORKLOADS = WORKLOADS + OPTIONAL_WORKLOADS
BASELINE = "baseline"

# The int4 scout's ranking gate (fixed): kept-block Jaccard against the int8 scout, every prompt.
JACCARD_MIN = 0.9
# Jaccard prompts: cut from the repository like the greedy ones, ~32K and ~100K tokens at 3.3
# chars/token, JACCARD_PROMPTS of each size.
JACCARD_CHARS = (105_000, 330_000)
JACCARD_PROMPTS = 3
# The server's selection defaults, for points that leave the knob out.
SELECTION_DEFAULTS = {"block": 64, "head": 256, "recent": 4096}

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


# ---- kept-block overlap (the int4 scout's gate) -----------------------------------------------

def kept_blocks(runs: Sequence[Sequence[int]], uncached: int, block: int, scored_only: bool = False,
                head: int = 0, recent: int = 0) -> set[int]:
    """Blocks of `block` tokens (the last one clipped to the span of `uncached` tokens) that the
    [begin, end) runs keep whole. `scored_only` leaves out the blocks the head and recent rules
    force (any block starting before `head` or ending in the last `recent` tokens), so only the
    blocks the scores chose are compared."""
    out: set[int] = set()
    for begin, end in runs:
        b = (begin + block - 1) // block
        while b * block < end:
            lo, hi = b * block, min(b * block + block, uncached)
            if hi > end:
                break
            if not (scored_only and (lo < head or hi + recent > uncached)):
                out.add(b)
            b += 1
    return out


def jaccard(a: set[int], b: set[int]) -> float | None:
    """None when both sets are empty: there is nothing to compare, so no overlap was measured."""
    either = len(a | b)
    return None if either == 0 else len(a & b) / either


def point_params(rows: list[dict[str, Any]], point: str) -> dict[str, str] | None:
    for row in rows:
        if row["point"] == point:
            return row.get("params") or {}
    return None


def scout_a4_partner(rows: list[dict[str, Any]], point: str) -> str | None:
    """The int8 point a `scout_a4=1` point is compared with: the same params with scout_a4 off
    (absent or 0). None for any other point, or when no such point has rows."""
    params = point_params(rows, point)
    if not params or params.get("scout_a4") != "1":
        return None
    want = {k: v for k, v in params.items() if k != "scout_a4"}
    for row in rows:
        other = row.get("params") or {}
        if row["point"] != point and other.get("scout_a4", "0") == "0" and \
                {k: v for k, v in other.items() if k != "scout_a4"} == want:
            return row["point"]
    return None


def jaccard_metrics(rows: list[dict[str, Any]], point: str) -> dict[str, Any] | None:
    """Per-prompt kept-block Jaccard of a scout_a4=1 point against its partner, from both points'
    `jaccard` rows. None unless both rows exist. A prompt missing from either side, or one that
    did not complete lossy prefill on both, is listed in `unmatched` and fails the gate."""
    partner = scout_a4_partner(rows, point)
    mine = _metric(rows, point, "jaccard")
    theirs = _metric(rows, partner, "jaccard") if partner else None
    if mine is None or theirs is None:
        return None
    params = point_params(rows, point) or {}
    block = int(params.get("block", SELECTION_DEFAULTS["block"]))
    head = int(params.get("head", SELECTION_DEFAULTS["head"]))
    recent = int(params.get("recent", SELECTION_DEFAULTS["recent"]))
    by_prompt = {r["prompt"]: r for r in theirs.get("kept") or []}
    values, scored, unmatched = [], [], []
    for record in mine.get("kept") or []:
        other = by_prompt.get(record["prompt"])
        if other is None or record.get("outcome") != "completed" or other.get("outcome") != "completed" \
                or record.get("uncached_tokens") != other.get("uncached_tokens"):
            unmatched.append(record["prompt"])
            continue
        n = record["uncached_tokens"]
        values.append(jaccard(kept_blocks(record["kept_runs"], n, block),
                              kept_blocks(other["kept_runs"], n, block)))
        scored.append(jaccard(kept_blocks(record["kept_runs"], n, block, True, head, recent),
                              kept_blocks(other["kept_runs"], n, block, True, head, recent)))
    unmatched += [p for p in by_prompt if p not in {r["prompt"] for r in mine.get("kept") or []}]
    # A prompt whose two runs hold no score-chosen block (a span within head + recent) measures
    # nothing; it is listed, and the gate is not measured rather than passed.
    unmeasured = [r["prompt"] for r, v in zip([r for r in mine.get("kept") or []
                                               if r["prompt"] not in unmatched], scored) if v is None]
    some, chosen = [v for v in values if v is not None], [v for v in scored if v is not None]
    return {"partner": partner, "prompts": len(values), "jaccard": values,
            "jaccard_min": min(some) if some else None,
            "jaccard_mean": statistics.fmean(some) if some else None,
            "scored": scored, "scored_min": min(chosen) if chosen else None,
            "scored_mean": statistics.fmean(chosen) if chosen else None,
            "unmatched": unmatched, "unmeasured": unmeasured}


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
    """An eval/ config (JSON, which is YAML) after eval/configs/qwen3_6_35b_needle_haystack.yaml
    (native_long): one evalscope needle_haystack job per context length, `depths` document depths
    0-100% in each of NEEDLE_SUBSETS, so len(NEEDLE_SUBSETS) x depths samples per length."""
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
                "subset_list": list(NEEDLE_SUBSETS),
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
                         min_context: int = 30000, max_context: int = 120000,
                         repo: Path | None = None, commit: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "tools.bench.agent_replay", "run", "--port", str(port),
           "--sessions", str(sessions), "--concurrency", "1",
           "--min-context", str(min_context), "--max-context", str(max_context),
           "--request-log", str(request_log), "--out", str(out)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if repo is not None:
        cmd += ["--repo", str(repo)]
    if commit is not None:
        cmd += ["--commit", commit]
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


def parse_needle_samples(summary: dict[str, Any]) -> dict[str, int]:
    """Completed sample count per needle job, from the same ninfer_eval summary.json."""
    return {str(r["job_id"]).removeprefix("needle_"): int((r.get("counts") or {}).get("completed") or 0)
            for r in summary.get("results", []) if str(r.get("job_id", "")).startswith("needle_")}


_ARTICLES = {"a", "an", "the"}


def _answer_tokens(text: str) -> list[str]:
    """Lower-case word tokens, CJK characters one by one, articles and punctuation dropped."""
    return [t for t in re.findall(r"[\u3400-\u9fff]|[a-z0-9']+", text.lower()) if t not in _ARTICLES]


def needle_answer_hit(prediction: str, reference: str, question: str) -> bool:
    """The needle judge: every word of the needle that the question does not already contain (the
    answer words) occurs in the prediction. evalscope's `rule` judge is an exact match of the whole
    normalised needle sentence, so a correct short answer ("Eat a sandwich and sit in Dolores Park
    on a sunny day." for "The best thing to do in San Francisco is eat a sandwich ...") scores 0;
    the system prompt asks for short answers, so it scored 1/22 on a baseline that answered 22/22."""
    asked = set(_answer_tokens(question))
    answer = [t for t in _answer_tokens(reference) if t not in asked] or _answer_tokens(reference)
    said = set(_answer_tokens(prediction))
    return bool(answer) and all(t in said for t in answer)


def rescore_needle_run(run: Path) -> dict[str, Any] | None:
    """Re-judge a ninfer_eval needle run from evalscope's per-sample reviews with needle_answer_hit:
    {"accuracy": {length: share}, "samples": {length: n}, "rule_accuracy": {...}}. None when a length
    has no reviews (the caller keeps evalscope's own numbers)."""
    accuracy: dict[str, float] = {}
    samples: dict[str, int] = {}
    rule: dict[str, float] = {}
    for length in NEEDLE_LENGTHS:
        files = sorted((run / "backends" / f"needle_{length}" / "reviews").glob("*/*.jsonl"))
        hits = rule_hits = n = 0
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                review = json.loads(line)
                score = (review.get("sample_score") or {}).get("score") or {}
                prediction = str(score.get("prediction") or "")
                users = [m for m in review.get("messages") or [] if m.get("role") == "user"]
                content = str(users[-1].get("content") or "") if users else ""
                match = re.search(r"<question>(.*?)</question>", content, re.S)
                question = match.group(1) if match else content[-500:]
                n += 1
                hits += needle_answer_hit(prediction, str(review.get("target") or ""), question)
                rule_hits += any(bool(v) for v in (score.get("value") or {}).values())
        if n == 0:
            return None
        accuracy[length], samples[length], rule[length] = hits / n, n, rule_hits / n
    return {"accuracy": accuracy, "samples": samples, "rule_accuracy": rule, "judge": "answer_words"}


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
                if isinstance(args, dict) and ok and name == "Edit" and turn.get("category") == "edit_function":
                    file_path, old = args.get("file_path"), args.get("old_string")
                    if isinstance(file_path, str) and isinstance(old, str) and old:
                        text = read_file(file_path.removeprefix("/repo/"))
                        if text is not None and old in text:
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


def check_speed_report(stall_report: dict[str, Any], max_context: int, min_tokens: int = 100_000) -> None:
    """Refuse a stall report that does not measure a 100K cold prefill: the long request must have
    succeeded (HTTP 200), been uncached, and be between `min_tokens` and the server's max context."""
    long = stall_report.get("long") or {}
    if long.get("status") != 200:
        raise RuntimeError(f"long request failed: status {long.get('status')}, error {long.get('error')}")
    if long.get("cached_tokens"):
        raise RuntimeError(f"long prompt was not cold: cached_tokens {long.get('cached_tokens')}")
    tokens = long.get("prompt_tokens")
    if not isinstance(tokens, int) or not min_tokens <= tokens <= max_context:
        raise RuntimeError(f"long prompt has {tokens} tokens, outside {min_tokens}..{max_context}; "
                           "tune --words-per-token / --speed-tokens")


def lossy_outcome_counts(request_log: Path, offset: int = 0) -> dict[str, int]:
    """Count request_done records after `offset` by `lossy_prefill.outcome`; `requests` counts all of
    them, `none` those that never took the path (no lossy_prefill key). The summary strategy's
    coldness probe answers are counted as `summary_probe_cold|warm|timeout`: a timeout left a cold
    prompt whole because the busy worker did not answer, which explains a summary point with few
    completions."""
    counts: dict[str, int] = {"requests": 0, "none": 0}
    if not request_log.exists():
        return counts
    with request_log.open("rb") as log:
        log.seek(offset)
        for raw in log:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if record.get("event") != "request_done":
                continue
            counts["requests"] += 1
            lossy = record.get("lossy_prefill")
            if isinstance(lossy, dict):
                outcome = str(lossy.get("outcome", "unknown"))
                counts[outcome] = counts.get(outcome, 0) + 1
                probe = lossy.get("summary_probe")
                if probe:
                    key = f"summary_probe_{probe}"
                    counts[key] = counts.get(key, 0) + 1
            else:
                counts["none"] += 1
    return counts


def speed_metrics(stall_report: dict[str, Any], lossy_record: dict[str, Any] | None) -> dict[str, Any]:
    """From run_prefill_stall's report and, for a lossy run, the long request's `lossy_prefill`
    request-log object (scout_ms, main_ms, kept/uncached tokens, outcome)."""
    long = stall_report.get("long") or {}
    during = (stall_report.get("stream") or {}).get("during_long_prompt") or {}
    out: dict[str, Any] = {
        "prompt_tokens": long.get("prompt_tokens"),
        "cached_tokens": long.get("cached_tokens"),
        "total_s": long.get("seconds"),
        "neighbour_gap_max_ms": during.get("gap_max_ms"),
        "neighbour_gap_p99_ms": during.get("gap_p99_ms"),
    }
    if lossy_record:
        for key in ("scout_ms", "main_ms", "kept_tokens", "uncached_tokens", "filter_layer", "outcome"):
            out[key] = lossy_record.get(key)
    return out


def jaccard_record(prompt: int, lossy: dict[str, Any] | None) -> dict[str, Any]:
    """One jaccard-workload prompt's kept set from its request_done `lossy_prefill` object, which
    must carry kept_runs (the server needs --request-log-lossy-detail)."""
    if not isinstance(lossy, dict):
        return {"prompt": prompt, "outcome": None, "uncached_tokens": None, "kept_runs": []}
    if "kept_runs" not in lossy:
        raise RuntimeError("lossy_prefill has no kept_runs: the server needs --request-log-lossy-detail")
    return {"prompt": prompt, "outcome": lossy.get("outcome"),
            "uncached_tokens": lossy.get("uncached_tokens"), "kept_tokens": lossy.get("kept_tokens"),
            "scout_ms": lossy.get("scout_ms"), "kept_runs": lossy["kept_runs"]}


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


def preflight_messages(text: str, nonce: str, turns: int = PREFLIGHT_TURNS) -> list[dict[str, str]]:
    """A conversation of `turns` user turns over `text` (each answered "Noted."), then a question.
    `nonce` leads the first turn so no earlier run's cached prefix can make it warm."""
    size = max(1, len(text) // turns)
    messages: list[dict[str, str]] = []
    for index in range(turns):
        chunk = text[index * size:(index + 1) * size if index + 1 < turns else len(text)]
        lead = f"[preflight {nonce}]\n" if index == 0 else ""
        messages.append({"role": "user", "content": f"{lead}Part {index + 1}:\n{chunk}"})
        messages.append({"role": "assistant", "content": "Noted."})
    messages.append({"role": "user", "content": "Name the first file shown above."})
    return messages


def check_preflight(record: dict[str, Any] | None) -> dict[str, Any]:
    """The preflight request's `lossy_prefill` object, which must report outcome "completed": a lossy
    server whose long cold request did not take the path would measure the baseline under a lossy
    label. Raises with what the record showed instead."""
    lossy = (record or {}).get("lossy_prefill")
    if not isinstance(lossy, dict):
        raise RuntimeError("preflight: the long cold request did not take the lossy path (no lossy_prefill "
                           "in its request_done record); the trigger cannot fire for this point")
    if lossy.get("outcome") != "completed":
        raise RuntimeError(f"preflight: lossy_prefill.outcome is {lossy.get('outcome')!r}, not 'completed' "
                           f"(uncached {lossy.get('uncached_tokens')}, kept {lossy.get('kept_tokens')}"
                           + (f", summary_probe {lossy.get('summary_probe')}" if "summary_probe" in lossy else "")
                           + ")")
    return lossy


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


def unresolved_failures(rows: list[dict[str, Any]], point: str) -> list[dict[str, Any]]:
    """Failed rows of a point that no later ok row supersedes. A later ok row for the same workload
    repairs it; a failed server start is repaired by any later ok row of the point."""
    failed = []
    for i, row in enumerate(rows):
        if row["point"] != point or row.get("ok"):
            continue
        later = [r for r in rows[i + 1:] if r["point"] == point and r.get("ok")]
        if row["workload"] == "server":
            repaired = bool(later)
        else:
            repaired = any(r["workload"] == row["workload"] for r in later)
        if not repaired:
            failed.append(row)
    return failed


def edit_rows(rows: list[dict[str, Any]], point: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """The Edit gate's rows: agent_edit (cold edit turns) of the point and of the baseline. No
    fallback to agent_replay (its Edit turns are warm, so the lossy path barely runs on them): when
    either lacks agent_edit the gate is not measured and both are None."""
    agent, base = _metric(rows, point, "agent_edit"), _metric(rows, BASELINE, "agent_edit")
    if agent and base:
        return agent, base, "cold"
    return None, None, "cold"


def paired_edit_turns(rows: list[dict[str, Any]], point: str) -> dict[str, int] | None:
    """Per-turn Edit comparison of a point against the baseline over the agent_edit turns both ran
    (matched by session and turn id): turns the baseline hit and the point missed, and the reverse."""
    agent, base = _metric(rows, point, "agent_edit"), _metric(rows, BASELINE, "agent_edit")
    if not agent or not base or not agent.get("turns") or not base.get("turns"):
        return None
    mine = {(t.get("session"), t.get("turn")): bool(t.get("hit")) for t in agent["turns"]}
    ref = {(t.get("session"), t.get("turn")): bool(t.get("hit")) for t in base["turns"]}
    common = mine.keys() & ref.keys()
    return {"paired": len(common),
            "lost": sum(ref[k] and not mine[k] for k in common),
            "gained": sum(mine[k] and not ref[k] for k in common)}


def agent_edit_turns(transcripts: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]],
                     min_prompt: int = AGENT_EDIT_MIN_PROMPT) -> list[dict[str, Any]]:
    """The agent_edit workload's requests from an agent_replay run: every edit_function turn whose
    prompt (agent_replay's turns.jsonl `prompt_tokens`, by response id) has at least `min_prompt`
    tokens, as {session, turn, messages (the prompt), tools, max_tokens}."""
    by_id = {s.get("response_id"): s for s in samples if s.get("response_id")}
    out = []
    for session in transcripts:
        for turn in session.get("turns", []):
            sample = by_id.get(turn.get("response_id")) or {}
            if turn.get("category") != "edit_function" or int(sample.get("prompt_tokens") or 0) < min_prompt:
                continue
            out.append({"session": session.get("session"), "turn": turn.get("turn"),
                        "messages": session["messages"][:turn["message_count"]], "tools": session.get("tools", []),
                        "max_tokens": int(sample.get("max_tokens") or 6000),
                        "baseline_prompt_tokens": int(sample.get("prompt_tokens") or 0)})
    return out


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
        samples = ((needle or {}).get("samples") or {}).get(length)
        base_samples = ((base_needle or {}).get("samples") or {}).get(length)
        if not samples or not base_samples or min(samples, base_samples) < NEEDLE_MIN_SAMPLES:
            checks.append(check(f"needle_{length}", None, None,
                                f"{samples or 0} / baseline {base_samples or 0} samples, need >= "
                                f"{NEEDLE_MIN_SAMPLES} each for the {NEEDLE_MIN_RATIO:.2f}x gate"))
            continue
        ratio = value / base
        checks.append(check(f"needle_{length}", ratio, ratio >= NEEDLE_MIN_RATIO - 1e-9,
                            f"{value:.3f} ({samples} samples) / baseline {base:.3f} ({base_samples}) "
                            f"(need >= {NEEDLE_MIN_RATIO:.2f}x)"))
    agent, base_agent, source = edit_rows(rows, point)
    if agent and base_agent and agent.get("edit_rate") is not None and base_agent.get("edit_rate") is not None:
        delta = (agent["edit_rate"] - base_agent["edit_rate"]) * 100.0
        turns, lossy_turns = agent.get("edit_turns"), agent.get("lossy_turns")
        detail = (f"{delta:+.1f} points vs baseline over {turns} {source} Edit "
                  f"turns (need >= -{EDIT_MAX_DROP_POINTS:.0f})")
        pairs = paired_edit_turns(rows, point)
        if pairs:
            detail += (f"; of {pairs['paired']} paired turns {pairs['lost']} hit by baseline only, "
                       f"{pairs['gained']} by the point only")
        if point != BASELINE and (not turns or lossy_turns != turns):
            # Ruling R24: a score is only evidence if lossy prefill ran on the turns it scores.
            checks.append(check("edit", None, None, f"lossy ran on {lossy_turns or 0}/{turns or 0} turns; " + detail))
        else:
            checks.append(check("edit", delta, delta >= -EDIT_MAX_DROP_POINTS - 1e-9, detail))
    else:
        checks.append(check("edit", None, None, "not measured (needs agent_edit on this point and the baseline)"))
    speed, base_speed = _metric(rows, point, "speed"), _metric(rows, BASELINE, "speed")
    if speed and base_speed and speed.get("total_s") and base_speed.get("total_s"):
        ratio = speed["total_s"] / base_speed["total_s"]
        checks.append(check("time", ratio, ratio <= TIME_MAX_RATIO + 1e-9,
                            f"{speed['total_s']:.1f} s / baseline {base_speed['total_s']:.1f} s "
                            f"(need <= {TIME_MAX_RATIO:.2f}x)"))
    else:
        checks.append(check("time", None, None, "not measured"))
    if (point_params(rows, point) or {}).get("scout_a4") == "1":
        overlap = jaccard_metrics(rows, point)
        if overlap is None or not overlap["prompts"]:
            checks.append(check("jaccard", None, None, "not measured (needs the jaccard workload on "
                                "this point and on its scout_a4=0 partner)"))
        else:
            # Ruling R11: the gate is over the score-chosen blocks only; the forced head and recent
            # blocks are kept identically by both scouts and would inflate the index.
            low = overlap["scored_min"] is not None and overlap["scored_min"] < JACCARD_MIN - 1e-9
            if overlap["unmatched"] or low:
                ok: bool | None = False
            elif overlap["unmeasured"]:
                ok = None  # some prompt had no score-chosen block on either side
            else:
                ok = True

            def fmt(v: float | None) -> str:
                return "-" if v is None else f"{v:.3f}"
            checks.append(check("jaccard", overlap["scored_min"], ok,
                                f"score-chosen blocks min {fmt(overlap['scored_min'])} over {overlap['prompts']} "
                                f"prompt(s) vs `{overlap['partner']}` (need >= {JACCARD_MIN:.2f}); all kept "
                                f"blocks {fmt(overlap['jaccard_min'])} (reference)"
                                + (f"; unmatched prompts {overlap['unmatched']}" if overlap["unmatched"] else "")
                                + (f"; no score-chosen block, not measured: prompts {overlap['unmeasured']}"
                                   if overlap["unmeasured"] else "")))
    if point != BASELINE:
        for workload in ALL_WORKLOADS:
            metrics = _metric(rows, point, workload)
            if metrics is None:
                continue
            outcomes = metrics.get("lossy_outcomes")
            completed = (outcomes or {}).get("completed", 0)
            timeouts = (outcomes or {}).get("summary_probe_timeout", 0)
            probe_note = (f"; {timeouts} summary probe timeout(s) left cold prompts whole"
                          if timeouts else "")
            checks.append(check(f"lossy_ran_{workload}", completed if outcomes is not None else None,
                                True if completed > 0 else None,
                                (f"{completed} request(s) completed lossy prefill" if completed > 0 else
                                 "no request completed lossy prefill (or no outcome counts): the gate "
                                 "did not exercise the feature") + probe_note))
    failed = unresolved_failures(rows, point)
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


def lossy_cell(rows: list[dict[str, Any]], point: str) -> str:
    cells = []
    for workload in WORKLOADS:
        outcomes = (_metric(rows, point, workload) or {}).get("lossy_outcomes")
        if outcomes is None:
            cells.append("-")
            continue
        timeouts = outcomes.get("summary_probe_timeout", 0)
        cells.append(str(outcomes.get("completed", 0)) + (f" (t{timeouts})" if timeouts else ""))
    return "/".join(cells)


def needle_samples(rows: list[dict[str, Any]], point: str) -> str:
    counts = (_metric(rows, point, "needle") or {}).get("samples") or {}
    return ", ".join(f"{k}:{v}" for k, v in counts.items()) or "-"


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Lossy prefill sweep",
        "",
        "Acceptance (fixed): needle >= 0.95x baseline at 32K/64K/100K; Edit within 2 points of "
        "baseline; 100K cold prefill <= 0.40x baseline time. Greedy agreement and the speed "
        "columns are reported, not gated.",
        "",
        "| point | needle 32k | needle 64k | needle 100k | Edit (pts) | Edit turns | Edit lost/gained vs baseline | replay Edit rate (ungated) | time (x) | scout ms | main ms "
        "| kept/uncached | neighbour gap ms | tool-call valid | greedy median div | lossy completed (needle/agent/edit/greedy/speed) | needle samples | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    evals = []
    for point in points_in(rows):
        ev = evaluate_point(rows, point)
        evals.append(ev)
        by = {c["name"]: c for c in ev["checks"]}
        speed = _metric(rows, point, "speed") or {}
        agent = _metric(rows, point, "agent_edit") or {}
        replay = _metric(rows, point, "agent_replay") or {}
        pairs = paired_edit_turns(rows, point)
        pair_cell = f"-{pairs['lost']}/+{pairs['gained']} of {pairs['paired']}" if pairs else "-"
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
            f"| {_mark(by['needle_100k'], '.3f')} | {_mark(by['edit'], '+.1f')} | {num(agent.get('edit_turns'), 'd')} | {pair_cell} | {num(replay.get('edit_rate'), '.3f')} | {_mark(by['time'], '.2f')} "
            f"| {num(speed.get('scout_ms'), '.0f')} | {num(speed.get('main_ms'), '.0f')} | {kept} "
            f"| {num(speed.get('neighbour_gap_max_ms'), '.0f')} | {num(replay.get('tool_call_validity') if replay.get('tool_call_validity') is not None else agent.get('tool_call_validity'), '.3f')} "
            f"| {num(agreement['first_divergence_median'], '.0f') if agreement else '-'} "
            f"| {lossy_cell(rows, point)} | {needle_samples(rows, point)} | {ev['verdict']} |")
    a4_points = [p for p in points_in(rows) if (point_params(rows, p) or {}).get("scout_a4") == "1"]
    if a4_points:
        lines += ["", "## Int4 scout (scout_a4=1 against its int8 partner)", "",
                  f"Gate: kept-block Jaccard over the score-chosen blocks (forced head and recent blocks "
                  f"left out) >= {JACCARD_MIN} on every prompt; quality as the table above "
                  "against the partner's row; scout GEMMs >= 1.4x (ninfer_q4_a4_linear_test PERF lines).", "",
                  "| point | partner | prompts | Jaccard min, score-chosen (gated) | mean, score-chosen | all kept blocks min (reference) | scout ms int8 / int4 (speed) | gate |",
                  "|---|---|---|---|---|---|---|---|"]
        for point in a4_points:
            overlap = jaccard_metrics(rows, point)
            partner = scout_a4_partner(rows, point)
            mine = (_metric(rows, point, "speed") or {}).get("scout_ms")
            theirs = (_metric(rows, partner, "speed") or {}).get("scout_ms") if partner else None
            scout = f"{theirs:.0f} / {mine:.0f} ({theirs / mine:.2f}x)" if mine and theirs else "-"
            gate = {c["name"]: c for c in evaluate_point(rows, point)["checks"]}["jaccard"]
            if overlap and overlap["prompts"]:
                def cell(v: float | None) -> str:
                    return "-" if v is None else f"{v:.3f}"
                lines.append(f"| {point} | {partner} | {overlap['prompts']} | {cell(overlap['scored_min'])} "
                             f"| {cell(overlap['scored_mean'])} | {cell(overlap['jaccard_min'])} | {scout} "
                             f"| {_mark(gate, '.3f')} |")
            else:
                lines.append(f"| {point} | {partner or '-'} | - | - | - | - | {scout} | - |")
    base_speed = _metric(rows, BASELINE, "speed") or {}
    lines += ["", "Baseline: " + (
        f"100K-class prefill {base_speed['total_s']:.1f} s, neighbour worst gap "
        f"{base_speed.get('neighbour_gap_max_ms') or 0:.0f} ms." if base_speed.get("total_s")
        else "no speed measurement; the time check cannot be evaluated.")]
    errors = [e for ev in evals for e in ev["errors"]] + unresolved_failures(rows, BASELINE)
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
        if port_serving(self.base_url):
            raise RuntimeError(f"{self.base_url} is already serving; refusing to measure another server")
        with open(directory / f"{tag}-server.log", "wb") as log:
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
        if point == BASELINE:
            todo = [w for w in todo if w != "jaccard"]  # lossy prefill is off there: nothing kept
        quality = [w for w in todo if w not in ("speed", "agent_edit")]
        groups = []
        if quality:
            # jaccard reads lossy_prefill.kept_runs, which only the detail flag logs.
            groups.append(("quality", quality,
                           ["--request-log-lossy-detail"] if "jaccard" in quality else []))
        if "agent_edit" in todo:
            # Every edit turn cold: no request may reuse another's conversation record.
            groups.append(("edit", ["agent_edit"], ["--no-prefix-reuse"]))
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
                if params is not None:
                    try:
                        lossy = self.preflight(request_log)
                    except Exception as exc:  # noqa: BLE001 - abort this point, keep sweeping
                        self.record(point, params, "preflight", False, error=f"{tag} server: {exc}")
                        return
                    self.record(point, params, "preflight", True,
                                {"server": tag, **{key: lossy.get(key) for key in (
                                    "outcome", "strategy", "uncached_tokens", "kept_tokens", "scout_ms",
                                    "main_ms")}})
                for name in names:
                    self.run_workload(point, params, name, request_log)
            finally:
                stop_server(proc)

    def run_workload(self, point: str, params: dict[str, str] | None, name: str, request_log: Path) -> None:
        offset = request_log.stat().st_size
        try:
            metrics = getattr(self, f"workload_{name}")(point, request_log)
        except Exception as exc:  # noqa: BLE001
            self.record(point, params, name, False, error=f"{type(exc).__name__}: {exc}")
        else:
            # Evidence the feature ran: lossy_prefill outcomes of this workload's requests.
            metrics["lossy_outcomes"] = lossy_outcome_counts(request_log, offset)
            self.record(point, params, name, True, metrics)

    def preflight(self, request_log: Path) -> dict[str, Any]:
        text = greedy_prompt(self.args.agent_repo, 5000, self.args.preflight_chars)
        messages = preflight_messages(text, f"{time.time_ns():x}")
        record, _ = chat_request_done(self.base_url, self.args.model, None, 1, request_log,
                                      request_log.stat().st_size, messages=messages)
        return check_preflight(record)

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
        document = json.loads(summaries[-1].read_text(encoding="utf-8"))
        metrics = {"accuracy": parse_needle_summary(document), "samples": parse_needle_samples(document),
                   "run": str(summaries[-1].parent)}
        rescored = rescore_needle_run(summaries[-1].parent)
        if rescored is None:
            raise RuntimeError(f"needle run {summaries[-1].parent} has no per-sample reviews to judge")
        metrics.update(rescored)
        return metrics

    def rescore_needle(self) -> None:
        """Append a re-judged needle row (needle_answer_hit) for every point whose latest ok needle
        row was scored by evalscope's rule judge; the last ok row wins in the summary."""
        rows = load_rows([self.results]) if self.results.exists() else []
        for point in dict.fromkeys(r["point"] for r in rows):
            metrics = _metric(rows, point, "needle")
            if not metrics or metrics.get("judge") == "answer_words":
                continue
            run = Path(metrics.get("run") or "")
            try:
                found = run.is_dir()
            except OSError:  # a path recorded on another machine
                found = False
            if not found:
                run = self.out / point / "needle-runs" / run.name
            rescored = rescore_needle_run(run) if run.is_dir() else None
            if rescored is None:
                print(f"[{point}] needle: no reviews under {run}; not re-judged", file=sys.stderr)
                continue
            params = point_params(rows, point)
            self.record(point, params if params else None, "needle", True,
                        {**metrics, **rescored, "rejudged": True})

    def workload_agent_replay(self, point: str, request_log: Path) -> dict[str, Any]:
        args = self.args
        out = self.out / point / "agent-replay"
        cmd = agent_replay_command(port=args.port, out=out, request_log=request_log,
                                   sessions=args.agent_sessions, seed=args.agent_seed,
                                   repo=args.agent_repo, commit=args.agent_commit)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        (self.out / point / "agent-replay.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        transcripts_path, summary_path = out / "transcripts.jsonl", out / "summary.json"
        if not transcripts_path.exists() or not summary_path.exists():
            raise RuntimeError(f"agent_replay exit {proc.returncode}; no output (see agent-replay.log)")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if proc.returncode != 0 or summary.get("complete") is not True:
            raise RuntimeError(f"agent_replay incomplete: exit {proc.returncode}, complete={summary.get('complete')}, "
                               f"{len(summary.get('request_log_missing') or [])} turn(s) missing from the request log "
                               "(see agent-replay.log)")
        commit = summary.get("corpus_commit") or args.agent_commit
        repo = str(args.agent_repo)

        def read_file(path: str) -> str | None:
            shown = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
            return shown.stdout.decode("utf-8", "replace") if shown.returncode == 0 else None

        transcripts = [json.loads(line) for line in transcripts_path.read_text(encoding="utf-8").splitlines() if line]
        metrics = edit_metrics(transcripts, read_file)
        metrics.update(complete=summary.get("complete"), replay_exit=proc.returncode,
                       turns=(summary.get("all") or {}).get("turns"))
        return metrics

    def workload_agent_edit(self, point: str, request_log: Path) -> dict[str, Any]:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from tools.bench.agent_replay.run import send
        from tools.ninfer_serve.client import NInferServeClient

        args = self.args
        source = args.agent_source or (self.out / BASELINE / "agent-replay")
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        transcripts = [json.loads(l) for l in (source / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l]
        samples = [json.loads(l) for l in (source / "turns.jsonl").read_text(encoding="utf-8").splitlines() if l]
        requests = agent_edit_turns(transcripts, samples)
        if not requests:
            raise RuntimeError(f"no edit_function turn of >= {AGENT_EDIT_MIN_PROMPT} tokens in {source}")
        commit = summary.get("corpus_commit") or args.agent_commit
        repo = str(args.agent_repo)
        cache: dict[str, str | None] = {}

        def read_file(path: str) -> str | None:
            if path not in cache:
                shown = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
                cache[path] = shown.stdout.decode("utf-8", "replace") if shown.returncode == 0 else None
            return cache[path]

        client = NInferServeClient(self.base_url, 3600.0)
        cursor = request_log.stat().st_size
        per_turn = []
        for item in requests:
            reply = send(client, args.model, item["messages"], item["max_tokens"])
            if reply.error is not None:
                raise RuntimeError(f"session {item['session']} turn {item['turn']}: {reply.error}")
            record, cursor = find_request_done(request_log, cursor, reply.response_id)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}} for i, c in enumerate(reply.tool_calls)]}
            m = edit_metrics([{"tools": item["tools"], "messages": item["messages"] + [assistant],
                               "turns": [{"message_count": len(item["messages"]), "category": "edit_function"}]}],
                             read_file)
            lossy = record.get("lossy_prefill") or {}
            per_turn.append({"session": item["session"], "turn": item["turn"], "hit": m["edit_rate"] == 1.0,
                             "prompt_tokens": int(reply.usage.get("prompt_tokens") or 0),
                             "cached_tokens": int((reply.usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                             "outcome": lossy.get("outcome"), "kept_tokens": lossy.get("kept_tokens"),
                             "uncached_tokens": lossy.get("uncached_tokens"), "valid_calls": m["tool_call_validity"]})
            if not per_turn[-1]["hit"]:  # what the model did instead, for the report
                per_turn[-1]["reply"] = {"content": reply.content[:500], "tool_calls": [
                    {"name": c["name"], "arguments": c["arguments"][:1500]} for c in reply.tool_calls]}
        hits = sum(t["hit"] for t in per_turn)
        return {"edit_turns": len(per_turn), "edit_rate": hits / len(per_turn), "turns": per_turn,
                "lossy_turns": sum(t["outcome"] == "completed" for t in per_turn),
                "tool_call_validity": statistics.fmean(t["valid_calls"] for t in per_turn
                                                       if t["valid_calls"] is not None)
                if any(t["valid_calls"] is not None for t in per_turn) else None,
                "source": str(source), "min_prompt": AGENT_EDIT_MIN_PROMPT}

    def workload_greedy_agreement(self, point: str, request_log: Path) -> dict[str, Any]:
        outputs = []
        cursor = request_log.stat().st_size
        for index in range(self.args.greedy_prompts):
            prompt = greedy_prompt(self.args.agent_repo, index, self.args.greedy_chars)
            ids, cursor = greedy_generate(self.base_url, self.args.model, prompt,
                                          self.args.greedy_max_tokens, request_log, cursor)
            outputs.append(ids)
        return {"outputs": outputs, "prompts": len(outputs)}

    def workload_jaccard(self, point: str, request_log: Path) -> dict[str, Any]:
        """The same long prompts on every point (cut from --agent-repo), one output token each; the
        kept runs of each come from its request_done record."""
        kept = []
        cursor = request_log.stat().st_size
        index = 0
        for chars in JACCARD_CHARS:
            for _ in range(self.args.jaccard_prompts):
                prompt = greedy_prompt(self.args.agent_repo, 1000 + index, chars)
                record, cursor = chat_request_done(self.base_url, self.args.model, prompt, 1,
                                                   request_log, cursor)
                kept.append(jaccard_record(index, record.get("lossy_prefill")))
                index += 1
        return {"kept": kept, "prompts": len(kept)}

    def workload_speed(self, point: str, request_log: Path) -> dict[str, Any]:
        out = self.out / point / "stall.json"
        out.unlink(missing_ok=True)  # never read a report an earlier run left behind
        offset = request_log.stat().st_size
        cmd = stall_command(base_url=self.base_url, model=self.args.model, output=out,
                            long_tokens=self.args.speed_tokens, words_per_token=self.args.words_per_token)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        (self.out / point / "stall.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if not out.exists():
            raise RuntimeError(f"run_prefill_stall exit {proc.returncode}: {proc.stderr[-300:]}")
        report = json.loads(out.read_text(encoding="utf-8"))
        check_speed_report(report, self.args.max_context)
        lossy = long_request_lossy_record(request_log, offset)
        return speed_metrics(report, lossy)


def port_serving(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=3):
            return True
    except urllib.error.HTTPError:
        return True  # something answered
    except (urllib.error.URLError, OSError):
        return False


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def greedy_prompt(repo: Path, index: int, chars: int) -> str:
    """A long, deterministic prompt per index: tracked source files in a seeded order, cut to `chars`
    characters, then a question."""
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
    # Cut at `chars`: the last file can be large (one prompt reached 505K characters, 134K tokens,
    # past --max-context 131072 at 330K requested).
    return "".join(parts)[:chars] + "\nList the first three file names above, in order, and say what each is for."


def greedy_generate(base_url: str, model: str, prompt: str, max_tokens: int, request_log: Path,
                    cursor: int, wait_seconds: float = 30.0) -> tuple[list[int], int]:
    """Greedy completion; returns the committed token ids from the request log record of this
    response and the log offset to continue from. No fallback to text: a missing record fails."""
    record, cursor = chat_request_done(base_url, model, prompt, max_tokens, request_log, cursor,
                                       wait_seconds)
    ids = (record.get("result") or {}).get("generated_token_ids")
    if not isinstance(ids, list):
        raise RuntimeError(f"request log record for {(record.get('request') or {}).get('response_id')} "
                           "has no generated_token_ids")
    return ids, cursor


def find_request_done(request_log: Path, cursor: int, response_id: str | None,
                      wait_seconds: float = 30.0) -> tuple[dict[str, Any], int]:
    """The request_done record of `response_id` written after `cursor`, and the offset after it."""
    deadline = time.monotonic() + wait_seconds
    while True:
        with request_log.open("rb") as log:
            log.seek(cursor)
            while True:
                start = log.tell()
                raw = log.readline()
                if not raw.endswith(b"\n"):
                    log.seek(start)
                    break
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if record.get("event") == "request_done" and \
                        (record.get("request") or {}).get("response_id") == response_id:
                    return record, log.tell()
        if time.monotonic() > deadline:
            raise RuntimeError(f"no request-log record for response {response_id} after {wait_seconds:.0f} s")
        time.sleep(0.5)


def chat_request_done(base_url: str, model: str, prompt: str | None, max_tokens: int, request_log: Path,
                      cursor: int, wait_seconds: float = 30.0,
                      messages: list[dict[str, str]] | None = None) -> tuple[dict[str, Any], int]:
    """Greedy completion of `prompt` (one user message) or `messages`; returns this response's
    request_done record from the request log and the log offset to continue from. A missing record
    fails."""
    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                       "messages": messages,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.load(response)
    response_id = reply.get("id")
    deadline = time.monotonic() + wait_seconds
    while True:
        with request_log.open("rb") as log:
            log.seek(cursor)
            while True:
                start = log.tell()
                raw = log.readline()
                if not raw.endswith(b"\n"):  # nothing more, or a record still being written
                    log.seek(start)
                    break
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if record.get("event") == "request_done" and (record.get("request") or {}).get("response_id") == response_id:
                    return record, log.tell()
            cursor = log.tell()
        if time.monotonic() > deadline:
            raise RuntimeError(f"no request-log record for response {response_id} after {wait_seconds:.0f} s")
        time.sleep(0.5)


# ---- CLI -------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--server-cmd", help="ninfer-serve command line (artifact, --kv-dtype rk4v4, "
                        "--max-context ...); the harness adds --port, --request-log-jsonl and the lossy flags")
    parser.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2",
                        help=f"keys: {', '.join(GRID_KEYS)}; omitted knobs keep the server default; "
                        "strategy=select,summary (summary ignores the knobs and runs once)")
    parser.add_argument("--workloads", default=",".join(WORKLOADS), help="comma list of "
                        + ", ".join(ALL_WORKLOADS) + f" (default {','.join(WORKLOADS)}; jaccard is the int4 "
                        "scout's kept-block overlap: run it on a scout_a4=0,1 grid)")
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
    parser.add_argument("--agent-sessions", type=int, default=26,
                        help="agent_replay sessions; ~2 Edit turns each (see README), so 26 gives ~50 for the Edit gate")
    parser.add_argument("--agent-seed", type=int, default=None)
    parser.add_argument("--agent-commit", default="HEAD", help="corpus commit for the replay and its Edit check")
    parser.add_argument("--max-context", type=int, default=131072,
                        help="the server's --max-context (the speed prompt must fit below it)")
    parser.add_argument("--agent-repo", type=Path, default=REPO_ROOT,
                        help="git repository the replay reads (and the greedy prompts are cut from)")
    parser.add_argument("--greedy-prompts", type=int, default=GREEDY_PROMPTS)
    parser.add_argument("--greedy-chars", type=int, default=330_000, help="prompt size, characters (~100K tokens at 3.3 chars/token)")
    parser.add_argument("--greedy-max-tokens", type=int, default=96)
    parser.add_argument("--jaccard-prompts", type=int, default=JACCARD_PROMPTS,
                        help="jaccard workload prompts per size (~32K and ~100K tokens)")
    parser.add_argument("--preflight-chars", type=int, default=PREFLIGHT_CHARS,
                        help="size of each lossy server's preflight conversation, characters (~60K tokens); "
                        "it must exceed the point's min_tokens")
    parser.add_argument("--agent-source", type=Path, default=None,
                        help="agent_replay output directory whose edit turns agent_edit replays cold "
                        "(default OUT/baseline/agent-replay)")
    parser.add_argument("--rescore-needle", action="store_true",
                        help="re-judge recorded needle runs (needle_answer_hit) and re-summarise; no server")
    parser.add_argument("--speed-tokens", type=int, default=SPEED_LONG_TOKENS)
    parser.add_argument("--words-per-token", type=float, default=None)
    args = parser.parse_args(argv)
    names = [w for w in args.workloads.split(",") if w]
    if any(w not in ALL_WORKLOADS for w in names):
        parser.error(f"--workloads: choose from {', '.join(ALL_WORKLOADS)}")
    args.workload_names = names
    if args.from_json is None and not args.rescore_needle:
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
    if args.rescore_needle:
        sweep.rescore_needle()
        rows = load_rows([sweep.results]) if sweep.results.exists() else []
        text = render_summary(rows)
        (args.out / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
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
