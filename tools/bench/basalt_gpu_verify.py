#!/usr/bin/env python3
"""Verify every option of the Basalt-learnings branch (issue #383) on a machine with the RTX 3090.

The branch adds about 25 options, each compiled and CPU-tested but never run on a GPU. This runs, in
order, everything a GPU session needs to say whether each one works, keeps its promise and pays:

  1. the GPU tests of every touched area (tools/run-tests.ps1 -GpuOnly -Regex on Windows, one lock
     hold per test; ctest -L gpu on Linux), with the model each group needs;
  2. identity runs: the forward harness or engine gates that compare an option on against off;
  3. quality: the decode-path KLD gate (20 chunks, then 100) with each numerics option off and on,
     and the window check (--window 1 against 3);
  4. speed: an ABBA A/B per option, n = 12, one build with the option off and on
     (run_interleaved_ab.py for the Engine bench, run_chat_decode.py for ninfer-serve workloads);
  5. a combined row with every winning option on;
  6. the ledger's GPU, Windows and real-model checks (server smoke, soak, launchers, packs, ...);
  7. the whole GPU suite without the slow tests, after the targeted runs;
then writes summary.md, the text to paste into the pull request.

The options, their kinds, test regexes and workloads are in basalt_options.json beside this file.
Do not run this directly on the shared Windows box: basalt_gpu_verify.ps1 runs it step by step and
holds the GPU lock (tools/gpu-lock.ps1) around each step that uses the card, released between steps.
basalt_gpu_verify.sh is the Linux equivalent (a rented box is single-tenant, so it takes no lock).

  python tools/bench/basalt_gpu_verify.py --build-dir build-ninja --dry-run
  python tools/bench/basalt_gpu_verify.py --build-dir build-ninja --flash-next M --qwen27b M --action steps
  python tools/bench/basalt_gpu_verify.py ... --action run --step ab:--moe-fused-route
  python tools/bench/basalt_gpu_verify.py ... --action summary

--dry-run (the same as --action plan) prints every step with its commands and checks that every
test regex matches a name `ctest -N` lists in --build-dir; it exits 1 if one does not. A step whose
input is missing (no --qwen35b, no KLD base, no nsys, ...) is SKIPPED with the reason, never failed.
Results go to --out (default profiles/bench/basalt-<date>/): results/<step>.json, logs/, inputs/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = REPO / "tests" / "fixtures"
OPTIONS_FILE = HERE / "basalt_options.json"
SLOW_TESTS = "softmax_attention|kv_materialize"  # AGENTS.md: the very slow op tests
KINDS = ("identity", "quality", "speed", "serving")
TOOLS = ("engine-bench", "chat-decode")
# A speed variant "wins" when its paired median is at least +1% and three quarters of its pairs are
# positive. The combined row is a measurement built from these, not a decision.
WIN_MEDIAN, WIN_SHARE = 1.01, 0.75
CHAT_PROMPTS = 8  # the first 8 of bench/fixtures/chat_prompts.jsonl, so a 27B ABBA set takes ~40 min


# --- configuration -----------------------------------------------------------------------------------

def load_config(path: Path = OPTIONS_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def serve_flags(source: Path = REPO / "src" / "serve" / "serve_options.cpp") -> set[str]:
    """Every flag ninfer-serve compares an argument with (`arg == "--name"`), to catch a misspelt option."""
    return set(re.findall(r'== "(--[a-z0-9-]+)"', source.read_text(encoding="utf-8")))


def validate_config(config: dict, known_flags: set[str] | None = None) -> list[str]:
    """Problems with the options file, one line each; empty when it is usable."""
    problems: list[str] = []
    models, workloads = config.get("models", {}), config.get("workloads", {})
    for name, workload in workloads.items():
        if workload.get("tool") not in TOOLS:
            problems.append(f"workload {name}: tool must be one of {TOOLS}")
        if workload.get("model") not in models:
            problems.append(f"workload {name}: unknown model {workload.get('model')!r}")
        if workload.get("tool") == "engine-bench" and workload.get("metric") not in ("decode", "prefill"):
            problems.append(f"workload {name}: an engine-bench workload needs metric decode or prefill")
    seen: set[str] = set()
    for option in config.get("options", []):
        flag = option.get("flag", "?")
        where = f"option {flag}"
        if flag in seen:
            problems.append(f"{where}: listed twice")
        seen.add(flag)
        if known_flags is not None and flag not in known_flags:
            problems.append(f"{where}: ninfer-serve does not parse this flag")
        for key in ("item", "model", "kind", "summary", "tests"):
            if key not in option:
                problems.append(f"{where}: missing {key!r}")
        if option.get("model") not in models:
            problems.append(f"{where}: unknown model {option.get('model')!r}")
        if option.get("kind") not in KINDS:
            problems.append(f"{where}: kind must be one of {KINDS}")
        try:
            re.compile(option.get("tests", ""))
        except re.error as error:
            problems.append(f"{where}: bad tests regex ({error})")
        if bool(option.get("ab")) == bool(option.get("ab_reason")):
            problems.append(f"{where}: give exactly one of ab (variants) and ab_reason")
        for variant in option.get("ab", []):
            if variant.get("workload") not in workloads:
                problems.append(f"{where}: unknown workload {variant.get('workload')!r}")
            if variant.get("a") == variant.get("b"):
                problems.append(f"{where}: an A/B variant whose arms are equal compares nothing")
        if option.get("kind") == "quality" and not option.get("kld") and not option.get("kld_reason"):
            problems.append(f"{where}: a quality option needs a kld harness flag or a kld_reason")
        for command in option.get("identity", []):
            if not command or not str(command[0]).startswith("ninfer_"):
                problems.append(f"{where}: an identity command starts with the ninfer_tests test name")
    ids = [check.get("id") for check in config.get("checks", [])]
    if len(ids) != len(set(ids)):
        problems.append("checks: an id is listed twice")
    return problems


# --- context ---------------------------------------------------------------------------------------

@dataclass
class Ctx:
    args: argparse.Namespace
    config: dict
    out: Path
    windows: bool
    build: Path

    def exe(self, relative: str) -> Path:
        return self.build / (relative + (".exe" if self.windows else ""))

    @property
    def tests(self) -> Path:
        return self.exe("tests/ninfer_tests")

    @property
    def serve(self) -> Path:
        return self.exe("apps/ninfer-serve")

    def model(self, name: str) -> str | None:
        flag = self.config["models"][name]["input"]
        value = getattr(self.args, flag.lstrip("-").replace("-", "_"))
        return str(value) if value else None

    def need_model(self, name: str) -> str:
        """The artifact of a model a step needs (missing_inputs has already checked it is given)."""
        path = self.model(name)
        if path is None:
            raise RuntimeError(f"{self.config['models'][name]['input']} is required here")
        return path

    def model_env(self, name: str) -> dict[str, str]:
        path = self.model(name) or ""
        return {key: value.replace("{model}", path) for key, value in self.config["models"][name]["env"].items()}

    def subst(self, token: str) -> str:
        return token.replace("{fixtures}", str(FIXTURES)).replace("{out}", str(self.out))


@dataclass
class Step:
    id: str
    lock: str  # none | per-test (the test runner locks each test) | hold (the wrapper holds the lock)
    title: str
    needs: list[str] = field(default_factory=list)  # inputs: model names, "kld-base-flash-next", ...
    show: list[str] = field(default_factory=list)  # the commands, for the plan
    run: Callable[["Ctx", "Step"], dict] | None = None
    data: dict = field(default_factory=dict)


def quote(argv: list[str] | str) -> str:
    if isinstance(argv, str):  # a command line already (cmd's own quoting)
        return argv
    return " ".join(f'"{a}"' if (" " in a or not a) else a for a in map(str, argv))


def missing_inputs(ctx: Ctx, step: Step) -> list[str]:
    missing = []
    for need in step.needs:
        if need in ctx.config["models"]:
            if not ctx.model(need):
                missing.append(f"{ctx.config['models'][need]['input']} (the {need} artifact)")
        elif need == "windows" and not ctx.windows:
            missing.append("Windows (this step is Windows-only)")
        elif need.startswith("--"):
            if not getattr(ctx.args, need.lstrip("-").replace("-", "_")):
                missing.append(need)
        elif need.startswith("exe:"):
            if not ctx.exe(need[4:]).is_file():
                missing.append(f"{ctx.exe(need[4:])} (not built)")
        elif need.startswith("path:"):
            if not shutil.which(need[5:]):
                missing.append(f"{need[5:]} on PATH")
    return missing


# --- running commands -------------------------------------------------------------------------------

# The test-only side-stream delay (src/core/side_stream_delay.h). The identity gates set their own
# (2 ms) when it is unset; a value inherited from the shell would replace it (0 turns the gates'
# missing-join check off), and in a timing A/B any delay slows the arms. So no child inherits it.
SIDE_STREAM_DELAY = "NINFER_TEST_SIDE_STREAM_DELAY_US"


def child_env(env: dict[str, str] | None) -> dict[str, str]:
    """os.environ with env on top, without the side-stream delay variable."""
    merged = {**os.environ, **(env or {})}
    merged.pop(SIDE_STREAM_DELAY, None)
    return merged


def run_logged(argv: list[str], log: Path, env: dict[str, str] | None = None, timeout: float | None = None,
               cwd: Path | None = None) -> tuple[int, str]:
    """Run argv to completion, keep its combined output in log, return (exit code, output)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {quote(argv)}", flush=True)
    started = time.monotonic()
    try:
        done = subprocess.run([str(a) for a in argv], capture_output=True, text=True, errors="replace",
                              env=child_env(env), timeout=timeout, cwd=cwd)
        code, output = done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired as expired:
        code = -1
        output = f"{expired.stdout or ''}{expired.stderr or ''}\n[timed out after {timeout} s]"
    except OSError as error:
        code, output = -2, f"could not start: {error}"
    log.write_text(f"$ {quote(argv)}\n# exit {code}, {time.monotonic() - started:.1f} s\n{output}", encoding="utf-8")
    return code, output


def tail(text: str, lines: int = 6) -> str:
    return " | ".join(line.strip() for line in text.strip().splitlines()[-lines:])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def http(port: int, path: str, *, host: str | None = None, headers: dict | None = None,
         body: dict | None = None, timeout: float = 600.0) -> tuple[int, str, dict]:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
    if host is not None:
        request.add_header("Host", host)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace"), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), dict(error.headers)


class Server:
    """A ninfer-serve (or launcher) process this script started, stopped by its own PID only.

    argv is a list, or on Windows a command line string (the launcher smoke hands cmd its own
    quoting). Process groups follow the real OS, not --platform.
    """

    def __init__(self, argv: list[str] | str, log: Path, env: dict[str, str] | None, port: int,
                 windows: bool):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log, self.port = log, port
        self.windows = os.name == "nt"
        self.handle = open(log, "w", encoding="utf-8", errors="replace")
        print(f"    $ {quote(argv)}  (server, log {log})", flush=True)
        command = argv if isinstance(argv, str) else [str(a) for a in argv]
        full_env = child_env(env)
        if self.windows:
            self.proc = subprocess.Popen(command, stdout=self.handle, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=full_env,
                                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        else:
            self.proc = subprocess.Popen(command, stdout=self.handle, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=full_env, start_new_session=True)

    def wait_ready(self, timeout: float = 1800.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                if http(self.port, "/health", timeout=5)[0] == 200:
                    return True
            except OSError:
                pass
            time.sleep(1.0)
        return False

    def output(self) -> str:
        self.handle.flush()
        return self.log.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        if self.proc.poll() is None:
            if self.windows:  # the launcher's cmd and the server it started: this tree only
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
            else:
                getattr(os, "killpg")(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                if not self.windows:
                    getattr(os, "killpg")(self.proc.pid, getattr(signal, "SIGKILL"))
                self.proc.wait(timeout=30)
        self.handle.close()


# --- parsing ---------------------------------------------------------------------------------------

CTEST_LINE = re.compile(r"Test\s+#\d+:\s+(\S+)\s+\.*\s*(\*{0,3}[A-Za-z][A-Za-z :]*?)\s+[\d.]+\s+sec")


def parse_ctest(output: str) -> dict[str, str]:
    """Each test's outcome from ctest output: Passed, Failed, Skipped, Not Run, Timeout, ..."""
    outcome: dict[str, str] = {}
    for name, status in CTEST_LINE.findall(output):
        word = status.strip("* ")
        outcome[name] = "Not Run" if word.startswith("Not Run") else word.split(" ")[0].rstrip(":")
    return outcome


def ctest_names(build: Path, label: str | None = None) -> list[str]:
    argv = ["ctest", "--test-dir", str(build), "-N"] + (["-L", f"^{label}$"] if label else [])
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ctest -N failed in {build}: {tail(done.stdout + done.stderr)}")
    return re.findall(r"^\s*Test\s+#\d+:\s+(\S+)", done.stdout, re.MULTILINE)


KLD_FIELDS = {"ppl": r"^PPL:\s+([\d.]+)", "kld_mean": r"^KLD mean:\s+([\d.eE+-]+)",
              "kld_median": r"^KLD median:\s+([\d.eE+-]+)", "kld_p99": r"^KLD p99:\s+([\d.eE+-]+)",
              "kld_max": r"^KLD max:\s+([\d.eE+-]+)", "same_top1": r"^same top-1:\s+([\d.]+)%"}


def parse_kld(output: str) -> dict[str, float]:
    found = {}
    for key, pattern in KLD_FIELDS.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            found[key] = float(match.group(1))
    return found


def compare_rows(a: Path, b: Path) -> dict:
    """Two rows.tsv files (the KLD gate's columns): identical bytes, or how far apart."""
    if a.read_bytes() == b.read_bytes():
        return {"identical": True}
    rows_a = [line.split("\t") for line in a.read_text(encoding="utf-8").splitlines()]
    rows_b = [line.split("\t") for line in b.read_text(encoding="utf-8").splitlines()]
    header = rows_a[0]
    nll, kld = header.index("nll"), header.index("kld")
    differing, worst_nll, worst_kld = 0, 0.0, 0.0
    for row_a, row_b in zip(rows_a[1:], rows_b[1:]):
        if row_a != row_b:
            differing += 1
            worst_nll = max(worst_nll, abs(float(row_a[nll]) - float(row_b[nll])))
            worst_kld = max(worst_kld, abs(float(row_a[kld]) - float(row_b[kld])))
    return {"identical": False, "rows": len(rows_a) - 1, "rows_b": len(rows_b) - 1, "differing": differing,
            "max_abs_dnll": worst_nll, "max_abs_dkld": worst_kld}


CHAT_RESULT = re.compile(r"^RESULT rep(\d+) (\S+) C\d+ decode=([\d.]+) tok/s .*?accept=([\d.]+)% "
                         r"tok/round=([\d.]+)", re.MULTILINE)


def parse_chat_results(output: str) -> dict:
    """run_chat_decode.py RESULT lines -> per-rep values of arms off/on and the paired ratios."""
    reps: dict[int, dict[str, dict]] = {}
    for rep, arm, decode, accept, per_round in CHAT_RESULT.findall(output):
        reps.setdefault(int(rep), {})[arm] = {"decode": float(decode), "accept": float(accept),
                                              "tok_per_round": float(per_round)}
    ratios = [r["on"]["decode"] / r["off"]["decode"] for r in reps.values()
              if "on" in r and "off" in r and r["off"]["decode"] > 0]
    out = {"reps": reps, "pairs": len(ratios)}
    if ratios:
        out.update(median=statistics.median(ratios), min=min(ratios), max=max(ratios),
                   positive=sum(1 for r in ratios if r > 1.0))
        for arm in ("off", "on"):
            values = [r[arm]["accept"] for r in reps.values() if arm in r]
            out[f"accept_{arm}"] = statistics.median(values) if values else None
    return out


def wins(result: dict) -> bool:
    pairs = result.get("pairs", 0)
    return pairs > 0 and result.get("median", 0.0) >= WIN_MEDIAN and result.get("positive", 0) >= WIN_SHARE * pairs


def union_args(arg_lists: list[list[str]]) -> list[str]:
    """Several options' arguments as one command line: each flag once, a valued flag's last value."""
    merged: dict[str, list[str]] = {}
    for args in arg_lists:
        i = 0
        while i < len(args):
            flag = args[i]
            value = [args[i + 1]] if i + 1 < len(args) and not args[i + 1].startswith("--") else []
            merged.pop(flag, None)
            merged[flag] = [flag, *value]
            i += 1 + len(value)
    return [token for tokens in merged.values() for token in tokens]


# --- results ---------------------------------------------------------------------------------------

def result_path(ctx: Ctx, step_id: str) -> Path:
    return ctx.out / "results" / (re.sub(r"[^A-Za-z0-9._-]+", "_", step_id).strip("_") + ".json")


def load_result(ctx: Ctx, step_id: str) -> dict | None:
    path = result_path(ctx, step_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def log_dir(ctx: Ctx, step: Step) -> Path:
    return ctx.out / "logs" / re.sub(r"[^A-Za-z0-9._-]+", "_", step.id).strip("_")


def result(status: str, detail: str, **data) -> dict:
    return {"status": status, "detail": detail, "data": data}


# --- step runners ----------------------------------------------------------------------------------

def tests_command(ctx: Ctx, regex: str, exclude: str | None) -> list[str]:
    if ctx.windows:
        argv = ["pwsh", "-NoProfile", "-File", str(REPO / "tools" / "run-tests.ps1"), "-BuildDir", str(ctx.build),
                "-GpuOnly", "-Regex", regex]
        return argv + (["-Exclude", exclude] if exclude else [])
    argv = ["ctest", "--test-dir", str(ctx.build), "-L", "^gpu$", "-R", regex, "--timeout", "3600",
            "--output-on-failure"]
    return argv + (["-E", exclude] if exclude else [])


def run_tests_step(ctx: Ctx, step: Step) -> dict:
    names = step.data.get("names")
    if names is None:  # resolved now, so tests an earlier group already ran are left out
        names = resolve_group_names(ctx, step)
    if not names:
        return result("SKIP", "no gpu test matches this group", names=[])
    regex = "^(" + "|".join(re.escape(n) for n in names) + ")$"
    exclude = step.data.get("exclude")
    code, output = run_logged(tests_command(ctx, regex, exclude), log_dir(ctx, step) / "ctest.log",
                              env=step.data.get("env"))
    outcome = parse_ctest(output)
    failed = sorted(n for n, s in outcome.items() if s not in ("Passed", "Skipped"))
    skipped = sorted(n for n, s in outcome.items() if s == "Skipped")
    passed = sorted(n for n, s in outcome.items() if s == "Passed")
    detail = f"{len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed"
    if skipped:
        detail += " (skipped: " + ", ".join(skipped) + ")"
    supplied = any((step.data.get("env") or {}).values())
    if failed or code != 0:
        status = "FAIL"
    elif skipped and supplied:  # the model was given, so a skip means a test never reached it
        status = "REVIEW"
        detail += "; a model was given, so each skip needs a look (wrong artifact, missing drafter?)"
    else:
        status = "PASS" if passed else "SKIP"
    return result(status, detail, names=names, passed=passed, skipped=skipped, failed=failed, outcome=outcome)


def resolve_group_names(ctx: Ctx, step: Step) -> list[str]:
    gpu = ctest_names(ctx.build, "gpu")
    pattern = re.compile(step.data["regex"])
    earlier: set[str] = set()
    for other in step.data.get("after", []):
        previous = load_result(ctx, other)
        if previous:
            earlier.update(previous["data"].get("names", []))
    names = [n for n in gpu if pattern.search(n) and n not in earlier]
    step.data["names"] = names
    return names


def run_identity_step(ctx: Ctx, step: Step) -> dict:
    option = step.data["option"]
    outcomes = []
    for index, command in enumerate(option["identity"]):
        argv = [str(ctx.tests)] + [ctx.subst(token) for token in command]
        code, output = run_logged(argv, log_dir(ctx, step) / f"run{index}.log", env=ctx.model_env(option["model"]))
        outcomes.append({"command": command, "exit": code, "tail": tail(output, 3)})
    if any(o["exit"] not in (0, 77) for o in outcomes):
        bad = next(o for o in outcomes if o["exit"] not in (0, 77))
        return result("FAIL", f"{' '.join(bad['command'][:2])} exited {bad['exit']}: {bad['tail']}", runs=outcomes)
    if any(o["exit"] == 77 for o in outcomes):  # the model was given (the step needs it), so 77 is wrong
        return result("FAIL", "a harness skipped (77) although the model was given: "
                      + next(o["tail"] for o in outcomes if o["exit"] == 77), runs=outcomes)
    return result("PASS", f"{len(outcomes)} harness run(s) passed", runs=outcomes)


def kld_tsv(ctx: Ctx, name: str) -> Path:
    return ctx.out / "kld" / f"{name}.tsv"


def run_kld_step(ctx: Ctx, step: Step) -> dict:
    d = step.data
    gate = d.get("gate")
    if gate:  # the 100-chunk run waits for its 20-chunk run
        prior = load_result(ctx, gate)
        if not prior or prior["status"] not in ("PASS", "MEASURED"):
            state = prior["status"] if prior else "not run"
            return result("SKIP", f"{gate} has not passed ({state}), so the 100-chunk run waits for it")
    tsv = kld_tsv(ctx, d["name"])
    tsv.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(ctx.tests), "ninfer_qwen4_exp_forward_real_test", "--kld-base", str(ctx.args.kld_base_flash_next),
            "--chunks", str(d["chunks"]), "--window", str(d["window"]), "--tsv", str(tsv)] + d.get("flags", [])
    code, output = run_logged(argv, log_dir(ctx, step) / "kld.log", env=ctx.model_env("flash-next"))
    summary = parse_kld(output)
    if code != 0 or "kld_mean" not in summary:
        return result("FAIL", f"exit {code}: {tail(output, 3)}", summary=summary)
    against = d.get("against")
    if not against:
        return result("MEASURED", f"KLD mean {summary['kld_mean']:.6f}, same top-1 {summary.get('same_top1', 0):.2f}%",
                      summary=summary)
    off = load_result(ctx, f"kld:{against}")
    if not off or not kld_tsv(ctx, against).is_file():
        return result("FAIL", f"the baseline run kld:{against} has no result", summary=summary)
    rows = compare_rows(kld_tsv(ctx, against), tsv)
    base = off["data"]["summary"]
    delta = {key: summary[key] - base[key] for key in summary if key in base}
    if d.get("expect") == "identical":
        status = "PASS" if rows["identical"] else "FAIL"
        detail = "rows identical" if rows["identical"] else f"{rows['differing']} of {rows['rows']} rows differ " \
                                                           f"(max |dnll| {rows['max_abs_dnll']:.3g})"
    else:  # an option under the gate: it must change something, and its numbers are the measurement
        status = "FAIL" if rows["identical"] else "MEASURED"
        detail = ("the rows equal the option-off run bit for bit: the option did not reach the forward"
                  if rows["identical"] else
                  f"KLD mean {summary['kld_mean']:.6f} ({delta.get('kld_mean', 0):+.6f} vs off), same top-1 "
                  f"{summary.get('same_top1', 0):.2f}% ({delta.get('same_top1', 0):+.2f}), "
                  f"PPL {summary.get('ppl', 0):.4f}")
    return result(status, detail, summary=summary, delta=delta, rows=rows)


def run_kld_qwen35_step(ctx: Ctx, step: Step) -> dict:
    window = step.data["window"]
    out = ctx.out / "kld" / f"qwen35-w{window}"
    if out.exists():
        shutil.rmtree(out)  # ninfer-perplexity refuses a non-empty output directory
    argv = [str(ctx.exe("apps/ninfer-perplexity")), ctx.need_model("qwen38-27b"), "--kld-base",
            str(ctx.args.kld_base_27b), "--chunks", "20", "--decode-window", str(window), "--output", str(out)]
    code, output = run_logged(argv, log_dir(ctx, step) / "perplexity.log")
    if code != 0:
        return result("FAIL", f"exit {code}: {tail(output, 3)}")
    report = json.loads((out / "report.json").read_text(encoding="utf-8")) if (out / "report.json").is_file() else {}
    if window == 1:
        return result("MEASURED", "window 1 scored", report=report)
    other = ctx.out / "kld" / "qwen35-w1" / "rows.tsv"
    if not other.is_file() or not (out / "rows.tsv").is_file():
        return result("MEASURED", "no rows.tsv pair to compare", report=report)
    rows = compare_rows(other, out / "rows.tsv")
    detail = "rows identical at T = 1 and T = 3" if rows["identical"] else \
        f"{rows['differing']} of {rows['rows']} rows differ, max |dnll| {rows['max_abs_dnll']:.3g}, max |dkld| " \
        f"{rows['max_abs_dkld']:.3g} (a different route per width; X-A's provisional bound is mean " \
        "|dlogprob| <= 0.05)"
    return result("MEASURED", detail, report=report, rows=rows)


def run_mma_timing_step(ctx: Ctx, step: Step) -> dict:
    code, output = run_logged([str(ctx.tests), "ninfer_linear_ggml_test", "--time"], log_dir(ctx, step) / "time.log")
    rows = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 6 and parts[0] == "time":
            rows.append({"problem": parts[1], "t": parts[2], "dp4a_us": float(parts[3].split()[1]),
                         "mma_us": float(parts[4].split()[1]), "ratio": float(parts[5].split()[1])})
    if code != 0 or not rows:
        return result("FAIL", f"exit {code}: {tail(output, 3)}", rows=rows)
    slower = sorted({r["problem"] for r in rows if r["ratio"] > 1.0})
    return result("MEASURED", f"{len(rows)} problem x T timings; the tensor-core route is slower on "
                              f"{len(slower)} problem(s)" + (": " + ", ".join(slower) if slower else ""),
                  rows=rows, slower=slower)


def run_mma_policy_step(ctx: Ctx, step: Step) -> dict:
    found = {}
    for arm, extra in (("off", []), ("on", ["--ggml-decode-mma"])):
        report = log_dir(ctx, step) / f"engine-{arm}"
        argv = ["nsys", "profile", "--force-overwrite", "true", "--trace", "cuda", "-o", str(report), str(ctx.tests),
                "ninfer_qwen4_exp_engine_real_test", "--bench", "--generate", "16", "--max-context", "8192",
                "--no-expert-state", *extra, "@" + str(FIXTURES / "qwen4_exp" / "ppl_tokens-chat.ids")]
        code, output = run_logged(argv, log_dir(ctx, step) / f"profile-{arm}.log", env=ctx.model_env("flash-next"))
        if code != 0:
            return result("FAIL", f"nsys profile ({arm}) exited {code}: {tail(output, 3)}")
        code, stats = run_logged(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                  str(report) + ".nsys-rep"], log_dir(ctx, step) / f"stats-{arm}.log")
        if code != 0:
            return result("FAIL", f"nsys stats ({arm}) exited {code}: {tail(stats, 3)}")
        found[arm] = "ggml_mma_decode_kernel" in stats
    ok = found["on"] and not found["off"]
    return result("PASS" if ok else "FAIL",
                  "ggml_mma_decode_kernel runs with --ggml-decode-mma and not without it" if ok else
                  f"ggml_mma_decode_kernel present: off {found['off']}, on {found['on']}", found=found)


def engine_metric(workload: dict) -> str:
    job = Path(workload["jobs"][-1]).name
    label = re.escape(job) + r": [^\n]*?"
    if workload["metric"] == "decode":
        return label + r"decode [0-9.]+ s \(([0-9.]+) tok/s\)"
    return label + r"prefill [0-9.]+ s \(([0-9.]+) tok/s\)"


# A job's label is its token file as given (@path), which may hold spaces.
JOB_LINE = re.compile(r"^(.+?): \d+ prompt tokens, [^\n]*ids ([0-9a-f]{16})$", re.MULTILINE)
SPEC_LINE = re.compile(r"^(.+?): (mtp rounds [^\n]*)$", re.MULTILINE)


def job_name(label: str) -> str:
    return Path(label.lstrip("@").replace("\\", "/")).name


def engine_identity(run_logs: list[Path]) -> dict:
    """The engine bench's per-job results across every run of an A/B (both arms).

    Each run prints, per job, an FNV hash of its greedy ids and, with MTP, a line of rounds, plain
    steps, drafts and acceptances. An option designed to keep the output must leave every job's hash
    and speculation line the same in every run, on and off (a drafter that diverged would change the
    second while the ids still agree)."""
    ids: dict[str, set[str]] = {}
    spec: dict[str, set[str]] = {}
    runs_seen = 0
    for log in run_logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        runs_seen += 1
        for label, value in JOB_LINE.findall(text):
            ids.setdefault(job_name(label), set()).add(value)
        for label, value in SPEC_LINE.findall(text):
            spec.setdefault(job_name(label), set()).add(value)
    complete = runs_seen > 0 and bool(ids) and all(
        len(re.findall(r"ids [0-9a-f]{16}", log.read_text(encoding="utf-8", errors="replace"))) == len(ids)
        for log in run_logs)
    return {"runs": runs_seen, "jobs": sorted(ids),
            "same_ids": complete and all(len(v) == 1 for v in ids.values()),
            "same_spec": (all(len(v) == 1 for v in spec.values()) and set(spec) == set(ids)) if spec else None,
            "ids_differ": sorted(job for job, v in ids.items() if len(v) > 1),
            "spec_differ": sorted(job for job, v in spec.items() if len(v) > 1)}


def run_ab_step(ctx: Ctx, step: Step) -> dict:
    d = step.data
    workload = ctx.config["workloads"][d["workload"]]
    logs = log_dir(ctx, step)
    same_spec: bool | None = None
    # The identity verdicts read every log under runs/: a rerun of the step (--resume after a
    # failure) must not mix in the logs of an earlier attempt.
    shutil.rmtree(logs / "runs", ignore_errors=True)
    if workload["tool"] == "engine-bench":
        ensure_inputs(ctx)
        command = ["{exe}", "ninfer_qwen4_exp_engine_real_test", *workload["args"],
                   *("@" + ctx.subst(job) for job in workload["jobs"])]
        # Each arm warm-starts from its own copy of the saved expert state and saves into that copy:
        # the arms never share a state and the user's <model>.expert-state is only read.
        states = logs / "state"
        states.mkdir(parents=True, exist_ok=True)
        saved = Path(ctx.need_model(workload["model"]) + ".expert-state")
        a_args, b_args = list(d["a"]), list(d["b"])
        for arm, extra in (("A", a_args), ("B", b_args)):
            copy = states / f"{arm}.expert-state"
            if saved.is_file() and not copy.is_file():
                shutil.copy2(saved, copy)
            extra += ["--expert-state", str(copy)]
        report = logs / "ab.json"
        argv = [sys.executable, str(HERE / "run_interleaved_ab.py"), "--arm-a", str(ctx.tests),
                "--reps", str(ctx.args.reps), f"--a-args={shlex.join(a_args)}", f"--b-args={shlex.join(b_args)}",
                "--config", "run:" + quote(command), "--metric-regex", engine_metric(workload),
                "--log-dir", str(logs / "runs"), "--json", str(report), "--timeout", "3600", "--quiet"]
        code, output = run_logged(argv, logs / "ab.log", env=ctx.model_env(workload["model"]))
        if not report.is_file():
            return result("FAIL", f"run_interleaved_ab.py exited {code}: {tail(output, 3)}")
        summary = json.loads(report.read_text(encoding="utf-8"))["configs"][0]
        identity = engine_identity(sorted((logs / "runs").glob("*.log")))
        same_ids, same_spec = identity["same_ids"], identity["same_spec"]
        stats = {k: summary.get(k) for k in ("pairs", "median", "min", "max", "positive")}
    else:
        prompts = ensure_inputs(ctx)
        serve = str(ctx.serve)
        arm = lambda name, extra: f"{name}={serve}" + (f";add={' '.join(extra)}" if extra else "")  # noqa: E731
        argv = [sys.executable, str(HERE / "run_chat_decode.py"), "--model", ctx.need_model(workload["model"]),
                "--prompts", str(prompts), "--out", str(logs / "runs"), "--reps", str(ctx.args.reps),
                "--arm", arm("off", d["a"]), "--arm", arm("on", d["b"]), "--port", str(free_port()),
                *workload["args"]]
        code, output = run_logged(argv, logs / "ab.log")
        stats = parse_chat_results(output)
        if code != 0 or not stats["pairs"]:
            return result("FAIL", f"run_chat_decode.py exited {code}: {tail(output, 3)}", **stats)
        args = workload["args"]
        greedy = "--temperature" in args and args[args.index("--temperature") + 1] in ("0", "0.0")
        same_ids = chat_outputs_equal(logs / "runs") if greedy else None
        identity = {}
    return ab_verdict(d, stats, code, output, same_ids, same_spec, identity)


def ab_verdict(d: dict, stats: dict, code: int, output: str, same_ids: bool | None, same_spec: bool | None,
               identity: dict) -> dict:
    """An A/B's result: the paired numbers, and for an identity option whether the outputs agreed."""
    status = "MEASURED"
    if stats.get("pairs"):
        detail = (f"B/A median {100 * (stats['median'] - 1):+.2f}%, positive {stats['positive']}/{stats['pairs']}, "
                  f"range {100 * (stats['min'] - 1):+.2f}%..{100 * (stats['max'] - 1):+.2f}%")
    else:
        detail = "no complete pairs"
    if not stats.get("pairs") or code != 0:
        status = "FAIL"
        detail += f"; the A/B tool exited {code}: {tail(output, 2)}"
    if same_ids is not None:
        jobs = len(identity.get("jobs", [])) or None
        detail += (f"; greedy ids of {'every job' if jobs is None else f'all {jobs} jobs'} identical in every run"
                   if same_ids else "; greedy ids DIFFER between runs"
                   + (f" ({', '.join(identity['ids_differ'])})" if identity.get("ids_differ") else ""))
    if same_spec is not None:
        detail += ("; MTP rounds, drafts and acceptances identical in every run" if same_spec else
                   "; MTP rounds, drafts or acceptances DIFFER between runs")
    if d["kind"] == "identity" and (same_ids is False or same_spec is False):
        status = "FAIL"
    if "accept_off" in stats:
        detail += f"; acceptance {stats['accept_off']:.1f}% -> {stats['accept_on']:.1f}%"
    return result(status, detail, wins=status == "MEASURED" and wins(stats), same_ids=same_ids,
                  same_spec=same_spec, identity=identity, a=d["a"], b=d["b"], workload=d["workload"], **stats)


def chat_outputs_equal(runs: Path) -> bool:
    """run_chat_decode.py's responses: every rep's 'on' answers equal its 'off' answers."""
    texts = {}
    for path in runs.glob("rep*_*/responses.json"):
        rep, arm = path.parent.name.split("_", 1)
        texts[(rep, arm)] = [r["choices"][0]["message"].get("content") for r in json.loads(path.read_text("utf-8"))]
    reps = {rep for rep, _ in texts}
    return bool(reps) and all(texts.get((rep, "off")) == texts.get((rep, "on")) for rep in reps)


def ensure_inputs(ctx: Ctx) -> Path:
    """The prompt files the workloads name: a 2,048-token prompt and the first chat prompts."""
    inputs = ctx.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    two_k = inputs / "prompt-2k.ids"
    if not two_k.is_file():
        ids = []
        for name in ("ppl_tokens-chat.ids", "ppl_tokens-doc.ids"):
            ids += re.split(r"[\s,]+", (FIXTURES / "qwen4_exp" / name).read_text(encoding="utf-8").strip())
        two_k.write_text(" ".join(ids[:2048]) + "\n", encoding="utf-8")
    chat = inputs / f"chat{CHAT_PROMPTS}.jsonl"
    if not chat.is_file():
        lines = [line for line in (REPO / "bench" / "fixtures" / "chat_prompts.jsonl").read_text("utf-8").splitlines()
                 if line.strip()]
        chat.write_text("\n".join(lines[:CHAT_PROMPTS]) + "\n", encoding="utf-8")
    return chat


def combined_choice(ctx: Ctx, step: Step) -> tuple[list[str], list[str]]:
    """The options a combined row turns on (each one's best winning variant) and their B arguments.

    Options of the step's model and kinds, from variants on the step's source workloads. Without
    "quality" in the kinds a variant whose B arguments carry a quality option's flag is left out
    (--hc-fused-read's B holds --hc-int8), so that row stays free of numerics changes."""
    d = step.data
    options = ctx.config["options"]
    quality_flags = {o["flag"] for o in options if o["kind"] == "quality"}
    chosen, args = [], []
    for option in options:
        if option["model"] != d["model"] or option["kind"] not in d["include"]:
            continue
        best = None
        for index, variant in enumerate(option.get("ab", [])):
            if variant["workload"] not in d["from"]:
                continue
            if "quality" not in d["include"] and quality_flags & set(variant["b"]):
                continue
            r = load_result(ctx, ab_id(option, index))
            if r and r["data"].get("wins") and (best is None or r["data"]["median"] > best[1]["data"]["median"]):
                best = (variant, r)
        if best:
            chosen.append(option["flag"])
            args.append(best[0]["b"])
    return chosen, union_args(args)


def run_combined_step(ctx: Ctx, step: Step) -> dict:
    chosen, b_args = combined_choice(ctx, step)
    if not chosen:
        return result("SKIP", "no option won its A/B (median >= +1%, 3/4 of pairs positive)", options=[])
    step.data.update(a=[], b=b_args, kind="combined")
    measured = run_ab_step(ctx, step)
    measured["data"]["options"] = chosen
    measured["detail"] = f"{len(chosen)} option(s) on ({' '.join(b_args)}): " + measured["detail"]
    return measured


def chat_prompt_lines() -> list[str]:
    lines = (REPO / "bench" / "fixtures" / "chat_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


def run_greedy_master_step(ctx: Ctx, step: Step) -> dict:
    """One prompt of one configuration, so each GPU-lock hold is two CLI runs, not a whole prompt set."""
    d = step.data
    prompts = ctx.out / "inputs" / f"prompt-{d['prompt']}.jsonl"
    prompts.parent.mkdir(parents=True, exist_ok=True)
    prompts.write_text(chat_prompt_lines()[d["prompt"] - 1] + "\n", encoding="utf-8")
    argv = [sys.executable, str(HERE / "greedy_identity.py"), "--a", str(ctx.args.master_cli),
            "--b", str(ctx.exe("apps/ninfer")), "--model", ctx.need_model(d["model"]), "--tokens", "128",
            "--prompts", str(prompts), f"--extra-args={d['extra']}"]
    code, output = run_logged(argv, log_dir(ctx, step) / "greedy.log")
    verdict = next((line for line in output.splitlines() if line.startswith("GREEDY")), tail(output, 2))
    return result({0: "PASS", 1: "FAIL"}.get(code, "ERROR"), verdict, config=d["config"], prompt=d["prompt"])


def no_state(model: str) -> list[str]:
    """Flash-Next servers save their expert state beside the model; a check must not touch it."""
    return ["--no-expert-state"] if model == "flash-next" else []


def run_relaxed_startup_step(ctx: Ctx, step: Step) -> dict:
    model = step.data["model"]
    port = free_port()
    argv = [str(ctx.serve), ctx.need_model(model), "--host", "127.0.0.1", "--port", str(port), "--max-context",
            "8192", "--spec", "mtp", "--draft-tokens", "3", "--relaxed-drafting", "0.5", *no_state(model)]
    server = Server(argv, log_dir(ctx, step) / "serve.log", None, port, ctx.windows)
    try:
        if not server.wait_ready():
            return result("FAIL", "the server did not start: " + tail(server.output(), 4))
        code, body, _ = http(port, "/v1/chat/completions", body={
            "messages": [{"role": "user", "content": "Write a short poem about rivers."}], "max_tokens": 96,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20})
        _, metrics, _ = http(port, "/metrics")
        kept = re.search(r"^ninfer_spec_decode_relaxed_kept_tokens_total\S*\s+([\d.]+)", metrics, re.MULTILINE)
        if code != 200:
            return result("FAIL", f"a sampled request returned {code}: {body[:200]}")
        return result("PASS", "started with --relaxed-drafting (added ids contiguous); relaxed_kept after one "
                              f"sampled request: {kept.group(1) if kept else 'not reported'}")
    finally:
        server.stop()


def pin_failure_line(line: str) -> bool:
    """A line reporting a failed pin, and not the page-lock cap warning (which says "fails to pin")."""
    if "cudaHostRegister" in line or "warning" in line.lower():
        return False
    return bool(re.search(r"\bpin|cudaHostAlloc", line, re.I) and re.search(r"fail|error|out of memory", line, re.I))


def run_partial_start_step(ctx: Ctx, step: Step) -> dict:
    port = free_port()
    argv = [str(ctx.serve), ctx.need_model("flash-next"), "--host", "127.0.0.1", "--port", str(port),
            "--partial-ram-residency", *no_state("flash-next")]
    server = Server(argv, log_dir(ctx, step) / "serve.log", None, port, ctx.windows)
    try:
        ready = server.wait_ready()
        text = server.output()
        lines = text.splitlines()
        warning = next((line for line in lines if "cudaHostRegister" in line), None)
        pin_failure = next((line for line in lines if pin_failure_line(line)), None)
        if ready:
            code, body, _ = http(port, "/v1/chat/completions", body={
                "messages": [{"role": "user", "content": "Count to five."}], "max_tokens": 32, "temperature": 0})
            ready = code == 200
        detail = ("started and answered" if ready else "did not start") + \
                 ("; page-lock cap warning shown" if warning else "; no page-lock cap warning") + \
                 (f"; pin failure: {pin_failure.strip()}" if pin_failure else "")
        return result("PASS" if ready else "FAIL", detail, warning=warning, pin_failure=pin_failure,
                      windows=ctx.windows, tail=tail(text, 8))
    finally:
        server.stop()


def run_default_state_step(ctx: Ctx, step: Step) -> dict:
    """A session saves a state of its own (not the user's), which is exported and started from."""
    model = ctx.need_model("flash-next")
    inputs = ctx.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    chat = "@" + str(FIXTURES / "qwen4_exp" / "ppl_tokens-chat.ids")
    bench = [str(ctx.tests), "ninfer_qwen4_exp_engine_real_test", "--bench", "--generate", "64", "--max-context",
             "8192"]
    session, default, fresh = (inputs / name for name in ("session.expert-state", "flash-next.expert-state.default",
                                                          "fresh-install.expert-state"))
    for path in (session, fresh):
        path.unlink(missing_ok=True)
    code, output = run_logged([*bench, "--expert-state", str(session), chat], log_dir(ctx, step) / "session.log",
                              env=ctx.model_env("flash-next"))
    if code != 0 or not session.is_file():
        return result("FAIL", f"the session run saved no state (exit {code}): {tail(output, 3)}")
    code, output = run_logged([str(ctx.exe("tools/ninfer-qwen4-exp-expert-state-export")), str(session),
                               str(default), model], log_dir(ctx, step) / "export.log")
    if code != 0:
        return result("FAIL", f"the exporter exited {code}: {tail(output, 3)}")
    code, output = run_logged([*bench, "--expert-state", str(fresh), "--default-expert-state", str(default), chat],
                              log_dir(ctx, step) / "engine.log", env=ctx.model_env("flash-next"))
    used = "default state shipped with the model" in output or "by the default expert state" in output
    if code != 0:
        return result("FAIL", f"the engine bench exited {code}: {tail(output, 3)}")
    return result("PASS" if used else "FAIL", "warm-started from the exported default state" if used else
                  "the log does not say the default state was used (see engine.log)")


def run_serve_smoke_step(ctx: Ctx, step: Step) -> dict:
    inputs = ctx.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    config = inputs / "serve-smoke.json"
    config.write_text(json.dumps({"max-context": 8192, "kv-dtype": "int8", "allowed-host": ["gpubox.lan"]},
                                 indent=2) + "\n", encoding="utf-8")
    port = free_port()
    argv = [str(ctx.serve), ctx.need_model("qwen38-27b"), "--config", str(config), "--host", "127.0.0.1",
            "--port", str(port), "--admission-affinity"]
    server = Server(argv, log_dir(ctx, step) / "serve.log", None, port, ctx.windows)
    checks: dict[str, bool] = {}
    try:
        if not server.wait_ready():
            return result("FAIL", "the server did not start: " + tail(server.output(), 4))
        code, body, _ = http(port, "/health", host=f"evil.example:{port}")
        checks["Host evil.example refused (403 host_not_allowed)"] = code == 403 and "host_not_allowed" in body
        code, body, _ = http(port, "/health", host=f"gpubox.lan:{port}")
        checks["Host gpubox.lan (--allowed-host in the config file) accepted"] = code == 200
        code, body, _ = http(port, "/v1/models", headers={"Origin": "http://evil.example"})
        checks["Origin of another site refused (403 origin_not_allowed)"] = code == 403 and "origin_not_allowed" in body
        code, body, _ = http(port, "/v1/chat/completions", body={
            "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0})
        checks["a chat request answers"] = code == 200
        code, body, _ = http(port, "/slots")
        slots = json.loads(body) if code == 200 else None
        checks["/slots is llama.cpp's array shape"] = isinstance(slots, list) and bool(slots) and all(
            {"id", "is_processing", "n_ctx", "state"} <= set(slot) for slot in slots)
        code, page, headers = http(port, "/dashboard")
        (ctx.out / "serve-smoke").mkdir(parents=True, exist_ok=True)
        (ctx.out / "serve-smoke" / "dashboard.html").write_text(page, encoding="utf-8")
        checks["/dashboard serves HTML"] = code == 200 and "html" in headers.get("Content-Type", "")
        _, metrics, _ = http(port, "/metrics")
        for name in ("ninfer_gpu_power_limit_watts", "ninfer_admission_overtakes_total",
                     "ninfer_spec_decode_relaxed_kept_tokens_total"):
            checks[f"/metrics has {name}"] = re.search(rf"^{name}\b", metrics, re.MULTILINE) is not None
        checks["the settings line names the config file"] = any(
            "settings |" in line and config.name in line for line in server.output().splitlines())
    finally:
        server.stop()
    failed = [name for name, ok in checks.items() if not ok]
    detail = f"{len(checks) - len(failed)}/{len(checks)} checks passed" + \
        (" (failed: " + "; ".join(failed) + ")" if failed else "") + \
        f". Look at {ctx.out / 'serve-smoke' / 'dashboard.html'} by opening /dashboard on a live server in a " \
        "browser at desktop and phone width: the page is not checked visually here."
    return result("FAIL" if failed else "REVIEW", detail, checks=checks)


def run_serve_soak_step(ctx: Ctx, step: Step) -> dict:
    port = free_port()
    argv = [str(ctx.serve), ctx.need_model("qwen38-27b"), "--host", "127.0.0.1", "--port", str(port), "--max-context",
            "16384", "--max-concurrency", "1", "--admission-affinity"]
    server = Server(argv, log_dir(ctx, step) / "serve.log", None, port, ctx.windows)
    try:
        if not server.wait_ready():
            return result("FAIL", "the server did not start: " + tail(server.output(), 4))
        report = ctx.out / "serve-soak" / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        code, output = run_logged([sys.executable, str(HERE / "serve_soak.py"), "--url", f"http://127.0.0.1:{port}",
                                   "--minutes", "5", "--out", str(report), "--pid", str(server.proc.pid),
                                   "--max-concurrency", "4"], log_dir(ctx, step) / "soak.log")
        _, metrics, _ = http(port, "/metrics")
        overtakes = re.search(r"^ninfer_admission_overtakes_total\S*\s+([\d.]+)", metrics, re.MULTILINE)
    finally:
        server.stop()
    count = overtakes.group(1) if overtakes else "not reported"
    status = {0: "PASS", 1: "FAIL"}.get(code, "ERROR")
    return result(status, f"soak exit {code} ({'clean' if code == 0 else 'findings, see ' + str(report)}); "
                          f"admission overtakes: {count}", overtakes=count)


def run_community_step(ctx: Ctx, step: Step) -> dict:
    argv = [str(ctx.exe("bench/ninfer_bench")), "--preset", "community-1", "--weights", ctx.need_model("qwen38-27b"),
            "--submitter", "basalt-gpu-verify"]
    code, output = run_logged(argv, log_dir(ctx, step) / "bench.log")
    row = next((line for line in reversed(output.splitlines()) if line.startswith('{"submitter"')), None)
    if code != 0 or row is None:
        return result("FAIL", f"ninfer_bench exited {code}: {tail(output, 3)}")
    rows = ctx.out / "community" / "rows.json"
    rows.parent.mkdir(parents=True, exist_ok=True)
    rows.write_text(json.dumps({"preset": "community-1", "runs": [json.loads(row)]}, indent=2, ensure_ascii=False)
                    + "\n", encoding="utf-8")
    code, check = run_logged([sys.executable, str(HERE / "check_bench_rows.py"), "--fix", str(rows)],
                             log_dir(ctx, step) / "check.log")
    power = json.loads(row).get("gpus", [{}])[0].get("power_limit_w")
    return result("PASS" if code == 0 and power else "FAIL",
                  f"row {'valid' if code == 0 else 'invalid: ' + tail(check, 2)}; power limit {power} W", row=row)


def run_pytests_step(ctx: Ctx, step: Step) -> dict:
    have = lambda module: subprocess.run([sys.executable, "-c", f"import {module}"],  # noqa: E731
                                         capture_output=True).returncode == 0
    if not have("pytest") or not have("numpy"):
        return result("SKIP", "pytest or numpy is not installed for this Python; install them to run the suites")
    suites = ["tests/artifact/test_ggml_codecs.py"]
    note = ""
    if have("torch"):
        suites += ["tests/convert/test_qwen4_exp.py", "tests/convert/test_qwen4_exp_real.py"]
    else:
        note = "; torch is not installed, so the converter suites (tests/convert/test_qwen4_exp*.py) were skipped"
    env = {"NINFER_TEST_GGUF": str(ctx.args.gguf)} if ctx.args.gguf else {}
    code, output = run_logged([sys.executable, "-m", "pytest", "-q", *suites], log_dir(ctx, step) / "pytest.log",
                              env=env, cwd=REPO)
    summary = next((line for line in reversed(output.splitlines()) if re.search(r"passed|failed|error", line)), "")
    status = "FAIL" if code != 0 else ("SKIP" if note else "PASS")  # SKIP: the converter suites did not run
    return result(status, summary.strip("= ") + note, suites=suites, torch=not note)


def run_iq3_step(ctx: Ctx, step: Step) -> dict:
    pack = step.data["pack"]
    env = {"NINFER_TEST_QWEN4EXP_ARTIFACT": pack, "NINFER_TEST_ARTIFACT": pack}
    chat = "@" + str(FIXTURES / "qwen4_exp" / "ppl_tokens-chat.ids")
    runs = {
        "forward": [str(ctx.tests), "ninfer_qwen4_exp_forward_real_test", chat, "--decode", "4"],
        "engine": [str(ctx.tests), "ninfer_qwen4_exp_engine_real_test", "--bench", "--generate", "64",
                   "--max-context", "8192", "--no-expert-state", chat],
        "engine-partial": [str(ctx.tests), "ninfer_qwen4_exp_engine_real_test", "--bench", "--generate", "64",
                           "--max-context", "8192", "--no-expert-state", "--partial-ram-residency", chat],
    }
    if ctx.args.kld_base_flash_next:
        runs["kld"] = [str(ctx.tests), "ninfer_qwen4_exp_forward_real_test", "--kld-base",
                       str(ctx.args.kld_base_flash_next), "--chunks", "20", "--window", "3", "--tsv",
                       str(ctx.out / "kld" / f"{Path(pack).stem}.tsv")]
    outcome, kld = {}, {}
    for name, argv in runs.items():
        code, output = run_logged(argv, log_dir(ctx, step) / f"{name}.log", env=env)
        outcome[name] = code
        if name == "kld":
            kld = parse_kld(output)
    failed = [name for name, code in outcome.items() if code != 0]
    detail = ", ".join(f"{name} exit {code}" for name, code in outcome.items())
    if kld:
        detail += f"; KLD mean {kld.get('kld_mean', 0):.6f}, same top-1 {kld.get('same_top1', 0):.2f}%"
    return result("FAIL" if failed else "PASS", detail, outcome=outcome, kld=kld)


LAUNCHER_RUNS = [  # (model key, profile, environment, model input)
    ("qwen38-27b", "tuned", {}, "qwen38-27b"),
    ("qwen38-27b", "tuned", {"NINFER_SPEC": "mtp"}, "qwen38-27b"),
    ("qwen38-27b", "tuned", {"NINFER_SPEC": "none"}, "qwen38-27b"),
    ("qwen38-27b", "int8", {}, "qwen38-27b"),
    ("qwen38-27b", "c8", {}, "qwen38-27b"),
    ("qwen36-35b-a3b", "tuned", {}, "qwen36-35b"),
    ("qwen38-flash-next", "tuned", {}, "flash-next"),  # run.bat only: run.sh does not serve Flash-Next
]


def smoke_dir(ctx: Ctx) -> Path:
    base = Path(os.environ.get("TEMP", "/tmp")) if ctx.windows else Path("/tmp")
    return base / "ninfer smoke (x86)"


def run_package_step(ctx: Ctx, step: Step) -> dict:
    code, output = run_logged(["pwsh", "-NoProfile", "-File", str(REPO / "scripts" / "package-release.ps1")],
                              log_dir(ctx, step) / "package.log", env={"NINFER_BUILD_ROOT": str(ctx.build)}, cwd=REPO)
    products = sorted((p for p in (REPO / "dist").glob("ninfer-rtx3090-windows-x64-*") if p.is_dir()),
                      key=lambda p: p.stat().st_mtime)
    if code != 0 or not products:
        return result("FAIL", f"package-release.ps1 exited {code}: {tail(output, 3)}")
    target = smoke_dir(ctx)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(products[-1], target)
    return result("PASS", f"packaged {products[-1].name}; copied to {target}", product=str(products[-1]))


def run_launcher_step(ctx: Ctx, step: Step) -> dict:
    key, profile, env, model = step.data["run"]
    port = free_port()
    target = smoke_dir(ctx)
    if ctx.windows:
        if not (target / "run.bat").is_file():
            return result("FAIL", f"{target / 'run.bat'} is missing: the launchers:package step must pass first")
        # One command line, as cmd parses it: /s strips the outer quotes and keeps the inner ones, so
        # a path with spaces and parentheses survives (a list would be re-quoted by list2cmdline).
        argv: list[str] | str = f'cmd /d /s /c ""{target / "run.bat"}" {key} {profile}"'
        extra = {}
    else:
        if not (target / "run.sh").is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / "scripts" / "run.sh", target / "run.sh")
            shutil.copytree(REPO / "scripts" / "profiles", target / "profiles", dirs_exist_ok=True)
        argv = ["bash", str(target / "run.sh"), key, profile]
        extra = {"NINFER_SERVER": str(ctx.serve)}
    full_env = {**env, **extra, "NINFER_MODEL": ctx.need_model(model), "NINFER_PORT": str(port)}
    server = Server(argv, log_dir(ctx, step) / "launcher.log", full_env, port, ctx.windows)
    try:
        ready = server.wait_ready()
        text = server.output()
    finally:
        server.stop()
    settings = next((line for line in text.splitlines() if "settings |" in line), None)
    profile_named = settings is not None and ".json" in settings
    status = "PASS" if ready and profile_named else "FAIL"
    return result(status, ("started" if ready else "did not start: " + tail(text, 3)) +
                  ("; " + settings.strip() if settings else "; no settings line"), env=env, ready=ready)


# --- the plan --------------------------------------------------------------------------------------

def ab_id(option: dict, index: int) -> str:
    return f"ab:{option['flag']}" + (f":{index + 1}" if len(option.get("ab", [])) > 1 else "")


def build_steps(ctx: Ctx) -> list[Step]:
    config, steps = ctx.config, []
    options = config["options"]

    steps.append(Step("preflight", "none", "record the build, the GPU and the inputs", run=run_preflight))

    # 1. GPU tests, grouped by the artifact their real tests read; a test runs in the first group that
    #    lists it (Flash-Next first: the offloaded MoE tests read its expert banks).
    group_ids = []
    for model in ("flash-next", "qwen38-27b"):
        regexes = [o["tests"] for o in options if o["model"] == model]
        regex = "|".join(f"(?:{r})" for r in regexes)
        step = Step(f"tests:{model}", "per-test", f"GPU tests of the {model} options", needs=[model],
                    show=[quote(tests_command(ctx, f"<the gpu tests matching {len(regexes)} option regexes>",
                                              None))], run=run_tests_step,
                    data={"regex": regex, "env": ctx.model_env(model), "after": list(group_ids)})
        steps.append(step)
        group_ids.append(step.id)
    context_rows = next(c for c in config["checks"] if c["id"] == "context-rows")
    for model, spec in (("qwen38-27b", "dflash2"), ("qwen36-35b", "mtp")):
        env = {**ctx.model_env(model), "NINFER_CONTEXT_ROWS_SPEC": spec}
        steps.append(Step(f"tests:context-rows-{model}-{spec}", "per-test",
                          f"draft-head context rows on the {model} with {spec}", needs=[model],
                          show=[f"NINFER_CONTEXT_ROWS_SPEC={spec} "
                                + quote(tests_command(ctx, context_rows["tests"], None))],
                          run=run_tests_step, data={"regex": context_rows["tests"], "env": env, "names": None}))

    # 2. identity runs, one lock hold per option.
    for option in options:
        if option.get("identity"):
            steps.append(Step(f"identity:{option['flag']}", "hold", f"{option['flag']} on against off in one process",
                              needs=[option["model"]],
                              show=[quote([str(ctx.tests)] + [ctx.subst(t) for t in c]) for c in option["identity"]],
                              run=run_identity_step, data={"option": option}))

    # 3. quality: the KLD gate. The rows are deterministic, so one option-off run serves every option.
    kld_options = [o for o in options if o.get("kld")]
    kld_cmd = lambda chunks, window, flags: quote(  # noqa: E731
        [str(ctx.tests), "ninfer_qwen4_exp_forward_real_test", "--kld-base", "<kld base>", "--chunks", str(chunks),
         "--window", str(window), "--tsv", "<out>/kld/<run>.tsv", *flags])
    for chunks in (20, 100):
        base = f"off-{chunks}"
        gate = {"gate": "kld:off-20"} if chunks == 100 else {}
        steps.append(Step(f"kld:{base}", "hold", f"KLD gate, every option off, {chunks} chunks, window 3",
                          needs=["flash-next", "--kld-base-flash-next"], show=[kld_cmd(chunks, 3, [])],
                          run=run_kld_step, data={"name": base, "chunks": chunks, "window": 3, **gate}))
        if chunks == 20:
            steps.append(Step("kld:off-20-w1", "hold", "the window check: window 1 must give window 3's rows",
                              needs=["flash-next", "--kld-base-flash-next"], show=[kld_cmd(20, 1, [])],
                              run=run_kld_step, data={"name": "off-20-w1", "chunks": 20, "window": 1,
                                                      "against": base, "expect": "identical"}))
        for option in kld_options:
            name = f"{option['flag'].lstrip('-')}-{chunks}"
            gate = {"gate": f"kld:{option['flag'].lstrip('-')}-20"} if chunks == 100 else {}
            steps.append(Step(f"kld:{name}", "hold", f"KLD gate with {option['flag']}, {chunks} chunks"
                              + (" (after its 20-chunk run passes)" if gate else ""),
                              needs=["flash-next", "--kld-base-flash-next"], show=[kld_cmd(chunks, 3, [option["kld"]])],
                              run=run_kld_step, data={"name": name, "chunks": chunks, "window": 3,
                                                      "flags": [option["kld"]], "against": base, **gate}))
    for window in (1, 3):
        steps.append(Step(f"kld-qwen35:w{window}", "hold", f"ninfer-perplexity --kld-base on the 27B, window {window}",
                          needs=["qwen38-27b", "--kld-base-27b"],
                          show=[quote([str(ctx.exe("apps/ninfer-perplexity")), "<27B>", "--kld-base", "<base>",
                                       "--chunks", "20", "--decode-window", str(window), "--output",
                                       f"<out>/kld/qwen35-w{window}"])],
                          run=run_kld_qwen35_step, data={"window": window}))
    steps.append(Step("ggml-mma:timing", "hold", "--ggml-decode-mma per shape, dp4a against the tensor cores",
                      show=[quote([str(ctx.tests), "ninfer_linear_ggml_test", "--time"])], run=run_mma_timing_step))
    steps.append(Step("ggml-mma:policy", "hold", "the served decode selects the tensor-core route (nsys)",
                      needs=["flash-next", "path:nsys"],
                      show=["nsys profile -o <out> " + quote([str(ctx.tests), "ninfer_qwen4_exp_engine_real_test",
                                                               "--bench", "--generate", "16", "[--ggml-decode-mma]",
                                                               "@ppl_tokens-chat.ids"]),
                            "nsys stats --report cuda_gpu_kern_sum: ggml_mma_decode_kernel only with the option"],
                      run=run_mma_policy_step))

    # 4. speed: one ABBA set per variant, each inside one lock hold.
    for option in options:
        for index, variant in enumerate(option.get("ab", [])):
            workload = config["workloads"][variant["workload"]]
            if workload["tool"] == "engine-bench":
                show = (f"run_interleaved_ab.py --arm-a {ctx.tests} --reps {ctx.args.reps} "
                        f"--a-args='{shlex.join(variant['a'])}' --b-args='{shlex.join(variant['b'])}' "
                        f"--config 'run:{{exe}} ninfer_qwen4_exp_engine_real_test {' '.join(workload['args'])} "
                        f"{' '.join('@' + Path(j).name for j in workload['jobs'])}'")
            else:
                show = (f"run_chat_decode.py --reps {ctx.args.reps} --arm off={ctx.serve}"
                        f"{';add=' + ' '.join(variant['a']) if variant['a'] else ''} --arm on={ctx.serve};add="
                        f"{' '.join(variant['b'])} {' '.join(workload['args'])} (first {CHAT_PROMPTS} chat prompts)")
            steps.append(Step(ab_id(option, index), "hold", f"{option['flag']} A/B on {variant['workload']}",
                              needs=[workload["model"]], show=[show], run=run_ab_step,
                              data={**variant, "kind": option["kind"]}))

    # 5. the combined rows: each turns on every winning option of its model and kinds at once.
    q27 = [name for name, w in config["workloads"].items() if w["model"] == "qwen38-27b"]
    for name, model, workload, sources, include in (
            ("identity", "flash-next", "fn-decode-mtp", ["fn-decode-mtp"], ("identity", "speed")),
            ("all", "flash-next", "fn-decode-mtp", ["fn-decode-mtp"], ("identity", "speed", "quality")),
            ("prefill", "flash-next", "fn-prefill-2k", ["fn-prefill-2k"], ("identity", "speed", "quality")),
            ("qwen38-27b", "qwen38-27b", "q27-chat-sampled-dflash2", q27, ("identity", "speed", "quality"))):
        steps.append(Step(f"combined:{name}", "hold",
                          f"every winning {'/'.join(include)} {model} option on at once, on {workload}",
                          needs=[model], show=[f"the {workload} A/B with B = the winners' B arguments"],
                          run=run_combined_step, data={"workload": workload, "model": model, "from": sources,
                                                       "include": include}))

    # 6. the ledger's checks. Every option off against master: one step per configuration and prompt,
    #    so each lock hold is two CLI processes, not a whole prompt set.
    prompt_count = len(chat_prompt_lines())
    for model, label, extra, limit in (("qwen38-27b", "dflash2", "--spec dflash2 --draft-tokens 6", None),
                                       ("qwen38-27b", "mtp", "--spec mtp --draft-tokens 3", None),
                                       ("qwen38-27b", "none", "", None),
                                       ("flash-next", "plain", "", 8),
                                       ("flash-next", "mtp", "--spec mtp --draft-tokens 3", 8)):
        config_id = f"{model}-{label}"
        for prompt in range(1, (limit or prompt_count) + 1):
            steps.append(Step(f"greedy-master:{config_id}:{prompt}", "hold",
                              f"every option off: greedy ids equal master's ({config_id}, prompt {prompt})",
                              needs=[model, "--master-cli", "exe:apps/ninfer"],
                              show=[f"greedy_identity.py --a <master ninfer> --b {ctx.exe('apps/ninfer')} "
                                    f"--model <{model}> --tokens 128 --prompts <out>/inputs/prompt-{prompt}.jsonl "
                                    f"--extra-args='{extra}'"] if prompt == 1 else [],
                              run=run_greedy_master_step,
                              data={"model": model, "extra": extra, "prompt": prompt, "config": config_id}))
    for model in ("qwen38-27b", "qwen36-35b", "flash-next"):
        steps.append(Step(f"relaxed-startup:{model}", "hold", f"ninfer-serve --relaxed-drafting starts on the {model}",
                          needs=[model], show=[f"{ctx.serve} <{model}> --spec mtp --relaxed-drafting 0.5"],
                          run=run_relaxed_startup_step, data={"model": model}))
    steps.append(Step("partial-start", "hold", "a --partial-ram-residency start (page-lock warning, pinning)",
                      needs=["flash-next"], show=[f"{ctx.serve} <flash-next> --partial-ram-residency"],
                      run=run_partial_start_step))
    steps.append(Step("default-expert-state", "hold", "a session's saved state, exported, then started from",
                      needs=["flash-next", "exe:tools/ninfer-qwen4-exp-expert-state-export"],
                      show=["engine bench --expert-state <out>/inputs/session.expert-state",
                            "ninfer-qwen4-exp-expert-state-export <session> <default> <model>",
                            "engine bench --expert-state <absent> --default-expert-state <default>"],
                      run=run_default_state_step))
    steps.append(Step("serve-smoke", "hold", "a real 27B server: Host/Origin checks, /slots, /dashboard, /metrics",
                      needs=["qwen38-27b"], show=[f"{ctx.serve} <27B> --config <out>/inputs/serve-smoke.json "
                                                  "--admission-affinity, then HTTP probes"], run=run_serve_smoke_step))
    steps.append(Step("serve-soak", "hold", "serve_soak.py for 5 minutes, 4 in flight on one lane",
                      needs=["qwen38-27b"], show=[f"{ctx.serve} <27B> --max-concurrency 1 --admission-affinity; "
                                                  "serve_soak.py --minutes 5 --max-concurrency 4"],
                      run=run_serve_soak_step))
    steps.append(Step("community-preset", "hold", "ninfer_bench --preset community-1 and the row checker",
                      needs=["qwen38-27b", "exe:bench/ninfer_bench"],
                      show=[f"{ctx.exe('bench/ninfer_bench')} --preset community-1 --weights <27B> --submitter ..."],
                      run=run_community_step))
    steps.append(Step("converter-pytests", "none", "the converter and codec pytest suites (skipped without torch)",
                      show=["python -m pytest -q tests/artifact/test_ggml_codecs.py tests/convert/test_qwen4_exp.py "
                            "tests/convert/test_qwen4_exp_real.py"], run=run_pytests_step))
    for index, pack in enumerate(ctx.args.iq3_pack or ["<--iq3-pack>"]):
        named = pack != "<--iq3-pack>"
        steps.append(Step(f"iq3:{index + 1}:tests", "per-test", f"the offloaded MoE tests on {Path(pack).name}'s banks",
                          needs=[] if named else ["--iq3-pack"],
                          show=[quote(tests_command(ctx, "^ninfer_offloaded_moe_(layer|wide)_test$", None))],
                          run=run_tests_step, data={"regex": "^ninfer_offloaded_moe_(layer|wide)_test$", "names": None,
                                                    "env": {"NINFER_TEST_ARTIFACT": pack}}))
        steps.append(Step(f"iq3:{index + 1}:run", "hold", f"{Path(pack).name}: forward, engine start, KLD",
                          needs=[] if named else ["--iq3-pack"],
                          show=["forward_real @chat --decode 4; engine bench (plain and --partial-ram-residency); "
                                "KLD gate at 20 chunks with --kld-base-flash-next"],
                          run=run_iq3_step, data={"pack": pack}))
    if ctx.windows:
        steps.append(Step("launchers:package", "none", "package-release.ps1, copied to a path with spaces and (x86)",
                          show=["pwsh scripts/package-release.ps1", f"copy to {smoke_dir(ctx)}"], run=run_package_step))
    for key, profile, env, model in LAUNCHER_RUNS:
        if key == "qwen38-flash-next" and not ctx.windows:
            continue
        label = key + "-" + profile + "".join(f"-{v.lower()}" for v in env.values())
        launcher = "run.bat" if ctx.windows else "run.sh"
        settings = " ".join(f"{k}={v}" for k, v in env.items())
        steps.append(Step(f"launchers:{label}", "hold", f"{launcher} {key} {profile} {settings}".strip(),
                          needs=[model], show=[f"{settings} \"{smoke_dir(ctx) / launcher}\" {key} {profile}".strip()],
                          run=run_launcher_step, data={"run": (key, profile, env, model)}))

    # 7. the whole GPU suite, after the targeted runs.
    env = {**ctx.model_env("qwen38-27b"), "NINFER_TEST_QWEN4EXP_ARTIFACT": ctx.model("flash-next") or ""}
    steps.append(Step("full-suite", "per-test", f"every other GPU test, without the slow ones ({SLOW_TESTS})",
                      show=[quote(tests_command(ctx, ".", SLOW_TESTS + "|<tests the targeted runs passed>"))],
                      run=run_full_suite_step, data={"env": env}))
    return steps


def run_full_suite_step(ctx: Ctx, step: Step) -> dict:
    done: set[str] = set()
    for path in (ctx.out / "results").glob("tests_*.json"):
        done.update(json.loads(path.read_text(encoding="utf-8"))["data"].get("passed", []))
    pattern = re.compile(SLOW_TESTS)
    step.data["names"] = [n for n in ctest_names(ctx.build, "gpu") if not pattern.search(n) and n not in done]
    measured = run_tests_step(ctx, step)
    measured["detail"] += f"; {len(done)} tests already passed in the targeted runs were not repeated"
    return measured


def run_preflight(ctx: Ctx, step: Step) -> dict:
    info: dict = {"date": dt.datetime.now().isoformat(timespec="seconds"), "platform": sys.platform,
                  "python": sys.version.split()[0], "build_dir": str(ctx.build)}
    rev = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty", "--abbrev=12"],
                         capture_output=True, text=True)
    info["commit"] = rev.stdout.strip()
    smi = None
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True)
    info["gpu"] = smi.stdout.strip() if smi and smi.returncode == 0 else "nvidia-smi unavailable"
    inputs: dict[str, object] = {name: ctx.model(name) for name in ctx.config["models"]}
    info["inputs"] = inputs
    for key in ("kld_base_flash_next", "kld_base_27b", "master_cli", "gguf", "iq3_pack"):
        value = getattr(ctx.args, key)
        inputs[key] = [str(v) for v in value] if isinstance(value, list) else (str(value) if value else None)
    for relative in ("tests/ninfer_tests", "apps/ninfer-serve", "apps/ninfer", "apps/ninfer-perplexity",
                     "bench/ninfer_bench", "tools/ninfer-qwen4-exp-expert-state-export"):
        info.setdefault("executables", {})[relative] = ctx.exe(relative).is_file()
    ensure_inputs(ctx)
    problems = input_problems(ctx) + [f"{exe} is not built" for exe in (ctx.tests, ctx.serve) if not exe.is_file()]
    if problems:
        return result("FAIL", "; ".join(problems), **info)
    return result("PASS", f"{info['commit']} on {info['gpu']}", **info)


def split_alternation(pattern: str) -> list[str]:
    """A regex split at its top-level `|` (not inside a group or a character class)."""
    parts, depth, start, i, in_class = [], 0, 0, 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append(pattern[start:i])
            start = i + 1
        i += 1
    return parts + [pattern[start:]]


def regex_alternatives(pattern: str) -> list[str]:
    """Every alternative a regex spells out, with each group's alternation expanded: ^x_(a|b)_y$ gives
    ^x_(?:a)_y$ and ^x_(?:b)_y$. A misspelt alternative then matches nothing on its own."""
    out = []
    for part in split_alternation(pattern):
        depth, open_at, i, in_class, expanded = 0, -1, 0, False, False
        while i < len(part):
            c = part[i]
            if c == "\\":
                i += 2
                continue
            if in_class:
                in_class = c != "]"
            elif c == "[":
                in_class = True
            elif c == "(":
                if depth == 0:
                    open_at = i
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    inner = part[open_at + 1:i]
                    inner = inner[2:] if inner.startswith("?:") else inner
                    alternatives = split_alternation(inner)
                    if len(alternatives) > 1:
                        for alternative in alternatives:
                            out += regex_alternatives(part[:open_at] + "(?:" + alternative + ")" + part[i + 1:])
                        expanded = True
                        break
            i += 1
        if not expanded:
            out.append(part)
    return out


def regex_problems(label: str, pattern: str, names: list[str]) -> list[str]:
    """Each alternative of the regex that matches no ctest name."""
    if not names:
        return []
    if not any(re.search(pattern, n) for n in names):
        return [f"{label}: tests regex {pattern!r} matches no ctest name"]
    return [f"{label}: the alternative {alt!r} of its tests regex matches no ctest name (a typo?)"
            for alt in regex_alternatives(pattern) if not any(re.search(alt, n) for n in names)]


def input_problems(ctx: Ctx) -> list[str]:
    """Given paths that do not exist: caught before any step waits for the GPU lock."""
    problems = []
    for name, model in ctx.config["models"].items():
        path = ctx.model(name)
        if path and not Path(path).is_file():
            problems.append(f"{model['input']} {path}: no such file")
    for key in ("kld_base_flash_next", "kld_base_27b", "master_cli", "gguf"):
        value = getattr(ctx.args, key)
        if value and not Path(value).is_file():
            problems.append(f"--{key.replace('_', '-')} {value}: no such file")
    for pack in ctx.args.iq3_pack or []:
        if not Path(pack).is_file():
            problems.append(f"--iq3-pack {pack}: no such file")
    if not ctx.build.is_dir():
        problems.append(f"--build-dir {ctx.build}: no such directory")
    return problems


def print_plan(ctx: Ctx, steps: list[Step]) -> int:
    config = ctx.config
    print(f"Basalt GPU verification plan (issue #383): {len(config['options'])} options, {len(steps)} steps, "
          f"results to {ctx.out}")
    print(f"platform {'windows' if ctx.windows else 'linux'}, build dir {ctx.build}, A/B reps {ctx.args.reps}")
    problems = validate_config(config, serve_flags()) + input_problems(ctx)
    try:
        names, gpu = ctest_names(ctx.build), set(ctest_names(ctx.build, "gpu"))
    except (RuntimeError, OSError) as error:
        problems.append(str(error))
        names, gpu = [], set()
    print(f"ctest -N lists {len(names)} tests in {ctx.build}, {len(gpu)} labelled gpu\n")
    print("Options:")
    for option in config["options"]:
        matched = [n for n in names if re.search(option["tests"], n)]
        problems += regex_problems(option["flag"], option["tests"], names)
        for command in option.get("identity", []):
            if names and command[0] not in names:
                problems.append(f"{option['flag']}: identity harness {command[0]} is not a ctest name")
        ab = "; ".join(f"{v['workload']} A+[{shlex.join(v['a'])}] B+[{shlex.join(v['b'])}]"
                       for v in option.get("ab", []))
        print(f"  {option['flag']:<24} {option['item']:<4} {option['model']:<11} {option['kind']:<8} "
              f"tests {len(matched)} ({len([n for n in matched if n in gpu])} gpu): {option['tests']}")
        print(f"  {'':<24} " + ("A/B: " + ab if ab else "no A/B: " + option["ab_reason"]))
        if option.get("kld"):
            print(f"  {'':<24} KLD gate flag {option['kld']}")
    for check in config["checks"]:
        if check.get("tests"):
            problems += regex_problems(f"check {check['id']}", check["tests"], names)
    print("\nSteps, in order (lock: hold = one GPU-lock hold for the step; per-test = the test runner locks each "
          "test; none = no GPU):")
    for number, step in enumerate(steps, 1):
        missing = missing_inputs(ctx, step)
        print(f"  {number:>3}. {step.id:<44} {step.lock:<8} {step.title}"
              + (f"  [SKIP: needs {', '.join(missing)}]" if missing else ""))
        for line in step.show:
            print(f"         {line}")
    print("\nThe ledger's checks:")
    for check in config["checks"]:
        print(f"  {check['id']:<22} {check.get('item', ''):<4} {check['about']}")
    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Check passed: every alternative of every option's test regex, and every identity harness, matches a "
          f"ctest name ({len(config['options'])} options, {sum(1 for c in config['checks'] if c.get('tests'))} "
          "check regex).")
    return 0


# --- summary ---------------------------------------------------------------------------------------

def write_summary(ctx: Ctx, steps: list[Step]) -> Path:
    results = {step.id: load_result(ctx, step.id) for step in steps}
    pre = results.get("preflight") or {"data": {}}
    lines = [f"## GPU verification of the Basalt options ({dt.date.today().isoformat()})", "",
             f"Build `{pre['data'].get('commit', '?')}` on {pre['data'].get('gpu', '?')}, "
             f"{'Windows' if ctx.windows else 'Linux'}; A/B: ABBA, n = {ctx.args.reps}, B/A paired median; "
             f"a win is >= +{100 * (WIN_MEDIAN - 1):.0f}% with {WIN_SHARE:.0%} of pairs positive. Raw results: "
             f"`{ctx.out}`.", "", "| Option | Kind | GPU tests | Identity | KLD 20 / 100 chunks | A/B |",
             "|---|---|---|---|---|---|"]
    status_of = lambda step_id: (results.get(step_id) or {}).get("status", "not run")  # noqa: E731
    detail_of = lambda step_id: (results.get(step_id) or {}).get("detail", "")  # noqa: E731
    outcome: dict[str, str] = {}  # every GPU test's outcome, from the targeted groups and the full suite
    for step_id, r in results.items():
        if r and (step_id.startswith("tests:") or step_id == "full-suite"):
            for name, state in r["data"].get("outcome", {}).items():
                outcome.setdefault(name, state)
    for option in ctx.config["options"]:
        mine = {n: state for n, state in outcome.items() if re.search(option["tests"], n)}
        bad = [f"{n} {state}" for n, state in mine.items() if state != "Passed"]
        tests = (f"{len(mine) - len(bad)}/{len(mine)} gpu passed" + (f" ({', '.join(bad)})" if bad else "")) \
            if mine else "not run (or cpu only)"
        identity = status_of(f"identity:{option['flag']}") if option.get("identity") else "-"
        kld = "-"
        if option.get("kld"):
            name = option["flag"].lstrip("-")
            kld = " / ".join(detail_of(f"kld:{name}-{c}") or status_of(f"kld:{name}-{c}") for c in (20, 100))
        ab = "; ".join(f"{status_of(ab_id(option, i))}: {detail_of(ab_id(option, i))}".rstrip(": ")
                       for i in range(len(option.get("ab", [])))) or f"none ({option['ab_reason'].split(';')[0]})"
        lines.append(f"| `{option['flag']}` ({option['item']}) | {option['kind']} | {tests} | {identity} | "
                     f"{kld} | {ab} |")
    lines += ["", "| Step | Status | Detail |", "|---|---|---|"]
    greedy: dict[str, list[tuple[int, dict | None]]] = {}
    for step in steps:
        if step.id.startswith("greedy-master:"):
            greedy.setdefault(step.data["config"], []).append((step.data["prompt"], results.get(step.id)))
    for config_id, runs in greedy.items():  # one row per configuration, over its prompts
        counts: dict[str, int] = {}
        for _, r in runs:
            counts[r["status"] if r else "not run"] = counts.get(r["status"] if r else "not run", 0) + 1
        differ = [f"prompt {n}: {r['detail']}" for n, r in runs if r and r["status"] != "PASS"]
        status = "FAIL" if counts.get("FAIL") or counts.get("ERROR") else ("PASS" if counts.get("PASS") == len(runs)
                                                                            else "incomplete")
        lines.append(f"| greedy-master:{config_id} | {status} | " + ", ".join(f"{v} {k}" for k, v in counts.items())
                     + f" of {len(runs)} prompts" + ("; " + "; ".join(differ[:3]) if differ else "") + " |")
    for step in steps:
        if step.id.startswith(("identity:", "ab:", "greedy-master:")) or (
                re.match(r"kld:[a-z]", step.id) and "off" not in step.id):
            continue  # in the tables above
        r = results.get(step.id)
        detail = r["detail"] if r else ("needs " + ", ".join(missing_inputs(ctx, step)) if missing_inputs(ctx, step)
                                        else "not run")
        lines.append(f"| {step.id} | {r['status'] if r else 'not run'} | {detail} |")
    path = ctx.out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- main ------------------------------------------------------------------------------------------

RESULTS_ROOT = REPO / "profiles" / "bench"
DONE = ("PASS", "MEASURED", "REVIEW")  # what --resume does not run again


def default_out(resume: bool) -> Path:
    """Today's directory; with --resume, the newest earlier run (its run.json manifest), so a rerun on
    a later day continues it."""
    if resume:
        runs = sorted((p for p in RESULTS_ROOT.glob("basalt-*") if (p / "run.json").is_file()),
                      key=lambda p: (p / "run.json").stat().st_mtime)
        if runs:
            return runs[-1]
    return RESULTS_ROOT / f"basalt-{dt.date.today():%Y-%m-%d}"


def make_ctx(args: argparse.Namespace) -> Ctx:
    windows = args.platform == "windows"
    default_build = REPO / ("build-ninja" if windows else "build")
    build = Path(args.build_dir).resolve() if args.build_dir else default_build
    out = Path(args.out).resolve() if args.out else default_out(getattr(args, "resume", False))
    return Ctx(args, load_config(Path(args.options)), out, windows, build)


def finished(ctx: Ctx, step: Step) -> bool:
    previous = load_result(ctx, step.id)
    return previous is not None and previous["status"] in DONE


def selected(steps: list[Step], only: str | None, skip: str | None) -> list[Step]:
    pick = lambda text: [p.strip() for p in text.split(",") if p.strip()] if text else []  # noqa: E731
    keep = [s for s in steps if not pick(only) or any(s.id.startswith(p) for p in pick(only))]
    return [s for s in keep if not any(s.id.startswith(p) for p in pick(skip))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--action", choices=("plan", "out", "steps", "run", "summary"), default="plan",
                        help="out prints the results directory, for the wrappers to pin it for every call")
    parser.add_argument("--dry-run", action="store_true", help="the same as --action plan")
    parser.add_argument("--step", help="with --action run: the step id")
    parser.add_argument("--platform", choices=("windows", "linux"), default="windows" if os.name == "nt" else "linux")
    parser.add_argument("--build-dir", help="the build tree (default build-ninja on Windows, build on Linux)")
    parser.add_argument("--options", default=str(OPTIONS_FILE))
    parser.add_argument("--out", help="results directory (default profiles/bench/basalt-<date>)")
    parser.add_argument("--reps", type=int, default=12, help="A/B repetitions (ABBA pairs), default 12")
    parser.add_argument("--flash-next", help="Qwen3.8-Flash-Next artifact with the MTP drafter")
    parser.add_argument("--qwen27b", help="Qwen3.8-27B artifact (the DFlash2 bundle)")
    parser.add_argument("--qwen35b", help="Qwen3.6-35B-A3B artifact (optional)")
    parser.add_argument("--kld-base-flash-next", help="llama-perplexity --kl-divergence-base file for Flash-Next")
    parser.add_argument("--kld-base-27b", help="llama-perplexity --kl-divergence-base file for the 27B")
    parser.add_argument("--master-cli", help="origin/master's ninfer CLI, beside this build's under another name")
    parser.add_argument("--gguf", help="first file of the Flash-Next GGUF, for tests/convert/test_qwen4_exp_real.py")
    parser.add_argument("--iq3-pack", action="append", help="a converted IQ3_XXS or IQ3_S pack (repeatable)")
    parser.add_argument("--only", help="comma-separated step id prefixes to keep")
    parser.add_argument("--skip", help="comma-separated step id prefixes to leave out")
    parser.add_argument("--resume", action="store_true",
                        help=f"leave out steps whose result is {', '.join(DONE)}; without --out, continue the "
                             "newest run")
    args = parser.parse_args(argv)
    if args.dry_run:
        args.action = "plan"
    ctx = make_ctx(args)
    steps = selected(build_steps(ctx), args.only, args.skip)
    if args.resume:
        steps = [step for step in steps if not finished(ctx, step)]
    if args.action == "plan":
        return print_plan(ctx, steps)
    if args.action == "out":
        print(ctx.out)
        return 0
    if args.action == "steps":
        problems = input_problems(ctx)
        if problems:  # before any step waits for the lock
            print("basalt_gpu_verify: " + "; ".join(problems), file=sys.stderr)
            return 2
        ctx.out.mkdir(parents=True, exist_ok=True)
        manifest = ctx.out / "run.json"
        if not manifest.is_file():
            manifest.write_text(json.dumps({"started": dt.datetime.now().isoformat(timespec="seconds"),
                                            "argv": sys.argv[1:]}, indent=2) + "\n", encoding="utf-8")
        for step in steps:
            missing = missing_inputs(ctx, step)
            # A step that will SKIP for a missing input needs no GPU, so it does not wait for the lock.
            lock = "none" if missing else step.lock
            print(f"{step.id}\t{lock}\t{step.title}" + (f" [SKIP: needs {', '.join(missing)}]" if missing else ""))
        return 0
    if args.action == "summary":
        path = write_summary(ctx, build_steps(ctx))
        print(path.read_text(encoding="utf-8"))
        print(f"written to {path}")
        return 0
    step = next((s for s in build_steps(ctx) if s.id == args.step), None)
    if step is None:
        parser.error(f"unknown step {args.step!r} (see --action steps)")
    missing = missing_inputs(ctx, step)
    print(f"== {step.id}: {step.title}", flush=True)
    if args.resume and finished(ctx, step):
        previous = load_result(ctx, step.id) or {}
        print(f"   already {previous.get('status')} (--resume): not run again", flush=True)
        return 0
    if missing:
        outcome = result("SKIP", "needs " + ", ".join(missing))
    elif step.run is None:
        outcome = result("ERROR", "the step has no runner")
    else:
        try:
            outcome = step.run(ctx, step)
        except Exception as error:  # a broken step is reported, and the next one still runs
            outcome = result("ERROR", f"{type(error).__name__}: {error}")
    outcome["id"] = step.id
    path = result_path(ctx, step.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outcome, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"   {outcome['status']}: {outcome['detail']}", flush=True)
    return 0 if outcome["status"] in ("PASS", "SKIP", "MEASURED", "REVIEW") else 1


if __name__ == "__main__":
    sys.exit(main())
