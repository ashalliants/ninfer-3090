"""Self-test of run_interleaved_ab.py against a fake executable. Run: python -I tools/bench/test_run_interleaved_ab.py

The fake is a small Python script behind an executable wrapper (a shebang file on POSIX, a .cmd on
Windows). It prints `decode speed X tok/s`, 100 normally and 110 when its arguments hold `--fast`,
so same-build mode (one executable, --a-args / --b-args) must show B about 10% ahead of A; it also
echoes its arguments so the per-run logs can be checked. Pure Python, no GPU.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
TOOL = HERE / "run_interleaved_ab.py"
failures = []


def check(name, condition, detail=""):
    if not condition:
        failures.append(f"{name}: {detail}")


FAKE = r'''
import sys
args = sys.argv[1:]
print("args " + " ".join(args))
if "--crash" in args:
    sys.exit(4)
print("summary decode speed %.1f tok/s" % (110.0 if "--fast" in args else 100.0), file=sys.stderr)
'''


def make_fake(directory, name):
    script = directory / (name + ".py")
    script.write_text(FAKE, encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / (name + ".cmd")
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = directory / name
        wrapper.write_text(f"#!{sys.executable}\nimport runpy, sys\n"
                           f"sys.argv[0] = {str(script)!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n",
                           encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def run(*argv):
    return subprocess.run([sys.executable, "-I", str(TOOL), *map(str, argv)], capture_output=True, text=True)


with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    fake = make_fake(root, "fake")
    logs, report = root / "logs", root / "ab.json"

    done = run("--arm-a", fake, "--reps", 4, "--b-args=--fast", "--config", "decode:{exe} --job x",
               "--log-dir", logs, "--json", report, "--quiet")
    check("same-build mode exits 0", done.returncode == 0, done.stdout + done.stderr)
    data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {"configs": [{}]}
    result = data["configs"][0]
    check("every repetition makes a pair", result.get("pairs") == 4, str(result))
    check("B (--fast) is 10% ahead in every pair", abs(result.get("median", 0) - 1.1) < 1e-9
          and result.get("positive") == 4, str(result))
    check("the JSON names the arm arguments", data.get("a_args") == [] and data.get("b_args") == ["--fast"], str(data))
    log_b = logs / "decode-B-rep1.log"
    log_a = logs / "decode-A-rep2.log"
    check("each run is logged", log_b.is_file() and log_a.is_file(), str(sorted(p.name for p in logs.glob("*"))))
    if log_b.is_file() and log_a.is_file():
        check("arm B ran with --b-args appended", "args --job x --fast" in log_b.read_text(encoding="utf-8"))
        check("arm A ran without them", "args --job x\n" in log_a.read_text(encoding="utf-8"))
    check("the arms' arguments are printed", "+ --fast" in done.stdout, done.stdout)

    done = run("--arm-a", fake, "--reps", 2, "--a-args=--x", "--b-args=--x", "--config", "d:{exe}")
    check("same-build mode refuses equal arms", done.returncode != 0 and "differ" in done.stderr, done.stderr)

    done = run("--arm-a", fake, "--reps", 2, "--config", "d:{exe}")
    check("no --arm-b and no arm arguments is refused", done.returncode != 0, done.stdout + done.stderr)

    done = run("--arm-a", fake, "--reps", 2, "--b-args=--crash", "--config", "d:{exe}", "--quiet")
    check("a failing arm makes exit 1", done.returncode == 1, done.stdout + done.stderr)

    second = make_fake(root, "fake2")
    done = run("--arm-a", fake, "--arm-b", second, "--reps", 2, "--config", "d:{exe} --job y", "--quiet")
    check("two-build mode still works", done.returncode == 0 and "+0.00%" in done.stdout, done.stdout + done.stderr)

if failures:
    print(f"{len(failures)} check(s) failed:")
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("run_interleaved_ab self-test: all checks passed")
