"""Self-test of greedy_identity.py against fake ninfer CLIs. Run: python -I tools/bench/test_greedy_identity.py

The fakes are small Python scripts behind an executable wrapper (a shebang file on POSIX, a .cmd on
Windows). They check the arguments the tool passes, then print the CLI's `tokens  generated ids`
line on stderr with ids derived from the prompt, so a build can be made to agree, to diverge at a
chosen token, to stop early, to crash, or to print nothing. Pure Python, no GPU.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(HERE))
import greedy_identity as gi  # noqa: E402

TOOL = HERE / "greedy_identity.py"
failures = []


def check(name, condition, detail=""):
    if not condition:
        failures.append(f"{name}: {detail}")


FAKE = r'''
import sys
argv = sys.argv[1:]
model, rest = argv[0], argv[1:]
BEHAVIOUR = %(behaviour)r
want = ["--max-new", "7", "--greedy", "--print-token-ids", "--foo", "bar"]
for flag in want:
    if flag not in rest:
        print("fake: missing " + flag + " in " + repr(rest), file=sys.stderr)
        sys.exit(3)
if model != "m.ninfer" or rest[rest.index("--max-new") + 1] != "7":
    print("fake: bad model or --max-new", file=sys.stderr)
    sys.exit(3)
prompt = rest[rest.index("--prompt") + 1]
ids = [(sum(map(ord, prompt)) * (i + 7)) %% 1000 for i in range(7)]
kind = BEHAVIOUR.get(prompt, "same")
if kind == "crash":
    print("CUDA error: out of memory", file=sys.stderr)
    sys.exit(5)
if kind.startswith("flip"):
    at = int(kind[4:])
    ids[at] += 1
if kind.startswith("short"):
    ids = ids[:int(kind[5:])]
if "--flip-me" in rest:
    ids[2] += 1
print("the answer text goes to stdout")
print("phase       detail                      elapsed/progress", file=sys.stderr)
if kind != "silent":
    print("tokens      generated ids             " + " ".join(map(str, ids)), file=sys.stderr)
print("summary decode speed 100.0 tok/s", file=sys.stderr)
'''


def make_fake(directory, name, behaviour):
    script = directory / (name + ".py")
    script.write_text(FAKE % {"behaviour": behaviour}, encoding="utf-8")
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


def run(a, b, prompts, *flags):
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), "--a", str(a), "--b", str(b), "--model", "m.ninfer",
         "--prompts", str(prompts), "--tokens", "7", "--extra-args=--foo bar", *flags],
        capture_output=True, text=True, timeout=120)


# 0. Parsing the ids line the way apps/cli/main.cpp prints it.
line = "phase  x\ntokens      " + "generated ids".ljust(26) + "1 22 333\nsummary y\n"
check("parse ids", gi.parse_ids(line) == [1, 22, 333], str(gi.parse_ids(line)))
check("parse empty ids", gi.parse_ids("tokens      " + "generated ids".ljust(26) + "\n") == [])
check("parse no line", gi.parse_ids("summary y\n") is None)
check("first difference", gi.first_difference([1, 2, 3], [1, 2, 4]) == 2
      and gi.first_difference([1, 2], [1, 2, 3]) == 2 and gi.first_difference([1, 2], [1, 2]) is None)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    prompts = root / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps({"prompt": p}) for p in ("alpha", "beta", "gamma")) + "\n\n",
                       encoding="utf-8")

    def pair(subdir, behaviour_a, behaviour_b):
        directory = root / subdir
        directory.mkdir()
        return make_fake(directory, "ninfer-a", behaviour_a), make_fake(directory, "ninfer-b", behaviour_b)

    # 1. Identical builds.
    a, b = pair("same", {}, {})
    result = run(a, b, prompts)
    check("identical exits 0", result.returncode == 0, result.stdout + result.stderr)
    check("identical prints GREEDY IDENTICAL", "GREEDY IDENTICAL (3 prompts, 21 tokens)" in result.stdout, result.stdout)

    # 2. B diverges at token 3 of the second prompt: exit 1, names the prompt and the index, and stops there.
    a, b = pair("flip", {}, {"beta": "flip3"})
    result = run(a, b, prompts)
    check("differing exits 1", result.returncode == 1, result.stdout + result.stderr)
    check("differing names prompt 2 and token 3",
          "GREEDY DIFFERS: prompt 2/3 ('beta'), token index 3" in result.stdout, result.stdout)
    check("differing stops at the first", "prompt 3/3" not in result.stdout, result.stdout)
    check("differing does not claim identity", "GREEDY IDENTICAL" not in result.stdout, result.stdout)

    # 3. B stops early: the first difference is where the shorter one ends.
    a, b = pair("short", {}, {"alpha": "short4"})
    result = run(a, b, prompts)
    check("early stop exits 1", result.returncode == 1, result.stdout + result.stderr)
    check("early stop index", "prompt 1/3 ('alpha'), token index 4" in result.stdout and "B [ended]" in result.stdout,
          result.stdout)

    # 4. A crash, and a build that prints no ids line, compare nothing: exit 2, naming the build.
    a, b = pair("crash", {}, {"gamma": "crash"})
    result = run(a, b, prompts)
    check("crash exits 2", result.returncode == 2, result.stdout + result.stderr)
    check("crash names the build and its message",
          "build B" in result.stderr and "exited 5" in result.stderr and "out of memory" in result.stderr,
          result.stderr)
    check("crash is not identical", "GREEDY IDENTICAL" not in result.stdout)
    a, b = pair("silent", {"alpha": "silent"}, {})
    result = run(a, b, prompts)
    check("no ids line exits 2", result.returncode == 2 and "build A" in result.stderr
          and "generated ids" in result.stderr, result.stdout + result.stderr)

    # 5. The flags reach both builds (the fakes exit 3 without them), so a missing --extra-args fails loudly.
    a, b = pair("flags", {}, {})
    result = subprocess.run([sys.executable, "-I", str(TOOL), "--a", str(a), "--b", str(b), "--model", "m.ninfer",
                             "--prompts", str(prompts), "--tokens", "7"], capture_output=True, text=True, timeout=120)
    check("a build missing a flag is a failure", result.returncode == 2 and "exited 3" in result.stderr,
          result.stdout + result.stderr)

    # 6. --limit.
    a, b = pair("limit", {}, {"gamma": "flip1"})
    result = run(a, b, prompts, "--limit", "2")
    check("--limit skips the differing prompt", result.returncode == 0 and "(2 prompts, 14 tokens)" in result.stdout,
          result.stdout + result.stderr)

    # 7. Preflight and input errors exit 2 before any run.
    other = root / "other"
    other.mkdir()
    stray = make_fake(other, "ninfer-b", {})
    a, b = pair("preflight", {}, {})
    result = run(a, stray, prompts)
    check("different directories refused", result.returncode == 2 and "different directories" in result.stderr,
          result.stdout + result.stderr)
    result = run(a, a, prompts)
    check("same file refused", result.returncode == 2 and "same file" in result.stderr, result.stderr)
    result = run(a, root / "missing", prompts)
    check("missing executable refused", result.returncode == 2 and "missing executable" in result.stderr, result.stderr)
    result = run(a, b, root / "no-such.jsonl")
    check("missing prompts refused", result.returncode == 2 and "missing prompts file" in result.stderr, result.stderr)
    bad = root / "bad.jsonl"
    bad.write_text('{"text": "x"}\n', encoding="utf-8")
    result = run(a, b, bad)
    check("a prompt line without \"prompt\" refused", result.returncode == 2 and "bad.jsonl:1" in result.stderr,
          result.stderr)

    # 8. Same-build mode: one executable, per-side arguments.
    same = root / "single"
    same.mkdir()
    exe = make_fake(same, "ninfer", {})

    def run_same(*flags):
        return subprocess.run([sys.executable, "-I", str(TOOL), "--a", str(exe), "--model", "m.ninfer",
                               "--prompts", str(prompts), "--tokens", "7", "--extra-args=--foo bar", *flags],
                              capture_output=True, text=True, timeout=120)

    result = run_same("--b-args=--harmless")
    check("same-build identical exits 0", result.returncode == 0 and "GREEDY IDENTICAL" in result.stdout,
          result.stdout + result.stderr)
    result = run_same("--b-args=--flip-me")
    check("same-build B args reach B only",
          result.returncode == 1 and "token index 2" in result.stdout, result.stdout + result.stderr)
    result = run_same("--a-args=--flip-me", "--b-args=--flip-me")
    check("same-build equal args refused", result.returncode == 2 and "nothing to compare" in result.stderr,
          result.stdout + result.stderr)
    result = run_same()
    check("same-build without per-side args refused", result.returncode == 2 and "nothing to compare" in result.stderr,
          result.stdout + result.stderr)
    a, b = pair("sided", {}, {})
    result = run(a, b, prompts, "--a-args=--flip-me")
    check("--a-args reaches A only with two builds", result.returncode == 1 and "token index 2" in result.stdout,
          result.stdout + result.stderr)

# 9. The default prompts file is the committed one.
check("default prompts exist", gi.DEFAULT_PROMPTS.is_file(), str(gi.DEFAULT_PROMPTS))
if gi.DEFAULT_PROMPTS.is_file():
    check("default prompts load", len(gi.load_prompts(gi.DEFAULT_PROMPTS, None)) == 24)

if failures:
    print("FAIL")
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("greedy_identity self-test ok")
