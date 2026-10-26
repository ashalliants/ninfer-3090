"""Self-test of check_stderr_logging.py. Run: python -I tools/test_check_stderr_logging.py

Builds temporary source trees (no git, so the tool walks src/) and checks the ratchet: a count that
rises or a new file fails, a lower count passes and --update lowers the allowlist, and comments and
string literals are not counted. Pure Python, no GPU.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(HERE))
import check_stderr_logging as tool  # noqa: E402

TOOL = HERE / "check_stderr_logging.py"
failures = []


def check(name, condition, detail=""):
    if not condition:
        failures.append(f"{name}: {detail}")


# 1. What counts as a use. Four real uses; everything else in the snippet is a comment, a string, or a
#    different identifier.
SNIPPET = r'''
// stderr in a line comment, std::cerr << "no";
// a continued comment \
   std::cerr << "still the comment";
/* stderr in a block
   comment spanning lines, perror("x") */
const char* a = "fprintf(stderr, \"escaped\") // not a comment";
const char* b = R"raw(stderr "quoted" )" still raw)raw";
char c = '"'; int big = 1'000'000; int hex = 0xFF'FF;
int d = ::isatty(STDERR_FILENO); spdlog::sinks::stderr_color_sink_st sink;
std::fprintf(stderr, "%d", 1);        // use 1
std::cerr << "x";                      // use 2
perror("p");                           // use 3
std::fflush(stderr);                   // use 4
'''
check("snippet count", tool.count_uses(SNIPPET) == 4, str(tool.count_uses(SNIPPET)))
check("clog and bare cerr", tool.count_uses("std::clog << 1; cerr << 2;") == 2)
check("perror needs a call", tool.count_uses("int perror_count = 0; // perror") == 0)
check("code after a closed block comment", tool.count_uses("/* a */ std::fputs(\"x\", stderr);") == 1)
check("unterminated string ends at the line", tool.count_uses('"abc\nstd::cerr << 1;') == 1)

# 2. The ratchet, on a temporary tree.
ONE = "void f() { std::fprintf(stderr, \"x\"); }\n"
TWO = "void f() { std::fprintf(stderr, \"x\"); std::cerr << 1; }\n"
QUIET = "void f() { /* stderr */ log(\"stderr\"); }\n"


def run(root, *flags):
    return subprocess.run([sys.executable, "-I", str(TOOL), "--root", str(root), *flags],
                          capture_output=True, text=True)


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def allow(root, text):
    write(root, "tools/stderr_logging.allow", text)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    write(root, "src/core/a.cpp", TWO)
    write(root, "src/core/b.cu", ONE)
    write(root, "src/core/quiet.cpp", QUIET)
    write(root, "src/core/notes.txt", "std::cerr\n")          # not a source file
    write(root, "apps/cli/main.cpp", TWO)                      # outside src/
    header = "# header\n# second line\n"
    allow(root, header + "2 src/core/a.cpp\n1 src/core/b.cu\n")

    result = run(root)
    check("at the allowlist passes", result.returncode == 0, result.stdout + result.stderr)
    check("summary printed", "ok (2 files, 3 uses)" in result.stdout, result.stdout)

    counts = run(root, "--counts")
    check("--counts format", counts.stdout == "2 src/core/a.cpp\n1 src/core/b.cu\n", counts.stdout)

    # A count that rises fails and names the file.
    write(root, "src/core/b.cu", TWO)
    result = run(root)
    check("a rise fails", result.returncode == 1, result.stdout + result.stderr)
    check("the rise is named", "src/core/b.cu: names stderr 2 times, allowed 1" in result.stderr, result.stderr)
    before = (root / "tools/stderr_logging.allow").read_text()
    result = run(root, "--update")
    check("--update refuses a rise", result.returncode == 1, result.stdout + result.stderr)
    check("--update leaves the allowlist alone on a rise",
          (root / "tools/stderr_logging.allow").read_text() == before)
    write(root, "src/core/b.cu", ONE)

    # A new file fails.
    write(root, "src/serve/new.cpp", ONE)
    result = run(root)
    check("a new file fails", result.returncode == 1, result.stdout + result.stderr)
    check("the new file is named", "src/serve/new.cpp" in result.stderr and "not in the allowlist" in result.stderr,
          result.stderr)
    (root / "src/serve/new.cpp").unlink()
    result = run(root)
    check("back to passing", result.returncode == 0, result.stdout + result.stderr)

    # A count that goes down passes, with a note; --update lowers the allowlist and drops a file at zero.
    write(root, "src/core/a.cpp", ONE)
    write(root, "src/core/b.cu", QUIET)
    result = run(root)
    check("a drop passes", result.returncode == 0, result.stdout + result.stderr)
    check("the drop is noted", "src/core/a.cpp: down to 1" in result.stdout
          and "src/core/b.cu: no longer names stderr" in result.stdout, result.stdout)
    check("a drop does not touch the allowlist", (root / "tools/stderr_logging.allow").read_text() == before)
    result = run(root, "--update")
    check("--update succeeds", result.returncode == 0, result.stdout + result.stderr)
    updated = (root / "tools/stderr_logging.allow").read_text()
    check("--update lowers and keeps the header", updated == header + "1 src/core/a.cpp\n", repr(updated))
    result = run(root)
    check("passes after --update", result.returncode == 0, result.stdout + result.stderr)

    # After the update the lowered count is the new ceiling.
    write(root, "src/core/a.cpp", TWO)
    check("the ceiling came down", run(root).returncode == 1)

    # An allowlist line that is not "<count> <path>" is refused.
    allow(root, "two src/core/a.cpp\n")
    result = run(root)
    check("a malformed allowlist is refused", result.returncode == 1 and "expected '<count> <path>'" in result.stderr,
          result.stdout + result.stderr)

if failures:
    print("FAIL")
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("check_stderr_logging self-test ok")
