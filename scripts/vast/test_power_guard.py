"""Tests for scripts/vast/power_guard.py. Pure Python, no GPU and no Vast account.

The inputs are nvidia-smi output captured from the Vast box the perf model was validated on (a 3090
limited to 220 W of its 350 W default) and the uncapped case.
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import power_guard  # noqa: E402

CAPPED = "220.00, 350.00\n"  # the v0.17 benchmark box
FULL = "350.00, 350.00\n"
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL:", message)


def run(text, min_pct):
    return power_guard.check(text, min_pct)


# The box that produced the 54-57% numbers fails every threshold above its 63%, including the 80% floor.
code, message = run(CAPPED, 80)
check(code == 1 and "220 W" in message and "(63%)" in message, "a 220 W limit passed an 80%% floor: %r" % message)
check(run(CAPPED, 100)[0] == 1, "a 220 W limit passed the 100% default")
check(run(CAPPED, 60)[0] == 0, "a 220 W (63%) limit failed a 60% floor")

# A full-power host passes at the default, and a 349 W reading of 350 W rounds up to 100%.
check(run(FULL, 100)[0] == 0, "a 350 W limit failed 100%")
check(run("349.00, 350.00\n", 100)[0] == 0, "349 W of 350 W failed 100%")
check(run("340.00, 350.00\n", 100)[0] == 1, "340 W of 350 W (97%) passed 100%")
check(run("340.00, 350.00\n", 95)[0] == 0, "340 W of 350 W (97%) failed 95%")

# 80% is the lowest the team accepts: 280 W of 350 W is exactly on it.
check(run("280.00, 350.00\n", 80)[0] == 0, "280 W of 350 W failed the 80% floor")
check(run("279.00, 350.00\n", 80)[0] == 0, "279 W of 350 W (79.7%, rounds to 80%) failed the 80% floor")
check(run("277.00, 350.00\n", 80)[0] == 1, "277 W of 350 W (79%) passed the 80% floor")

# A card whose limit is above its default (partner cards) is fine.
check(run("390.00, 350.00\n", 100)[0] == 0, "a limit above the default failed")

# Unreadable output is a failure unless the check is off; that is the point of the guard.
for bad in ("", "[N/A], [N/A]\n", "[Not Supported], 350.00\n", "NVIDIA-SMI has failed\n", "0.00, 350.00\n", "220.00\n"):
    check(run(bad, 100)[0] == 2, "unreadable output %r did not fail closed" % bad)
    check(run(bad, 0)[0] == 0, "--min-pct 0 did not turn the check off for %r" % bad)

# Extra lines (a banner, a second GPU's row after the first) do not hide a readable row.
check(power_guard.parse_limits("some warning\n220.00, 350.00\n") == (220.0, 350.0), "a banner line hid the reading")

# The command line: exit code and the one printed line, as vast-up.sh uses them.
script = os.path.join(HERE, "power_guard.py")
for text, args, want in ((CAPPED, [], 1), (FULL, [], 0), (CAPPED, ["--min-pct", "0"], 0), ("junk", [], 2)):
    done = subprocess.run([sys.executable, "-I", script] + args, input=text, capture_output=True, text=True)
    check(done.returncode == want, "cli %r %r exited %d, wanted %d" % (text, args, done.returncode, want))
    check(len(done.stdout.strip().splitlines()) == 1, "cli printed %r, wanted one line" % done.stdout)
check(subprocess.run([sys.executable, "-I", script, "--min-pct", "101"], input=FULL, capture_output=True,
                     text=True).returncode == 2, "an out-of-range --min-pct was accepted")

if FAILURES:
    print("%d failure(s)" % len(FAILURES))
    sys.exit(1)
print("ok")
