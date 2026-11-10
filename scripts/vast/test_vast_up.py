"""Tests for scripts/vast/vast-up.sh's rent / check / destroy loop, against a fake `vastai` and a fake ssh.

vast-up.sh bills by the minute, so what it must get right is the lifecycle: a rejected host is destroyed,
the next candidate is tried, a run that finds no good host leaves nothing billing and no state file, and a
destroy that fails leaves the instance tracked (so vast-destroy.sh can retry) without renting another.
Nothing here touches Vast or a GPU; it needs bash and git, so it is registered on POSIX hosts only.

The fakes are written into a temporary directory:
  vastai          offers 1001-1003 in price order; `create` hands out instance ids 1001, 1002, ...;
                  `show instance` (FAKE_BOOT, comma-separated) and the card's power (FAKE_POWER,
                  semicolon-separated, since a reading is "limit, default") come from the FAKE_* variables,
                  one entry per instance in the order rented; every call is appended to calls.log
  vast-dev-ssh.sh answers the power query from FAKE_POWER and succeeds at everything else
  sleep           returns at once, so the 8-minute boot wait costs nothing
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FAILURES = []

FAKE_VASTAI = r'''#!/usr/bin/env python3
import json, os, sys
root = os.environ["FAKE_ROOT"]
args = sys.argv[1:]
with open(os.path.join(root, "calls.log"), "a") as log:
    log.write(" ".join(args) + "\n")
def offer(i, price):
    return {"id": i, "dph_base": price, "storage_cost": 0.1, "inet_down_cost": 0.01, "inet_up_cost": 0.01,
            "cpu_cores_effective": 24, "geolocation": "Test", "inet_down": 2000, "gpu_max_power": 350.0}
if args[:2] == ["search", "offers"]:
    print(json.dumps([] if os.environ.get("FAKE_NO_OFFERS") else [offer(1001, 0.20), offer(1002, 0.25), offer(1003, 0.30)]))
elif args[:2] == ["create", "instance"]:
    if os.environ.get("FAKE_CREATE_FAIL"):
        sys.exit(1)
    counter = os.path.join(root, "counter")
    n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1001
    open(counter, "w").write(str(n))
    print(json.dumps({"new_contract": n}))
elif args[:2] == ["show", "instance"]:
    boots = os.environ.get("FAKE_BOOT", "").split(",")
    index = int(args[2]) - 1001
    print(json.dumps({"actual_status": boots[index] if index < len(boots) and boots[index] else "running"}))
elif args[:2] == ["destroy", "instance"]:
    sys.stdin.read()
    if os.environ.get("FAKE_DESTROY_FAIL"):
        sys.exit(1)
'''

FAKE_SSH = r'''#!/usr/bin/env python3
import os, sys
root = os.environ["FAKE_ROOT"]
command = " ".join(sys.argv[1:])
if "power.limit" in command:
    instance = int(open(os.path.join(os.environ["NINFER_VAST_STATE"], "instance_id")).read())
    readings = os.environ.get("FAKE_POWER", "").split(";")
    index = instance - 1001
    print(readings[index] if index < len(readings) else readings[-1])
elif not sys.stdin.isatty():
    sys.stdin.read()
'''


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print("FAIL:", message)


def run(name, env=None, extra=None):
    """Run vast-up.sh in a fresh sandbox; returns (exit code, output, calls, state file contents or None, leftovers)."""
    sandbox = tempfile.mkdtemp(prefix="vast-up-test-")
    try:
        work, bin_dir, state, tmp = (os.path.join(sandbox, d) for d in ("work", "bin", "state", "tmp"))
        for d in (work, bin_dir, state, tmp):
            os.mkdir(d)
        for f in ("vast-up.sh", "vast-common.sh", "power_guard.py"):
            shutil.copy(os.path.join(HERE, f), work)
        for path, text in ((os.path.join(bin_dir, "vastai"), FAKE_VASTAI),
                           (os.path.join(work, "vast-dev-ssh.sh"), FAKE_SSH),
                           (os.path.join(bin_dir, "sleep"), "#!/bin/sh\nexit 0\n")):
            with open(path, "w") as f:
                f.write(text)
            os.chmod(path, 0o755)
        key = os.path.join(sandbox, "key")
        open(key, "w").close()
        full = dict(os.environ, FAKE_ROOT=sandbox, NINFER_VAST_STATE=state, NINFER_VAST_KEY=key, TMPDIR=tmp,
                    PATH=bin_dir + os.pathsep + os.environ["PATH"], FAKE_POWER="350.00, 350.00")
        for var in [v for v in full if v.startswith("NINFER_VAST_") and v not in ("NINFER_VAST_STATE", "NINFER_VAST_KEY")]:
            del full[var]
        full.update(env or {})
        done = subprocess.run(["bash", os.path.join(work, "vast-up.sh"), REPO, "1"] + (extra or []), env=full,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        calls_path = os.path.join(sandbox, "calls.log")
        calls = open(calls_path).read().splitlines() if os.path.exists(calls_path) else []
        state_path = os.path.join(state, "instance_id")
        state_text = open(state_path).read().strip() if os.path.exists(state_path) else None
        return done.returncode, done.stdout + done.stderr, calls, state_text, os.listdir(tmp)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def ids(calls, verb):
    """Instance or offer ids the fake saw for `create instance` (offer ids) or `destroy instance`."""
    return [c.split()[2] for c in calls if c.startswith(verb + " instance")]


CAPPED, FULL = "220.00, 350.00", "350.00, 350.00"

# 1. The first host is capped, the second is not: the first is destroyed and the second kept.
rc, out, calls, state, tmpfiles = run("capped-then-full", {"FAKE_POWER": CAPPED + ";" + FULL})
check(rc == 0, "capped-then-full failed (%d): %s" % (rc, out[-400:]))
check(ids(calls, "create") == ["1001", "1002"], "capped-then-full rented %r" % ids(calls, "create"))
check(ids(calls, "destroy") == ["1001"], "capped-then-full destroyed %r, wanted only the capped host" % ids(calls, "destroy"))
check(state == "1002", "capped-then-full left state %r, wanted the kept host" % state)
check("63%" in out and "rejected" in out, "capped-then-full did not say why it rejected the host: %s" % out[-400:])
check(tmpfiles == [], "capped-then-full left temporary files %r" % tmpfiles)
check(any(c.startswith("search offers") and "gpu_max_power>=350" in c for c in calls),
      "the search did not carry gpu_max_power>=350: %r" % [c for c in calls if c.startswith("search")])

# 2. Every host is capped: all are destroyed, the run fails, and nothing is left tracked.
rc, out, calls, state, tmpfiles = run("all-capped", {"FAKE_POWER": CAPPED})
check(rc != 0, "all-capped exited 0")
check(ids(calls, "create") == ["1001", "1002", "1003"], "all-capped rented %r" % ids(calls, "create"))
check(ids(calls, "destroy") == ["1001", "1002", "1003"], "all-capped destroyed %r, every rented host must go" % ids(calls, "destroy"))
check(state is None, "all-capped left state %r, wanted none" % state)
check("None of the 3 offers" in out, "all-capped did not report how many it tried: %s" % out[-300:])
check(tmpfiles == [], "all-capped left temporary files %r" % tmpfiles)

# 3. NINFER_VAST_MAX_HOSTS bounds how many are rented.
rc, out, calls, state, _ = run("max-hosts", {"FAKE_POWER": CAPPED, "NINFER_VAST_MAX_HOSTS": "2"})
check(rc != 0 and len(ids(calls, "create")) == 2 and state is None, "max-hosts 2 rented %r, state %r" % (ids(calls, "create"), state))

# 4. A host that never leaves "loading" is destroyed and the next one tried.
rc, out, calls, state, _ = run("boot-fail", {"FAKE_BOOT": "loading,running"})
check(rc == 0 and ids(calls, "destroy") == ["1001"] and state == "1002", "boot-fail: rc %d, destroyed %r, state %r" % (rc, ids(calls, "destroy"), state))

# 5. A power reading that cannot be parsed fails closed; turning the check off accepts the host.
rc, out, calls, state, _ = run("unreadable", {"FAKE_POWER": "[N/A], [N/A]"})
check(rc != 0 and state is None and ids(calls, "destroy") == ["1001", "1002", "1003"], "unreadable did not fail closed: rc %d, state %r" % (rc, state))
rc, out, calls, state, _ = run("check-off", {"FAKE_POWER": CAPPED, "NINFER_VAST_MIN_POWER_PCT": "0"})
check(rc == 0 and state == "1001" and not ids(calls, "destroy"), "check-off: rc %d, state %r, destroyed %r" % (rc, state, ids(calls, "destroy")))
check(not any("gpu_max_power" in c for c in calls if c.startswith("search")), "check-off still filtered the search by power")

# 6. The floor is a percentage of the default: 63% passes 60 and fails 80; the search filter follows it.
rc, out, calls, state, _ = run("floor-60", {"FAKE_POWER": CAPPED, "NINFER_VAST_MIN_POWER_PCT": "60"})
check(rc == 0 and state == "1001", "a 63%% card failed a 60%% floor: rc %d, state %r" % (rc, state))
rc, out, calls, state, _ = run("floor-80", {"FAKE_POWER": CAPPED + ";" + FULL, "NINFER_VAST_MIN_POWER_PCT": "80"})
check(rc == 0 and state == "1002", "a 63%% card passed an 80%% floor: rc %d, state %r" % (rc, state))
check(any("gpu_max_power>=280" in c for c in calls if c.startswith("search")), "the 80%% floor did not make the search ask for 280 W")

# 7. A leading zero is decimal ("080" is 80, not an invalid octal number).
rc, out, calls, state, _ = run("leading-zero", {"FAKE_POWER": FULL, "NINFER_VAST_MIN_POWER_PCT": "080"})
check(rc == 0 and state == "1001", "NINFER_VAST_MIN_POWER_PCT=080 was refused: rc %d: %s" % (rc, out[-300:]))
check(any("gpu_max_power>=280" in c for c in calls if c.startswith("search")), "080 did not become the 80%% filter")

# 8. Bad settings are refused before anything is rented.
for var, value in (("NINFER_VAST_MIN_POWER_PCT", "150"), ("NINFER_VAST_MIN_POWER_PCT", "abc"),
                   ("NINFER_VAST_MIN_POWER_PCT", "1000"), ("NINFER_VAST_MAX_HOSTS", "0"), ("NINFER_VAST_MAX_HOSTS", "x")):
    rc, out, calls, state, tmpfiles = run("bad-" + value, {var: value})
    check(rc != 0 and not ids(calls, "create") and state is None, "%s=%s was not refused up front (rc %d)" % (var, value, rc))
    check("must be" in out, "%s=%s gave no explanation: %s" % (var, value, out[-200:]))

# 9. No offers at all: a clear message, nothing rented.
rc, out, calls, state, _ = run("no-offers", {"FAKE_NO_OFFERS": "1"})
check(rc != 0 and not ids(calls, "create") and "No suitable offer" in out and "gpu_max_power" in out,
      "an empty search was not explained: rc %d: %s" % (rc, out[-300:]))

# 10. A destroy that fails keeps the instance tracked for vast-destroy.sh and rents nothing more.
rc, out, calls, state, _ = run("destroy-fails", {"FAKE_POWER": CAPPED, "FAKE_DESTROY_FAIL": "1"})
check(rc != 0, "a failed destroy exited 0")
check(state == "1001", "a failed destroy left state %r, wanted the instance still tracked" % state)
check(ids(calls, "create") == ["1001"], "rented %r after a destroy failed, while an instance was still billing" % ids(calls, "create"))
check("Destroy FAILED" in out, "a failed destroy was not reported: %s" % out[-300:])

# 11. A rental that fails outright leaves no state and no temporary file.
rc, out, calls, state, tmpfiles = run("create-fails", {"FAKE_CREATE_FAIL": "1"})
check(rc != 0 and state is None, "a failed create: rc %d, state %r" % (rc, state))
check(tmpfiles == [], "a failed create left the on-start script behind: %r" % tmpfiles)

if FAILURES:
    print("%d failure(s)" % len(FAILURES))
    sys.exit(1)
print("ok")
