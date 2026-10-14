"""Self-test of replay_host_arena.py on synthetic traces. Run: python -I tools/bench/test_replay_host_arena.py"""
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_host_arena as ra  # noqa: E402

UNIT = 256


def emit(lines, tag, **kw):
    lines.append("[trace] " + tag + " " + " ".join(f"{k}={v}" for k, v in kw.items()))


# 1. The failure shape the tool exists to study: a full arena of 4 state images (4 units) and 4 KV runs
#    (1 unit) allocated in alternation, then the KV runs are freed, then a state image arrives. Carving
#    everything from the low end leaves four unusable 1-unit holes; keeping KV at the high end leaves
#    one run. The expected numbers are worked out by hand, independently of the replay code.
def scenario(policy):
    cap = 20 * UNIT
    arena = ra.Arena(cap)
    place = ra.make_policy(policy, cap)
    live = []
    for _ in range(4):
        live.append(("S", place(arena, 4 * UNIT, "S"), 4 * UNIT))
        live.append(("K", place(arena, 1 * UNIT, "K"), 1 * UNIT))
    for kind, off, n in live:
        if kind == "K":
            arena.release(off, n)
    return arena.largest(), len(arena.free), place(arena, 4 * UNIT, "S") is not None


largest, runs, fits = scenario("legacy")
assert (largest, runs, fits) == (UNIT, 4, False), ("legacy", largest, runs, fits)
largest, runs, fits = scenario("twoended")
assert (largest, runs, fits) == (4 * UNIT, 1, True), ("twoended", largest, runs, fits)
print("scenario: legacy leaves 4 one-unit holes, twoended leaves one run")

# 2. Random workload under legacy placement -> trace -> replay must reproduce it exactly.
rng = random.Random(7)
cap = 4096 * UNIT
arena = ra.Arena(cap)
lines, live = [], {}
state_units, kv_units = 300, [8, 16, 24, 48]
for step in range(6000):
    t = step
    if live and rng.random() < 0.48:
        off = rng.choice(list(live))
        n = live.pop(off)
        arena.release(off, n)
        emit(lines, "R", t=t, off=off, bytes=n)
        continue
    kind = "S" if rng.random() < 0.12 else "K"
    n = (state_units if kind == "S" else rng.choice(kv_units)) * UNIT
    off = arena.alloc(n, "first")
    if off is None:
        emit(lines, "F", t=t, bytes=n, kind=kind, free=arena.free_bytes(), largest=arena.largest(),
             extents=len(arena.free), cause="geometry" if arena.free_bytes() >= n else "shortage")
    else:
        live[off] = n
        emit(lines, "A", t=t, off=off, bytes=n, kind=kind)
with tempfile.NamedTemporaryFile("w", suffix=".err", delete=False) as fh:
    fh.write("\n".join(lines) + "\n")
    path = fh.name
events = ra.parse(path)
stats, mismatch, _ = ra.replay(events, cap, "legacy")
fails = sum(s["real_failures"] for s in stats.values())
assert mismatch == 0 and fails > 0, (mismatch, fails)
assert all(s["regression"] == 0 and s["would_fit"] == 0 for s in stats.values()), dict(stats)
print(f"legacy replay reproduces a {len(events)}-event synthetic trace exactly "
      f"({fails} real failures, 0 mismatches, 0 spurious fits)")

# 3. Two-ended on the same trace should refuse state images no more often than legacy.
legacy_s = ra.replay(events, cap, "legacy")[0]["S"]["real_failures"]
two = ra.replay(events, cap, "twoended")[0]["S"]
print(f"synthetic: state failures {legacy_s}; twoended would place {two['would_fit']} of them, "
      f"{two['regression']} regressions")
os.unlink(path)

# 4. The command line refuses a trace the legacy replay cannot reproduce, and --force overrides it.
#    A trace placed two-ended (KV carved from the high end) is not what `legacy` would have done.
arena = ra.Arena(cap)
place = ra.make_policy("twoended", cap)
lines, live = [], {}
for step in range(400):
    kind = "S" if step % 5 == 0 else "K"
    n = (state_units if kind == "S" else 8) * UNIT
    off = place(arena, n, kind)
    if off is not None:
        live[off] = n
        emit(lines, "A", t=step, off=off, bytes=n, kind=kind)
    if step % 3 == 2 and live:
        victim = sorted(live)[len(live) // 2]
        arena.release(victim, live.pop(victim))
        emit(lines, "R", t=step, off=victim, bytes=n)
with tempfile.NamedTemporaryFile("w", suffix=".err", delete=False) as fh:
    fh.write("\n".join(lines) + "\n")
    two_ended_trace = fh.name
tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_host_arena.py")
refused = subprocess.run([sys.executable, "-I", tool, two_ended_trace, "--capacity-mib",
                          str(cap // (1024 * 1024) or 1)], capture_output=True, text=True)
assert refused.returncode != 0 and "refusing" in refused.stderr, (refused.returncode, refused.stderr)
forced = subprocess.run([sys.executable, "-I", tool, two_ended_trace, "--capacity-mib",
                         str(cap // (1024 * 1024) or 1), "--force", "--policies", "legacy"],
                        capture_output=True, text=True)
assert forced.returncode == 0 and "DOES NOT REPRODUCE" in forced.stdout, (forced.returncode, forced.stdout)
os.unlink(two_ended_trace)
print("a trace the legacy replay cannot reproduce is refused, and --force reports it")
print("OK")
