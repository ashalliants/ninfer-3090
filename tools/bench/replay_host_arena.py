#!/usr/bin/env python3
"""Replays a Host-arena allocation trace under different placement policies, to predict how many
state-image and KV allocations each policy would have refused on the same workload, without a GPU
and without re-running the server.

    python tools/bench/replay_host_arena.py server.err --capacity-mib 8192 \
        [--policies legacy,twoended,bestfit,twoended_best,region30,region50,region70] [--force]

The trace is `[trace]` lines on a measurement build's stderr (HostContextArena, src/core/
host_context_arena.cpp, printed from allocate / release / split). The shipped server does not print
them; tools/bench/README.md ("Host arena allocation replay") shows the few lines to add. Kind S is
a state image, K is a KV run; `bytes` are the 256-aligned sizes the arena actually carved:

    [trace] A t= off= bytes= kind=     allocation made
    [trace] F t= bytes= kind= ...      allocation refused (a real failure)
    [trace] R t= off= bytes=           release
    [trace] P t= off= left= right=     one allocation split in two

How the replay is scored (an approximation, stated so nobody reads more into it):
  * An A event is replayed as an allocation under the policy. If the policy cannot place it, that is
    a REGRESSION (the real run placed it) and it is simply absent from the replayed live set.
  * An F event is replayed as a PROBE: would the policy have placed it right now? The probe is not
    kept, so the live set stays the real run's. A policy that fixes failures would in reality change
    what the engine does next (it would keep the demoted data); that feedback is not modelled.
  * `legacy` (first-fit by address, what the arena did when this was written) must reproduce the real
    run exactly: same offsets, every real failure refused. The script checks this first and exits
    without reporting the other policies if it does not, because numbers from a trace the baseline
    cannot reproduce are not evidence. `--force` reports anyway (e.g. a trace taken under another
    placement), at your own risk.

Measured use (2026-10): on a hosted-27B trace at 8 GiB the legacy replay reproduced the run exactly
(0 offset mismatches); best-fit within each end of the arena would have placed 75% of the refused
state images with no regressions, plain two-ended 28%, and fixed regions starved KV. A live A/B of
plain two-ended placement then showed no clear change, so treat a replay as a hypothesis generator,
not a result. See issue #304.
"""
import argparse
import bisect
import collections
import re
import sys

ALIGN = 256

A_RE = re.compile(r"\[trace\] A t=(\d+) off=(\d+) bytes=(\d+) kind=(\w)")
F_RE = re.compile(r"\[trace\] F t=(\d+) bytes=(\d+) kind=(\w) free=(\d+) largest=(\d+) extents=(\d+) (\w+)")
R_RE = re.compile(r"\[trace\] R t=(\d+) off=(\d+) bytes=(\d+)")
P_RE = re.compile(r"\[trace\] P t=(\d+) off=(\d+) left=(\d+) right=(\d+)")


class Arena:
    """Free list of (offset, bytes), sorted by offset, coalescing on release."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.free = [[0, capacity]]

    def free_bytes(self):
        return sum(b for _, b in self.free)

    def largest(self):
        return max((b for _, b in self.free), default=0)

    def _carve(self, index, nbytes, from_end):
        off, size = self.free[index]
        if from_end:
            start = off + size - nbytes
            self.free[index][1] -= nbytes
        else:
            start = off
            self.free[index][0] += nbytes
            self.free[index][1] -= nbytes
        if self.free[index][1] == 0:
            del self.free[index]
        return start

    def alloc(self, nbytes, mode, lo=0, hi=None):
        """mode: first | last | best | best_end.  lo/hi restrict to a region [lo, hi)."""
        hi = self.capacity if hi is None else hi
        pick = None
        if mode in ("first", "last"):
            order = range(len(self.free)) if mode == "first" else range(len(self.free) - 1, -1, -1)
            for i in order:
                off, size = self.free[i]
                s, e = max(off, lo), min(off + size, hi)
                if e - s >= nbytes:
                    pick = (i, s, e)
                    break
        else:  # best fit: smallest extent (within the region) that fits
            best = None
            for i, (off, size) in enumerate(self.free):
                s, e = max(off, lo), min(off + size, hi)
                if e - s >= nbytes and (best is None or e - s < best[0]):
                    best = (e - s, i, s, e)
            if best:
                pick = (best[1], best[2], best[3])
        if pick is None:
            return None
        i, s, e = pick
        # carve inside the (possibly region-clipped) window [s, e) of extent i
        off, size = self.free[i]
        from_end = mode in ("last", "best_end")
        start = (e - nbytes) if from_end else s
        self._remove(i, start, nbytes)
        return start

    def _remove(self, i, start, nbytes):
        off, size = self.free[i]
        left = start - off
        right = (off + size) - (start + nbytes)
        pieces = []
        if left > 0:
            pieces.append([off, left])
        if right > 0:
            pieces.append([start + nbytes, right])
        self.free[i:i + 1] = pieces

    def release(self, off, nbytes):
        idx = bisect.bisect_left([f[0] for f in self.free], off)
        self.free.insert(idx, [off, nbytes])
        if idx > 0 and self.free[idx - 1][0] + self.free[idx - 1][1] == self.free[idx][0]:
            self.free[idx - 1][1] += self.free[idx][1]
            del self.free[idx]
            idx -= 1
        if idx + 1 < len(self.free) and self.free[idx][0] + self.free[idx][1] == self.free[idx + 1][0]:
            self.free[idx][1] += self.free[idx + 1][1]
            del self.free[idx + 1]


def make_policy(name, capacity):
    """Returns f(arena, nbytes, kind) -> offset or None, without committing for probes: callers
    commit by passing probe=False. Implemented by copying free lists for probes."""
    if name == "legacy":
        return lambda a, n, k: a.alloc(n, "first")
    if name == "twoended":
        return lambda a, n, k: a.alloc(n, "last") if k == "K" else a.alloc(n, "first")
    if name == "bestfit":
        return lambda a, n, k: a.alloc(n, "best")
    if name == "twoended_best":
        return lambda a, n, k: a.alloc(n, "best_end") if k == "K" else a.alloc(n, "best")
    if name.startswith("region"):
        # regionNN: states live in the bottom NN percent, KV in the rest (first-fit each).
        pct = int(name[len("region"):])
        cut = (capacity * pct // 100) // ALIGN * ALIGN
        return lambda a, n, k: a.alloc(n, "first", 0, cut) if k == "S" else a.alloc(n, "first", cut)
    raise SystemExit(f"unknown policy {name}")


def parse(path):
    events = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "[trace]" not in line:
                continue
            m = A_RE.search(line)
            if m:
                events.append(("A", int(m[1]), int(m[2]), int(m[3]), m[4]))
                continue
            m = F_RE.search(line)
            if m:
                events.append(("F", int(m[1]), int(m[2]), m[3], int(m[4]), int(m[5]), int(m[6]), m[7]))
                continue
            m = R_RE.search(line)
            if m:
                events.append(("R", int(m[1]), int(m[2]), int(m[3])))
                continue
            m = P_RE.search(line)
            if m:
                events.append(("P", int(m[1]), int(m[2]), int(m[3]), int(m[4])))
    return events


def replay(events, capacity, policy_name):
    place = make_policy(policy_name, capacity)
    arena = Arena(capacity)
    live = {}  # real offset -> (sim offset, bytes)
    stats = collections.defaultdict(lambda: collections.Counter())
    offset_mismatch = 0
    extents_samples = []
    for ev in events:
        kind_ev = ev[0]
        if kind_ev == "A":
            _, _, off, nbytes, kind = ev
            got = place(arena, nbytes, kind)
            if got is None:
                stats[kind]["regression"] += 1  # the real run placed it, the policy cannot
            else:
                live[off] = (got, nbytes)
                stats[kind]["placed"] += 1
                if got != off:
                    offset_mismatch += 1
            if len(extents_samples) < 100000:
                extents_samples.append(len(arena.free))
        elif kind_ev == "F":
            _, _, nbytes, kind, *_rest = ev
            stats[kind]["real_failures"] += 1
            probe = Arena(capacity)
            probe.free = [list(x) for x in arena.free]
            if place(probe, nbytes, kind) is not None:
                stats[kind]["would_fit"] += 1
        elif kind_ev == "R":
            _, _, off, nbytes = ev
            if off in live:
                sim_off, sim_bytes = live.pop(off)
                arena.release(sim_off, sim_bytes)
        elif kind_ev == "P":
            _, _, off, left, right = ev
            if off in live:
                sim_off, sim_bytes = live.pop(off)
                live[off] = (sim_off, left)
                live[off + left] = (sim_off + left, right)
    return stats, offset_mismatch, extents_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--capacity-mib", type=int, required=True)
    ap.add_argument("--policies", default="legacy,twoended,bestfit,twoended_best,region30,region50,region70")
    ap.add_argument("--force", action="store_true",
                    help="report even when the legacy replay does not reproduce the trace")
    args = ap.parse_args()
    capacity = args.capacity_mib * 1024 * 1024
    events = parse(args.trace)
    counts = collections.Counter(e[0] for e in events)
    print(f"trace events: {dict(counts)}   capacity {args.capacity_mib} MiB")
    if not events:
        sys.exit("no [trace] lines: was this a measurement build?")

    # Validation: the legacy policy on a legacy-placement trace must reproduce the real run.
    stats, mism, _ = replay(events, capacity, "legacy")
    ok = mism == 0 and all(s["regression"] == 0 for s in stats.values()) \
        and all(s["would_fit"] == 0 for s in stats.values())
    print(f"\nvalidation (legacy policy vs the real run): offset mismatches={mism}, "
          f"regressions={sum(s['regression'] for s in stats.values())}, "
          f"real failures the replay would have placed={sum(s['would_fit'] for s in stats.values())} "
          f"-> {'REPRODUCES THE RUN' if ok else 'DOES NOT REPRODUCE (trace is not from legacy placement, or incomplete)'}")
    if not ok and not args.force:
        sys.exit("refusing to report other policies from a trace the legacy replay cannot reproduce "
                 "(use --force to override)")

    print(f"\n{'policy':<14}{'kind':<6}{'placed':>9}{'real fails':>12}{'would fit':>11}{'regressions':>13}"
          f"{'  mean free runs':>16}")
    for name in args.policies.split(","):
        stats, _, ext = replay(events, capacity, name)
        mean_ext = sum(ext) / len(ext) if ext else 0
        for kind, label in (("S", "state"), ("K", "kv")):
            s = stats[kind]
            print(f"{name:<14}{label:<6}{s['placed']:>9}{s['real_failures']:>12}{s['would_fit']:>11}"
                  f"{s['regression']:>13}{mean_ext:>16.1f}" if kind == "S" else
                  f"{'':<14}{label:<6}{s['placed']:>9}{s['real_failures']:>12}{s['would_fit']:>11}{s['regression']:>13}")


if __name__ == "__main__":
    main()
