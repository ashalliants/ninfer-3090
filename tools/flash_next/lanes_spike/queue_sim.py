"""Queue simulation: the eight captured requests all queued at t = 0 (an agent burst), served FIFO by
C plain lanes with the step model's round times, against the measured C = 1 runs (MTP and plain).
Prefill calls run alone (the Program serializes them; the other lanes wait), with the measured
prefill seconds of the plain C = 1 run. Prints makespan, mean and max completion, mean TTFT.

usage: python queue_sim.py <client_serveP.log> <client_serveM.log> <serveP.stderr.log>
"""
import json
import re
import sys

plain = [json.loads(l) for l in open(sys.argv[1])]
mtp = [json.loads(l) for l in open(sys.argv[2])]
prefill = [float(m.group(1)) for m in (re.search(r"request: \d+ prompt \+ \d+ generated tokens; prefill ([\d.]+) s", l)
                                       for l in open(sys.argv[3], encoding="utf-8", errors="replace")) if m][1:]  # skip warm-up
assert len(prefill) == len(plain)
gen = [r["usage"]["completion_tokens"] for r in plain]

# Round time (ms) with k lanes decoding: step_model.py's plain-32K rows (mid, low, high) as printed for
# the measured session; rerun step_model.py and update these after a new capture.
R = {"mid": {1: 13.2, 2: 18.3, 3: 24.2, 4: 31.4}, "low": {1: 13.2, 2: 17.9, 3: 23.4, 4: 30.2},
     "high": {1: 13.2, 2: 19.3, 3: 26.2, 4: 34.4}}


def measured(rows):
    t, done, ttft = 0.0, [], []
    for r, p in zip(rows, prefill):
        ttft.append(t + p)
        t += r["seconds"]
        done.append(t)
    return t, done, ttft


def sim(c, variant):
    queue = list(range(len(gen)))
    lanes = []  # [request, tokens left]
    t, done, ttft = 0.0, {}, {}
    while queue or lanes:
        while queue and len(lanes) < c:
            i = queue.pop(0)
            t += prefill[i]
            ttft[i] = t
            lanes.append([i, gen[i] - 1])
        k = len(lanes)
        # advance to the next finish
        step = min(l[1] for l in lanes)
        t += step * R[variant][k] / 1000
        for l in lanes:
            l[1] -= step
        for l in [l for l in lanes if l[1] == 0]:
            done[l[0]] = t
            lanes.remove(l)
    d = [done[i] for i in range(len(gen))]
    return t, d, [ttft[i] for i in range(len(gen))]


def row(name, res):
    t, d, f = res
    print(f"{name:28s} makespan {t:6.1f} s  mean completion {sum(d) / len(d):6.1f} s  max {max(d):6.1f}  "
          f"mean TTFT {sum(f) / len(f):6.1f} s  first finish {min(d):5.1f} s")


row("C=1 MTP (measured)", measured(mtp))
row("C=1 plain (measured)", measured(plain))
for v in ("mid", "low", "high"):
    row(f"C=1 plain (model, {v})", sim(1, v))
    for c in (2, 3, 4):
        row(f"C={c} plain (model, {v})", sim(c, v))
