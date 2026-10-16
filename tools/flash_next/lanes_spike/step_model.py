"""Step-time model for C plain lanes, from (1) a fit of the measured C = 1 rounds against the replayed
misses per round, (2) the lanes Op bench, (3) PR 13's nsys verification cost. Prints the table of the
spike document.

usage: python step_model.py <serve stderr log of the C = 1 plain run> <replay sweep csv>
"""
import csv
import re
import sys

log, sweep = sys.argv[1], sys.argv[2]

# (1) C = 1: measured decode seconds per token against misses per round (480 routed per token).
pts = []
for line in open(log, encoding="utf-8", errors="replace"):
    m = re.search(r"request: (\d+) prompt \+ (\d+) generated.*?decode ([\d.]+) s \(([\d.]+) tok/s\).*?decode ([\d.]+)% of (\d+) hit", line)
    if m and int(m.group(2)) >= 512:
        gen, sec, hit, routed = int(m.group(2)), float(m.group(3)), float(m.group(5)), int(m.group(6))
        rounds = gen - 1
        pts.append((routed / rounds * (1 - hit / 100), 1000 * sec / rounds))
n = len(pts)
mx = sum(p[0] for p in pts) / n
my = sum(p[1] for p in pts) / n
b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
a = my - b * mx
res = [p[1] - (a + b * p[0]) for p in pts]
print(f"C=1 fit over {n} requests: round_ms = {a:.2f} + {b * 1000:.1f} us * misses/round "
      f"(residual max {max(abs(r) for r in res):.2f} ms); points {[(round(x, 1), round(y, 2)) for x, y in pts]}")

# (2) bench: per layer call, us = 15.6 + 3.67 D + 55.8 m + 4.3 C (pooled over the three formats, staged)
D_US, MISS_US, COL_US = 3.67, 55.8, 4.3
# (3) PR 13: +3.4 ms per verification column of which 2.0 ms experts -> 1.4 ms non-expert per column.
NONEXP = {"low": 1.0, "mid": 1.4, "high": 1.8}
NGRAM = {"low": 0.0, "mid": 0.0, "high": 0.6}  # exposed n-gram row read per extra column (ms)

rows = list(csv.DictReader(open(sweep)))
base = {}
print("\nscenario,C,frames,hit,misses/round,distinct/layer,round_ms[low mid high],aggregate tok/s[mid; range],"
      "per-stream tok/s[mid]")
for r in rows:
    c = int(r["C"])
    if int(float(r["tokens_per_round"]) + 0.5) != c:
        continue  # plain scenarios only
    D, M = float(r["distinct_per_layer"]), float(r["misses_per_round"])
    if c == 1:
        base[r["scenario"]] = (D, M)
    D1, M1 = base[r["scenario"]]
    out = {}
    for k in ("low", "mid", "high"):
        expert_delta = 48 * (D_US * (D - D1) + COL_US * (c - 1)) / 1000
        miss = (b * M1 + MISS_US / 1000 * (M - M1)) if c > 1 else b * M
        out[k] = a + (NONEXP[k] + NGRAM[k]) * (c - 1) + expert_delta + miss
    agg = {k: 1000 * c / v for k, v in out.items()}
    print(f"{r['scenario']},{c},{r['frames']},{r['hit_med']}%,{M:.1f},{D:.2f},"
          f"{out['low']:.1f} {out['mid']:.1f} {out['high']:.1f},{agg['mid']:.0f} [{agg['high']:.0f}-{agg['low']:.0f}],"
          f"{agg['mid'] / c:.1f}")
