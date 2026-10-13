"""Runs lanes_replay over C = 1..4, eight rotations of the request list each, for the scenarios of
docs/maintainer/flash-next-lanes-spike.md, and prints one CSV row per scenario and C (medians over
the rotations; the hit rate's range too).

usage: python sweep.py <lanes_replay.exe> <trace dir> [scenario ...]

The trace dir holds client.py's trace_<i>_<name>.bin files of one plain C = 1 session; their prompt
token counts are listed in TRACES below (the client's usage.prompt_tokens of that session).
"""
import re
import statistics
import subprocess
import sys

EXE, TRACE_DIR = sys.argv[1], sys.argv[2]
TRACES = ["trace_0_codegen.bin=109", "trace_1_review.bin=7106", "trace_2_story.bin=101", "trace_3_tools.bin=3235",
          "trace_4_summary.bin=10478", "trace_5_math.bin=185", "trace_6_sql.bin=112", "trace_7_chinese.bin=99"]
PAT = re.compile(r"rounds (\d+), hit ([\d.]+)%, misses/round ([\d.]+), distinct experts/layer ([\d.]+), layer calls with "
                 r"a miss ([\d.]+)%, with >2 misses ([\d.]+)%, promotions/round ([\d.]+), tokens/round ([\d.]+), "
                 r"columns/round ([\d.]+)")
KEYS = ["rounds", "hit", "misses", "distinct", "with_miss", "over2", "promotions", "tokens", "columns"]

# name: (frames at C = 1, frames each further lane costs, width, advance). Frames per lane: the plain
# Program's KV at 32K (480,247,808 B) plus its recurrent state (115,651,840 B) over one 1,510,400 B frame
# = 395; at 64K the KV is ~921 MB (estimated from the 32K plan) = 687; with MTP the 32K KV is
# 520,093,696 B, the state 115,652,608 B and the verification memory 6,841,600 B = 426. The C = 1 counts
# are the startup sizes of the measured sessions (12,258 plain, 11,614 MTP; 64K: 12,258 less 292).
SCENARIOS = {
    "plain-32K": (12258, 395, 1, 1),
    "plain-32K-same-frames": (12258, 0, 1, 1),
    "plain-64K": (11966, 687, 1, 1),
    "mtp-proxy-32K-w3a2": (11614, 426, 3, 2),
}


def run(frames, lanes, width, advance, offset, warmup=200):
    out = subprocess.run([EXE, "--frames", str(frames), "--lanes", str(lanes), "--width", str(width), "--advance",
                          str(advance), "--warmup", str(warmup), "--offset", str(offset)] + TRACES,
                         cwd=TRACE_DIR, capture_output=True, text=True, check=True).stdout
    m = PAT.search(out)
    if not m:
        raise RuntimeError(out)
    return dict(zip(KEYS, map(float, m.groups())))


print("scenario,C,frames,n,hit_med,hit_min,hit_max,misses_per_round,distinct_per_layer,with_miss_pct,over2_pct,"
      "promotions_per_round,tokens_per_round,columns_per_round,rounds_med")
for name in sys.argv[3:] or list(SCENARIOS):
    f1, per, w, a = SCENARIOS[name]
    for c in (1, 2, 3, 4):
        frames = f1 - (c - 1) * per
        rs = [run(frames, c, w, a, off) for off in range(len(TRACES))]
        med = lambda key: statistics.median(r[key] for r in rs)  # noqa: E731
        print(f"{name},{c},{frames},{len(rs)},{med('hit'):.2f},{min(r['hit'] for r in rs):.2f},"
              f"{max(r['hit'] for r in rs):.2f},{med('misses'):.2f},{med('distinct'):.2f},{med('with_miss'):.1f},"
              f"{med('over2'):.1f},{med('promotions'):.2f},{med('tokens'):.2f},{med('columns'):.2f},{med('rounds'):.0f}",
              flush=True)
