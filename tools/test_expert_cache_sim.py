"""Self-test of expert_cache_sim.py. Run: python -I tools/test_expert_cache_sim.py

Every trace here is SYNTHETIC and tiny; the expected numbers were worked out by hand (the
derivations are in the comments). They prove the simulator counts hits, promotions and swaps
correctly. They say nothing about how any policy performs on real routing: the repository has no
real routing trace. The LFRU model is also checked against the C++ policy's own conformance
fixture (tests/fixtures/expert_cache/lfru_conformance.txt), whose victims come from the replay tool
the C++ was ported against.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expert_cache_sim as sim  # noqa: E402

FIXTURE = os.path.join(HERE, "..", "tests", "fixtures", "expert_cache", "lfru_conformance.txt")


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def rounds_of(experts):
    """One layer, one token per round, one routed expert per round."""
    return [(1, [np.array([e], dtype=np.int64)]) for e in experts]


def test_lfru_hand_computed():
    # 1 layer, 4 experts, 2 frames, interval 1. Trace 0 0 1 0 2 1.
    # r1 0 miss, fill-admit. r2 0 hit. r3 1 miss, fill-admit (2 promotions = frames, so fill is over).
    # r4 0 hit. r5 2 miss; now=5: score(0)=3/(5-4+1)=1.5, score(1)=1/3, score(2)=1/1=1 > 1/3: evict 1.
    # r6 1 miss; now=6: score(0)=3/3=1, score(2)=1/2, score(1)=2/1=2 > 1/2: evict 2.
    # hits 2 of 6 routed, promotions 4 (0, 1, 2, 1).
    r = sim.LfruSim(1, 4, 2, interval=1).run(rounds_of([0, 0, 1, 0, 2, 1]))
    check(int(r.routed.sum()) == 6 and int(r.hits.sum()) == 2, f"hand LFRU hits: {r.hits.sum()}/{r.routed.sum()}")
    check(r.promotions == 4, f"hand LFRU promotions: {r.promotions}")
    # Interval 4: rounds 5 and 6 bank only 1 and 2 tokens, so nothing is due and the cache stays {0, 1}:
    # r6 (expert 1) is then a hit. 3 hits, 2 promotions.
    r = sim.LfruSim(1, 4, 2, interval=4).run(rounds_of([0, 0, 1, 0, 2, 1]))
    check(int(r.hits.sum()) == 3 and r.promotions == 2, "hand LFRU interval 4")
    # Trace 0 0 1 0 2 2, interval 2: banking starts at r4 (r1-r3 are the fill phase: promotions < frames
    # until r3 ends), r4 banks 1, r5 banks 2 -> due 1. At r5 now=5: miss 2 has score 1/1 = 1, victim 1
    # scores 1/(5-3+1) = 1/3 (expert 0: 3/(5-4+1) = 1.5): 2 replaces 1. r6 then hits 2. Hits r2, r4, r6.
    r = sim.LfruSim(1, 4, 2, interval=2).run(rounds_of([0, 0, 1, 0, 2, 2]))
    check(int(r.hits.sum()) == 3 and r.promotions == 3, f"hand LFRU interval 2: {r.hits.sum()} {r.promotions}")


def test_lfru_protects_group_and_budget():
    # 2 layers, 3 experts, 1 frame. Round 1 routes layer0 {0,1}, layer1 {2}: fill budget 1/layer/token.
    # layer0: misses 0,1 both score 1; key order: admit 0 (budget 1). layer1: key 3+2=5 admitted? frames=1
    # and 0 is resident and outside the group, equal count 1: score(0)=1/(2-1+1)=0.5 < score(5)=1/1: evict.
    sim_ = sim.LfruSim(2, 3, 1, interval=1)
    r = sim_.run([(1, [np.array([0, 1]), np.array([2])])])
    check(r.promotions == 2 and sim_.resident[5] and not sim_.resident[0], "budget 1 per layer, victim outside group")
    check(int(r.hits.sum()) == 0 and int(r.routed.sum()) == 3, "nothing resident at the start of round 1")


def test_tokens_per_round():
    # An MTP-style round advancing 3 tokens in the fill phase has budget 3 per layer.
    r = sim.LfruSim(1, 8, 8, interval=4).run([(3, [np.array([0, 1, 2, 3])])])
    check(r.promotions == 3 and r.tokens == 3, "fill budget scales with tokens")
    # After the fill phase, 3 tokens + 1 token reach interval 4 on the second round only.
    s = sim.LfruSim(1, 4, 1, interval=4)
    r = s.run([(1, [np.array([0])]), (3, [np.array([1])]), (1, [np.array([1])]), (1, [np.array([1])])])
    # r1 fills (promotion 1 = frames). r2: banks 3, nothing due. r3: banks 4 -> due 1: key 1 (count 2,
    # last 3 -> score 2/1) beats 0 (count 1, last 1 -> 1/(3-1+1)): promoted at r3. r4: hit.
    check(r.promotions == 2 and int(r.hits.sum()) == 1, f"interval banking: {r.promotions} {r.hits.sum()}")


def test_windowed_hand_computed():
    # 1 layer, 4 experts, 2 frames, N=3, M=2, max_swaps 1. Trace 0 0 1 0 2 1 2 2 2.
    # Fill: r1 admits 0, r3 admits 1. Full after r3. Re-selection at even rounds once full:
    #  r4 window(r2..r4): 0:2, 1:1; no non-resident with hits -> none.
    #  r6 window(r4..r6): 0:1, 2:1, 1:1; candidate 2 (1) vs worst resident (1): gain 0 < 1 -> none.
    #  r8 window(r6..r8): 1:1, 2:2, 0:0; candidate 2 (2) vs resident 0 (0): gain 2 >= 1 -> swap.
    # Hits (resident at the start of the round): r2 (0), r4 (0), r6 (1); r7, r8 miss; r9 hits 2. = 4 of 9.
    # Promotions 3.
    trace = rounds_of([0, 0, 1, 0, 2, 1, 2, 2, 2])
    r = sim.WindowedSim(1, 4, 2, window=3, every=2, min_gain=1, max_swaps=1).run(trace)
    check(int(r.hits.sum()) == 4 and r.promotions == 3, f"hand windowed: {r.hits.sum()}, {r.promotions}")
    # Hysteresis: min_gain 3 is never met (best gain is 2), so no swap: hits r2, r4, r6 = 3, promotions 2.
    r = sim.WindowedSim(1, 4, 2, window=3, every=2, min_gain=3, max_swaps=1).run(trace)
    check(int(r.hits.sum()) == 3 and r.promotions == 2, "hysteresis blocks the swap")
    # Relative hysteresis: resident 0 has window 0 so a 100% margin does not stop the swap (2 - 0*2 >= 1).
    r = sim.WindowedSim(1, 4, 2, window=3, every=2, hysteresis=1.0, min_gain=1, max_swaps=1).run(trace)
    check(r.promotions == 3, "relative margin on a zero-hit resident")


def test_swap_budget_and_window_expiry():
    # 1 layer, 6 experts, 2 frames, N=2, M=2, max_swaps 1. Trace 0 1 4 4 5 5.
    # Fill: r1 admits 0, r2 admits 1 (r2 itself is a fill round, so no re-selection there).
    # r4: window(r3..r4) has 4:2; both residents have 0 hits, the lower key (0) goes: swap, 3 promotions.
    # r6: window(r5..r6) has 5:2 and 4 has expired to 0; residents 1 and 4 tie at 0, key 1 goes: 4 in total.
    # Routed 4 at r4 and 5 at r5/r6 were never resident when asked: 0 hits, final residents {4, 5}.
    trace = rounds_of([0, 1, 4, 4, 5, 5])
    s = sim.WindowedSim(1, 6, 2, window=2, every=2, min_gain=1, max_swaps=1)
    r = s.run(trace)
    check(r.promotions == 4 and int(r.hits.sum()) == 0, f"{r.promotions} promotions, {r.hits.sum()} hits")
    check(sorted(np.flatnonzero(s.resident)) == [4, 5], "the window slid: residents are 4 and 5")
    # max_swaps 2 with one window over the whole trace, M=6: at r6 window has 4:2, 5:2; both swap in
    # for residents 0 and 1 (windows 1 each, gain 1 >= 1). 2 fill + 2 swaps = 4.
    r2 = sim.WindowedSim(1, 6, 2, window=6, every=6, min_gain=1, max_swaps=2).run(trace)
    check(r2.promotions == 4, f"two swaps at r6: {r2.promotions}")
    r3 = sim.WindowedSim(1, 6, 2, window=6, every=6, min_gain=1, max_swaps=1).run(trace)
    check(r3.promotions == 3, f"swaps never exceed max_swaps: {r3.promotions}")


def test_curve_and_per_layer():
    r = sim.LfruSim(2, 4, 8, interval=1).run([(1, [np.array([0]), np.array([1])])] * 4)
    # Round 1 misses both layers (fill-admits both); rounds 2-4 hit: 2 + 6 hits of 8 routed.
    check(r.hit_rate == 6 / 8 and r.per_layer_hit_rate() == [0.75, 0.75], "hit rates")
    curve = r.windowed_curve(4)
    check([round(h, 3) for _, h, _ in curve] == [0.0, 1.0, 1.0, 1.0], f"cold-start curve {curve}")
    check(abs(curve[0][2] - 2.0) < 1e-9 and curve[1][2] == 0.0, "promotions per token in the first slice")


def test_trace_roundtrip():
    rounds = [(1, [np.array([3, 1]), np.array([0])]), (2, [np.array([], dtype=np.int64), np.array([2])])]
    with tempfile.TemporaryDirectory() as d:
        for name in ("t.jsonl", "t.jsonl.gz"):
            path = os.path.join(d, name)
            sim.write_trace(path, 2, 4, rounds)
            layers, experts, back = sim.read_trace(path)
            check((layers, experts) == (2, 4) and len(back) == 2, "trace header and length")
            check(back[0][0] == 1 and list(back[0][1][0]) == [1, 3] and list(back[1][1][1]) == [2], "trace content")
            check(back[1][0] == 2 and len(back[1][1][0]) == 0, "tokens and empty layer")
        bad = os.path.join(d, "bad.jsonl")
        with open(bad, "w") as f:
            f.write('{"format": "ninfer-routing-trace", "version": 1, "layers": 1, "experts": 2}\n')
            f.write('{"t": 1, "r": [[5]]}\n')
        try:
            sim.read_trace(bad)
            check(False, "out-of-range expert id is rejected")
        except ValueError:
            pass


def test_lfru_matches_cpp_conformance_fixture():
    # The C++ LfruPolicy is held to this file decision for decision (test_expert_cache.cpp). The
    # simulator's on-demand mode must produce the same victims, in the same order, and the same
    # hit and miss totals: that ties the Python model of policy (a) to the shipped code.
    with open(FIXTURE) as f:
        header = f.readline().split()
        keys, capacity, hits, misses = int(header[2]), int(header[4]), int(header[6]), int(header[8])
        model = sim.LfruSim(1, keys, capacity)
        got_hits = got_misses = groups = mismatches = 0
        for line in f:
            left, _, right = line[1:].partition("|")
            group = np.array([int(x) for x in left.split()], dtype=np.int64)
            expected = [int(x) for x in right.split()]
            was = model.resident[group].copy()
            model._step(group, None)
            got_hits += int(was.sum())
            got_misses += int((~was).sum())
            mismatches += model.last_victims != expected
            groups += 1
    check(groups == 1200, f"fixture groups: {groups}")
    check(mismatches == 0, f"{mismatches} of {groups} steps differ from the C++ fixture")
    check((got_hits, got_misses) == (hits, misses), f"fixture totals {got_hits}/{got_misses} vs {hits}/{misses}")


def test_invalid_trace_metadata_rejected():
    header = '{"format": "ninfer-routing-trace", "version": 1, "layers": %s, "experts": %s}\n'
    cases = [(header % ("1.9", "4"), ""), (header % ("0", "4"), ""), (header % ("true", "4"), ""),
             (header % ("1", "4"), '{"t": 1.5, "r": [[0]]}\n'), (header % ("1", "4"), '{"t": 0, "r": [[0]]}\n'),
             (header % ("1", "4"), '{"t": 1, "r": [[0.5]]}\n'), (header % ("1", "4"), '{"t": 1, "r": [["2"]]}\n')]
    with tempfile.TemporaryDirectory() as d:
        for i, (head, body) in enumerate(cases):
            path = os.path.join(d, f"bad{i}.jsonl")
            with open(path, "w") as f:
                f.write(head + body)
            try:
                sim.read_trace(path)
            except ValueError:
                continue
            check(False, f"invalid trace case {i} was accepted")


def test_default_swap_cap_follows_tokens():
    # 2 layers, interval 4: 8 tokens between re-selections allows round(2 * 8 / 4) = 4 swaps, not layers*every/interval.
    s = sim.WindowedSim(2, 8, 4, window=4, every=2, interval=4)
    s.resident[:4] = True
    s.n_resident = 4
    s.win[4:8] = 5
    check(s._reselect(8) == 4, "token-based default cap allows 4 swaps")
    s2 = sim.WindowedSim(2, 8, 4, window=4, every=2, interval=4)
    s2.resident[:4] = True
    s2.n_resident = 4
    s2.win[4:8] = 5
    check(s2._reselect(2) == 1, "2 tokens elapsed allow 1 swap")


def test_cli_rejects_invalid_parameters():
    for bad in (["--frames", "0"], ["--window", "-1"], ["--every", "0"], ["--interval", "0"],
                ["--min-gain", "-1"], ["--hysteresis", "-0.5"], ["--max-swaps", "-2"], ["--expert-bytes", "-1"], ["--hysteresis", "nan"], ["--hysteresis", "inf"], ["--expert-bytes", "nan"]):
        try:
            sim.main(["unused-trace"] + bad)
        except SystemExit as e:
            check(e.code == 2, f"{bad}: exit code {e.code}")
        else:
            check(False, f"{bad} was accepted")


if __name__ == "__main__":
    test_lfru_hand_computed()
    test_lfru_protects_group_and_budget()
    test_tokens_per_round()
    test_windowed_hand_computed()
    test_swap_budget_and_window_expiry()
    test_curve_and_per_layer()
    test_trace_roundtrip()
    test_lfru_matches_cpp_conformance_fixture()
    test_invalid_trace_metadata_rejected()
    test_default_swap_cap_follows_tokens()
    test_cli_rejects_invalid_parameters()
    print("expert_cache_sim: all checks passed")
