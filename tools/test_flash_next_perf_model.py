"""Self-test of flash_next_perf_model.py. Run: python -I tools/test_flash_next_perf_model.py

The model is anchored on the 8K plain step, the W = 2 / 3 / 4 verification costs and the physical
price of a miss. Everything else here is a check against a measurement in
docs/maintainer/flash-next-plan.md or the lanes spike (#284) that the anchors did NOT set, so a
tight match is evidence and a loose one is stated as loose.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flash_next_perf_model as fn  # noqa: E402


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def near(value, expected, tolerance):
    return abs(value - expected) <= tolerance * abs(expected)


def test_a_miss_is_priced_by_physics_and_corroborated_by_the_tiers():
    # 1,510,400 B frame / 24.2 GB/s = 62.4 us.
    check(near(fn.CAL.miss_ms * 1e3, 62.4, 0.01), f"miss cost {fn.CAL.miss_ms * 1e3:.1f} us")
    # Independent estimate: the plain-decode tiers' slope of step time against misses per token.
    (h1, t1), (h32, t32) = fn.PLAIN_TIERS["1K"], fn.PLAIN_TIERS["32K"]
    slope_us = (1e3 / t32 - 1e3 / t1) / (fn.misses_per_token(h32) - fn.misses_per_token(h1)) * 1e3
    check(near(slope_us, fn.CAL.miss_ms * 1e3, 0.25), f"tier slope {slope_us:.0f} us per miss")


def test_plain_decode_tiers_follow_the_hit_rate():
    # 8K sets the anchor; the 1K and 32K tiers (PR 11 gate table) are predictions.
    for tier, (hit, measured) in fn.PLAIN_TIERS.items():
        got = fn.plain(hit)["tok_s"]
        check(near(got, measured, 0.015), f"{tier}: {got:.2f} tok/s at {hit:.1%}, measured {measured}")
    check(fn.plain(0.98)["tok_s"] > fn.plain(0.972)["tok_s"] > fn.plain(0.95)["tok_s"],
          "a better cache must decode faster")


def test_a_cold_session_matches_the_lanes_spike_plain_baseline():
    # #284: on its 8 cold-session requests the server's plain decode was 74.5 tok/s [70.2-77.2] at a
    # 93.3% per-expert hit rate (31.8-32 misses per token).
    got = fn.plain(0.933)
    check(near(got["misses_per_token"], 32.0, 0.01), "93.3% hit is 32 misses per token")
    check(near(got["tok_s"], 74.5, 0.04), f"cold plain {got['tok_s']:.1f} tok/s, measured 74.5")


def test_mtp_depth_sweep_at_8k():
    # PR 13: fixed drafts 1-5 at 8K measured 102.1, 112.0, 111.9, 96.0, 90.0 tok/s.
    for k in (1, 2, 3):
        got = fn.mtp(k, fn.ACCEPTANCE_8K[k])["tok_s"]
        check(near(got, fn.MTP_SWEEP_8K[k], 0.05), f"MTP{k}: {got:.1f} vs measured {fn.MTP_SWEEP_8K[k]}")
    for k in (4, 5):
        got = fn.mtp(k, fn.ACCEPTANCE_8K[k])["tok_s"]
        check(fn.MTP_SWEEP_8K[k] < got < 1.2 * fn.MTP_SWEEP_8K[k],
              f"MTP{k}: {got:.1f}; known to run 10-15% fast against {fn.MTP_SWEEP_8K[k]}")
    best = max(range(1, 6), key=lambda k: fn.mtp(k, fn.ACCEPTANCE_8K[k])["tok_s"])
    check(best in (2, 3), f"the optimum draft count should be 2 or 3 (measured 2), got {best}")


def test_tokens_per_round_reproduces_the_measured_table():
    # PR 13: 82 / 76 / 65 / 58 / 51% acceptance gave 1.82 / 2.51 / 2.94 / 3.30 / 3.52 tokens per round.
    measured = {1: 1.82, 2: 2.51, 3: 2.94, 4: 3.30, 5: 3.52}
    for k, tokens in measured.items():
        got = fn.mtp(k, fn.ACCEPTANCE_8K[k])["tokens_per_round"]
        check(near(got, tokens, 0.02), f"{k} drafts: {got:.2f} tokens per round, measured {tokens}")


def test_verification_costs_are_the_pr13_nsys_numbers():
    for columns, ms in ((1, 12.5), (2, 17.0), (3, 20.0), (4, 23.8)):
        check(near(fn.step_ms(columns), ms, 1e-9), f"W={columns} should cost {ms} ms")
    check(fn.step_ms(5) > fn.step_ms(4), "more columns must cost more")


def test_independent_lanes_share_almost_no_experts():
    # #284 measured 19.6 distinct experts per layer at two lanes from real route traces.
    check(near(fn.distinct_experts(2), 19.6, 0.02), f"distinct experts {fn.distinct_experts(2):.2f}")
    check(near(fn.distinct_experts(1), fn.TOP_K, 1e-12), "one token selects top_k experts")
    check(fn.independent_new_fraction(2) > 2 * fn.CAL.same_sequence_new_fraction,
          "an independent row brings far more new experts than an MTP column")


def test_two_lanes_land_in_the_measured_range():
    # PR 16 gate: a pair decoded at 47-52 tok/s per stream, about 95-104 aggregate, and the spike
    # had modelled 109 (104-112).
    warm = fn.lanes(2)
    check(90 <= warm["tok_s"] <= 112, f"warm aggregate {warm['tok_s']:.1f} tok/s, measured 95-104")
    check(46 <= warm["per_stream_tok_s"] <= 56, f"per stream {warm['per_stream_tok_s']:.1f}")
    cold = fn.lanes(2, 0.933)
    check(cold["tok_s"] < warm["tok_s"], "cold caches must slow the pair")
    check(85 <= cold["tok_s"] <= 110, f"cold aggregate {cold['tok_s']:.1f}; the burst measured 95-104")


def test_lanes_pay_only_where_mtp_acceptance_is_low():
    crossover = fn.crossover_acceptance(2, 3)
    check(crossover is not None and 0.4 < crossover < 0.75, f"crossover acceptance {crossover}")
    # By construction MTP and the lanes tie there, and MTP wins above it.
    at = fn.mtp(3, crossover)["tok_s"]
    check(near(at, fn.lanes(2)["tok_s"], 1e-9), "the two rates must be equal at the crossover")
    check(fn.mtp(3, crossover + 0.1)["tok_s"] > fn.lanes(2)["tok_s"], "above it MTP wins")
    check(fn.mtp(3, crossover - 0.1)["tok_s"] < fn.lanes(2)["tok_s"], "below it the lanes win")
    # The measured 8K acceptance at 3 drafts (65%) is above the crossover: MTP wins a warm lone stream.
    check(fn.ACCEPTANCE_8K[3] > crossover, "at 65% acceptance one MTP lane beats two plain lanes")


def test_levers_are_ranked_and_scale_with_the_cache():
    for mode in ("plain", "mtp", "lanes"):
        rows = fn.rank_levers(mode, 0.972, 3, 0.65)
        gains = [r["gain_pct"] for r in rows]
        check(gains == sorted(gains, reverse=True), f"{mode}: levers must be ranked")
        by = {r["name"]: r["gain_pct"] for r in rows}
        check(by["expert cache hit rate +2 points"] > by["expert cache hit rate +1 point"] > 0,
              f"{mode}: more hit rate is worth more")
        check(all(r["gain_pct"] > 0 for r in rows), f"{mode}: every lever should help")
        check(("MTP acceptance +5 points" in by) == (mode == "mtp"), f"{mode}: acceptance lever only for MTP")
    # +1 point of hit rate is 4.8 misses per token at 62 us, about 2.4% of a 12.5 ms step.
    plain_by = {r["name"]: r["gain_pct"] for r in fn.rank_levers("plain", 0.972, 3, 0.65)}
    check(near(plain_by["expert cache hit rate +1 point"], 2.4, 0.15), "1 point of hit rate ~ 2.4%")


def test_miss_cost_scaling_is_consistent():
    cal = fn._miss_cost_scaled(fn.CAL, 0.5)
    check(near(cal.miss_ms, fn.CAL.miss_ms / 2, 1e-12), "halving the cost halves a miss")
    check(near(fn.plain(0.972, cal)["tok_s"], 1e3 / (12.5 - fn.misses_per_token(0.972) * fn.CAL.miss_ms / 2),
               1e-9), "at the reference hit rate the saving is half the misses' cost")
    unchanged = fn._miss_cost_scaled(fn.CAL, 1.0)
    check(near(fn.plain(0.95, unchanged)["tok_s"], fn.plain(0.95)["tok_s"], 1e-12), "scale 1 is a no-op")


def test_cli_text_and_json():
    import contextlib
    import io
    import json

    def run(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            check(fn.main(list(argv)) == 0, "main should return 0")
        return out.getvalue()

    text = run("--mtp-k", "3", "--lanes", "2", "--levers", "--tok-s", "79.9")
    for needle in ("PLAIN decode", "MTP, 3 drafts", "2 LANES", "acceptance is above", "LEVERS for mtp",
                   "measured 79.9"):
        check(needle in text, f"report is missing {needle!r}")
    check("validated at two lanes only" in run("--lanes", "3"), "more than two lanes must carry the caveat")
    check("validated at two lanes only" not in run("--lanes", "2"), "two lanes is the validated case")
    report = json.loads(run("--hit-rate", "0.95", "--json", "--lanes", "2"))
    check(report["lanes"]["rows"] == 2 and "mtp" not in report, "JSON should carry only what was asked")
    check(report["plain"]["tok_s"] < fn.plain()["tok_s"], "a worse cache decodes slower in the JSON too")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fn.parse_args(["--hit-rate", "1.2"])
    except SystemExit:
        pass
    else:
        check(False, "an impossible hit rate must be refused")


if __name__ == "__main__":
    for name, fn_ in sorted(globals().items()):
        if name.startswith("test_") and callable(fn_):
            fn_()
    print("flash_next_perf_model: all checks passed")
