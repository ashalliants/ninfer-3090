"""Self-test of dsv4_perf_model.py. Run: python -I tools/test_dsv4_perf_model.py

The model has no V4.1 measurement to be checked against, so these tests pin what can be pinned: the
probability arithmetic (exact), the calibration to Flash-Next's measured hit rate, the one Flash-Next
time it can reproduce, and the directions every knob must move the answer.
"""
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsv4_perf_model as dm  # noqa: E402

M = dm.Machine()


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def near(value, expected, tolerance):
    return abs(value - expected) <= tolerance * abs(expected)


def test_inclusion_probabilities_sum_to_top_k_and_stay_probabilities():
    for s in (0.0, 0.7, 1.5, 3.0):
        q = dm.inclusion_probs(384, 6, s)
        check(abs(sum(q) - 6) < 1e-6, f"s={s}: sums to {sum(q)}")
        check(all(0.0 <= x <= 1.0 + 1e-12 for x in q), f"s={s}: out of range")
        check(all(a >= b - 1e-12 for a, b in zip(q, q[1:])), f"s={s}: not ranked")


def test_uniform_routing_hits_in_proportion_to_cache_size():
    check(near(dm.hit_rate(384, 6, 0.0, 0.25), 0.25, 0.01), "uniform: hit rate == cached fraction")


def test_more_skew_means_more_hits():
    rates = [dm.hit_rate(384, 6, s, 0.10) for s in (0.0, 0.5, 1.0, 1.5, 2.0)]
    check(rates == sorted(rates) and rates[-1] > 0.9, f"{rates}")


def test_calibration_reproduces_flash_nexts_measured_hit_rate():
    s = dm.calibrate_skew()
    check(near(dm.hit_rate(dm.FN_EXPERTS, dm.FN_TOP_K, s, dm.FN_CACHE_FRACTION), dm.FN_HIT_RATE, 0.001), f"s={s}")


def test_flash_nexts_routed_expert_time_is_reproduced_within_a_quarter():
    chk = dm.fn_routed_check(M)
    # 4.27 ms measured (PR 9). The model leaves out launch tails, so it lands low; 25% is the bound.
    check(near(chk["model_ms"], chk["measured_ms"], 0.25), f"{chk}")
    check(chk["model_ms"] < chk["measured_ms"], "the model has no launch overhead, so it must not be the slower")


def test_the_measured_v41_routing_curve_is_valid_and_flatter_than_flash_next():
    q = dm.profile_inclusion_probs()
    check(len(q) == dm.EXPERTS and abs(sum(q) - dm.TOP_K) < 1e-6, f"{len(q)} ranks summing to {sum(q)}")
    check(all(a >= b for a, b in zip(q, q[1:])), "ranked by popularity")
    top10 = sum(q[:38]) / dm.TOP_K
    check(near(top10, 0.318, 0.01), f"top 10% of experts carry {top10:.3f} of routing (Strata profile: 31.8%)")
    check(top10 < dm.hit_rate(dm.EXPERTS, dm.TOP_K, dm.calibrate_skew(), 0.10),
          "V4.1 is less skewed than the Zipf fit of Flash-Next's cache behaviour")


def test_prefill_reproduces_stratas_published_rtx_5070_numbers_within_40_percent():
    # Strata docs/DEEPSEEK41.md: RTX 5070 (12 GB), 64 GB RAM, 3.6 GB/s NVMe, the 2.00 bpw build:
    # prefill 121 tok/s at 4K and 231 tok/s at 8K. A different engine, so this is a sanity bound.
    m = dm.Machine(vram_gib=12.0, ram_gib=64.0, ssd_gb_s=3.6)
    plan, q = dm.make_plan(1.944, m), dm.profile_inclusion_probs()
    for chunk, published in ((4096, 121.0), (8192, 231.0)):
        got = dm.prefill(chunk, plan, q, m)
        check(near(got["tok_s"], published, 0.40), f"{chunk}: model {got['tok_s']:.0f} vs published {published}")
        check(got["bound"] == "streaming", "on this box prefill is bound by streaming the experts")
    check(dm.prefill(8192, plan, q, m)["tok_s"] > 1.7 * dm.prefill(4096, plan, q, m)["tok_s"],
          "doubling the chunk nearly doubles prefill: the model is read once per chunk whatever its size")


def test_a_routing_curve_and_an_equal_zipf_agree():
    q = dm.inclusion_probs(dm.EXPERTS, dm.TOP_K, 0.8)
    plan = dm.make_plan(1.944, M)
    a, b = dm.round_time(2, plan, 0.8, M), dm.round_time(2, plan, q, M)
    check(a["round_ms"] == b["round_ms"], "a float is just a Zipf curve")


def test_a_single_lane_reads_exactly_top_k_experts_per_layer():
    plan = dm.make_plan(2.3125, M)
    r = dm.round_time(1, plan, 1.0, M)
    check(near(sum(r["reads"]), dm.LAYERS * dm.TOP_K, 1e-6), f"{r['reads']}")


def test_lanes_trade_per_lane_speed_for_aggregate_speed():
    plan = dm.make_plan(2.3125, M)
    s = dm.calibrate_skew()
    rows = [dm.round_time(b, plan, s, M) for b in (1, 2, 4, 8)]
    agg = [r["tok_s_aggregate"] for r in rows]
    per = [r["tok_s_per_lane"] for r in rows]
    check(agg == sorted(agg), f"aggregate must not fall with lanes: {agg}")
    check(per == sorted(per, reverse=True), f"per-lane speed must not rise with lanes: {per}")
    check(agg[-1] < 8 * agg[0], "experts are not shared, so 8 lanes cannot be 8x")


def test_the_dense_path_is_paid_once_per_round_whatever_the_lane_count():
    plan = dm.make_plan(2.3125, M)
    a, b = dm.round_time(1, plan, 1.0, M), dm.round_time(8, plan, 1.0, M)
    check(a["dense_ms"] == b["dense_ms"], "dense bytes are read once per round")


def test_knobs_move_the_answer_the_right_way():
    plan = dm.make_plan(2.3125, M)
    s = 1.0
    base = dm.round_time(1, plan, s, M)["tok_s_aggregate"]
    faster_ssd = dm.round_time(1, plan, s, dataclasses.replace(M, ssd_gb_s=10.0))["tok_s_aggregate"]
    hidden = dm.round_time(1, plan, s, dataclasses.replace(M, overlap=0.5))["tok_s_aggregate"]
    check(faster_ssd > base and hidden > base, "faster SSD and overlap must both help")
    check(dm.round_time(1, plan, 1.5, M)["tok_s_aggregate"] > dm.round_time(1, plan, 0.5, M)["tok_s_aggregate"],
          "more skew must help")


def test_plan_tiers_partition_the_experts_and_more_ram_shrinks_the_ssd_tier():
    p = dm.make_plan(2.3125, M)
    check(p.vram_experts + p.ram_experts <= p.total_experts and abs(p.vram_fraction + p.ram_fraction + p.ssd_fraction - 1) < 1e-12,
          "partition")
    big = dm.make_plan(2.3125, dataclasses.replace(M, ram_gib=128.0))
    check(big.ssd_fraction < p.ssd_fraction, "more RAM, less SSD")
    check(dm.make_plan(1.75, M).ssd_fraction < p.ssd_fraction, "a smaller format, less SSD")
    check(dm.make_plan(4.25, M).total_experts == 15360, "15,360 routed experts")


def test_everything_resident_has_no_staging_cost():
    huge = dataclasses.replace(M, vram_gib=1000.0)
    plan = dm.make_plan(2.3125, huge)
    r = dm.round_time(4, plan, 1.0, huge)
    check(r["ram_ms"] == 0.0 and r["ssd_ms"] == 0.0, f"{r}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
