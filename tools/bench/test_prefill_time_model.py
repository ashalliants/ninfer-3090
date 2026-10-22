"""Self-test of prefill_time_model.py. Run: python -I tools/bench/test_prefill_time_model.py

The model is checked against measurements the repo documents (docs/performance.md) and against
values computed by the C++ `PrefillPacing::step_cap` for the same cost, so a drift in either the
fitted coefficients or the cap logic shows up here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_time_model as m  # noqa: E402


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def near(value, expected, tolerance):
    return abs(value - expected) <= tolerance * expected


def test_coefficients_come_from_the_engine_source():
    cost = m.load_cost("27b")
    check(near(cost.chunk_ns, 107_758_083, 1e-9), "27B chunk_ns was not read from the Engine's table")
    check(near(cost.token_ns, 284_500, 0.01), f"27B per-token cost {cost.token_ns}")
    check(near(cost.pair_ns, 4.54, 0.01), f"27B attention pair cost {cost.pair_ns}")
    moe = m.load_cost("35b-a3b")
    check(moe.token_ns < cost.token_ns / 2, "the MoE should be far cheaper per token")


def test_model_matches_documented_prefill_speeds():
    # docs/performance.md, ninfer_bench, rk4v4 + cuBLAS prefill, chunk 4096, Qwen3.8-27B.
    cost = m.load_cost("27b")
    for prompt, measured in ((4096, 3143), (16384, 2893), (65536, 2172)):
        predicted = m.prefill_profile(cost, prompt, 4096)["tokens_per_second"]
        check(near(predicted, measured, 0.10),
              f"pp{prompt}: model {predicted:.0f} tok/s against the documented {measured}")


def test_shares_add_up_and_attention_grows_with_the_prompt():
    cost = m.load_cost("27b")
    previous = 0.0
    for prompt in (4096, 16384, 65536, 131072):
        p = m.prefill_profile(cost, prompt, 4096)
        check(near(p["fixed_share"] + p["linear_share"] + p["attention_share"], 1.0, 1e-9),
              "time shares do not add up")
        check(p["attention_share"] > previous, "attention share did not grow with the prompt")
        previous = p["attention_share"]
    check(previous > 0.4, "a 128K prompt should spend a large share in attention")


def test_larger_chunks_amortise_the_fixed_cost():
    cost = m.load_cost("27b")
    rates = [m.prefill_profile(cost, 100_000, c)["tokens_per_second"] for c in (1024, 2048, 4096, 8192)]
    check(rates == sorted(rates), "a larger chunk was predicted slower")


def test_step_cap_matches_the_expected_values_of_the_cpp_policy():
    # A Python expected-value test. The same table is asserted by the C++ test
    # `test_step_cap_expected_values` in tests/runtime/test_prefill_pacing.cpp (the stall-budget PR),
    # so a change to either side that is not made to the other fails one of them.
    # Values printed by PrefillPacing::step_cap (src/runtime/engine/prefill_pacing.h) for this cost:
    # 100 ms per step, 0.31 ms per token, 0.5 ns per attention pair, chunk 4096.
    cost = m.Cost(chunk_ns=100e6, token_ns=310_000, pair_ns=0.5)
    cpp = {100: (512, 512), 200: (512, 512), 300: (640, 512), 500: (1280, 1024),
           800: (2176, 1920), 1200: (3456, 3072), 1500: (0, 3840)}
    for budget, (early, deep) in cpp.items():
        check(m.step_cap(cost, 0, 4096, budget) == early, f"cap at budget {budget} ms, prefix 0")
        check(m.step_cap(cost, 90_000, 4096, budget) == deep, f"cap at budget {budget} ms, prefix 90k")
    check(m.step_cap(cost, 0, 4096, 0) == 0 and m.step_cap(cost, 0, 512, 100) == 0,
          "an off budget or a step at the minimum was capped")


def run(scenario, **kwargs):
    return m.run_scenario(m.load_cost("27b"), 100_000, 4096, 45.0, scenario, **kwargs)


def test_scenarios_behave():
    base = run(m.Scenario("base"))
    one = run(m.Scenario("one slice", slices=1))
    check(one == {**base, "scenario": "one slice"}, "one slice should equal the baseline")
    gaps = [run(m.Scenario(f"s{n}", slices=n))["worst_stream_gap_ms"] for n in (1, 2, 4, 8, 16)]
    check(gaps == sorted(gaps, reverse=True), "more slices should shorten the worst gap")
    check(gaps[2] < gaps[0] / 3, "four slices should cut the worst gap by more than three")
    times = [run(m.Scenario(f"s{n}", slices=n))["prompt_seconds"] for n in (1, 2, 4, 8)]
    check(times == sorted(times), "more slices should not make the prompt faster")
    budget = run(m.Scenario("budget", budget_ms=500))
    check(budget["worst_stream_gap_ms"] < base["worst_stream_gap_ms"] / 3, "the budget did not cut the gap")
    check(budget["prompt_seconds"] > base["prompt_seconds"], "the budget should cost prompt time")
    share = run(m.Scenario("share", share_percent=50))
    check(share["stream_rounds_per_second"] > 10 * base["stream_rounds_per_second"],
          "a decode share should give the stream far more rounds per second")
    check(near(share["worst_stream_gap_ms"], base["worst_stream_gap_ms"], 1e-9),
          "a decode share does not shorten the worst gap")
    # Slices beat a budget on prompt time for a similar gap: the point of slicing.
    slices = run(m.Scenario("s8", slices=8))
    check(slices["prompt_seconds"] < budget["prompt_seconds"], "slicing should beat the budget on prompt time")


def test_yield_turns_a_wait_for_the_prompt_into_a_wait_for_a_step():
    y = m.yield_wait(m.load_cost("27b"), 100_000, 4096, 0.2)
    check(y["wait_without_yield_s"] > 30 and y["wait_with_yield_s"] < 3, f"yield wait {y}")


def test_unusable_numbers_are_an_argument_error_not_a_hang():
    for bad in (["--chunk", "0"], ["--chunks", "1024", "0"], ["--prompt", "0"], ["--prompt", "-5"],
                ["--decode-round-ms", "0"], ["--yield-arrival", "0"], ["--yield-arrival", "1"]):
        try:
            m.parse_args(bad)
        except SystemExit as error:
            check(error.code == 2, f"{bad}: exit code {error.code}")
        else:
            check(False, f"{bad} was accepted")
    m.parse_args([])


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
    print("OK prefill time model")
