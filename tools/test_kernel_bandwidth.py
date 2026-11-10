"""Self-test of kernel_bandwidth.py. Run: python -I tools/test_kernel_bandwidth.py

Two kinds of input. Synthetic CSVs in the layout `nsys stats --report cuda_gpu_kern_sum --format csv`
writes, with kernel names taken from src/ops, where each family's time is chosen to come out at a known
GB/s: that checks the units, the grouping and the arithmetic. And the first REAL captures
(tests/fixtures/perf_model/, one power-capped RTX 3090, Qwen3.8-27B plain decode and MTP3), which check
what synthetic data cannot: that the classification holds on the kernels a real run contains, that
--steps is the right divisor, that kernel time matches the measured throughput, and how the byte
model behaves on a host it was never fitted on. They cover a dense 27B only: no MoE, no prefill.
"""
import contextlib
import csv
import io
import json
import os
import pathlib
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kernel_bandwidth as kb  # noqa: E402
import perf_model as pm  # noqa: E402

COLUMNS = ["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)",
           "StdDev (ns)", "Name"]
STEPS = 128


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def near(value, expected, tolerance):
    return abs(value - expected) <= tolerance * abs(expected)


def test_short_name_strips_templates_namespaces_and_arguments():
    cases = {
        "void ninfer::ops::q4_a16_gemv_kernel<4, 128>(const half*, int)": "q4_a16_gemv_kernel",
        "q5_small_t_mma_kernel": "q5_small_t_mma_kernel",
        "void ninfer::detail::foo_kernel<std::pair<int, int>, 8>(float*)": "foo_kernel",
        "sparse_moe_d3_nine_warp_kernel(int*)": "sparse_moe_d3_nine_warp_kernel",
    }
    for raw, expected in cases.items():
        check(kb.short_name(raw) == expected, f"{raw!r} -> {kb.short_name(raw)!r}, want {expected!r}")


def test_classification_follows_the_kernel_names_in_src_ops():
    cases = {
        "q4_a16_gemv_kernel": "q4",
        "void ninfer::ops::q5_small_t_mma_kernel<3>(int)": "q5",
        "q6_a16_mma_kernel": "q6",
        "q8_a16_gemv_kernel": "q8",
        "bf16_gdn_gating_proj_gemv_kernel": "bf16",  # the a/b projections are BF16 weights
        "nvfp4_a16_gemv_kernel": "nvfp4",
        "sparse_moe_d3_nine_warp_kernel": "moe",
        "int8_kv_tiled_mma_kernel": "attention",
        "kv_cache_append_full_bf16_kernel": "attention",
        "causal_attention_prompt_i8_fa2_kernel": "attention",
        "recurrent_direct_kernel": "gdn",
        "gdn_projected_conv_kernel": "gdn",
        "chunk_recurrence_kernel": "gdn",
        # Review finding: cache and recurrence kernels are named after the format they store, and
        # were claimed by the weight-format prefix rule. Real names from src/ops:
        "bf16_kv_grouped_mma_kernel": "attention",
        "fp8_kv_grouped_mma_kernel": "attention",
        "int8_kv_grouped_mma_kernel": "attention",
        "k8v4_kv_grouped_mma_kernel": "attention",
        "nvfp4_kv_tiled_mma_kernel": "attention",
        "mxfp8_kv_tiled_mma_kernel": "attention",
        "nvfp4_gdn_conv_post_kernel": "gdn",
        # A fused pair of stored formats is its own family, not whichever format comes first.
        "q4_q5_attn_input_small_t_kernel": "q4_q5",
        # Streams no weights, whatever its name starts with.
        "q4_small_t_quantize_activations_kernel": "other",
        "fp8_a8_tma_split_k_reduce": "other",
        "bf16_split_k_reduce_kernel": "other",
        "rmsnorm_rope_d128_kernel": "other",
        "ggml_mmvq_kernel": "other",
    }
    for raw, expected in cases.items():
        check(kb.classify(raw) == expected, f"{raw}: {kb.classify(raw)}, want {expected}")


def write_csv(path, entries):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for name, total_ns, instances in entries:
            writer.writerow(["0.0", f"{total_ns:.0f}", instances, "0", "0", "0", "0", "0", name])


def model_bytes(preset="27b", recipe="qwen3_8_27b", depth=0):
    geo = pm.Geometry.from_config(pm.PRESETS[preset])
    weights = pm.build_weights(geo, pm.RECIPES[recipe])
    step = pm.decode_step(geo, weights, depth=depth)
    groups = pm.bytes_by_group(weights, geo)
    groups["attention"] = step.kv
    groups["gdn"] = step.state_rw
    return geo, weights, groups


def ms_for_rate(nbytes, gb_s):
    return nbytes / (gb_s * 1e9) * 1e3


def test_join_recovers_the_rate_each_family_was_built_to_run_at():
    _, _, groups = model_bytes(depth=8192)
    # q4 at 800 GB/s (good), q5 at 500 (poor, split over two kernels), attention at 300.
    q4_ms, q5_ms, attn_ms = (ms_for_rate(groups["q4"], 800), ms_for_rate(groups["q5"], 500),
                             ms_for_rate(groups["attention"], 300))
    rows = [
        {"name": "void ninfer::ops::q4_a16_gemv_kernel<4>(int)", "total_ns": q4_ms * 1e6 * STEPS,
         "instances": 100 * STEPS},
        {"name": "q5_small_t_mma_kernel", "total_ns": q5_ms * 0.75e6 * STEPS, "instances": 50 * STEPS},
        {"name": "q5_a16_gemv_kernel", "total_ns": q5_ms * 0.25e6 * STEPS, "instances": 10 * STEPS},
        {"name": "int8_kv_tiled_mma_kernel", "total_ns": attn_ms * 1e6 * STEPS, "instances": 16 * STEPS},
        {"name": "rmsnorm_rope_d128_kernel", "total_ns": 0.5e6 * STEPS, "instances": 64 * STEPS},
    ]
    joined = kb.join(rows, STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)
    by = {r["family"]: r for r in joined["families"]}
    check(near(by["q4"]["gb_s"], 800, 1e-6), f"q4 rate {by['q4']['gb_s']}")
    check(near(by["q5"]["gb_s"], 500, 1e-6), "two q5 kernels must be summed into one family")
    check(near(by["q5"]["launches"], 60, 1e-9), "launches are per step")
    check(near(by["q5"]["pct"], 500 / pm.ACHIEVABLE_READ_GB_S * 100, 1e-6), "percent of sustained read")
    check(by["q5"]["top_kernel"] == "q5_small_t_mma_kernel", "the biggest kernel is named")
    check(near(by["attention"]["gb_s"], 300, 1e-6), "attention is compared with the KV bytes")
    good_ms = ms_for_rate(groups["q5"], pm.ACHIEVABLE_READ_GB_S * 0.9)
    check(near(by["q5"]["recoverable_ms"], q5_ms - good_ms, 1e-6), "saving to reach 90% of the rate")
    check(by["q4"]["recoverable_ms"] == 0.0, "a family already above the target saves nothing")
    check(by["other"]["bytes"] is None and by["other"]["gb_s"] is None, "`other` streams no bytes")
    check(joined["families"][0]["ms"] >= joined["families"][-1]["ms"], "families are sorted by time")
    check(near(joined["kernel_ms"], sum(r["ms"] for r in joined["families"]), 1e-12), "total")


def test_kv_kernels_are_joined_to_the_kv_bytes_not_to_weights():
    _, _, groups = model_bytes(depth=8192)
    ms = ms_for_rate(groups["attention"], 400)
    rows = [{"name": "void ninfer::ops::bf16_kv_grouped_mma_kernel<8>(int)", "total_ns": ms * 1e6 * STEPS,
             "instances": 16 * STEPS}]
    by = {r["family"]: r for r in kb.join(rows, STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)["families"]}
    check(set(by) == {"attention"}, f"a bf16 KV kernel must not land in the bf16 weight bucket: {set(by)}")
    check(near(by["attention"]["gb_s"], 400, 1e-6), "and is compared with the KV bytes")


def test_a_fused_format_kernel_gets_its_own_row_and_a_caveat():
    _, _, groups = model_bytes()
    rows = [{"name": "q4_q5_attn_input_small_t_kernel", "total_ns": 1e6 * STEPS, "instances": STEPS},
            {"name": "q4_a16_gemv_kernel", "total_ns": 2e6 * STEPS, "instances": STEPS}]
    joined = kb.join(rows, STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)
    by = {r["family"]: r for r in joined["families"]}
    check(by["q4_q5"]["bytes"] is None and by["q4_q5"]["gb_s"] is None, "no byte claim for a fused kernel")
    check(joined["caveats"] and "q4_q5" in joined["caveats"][0] and "HIGH" in joined["caveats"][0],
          "the rows it distorts are named, with the direction")
    clean = kb.join(rows[1:], STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)
    check(clean["caveats"] == [], "no fused kernel, no caveat")


def test_the_calibration_rate_counts_a_fused_kernels_time():
    # Review finding: a fused kernel's bytes are inside the q4 and q5 rows, but its time was left out of
    # the combined weight rate, which then read high and made a calibrated model underpredict decode.
    _, _, groups = model_bytes()
    t4, t5 = ms_for_rate(groups["q4"], 800), ms_for_rate(groups["q5"], 800)
    fused_ms = 3.0
    rows = [{"name": "q4_a16_gemv_kernel", "total_ns": t4 * 1e6 * STEPS, "instances": STEPS},
            {"name": "q5_small_t_mma_kernel", "total_ns": t5 * 1e6 * STEPS, "instances": STEPS},
            {"name": "q4_q5_attn_input_small_t_kernel", "total_ns": fused_ms * 1e6 * STEPS,
             "instances": STEPS}]
    joined = kb.join(rows, STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)
    expected = (groups["q4"] + groups["q5"]) / ((t4 + t5 + fused_ms) / 1e3) / 1e9
    check(near(joined["weights_gb_s"], expected, 1e-9),
          f"combined rate {joined['weights_gb_s']:.1f}, expected {expected:.1f} with the fused time counted")
    check(joined["weights_gb_s"] < 800, "counting the fused time must lower the rate below the clean 800")
    clean = kb.join(rows[:2], STEPS, groups, pm.ACHIEVABLE_READ_GB_S, 90.0)
    check(near(clean["weights_gb_s"], 800, 1e-9), "with no fused kernel the rate is the families' own 800")


def test_numeric_options_and_recipe_choice_follow_perf_model():
    import io
    for argv in (["--steps", "0"], ["--steps", "-1"], ["--columns", "0"], ["--depth", "-1"],
                 ["--target-pct", "0"], ["--target-pct", "101"], ["--target-pct", "nan"],
                 ["--bandwidth-gb-s", "0"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                kb.parse_args(["x.csv", "--steps", "128", *argv])
        except SystemExit:
            continue
        check(False, f"{argv} must be refused by the parser")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "k.csv")
        write_csv(path, [("sparse_moe_d3_nine_warp_kernel", 1e6 * STEPS, STEPS)])
        report = json.loads(run_cli(path, "--steps", str(STEPS), "--model", "35b-a3b", "--json"))
        check(report["recipe"] == "qwen3_6_35b_a3b", "an MoE profile uses the MoE recipe by default")
        check("moe" in {r["family"] for r in report["families"]}, "and joins its expert kernels")


# ---------------------------------------------------------------------------------------------------
# Real captures: tests/fixtures/perf_model/. One rented RTX 3090 (power-capped at 220 W, so its absolute
# speed is far below a 315-350 W card), Qwen3.8-27B, `scripts/sweeps/decode-step-profile.sh` output.
# ---------------------------------------------------------------------------------------------------
FIXTURES = os.path.join(HERE, "..", "tests", "fixtures", "perf_model")
with open(os.path.join(FIXTURES, "rtx3090-vast-220w.json"), encoding="utf-8") as _handle:
    FACTS = json.load(_handle)
PROBE_READ_GB_S = FACTS["bandwidth_probe_gb_s"]["kernel_uint4_read_best"]


def real_join(csv_name):
    capture = FACTS["captures"][csv_name]
    geo = pm.Geometry.from_config(pm.PRESETS["27b"])
    weights = pm.build_weights(geo, pm.RECIPES["qwen3_8_27b"])
    step = pm.decode_step(geo, weights, tokens=capture["columns"], depth=capture["depth"])
    groups = pm.bytes_by_group(weights, geo, capture["columns"])
    groups["attention"] = step.kv
    groups["gdn"] = step.state_rw
    rows = kb.read_kernel_summary(pathlib.Path(FIXTURES, csv_name))
    return kb.join(rows, capture["steps"], groups, PROBE_READ_GB_S, 90.0), capture


PLAIN_CSV = "prof_27b-decode_cuda_gpu_kern_sum.csv"
MTP_CSV = "prof_27b-decode-mtp3_cuda_gpu_kern_sum.csv"


def test_real_kernel_names_all_land_in_known_families():
    for name in (PLAIN_CSV, MTP_CSV):
        joined, _ = real_join(name)
        by = {r["family"]: r for r in joined["families"]}
        check(set(by) <= {"q4", "q5", "q6", "q8", "gdn", "attention", "other"},
              f"{name}: unexpected families {set(by)}")
        for family in ("q4", "q5", "q6", "gdn", "attention"):
            check(family in by, f"{name}: no {family} kernels found")
        check(by["other"]["ms"] < 0.05 * joined["kernel_ms"],
              f"{name}: `other` is {100 * by['other']['ms'] / joined['kernel_ms']:.1f}% of kernel time; "
              f"biggest: {by['other']['top_kernel']}")
        check(joined["caveats"] == [], f"{name}: no fused-format kernel is expected in this capture")


def test_real_launch_counts_per_step_match_the_run():
    joined, _ = real_join(PLAIN_CSV)
    by = {r["family"]: r for r in joined["families"]}
    # The LM head runs once per step and the 16 full-attention layers once each: the proof that
    # --steps (128 for `-n 128`) is the right divisor. The one-token seed needs one extra head pass
    # (129 launches for 128 steps), hence a 1.5% tolerance rather than exactly 1.
    check(near(by["q6"]["launches"], 1.0, 0.015), f"head launches per step {by['q6']['launches']}")
    check(near(by["attention"]["launches"], 16.0, 0.015), f"attention launches {by['attention']['launches']}")
    joined, capture = real_join(MTP_CSV)
    by = {r["family"]: r for r in joined["families"]}
    check(near(by["q6"]["launches"], 1.0, 0.03), f"head launches per round {by['q6']['launches']}")
    check(capture["steps"] * pm.tokens_per_round(3, capture["acceptance"]) > 127.9,
          "48 rounds at 55.6% acceptance with 3 drafts produce the 128 tokens generated")


def test_real_kernel_time_matches_the_measured_throughput():
    joined, capture = real_join(PLAIN_CSV)
    check(near(joined["kernel_ms"], 1e3 / capture["tok_s_in_capture"], 0.03),
          f"plain: {joined['kernel_ms']:.2f} ms of kernels against {1e3 / capture['tok_s_in_capture']:.2f} ms/token")
    joined, capture = real_join(MTP_CSV)
    round_ms = capture["tokens_generated"] / capture["steps"] / capture["tok_s_in_capture"] * 1e3
    check(near(joined["kernel_ms"], round_ms, 0.03),
          f"MTP3: {joined['kernel_ms']:.2f} ms of kernels against a {round_ms:.2f} ms round")


def test_real_time_shares_track_byte_shares():
    # At a power cap the weight kernels all run at about the same fraction of the card's bandwidth
    # (here 54-57%), so a family's share of time follows its share of bytes. That is the model's
    # byte budget holding on a host it was never fitted on.
    joined, _ = real_join(PLAIN_CSV)
    by = {r["family"]: r for r in joined["families"]}
    total_bytes = sum(r["bytes"] for r in joined["families"] if r["bytes"])
    for family in ("q4", "q5", "q6"):
        time_share = by[family]["ms"] / joined["kernel_ms"]
        byte_share = by[family]["bytes"] / total_bytes
        check(abs(time_share - byte_share) < 0.06,
              f"{family}: {100 * time_share:.1f}% of time against {100 * byte_share:.1f}% of bytes")
    rates = [by[f]["pct"] for f in ("q4", "q5", "q6")]
    check(max(rates) - min(rates) < 6, f"the weight families should run at similar fractions: {rates}")


def test_calibrating_bandwidth_to_the_hosts_streaming_rate_reproduces_its_decode():
    joined, capture = real_join(PLAIN_CSV)
    measured = capture["tok_s_in_capture"]
    geo = pm.Geometry.from_config(pm.PRESETS["27b"])
    weights = pm.build_weights(geo, pm.RECIPES["qwen3_8_27b"])

    def predicted(bandwidth):
        hardware = pm.Hardware("test", bandwidth, 936.1, 316.0)
        return 1e3 / pm.decode_step(geo, weights, depth=capture["depth"], hardware=hardware).step_ms

    # The constants are fitted on a 315-350 W card, so the stock prediction is far too fast here ...
    check(predicted(pm.ACHIEVABLE_READ_GB_S) > 1.5 * measured, "stock model should overpredict a 220 W host")
    # ... and even the probe's own read rate does not help: it is a short, light kernel.
    check(predicted(PROBE_READ_GB_S) > 1.4 * measured, "the probe rate does not explain the decode either")
    # The weight kernels' achieved streaming rate, which the tool prints, does.
    got = predicted(joined["weights_gb_s"])
    check(near(got, measured, 0.06), f"calibrated prediction {got:.1f} against measured {measured}")


def test_speculation_speedup_is_the_same_on_a_slow_host():
    # tok/s with MTP3 over plain, measured here (60.76 / 27.60 = 2.20) and in docs/performance.md on a
    # 315 W card (107.1 / 49.0 = 2.19). The model says 2.32 on both, so +5-6% each time.
    geo = pm.Geometry.from_config(pm.PRESETS["27b"])
    weights = pm.build_weights(geo, pm.RECIPES["qwen3_8_27b"])
    plain = pm.decode_step(geo, weights)
    verify = pm.decode_step(geo, weights, tokens=4)
    bench = FACTS["ninfer_bench"]
    spec = pm.speculative(plain, verify, 3, bench["mtp3_acceptance"])
    measured = bench["mtp3_decode_tok_s"] / bench["plain_decode_tok_s"]
    check(near(spec["speedup"], measured, 0.08), f"model {spec['speedup']:.2f}x against measured {measured:.2f}x")


def test_a_round_costs_more_than_the_modelled_ratio_on_this_host():
    plain, _ = real_join(PLAIN_CSV)
    mtp, _ = real_join(MTP_CSV)
    ratio = mtp["kernel_ms"] / plain["kernel_ms"]
    # The model assumes 1.15x (measured on a 315 W card). Here 2.2 ms of a round is the drafter's q8
    # weights, which the byte model does not include, and the recurrence records cost more.
    check(1.15 < ratio < 1.30, f"round / plain step = {ratio:.3f}")
    q8 = {r["family"]: r for r in mtp["families"]}["q8"]
    check(q8["bytes"] is None and q8["ms"] > 1.5, "the drafter's q8 weights are visible as time with no bytes")


def test_shallow_attention_is_latency_bound_not_a_bandwidth_failure():
    joined, _ = real_join(PLAIN_CSV)
    attention = {r["family"]: r for r in joined["families"]}["attention"]
    check(attention["latency_bound"] and attention["pct"] is None and attention["recoverable_ms"] is None,
          "a few KB of KV in 0.36 ms must not print as '1% of peak' with the whole kernel recoverable")


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        check(kb.main(list(argv)) == 0, "main should return 0")
    return out.getvalue()


def test_cli_end_to_end_text_and_json():
    _, _, groups = model_bytes(depth=4096)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prof_27b-decode_cuda_gpu_kern_sum.csv")
        write_csv(path, [
            ("q5_a16_gemv_kernel", ms_for_rate(groups["q5"], 700) * 1e6 * STEPS, 90 * STEPS),
            ("q4_a16_gemv_kernel", ms_for_rate(groups["q4"], 780) * 1e6 * STEPS, 120 * STEPS),
            ("mystery_kernel", 2e6 * STEPS, 3 * STEPS),
        ])
        text = run_cli(path, "--steps", str(STEPS), "--model", "27b", "--depth", "4096")
        for needle in ("family", "q5", "q4", "other", "would save about"):
            check(needle in text, f"report is missing {needle!r}")
        check("above the card's read rate" not in text, "plausible rates must not be flagged")
        implausible = os.path.join(tmp, "implausible.csv")
        write_csv(implausible, [("q5_a16_gemv_kernel", ms_for_rate(groups["q5"], 2000) * 1e6 * STEPS, 1)])
        check("above the card's read rate" in run_cli(implausible, "--steps", str(STEPS)),
              "a family faster than the card can read must be flagged, not printed quietly")
        report = json.loads(run_cli(path, "--steps", str(STEPS), "--depth", "4096", "--json",
                                    "--target-pct", "80"))
        check(report["target_pct"] == 80.0 and report["steps"] == STEPS, "JSON carries the arguments")
        check(near(report["kernel_ms"], sum(r["ms"] for r in report["families"]), 1e-9), "JSON total")


def test_columns_scale_the_experts_a_speculative_round_touches():
    geo, weights, _ = model_bytes("35b-a3b", "qwen3_6_35b_a3b")
    one = pm.bytes_by_group(weights, geo, 1)
    four = pm.bytes_by_group(weights, geo, 4)
    check(four["moe"] > one["moe"], "a 4-column round reads more distinct experts")
    # q6 is only the output head in this recipe: a whole matrix, read once however many columns.
    check(four["q6"] == one["q6"], "dense weights are read once however many columns")


def test_a_file_that_is_not_an_nsys_summary_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        with open(path, "w") as handle:
            handle.write("a,b\n1,2\n")
        try:
            kb.read_kernel_summary(pathlib.Path(path))
        except SystemExit as error:
            check("cuda_gpu_kern_sum" in str(error), "the error should say what was expected")
        else:
            check(False, "a CSV without nsys columns must be refused")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("kernel_bandwidth: all checks passed")
