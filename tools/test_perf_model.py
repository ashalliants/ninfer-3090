"""Self-test of perf_model.py. Run: python -I tools/test_perf_model.py

The model is only worth having if it reproduces what the repository has already measured, so these
checks pin it to three kinds of evidence:

  * byte counts the artifact-header tool (`decode_byte_accounting.py`) printed for real artifacts;
  * the RTX 3090 rows of docs/performance.md (decode, prefill, MTP3 speculation);
  * the Engine's own fitted prefill cost table, through `tools/bench/prefill_time_model.py`.

Two calibration constants are fitted (per-layer decode overhead; prefill efficiency and step cost),
so a tight match on the fitted points proves little. The held-out points are marked below.
"""
import contextlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bench"))
import perf_model as m  # noqa: E402
import prefill_time_model as fitted  # noqa: E402


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def near(value, expected, tolerance):
    return abs(value - expected) <= tolerance * abs(expected)


def model(preset, recipe, overrides=None):
    geo = m.Geometry.from_config(m.PRESETS[preset])
    return geo, m.build_weights(geo, m.RECIPES[recipe], overrides)


def test_dense_bytes_match_the_artifact_header_count():
    # decode_byte_accounting.py, Qwen3.8-27B dense, 2026-09-09: 15.743 GB per token. That artifact had
    # a Q8 output head; today's recipes use Q6, so the head is overridden to reproduce it.
    geo, weights = model("27b", "qwen3_6_27b", {"output_head": "q8"})
    step = m.decode_step(geo, weights)
    check(near(step.weights, 15.743e9, 5e-4), f"27B weights per token {step.weights / 1e9:.4f} GB")
    check(near(sum(w.params for w in weights), 26.9e9, 0.01), "27B should have 26.9B parameters")


def test_moe_bytes_match_the_artifact_header_count():
    # Same tool, Qwen3.6-35B-A3B: 1.943 GB dense + 0.580 GB of routed experts read (8 of 256), out
    # of 18.556 GB of experts resident; 2.522 GB per token.
    geo, weights = model("35b-a3b", "qwen3_6_35b_a3b")
    step = m.decode_step(geo, weights)
    routed = sum(b for role, b in step.by_role.items() if role.startswith("experts."))
    resident = sum(w.resident_bytes for w in weights if w.experts)
    check(near(resident, 18.556e9, 1e-3), f"routed experts resident {resident / 1e9:.3f} GB")
    check(near(routed, 0.580e9, 2e-3), f"routed experts read {routed / 1e9:.3f} GB")
    check(near(step.weights - routed, 1.943e9, 1e-3), "dense part of the MoE token")
    check(near(step.weights, 2.522e9, 1e-3), f"MoE weights per token {step.weights / 1e9:.4f} GB")
    check(m.kv_bytes_per_token(geo, "int8") == 10560, "the 35B int8 KV figure that run passed in")


def test_plain_decode_matches_the_3090_measurements():
    # docs/performance.md, ninfer_bench, no speculation. Both points fit the one per-layer constant.
    geo, weights = model("27b", "qwen3_8_27b")
    tok_s = 1e3 / m.decode_step(geo, weights).step_ms
    check(near(tok_s, 49.0, 0.03), f"27B plain decode {tok_s:.1f} tok/s, measured 49.0")
    geo, weights = model("35b-a3b", "qwen3_6_35b_a3b")
    tok_s = 1e3 / m.decode_step(geo, weights).step_ms
    check(near(tok_s, 204.5, 0.03), f"35B plain decode {tok_s:.1f} tok/s, measured 204.5")


def test_decode_slows_with_depth_by_the_kv_bytes():
    geo, weights = model("27b", "qwen3_8_27b")
    shallow = m.decode_step(geo, weights, depth=0)
    deep = m.decode_step(geo, weights, depth=32768)
    check(deep.kv == (32768 + 1) * m.kv_bytes_per_token(geo, "int8"),
          "a plain step attends over the cache plus its own entry")
    check(deep.step_ms > shallow.step_ms, "a deeper cache must cost time")
    bf16 = m.decode_step(geo, weights, depth=32768, kv="bf16")
    check(near(bf16.kv / deep.kv, 2 / (33 / 32), 1e-9), "BF16 KV is 2 bytes per element")
    fp16_state = m.decode_step(geo, weights, gdn_state_bytes_per=2)
    check(near(fp16_state.state_rw, shallow.state_rw / 2, 1e-9), "FP16 GDN state halves the traffic")


def test_a_verification_reads_its_own_entries_but_not_one_pass_per_column():
    # Review finding (codepulse, PR #438): the byte model ignored the K/V entries a round appends before
    # it attends. The reviewer's proposed fix, scaling by every column's own context, would be wrong:
    # the small-T attention kernel loads each K/V tile once for all columns (a real 4-column MTP3
    # capture launches attention 19 times over 16 layers, not 64). The right count is depth + columns.
    geo, weights = model("27b", "qwen3_8_27b")
    per_token = m.kv_bytes_per_token(geo, "int8")
    depth = 8192
    plain = m.decode_step(geo, weights, depth=depth)
    verify = m.decode_step(geo, weights, tokens=4, depth=depth)
    check(plain.kv == (depth + 1) * per_token, "plain: the cache plus its own entry")
    check(verify.kv == (depth + 4) * per_token, "a 4-column round attends over depth + 4 entries")
    check(verify.kv - plain.kv == 3 * per_token, "three entries more than a plain step, and no more")
    check(verify.kv < 1.01 * plain.kv, "the columns share each K/V read: not four passes over the cache")
    check(m.decode_step(geo, weights).kv == per_token, "even at depth 0 a step reads its own entry")
    # And the table still accounts for every byte, kv row included.
    report = json.loads(run_cli("--depth", "0", "--json"))
    check(near(sum(r["share"] for r in report["decode"][0]["table"]), 1.0, 1e-9), "shares sum to 1 at depth 0")


def test_speculation_matches_the_3090_mtp3_measurement():
    # 27B, MTP3, INT8 KV, ninfer_bench decode 128: 107.1 tok/s at 55.6% acceptance (docs/performance.md,
    # the INT8-vs-rk4v4 table). The 1.15x round cost is itself from the 27B, so this is a sanity check
    # of how the pieces compose, not an independent validation.
    geo, weights = model("27b", "qwen3_8_27b")
    plain = m.decode_step(geo, weights)
    verify = m.decode_step(geo, weights, tokens=4)
    spec = m.speculative(plain, verify, k=3, acceptance=0.556)
    check(near(spec["tokens_per_round"], 1 + 0.556 * 3, 1e-12), "tokens per round = 1 + acceptance x k")
    check(near(spec["tok_s"], 107.1, 0.08), f"MTP3 prediction {spec['tok_s']:.1f} tok/s, measured 107.1")
    check(spec["speedup"] > 1.0, "speculation at 55% acceptance should help a bandwidth-bound decode")
    zero = m.speculative(plain, verify, k=3, acceptance=0.0)
    check(zero["speedup"] < 1.0, "with no accepted drafts the verify round is pure overhead")


def test_moe_verify_reads_more_experts_but_never_more_than_all():
    geo, weights = model("35b-a3b", "qwen3_6_35b_a3b")
    check(near(m.expert_fraction(256, 8, 1), 8 / 256, 1e-12), "one token touches top_k / experts")
    previous = 0.0
    for tokens in (1, 2, 4, 8, 64, 4096):
        fraction = m.expert_fraction(256, 8, tokens)
        check(previous < fraction <= 1.0, f"expert fraction at T={tokens}: {fraction}")
        previous = fraction
    one = m.decode_step(geo, weights, tokens=1)
    four = m.decode_step(geo, weights, tokens=4)
    check(four.weights > one.weights, "a 4-column verify must read more experts than one token")
    dense_geo, dense_weights = model("27b", "qwen3_8_27b")
    one = m.decode_step(dense_geo, dense_weights, tokens=1).by_role
    four = m.decode_step(dense_geo, dense_weights, tokens=4).by_role
    check({r: b for r, b in four.items() if r != "embedding"}
          == {r: b for r, b in one.items() if r != "embedding"},
          "a dense verify streams the same weights as one token; only the embedding rows grow")
    check(near(four["embedding"], 4 * one["embedding"], 1e-12), "one embedding row per column")


def test_prefill_matches_the_3090_measurements():
    # docs/performance.md ninfer_bench prefill rows. 512 and 16,384 fit the two constants; 4,096 is
    # held out.
    rows = {
        ("27b", "qwen3_8_27b"): {512: 2057, 4096: 3143, 16384: 2893},
        ("35b-a3b", "qwen3_6_35b_a3b"): {512: 5479, 4096: 8468, 16384: 7749},
    }
    for (preset, recipe), measured in rows.items():
        geo, weights = model(preset, recipe)
        for prompt, tok_s in measured.items():
            got = m.prefill(geo, weights, prompt)["tok_s"]
            check(near(got, tok_s, 0.02), f"{preset} prefill {prompt}: {got:.0f} vs measured {tok_s}")


def test_prefill_agrees_with_the_engines_fitted_cost_table():
    # An independent fit, made by a different route (ninfer_context_cost_bench), so agreement means
    # the first-principles operation count and the Engine's pricing describe the same machine.
    geo, weights = model("27b", "qwen3_8_27b")
    cost = fitted.load_cost("27b")
    # 65,536 and 131,072 tokens are far beyond anything either side was fitted on.
    for prompt in (512, 4096, 16384, 65536, 131072):
        engine = fitted.prefill_profile(cost, prompt, 4096)["tokens_per_second"]
        ours = m.prefill(geo, weights, prompt)["tok_s"]
        check(near(ours, engine, 0.03), f"prefill {prompt}: model {ours:.0f} vs Engine fit {engine:.0f}")


def test_prefill_is_never_above_the_tensor_peak():
    for preset, recipe in (("27b", "qwen3_8_27b"), ("35b-a3b", "qwen3_6_35b_a3b")):
        geo, weights = model(preset, recipe)
        p = m.prefill(geo, weights, 8192)
        check(p["tok_s"] < p["ceiling_tok_s"], f"{preset}: predicted rate must sit below the peak")


def test_format_what_if_changes_exactly_the_bytes_it_should():
    geo, base = model("27b", "qwen3_8_27b")
    _, narrow = model("27b", "qwen3_8_27b", {"output_head": "q4"})
    saved = (m.decode_step(geo, base).weights - m.decode_step(geo, narrow).weights)
    expected = geo.vocab * geo.hidden * (m.FORMAT_BITS["q6"] - m.FORMAT_BITS["q4"]) / 8
    check(near(saved, expected, 1e-9), "q6 -> q4 head should save vocab x hidden x 2 bits")
    try:
        model("27b", "qwen3_8_27b", {"no.such.role": "q4"})
    except SystemExit as error:
        check("no.such.role" in str(error), "the error should name the bad role")
    else:
        check(False, "an unknown role must be refused, not silently ignored")
    try:
        m.parse_format("output_head=q3")
    except SystemExit:
        pass
    else:
        check(False, "an unknown format must be refused")


def levers(preset, recipe, overrides=None, **kwargs):
    geo = m.Geometry.from_config(m.PRESETS[preset])
    return {row["name"]: row for row in m.rank_levers(geo, m.RECIPES[recipe], overrides or {}, **kwargs)}


def test_levers_agree_with_the_measured_quality_trades():
    # docs/maintainer/quality-trade-experiments.md: with the head at q4, plain decode moved +1.4% and
    # +2.9% in two interleaved reps, "close to the naive bandwidth-share prediction". The FP16 GDN state
    # was inside the C1 noise and +2.0% at C8.
    found = levers("27b", "qwen3_6_27b")
    head = found["output head -> q4"]["gain_pct"]
    check(1.0 < head < 3.5, f"q4 head predicted {head:+.2f}%, measured +1.4% to +2.9%")
    state = found["GDN state in FP16"]["gain_pct"]
    check(0.0 < state < head, f"FP16 state should help less than the head here, got {state:+.2f}%")
    check("perplexity" in found["output head -> q4"]["evidence"], "measured levers cite their evidence")


def test_levers_are_ranked_and_the_kernel_ones_nest():
    found = m.rank_levers(m.Geometry.from_config(m.PRESETS["27b"]), m.RECIPES["qwen3_8_27b"], {})
    gains = [row["gain_pct"] for row in found]
    check(gains == sorted(gains, reverse=True), "levers must be ranked best first")
    by = {row["name"]: row["gain_pct"] for row in found}
    check(by["kernels: 25% less per-layer overhead"] < by["kernels: 50% less per-layer overhead"]
          < by["kernels: bandwidth floor (no overhead)"], "more kernel work must be worth more")
    check(by["every matrix -> q4"] > by["MLP weights -> q4"] > 0, "narrower sets save less")
    # The 3090 recipe keeps mlp.down at q5 in layers 0-35 only; the lever must still see it.
    check("MLP weights -> q4" in by, "a role with mixed formats across layers still has bytes to save")


def test_a_lever_with_nothing_left_to_save_is_left_out():
    already = levers("27b", "qwen3_8_27b", {"output_head": "q4"})
    check("output head -> q4" not in already, "a head already at q4 has nothing to gain")
    check("kernels: bandwidth floor (no overhead)" in already, "kernel levers always apply")
    fp16 = levers("27b", "qwen3_8_27b", gdn_state_bytes_per=2)
    check("GDN state in FP16" not in fp16, "state already FP16")
    moe = levers("35b-a3b", "qwen3_6_35b_a3b")
    check("routed experts -> q4" in moe and "MLP weights -> q4" not in moe,
          "an MoE has experts to shrink and no dense MLP")


def test_config_json_matches_the_preset():
    preset = m.PRESETS["27b"]
    nested = {"architectures": ["Qwen3_5ForCausalLM"], "text_config": dict(preset)}
    by_interval = m.Geometry.from_config(nested)
    explicit = dict(preset)
    explicit["layer_types"] = ["full_attention" if i % 4 == 3 else "linear_attention"
                               for i in range(preset["num_hidden_layers"])]
    check(m.Geometry.from_config(explicit) == by_interval == m.Geometry.from_config(preset),
          "layer_types, the default interval and a nested text_config must agree")
    check(by_interval.full_layers == 16, "the 27B has 16 full-attention layers")
    bad = dict(explicit)
    bad["layer_types"] = bad["layer_types"][:-1]
    try:
        m.Geometry.from_config(bad)
    except SystemExit:
        pass
    else:
        check(False, "a short layer_types array must be refused")


def test_single_family_configs_load_without_the_other_familys_fields():
    # Review finding: both mixer families' keys were read unconditionally. src/models/qwen3_5/config.cpp
    # requires a family's fields only when a layer uses it.
    full_only = dict(m.PRESETS["27b"])
    for key in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                "linear_value_head_dim", "linear_conv_kernel_dim"):
        del full_only[key]
    full_only["layer_types"] = ["full_attention"] * full_only["num_hidden_layers"]
    geo = m.Geometry.from_config(full_only)
    check(geo.full_layers == geo.layers and geo.gdn_value_heads == 0, "an all-attention model")
    weights = m.build_weights(geo, m.RECIPES["qwen3_8_27b"])
    check(not any(w.role.startswith("gdn.") for w in weights), "no GDN tensors without GDN layers")
    check(m.decode_step(geo, weights).state_rw == 0, "no recurrent state without GDN layers")

    linear_only = dict(m.PRESETS["27b"])
    for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        del linear_only[key]
    linear_only["layer_types"] = ["linear_attention"] * linear_only["num_hidden_layers"]
    geo = m.Geometry.from_config(linear_only)
    check(geo.full_layers == 0 and m.kv_bytes_per_token(geo, "int8") == 0, "an all-GDN model has no KV")
    check(m.prefill(geo, m.build_weights(geo, m.RECIPES["qwen3_8_27b"]), 1024)["attention_s"] == 0.0,
          "and no attention work in prefill")

    # A family that is used must still define its fields, and the error must say which.
    missing = dict(m.PRESETS["27b"])
    del missing["linear_conv_kernel_dim"]
    try:
        m.Geometry.from_config(missing)
    except SystemExit as error:
        check("linear_conv_kernel_dim" in str(error), "the error should name the missing field")
    else:
        check(False, "a used family with missing fields must be refused")


def test_unknown_layer_labels_are_refused_not_reclassified():
    typo = dict(m.PRESETS["27b"])
    typo["layer_types"] = ["full_atention"] * typo["num_hidden_layers"]
    try:
        m.Geometry.from_config(typo)
    except SystemExit as error:
        check("full_atention" in str(error), "the error should name the bad label")
    else:
        check(False, "a mistyped layer type must not silently become a GDN layer")


def test_the_default_recipe_follows_the_model_and_a_mismatch_is_refused():
    # Review finding: --model 35b-a3b with the dense default succeeded with wrong weights.
    dense = m.Geometry.from_config(m.PRESETS["27b"])
    moe = m.Geometry.from_config(m.PRESETS["35b-a3b"])
    check(m.choose_recipe(dense, None) == "qwen3_8_27b", "dense default")
    check(m.choose_recipe(moe, None) == "qwen3_6_35b_a3b", "MoE default")
    check(m.choose_recipe(moe, "qwen3_6_35b_a3b") == "qwen3_6_35b_a3b", "a matching explicit recipe")
    for geo, recipe in ((moe, "qwen3_8_27b"), (dense, "qwen3_6_35b_a3b")):
        try:
            m.choose_recipe(geo, recipe)
        except SystemExit as error:
            check(recipe in str(error), "the error should name the recipe")
        else:
            check(False, f"{recipe} on the wrong family must be refused")
    # End to end: no --recipe on the MoE preset gives the MoE's bytes, not a dense recipe's fallback.
    report = json.loads(run_cli("--model", "35b-a3b", "--json"))
    check(report["recipe"] == "qwen3_6_35b_a3b", "the report names the recipe it used")
    check(near(report["decode"][0]["total_gb"], 2.648, 1e-3), "and the MoE token is 2.648 GB")


def test_numeric_options_are_validated_at_the_command_line():
    import io
    # Review finding: --tok-s 0, --chunk 0 and --prompt 0 divided by zero; --depth -5 was accepted;
    # --bandwidth-gb-s 0 silently fell back to the default card.
    bad = (["--tok-s", "0"], ["--tok-s", "-1"], ["--tok-s", "nan"], ["--chunk", "0"], ["--chunk", "-4"],
           ["--prompt", "0"], ["--depth", "-5"], ["--bandwidth-gb-s", "0"], ["--bandwidth-gb-s", "inf"],
           ["--int8-tops", "-3"], ["--spec-k", "-1"], ["--acceptance", "1.5"], ["--top", "0"],
           ["--kv-bytes-per-token", "-1"], ["--depth", "1.5"])
    for argv in bad:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                m.parse_args(argv)
        except SystemExit:
            continue
        check(False, f"{argv} must be refused by the parser")
    ok = m.parse_args(["--depth", "0", "--acceptance", "0", "--spec-k", "0", "--kv-bytes-per-token", "0"])
    check(ok.depth == [0] and ok.acceptance == 0.0, "zero is a valid depth, acceptance and draft count")
    # A zero depth is honoured, not replaced by a default.
    report = json.loads(run_cli("--depth", "0", "--json"))
    check(report["decode"][0]["depth"] == 0, "depth 0 stays 0")


def test_overridden_hardware_is_not_reported_as_the_default_card():
    stock = json.loads(run_cli("--json"))
    check(stock["hardware"] == "RTX 3090", "the default card keeps its name")
    custom = json.loads(run_cli("--json", "--bandwidth-gb-s", "1008"))
    check("custom" in custom["hardware"] and custom["read_gb_s"] == 1008.0,
          f"overridden rates must say so: {custom['hardware']!r}")
    check(custom["int8_tops"] == stock["int8_tops"], "the other rate keeps its default")
    check("custom" in json.loads(run_cli("--json", "--int8-tops", "500"))["hardware"], "either flag")


def test_recipes_still_exist_in_the_converter():
    # The recipes here mirror tools/convert/official_recipes.py. A rename there should fail loudly.
    with open(os.path.join(HERE, "convert", "official_recipes.py"), encoding="utf-8") as handle:
        source = handle.read()
    for name in m.RECIPES:
        check(re.search(rf"^def {name}\(", source, re.MULTILINE), f"{name} is gone from official_recipes.py")


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        check(m.main(list(argv)) == 0, "main should return 0")
    return out.getvalue()


def test_cli_text_and_json():
    text = run_cli("--model", "27b", "--spec-k", "3", "--tok-s", "49.0", "--depth", "4096", "--levers")
    for needle in ("DECODE one token", "bandwidth floor", "speculative k=3", "MEASURED 49.0", "PREFILL",
                   "LEVERS ranked", "output head -> q4"):
        check(needle in text, f"report is missing {needle!r}")
    report = json.loads(run_cli("--model", "35b-a3b", "--recipe", "qwen3_6_35b_a3b", "--json",
                                "--tok-s", "204.5", "--prompt", "512", "--prompt", "4096"))
    check(report["moe"] and len(report["prefill"]) == 2, "JSON should describe the MoE and both prompts")
    check(abs(report["measured"]["model_error_pct"]) < 3, "JSON should carry the measured comparison")
    shares = sum(row["share"] for row in report["decode"][0]["table"])
    check(near(shares, 1.0, 1e-9), "the byte table must account for every byte")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("perf_model: all checks passed")
