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
    check(deep.kv == 32768 * m.kv_bytes_per_token(geo, "int8"), "KV bytes should be depth x per-token")
    check(deep.step_ms > shallow.step_ms, "a deeper cache must cost time")
    bf16 = m.decode_step(geo, weights, depth=32768, kv="bf16")
    check(near(bf16.kv / deep.kv, 2 / (33 / 32), 1e-9), "BF16 KV is 2 bytes per element")
    fp16_state = m.decode_step(geo, weights, gdn_state_bytes_per=2)
    check(near(fp16_state.state_rw, shallow.state_rw / 2, 1e-9), "FP16 GDN state halves the traffic")


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


def test_recipes_still_exist_in_the_converter():
    # The recipes here mirror tools/convert/official_recipes.py. A rename there should fail loudly.
    source = open(os.path.join(HERE, "convert", "official_recipes.py"), encoding="utf-8").read()
    for name in m.RECIPES:
        check(re.search(rf"^def {name}\(", source, re.MULTILINE), f"{name} is gone from official_recipes.py")


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        check(m.main(list(argv)) == 0, "main should return 0")
    return out.getvalue()


def test_cli_text_and_json():
    text = run_cli("--model", "27b", "--spec-k", "3", "--tok-s", "49.0", "--depth", "4096")
    for needle in ("DECODE one token", "bandwidth floor", "speculative k=3", "MEASURED 49.0", "PREFILL"):
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
