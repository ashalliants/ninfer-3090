"""Self-test of the DeepSeek V4.1 tools. Run: python -I tools/dsv4/test_dsv4.py

Covers the reference maths against hand-computed values and an independent formulation, the layout
arithmetic against numbers read from the real V4.1-Flash safetensors headers (2026-11-11), and the
golden fixture's consistency. No network, no weights.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_sizes as gs  # noqa: E402
import layout as lo  # noqa: E402
import profile_to_curve as pc  # noqa: E402
import reference as ref  # noqa: E402

# text_config of deepseek-ai/DeepSeek-V4.1-Flash, the fields the layout reads.
V41 = {"text_config": {"hidden_size": 5120, "moe_intermediate_size": 2304, "num_hidden_layers": 40,
                       "n_routed_experts": 384, "num_experts_per_tok": 6}}


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def test_fp4_table_and_nibble_order():
    # Byte 0x21: low nibble 1 -> 0.5 (even column), high nibble 2 -> 1.0 (odd column).
    got = ref.unpack_fp4(np.array([[0x21, 0xF7]], dtype=np.uint8))
    check(got.tolist() == [[0.5, 1.0, 6.0, -6.0]], f"nibble order: {got}")
    check(len(set(ref.FP4_TABLE[:8].tolist())) == 8, "positive codes are distinct")
    check(np.array_equal(ref.FP4_TABLE[8:], -ref.FP4_TABLE[:8]), "sign bit negates")


def test_pack_unpack_round_trip():
    codes = np.random.default_rng(1).integers(0, 16, size=(5, 64))
    again = ref.unpack_fp4(ref.pack_fp4(codes))
    check(np.array_equal(again, ref.FP4_TABLE[codes]), "pack/unpack")


def test_e8m0_is_a_bare_exponent():
    got = ref.e8m0_to_float(np.array([127, 128, 120, 0, 255], dtype=np.uint8))
    check(got[:3].tolist() == [1.0, 2.0, 2.0 ** -7], f"{got}")
    check(got[3] == 2.0 ** -127 and np.isnan(got[4]), "ends: 2^-127 and NaN")


def test_dequant_scales_each_block_of_32():
    packed = np.full((1, 32), 0x22, dtype=np.uint8)  # 64 weights of 1.0
    scale = np.array([[127, 129]], dtype=np.uint8)  # blocks of 32: x1 and x4
    w = ref.dequant_fp4(packed, scale)
    check(w[0, :32].tolist() == [1.0] * 32 and w[0, 32:].tolist() == [4.0] * 32, "block scales")
    try:
        ref.dequant_fp4(packed, np.zeros((1, 3), dtype=np.uint8))
    except ValueError:
        return
    check(False, "a wrong scale shape must be refused")


def test_e4m3_decode_known_values():
    # 0x38 = exp 7 man 0 -> 1.0; 0x40 -> 2.0; 0x7E -> 448 (max); 0x01 -> 2^-9 (smallest subnormal);
    # 0xB8 -> -1.0; 0x7F is NaN.
    got = ref.decode_e4m3(np.array([0x38, 0x40, 0x7E, 0x01, 0xB8, 0x7F, 0x00], dtype=np.uint8))
    check(got[:5].tolist() == [1.0, 2.0, 448.0, 2.0 ** -9, -1.0], f"{got}")
    check(np.isnan(got[5]) and got[6] == 0.0, "NaN and zero")


def test_expert_matches_an_independent_loop():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w1, w3 = (rng.standard_normal((4, 8)).astype(np.float32) for _ in range(2))
    w2 = rng.standard_normal((8, 4)).astype(np.float32)
    got = ref.expert_forward(x, w1, w2, w3, swiglu_limit=1.0)
    for t in range(2):
        h = np.zeros(4)
        for j in range(4):
            g, u = float(x[t] @ w1[j]), float(x[t] @ w3[j])
            g, u = min(g, 1.0), max(-1.0, min(u, 1.0))
            h[j] = g / (1 + np.exp(-g)) * u
        check(np.allclose(got[t], h @ w2.T, atol=1e-5), f"token {t}")


def test_swiglu_clamps_are_asymmetric():
    # Up is clamped both sides, gate only from above: a very negative gate is NOT clamped.
    x = np.ones((1, 1), dtype=np.float32)
    big = np.array([[100.0]], dtype=np.float32)
    out_pos = ref.expert_forward(x, big, np.ones((1, 1), np.float32), big, swiglu_limit=10.0)
    check(np.isclose(out_pos[0, 0], 10.0 / (1 + np.exp(-10.0)) * 10.0), f"{out_pos}")
    out_neg_gate = ref.expert_forward(x, -big, np.ones((1, 1), np.float32), -big, swiglu_limit=10.0)
    # gate -100 (unclamped, silu ~ 0), up -100 clamps to -10 -> output ~ 0 from below
    check(abs(out_neg_gate[0, 0]) < 1e-30, f"{out_neg_gate}")


def test_gate_bias_steers_selection_but_not_weights():
    x = np.array([[1.0, 0.0]], dtype=np.float32)
    weight = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    # Unbiased, expert 2 scores highest. A big bias on expert 0 makes it the pick instead.
    w, idx = ref.gate_forward(x, weight, np.zeros(3, np.float32), topk=1, route_scale=1.0, norm_topk_prob=False)
    check(idx[0, 0] == 2, "highest score wins without bias")
    w2, idx2 = ref.gate_forward(x, weight, np.array([10.0, 0.0, 0.0], np.float32), 1, 1.0, False)
    check(idx2[0, 0] == 0, "bias steers selection")
    check(np.isclose(w2[0, 0], np.sqrt(np.log(2.0))), f"weight is the UNBIASED sqrt-softplus score: {w2}")


def test_gate_weights_normalise_to_the_route_scale():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((6, 16)).astype(np.float32)
    w, idx = ref.gate_forward(x, rng.standard_normal((12, 16)).astype(np.float32), np.zeros(12, np.float32), 4, 1.5)
    check(np.allclose(w.sum(-1), 1.5, atol=1e-5), f"{w.sum(-1)}")
    check(all(len(set(row)) == 4 for row in idx.tolist()), "distinct experts per token")


def test_engram_row_addressing():
    check(ref.engram_row_offsets(10, 256) == (2560, 80), "row 10 of a 256-wide table")
    table = np.full((3, 64), 0x38, dtype=np.uint8)  # all 1.0
    scale = np.array([[127, 127], [128, 128], [126, 126]], dtype=np.uint8)
    check(ref.engram_row(table, scale, 1).tolist() == [2.0] * 64, "row 1 scale x2")
    check(ref.engram_row(table, scale, 2).tolist() == [0.5] * 64, "row 2 scale x0.5")


def test_golden_fixture_is_deterministic_and_self_consistent():
    a, b = ref.golden(), ref.golden()
    check(a == b, "same seed, same fixture")
    w = {k: ref.dequant_fp4(np.array(a[k]["packed"], np.uint8), np.array(a[k]["scale"], np.uint8)) for k in ("w1", "w2", "w3")}
    x, route = np.array(a["x"], np.float32), np.array(a["route_weight"], np.float32)
    again = ref.expert_forward(x, w["w1"], w["w2"], w["w3"], a["swiglu_limit"], route)
    check(np.allclose(again, a["expert_out"]), "expert_out reproduces")
    check(np.allclose(np.array(a["expert_out"]), route * np.array(a["expert_out_unweighted"]), rtol=1e-4, atol=1e-6),
          "routing weight scales linearly through the down projection")


def test_committed_golden_fixture_is_current():
    path = Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "golden.json"
    committed = json.loads(path.read_text())
    check(committed == json.loads(json.dumps(ref.golden())), f"{path} is stale: rerun reference.py --emit-golden")


def test_expert_size_matches_the_real_v41_headers():
    es = lo.expert_size(V41)
    # Read from the headers: w1/w3 [2304, 2560] i8 and w2 [5120, 1152] i8, scales [2304, 160] / [5120, 72].
    check(es.fp4_data_bytes == 2 * 2304 * 2560 + 5120 * 1152, f"{es.fp4_data_bytes}")
    check(es.fp4_scale_bytes == 2 * 2304 * 160 + 5120 * 72, f"{es.fp4_scale_bytes}")
    check(es.fp4_bytes == 17_694_720 + 1_105_920, f"{es.fp4_bytes}")
    # The whole routed-expert group in the real checkpoint is 288.778 GB.
    check(round(es.fp4_bytes * 40 * 384 / 1e9, 3) == 288.778, f"{es.fp4_bytes * 40 * 384 / 1e9}")


def test_gguf_sizes_from_an_allocation_list():
    alloc = "\n".join([
        "# comment", "output.weight: Q5_K", "token_embd.weight: Q5_K", "per_layer_token_embd.weight: IQ4_NL",
        "blk.0.ffn_gate_exps.weight: IQ2_XXS", "blk.0.ffn_up_exps.weight: IQ2_XXS", "blk.0.ffn_down_exps.weight: IQ1_M",
        "blk.0.attn_q_a.weight: Q8_0", "blk.0.attn_norm.weight: F32",
    ])
    s = gs.summarize(gs.parse_allocation(alloc))
    one_matrix = 2304 * 5120
    expect = one_matrix * 2.0625 / 8 * 2 + one_matrix * 1.75 / 8
    check(abs(s["expert_bytes_by_layer"][0] - expect) < 1e-6, f"{s['expert_bytes_by_layer']}")
    check(s["engram_table_type"] == "IQ4_NL", "engram table is reported, not sized")
    # Dense = output head + attn_q_a; the embedding is a gather, kept out; norms are the flat allowance.
    check(abs(s["dense_resident_bytes"] - (129280 * 5120 * 5.5 / 8 + 1280 * 5120 * 8.5 / 8)) < 1e-6, f"{s}")
    check(gs.tensor_bytes("blk.3.attn_norm.weight", "F32") is None, "unmapped tensors are not sized")


def test_the_committed_routing_curve_matches_its_source_statistics():
    curve = json.loads((Path(HERE) / "data" / "routing_rank_curve.json").read_text())
    r = curve["rank_share"]
    check(curve["experts"] == 384 and len(r) == 384 and abs(sum(r) - 1.0) < 1e-9, "384 shares summing to 1")
    check(all(a >= b for a, b in zip(r, r[1:])), "ranked")
    check(abs(sum(r[:38]) - 0.318) < 0.002, f"top 10% carries {sum(r[:38]):.4f}")  # Strata profile-2.00: 31.8%
    check(len(curve["top10pct_mass_by_layer"]) == 40 and "Strata" in curve["source"], "40 layers, attributed")


def test_rank_curve_from_counts():
    counts = np.array([[1.0, 3.0, 6.0], [10.0, 0.0, 10.0]])
    c = pc.rank_curve(counts)
    # Layer 0 shares sorted: .6 .3 .1; layer 1: .5 .5 0. Mean per rank: .55 .4 .05.
    check(np.allclose(c["rank_share"], [0.55, 0.4, 0.05]), c)
    try:
        pc.rank_curve(np.array([[0.0, 0.0]]))
    except ValueError:
        return
    check(False, "a layer with no routing counts must be refused")


def test_group_names():
    cases = {
        "layers.7.ffn.experts.300.w2.weight": "routed_experts",
        "layers.7.ffn.experts.300.w2.scale": "routed_experts",
        "mtp.1.ffn.experts.2.w1.weight": "mtp_routed_experts",
        "layers.14.engram.embed.weight": "engram_tables",
        "layers.14.engram.wkv.weight": "engram_projection",
        "layers.7.ffn.shared_experts.w1.weight": "shared_experts",
        "layers.7.ffn.gate.bias": "router",
        "layers.7.attn.wq_b.weight": "attention",
        "layers.7.hc_attn_fn": "other_dense",
        "embed.weight": "embed_and_head",
        "vision.blocks.0.norm1.weight": "vision",
        "mtp.0.attn.wq_a.weight": "mtp_dense",
    }
    for name, group in cases.items():
        check(lo.group_of(name) == group, f"{name}: {lo.group_of(name)} != {group}")


def test_tier_split():
    s = lo.split_tiers(100, 10, vram_bytes=250, ram_bytes=400)
    check((s.vram, s.ram, s.ssd) == (25, 40, 35), f"{s}")
    s = lo.split_tiers(100, 10, vram_bytes=5000, ram_bytes=5000)
    check((s.vram, s.ram, s.ssd) == (100, 0, 0), "everything fits in VRAM")
    check(abs(sum(lo.split_tiers(15360, 10_230_000, 17 << 30, 46 << 30).fractions()) - 1.0) < 1e-12, "fractions sum to 1")


def test_report_cross_checks_config_against_headers():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "headers").mkdir()
        (d / "config.json").write_text(json.dumps(V41))
        cursor = 0
        header = {}
        for w, (o, i) in {"w1": (2304, 5120), "w3": (2304, 5120), "w2": (5120, 2304)}.items():
            for kind, shape, size in (("weight", [o, i // 2], o * i // 2), ("scale", [o, i // 32], o * i // 32)):
                header[f"layers.3.ffn.experts.0.{w}.{kind}"] = {
                    "dtype": "I8", "shape": shape, "data_offsets": [cursor, cursor + size]}
                cursor += size
        (d / "headers" / "a.json").write_text(json.dumps(header))
        text = lo.report(d, (17, 46), None)
        check("18.80 MB native" in text and "routed_experts" in text, text)
        header["layers.3.ffn.experts.0.w1.scale"]["data_offsets"][1] += 1  # now disagrees with the config
        (d / "headers" / "a.json").write_text(json.dumps(header))
        try:
            lo.report(d, None, None)
        except AssertionError:
            return
        check(False, "a config/header size disagreement must be an error")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
