"""Self-test of the DeepSeek V4.1 tools. Run: python -I tools/dsv4/test_dsv4.py

Covers the reference maths against hand-computed values and an independent formulation, the layout
arithmetic against numbers read from the real V4.1-Flash safetensors headers (2026-11-11), and the
golden fixture's consistency. No network, no weights.
"""
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention as att  # noqa: E402
import engram_ref as eng  # noqa: E402
import gguf_headers as gh  # noqa: E402
import hyper_connection as hcr  # noqa: E402
import model_golden as mg  # noqa: E402


import moe as moe_ref  # noqa: E402
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


def _same(a, b, rtol=1e-4, atol=1e-6):
    """Equal, with floats compared to a tolerance: float32 matmul and tanh differ in the last bits
    between NumPy/BLAS builds, while the packed bytes, shapes and seeds must match exactly."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def test_committed_golden_fixture_is_current():
    path = Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "golden.json"
    committed = json.loads(path.read_text())
    check(_same(committed, json.loads(json.dumps(ref.golden()))), f"{path} is stale: rerun reference.py --emit-golden")


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


def test_truncated_profile_is_refused():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profile.bin"
        np.ones((3, 8), dtype="<u4").tofile(path)
        try:
            pc.load_counts(path, 8, 5)
        except ValueError:
            pass
        else:
            check(False, "3 layers of counts must not pass for 5 expected")
        check(pc.load_counts(path, 8, 3).shape == (3, 8), "a complete profile loads")


def test_fetch_refuses_a_directory_for_another_repo():
    import fetch_headers as fh
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "repo.txt").write_text("someone/else\n")
        try:
            fh.fetch("deepseek-ai/other", Path(d), None, 1)
        except SystemExit:
            return
        check(False, "a cache stamped for another repository must be refused")


def test_fetch_refuses_an_unstamped_non_empty_directory():
    import fetch_headers as fh
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text("{}")
        try:
            fh.fetch("deepseek-ai/other", Path(d), None, 1)
        except SystemExit:
            check(not (Path(d) / "repo.txt").exists(), "a refused directory must not get a stamp")
            return
        check(False, "an unstamped directory with cached files must be refused")


def test_fetch_pins_one_commit_and_reuses_it():
    import fetch_headers as fh
    sha, other = "a" * 40, "b" * 40
    seen = []
    real = (fh._get, fh.resolve_revision)
    index = json.dumps({"weight_map": {"w": "m-1.safetensors"}}).encode()
    header = json.dumps({"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()

    def fake_get(url, start=None, end=None):
        seen.append(url)
        if url.endswith(".safetensors"):
            return struct.pack("<Q", len(header)) if end == 7 else header
        return index if url.endswith("index.json") else b"{}"
    try:
        fh._get = fake_get
        for revision in (sha, other):  # the second run must ignore a newer remote commit
            fh.resolve_revision = lambda repo, revision=revision: revision
            with tempfile.TemporaryDirectory() as d:
                fh.fetch("o/r", Path(d), None, 1)
                fh.resolve_revision = lambda repo: "c" * 40
                fh.fetch("o/r", Path(d), None, 1)
                check((Path(d) / "repo.txt").read_text().strip() == f"o/r@{revision}", "stamp names the commit")
                check(fh.check_stamp("o/r", Path(d)) == revision, "check_stamp returns the commit")
            check(all(f"/resolve/{revision}/" in u for u in seen), f"every request uses the commit: {seen}")
            seen.clear()
    finally:
        fh._get, fh.resolve_revision = real


def test_range_reads_are_bounded_and_refuse_a_server_that_ignores_range():
    import fetch_headers as fh

    class Reply:
        def __init__(self, status, body, content_range=None):
            self.status, self.body, self.headers = status, body, {"Content-Range": content_range} if content_range else {}
            self.read_sizes = []

        def read(self, n=-1):
            self.read_sizes.append(n)
            return self.body if n < 0 else self.body[:n]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    real = fh.urllib.request.urlopen
    try:
        cases = [(Reply(206, b"abcd", "bytes 10-13/99"), b"abcd"),
                 (Reply(200, b"x" * 1000), None),                 # ignored Range: the whole file
                 (Reply(206, b"abcd", "bytes 0-3/99"), None),     # the wrong range
                 (Reply(206, b"abcdefgh", "bytes 10-13/99"), None),  # more than asked for
                 (Reply(206, b"ab", "bytes 10-11/99"), None)]        # a body cut short
        for reply, expected in cases:
            fh.urllib.request.urlopen = lambda request, timeout=0, reply=reply: reply
            if expected is None:
                try:
                    fh._get("http://x/f", 10, 13)
                except ValueError:
                    check(-1 not in reply.read_sizes, "a refused reply must not be read unbounded")
                    continue
                check(False, "a reply that is not exactly the requested range must be refused")
            else:
                check(fh._get("http://x/f", 10, 13) == expected, "a correct 206 returns its bytes")
                check(reply.read_sizes == [5], "the read is bounded by the range")
        whole = Reply(200, b"{}")
        fh.urllib.request.urlopen = lambda request, timeout=0: whole
        check(fh._get("http://x/index.json") == b"{}", "an unranged request still reads the whole body")
    finally:
        fh.urllib.request.urlopen = real


def test_an_interrupted_write_leaves_no_cache_file_and_the_next_fetch_recovers():
    import fetch_headers as fh
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "config.json"
        real = os.replace

        def killed(src, dst):
            raise KeyboardInterrupt  # the process dies after the data is written, before the rename

        os.replace = killed
        try:
            fh.write_atomic(target, b"{}")
        except KeyboardInterrupt:
            pass
        finally:
            os.replace = real
        check(not target.exists(), "an interrupted write must not leave the final file")
        check((Path(d) / "config.json.part").exists(), "the partial data is only in the .part file")
        fh.write_atomic(target, b'{"a": 1}')
        check(json.loads(target.read_text()) == {"a": 1}, "a later write replaces it whole")
    with tempfile.TemporaryDirectory() as d:  # a run killed while stamping must not make the directory look used
        (Path(d) / "repo.txt.part").write_text("o/r@")
        real_resolve, real_get = fh.resolve_revision, fh._get
        fh.resolve_revision = lambda repo: "a" * 40
        fh._get = lambda url, start=None, end=None: b'{"weight_map": {}}'
        try:
            fh.fetch("o/r", Path(d), None, 1)
        finally:
            fh.resolve_revision, fh._get = real_resolve, real_get
        check(fh.check_stamp("o/r", Path(d)) == "a" * 40, "fetch recovers over a stray .part file")


def test_a_stamp_without_a_commit_is_refused():
    import fetch_headers as fh
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "repo.txt").write_text("o/r\n")
        try:
            fh.check_stamp("o/r", Path(d))
        except SystemExit:
            return
        check(False, "a stamp that names no commit must be refused")


def test_shard_names_from_the_index_cannot_leave_the_cache():
    import fetch_headers as fh
    for bad in ("../x.safetensors", "a/b.safetensors", "..", "/etc/passwd", "a\\b", ""):
        try:
            fh.shard_name(bad)
        except ValueError:
            continue
        check(False, f"{bad!r} must be refused")
    check(fh.shard_name("model-00001-of-00046.safetensors") == "model-00001-of-00046.safetensors", "a real name passes")


def test_read_tensor_requires_the_matching_stamp():
    import fetch_headers as fh
    with tempfile.TemporaryDirectory() as d:
        for stamp in (None, "someone/else"):
            if stamp:
                (Path(d) / "repo.txt").write_text(stamp + "\n")
            try:
                fh.read_tensor("deepseek-ai/other", Path(d), "w")
            except SystemExit:
                continue
            check(False, f"stamp {stamp!r} must be refused before any read")


def _gguf_string(text):
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def test_gguf_header_parse():
    """A synthetic header: a scalar, a string, a short array, a long array (recorded by length) and two tensors."""
    kv = [
        (_gguf_string("a.count") + struct.pack("<II", 4, 40)),
        (_gguf_string("a.name") + struct.pack("<I", 8) + _gguf_string("hi")),
        (_gguf_string("a.list") + struct.pack("<IIQ", 9, 5, 3) + struct.pack("<3i", 1, 2, 3)),
        (_gguf_string("a.tokens") + struct.pack("<IIQ", 9, 8, 100) + b"".join(_gguf_string("t%d" % i) for i in range(100))),
        (_gguf_string("a.floats") + struct.pack("<IIQ", 9, 6, 100) + struct.pack("<100f", *range(100))),
    ]
    tensors = [
        _gguf_string("blk.0.w") + struct.pack("<I", 2) + struct.pack("<QQ", 5120, 384) + struct.pack("<IQ", 13, 0),
        _gguf_string("norm") + struct.pack("<I", 1) + struct.pack("<Q", 64) + struct.pack("<IQ", 0, 32),
    ]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv) + b"".join(tensors)
    meta, ts = gh.parse_header(data)
    check(meta == {"a.count": 40, "a.name": "hi", "a.list": [1, 2, 3], "a.tokens": {"array_length": 100},
                   "a.floats": {"array_length": 100}}, f"metadata parsed wrongly: {meta}")
    check(ts == [{"name": "blk.0.w", "ne": [5120, 384], "type": 13, "offset": 0},
                 {"name": "norm", "ne": [64], "type": 0, "offset": 32}], f"tensors parsed wrongly: {ts}")
    for bad in (b"GGUX" + data[4:], data[:4] + struct.pack("<I", 2) + data[8:]):
        try:
            gh.parse_header(bad)
        except gh.HeaderError:
            continue
        check(False, "a bad magic or version must be refused")
    try:
        gh.parse_header(data[:-3])
    except gh.HeaderError:
        pass
    else:
        check(False, "a header cut short must be refused, not parsed")


def test_gguf_headers_fetch_refuses_what_could_hide_missing_or_extra_data():
    """A server that ignores Range (a 200) is refused, and so is an inventory short of split.tensors.count."""
    from unittest import mock

    class Response:
        def __init__(self, status, body, content_range):
            self.status, self._body = status, body
            self.headers = {"Content-Range": content_range} if content_range else {}

        def read(self, n):
            return self._body[:n]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    for response in (Response(200, header, None), Response(206, header, "bytes 5-9/10"), Response(206, header, None)):
        with mock.patch.object(gh.urllib.request, "urlopen", return_value=response):
            try:
                gh._range_get("https://example.invalid/x.gguf")
            except SystemExit:
                continue
        check(False, f"a {response.status} without a range from byte 0 must be refused")
    with mock.patch.object(gh.urllib.request, "urlopen", return_value=Response(206, header, "bytes 0-12/13")):
        check(gh._range_get("https://example.invalid/x.gguf") == header, "a 206 from byte 0 must be accepted")

    kv = _gguf_string("split.count") + struct.pack("<II", 4, 3) + _gguf_string("split.tensors.count") + struct.pack("<IQ", 10, 5)
    first = b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kv  # advertises 5 tensors, holds none
    with mock.patch.object(gh, "_revision", return_value="0" * 40), mock.patch.object(
        gh, "_range_get", return_value=first
    ):
        try:
            gh.fetch("a/b", "stem", None, 3)
        except SystemExit as error:
            check("advertises 5" in str(error), f"the count mismatch must be named: {error}")
        else:
            check(False, "a tensor inventory short of split.tensors.count must be refused")


def _scalar_sinkhorn(logits, scale, base, hc, iters, eps):
    """One token, plain Python loops: an independent formulation of the kernel's comb."""
    import math

    rows = []
    for j in range(hc):
        row = [logits[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k] for k in range(hc)]
        top = max(row)
        e = [math.exp(v - top) for v in row]
        total = sum(e)
        rows.append([v / total + eps for v in e])
    col = [sum(rows[j][k] for j in range(hc)) for k in range(hc)]
    rows = [[rows[j][k] / (col[k] + eps) for k in range(hc)] for j in range(hc)]
    for _ in range(iters - 1):
        rows = [[v / (sum(r) + eps) for v in r] for r in rows]
        col = [sum(rows[j][k] for j in range(hc)) for k in range(hc)]
        rows = [[rows[j][k] / (col[k] + eps) for k in range(hc)] for j in range(hc)]
    return rows


def test_hc_sinkhorn_matches_a_scalar_formulation_and_is_doubly_stochastic():
    rng = np.random.default_rng(3)
    hc, eps = 4, 1e-6
    mixes = rng.standard_normal((5, (2 + hc) * hc)).astype(np.float32)
    scale = np.array([0.7, 0.4, 1.1], np.float32)
    base = (rng.standard_normal((2 + hc) * hc) * 0.3).astype(np.float32)
    pre, post, comb = hcr.hc_split_sinkhorn(mixes, scale, base, hc, 20, eps)
    for t in range(5):
        want = _scalar_sinkhorn(mixes[t].astype(float).tolist(), scale.astype(float).tolist(), base.astype(float).tolist(), hc, 20, eps)
        check(np.allclose(comb[t], np.array(want), atol=2e-6), f"comb of token {t} differs from the scalar formulation")
    check(np.allclose(comb.sum(axis=-2), 1.0, atol=1e-4), "columns of comb must sum to 1 after the last step")
    check(np.allclose(comb.sum(axis=-1), 1.0, atol=1e-2), "rows of comb must be close to 1 (doubly stochastic)")
    check((pre > eps).all() and (pre < 1 + eps + 1e-6).all(), "pre is a sigmoid plus eps")
    check((post > 0).all() and (post < 2).all(), "post is 2 * a sigmoid")
    # More iterations tighten the rows.
    _, _, loose = hcr.hc_split_sinkhorn(mixes, scale, base, hc, 1, eps)
    check(np.abs(loose.sum(axis=-1) - 1).max() > np.abs(comb.sum(axis=-1) - 1).max(), "iterating must tighten the rows")
    for bad in (lambda: hcr.hc_split_sinkhorn(mixes[:, :5], scale, base, hc, 20, eps),
                lambda: hcr.hc_split_sinkhorn(mixes, scale, base, hc, 0, eps)):
        try:
            bad()
        except ValueError:
            continue
        check(False, "a wrong mixes width and zero iterations must be refused")


def test_hc_pre_and_post_semantics():
    rng = np.random.default_rng(4)
    n, hc, d = 2, 3, 5
    x = rng.standard_normal((n, hc, d)).astype(np.float32)
    pre = np.zeros((n, hc), np.float32)
    pre[:, 1] = 1.0
    check(np.array_equal(hcr.hc_pre(x, pre), x[:, 1]), "a one-hot pre must pick that copy")
    sub = rng.standard_normal((n, d)).astype(np.float32)
    ident = np.broadcast_to(np.eye(hc, dtype=np.float32), (n, hc, hc))
    check(np.allclose(hcr.hc_post(sub, x, np.zeros((n, hc), np.float32), ident), x), "comb = I, post = 0 must return the residual")
    out = hcr.hc_post(sub, x, np.ones((n, hc), np.float32), np.zeros((n, hc, hc), np.float32))
    check(np.allclose(out, np.broadcast_to(sub[:, None, :], (n, hc, d))), "comb = 0, post = 1 must copy the sublayer output")
    # comb[i, j] sends copy i of the residual to copy j.
    comb = np.zeros((n, hc, hc), np.float32)
    comb[:, 0, 2] = 1.0
    out = hcr.hc_post(np.zeros((n, d), np.float32), x, np.zeros((n, hc), np.float32), comb)
    check(np.allclose(out[:, 2], x[:, 0]) and not out[:, [0, 1]].any(), "comb[i, j] must route residual copy i to copy j")


def test_hc_mixes_depend_on_the_stream_through_one_flat_statistic():
    rng = np.random.default_rng(5)
    hc, d = 4, 8
    x = rng.standard_normal((2, hc, d)).astype(np.float32)
    fn = (rng.standard_normal(((2 + hc) * hc, hc * d)) * 0.3).astype(np.float32)
    scale, base = np.array([1.0, 1.0, 1.0], np.float32), np.zeros((2 + hc) * hc, np.float32)
    a = hcr.hc_mixes(x, fn, scale, base, 1e-6, 20, 1e-6)
    b = hcr.hc_mixes(x * 7.0, fn, scale, base, 1e-6, 20, 1e-6)
    check(all(np.allclose(u, v, atol=1e-4) for u, v in zip(a, b)), "mixes are scale invariant in the stream (RMS-normalised)")
    # One statistic per token over the whole hc*d stream: scaling one copy changes the mixes.
    y = x.copy()
    y[:, 0] *= 3.0
    c = hcr.hc_mixes(y, fn, scale, base, 1e-6, 20, 1e-6)
    check(not np.allclose(a[0], c[0]), "the statistic is over the whole stream, not per copy")


def test_a_block_chains_each_sublayers_pre_into_the_next_one():
    rng = np.random.default_rng(6)
    n, hc, d = 2, 4, 6
    mix = (2 + hc) * hc

    def params():
        return ((rng.standard_normal((mix, hc * d)) * 0.3).astype(np.float32), np.array([0.9, 0.6, 1.2], np.float32),
                (rng.standard_normal(mix) * 0.2).astype(np.float32))

    attn_p, ffn_p = params(), params()
    x = rng.standard_normal((n, hc, d)).astype(np.float32)
    pre_mix = rng.random((n, hc)).astype(np.float32)
    seen = {}

    def attention(h):
        seen["attn_in"] = h.copy()
        return h * 2.0

    def ffn(h):
        seen["ffn_in"] = h.copy()
        return h - 1.0

    out, next_pre = hcr.block_streams(x, pre_mix, attn_p, ffn_p, attention, ffn, 1e-6, 20, 1e-6)
    a_pre, a_post, a_comb = hcr.hc_mixes(x, *attn_p, 1e-6, 20, 1e-6)
    check(np.allclose(seen["attn_in"], hcr.hc_pre(x, pre_mix)), "attention reads the previous layer's pre_mix")
    after_attn = hcr.hc_post(attention(hcr.hc_pre(x, pre_mix)), x, a_post, a_comb)
    check(np.allclose(seen["ffn_in"], hcr.hc_pre(after_attn, a_pre)), "the FFN reads the attention's pre, computed from the block input")
    f_pre, f_post, f_comb = hcr.hc_mixes(after_attn, *ffn_p, 1e-6, 20, 1e-6)
    check(np.allclose(next_pre, f_pre), "the block returns the next block's pre_mix: its FFN's pre")
    check(np.allclose(out, hcr.hc_post(ffn(seen["ffn_in"]), after_attn, f_post, f_comb)), "the block output mixes the residual through the FFN's comb")


def test_the_committed_hyper_connection_golden_matches_the_reference():
    golden = json.loads((Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "hyper_connection_golden.json").read_text())
    x, fn = np.array(golden["x"], np.float32), np.array(golden["hc_fn"], np.float32)
    scale, base = np.array(golden["hc_scale"], np.float32), np.array(golden["hc_base"], np.float32)
    pre, post, comb = hcr.hc_mixes(x, fn, scale, base, golden["norm_eps"], golden["sinkhorn_iters"], golden["eps"])
    for name, got in (("pre", pre), ("post", post), ("comb", comb)):
        check(np.allclose(got, np.array(golden[name], np.float32), atol=1e-6), f"golden {name} differs from the reference")
    check(np.allclose(hcr.hc_pre(x, pre), np.array(golden["collapsed"], np.float32), atol=1e-6), "golden collapse differs")
    check(np.allclose(hcr.hc_post(np.array(golden["sublayer"], np.float32), x, post, comb),
                      np.array(golden["expanded"], np.float32), atol=1e-6), "golden expansion differs")
    fresh = hcr._golden()
    same_shape = golden.keys() == fresh.keys() and all(
        golden[k] == fresh[k] for k in ("hc", "d", "n", "sinkhorn_iters", "eps", "norm_eps")
    )
    close = all(np.allclose(np.array(golden[k], np.float32), np.array(fresh[k], np.float32), atol=1e-6)
                for k in ("x", "hc_fn", "hc_scale", "hc_base", "pre", "post", "comb", "collapsed", "sublayer", "expanded"))
    # Not exact equality: the matmul and exp of another BLAS or CPU differ in the last float32 bits.
    check(same_shape and close, "the committed golden is not what --emit-golden produces; regenerate it")


def test_attention_stack_matches_the_model_repos_own_code():
    """The golden is the model repo's Attention/Compressor/Indexer run on CPU (tools/dsv4/model_golden.py):
    window-only layers, a gated ratio-2 compressor, ratio-1 layers, kv and index sources, a candidate source
    with two later layers searching inside its blocks, YaRN active."""
    golden = json.loads((Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "attention_golden.json").read_text())
    check(golden["seed"] == att.STACK_SEED and golden["seq"] == att.STACK_SEQ, "the golden is for another seed or length")
    check(golden["model_repo"] == mg.REPO and len(golden["revision"]) == 40 and set(golden["revision"]) <= set("0123456789abcdef"),
          "the golden must record the commit of the model repo it was produced from")
    mine = att.run_stack()
    check(len(mine) == len(golden["outputs"]) == att.STACK_LAYERS, "layer count")
    for layer, (want, got) in enumerate(zip(golden["outputs"], mine)):
        err = float(np.abs(np.array(want, np.float32) - got).max())
        check(err < 5e-6, f"layer {layer} differs from the model repo's output by {err}")


def test_attention_is_causal_row_by_row():
    """Row i of a longer sequence equals the last row of its length-(i+1) prefix: decode is this function
    applied to the prefix. It includes the compressor's groups, the indexer's reachability and the candidate blocks."""
    full = att.run_stack()
    for n in (1, 2, 3, 5, 8, 10):
        prefix = att.run_stack(tokens=n)
        for layer, (a, b) in enumerate(zip(prefix, full)):
            check(np.allclose(a, b[:n], atol=2e-6), f"layer {layer}: a {n}-token prefix differs from the first {n} rows of the full run")


def test_rope_is_a_rotation_and_yarn_only_slows_the_low_frequencies():
    import math

    f = att.rope_freqs(8, 6, 0, 10000.0, 4.0, 32, 1)
    check(np.allclose(f[0], 1.0) and np.allclose(np.abs(f), 1.0, atol=1e-6), "position 0 is the identity and every rotation has modulus 1")
    x = np.random.default_rng(0).standard_normal((6, 8)).astype(np.float32)
    r = att.apply_rope(x, f)
    check(np.allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-5), "a rotation preserves the norm")
    check(np.allclose(att.apply_rope(r, f, inverse=True), x, atol=1e-5), "the inverse undoes the rotation")
    # adjacent pairs: position 1, pair 0 rotates by angle 1 (theta^0), which is cos/sin on elements 0 and 1
    check(np.allclose(r[1, :2], [x[1, 0] * math.cos(1) - x[1, 1] * math.sin(1), x[1, 0] * math.sin(1) + x[1, 1] * math.cos(1)], atol=1e-5), "pair order")
    # YaRN for dim 8, base 10000, a context of 8 turns and factor 4: low = 0, high = 1, so the ramp is
    # [0, 1, 1, 1]: the first (fastest) dimension keeps its frequency, the other three are divided by 4.
    y = att.rope_freqs(8, 6, 8, 10000.0, 4.0, 32, 1)
    check(np.allclose(np.angle(y[1])[0], np.angle(f[1])[0], atol=1e-5), "the fastest dimension is unchanged by YaRN")
    check(np.allclose(np.angle(y[1])[1:], np.angle(f[1])[1:] / 4, atol=1e-5), "the slower dimensions are divided by the factor")


def test_sparse_attention_matches_a_dense_formulation():
    rng = np.random.default_rng(8)
    s, h, d, n = 4, 3, 5, 6
    q = rng.standard_normal((s, h, d)).astype(np.float32)
    kv = rng.standard_normal((n, d)).astype(np.float32)
    sink = rng.standard_normal(h).astype(np.float32)
    idxs = np.array([[0, 2, 5, -1], [1, 3, -1, -1], [-1, -1, -1, -1], [4, 0, 1, 2]])
    out = att.sparse_attention(q, kv, sink, idxs, 0.4)
    for i in range(s):
        rows = [j for j in idxs[i] if j >= 0]
        if not rows:
            check(not out[i].any(), "a query with no valid row gets zeros")
            continue
        for hh in range(h):
            logits = np.append((kv[rows] @ q[i, hh]) * 0.4, sink[hh])  # the sink is one more logit with no value
            p = np.exp(logits - logits.max())
            p /= p.sum()
            check(np.allclose(out[i, hh], p[:-1] @ kv[rows], atol=1e-5), f"query {i} head {hh} differs from softmax over rows + sink")
    big = att.sparse_attention(q, kv, np.full(h, 60.0, np.float32), idxs, 0.4)
    check(np.abs(big).max() < 1e-6, "a huge sink absorbs the attention: the output tends to zero")


def test_window_attention_equals_dense_sliding_window_attention():
    """A window-only layer, written out with explicit loops from the pieces (RoPE, shared kv, grouped output)."""
    cfgs, weights, inputs = att.stack_inputs()
    cfg, w, x = cfgs[0], weights[0], inputs[0]
    got = att.attention(x, w, cfg, att.Shared())
    s, h, hd, rd = x.shape[0], cfg.n_heads, cfg.head_dim, cfg.rope_dim
    freqs = att.rope_freqs(rd, s, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    qr = att.rms_norm(x @ w.wq_a.T, w.q_norm, cfg.norm_eps)
    q = att._rope_tail((qr @ w.wq_b.T).reshape(s, h, hd), freqs, rd)
    kv = att._rope_tail(att.rms_norm(x @ w.wkv.T, w.kv_norm, cfg.norm_eps), freqs, rd)
    o = np.zeros((s, h, hd), np.float32)
    for i in range(s):
        lo = max(0, i - cfg.window + 1)
        for hh in range(h):
            logits = np.append((kv[lo : i + 1] @ q[i, hh]) * hd ** -0.5, w.attn_sink[hh])
            p = np.exp(logits - logits.max())
            p /= p.sum()
            o[i, hh] = p[:-1] @ kv[lo : i + 1]
    o = att._rope_tail(o, freqs, rd, inverse=True)
    g = cfg.o_groups
    group_out = [o[:, k * (h // g) : (k + 1) * (h // g)].reshape(s, -1) @ w.wo_a.reshape(g, cfg.o_lora_rank, -1)[k].T for k in range(g)]
    want = np.concatenate(group_out, axis=1) @ w.wo_b.T
    check(np.allclose(got, want, atol=2e-5), "the window-only layer differs from the loop formulation")


def test_compressor_pools_complete_groups_with_a_per_channel_softmax():
    cfgs, weights, inputs = att.stack_inputs()
    cfg, w, x = cfgs[2], weights[2], inputs[2]  # ratio 2, with a gate
    latent = att.compressor(x, w, cfg)
    check(latent.shape == (5, cfg.head_dim), "11 tokens at ratio 2 give 5 groups; the trailing token is dropped")
    kv, gate = x @ w.comp_wkv.T, x @ w.comp_wgate.T
    for g in range(5):
        a, b = 2 * g, 2 * g + 1
        e = np.exp(np.stack([gate[a], gate[b]]) - np.maximum(gate[a], gate[b]))
        pooled = (kv[a] * e[0] + kv[b] * e[1]) / (e[0] + e[1])  # softmax over the two tokens, per channel
        check(np.allclose(latent[g], att.rms_norm(pooled, w.comp_norm, cfg.norm_eps), atol=1e-5), f"group {g}")
    check(att.compressor(x[:1], w, cfg) is None, "fewer tokens than the ratio give no latent")
    one = cfgs[4]  # ratio 1: a plain projection and norm, one latent per token
    check(np.allclose(att.compressor(inputs[4], weights[4], one), att.rms_norm(inputs[4] @ weights[4].comp_wkv.T, weights[4].comp_norm, one.norm_eps), atol=1e-6), "ratio 1 is a projection")


def test_indexer_selection_is_reachable_ordered_and_padded():
    cfgs, weights, inputs = att.stack_inputs()
    shared = att.Shared()
    for i in range(3):
        att.attention(inputs[i], weights[i], cfgs[i], shared)
    idxs, s, ratio, offset = shared.topk_idxs, att.STACK_SEQ, 2, att.STACK_SEQ
    check(idxs.shape == (s, 3), "index_topk columns")
    for i in range(s):
        row = idxs[i][idxs[i] >= 0]
        check((np.diff(row) > 0).all(), "selected positions are in increasing order")
        check(((row - offset) < (i + 1) // ratio).all(), f"query {i} selects a group that is not complete yet")
        check(len(row) == min(3, (i + 1) // ratio), f"query {i} selects min(top-k, reachable groups) positions")
        check((idxs[i][len(row):] == -1).all(), "unreachable slots are -1, after the real ones")
    # The candidate source keeps whole blocks: a later layer's selection stays inside them.
    for i in range(3, 5):
        att.attention(inputs[i], weights[i], cfgs[i], shared)
    cand = shared.candidates
    check(cand.shape == (s, s) and cand.dtype == bool, "candidate mask is [queries, groups]")
    for i in range(s):
        blocks = {j // 2 for j in np.flatnonzero(cand[i])}
        check(len(blocks) <= 2 and (i // 2) in blocks, "at most topk_blocks blocks, and the one holding the newest position")
    att.attention(inputs[5], weights[5], cfgs[5], shared)
    row = shared.topk_idxs
    for i in range(s):
        for j in row[i][row[i] >= 0] - s:
            check(cand[i, j], f"layer 5 query {i} selected group {j} outside the candidate blocks")


def test_model_golden_fetch_pins_one_commit_and_is_all_or_nothing():
    import io
    import tempfile
    from unittest import mock

    sha = "ab" * 20

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    urls = []

    def urlopen(url, timeout=0):
        urls.append(url)
        if url.endswith("/api/models/" + mg.REPO):
            return Response(json.dumps({"sha": sha}).encode())
        return Response(b"# " + url.encode())

    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "inference"
        with mock.patch.object(mg.urllib.request, "urlopen", side_effect=urlopen):
            check(mg.fetch(target) == sha, "fetch returns the commit")
        files = sorted(f.name for f in target.iterdir())
        check(files == sorted([*mg.FILES, mg.REVISION_FILE]), f"fetch writes every file and the revision: {files}")
        check("engram.py" in mg.FILES, "engram.py is imported by --emit-engram and --emit-model, so it must be fetched")
        check(all(f"/resolve/{sha}/inference/" in u for u in urls if "/resolve/" in u), "every file is fetched at the pinned commit, not main")
        check(mg.revision_of(target) == sha, "the directory records its commit")
        try:
            mg.fetch(target)
        except SystemExit:
            pass
        else:
            check(False, "fetching into a non-empty directory must be refused")
        # A fetch that fails part-way leaves nothing behind: no mixed revision, no staging directory.
        other = Path(d) / "other"
        calls = {"n": 0}

        def flaky(url, timeout=0):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("network down")
            return urlopen(url, timeout)

        with mock.patch.object(mg.urllib.request, "urlopen", side_effect=flaky):
            try:
                mg.fetch(other)
            except OSError:
                pass
            else:
                check(False, "a failed fetch must raise")
        check(not other.exists() or not any(other.iterdir()), "a failed fetch leaves no files in the target")
        check(not list(Path(d).glob("*.staging-*")), "a failed fetch removes its staging directory")
        # A failure at the publishing step (the rename) leaves no destination and no staging either.
        third = Path(d) / "third"
        with mock.patch.object(mg.urllib.request, "urlopen", side_effect=urlopen), mock.patch.object(
            Path, "rename", side_effect=OSError("disk error")
        ):
            try:
                mg.fetch(third)
            except OSError:
                pass
            else:
                check(False, "a failed publication must raise")
        check(not third.exists(), "a failed publication leaves no destination directory")
        check(not list(Path(d).glob("*.staging-*")), "a failed publication removes its staging directory")
        # An empty existing directory is fine: it is replaced by the staged one.
        empty = Path(d) / "empty"
        empty.mkdir()
        with mock.patch.object(mg.urllib.request, "urlopen", side_effect=urlopen):
            mg.fetch(empty)
        check(sorted(f.name for f in empty.iterdir()) == sorted([*mg.FILES, mg.REVISION_FILE]), "an empty target directory is filled whole")
        try:
            mg.revision_of(Path(d) / "nowhere")
        except SystemExit:
            pass
        else:
            check(False, "a directory without a recorded commit must be refused")


def test_moe_matches_the_model_repos_own_code():
    """The golden is the model repo's MoE (Gate, Expert, shared expert) run on CPU (tools/dsv4/model_golden.py):
    one layer with swiglu clamps, top-2 and a route scale; one without clamps, top-3 and no normalisation."""
    golden = json.loads((Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "moe_golden.json").read_text())
    check(golden["seed"] == moe_ref.GOLDEN_SEED, "the golden is for another seed")
    check(golden["model_repo"] == mg.REPO and len(golden["revision"]) == 40, "the golden must record the model repo commit")
    for layer, (want, got) in enumerate(zip(golden["outputs"], moe_ref.run_golden())):
        want = np.array(want, np.float32)
        check(np.allclose(want, got, rtol=2e-6, atol=2e-6 * float(np.abs(want).max())), f"moe layer {layer} differs from the model repo's output")


def test_moe_is_the_weighted_routed_sum_plus_the_shared_expert():
    x, w, cfg = moe_ref.golden_inputs()[0]
    got = moe_ref.moe_forward(x, w, cfg)
    weights, indices = moe_ref.route(x, w, cfg)
    for t in range(x.shape[0]):  # one token at a time, with the loop spelled out
        want = ref.expert_forward(x[t : t + 1], w.shared.w1, w.shared.w2, w.shared.w3, cfg.clamp_shared)[0].copy()
        routed = np.zeros_like(want)
        for slot in range(cfg.topk):
            e = w.experts[indices[t, slot]]
            routed += ref.expert_forward(x[t : t + 1], e.w1, e.w2, e.w3, cfg.clamp_experts, weights[t : t + 1, slot : slot + 1])[0]
        check(np.allclose(got[t], routed + want, atol=1e-5), f"token {t} is not routed sum + shared expert")
    check(np.allclose(weights.sum(axis=1), cfg.route_scale, atol=1e-5), "normalised routing weights sum to the route scale")
    # An expert nobody picked contributes nothing: perturbing it changes no output.
    unused = [e for e in range(6) if e not in set(indices.ravel().tolist())]
    check(bool(unused), "the golden should leave an expert unpicked to exercise the skip")
    w.experts[unused[0]].w2 = w.experts[unused[0]].w2 + 5.0
    check(np.array_equal(got, moe_ref.moe_forward(x, w, cfg)), "an unpicked expert must not affect the output")


def test_moe_clamps_and_shared_expert_are_separate_knobs():
    x, w, cfg = moe_ref.golden_inputs()[0]
    base = moe_ref.moe_forward(x, w, cfg)
    import dataclasses

    no_routed_clamp = moe_ref.moe_forward(x, w, dataclasses.replace(cfg, clamp_experts=0.0))
    no_shared_clamp = moe_ref.moe_forward(x, w, dataclasses.replace(cfg, clamp_shared=0.0))
    check(not np.allclose(base, no_routed_clamp, atol=1e-4), "the routed experts' clamp must matter in the golden")
    check(not np.allclose(base, no_shared_clamp, atol=1e-4), "the shared expert's clamp must matter in the golden")
    for bad in (dataclasses.replace(cfg, topk=0), dataclasses.replace(cfg, topk=7)):
        try:
            moe_ref.moe_forward(x, w, bad)
        except ValueError:
            continue
        check(False, "a top-k outside 1..experts must be refused")
    short = moe_ref.MoEWeights(w.router, w.bias, w.experts[:-1], w.shared)
    try:
        moe_ref.moe_forward(x, short, cfg)
    except ValueError:
        pass
    else:
        check(False, "a router that scores more experts than there are must be refused")


def test_engram_matches_the_model_repos_own_code():
    """The golden is the repo's EngramLayout, compute_hash_multipliers, NgramHashState and Engram run on CPU
    (tools/dsv4/model_golden.py): hash ids must match exactly, the gated residual write to float32 rounding."""
    golden = json.loads((Path(HERE).parent.parent / "tests" / "fixtures" / "dsv4" / "engram_golden.json").read_text())
    check(golden["seed"] == eng.GOLDEN_SEED, "the golden is for another seed")
    check(golden["model_repo"] == mg.REPO and len(golden["revision"]) == 40, "the golden must record the model repo commit")
    ids, outputs = eng.run_golden()
    check(np.array_equal(np.array(golden["hash_ids"]), ids), "hash ids differ from the model repo's")
    x = eng.golden_case()["x"]
    for layer, (want, got) in enumerate(zip(golden["outputs"], outputs)):
        want = np.array(want, np.float32)
        check(np.allclose(want, got, rtol=2e-6, atol=2e-6 * float(np.abs(want).max())), f"engram layer {layer} differs from the model repo's output")
        check(float(np.abs(got - x[layer]).max()) > 1.0, "the golden must actually change the stream")


def test_engram_hash_ids_structure():
    case = eng.golden_case()
    ids, primes, offsets = eng.ngram_hash_ids(case["input_ids"], case["token_map"], case["pad_id"], case["multipliers"], case["primes"], case["offsets"], case["token_mask"]), case["primes"], case["offsets"]
    flat = primes.reshape(primes.shape[0], -1)  # [layers, cols]: the prime modulus of each (order, head)
    for layer in range(ids.shape[1]):
        check(((ids[:, layer] >= offsets[layer]) & (ids[:, layer] < offsets[layer] + flat[layer])).all(), "every id stays inside its (order, head) bucket range")
        ends = offsets[layer] + flat[layer]
        check((offsets[layer][1:] == ends[:-1]).all(), "bucket ranges are consecutive and disjoint")
    allp = primes.reshape(-1).tolist()
    check(len(set(allp)) == len(allp) and all(eng._is_prime(v) for v in allp), "bucket moduli are distinct primes (never reused across layers)")
    check((case["multipliers"] % 2 == 1).all(), "multipliers are odd")
    # A masked token takes part in no n-gram: changing it changes no id, and the n-grams after it see the pad id.
    changed = case["input_ids"].copy()
    changed[4] = (changed[4] + 7) % eng.G_VOCAB
    again = eng.ngram_hash_ids(changed, case["token_map"], case["pad_id"], case["multipliers"], case["primes"], case["offsets"], case["token_mask"])
    check(np.array_equal(ids, again), "a masked token must not influence any hash id")
    # Look-back stops at the start of the sequence: position 0 is the 1-gram of its own token against pad slots.
    one = eng.ngram_hash_ids(case["input_ids"][:1], case["token_map"], case["pad_id"], case["multipliers"], case["primes"], case["offsets"])
    check(np.array_equal(one[0], eng.ngram_hash_ids(case["input_ids"], case["token_map"], case["pad_id"], case["multipliers"], case["primes"], case["offsets"])[0]), "position 0 sees only itself")
    # Causal: a prefix gives the same ids as the full sequence.
    prefix = eng.ngram_hash_ids(case["input_ids"][:7], case["token_map"], case["pad_id"], case["multipliers"], case["primes"], case["offsets"], case["token_mask"][:7])
    check(np.array_equal(prefix, ids[:7]), "hash ids are causal")
    # The XOR chain is the hash of the (i+1)-gram: column order 0 uses tokens t and t-1 only.
    t = case["token_map"][case["input_ids"]]
    pos = 8  # three tokens past the masked 5, so no look-back of this 2-gram touches a masked token
    mult = case["multipliers"][0]
    want = ((t[pos] * mult[0]) ^ (t[pos - 1] * mult[1])) % case["primes"][0, 0, 0] + case["offsets"][0, 0]
    check(int(ids[pos, 0, 0]) == int(want), "the 2-gram id is (t*m0 xor t-1*m1) mod prime + offset")


def test_engram_gate_semantics():
    rng = np.random.default_rng(2)
    L, hc, d, cols, row = 4, 2, 8, 3, 4
    x = rng.standard_normal((L, hc, d)).astype(np.float32)
    rows = rng.standard_normal((L, cols, row)).astype(np.float32)
    wkv = rng.standard_normal(((hc + 1) * d, cols * row)).astype(np.float32) * 0.3
    weight = (1 + 0.1 * rng.standard_normal((hc, d))).astype(np.float32)
    out = eng.engram_forward(x, rows, wkv, weight, 1e-6)
    # One token with explicit loops: the gate is the sigmoid of the signed sqrt of the normalised dot, per copy.
    t = 2
    kv = wkv @ rows[t].reshape(-1)
    key, value = kv[: hc * d].reshape(hc, d), kv[hc * d :]
    for c in range(hc):
        rstd = 1 / np.sqrt((x[t, c] ** 2).mean() + 1e-6) / np.sqrt((key[c] ** 2).mean() + 1e-6)
        dot = float((x[t, c] * weight[c] * key[c]).sum() * rstd / np.sqrt(d))
        gate = 1 / (1 + np.exp(-np.sign(dot) * np.sqrt(max(abs(dot), 1e-6))))
        check(np.allclose(out[t, c], x[t, c] + gate * value, atol=1e-5), f"copy {c} of token {t}")
    # The gate is a per-copy statistic: scaling one copy of the stream leaves the other's output increment unchanged.
    y = x.copy()
    y[:, 0] *= 5.0
    b = eng.engram_forward(y, rows, wkv, weight, 1e-6)
    check(np.allclose(b[:, 1] - y[:, 1], out[:, 1] - x[:, 1], atol=1e-5), "the statistic is per copy, not joint")
    # A masked token passes through untouched; a zero dot sits at the clamp: sigmoid(sqrt(1e-6)).
    mask = np.array([True, False, True, True])
    m = eng.engram_forward(x, rows, wkv, weight, 1e-6, mask)
    check(np.array_equal(m[1], x[1]) and np.allclose(m[[0, 2, 3]], out[[0, 2, 3]]), "a masked position is unchanged")
    zero = eng.engram_forward(x, np.zeros_like(rows), wkv, weight, 1e-6)
    check(np.allclose(zero, x), "zero rows give zero key and value: no change")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
