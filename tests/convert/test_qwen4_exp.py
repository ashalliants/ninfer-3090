from __future__ import annotations

import json
import os
import shutil

import pytest
import torch

from tools.artifact import ngram_volume
from tools.convert import qwen4_exp
from tools.convert.__main__ import main
from tools.convert.official_recipes import qwen4_exp_gguf
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufError, GgufModel

from . import qwen4_exp_fixture as fixture
from .qwen4_exp_verify import verify

# Infernix a3edb450 parse_config's text keys (src/models/qwen4_exp/config.cpp:115-127).
INFERNIX_TEXT_KEYS = {
    "architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
    "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps", "layer_types",
    "num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters",
    "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
    "linear_value_head_dim", "linear_conv_kernel_dim", "output_gate_type", "num_experts",
    "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
    "norm_topk_prob", "hc_count", "hc_lowrank", "indexer_n_heads", "indexer_kv_heads",
    "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "ple_layer_ids",
    "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
    "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "seed",
    "split_ngram_parts", "eos_token_id", "ngram_table",
}
# Its n-gram table keys (config.cpp:79-82).
INFERNIX_TABLE_KEYS = {
    "format", "rows", "row_bytes", "rows_per_block", "block_bytes", "header_bytes", "blocks",
    "file_bytes", "volume_id",
}


@pytest.fixture
def gguf_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def test_config_has_infernix_keys_from_metadata(gguf_path):
    with GgufModel(gguf_path) as gguf:
        config = qwen4_exp.text_config(gguf.metadata)
        model = qwen4_exp.build_model(gguf)
    assert set(model.config) == INFERNIX_TEXT_KEYS
    assert set(config) == INFERNIX_TEXT_KEYS - {"ngram_table"}
    assert config["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert config["rope_parameters"] == {
        "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1]
    }
    assert (config["linear_num_value_heads"], config["linear_value_head_dim"]) == (4, 16)
    assert (config["num_experts"], config["moe_intermediate_size"]) == (4, fixture.F)
    assert (config["hc_lowrank"], config["indexer_budget"], config["indexer_n_heads"]) == (
        fixture.LOW, 8, 2
    )
    # GGUF ple.layers=[1] is zero-based; Infernix's ple_layer_ids is one-based.
    assert config["ple_layer_ids"] == [2] and config["eos_token_id"] == 10
    assert config["ple_embed_dim"] == fixture.PLE_HEADS * fixture.PLE_DIM
    assert (
        config["ngram_vocab_size_base"],
        config["make_ngram_vocab_size_divisible_by"],
        config["seed"],
    ) == (98, 128, 1234)  # 98 is the smallest base whose next prime is 101 (after 97)
    assert qwen4_exp.ngram_table_rows(config) == fixture.PLE_ROWS
    table = model.config["ngram_table"]
    assert set(table) == INFERNIX_TABLE_KEYS and table["volume_id"] is None
    assert (table["format"], table["rows"], table["row_bytes"], table["rows_per_block"]) == (
        "ggml_iq4_nl", fixture.PLE_ROWS, 18, 227
    )
    assert (table["blocks"], table["file_bytes"]) == (3, 4 * 4096)


def test_hash_derivation_reproduces_the_real_gguf_literals():
    # Literal values of the Qwen3.8-Flash-Next GGUF metadata (scan of shard 1).
    assert qwen4_exp.layer_multipliers(248_320, 3, 1234, 0) == [
        23703573157769, 20109073645365, 8052911324071
    ]
    sizes, offsets, rows = qwen4_exp.head_tables(20_000_000, 16, 128, 0)
    assert sizes[:3] == [20000003, 20000023, 20000033] and sizes[-1] == 20000171
    assert offsets[1] == 20000003 and offsets[-1] == 300001275
    assert rows == 320_001_536


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"general.architecture": (8, "qwen35")}, "general.architecture"),
        ({"qwen4exp.attention.compress_ratios": (9, (5, [0, 0, 4, 0]))}, "compress_ratios"),
        ({"qwen4exp.ple.head_offsets": (9, (11, [0, 101, 205, 311]))}, "consecutive primes"),
        ({"qwen4exp.ple.head_vocab_sizes": (9, (11, [101, 103, 109, 113]))}, "consecutive primes"),
        ({"qwen4exp.ple.layer_multipliers": (9, (11, [11, 13, 17]))}, "splitmix64"),
        ({"qwen4exp.ple.layers": (9, (5, [3]))}, "linear-attention"),
        ({"qwen4exp.ple.layers": (9, (5, [0, 1]))}, "exactly one PLE layer"),
        ({"qwen4exp.rope.dimension_count": (4, 10)}, "rope sections"),
        ({"qwen4exp.expert_used_count": (4, 5)}, "expert_used_count"),
        ({"qwen4exp.attention.indexer.top_k": (4, 6)}, "indexer"),
    ],
)
def test_inconsistent_metadata_is_rejected(tmp_path, changes, message):
    path = fixture.write(tmp_path / "bad.gguf", metadata_changes=changes)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match=message):
        qwen4_exp.text_config(gguf.metadata)


def test_name_map_covers_every_tensor_once(gguf_path):
    with GgufModel(gguf_path) as gguf:
        model = qwen4_exp.build_model(gguf)
        mappings = qwen4_exp.name_map(gguf, model.config)
        assert {item.gguf for item in mappings} == set(gguf.tensors)
    banks = [name for name in model.parameters if name.endswith("/moe/experts")]
    assert banks == [qwen4_exp.expert_bank(layer) for layer in range(4)]
    # Three expert tensors become one bank per layer; the PLE table is the n-gram volume.
    assert len(model.parameters) == len(gguf.tensors) - 3 * 4 + 4 - 1
    # Each mixer's down and inject rows form one parent (the final mixer has no inject), as do
    # the GDN a and b projections and the PLE key and value projections.
    groups = [tuple(name.split("/", 3)[3] for name in g) for g in model.packing_groups]
    assert groups == [
        ("attn_hc/down", "attn_hc/inject"), ("mlp_hc/down", "mlp_hc/inject"),
        ("gdn/a_projection", "gdn/b_projection"),
        ("attn_hc/down", "attn_hc/inject"), ("mlp_hc/down", "mlp_hc/inject"),
        ("gdn/a_projection", "gdn/b_projection"), ("ple/key", "ple/value"),
        ("attn_hc/down", "attn_hc/inject"), ("mlp_hc/down", "mlp_hc/inject"),
        ("gdn/a_projection", "gdn/b_projection"),
        ("attn_hc/down", "attn_hc/inject"), ("mlp_hc/down", "mlp_hc/inject"),
    ]
    assert "text/ple/table" not in model.parameters
    assert model.parameters["text/layers/3/attention/query_norm"].direct_format == "bf16"
    conv = model.parameters["text/layers/0/gdn/convolution"]
    assert conv.direct_format == "bf16" and conv.shape == (4, 128)  # [K, C], the GGUF transposed
    assert model.parameters["text/layers/0/attn_hc/norm"].direct_format == "fp32"
    bank = model.parameters["text/layers/0/moe/experts"]
    assert bank.shape == (fixture.E, fixture.H, fixture.F)
    assert bank.inputs == ("text/layers/0/ffn_input",)
    assert qwen4_exp.expert_record_source(bank).format == "ggml_rec_iq2_s_q2_0"
    assert model.parameters["text/layers/0/moe/shared_score"].shape == (1, fixture.H)
    assert model.parameters["text/layers/0/gdn/output"].inputs == ("text/layers/0/gdn/gated_output",)
    assert model.token_count == 14 and model.special_token_ids == (10, 11, 12)


@pytest.mark.parametrize(
    ("drop", "extra", "message"),
    [
        (("blk.1.ple_key.weight",), (), "layer 1 is missing"),
        (("output_hc_up.weight",), (), "output_hc_up.weight' is missing"),
        (("per_layer_token_embd.weight",), (), "per_layer_token_embd.weight' is missing"),
        ((), (("blk.0.attn_q.weight", 8, (64, 128)),), "unexpected GGUF tensor"),
        ((), (("blk.2.ple_key.weight", 30, (64, 128)),), "unexpected GGUF tensor"),
        ((), (("rope_freqs.weight", 0, (4,)),), "unknown GGUF tensor"),
        (("blk.0.ssm_out.weight",), (("blk.0.ssm_out.weight", 8, (32, 64)),), "ssm_out.weight: shape"),
        (("blk.1.ple_key.weight",), (("blk.1.ple_key.weight", 30, (64, 128)),), "ple_key.weight: shape"),
        (
            ("blk.3.indexer.k_proj.weight",),
            (("blk.3.indexer.k_proj.weight", 30, (64, 32)),),
            "k_proj.weight: shape",
        ),
        (
            ("per_layer_token_embd.weight",),
            (("per_layer_token_embd.weight", 20, (32, 500)),),
            "per_layer_token_embd.weight: shape",
        ),
    ],
)
def test_name_map_rejects_missing_misplaced_and_unknown_tensors(tmp_path, drop, extra, message):
    path = fixture.write(tmp_path / "bad.gguf", drop=drop, extra=extra)
    with GgufModel(path) as gguf, pytest.raises(GgufError, match=message):
        qwen4_exp.build_model(gguf)


def test_narrowing_is_exact_or_refused():
    gamma = torch.tensor([0.5, -0.25, 0.0078125, -0.4375], dtype=torch.bfloat16).float()
    stored = (torch.tensor(1.0) + gamma).float()
    assert torch.equal(qwen4_exp.narrow_exactly("n", stored, "gamma").float(), gamma)
    assert torch.equal(qwen4_exp.narrow_exactly("n", gamma, "words").float(), gamma)
    # fl32(1.1) - 1 needs 21 significant bits: no BF16 gamma reproduces it.
    with pytest.raises(GgufError, match="no exact BF16 gamma"):
        qwen4_exp.narrow_exactly("n", torch.tensor([1.1]), "gamma")
    with pytest.raises(GgufError, match="no exact BF16 words"):
        qwen4_exp.narrow_exactly("n", torch.tensor([0.1]), "words")


@pytest.mark.parametrize(
    ("suffix", "kind", "ne", "message"),
    [
        ("ffn_up_exps.weight", fixture.IQ2_XXS, (fixture.H, fixture.F, fixture.E),
         "gate experts are ggml_iq2_s but up experts ggml_iq2_xxs"),
        ("ffn_down_exps.weight", fixture.Q8_0, (fixture.F, fixture.H, fixture.E),
         "no expert record format holds ggml_iq2_s gate/up with ggml_q8_0 down"),
    ],
)
def test_expert_types_must_form_a_record_format(tmp_path, suffix, kind, ne, message):
    name = "blk.0." + suffix
    path = fixture.write(tmp_path / "t.gguf", drop=(name,), extra=((name, kind, ne),))
    with GgufModel(path) as gguf, pytest.raises(GgufError, match=message):
        qwen4_exp.build_model(gguf)


def _record_gguf(tmp_path):
    return fixture.write(tmp_path / "records.gguf")


def _convert(gguf, out, subset=None, volume=True):
    model = qwen4_exp.build_model(gguf, subset=subset)
    qwen4_exp.bind_ngram_volume(model, os.urandom(16))
    recipe = Recipe(model)
    qwen4_exp_gguf(model, recipe, {"base": gguf})
    report = convert(model, recipe, out, device="cpu")
    if volume:
        qwen4_exp.write_ngram_volume(gguf, model, str(out) + ".ngram")
    return report


def test_conversion_stores_gguf_bytes_exactly(tmp_path):
    path = _record_gguf(tmp_path)
    out = tmp_path / "tiny.ninfer"
    with GgufModel(path) as gguf:
        report = _convert(gguf, out)
        summary = verify(out, gguf, volume=str(out) + ".ngram")
    # Narrowed: ssm_norm and ssm_conv1d of three GDN layers, four q/k norms of the QSA layer.
    assert summary["records"] == 4 and summary["widened"] == 1 and summary["narrowed"] == 10
    hc = [m for m in report["methods"] if "text/layers/0/attn_hc/down" in m["parameters"]]
    assert [tuple(m["parameters"]) for m in hc] == [
        ("text/layers/0/attn_hc/down", "text/layers/0/attn_hc/inject")
    ]
    assert tuple(hc[0]["shape"]) == (fixture.LOW + fixture.HC, fixture.HC * fixture.H)
    assert summary["volume_rows"] == fixture.PLE_ROWS
    assert summary["objects"] == len(report["methods"])
    banks = {
        item["parameters"][0]: (item["format"], item["layout"], tuple(item["shape"]))
        for item in report["methods"]
        if item["parameters"][0].endswith("/moe/experts")
    }
    shape = (fixture.E, fixture.H, fixture.F)
    assert banks == {
        "text/layers/0/moe/experts": ("ggml_rec_iq2_s_q2_0", "ggml_expert_record_v1", shape),
        "text/layers/1/moe/experts": ("ggml_rec_iq2_xxs_q2_0", "ggml_expert_record_v1", shape),
        "text/layers/2/moe/experts": ("ggml_rec_iq1_m_q2_0", "ggml_expert_record_v1", shape),
        "text/layers/3/moe/experts": ("ggml_rec_iq2_s_q2_0", "ggml_expert_record_v1", shape),
    }
    formats = {item["format"] for item in report["methods"]}
    assert {"ggml_q8_0", "ggml_iq4_nl", "ggml_q2_0", "bf16", "fp32"} <= formats


def test_conversion_requires_a_bound_volume(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        model = qwen4_exp.build_model(gguf)
        with pytest.raises(ValueError, match="bind the n-gram volume"):
            qwen4_exp_gguf(model, Recipe(model), {"base": gguf})


@pytest.mark.parametrize("target", ("record", "volume_row", "volume_tail"))
def test_verification_detects_a_changed_byte(tmp_path, target):
    from tools.artifact.reader import Artifact

    path = _record_gguf(tmp_path)
    out = tmp_path / "tiny.ninfer"
    with GgufModel(path) as gguf:
        report = _convert(gguf, out)
    if target == "record":
        (object_id,) = [
            m["object"] for m in report["methods"]
            if m["parameters"] == ("text/layers/2/moe/experts",)
        ]
        with Artifact(out) as artifact:
            obj = artifact.by_id[object_id]
            # A byte of expert 3's down rows: the end of the last record.
            changed, at = out, artifact.payload_offset + obj.offset + obj.bytes - 5
    else:
        changed = tmp_path / "tiny.ninfer.ngram"
        # Row 1 of block 0, or the zero tail of block 0.
        at = 4096 + (18 + 3 if target == "volume_row" else 4095)
    data = bytearray(changed.read_bytes())
    data[at] ^= 0x40
    changed.write_bytes(data)
    with GgufModel(path) as gguf, pytest.raises((AssertionError, ValueError)):
        verify(out, gguf, volume=str(out) + ".ngram")


def test_ggml_projections_permit_a8_and_bf16_projections_stay_a16(tmp_path):
    from tools.artifact.reader import Artifact

    path = _record_gguf(tmp_path)
    with GgufModel(path) as gguf:
        _convert(gguf, tmp_path / "tiny.ninfer", volume=False)
    with Artifact(tmp_path / "tiny.ninfer") as artifact:
        policies = {(u["parameter"], u["input"]): u["activation_policy"] for u in artifact.directory.uses}
    assert policies[("text/layers/0/gdn/qkv", "text/layers/0/mixer_input")] == "AllowA8"
    assert policies[("text/layers/0/moe/experts", "text/layers/0/ffn_input")] == "AllowA8"
    assert policies[("text/layers/0/moe/router", "text/layers/0/ffn_input")] == "A16Only"
    assert policies[("text/output_head", "text/final_hidden")] == "AllowA8"


def test_subset_keeps_chosen_tensors_and_first_ple_rows(tmp_path):
    subset = qwen4_exp.Subset(
        tensors=frozenset({"token_embd.weight", "blk.3.attn_q.weight"}),
        direct_layers=frozenset({1}),
        expert_layers=frozenset({2}),
        ple_rows=227,  # exactly one volume block
    )
    path = _record_gguf(tmp_path)
    out = tmp_path / "subset.ninfer"
    with GgufModel(path) as gguf:
        model = qwen4_exp.build_model(gguf, subset=subset)
        assert model.config["ngram_table"]["rows"] == 227
        assert model.config["ngram_table"]["blocks"] == 1
        assert "text/layers/3/attention/query_gate" in model.parameters
        assert "text/layers/1/ple/key" in model.parameters
        assert "text/layers/0/gdn/qkv" not in model.parameters
        assert "text/layers/1/gdn/qkv" not in model.parameters  # block format, not selected
        assert [n for n in model.parameters if "/experts" in n] == ["text/layers/2/moe/experts"]
        _convert(gguf, out, subset)
        summary = verify(out, gguf, subset, volume=str(out) + ".ngram")
    assert summary["records"] == 1 and summary["volume_rows"] == 227


def test_volume_reuse_accepts_the_same_table_and_refuses_another(tmp_path):
    path = _record_gguf(tmp_path)
    first = tmp_path / "a.ninfer"
    with GgufModel(path) as gguf:
        _convert(gguf, first)
        model = qwen4_exp.build_model(gguf)
        volume = tmp_path / "a.ninfer.ngram"
        stored = ngram_volume.read_header(volume).volume_id
        assert qwen4_exp.read_ngram_volume_id(gguf, model, volume) == stored

        # Same geometry, one sampled row changed: a volume of another table.
        other = tmp_path / "other.ngram"
        shutil.copy(volume, other)
        data = bytearray(other.read_bytes())
        g = ngram_volume.read_header(other).geometry
        data[g.row_offset(ngram_volume.reuse_samples(g.rows)[7])] ^= 1
        other.write_bytes(data)
        with pytest.raises(ValueError, match="differs from this table"):
            qwen4_exp.read_ngram_volume_id(gguf, model, other)

        # A subset's volume has another geometry.
        subset = qwen4_exp.Subset(frozenset(), frozenset(), frozenset(), 100)
        _convert(gguf, tmp_path / "s.ninfer", subset)
        with pytest.raises(ValueError, match="not an n-gram volume of this table"):
            qwen4_exp.read_ngram_volume_id(gguf, model, tmp_path / "s.ninfer.ngram")


def test_cli_writes_then_reuses_a_volume(tmp_path):
    from tools.artifact.reader import Artifact

    path = _record_gguf(tmp_path)
    subset = ["--subset", "dev"]
    dev = qwen4_exp.Subset(frozenset({"blk.0.attn_qkv.weight"}), frozenset(), frozenset({0}), 10)
    out, again = tmp_path / "cli.ninfer", tmp_path / "again.ninfer"
    base = ["--model", str(path), "--recipe", "qwen4_exp_gguf", "--device", "cpu"]
    original = qwen4_exp.SUBSETS["dev"]
    qwen4_exp.SUBSETS["dev"] = dev
    try:
        main([*base, *subset, "--out", str(out)])
        volume = tmp_path / "cli.ninfer.ngram"
        report = json.loads((tmp_path / "cli.ninfer.conversion.json").read_text())
        assert report["provenance"]["subset"] == "dev"
        assert report["provenance"]["sources"]["base"]["paths"] == [str(path)]
        with GgufModel(path) as gguf:
            verify(out, gguf, dev, volume=volume)
        main([*base, *subset, "--ngram-reuse", str(volume), "--out", str(again)])
        assert not (tmp_path / "again.ninfer.ngram").exists()
        ids = []
        for artifact_path in (out, again):
            with Artifact(artifact_path) as artifact:
                ids.append(artifact.directory.components["text"]["config"]["ngram_table"]["volume_id"])
        assert ids[0] == ids[1] == ngram_volume.read_header(volume).volume_id.hex()
        with pytest.raises(FileExistsError, match="already exists"):
            main([*base, *subset, "--ngram-out", str(volume), "--out", str(tmp_path / "x.ninfer")])
        with pytest.raises(ValueError, match="exclusive"):
            main([*base, *subset, "--ngram-out", str(tmp_path / "v"), "--ngram-reuse", str(volume),
                  "--out", str(tmp_path / "y.ninfer")])
    finally:
        qwen4_exp.SUBSETS["dev"] = original
    with pytest.raises(ValueError, match="only the text component"):
        main([*base, "--components", "text,vision", "--out", str(tmp_path / "z.ninfer")])


def test_synthesized_tokenizer_and_override_check(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        config = qwen4_exp.text_config(gguf.metadata)
        resources = qwen4_exp.tokenizer_resources(gguf.metadata, config)
        tokenizer = json.loads(resources["tokenizer.json"])
        assert tokenizer["model"]["vocab"] == {chr(ord("a") + i): i for i in range(10)}
        assert tokenizer["model"]["merges"] == ["a b", "ab c"]
        assert [(t["id"], t["special"]) for t in tokenizer["added_tokens"]] == [
            (10, True), (11, True), (12, True), (13, False)
        ]
        assert tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == (
            qwen4_exp.QWEN35_SPLIT_PATTERN
        )
        settings = json.loads(resources["tokenizer_config.json"])
        assert (settings["eos_token"], settings["pad_token"]) == ("<|im_end|>", "<|endoftext|>")
        assert json.loads(resources["generation_config.json"])["eos_token_id"] == [12, 10]

        # A Hugging Face tokenizer may clear "special" on a <|...|> token llama.cpp made CONTROL.
        candidate = json.loads(resources["tokenizer.json"])
        candidate["added_tokens"][1]["special"] = False
        assert qwen4_exp.check_tokenizer(candidate, tokenizer) == [11]
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        model = qwen4_exp.build_model(gguf, resource_overrides={"tokenizer.json": path})
        assert model.special_token_ids == (10, 12)
        for change, message in (
            (lambda t: t["added_tokens"][3].update(special=True), "special flag conflicts"),
            (lambda t: t["model"]["merges"].pop(), "model.merges"),
            (lambda t: t["added_tokens"][0].update(content="<|eot|>"), "added tokens differ"),
        ):
            bad = json.loads(resources["tokenizer.json"])
            change(bad)
            with pytest.raises(ValueError, match=message):
                qwen4_exp.check_tokenizer(bad, tokenizer)


def test_unsupported_tokenizer_is_rejected(tmp_path):
    path = fixture.write(tmp_path / "t.gguf", metadata_changes={"tokenizer.ggml.pre": (8, "llama3")})
    with GgufModel(path) as gguf, pytest.raises(GgufError, match="qwen35"):
        qwen4_exp.build_model(gguf)
