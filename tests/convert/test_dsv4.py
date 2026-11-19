"""DeepSeek-V4.1-Flash's config and tensor map, against the real GGUF headers of ISTA-DASLab's
GSQ-RCO IQ2_XXS build (tests/fixtures/dsv4/ista_iq2_xxs_header.json: all 1,043 tensors' names,
shapes and GGML types, and the metadata) and against mutations of it that must each be refused."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.convert import dsv4

# GGML type ids in the fixture (ggml.h). Not imported from the GGUF reader, which needs torch.
GGML_NAMES = {0: "F32", 8: "Q8_0", 13: "Q5_K", 16: "IQ2_XXS", 17: "IQ2_XS", 20: "IQ4_NL", 29: "IQ1_M", 30: "BF16"}

FIXTURE = Path(__file__).parent.parent / "fixtures" / "dsv4" / "ista_iq2_xxs_header.json"


@dataclass(frozen=True)
class Tensor:
    ne: tuple[int, ...]
    type_name: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.ne))


@pytest.fixture(scope="module")
def header() -> dict:
    return json.loads(FIXTURE.read_text())


def tensors_of(header: dict) -> dict[str, Tensor]:
    return {
        t["name"]: Tensor(tuple(t["ne"]), GGML_NAMES[t["type"]])
        for t in header["tensors"]
    }


@pytest.fixture
def config(header) -> dict:
    return dsv4.text_config(header["metadata"], vocab_size=129280)


@pytest.fixture
def tensors(header) -> dict[str, Tensor]:
    return tensors_of(header)


def test_config_from_real_metadata(config):
    assert config["architecture"] == "deepseek41"
    assert (config["num_hidden_layers"], config["hidden_size"], config["vocab_size"]) == (40, 5120, 129280)
    assert (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (64, 1, 512)
    assert (config["q_lora_rank"], config["o_groups"], config["o_lora_rank"]) == (1280, 8, 1024)
    assert (config["num_experts"], config["num_experts_per_tok"], config["moe_intermediate_size"]) == (384, 6, 2304)
    assert (config["hc_count"], config["hc_sinkhorn_iterations"]) == (4, 20)
    assert config["kv_source_layers"] == [2, 8, 14, 20]
    assert config["indexer_source_layers"] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert config["engram_layers"] == [1, 14]
    assert len(config["compress_ratios"]) == 40 and len(config["swiglu_clamp_experts"]) == 40
    assert config["routed_scaling_factor"] == 1.5 and config["norm_topk_prob"] is True


def test_every_real_tensor_maps(header, config, tensors):
    mapping = dsv4.name_map(tensors, config)
    assert len(mapping) == 1043 == len(header["tensors"])
    assert len({m.gguf for m in mapping}) == 1043
    banks = {m.parameter for m in mapping if m.part}
    assert banks == {dsv4.expert_bank(layer) for layer in range(40)}
    assert sum(1 for m in mapping if m.part) == 120  # gate, up, down of 40 layers
    volume = [m for m in mapping if m.parameter is None]
    assert [m.gguf for m in volume] == [dsv4.ENGRAM_TABLE]
    assert volume[0].shape == (sum(config["engram_head_vocab_sizes"]), 256)
    # Layer 14 holds everything optional: compressor with gate, both indexer sides and Engram.
    layer14 = {m.parameter.split("/", 3)[3] for m in mapping if m.parameter and m.parameter.startswith("text/layers/14/")}
    assert {"attention/compressor/gate", "indexer/k", "indexer/q_b", "engram/kv", "engram/gate"} <= layer14
    # Layer 20's compressor has no gate (ratio 1); layer 24 has the indexer's query side only.
    names = {m.gguf for m in mapping}
    assert "blk.20.attn_compressor_kv.weight" in names and "blk.20.attn_compressor_gate.weight" not in names
    assert "blk.24.indexer.attn_q_b.weight" in names and "blk.24.indexer.attn_k.weight" not in names


def test_expert_record_types_are_per_layer(header, tensors):
    types = {name: t.type_name for name, t in tensors.items()}
    pairs = [dsv4.expert_record_types(types, layer) for layer in range(40)]
    # GSQ-RCO picks a type per layer: this build mixes IQ1_M, IQ2_XXS and IQ2_XS across layers.
    assert set(pairs) == {
        ("IQ1_M", "IQ1_M"), ("IQ2_XXS", "IQ1_M"), ("IQ2_XXS", "IQ2_XXS"), ("IQ2_XS", "IQ1_M"),
        ("IQ2_XXS", "IQ2_XS"), ("IQ1_M", "IQ2_XXS"), ("IQ2_XS", "IQ2_XXS"), ("IQ2_XS", "IQ2_XS"),
    }


def test_gate_and_up_must_share_a_type(tensors):
    types = {name: t.type_name for name, t in tensors.items()}
    types["blk.3.ffn_up_exps.weight"] = "Q8_0"
    with pytest.raises(dsv4.Dsv4Error, match="layer 3: gate experts are"):
        dsv4.expert_record_types(types, 3)


def _refused(tensors, config, match):
    with pytest.raises(dsv4.Dsv4Error, match=match):
        dsv4.name_map(tensors, config)


def test_a_missing_tensor_is_refused(tensors, config):
    del tensors["blk.7.attn_q_b.weight"]
    _refused(tensors, config, r"layer 7 is missing \['attn_q_b.weight'\]")


def test_a_missing_expert_tensor_is_refused(tensors, config):
    del tensors["blk.9.ffn_down_exps.weight"]
    _refused(tensors, config, r"layer 9 is missing \['ffn_down_exps.weight'\]")


def test_a_tensor_on_the_wrong_layer_is_refused(tensors, config):
    tensors["blk.5.engram_kv.weight"] = tensors["blk.1.engram_kv.weight"]
    _refused(tensors, config, r"layer 5 has unexpected tensors \['engram_kv.weight'\]")


def test_a_compressor_gate_where_the_ratio_is_one_is_refused(tensors, config):
    tensors["blk.20.attn_compressor_gate.weight"] = tensors["blk.14.attn_compressor_gate.weight"]
    _refused(tensors, config, r"layer 20 has unexpected tensors \['attn_compressor_gate.weight'\]")


def test_an_unknown_tensor_is_refused(tensors, config):
    tensors["mystery.weight"] = Tensor((4,), "F32")
    _refused(tensors, config, "unknown GGUF tensors")


def test_a_layer_beyond_the_config_is_refused(tensors, config):
    tensors["blk.40.attn_norm.weight"] = tensors["blk.0.attn_norm.weight"]
    _refused(tensors, config, "unknown GGUF tensors")


@pytest.mark.parametrize(
    "name",
    [
        "token_embd.weight", "output.weight", "blk.0.attn_q_a.weight", "blk.0.hc_attn_fn.weight",
        "blk.2.indexer.attn_k.weight", "blk.14.engram_kv.weight", "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_down_exps.weight", "blk.0.exp_probs_b.bias", dsv4.ENGRAM_TABLE,
    ],
)
def test_a_wrong_shape_is_refused(tensors, config, name):
    ne = tensors[name].ne
    tensors[name] = Tensor((ne[0] + 1, *ne[1:]), tensors[name].type_name)
    _refused(tensors, config, "shape")


def test_a_missing_global_tensor_is_refused(tensors, config):
    del tensors["output_norm.weight"]
    _refused(tensors, config, "'output_norm.weight' is missing")
    tensors["output_norm.weight"] = Tensor((5120,), "F32")
    del tensors[dsv4.ENGRAM_TABLE]
    _refused(tensors, config, "per_layer_token_embd.weight' is missing")


def test_metadata_is_validated(header):
    def bad(**changes):
        metadata = copy.deepcopy(header["metadata"])
        for key, value in changes.items():
            metadata["deepseek41." + key.replace("__", ".")] = value
        return metadata

    with pytest.raises(dsv4.Dsv4Error, match="general.architecture"):
        dsv4.text_config({**header["metadata"], "general.architecture": "qwen4exp"}, 129280)
    with pytest.raises(dsv4.Dsv4Error, match="hash_layer_count"):
        dsv4.text_config(bad(hash_layer_count=3), 129280)
    with pytest.raises(dsv4.Dsv4Error, match="kv_source_layers"):
        dsv4.text_config(bad(attention__kv_source_layers=[2, 2]), 129280)
    with pytest.raises(dsv4.Dsv4Error, match="kv_source_layers"):
        dsv4.text_config(bad(attention__kv_source_layers=[2, 99]), 129280)
    with pytest.raises(dsv4.Dsv4Error, match="expert_used_count"):
        dsv4.text_config(bad(expert_used_count=385), 129280)
    with pytest.raises(dsv4.Dsv4Error, match="is missing"):
        metadata = copy.deepcopy(header["metadata"])
        del metadata["deepseek41.expert_count"]
        dsv4.text_config(metadata, 129280)
    with pytest.raises(dsv4.Dsv4Error, match="swiglu_clamp_exp"):
        dsv4.text_config(bad(swiglu_clamp_exp=[10.0] * 5), 129280)
