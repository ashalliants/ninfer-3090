"""Inkling-2-Mini GGUF config and tensor map, against the real Unsloth headers (tests/fixtures/inkling2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.convert import inkling2
from tools.convert.inkling2 import Inkling2Error

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "inkling2"
VARIANTS = ("ud_q4_k_xl", "ud_q8_k_xl", "ud_q4_k_m")

# config.json of thinkingmachines/Inkling-2-Mini (text_config), the independent statement of the shape.
LOCAL_LAYER_IDS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    type: str


def header(variant: str) -> dict:
    return json.loads((FIXTURES / f"{variant}_header.json").read_text())


def tensors_of(h: dict) -> dict[str, Tensor]:
    return {
        t["name"]: Tensor(tuple(reversed(t["ne"])), inkling2.GGML_TYPE_NAMES[t["type"]]) for t in h["tensors"]
    }


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request):
    return request.param


@pytest.fixture(scope="module")
def q4() -> dict:
    return header("ud_q4_k_xl")


@pytest.fixture(scope="module")
def config(q4) -> dict:
    return inkling2.text_config(q4["metadata"])


@pytest.fixture()
def tensors(q4) -> dict[str, Tensor]:
    return tensors_of(q4)


def test_config_from_unsloth_metadata(config):
    assert config["hidden_size"] == 2304
    assert config["num_hidden_layers"] == 24 and config["num_layer_visits"] == 48
    assert (config["loop_core_start"], config["loop_core_end"], config["loop_num_passes"]) == (6, 18, 3)
    assert config["loop_gate_rank"] == 64
    assert config["local_layers"] == LOCAL_LAYER_IDS
    assert (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (24, 8, 128)
    assert (config["d_rel"], config["rel_extent"], config["sliding_window"]) == (16, 1024, 1024)
    assert config["qk_clip"] == pytest.approx(22.627416997969522, rel=1e-7)
    assert config["rel_clip"] == 8.0
    assert config["log_scaling_n_floor"] == 8192 and config["log_scaling_alpha"] == pytest.approx(0.1)
    assert (config["num_experts"], config["num_experts_per_tok"], config["moe_intermediate_size"]) == (128, 8, 1152)
    assert (config["route_scale"], config["mlp_gate_softcap"], config["mlp_up_softcap"]) == (8.0, 4.0, 25.0)
    assert config["final_logit_softcapping"] == 30.0
    assert (config["vocab_size"], config["unpadded_vocab_size"]) == (201024, 200058)
    assert config["conv_kernel_size"] == 4 and config["attn_res_block_size"] == 6


def test_every_unsloth_tensor_maps(variant):
    h = header(variant)
    config = inkling2.text_config(h["metadata"])
    tensors = tensors_of(h)
    mapped = inkling2.name_map(tensors, config)
    assert len(mapped) == len(tensors) == 734
    banks = {m.parameter for m in mapped if m.part is not None}
    assert banks == {inkling2.expert_bank(layer) for layer in range(24)}
    loops = {m.parameter.split("/")[2] for m in mapped if "/loop/" in m.parameter and m.parameter.startswith("text/layers")}
    assert loops == {str(layer) for layer in range(6, 18)}


EXPERT_TYPES = {  # (layers 0-19, layers 20-23): the checkpoint's NVFP4 and MXFP8 experts, as each build stores them
    "ud_q4_k_xl": (("NVFP4", "NVFP4"), ("Q8_0", "Q8_0")),
    "ud_q8_k_xl": (("NVFP4", "NVFP4"), ("BF16", "BF16")),
    "ud_q4_k_m": (("NVFP4", "NVFP4"), ("Q5_K", "Q8_0")),
}


def test_expert_record_types(variant):
    types = {name: t.type for name, t in tensors_of(header(variant)).items()}
    nvfp4, last = EXPERT_TYPES[variant]
    for layer in range(24):
        assert inkling2.expert_record_types(types, layer) == (nvfp4 if layer < 20 else last)


def test_expert_scales_exactly_on_nvfp4_parts(variant):
    h = header(variant)
    mapped = inkling2.name_map(tensors_of(h), inkling2.text_config(h["metadata"]))
    scaled = {(m.parameter, m.part) for m in mapped if m.scale}
    assert scaled == {(inkling2.expert_bank(layer), part) for layer in range(20) for part in ("gate_up", "down")}


def test_visit_schedule_matches_llama_cpp(config):
    # llama.cpp loop_layer: identity before the core, (v - 6) mod 12 + 6 inside 3 passes, v - 24 after.
    expected = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    assert [inkling2.physical_layer(config, v) for v in range(48)] == expected
    assert inkling2.visits_of(config, 11) == [11, 23, 35]
    assert inkling2.visits_of(config, 20) == [44]
    # DSpark taps count visits: 23 is physical 11 in pass 1, 41 physical 17 in pass 2.
    assert (inkling2.physical_layer(config, 23), inkling2.visit_pass(config, 23)) == (11, 1)
    assert (inkling2.physical_layer(config, 41), inkling2.visit_pass(config, 41)) == (17, 2)
    with pytest.raises(Inkling2Error):
        inkling2.physical_layer(config, 48)


def test_state_sizes(config):
    assert inkling2.kv_bytes_per_token(config) == 8 * 2 * 8 * 128 * 2  # 8 global visits, FP16 K and V
    assert inkling2.local_kv_bytes(config) == 40 * 1024 * 4096
    assert inkling2.conv_state_bytes(config) == 48 * (2048 + 4608) * 3 * 4


def test_sizes_match_the_file(variant):
    h = header(variant)
    tensors = tensors_of(h)
    total = sum(inkling2.tensor_bytes(t.shape, t.type) for t in tensors.values())
    expected = {"ud_q4_k_xl": 17617780736, "ud_q8_k_xl": 22855473152, "ud_q4_k_m": 16598564864}[variant]
    assert total == expected  # the data section, without alignment padding


def test_decode_bytes_per_token(q4, config):
    groups = inkling2.decode_bytes_per_token(config, tensors_of(q4))
    assert groups["head"] == 492_106_752
    assert 1.80e9 < groups["experts"] < 1.90e9
    assert 1.40e9 < groups["dense"] < 1.50e9  # the core's attention weights stream three times
    q8 = inkling2.decode_bytes_per_token(config, tensors_of(header("ud_q8_k_xl")))
    assert sum(q8.values()) / sum(groups.values()) == pytest.approx(1.44, abs=0.02)


# --------------------------------------------------------------------------- refusals


def _refused(tensors, config, match):
    with pytest.raises(Inkling2Error, match=match):
        inkling2.name_map(tensors, config)


def test_a_missing_tensor_is_refused(tensors, config):
    del tensors["blk.7.attn_rel_bias.weight"]
    _refused(tensors, config, "missing")


def test_a_missing_global_tensor_is_refused(tensors, config):
    del tensors["loop_kv_res_query.weight"]
    _refused(tensors, config, "loop_kv_res_query")


def test_an_unknown_tensor_is_refused(tensors, config):
    tensors["blk.3.attn_q_norm.weight"] = Tensor((128,), "F32")
    _refused(tensors, config, "unexpected")


def test_a_stray_global_tensor_is_refused(tensors, config):
    tensors["rope_freqs.weight"] = Tensor((64,), "F32")
    _refused(tensors, config, "unknown GGUF tensors")


def test_loop_controllers_outside_the_core_are_refused(tensors, config):
    tensors["blk.18.loop_k_gain.weight"] = tensors["blk.6.loop_k_gain.weight"]
    _refused(tensors, config, "layer 18 has unexpected")


def test_a_layer_zero_residual_query_is_refused(tensors, config):
    tensors["blk.0.attn_res_query.weight"] = Tensor((2304,), "F32")
    _refused(tensors, config, "layer 0 has unexpected")


def test_global_attention_tensors_on_a_local_layer_are_refused(tensors, config):
    tensors["blk.4.attn_k_conv_proj.weight"] = tensors["blk.5.attn_k_conv_proj.weight"]
    _refused(tensors, config, "layer 4 has unexpected")


def test_an_nvfp4_expert_without_its_scale_is_refused(tensors, config):
    del tensors["blk.3.ffn_down_exps.scale"]
    _refused(tensors, config, "missing")


def test_a_scale_on_a_block_quantized_expert_is_refused(tensors, config):
    tensors["blk.21.ffn_down_exps.scale"] = Tensor((128,), "F32")
    _refused(tensors, config, "unexpected")


@pytest.mark.parametrize(
    "name, shape",
    [
        ("blk.5.attn_res_query.weight", (2304,)),  # a global layer mixes three queries
        ("blk.6.attn_res_query.weight", (3, 2304)),
        ("blk.2.ffn_gate_up_exps.weight", (128, 1152, 2304)),
        ("blk.9.loop_v_down.weight", (64, 2304)),
        ("output.weight", (200058, 2304)),
    ],
)
def test_a_wrong_shape_is_refused(tensors, config, name, shape):
    tensors[name] = Tensor(shape, tensors[name].type)
    _refused(tensors, config, "shape")


def _metadata(q4, **changes):
    m = dict(q4["metadata"])
    for key, value in changes.items():
        key = "inkling2." + key.replace("__", ".")
        if value is None:
            del m[key]
        else:
            m[key] = value
    return m


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"num_loops": 1}, "looped variant"),
        ({"loop__core_start": 4}, "whole attention-residual blocks"),
        ({"loop__core_end": 17}, "whole attention-residual blocks"),
        ({"rope__dimension_count": 64}, "no RoPE"),
        ({"vocab_size_unpadded": 300000}, "exceeds"),
        ({"attention__key_length_swa": 64}, "must equal"),
        ({"loop__gate_rank": None}, "missing"),
        ({"attention__qk_clip": float("nan")}, "finite"),
    ],
)
def test_metadata_is_validated(q4, changes, match):
    with pytest.raises(Inkling2Error, match=match):
        inkling2.text_config(_metadata(q4, **changes))


def test_a_visit_that_differs_from_its_layer_is_refused(q4):
    pattern = list(q4["metadata"]["inkling2.attention.sliding_window_pattern"])
    assert pattern[23] is False  # physical 11 in pass 1, a global layer
    pattern[23] = True
    with pytest.raises(Inkling2Error, match="differs from the first visit"):
        inkling2.text_config(_metadata(q4, attention__sliding_window_pattern=pattern))


def test_head_counts_must_be_uniform(q4):
    heads = list(q4["metadata"]["inkling2.attention.head_count"])
    heads[30] = 16
    with pytest.raises(Inkling2Error, match="same for every layer visit"):
        inkling2.text_config(_metadata(q4, attention__head_count=heads))


def test_another_architecture_is_refused(q4):
    m = dict(q4["metadata"])
    m["general.architecture"] = "inkling"
    with pytest.raises(Inkling2Error, match="not 'inkling2'"):
        inkling2.text_config(m)
