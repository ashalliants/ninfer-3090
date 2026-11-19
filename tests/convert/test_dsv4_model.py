"""The exact repack of a (tiny, synthetic) deepseek41 GGUF into a v3 artifact, read back and compared
byte for byte with the GGUF, and the refusals of its model builder."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from tools.artifact.codecs.ggml_blocks import unpack_expert_records
from tools.artifact.layouts import ggml_expert_record_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import dsv4, dsv4_model
from tools.convert.official_recipes import RECIPES
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufModel

from . import dsv4_fixture as fixture

FIXTURES = Path(__file__).parent.parent / "fixtures" / "dsv4"


@pytest.fixture
def gguf_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def _convert(gguf, out):
    model = dsv4_model.build_model(gguf)
    recipe = Recipe(model)
    RECIPES["dsv4_gguf"](model, recipe, {"base": gguf})
    return model, convert(model, recipe, out, device="cpu")


def _sha(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _verify(out, gguf) -> dict:
    """Every tensor object equals its GGUF bytes; banks are unpacked to each expert's rows."""
    mappings = defaultdict(list)
    for item in dsv4.name_map(gguf.tensors, dsv4.text_config(gguf.metadata, fixture.VOCAB)):
        mappings[item.parameter].append(item)
    summary = defaultdict(int)
    with Artifact(out) as artifact:
        parts = defaultdict(list)
        for name, binding in artifact.directory.bindings.items():
            for object_id, begin, end in binding_parts(binding, artifact.by_id, name):
                parts[object_id].append((begin, end, name))
        for obj in artifact.directory.objects:
            if not isinstance(obj, TensorObject):
                continue
            ((_, _, name),) = sorted(parts[obj.id])  # one parameter per object: nothing is packed
            if obj.layout == "ggml_expert_record_v1":
                items = sorted(mappings[name], key=lambda item: {"gate": 0, "up": 1, "down": 2}[item.part])
                g = ggml_expert_record_geometry(obj.format, obj.shape)
                payload = artifact.read_range(obj.offset, g.experts * g.record_stride)
                stored = unpack_expert_records(payload, obj.format, (g.experts, g.hidden, g.intermediate))
                for item, rows, (role, _, n_rows, _) in zip(items, stored, g.parts()):
                    tensor = gguf.tensor(item.gguf)
                    source = gguf.read_rows(tensor, 0, g.experts * n_rows)
                    assert np.array_equal(rows.reshape(-1, rows.shape[-1]), source), f"{name}: {role} differs"
                summary["banks"] += 1
                summary[obj.format] += 1
            else:
                tensor = gguf.tensor(mappings[name][0].gguf)
                assert obj.shape == tensor.shape, name
                assert _sha(artifact.iter_object(obj.id)) == _sha(gguf.iter_range(tensor, 0, tensor.bytes)), name
                summary["plain"] += 1
                summary[obj.format] += 1
    return dict(summary)


def test_conversion_stores_gguf_bytes_exactly(gguf_path, tmp_path):
    out = tmp_path / "tiny.ninfer"
    with GgufModel(gguf_path) as gguf:
        model, report = _convert(gguf, out)
        summary = _verify(out, gguf)
    # Every GGUF tensor but the Engram table is one object: the experts of each layer are one bank.
    assert len(model.parameters) == len(gguf.tensors) - 1 - 2 * fixture.LAYERS
    assert summary["banks"] == fixture.LAYERS
    # Each layer's own pair of GGML types became its own record format.
    assert [summary[f"ggml_rec_{a}_{b}"] for a, b in (
        ("iq1_m", "iq1_m"), ("iq2_xxs", "iq1_m"), ("iq2_xxs", "iq2_xs"), ("iq2_xs", "iq2_xxs"), ("iq2_xs", "iq2_xs"),
    )] == [1, 1, 1, 1, 1]
    # Block tensors keep their GGML blocks, direct ones their words.
    assert summary["ggml_q8_0"] > 0 and summary["fp32"] > 0 and summary["bf16"] > 0
    assert report["formats"]["ggml_rec_iq2_xs_iq2_xs"] == 1


def test_the_text_config_and_parameter_names_are_recorded(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        model, _ = _convert(gguf, tmp_path / "tiny.ninfer")
    config = model.config
    assert config["model_type"] == "deepseek41_text" and config["vocab_size"] == fixture.VOCAB
    assert config["kv_source_layers"] == fixture.KV_LAYERS and config["engram_layers"] == fixture.ENGRAM_LAYERS
    names = set(model.parameters)
    assert "text/layers/2/attention/compressor/gate" in names
    assert "text/layers/4/attention/compressor/gate" not in names  # ratio 1: no gate
    assert "text/layers/3/indexer/q_b" in names and "text/layers/3/indexer/k" not in names
    assert "text/layers/1/engram/kv" in names and "text/layers/0/engram/kv" not in names
    assert {dsv4.expert_bank(layer) for layer in range(fixture.LAYERS)} <= names
    # Projections carry an input (so GGML ones may take A8 activations); norms and mixers do not.
    assert model.parameters["text/layers/0/attention/q_a"].projection
    assert not model.parameters["text/layers/0/attention/norm"].projection
    assert model.parameters["text/head"].projection


def test_the_engram_table_is_not_a_parameter(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        model, _ = _convert(gguf, tmp_path / "tiny.ninfer")
    assert not any("engram_table" in n or n.endswith("per_layer_token_embd") for n in model.parameters)
    assert all(p.source.shape != (fixture.ENGRAM_ROWS, fixture.ENGRAM_ROW) for p in model.parameters.values())


def test_an_expert_pair_without_a_record_format_is_refused(tmp_path):
    # IQ2_XXS gate/up with an IQ4_NL down is a Flash-Next pair, but IQ1_M with IQ4_NL is registered nowhere.
    bad = fixture.EXPERT_TYPES[:4] + ((fixture.IQ1_M, fixture.IQ4_NL),)
    path = fixture.write(tmp_path / "bad.gguf", experts=bad)
    with GgufModel(path) as gguf:
        with pytest.raises(dsv4.Dsv4Error, match="layer 4: no expert record format"):
            dsv4_model.build_model(gguf)


def test_a_malformed_gguf_is_refused_before_conversion(tmp_path):
    path = fixture.write(tmp_path / "missing.gguf", drop=("blk.3.attn_q_b.weight",))
    with GgufModel(path) as gguf:
        with pytest.raises(dsv4.Dsv4Error, match="layer 3 is missing"):
            dsv4_model.build_model(gguf)


def test_the_recipe_refuses_another_model(gguf_path):
    with GgufModel(gguf_path) as gguf:
        model = dsv4_model.build_model(gguf)
    model.components["text"]["config"]["model_type"] = "qwen4_exp_text"
    with pytest.raises(ValueError, match="requires the deepseek41 GGUF model"):
        dsv4_model.dsv4_gguf(model, Recipe(model))


def test_every_layer_of_the_real_iq2_xxs_build_has_a_record_format():
    """The real build's (gate/up, down) pair of each of its 40 layers is a registered record format,
    and the banks add up to the plan's 132.1 GB of routed experts (to 0.5%)."""
    from tools.artifact.formats import ggml_record_format

    header = json.loads((FIXTURES / "ista_iq2_xxs_header.json").read_text())
    names = {16: "IQ2_XXS", 17: "IQ2_XS", 29: "IQ1_M"}
    types = {t["name"]: names[t["type"]] for t in header["tensors"] if "_exps" in t["name"]}
    total = 0
    for layer in range(40):
        gate_up, down = dsv4.expert_record_types(types, layer)
        record = ggml_record_format(f"ggml_{gate_up.lower()}", f"ggml_{down.lower()}")
        total += ggml_expert_record_geometry(record.name, (384, 5120, 2304)).record_stride * 384
    assert abs(total / 132.1e9 - 1) < 0.005, total
