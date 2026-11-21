"""The exact repack of a tiny synthetic Inkling-2 GGUF into a v3 artifact, read back byte for byte, its resources
and the refusals of the model builder and CLI."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from tools.artifact.codecs.ggml_blocks import decode_blocks, unpack_expert_records
from tools.artifact.layouts import ggml_expert_record_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import inkling2_model
from tools.convert.inkling2 import Inkling2Error
from tools.convert.official_recipes import RECIPES
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufModel

from . import inkling2_fixture as fixture

FIXTURES = Path(__file__).parent.parent / "fixtures" / "inkling2"


@pytest.fixture
def gguf_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def _convert(gguf, out):
    model = inkling2_model.build_model(gguf)
    recipe = Recipe(model)
    RECIPES["inkling2_gguf"](model, recipe, {"base": gguf})
    return model, convert(model, recipe, out, device="cpu")


def _sha(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _objects(out):
    with Artifact(out) as artifact:
        names = {}
        for name, binding in artifact.directory.bindings.items():
            for object_id, _, _ in binding_parts(binding, artifact.by_id, name):
                names[object_id] = name
        for obj in artifact.directory.objects:
            if isinstance(obj, TensorObject):
                yield artifact, names[obj.id], obj


def test_conversion_stores_gguf_bytes_exactly(gguf_path, tmp_path):
    out = tmp_path / "tiny.ninfer"
    with GgufModel(gguf_path) as gguf:
        model, report = _convert(gguf, out)
        source = {p.name: p.source.label for p in model.parameters.values()}
        summary = defaultdict(int)
        for artifact, name, obj in _objects(out):
            if obj.layout == "ggml_expert_record_v1":
                layer = int(name.split("/")[2])
                g = ggml_expert_record_geometry(obj.format, obj.shape)
                payload = artifact.read_range(obj.offset, g.experts * g.record_stride)
                gate, up, down = unpack_expert_records(payload, obj.format, (g.experts, g.hidden, g.intermediate))
                fused = gguf.read_rows(gguf.tensor(f"blk.{layer}.ffn_gate_up_exps.weight"), 0, g.experts * 2 * g.intermediate)
                fused = fused.reshape(g.experts, 2, g.intermediate, -1)  # each expert's gate rows, then its up rows
                assert np.array_equal(gate, fused[:, 0]) and np.array_equal(up, fused[:, 1]), name
                rows = gguf.read_rows(gguf.tensor(f"blk.{layer}.ffn_down_exps.weight"), 0, g.experts * g.hidden)
                assert np.array_equal(down.reshape(rows.shape), rows), name
                summary[obj.format] += 1
            else:
                tensor = gguf.tensor(source[name])
                assert obj.shape == tensor.shape, name
                assert _sha(artifact.iter_object(obj.id)) == _sha(gguf.iter_range(tensor, 0, tensor.bytes)), name
                summary[obj.format] += 1
    # Layers 0-3 are NVFP4 banks, layers 4 and 5 two BF16 tensors each.
    assert summary["ggml_rec_nvfp4_nvfp4"] == 4
    assert report["formats"]["ggml_rec_nvfp4_nvfp4"] == 4
    # Every GGUF tensor is one object, except that an NVFP4 layer's two expert tensors are one bank.
    assert len(model.parameters) == len(gguf.tensors) - 4
    assert summary["ggml_q8_0"] > 0 and summary["bf16"] > 0 and summary["fp32"] > 0


def test_parameter_names_and_roles(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        model = inkling2_model.build_model(gguf)
    p = model.parameters
    assert model.config["model_type"] == "inkling2_text" and model.config["num_layer_visits"] == fixture.VISITS
    assert p["text/layers/0/moe/experts"].shape == (fixture.EXPERTS, fixture.HIDDEN, fixture.INTER)
    assert p["text/layers/3/moe/experts_gate_up_scale"].shape == (fixture.EXPERTS,)
    assert "text/layers/4/moe/experts_gate_up_scale" not in p and "text/layers/4/moe/experts" not in p
    assert p["text/layers/5/moe/experts_gate_up"].shape == (fixture.EXPERTS, 2 * fixture.INTER, fixture.HIDDEN)
    assert p["text/layers/5/moe/experts_down"].shape == (fixture.EXPERTS, fixture.HIDDEN, fixture.INTER)
    assert "text/layers/2/loop/k/gain" in p and "text/layers/4/loop/k/gain" not in p
    assert "text/layers/1/attention/k_conv_proj" in p and "text/layers/0/attention/k_conv_proj" not in p
    assert p["text/layers/1/attention/q"].projection and not p["text/layers/1/attention/rel_bias"].projection
    assert p["text/head"].shape == (fixture.VOCAB, fixture.HIDDEN)  # padded rows kept


def test_block_experts_without_a_record_format_are_refused(tmp_path):
    # UD-Q4_K_XL's layers 20-23 (Q8_0 experts): no record format is registered for them yet.
    path = fixture.write(tmp_path / "q8.gguf", {**fixture.EXPERT_TYPES, 4: fixture.Q8_0})
    with GgufModel(path) as gguf, pytest.raises(Inkling2Error, match="layer 4: no expert record format"):
        inkling2_model.build_model(gguf)


def test_expert_scales_must_be_f32(tmp_path):
    path = fixture.write(tmp_path / "bf16_scale.gguf", scale_type=fixture.BF16)
    with GgufModel(path) as gguf, pytest.raises(Inkling2Error, match="ffn_down_exps.scale: type BF16, expected F32"):
        inkling2_model.build_model(gguf)


def test_nvfp4_values_survive(gguf_path, tmp_path):
    """The stored NVFP4 blocks decode to ggml's values (the codec is checked against ggml separately)."""
    out = tmp_path / "tiny.ninfer"
    with GgufModel(gguf_path) as gguf:
        _convert(gguf, out)
        expected = decode_blocks("ggml_nvfp4", gguf.read_range(gguf.tensor("blk.0.ffn_down_exps.weight"), 0, 36 * 4))
    for artifact, name, obj in _objects(out):
        if name == "text/layers/0/moe/experts":
            g = ggml_expert_record_geometry(obj.format, obj.shape)
            payload = artifact.read_range(obj.offset, g.experts * g.record_stride)
            _, _, down = unpack_expert_records(payload, obj.format, (g.experts, g.hidden, g.intermediate))
            assert np.array_equal(decode_blocks("ggml_nvfp4", down[0].reshape(-1)[: 36 * 4]), expected)
            return
    raise AssertionError("no layer-0 bank")


def test_resources_from_the_checkpoint_tokenizer(gguf_path, tmp_path):
    with GgufModel(gguf_path) as gguf:
        metadata = dict(gguf.metadata)
    config = {"unpadded_vocab_size": fixture.UNPADDED}
    files = fixture.tokenizer_files(tmp_path)
    out = inkling2_model.resources(metadata, config, files)
    assert set(out) == {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}
    assert out["tokenizer.json"] == files["tokenizer.json"].read_bytes()
    assert out["chat_template.jinja"] == b"{{ messages }}"
    generation = json.loads(out["generation_config.json"])
    tokens = metadata["tokenizer.ggml.tokens"]
    assert tokens[generation["eos_token_id"][0]] == "<|content_model_end_sampling|>"
    assert tokens[generation["pad_token_id"]] == "<|endoftext|>"
    with pytest.raises(Inkling2Error, match="tokenizer_config.json"):
        inkling2_model.resources(metadata, config, {"tokenizer.json": files["tokenizer.json"]})
    files["tokenizer_config.json"].write_text(json.dumps({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}))
    with pytest.raises(Inkling2Error, match="eos_token"):
        inkling2_model.resources(metadata, config, files)
    swapped = fixture.tokenizer()
    swapped["model"]["vocab"].update(t3=4, t4=3)
    files["tokenizer.json"].write_text(json.dumps(swapped))
    files["tokenizer_config.json"].write_text(json.dumps({"eos_token": "<|content_model_end_sampling|>",
                                                          "pad_token": "<|endoftext|>"}))
    with pytest.raises(Inkling2Error, match="in tokenizer.json but"):
        inkling2_model.resources(metadata, config, files)


def test_an_unknown_resource_is_refused(gguf_path):
    with GgufModel(gguf_path) as gguf, pytest.raises(ValueError, match="no consumer"):
        inkling2_model.build_model(gguf, resource_overrides={"vocab.txt": "x"})


def test_the_recipe_refuses_another_model(gguf_path):
    with GgufModel(gguf_path) as gguf:
        model = inkling2_model.build_model(gguf)
    model.config["model_type"] = "deepseek41_text"
    with pytest.raises(ValueError, match="requires the inkling2 GGUF model"):
        RECIPES["inkling2_gguf"](model, Recipe(model), {})


def test_the_public_cli_converts_an_inkling2_gguf(gguf_path, tmp_path):
    from tools.convert.__main__ import main

    files = fixture.tokenizer_files(tmp_path)
    resources = [arg for name, path in files.items() for arg in ("--resource", f"{name}={path}")]
    out = tmp_path / "cli.ninfer"
    main(["--model", str(gguf_path), "--recipe", "inkling2_gguf", "--device", "cpu", "--out", str(out), *resources])
    report = json.loads((tmp_path / "cli.ninfer.conversion.json").read_text())
    assert report["provenance"]["recipe"] == "inkling2_gguf" and report["formats"]["ggml_rec_nvfp4_nvfp4"] == 4
    with Artifact(out) as artifact:
        stored = {artifact.read_object(obj.id) for obj in artifact.directory.objects if obj.kind == "resource"}
    assert len(stored) == 4 and files["tokenizer.json"].read_bytes() in stored
    with pytest.raises(ValueError, match="tokenizer_config.json"):
        main(["--model", str(gguf_path), "--recipe", "inkling2_gguf", "--device", "cpu",
              "--out", str(tmp_path / "y.ninfer"), *resources[:2]])
    with pytest.raises(ValueError, match="takes none of"):
        main(["--model", str(gguf_path), "--recipe", "inkling2_gguf", "--device", "cpu", "--mmproj", str(gguf_path),
              "--out", str(tmp_path / "x.ninfer")])


def test_the_reference_reads_the_same_weights_from_the_artifact(gguf_path, tmp_path):
    """tools/inkling2/artifact_weights.py gives the NumPy reference every value the GGUF loader gives it (NVFP4
    records, BF16 banks, loop controllers, the shared tensors), so the reference computes the same on either. (The
    fixture's Q8_0 blocks are random bytes, some with NaN scales, so a forward pass on it is not finite; the values
    are compared instead, NaN where the GGUF has NaN.)"""
    from tools.inkling2 import artifact_weights, gguf_weights

    out = tmp_path / "tiny.ninfer"
    _convert(GgufModel(gguf_path), out)
    from_gguf, config = gguf_weights.load(gguf_path)
    from_artifact, artifact_config = artifact_weights.load(out)
    assert artifact_config == config
    for p, (a, b) in enumerate(zip(from_artifact["layers"], from_gguf["layers"])):
        assert set(a.names) == set(b.names), p
        for key in b.names:
            if key in ("gate_up", "down"):
                for e in range(fixture.EXPERTS):
                    np.testing.assert_array_equal(a[key][e], b[key][e], err_msg=f"{p} {key} {e}")
            elif key == "loop":
                for m, weights in b["loop"].items():
                    for x, y in zip(a["loop"][m], weights):
                        np.testing.assert_array_equal(x, y)
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{p} {key}")
    tokens = np.arange(fixture.VOCAB)
    np.testing.assert_array_equal(from_artifact["embed"][tokens], from_gguf["embed"][tokens])
    assert set(from_artifact) == set(from_gguf)
    for key in set(from_gguf) - {"embed", "layers"}:
        np.testing.assert_array_equal(from_artifact[key], from_gguf[key], err_msg=key)
