"""The DSpark drafter (component ``dspark``) from a tiny synthetic GGUF: its config, its exact repack beside the
text model, and the builder's refusals."""

from __future__ import annotations

import hashlib
import json

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import inkling2_dspark, inkling2_model
from tools.convert.inkling2 import Inkling2Error
from tools.convert.official_recipes import RECIPES
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufModel

from . import inkling2_fixture as fixture

Q8_0, BF16, F32 = fixture.Q8_0, fixture.BF16, fixture.F32


@pytest.fixture
def text_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def _sha(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def test_config_from_the_real_header():
    header = json.loads((fixture._PIPELINE.parent / "dspark_q8_0_header.json").read_text())
    text = {"hidden_size": 2304, "vocab_size": 201024, "unpadded_vocab_size": 200058, "num_layer_visits": 48,
            "final_logit_softcapping": 30.0}
    config = inkling2_dspark.dspark_config(header["metadata"], text)
    assert config["num_layers"] == 3 and config["local_layers"] == [0, 2] and config["block_size"] == 8
    assert config["target_visits"] == [5, 23, 41, 48] and config["markov_rank"] == 256
    assert config["intermediate_size"] == 9216 and config["sliding_window"] == 1024 and config["mask_token_id"] == 0
    shapes = inkling2_dspark.expected_shapes({**config, "vocab_size": 201024})
    real = {t["name"]: tuple(reversed(t["ne"])) for t in header["tensors"]}
    assert shapes == real


def test_the_drafter_is_stored_exactly_beside_the_text_model(text_path, tmp_path):
    dspark_path = fixture.write_dspark(tmp_path / "dspark.gguf")
    out = tmp_path / "with-dspark.ninfer"
    with GgufModel(text_path) as text, GgufModel(dspark_path) as dspark:
        model = inkling2_model.build_model(text, dspark=dspark)
        recipe = Recipe(model)
        RECIPES["inkling2_gguf"](model, recipe, {"base": text})
        convert(model, recipe, out, device="cpu")
        names = {inkling2_dspark.parameter_name(name): name for name in dspark.tensors}
        assert len(names) == len(dspark.tensors)
        assert "dspark/final_hidden" in model.parameters["text/head"].inputs
        with Artifact(out) as artifact:
            assert artifact.directory.components["dspark"]["config"]["target_visits"] == fixture.D_TAPS
            ids = {}
            for name, binding in artifact.directory.bindings.items():
                for object_id, _, _ in binding_parts(binding, artifact.by_id, name):
                    ids[object_id] = name
            seen = set()
            for obj in artifact.directory.objects:
                if not isinstance(obj, TensorObject) or not ids[obj.id].startswith("dspark/"):
                    continue
                tensor = dspark.tensor(names[ids[obj.id]])
                assert obj.shape == tensor.shape, ids[obj.id]
                assert _sha(artifact.iter_object(obj.id)) == _sha(dspark.iter_range(tensor, 0, tensor.bytes))
                expected = {"Q8_0": "ggml_q8_0", "BF16": "bf16", "F32": "fp32"}[tensor.type.name]
                assert obj.format == expected, ids[obj.id]
                seen.add(ids[obj.id])
            assert seen == set(names)


@pytest.mark.parametrize("changes, match", [
    ({"general.architecture": "llama"}, "general.architecture"),
    ({"dflash.embedding_length": 128}, "embedding_length"),
    ({"dflash.vocab_size_unpadded": 91}, "vocab_size_unpadded"),
    ({"dflash.attention.causal": True}, "non-causal"),
    ({"dflash.has_confidence_head": True}, "confidence head"),
    ({"dflash.target_layers": (fixture.I32, [1, 4, 7, 9])}, "target_layers"),
    ({"dflash.attention.sliding_window_pattern": (fixture.BOOL, [True])}, "sliding_window_pattern"),
    ({"dflash.dspark.markov_rank": None}, "markov_rank"),
    ({"tokenizer.ggml.mask_token_id": 95}, "mask_token_id"),
])
def test_metadata_refusals(text_path, tmp_path, changes, match):
    dspark_path = fixture.write_dspark(tmp_path / "bad.gguf", metadata_changes=changes)
    with GgufModel(text_path) as text, GgufModel(dspark_path) as dspark, pytest.raises(Inkling2Error, match=match):
        inkling2_model.build_model(text, dspark=dspark)


@pytest.mark.parametrize("changes, match", [
    ({"blk.0.attn_q.weight": (Q8_0, (fixture.HIDDEN, 64))}, "shape"),
    ({"blk.1.ffn_up.weight": (F32, (fixture.HIDDEN, fixture.D_FF))}, "expected Q8_0 or BF16"),
    ({"mask_embd.weight": (BF16, (fixture.HIDDEN,))}, "expected F32"),
])
def test_tensor_refusals(text_path, tmp_path, changes, match):
    dspark_path = fixture.write_dspark(tmp_path / "bad.gguf", tensor_changes=changes)
    with GgufModel(text_path) as text, GgufModel(dspark_path) as dspark, pytest.raises(Inkling2Error, match=match):
        inkling2_model.build_model(text, dspark=dspark)


def test_the_cli_adds_the_drafter(text_path, tmp_path):
    from tools.convert.__main__ import main

    files = fixture.tokenizer_files(tmp_path)
    resources = [arg for name, path in files.items() for arg in ("--resource", f"{name}={path}")]
    out = tmp_path / "cli.ninfer"
    main(["--model", str(text_path), "--recipe", "inkling2_gguf", "--device", "cpu", "--out", str(out),
          "--mtp-gguf", str(fixture.write_dspark(tmp_path / "dspark.gguf")), *resources])
    with Artifact(out) as artifact:
        assert "dspark" in artifact.directory.components
        assert any(name.startswith("dspark/layers/1/") for name in artifact.directory.bindings)
