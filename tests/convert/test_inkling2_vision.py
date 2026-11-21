"""The vision tower (component ``vision``) from a tiny synthetic mmproj GGUF: its config, its exact repack beside
the text model, and the builder's refusals."""

from __future__ import annotations

import hashlib
import json

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import inkling2_model, inkling2_vision
from tools.convert.inkling2 import Inkling2Error
from tools.convert.official_recipes import RECIPES
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GgufModel

from . import inkling2_fixture as fixture

BF16, F32 = fixture.BF16, fixture.F32


@pytest.fixture
def text_path(tmp_path):
    return fixture.write(tmp_path / "tiny.gguf")


def _sha(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def test_config_from_the_real_header():
    header = json.loads((fixture._PIPELINE.parent / "mmproj_bf16_header.json").read_text())
    config = inkling2_vision.vision_config(header["metadata"], {"hidden_size": 2304})
    assert config["depth"] == 23 and config["hidden_size"] == 1024 and config["num_heads"] == 16
    assert config["intermediate_size"] == 4096 and config["patch_size"] == 20 and config["position_grid"] == 32
    assert config["spatial_merge_size"] == 2 and config["projection_size"] == 2304
    assert config["rope_theta"] == 10000.0 and config["max_image_tokens"] == 4096
    assert config["image_mean"] == [0.5, 0.5, 0.5] and config["image_std"] == [0.5, 0.5, 0.5]
    shapes = inkling2_vision.expected_shapes(config)
    real = {t["name"]: tuple(reversed(t["ne"])) for t in header["tensors"]}
    assert shapes == real
    assert {t["name"]: t["type"] for t in header["tensors"]}["v.patch_embd.weight"] == F32


def test_the_tower_is_stored_exactly_beside_the_text_model(text_path, tmp_path):
    vision_path = fixture.write_vision(tmp_path / "mmproj.gguf")
    out = tmp_path / "with-vision.ninfer"
    with GgufModel(text_path) as text, GgufModel(vision_path) as vision:
        model = inkling2_model.build_model(text, vision=vision)
        recipe = Recipe(model)
        RECIPES["inkling2_gguf"](model, recipe, {"base": text})
        convert(model, recipe, out, device="cpu")
        names = {inkling2_vision.parameter_name(name): name for name in vision.tensors}
        assert len(names) == len(vision.tensors)
        assert model.parameters["vision/layers/1/mlp/fc2"].inputs == ("vision/layers/1/mlp_activation",)
        with Artifact(out) as artifact:
            component = artifact.directory.components["vision"]
            assert component["config"]["position_grid"] == fixture.V_GRID and component["target"] == "text"
            ids = {}
            for name, binding in artifact.directory.bindings.items():
                for object_id, _, _ in binding_parts(binding, artifact.by_id, name):
                    ids[object_id] = name
            seen = set()
            for obj in artifact.directory.objects:
                if not isinstance(obj, TensorObject) or not ids[obj.id].startswith("vision/"):
                    continue
                tensor = vision.tensor(names[ids[obj.id]])
                assert obj.shape == tensor.shape, ids[obj.id]
                assert _sha(artifact.iter_object(obj.id)) == _sha(vision.iter_range(tensor, 0, tensor.bytes))
                assert obj.format == {"BF16": "bf16", "F32": "fp32"}[tensor.type.name], ids[obj.id]
                seen.add(ids[obj.id])
            assert seen == set(names)


@pytest.mark.parametrize("changes, match", [
    ({"general.architecture": "llama"}, "general.architecture"),
    ({"clip.vision.projector_type": "qwen3vl_merger"}, "projector_type"),
    ({"clip.vision.projection_dim": 128}, "projection_dim"),
    ({"clip.vision.attention.head_count": 3}, "heads"),
    ({"clip.vision.attention.head_count": 4}, "heads of 64"),
    ({"clip.vision.spatial_merge_size": 3}, "spatial_merge_size"),
    ({"clip.vision.image_std": (fixture.F32V, [0.5, 0.5])}, "image_std"),
    ({"clip.vision.attention.layer_norm_epsilon": None}, "epsilon"),
    ({"clip.vision.block_count": None}, "block_count"),
])
def test_metadata_refusals(text_path, tmp_path, changes, match):
    vision_path = fixture.write_vision(tmp_path / "bad.gguf", metadata_changes=changes)
    with GgufModel(text_path) as text, GgufModel(vision_path) as vision, pytest.raises(Inkling2Error, match=match):
        inkling2_model.build_model(text, vision=vision)


@pytest.mark.parametrize("changes, match", [
    ({"v.blk.0.attn_q.weight": (BF16, (fixture.V_HIDDEN, 16))}, "shape"),
    ({"v.blk.1.ffn_up.weight": (F32, (fixture.V_HIDDEN, fixture.V_FF))}, "expected BF16"),
    ({"v.patch_embd.weight": (BF16, (fixture.V_PATCH, fixture.V_PATCH, 3, fixture.V_HIDDEN))}, "expected F32"),
])
def test_tensor_refusals(text_path, tmp_path, changes, match):
    vision_path = fixture.write_vision(tmp_path / "bad.gguf", tensor_changes=changes)
    with GgufModel(text_path) as text, GgufModel(vision_path) as vision, pytest.raises(Inkling2Error, match=match):
        inkling2_model.build_model(text, vision=vision)


def test_the_cli_adds_the_tower(text_path, tmp_path):
    from tools.convert.__main__ import main

    files = fixture.tokenizer_files(tmp_path)
    resources = [arg for name, path in files.items() for arg in ("--resource", f"{name}={path}")]
    out = tmp_path / "cli.ninfer"
    main(["--model", str(text_path), "--recipe", "inkling2_gguf", "--device", "cpu", "--out", str(out),
          "--mmproj", str(fixture.write_vision(tmp_path / "mmproj.gguf")), *resources])
    with Artifact(out) as artifact:
        assert "vision" in artifact.directory.components
        assert any(name.startswith("vision/layers/1/") for name in artifact.directory.bindings)
