from __future__ import annotations

from pathlib import Path

import pytest

from tools.artifact.add_component import add_component
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def _pattern(count: int, seed: int) -> bytes:
    return bytes((index * 37 + seed) % 251 for index in range(count))


def _write(path: Path, objects: dict, components: dict, bindings: dict, uses: list) -> dict:
    specs = [TensorSpec(name, shape, "bf16", "contiguous_le_v1") for name, shape in objects.items()]
    data = {name: _pattern(2 * shape[0] * shape[1], len(name)) for name, shape in objects.items()}
    with ArtifactWriter(path, specs, components=components, bindings=bindings, uses=uses) as writer:
        for name, payload in data.items():
            writer.write_object(name, payload)
    return data


TEXT = {"text": {"config": {"architectures": ["Qwen3_5ForCausalLM"]}}}
DRAFT = {"dflash2": {"config": {"architectures": ["DFlash2DraftModel"]}, "target": "text"}}
HEAD_USE = {"parameter": "text/output_head", "input": "text/final_hidden", "activation_policy": "AllowA4"}


def _base(path: Path) -> dict:
    return _write(
        path,
        {"weight/0": (4, 8), "weight/1": (8, 4)},
        TEXT,
        {"text/embed": {"object": "weight/0"}, "text/output_head": {"object": "weight/1"}},
        [HEAD_USE],
    )


def _donor(path: Path, draft_head_use: bool = True) -> dict:
    uses = [{"parameter": "dflash2/proj", "input": "dflash2/x", "activation_policy": "A16Only"}]
    if draft_head_use:
        uses.append(
            {"parameter": "text/output_head", "input": "dflash2/final_hidden", "activation_policy": "A16Only"}
        )
    return _write(
        path,
        {"weight/0": (2, 8), "weight/1": (8, 4), "weight/2": (4, 4), "weight/3": (2, 4)},
        {**TEXT, **DRAFT},
        {
            "text/embed": {"object": "weight/0"},
            "text/output_head": {"object": "weight/1"},
            "dflash2/proj": {"object": "weight/2"},
            "dflash2/split/a": {"parts": [{"object": "weight/3", "range": [0, 4]}]},
            "dflash2/split/b": {"parts": [{"object": "weight/3", "range": [4, 8]}]},
        },
        uses,
    )


def test_adds_the_component_and_keeps_the_base(tmp_path: Path) -> None:
    base_data = _base(tmp_path / "base.ninfer")
    donor_data = _donor(tmp_path / "donor.ninfer")
    add_component(tmp_path / "base.ninfer", tmp_path / "donor.ninfer", tmp_path / "out.ninfer", "dflash2")
    with Artifact(tmp_path / "out.ninfer") as out, Artifact(tmp_path / "base.ninfer") as base:
        d = out.directory
        assert set(d.components) == {"text", "dflash2"}
        for name, binding in base.directory.bindings.items():
            assert d.bindings[name] == binding
            assert out.read_object(binding["object"]) == base_data[binding["object"]]
        assert d.bindings["dflash2/proj"] == {"object": "dflash2/weight/2"}
        assert out.read_object("dflash2/weight/2") == donor_data["weight/2"]
        assert d.bindings["dflash2/split/b"]["parts"][0]["object"] == "dflash2/weight/3"
        assert out.read_object("dflash2/weight/3") == donor_data["weight/3"]
        assert list(d.uses)[0] == HEAD_USE
        assert len(d.uses) == 3
        assert d.provenance["added_components"][0]["component"] == "dflash2"


def test_refuses_a_shared_parameter_of_another_shape(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.ninfer",
        {"weight/0": (4, 8), "weight/1": (8, 8)},
        TEXT,
        {"text/embed": {"object": "weight/0"}, "text/output_head": {"object": "weight/1"}},
        [],
    )
    _donor(tmp_path / "donor.ninfer")
    with pytest.raises(ArtifactError, match="shapes differ"):
        add_component(tmp_path / "base.ninfer", tmp_path / "donor.ninfer", tmp_path / "out.ninfer", "dflash2")
    assert not (tmp_path / "out.ninfer").exists()


def test_refuses_a_base_that_has_the_component(tmp_path: Path) -> None:
    _donor(tmp_path / "donor.ninfer")
    with pytest.raises(ArtifactError, match="already has"):
        add_component(tmp_path / "donor.ninfer", tmp_path / "donor.ninfer", tmp_path / "out.ninfer", "dflash2")


def test_copies_the_component_resources(tmp_path: Path) -> None:
    _base(tmp_path / "base.ninfer")
    specs = [TensorSpec("weight/0", (4, 4), "bf16", "contiguous_le_v1"), ResourceSpec("resource/0", 5)]
    components = {**TEXT, "dflash2": {**DRAFT["dflash2"], "resources": {"config.json": "resource/0"}}}
    with ArtifactWriter(tmp_path / "donor.ninfer", specs, components=components,
                        bindings={"dflash2/proj": {"object": "weight/0"}}) as writer:
        writer.write_object("weight/0", _pattern(32, 1))
        writer.write_object("resource/0", b"hello")
    add_component(tmp_path / "base.ninfer", tmp_path / "donor.ninfer", tmp_path / "out.ninfer", "dflash2")
    with Artifact(tmp_path / "out.ninfer") as out:
        name = out.directory.components["dflash2"]["resources"]["config.json"]
        assert name == "dflash2/resource/0"
        assert out.read_object(name) == b"hello"


def test_refuses_a_component_use_with_auxiliaries(tmp_path: Path) -> None:
    _base(tmp_path / "base.ninfer")
    _write(
        tmp_path / "donor.ninfer",
        {"weight/0": (4, 4), "weight/1": (1, 4)},
        DRAFT | TEXT,
        {"dflash2/proj": {"object": "weight/0"}},
        [{"parameter": "dflash2/proj", "input": "dflash2/x", "activation_policy": "AllowA4",
          "auxiliaries": {"activation_input_divisor": {"object": "weight/1"}}}],
    )
    with pytest.raises(ArtifactError, match="auxiliaries"):
        add_component(tmp_path / "base.ninfer", tmp_path / "donor.ninfer", tmp_path / "out.ninfer", "dflash2")
    assert not (tmp_path / "out.ninfer").exists()
