"""The sidecar writer: volume + token map + manifest beside an artifact, its validation and what it refuses."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.artifact import ngram_volume
from tools.convert import dsv4_engram_volume as sidecar
from tools.convert.sources.gguf import GgufModel

from . import dsv4_fixture as fixture
from .test_dsv4_model import gguf_path  # noqa: F401  (the fixture of the model tests)


def _written(gguf_path, directory):  # noqa: F811
    with GgufModel(gguf_path) as gguf:
        return sidecar.write(gguf, directory, log=lambda _: None)


def test_the_folder_holds_the_volume_the_token_map_and_the_hash_parameters(gguf_path, tmp_path):  # noqa: F811
    manifest = _written(gguf_path, tmp_path / "engram")
    folder = tmp_path / "engram"
    assert sorted(p.name for p in folder.iterdir()) == sorted([sidecar.VOLUME_FILE, sidecar.TOKEN_MAP_FILE, sidecar.MANIFEST_FILE])
    header = ngram_volume.read_header(folder / sidecar.VOLUME_FILE)
    assert manifest["volume"]["table"]["volume_id"] == header.volume_id.hex()
    assert manifest["volume"]["table"]["rows"] == fixture.ENGRAM_ROWS
    h = manifest["hash"]
    assert h["layers"] == fixture.ENGRAM_LAYERS and h["ngram_size"] == fixture.NGRAM and h["heads"] == fixture.ENGRAM_HEADS
    assert h["multipliers"] == fixture.MULTIPLIERS and h["head_vocab_sizes"] == fixture.HEAD_VOCAB and h["pad_token_id"] == 2
    assert np.fromfile(folder / sidecar.TOKEN_MAP_FILE, "<i4").tolist() == fixture.TOKEN_MAP
    assert json.loads((folder / sidecar.MANIFEST_FILE).read_text()) == manifest


def test_validation_passes_on_a_good_folder_and_checks_every_row_of_a_small_table(gguf_path, tmp_path):  # noqa: F811
    _written(gguf_path, tmp_path / "e")
    with GgufModel(gguf_path) as gguf:
        result = sidecar.validate(gguf, tmp_path / "e", samples=1000)
    assert result["rows_checked"] <= fixture.ENGRAM_ROWS and result["rows_checked"] > 100


@pytest.mark.parametrize("damage", ["row", "map", "manifest", "id"])
def test_validation_refuses_a_damaged_folder(gguf_path, tmp_path, damage):  # noqa: F811
    folder = tmp_path / "e"
    _written(gguf_path, folder)
    if damage == "row":
        with (folder / sidecar.VOLUME_FILE).open("r+b") as f:
            f.seek(4096 + 5)
            f.write(b"\xff")
    elif damage == "map":
        data = bytearray((folder / sidecar.TOKEN_MAP_FILE).read_bytes())
        data[4] ^= 1
        (folder / sidecar.TOKEN_MAP_FILE).write_bytes(data)
    elif damage == "manifest":
        manifest = json.loads((folder / sidecar.MANIFEST_FILE).read_text())
        manifest["hash"]["multipliers"][0] += 2
        (folder / sidecar.MANIFEST_FILE).write_text(json.dumps(manifest))
    else:
        with (folder / sidecar.VOLUME_FILE).open("r+b") as f:
            f.seek(44)
            f.write(bytes(16))
    with GgufModel(gguf_path) as gguf, pytest.raises(sidecar.EngramVolumeError):
        sidecar.validate(gguf, folder, samples=1000)


def test_a_folder_with_files_is_not_overwritten(gguf_path, tmp_path):  # noqa: F811
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "keep").write_text("x")
    with GgufModel(gguf_path) as gguf, pytest.raises(sidecar.EngramVolumeError):
        sidecar.write(gguf, tmp_path / "e", log=lambda _: None)
    assert (tmp_path / "e" / "keep").read_text() == "x"


def test_the_digest_is_the_canonical_utf8_form_the_runtime_recomputes():
    import hashlib

    manifest = {"b": 1, "a": {"file": "mod\u00e8le-\u4e2d.gguf"}, "digest": "x"}
    expected = hashlib.sha256('{"a":{"file":"mod\u00e8le-\u4e2d.gguf"},"b":1}'.encode("utf-8")).hexdigest()
    assert sidecar.manifest_digest(manifest) == expected


def test_a_non_first_shard_is_refused(tmp_path):
    from tools.convert.sources.gguf import GgufError

    with pytest.raises(GgufError):
        GgufModel(tmp_path / "m-00002-of-00006.gguf")
