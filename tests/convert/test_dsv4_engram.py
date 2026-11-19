"""The Engram table as an n-gram volume: its geometry in the text config, a volume whose rows are the GGUF's
rows byte for byte, the id tying the artifact to it, reuse of an existing volume and the refusals."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.artifact import ngram_volume
from tools.artifact.reader import Artifact
from tools.convert import dsv4, dsv4_engram, dsv4_model
from tools.convert.__main__ import main
from tools.convert.sources.gguf import GgufModel

from . import dsv4_fixture as fixture
from .test_dsv4_model import gguf_path, tokenizer_file  # noqa: F401  (the fixture and helper of the model tests)


def _cli(gguf, out, tmp_path, *extra):
    main(["--model", str(gguf), "--recipe", "dsv4_gguf", "--device", "cpu", "--out", str(out),
          "--resource", f"tokenizer.json={tokenizer_file(tmp_path)}", *extra])


def test_the_config_records_the_volume_geometry_unbound(gguf_path):  # noqa: F811
    with GgufModel(gguf_path) as gguf:
        model = dsv4_model.build_model(gguf)
    record = model.config[dsv4_engram.CONFIG_KEY]
    assert record["rows"] == fixture.ENGRAM_ROWS and record["volume_id"] is None
    assert record["format"] == "ggml_iq4_nl" and record["row_bytes"] == fixture.ENGRAM_ROW // 32 * 18


def test_the_cli_writes_a_volume_of_the_gguf_rows_and_binds_its_id(gguf_path, tmp_path):  # noqa: F811
    out, volume = tmp_path / "m.ninfer", tmp_path / "m.engram"
    _cli(gguf_path, out, tmp_path, "--ngram-out", str(volume))
    header = ngram_volume.read_header(volume)
    assert header.geometry.rows == fixture.ENGRAM_ROWS and header.geometry.format == "ggml_iq4_nl"
    report = json.loads((tmp_path / "m.ninfer.conversion.json").read_text())
    assert report["provenance"]["recipe"] == "dsv4_gguf"
    with GgufModel(gguf_path) as gguf:
        table = gguf.tensor(dsv4.ENGRAM_TABLE)
        expected = gguf.read_rows(table, 0, table.rows)
        stored = ngram_volume.read_rows(volume, header.geometry, range(table.rows))
        assert b"".join(stored[r] for r in range(table.rows)) == expected.tobytes()
    assert header.volume_id.hex() == _bound_id(out)


def _bound_id(out) -> str | None:
    with Artifact(out) as artifact:
        return artifact.directory.components["text"]["config"]["engram_table"]["volume_id"]


def test_reuse_binds_the_existing_volume_and_a_different_table_is_refused(gguf_path, tmp_path):  # noqa: F811
    volume = tmp_path / "v.engram"
    _cli(gguf_path, tmp_path / "a.ninfer", tmp_path, "--ngram-out", str(volume))
    header = ngram_volume.read_header(volume)
    _cli(gguf_path, tmp_path / "b.ninfer", tmp_path, "--ngram-reuse", str(volume))
    assert _bound_id(tmp_path / "b.ninfer") == header.volume_id.hex()
    # A volume written from another table (one byte of a sampled row changed) is not adopted.
    with volume.open("r+b") as stream:
        stream.seek(header.geometry.row_offset(0))
        first = stream.read(1)
        stream.seek(header.geometry.row_offset(0))
        stream.write(bytes([first[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="differs from this table"):
        _cli(gguf_path, tmp_path / "c.ninfer", tmp_path, "--ngram-reuse", str(volume))
    assert not (tmp_path / "c.ninfer").exists()


def test_an_existing_volume_is_never_overwritten(gguf_path, tmp_path):  # noqa: F811
    volume = tmp_path / "taken.engram"
    volume.write_bytes(b"precious")
    with pytest.raises(FileExistsError, match="already exists"):
        _cli(gguf_path, tmp_path / "d.ninfer", tmp_path, "--ngram-out", str(volume))
    assert volume.read_bytes() == b"precious" and not (tmp_path / "d.ninfer").exists()


def test_a_volume_path_that_is_one_of_the_conversions_outputs_is_refused_before_anything_is_written(gguf_path, tmp_path):  # noqa: F811
    out = tmp_path / "g.ninfer"
    for name in ("g.ninfer", "g.ninfer.conversion.json", "g.ninfer.part-0001"):
        with pytest.raises(ValueError, match="one of the conversion's own output files"):
            _cli(gguf_path, out, tmp_path, "--ngram-out", str(tmp_path / name))
    assert not list(tmp_path.glob("g.ninfer*"))
    # The same files by another spelling: a `..` detour and a symlinked directory.
    (tmp_path / "sub").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    for spelling in (tmp_path / "sub" / ".." / "g.ninfer", tmp_path / "alias" / "g.ninfer.conversion.json"):
        with pytest.raises(ValueError, match="one of the conversion's own output files"):
            _cli(gguf_path, out, tmp_path, "--ngram-out", str(spelling))
    assert not list(tmp_path.glob("g.ninfer*"))
    # Another file beside it is fine.
    _cli(gguf_path, out, tmp_path, "--ngram-out", str(tmp_path / "g.engram"))
    assert out.exists() and (tmp_path / "g.engram").exists()


def test_the_two_volume_options_are_exclusive(gguf_path, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="exclusive"):
        _cli(gguf_path, tmp_path / "e.ninfer", tmp_path, "--ngram-out", str(tmp_path / "x"), "--ngram-reuse", str(tmp_path / "y"))


def test_without_a_volume_option_none_is_written_and_the_id_stays_unbound(gguf_path, tmp_path):  # noqa: F811
    _cli(gguf_path, tmp_path / "f.ninfer", tmp_path)
    assert not list(tmp_path.glob("f.ninfer.*ngram*")) and not list(tmp_path.glob("*.engram"))
    assert _bound_id(tmp_path / "f.ninfer") is None


def test_a_table_that_is_not_ggml_rows_is_refused():
    class Table:
        class type:  # noqa: N801
            dtype, name = np.float32, "F32"

    class Fake:
        tensors = {dsv4.ENGRAM_TABLE: None}

        @staticmethod
        def tensor(name):
            return Table

    with pytest.raises(dsv4.Dsv4Error, match="expected GGML block rows"):
        dsv4_engram.geometry(Fake)
