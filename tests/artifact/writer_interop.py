"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.formats import GGML_BLOCK_FORMATS
from tools.artifact.layouts import ROW_PAGE_BYTES, ggml_row_page_geometry
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter

# The C++ side expects every GGML object's unpaged row bytes to be this sequence.
def _pattern(count: int) -> bytes:
    return bytes((index * 37 + 11) % 251 for index in range(count))


def _ggml_objects() -> tuple[list[TensorSpec], dict[str, bytes], dict[str, dict]]:
    """Three rows of two blocks per GGML format, plus one paged table with a partial last page."""
    specs, data, bindings = [], {}, {}
    for name, spec in sorted(GGML_BLOCK_FORMATS.items()):
        shape = (3, 2 * spec.block_elems)
        specs.append(TensorSpec(name, shape, name, "ggml_blocks_v1"))
        data[name] = _pattern(3 * 2 * spec.block_bytes)
        bindings[name] = {"object": name}
    # Rows 1..2 of one parent: a partial region of a GGML block parent.
    k = 2 * GGML_BLOCK_FORMATS["ggml_q6_k"].block_elems
    bindings["ggml_q6_k_rows"] = {"parts": [{"object": "ggml_q6_k", "range": [k, 3 * k]}]}
    shape = (50, 160)  # 45 rows in page 0, five in page 1
    geometry = ggml_row_page_geometry("ggml_iq4_nl", shape)
    rows = _pattern(shape[0] * geometry.row_bytes)
    paged = bytearray(geometry.payload_bytes)
    for row in range(shape[0]):
        paged[geometry.row_offset(row) : geometry.row_offset(row) + geometry.row_bytes] = rows[
            row * geometry.row_bytes : (row + 1) * geometry.row_bytes
        ]
    assert len(paged) == 2 * ROW_PAGE_BYTES
    specs.append(TensorSpec("ple_paged", shape, "ggml_iq4_nl", "ggml_rows_page4k_v1"))
    data["ple_paged"] = bytes(paged)
    bindings["ple_paged"] = {"object": "ple_paged"}
    return specs, data, bindings


def main() -> int:
    executable = sys.argv[1]
    matrix = _pattern(130 * 130 * 2)
    ggml_specs, ggml_data, ggml_bindings = _ggml_objects()
    with tempfile.TemporaryDirectory(prefix="ninfer-writer-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1"), *ggml_specs],
                components={"text": {"config": {}}},
                bindings={
                    "whole": {"object": "matrix"},
                    "reordered_rows": {
                        "parts": [
                            {"object": "matrix", "range": [16770, 16900]},
                            {"object": "matrix", "range": [130, 260]},
                        ]
                    },
                    **ggml_bindings,
                },
                max_file_bytes=limit,
            ) as writer:
                writer.write_object("matrix", matrix)
                for name, payload in ggml_data.items():
                    writer.write_object(name, payload)
            result = subprocess.run([executable, "--writer-fixture", str(path)])
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
