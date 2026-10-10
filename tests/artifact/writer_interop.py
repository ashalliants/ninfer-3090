"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.formats import GGML_BLOCK_FORMATS, GGML_RECORD_FORMATS
from tools.artifact.layouts import ggml_expert_record_geometry
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter


# The C++ side expects every GGML object's rows, and every expert record's parts in record
# order, to be consecutive runs of this sequence (gaps between record parts are zero).
def _pattern(count: int) -> bytes:
    return bytes((index * 37 + 11) % 251 for index in range(count))


def _records(name: str, shape: tuple[int, int, int]) -> bytes:
    g = ggml_expert_record_geometry(name, shape)
    payload, cursor = bytearray(g.payload_bytes), 0
    for expert in range(g.experts):
        for _, offset, rows, row_bytes in g.parts():
            size = rows * row_bytes
            start = expert * g.record_stride + offset
            payload[start : start + size] = _pattern(cursor + size)[cursor:]
            cursor += size
    return bytes(payload)


def _ggml_objects() -> tuple[list[TensorSpec], dict[str, bytes], dict[str, dict]]:
    """Three rows of two blocks per GGML format, and two experts of each record format."""
    specs, data, bindings = [], {}, {}
    for name, spec in sorted(GGML_BLOCK_FORMATS.items()):
        shape = (3, 2 * spec.block_elems)
        specs.append(TensorSpec(name, shape, name, "ggml_blocks_v1"))
        data[name] = _pattern(3 * 2 * spec.block_bytes)
        bindings[name] = {"object": name}
    # Rows 1..2 of one parent: a partial region of a GGML block parent.
    k = 2 * GGML_BLOCK_FORMATS["ggml_q6_k"].block_elems
    bindings["ggml_q6_k_rows"] = {"parts": [{"object": "ggml_q6_k", "range": [k, 3 * k]}]}
    # [2 experts, hidden 256, intermediate 64]: IQ2_S and IQ2_XXS gate parts leave a gap.
    for name in sorted(GGML_RECORD_FORMATS):
        shape = (2, 256, 64)
        specs.append(TensorSpec(name, shape, name, "ggml_expert_record_v1"))
        data[name] = _records(name, shape)
        bindings[name] = {"object": name}
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
