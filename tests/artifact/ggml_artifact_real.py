"""A real GGML artifact through the C++ loader, decoded exactly against the Python decoders.

Usage: ggml_artifact_real.py MATERIALIZATION_TEST_EXECUTABLE, with NINFER_TEST_ARTIFACT naming a
.ninfer that holds GGML block tensors (such as the qwen4exp ``--subset dev`` conversion).

The C++ test materializes every GGML object (rotating Device, Pinned and Host residency), checks
each one whole against the file, and decodes sampled blocks with tests/ops/ggml_blocks_decode.h.
This script then reads the same blocks independently through the Python reader and layout
geometry, checks the bytes the C++ side addressed, and requires the Python decoder's FP32 bit
patterns to equal the C++ ones. Exits 77 when the variable is unset or the artifact has no GGML
tensor.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SKIP = 77


def _records(data: bytes):
    position = 0

    def take(count: int) -> bytes:
        nonlocal position
        if position + count > len(data):
            raise ValueError("truncated sample file")
        chunk = data[position : position + count]
        position += count
        return chunk

    while position < len(data):
        (id_bytes,) = struct.unpack("<I", take(4))
        object_id = take(id_bytes).decode()
        (residency,) = struct.unpack("<I", take(4))
        (offset,) = struct.unpack("<Q", take(8))
        (name_bytes,) = struct.unpack("<I", take(4))
        format = take(name_bytes).decode()
        (block_bytes,) = struct.unpack("<I", take(4))
        encoded = take(block_bytes)
        (count,) = struct.unpack("<I", take(4))
        values = take(4 * count)
        yield object_id, residency, offset, format, encoded, values


def main() -> int:
    path = os.environ.get("NINFER_TEST_ARTIFACT")
    if not path:
        print("NINFER_TEST_ARTIFACT is not set; skipped")
        return SKIP
    try:
        import numpy as np
    except ImportError:
        print("numpy is not available to the test interpreter; skipped")
        return SKIP
    from tools.artifact.codecs.ggml_blocks import decode_blocks
    from tools.artifact.formats import GgmlExpertRecordFormat, get_format
    from tools.artifact.layouts import ggml_blocks_geometry, ggml_expert_record_geometry
    from tools.artifact.reader import Artifact

    def block_format(item, offset: int) -> str:
        """The GGML block format the Python geometry places at payload *offset* of *item*."""
        spec = get_format(item.format)
        if not isinstance(spec, GgmlExpertRecordFormat):
            geometry = ggml_blocks_geometry(spec, item.shape)
            if offset % spec.block_bytes or offset >= geometry.payload_bytes:
                raise ValueError(f"{item.id}: offset {offset} is not a block")
            return spec.name
        g = ggml_expert_record_geometry(spec, item.shape)
        within = offset % g.record_stride
        for role, start, rows, row_bytes in g.parts():
            part = spec.down if role == "down" else spec.gate_up
            if start <= within < start + rows * row_bytes:
                if (within - start) % part.block_bytes:
                    break
                return part.name
        raise ValueError(f"{item.id}: offset {offset} is not a block of a record part")

    with tempfile.TemporaryDirectory(prefix="ninfer-ggml-real-") as temporary:
        samples = Path(temporary) / "samples.bin"
        result = subprocess.run([sys.argv[1], "--ggml-samples", path, str(samples)])
        if result.returncode:
            return result.returncode
        data = samples.read_bytes()

    checked: dict[str, int] = {}
    residencies: set[int] = set()
    with Artifact.open(path) as artifact:
        for object_id, residency, offset, format, encoded, values in _records(data):
            item = artifact.object(object_id)
            if block_format(item, offset) != format:
                print(f"{object_id} at {offset}: C++ decoded {format}, the geometry says otherwise")
                return 1
            spec = get_format(format)
            stored = artifact.read_range(item.offset + offset, spec.block_bytes)
            if stored != encoded:
                print(f"{object_id} at {offset}: C++ addressed different bytes than the file")
                return 1
            expected = decode_blocks(spec, np.frombuffer(stored, np.uint8)).view(np.uint32)
            actual = np.frombuffer(values, "<u4")
            if not np.array_equal(expected, actual):
                first = int(np.argmax(expected != actual))
                print(
                    f"{object_id} at {offset} value {first}: C++ 0x{actual[first]:08x}, "
                    f"Python 0x{expected[first]:08x}"
                )
                return 1
            checked[format] = checked.get(format, 0) + 1
            residencies.add(residency)
    for name, count in sorted(checked.items()):
        print(f"{name}: {count} sampled blocks bit-exact")
    print(f"{sum(checked.values())} blocks over {len(residencies)} residencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
