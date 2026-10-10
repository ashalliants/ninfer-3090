"""Check a converted qwen4exp artifact and its n-gram volume byte for byte against the GGUF."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.artifact import ngram_volume
from tools.artifact.codecs.ggml_blocks import unpack_expert_records
from tools.artifact.layouts import ggml_expert_record_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert import qwen4_exp
from tools.convert.sources.gguf import GgufModel

# Experts compared per read, bounding memory on the real banks (~48 MiB of records).
_EXPERTS_PER_READ = 32


def _hash(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _verify_bank(artifact: Artifact, obj: TensorObject, gguf: GgufModel, tensors) -> None:
    """Invert the records exactly: every expert's gate, up and down rows equal the GGUF's."""
    g = ggml_expert_record_geometry(obj.format, obj.shape)
    for first in range(0, g.experts, _EXPERTS_PER_READ):
        count = min(_EXPERTS_PER_READ, g.experts - first)
        payload = artifact.read_range(obj.offset + first * g.record_stride, count * g.record_stride)
        parts = unpack_expert_records(payload, obj.format, (count, g.hidden, g.intermediate))
        for tensor, stored, (role, _, rows, _) in zip(tensors, parts, g.parts()):
            source = gguf.read_rows(tensor, first * rows, (first + count) * rows)
            assert np.array_equal(stored.reshape(-1, stored.shape[-1]), source), (
                f"{obj.id}: {role} rows of experts [{first},{first + count}) differ"
            )


def verify_volume(path, gguf: GgufModel, table: dict) -> int:
    """The volume's header equals the artifact's ``ngram_table`` and its rows the GGUF's rows.

    Every block is unpacked (tails and the rows past the table must be zero). Returns the rows.
    """
    header = ngram_volume.read_header(path)
    g = header.geometry
    assert {**g.config(header.volume_id)} == table, "volume header differs from ngram_table"
    source = gguf.tensor(qwen4_exp.PLE_TABLE)
    step = 4096  # blocks per read
    with Path(path).open("rb") as stream:
        for block in range(0, g.blocks, step):
            count = min(step, g.blocks - block)
            stream.seek(ngram_volume.BLOCK_BYTES * (1 + block))
            rows = ngram_volume.unpack_blocks(
                g, stream.read(count * ngram_volume.BLOCK_BYTES), block
            )
            first = block * g.rows_per_block
            expected = gguf.read_rows(source, first, first + rows.shape[0])
            assert np.array_equal(rows, expected), f"volume rows from {first} differ"
    return g.rows


def verify(path, gguf: GgufModel, subset: qwen4_exp.Subset | None = None, volume=None) -> dict:
    """Every tensor object equals its GGUF ranges; returns counts of what was compared.

    Plain objects are compared by SHA-256 over the GGUF bytes of their bound parameters in
    binding order. Layout transforms are inverted exactly: expert records are unpacked to each
    expert's gate, up and down rows, and F16 widened to FP32 must narrow back to the original
    words. With *volume*, the n-gram volume is checked against the GGUF table too.
    """
    config = qwen4_exp.text_config(gguf.metadata)
    mappings = defaultdict(list)
    for item in qwen4_exp.name_map(gguf, config):
        mappings[item.parameter].append(item)
    summary = defaultdict(int)
    with Artifact(path) as artifact:
        directory = artifact.directory
        text = directory.components["text"]["config"]
        assert text["ngram_table"]["volume_id"] is not None, "no n-gram volume is bound"
        parts = defaultdict(list)
        for name, binding in directory.bindings.items():
            for object_id, begin, end in binding_parts(binding, artifact.by_id, name):
                parts[object_id].append((begin, end, name))
        for obj in directory.objects:
            if not isinstance(obj, TensorObject):
                continue
            bound = sorted(parts[obj.id])
            assert bound, f"{obj.id} has no binding"
            if obj.layout == "ggml_expert_record_v1":
                ((_, _, name),) = bound
                order = {"gate": 0, "up": 1, "down": 2}
                items = sorted(mappings[name], key=lambda item: order[item.part])
                _verify_bank(artifact, obj, gguf, [gguf.tensor(item.gguf) for item in items])
                summary["records"] += 1
            else:
                tensors = [gguf.tensor(mappings[name][0].gguf) for _, _, name in bound]
                if obj.format == "fp32" and tensors[0].type.name == "F16":
                    (tensor,) = tensors
                    source = gguf.read_range(tensor, 0, tensor.bytes)
                    widened = np.frombuffer(artifact.read_object(obj.id), dtype="<f4")
                    assert widened.tobytes() == np.frombuffer(source, "<f2").astype("<f4").tobytes()
                    assert widened.astype("<f2").tobytes() == source, obj.id
                    summary["widened"] += 1
                else:
                    source = _hash(
                        chunk for tensor in tensors for chunk in gguf.iter_range(tensor, 0, tensor.bytes)
                    )
                    assert _hash(artifact.iter_object(obj.id)) == source, obj.id
                    summary["identical"] += 1
            summary["objects"] += 1
            summary["bytes"] += obj.bytes
        resources = directory.components["text"]["resources"]
        synthesized = qwen4_exp.tokenizer_resources(gguf.metadata, config)
        for role, object_id in resources.items():
            data = artifact.read_object(object_id)
            if role.endswith(".json"):
                assert json.loads(data) == json.loads(synthesized[role]), role
            else:
                assert data == synthesized[role], role
    rows = gguf.tensor(qwen4_exp.PLE_TABLE).rows if subset is None else subset.ple_rows
    assert text["ngram_table"]["rows"] == rows
    if volume is not None:
        summary["volume_rows"] = verify_volume(volume, gguf, text["ngram_table"])
    return dict(summary)
