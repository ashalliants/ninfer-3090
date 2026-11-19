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


def _bf16_words(gguf: GgufModel, tensor) -> bytes:
    """The BF16 words a tensor read as BF16 is stored as: its exact FP32 values, an F32 source
    required to be BF16 words already, any other rounded once to nearest even (integer arithmetic,
    not the converter's torch cast)."""
    wide = gguf.read_values(tensor, 0, tensor.elements).to("cpu").numpy().astype(np.float32, copy=False)
    bits = wide.view(np.uint32).astype(np.uint64)
    if tensor.type.name == "F32":
        assert not (bits & 0xFFFF).any(), f"{tensor.name}: F32 values that are not BF16 words"
        words = bits >> 16
    else:
        assert np.isfinite(wide).all(), tensor.name
        words = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return words.astype("<u2").tobytes()


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


def _mtp_sources() -> dict[str, tuple[str, object]]:
    """Each MTP parameter's GGUF tensor and how it is stored (None: the tensor's bytes)."""
    out = {name: (gguf_name, rule) for gguf_name, (name, rule) in qwen4_exp._MTP_NORMS.items()}
    out.update({name: (gguf_name, None) for gguf_name, (name, _) in qwen4_exp._MTP_MATRICES.items()})
    out["mtp/layers/0/indexer/query"] = (qwen4_exp._MTP_INDEX, "query rows")
    out["mtp/layers/0/indexer/key"] = (qwen4_exp._MTP_INDEX, "key rows")
    out[qwen4_exp.MTP_EXPERTS] = (qwen4_exp._MTP_GATE_UP, "records")
    return out


def _verify_mtp_object(artifact: Artifact, obj: TensorObject, bound, mtp: GgufModel, config: dict) -> str:
    """One MTP object against the drafter GGUF; returns the summary key it counts under."""
    sources = _mtp_sources()
    if obj.layout == "ggml_expert_record_v1":
        ((_, _, name),) = bound
        assert name == qwen4_exp.MTP_EXPERTS, name
        g = ggml_expert_record_geometry(obj.format, obj.shape)
        gate_up, down = mtp.tensor(qwen4_exp._MTP_GATE_UP), mtp.tensor(qwen4_exp._MTP_DOWN)
        for first in range(0, g.experts, _EXPERTS_PER_READ):
            count = min(_EXPERTS_PER_READ, g.experts - first)
            payload = artifact.read_range(obj.offset + first * g.record_stride, count * g.record_stride)
            gate, up, dn = unpack_expert_records(payload, obj.format, (count, g.hidden, g.intermediate))
            fused = mtp.read_rows(gate_up, first * 2 * g.intermediate, (first + count) * 2 * g.intermediate)
            fused = fused.reshape(count, 2 * g.intermediate, -1)
            assert np.array_equal(gate, fused[:, : g.intermediate]), f"{obj.id}: gate rows differ"
            assert np.array_equal(up, fused[:, g.intermediate :]), f"{obj.id}: up rows differ"
            source = mtp.read_rows(down, first * g.hidden, (first + count) * g.hidden)
            assert np.array_equal(dn.reshape(-1, dn.shape[-1]), source), f"{obj.id}: down rows differ"
        return "mtp_records"
    chunks = []
    for _, _, name in bound:
        gguf_name, rule = sources[name]
        tensor = mtp.tensor(gguf_name)
        raw = mtp.read_range(tensor, 0, tensor.bytes)
        if rule in ("query rows", "key rows"):
            iq = config["indexer_n_heads"] * config["indexer_head_dim"]
            rows = (0, iq) if rule == "query rows" else (iq, tensor.rows)
            raw = raw[rows[0] * tensor.row_bytes : rows[1] * tensor.row_bytes]
        elif rule is not None:
            gamma = np.frombuffer(raw, dtype="<f4")
            if rule == "words":
                assert obj.format == "bf16", obj.id
                raw = (gamma.view(np.uint32) >> 16).astype("<u2").tobytes()
                assert ((gamma.view(np.uint32) & 0xFFFF) == 0).all(), obj.id
            else:
                assert obj.format == "fp32", obj.id
                raw = (np.float32(1.0) + gamma).astype("<f4").tobytes()
        chunks.append(raw)
    assert artifact.read_object(obj.id) == b"".join(chunks), obj.id
    return "mtp_identical"


def _vision_source(name: str, depth: int) -> tuple[str, str, int]:
    """The mmproj tensor of one vision parameter, how its bytes are formed, and the third of a
    fused QKV tensor it takes (-1: all). Rules: ``bf16`` the tensor's words; ``words`` the high
    halves of F32 BF16 words; ``patch`` the temporal slices interleaved per output channel."""
    fixed = {
        "vision/patch_embedding": ("v.patch_embd.weight", "patch", -1),
        "vision/patch_embedding_bias": ("v.patch_embd.bias", "words", -1),
        "vision/position_embedding": ("v.position_embd.weight", "words", -1),
        "vision/merger/norm_weight": ("v.post_ln.weight", "words", -1),
        "vision/merger/norm_bias": ("v.post_ln.bias", "words", -1),
        "vision/merger/fc1": ("mm.0.weight", "bf16", -1),
        "vision/merger/fc1_bias": ("mm.0.bias", "words", -1),
        "vision/merger/fc2": ("mm.2.weight", "bf16", -1),
        "vision/merger/fc2_bias": ("mm.2.bias", "words", -1),
    }
    if name in fixed:
        return fixed[name]
    layer, role = name.removeprefix("vision/layers/").split("/", 1)
    assert int(layer) < depth, name
    g = f"v.blk.{layer}."
    table = {
        "norm1_weight": ("ln1.weight", "words"), "norm1_bias": ("ln1.bias", "words"),
        "norm2_weight": ("ln2.weight", "words"), "norm2_bias": ("ln2.bias", "words"),
        "attention/output": ("attn_out.weight", "bf16"), "attention/output_bias": ("attn_out.bias", "words"),
        "mlp/fc1": ("ffn_up.weight", "bf16"), "mlp/fc1_bias": ("ffn_up.bias", "words"),
        "mlp/fc2": ("ffn_down.weight", "bf16"), "mlp/fc2_bias": ("ffn_down.bias", "words"),
    }
    for third, part in enumerate(("query", "key", "value")):
        table[f"attention/{part}"] = ("attn_qkv.weight", "bf16", third)
        table[f"attention/{part}_bias"] = ("attn_qkv.bias", "words", third)
    tensor, rule, *third = table[role]
    return g + tensor, rule, third[0] if third else -1


def _vision_bytes(mmproj: GgufModel, name: str, depth: int) -> bytes:
    gguf_name, rule, third = _vision_source(name, depth)
    tensor = mmproj.tensor(gguf_name)
    raw = mmproj.read_range(tensor, 0, tensor.bytes)
    if rule == "patch":
        slices = [np.frombuffer(raw, "<f4")]
        extra = gguf_name + ".1"
        if extra in mmproj.tensors:
            more = mmproj.tensor(extra)
            slices.append(np.frombuffer(mmproj.read_range(more, 0, more.bytes), "<f4"))
        out = tensor.shape[0]
        # [out, 3, t, ps, ps]: per output channel and colour channel, every frame's ps x ps block.
        blocks = [s.reshape(out, 3, -1) for s in slices]
        values = np.stack(blocks, axis=2).reshape(-1)
    elif rule == "words":
        values = np.frombuffer(raw, "<f4")
    else:
        words = np.frombuffer(raw, "<u2")
        if third >= 0:
            rows = tensor.shape[0] // 3
            words = words.reshape(tensor.shape[0], -1)[third * rows : (third + 1) * rows].reshape(-1)
        return words.tobytes()
    bits = values.view(np.uint32)
    assert ((bits & 0xFFFF) == 0).all(), f"{gguf_name}: an F32 value is not a BF16 word"
    words = (bits >> 16).astype("<u2")
    if third >= 0:
        words = words[third * (words.size // 3) : (third + 1) * (words.size // 3)]
    return words.tobytes()


def verify(path, gguf: GgufModel, subset: qwen4_exp.Subset | None = None, volume=None,
           mtp: GgufModel | None = None, vision: GgufModel | None = None) -> dict:
    """Every tensor object equals its GGUF ranges; returns counts of what was compared.

    Plain objects are compared by SHA-256 over the GGUF bytes of their bound parameters in
    binding order. Layout transforms are inverted exactly: expert records are unpacked to each
    expert's gate, up and down rows, F16 widened to FP32 must narrow back to the original words,
    and a narrowed BF16 tensor must widen back to the GGUF's F32 words (``1 + gamma`` for the
    unit-offset norms). With *volume*, the n-gram volume is checked against the GGUF table too.
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
            if bound[0][2].startswith("mtp/"):
                assert mtp is not None, f"{obj.id}: an MTP object needs the drafter GGUF"
                assert all(name.startswith("mtp/") for _, _, name in bound), obj.id
                summary[_verify_mtp_object(artifact, obj, bound, mtp, config)] += 1
                summary["objects"] += 1
                summary["bytes"] += obj.bytes
                continue
            if bound[0][2].startswith("vision/"):
                assert vision is not None, f"{obj.id}: a vision object needs the mmproj GGUF"
                assert all(name.startswith("vision/") for _, _, name in bound), obj.id
                assert obj.format == "bf16", obj.id
                depth = directory.components["vision"]["config"]["depth"]
                expected = b"".join(_vision_bytes(vision, name, depth) for _, _, name in bound)
                assert artifact.read_object(obj.id) == expected, obj.id
                summary["vision_identical"] += 1
                summary["objects"] += 1
                summary["bytes"] += obj.bytes
                continue
            if obj.layout == "ggml_expert_record_v1":
                ((_, _, name),) = bound
                order = {"gate": 0, "up": 1, "down": 2}
                items = sorted(mappings[name], key=lambda item: order[item.part])
                _verify_bank(artifact, obj, gguf, [gguf.tensor(item.gguf) for item in items])
                summary["records"] += 1
            else:
                tensors = [gguf.tensor(mappings[name][0].gguf) for _, _, name in bound]
                rule = qwen4_exp.NARROWED.get(qwen4_exp.layer_suffix(tensors[0].name))
                if rule is not None:
                    (tensor,) = tensors
                    words = np.frombuffer(artifact.read_object(obj.id), dtype="<u2")
                    stored = (words.astype(np.uint32) << 16).view(np.float32)
                    if rule == "gamma":
                        stored = np.float32(1.0) + stored
                    if qwen4_exp.layer_suffix(tensor.name) in qwen4_exp.TRANSPOSED:
                        stored = stored.reshape(obj.shape).T.copy()
                    source = gguf.read_range(tensor, 0, tensor.bytes)
                    assert obj.format == "bf16" and stored.tobytes() == source, obj.id
                    summary["narrowed"] += 1
                elif (obj.format == "bf16" and tensors[0].type.name != "BF16"
                      and qwen4_exp.layer_suffix(tensors[0].name) in qwen4_exp.BF16_PARAMETERS):
                    expected = b"".join(_bf16_words(gguf, tensor) for tensor in tensors)
                    assert artifact.read_object(obj.id) == expected, obj.id
                    summary[f"bf16_from_{tensors[0].type.name.lower()}"] += 1
                elif obj.format == "fp32" and tensors[0].type.name == "F16":
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
        if vision is not None:
            component = directory.components["vision"]
            assert component["config"] == qwen4_exp.vision_config(vision, config), "vision config"
            for role, expected in (("preprocessor_config.json", qwen4_exp.PREPROCESSOR_CONFIG),
                                   ("video_preprocessor_config.json", qwen4_exp.VIDEO_PREPROCESSOR_CONFIG)):
                assert json.loads(artifact.read_object(component["resources"][role])) == expected, role
        else:
            assert "vision" not in directory.components, "a vision component needs the mmproj GGUF"
    rows =gguf.tensor(qwen4_exp.PLE_TABLE).rows if subset is None else subset.ple_rows
    assert text["ngram_table"]["rows"] == rows
    if volume is not None:
        summary["volume_rows"] = verify_volume(volume, gguf, text["ngram_table"])
    return dict(summary)
