#!/usr/bin/env python3
"""Export one DeepSeek-V4.1-Flash attention layer's REAL weights and the NumPy oracle's activations.

The GPU integration test `ninfer_dsv4_attention_layer_test` (tests/ops/test_dsv4_attention_layer.cpp)
runs the layer through the Ops on the card and compares with what this writes. It cannot read a GGUF
itself, so this script does the read-and-run-the-oracle half:

  python3 tools/dsv4/export_layer_fixture.py FIRST_SHARD.gguf OUT_DIR [--layer 0] [--tokens 200] [--seed 1]

OUT_DIR (must not exist or be empty) gets, per tensor, the GGUF's own bytes (`<name>.bin`: the
quantised blocks of the projections, BF16 words of the small projections, FP32 for norms and sinks),
`x.bin` (BF16 input, [tokens, hidden]), `inv_freq.bin` (the layer's RoPE frequencies), the oracle's
activations as FP32 (`qr`, `q`, `kv`, `o_pre`, `o`, `projected`, `out`, and for a compressing layer
`latent`, `ckv`, `index_k` and the selection `topk`, each [tokens or groups, ...]) and `config.txt`
(one `key value` per line).

Supported layers: window-only (compress ratio 0) and ratio-2 layers that own their compressed KV and
their indexer (kv source and index source: layers 2, 8 and 14 of V4.1-Flash). Layer 20 (ratio 1, the
candidate-block source) and layers that read another layer's compressed KV need state this exporter
does not model and are refused.

The oracle is tools/dsv4/attention.py on the weights decoded exactly as ggml does
(tools/artifact/codecs/ggml_blocks.py), in FP32: the device's activation casts (BF16 and Q8_1) are the
difference the test's criterion allows for.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.codecs.ggml_blocks import decode_ggml_blocks  # noqa: E402
from tools.convert.sources.gguf import GgufModel  # noqa: E402
from tools.dsv4 import attention as att  # noqa: E402

# GGUF tensor name (without "blk.N.") -> oracle field, per group.
_PROJECTIONS = {
    "attn_q_a.weight": "wq_a",
    "attn_q_b.weight": "wq_b",
    "attn_kv.weight": "wkv",
    "attn_output_a.weight": "wo_a",
    "attn_output_b.weight": "wo_b",
}
_VECTORS = {"attn_q_a_norm.weight": "q_norm", "attn_kv_a_norm.weight": "kv_norm", "attn_sinks.weight": "attn_sink"}
_COMPRESSOR = {"attn_compressor_kv.weight": "comp_wkv", "attn_compressor_gate.weight": "comp_wgate"}
_COMPRESSOR_VECTORS = {"attn_compressor_norm.weight": "comp_norm"}
_INDEXER = {"indexer.attn_q_b.weight": "idx_wq_b", "indexer.proj.weight": "idx_weights_proj", "indexer.attn_k.weight": "idx_wk"}
_INDEXER_VECTORS = {"indexer.k_norm.weight": "idx_k_norm"}


def _bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(_bf16_round(x)).view(np.uint32) >> 16).astype(np.uint16)


def _read_matrix(gguf: GgufModel, name: str, out: Path, field: str, config: dict) -> np.ndarray:
    """Write a projection's own bytes; return its values decoded to FP32 [rows, cols]."""
    tensor = gguf.tensor(name)
    raw = np.ascontiguousarray(gguf.read_rows(tensor, 0, tensor.rows), np.uint8)
    (out / f"{field}.bin").write_bytes(raw.tobytes())
    config[f"{field}.format"] = tensor.type.name
    config[f"{field}.rows"], config[f"{field}.cols"] = tensor.shape
    print(f"{field}: {tensor.type.name} {list(tensor.shape)}", flush=True)
    if tensor.type.dtype is not None:  # direct words (BF16)
        return gguf.read_values(tensor, 0, tensor.elements).float().numpy().reshape(tensor.shape)
    return decode_ggml_blocks(raw, tensor.type.format, tensor.shape)


def _read_vector(gguf: GgufModel, name: str, out: Path, field: str) -> np.ndarray:
    tensor = gguf.tensor(name)
    values = gguf.read_values(tensor, 0, tensor.elements).float().numpy().astype("<f4")
    values.tofile(out / f"{field}.bin")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf", type=Path, help="the first shard of the split GGUF")
    parser.add_argument("out", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    final = args.out
    if final.exists() and any(final.iterdir()):
        parser.error(f"{final} is not empty")
    # Written beside the target and renamed once complete, so a failed export never leaves a half fixture there.
    final.parent.mkdir(parents=True, exist_ok=True)
    args.out = Path(tempfile.mkdtemp(prefix=final.name + ".partial-", dir=final.parent))  # unique, owned by this run
    _OWNED.append(args.out)

    weights: dict[str, np.ndarray] = {}
    config: dict = {"layer": args.layer, "tokens": args.tokens}
    with GgufModel(args.gguf) as gguf:
        meta = gguf.metadata
        key = "deepseek41."
        ratio = int(meta[key + "attention.compress_ratios"][args.layer])
        kv_source = args.layer in [int(v) for v in meta[key + "attention.kv_source_layers"]]
        index_source = args.layer in [int(v) for v in meta[key + "attention.indexer.source_layers"]]
        if ratio not in (0, 2) or (ratio and not (kv_source and index_source)):
            parser.error(f"layer {args.layer} (ratio {ratio}, kv source {kv_source}, index source {index_source}) is not exportable")
        if ratio and args.tokens < ratio:
            parser.error(f"--tokens {args.tokens} is below the compress ratio {ratio}: the compressor would produce no latents")
        cfg = att.AttentionConfig(
            dim=int(meta[key + "embedding_length"]),
            n_heads=int(meta[key + "attention.head_count"]),
            head_dim=int(meta[key + "attention.key_length"]),
            rope_dim=int(meta[key + "rope.dimension_count"]),
            q_lora_rank=int(meta[key + "attention.q_lora_rank"]),
            o_groups=int(meta[key + "attention.output_group_count"]),
            o_lora_rank=int(meta[key + "attention.output_lora_rank"]),
            window=int(meta[key + "attention.sliding_window"]),
            norm_eps=float(np.float32(meta[key + "attention.layer_norm_rms_epsilon"])),
            compress_ratio=ratio,
            rope_theta=float(meta[key + "rope.freq_base"]),
            compress_rope_theta=float(meta[key + "attention.compress_rope_freq_base"]),
            original_seq_len=int(meta[key + "rope.scaling.original_context_length"]),
            rope_factor=float(meta[key + "rope.scaling.factor"]),
            beta_fast=float(meta[key + "rope.scaling.yarn_beta_fast"]),
            beta_slow=float(meta[key + "rope.scaling.yarn_beta_slow"]),
            is_kv_source=kv_source,
            is_index_source=index_source,
            index_n_heads=int(meta[key + "attention.indexer.head_count"]),
            index_head_dim=int(meta[key + "attention.indexer.key_length"]),
            index_topk=int(meta[key + "attention.indexer.top_k"]),
            layer_id=args.layer,
            candidate_source_layer=int(meta[key + "attention.indexer.candidate_source_layer"]),
            candidate_topk_blocks=int(meta[key + "attention.indexer.candidate_top_k_blocks"]),
            candidate_block_size=int(meta[key + "attention.indexer.candidate_block_size"]),
        )
        if ratio and cfg.candidate_source_layer == args.layer:
            parser.error("a candidate-source layer needs the two-level selection, which is not exported")
        config.update(
            ratio=ratio,
            window=cfg.window,
            rope_dim=cfg.rope_dim,
            index_topk=cfg.index_topk,
            index_heads=cfg.index_n_heads,
            index_head_dim=cfg.index_head_dim,
            norm_eps=cfg.norm_eps,
        )
        prefix = f"blk.{args.layer}."
        for suffix, field in _PROJECTIONS.items():
            weights[field] = _read_matrix(gguf, prefix + suffix, args.out, field, config)
        for suffix, field in _VECTORS.items():
            weights[field] = _read_vector(gguf, prefix + suffix, args.out, field)
        if ratio:
            for suffix, field in _COMPRESSOR.items():
                if field == "comp_wgate" and ratio == 1:
                    continue
                weights[field] = _read_matrix(gguf, prefix + suffix, args.out, field, config)
            for suffix, field in _COMPRESSOR_VECTORS.items():
                weights[field] = _read_vector(gguf, prefix + suffix, args.out, field)
            for suffix, field in _INDEXER.items():
                weights[field] = _read_matrix(gguf, prefix + suffix, args.out, field, config)
            for suffix, field in _INDEXER_VECTORS.items():
                weights[field] = _read_vector(gguf, prefix + suffix, args.out, field)

    w = att.AttentionWeights(**weights)
    rng = np.random.default_rng(args.seed)
    x = _bf16_round(rng.standard_normal((args.tokens, cfg.dim)).astype(np.float32))
    _bf16_bits(x).tofile(args.out / "x.bin")

    # The oracle, stage by stage (the same composition as att.attention).
    s, h, hd, rd = args.tokens, cfg.n_heads, cfg.head_dim, cfg.rope_dim
    if ratio:
        base, original = cfg.compress_rope_theta, cfg.original_seq_len
    else:  # no YaRN and the base theta in pure sliding-window attention
        base, original = cfg.rope_theta, 0
    att.rope_inv_freqs(rd, original, base, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow).astype("<f4").tofile(args.out / "inv_freq.bin")
    freqs = att.rope_freqs(rd, s, original, base, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    qr = att.rms_norm(x @ w.wq_a.T, w.q_norm, cfg.norm_eps)
    q = att._rope_tail((qr @ w.wq_b.T).reshape(s, h, hd), freqs[:s], rd)
    kv = att._rope_tail(att.rms_norm(x @ w.wkv.T, w.kv_norm, cfg.norm_eps), freqs[:s], rd)
    idxs = att.window_indices(s, cfg.window)
    all_kv = kv
    stages = {"qr": qr, "q": q, "kv": kv}
    shared = att.Shared()
    if ratio:
        latent = att.compressor(x, w, cfg)
        shared.compress_kv = att._rope_tail(latent, freqs[: s - s % ratio : ratio], rd)
        shared.topk_idxs = att.indexer(x, qr, latent, w, cfg, freqs, shared, offset=s)
        groups = s // ratio
        all_kv = np.concatenate([kv, shared.compress_kv[:groups]], axis=0)
        idxs = np.concatenate([idxs, shared.topk_idxs], axis=1)
        stages.update(latent=latent, ckv=shared.compress_kv[:groups], index_k=shared.index_k[:groups])
        np.ascontiguousarray(shared.topk_idxs, "<i4").tofile(args.out / "topk.bin")
        config["topk_width"] = int(shared.topk_idxs.shape[1])
    o_pre = att.sparse_attention(q, all_kv, w.attn_sink, idxs, hd**-0.5)
    o = att._rope_tail(o_pre, freqs[:s], rd, inverse=True)
    grouped = o.reshape(s, cfg.o_groups, -1)
    projected = np.einsum("sgd,grd->sgr", grouped, w.wo_a.reshape(cfg.o_groups, cfg.o_lora_rank, -1))
    out = (projected.reshape(s, -1) @ w.wo_b.T).astype(np.float32)
    reference = att.attention(x, w, cfg, att.Shared())
    assert np.array_equal(out, reference), "the staged oracle differs from attention()"
    stages.update(o_pre=o_pre, o=o, projected=projected.reshape(s, -1), out=out)
    for name, value in stages.items():
        np.ascontiguousarray(value, "<f4").tofile(args.out / f"{name}.bin")
    (args.out / "config.txt").write_text("".join(f"{k} {v}\n" for k, v in config.items()))
    if final.exists():
        final.rmdir()  # empty, checked above
    args.out.rename(final)
    print(f"wrote {final}", flush=True)
    return 0


_OWNED: list[Path] = []  # the temporary directories this run created; removed if the export fails

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the cleanup below on termination too
    try:
        raise SystemExit(main())
    except BaseException:
        for path in _OWNED:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        raise
