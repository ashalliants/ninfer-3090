#!/usr/bin/env python3
"""Golden outputs of the model repo's own attention code, for the NumPy reference to be checked against.

`inference/model.py` of the DeepSeek-V4.1-Flash repo imports GPU kernels (TileLang) and the vision and
Engram modules at import time. This runs its real `Attention`, `Compressor` and `Indexer` classes on
CPU with those imports stubbed: the quantisation round trips (`act_quant`, `fp4_act_quant`) become the
identity, and `sparse_attn` is a PyTorch rendering of the kernel's documented semantics (softmax with a
per-head sink in the denominator, a row with no valid index giving zeros). Everything else, the
projections, RoPE and YaRN, the compressor's pooling, the indexer's scoring and the two-level top-k, the
window ring and the shared state between layers, is the repo's code unchanged.

It needs `torch` and a directory holding the repo's `inference/` files (fetch with `--fetch`):

    python tools/dsv4/model_golden.py --fetch DIR            # the repo's inference files at one commit, recorded in DIR/revision.txt
    python tools/dsv4/model_golden.py --model-dir DIR --emit tests/fixtures/dsv4/attention_golden.json

The weights and inputs are `attention.stack_inputs()`, so the NumPy side regenerates them from the seed
and the golden file holds only the outputs and the commit of the model repo they came from (`revision`): a
regeneration from another commit is visible in the diff, and `--fetch` pins one commit for all its files.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attention as att  # noqa: E402
import moe as moe_ref  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# What the goldens import: model.py and kernel.py (stubbed), engram.py (the real hash code), convert.py (not imported
# by the goldens, fetched so the directory is the repo's whole inference set for reading).
FILES = ("model.py", "kernel.py", "engram.py", "convert.py")
REVISION_FILE = "revision.txt"


def _sha(text: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise SystemExit(f"{text!r} is not a 40-hex commit")
    return text


def revision_of(directory: Path) -> str:
    """The commit of the model repo that `directory` was fetched at; every golden records it."""
    path = directory / REVISION_FILE
    if not path.exists():
        raise SystemExit(f"{directory} has no {REVISION_FILE}: fetch it with --fetch so its commit is recorded")
    return _sha(path.read_text().strip())


def fetch(directory: Path) -> str:
    """All of FILES at ONE commit of the model repo, staged in a sibling directory and published by a single
    rename only when every file arrived, so a failed or interrupted fetch leaves no mixed or partial directory
    (an interruption can leave the staging directory, which is never read, and a later fetch refuses to reuse it
    silently: it names it). Returns the commit."""
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"{directory} is not empty: fetch into a fresh directory")
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}", timeout=60) as response:  # noqa: S310
        sha = _sha(json.load(response)["sha"])
    staging = directory.with_name(directory.name + f".staging-{sha[:12]}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name in FILES:
            url = f"https://huggingface.co/{REPO}/resolve/{sha}/inference/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                (staging / name).write_bytes(response.read())
            print(f"fetched {name}")
        (staging / REVISION_FILE).write_text(sha + "\n")
        if directory.exists():
            directory.rmdir()  # empty (checked above); the rename below is then the only step that publishes
        staging.rename(directory)  # one atomic rename: the directory appears whole or not at all
    finally:
        if staging.exists():  # the rename did not happen: remove the staging files
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
    print(f"model repo {REPO} at {sha}")
    return sha


def _stub_modules(torch) -> None:
    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, s, h, d = q.shape
        out = torch.zeros_like(q)
        for bi in range(b):
            for i in range(s):
                idx = topk_idxs[bi, i]
                valid = idx[idx >= 0].long()
                if valid.numel() == 0:
                    continue
                scores = torch.einsum("hd,kd->hk", q[bi, i], kv[bi, valid]) * softmax_scale
                m = torch.maximum(scores.max(dim=1).values, attn_sink)
                e = torch.exp(scores - m[:, None])
                denom = e.sum(dim=1) + torch.exp(attn_sink - m)
                out[bi, i] = (e @ kv[bi, valid]) / denom[:, None]
        return out

    kernel = types.ModuleType("kernel")
    kernel.act_quant = lambda x, *a, **k: None  # in-place round trip: identity here
    kernel.fp4_act_quant = lambda x, *a, **k: None
    for name in ("fp4_gemm", "fp8_gemm", "hc_split_sinkhorn"):
        setattr(kernel, name, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not stubbed")))
    kernel.sparse_attn = sparse_attn
    engram = types.ModuleType("engram")
    engram.EngramLayout = engram.NgramHashState = object
    image = types.ModuleType("image_processor")
    image.IMAGE = image.IMAGE_END = image.IMAGE_NEW_LINE = image.IMAGE_START = 0
    vision = types.ModuleType("vision")
    vision.Aligner = vision.ViT = object
    for module in (kernel, engram, image, vision):
        sys.modules[module.__name__] = module


def run(model_dir: Path) -> list[np.ndarray]:
    import torch

    _stub_modules(torch)
    sys.path.insert(0, str(model_dir))
    model = importlib.import_module("model")
    model.default_dtype = torch.float32
    cfgs, weights, inputs = att.stack_inputs()
    c0 = cfgs[0]
    args = model.ModelArgs(
        max_batch_size=1, max_seq_len=16, dim=c0.dim, n_layers=len(cfgs), n_heads=c0.n_heads, q_lora_rank=c0.q_lora_rank,
        head_dim=c0.head_dim, rope_head_dim=c0.rope_dim, norm_eps=c0.norm_eps, o_groups=c0.o_groups,
        o_lora_rank=c0.o_lora_rank, window_size=c0.window,
        compress_ratios=tuple(c.compress_ratio for c in cfgs),
        kv_source_layers=tuple(i for i, c in enumerate(cfgs) if c.is_kv_source),
        index_source_layers=tuple(i for i, c in enumerate(cfgs) if c.is_index_source),
        compress_rope_theta=c0.compress_rope_theta, original_seq_len=c0.original_seq_len, rope_theta=c0.rope_theta,
        rope_factor=c0.rope_factor, beta_fast=c0.beta_fast, beta_slow=c0.beta_slow,
        index_n_heads=c0.index_n_heads, index_head_dim=c0.index_head_dim, index_topk=c0.index_topk,
        candidate_source_layer=c0.candidate_source_layer, candidate_topk_blocks=c0.candidate_topk_blocks,
        candidate_block_size=c0.candidate_block_size,
    )

    def put(linear, array):
        linear.weight = torch.nn.Parameter(torch.tensor(array, dtype=torch.float32))

    outputs = []
    with torch.no_grad():
        for cfg, w, x in zip(cfgs, weights, inputs):
            layer = model.Attention(cfg.layer_id, args)
            put(layer.wq_a, w.wq_a); layer.q_norm.weight.data = torch.tensor(w.q_norm)
            put(layer.wq_b, w.wq_b); put(layer.wkv, w.wkv); layer.kv_norm.weight.data = torch.tensor(w.kv_norm)
            put(layer.wo_a, w.wo_a); put(layer.wo_b, w.wo_b); layer.attn_sink.data = torch.tensor(w.attn_sink)
            if layer.compressor is not None:
                put(layer.compressor.wkv, w.comp_wkv); layer.compressor.norm.weight.data = torch.tensor(w.comp_norm)
                if w.comp_wgate is not None:
                    put(layer.compressor.wgate, w.comp_wgate)
            if layer.indexer is not None:
                put(layer.indexer.wq_b, w.idx_wq_b); put(layer.indexer.weights_proj, w.idx_weights_proj)
                if w.idx_wk is not None:
                    put(layer.indexer.wk, w.idx_wk); layer.indexer.k_norm.weight.data = torch.tensor(w.idx_k_norm)
            out = layer(torch.tensor(x)[None], 0)
            outputs.append(out[0].numpy())
    return outputs


def run_moe(model_dir: Path) -> list[np.ndarray]:
    """The repo's real `MoE` (router, routed experts, shared expert) on `moe.golden_inputs()`."""
    import torch

    _stub_modules(torch)
    sys.path.insert(0, str(model_dir))
    model = importlib.import_module("model")
    model.default_dtype = torch.float32
    outputs = []
    with torch.no_grad():
        for x, w, cfg in moe_ref.golden_inputs():
            args = model.ModelArgs(
                dim=moe_ref.GOLDEN_DIM, moe_inter_dim=moe_ref.GOLDEN_INTER, n_layers=1,
                n_routed_experts=moe_ref.GOLDEN_EXPERTS, n_shared_experts=1, n_activated_experts=cfg.topk,
                score_func="sqrtsoftplus", gate_temp=cfg.gate_temp, norm_topk_prob=cfg.norm_topk_prob,
                route_scale=cfg.route_scale, swiglu_limit=cfg.clamp_experts, expert_dtype=None,
            )
            assert cfg.clamp_experts == cfg.clamp_shared, "the repo's MoE has one swiglu limit for both"
            layer = model.MoE(0, args)
            layer.gate.weight = torch.nn.Parameter(torch.tensor(w.router))
            layer.gate.bias = torch.nn.Parameter(torch.tensor(w.bias))
            for target, e in [*zip(layer.experts, w.experts), (layer.shared_experts, w.shared)]:
                for name, array in (("w1", e.w1), ("w2", e.w2), ("w3", e.w3)):
                    getattr(target, name).weight = torch.nn.Parameter(torch.tensor(array))
            outputs.append(layer(torch.tensor(x)[None])[0].numpy())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", type=Path, metavar="DIR", help="download the repo's inference files into DIR")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--emit", type=Path, help="write the attention stack's outputs")
    parser.add_argument("--emit-moe", type=Path, help="write the MoE layers' outputs")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.fetch)
    if args.emit:
        if not args.model_dir:
            parser.error("--emit needs --model-dir")
        outputs = run(args.model_dir)
        args.emit.write_text(json.dumps({"model_repo": REPO, "revision": revision_of(args.model_dir), "seed": att.STACK_SEED,
                                         "seq": att.STACK_SEQ, "outputs": [o.tolist() for o in outputs]}) + "\n")
        print(f"wrote {args.emit} ({len(outputs)} layers)")
    if args.emit_moe:
        if not args.model_dir:
            parser.error("--emit-moe needs --model-dir")
        outputs = run_moe(args.model_dir)
        args.emit_moe.write_text(json.dumps({"seed": moe_ref.GOLDEN_SEED, "outputs": [o.tolist() for o in outputs]}) + "\n")
        print(f"wrote {args.emit_moe} ({len(outputs)} layers)")


if __name__ == "__main__":
    main()
