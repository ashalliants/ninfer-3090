#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up, docs/maintainer/inkling2-model.md "Validation").
"""Run the NumPy reference on the real GGUF weights and compare it with a llama.cpp golden (llama_golden.cpp).

    python -m tools.inkling2.compare_golden model.gguf golden_dir/p0 [--chunk 4] [--json out.json]

For every traced tensor (``attn_out-v``, ``ffn_inp-v``, ``l_out-v`` per layer visit v, ``result_norm``) it
prints the relative error ||ref - llama|| / ||llama|| and the cosine; for the logits, the largest absolute
difference, the per-position KL divergence KL(llama || ref) and whether the argmax agrees. The reference is
FP64 on the stored weights; llama.cpp's CPU path rounds activations (BF16 for BF16 weights, Q8 blocks for
NVFP4 experts), so agreement is expected near 1e-2 relative, not exactly. A structural error (a wrong loop
restart, message, tap order) shows as errors of order 1.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from tools.inkling2 import reference
from tools.inkling2.gguf_weights import load


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def compare(model_path: Path, golden: Path, chunk: int = 0) -> dict:
    meta = json.loads((golden / "tokens.json").read_text())
    tokens = np.asarray(meta["tokens"])
    weights, config = load(model_path)
    model = reference.Model(config, weights)
    state = model.new_state()
    traces: dict[str, list[np.ndarray]] = {}
    logits = []
    step = chunk or len(tokens)
    began = time.time()
    for start in range(0, len(tokens), step):
        trace: dict = {}
        out, _ = model.forward(tokens[start : start + step], state, start, trace)
        logits.append(out)
        for name, rows in trace.items():
            traces.setdefault(name, []).append(rows)
    ref_logits = np.concatenate(logits)
    report = {"tokens": len(tokens), "chunk": chunk, "seconds": round(time.time() - began, 1), "tensors": {}}
    H = config["hidden"]
    for name, parts in traces.items():
        path = golden / f"{name}.f32"
        if not path.exists():
            continue
        ours = np.concatenate(parts)
        theirs = np.fromfile(path, np.float32).astype(np.float64).reshape(-1, H)
        rel = float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
        cos = float(np.sum(ours * theirs) / (np.linalg.norm(ours) * np.linalg.norm(theirs)))
        per_token = np.linalg.norm(ours - theirs, axis=1) / np.linalg.norm(theirs, axis=1)
        report["tensors"][name] = {"rel": rel, "cos": cos, "rel_per_token": per_token.round(5).tolist()}
    theirs = np.fromfile(golden / "logits.f32", np.float32).astype(np.float64).reshape(len(tokens), -1)
    p_log, q_log = _log_softmax(theirs), _log_softmax(ref_logits)
    kld = np.sum(np.exp(p_log) * (p_log - q_log), axis=-1)
    report["logits"] = {
        "max_abs": float(np.abs(theirs - ref_logits).max()),
        "kld": kld.tolist(),
        "argmax_agree": (theirs.argmax(-1) == ref_logits.argmax(-1)).tolist(),
        "argmax": ref_logits.argmax(-1).tolist(),
    }
    return report


def _order(name: str) -> tuple:
    m = re.match(r"(\w+)-(\d+)$", name)
    return (int(m.group(2)), m.group(1)) if m else (10**6, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = compare(args.model, args.golden, args.chunk)
    for name in sorted(report["tensors"], key=_order):
        r = report["tensors"][name]
        print(f"{name:16s} rel {r['rel']:.3e}  cos {r['cos']:.6f}")
    lg = report["logits"]
    print(f"logits max|d| {lg['max_abs']:.3f}  KLD {['%.2e' % k for k in lg['kld']]}  argmax agree {lg['argmax_agree']}")
    print(f"{report['tokens']} tokens, chunk {report['chunk']}, {report['seconds']} s")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
