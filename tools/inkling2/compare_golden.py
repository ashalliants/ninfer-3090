#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up, docs/maintainer/inkling2-model.md "Validation").
"""Run the NumPy reference on the real weights and compare it with a golden: llama.cpp's (llama_golden.cpp) or the
engine's (tests/models/inkling2/test_golden.cpp with NINFER_INKLING2_DUMP, same layout).

    python -m tools.inkling2.compare_golden model.gguf|model.ninfer golden_dir/p0 [--chunk 4] [--json out.json]
        [--routes dir]

The weights come from the GGUF or from the converted artifact, which holds the same bytes. ``--routes`` makes every
layer visit's MoE use the experts in ``dir/route_ids-v.i32`` (int32 [T, k], the engine dump writes them), so a
comparison past a near-tied routing decision measures the remaining arithmetic.

For every traced tensor (``attn_out-v``, ``ffn_inp-v``, ``l_out-v`` per layer visit v, ``result_norm``; all are
required, see :func:`expected_traces`) it
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

from tools.inkling2 import artifact_weights, gguf_weights, reference


def load(path: Path) -> tuple[dict, dict]:
    return (artifact_weights if path.suffix == ".ninfer" else gguf_weights).load(path)


def read_routes(directory: Path, visits: int, top_k: int) -> dict[int, np.ndarray]:
    routes = {}
    for visit in range(visits):
        path = directory / f"route_ids-{visit}.i32"
        if not path.exists():
            raise ValueError(f"{directory}: no route_ids-{visit}.i32")
        routes[visit] = np.fromfile(path, np.int32).reshape(-1, top_k)
    return routes


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def expected_traces(model: reference.Model) -> list[str]:
    """Every activation the golden must hold: ``ffn_inp`` and ``l_out`` of each visit, ``attn_out`` of each visit
    that does not start a residual block (there the attention output *is* ``ffn_inp``, and llama.cpp's callback
    names the one tensor ``ffn_inp``), and ``result_norm``."""
    names = []
    for visit in range(model.visits):
        if model.physical(visit) % model.config["block"]:
            names.append(f"attn_out-{visit}")
        names += [f"ffn_inp-{visit}", f"l_out-{visit}"]
    return names + ["result_norm"]


def compare(model_path: Path, golden: Path, chunk: int = 0, routes: Path | None = None) -> dict:
    meta = json.loads((golden / "tokens.json").read_text())
    tokens = np.asarray(meta["tokens"])
    weights, config = load(model_path)
    model = reference.Model(config, weights)
    forced = None if routes is None else read_routes(routes, model.visits, config["top_k"])
    state = model.new_state()
    traces: dict[str, list[np.ndarray]] = {}
    logits = []
    step = chunk or len(tokens)
    began = time.time()
    for start in range(0, len(tokens), step):
        trace: dict = {}
        part = None if forced is None else {v: ids[start : start + step] for v, ids in forced.items()}
        out, _ = model.forward(tokens[start : start + step], state, start, trace, part)
        logits.append(out)
        for name, rows in trace.items():
            traces.setdefault(name, []).append(rows)
    ref_logits = np.concatenate(logits)
    report = {"tokens": len(tokens), "chunk": chunk, "seconds": round(time.time() - began, 1), "tensors": {}}
    H = config["hidden"]
    for name in expected_traces(model):
        path = golden / f"{name}.f32"
        if not path.exists():
            raise ValueError(f"{golden}: the golden has no {name} (llama_golden.cpp captures it)")
        theirs = np.fromfile(path, np.float32).astype(np.float64)
        if theirs.size != len(tokens) * H:
            raise ValueError(f"{path}: {theirs.size} values, expected {len(tokens)} x {H}")
        theirs = theirs.reshape(-1, H)
        ours = np.concatenate(traces[name])
        rel = float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
        cos = float(np.sum(ours * theirs) / (np.linalg.norm(ours) * np.linalg.norm(theirs)))
        per_token = np.linalg.norm(ours - theirs, axis=1) / np.linalg.norm(theirs, axis=1)
        report["tensors"][name] = {"rel": rel, "cos": cos, "rel_per_token": per_token.round(5).tolist()}
    # A golden with the routing it used (the engine's dump): where this reference picks other experts, and how close
    # its k-th and next scores were there.
    if (golden / "route_ids-0.i32").exists():
        theirs_routes = read_routes(golden, model.visits, config["top_k"])
        differ = []
        for visit, ids in theirs_routes.items():
            ours = np.concatenate(traces[f"route_ids-{visit}"])
            gaps = np.concatenate(traces[f"route_gap-{visit}"])
            for t in range(len(tokens)):
                if set(ours[t].tolist()) != set(ids[t].tolist()):
                    differ.append({"visit": visit, "token": t, "gap": float(gaps[t]),
                                   "swapped": len(set(ours[t].tolist()) - set(ids[t].tolist()))})
        report["routes"] = {"decisions": model.visits * len(tokens), "differ": differ}
    theirs = np.fromfile(golden / "logits.f32", np.float32).astype(np.float64)
    if theirs.size != len(tokens) * ref_logits.shape[1]:
        raise ValueError(f"{golden}/logits.f32: {theirs.size} values, expected {len(tokens)} x {ref_logits.shape[1]}")
    theirs = theirs.reshape(len(tokens), -1)
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
    parser.add_argument("--routes", type=Path)
    args = parser.parse_args(argv)
    report = compare(args.model, args.golden, args.chunk, args.routes)
    for name in sorted(report["tensors"], key=_order):
        r = report["tensors"][name]
        print(f"{name:16s} rel {r['rel']:.3e}  cos {r['cos']:.6f}")
    lg = report["logits"]
    print(f"logits max|d| {lg['max_abs']:.3f}  KLD {['%.2e' % k for k in lg['kld']]}  argmax agree {lg['argmax_agree']}")
    if "routes" in report:
        # The first difference in visit order is the seed; later ones also follow from it (attention and the
        # convolutions carry it to later tokens), so their margins say less.
        d = sorted(report["routes"]["differ"], key=lambda x: (x["visit"], x["token"]))
        print(f"routing differs at {len(d)} of {report['routes']['decisions']} (visit, token) decisions"
              + (f"; the first at visit {d[0]['visit']}, token {d[0]['token']}, margin {d[0]['gap']:.2e}" if d else ""))
    print(f"{report['tokens']} tokens, chunk {report['chunk']}, {report['seconds']} s")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
