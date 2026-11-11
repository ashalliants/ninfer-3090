#!/usr/bin/env python3
"""Turn a DeepSeek V4.1 expert-routing profile into the rank curve `dsv4_perf_model.py` reads.

Strata (https://github.com/Niko1221/Strata, MIT) ships `data/dsv41/profile-2.00.bin`: for every layer
(40 backbone + 3 MTP) and every routed expert, a little-endian uint32 count of how often the router
chose it over a calibration corpus. Every backbone layer's counts sum to the same number
(10,661,238 = 6 experts x 1,776,873 tokens), so the file is a real routing trace in aggregate: the
only measurement of V4.1's expert popularity that exists without running the model.

The perf model needs one thing from it: how much of the routing mass the k-th most popular expert of a
layer carries. This averages the per-layer sorted shares into one 384-point curve and records the
spread across layers.

    python tools/dsv4/profile_to_curve.py profile-2.00.bin tools/dsv4/data/routing_rank_curve.json \\
        --source "Niko1221/Strata data/dsv41/profile-2.00.bin (MIT)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rank_curve(counts: np.ndarray) -> dict:
    """counts [layers, experts] -> mean per-rank share of routing mass and per-layer top-10% mass."""
    if counts.ndim != 2 or (counts.sum(axis=1) == 0).any():
        raise ValueError("expected [layers, experts] counts with every layer used")
    shares = counts / counts.sum(axis=1, keepdims=True)
    ranked = -np.sort(-shares, axis=1)
    k10 = round(0.10 * counts.shape[1])
    return {
        "layers": int(counts.shape[0]),
        "experts": int(counts.shape[1]),
        "rank_share": [float(x) for x in ranked.mean(axis=0)],
        "top10pct_mass_by_layer": [float(x) for x in ranked[:, :k10].sum(axis=1)],
    }


def load_counts(path: Path, experts: int, backbone_layers: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u4")
    if raw.size % experts:
        raise ValueError(f"{path}: {raw.size} counts is not a multiple of {experts} experts")
    return raw.reshape(-1, experts)[:backbone_layers].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--layers", type=int, default=40, help="backbone layers (the MTP layers' counts are dropped)")
    ap.add_argument("--source", required=True, help="where the profile came from, recorded in the output")
    args = ap.parse_args()
    curve = rank_curve(load_counts(args.profile, args.experts, args.layers))
    curve["source"] = args.source
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curve, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
