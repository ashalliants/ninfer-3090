# Adapted from Infernix a3edb450 tools/flash_next/block_check.py (Apache-2.0).
# Modified for NInfer-3090: the reference reads the converted artifact (tools/flash_next/reference.py)
# and the taps come from ninfer_qwen4_exp_forward_real_test; per-position errors are summarized.
"""Check each NInfer Qwen4Exp block's op chains against the FP64 reference on identical inputs.

    python -m tools.flash_next.block_check --artifact OUT.ninfer --tokens 1,2,3 --blocks blocks.bin \\
        --residuals residuals.bin [--layers 0,1,3]

``blocks.bin`` and ``residuals.bin`` are the ``--blocks`` and ``--residuals`` taps of
ninfer_qwen4_exp_forward_real_test: BF16 ``[layers][mixer in, mixer out, MoE in, MoE out][T][H]``
and BF16 ``[layers][T][S*H]`` (the residual after each block). For every selected layer the
reference recomputes, from the engine's own inputs:

- ``attn_mix``: the PLE injection (its layer only) and the attention-side hyper-connection mixer,
  from the residual entering the block;
- ``mixer``: GDN or QSA from the engine's mixer input;
- ``mlp_mix``: the injection of the engine's mixer output and the MLP-side mixer;
- ``moe``: the MoE from the engine's MoE input;
- ``residual``: the injection of the engine's MoE output,

so each reported error belongs to that op chain alone rather than to accumulated drift. Each value
is the relative L2 error per position; the line reports its maximum and median over positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.flash_next.reference import F64, Reference


def bf16_file(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(raw.view(np.float32).reshape(shape)).to(F64)


def relative(got: torch.Tensor, ref: torch.Tensor) -> dict:
    error = ((got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).numpy()
    return {"max": round(float(error.max()), 5), "median": round(float(np.median(error)), 5)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--volume", type=Path)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids, or @FILE")
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--layers", help="comma-separated layer indices (default: all)")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    text = Path(args.tokens[1:]).read_text() if args.tokens.startswith("@") else args.tokens
    ids = [int(v) for v in text.replace("\n", ",").split(",") if v.strip()]
    ref = Reference(args.artifact, volume=args.volume)
    layers_total, hidden, T = ref.c["num_hidden_layers"], ref.H, len(ids)
    taps = bf16_file(args.blocks, (layers_total, 4, T, hidden))
    residuals = bf16_file(args.residuals, (layers_total, T, ref.hc * hidden))
    embedded = ref.embed(ids)
    layers = [int(v) for v in args.layers.split(",")] if args.layers else range(layers_total)
    positions = torch.arange(T)
    for layer in layers:
        x, y, xm, ym = taps[layer]
        R = embedded.repeat(1, ref.hc) if layer == 0 else residuals[layer - 1]
        if layer + 1 in ref.ple_layers:
            out, _ = ref.ple(layer, R, ids)
            R = ref.round(R + out)
        x_ref, inject = ref.hc_mix(f"layers/{layer}/attn_hc/", R)
        mixer = ref.gdn(layer, x) if ref.layer_types[layer] == "linear" else ref.qsa(layer, x, positions)
        R = ref.inject(R, y, inject)
        xm_ref, inject = ref.hc_mix(f"layers/{layer}/mlp_hc/", R)
        moe, _ = ref.moe(layer, xm)
        R = ref.inject(R, ym, inject)
        ref.drop_cache()
        print(json.dumps({"layer": layer, "mixer": ref.layer_types[layer], "attn_mix": relative(x, x_ref),
                          "mixer_rel_error": relative(y, mixer), "mlp_mix": relative(xm, xm_ref),
                          "moe_rel_error": relative(ym, moe), "residual": relative(residuals[layer], R)}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
