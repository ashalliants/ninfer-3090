#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up). The mathematics is docs/maintainer/inkling2-model.md, "Vision
# tower", which cites its sources: llama.cpp's tools/mtmd/models/inkling2v.cpp and transformers'
# Inkling2VisionModel. Nothing here is copied from either; it restates the formulas.
"""An FP64 (torch) reference of Inkling-2-Mini's vision tower, read straight from the mmproj GGUF.

    python tools/inkling2/vision_reference.py MMPROJ.gguf IMAGE.rgb WIDTH HEIGHT OUT.emb

IMAGE.rgb holds WIDTH x HEIGHT pixels of 3 bytes (rows top to bottom); the image is padded, not resized (it must
fit the tower's token limit). OUT.emb gets int32 tokens, int32 width, then the FP32 embeddings, one row per image
token: what ninfer_inkling2_vision_test compares the engine against.
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.sources.gguf import GgufModel  # noqa: E402
mm, rgb, W, H, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
g = GgufModel(mm)
def t(name):
    x = g.tensor(name); return g.read_values(x, 0, x.elements).to(torch.float64).reshape(x.shape)
m = g.metadata
P, D, NH, L = m["clip.vision.patch_size"], m["clip.vision.embedding_length"], m["clip.vision.attention.head_count"], m["clip.vision.block_count"]
eps = float(np.float32(m["clip.vision.attention.layer_norm_epsilon"]))
A = 2 * P
Wp_, Hp_ = (W + A - 1) // A * A, (H + A - 1) // A * A
img = np.frombuffer(open(rgb, "rb").read(), np.uint8).reshape(H, W, 3).astype(np.float32)
pix = np.full((Hp_, Wp_, 3), -1.0, np.float32); pix[:H, :W] = img
mean = np.float32(m["clip.vision.image_mean"]) * np.float32(255)  # FP32, as llama.cpp's normalize
std = np.float32(m["clip.vision.image_std"]) * np.float32(255)
pix = (pix - mean) / std
x = torch.from_numpy(pix.astype(np.float64)).permute(2, 0, 1)  # [3, H, W]
gw, gh = Wp_ // P, Hp_ // P
patches = x.reshape(3, gh, P, gw, P).permute(1, 3, 0, 2, 4).reshape(gh * gw, 3 * P * P)
cur = patches @ t("v.patch_embd.weight").reshape(D, -1).T      # [N, D]
pos = t("v.position_embd.weight")                               # [n_pos, D]
n = math.isqrt(pos.shape[0]); pos = pos.reshape(n, n, D)        # [y, x, D]
if (gw, gh) != (n, n):
    def coords(dst, src):
        out = []
        for i in range(dst):
            f = (i + 0.5) / (dst / src) - 0.5
            i0 = math.floor(f); i1 = i0 + 1
            c0, c1 = min(max(i0, 0), src - 1), min(max(i1, 0), src - 1)
            out.append((c0, c1, min(max(f - c0, 0.0), 1.0)))
        return out
    cx, cy = coords(gw, n), coords(gh, n)
    rp = torch.empty(gh, gw, D, dtype=torch.float64)
    for yi, (y0, y1, dy) in enumerate(cy):
        for xi, (x0, x1, dx) in enumerate(cx):
            rp[yi, xi] = (pos[y0, x0] * (1 - dx) * (1 - dy) + pos[y0, x1] * dx * (1 - dy) + pos[y1, x0] * (1 - dx) * dy
                          + pos[y1, x1] * dx * dy)
    pos = rp
cur = cur + pos.reshape(gh * gw, D)
cls = t("v.class_embd") + t("v.class_position_embd.weight")
h = torch.cat([cls[None], cur])                                  # [1 + N, D]
pw = torch.tensor([0] + [xx + 1 for yy in range(gh) for xx in range(gw)], dtype=torch.float64)
ph = torch.tensor([0] + [yy + 1 for yy in range(gh) for xx in range(gw)], dtype=torch.float64)
def ln(v, w, b):
    mu = v.mean(-1, keepdim=True); var = ((v - mu) ** 2).mean(-1, keepdim=True)
    return (v - mu) / torch.sqrt(var + eps) * w + b
hd = D // NH; half = hd // 2
inv = 10000.0 ** (-torch.arange(0, half, 2, dtype=torch.float64) / half)   # pair i: base^(-2i/half)
def rope(q):  # q [T, NH, hd]
    q = q.clone()
    for off, p in ((0, pw), (half, ph)):
        a = p[:, None] * inv[None]          # [T, half/2]
        c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
        x0, x1 = q[..., off:off + half:2].clone(), q[..., off + 1:off + half:2].clone()
        q[..., off:off + half:2] = x0 * c - x1 * s
        q[..., off + 1:off + half:2] = x0 * s + x1 * c
    return q
h = ln(h, t("v.pre_ln.weight"), t("v.pre_ln.bias"))
T = h.shape[0]
gelu = lambda v: 0.5 * v * (1 + torch.erf(v / math.sqrt(2)))
for i in range(L):
    b = f"v.blk.{i}."
    y = ln(h, t(b + "ln1.weight"), t(b + "ln1.bias"))
    q = (y @ t(b + "attn_q.weight").T + t(b + "attn_q.bias")).reshape(T, NH, hd)
    k = (y @ t(b + "attn_k.weight").T + t(b + "attn_k.bias")).reshape(T, NH, hd)
    v = (y @ t(b + "attn_v.weight").T + t(b + "attn_v.bias")).reshape(T, NH, hd)
    q, k = rope(q), rope(k)
    s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
    o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v).reshape(T, D)
    o = o @ t(b + "attn_out.weight").T + t(b + "attn_out.bias")
    h = h + o * t(b + "ls1.weight")
    y = ln(h, t(b + "ln2.weight"), t(b + "ln2.bias"))
    y = gelu(y @ t(b + "ffn_up.weight").T + t(b + "ffn_up.bias")) @ t(b + "ffn_down.weight").T + t(b + "ffn_down.bias")
    h = h + y * t(b + "ls2.weight")
# A merged token's 4 D values: patch (2X + dx, 2Y + dy) at e + D dx + 2 D dy.
p = h[1:].reshape(gh // 2, 2, gw // 2, 2, D).permute(0, 2, 1, 3, 4).reshape(-1, 4 * D)
e = gelu(p @ t("mm.1.weight").T) @ t("mm.2.weight").T
with open(out, "wb") as f:
    f.write(struct.pack("<ii", e.shape[0], e.shape[1])); f.write(e.to(torch.float32).numpy().tobytes())
print("tokens", e.shape, "grid", gw, gh, "norm", float(e.norm()))
