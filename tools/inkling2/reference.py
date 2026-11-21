#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up). The mathematics is docs/maintainer/inkling2-model.md, which
# cites its sources: transformers' modeling_inkling2.py (non-loop ops) and unslothai/llama.cpp 8aa2f23f
# src/models/inkling2.cpp (the loop). Nothing here is copied from either; it restates the formulas.
"""A NumPy (FP64) reference for Inkling-2-Mini's text model: every op, and a whole looped forward with state.

It is the oracle the engine's Ops and Program are qualified against, at toy or real shapes:

  * ``decode_nvfp4``           GGML NVFP4 blocks -> float (ggml ``dequantize_row_nvfp4``),
  * ``nvfp4_integer_form``     the same blocks as int8 codes and one FP32 scale per 16 values (the dp4a form),
  * ``repack_checkpoint_nvfp4``  the checkpoint's NVFP4 rows (low nibble first, E4M3 scale per 16) -> GGML blocks,
  * ``l2_clip``, ``log_scale``, ``relative_bias``, ``attention``, ``short_conv``, ``mix``, ``control``, ``route``,
    ``situ``, ``moe``: the ops of the spec, one token row per row,
  * ``Model.forward``          a chunk of tokens through all layer visits, carrying per-visit KV and convolution
    state, so a sequence may be fed whole or in pieces (the two must agree),
  * ``Model.random``           a toy model with the real structure (loop, block residuals, local/global layers).

    python tools/inkling2/reference.py --emit-golden tests/fixtures/inkling2/golden.json

writes deterministic toy vectors the C++ oracle test (tests/ops/inkling2_reference_test.cpp) compares against.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- NVFP4 (GGML type 40)

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)  # 2 * E2M1
QK_NVFP4, NVFP4_BLOCK_BYTES = 64, 36


def e4m3(byte: np.ndarray) -> np.ndarray:
    """E4M3FN words -> float64 (NaN for 0x7F/0xFF)."""
    b = np.asarray(byte, dtype=np.int64)
    exponent, mantissa = (b >> 3) & 0xF, b & 7
    value = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8.0) * np.exp2(exponent - 7.0))
    value = np.where(b & 0x80, -value, value)
    return np.where((b & 0x7F) == 0x7F, np.nan, value)


def ue4m3_half(byte: np.ndarray) -> np.ndarray:
    """ggml_ue4m3_to_fp32: bytes 0x00 and 0x7F decode to 0; otherwise bits 6..0 as E4M3 (no NaN), times 0.5.

    The sign bit is ignored, so 0x80 is 0 and 0xFF is 240 (ggml decodes rather than rejects them).
    """
    b = np.asarray(byte, dtype=np.int64)
    exponent, mantissa = (b >> 3) & 0xF, b & 7
    value = np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8.0) * np.exp2(exponent - 7.0)) * 0.5
    return np.where((b == 0) | (b == 0x7F), 0.0, value)


def decode_nvfp4(blocks: bytes | np.ndarray) -> np.ndarray:
    """GGML NVFP4 blocks ``{u8 d[4]; u8 qs[32]}`` -> flat float64 values, 64 per block."""
    raw = np.frombuffer(bytes(blocks), dtype=np.uint8) if not isinstance(blocks, np.ndarray) else blocks.reshape(-1)
    if raw.size % NVFP4_BLOCK_BYTES:
        raise ValueError(f"{raw.size} bytes is not whole NVFP4 blocks")
    codes, scales = nvfp4_integer_form(raw)
    return (codes.astype(np.float64).reshape(-1, 4, 16) * scales.reshape(-1, 4, 1)).reshape(-1)


def nvfp4_integer_form(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(int8 codes in [-12, 12] per value, FP64 scale per 16 values): value = code * scale exactly.

    This is the form an A8 ``dp4a`` kernel consumes: the codes are twice E2M1 and the scale is ``ue4m3_half``.
    Sub-block s of a block holds values 16s..16s+15; value j < 8 is the low nibble of ``qs[8s + j]``, value
    j + 8 its high nibble.
    """
    b = np.asarray(blocks, dtype=np.uint8).reshape(-1, NVFP4_BLOCK_BYTES)
    d, qs = b[:, :4], b[:, 4:].reshape(-1, 4, 8)
    nibbles = np.concatenate((qs & 0xF, qs >> 4), axis=2)  # [block, sub, 16]
    return KVALUES_MXFP4[nibbles].reshape(-1), ue4m3_half(d).reshape(-1)


def repack_checkpoint_nvfp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Checkpoint NVFP4 rows -> GGML blocks, bit exact.

    ``packed`` is uint8 [N, K/2] with element 2i in the low nibble of byte i; ``scales`` is E4M3 uint8 [N, K/16].
    Scales must be non-negative and finite (a GGML block cannot carry a sign or NaN in its UE4M3 scale).
    """
    n, half = packed.shape
    groups = scales.shape[1]
    if half * 2 != groups * 16 or groups % 4:
        raise ValueError("NVFP4 rows must be whole 64-value blocks")
    if np.any(scales & 0x80) or np.any(scales == 0x7F):
        raise ValueError("NVFP4 block scales must be non-negative and finite")
    values = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(n, groups, 16)
    qs = (values[:, :, :8] | (values[:, :, 8:] << 4)).astype(np.uint8)
    out = np.concatenate((scales.reshape(n, groups // 4, 4), qs.reshape(n, groups // 4, 32)), axis=-1)
    return out.reshape(n, groups // 4 * NVFP4_BLOCK_BYTES)


def decode_checkpoint_nvfp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The checkpoint's own meaning of its NVFP4 rows: E2M1(code) * E4M3(scale), [N, K]."""
    n = packed.shape[0]
    codes = np.stack((packed & 0xF, packed >> 4), axis=-1).reshape(n, -1)
    return E2M1[codes] * np.repeat(e4m3(scales), 16, axis=1)


# --------------------------------------------------------------------------- ops (rows are tokens)

EPS = 1e-6


def unit_rms(x: np.ndarray) -> np.ndarray:
    """RMSNorm without a weight, eps 1e-6, over the last axis."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)


def rms(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return unit_rms(x) * gamma


def l2_clip(x: np.ndarray, threshold: float, scale: np.ndarray | None = None) -> np.ndarray:
    """Per head vector (last axis): optional learned scale, then shrink to L2 norm ``threshold`` if longer."""
    if scale is not None:
        x = x * scale
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(1.0, threshold / (norm + 1e-6))


def log_scale(positions: np.ndarray, floor: int, alpha: float) -> np.ndarray:
    """tau = 1 + alpha * ln(max(1, (pos + 1) / floor)) for the global layers' q and r."""
    return 1.0 + alpha * np.log(np.maximum(1.0, (np.asarray(positions, dtype=np.float64) + 1.0) / floor))


def relative_bias(r: np.ndarray, table: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """bias[h, j] = sum_c r[h, c] * table[d_j, c] for 0 <= d_j < len(table), else 0.  table: [extent, d_rel]."""
    extent = table.shape[0]
    inside = (distance >= 0) & (distance < extent)
    profile = r @ table.T  # [heads, extent]
    return np.where(inside[None, :], profile[:, np.clip(distance, 0, extent - 1)], 0.0)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray, sink: np.ndarray, visible: np.ndarray
) -> np.ndarray:
    """One query position. q [heads, d]; k, v [keys, kv_heads, d]; bias [heads, keys]; sink [heads] logits.

    out_h = sum_j e^{s_hj} v_j / (e^{t_h} + sum_j e^{s_hj}),  s_hj = <q_h, k_j>/sqrt(d) + bias_hj over visible keys.
    """
    heads, d = q.shape
    group = heads // k.shape[1]
    out = np.zeros_like(q)
    for h in range(heads):
        s = k[:, h // group, :] @ q[h] / np.sqrt(d) + bias[h]
        s = np.where(visible, s, -np.inf)
        top = max(np.max(s), sink[h])
        e = np.exp(s - top)
        out[h] = e @ v[:, h // group, :] / (np.exp(sink[h] - top) + e.sum())
    return out


def short_conv(x: np.ndarray, history: np.ndarray, taps: np.ndarray, dynamic: np.ndarray | None = None,
               head_width: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Causal depthwise convolution with the input added back and a fixed scale; returns (y, new history).

    x [T, C]; history [K-1, C] (oldest first); taps [C, K] with tap K-1 the current token. ``dynamic`` [T, heads*K]
    adds per-token taps shared by the ``head_width`` channels of a head (row = head * K + lag).
    y_t = (x_t + sum_l (taps[:, K-1-l] + dyn_t[head, l]) x_{t-l}) / sqrt((2 if dynamic else 1) K + 1)
    """
    t_len, channels = x.shape
    k = taps.shape[1]
    full = np.concatenate((history, x), axis=0)
    y = np.empty_like(x)
    for t in range(t_len):
        acc = x[t].copy()
        for lag in range(k):
            past = full[t + k - 1 - lag]
            w = taps[:, k - 1 - lag].copy()
            if dynamic is not None:
                w += np.repeat(dynamic[t].reshape(-1, k)[:, lag], head_width)
            acc += w * past
        y[t] = acc / np.sqrt((2 if dynamic is not None else 1) * k + 1)
    return y, full[-(k - 1):].copy()


def mix(sources: list[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Depth attention: each query attends over the unit-RMS-normalized sources (values normalized too).

    sources: list of [T, H]; queries [Q, H] -> [Q, T, H].
    """
    s = unit_rms(np.stack(sources, axis=0))  # [S, T, H]
    h = s.shape[-1]
    logits = np.einsum("qh,sth->qts", queries, s) / np.sqrt(h)
    a = np.exp(logits - logits.max(axis=-1, keepdims=True))
    a /= a.sum(axis=-1, keepdims=True)
    return np.einsum("qts,sth->qth", a, s)


def control(current: np.ndarray, message: np.ndarray, down: np.ndarray, gain: np.ndarray, forget: np.ndarray) -> np.ndarray:
    """Loop message controller: (1 - tanh(F h)) * c + tanh(G h) * m, h = D [u(c); u(m)].  D [R, 2H], G, F [H, R]."""
    hidden = np.concatenate((unit_rms(current), unit_rms(message)), axis=-1) @ down.T
    return (1.0 - np.tanh(hidden @ forget.T)) * current + np.tanh(hidden @ gain.T) * message


def route(z: np.ndarray, router: np.ndarray, bias: np.ndarray, top_k: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid scores, bias-steered top-k (lower id wins ties), weights normalized over the chosen and scaled.

    z [T, H] -> (ids [T, k], weights [T, k]).
    """
    scores = 1.0 / (1.0 + np.exp(-(z @ router.T)))
    steered = scores + bias
    ids = np.argsort(-steered, axis=-1, kind="stable")[:, :top_k]
    chosen = np.take_along_axis(scores, ids, axis=-1)
    return ids, scale * chosen / (chosen.sum(axis=-1, keepdims=True) + 1e-20)


def situ(gate: np.ndarray, up: np.ndarray, gate_cap: float, up_cap: float) -> np.ndarray:
    """Softcapped SwiGLU: cap_g tanh(g / cap_g) * sigmoid(g) * cap_u tanh(u / cap_u)."""
    return gate_cap * np.tanh(gate / gate_cap) / (1.0 + np.exp(-gate)) * up_cap * np.tanh(up / up_cap)


def moe(z: np.ndarray, layer: dict, config: dict) -> np.ndarray:
    """Routed experts; gate_up [E, 2I, H] (gate rows first), down [E, H, I], optional per-expert multipliers.

    Each chosen expert is read once (``layer["gate_up"][e]``), so a bank may decode experts lazily.
    """
    ids, weights = route(z, layer["router"], layer["router_bias"], config["top_k"], config["route_scale"])
    inter = config["inter"]
    gu_scale, down_scale = layer.get("gate_up_scale"), layer.get("down_scale")
    out = np.zeros_like(z)
    for e in np.unique(ids):
        rows, slots = np.nonzero(ids == e)
        gu = z[rows] @ np.asarray(layer["gate_up"][e], dtype=np.float64).T
        if gu_scale is not None:
            gu = gu * gu_scale[e]
        h = situ(gu[:, :inter], gu[:, inter:], config["gate_cap"], config["up_cap"])
        y = h @ np.asarray(layer["down"][e], dtype=np.float64).T
        if down_scale is not None:
            y = y * down_scale[e]
        out[rows] += weights[rows, slots][:, None] * y
    return out


# --------------------------------------------------------------------------- model


@dataclass
class VisitState:
    """What one layer visit keeps across chunks: its K/V rows with positions (a local visit keeps only the rows a
    later query can still see), and four convolution histories."""

    positions: list = field(default_factory=list)
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    conv: dict = field(default_factory=dict)  # "k", "v", "attn", "mlp" -> [K-1, C]


@dataclass
class Model:
    config: dict
    weights: dict  # "embed", "embed_norm", "head", "output_norm", "output_query", "loop_query", "loop_kv_query", "layers"

    def physical(self, visit: int) -> int:
        c = self.config
        start, core = c["core_start"], c["core_end"] - c["core_start"]
        if visit < start:
            return visit
        if visit < start + c["passes"] * core:
            return start + (visit - start) % core
        return visit - (c["passes"] - 1) * core

    @property
    def visits(self) -> int:
        c = self.config
        return c["layers"] + (c["passes"] - 1) * (c["core_end"] - c["core_start"])

    def new_state(self) -> list[VisitState]:
        return [VisitState() for _ in range(self.visits)]

    def forward(self, tokens: np.ndarray, state: list[VisitState], start: int,
                trace: dict | None = None) -> tuple[np.ndarray, dict]:
        """Tokens at positions start.. -> (logits [T, unpadded vocab], taps {visit: cur at the visit's input}).

        ``trace``, when given, receives llama.cpp's callback names per visit v: ``attn_out-v`` (after the output
        convolution), ``ffn_inp-v`` and ``l_out-v``, and ``result_norm``; each is [T, H].
        """
        c, w = self.config, self.weights
        tokens = np.asarray(tokens)
        positions = np.arange(start, start + len(tokens))
        cur = rms(w["embed"][tokens], w["embed_norm"])
        blocks: list[np.ndarray] = []
        prelude_blocks: list[np.ndarray] = []
        prelude_cur = bank_last = None
        bank: list[np.ndarray] = []
        taps = {}
        core = c["core_end"] - c["core_start"]
        for visit in range(self.visits):
            p = self.physical(visit)
            layer = w["layers"][p]
            in_core = c["core_start"] <= p < c["core_end"]
            k_pass = (visit - c["core_start"]) // core if in_core else 0
            if in_core and p == c["core_start"]:
                if k_pass == 0:
                    prelude_blocks, prelude_cur = list(blocks), cur
                else:
                    bank, bank_last = blocks[len(prelude_blocks) + 1:], cur
                    blocks, cur = list(prelude_blocks), prelude_cur
            messages = None
            if in_core and k_pass > 0:
                j = p - c["core_start"]
                m_attn = mix(bank + [bank_last], w["loop_query"][2 * j: 2 * j + 2])
                m_kv = mix(bank + [bank_last], w["loop_kv_query"][2 * j: 2 * j + 2])
                messages = {"attn": m_attn[0], "mlp": m_attn[1], "k": m_kv[0], "v": m_kv[1]}
            taps[visit] = cur
            cur, blocks = self._visit(visit, p, layer, cur, blocks, messages, positions, state[visit], trace)
            if trace is not None:
                trace[f"l_out-{visit}"] = cur
        x = rms(mix(blocks + [cur], w["output_query"][None, :])[0], w["output_norm"])
        cap = c["final_softcap"]
        logits = cap * np.tanh((x @ w["head"].T) / cap)
        taps[self.visits] = x
        if trace is not None:
            trace["result_norm"] = x
        return logits[:, : c["unpadded_vocab"]], taps

    def _visit(self, visit, p, layer, cur, blocks, messages, positions, st: VisitState, trace=None):
        c = self.config
        local = p in c["local_layers"]
        if p == 0:
            xq = xk = xv = cur
        else:
            m = mix(blocks + [cur], layer["res_query"].reshape(-1, c["hidden"]))
            xq, xk, xv = (m[0], m[0], m[0]) if local else (m[0], m[1], m[2])
        if messages is not None:
            xq = control(xq, messages["attn"], *layer["loop"]["attn"])
            xk = control(xk, messages["k"], *layer["loop"]["k"])
            xv = control(xv, messages["v"], *layer["loop"]["v"])
        if local:
            xq, xk, xv = (rms(x, layer["attn_norm"]) for x in (xq, xk, xv))
        else:
            xq, xk, xv = rms(xq, layer["q_norm"]), rms(xk, layer["k_norm"]), rms(xv, layer["v_norm"])

        heads, kv, d, rel = c["heads"], c["kv_heads"], c["head_dim"], c["d_rel"]
        t_len = len(positions)
        hist = st.conv or {}
        zeros = lambda ch: np.zeros((c["conv"] - 1, ch))
        k_dyn = None if local else xk @ layer["k_conv_proj"].T
        v_dyn = None if local else xv @ layer["v_conv_proj"].T
        k, hist_k = short_conv(xk @ layer["wk"].T, hist.get("k", zeros(kv * d)), layer["k_conv"], k_dyn, d)
        v, hist_v = short_conv(xv @ layer["wv"].T, hist.get("v", zeros(kv * d)), layer["v_conv"], v_dyn, d)
        q = l2_clip((xq @ layer["wq"].T).reshape(t_len, heads, d), c["qk_clip"], layer["q_clip"])
        k = l2_clip(k.reshape(t_len, kv, d), c["qk_clip"])
        v = unit_rms(v.reshape(t_len, kv, d))
        r = l2_clip((xq @ layer["wr"].T).reshape(t_len, heads, rel), c["rel_clip"], layer["rel_clip"])
        if not local:
            tau = log_scale(positions, c["log_floor"], c["log_alpha"])[:, None, None]
            q, r = q * tau, r * tau
        sink = xq @ layer["sink"].T
        gate = xq @ layer["gate"].T

        st.positions.extend(positions.tolist())
        st.keys.extend(list(k))
        st.values.extend(list(v))
        key_pos = np.array(st.positions)
        keys, values = np.stack(st.keys), np.stack(st.values)
        out = np.empty((t_len, heads * d))
        for t in range(t_len):
            dist = positions[t] - key_pos
            visible = dist >= 0
            if local:
                visible &= dist < c["window"]
            o = attention(q[t], keys, values, relative_bias(r[t], layer["rel_bias"], dist), sink[t], visible)
            out[t] = o.reshape(-1)
        if local:
            # A ring of the last `window - 1` positions: the next query (positions[-1] + 1) sees a key only at
            # distance < window. Positions are ascending, so the expired keys are a prefix.
            drop = sum(1 for pos in st.positions if positions[-1] + 1 - pos >= c["window"])
            del st.positions[:drop], st.keys[:drop], st.values[:drop]
        a = (out / (1.0 + np.exp(-gate))) @ layer["wo"].T
        a, hist_a = short_conv(a, hist.get("attn", zeros(c["hidden"])), layer["post_conv"])

        if p % c["block"] == 0:
            blocks = blocks + [cur]
            ffn_inp = a
        else:
            ffn_inp = cur + a
        if trace is not None:
            trace[f"attn_out-{visit}"], trace[f"ffn_inp-{visit}"] = a, ffn_inp
        z = mix(blocks + [ffn_inp], layer["ffn_query"][None, :])[0]
        if messages is not None:
            z = control(z, messages["mlp"], *layer["loop"]["mlp"])
        z = rms(z, layer["ffn_norm"])
        y, hist_m = short_conv(moe(z, layer, c), hist.get("mlp", zeros(c["hidden"])), layer["mlp_conv"])
        st.conv = {"k": hist_k, "v": hist_v, "attn": hist_a, "mlp": hist_m}
        return ffn_inp + y, blocks

    @staticmethod
    def toy_config(**changes) -> dict:
        config = {
            "hidden": 32, "layers": 6, "block": 2, "core_start": 2, "core_end": 4, "passes": 3, "rank": 4,
            "heads": 4, "kv_heads": 2, "head_dim": 8, "d_rel": 4, "rel_extent": 5, "window": 3,
            "local_layers": [0, 2, 4], "conv": 4, "experts": 6, "top_k": 2, "inter": 8,
            "route_scale": 8.0, "gate_cap": 4.0, "up_cap": 25.0, "vocab": 24, "unpadded_vocab": 21,
            "final_softcap": 30.0, "qk_clip": 16 * 2**0.5, "rel_clip": 8.0, "log_floor": 4, "log_alpha": 0.1,
        }
        config.update(changes)
        return config

    @classmethod
    def random(cls, config: dict, seed: int = 0, loop_gain: float = 1.0) -> "Model":
        """Random weights at ``config``; ``loop_gain`` 0 makes every controller the identity."""
        rng = np.random.default_rng(seed)
        h, heads, kv, d, rel = config["hidden"], config["heads"], config["kv_heads"], config["head_dim"], config["d_rel"]
        e, i, k, rank = config["experts"], config["inter"], config["conv"], config["rank"]
        g = lambda *shape, s=1.0: rng.standard_normal(shape) * s
        core = config["core_end"] - config["core_start"]
        layers = []
        for p in range(config["layers"]):
            local = p in config["local_layers"]
            layer = {
                "wq": g(heads * d, h, s=h**-0.5), "wk": g(kv * d, h, s=h**-0.5), "wv": g(kv * d, h, s=h**-0.5),
                "wr": g(heads * rel, h, s=h**-0.5), "wo": g(h, heads * d, s=(heads * d) ** -0.5),
                "gate": g(heads * d, h, s=h**-0.5), "sink": g(heads, h, s=h**-0.5),
                "q_clip": 1 + g(heads, d, s=0.1), "rel_clip": 1 + g(heads, rel, s=0.1),
                "rel_bias": g(config["window"] if local else config["rel_extent"], rel, s=0.3),
                "k_conv": g(kv * d, k, s=0.3), "v_conv": g(kv * d, k, s=0.3), "post_conv": g(h, k, s=0.3),
                "mlp_conv": g(h, k, s=0.3), "ffn_query": g(h, s=0.5), "ffn_norm": 1 + g(h, s=0.1),
                "router": g(e, h, s=h**-0.5), "router_bias": g(e, s=0.1),
                "gate_up": g(e, 2 * i, h, s=h**-0.5), "down": g(e, h, i, s=i**-0.5),
                "gate_up_scale": np.exp(g(e, s=0.2)), "down_scale": np.exp(g(e, s=0.2)),
            }
            if local:
                layer["attn_norm"] = 1 + g(h, s=0.1)
            else:
                layer |= {"q_norm": 1 + g(h, s=0.1), "k_norm": 1 + g(h, s=0.1), "v_norm": 1 + g(h, s=0.1),
                          "k_conv_proj": g(kv * k, h, s=0.1 * h**-0.5), "v_conv_proj": g(kv * k, h, s=0.1 * h**-0.5)}
            if p > 0:
                layer["res_query"] = g(1 if local else 3, h, s=0.5)
            if config["core_start"] <= p < config["core_end"]:
                layer["loop"] = {
                    m: (g(rank, 2 * h, s=(2 * h) ** -0.5), loop_gain * g(h, rank, s=rank**-0.5),
                        loop_gain * g(h, rank, s=rank**-0.5))
                    for m in ("attn", "k", "v", "mlp")
                }
            layers.append(layer)
        weights = {
            "embed": g(config["vocab"], h), "embed_norm": 1 + g(h, s=0.1), "head": g(config["vocab"], h, s=h**-0.5),
            "output_norm": 1 + g(h, s=0.1), "output_query": g(h, s=0.5),
            "loop_query": g(2 * core, h, s=0.5), "loop_kv_query": g(2 * core, h, s=0.5), "layers": layers,
        }
        return cls(config, weights)


# --------------------------------------------------------------------------- golden vectors for the C++ oracle


def golden(seed: int = 7) -> dict:
    """Small deterministic op vectors: inputs and FP64 outputs, as nested lists."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    x = rng.standard_normal((3, 8)) * 5
    scale = 1 + rng.standard_normal(8) * 0.2
    out["l2_clip"] = {"x": x.tolist(), "scale": scale.tolist(), "threshold": 6.0, "y": l2_clip(x, 6.0, scale).tolist()}
    pos = [0, 7, 8190, 8191, 100000, 1048575]
    out["log_scale"] = {"pos": pos, "tau": log_scale(np.array(pos), 8192, 0.1).tolist()}
    xs = rng.standard_normal((5, 6))
    hist = rng.standard_normal((3, 6))
    taps = rng.standard_normal((6, 4)) * 0.3
    dyn = rng.standard_normal((5, 2 * 4)) * 0.2
    y, h2 = short_conv(xs, hist, taps)
    yd, _ = short_conv(xs, hist, taps, dyn, head_width=3)
    out["short_conv"] = {"x": xs.tolist(), "history": hist.tolist(), "taps": taps.tolist(), "dynamic": dyn.tolist(),
                         "head_width": 3, "y": y.tolist(), "y_dynamic": yd.tolist(), "new_history": h2.tolist()}
    sources = [rng.standard_normal((2, 6)) * s for s in (1.0, 3.0, 0.5)]
    queries = rng.standard_normal((3, 6))
    out["mix"] = {"sources": [s.tolist() for s in sources], "queries": queries.tolist(), "y": mix(sources, queries).tolist()}
    cur, msg = rng.standard_normal((2, 6)), rng.standard_normal((2, 6))
    down, gain, forget = rng.standard_normal((3, 12)) * 0.3, rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
    out["control"] = {"current": cur.tolist(), "message": msg.tolist(), "down": down.tolist(), "gain": gain.tolist(),
                      "forget": forget.tolist(), "y": control(cur, msg, down, gain, forget).tolist()}
    z, router, rbias = rng.standard_normal((2, 6)), rng.standard_normal((8, 6)), rng.standard_normal(8) * 0.1
    ids, weights = route(z, router, rbias, 3, 8.0)
    out["route"] = {"z": z.tolist(), "router": router.tolist(), "bias": rbias.tolist(), "top_k": 3, "scale": 8.0,
                    "ids": ids.tolist(), "weights": weights.tolist()}
    g_, u_ = rng.standard_normal(6) * 10, rng.standard_normal(6) * 40
    out["situ"] = {"gate": g_.tolist(), "up": u_.tolist(), "y": situ(g_, u_, 4.0, 25.0).tolist()}
    # attention: 3 query heads over 2 kv heads? use 4 heads, 2 kv heads, 5 keys, a window that hides one key
    q = rng.standard_normal((4, 8)); k = rng.standard_normal((5, 2, 8)); v = rng.standard_normal((5, 2, 8))
    r = rng.standard_normal((4, 3)); table = rng.standard_normal((4, 3)); sink = rng.standard_normal(4)
    dist = np.array([4, 3, 2, 1, 0]); visible = dist < 4
    bias = relative_bias(r, table, dist)
    out["attention"] = {"q": q.tolist(), "k": k.tolist(), "v": v.tolist(), "r": r.tolist(), "table": table.tolist(),
                        "sink": sink.tolist(), "distance": dist.tolist(), "window": 4,
                        "bias": bias.tolist(), "y": attention(q, k, v, bias, sink, visible).tolist()}
    blocks = rng.integers(0, 256, size=(2, NVFP4_BLOCK_BYTES), dtype=np.uint8)
    blocks[:, :4] = rng.integers(0, 0x7E, size=(2, 4))  # finite scales
    blocks[0, 0], blocks[1, 3] = 0x00, 0x7F  # both zero-scale encodings
    out["nvfp4"] = {"blocks": blocks.reshape(-1).tolist(), "values": decode_nvfp4(blocks).tolist()}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--emit-golden", type=Path, required=True)
    args = parser.parse_args(argv)
    args.emit_golden.parent.mkdir(parents=True, exist_ok=True)
    args.emit_golden.write_text(json.dumps(golden(), indent=None, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
