# Adapted from Infernix a3edb450 tools/flash_next/reference.py (Apache-2.0).
# Modified for NInfer-3090: reads the converted .ninfer and its n-gram volume through the exact
# GGML decoders (tools/artifact/codecs/ggml_blocks.py) instead of NVIDIA's safetensors; GGML
# projections and experts take the canonical A8 activation cast; the GGUF's layouts (stored norm
# multipliers, BF16 gamma attention norms, tiled GDN value heads, the stored GDN decay, the [K, C]
# GDN convolution) are read as stored.
"""Independent FP64 reference forward of Qwen3.8-Flash-Next (Qwen4Exp) on a converted artifact.

    python -m tools.flash_next.reference --artifact OUT.ninfer --tokens 248045,9707,11 --out ref.npz

The model-level oracle of NInfer's qwen4_exp forward (AGENTS.md "Verification"). It is written from
the upstream ``qwen4_exp`` mathematics (transformers ``modular_qwen4_exp.py``, through Infernix's
reference) and reads the artifact's stored values directly, never the engine:

- every weight is decoded exactly (``decode_blocks``, bit-exact against ggml b11316) or widened
  exactly from BF16/FP32, and every product is formed in binary64;
- the activations of GGML-format weights (dense projections, the LM head and the routed experts)
  take the canonical A8 cast first: per 32 values ``d = fl32(amax / 127)`` and
  ``q = clamp(rne(fl32(x * fl32(127 / amax))), +-127)``, the cast every GGML route applies
  (include/ninfer/ops/linear.h, offloaded_sparse_moe.h), so differences are kernel arithmetic, not a
  different quantization;
- the PLE n-gram rows are the volume's IQ4_NL rows, decoded exactly and rounded to BF16.

``--bf16-boundaries`` (default on) rounds the activations the engine stores in BF16: residual
streams, every block input and output, projection outputs. The K/V cache is exact here; the engine
stores INT8-G64 K/V, so attention outputs differ by that quantization. QSA selection is evaluated
exactly; for prompts up to ``indexer_budget + compress_ratio - 1`` tokens every token is selected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.artifact.codecs.ggml_blocks import decode_blocks, unpack_expert_records
from tools.artifact.formats import GgmlBlockFormat, get_format
from tools.artifact.layouts import ggml_blocks_geometry, ggml_expert_record_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.flash_next.ngram import NgramConfig, row_ids

F64 = torch.float64
F32 = torch.float32


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 (nearest even) and widen back to binary64."""
    return x.to(F32).to(torch.bfloat16).to(F64)


def a8(x: torch.Tensor) -> torch.Tensor:
    """The canonical A8 activation cast along the last axis (groups of 32), dequantized in FP64."""
    shape = x.shape
    v = x.to(F32).reshape(*shape[:-1], -1, 32)
    a = v.abs().amax(-1, keepdim=True)
    safe = torch.where(a > 0, a, torch.ones_like(a))
    d = torch.where(a > 0, safe / torch.tensor(127.0, dtype=F32), torch.zeros_like(a))
    inv = torch.tensor(127.0, dtype=F32) / safe
    q = torch.where(a > 0, torch.round(v * inv).clamp(-127, 127), torch.zeros_like(v))
    return (q.to(F64) * d.to(F64)).reshape(shape)


class Weights:
    """The artifact's parameters by logical name, decoded exactly on demand."""

    def __init__(self, path: Path, volume: Path | None = None):
        self.artifact = Artifact(path)
        self.directory = self.artifact.directory
        self.config = self.directory.components["text"]["config"]
        self.volume = Path(volume) if volume is not None else Path(str(path) + ".ngram")

    def _parts(self, name: str):
        return binding_parts(self.directory.bindings[name], self.artifact.by_id, name)

    def get(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        """A whole parameter in FP64 (direct words widened, GGML blocks decoded)."""
        chunks = []
        for object_id, begin, end in self._parts(name):
            obj = self.artifact.object(object_id)
            spec = get_format(obj.format)
            if isinstance(spec, GgmlBlockFormat):
                k = obj.shape[-1]
                g = ggml_blocks_geometry(obj.format, obj.shape)
                if begin % k or end % k:
                    raise ValueError(f"{name}: a GGML binding must cover whole rows")
                raw = self.artifact.read_range(obj.offset + begin // k * g.row_bytes, (end - begin) // k * g.row_bytes)
                chunks.append(torch.from_numpy(decode_blocks(obj.format, raw).astype(np.float64)))
            else:
                width = spec.word_bytes
                raw = self.artifact.read_range(obj.offset + begin * width, (end - begin) * width)
                if obj.format == "bf16":
                    words = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
                    chunks.append(torch.from_numpy(words.view(np.float32).astype(np.float64)))
                elif obj.format == "fp32":
                    chunks.append(torch.from_numpy(np.frombuffer(raw, "<f4").astype(np.float64)))
                else:
                    raise ValueError(f"{name}: unexpected direct format {obj.format}")
        return torch.cat(chunks).reshape(shape)

    def rows(self, name: str, first: int, last: int, k: int) -> torch.Tensor:
        """Rows [first, last) of a one-object GGML matrix with K = k."""
        ((object_id, begin, _),) = self._parts(name)
        obj = self.artifact.object(object_id)
        g = ggml_blocks_geometry(obj.format, obj.shape)
        row0 = begin // k + first
        raw = self.artifact.read_range(obj.offset + row0 * g.row_bytes, (last - first) * g.row_bytes)
        return torch.from_numpy(decode_blocks(obj.format, raw).astype(np.float64)).reshape(last - first, k)

    def text(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        return self.get("text/" + name, shape)

    def expert(self, layer: int, expert: int):
        """Expert e of a layer's bank: gate [I, H], up [I, H], down [H, I], decoded exactly."""
        ((object_id, _, _),) = self._parts(f"text/layers/{layer}/moe/experts")
        obj = self.artifact.object(object_id)
        g = ggml_expert_record_geometry(obj.format, obj.shape)
        record = self.artifact.read_range(obj.offset + expert * g.record_stride, g.record_stride)
        gate, up, down = unpack_expert_records(record, obj.format, (1, g.hidden, g.intermediate))
        spec = get_format(obj.format)
        decode = lambda part, fmt, rows, k: torch.from_numpy(  # noqa: E731
            decode_blocks(fmt, part).astype(np.float64)).reshape(rows, k)
        return (decode(gate, spec.gate_up, g.intermediate, g.hidden),
                decode(up, spec.gate_up, g.intermediate, g.hidden),
                decode(down, spec.down, g.hidden, g.intermediate))

    def ngram_rows(self, rows: list[int], row_values: int) -> torch.Tensor:
        """The volume's rows (IQ4_NL), decoded exactly: [len(rows), row_values]."""
        table = self.config["ngram_table"]
        out = []
        with self.volume.open("rb") as stream:
            for r in rows:
                block, slot = divmod(r, table["rows_per_block"])
                stream.seek(table["header_bytes"] + block * table["block_bytes"] + slot * table["row_bytes"])
                out.append(decode_blocks(table["format"], stream.read(table["row_bytes"])))
        return torch.from_numpy(np.stack(out).astype(np.float64)).reshape(len(rows), row_values)


# ---------------------------------------------------------------------------- primitives


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * weight, optionally per group of ``group``; weight is the
    full multiplier (the GGUF's stored 1 + gamma, or 1 + gamma formed from a stored gamma)."""
    shape = x.shape
    if group is not None:
        x = x.reshape(*shape[:-1], -1, group)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y.reshape(shape) * weight


def silu(x):
    return x * torch.sigmoid(x)


def softplus(x):
    return torch.nn.functional.softplus(x)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float, rotary: int) -> torch.Tensor:
    """Text-token MRoPE: every section uses the same position, so it is plain half-split RoPE on
    the first ``rotary`` dimensions. x: [T, H, D]."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=F64) / rotary))
    freqs = positions.to(F64)[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    xr, xp = x[..., :rotary], x[..., rotary:]
    half = rotary // 2
    rotated = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
    return torch.cat((xr * cos + rotated * sin, xp), dim=-1)


# ---------------------------------------------------------------------------- model


class Reference:
    def __init__(self, artifact: Path, *, volume: Path | None = None, bf16_boundaries: bool = True):
        self.w = Weights(artifact, volume)
        self.c = self.w.config
        self.eps = float(self.c["rms_norm_eps"])
        self.hc = int(self.c["hc_count"])
        self.rank = int(self.c["hc_lowrank"])
        self.H = int(self.c["hidden_size"])
        self.round = bf16 if bf16_boundaries else (lambda x: x)
        rope_parameters = self.c["rope_parameters"]
        self.theta = float(rope_parameters["rope_theta"])
        self.rotary = int(self.c["head_dim"] * rope_parameters["partial_rotary_factor"])
        self.layer_types = ["full" if kind == "full_attention" else "linear" for kind in self.c["layer_types"]]
        self.eos = int(self.c["eos_token_id"])
        self.ple_layers = list(self.c["ple_layer_ids"])
        self._cache: dict[str, torch.Tensor] = {}

    def p(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if name not in self._cache:
            self._cache[name] = self.w.text(name, shape)
        return self._cache[name]

    def drop_cache(self) -> None:
        self._cache.clear()

    def ggml_linear(self, x: torch.Tensor, name: str, shape: tuple[int, int]) -> torch.Tensor:
        """A GGML projection: the A8 cast of x, then the exact product, rounded at the BF16 output."""
        return self.round(a8(x) @ self.p(name, shape).T)

    def bf16_linear(self, x: torch.Tensor, name: str, shape: tuple[int, int]) -> torch.Tensor:
        return self.round(x @ self.p(name, shape).T)

    # -- hyper-connections
    def hc_mix(self, prefix: str, R: torch.Tensor, combine: bool = True):
        W = self.hc * self.H
        Rn = self.round(rmsnorm(R, self.p(prefix + "norm", (W,)), self.eps, group=self.H))
        z = Rn @ self.p(prefix + "down", (self.rank, W)).T
        m = silu(z / self.hc)
        u = torch.sigmoid(m @ self.p(prefix + "up", (W, self.rank)).T)
        T = R.shape[0]
        x = (u.reshape(T, self.hc, self.H) * Rn.reshape(T, self.hc, self.H)).mean(dim=1)
        if not combine:
            return self.round(x)
        inj = 2.0 * torch.sigmoid((Rn @ self.p(prefix + "inject", (self.hc, W)).T) / self.hc)
        return self.round(x), inj

    def inject(self, R: torch.Tensor, y: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        T = R.shape[0]
        out = R.reshape(T, self.hc, self.H) + y[:, None, :] * inj[:, :, None]
        return self.round(out.reshape(T, self.hc * self.H))

    # -- GDN
    def gdn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c = self.c
        p = f"layers/{layer}/gdn/"
        nk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        T, H = x.shape[0], self.H
        C = 2 * nk * dk + nv * dv
        qkv = self.ggml_linear(x, p + "qkv", (C, H))
        z = self.ggml_linear(x, p + "z", (nv * dv, H))
        ab = self.bf16_linear(x, p + "a_projection", (nv, H)), self.bf16_linear(x, p + "b_projection", (nv, H))
        conv = self.p(p + "convolution", (c["linear_conv_kernel_dim"], C))  # [K, C], tap-major
        taps = conv.shape[0]
        padded = torch.cat((torch.zeros(taps - 1, C, dtype=F64), qkv), dim=0)
        mixed = self.round(silu(sum(conv[j][None, :] * padded[j : j + T] for j in range(taps))))
        q, k, v = mixed.split([nk * dk, nk * dk, nv * dv], dim=-1)
        # Tiled value-head order: value head h reads key head h % nk.
        q = q.reshape(T, nk, dk).repeat(1, nv // nk, 1)
        k = k.reshape(T, nk, dk).repeat(1, nv // nk, 1)
        v = v.reshape(T, nv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        a, b = ab
        beta = torch.sigmoid(b)
        g = self.p(p + "a", (nv,)) * softplus(a + self.p(p + "dt_bias", (nv,)))  # stored decay -exp(A_log)
        S = torch.zeros(nv, dk, dv, dtype=F64)
        out = torch.zeros(T, nv, dv, dtype=F64)
        for t in range(T):
            S = S * g[t].exp()[:, None, None]
            kv = (S * k[t][:, :, None]).sum(dim=1)
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = (S * q[t][:, :, None]).sum(dim=1)
        out = self.round(out)
        on = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        gate = torch.sigmoid if c["output_gate_type"] == "sigmoid" else silu
        on = self.round(self.p(p + "norm", (dv,)) * on * gate(z.reshape(T, nv, dv)))
        return self.ggml_linear(on.reshape(T, nv * dv), p + "output", (H, nv * dv))

    # -- QSA
    def qsa(self, layer: int, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        c = self.c
        p = f"layers/{layer}/attention/"
        nh, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        T, H = x.shape[0], self.H
        qg = self.ggml_linear(x, p + "query_gate", (2 * nh * d, H)).reshape(T, nh, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(T, nh * d)
        k = self.ggml_linear(x, p + "key", (nkv * d, H)).reshape(T, nkv, d)
        v = self.ggml_linear(x, p + "value", (nkv * d, H)).reshape(T, nkv, d)
        q = rope(self.round(rmsnorm(q, 1.0 + self.p(p + "query_norm", (d,)), self.eps)), positions, self.theta, self.rotary)
        k = rope(self.round(rmsnorm(k, 1.0 + self.p(p + "key_norm", (d,)), self.eps)), positions, self.theta, self.rotary)
        selected = self.qsa_select(layer, x, positions)
        rep = nh // nkv
        out = torch.zeros(T, nh, d, dtype=F64)
        for t in range(T):
            idx = selected[t]
            kk = k[idx].repeat_interleave(rep, dim=1)  # [n, nh, d]
            vv = v[idx].repeat_interleave(rep, dim=1)
            scores = torch.einsum("hd,nhd->hn", q[t], kk) / math.sqrt(d)
            prob = torch.softmax(scores, dim=-1)
            out[t] = torch.einsum("hn,nhd->hd", prob, vv)
        o = self.round(self.round(out).reshape(T, nh * d) * torch.sigmoid(gate))
        return self.ggml_linear(o, p + "output", (H, nh * d))

    def qsa_select(self, layer: int, x: torch.Tensor, positions: torch.Tensor) -> list[torch.Tensor]:
        """Upstream Qwen4Exp indexer: selected token positions per query (prefix of length T)."""
        c = self.c
        p = f"layers/{layer}/indexer/"
        nh, d, ratio = c["indexer_n_heads"], c["indexer_head_dim"], c["indexer_compress_ratio"]
        budget = c["indexer_budget"] // ratio
        T, H = x.shape[0], self.H
        q = self.bf16_linear(x, p + "query", (nh * d, H)).reshape(T, nh, d)
        raw = self.bf16_linear(x, p + "key", (d, H))  # [T, d], cached in BF16
        q = rope(rmsnorm(q, 1.0 + self.p(p + "query_norm", (d,)), self.eps), positions, self.theta, self.rotary)
        key_norm = 1.0 + self.p(p + "key_norm", (d,))
        out = []
        for t in range(T):
            visible = t + 1
            blocks = visible // ratio
            chosen = []
            if blocks > 0:
                pooled = self.round(raw[: blocks * ratio].reshape(blocks, ratio, d).mean(1))
                pooled = rmsnorm(pooled, key_norm, self.eps)
                starts = positions[torch.arange(blocks) * ratio]
                pooled = rope(pooled[:, None, :], starts, self.theta, self.rotary)[:, 0]
                scores = torch.relu(q[t] @ pooled.T).sum(0) / math.sqrt(d)
                count = min(budget, blocks)
                order = sorted(range(blocks), key=lambda b: (-float(scores[b]), b))[:count]
                for b in sorted(order):
                    chosen.extend(range(b * ratio, b * ratio + ratio))
            chosen.extend(range(blocks * ratio, visible))
            out.append(torch.tensor(chosen, dtype=torch.long))
        return out

    # -- MoE
    def expert(self, layer: int, e: int, x_bf16: torch.Tensor) -> torch.Tensor:
        """One routed expert: A8(x), gate/up rowdots, SiLU(gate) * up in FP32, A8(h), the down
        rowdot rounded to BF16 (offloaded_sparse_moe.h)."""
        wg, wu, wd = self.w.expert(layer, e)
        xq = a8(x_bf16)
        h = (silu(xq @ wg.T) * (xq @ wu.T)).to(F32).to(F64)
        return bf16(a8(h) @ wd.T)

    def moe(self, layer: int, x: torch.Tensor):
        c = self.c
        p = f"layers/{layer}/moe/"
        E, H, I = c["num_experts"], self.H, c["shared_expert_intermediate_size"]
        # FP32 router logits (not rounded); exact ties go to the lower expert id.
        logits = x @ self.p(p + "router", (E, H)).T
        probs = torch.softmax(logits, dim=-1)
        order = torch.sort(probs, dim=-1, descending=True, stable=True).indices
        idx = order[:, : c["num_experts_per_tok"]]
        top = torch.gather(probs, -1, idx)
        top = top / top.sum(-1, keepdim=True)
        T = x.shape[0]
        routed = torch.zeros(T, H, dtype=F64)
        for e in torch.unique(idx).tolist():
            rows, slots = torch.where(idx == e)
            y = self.expert(layer, e, x[rows])
            routed.index_add_(0, rows, y * top[rows, slots][:, None])
        g = self.ggml_linear(x, p + "shared/gate", (I, H))
        u = self.ggml_linear(x, p + "shared/up", (I, H))
        shared = self.ggml_linear(self.round(silu(g) * u), p + "shared/down", (H, I))
        shared = shared * torch.sigmoid(x @ self.p(p + "shared_score", (1, H)).T)
        return self.round(routed + shared), idx

    # -- PLE
    def ple(self, layer: int, R: torch.Tensor, ids: list[int]):
        c = self.c
        p = f"layers/{layer}/ple/"
        spec = NgramConfig(vocab_size=c["vocab_size"], eos_token_id=self.eos, ngram_size=c["ngram_size"],
                           heads_per_ngram=c["heads_per_ngram"], ngram_vocab_size_base=c["ngram_vocab_size_base"],
                           make_ngram_vocab_size_divisible_by=c["make_ngram_vocab_size_divisible_by"],
                           seed=c["seed"])
        history = [self.eos] * (spec.ngram_size - 1) + list(ids)
        rows = row_ids(spec, 0, history, len(ids))
        heads = (spec.ngram_size - 1) * spec.heads_per_ngram
        width = c["ple_embed_dim"] // heads
        T, H, W = R.shape[0], self.H, self.hc * self.H
        e = bf16(self.w.ngram_rows([r for row in rows for r in row], width)).reshape(T, heads * width)
        key = self.bf16_linear(e, p + "key", (W, c["ple_embed_dim"]))
        value = self.bf16_linear(e, p + "value", (H, c["ple_embed_dim"]))
        key = rmsnorm(key, self.p(p + "key_norm", (W,)), self.eps, group=H)
        query = rmsnorm(R, self.p(p + "query_norm", (W,)), self.eps, group=H)
        gate = (key.reshape(T, self.hc, H) * query.reshape(T, self.hc, H)).sum(-1) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate)[:, :, None] * value[:, None, :]).reshape(T, W)
        normed = self.round(rmsnorm(gated, self.p(p + "conv_norm", (W,)), self.eps, group=H))
        kernel = self.p(p + "convolution", (W, c["ple_conv_kernel_size"]))  # [C, K]: channel-major taps
        taps, dilation = kernel.shape[1], c["ngram_size"]
        span = (taps - 1) * dilation
        padded_in = torch.cat((torch.zeros(span, W, dtype=F64), normed), dim=0)
        conv = sum(kernel[:, j][None, :] * padded_in[j * dilation : j * dilation + T] for j in range(taps))
        return self.round(self.round(gated) + silu(conv)), rows

    # -- embedding and head
    def embed(self, ids: list[int]) -> torch.Tensor:
        return torch.stack([bf16(self.w.rows("text/token_embedding", i, i + 1, self.H)[0]) for i in ids])

    def head(self, xf: torch.Tensor) -> torch.Tensor:
        V = self.c["vocab_size"]
        xq = a8(xf)
        chunks = []
        for begin in range(0, V, 16384):
            end = min(V, begin + 16384)
            chunks.append(xq @ self.w.rows("text/output_head", begin, end, self.H).T)
        return torch.cat(chunks, dim=-1)

    # -- full forward
    def forward(self, ids: list[int], *, layers: int | None = None, log=print) -> dict:
        c = self.c
        positions = torch.arange(len(ids))
        R = self.embed(ids).repeat(1, self.hc)
        count = c["num_hidden_layers"] if layers is None else layers
        record = {"layer_residual_rms": [], "layer_residuals": [], "routed": [], "ngram_rows": None}
        start = time.time()
        for layer in range(count):
            if layer + 1 in self.ple_layers:
                out, rows = self.ple(layer, R, ids)
                R = self.round(R + out)
                record["ngram_rows"] = np.array(rows, dtype=np.int64)
            xa, inj = self.hc_mix(f"layers/{layer}/attn_hc/", R)
            y = self.gdn(layer, xa) if self.layer_types[layer] == "linear" else self.qsa(layer, xa, positions)
            R = self.inject(R, y, inj)
            xm, inj = self.hc_mix(f"layers/{layer}/mlp_hc/", R)
            y, idx = self.moe(layer, xm)
            R = self.inject(R, y, inj)
            record["layer_residual_rms"].append(float(R.pow(2).mean().sqrt()))
            record["routed"].append(idx.numpy())
            record["layer_residuals"].append(R.to(F32).numpy())
            self.drop_cache()
            log(f"layer {layer:2d} {self.layer_types[layer]:6s} rms {record['layer_residual_rms'][-1]:.5f} "
                f"({time.time() - start:.0f}s)")
        record["final_residual"] = R.to(F32).numpy()
        if layers is None or layers == c["num_hidden_layers"]:
            xf = self.hc_mix("final_mixer/", R, combine=False)
            record["logits"] = self.head(xf).to(F32).numpy()
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--volume", type=Path, help="the n-gram volume (default: ARTIFACT.ngram)")
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--layers", type=int, help="stop after this many layers (no logits)")
    parser.add_argument("--no-bf16-boundaries", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    ids = [int(v) for v in args.tokens.split(",") if v]
    ref = Reference(args.artifact, volume=args.volume, bf16_boundaries=not args.no_bf16_boundaries)
    record = ref.forward(ids, layers=args.layers)
    arrays = {"tokens": np.array(ids, dtype=np.int64), "final_residual": record["final_residual"]}
    if "logits" in record:
        arrays["logits"] = record["logits"]
        top = np.argsort(-record["logits"], axis=-1)[:, :5]
        print("top-5 next tokens per position:", top.tolist())
    if record["ngram_rows"] is not None:
        arrays["ngram_rows"] = record["ngram_rows"]
    arrays["routed"] = np.stack(record["routed"])
    arrays["layer_residuals"] = np.stack(record["layer_residuals"])
    np.savez(args.out, **arrays)
    Path(str(args.out) + ".json").write_text(json.dumps({"layer_residual_rms": record["layer_residual_rms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
