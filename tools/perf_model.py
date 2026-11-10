#!/usr/bin/env python3
"""A first-principles performance model of NInfer on a card you do not have, for a Mac.

It answers "where do the bytes and the operations go, and how fast should this be?" from nothing but
a model's shape and the recipe that quantises it. Pure Python standard library: no GPU, no CUDA, no
torch, no artifact file.

    python tools/perf_model.py --model 27b --recipe qwen3_8_27b --depth 0 --depth 32768
    python tools/perf_model.py --model 27b --recipe qwen3_8_27b --levers
    python tools/perf_model.py --model 35b-a3b --recipe qwen3_6_35b_a3b --spec-k 3 --acceptance 0.6
    python tools/perf_model.py --model 27b --recipe qwen3_8_27b --format output_head=q4
    python tools/perf_model.py --config /path/to/config.json --recipe qwen3_8_27b --tok-s 49.0

What it models, and how far to trust each part:

  * **Decode bytes** (exact arithmetic). One decoded token streams every dense weight, one embedding
    row, 8 of 256 routed experts per layer, the recurrent GDN state in and out, and the KV cache up to
    the current depth. The weight arithmetic reproduces the artifact-header count of
    `decode_byte_accounting.py` to within 0.02% (27B with the Q8 head that artifact had, and the
    35B-A3B), without the artifact. `test_perf_model.py` pins both.
  * **Decode speed** (a fitted model). `bytes / sustained read bandwidth + layers x a fixed cost per
    layer`. The per-layer cost is the one fitted constant: it is chosen so the 27B (49.0 tok/s) and
    the 35B-A3B (204.5 tok/s) measured on the RTX 3090 both come out within a few percent. Two points
    and one constant make it consistent with those runs, not proven for a third model.
  * **Speculative decode** (a model of a model). `tokens per round = 1 + acceptance x k`, `round time
    = step time at k+1 tokens x the round-cost ratio`. Acceptance is accepted drafts over drafted
    tokens, the definition `docs/performance/methodology.md` publishes. It is an input: no CPU-side
    model predicts it, a real draft model on real text does.
  * **Prefill** (a calibrated ceiling). `2 x active parameters` operations per token against the INT8
    tensor peak, times an efficiency measured per family (dense 27B, MoE 35B-A3B), plus attention.
    Cross-checked against the Engine's own fitted cost table in `prefill_time_model.py`.

What it can NOT tell you: anything that needs the hardware to run. Bank conflicts, occupancy,
register spills, how well memory and compute overlap, launch gaps and L2 behaviour all show up only
as the gap between this model and a measurement. That gap is the point: pass `--tok-s` and the tool
prints it in milliseconds, which says how much a kernel-level win could still be worth before you
rent a card. Whenever the model disagrees with a measurement, the measurement is right.

`--levers` ranks a fixed menu of what-ifs (narrower formats, FP16 GDN state, less kernel overhead) by
predicted gain. Two tools build on this one: `kernel_bandwidth.py` joins an `nsys` capture to these
byte counts to find the kernel families that run well below the card's bandwidth, and
`flash_next_perf_model.py` models Qwen3.8-Flash-Next, which is not bandwidth-bound. The guide is
`docs/maintainer/perf-model.md`.

The recipes below mirror `tools/convert/official_recipes.py`; keep them in step when it changes.
Norm weights and biases are ignored (well under 0.1% of the bytes).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_byte_accounting import (  # noqa: E402  (shared constants: one source for the card)
    ACHIEVABLE_READ_GB_S,
    ADVERTISED_GB_S,
    SPARSE_MOE_EXPERTS,
    SPARSE_MOE_TOP_K,
)

# ---------------------------------------------------------------------------------------------
# Formats. Bits per weight include the scale, from docs/maintainer/tensor-formats.md.
# ---------------------------------------------------------------------------------------------
FORMAT_BITS = {
    "q4": 4.25,  # q4_g64_fp16
    "q5": 5.25,  # q5_g64_fp16
    "q6": 6.25,  # q6_g64_fp16
    "q8": 8.5,  # q8_g32_fp16
    "t2": 2.125,  # t2_g128_fp16
    "nvfp4": 4.5,  # E2M1 plus one E4M3 scale per 16 (the FP32 divisor is per tensor)
    "fp8": 8.0,  # fp8_e4m3fn_row_bf16 (one BF16 per row, negligible)
    "bf16": 16.0,
    "fp32": 32.0,
}

# Bytes per KV element, including the scale. 33/32 reproduces the 10,560 B/token that the 35B-A3B
# roofline run passed to decode_byte_accounting.py (10 attention layers x 2 x 2 heads x 256).
KV_BYTES_PER_ELEMENT = {"bf16": 2.0, "int8": 33 / 32}


@dataclass(frozen=True)
class Hardware:
    name: str
    read_gb_s: float  # sustained read, the ceiling decode streams against
    advertised_gb_s: float
    int8_tops: float  # dense INT8 tensor peak, the prefill matmul ceiling


# Measured on this fork's card: tools/hbm_bandwidth_probe.cu and tools/tensor_core_rate_probe.cu
# (docs/performance/bonsai2-27b-rtx3090.md: "int8 m16n8k32 peak at 316 TOPS (1,695 MHz)").
RTX_3090 = Hardware("RTX 3090", ACHIEVABLE_READ_GB_S, ADVERTISED_GB_S, 316.0)


@dataclass(frozen=True)
class Calibration:
    """Constants fitted to measurements. Each says where it came from; change them with a new run."""

    # Fixed cost per transformer block per decoded step: norms, small kernels, tails of waves.
    # Fitted so the 27B (64 layers, 49.0 tok/s) and the 35B-A3B (40 layers, 204.5 tok/s) plain
    # decodes of docs/performance.md ("ninfer_bench, Engine route") both land within 2%.
    per_layer_ms: float = 0.0425
    # Prefill = steps x a fixed cost per step + matmul ops / (INT8 peak x efficiency) + attention.
    # Both constants, per family, are solved from the 512- and 16,384-token rows of
    # docs/performance.md (ninfer_bench, chunk 4096): 27B 2,057 and 2,893 tok/s, 35B-A3B 5,479 and
    # 7,749. The 4,096-token rows (3,143 and 8,468) are held out and come out within 1%. The 27B fixed
    # cost, 103 ms, agrees with the 107.8 ms the Engine fitted independently (context_cost_defaults).
    # The MoE's efficiency is low because grouped expert GEMMs of a few hundred rows each cannot fill
    # the tensor cores.
    prefill_eff_dense: float = 0.544
    prefill_eff_moe: float = 0.149
    prefill_step_ms_dense: float = 103.3
    prefill_step_ms_moe: float = 40.1
    # Attention in prefill, from the Engine's fitted 27B pair cost (4.54 ns/pair, context_cost_defaults).
    prefill_attention_eff: float = 0.274
    # Speculative round time over the plain step at the same bytes: 22.3 ms step, 1.15x
    # (docs/performance.md, "MTP round/plain step ratio"). Measured on the 27B only. A 220 W-capped
    # host measured 1.22 (tests/fixtures/perf_model/): the drafter's q8 weights and the recurrence
    # records are not in the byte model, and cost more when the SM clock is low.
    spec_round_ratio: float = 1.15


CALIBRATION = Calibration()

# ---------------------------------------------------------------------------------------------
# Geometry: the fields of a Qwen3.5 text config (src/models/qwen3_5/config.cpp).
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    hidden: int
    layer_full: tuple[bool, ...]  # True = gated full attention, False = GDN
    q_heads: int
    kv_heads: int
    head_dim: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_key_dim: int
    gdn_value_dim: int
    conv_width: int
    dense_ffn: int  # 0 for an MoE
    experts: int  # 0 for a dense model
    top_k: int
    moe_ffn: int
    shared_ffn: int
    vocab: int

    @property
    def layers(self) -> int:
        return len(self.layer_full)

    @property
    def is_moe(self) -> bool:
        return self.experts > 0

    @property
    def full_layers(self) -> int:
        return sum(self.layer_full)

    @classmethod
    def from_config(cls, config: dict) -> "Geometry":
        """Reads a HuggingFace-style config.json (the text part, possibly under `text_config`)."""
        cfg = config.get("text_config", config)
        layers = int(cfg["num_hidden_layers"])
        if "layer_types" in cfg:
            labels = list(cfg["layer_types"])
            unknown = sorted({str(t) for t in labels} - {"full_attention", "linear_attention"})
            if unknown:
                # src/models/qwen3_5/config.cpp: "unknown text layer_type".
                raise SystemExit(f"unknown layer_types {unknown}; a Qwen3.5 layer is full_attention "
                                 f"or linear_attention")
            full = tuple(t == "full_attention" for t in labels)
        else:
            interval = int(cfg.get("full_attention_interval", 4))
            full = tuple(i % interval == interval - 1 for i in range(layers))
        if len(full) != layers:
            raise SystemExit("layer_types does not have num_hidden_layers entries")

        # Each mixer family's fields are required only when a layer uses it, as in config.cpp: a
        # model with no GDN layers need not carry linear_* keys, nor an all-GDN one the attention keys.
        def family(present: bool, *keys: str) -> list[int]:
            if present:
                missing = [k for k in keys if k not in cfg]
                if missing:
                    raise SystemExit(f"config has layers that need {missing} but does not define them")
                return [int(cfg[k]) for k in keys]
            return [int(cfg.get(k, 0)) for k in keys]

        q_heads, kv_heads, head_dim = family(
            any(full), "num_attention_heads", "num_key_value_heads", "head_dim")
        gdn_key_heads, gdn_value_heads, gdn_key_dim, gdn_value_dim, conv_width = family(
            not all(full), "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim")
        return cls(
            hidden=int(cfg["hidden_size"]),
            layer_full=full,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            gdn_key_heads=gdn_key_heads,
            gdn_value_heads=gdn_value_heads,
            gdn_key_dim=gdn_key_dim,
            gdn_value_dim=gdn_value_dim,
            conv_width=conv_width,
            dense_ffn=int(cfg.get("intermediate_size", 0)) if "num_experts" not in cfg else 0,
            experts=int(cfg.get("num_experts", 0)),
            top_k=int(cfg.get("num_experts_per_tok", 0)),
            moe_ffn=int(cfg.get("moe_intermediate_size", 0)),
            shared_ffn=int(cfg.get("shared_expert_intermediate_size", 0)),
            vocab=int(cfg["vocab_size"]),
        )


# The two official geometries, from docs/maintainer/qwen3_5-model.md "Current official geometries".
PRESETS = {
    "27b": {
        "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "intermediate_size": 17408, "vocab_size": 248320,
    },
    "35b-a3b": {
        "hidden_size": 2048, "num_hidden_layers": 40, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "num_experts": SPARSE_MOE_EXPERTS,
        "num_experts_per_tok": SPARSE_MOE_TOP_K,
        "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        "vocab_size": 248320,
    },
}

# ---------------------------------------------------------------------------------------------
# Weights: every matrix the decode or prefill path touches, by role.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Weight:
    role: str
    layer: int | None
    rows: int  # for experts: rows of the whole bank (experts x per-expert rows)
    cols: int
    fmt: str
    experts: int = 0  # routed bank: selection scales the bytes read
    embedding: bool = False  # a lookup table: one row per token

    @property
    def params(self) -> int:
        return self.rows * self.cols

    @property
    def resident_bytes(self) -> float:
        return self.params * FORMAT_BITS[self.fmt] / 8

    def read_bytes(self, geo: Geometry, tokens: int) -> float:
        """Bytes streamed for one step over `tokens` columns, assuming uniform expert choice."""
        if self.embedding:
            return self.resident_bytes * tokens / geo.vocab
        if self.experts:
            return self.resident_bytes * expert_fraction(geo.experts, geo.top_k, tokens)
        return self.resident_bytes

    def active_params(self, geo: Geometry) -> float:
        """Parameters that multiply each token: the matmul work in prefill."""
        if self.embedding:
            return 0.0
        if self.experts:
            return self.params * geo.top_k / geo.experts
        return float(self.params)


def expert_fraction(experts: int, top_k: int, tokens: int) -> float:
    """Expected share of a routed bank one step touches, if every token picks its `top_k` experts
    independently and uniformly. This is an estimate, not a bound: with no route data it is the
    natural prior. Evidence says which way it errs. Unrelated sequences land on it (the lanes spike
    measured 19.6 distinct experts per layer at two lanes against 19.8 here). Columns of ONE sequence
    overlap more, so it overstates their union (flash-next-plan.md: a verification column brings 47%
    new experts, uniform predicts 98%). Routes that deliberately diverge could exceed it."""
    return 1.0 - (1.0 - top_k / experts) ** tokens


# A recipe maps (role, layer) to a format. Mirrors tools/convert/official_recipes.py.
Recipe = Callable[[str, Optional[int]], str]
ALWAYS_BF16 = {"gdn.a", "gdn.b", "gdn.convolution", "moe.router", "moe.shared_score"}


def _dense_groupwise(embedding: str, head: str, q4_tail: bool) -> Recipe:
    q4_roles = {"attention.query", "attention.key", "gdn.query", "gdn.key", "mlp.gate", "mlp.up"}
    tail_roles = {"attention.output", "gdn.output", "mlp.down"}

    def fmt(role: str, layer: int | None) -> str:
        if role == "embedding":
            return embedding
        if role == "output_head":
            return head
        if role in ALWAYS_BF16:
            return "bf16"
        if role in q4_roles or (q4_tail and role in tail_roles and (layer or 0) >= 36):
            return "q4"
        return "q5"

    return fmt


def _moe_35b(role: str, layer: int | None) -> str:
    if role == "embedding":
        return "q8"
    if role == "output_head":
        return "q6"
    if role in ALWAYS_BF16:
        return "bf16"
    if role == "experts.down":
        return "q6" if layer in (34, 38, 39) else "q5"
    if role in ("experts.gate", "experts.up"):
        return "q4"
    return "q8"


RECIPES: dict[str, Recipe] = {
    "qwen3_6_27b": _dense_groupwise("q6", "q6", q4_tail=False),
    "qwen3_8_27b": _dense_groupwise("q4", "q6", q4_tail=True),
    "qwen3_6_35b_a3b": _moe_35b,
}
# Each official recipe requires one family, as official_recipes.py checks ("requires Qwen3.5 Dense
# mathematics" / "...MoE mathematics"). The default for a model is its family's current recipe.
RECIPE_FAMILY = {"qwen3_6_27b": "dense", "qwen3_8_27b": "dense", "qwen3_6_35b_a3b": "moe"}
DEFAULT_RECIPE = {"dense": "qwen3_8_27b", "moe": "qwen3_6_35b_a3b"}


def build_weights(geo: Geometry, recipe: Recipe, overrides: dict[str, str] | None = None) -> list[Weight]:
    overrides = overrides or {}
    out: list[Weight] = []

    def add(role: str, layer: int | None, rows: int, cols: int, *, experts: int = 0,
            embedding: bool = False) -> None:
        fmt = overrides.get(role) or recipe(role, layer)
        out.append(Weight(role, layer, rows, cols, fmt, experts, embedding))

    h = geo.hidden
    add("embedding", None, geo.vocab, h, embedding=True)
    q_width = geo.q_heads * geo.head_dim
    kv_width = geo.kv_heads * geo.head_dim
    gk = geo.gdn_key_heads * geo.gdn_key_dim
    gv = geo.gdn_value_heads * geo.gdn_value_dim
    for i, full in enumerate(geo.layer_full):
        if full:
            add("attention.query", i, q_width, h)
            add("attention.gate", i, q_width, h)
            add("attention.key", i, kv_width, h)
            add("attention.value", i, kv_width, h)
            add("attention.output", i, h, q_width)
        else:
            add("gdn.query", i, gk, h)
            add("gdn.key", i, gk, h)
            add("gdn.value", i, gv, h)
            add("gdn.z", i, gv, h)
            add("gdn.a", i, geo.gdn_value_heads, h)
            add("gdn.b", i, geo.gdn_value_heads, h)
            add("gdn.convolution", i, 2 * gk + gv, geo.conv_width)
            add("gdn.output", i, h, gv)
        if geo.is_moe:
            add("moe.router", i, geo.experts, h)
            add("moe.shared_score", i, 1, h)
            add("moe.shared.gate", i, geo.shared_ffn, h)
            add("moe.shared.up", i, geo.shared_ffn, h)
            add("moe.shared.down", i, h, geo.shared_ffn)
            add("experts.gate", i, geo.experts * geo.moe_ffn, h, experts=geo.experts)
            add("experts.up", i, geo.experts * geo.moe_ffn, h, experts=geo.experts)
            add("experts.down", i, geo.experts * h, geo.moe_ffn, experts=geo.experts)
        else:
            add("mlp.gate", i, geo.dense_ffn, h)
            add("mlp.up", i, geo.dense_ffn, h)
            add("mlp.down", i, h, geo.dense_ffn)
    add("output_head", None, geo.vocab, h)

    unknown = set(overrides) - {w.role for w in out}
    if unknown:
        raise SystemExit(f"--format names roles this model does not have: {sorted(unknown)}; "
                         f"valid roles: {sorted({w.role for w in out})}")
    return out


# ---------------------------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------------------------


def kv_bytes_per_token(geo: Geometry, kv: str) -> float:
    return geo.full_layers * 2 * geo.kv_heads * geo.head_dim * KV_BYTES_PER_ELEMENT[kv]


def gdn_state_bytes(geo: Geometry, state_bytes: int) -> float:
    """The recurrent state S[Dv, Dk] per value head, per GDN layer, FP32 unless --gdn-state-fp16."""
    gdn_layers = geo.layers - geo.full_layers
    return gdn_layers * geo.gdn_value_heads * geo.gdn_value_dim * geo.gdn_key_dim * state_bytes


def gdn_conv_bytes(geo: Geometry) -> float:
    """The convolution history every GDN layer also keeps: the preceding `conv_width - 1` columns of the
    projected Q/K/V channels, always BF16 (docs/maintainer/replayssm-gdn.md: Q = 2 L_g C_p (W - 1)).
    It is read and written every step like the recurrent matrix, but --gdn-state-fp16 does not shrink it."""
    gdn_layers = geo.layers - geo.full_layers
    channels = 2 * geo.gdn_key_heads * geo.gdn_key_dim + geo.gdn_value_heads * geo.gdn_value_dim
    return gdn_layers * channels * max(geo.conv_width - 1, 0) * 2


@dataclass
class DecodeStep:
    tokens: int
    depth: int
    by_role: dict[str, float]  # bytes per role, weights only
    weights: float
    state_rw: float  # read plus write of the recurrent state, once per step
    kv: float
    layers: int
    hardware: Hardware
    cal: Calibration

    @property
    def total_bytes(self) -> float:
        return self.weights + self.state_rw + self.kv

    @property
    def floor_ms(self) -> float:
        return self.total_bytes / (self.hardware.read_gb_s * 1e9) * 1e3

    @property
    def overhead_ms(self) -> float:
        return self.layers * self.cal.per_layer_ms

    @property
    def step_ms(self) -> float:
        return self.floor_ms + self.overhead_ms

    def ms_for(self, nbytes: float) -> float:
        return nbytes / (self.hardware.read_gb_s * 1e9) * 1e3


def decode_step(geo: Geometry, weights: Sequence[Weight], *, tokens: int = 1, depth: int = 0,
                kv: str = "int8", kv_bpt: float | None = None, gdn_state_bytes_per: int = 4,
                hardware: Hardware = RTX_3090, cal: Calibration = CALIBRATION) -> DecodeStep:
    """One decode round over `tokens` columns with `depth` tokens already in the cache.

    The round first appends its own `tokens` K/V entries, then attends over all `depth + tokens` of
    them, so a speculative verification reads a few more entries than a plain step at the same depth.
    The columns of one round do NOT each re-read the cache: the small-T attention kernel takes them
    together and loads each K/V tile once per layer (a real 4-column MTP3 capture launches attention
    19 times over 16 layers, not 64), so KV traffic is not `tokens * depth`."""
    by_role: dict[str, float] = {}
    for w in weights:
        by_role[w.role] = by_role.get(w.role, 0.0) + w.read_bytes(geo, tokens)
    per_token_kv = kv_bytes_per_token(geo, kv) if kv_bpt is None else kv_bpt
    return DecodeStep(
        tokens=tokens, depth=depth, by_role=by_role, weights=sum(by_role.values()),
        state_rw=2 * (gdn_state_bytes(geo, gdn_state_bytes_per) + gdn_conv_bytes(geo)),
        kv=per_token_kv * (depth + tokens),
        layers=geo.layers, hardware=hardware, cal=cal)


def tokens_per_round(k: int, acceptance: float) -> float:
    """Tokens one speculative round yields: the verified token plus the accepted drafts.
    `acceptance` is accepted drafts / drafted tokens (docs/performance/methodology.md). Holds for the
    Flash-Next MTP table too: 82 / 76 / 65% at 1 / 2 / 3 drafts give its measured 1.82 / 2.51 / 2.94."""
    return 1.0 + acceptance * k


def bytes_by_group(weights: Sequence[Weight], geo: Geometry, tokens: int = 1) -> dict[str, float]:
    """Bytes one step streams, grouped by the kernel family that streams them: the stored weight
    format (`q4`, `q5`, ...), or `moe` for routed experts. Kernel names carry the format, not the
    role, which is why `kernel_bandwidth.py` joins on this grouping."""
    groups: dict[str, float] = {}
    for w in weights:
        group = "moe" if w.experts else w.fmt
        groups[group] = groups.get(group, 0.0) + w.read_bytes(geo, tokens)
    return groups


def speculative(plain_bytes_step: DecodeStep, verify_step: DecodeStep, k: int, acceptance: float,
                cal: Calibration = CALIBRATION) -> dict[str, float]:
    """One verify round of k drafts. `acceptance` is accepted drafts / drafted tokens."""
    tokens_per_round_ = tokens_per_round(k, acceptance)
    round_ms = verify_step.step_ms * cal.spec_round_ratio
    tok_s = tokens_per_round_ / round_ms * 1e3
    plain_tok_s = 1e3 / plain_bytes_step.step_ms
    return {"k": k, "acceptance": acceptance, "tokens_per_round": tokens_per_round_,
            "round_ms": round_ms, "round_over_plain": round_ms / plain_bytes_step.step_ms,
            "tok_s": tok_s, "speedup": tok_s / plain_tok_s}


# ---------------------------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------------------------


def prefill(geo: Geometry, weights: Sequence[Weight], prompt: int, hardware: Hardware = RTX_3090,
            cal: Calibration = CALIBRATION, chunk: int = 4096) -> dict[str, float]:
    """A prompt processed in chunks. The fixed cost is paid once per chunk, so it dominates short
    prompts; the constants were measured at chunk 4096, and other chunk sizes are an extrapolation."""
    active = sum(w.active_params(geo) for w in weights if w.role != "output_head")
    ops_per_token = 2.0 * active
    matmul_ops = ops_per_token * prompt
    pairs = prompt * (prompt + 1) / 2
    attention_ops = pairs * 4 * geo.q_heads * geo.head_dim * geo.full_layers
    eff = cal.prefill_eff_moe if geo.is_moe else cal.prefill_eff_dense
    step_ms = cal.prefill_step_ms_moe if geo.is_moe else cal.prefill_step_ms_dense
    steps = -(-prompt // chunk)
    peak = hardware.int8_tops * 1e12
    fixed_s = steps * step_ms / 1e3
    matmul_s = matmul_ops / (peak * eff)
    attention_s = attention_ops / (peak * cal.prefill_attention_eff)
    total_s = fixed_s + matmul_s + attention_s
    ceiling_s = (matmul_ops + attention_ops) / peak
    return {"prompt": prompt, "chunk": chunk, "steps": steps, "ops_per_token": ops_per_token,
            "fixed_s": fixed_s, "matmul_s": matmul_s, "attention_s": attention_s,
            "total_s": total_s, "tok_s": prompt / total_s, "ceiling_tok_s": prompt / ceiling_s,
            "matmul_efficiency": eff}


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Levers: a fixed menu of what-ifs, ranked
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lever:
    name: str
    kind: str  # "bytes": fewer bytes streamed (changes numerics); "kernel": less time per byte
    roles: tuple[str, ...] = ()  # bytes levers: move these roles to `fmt` where it saves bytes
    fmt: str = "q4"
    gdn_state_fp16: bool = False
    overhead_scale: float | None = None  # kernel levers: multiply the per-layer overhead by this
    evidence: str = ""  # what the repository already knows, with where


_ATTENTION_GDN = ("attention.query", "attention.gate", "attention.key", "attention.value",
                  "attention.output", "gdn.query", "gdn.key", "gdn.value", "gdn.z", "gdn.output")
_EVERY_MATRIX = _ATTENTION_GDN + ("mlp.gate", "mlp.up", "mlp.down", "experts.gate", "experts.up",
                                  "experts.down", "moe.shared.gate", "moe.shared.up",
                                  "moe.shared.down", "output_head")

# Evidence strings cite docs/maintainer/quality-trade-experiments.md (measured 2026-09-12). A lever
# with no evidence is an unmeasured quality cost: the byte saving is arithmetic, the damage is not.
LEVERS = (
    Lever("output head -> q4", "bytes", ("output_head",), "q4",
          evidence="measured: perplexity +0.69%; plain decode +1.4% to +2.9%, C1 MTP3 within noise, C8 +3.2%"),
    Lever("GDN state in FP16", "bytes", gdn_state_fp16=True,
          evidence="measured: perplexity -0.003%; state drift 0.34% worst; C1 within noise, C8 +2.0%"),
    Lever("MLP weights -> q4", "bytes", ("mlp.gate", "mlp.up", "mlp.down"), "q4",
          evidence="quality not measured"),
    Lever("routed experts -> q4", "bytes", ("experts.gate", "experts.up", "experts.down"), "q4",
          evidence="quality not measured"),
    Lever("attention + GDN projections -> q4", "bytes", _ATTENTION_GDN, "q4",
          evidence="quality not measured"),
    Lever("every matrix -> q4", "bytes", _EVERY_MATRIX, "q4",
          evidence="an upper bound for weight quantisation; quality not measured"),
    Lever("kernels: 25% less per-layer overhead", "kernel", overhead_scale=0.75,
          evidence="no quality cost; needs kernel work (launch gaps, wave tails, fusion)"),
    Lever("kernels: 50% less per-layer overhead", "kernel", overhead_scale=0.5,
          evidence="no quality cost; needs kernel work"),
    Lever("kernels: bandwidth floor (no overhead)", "kernel", overhead_scale=0.0,
          evidence="the ceiling for any kernel work at these bytes; not reachable"),
)


def rank_levers(geo: Geometry, recipe: Recipe, overrides: dict[str, str], *, depth: int = 0,
                kv: str = "int8", kv_bpt: float | None = None, gdn_state_bytes_per: int = 4,
                hardware: Hardware = RTX_3090, cal: Calibration = CALIBRATION) -> list[dict]:
    """Predicted plain-decode gain of each lever over the current configuration, best first. A
    bytes lever that would not shrink anything here (the role is absent, or already at or below the
    target) is left out rather than shown as a zero."""
    def tok_s(weights: Sequence[Weight], state: int, calibration: Calibration) -> float:
        return 1e3 / decode_step(geo, weights, depth=depth, kv=kv, kv_bpt=kv_bpt,
                                 gdn_state_bytes_per=state, hardware=hardware,
                                 cal=calibration).step_ms

    base_weights = build_weights(geo, recipe, overrides)
    base = tok_s(base_weights, gdn_state_bytes_per, cal)
    rows = []
    for lever in LEVERS:
        weights, state, calibration = base_weights, gdn_state_bytes_per, cal
        if lever.kind == "kernel":
            calibration = dataclasses.replace(cal, per_layer_ms=cal.per_layer_ms * lever.overhead_scale)
        else:
            if lever.gdn_state_fp16:
                if gdn_state_bytes_per <= 2:
                    continue
                state = 2
            else:
                # A role can mix formats across layers (the 3090 recipe keeps mlp.down at q5 in
                # layers 0-35 and q4 after), so ask whether any layer would shrink.
                shrinks = {w.role for w in base_weights
                           if w.role in lever.roles and w.fmt not in ("bf16", "fp32")
                           and FORMAT_BITS[w.fmt] > FORMAT_BITS[lever.fmt]}
                changes = {role: lever.fmt for role in shrinks}
                if not changes:
                    continue
                weights = build_weights(geo, recipe, {**overrides, **changes})
        value = tok_s(weights, state, calibration)
        rows.append({"name": lever.name, "kind": lever.kind, "tok_s": value,
                     "gain_pct": 100 * (value / base - 1), "evidence": lever.evidence})
    return sorted(rows, key=lambda r: -r["gain_pct"])


def _decode_table(step: DecodeStep, top: int) -> list[dict]:
    rows = sorted(step.by_role.items(), key=lambda kv: -kv[1])
    shown, rest = rows[:top], rows[top:]
    table = [{"role": r, "bytes": b} for r, b in shown]
    if rest:
        table.append({"role": f"({len(rest)} smaller roles)", "bytes": sum(b for _, b in rest)})
    table.append({"role": "gdn state (read + write)", "bytes": step.state_rw})
    if step.kv:
        table.append({"role": f"kv cache @ {step.depth:,} tokens + step", "bytes": step.kv})
    for row in table:
        row["share"] = row["bytes"] / step.total_bytes
        row["ms"] = step.ms_for(row["bytes"])
        # tok/s if this cost no bytes at all: the ceiling on what optimising it can buy.
        row["tok_s_if_free"] = 1e3 / (step.step_ms - row["ms"])
    return table


def _number(text: str, *, minimum: float, inclusive: bool, integer: bool, what: str) -> float:
    try:
        value = int(text) if integer else float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not {'an integer' if integer else 'a number'}") from None
    if not math.isfinite(value) or value < minimum or (value == minimum and not inclusive):
        raise argparse.ArgumentTypeError(f"{text} is not {what}")
    return value


# argparse types shared by every tool built on this one: a zero, negative or non-finite value is
# refused at the command line instead of dividing by zero later or silently changing the what-if.
def positive_float(text: str) -> float:
    return _number(text, minimum=0.0, inclusive=False, integer=False, what="a positive number")


def non_negative_float(text: str) -> float:
    return _number(text, minimum=0.0, inclusive=True, integer=False, what="zero or more")


def positive_int(text: str) -> int:
    return _number(text, minimum=0, inclusive=False, integer=True, what="a positive integer")


def non_negative_int(text: str) -> int:
    return _number(text, minimum=0, inclusive=True, integer=True, what="zero or more")


def probability(text: str) -> float:
    value = _number(text, minimum=0.0, inclusive=True, integer=False, what="a fraction in [0, 1]")
    if value > 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not a fraction in [0, 1]")
    return value


def choose_recipe(geo: Geometry, name: Optional[str]) -> str:
    """The recipe for `geo`: the family's default if none is named, and an error if the one named is
    for the other family. A dense recipe applied to an MoE has no rules for the expert and shared
    tensors, so it would fall through to arbitrary formats and still print a credible-looking report."""
    family = "moe" if geo.is_moe else "dense"
    if name is None:
        return DEFAULT_RECIPE[family]
    if RECIPE_FAMILY[name] != family:
        raise SystemExit(f"recipe {name} is for {RECIPE_FAMILY[name]} models but this model is {family}; "
                         f"use {', '.join(r for r, f in sorted(RECIPE_FAMILY.items()) if f == family)}")
    return name


class Resolved(NamedTuple):
    geo: Geometry
    label: str
    recipe: str
    weights: list
    hardware: Hardware
    overrides: dict


def resolve(args: argparse.Namespace) -> Resolved:
    """The model, recipe, overrides and card an invocation names. Shared by every tool built on this
    one, so `--model`, `--config`, `--format` and the card flags mean the same thing everywhere."""
    if args.config:
        geo = Geometry.from_config(json.loads(Path(args.config).read_text()))
        label = Path(args.config).name
    else:
        geo = Geometry.from_config(PRESETS[args.model])
        label = args.model
    recipe = choose_recipe(geo, args.recipe)
    overrides = dict(parse_format(item) for item in args.format)
    weights = build_weights(geo, RECIPES[recipe], overrides)
    custom = args.bandwidth_gb_s is not None or args.int8_tops is not None
    hardware = Hardware(
        f"custom card (rates overridden; fitted constants are the {RTX_3090.name}'s)" if custom
        else RTX_3090.name,
        RTX_3090.read_gb_s if args.bandwidth_gb_s is None else args.bandwidth_gb_s,
        RTX_3090.advertised_gb_s,
        RTX_3090.int8_tops if args.int8_tops is None else args.int8_tops)
    return Resolved(geo, label, recipe, weights, hardware, overrides)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags that choose a model, a recipe, a KV format and a card; see `resolve`."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", choices=sorted(PRESETS), default="27b")
    source.add_argument("--config", help="a model's config.json instead of a preset")
    parser.add_argument("--recipe", choices=sorted(RECIPES),
                        help="default: qwen3_8_27b for a dense model, qwen3_6_35b_a3b for an MoE")
    parser.add_argument("--format", action="append", default=[], metavar="ROLE=FORMAT",
                        help="what-if: store ROLE (e.g. output_head, mlp.down) in FORMAT "
                             f"({', '.join(sorted(FORMAT_BITS))}); repeatable")
    parser.add_argument("--kv", choices=sorted(KV_BYTES_PER_ELEMENT), default="int8")
    parser.add_argument("--kv-bytes-per-token", type=non_negative_float,
                        help="exact value from ninfer_bench; overrides --kv (needed for rk4v4)")
    parser.add_argument("--gdn-state-fp16", action="store_true")
    parser.add_argument("--bandwidth-gb-s", type=positive_float,
                        help="sustained read rate of another card")
    parser.add_argument("--int8-tops", type=positive_float, help="INT8 tensor peak of another card")


def build_report(args: argparse.Namespace) -> dict:
    geo, label, recipe, weights, hardware, overrides = resolve(args)
    state_bytes = 2 if args.gdn_state_fp16 else 4
    depths = [0] if args.depth is None else args.depth

    report: dict = {
        "model": label, "recipe": recipe, "overrides": overrides, "hardware": hardware.name,
        "read_gb_s": hardware.read_gb_s, "int8_tops": hardware.int8_tops,
        "layers": geo.layers, "moe": geo.is_moe,
        "resident_gb": sum(w.resident_bytes for w in weights) / 1e9,
        "decode": [],
    }
    for depth in depths:
        step = decode_step(geo, weights, depth=depth, kv=args.kv, kv_bpt=args.kv_bytes_per_token,
                           gdn_state_bytes_per=state_bytes, hardware=hardware)
        entry = {"depth": depth, "table": _decode_table(step, args.top),
                 "total_gb": step.total_bytes / 1e9, "floor_ms": step.floor_ms,
                 "floor_tok_s": 1e3 / step.floor_ms, "overhead_ms": step.overhead_ms,
                 "step_ms": step.step_ms, "tok_s": 1e3 / step.step_ms}
        if args.spec_k:
            verify = decode_step(geo, weights, tokens=args.spec_k + 1, depth=depth, kv=args.kv,
                                 kv_bpt=args.kv_bytes_per_token, gdn_state_bytes_per=state_bytes,
                                 hardware=hardware)
            entry["speculative"] = speculative(step, verify, args.spec_k, args.acceptance)
        report["decode"].append(entry)
    if args.tok_s is not None:
        first = report["decode"][0]
        measured_ms = 1e3 / args.tok_s
        report["measured"] = {"tok_s": args.tok_s, "ms": measured_ms,
                              "model_error_pct": 100 * (first["tok_s"] / args.tok_s - 1),
                              "gap_over_floor_ms": measured_ms - first["floor_ms"],
                              "pct_of_floor": 100 * first["floor_ms"] / measured_ms}
    report["prefill"] = [prefill(geo, weights, n, hardware, chunk=args.chunk)
                         for n in ([4096] if args.prompt is None else args.prompt)]
    if args.levers:
        report["levers"] = rank_levers(geo, RECIPES[recipe], overrides, depth=depths[0],
                                       kv=args.kv, kv_bpt=args.kv_bytes_per_token,
                                       gdn_state_bytes_per=state_bytes, hardware=hardware)
    return report


def parse_format(item: str) -> tuple[str, str]:
    role, sep, fmt = item.partition("=")
    if not sep or fmt not in FORMAT_BITS:
        raise SystemExit(f"--format wants ROLE=FORMAT with FORMAT one of {sorted(FORMAT_BITS)}; got {item!r}")
    return role, fmt


def print_report(r: dict, spec_k: int) -> None:
    print(f"{r['model']} / recipe {r['recipe']}"
          + (f" / overrides {r['overrides']}" if r["overrides"] else "")
          + f"   on {r['hardware']}: {r['read_gb_s']:.1f} GB/s read, {r['int8_tops']:.0f} INT8 TOPS")
    print(f"resident weights {r['resident_gb']:.2f} GB, {r['layers']} layers"
          f"{', MoE' if r['moe'] else ', dense'}")
    for d in r["decode"]:
        print(f"\nDECODE one token, depth {d['depth']:,}")
        print(f"  {'bytes read':<30}{'GB':>8}{'share':>8}{'ms':>7}{'tok/s if free':>15}")
        for row in d["table"]:
            print(f"  {row['role']:<30}{row['bytes'] / 1e9:>8.3f}{100 * row['share']:>7.1f}%"
                  f"{row['ms']:>7.2f}{row['tok_s_if_free']:>15.1f}")
        print(f"  {'total':<30}{d['total_gb']:>8.3f}")
        print(f"  bandwidth floor        {d['floor_ms']:6.2f} ms  = {d['floor_tok_s']:7.1f} tok/s")
        print(f"  + per-layer overhead   {d['overhead_ms']:6.2f} ms")
        print(f"  predicted              {d['step_ms']:6.2f} ms  = {d['tok_s']:7.1f} tok/s")
        spec = d.get("speculative")
        if spec:
            print(f"  speculative k={spec['k']}, acceptance {spec['acceptance']:.0%}: "
                  f"{spec['tokens_per_round']:.2f} tokens/round, round {spec['round_ms']:.2f} ms "
                  f"({spec['round_over_plain']:.2f}x a plain step) = {spec['tok_s']:.1f} tok/s, "
                  f"{spec['speedup']:.2f}x plain")
    if "measured" in r:
        m = r["measured"]
        print(f"\nMEASURED {m['tok_s']:.1f} tok/s ({m['ms']:.2f} ms/token) at depth "
              f"{r['decode'][0]['depth']:,}")
        print(f"  model is {m['model_error_pct']:+.1f}% off; the measurement runs at "
              f"{m['pct_of_floor']:.0f}% of the bandwidth floor,")
        print(f"  leaving {m['gap_over_floor_ms']:.2f} ms/token that bytes do not explain: that is "
              f"what kernel work can still win.")
    for p in r["prefill"]:
        print(f"\nPREFILL {p['prompt']:,}-token prompt: {p['tok_s']:,.0f} tok/s predicted "
              f"({p['total_s']:.2f} s = {p['steps']} x fixed {p['fixed_s'] / p['steps']:.3f} s + "
              f"matmul {p['matmul_s']:.2f} s at {100 * p['matmul_efficiency']:.0f}% of peak + "
              f"attention {p['attention_s']:.2f} s)")
        print(f"  {p['ops_per_token'] / 1e9:.1f} GOP/token; the INT8 peak alone would give "
              f"{p['ceiling_tok_s']:,.0f} tok/s")
    if r.get("levers"):
        print(f"\nLEVERS ranked by predicted plain-decode gain (depth {r['decode'][0]['depth']:,})")
        for row in r["levers"]:
            print(f"  {row['gain_pct']:+6.1f}%  {row['tok_s']:7.1f} tok/s  [{row['kind']:<6}] "
                  f"{row['name']}")
            print(f"{'':>34}{row['evidence']}")
        print("  'bytes' levers change numerics: the saving is arithmetic, the quality cost is not.")
    print("\nA model, not a measurement. Disagreement with a real run is the finding.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--depth", type=non_negative_int, action="append",
                        help="KV depth in tokens; repeatable")
    parser.add_argument("--levers", action="store_true",
                        help="rank a fixed menu of what-ifs by predicted plain-decode gain")
    parser.add_argument("--spec-k", type=non_negative_int, default=0,
                        help="drafts per round; 0 = no speculation")
    parser.add_argument("--acceptance", type=probability, default=0.6,
                        help="accepted drafts / drafted tokens, from a real run")
    parser.add_argument("--tok-s", type=positive_float,
                        help="measured plain-decode tok/s at the first depth")
    parser.add_argument("--prompt", type=positive_int, action="append",
                        help="prompt tokens; repeatable")
    parser.add_argument("--chunk", type=positive_int, default=4096,
                        help="prefill chunk size in tokens")
    parser.add_argument("--top", type=positive_int, default=8, help="roles to list individually")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, args.spec_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
