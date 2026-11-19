#!/usr/bin/env python3
"""A NumPy reference for DeepSeek V4's MoE layer: the router chain, the routed experts and the shared expert.

It mirrors `Gate`, `Expert` and `MoE` of the model repo's `inference/model.py`, in FP32 and without a GPU,
built from the pieces `reference.py` already checks (`gate_forward`, `expert_forward`):

  * the router: sqrt-softplus scores, an aux-loss-free bias that picks the experts but does not weight
    them, top-k, weights from the unbiased scores normalised over the chosen k (with the `1e-20`
    of training, not the norm eps) and scaled by `route_scale`;
  * each chosen expert is a SwiGLU FFN whose clamps come from training (up both sides, gate from above);
    the routing weight multiplies the activation before the down projection;
  * every token also goes through one shared expert (same shape, unweighted);
  * the routed contributions are summed in FP32 in expert order, then the shared expert is added.

The clamps are separate for the routed and the shared experts (the GGUF carries a list per layer for each).
The vision routing bias (`bias_vl`, image spans) is not modelled: text only.

Shapes: x [tokens, dim]; routed experts w1, w3 [inter, dim] and w2 [dim, inter]; router [experts, dim].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reference import expert_forward, gate_forward


@dataclass
class ExpertWeights:
    w1: np.ndarray  # gate projection [inter, dim]
    w2: np.ndarray  # down projection [dim, inter]
    w3: np.ndarray  # up projection [inter, dim]


@dataclass
class MoEWeights:
    router: np.ndarray  # [experts, dim]
    bias: np.ndarray  # [experts]
    experts: list[ExpertWeights]
    shared: ExpertWeights


@dataclass(frozen=True)
class MoEConfig:
    topk: int
    route_scale: float = 1.0
    norm_topk_prob: bool = True
    gate_temp: float = 1.0
    clamp_experts: float = 0.0  # swiglu limit of the routed experts; 0 disables
    clamp_shared: float = 0.0  # and of the shared expert


def route(x: np.ndarray, w: MoEWeights, cfg: MoEConfig) -> tuple[np.ndarray, np.ndarray]:
    """(weights, indices) [tokens, topk]: the router chain alone."""
    return gate_forward(x, w.router, w.bias, cfg.topk, cfg.route_scale, cfg.norm_topk_prob, cfg.gate_temp)


def moe_forward(x: np.ndarray, w: MoEWeights, cfg: MoEConfig) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if len(w.experts) != w.router.shape[0]:
        raise ValueError(f"{len(w.experts)} experts but the router scores {w.router.shape[0]}")
    if not 1 <= cfg.topk <= len(w.experts):
        raise ValueError(f"topk {cfg.topk} is not in 1..{len(w.experts)}")
    weights, indices = route(x, w, cfg)
    y = np.zeros(x.shape, np.float32)
    for e, expert in enumerate(w.experts):
        rows, slot = np.nonzero(indices == e)
        if rows.size == 0:
            continue
        y[rows] += expert_forward(x[rows], expert.w1, expert.w2, expert.w3, cfg.clamp_experts, weights[rows, slot][:, None])
    y += expert_forward(x, w.shared.w1, w.shared.w2, w.shared.w3, cfg.clamp_shared)
    return y.astype(np.float32)


def random_moe(rng: np.random.Generator, dim: int, inter: int, experts: int) -> MoEWeights:
    """Deterministic weights (the draw order is part of the contract: the committed golden used the same draws)."""
    def mat(rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
        return (scale * rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)

    def expert() -> ExpertWeights:
        return ExpertWeights(mat(inter, dim, 3.0), mat(dim, inter), mat(inter, dim, 3.0))

    router = mat(experts, dim)
    bias = (0.3 * rng.standard_normal(experts)).astype(np.float32)
    return MoEWeights(router, bias, [expert() for _ in range(experts)], expert())


# Two layers of the golden: one with clamps, a smaller top-k and a route scale; one without clamps and a larger top-k.
GOLDEN_SEED = 5
GOLDEN_TOKENS, GOLDEN_DIM, GOLDEN_INTER, GOLDEN_EXPERTS = 9, 32, 16, 6
GOLDEN_CONFIGS = (
    MoEConfig(topk=2, route_scale=1.5, norm_topk_prob=True, clamp_experts=1.5, clamp_shared=1.5),
    MoEConfig(topk=3, route_scale=1.0, norm_topk_prob=False, clamp_experts=0.0, clamp_shared=0.0),
)


def golden_inputs(seed: int = GOLDEN_SEED) -> list[tuple[np.ndarray, MoEWeights, MoEConfig]]:
    rng = np.random.default_rng(seed)
    out = []
    for cfg in GOLDEN_CONFIGS:
        w = random_moe(rng, GOLDEN_DIM, GOLDEN_INTER, GOLDEN_EXPERTS)
        x = rng.standard_normal((GOLDEN_TOKENS, GOLDEN_DIM)).astype(np.float32)
        out.append((x, w, cfg))
    return out


def run_golden(seed: int = GOLDEN_SEED) -> list[np.ndarray]:
    return [moe_forward(x, w, cfg) for x, w, cfg in golden_inputs(seed)]
