#!/usr/bin/env python3
"""A first-principles decode model of DeepSeek V4.1-Flash on one RTX 3090 with VRAM / RAM / SSD expert tiers.

`flash_next_perf_model.py` is anchored on measured Flash-Next steps. V4.1-Flash has no run on this
card yet, so this one is built from bytes, the few efficiencies Flash-Next did measure, and the one
real piece of V4.1 routing data that exists, and is meant to answer, before any GPU time is spent:

  * How fast could decode be, and where does the time go (dense path, VRAM hits, RAM misses, SSD)?
  * How sensitive is that to how skewed expert use is? (`--sweep`)
  * How do concurrent **lanes** scale? The dense weights are read once per round whatever the lane
    count, the experts are not shared: each lane brings its own top-6 per layer.

    python tools/dsv4_perf_model.py                              # GSQ IQ2_XXS, 3090 + 64 GB RAM, measured routing
    python tools/dsv4_perf_model.py --format gsq_iq1_m --ssd-gb-s 3.6
    python tools/dsv4_perf_model.py --sweep                      # routing skew x SSD speed, at 1 and 4 lanes
    python tools/dsv4_perf_model.py --calibrate                  # what Flash-Next's 97% implies, for comparison

How it works. A token includes expert i of a layer with probability q_i (sum q = top-k = 6). By default
q comes from the measured rank curve in `dsv4/data/routing_rank_curve.json` (router counts over 1.78 M
tokens, from Strata's shipped profile; see `dsv4/profile_to_curve.py`); `--skew s` swaps in a Zipf(s)
curve instead. The caches hold the most popular experts: the VRAM tier the top slice, the RAM tier the
next, the SSD tier the rest (a STATIC ranking: real LRU/LFRU caches also exploit locality within a
prompt, so this is a floor on hit rates, not a ceiling). A round of B lanes touches expert i with
probability 1 - (1 - q_i)^B, so lanes widen the working set and push more reads to slower tiers.
Time per round = dense path (once) + per-lane overhead + VRAM-hit expert reads + staged RAM misses
over PCIe + SSD misses at the drive's rate, summed (a miss stalls the layer; `--overlap` hides a share).

How far to trust it. The routing curve is real but global: V4.1's top 10% of experts carry 31.8% of
routing mass (Zipf s ~ 0.5), far flatter than Flash-Next's cache behaviour implies (a Zipf fit of its
97.2% hit rate at a 48.6% cache gives s ~ 1.5), so V4.1 leans on the slow tiers much harder. The
narrow-route efficiency (29% of the card) and the PCIe rate are Flash-Next measurements; the
dense-path efficiency is an assumption (`--dense-eff`). Flash-Next's routed-expert time is reproduced
within ~25% (tested). Strata's published decode speeds on other hardware (see
docs/maintainer/dsv4-flash-plan.md) are the only end-to-end anchors. Not modelled: prefill,
MTP/DSpark speculation, CPU-side expert compute, the Engram row reads (about 48 random 256 B rows per
token, prefetchable from token ids), 1M contexts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "dsv4"))
import flash_next_perf_model as fn  # noqa: E402
import gguf_sizes as gs  # noqa: E402

GIB = 1 << 30

# V4.1-Flash, from the safetensors headers (tools/dsv4/layout.py, 2026-11-11).
LAYERS = 40
EXPERTS = 384
TOP_K = 6
EXPERT_PARAMS = 3 * 5120 * 2304  # 35.4 M weights per expert
# Always-resident, read every token (bytes). In the ISTA-DASLab GSQ-RCO builds (Q5_K / Q8_0 dense
# tensors) that is 5.55 GB (tools/dsv4/gguf_sizes.py on the allocation lists, identical in all three
# builds) plus about 0.2 GB of norms and mixers. The native FP8 checkpoint's equivalent is 8.4 GB
# (attention 5.15, shared experts 1.42, BF16 head 1.32, the rest 0.6). The embedding table is a row
# gather and lives in RAM.
DENSE_BYTES = 5.55e9 + 0.2e9
# Per-lane state is tiny: a 128-row window KV per layer (40 x 128 x 512 x 2 B = 5.2 MB) plus a compressed
# cache on four layers (about 2 KB per context token).
# Average routed-expert bits per weight. `gsq_*` are the real ISTA-DASLab GSQ-RCO builds, whose
# per-layer types average to these (gguf_sizes.py: 8.60 / 9.70 / 7.46 MB per expert); the others are
# uniform GGML types for comparison.
BITS_PER_WEIGHT = {"fp4_native": 4.25} | {name.lower(): gs.BPW[name] for name in ("IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M")} | {
    "gsq_iq2_xxs": 1.944, "gsq_iq2_xs": 2.194, "gsq_iq1_m": 1.686}

# Flash-Next, the calibration target (flash-next-plan.md PR 9): 512 experts, top-10, 48 layers; a VRAM
# cache of 11,951 of 24,576 experts gave a 97.2% decode hit rate at 8K.
FN_EXPERTS, FN_TOP_K, FN_LAYERS = fn.EXPERTS, fn.TOP_K, fn.LAYERS
FN_CACHE_FRACTION = 11_951 / (FN_LAYERS * FN_EXPERTS)
FN_HIT_RATE = 0.972
NARROW_ROUTE_EFFICIENCY = 0.29  # PR 7: 28-30% of 936 GB/s on the resident T=1 route

ROUTING_CURVE_PATH = Path(__file__).resolve().parent / "dsv4" / "data" / "routing_rank_curve.json"


def profile_inclusion_probs(path: Path = ROUTING_CURVE_PATH) -> list[float]:
    """Per-token probability that the k-th most popular expert of a layer is chosen, from the measured
    routing mass per rank (the shares sum to 1 and each token chooses TOP_K experts)."""
    curve = json.loads(path.read_text())
    if curve["experts"] != EXPERTS:
        raise ValueError(f"{path}: curve is for {curve['experts']} experts, the model has {EXPERTS}")
    return [min(1.0, TOP_K * share) for share in curve["rank_share"]]


@dataclasses.dataclass(frozen=True)
class Machine:
    """The box and the assumptions about it."""

    vram_gib: float = 24.0
    vram_reserve_gib: float = 1.8  # CUDA context, graphs, workspaces, staging slots
    ram_gib: float = 64.0
    ram_reserve_gib: float = 14.0  # OS, the embedding table, staging, Engram row cache, pinned buffers
    ssd_gb_s: float = 5.0  # NVMe gen4 sequential-ish 10 MB reads; SATA is 0.5
    pcie_gb_s: float = fn.PCIE_GB_S
    card_gb_s: float = fn.CARD_GB_S
    dense_eff: float = 0.60  # ASSUMPTION: dense GEMVs reach 60% of the card; measure and replace
    expert_eff: float = NARROW_ROUTE_EFFICIENCY
    lane_ms: float = 0.4  # ASSUMPTION: window attention, sampling, launch tails per extra lane
    overlap: float = 0.0  # share of miss transfer time hidden behind GPU work


def inclusion_probs(n: int, k: int, s: float) -> list[float]:
    """Per-token probability that each of `n` Zipf(s)-ranked experts is in the top-k (sums to k)."""
    w = [(i + 1) ** -s for i in range(n)]
    lo, hi = 0.0, 1e9
    for _ in range(200):  # find c so that sum(min(1, c * w)) == k
        c = (lo + hi) / 2
        if sum(min(1.0, c * x) for x in w) < k:
            lo = c
        else:
            hi = c
    return [min(1.0, hi * x) for x in w]


def hit_rate(n: int, k: int, s: float, cached_fraction: float) -> float:
    """Share of a token's expert reads served by a static cache of the most popular experts."""
    q = inclusion_probs(n, k, s)
    return sum(q[: round(cached_fraction * n)]) / k


def calibrate_skew() -> float:
    """The Zipf exponent at which Flash-Next's cache fraction reproduces its measured hit rate."""
    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if hit_rate(FN_EXPERTS, FN_TOP_K, mid, FN_CACHE_FRACTION) < FN_HIT_RATE:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclasses.dataclass(frozen=True)
class Plan:
    """Where the experts live, for one quantisation and machine."""

    bits: float
    expert_bytes: int
    vram_experts: int
    ram_experts: int
    total_experts: int

    @property
    def vram_fraction(self) -> float:
        return self.vram_experts / self.total_experts

    @property
    def ram_fraction(self) -> float:
        return self.ram_experts / self.total_experts

    @property
    def ssd_fraction(self) -> float:
        return 1.0 - self.vram_fraction - self.ram_fraction


def make_plan(bits: float, m: Machine, dense_vram_bytes: float = DENSE_BYTES, lanes: int = 1) -> Plan:
    expert_bytes = -(-int(EXPERT_PARAMS * bits) // 8)
    lane_state = lanes * 6e6
    vram_free = (m.vram_gib - m.vram_reserve_gib) * GIB - dense_vram_bytes - lane_state
    ram_free = (m.ram_gib - m.ram_reserve_gib) * GIB
    total = LAYERS * EXPERTS
    vram = max(0, min(total, int(vram_free // expert_bytes)))
    ram = max(0, min(total - vram, int(ram_free // expert_bytes)))
    return Plan(bits, expert_bytes, vram, ram, total)


def resolve_routing(routing: float | list[float]) -> list[float]:
    """`routing` is a Zipf exponent or already a per-rank inclusion-probability list."""
    return inclusion_probs(EXPERTS, TOP_K, routing) if isinstance(routing, (int, float)) else list(routing)


def round_time(lanes: int, plan: Plan, routing: float | list[float], m: Machine) -> dict:
    """One decode round of `lanes` independent single-token lanes."""
    q = resolve_routing(routing)
    edges = (0, round(plan.vram_fraction * EXPERTS), round((plan.vram_fraction + plan.ram_fraction) * EXPERTS), EXPERTS)

    def touched(a: int, b: int) -> float:  # expected distinct experts one layer touches in rank [a, b)
        return sum(1.0 - (1.0 - qi) ** lanes for qi in q[a:b])

    reads = [LAYERS * touched(edges[i], edges[i + 1]) for i in range(3)]  # per round: vram, ram, ssd
    eb = plan.expert_bytes
    dense_ms = DENSE_BYTES / (m.dense_eff * m.card_gb_s * 1e9) * 1e3
    vram_ms = reads[0] * eb / (m.expert_eff * m.card_gb_s * 1e9) * 1e3
    # A RAM or SSD expert is staged to the card, then read like a hit; the stage is the exposed cost.
    ram_ms = reads[1] * eb / (m.pcie_gb_s * 1e9) * 1e3 * (1.0 - m.overlap)
    ssd_ms = reads[2] * eb * (1.0 / (m.ssd_gb_s * 1e9) + 1.0 / (m.pcie_gb_s * 1e9)) * 1e3 * (1.0 - m.overlap)
    staged_compute_ms = (reads[1] + reads[2]) * eb / (m.expert_eff * m.card_gb_s * 1e9) * 1e3
    lane_ms = (lanes - 1) * m.lane_ms
    total = dense_ms + vram_ms + staged_compute_ms + ram_ms + ssd_ms + lane_ms
    return {
        "lanes": lanes,
        "round_ms": total,
        "tok_s_aggregate": lanes / total * 1e3,
        "tok_s_per_lane": 1e3 / total,
        "dense_ms": dense_ms,
        "vram_ms": vram_ms + staged_compute_ms,
        "ram_ms": ram_ms,
        "ssd_ms": ssd_ms,
        "lane_ms": lane_ms,
        "reads": reads,
        "single_token_hit_rate": sum(q[: edges[1]]) / TOP_K,
        "single_token_ssd_share": sum(q[edges[2]:]) / TOP_K,
    }


PREFILL_COMPUTE_TOK_S = 1500.0  # ASSUMPTION: GPU-bound prefill with every expert resident; Flash-Next
# measured 650-2,400 tok/s on a model with ~1.5 MB experts and 10 per token, so V4.1 is plausibly lower.


def prefill(chunk: int, plan: Plan, routing: float | list[float], m: Machine,
            compute_tok_s: float = PREFILL_COMPUTE_TOK_S) -> dict:
    """One layer-major prefill chunk: every expert any of its `chunk` tokens routes to is streamed in
    once, so a big chunk touches nearly all 15,360 and the SSD tier is read almost whole. Streaming
    runs ahead of compute (a ring of lent frames, as Flash-Next PR 10), so the chunk takes the longer
    of the two."""
    r = round_time(chunk, plan, routing, m)  # `lanes` = tokens: reads[] are the distinct experts touched
    eb = plan.expert_bytes
    transfer_ms = (r["reads"][1] * eb / (m.pcie_gb_s * 1e9)
                   + r["reads"][2] * eb * (1.0 / (m.ssd_gb_s * 1e9) + 1.0 / (m.pcie_gb_s * 1e9))) * 1e3
    compute_ms = chunk / compute_tok_s * 1e3
    ms = max(transfer_ms, compute_ms)
    return {"chunk": chunk, "ms": ms, "tok_s": chunk / ms * 1e3, "transfer_ms": transfer_ms,
            "compute_ms": compute_ms, "bound": "streaming" if transfer_ms >= compute_ms else "compute"}


def fn_routed_check(m: Machine) -> dict:
    """This model's routed-expert time for Flash-Next at one lane, against PR 9's measured 4.27 ms.

    Flash-Next: 480 expert reads per token at 1,510,400 B, 97.2% served from the VRAM cache."""
    reads = FN_LAYERS * FN_TOP_K
    hits = FN_HIT_RATE * reads
    misses = (1 - FN_HIT_RATE) * reads
    hit_ms = hits * fn.EXPERT_FRAME_BYTES / (m.expert_eff * m.card_gb_s * 1e9) * 1e3
    miss_ms = misses * fn.EXPERT_FRAME_BYTES / (m.pcie_gb_s * 1e9) * 1e3
    return {"model_ms": hit_ms + miss_ms, "measured_ms": 4.27}


def table(plan: Plan, s: float | list[float], m: Machine, lane_counts) -> str:
    rows = [f"{'lanes':>5} {'round ms':>9} {'tok/s/lane':>11} {'tok/s total':>12} | "
            f"{'dense':>6} {'vram':>6} {'ram':>6} {'ssd':>6} {'lanes':>6}"]
    for b in lane_counts:
        r = round_time(b, plan, s, m)
        rows.append(f"{b:5d} {r['round_ms']:9.1f} {r['tok_s_per_lane']:11.1f} {r['tok_s_aggregate']:12.1f} | "
                    f"{r['dense_ms']:6.1f} {r['vram_ms']:6.1f} {r['ram_ms']:6.1f} {r['ssd_ms']:6.1f} {r['lane_ms']:6.1f}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=sorted(BITS_PER_WEIGHT), default="gsq_iq2_xxs")
    ap.add_argument("--skew", type=float, help="Zipf exponent instead of the measured V4.1 routing curve")
    ap.add_argument("--lanes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ssd-gb-s", type=float, default=Machine.ssd_gb_s)
    ap.add_argument("--ram-gib", type=float, default=Machine.ram_gib)
    ap.add_argument("--dense-eff", type=float, default=Machine.dense_eff)
    ap.add_argument("--overlap", type=float, default=Machine.overlap)
    ap.add_argument("--sweep", action="store_true", help="skew x SSD speed at 1 and 4 lanes")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    fitted = calibrate_skew()
    if a.calibrate:
        chk = fn_routed_check(Machine())
        print(f"Flash-Next: {FN_CACHE_FRACTION:.1%} of experts cached, {FN_HIT_RATE:.1%} hit rate -> Zipf s = {fitted:.3f}")
        print(f"routed-expert time check: model {chk['model_ms']:.2f} ms vs measured {chk['measured_ms']:.2f} ms")
        return
    m = Machine(ram_gib=a.ram_gib, ssd_gb_s=a.ssd_gb_s, dense_eff=a.dense_eff, overlap=a.overlap)
    bits = BITS_PER_WEIGHT[a.format]
    plan = make_plan(bits, m, lanes=max(a.lanes))
    print(f"{a.format} ({bits} bpw): {plan.expert_bytes / 1e6:.2f} MB/expert; experts in VRAM {plan.vram_experts:,} "
          f"({plan.vram_fraction:.1%}), RAM {plan.ram_experts:,} ({plan.ram_fraction:.1%}), SSD "
          f"{plan.total_experts - plan.vram_experts - plan.ram_experts:,} ({plan.ssd_fraction:.1%})")
    measured = profile_inclusion_probs()
    if a.sweep:
        cases = [("uniform (s=0)", 0.0), ("Zipf 0.4", 0.4), ("MEASURED V4.1", measured), ("Zipf 0.8", 0.8),
                 ("Zipf 1.2", 1.2), (f"Flash-Next fit ({fitted:.2f})", fitted)]
        for lanes in (1, 4):
            print(f"\ntok/s per lane at {lanes} lane(s): rows = expert routing, columns = SSD GB/s")
            speeds = (0.5, 2.0, 3.5, 7.0)
            print(f"{'routing':>22} " + " ".join(f"{v:>7}" for v in speeds) + "   single-token hit / SSD share")
            for label, routing in cases:
                cells = [f"{round_time(lanes, plan, routing, dataclasses.replace(m, ssd_gb_s=v))['tok_s_per_lane']:7.1f}"
                         for v in speeds]
                r = round_time(lanes, plan, routing, m)
                print(f"{label:>22} " + " ".join(cells) + f"   {r['single_token_hit_rate']:.0%} / {r['single_token_ssd_share']:.0%}")
        return
    s = a.skew if a.skew is not None else measured
    print(f"routing: {'Zipf ' + format(s, '.2f') if a.skew is not None else 'measured V4.1 rank curve'}, "
          f"SSD {m.ssd_gb_s} GB/s, dense eff {m.dense_eff:.0%}\n")
    print(table(plan, s, m, a.lanes))
    one = round_time(1, plan, s, m)
    print(f"\none token: {one['single_token_hit_rate']:.1%} of expert reads hit VRAM, "
          f"{one['single_token_ssd_share']:.1%} fall to the SSD")
    print("\nprefill (layer-major, experts streamed once per chunk):")
    for chunk in (1024, 4096, 8192, 16384):
        p = prefill(chunk, plan, s, m)
        print(f"  {chunk:6d}-token chunks: {p['tok_s']:7.0f} tok/s ({p['bound']}-bound, "
              f"{p['transfer_ms'] / 1e3:.1f} s streaming vs {p['compute_ms'] / 1e3:.1f} s compute per chunk)")


if __name__ == "__main__":
    main()
