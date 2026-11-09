#!/usr/bin/env python3
"""A component-time model of Qwen3.8-Flash-Next (`qwen4exp`) decode on the RTX 3090, for a Mac.

`perf_model.py` is a bandwidth model, and Flash-Next does not run at the bandwidth floor: its
decode step is a chain of small, latency-bound kernels (hyper-connection mixers, a two-kernel narrow
expert route at 28-30% of bandwidth, GGML GEMVs) plus PCIe traffic whenever an expert is not in the
VRAM cache. So this model is built from the measured pieces instead of from bytes, and answers the
questions the Flash-Next work keeps asking:

  * What does a change in the **expert-cache hit rate** cost? (`--hit-rate`)
  * What should **MTP** give at a draft count and acceptance? (`--mtp-k`, `--acceptance`)
  * Should a second **lane** beat MTP at one lane, and at what acceptance does it stop paying?
    (`--lanes`)
  * Where would kernel or cache work buy the most? (`--levers`)
  * Is a measurement what the model expects? (`--tok-s`)

    python tools/flash_next_perf_model.py --hit-rate 0.972 --mtp-k 3 --acceptance 0.65 --lanes 2
    python tools/flash_next_perf_model.py --levers
    python tools/flash_next_perf_model.py --tok-s 77.5 --hit-rate 0.972     # a plain-decode run

Everything is anchored on three measurements in `docs/maintainer/flash-next-plan.md` and the lanes
spike (#284): the 8K plain step (12.5 ms at a 97.2% decode hit rate), the cost of a W-column
verification (17.0 / 20.0 / 23.8 ms at W = 2 / 3 / 4), and the size of an expert frame
(1,510,400 B) over the effective PCIe rate (24.2 GB/s), which prices a miss at 62 us.

How far to trust it. The 1K and 32K plain-decode tiers (98.0% and 96.6% hit rate) are PREDICTED from
the 8K anchor and the physical miss cost, and land within 1%. MTP at 1 to 3 drafts reproduces the
measured depth sweep within 5%; 4 and 5 drafts come out 10-15% too fast because the verification
cost per extra column rises faster than the line this extrapolates. The two-lane aggregate lands
inside the measured range (95-104 tok/s) with warm caches and a little low with cold ones. Whenever
it disagrees with a run on the card, the run is right.

Not modelled: prefill, vision, the context cache, 128K contexts, and any C above 2 (the lanes spike
measured a hit-rate collapse there that this model does not reproduce).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_model as pm  # noqa: E402

LAYERS = 48
EXPERTS = 512
TOP_K = 10
EXPERT_FRAME_BYTES = 1_510_400  # lanes spike (#284): one expert record in the VRAM cache
PCIE_GB_S = 24.2  # effective staged-copy rate, flash-next-plan.md "PR 7 measurements"
NARROW_ROUTE_BYTES = 10 * EXPERT_FRAME_BYTES * LAYERS  # the ten selected experts of every layer
HC_MIXER_BYTES = 97 * 13.2e6  # 97 fused mixer calls over 13.2 MB of BF16 each (PR 6 / PR 9)
LANE_STATE_BYTES = 115.65e6  # recurrent state per lane (#284)
CARD_GB_S = 936.1  # the spec-sheet figure the plan's "% of 936 GB/s" columns use


@dataclass(frozen=True)
class Calibration:
    """Constants, each with where it came from."""

    # Plain decode step at 8K, warm cache. flash-next-plan.md PR 13 ("~12.5 ms plain step"), which
    # matches the wall-clock 79.88 tok/s of the PR 11 gate (12.52 ms).
    step_ms: tuple[float, ...] = (12.5, 17.0, 20.0, 23.8)  # W = 1..4 columns of one sequence (PR 13 nsys)
    extra_column_ms: float = 3.4  # beyond W = 4: PR 13's "+3.4 ms per column", an under-estimate
    ref_hit_rate: float = 0.972  # decode hit rate at 8K, PR 9
    # PR 13: of the 3.4 ms a verification column adds, 2.0 ms is routed experts, because a column
    # touches new experts. Against the 4.27 ms the routed experts cost a plain token (PR 9) that is
    # a 46.8% share, which is how many of a new column's 10 experts are not already selected.
    routed_column_ms: float = 2.0
    same_sequence_new_fraction: float = 2.0 / 4.27
    drafter_ms: float = 0.88  # one drafter step (PR 13)
    lane_extra_ms: float = 2 * LANE_STATE_BYTES / (pm.ACHIEVABLE_READ_GB_S * 1e9) * 1e3  # state r+w
    pcie_gb_s: float = PCIE_GB_S

    @property
    def miss_ms(self) -> float:
        """One staged expert miss: its frame over the effective PCIe rate (62.4 us)."""
        return EXPERT_FRAME_BYTES / (self.pcie_gb_s * 1e9) * 1e3


CAL = Calibration()

# Measured MTP acceptance by fixed draft count at 8K (PR 13): accepted drafts / drafted tokens.
ACCEPTANCE_8K = {1: 0.82, 2: 0.76, 3: 0.65, 4: 0.58, 5: 0.51}
# Measured decode tok/s for the same sweep, and the PR 11 gate's plain tiers (hit rate, tok/s).
MTP_SWEEP_8K = {1: 102.1, 2: 112.0, 3: 111.9, 4: 96.0, 5: 90.0}
PLAIN_TIERS = {"1K": (0.980, 81.15), "8K": (0.972, 79.88), "32K": (0.966, 78.96)}


def distinct_experts(rows: int) -> float:
    """Experts one layer touches for `rows` independent tokens with uniform routing. The lanes spike
    measured 19.6 at two lanes from real traces; this gives 19.8."""
    return EXPERTS * (1.0 - (1.0 - TOP_K / EXPERTS) ** rows)


def independent_new_fraction(row: int) -> float:
    """Of the ten experts the `row`-th independent token selects, the share nobody selected yet."""
    return (distinct_experts(row) - distinct_experts(row - 1)) / TOP_K


def misses_per_token(hit_rate: float) -> float:
    return (1.0 - hit_rate) * LAYERS * TOP_K


def _table_ms(cal: Calibration, columns: int) -> float:
    table = cal.step_ms
    if columns <= len(table):
        return table[columns - 1]
    return table[-1] + (columns - len(table)) * cal.extra_column_ms


def step_ms(columns: int = 1, hit_rate: float = CAL.ref_hit_rate, *, independent: bool = False,
            cal: Calibration = CAL) -> float:
    """One decode round over `columns` tokens.

    Same-sequence columns (an MTP verification) share most of their experts. Independent rows (two
    lanes) share almost none, so each costs more routed-expert time, and each lane reads and writes
    its own recurrent state. The hit rate moves only the staged misses: more distinct experts means
    proportionally more lookups, each missing with probability (1 - hit rate)."""
    ms = _table_ms(cal, columns)
    distinct_ratio = 1.0
    if columns > 1:
        if independent:
            for row in range(2, columns + 1):
                ratio = independent_new_fraction(row) / cal.same_sequence_new_fraction
                ms += cal.routed_column_ms * (ratio - 1.0) + cal.lane_extra_ms
            distinct_ratio = distinct_experts(columns) / TOP_K
        else:
            distinct_ratio = 1.0 + cal.same_sequence_new_fraction * (columns - 1)
    extra_misses = (cal.ref_hit_rate - hit_rate) * LAYERS * TOP_K * distinct_ratio
    return ms + extra_misses * cal.miss_ms


def plain(hit_rate: float = CAL.ref_hit_rate, cal: Calibration = CAL) -> dict:
    ms = step_ms(1, hit_rate, cal=cal)
    return {"hit_rate": hit_rate, "step_ms": ms, "tok_s": 1e3 / ms,
            "misses_per_token": misses_per_token(hit_rate)}


def mtp(k: int, acceptance: float, hit_rate: float = CAL.ref_hit_rate, cal: Calibration = CAL) -> dict:
    """MTP at a fixed draft count: a (k+1)-column verification plus k drafter steps."""
    tokens = pm.tokens_per_round(k, acceptance)
    round_ms = step_ms(k + 1, hit_rate, cal=cal) + k * cal.drafter_ms
    return {"k": k, "acceptance": acceptance, "tokens_per_round": tokens, "round_ms": round_ms,
            "tok_s": tokens / round_ms * 1e3,
            "speedup": tokens / round_ms * 1e3 / plain(hit_rate, cal)["tok_s"]}


def lanes(rows: int, hit_rate: float = CAL.ref_hit_rate, cal: Calibration = CAL) -> dict:
    """`rows` independent requests decoding in one plain batched round (no MTP, as shipped)."""
    ms = step_ms(rows, hit_rate, independent=True, cal=cal)
    return {"rows": rows, "round_ms": ms, "tok_s": rows / ms * 1e3, "per_stream_tok_s": 1e3 / ms,
            "distinct_experts_per_layer": distinct_experts(rows)}


def crossover_acceptance(rows: int, k: int, hit_rate: float = CAL.ref_hit_rate,
                         cal: Calibration = CAL) -> float | None:
    """The MTP acceptance at which one lane with `k` drafts decodes exactly as fast, in aggregate, as
    `rows` plain lanes. Below it the lanes win decode throughput; above it MTP wins. None if even
    perfect acceptance cannot reach the lanes (acceptance would exceed 1)."""
    round_ms = step_ms(k + 1, hit_rate, cal=cal) + k * cal.drafter_ms
    needed_tokens = lanes(rows, hit_rate, cal)["tok_s"] * round_ms / 1e3
    acceptance = (needed_tokens - 1.0) / k
    return acceptance if acceptance <= 1.0 else None


# ---------------------------------------------------------------------------------------------
# Levers
# ---------------------------------------------------------------------------------------------


def _saving_ms_at_bandwidth(component_ms: float, nbytes: float, fraction: float) -> float:
    """Time saved if a component moving `nbytes` in `component_ms` ran at `fraction` of the card."""
    return max(0.0, component_ms - nbytes / (fraction * CARD_GB_S * 1e9) * 1e3)


def _shifted(cal: Calibration, saved_ms: float) -> Calibration:
    """The same calibration with every step `saved_ms` shorter (the component is in every column)."""
    return dataclasses.replace(cal, step_ms=tuple(max(0.1, v - saved_ms) for v in cal.step_ms))


def rank_levers(mode: str, hit_rate: float, k: int, acceptance: float,
                cal: Calibration = CAL) -> list[dict]:
    """What-ifs ranked by their effect on `mode`'s throughput (plain, mtp or lanes)."""
    def rate(c: Calibration, hit: float, accept: float) -> float:
        if mode == "mtp":
            return mtp(k, accept, hit, c)["tok_s"]
        if mode == "lanes":
            return lanes(2, hit, c)["tok_s"]
        return plain(hit, c)["tok_s"]

    base = rate(cal, hit_rate, acceptance)
    # Component times from PR 9's decode breakdown (8K, warm cache): mixers 2.78, narrow expert route
    # (gate_up 1.46 + down 0.73) 2.19.
    hc_saving = _saving_ms_at_bandwidth(2.78, HC_MIXER_BYTES, 0.80)
    route_saving = _saving_ms_at_bandwidth(2.19, NARROW_ROUTE_BYTES, 0.70)
    candidates = [
        ("expert cache hit rate +1 point", "cache", rate(cal, min(1.0, hit_rate + 0.01), acceptance),
         "a bigger or smarter expert cache; the cache grows to ~16.8 GiB already"),
        ("expert cache hit rate +2 points", "cache", rate(cal, min(1.0, hit_rate + 0.02), acceptance),
         "as above"),
        ("each miss costs half (prefetch, overlap)", "cache",
         rate(_miss_cost_scaled(cal, 0.5), hit_rate, acceptance),
         "PR 11 tried a forked copy stream (slower) and a CPU leg (not wired)"),
        ("hyper-connection mixers at 80% of bandwidth", "kernel",
         rate(_shifted(cal, hc_saving), hit_rate, acceptance),
         "PR 6: 50% as shipped; a loads-only probe reached 81% but the one-kernel variant was rejected"),
        ("narrow expert route at 70% of bandwidth", "kernel",
         rate(_shifted(cal, route_saving), hit_rate, acceptance),
         "PR 11 tried a persistent kernel, L2 warming, grid gathers: at most ~15%; "
         "it is per-CTA latency, not bandwidth, so treat this as a ceiling"),
    ]
    if mode == "mtp":
        candidates.append(("MTP acceptance +5 points", "model",
                           rate(cal, hit_rate, min(1.0, acceptance + 0.05)),
                           "a better drafter or a distillation pass; text-dependent (summaries run lower)"))
        candidates.append(("drafter step twice as fast", "kernel",
                           rate(dataclasses.replace(cal, drafter_ms=cal.drafter_ms / 2), hit_rate, acceptance),
                           "0.40 of its 0.88 ms is the IQ4_XS head (PR 13)"))
    rows = [{"name": n, "kind": kind, "tok_s": v, "gain_pct": 100 * (v / base - 1), "note": note}
            for n, kind, v, note in candidates]
    return sorted(rows, key=lambda r: -r["gain_pct"])


def _miss_cost_scaled(cal: Calibration, scale: float) -> Calibration:
    """The calibration if every staged miss cost `scale` times as much. The measured steps already
    contain the misses of the reference hit rate, so remove the difference from each column, and let
    the hit-rate term price misses at the new rate."""
    saved = (1.0 - scale) * cal.miss_ms * misses_per_token(cal.ref_hit_rate)
    steps = tuple(v - saved * (1.0 + cal.same_sequence_new_fraction * column)
                  for column, v in enumerate(cal.step_ms))
    return dataclasses.replace(cal, step_ms=steps, pcie_gb_s=cal.pcie_gb_s / scale)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def build_report(args: argparse.Namespace) -> dict:
    report: dict = {"hit_rate": args.hit_rate, "plain": plain(args.hit_rate)}
    if args.mtp_k:
        acceptance = args.acceptance if args.acceptance is not None else ACCEPTANCE_8K.get(args.mtp_k, 0.6)
        report["mtp"] = mtp(args.mtp_k, acceptance, args.hit_rate)
    if args.lanes > 1:
        report["lanes"] = lanes(args.lanes, args.hit_rate)
        if args.mtp_k:
            report["lanes"]["crossover_acceptance"] = crossover_acceptance(
                args.lanes, args.mtp_k, args.hit_rate)
    if args.tok_s is not None:
        modelled = report["plain"]["tok_s"]
        report["measured"] = {"tok_s": args.tok_s, "model_error_pct": 100 * (modelled / args.tok_s - 1),
                              "gap_ms": 1e3 / args.tok_s - report["plain"]["step_ms"]}
    if args.levers:
        mode = "mtp" if args.mtp_k else ("lanes" if args.lanes > 1 else "plain")
        acceptance = (args.acceptance if args.acceptance is not None
                      else ACCEPTANCE_8K.get(args.mtp_k, 0.6))
        report["levers"] = {"mode": mode,
                            "rows": rank_levers(mode, args.hit_rate, args.mtp_k or 3, acceptance)}
    return report


def print_report(r: dict) -> None:
    p = r["plain"]
    print(f"Qwen3.8-Flash-Next on an RTX 3090, expert-cache decode hit rate {r['hit_rate']:.1%}")
    print(f"\nPLAIN decode: {p['step_ms']:.2f} ms/token = {p['tok_s']:.1f} tok/s "
          f"({p['misses_per_token']:.1f} expert misses per token, "
          f"{CAL.miss_ms * 1e3:.0f} us each)")
    if "measured" in r:
        m = r["measured"]
        print(f"  measured {m['tok_s']:.1f} tok/s: the model is {m['model_error_pct']:+.1f}% off "
              f"({m['gap_ms']:+.2f} ms/token not explained)")
    if "mtp" in r:
        t = r["mtp"]
        print(f"\nMTP, {t['k']} drafts at {t['acceptance']:.0%} acceptance: {t['tokens_per_round']:.2f} "
              f"tokens per round, {t['round_ms']:.1f} ms per round = {t['tok_s']:.1f} tok/s "
              f"({t['speedup']:.2f}x plain)")
        if t["k"] > 3:
            print("  (above 3 drafts the model runs 10-15% fast: verification cost per column rises)")
    if "lanes" in r:
        n = r["lanes"]
        print(f"\n{n['rows']} LANES, plain batched round: {n['round_ms']:.1f} ms per round = "
              f"{n['tok_s']:.1f} tok/s aggregate, {n['per_stream_tok_s']:.1f} per stream "
              f"({n['distinct_experts_per_layer']:.1f} distinct experts per layer)")
        if n["rows"] > 2:
            print("  validated at two lanes only: the spike measured the hit rate falling at three "
                  "and four, which this model does not reproduce, so read this as an upper bound")
        if "crossover_acceptance" in n:
            c = n["crossover_acceptance"]
            mtp_k = r["mtp"]["k"]
            print("  " + (f"one lane with MTP{mtp_k} decodes faster than these lanes whenever its "
                          f"acceptance is above {c:.0%}; below that the lanes win decode throughput"
                          if c is not None else "even perfect acceptance does not reach these lanes"))
    if "levers" in r:
        print(f"\nLEVERS for {r['levers']['mode']} decode, ranked")
        for row in r["levers"]["rows"]:
            print(f"  {row['gain_pct']:+6.1f}%  {row['tok_s']:7.1f} tok/s  [{row['kind']:<6}] {row['name']}")
            print(f"{'':>34}{row['note']}")
    print("\nA model, not a measurement. Disagreement with a run on the card is the finding.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hit-rate", type=float, default=CAL.ref_hit_rate,
                        help="expert-cache decode hit rate; 97.2%% is a warm 8K run, 93%% a cold session")
    parser.add_argument("--mtp-k", type=int, default=0, help="MTP drafts per round; 0 = none")
    parser.add_argument("--acceptance", type=float,
                        help="accepted drafts / drafted tokens (default: the measured 8K value for K)")
    parser.add_argument("--lanes", type=int, default=1, help="plain lanes decoding together (the model "
                        "is validated at 2)")
    parser.add_argument("--tok-s", type=float, help="measured plain-decode tok/s at --hit-rate")
    parser.add_argument("--levers", action="store_true", help="rank what-ifs for the mode chosen")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not 0.5 <= args.hit_rate <= 1.0:
        parser.error("--hit-rate is a fraction between 0.5 and 1")
    if args.mtp_k < 0 or args.lanes < 1 or (args.acceptance is not None and not 0 <= args.acceptance <= 1):
        parser.error("--mtp-k >= 0, --lanes >= 1, --acceptance within [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
