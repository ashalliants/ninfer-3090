#!/usr/bin/env python3
"""A CPU model of long-prompt prefill time and of what the prefill scheduling options do to it.

It needs no GPU. It reads the RTX 3090 cost coefficients the Engine itself prices prefill with
(`src/runtime/engine/context_cache/context_cost_defaults.cpp`, fitted on the card with
`ninfer_context_cost_bench --suite prefill` under the launcher's prefill flags) and applies them:

    step(prefix, tokens) = chunk_ns + token_ns * tokens + attention_pair_ns * pairs
    pairs                = prefix * tokens + tokens * (tokens + 1) / 2

From that it prints, for a prompt of a given length:

  * where the time goes (the per-step fixed cost, the linear layers, attention) and the prompt's
    speed, for several chunk sizes; and
  * for the scheduling options (--prefill-stall-budget-ms, --prefill-decode-share,
    --prefill-layer-slices, --yield-long-prefill-ms) the PREDICTED prompt time, the longest gap a
    neighbouring stream sees, and how many tokens that stream gets while the prompt runs.

THIS IS A MODEL, NOT A MEASUREMENT. The coefficients are fitted at chunk 4096 on a 27B with the
cuBLAS prefill route, a step is assumed to cost what the fit says at any size (a small step really
pays a larger per-token price, which the fit's per-step term only partly captures), and a decode
round is a constant you supply. Use it to know what to expect before a run, to notice when a
measurement disagrees with the fit, and to choose settings worth measuring. Whenever it disagrees
with a measurement, the measurement is right and the disagreement is the finding.

    python tools/bench/prefill_time_model.py --prompt 100000
    python tools/bench/prefill_time_model.py --prompt 100000 --decode-round-ms 45 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_CPP = REPO_ROOT / "src/runtime/engine/context_cache/context_cost_defaults.cpp"
Q32 = float(1 << 32)
ALIGN = 128
MIN_PACED = 512  # kMinimumPacedPrefillTokens in src/runtime/engine/prefill_pacing.h

MODELS = {"27b": "kGroupwise27bPrefillSm86", "35b-a3b": "kGroupwise35bA3bPrefillSm86"}


@dataclass(frozen=True)
class Cost:
    chunk_ns: float
    token_ns: float
    pair_ns: float

    def step_ns(self, prefix: int, tokens: int) -> float:
        pairs = prefix * tokens + tokens * (tokens + 1) / 2
        return self.chunk_ns + self.token_ns * tokens + self.pair_ns * pairs

    def parts_ns(self, prefix: int, tokens: int) -> tuple[float, float, float]:
        pairs = prefix * tokens + tokens * (tokens + 1) / 2
        return self.chunk_ns, self.token_ns * tokens, self.pair_ns * pairs


def load_cost(model: str = "27b", path: Path = DEFAULTS_CPP) -> Cost:
    """The fitted prefill coefficients for `model`, read from the Engine's own source."""
    name = MODELS[model]
    text = path.read_text()
    block = re.search(r"constexpr ContextPrefillCost " + name + r"\s*\{(.*?)\};", text, re.S)
    if not block:
        raise RuntimeError(f"{name} not found in {path}")
    fields = {key: int(value.replace("'", "")) for key, value in
              re.findall(r"\.(\w+)\s*=\s*([\d']+)", block.group(1))}
    return Cost(
        chunk_ns=float(fields["chunk_ns"]),
        token_ns=fields["token_ns_q32"] / Q32,
        pair_ns=fields["attention_pair_ns_q32"] / Q32,
    )


# ---- a whole prompt -------------------------------------------------------------------------


def prefill_profile(cost: Cost, prompt: int, chunk: int, start: int = 0) -> dict:
    """Time of a prompt prefilled in full chunks from `start` cached tokens."""
    fixed = linear = attention = 0.0
    prefix, steps = start, 0
    while prefix < prompt:
        tokens = min(chunk, prompt - prefix)
        a, b, c = cost.parts_ns(prefix, tokens)
        fixed, linear, attention = fixed + a, linear + b, attention + c
        prefix += tokens
        steps += 1
    total = fixed + linear + attention
    return {
        "prompt": prompt, "chunk": chunk, "steps": steps,
        "seconds": total / 1e9, "tokens_per_second": (prompt - start) / (total / 1e9),
        "fixed_share": fixed / total, "linear_share": linear / total,
        "attention_share": attention / total,
    }


# ---- the scheduling options -----------------------------------------------------------------


def step_cap(cost: Cost, prefix: int, chunk: int, budget_ms: float, correction: float = 1.0) -> int:
    """prefill_pacing.h `PrefillPacing::step_cap`: 0 means the step is not cut."""
    if budget_ms <= 0 or chunk <= MIN_PACED:
        return 0
    budget = budget_ms * 1e6 / correction
    if cost.step_ns(prefix, chunk) <= budget:
        return 0
    low, high = MIN_PACED // ALIGN, (chunk - 1) // ALIGN
    if high < low:
        return 0
    while low < high:
        middle = low + (high - low + 1) // 2
        if cost.step_ns(prefix, middle * ALIGN) <= budget:
            low = middle
        else:
            high = middle - 1
    return low * ALIGN


@dataclass(frozen=True)
class Scenario:
    name: str
    budget_ms: float = 0.0
    share_percent: float = 0.0
    slices: int = 1


def run_scenario(cost: Cost, prompt: int, chunk: int, decode_round_ms: float, scenario: Scenario,
                 slice_overhead_ms: float = 0.2) -> dict:
    """Prompt time and what one neighbouring stream sees, with it decoding throughout.

    The stream gets one decode round per worker cycle. A cycle is a decode round and then one prefill
    turn: a whole step normally, or one slice of a sliced step. The decode share holds prefill turns
    back until decode rounds have used their share of the step's time. All times are in ms.
    """
    prefix, prompt_ms, stream_rounds, worst_gap = 0, 0.0, 0, 0.0
    d = decode_round_ms
    while prefix < prompt:
        planned = min(chunk, prompt - prefix)
        cap = step_cap(cost, prefix, planned, scenario.budget_ms)
        tokens = min(planned, cap) if cap else planned
        step_ms = cost.step_ns(prefix, tokens) / 1e6
        n = max(1, scenario.slices)
        # Each slice is a prefill turn preceded by a decode round; two residual copies per boundary.
        slice_ms = step_ms / n + (slice_overhead_ms if n > 1 else 0.0)
        for _ in range(n):
            prompt_ms += d + slice_ms
            stream_rounds += 1
            worst_gap = max(worst_gap, d + slice_ms)
        if scenario.share_percent > 0:
            # After the step, rounds repay share/(100-share) of its time before the next one.
            owed = step_ms * scenario.share_percent / (100.0 - scenario.share_percent)
            extra = int(owed // d)
            prompt_ms += extra * d
            stream_rounds += extra
            # The longest stall is still a slice; the burst is what the share adds.
        prefix += tokens
    seconds = prompt_ms / 1e3
    # The prompt's own work, without the neighbour's rounds, for the speed comparison.
    own = prefill_profile(cost, prompt, chunk)["seconds"]
    return {
        "scenario": scenario.name, "prompt_seconds": seconds, "own_work_seconds": own,
        "prompt_tokens_per_second": prompt / seconds, "worst_stream_gap_ms": worst_gap,
        "stream_rounds": stream_rounds,
        "stream_rounds_per_second": stream_rounds / seconds,
    }


def default_scenarios() -> list[Scenario]:
    return [
        Scenario("baseline (one round per step)"),
        Scenario("stall budget 500 ms", budget_ms=500),
        Scenario("stall budget 250 ms", budget_ms=250),
        Scenario("decode share 50%", share_percent=50),
        Scenario("slices 4", slices=4),
        Scenario("slices 8", slices=8),
        Scenario("slices 4 + decode share 50%", slices=4, share_percent=50),
        Scenario("budget 500 ms + decode share 50%", budget_ms=500, share_percent=50),
    ]


def yield_wait(cost: Cost, prompt: int, chunk: int, arrives_at_fraction: float) -> dict:
    """What a short request waits at one lane: the rest of the long prompt, or one step with yield."""
    arrive = int(prompt * arrives_at_fraction) // chunk * chunk
    rest = prefill_profile(cost, prompt, chunk, arrive)["seconds"]
    return {"arrives_after_tokens": arrive, "wait_without_yield_s": rest,
            "wait_with_yield_s": cost.step_ns(arrive, chunk) / 1e9}


# ---- output ---------------------------------------------------------------------------------


def report(args: argparse.Namespace) -> dict:
    cost = load_cost(args.model)
    out: dict = {
        "model": args.model,
        "coefficients": {"chunk_ms": cost.chunk_ns / 1e6, "token_us": cost.token_ns / 1e3,
                         "attention_pair_ns": cost.pair_ns},
        "profiles": [prefill_profile(cost, args.prompt, c) for c in args.chunks],
        "scenarios": [run_scenario(cost, args.prompt, args.chunk, args.decode_round_ms, s)
                      for s in default_scenarios()],
        "yield": yield_wait(cost, args.prompt, args.chunk, args.yield_arrival),
        "assumptions": {"decode_round_ms": args.decode_round_ms, "chunk": args.chunk,
                        "slice_overhead_ms": 0.2},
    }
    return out


def print_report(r: dict) -> None:
    c = r["coefficients"]
    print(f"MODEL, NOT A MEASUREMENT. {r['model']}: step = {c['chunk_ms']:.1f} ms + "
          f"{c['token_us']:.1f} us/token + {c['attention_pair_ns']:.2f} ns/attention-pair")
    print("\nWhere a prompt's time goes")
    print(f"{'prompt':>8} {'chunk':>6} {'steps':>5} {'seconds':>8} {'tok/s':>7} "
          f"{'fixed':>6} {'linear':>7} {'attention':>9}")
    for p in r["profiles"]:
        print(f"{p['prompt']:>8} {p['chunk']:>6} {p['steps']:>5} {p['seconds']:>8.1f} "
              f"{p['tokens_per_second']:>7.0f} {p['fixed_share']:>6.1%} {p['linear_share']:>7.1%} "
              f"{p['attention_share']:>9.1%}")
    a = r["assumptions"]
    print(f"\nScheduling options on a {r['profiles'][0]['prompt']:,}-token prompt, one neighbour "
          f"decoding ({a['decode_round_ms']:.0f} ms rounds, chunk {a['chunk']})")
    print(f"{'scenario':<36} {'prompt s':>9} {'tok/s':>7} {'worst gap ms':>13} {'stream rounds/s':>16}")
    for s in r["scenarios"]:
        print(f"{s['scenario']:<36} {s['prompt_seconds']:>9.1f} {s['prompt_tokens_per_second']:>7.0f} "
              f"{s['worst_stream_gap_ms']:>13.0f} {s['stream_rounds_per_second']:>16.1f}")
    y = r["yield"]
    print(f"\nA short request arriving after {y['arrives_after_tokens']:,} tokens at one lane waits "
          f"{y['wait_without_yield_s']:.1f} s without --yield-long-prefill-ms, "
          f"about {y['wait_with_yield_s']:.1f} s (one step) with it")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="27b")
    parser.add_argument("--prompt", type=int, default=100_000, help="prompt tokens")
    parser.add_argument("--chunk", type=int, default=4096, help="chunk for the scenarios")
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 2048, 4096, 8192],
                        help="chunk sizes for the profile table")
    parser.add_argument("--decode-round-ms", type=float, default=45.0,
                        help="one decode round of the neighbour (DFlash2/MTP: tens of ms)")
    parser.add_argument("--yield-arrival", type=float, default=0.2,
                        help="fraction of the prompt done when the short request arrives")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    # The simulation steps through the prompt a chunk at a time and divides by the prompt, the decode
    # round and the arrival point, so each of them has to be a usable number.
    if args.prompt <= 0:
        parser.error("--prompt must be positive")
    if args.chunk <= 0 or any(c <= 0 for c in args.chunks):
        parser.error("--chunk and --chunks must be positive")
    if args.decode_round_ms <= 0:
        parser.error("--decode-round-ms must be positive")
    if not 0.0 < args.yield_arrival < 1.0:
        parser.error("--yield-arrival must be strictly between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = report(args)
    if args.json:
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
