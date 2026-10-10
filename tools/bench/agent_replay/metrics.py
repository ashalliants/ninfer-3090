"""Copy-overlap metric and per-category aggregation for agent replay samples."""

from __future__ import annotations

import re
import statistics
from typing import Any, Iterable

from tools.bench.agent_replay.workload import CATEGORIES, COPY_CATEGORIES, strip_line_numbers

# Word-ish units: identifiers/numbers, or one punctuation character. Whitespace is ignored, so
# re-indented copies still match; `cat -n` prefixes are stripped from sources and outputs.
UNIT = re.compile(r"\w+|[^\w\s]")
OVERLAP_N = 8
SHORT_TURN_TOKENS = 128


def units(text: str) -> list[str]:
    return UNIT.findall(strip_line_numbers(text))


class SourceIndex:
    """Every OVERLAP_N-unit window of the text a session's prompt has contained so far."""

    def __init__(self, n: int = OVERLAP_N) -> None:
        self.n = n
        self._grams: set[int] = set()

    def add(self, text: str) -> None:
        seq = units(text)
        n = self.n
        self._grams.update(hash(tuple(seq[i:i + n])) for i in range(len(seq) - n + 1))

    def overlap(self, text: str) -> tuple[int, int]:
        """(units covered by some n-gram present in the sources, total output units)."""
        seq = units(text)
        n = self.n
        covered = [False] * len(seq)
        for i in range(len(seq) - n + 1):
            if hash(tuple(seq[i:i + n])) in self._grams:
                for j in range(i, i + n):
                    covered[j] = True
        return sum(covered), len(seq)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    # Linear interpolation between order statistics (numpy's default); 0.5 is the median.
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def flatten_counters(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Collect numeric leaves (ints/floats and numeric lists) under dotted names.

    Generic on purpose: new server counters such as `ngram_*` appear without code changes.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten_counters(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, list) and value and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
    ):
        out[prefix] = list(value)


def _sum_counters(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for sample in samples:
        for key, value in (sample.get("counters") or {}).items():
            if isinstance(value, list):
                current = total.get(key) or []
                size = max(len(current), len(value))
                total[key] = [
                    (current[i] if i < len(current) else 0) + (value[i] if i < len(value) else 0)
                    for i in range(size)
                ]
            else:
                total[key] = total.get(key, 0) + value
    return total


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def summarize_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in samples if s.get("error") is None]
    out: dict[str, Any] = {"turns": len(samples), "errors": len(samples) - len(ok)}
    if not ok:
        return out
    output_tokens = sum(s["output_tokens"] for s in ok)
    decode_seconds = sum(s["decode_s"] for s in ok if s.get("decode_s"))
    timed_tokens = sum(s["output_tokens"] for s in ok if s.get("decode_s"))
    ttfts = [s["ttft_s"] for s in ok if s.get("ttft_s") is not None]
    rates = [s["decode_tps"] for s in ok if s.get("decode_tps")]
    covered = sum(s["overlap_covered"] for s in ok)
    words = sum(s["overlap_units"] for s in ok)
    counters = _sum_counters(ok)
    out.update(
        prompt_tokens_mean=statistics.fmean(s["prompt_tokens"] for s in ok),
        prompt_tokens_max=max(s["prompt_tokens"] for s in ok),
        prompt_tokens_sum=sum(s["prompt_tokens"] for s in ok),
        cached_tokens_sum=sum(s["cached_tokens"] for s in ok),
        output_tokens_sum=output_tokens,
        output_tokens_mean=output_tokens / len(ok),
        length_capped_turns=sum(1 for s in ok if s.get("finish_reason") == "length"),
        # A copy turn this short means the model did something else (usually another tool call).
        short_turns=sum(1 for s in ok if s["output_tokens"] < SHORT_TURN_TOKENS),
        ttft_s_p50=percentile(ttfts, 0.5),
        ttft_s_mean=statistics.fmean(ttfts) if ttfts else None,
        ttft_s_max=max(ttfts) if ttfts else None,
        decode_s_sum=decode_seconds,
        # Token-weighted: committed output tokens over decode wall time.
        decode_tps=_ratio(timed_tokens, decode_seconds),
        decode_tps_p50=percentile(rates, 0.5),
        decode_tps_min=min(rates) if rates else None,
        copy_overlap=_ratio(covered, words),
        counters=counters,
    )
    drafted = counters.get("speculative.drafted_tokens")
    accepted = counters.get("speculative.accepted_tokens")
    rounds = counters.get("speculative.rounds")
    out["spec_acceptance"] = _ratio(accepted, drafted)
    out["spec_tokens_per_round"] = _ratio((accepted or 0) + (rounds or 0), rounds) if rounds else None
    if "timings.draft_n" in counters:
        out["usage_draft_acceptance"] = _ratio(counters.get("timings.draft_n_accepted"),
                                               counters.get("timings.draft_n"))
    # Copy-drafting counters added by later PRs, reported as shares of committed output.
    for key, value in counters.items():
        if "ngram" in key and key.endswith("accepted_tokens") and isinstance(value, (int, float)):
            out[f"{key}.share_of_output"] = _ratio(value, output_tokens)
    return out


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_category = {
        category: summarize_group([s for s in samples if s["category"] == category])
        for category in CATEGORIES
        if any(s["category"] == category for s in samples)
    }
    return {
        "by_category": by_category,
        "copy": summarize_group([s for s in samples if s["category"] in COPY_CATEGORIES]),
        "non_copy": summarize_group([s for s in samples if s["category"] not in COPY_CATEGORIES]),
        "all": summarize_group(samples),
    }
