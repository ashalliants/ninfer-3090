"""Summarise ``ninfer-serve`` benchmark runs: decode and prefill rates per prompt length, from the log.

    python -m tools.flash_next.bench_summary --dir DIR LABEL [LABEL ...]

DIR holds ``LABEL.err.log``: the stderr of one ``ninfer-serve`` run that was sent the request sequence
below. The server logs one line per request::

    [qwen3.8-flash-next] request: 940 prompt + 512 generated tokens; prefill 0.84 s, decode 6.1 s (84.2 tok/s); ...

The run is one request the server makes for itself at startup, ``--warmups`` (12) requests that fill the
expert cache, then ``--reps`` requests at each of the three prompt lengths. Prefill tok/s is the prompt
tokens over the server's own prefill time (a fresh prompt every request, so no prefix reuse helps); decode
tok/s is the server's own figure for the request. Medians with [min-max] over the repetitions.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

REQUEST = re.compile(
    r"request: (\d+) prompt \+ (\d+) generated tokens; prefill ([\d.]+) s, decode ([\d.]+) s \(([\d.]+) tok/s\)"
)
LENGTHS = ("1K", "8K", "30K")
# The prompt sizes (tokens) each group may have: the nominal 1K / 8K / 30K prompts tokenise to a little
# under or over their name, so these are bands, not exact values.
PROMPT_BANDS = {"1K": (700, 1500), "8K": (6000, 10000), "30K": (24000, 36000)}
SERVER_STARTUP_REQUESTS = 1  # the warm-up request ninfer-serve makes for itself


def parse_requests(text: str) -> list[tuple[int, int, float, float, float]]:
    """(prompt tokens, generated tokens, prefill s, decode s, decode tok/s) per logged request."""
    found = []
    for line in text.splitlines():
        match = REQUEST.search(line)
        if match:
            prompt, generated, prefill, decode, rate = match.groups()
            found.append((int(prompt), int(generated), float(prefill), float(decode), float(rate)))
    return found


def summarize(requests: list, warmups: int = 12, reps: int = 5) -> dict[str, dict]:
    """Per prompt length: median and range of the decode and prefill rates, and the typical sizes."""
    measured = requests[SERVER_STARTUP_REQUESTS + warmups:]
    if len(measured) != reps * len(LENGTHS):
        raise ValueError(
            f"expected {reps * len(LENGTHS)} measured requests after {SERVER_STARTUP_REQUESTS + warmups} "
            f"warm-ups, found {len(measured)} (of {len(requests)} logged)"
        )
    out = {}
    for index, length in enumerate(LENGTHS):
        group = measured[index * reps:(index + 1) * reps]
        low, high = PROMPT_BANDS[length]
        wrong = [r[0] for r in group if not low <= r[0] <= high]
        if wrong:
            raise ValueError(
                f"the {length} group should hold prompts of {low}-{high} tokens but has {wrong}: "
                "a request is missing, duplicated or out of order"
            )
        decode = [r[4] for r in group]
        prefill = [r[0] / r[2] for r in group]
        out[length] = {
            "prompt_tokens": int(statistics.median(r[0] for r in group)),
            "generated_tokens": int(statistics.median(r[1] for r in group)),
            "decode": (statistics.median(decode), min(decode), max(decode)),
            "prefill": (statistics.median(prefill), min(prefill), max(prefill)),
        }
    return out


def _fmt(values: tuple[float, float, float], places: int = 1) -> str:
    median, low, high = values
    return f"{median:.{places}f} [{low:.{places}f}-{high:.{places}f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, required=True, help="the folder holding LABEL.err.log")
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args(argv)
    for label in args.labels:
        requests = parse_requests((args.dir / f"{label}.err.log").read_text(errors="replace"))
        print(f"== {label}: {len(requests)} requests logged")
        for length, row in summarize(requests, args.warmups, args.reps).items():
            print(f"  {length:>3}: prompt {row['prompt_tokens']:>6} tok, generated {row['generated_tokens']:>4}, "
                  f"prefill {_fmt(row['prefill'], 0)} tok/s, decode {_fmt(row['decode'])} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
