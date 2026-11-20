"""Teacher-forced comparison of two Flash-Next builds' logit dumps over the same fixed texts.

    python -m tools.flash_next.compare_dumps --ids tests/fixtures/qwen4_exp --a DIR_A --b DIR_B

Each DIR holds, per set and text, ``<set>-<name>.bin``: the ``--dump-logits`` output of
``ninfer_qwen4_exp_forward_real_test`` run on that build's artifact (tests/README.md describes it), in the
layout ``strata_compare`` reads. ``--ids`` holds the token ids those dumps scored (``<set>-<name>.ids``,
comma-separated, as in tests/fixtures/qwen4_exp). Position ``t`` of a dump predicts ``ids[t + 1]``.

Per text, per set and overall it reports each build's perplexity and top-1 next-token accuracy, the
paired mean NLL difference (B - A) with its standard error, how often the two top tokens agree, and the
mean KL(A || B) over the whole vocabulary. Negative dNLL means B models the text better. This ranks two
builds against each other on the given texts; with no full-precision reference it says nothing of either
one's distance from the original model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .strata_compare import engine_rows, log_softmax

DEFAULT_SETS = ("ppl_tokens", "infernix_texts")
DEFAULT_NAMES = ("code", "doc", "chat")


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split(",") if x.strip()]


def score(ids: list[int], rows_a, rows_b) -> dict:
    """Per-position results of two dumps over one text (position t predicts ids[t + 1])."""
    n = len(ids) - 1
    if n < 2:
        raise ValueError("a text needs at least three tokens")
    if rows_a.shape[0] < n or rows_b.shape[0] < n:
        raise ValueError(f"the dumps hold {rows_a.shape[0]} and {rows_b.shape[0]} rows for {n} predictions")
    if rows_a.shape[1] != rows_b.shape[1]:
        raise ValueError("the two dumps have different vocabularies")
    vocab = rows_a.shape[1]
    bad = [i for i in ids[1:n + 1] if not 0 <= i < vocab]
    if bad:
        raise ValueError(f"target token id {bad[0]} is outside the dumps' {vocab}-token vocabulary")
    nll_a, nll_b, kl, top_a, top_b, agree = [], [], [], 0, 0, 0
    for t in range(n):
        la, lb = log_softmax(np.asarray(rows_a[t])), log_softmax(np.asarray(rows_b[t]))
        target = ids[t + 1]
        nll_a.append(-la[target])
        nll_b.append(-lb[target])
        kl.append(float((np.exp(la) * (la - lb)).sum()))
        ta, tb = int(la.argmax()), int(lb.argmax())
        top_a += ta == target
        top_b += tb == target
        agree += ta == tb
    return {"nll_a": nll_a, "nll_b": nll_b, "kl": kl, "top_a": top_a, "top_b": top_b, "agree": agree}


def combine(parts: list[dict]) -> dict:
    out = {"nll_a": [], "nll_b": [], "kl": [], "top_a": 0, "top_b": 0, "agree": 0}
    for part in parts:
        for key in ("nll_a", "nll_b", "kl"):
            out[key] += part[key]
        for key in ("top_a", "top_b", "agree"):
            out[key] += part[key]
    return out


def summarize(result: dict) -> dict:
    a, b = np.array(result["nll_a"]), np.array(result["nll_b"])
    n = len(a)
    diff = b - a
    return {
        "positions": n,
        "ppl_a": math.exp(a.mean()),
        "ppl_b": math.exp(b.mean()),
        "dnll": float(diff.mean()),
        "dnll_se": float(diff.std(ddof=1) / math.sqrt(n)),
        "top1_a": result["top_a"] / n,
        "top1_b": result["top_b"] / n,
        "top1_agree": result["agree"] / n,
        "kl_a_to_b": float(np.mean(result["kl"])),
    }


def format_row(label: str, s: dict) -> str:
    return (f"{label:<24}{s['positions']:>6}  ppl A {s['ppl_a']:>8.4f}  ppl B {s['ppl_b']:>8.4f}  "
            f"dNLL(B-A) {s['dnll']:>+8.4f} +-{s['dnll_se']:.4f}  top1 A {100 * s['top1_a']:5.1f}% "
            f"B {100 * s['top1_b']:5.1f}%  agree {100 * s['top1_agree']:5.1f}%  KL(A||B) {s['kl_a_to_b']:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", type=Path, required=True, help="folder of <set>-<name>.ids")
    parser.add_argument("--a", type=Path, required=True, help="folder of build A's <set>-<name>.bin dumps")
    parser.add_argument("--b", type=Path, required=True, help="folder of build B's dumps")
    parser.add_argument("--sets", default=",".join(DEFAULT_SETS))
    parser.add_argument("--names", default=",".join(DEFAULT_NAMES))
    parser.add_argument("--json", type=Path, help="write the numbers here")
    args = parser.parse_args(argv)
    everything, report = [], {}
    for text_set in args.sets.split(","):
        in_set = []
        for name in args.names.split(","):
            stem = f"{text_set}-{name}"
            result = score(read_ids(args.ids / f"{stem}.ids"), engine_rows(args.a / f"{stem}.bin"),
                           engine_rows(args.b / f"{stem}.bin"))
            in_set.append(result)
            report[stem] = summarize(result)
            print(format_row(f"{text_set} {name}", report[stem]))
        report[f"{text_set} ALL"] = summarize(combine(in_set))
        print(format_row(f"{text_set} ALL", report[f"{text_set} ALL"]))
        everything += in_set
    report["EVERYTHING"] = summarize(combine(everything))
    print(format_row("EVERYTHING", report["EVERYTHING"]))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
