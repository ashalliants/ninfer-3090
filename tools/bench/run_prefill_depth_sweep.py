#!/usr/bin/env python3
"""Measure prefill speed against prompt depth on a real card, and say where the cost model is wrong.

Runs `ninfer_bench` once over a list of prompt lengths (one model load), then compares each
measured prefill time with tools/bench/prefill_time_model.py (the Engine's own fitted costs) and
fits two scale factors by least squares: the per-step-and-per-token cost together, and the attention
(per-pair) cost. A fitted attention scale of 1.2 means attention at depth costs 20% more than
the fit says. That is the number that decides whether long-context prefill attention is worth tuning:
at 100K tokens attention is about 42% of the prompt's time on the 27B, so a 20% attention gain is
about 8% off the prompt. Needs a GPU and an artifact; run it under the GPU lock (CLAUDE.md).

    python tools/bench/run_prefill_depth_sweep.py \\
        --ninfer-bench build-ninja/bench/ninfer_bench.exe --weights out/qwen3_8_27b.ninfer \\
        --lengths 4096,16384,32768,65536,98304 --chunk 4096 \\
        -- --kv-dtype rk4v4 --prefill-cublas --gdn-state-fp16

Arguments after `--` go to ninfer_bench unchanged (match the serving flags you care about).
`--print-attention-commands` prints the op-bench commands for the attention kernel at the chunk's
width, for a kernel-level look at the same depths.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefill_time_model as model  # noqa: E402


def fit_scales(rows: list[dict]) -> dict:
    """Least-squares scales on the Engine's coefficients: `base` (the per-step and per-token costs
    together) and `attention` (the per-pair cost).

    At one chunk size the per-step and per-token terms grow in proportion to each other across
    prompts, so they cannot be told apart from this data; they are fitted as one. (Sweep --chunk to
    separate them.) Relative error is minimised, so the 100K prompt does not drown the 4K one. A
    single prompt length cannot separate the two scales, so then only attention is fitted.
    """
    base = [r["fixed_s"] + r["linear_s"] for r in rows]
    attn = [r["attention_s"] for r in rows]
    measured = [r["measured_s"] for r in rows]
    if len({r["prompt"] for r in rows}) >= 2:
        # Minimise sum((b*base_i + a*attn_i) / t_i - 1)^2: 2x2 normal equations.
        w = [(x / t, y / t) for x, y, t in zip(base, attn, measured)]
        bb = sum(u * u for u, _ in w)
        ba = sum(u * v for u, v in w)
        aa = sum(v * v for _, v in w)
        rb = sum(u for u, _ in w)
        ra = sum(v for _, v in w)
        det = bb * aa - ba * ba
        if abs(det) > 1e-18:
            return {"base": (rb * aa - ra * ba) / det, "attention": (ra * bb - rb * ba) / det,
                    "free": ["base", "attention"]}
    num = sum((t - x) * y / t**2 for x, y, t in zip(base, attn, measured))
    den = sum((y / t) ** 2 for y, t in zip(attn, measured))
    return {"base": 1.0, "attention": num / den if den else float("nan"), "free": ["attention"]}


def parse_bench_json(path: Path) -> dict[int, float]:
    """prompt tokens -> mean prefill seconds, from ninfer_bench's JSON for its pp tests."""
    document = json.loads(path.read_text())
    out: dict[int, float] = {}
    for test in document.get("tests", []):
        if test.get("kind") not in ("pp", "prompt", "prefill") and not str(test.get("label", "")).startswith("pp"):
            continue
        if test.get("n_gen", 0) not in (0, 1) and "+" in str(test.get("label", "")):
            continue
        seconds = test.get("prefill_seconds_mean")
        if seconds:
            out[int(test["n_prompt"])] = float(seconds)
    return out


def analyse(measured: dict[int, float], chunk: int, model_name: str) -> dict:
    cost = model.load_cost(model_name)
    rows = []
    for prompt in sorted(measured):
        profile = model.prefill_profile(cost, prompt, chunk)
        fixed = linear = attention = 0.0
        prefix = 0
        while prefix < prompt:
            tokens = min(chunk, prompt - prefix)
            a, b, c = cost.parts_ns(prefix, tokens)
            fixed, linear, attention = fixed + a, linear + b, attention + c
            prefix += tokens
        rows.append({
            "prompt": prompt, "measured_s": measured[prompt], "model_s": profile["seconds"],
            "ratio": measured[prompt] / profile["seconds"],
            "measured_tokens_per_second": prompt / measured[prompt],
            "fixed_s": fixed / 1e9, "linear_s": linear / 1e9, "attention_s": attention / 1e9,
        })
    scales = fit_scales(rows)
    for row in rows:
        fitted = scales["base"] * (row["fixed_s"] + row["linear_s"]) + scales["attention"] * row["attention_s"]
        row["fitted_attention_share"] = scales["attention"] * row["attention_s"] / fitted
    return {"chunk": chunk, "model": model_name, "rows": rows, "scales": scales}


def print_analysis(result: dict) -> None:
    print(f"{'prompt':>8} {'measured s':>11} {'model s':>8} {'ratio':>6} {'tok/s':>7} {'attn share (fitted)':>20}")
    for r in result["rows"]:
        flag = "  <-- model off by >10%" if abs(r["ratio"] - 1) > 0.10 else ""
        print(f"{r['prompt']:>8} {r['measured_s']:>11.2f} {r['model_s']:>8.2f} {r['ratio']:>6.2f} "
              f"{r['measured_tokens_per_second']:>7.0f} {r['fitted_attention_share']:>20.1%}{flag}")
    s = result["scales"]
    print(f"\nFitted scale on the Engine's coefficients (1.00 = the fit holds); free: {', '.join(s['free'])}")
    print(f"  per-step and per-token together {s['base']:.2f}   attention-per-pair {s['attention']:.2f}")
    if s["attention"] > 1.10:
        print("Attention at depth costs more than the fit says: the long-context attention path is "
              "worth a kernel-level look (--print-attention-commands).")
    elif s["attention"] < 0.90:
        print("Attention at depth is cheaper than the fit says; the Engine's cost model is pessimistic.")
    else:
        print("The fit holds at these depths; attention is not hiding an extra cost.")


def attention_commands(chunk: int, lengths: Sequence[int]) -> list[str]:
    contexts = ",".join(str(max(0, n - chunk)) for n in lengths if n > chunk)
    base = ("ninfer_benches ninfer_causal_softmax_attention_bench --entry append --geometry d256-h24-kv4 "
            f"--kv-dtype rk4v4 --batch 1 --tokens {chunk} --context {contexts} --execution eager "
            "--cache cold --warmup 5 --repeat 21")
    return [base, base.replace("rk4v4", "int8"), base.replace("d256-h24-kv4", "d256-h16-kv2")]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--ninfer-bench", type=Path, help="the ninfer_bench executable")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--lengths", default="4096,16384,32768,65536,98304")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--model", choices=sorted(model.MODELS), default="27b")
    parser.add_argument("-r", "--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, help="write the analysis as JSON")
    parser.add_argument("--from-json", type=Path, help="analyse an existing ninfer_bench JSON instead of running")
    parser.add_argument("--print-attention-commands", action="store_true")
    parser.add_argument("bench_args", nargs=argparse.REMAINDER, help="after --: flags for ninfer_bench")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    if args.print_attention_commands:
        print("\n".join(attention_commands(args.chunk, lengths)))
        return 0
    extra = [x for x in args.bench_args if x != "--"]
    if args.from_json:
        measured = parse_bench_json(args.from_json)
    else:
        if not args.ninfer_bench or not args.weights:
            raise SystemExit("--ninfer-bench and --weights are required (or --from-json)")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "bench.json"
            command = [str(args.ninfer_bench), "--weights", str(args.weights),
                       "-p", ",".join(map(str, lengths)), "-r", str(args.repetitions),
                       "--warmup", str(args.warmup), "--prefill-chunk", str(args.chunk),
                       "--max-ctx", str(max(lengths) + 1024), "-o", "json", "--output-file", str(out),
                       *extra]
            print("+", " ".join(command), file=sys.stderr)
            subprocess.run(command, check=True)
            measured = parse_bench_json(out)
    if not measured:
        raise SystemExit("no pp results found in the benchmark output")
    result = analyse(measured, args.chunk, args.model)
    print_analysis(result)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
