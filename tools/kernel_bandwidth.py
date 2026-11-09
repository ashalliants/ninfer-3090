#!/usr/bin/env python3
"""Which kernels run well below the card's bandwidth? Joins an `nsys` kernel summary to the byte model.

`perf_model.py` knows how many bytes a decode step must stream; `scripts/sweeps/decode-step-profile.sh`
measures how long each kernel took. Dividing one by the other says, per kernel family, how close it
runs to the card's sustained read rate, and what bringing it to a good level would save. That is
where kernel work pays, and it is the part of the gap that a byte count alone cannot see.

    scripts/sweeps/decode-step-profile.sh 27b-decode          # on the card; writes the CSVs
    python tools/kernel_bandwidth.py profiles/sweeps/prof_27b-decode_cuda_gpu_kern_sum.csv \\
        --steps 128 --model 27b --recipe qwen3_8_27b --depth 8192

`--steps` is the number of decode steps (or speculative rounds) in the capture: 128 for `-n 128`.
For a speculative round pass `--columns K+1`, which scales the embedding rows and, for an MoE, the
experts a step touches.

Kernel names carry the stored weight format (`q4_...`, `q5_...`) but not the role, and fused ops
read several roles at once, so the join is by FORMAT, not by role: every byte of a q5 weight is read
by a `q5_` kernel. Routed-expert kernels (`sparse_moe_...`) are their own family. Attention kernels
are compared with the KV bytes at `--depth`, GDN recurrence kernels with the state read and write.
Everything else (norms, RoPE, quantize, reduce, copies) streams no weights and is listed as time only.

Limits. Times are the profiler's, which run about 5% long. The summed kernel time double-counts
kernels that overlap on different streams, so a family can look slower than it was. A quantize
kernel is counted as `other`, not with its GEMV, even though it is launched for it. Kernels the rules
do not recognise land in `other`; the biggest are printed so you can extend `classify`.

Cache and recurrence kernels are named after the format they store (`bf16_kv_grouped_mma_kernel`,
`nvfp4_gdn_conv_post_kernel`), so `classify` recognises them before the weight-format prefix rule. A
kernel fused over two stored formats (`q4_q5_...`) gets its own row with no byte claim, and the report
says the q4 and q5 rows then read high: a name cannot say which roles it streams.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_model as pm  # noqa: E402

WEIGHT_PREFIXES = ("q4", "q5", "q6", "q8", "t2", "nvfp4", "fp8", "bf16")
# Streams no weights, even when its name starts with a format: the activation quantise before a
# GEMV, a split-K reduction, a tile-preparation pass.
NOT_STREAMING = ("quantize", "reduce", "prepare", "merge")


def short_name(raw: str) -> str:
    """`void ninfer::ops::q4_a16_gemv_kernel<4, 128>(const T*, ...)` -> `q4_a16_gemv_kernel`."""
    depth, out = 0, []
    for ch in raw.split("(", 1)[0]:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    name = "".join(out).strip()
    if name.startswith("void "):
        name = name[5:]
    return name.rsplit("::", 1)[-1].strip()


def leading_formats(name: str) -> list[str]:
    """The weight formats a kernel name opens with: `q4_q5_attn_input_x` -> [q4, q5]."""
    found = []
    for token in name.split("_"):
        if token not in WEIGHT_PREFIXES:
            break
        found.append(token)
    return found


def classify(raw: str) -> str:
    """The kernel family a launch belongs to: a weight format (`q4`), a fused pair of formats
    (`q4_q5`), `moe`, `attention`, `gdn` or `other`.

    Order matters. Cache and recurrence kernels are named after the format they store
    (`bf16_kv_grouped_mma_kernel`, `nvfp4_kv_tiled_mma_kernel`, `nvfp4_gdn_conv_post_kernel`), so they
    are recognised BEFORE the format prefix rule, or their time would be joined to weight bytes."""
    name = short_name(raw)
    if any(word in name for word in NOT_STREAMING):
        return "other"
    if name.startswith("sparse_moe"):
        return "moe"
    if re.search(r"kv_(tiled|grouped)|kv_cache|attention", name):
        return "attention"
    if re.search(r"recurren|chunk_recurrence|gdn_conv|conv_post|gdn_projected|gdn_norm", name):
        return "gdn"
    formats = leading_formats(name)
    if formats:
        return "_".join(formats)
    if "gdn" in name:
        return "gdn"
    return "other"


def read_kernel_summary(path: Path) -> list[dict]:
    """Rows of `nsys stats --report cuda_gpu_kern_sum --format csv`."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"Total Time (ns)", "Instances", "Name"}
    if not rows or not needed <= set(rows[0]):
        raise SystemExit(f"{path} is not an nsys cuda_gpu_kern_sum CSV (wants {sorted(needed)})")
    return [{"name": row["Name"], "total_ns": float(row["Total Time (ns)"]),
             "instances": int(row["Instances"])} for row in rows]


def join(rows: Sequence[dict], steps: int, group_bytes: dict[str, float], bandwidth_gb_s: float,
         target_pct: float) -> dict:
    """Per family: time and launches per step, bytes, achieved rate, and the time bringing it to
    `target_pct` of the sustained read rate would save."""
    families: dict[str, dict] = {}
    for row in rows:
        family = classify(row["name"])
        entry = families.setdefault(family, {"ms": 0.0, "launches": 0.0, "kernels": {}})
        entry["ms"] += row["total_ns"] / steps / 1e6
        entry["launches"] += row["instances"] / steps
        kernel = short_name(row["name"])
        entry["kernels"][kernel] = entry["kernels"].get(kernel, 0.0) + row["total_ns"] / steps / 1e6
    table = []
    for family, entry in families.items():
        nbytes = group_bytes.get(family)
        line = {"family": family, "ms": entry["ms"], "launches": entry["launches"],
                "bytes": nbytes, "gb_s": None, "pct": None, "recoverable_ms": None,
                "top_kernel": max(entry["kernels"], key=entry["kernels"].get)}
        if nbytes:
            line["gb_s"] = nbytes / (entry["ms"] / 1e3) / 1e9
            line["pct"] = 100 * line["gb_s"] / bandwidth_gb_s
            good_ms = nbytes / (bandwidth_gb_s * 1e9 * target_pct / 100) * 1e3
            line["recoverable_ms"] = max(0.0, entry["ms"] - good_ms)
        table.append(line)
    table.sort(key=lambda r: -r["ms"])
    # A kernel fused over two stored formats (q4_q5_...) reads weights that the q4 and q5 rows count
    # as bytes, but its time is in its own row. Which roles it reads cannot be told from a name, so
    # say that those rows read low instead of guessing a split.
    fused = sorted(r["family"] for r in table if "_" in r["family"])
    caveats = [f"{family} kernels read more than one stored format: their time is listed on their own "
               f"row, so the {' and '.join(family.split('_'))} rows count their bytes but not their "
               f"time and read HIGH in GB/s" for family in fused]
    return {"families": table, "kernel_ms": sum(r["ms"] for r in table),
            "launches": sum(r["launches"] for r in table), "caveats": caveats}


def build(args: argparse.Namespace) -> dict:
    geo, label, recipe, weights, hardware, _ = pm.resolve(args)
    state_bytes = 2 if args.gdn_state_fp16 else 4
    step = pm.decode_step(geo, weights, tokens=args.columns, depth=args.depth, kv=args.kv,
                          kv_bpt=args.kv_bytes_per_token, gdn_state_bytes_per=state_bytes,
                          hardware=hardware)
    group_bytes = pm.bytes_by_group(weights, geo, args.columns)
    group_bytes["attention"] = step.kv
    group_bytes["gdn"] = step.state_rw
    joined = join(read_kernel_summary(args.csv), args.steps, group_bytes, hardware.read_gb_s,
                  args.target_pct)
    joined.update({"model": label, "recipe": recipe, "depth": args.depth, "steps": args.steps,
                   "columns": args.columns, "read_gb_s": hardware.read_gb_s,
                   "target_pct": args.target_pct, "floor_ms": step.floor_ms,
                   "predicted_ms": step.step_ms, "bytes_gb": step.total_bytes / 1e9})
    joined["recoverable_ms"] = sum(r["recoverable_ms"] or 0.0 for r in joined["families"])
    return joined


def print_report(r: dict) -> None:
    print(f"{r['model']} / {r['recipe']}, depth {r['depth']:,}, {r['steps']} steps of "
          f"{r['columns']} column(s); card {r['read_gb_s']:.1f} GB/s sustained read")
    print(f"  {'family':<10}{'ms/step':>9}{'launches':>10}{'GB':>8}{'GB/s':>8}{'% of bw':>9}"
          f"{'saves @' + format(r['target_pct'], '.0f') + '%':>11}   biggest kernel")
    for row in r["families"]:
        gb = f"{row['bytes'] / 1e9:8.3f}" if row["bytes"] else f"{'-':>8}"
        rate = f"{row['gb_s']:8.0f}{row['pct']:8.0f}%" if row["gb_s"] else f"{'-':>8}{'-':>9}"
        saves = f"{row['recoverable_ms']:10.2f}" if row["recoverable_ms"] is not None else f"{'-':>10}"
        flag = ("  <- above the card's read rate: check --steps, --model and --recipe"
                if row["pct"] and row["pct"] > 105 else "")
        print(f"  {row['family']:<10}{row['ms']:9.2f}{row['launches']:10.0f}{gb}{rate}{saves}   "
              f"{row['top_kernel'][:46]}{flag}")
    print(f"  {'total':<10}{r['kernel_ms']:9.2f}{r['launches']:10.0f}{r['bytes_gb']:8.3f}")
    print(f"\nkernel time {r['kernel_ms']:.2f} ms/step against a model bandwidth floor of "
          f"{r['floor_ms']:.2f} ms and a predicted step of {r['predicted_ms']:.2f} ms.")
    print(f"Bringing every family with bytes to {r['target_pct']:.0f}% of the sustained rate would "
          f"save about {r['recoverable_ms']:.2f} ms/step ({100 * r['recoverable_ms'] / r['kernel_ms']:.0f}%"
          f" of kernel time).")
    for caveat in r.get("caveats", []):
        print(f"CAVEAT: {caveat}")
    print("Times are the profiler's (about 5% long). Concurrent kernels double-count, so a family can "
          "look slower than it ran.")


def percent(text: str) -> float:
    value = pm.positive_float(text)
    if value > 100.0:
        raise argparse.ArgumentTypeError(f"{text} is not a percentage in (0, 100]")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="nsys stats cuda_gpu_kern_sum CSV")
    parser.add_argument("--steps", type=pm.positive_int, required=True,
                        help="decode steps (or rounds) in the capture: 128 for `ninfer_bench -n 128`")
    parser.add_argument("--columns", type=pm.positive_int, default=1,
                        help="tokens per step: 1 for plain decode, K+1 for a speculative round")
    parser.add_argument("--depth", type=pm.non_negative_int, default=0,
                        help="KV depth the capture ran at")
    parser.add_argument("--target-pct", type=percent, default=90.0,
                        help="what counts as a good kernel, in percent of the sustained read rate")
    pm.add_model_arguments(parser)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
