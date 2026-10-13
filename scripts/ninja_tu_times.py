#!/usr/bin/env python3
"""Rank translation units by compile wall time from a Ninja build's .ninja_log.

Usage: scripts/ninja_tu_times.py <build-dir> [--top N] [--min-seconds S]

Wall time is inflated by contention when many jobs share the cores, so compare TUs within one
build, not across machines. For a clean per-TU cost, rebuild one object alone with
`ninja -C <build-dir> <object>`. Only the last entry per output is used (an interrupted or
repeated build appends several).
"""
import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("build_dir", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--min-seconds", type=float, default=0.0)
    args = parser.parse_args()

    log = args.build_dir / ".ninja_log"
    if not log.is_file():
        print(f"no .ninja_log in {args.build_dir}", file=sys.stderr)
        return 1

    last: dict[str, float] = {}
    for line in log.read_text().splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        start_ms, end_ms, output = int(fields[0]), int(fields[1]), fields[3]
        if output.endswith((".o", ".obj")):
            last[output] = (end_ms - start_ms) / 1000.0

    rows = sorted(((s, o) for o, s in last.items() if s >= args.min_seconds), reverse=True)
    total = sum(last.values())
    print(f"{len(last)} objects, {total / 60:.0f} min summed wall time")
    for seconds, output in rows[: args.top]:
        print(f"{seconds:8.0f}s  {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
