"""Is a rented GPU allowed to draw enough power? Used by vast-up.sh right after a box boots.

A Vast host can lower the card's enforced power limit, and the offer listing does not show it. The v0.17
benchmark box was a 3090 limited to 220 W of its 350 W default: under decode load the SM clock sat at
675-750 MHz of 2100 and the weight kernels read at 54-57% of the card's bandwidth, so every number
measured there was a capped-card number (docs/maintainer/perf-model.md, "This host is not a typical
3090"). The container can read the limit but not change it, so the only defence is to look after renting
and take another host.

Usage, on the output of
    nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader,nounits
piped to stdin:
    power_guard.py [--min-pct N]       (N defaults to 100; 0 turns the check off)
Exit 0: the limit is at least N% of the card's default limit (or the check is off).
Exit 1: the card is capped below that.
Exit 2: the limits could not be read, which is treated as a failure, because an unverified host is the
        case this guard exists for. `--min-pct 0` is the way to rent such a host anyway.
The one line it prints says what it found either way.
"""
import argparse
import sys
from typing import Optional, Tuple


def parse_limits(text: str) -> Optional[Tuple[float, float]]:
    """(power.limit, power.default_limit) in watts from nvidia-smi's CSV line, or None if unreadable."""
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            limit, default = float(fields[0]), float(fields[1])
        except ValueError:  # "[N/A]", "[Not Supported]", an error message
            continue
        if limit > 0 and default > 0:
            return limit, default
    return None


def percent(limit: float, default: float) -> int:
    """The limit as a whole percent of the default (round half up: 349 W of 350 W is 100%)."""
    return int(100.0 * limit / default + 0.5)


def check(text: str, min_pct: int) -> Tuple[int, str]:
    """(exit code, message) for nvidia-smi's output and the lowest acceptable percentage."""
    if min_pct <= 0:
        return 0, "power-limit check is off"
    limits = parse_limits(text)
    if limits is None:
        return 2, "could not read the power limit (nvidia-smi said: %r)" % text.strip()[:120]
    limit, default = limits
    pct = percent(limit, default)
    found = "power limit %.0f W of the %.0f W default (%d%%)" % (limit, default, pct)
    if pct < min_pct:
        return 1, "%s, below the required %d%%" % (found, min_pct)
    return 0, found


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-pct", type=int, default=100,
                        help="lowest acceptable power limit as a percent of the default (default 100; 0 = off)")
    args = parser.parse_args(argv)
    if not 0 <= args.min_pct <= 100:
        parser.error("--min-pct must be between 0 and 100")
    code, message = check(sys.stdin.read(), args.min_pct)
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
