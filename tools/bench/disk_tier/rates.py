import re
import statistics
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("server-*.err")):
    pre, dec = [], []
    for line in f.read_text(errors="replace").splitlines():
        if " done | " not in line:
            continue
        m = re.search(r"prefill ([0-9.]+)(k?) tok/s", line)
        if m:
            pre.append(float(m.group(1)) * (1000 if m.group(2) else 1))
        m = re.search(r"decode ([0-9.]+) tok/s", line)
        if m:
            dec.append(float(m.group(1)))
    if pre:
        print(f"{f.stem}: n={len(pre)} prefill median={statistics.median(pre):.0f} tok/s "
              f"decode median={statistics.median(dec):.1f} tok/s (per request, from the server log)")
