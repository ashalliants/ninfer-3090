import json
import sys
from pathlib import Path

invalid = 0
for f in sorted(Path(sys.argv[1]).glob("cont-*.json")):
    rows = json.loads(f.read_text())
    failed = [r for r in rows if r.get("error") or r.get("ttft") is None]
    if failed:
        # A run with a failed follow-up is not a measurement; never summarise it as one.
        invalid += 1
        print(f.stem, f"INVALID: {len(failed)} of {len(rows)} continuations failed")
        continue
    print(f.stem, " ".join(f"{r['prompt']}:{r['cached']}:{r['ttft']:.2f}" for r in rows))
sys.exit(1 if invalid else 0)
