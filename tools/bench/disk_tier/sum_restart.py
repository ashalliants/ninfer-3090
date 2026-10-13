import json
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("cont-*.json")):
    rows = json.loads(f.read_text())
    print(f.stem, " ".join(f"{r['prompt']}:{r['cached']}:{r['ttft']:.2f}" for r in rows))
