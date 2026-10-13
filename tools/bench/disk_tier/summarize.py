import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else "result-"
rows = []
for f in sorted(run.glob(prefix + "*.json")):
    d = json.loads(f.read_text())
    s = d["summary"]
    st = s.get("store", {})
    g = lambda k: st.get(k, 0) or 0
    rows.append((f.stem.replace(prefix, ""), s))
    print(f"{f.stem.replace(prefix,''):>5} n={s['requests']} err={s['errors']} wall={s['wall_s']:.0f}s hit={s['hit_rate']:.3f} "
          f"reprefill={s['reprefilled_tokens']} ttft p50={s['ttft_p50']:.2f} p95={s['ttft_p95']:.2f} p99={s['ttft_p99']:.2f} "
          f"mean={s['ttft_mean']:.2f} dec/stream={s['decode_tok_s_per_stream']:.1f} "
          f"wrGB/h={s['store_bytes_written_per_hour']/1e9:.1f} writes={g('context_store_writes_total'):.0f} "
          f"evw={g('context_store_eviction_writes_total'):.0f} evs={g('context_store_eviction_export_seconds_total'):.2f} "
          f"shw={g('context_store_shared_writes_total'):.0f} hyd={g('context_store_hydrations_total'):.0f} "
          f"hydtok={g('context_store_hydrated_tokens_total'):.0f} hydfail={g('context_store_hydration_failures_total'):.0f} "
          f"hyd_s={g('context_store_hydration_seconds_total'):.1f} dropped={g('context_store_dropped_total'):.0f} "
          f"rate={(s.get('read_rate') or 0)/1e9:.2f}GB/s")
