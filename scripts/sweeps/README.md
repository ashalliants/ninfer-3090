# KV and speculation sweeps

The PowerShell behind every number in TODO §2b, README's "Choosing a KV format" table, and
`docs/config-calculator.html`. Committed because §2b asks for re-measurement and reconstructing
these is an hour of work, not because they are polished tooling.

All of them expect a Release `build-ninja` and run from the repository root:

```powershell
$env:NINFER_MODEL_DIR = 'C:\Ninefer-3090\models'   # default; override for another box
$env:NINFER_SWEEP_OUT = 'profiles\sweeps'          # default; gitignored
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\sweeps\kv-decode-vs-depth.ps1
```

They print a CSV summary to stdout and leave per-run `ninfer_bench` CSVs and logs under
`$NINFER_SWEEP_OUT`. **Read the CSVs, not the console summary** — the summary is a convenience and
has been wrong before (see the note at the bottom).

Each takes a while and holds the GPU exclusively. Run one at a time; `kv-decode-vs-depth.ps1` took
about three hours for twelve model/format combinations.

## Which script answers which question

| script | question | roughly |
|---|---|---|
| `kv-decode-vs-depth.ps1` | How does each KV format's decode rate change with cache depth? | 3 h |
| `kv-decode-with-speculation.ps1` | Same, with MTP3 + draft head, capturing acceptance | 2 h |
| `kv-perplexity.ps1` | What does each KV format cost in perplexity? | 1 h |
| `memory-linearity-and-dflash-residency.ps1` | Is the non-KV sequence cost per-token or fixed? | 20 min |
| `dflash-residency-and-auto-capacity.ps1` | Does carrying DFlash cost VRAM when unused? | 15 min |
| `speculation-matrix.ps1` | What does each speculative backend cost and buy? | 1 h |
| `decode-roofline.ps1` | What fraction of the card's 936.2 GB/s does decode reach? | instant |

`decode-roofline.ps1` needs no GPU: it reads the CSVs `kv-decode-vs-depth.ps1` leaves behind and
divides achieved bandwidth by peak. Run that sweep first. Note it only means anything for the
**dense** model -- applied to the A3B MoE it returns 391% of peak, which is not a result but a
demonstration that the formula does not hold there, since an MoE never reads its resident weights
per token. See TODO section 2c.

## Things these got wrong, so you do not repeat them

**Decode must be measured at depth.** A `tg128` run seeds one token, so the cache is ~128 deep and
every KV format looks identical. Use `-pg P,128`: prefill *P* tokens, then time 128 decode steps on
top of that cache. Prefill and decode are timed separately, so the result is decode-only at a real
depth. This is the whole reason the sweeps take hours.

**`sequence_capacity_bytes` contains `kv_payload_bytes`.** They are not additive. Summing them
double-counts the entire cache.

**Never derive a per-token cost from one context length.** `sequence - kv_payload` at a single
context looks exactly like 4,063.5 B/token on the 27B. Measured at 8K/16K/32K/64K it resolves to
`166,438,656 + 0.0625 × ctx` — a *fixed* 158.7 MiB block plus a sixteenth of a byte per token. The
single-point reading understates memory by 127 MiB at 8K and overstates it by 349 MiB at 131K.
That is what `memory-linearity-and-dflash-residency.ps1` exists to settle.

**`ninfer_bench` cannot measure the automatic-sizing context.** With `-n 1` and no `-p`, auto-sizing
follows the *workload* and resolves `max_context` to **3**. It lands in the CSV looking like a real
answer. Use the serving path instead: `apps\ninfer --prompt hi --max-new 1 --kv-capacity auto`
prints `KV capacity` in its summary. Note the answer moves with free VRAM, so it describes the box
at that moment, not the format.

**The loader rejects any file not ending `.ninfer`.** The pre-DFlash 35B artifact was kept as
`.ninfer.pre-dflash.bak` and silently could not be loaded at all — "NInfer accepts only .ninfer
artifacts". It now lives as `qwen3_6_35b_a3b_v1_no_dflash.ninfer`.

**`kind` is `pp+tg`, not `prefill_decode`.** The original summary filter matched nothing, so these
scripts printed an empty table while writing perfectly good CSVs. Fixed here, but it is why the
advice above is to trust the CSVs.

**The engine will check your arithmetic.** Ask for a context that does not fit and the refusal names
the exact requirement: `minimum Engine runtime reservation requires 9197389568 bytes in addition to
1073741824 bytes of automatic headroom`. That equals
`kv_per_token × ctx + fixed sequence block + 0.0625 × ctx + workspace + graph allowance`, to the
byte. Cheaper than any probe, and it is the number the runtime actually applies.
