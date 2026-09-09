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
| `decode-step-profile.ps1` | Where does a decode step's time go -- occupancy, launch gaps, or bandwidth? | a few min |
| `power-and-clocks.ps1` | Does the 315 W cap bound these numbers? | a few min |
| `admin-profile.ps1` | Everything needing an elevated shell: `ncu` counters, and what 350 W buys | a few min |
| `vision-encode-throughput.ps1` | What does Vision cost in time, and what does overlay residency cost? | 20 min |

`decode-roofline.ps1` needs no GPU: it reads the CSVs `kv-decode-vs-depth.ps1` leaves behind and
divides achieved bandwidth by peak. Run that sweep first. Note it only means anything for the
**dense** model -- applied to the A3B MoE it returns 391% of peak, which is not a result but a
demonstration that the formula does not hold there, since an MoE never reads its resident weights
per token. See TODO section 2c.

`vision-encode-throughput.ps1` needs a vision-capable artifact and nothing else. It resizes the
committed `bench/fixtures/ttft/media` photograph rather than generating synthetic images, because a
flat image compresses trivially and a random one defeats every cache, and neither is
representative. The headline it produces: **prefilling an image's tokens costs 4.4-13.7x what
encoding them does**, so the Vision tower is about 15% of time-to-first-token at 1024px and is not
where to look for a win. Watch for plateaus in the token count -- the preprocessor snaps to a
multiple of the merged patch size, so nearby resolutions can produce identical token counts.

**`admin-profile.ps1` must be run from an administrator shell** and refuses to start otherwise.
It is the only script here that does. `ncu` fails with `ERR_NVGPUCTRPERM` as a normal user because
GPU performance counters are administrator-only by default on Windows, and `nvidia-smi -pl`/`-lgc`
are refused outright. Running elevated satisfies the counter permission without the persistent
`RmProfilingAdminOnly=0` registry change and without a reboot, which is why the script exists
rather than an instruction to reconfigure the driver. It changes the power limit and clock lock and
restores both in a `finally` block, including on Ctrl+C; pass `-SkipPower` to leave the power limit
alone entirely.

`power-and-clocks.ps1` needs neither a model sweep nor `nsys`, only `nvidia-smi`: it samples power,
clocks and throttle reasons through one decode and one prefill. The answer as of 2026-09-09 is that
the cap binds in essentially every busy sample but the **memory clock never leaves 9,501 MHz**, so
the bandwidth figures these sweeps produce are not capped. Read the two phases separately -- decode
and prefill sit ~135 MHz apart on SM clock for the same power -- and ignore the maxima, which are
ramp-up spikes on a cold card.

`decode-step-profile.ps1` needs `nsys` (Nsight Systems) rather than a plain GPU run: it captures one
measured decode repetition per configuration and reports GPU-busy vs. idle time from the trace, plus
the top kernels by total time. Set `NINFER_NSYS` if it is not at the default install path.

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
