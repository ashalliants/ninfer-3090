# Qwen3.8-Flash-Next on the RTX 3090: working plan

Temporary. This file tracks the staged work to run Qwen3.8-Flash-Next (GGUF architecture
`qwen4exp`) from the ISTA DASLab GSQ-RCO IQ2_XS GGUF. Delete it when the last milestone lands and
its stable facts have moved to their references (a `qwen4-exp-model.md`, the format and layout
references, `engine-architecture.md`).

## Status

| PR | Scope | Status |
|---|---|---|
| 0 | Strata baselines: decode, MTP, prefill, teacher-forced log-probs on frozen token ids | done (measurement; token sets in `tests/fixtures/qwen4_exp/`) |
| 1 | GGUF reader, nine exact GGML block formats, `ggml_blocks_v1` / `ggml_rows_page4k_v1`, `qwen4_exp` name map, tokenizer synthesis, `--subset dev` | done (Python only) |
| 2 | C++ registration and materialization of the GGML formats and layouts; host exact decoder | done |
| 2b | Infernix-aligned artifact: `ggml_expert_record_v1` expert banks and three record formats (Python and C++), the n-gram table as an Infernix-format volume with IQ4_NL rows, Infernix's text config keys; replaces PR 1's two-object banks and `ggml_rows_page4k_v1` | done |
| 3 | Dense GGML linears: `linear`/`linear_add` for IQ4_XS, IQ3_S, Q6_K, IQ4_NL, Q8_0, Q2_0 at the model's 19 dense problems, Q8_1 activation profile | done (perf gates below) |
| 4 | PLE frontend and stream residency | not started |
| 5 | QSA Ops from Infernix: index query, pooled keys, block selection, attention (Int8Group64) | done |
| 6 | Ops from Infernix: `hyper_connection` (mix, inject, expand), `ple` (IQ4_NL `ple_embed`, gate, stateful convolution, commit), `projection_fp32` (BF16 router; IQ4_XS head on PR 3's decode route), `rows` | done (HC perf gate missed, below) |
| 7a | `offloaded_sparse_moe` contract, routing, dispatch, combine, canonical A8 arithmetic, CPU expert engine (scalar, AVX2, worker team) | done |
| 7b | `offloaded_sparse_moe` GPU narrow route (device frames, staging, zero-copy), CPU miss channel and service | done (perf gate missed, see below) |
| 8 | M0: the qwen4exp model from Infernix (config, binding, forward, n-gram hash, n-gram volume reader with keep-alive), the full conversion, the FP64 reference and block check, `forward_real`; the quality gate | done (gate below) |
| 9 | M1: Program and Engine integration (one lane, elastic VMM KV, VRAM expert cache with LFRU and warm start, RAM/VRAM ledgers, decode CUDA graph, `EngineCore<Qwen4ExpInstance>`), serving and the `qwen38-flash-next` launcher profile | done (measurements below) |
| 10 | Prefill: exact GPU wide expert route (int8 `mma.sync`, cp.async), prefill expert streaming into lent frames, 4,096-token calls, n-gram row prefetch | done (prefill gate missed by 3-5%, below) |
| 11 | Decode (M2 gate): few misses zero-copy (11a), fewer decode-graph nodes and the GGML `linear_pair` (11b), the decode graph split around the n-gram row read and the startup frame retry (11c) | done (measurements below) |
| 12 | 128K through the elastic KV: growth takes expert frames before it maps (12a, the crash and 128K prefill fix); prefill lending restores the experts it displaced and is sized to the prompt (12b, the decode fix) | done (measurements below) |
| 13 | MTP speculation: the drafter imported exactly from Strata's `mtp-q2_0.gguf` (13a, `--mtp-gguf`, `ggml_rec_q2_0_q2_0`); Infernix's chained drafts, exact W-column verification and record commit (13b); the draft ceiling and the launcher (13c) | done (spec gate missed, below) |
| 14 | Vision: the tower imported exactly from the model's mmproj GGUF (14a, `--mmproj`, Qwen3.5's vision component); bound pinned in host RAM and encoded through Qwen3.5's streamed tower in a lent window, against an FP64 oracle (14b); M-RoPE positions, the embedding scatter (text and drafter), the Engine, serving and the launcher (14c) | done (below) |
| 15 | Prefix reuse: NInfer's context cache contract in this Program with a Host-only tier sized by the RAM ledger, on by default in the launcher ([design and measurements](flash-next-context-cache.md)); then `--context-store` images and an NVMe tier behind a RAM cache | done |
| 16 | Two lanes (`--max-concurrency 2`): slotted state and KV, a batched plain round for two decoding lanes (MTP stays at one), per-lane context cache, graphs and leases ([measurements](#pr-16-two-lanes), spike [#284](https://github.com/ashalliants/ninfer-3090/pull/284)) | done, opt-in (aggregate gate missed, below) |

The C++ loader reads and places every `ggml_*` and `ggml_rec_*` object. Linear and LinearAdd execute
the six dense formats (PR 3); offloaded_sparse_moe executes the three expert record formats (PR 7).

## PR 3 measurements

RTX 3090, CUDA 12.8, cold L2. Decode is IQ4_XS [10240, 2560] (13.93 MB); wide compares the Q8_1
cast plus int8 MMA with dequantize-to-FP16 plus cuBLAS (`cublasGemmEx`, FP32 compute).

| Workload | Result |
|---|---|
| Decode T = 1, kernel (nsys) | 20.2 us = 74% of the 936 GB/s spec; a plain read of the same bytes takes 17.2-17.7 us (79-81%) |
| Decode T = 1, public Op in a graph (events) | 22.5 us = 66% |
| Decode vs Strata `native_mmvq`, same bytes (nsys) | T = 1: 20.2 vs 21.1 us (+2.1 us Strata quantize); T = 8: 30.9 vs 62.9 us |
| Wide, T = 512 / 4096, ours over dequant + cuBLAS | 1.02-2.02x for five formats; Q6_K 0.96x / 0.77x |

The 80% decode gate is not met; see the PR 3 description for the tuning that was tried.

## PR 6 measurements

RTX 3090, CUDA 12.8, L2 flushed before each graph replay. "Chained" is 8 calls per graph on
distinct weight copies (the per-call cost inside a decode graph).

| Workload | Result |
|---|---|
| `hyper_connection_mix`, T = 1, 13.2 MB BF16 (block mixer / final mixer) | chained 28.2 / 26.8 us = 50% / 52% of 936 GB/s; single call 30.7 / 28.7 us |
| Same bytes, loads only (a probe of a rejected one-kernel variant, chained) | 17.3 us = 81% |
| `hyper_connection_mix`, fused route T = 2 / 4 / 8 / 16; composed route T = 17 / 64 | 35 / 49 / 76 / 122 us; 34 / 66 us |
| `projection_fp32` IQ4_XS head [248320, 2560], T = 1 / 2 / 8 | 392 / 396-400 / 583 us = 92% / 91% / 63% |
| `projection_fp32` BF16 router + shared gate [513, 2560], T = 1 / 8 / 9 | 10.2 / 14.3 / 26.6 us |

The 80% HC gate is not met: the fused mixer is latency-bound on sm_86, and above T = 2 it is
slower than the composed route. See the PR 6 description.

Rejected in PR 11 (same bench, ABBA against the two-kernel route, which measured 26.6-27.0 us):
- **One kernel of persistent CTAs** (each claims an A item and a B item from ticket counters and
  issues both weights' loads up front; B waits on a count of finished A items and runs unclaimed A
  items itself, so it cannot deadlock without co-residency): **47-50 us**. Per-CTA timestamps show
  the A partials finishing only once most of both weights have landed (W_down does not arrive ahead
  of W_up), then ~10 us of B work after the wait. Without the wait (timing probe) it was still 44 us.
- **Activations first** (R, the norm weights, partials and the residual slice copied to shared
  memory with `cp.async` before the weight loads, T <= 2): 27.4 us, no gain.

## Files ported from Infernix

Infernix (Apache-2.0) at commit `a3edb450`. Each file carries its own adaptation notice.

| Ours | Infernix |
|---|---|
| `include/ninfer/ops/hyper_connection.h` | `include/infernix/ops/hyper_connection.h` |
| `src/ops/hyper_connection/hyper_connection.cu` | `src/ops/hyper_connection/hyper_connection.cu` |
| `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` |
| `include/ninfer/ops/ple.h`, `src/ops/ple/ple.cu` | `include/infernix/ops/ple.h`, `src/ops/ple/ple.cu` |
| `include/ninfer/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | `include/infernix/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` |
| `include/ninfer/ops/rows.h`, `src/ops/rows/rows.cu` | `include/infernix/ops/rows.h`, `src/ops/rows/rows.cu` |
| `tests/ops/test_hyper_connection.cpp`, `test_projection_fp32.cpp`, `test_rows.cpp` | `tests/ops/test_hyper_connection.cpp`, `tests/ops/linear/test_projection_fp32.cpp`, `tests/ops/test_rows.cpp` |


## PR 7 measurements

RTX 3090, CUDA 12.8, 5950X with DDR4-2133; random records with finite scales (record bytes are
fixed per format); 10 experts each routed every column; cold L2; medians. Bit-exactness (CPU scalar
= AVX2 = team at 1-16 workers = GPU in device frames, staged and zero-copy, and CPU-served) holds on
synthetic and on the subset artifact's real IQ2_S, IQ2_XXS and IQ1_M banks.

| Workload | IQ2_S | IQ2_XXS | IQ1_M | Gate |
|---|---|---|---|---|
| Narrow route T = 1, resident, graph replay | 53.2 us, 284 GB/s (30%) | 46.1 us, 283 GB/s (30%) | 45.1 us, 261 GB/s (28%) | >= 70% of 936 GB/s: **missed** |
| Same, kernels (nsys median): gate/up + down | 35.4 + 16.9 us | 33.4 + 16.9 us | 28.8 + 16.9 us | |
| Plain read of the same bytes | 22.5 us (670 GB/s) | 19.5 us | 17.4 us | |
| Strata native grouped kernels, same records (quantize + grouped) | 61.4 us | 53.2 us | 52.2 us | |
| Canonical / native, T = 1, 2, 4, 8 | 0.90, 0.99, 0.88, 0.90 | 0.92, 1.02, 0.89, 0.76 | 0.90, 0.93, 0.81, 0.71 | K7: within 5% of native, **not triggered** |
| Staged misses T = 1 (10 misses), effective PCIe | 24.2 GB/s | 24.2 GB/s | 15.8-23.8 GB/s (one outlier run) | >= 18 GB/s: met (IQ1_M once below) |
| One cold CPU expert T = 1, 16 workers (sweep 1-16 in the PR) | 72.6 us, 20.8 GB/s | 88.9 us, 14.7 GB/s | 122.5 us, 9.6 GB/s | recorded |

The narrow route stays at the speed of Strata's native MMVQ grouped kernels on the same records:
both run two dependent kernels (gate/up, then down) over 15 MB, and neither reaches the read
roofline at T = 1. A sweep of CTA geometry and grids in shared memory moved T = 1 by at most ~15%.

PR 11 tried the persistent two-phase kernel and found the route is not bandwidth-bound:
- **Persistent kernel** (CTAs claim gate/up then down tiles in ticket order through a ring of
  `cp.async` slots; down tiles wait on per-job counters; bit-identical outputs): 2 slots matched the
  two-kernel route (IQ2_S / IQ2_XXS / IQ1_M T = 1: 55 / 47 / 48 us), 3 slots was slower, 4 slots
  (one CTA per SM) 70-74 us. Zero-copy calls were 2.4x slower (fewer bytes in flight over PCIe).
- **Weights in L2**: with every job reading the same frame (about 1.5 MB, L2-resident) the
  two-kernel route still takes 43 / 41 / 42 us at T = 1, against 47 / 46 / 45 us from DRAM. At most
  ~10% of the time is DRAM.
- **Grid gathers**: forcing every IQ grid index to 0 (timing only): 54 → 48 us (IQ2_S), unchanged
  (IQ2_XXS), 48 → 44 us (IQ1_M).
- **Activation reads first** (x prefetched to registers, h copied ahead of the down weights): T = 1
  unchanged, T = 4 / 8 5-12% slower.
- Per-CTA timestamps (gate/up, IQ2_S, T = 1): the first wave's slices land at ~11 us, then a CTA
  computes for ~12 us at three CTAs per SM, and the second wave follows; down adds ~12 us.

So the bound is per-CTA arithmetic latency at low occupancy, not memory, phase overlap or grid
gathers. Not yet attributed per instruction (Nsight Compute needs an elevated shell).

## PR 8 (M0) measurements

RTX 3090 (WDDM, sm_86), CUDA 12.8, Ryzen 9 5950X, DDR4-2133. Artifact and volume on K: (Samsung 860
QVO SATA); every routed expert in one 33.02 GiB pinned host block, served by the GPU narrow route
through 64 staging slots; INT8-G64 KV; one sequence. No speed gate applies to M0; these are recorded.

| Workload | Result |
|---|---|
| Full conversion (GGUF on C:, output to K:) | artifact 39.2 GB in two files and volume 29.1 GB; 802 s artifact + 346 s volume on the first run; 906 s for the artifact alone reusing the volume (QLC write cache exhausted); peak working set 0.62 GiB, peak private 1.73 GiB |
| Full artifact and volume against the GGUF (`qwen4_exp_verify`) | exact: 994 objects (48 record banks inverted, 825 identical, 120 narrowed, 1 widened), all 320,001,536 volume rows; 260 s |
| Load from K: | 93-104 s: 3.50 GiB device, 33.02 GiB pinned; process private 37.2 GiB; system available fell from ~44 to ~11.6 GiB |
| Teacher-forced scoring, chunks of 256 | 126-134 tok/s (1,023-position texts), including n-gram row reads (0.4 s per text) |
| Prefill of 300 tokens as one call | 3.13-3.24 s (~95 tok/s) |
| Decode, one call per token after 268 tokens (32 steps) | median 43.5 ms per token (23.0 tok/s) through 64 staging slots, 45.9 ms (21.8 tok/s) zero-copy; uncaptured, no expert resident in VRAM |
| Determinism and placement | two loads score bit-identical logits on all six texts; staged and zero-copy runs give bit-identical logits, block taps and routes |

Quality gate (spec 8.4) against Strata 0.1.38 IQ2_XS (`tools/flash_next/strata_compare.py`, ΔNLL =
NInfer - Strata, ± its paired standard error):

| Set / text | Positions | ppl NInfer | ppl Strata | ΔNLL | top-1 agree | KL mean / p99 |
|---|---:|---:|---:|---:|---:|---:|
| PR 0 code | 511 | 1.5225 | 1.5259 | -0.0022 ± 0.0036 | 99.6% | 0.0014 / 0.025 |
| PR 0 document | 1,023 | 21.4595 | 21.4299 | +0.0014 ± 0.0043 | 96.1% | 0.0050 / 0.035 |
| PR 0 chat | 1,023 | 6.5936 | 6.4902 | +0.0158 ± 0.0050 | 97.0% | 0.0048 / 0.050 |
| **PR 0 all** | 2,557 | 7.8876 | 7.8371 | **+0.0064 ± 0.0027** | **97.1%** | 0.0042 / 0.039 |
| Infernix code | 511 | 1.9616 | 1.9526 | +0.0046 ± 0.0035 | 98.4% | 0.0017 / 0.023 |
| Infernix document | 1,023 | 7.8517 | 7.8698 | -0.0023 ± 0.0043 | 96.3% | 0.0061 / 0.042 |
| Infernix chat (substitute) | 1,023 | 5.2231 | 5.2137 | +0.0018 ± 0.0043 | 98.0% | 0.0034 / 0.031 |
| **Infernix all** | 2,557 | **5.0555** | 5.0518 | **+0.0007 ± 0.0025** | **97.4%** | 0.0041 / 0.038 |

Strata against itself (PR 0, same metrics): top-1 agreement 96.5-99.2%, KL mean up to 0.064. Every
ΔNLL is within +0.02 nats. Two are more than 2 SE above zero: PR 0 chat (+0.0158, 3.2 SE) and the
PR 0 set overall (+0.0064, 2.4 SE); the Infernix set is within 1.2 SE on every text.

FP64 block check (`tools/flash_next/block_check.py`, 48 chat tokens; relative L2 per position, max
over positions): hyper-connection mixers 0.2-1.0%, GDN 0.7-1.0%, QSA 2.5-2.9% (the reference keeps
exact K/V, the engine INT8-G64), MoE <= 0.07% (most positions bit-equal after BF16 rounding),
residual updates <= 0.6%, layers 0-3, 7-9, 46 and 47.

## PR 9 (M1) measurements

RTX 3090 (WDDM, display attached), CUDA 12.8, Ryzen 9 5950X, DDR4-2133; artifact and volume on K:
(SATA). Through the public Engine (`ninfer_qwen4_exp_engine_real_test`) and `ninfer-serve`.

**Gates.**

| Gate | Result |
|---|---|
| Serving smoke (`run.bat qwen38-flash-next` + chat completions, streaming, tool call required and auto, the tool result answered, Anthropic Messages, greedy repeat) | 7/7 |
| Greedy ids identical at C = 1 across expert-cache states | identical: cold, warm (same Engine), warm start from the saved state, a 1 GiB cache (710 frames, evicting every round) with a 64-token KV base and no CUDA graph; a 5,120-token prompt (past QSA's 2,048-token budget, the KV base growing) identical between the default and the small-cache Engine |
| RAM ledger refuses cleanly | an Engine keeping 1 TiB free refuses before pinning: "... 1017.0 GiB short. Close other programs ..." |

**Decode, plain, 512 greedy tokens** after Strata's PR 0 prompts (the same token ids), one model
load, a warm-up request then 18 requests in ABBA order over the tiers (6 per tier), the expert cache
warm-started from an earlier session. Wall clock after the prompt (every host step included):

| Prompt | NInfer tok/s, median [min-max] | Strata plain (PR 0) | Strata MTP (PR 0) | decode hit rate |
|---|---:|---:|---:|---:|
| 1K | **78.1** [75.5-79.5] | 71.3 | 105.1 | 98.0% |
| 8K | **76.7** [75.7-78.7] | 69.6 | 112.4 | 97.2% |
| 32K | **75.6** [74.1-77.1] | 66.2 | 107.1 | 96.6% |

- 1.10×, 1.10× and 1.14× Strata's plain decode; 0.68-0.74× its MTP decode. The Infernix-normalized
  8K range (spec 8.2: 80-120) is not reached: 76.7 is 4% below its bottom.
- An earlier identical run (Program-timed) gave 77.9 / 76.8 / 75.9 by the same wall formula.
- Prefill, not gated here (PR 10): 650-700 tok/s from a warm cache at 1K-32K (Strata 1,900-2,400).
- Expert cache: 11,063 frames at startup, grown by the VRAM control law to 11,951 (16.8 GiB)
  within the first minute; Strata holds 12,375-12,583. Decode promotes ~2-3 experts per token.

**Where a decode token's time goes** (8K, warm cache, nsys with graph nodes traced; kernel times per
round, which the profiler inflates by roughly 5%: the GPU is busy ~13.7 ms of a 13 ms token, so
decode is GPU-bound and the host adds ~0.08 ms per token outside the rounds):

| Work | ms / token | Note |
|---|---:|---|
| Routed experts | 4.27 | `stage_kernel` 2.08 (48 launches; PR 11 found this is the misses' PCIe copy, not a fixed cost: half the launches take 2 µs, the rest 40-400 µs), `gate_up` 1.46, `down` 0.73 (#239's narrow route at 28-30% of bandwidth) |
| Dense GGML linears | 3.89 | 301 MMVQ launches (shared experts, attention, GDN, head) |
| HC mixer and inject | 2.78 | #236's 97 fused calls |
| MoE route, dispatch, combine | 0.55 | |
| FP32 projections (router, indexer) | 0.51 | |
| QSA (prep, select, attention), GDN, norms, copies | ~1.7 | QSA attention 15 µs per layer here |

The narrow route's bandwidth and the HC mixer are PR 11's (spec 8.2); PR 11 below corrects the
stage attribution.

## PR 11 (M2 decode) measurements

RTX 3090 (WDDM, display attached), CUDA 12.8, MSVC 14.44, Ryzen 9 5950X, DDR4-2133; artifact and
volume on K: (SATA). Decode is 512 greedy tokens after Strata's PR 0 prompts, wall clock after the
prompt; one model load per process, a warm-up request, then the requests named. Every GPU run held
the shared GPU lock.

**Gate (spec 8.4), ABBA over builds:** base (07457ce9) and PR 11 processes in the order A B B A, each
process a warm-up then 18 requests in ABBA order over the tiers; 12 requests per tier per build.
Greedy ids are identical between the builds on every request.

| Prompt | PR 9 base, median [min-max] | PR 11, median [min-max] | change | Strata plain | PR 11 / Strata |
|---|---:|---:|---:|---:|---:|
| 1K | 77.42 [74.90-78.80] | **81.15** [77.18-82.77] | +4.8% | 71.3 | 1.14× |
| 8K | 76.30 [74.69-77.95] | **79.88** [77.99-81.62] | +4.7% | 69.6 | 1.15× |
| 32K | 75.49 [70.97-77.08] | **78.96** [75.24-81.11] | +4.6% | 66.2 | 1.19× |

The blocking gate (≥ 1.0× Strata plain at 1K/8K/32K) holds. The non-blocking 8K target of 80 tok/s
is met by the mean of an earlier 16-request run (80.14) and missed by this run's median (79.88,
mean 79.92); the stretch (1.3× Strata, ~91) is not reached.

**Attribution, 8K** (16 requests per row unless noted, two processes in ABBA order):

| Build | tok/s, median [min-max] |
|---|---:|
| base | 75.36 [70.12-77.14] |
| + route log in the dispatch, PLE slot as state (11b, two of three parts) | 76.98 [73.83-78.56] |
| + zero-copy rule, threshold 1 / 2 / 4 (11a) | 77.76 / 77.90 / 77.91 |
| + `linear_pair` for shared gate/up and attention key/value (11b) | 79.51 [75.94-80.80] |
| + decode graph split around the n-gram read, startup frame retry (11c) | 80.47 [75.92-82.14] |

The last row's +1.2% (+0.96 tok/s) is 1.6 pooled standard errors: suggestive, not established. Its
processes started with 12,268 frames against 10,796-10,930 for the row above.

**Zero-copy rule (11a).** `offloaded_moe_bench --misses N`, T = 1, 10 experts, graph µs
(IQ2_S / IQ2_XXS / IQ1_M): one miss staged 122 / 120 / 93, under the rule 90 / 80 / 78; two misses
165 / 159 / 144 vs 162 / 142 / 138; three and four equal within noise. Rejected: the stage copy on
a forked high-priority stream beside the resident experts (with the shared expert in between) was
slower at every miss count (one miss 157 / 144 / 139 µs): the resident pass's 400 CTAs occupy the
SMs before the copy starts (~25 µs later in nsys) and the join edge adds 15-40 µs.

**Miss split and worker count (spec 3.3).** One cold expert on the CPU worker team at T = 1: 119 /
98 / 146 µs at 6 workers, 70 / 85 / 82 µs at 12, 67 / 70 / 84 µs at 16 (IQ2_S / IQ2_XXS / IQ1_M,
GPU idle). Zero-copy adds 30-45 µs to a layer for one miss, so a CPU leg that waits ≥ 67 µs plus the
handshake cannot win for the one or two misses a decode layer usually has (half the layer calls
have none; ~10% have three or more). The CPU route stays unwired; not measured end to end.

**Promotion cadence.** Interval 2 / 4 (the link-derived default) / 8 rounds once the frames are
full: 80.28 / 80.14 / 79.73 tok/s at 8K (16 each; the 8-round runs include one 66 tok/s request);
within noise, so the default stays.

**Startup frames.** The pool stopped growing at startup when one 64 MiB chunk read as placed in
system memory (free memory did not drop) with ~2.1 GB free; the control law grew it a minute later
anyway. Retrying such a chunk up to three times (100-300 ms apart) gave the full sizing in the
runs measured: 12,268 frames at startup (was 10,796-11,107). The steady state still settles at
11,9xx frames: the control law keeps the 512 MiB display headroom and the reserve free, and frames
are sized for the widest record (IQ2_S, 1.51 MB), which wastes ~3.5% of the pool on the IQ2_XXS and
IQ1_M layers. Strata holds 12,375-12,583.

**Host and n-gram rows (11c).** The decode round now runs the layers before the PLE layer (two of
48) as one graph while the host reads the round's n-gram rows from the volume (median 0.6 ms, p90
1.2 ms per token under nsys), then the rest as a second graph.

**Not changed, with reasons.** Power throttling: the opt-out exists since PR 9. WDDM event batching:
a decode round queries at most a few promotion events and records two per promotion batch; nothing
to batch. L2-warm replacement: Infernix warms L2 while the host computes CPU-served misses; without
the CPU route there is no window.

**Where a token goes now** (8K, a warm request after an identical one, 114 rounds, nsys with graph
nodes traced; kernel time per round, which the profiler inflates; the GPU is busy 11.8 ms of a
13.9 ms profiled round):

| Work | ms / token | Note |
|---|---:|---|
| Dense GGML linears | 3.82 | 181 MMVQ launches + 60 pair launches (0.43); the large IQ4_XS ones near PR 3's 74%, IQ3_S at ~42% |
| HC mixer and inject | 2.80 | unchanged (#236; the rewrites in #247 were slower) |
| Routed experts | 2.63 | `gate_up` 1.63, `down` 0.77, stage 0.22 (this request's hit rate was high; most layers have no miss) |
| MoE route, dispatch, combine; FP32 projections | 1.08 | |
| QSA, GDN, norms, copies | ~1.5 | 216 GDN head-tiling copy nodes, 0.25 |
| GPU idle waiting for the host's n-gram rows | 1.8 under nsys | the front graph (two layers, ~0.5 ms) hides part of the row read (median 0.6 ms, p90 1.2 ms); graph launches are much slower under the profiler, so the real exposed share is smaller (11c measured +1.2%) |

Remaining levers, largest first: the HC mixer and the narrow expert route (both latency-bound per
#247; instruction-level work with Nsight Compute), IQ3_S decode in the GEMV (~0.5 ms at 42% of
bandwidth), the row read (a host row cache or prefetching the next token's candidates), the
head-tiling copies, and per-format frame pools (~3.5% more frames).

One profiling run of the final build (three 8K requests of 160 tokens) ended with an access
violation after the trace was written; the same arguments ran cleanly without the profiler and
under nsys at 48 and 128 tokens, and every bench and test process exited cleanly. Not reproduced.



## PR 10 measurements

RTX 3090 (WDDM, display attached), CUDA 12.8, MSVC 14.44, Ryzen 9 5950X, DDR4-2133; artifact and
n-gram volume on K: (Samsung 860 QVO, SATA). Strata's PR 0 runs read their PLE rows from C: (NVMe).

**Prefill** through the Engine (`ninfer_qwen4_exp_engine_real_test --bench`), Strata's PR 0 prompt
ids, 4 generated tokens, one model load: a 4K warm-up then 6 requests per tier in ABBA order; 128K in
its own load (135,168-token context), 3 requests. Prompt tokens over the Program's prefill time:

| Prompt | PR 9 (same session, same method) | PR 10, median [min-max] | Strata (PR 0) | PR 10 / Strata |
|---|---:|---:|---:|---:|
| 4K | 524 [188-585] | **1,824** [1,348-2,120] | 1,912 | **0.95** |
| 32K | 503 [253-547] | **2,329** [1,817-2,371] | 2,397 | **0.97** |
| 128K | not run | **2,069** [1,906-2,329] (n = 3) | 2,184 | 0.95 |

- The prefill gate (≥ 1.0× Strata at 4K and 32K) is **missed by 5% and 3%**. Each tier's first
  requests are the slow ones (cold n-gram rows and expert cache); the best 4K run is 2,120.
- Greedy ids were identical for every repeat of a prompt, and the Engine test's 5,120-token prompt
  gives the same ids through the wide streamed path and the small-cache static path.

**Where a 32K prompt's prefill goes** (nsys, kernel time summed per prompt):

| | PR 9: 32 calls of 1,024 | PR 10: 8 calls of 4,096 |
|---|---:|---:|
| Routed experts | 51.2 s: staging 24.8, narrow gate/up 17.8, narrow down 8.6 | 5.74 s: wide gate/up 4.45, wide down 1.29 |
| Dense GGML linears (MMQ) | 3.00 s | 2.62 s |
| BF16 linears (HC composed route, PLE) | 1.75 s | 1.32 s |
| QSA attention, FP32 projections, row splits, HC, GDN, norms, other | ~3.6 s | ~3.2 s |
| Kernel total | 59.5 s | 12.9 s |
| Host time before each call's kernels (its n-gram rows) | 1-2.9 s per call under nsys | none after the first call: the next call's rows are read while the GPU runs |
| Expert copies (H2D) | none (staged by SMs) | 23 GB per call at 26 GB/s on the copy engine, overlapped |

**Wide expert route** (512 resident experts, near-uniform routing, `--bench`): T = 4,096 in
14.5-15.8 ms per layer call (26-28 TOPS of routed multiply-adds), T = 2,048 in 7.2-7.6 ms. Its
outputs equal the CPU engine's bits (`ninfer_offloaded_moe_wide_test`, every format and placement,
the 48 real banks).

## PR 12 measurements

RTX 3090 (WDDM, display attached), CUDA 12.8, MSVC 14.44, Ryzen 9 5950X, DDR4-2133; artifact and
n-gram volume on K: (SATA). Process-level ABBA against the integration build (`integ/qwen4exp-m1`,
#256) and #253 (decode); greedy ids, 512 generated tokens for decode, 4 for prefill.

**The crash and the 128K prefill collapse had one cause.** The KV pool grew by mapping new memory
first and took expert frames only if a map failed. On a VRAM the startup sizing had filled (~400 MiB
free), a 128K growth (+1,344 MiB) put the process ~900 MiB over its WDDM budget and past physical
memory, so the OS paged device memory (128K prefill at 447 tok/s), and when the VRAM monitor's
`cudaMemGetInfo` ran meanwhile, `cuMemSetAccess` returned `CUDA_ERROR_NOT_READY`. A standalone probe
reproduces it (map-first growth past the budget with a concurrent `cudaMemGetInfo` fails at the
physical limit, 2 of 3 runs; none with only DXGI queries, within the budget, or with take-first growth in 48 cycles).
The growth now releases frames first (Infernix's `grow_kv` order).

| Check | Integration (#256) | PR 12 |
|---|---|---|
| 4K/32K/128K alternating, 21 requests + warm-up, max-context 135,168 | every process crashed (2-3 growths) | no crash; process at 22.8-22.9 GiB dedicated, shared memory flat (no demotion), adapter ≤ 23.4 of 24 GiB |
| 128K prefill in that alternation (n = 7) | 447, 1,243 (n = 2) | **2,105** [1,949-2,348] tok/s = 0.96× Strata (2,184) |
| Decode after a 128K prompt / after a 32K prompt (wall, 512 tokens) | 21-42 tok/s for the first 64 tokens | **74.0** [72.3-75.2] / 78.1 [75.9-78.4] = 0.95×; first 64 tokens 74.7 |
| Needles (10/50/90% depth) at 32K, 64K, 128K | not run (crash) | 9 of 9 retrieved |

**Decode** (1K/8K/32K, D I B B I D, 6 requests per tier per process):

| Prompt | #253 | Integration | PR 12 | PR 12 / #253 (paired) | PR 12 / integration (paired) | decode hits #253 / integ / PR 12 |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 81.2 | 79.8 | **81.9** | 1.002 ± 0.003 | 1.021 ± 0.003 | 98.0 / 96.8 / 98.2% |
| 8K | 80.1 | 78.6 | **80.3** | 1.005 ± 0.002 | 1.027 ± 0.003 | 97.1 / 95.9 / 97.5% |
| 32K | 79.0 | 76.9 | **78.8** | 0.998 ± 0.006 | 1.028 ± 0.002 | 96.5 / 95.2 / 96.8% |

The integration's refill gave the lent frames to the experts the prompt's own calls had just used;
restoring the displaced ones returns the cache to the set decode had (prefill calls promote nothing).
The 8K target (80 tok/s) is met at 80.3 (range 79.2-83.8).

**Prefill** (4K warm-up + 12 requests, 4 tokens, max-context 33,792; three sessions, I B B I twice and
B I I B once): PR 12 / integration paired 4K **1.000** (0.998, 1.000, 1.001), 32K **0.994** (0.989 ±
0.006, 0.991 ± 0.006, 1.000 ± 0.003). The I B B I deficit was mostly the second process of the session
(slow in either build); what remains is the first 32K prompt after startup, 1-5% slower in every
session (the cache holds the restored decode set instead of the warm-up prompt's experts); later
32K prompts are equal. 0.95× / 0.97× Strata at 4K / 32K, as before.

Quality (#243's method): logit dumps byte-identical to the integration's on all six texts (the
forward does not use the Program). Greedy ids equal the integration's for every prompt measured.

## PR 13 measurements

MTP speculation, 2026-10-12. RTX 3090, CUDA 12.8, desktop attached; s60's prompts, 512 greedy tokens;
#258 plain brackets the MTP processes (18 measured requests each, n = 12 per cell). Strata (PR 0):
MTP 105 / 112 / 107, plain 71 / 70 / 66.

| Tier | #258 plain | MTP policy, ceiling 3 | MTP fixed 2 | fixed 2 / plain | / Strata MTP |
|---|---:|---:|---:|---:|---:|
| 1K | 81.4 [78.3-83.0] | 108.0 [104.5-111.3] | 110.5 [106.0-113.6] | 1.36 | 1.05 |
| 8K | 80.2 [78.7-83.5] | 110.7 [102.4-116.0] | 112.6 [104.6-118.2] | 1.39 | 1.01 |
| 32K | 78.2 [75.9-79.8] | 111.8 [106.3-116.3] | 111.9 [104.7-114.7] | 1.42 | 1.05 |

- **Gate (spec: 8K >= 1.2x Strata MTP = 134 tok/s): missed** at 112.6. Decode after a 128K prompt:
  102.6 [101.7-108.9] (policy, n = 3; PR 12 plain 74.0).
- Greedy ids equal plain decoding on every request (108 of 108 above, and `--mtp-exact`).
- Depth (fixed drafts 1-5 at 8K): 102.1, 112.0, 111.9, 96.0, 90.0 tok/s; acceptance 82 / 76 / 65 /
  58 / 51% of drafts, 1.82 / 2.51 / 2.94 / 3.30 / 3.52 tokens per round. Two drafts is the optimum.
- Cost model (nsys, 8K): a verification of W columns costs 17.0 / 20.0 / 23.8 ms at W = 2 / 3 / 4,
  +3.4 ms per column against a ~12.5 ms plain step; 2.0 ms of each column is routed experts (more
  distinct experts per layer, and their PCIe misses), 0.6 ms the mixers. A drafter step is 0.88 ms,
  0.40 of it the IQ4_XS head. The round has no host gap to recover.
- Plain decode and prefill unchanged against #258 (decode 1.003 / 1.000 / 0.993, prefill 0.999 /
  1.000 at 4K / 32K; ids identical). With MTP on, prefill is 1-4% slower (the drafter's cells, and a
  ~5% smaller expert cache).
- 4K / 32K / 128K alternation with MTP: no crash, adapter at most 23.3 of 24 GiB, no demotion.
- **Ceiling (13c).** The policy with ceiling 3 against ceiling 2 (P2 P3 P3 P2, n = 12): 0.996 /
  1.005 / 1.026 +- 0.015 at 1K / 8K / 32K; ceiling 2 drafts two in fewer rounds (1K: 2.16 tokens
  per round). Fixed 2 drafts is 1-2% faster than either at 1K / 8K, but a fixed depth is a
  measurement switch, not a serving option. The launcher uses ceiling 3 with the `-mtp` artifact.

**Reproduction on the merged build, 2026-10-13.** The worktree build of master at `04901788` (all of
PRs #247-#277), model and volume on the C: NVMe drive, launcher defaults (INT8 KV, 32,768 context,
vision on), the card shared with the desktop. One server per arm in the order MTP, plain, plain,
MTP inside one GPU-lock hold; each ran four requests per tier of 512 greedy tokens after a prompt
built from this repository's markdown (1,008 / 8,000 / 30,047 tokens, not the earlier prompt
set), one warm-up request first. Medians per arm [min-max of the 4 requests]:

| Tier | Plain, run 1 / run 2 | MTP, run 1 / run 4 | MTP / plain | Prefill, plain / MTP |
|---|---:|---:|---:|---:|
| 1K | 81.2 / 82.0 | 94.3 / 99.5 | 1.19 | 1,260 / 1,050-1,180 |
| 8K | 82.7 / 82.5 | 93.9 / 98.3 | 1.16 | 2,452 / 2,326-2,422 |
| 30K | 79.6 / 81.7 | 93.5 / 102.7 | 1.22 | 2,351-2,378 / 2,261-2,360 |

Plain decode and warm prefill agree with the figures above. MTP decode is 96-98 tok/s here against
108-113 on the earlier prompts: acceptance depends on the text, and these prompts ask for a
summary of dense technical notes. The two MTP runs differ by up to 5% from each other (the second
was faster at every tier), so this run does not support a tuning claim in either direction. Quality
was not re-measured. The harness is a small streaming client that is not in the tree; the earlier
prompt set was not kept either, so the Strata comparison in the README stays on the earlier
measurements.


## PR 14 measurements

Image input, 2026-10-12. RTX 3090, CUDA 12.8, desktop attached; artifact
`qwen4exp-iq2xs-mtp-vision.ninfer` (41.0 GB, the full artifact verified byte for byte against the
three GGUFs).

- **Tower against FP64** (`ninfer_qwen4_exp_vision_real_test`, 32 encodes of synthetic images over
  14 grids from 2 x 2 to 12 x 40 patches, the last 8 under the final criterion): normwise 1.3e-2 to 5.5e-2 on every image the processor can produce (2.0e-1 on a
  2 x 4 grid it cannot), within the criterion `max(2^-4, 2 E)` where E is the error of the oracle
  itself with BF16 storage (0.017 to 0.17); the GPU is 1.4e-2 to 3.7e-2 from that BF16 estimate. The
  tower stores its residual in BF16 and is ill-conditioned with respect to that storage; llama.cpp
  keeps F32 activations, so Strata's image embeddings sit nearer FP64. Two wrong-mapping controls
  land at 0.69 to 1.05. The criterion was widened twice after measuring (the PR 14b body has why).
- **Answers** (example images): "NIFER VISION 731; three; left", "24; right", "2; 3; star", the
  wallpaper and a 128-token scene description, all correct. MTP and plain greedy ids identical with
  images in context (185 ids, 88 verification rounds).
- **Against Strata 0.1.38** (same server requests, greedy): prompt tokens equal wherever both make
  the same grid (443 / 426 / 538 / 418 / 1,021); llama.cpp caps a 1920 x 1200 image at 1,000 tokens
  where the processor makes 2,280. Identical answers on 3 of 5 comparable cases; the others diverge
  at token 10 ("three" / "3") and 17 ("." / ":"). Strata's own two runs diverge at tokens 4 and 17-26
  on two of the cases; ours repeat exactly.
- **Time** (warm): encode 0.05-0.10 s for a 768 x 512 image (384 tokens), 0.15-0.17 s for two
  640 x 384 ones, 0.41-0.46 s for 1920 x 1200 (2,280 tokens); the whole prefill of a 426-443-token
  image prompt 0.7-1.4 s, of the 2,301-token one 1.5-2.3 s. The first image of a process took
  2.1-7.1 s more before the startup warm-up (now 2.2-6.9 s at startup).
- **Memory**: no VRAM while idle (the tower is pinned host RAM, 0.84 GiB); a request lends ~130 MiB
  plus the encode workspace and the handoff from the expert cache. The KV grow/shrink cycle with an
  image in a 41,728-token prompt: 0 MiB over the OS budget, peak 22,856 MiB, frames returned.
  The RAM ledger needs ~2.2 GiB more with `--vision` (the tower plus one prompt's decoded media).
- **Text unchanged** (#264's engine against this branch with `--vision`, A B B A, s60's prompts):
  greedy ids identical; plain decode 0.997 / 0.996 / 0.998 (+-0.002-0.003, 1K / 8K / 32K), MTP decode
  1.001 / 0.996 / 0.999, prefill 0.998 / 1.003 (4K / 32K).

## PR 16 (two lanes)

`--max-concurrency 2` for Flash-Next. Two requests decode in the same rounds as one plain batched round
(`Forward::run_rows`, from the spike's slice 1, #286); one decoding lane runs today's single-row round
with MTP. Design: the PR description of #291. The launcher stays at one lane (`NINFER_CONCURRENCY=2` opts in).

Gates (RTX 3090, WDDM, artifact `qwen4exp-iq2xs-mtp-vision.ninfer`, every run under the shared GPU lock):

1. **Exactness: passed.** `ninfer_qwen4_exp_lanes_real_test --exact --mtp 3 [--cache]`, 96 greedy tokens: 3 pairs
   (all overlapped), 2 staggered pairs and each of three prompts (700 / 3,000 / 8,192 or 9,000 tokens) alone on the
   two-lane Engine, every one identical to its one-lane run. With the cache: the 8,192-token prompt cached, its
   +600-token suffix resumed in one lane beside a short prompt (identical to the cold run), a lane cancelled after 8
   tokens (the other lane and the next request unchanged); also at `--concurrency 1`. A resume where the cold prefill
   has no call boundary (a 9,000-token prompt at chunk 1,024) differs in greedy ids on its own, with the cache
   at any lane count; `test_engine_real.cpp` documents it, so the gate resumes at 8,192.
2. **Aggregate decode, >= 1.15x of MTP at one lane on the spike's 8-request burst: MISSED.** `ninfer-serve --spec
   mtp --draft-tokens 3 --no-expert-state --no-prefix-reuse --max-context 32768 --prefill-chunk 1024`, the spike's
   eight conversations (`tools/flash_next/lanes/burst_client.py`, all queued at t = 0, 1,024 greedy tokens each, 7,242
   in all), a fresh server per run, order C1 C2 C2 C1 in one lock hold (`burst_gate.ps1`):

   | | C = 1 (two runs) | C = 2 (two runs) | C = 2 / C = 1 |
   |---|---:|---:|---:|
   | makespan | 101.4, 97.4 s | 93.5, 92.7 s | 0.94 (6.3% sooner) |
   | tokens / makespan | 71.4, 74.4 | 77.5, 78.2 | **1.07x** |
   | decode only (tokens / (makespan - prefill)) | 84.8 | 91.6 | 1.08x |
   | mean completion | 61.6, 59.1 s | 60.1, 59.4 s | 0.99 |
   | mean time to first token | 50.7, 48.7 s | 40.4, 39.9 s | 0.81 (19% lower) |

   A pair decodes at 47-52 tok/s per stream (about 95-104 aggregate, the spike modelled 109), against 80-95 for one
   stream with MTP; the last request of the burst runs alone and slower (72 tok/s) than the one-lane MTP rate.
   The one-lane MTP figures here (median 87-91 per request) are close to the spike's 92.6. The workload is queued
   prefill-heavy (about 14 s of prefill in 97 s), which a second lane cannot shrink.
3. **4K / 32K / 128K alternation at two lanes: passed.** `--bench --concurrency 2 --mtp 3 --max-context 135168`,
   ten requests (4,096 / 32,000 / 130,000 tokens, 64 generated), no crash or error, KV pool growing and shrinking
   (up to 2,560 pages), the process at most 22.7 GiB of the 23.1 GiB OS budget (VRAM free 375-786 MiB at the
   growths), all ten finished.
4. **A lone request at two lanes within 2% of one lane and of master: not separable from noise.** The eight
   conversations one at a time (`lone_gate.ps1`), median decode tok/s per full-length request, order M L1 L2 L2 L1 M:
   master 90.2 and 91.6, this branch at one lane 85.9 and 91.6, at two lanes 86.2 and 91.1. The first two runs of the
   branch were slow and the last two equal to master, so a drift in the sitting is as likely as a cost. Prefill of the
   7,106 / 3,233 / 10,527-token prompts: master 3.22 / 3.68 / 4.60 s, branch 3.20-3.35 / 3.60-3.77 / 4.58-4.78 s.
5. **Existing tests: passed.** The Flash-Next cache gates (`tools/bench/qwen4exp_cache_gates.ps1` runs: plain, MTP 3,
   eviction at 512 MiB, vision cache), `ctest -R qwen4_exp` (12 tests, all pass; `forward_real` and `lanes_real` skip without
   arguments, so `forward_real` ran directly: single-sequence decode and `--rows 32` with three prompts, greedy ids and
   logits bit-identical to each prompt alone at B = 1..3 on the slotted forward) and `ctest` over the built
   `ninfer_tests` bundle: 193 of 207 ran green or skipped; the 14 failures are the known `jsonschema` oracles and
   `ninfer_artifact_writer_interop_test`, the tests whose standalone executables were not built (`Not Run`), the
   chat-template renderer and `ninfer_openai_responses_test` (exit 0xc0000409; this branch touches no serve code).

Decision: C = 2 beats C = 1 in makespan and queued time to first token but not by the aggregate decode the
spike promised (about 1.07x against 1.18x modelled). It ships opt-in; the launcher default stays at one lane.

## Artifact decisions

Taken in PR 1 and revised in PR 2b, which aligned the artifact with Infernix (the model code and
expert cache are lifted from it in PRs 7-9).

- Logical parameters mirror the GGUF tensors one to one, named under `text/`; shapes are the
  reversed GGUF dimensions, so every object holds the GGUF bytes unchanged. The exceptions: a
  layer's three routed-expert tensors are one expert bank, and `ffn_gate_inp_shexp` becomes
  `moe/shared_score` of shape `[1, 2560]`. The map is the table in
  [`tools/convert/qwen4_exp.py`](../../tools/convert/qwen4_exp.py); unknown, missing or misplaced
  GGUF tensors are errors.
- **Expert banks (PR 2b).** Each layer's routed experts are one parameter
  `text/layers/L/moe/experts` of shape `[512, 2560, 640]` in `ggml_expert_record_v1`: one record
  per expert holding its gate, up and down blocks, in one of three record formats
  (`ggml_rec_iq2_s_q2_0`, `ggml_rec_iq2_xxs_q2_0`, `ggml_rec_iq1_m_q2_0`). Infernix's frame pool,
  residency table, staging and CPU jobs treat an expert as one opaque record with one
  `record_stride`; packing records on the host at load would be runtime repacking, so the converter
  writes them. Records are 1,510,400 / 1,305,600 / 1,177,600 bytes, back to back: the stride is
  rounded to 256 bytes, not Infernix's 4 KiB, because only the deferred SSD expert tier reads
  records in place with direct I/O, and 4 KiB rounding would cost 1.5% (0.5 GiB) of host RAM. This
  replaces PR 1's two objects per layer (gate|up interleaved, and down).
- **n-gram volume (PR 2b).** The PLE n-gram table is not an artifact object. It is Infernix's
  `NINFERNG` volume, version 2 with a row-format field (`ggml_iq4_nl`): 45 rows of 90 bytes per
  4 KiB block, exactly PR 1's page geometry, after a 4 KiB header; 29,127,258,112 bytes. The
  artifact's `ngram_table` records the geometry and the volume id; `--ngram-out` and
  `--ngram-reuse` follow Infernix, including its 512-row reuse check. This replaces
  `ggml_rows_page4k_v1`, which no longer exists.
- **Text config (PR 2b).** The keys are exactly Infernix `parse_config`'s (`hc_lowrank`,
  `indexer_n_heads`, `indexer_budget`, one-based `ple_layer_ids`, `ple_embed_dim` = 16 heads x 160,
  `eos_token_id` = the PLE EOS 248044, and so on). The GGUF's literal hash tables are replaced by
  Infernix's derivation parameters (seed 1234, base 20,000,000, padding 128), which the converter
  proves equal to the literals; a GGUF with other literals is refused. Facts the GGUF does not
  record are fixed by the architecture: `output_gate_type` sigmoid (llama.cpp's and Strata's GDN
  both apply a sigmoid gate), `norm_topk_prob` true, one indexer key head (checked against
  `indexer.k_proj`), and `split_ngram_parts` 1 (the GGUF holds the table as one tensor; Infernix
  parses the key and does not use it). The GGUF's `ple.image_token_id` is not carried: Infernix has
  no such key and vision is PR 14.
- GGML-format projections and the expert banks record `AllowA8` (ggml's Q8_1 activation path);
  BF16 projections keep `A16Only`.
- **Operand layouts (PR 8).** BF16 projections over one input share a parent: each mixer's
  `down|inject` ([324, 10240], the HC Op's operand), the GDN `a|b` ([96, 2560]) and the PLE
  `key|value` ([12800, 2560]); registered BF16 Linear shapes. The attention and indexer q/k norms
  are stored as BF16 `gamma` (unit offset), `gdn/norm` and `gdn/convolution` as BF16, after an
  exactness check that refuses any rounding; `gdn/convolution` is transposed to the `[K, C]`
  operand of the convolution Op. The PLE convolution stays the exact FP32 widening of the F16
  taps; `ple_conv_inject` now takes FP32 weights.
- The development subset is a builder option (`--subset dev`), not a second recipe, because a
  recipe cannot drop parameters or shorten the n-gram table. It is 2,928,979,712 bytes plus a
  413,696-byte volume of 4,500 rows: the token embedding and output head alone are 675 MB and the
  three expert banks 2.04 GB.

## Code adapted from Infernix

Infernix is Apache-2.0; each adapted file carries the notice the spec's licensing rule fixes.

| NInfer file | Infernix source (`a3edb450`) | Adaptation |
|---|---|---|
| `tools/artifact/ngram_volume.py` | `tools/convert/qwen4_exp.py` (`ngram_geometry`, `read_ngram_volume_id`, `write_ngram_volume`) | Version 2 header with a row-format field; streams GGML rows from any source |
| `tools/convert/qwen4_exp.py` (config keys, `layer_multipliers`, `head_tables`, expert bank parameter, volume binding) | `src/models/qwen4_exp/config.cpp`, `tools/flash_next/ngram.py`, `tools/convert/qwen4_exp.py` (`ExpertBankSource`, `import_expert_bank`), `tools/convert/__main__.py` | Sourced from GGUF metadata; GGML expert records instead of NVFP4 banks |
| `include/ninfer/ops/offloaded_sparse_moe.h` | `include/infernix/ops/offloaded_sparse_moe.h` | GGML record formats, no per-expert scales; an exact wide route (PR 10); SSD tier, fetch channel, landing, streamed records, overlap/fork streams and L2 warming not carried |
| `src/ops/common/canonical_math.h` | `src/ops/common/canonical_math.h` (`0cf68068`) | IEEE helpers, BF16, exp and SiLU only |
| `src/ops/offloaded_sparse_moe/cpu/expert_team.cpp` | `src/ops/offloaded_sparse_moe/cpu/expert_team.{h,cpp}` | GGML jobs; units of 32 intermediates; no A16, AVX-VNNI or AVX-512 |
| `src/ops/offloaded_sparse_moe/cpu/miss_service.cpp` | `src/ops/offloaded_sparse_moe/cpu/miss_service.{h,cpp}` | No tiered requests; fixes a startup race (the first request could be taken as already answered) |
| `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`398cf2cc`) | Route, dispatch, staging, CPU plan/wait and combine kept; narrow kernels rewritten for GGML sub-blocks |
| `src/ops/offloaded_sparse_moe/cuda/wide_expert.{h,cu}` (PR 10) | `src/ops/offloaded_sparse_moe/cuda/wide_expert.{h,cu}` (`17681db5`) | The plan kernel (tiles per pass) kept; the GEMMs are new: exact int8 `mma.sync` over decoded GGML codes in the canonical leaf order, cp.async instead of TMA/mbarrier, the A8 cast of h fused into gate/up |
| `src/models/qwen4_exp/execution/expert_stream.{h,cpp}` (PR 10) | `src/models/qwen4_exp/execution/expert_stream.{h,cpp}` | The ring is lent frames, published as per-layer frame tables (no `prefetched` source); 2-D run copies; no gated chunks or SSD holds |
| `include/ninfer/ops/hyper_connection.h` | `include/infernix/ops/hyper_connection.h` | FP32 multiplier norm weight; no Q8 weight form |
| `src/ops/hyper_connection/hyper_connection.cu` | `src/ops/hyper_connection/hyper_connection.cu` | BF16 weights only |
| `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | Grids re-fitted to 82 SMs |
| `include/ninfer/ops/ple.h`, `src/ops/ple/ple.cu` | `include/infernix/ops/ple.h`, `src/ops/ple/ple.cu` | IQ4_NL `ple_embed`; FP32 multiplier norms; FP32 `[K, C]` conv weight (the F16 taps widened) |
| `include/ninfer/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | `include/infernix/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | BF16 segment form plus a GGML IQ4_XS head |
| `include/ninfer/ops/rows.h`, `src/ops/rows/rows.cu` | `include/infernix/ops/rows.h`, `src/ops/rows/rows.cu` | Verbatim |
| `tests/ops/test_hyper_connection.cpp`, `test_projection_fp32.cpp`, `test_rows.cpp` | `tests/ops/test_hyper_connection.cpp`, `tests/ops/linear/test_projection_fp32.cpp`, `tests/ops/test_rows.cpp` | Ported with their Ops |
| `src/core/read_only_file{.h,_win32.cpp,_posix.cpp}` | `src/core/read_only_file*` | Unbuffered reads only (no mapping); each `read_direct` waits on its own event |
| `src/models/qwen4_exp/config.{h,cpp}` | `src/models/qwen4_exp/config.{h,cpp}` | IQ4_NL volume rows; no load options, MTP or YaRN; the vision config is Qwen3.5's (PR 14) |
| `src/models/qwen4_exp/frontend/ngram_hash.{h,cpp}` | same path | Verbatim |
| `src/models/qwen4_exp/weights.h`, `model.h`, `load.cpp` | `weights.h`, `model.h`, `load.{h,cpp}` | GGUF-mirroring parameters; record banks with no scales |
| `src/models/qwen4_exp/execution/parameters.{h,cpp}` | same path | One Linear per GGUF tensor |
| `src/models/qwen4_exp/execution/forward.{h,cpp}` | same path | One sequence; NInfer's single-sequence QSA Ops; interleaved query/gate rows; tiled GDN heads |
| `src/models/qwen4_exp/program/ngram_volume.{h,cpp}` | same path | Version 2 volume; keep-alive after Strata's PLE reader |
| `tests/models/qwen4_exp/test_{ngram_hash,ngram_volume,forward_real}.cpp`, `tests/fixtures/qwen4_exp/ngram_rows.txt` | same paths (`forward_real` with `45f20fb9`'s taps) | One sequence; INT8 KV built directly; timed decode |
| `tools/flash_next/{ngram,reference,block_check,strata_compare}.py` | same paths | The reference reads the `.ninfer` through the exact GGML decoders, with the canonical A8 cast |
| `src/core/{vmm_arena.{h,cu},cuda_vmm.h,vram_budget.{h,cpp},copy_batch.{h,cpp},link_probe.{h,cpp},power_throttling.{h,cpp}}` | same paths | Namespace; `CopyBatch` takes CUDA 12.8's `cudaMemcpyBatchAsync` form and falls back to single copies; a `cuMemSetAccess` that WDDM cannot make resident (`CUDA_ERROR_NOT_READY`) is a refusal, like an out-of-memory `cuMemCreate` (PR 12) |
| `src/core/host_memory.{h,cpp}` | same path | The snapshot and working-set reserve only |
| `src/core/pinned_upload.{h,cu}` | `src/core/device.{h,cu}` (`upload_pinned`) | Own translation unit |
| `src/models/qwen4_exp/memory_plan.{h,cpp}` | same path | Full RAM mode only (fits or refuses; no SSD tier, no `--expert-ram-mib`); the VRAM sizing and control law unchanged; `kv_growth_frames` is `grow_kv`'s arithmetic as a pure function (PR 12) |
| `src/models/qwen4_exp/program/expert_cache/expert_{cache,state}.{h,cpp}` | same paths | Verbatim but for `lend` reporting the keys it evicts (PR 12) |
| `src/models/qwen4_exp/program/expert_residency.{h,cpp}` | same path | Per-layer record strides in one frame size; no SSD tier or landing; frame lending (PR 10) whose give-back restores the displaced experts first (PR 12); a grow refills its new frames by score |
| `src/models/qwen4_exp/program/vram_monitor.{h,cpp}` | same path | Verbatim |
| `src/models/qwen4_exp/program/program.{h,cpp}` | `program/program.{h,cpp}`, `program/program_impl.h` | NInfer's execution contract (one lane; prefix reuse through NInfer's context cache with a Host or NVMe tier and context store images, [design](flash-next-context-cache.md), not Infernix's hybrid cache); device plan, elastic KV (PR 12: `grow_kv`'s order, frames released before the pool maps; the pool shrinks at the next admission that needs less instead of after 60 s idle, and a growth that does not fit fails the request), decode budget, warm start, VRAM control, link probe and decode graph from Infernix; PR 13: `mtp_draft`, `verify`, `commit_verified`, `mtp_catch_up` and `choose_draft_length` for one lane, the drafts read back for the n-gram rows, graphs per width |
| `src/models/qwen4_exp/execution/forward.{h,cpp}` (PR 13) | same path (`run_mtp`, `mtp_block`, the verification records) | Drafter experts through `offloaded_sparse_moe` from a device bank, the text IQ4_XS head as the draft head; verification records for the single-sequence Ops; the PLE and GDN a/b projections column by column in verification |
| `src/ops/qsa/prep.cu` `commit_tail_kernel` (PR 13) | `src/ops/qsa/qsa.cu` | One sequence; writes every committed slot in order (a commit across two blocks equals the one-column calls bit for bit) |
| `src/runtime/engine/qwen4_exp_instance.{h,cpp}` | same path | NInfer's EngineCore and ResourceManager; the context cache's Host tier sized in the RAM ledger; refuses unsupported options |
| `src/models/qwen4_exp/execution/vision.{h,cpp}` (PR 14) | `program/vision_program.cpp` (`vision_step`), `program/vision_window.h` | Qwen3.5's per-item tower and streamed weights in one lent window; no layer-major pass or stepped placement |
| `src/models/qwen4_exp/program/rope_positions.{h,cpp}`, `tests/models/qwen4_exp/test_rope_positions.cpp` (PR 14) | `program/rope_positions.{h,cpp}`, `program/vision_window.cpp` (`visual_slice`), their tests | Verbatim but for the namespace; the window placement not ported |
| `src/models/qwen4_exp/program/program.cpp` (PR 14) | `program/vision_program.cpp` (`plan_vision`, `vision_reserve`, `vision_release`, `stage_vision`, `stage_mtp_vision`) | The whole encode in the first prefill unit; a startup warm-up encode |
| `tests/models/qwen4_exp/test_{expert_cache,memory_plan}.cpp`, `tests/fixtures/expert_cache/lfru_conformance.txt` | same paths | The memory plan's RAM tests rewritten for full mode |

## Export conventions of this GGUF

Settled by evidence in PR 1 (`tests/convert/test_qwen4_exp_real.py` reproduces the data checks):

| Convention | Finding | Evidence |
|---|---|---|
| Unit-offset norms | Every F32 norm except `ssm_norm` stores `1 + gamma` of a BF16 `gamma`: `v - 1` is an exact BF16 word for 100% of values in all ten families, while only 8-67% of the stored `v` are (the BF16 checkpoint plus 1.0 in FP32). All 4,608 `ssm_norm` values are exact BF16 words, so it is the checkpoint's own multiplier. The model multiplies by the stored vector; nothing is subtracted. | data check; llama.cpp `conversion/qwen.py` (+1 on `norm.weight` except `linear_attn.norm`) and `conversion/qwen4exp.py` (+1 on PLE and indexer norms); Strata `gr.hpp`, `ple.hpp` and `gdn.hpp` apply `rms(x) * w` |
| `ssm_a` | `-exp(A_log)`; all 1,728 values are negative | data check; llama.cpp `qwen.py` |
| PLE hash constants | The literal multipliers equal splitmix64 with seed 1234, PLE index 0 and the 248,320-row vocabulary; the head sizes are the 16 consecutive primes from 20,000,000; the table rows are their sum rounded up to 128 | data check against the Infernix derivation |
| `attention/query_gate` | per head, 256 query rows then 256 gate rows | llama.cpp graph, Strata `layer.cpp`, Infernix mapping |

Settled in PR 8 by running the whole model (teacher-forced on the PR 0 set, 2,557 positions,
against Strata), changing one convention at a time from the shipped choice:

| Convention | Shipped | Alternative tried | Result of the alternative (shipped: ppl 7.888, ΔNLL +0.0064) |
|---|---|---|---|
| GDN value-head pairing | tiled: value head `h` reads key head `h % 16` | grouped (`h / 3`) | ppl 9,429, ΔNLL +7.09, top-1 agreement 7.3% |
| GDN output gate | sigmoid | SiLU | ppl 505,835, ΔNLL +11.08 |
| PLE convolution taps | tap `j` reads `nrm[t - (K-1-j) * 3]` (GGUF order) | reversed | ppl 8.232, ΔNLL +0.049 ± 0.012, top-1 agreement 86.9% |
| Attention query/gate rows | interleaved per head (256 + 256) | blocked (all query rows, then all gate rows) | ppl 213.5, ΔNLL +3.30, top-1 agreement 34.1% |
| GDN convolution taps | GGUF order | reversed | ppl 3,936, ΔNLL +6.22, top-1 agreement 4.7% |

The hyper-connection `down|inject` parent (down rows, then injection rows) and the `1 + gamma`
norms are confirmed by the shipped result itself (any error there is not a small loss), and the
FP64 block check agrees per op chain.

## Tokenizer

The synthesized `tokenizer.json` equals the Qwen3.8-27B artifact's in vocabulary (248,044 entries),
merges (247,587), pre-tokenizer, normalizer, decoder and the ids and contents of all 33 added
tokens. The `special` flag differs on six tokens (`<|fim_prefix|>`, `<|fim_middle|>`,
`<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`): Hugging Face has them non-special,
and llama.cpp's converter stores every `<|...|>` added token as a control token, so the GGUF cannot
say. `--resource tokenizer.json=PATH` keeps the Hugging Face flags after checking the file against
the GGUF. Whether Flash-Next's own Hugging Face tokenizer equals Qwen3.8-27B's is assumed, not
checked. The GGUF's chat template is used as is; it differs from the NInfer Qwen3.8-27B template.

## Measurements

Host: Ryzen 9 5950X, Samsung 980 PRO (C:), Python 3.12.7, numpy 2.5.3, torch 2.11 (CPU).

| Workload | Result |
|---|---|
| `--subset dev` conversion, GGUF on C: to C: | 2.93 GB in 20 s on the first run, 11 s with the GGUF ranges cached; peak working set 0.69 GiB (0.48 GiB of it is the interpreter with torch loaded) |
| Full n-gram table paging, output discarded | 28.80 GB read in 43 s (0.67 GB/s, CPU-bound in Python); peak working set 0.57 GiB |
| GGUF header parse (73 keys, 248,320-token vocabulary, 1,224 tensors) | 0.7 s |

PR 2b, same host, GGUF on C: to C:, peak working set from `psutil` after the run:

| Workload | Result |
|---|---|
| `--subset dev` conversion with its 4,500-row volume | 2.93 GB artifact and 0.41 MB volume in 20.0 s; peak working set 0.62 GiB |
| Whole n-gram volume through the writer, output discarded | 320,001,536 rows, 7,111,146 blocks, 29,127,258,112 bytes in 51.4 s (0.56 GB/s read); peak working set 0.60 GiB |

The full-model conversion ran in PR 8 (to K:; see the PR 8 measurements).
