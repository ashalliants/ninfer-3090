# Tesla V100 (sm_70) port

Status: **runs on a V100**, Qwen3.8-27B and Qwen3.8-Flash-Next both tuned. Decode is tuned (see
"Decode"); prefill's wide GEMMs run on the tensor cores through cuBLAS and prompt attention on FP16
tensor cores (see "Prefill"); the GDN recurrence runs the sequential FP32 kernel. The user-facing
summary (requirements, launcher defaults, current numbers) is [docs/tesla-v100.md](../tesla-v100.md);
this page is the engineering record behind it.

The engine's kernels are written against Ampere: `cp.async`, `ldmatrix`, `mma.sync.m16n8kK` for
BF16, FP16, TF32, INT8 and INT4. Volta has none of them (its tensor cores run FP16 `m8n8k4` only),
plus a 96 KiB shared-memory limit instead of GA10x's 99 KiB. The port keeps every kernel and
emulates the instructions underneath it.

## What the port changes

| Piece | Where | What it does on sm_70 |
|---|---|---|
| Instruction emulation | `src/ops/common/volta_compat.cuh`, dispatched from `mma.cuh`, `memory.cuh`, `math.cuh` | `ldmatrix` x2/x4 (plain and `.trans`) from warp shuffles and `ld.shared`; `mma` m16n8k16 BF16 and FP16, m16n8k8 TF32, m16n8k32 and k16 S8 (`dp4a`), m16n8k64 S4 (nibbles split to bytes, `dp4a`), the FP16-accumulate form; `cp.async` as a synchronous copy |
| Warp reductions | `masked_reduce_{max,and,or,add}` in `ops/common/warp.cuh` | `__reduce_*_sync` on sm_80+, shuffles below (the sampler's partial-lane mask included) |
| Raw asm call sites | `memory.cuh` `cp_async_mem`, `cp_async_zfill_mem`, `cp_commit_mem`, `cp_wait_mem` | kernels that staged with their own `cp.async` asm now use these (same sm_80+ code, a synchronous copy on Volta) |
| Shared-memory limit | `ops/common/smem_limit.h`, `NINFER_SM70` from CMake | `kMaxDynamicSharedBytes` is 96 KiB on Volta and 99 KiB elsewhere. Every schedule static-asserts against it, so a schedule that cannot launch fails to compile |
| Build gates | root `CMakeLists.txt`, `src/ops/CMakeLists.txt`, `softmax_attention/sources.cmake`, the Qwen3.5 startup check | arch 70 takes the sm_8x path (no PDL, FP8 A8 and NVFP4 A4 stubbed) |
| `sigmoid_approx` | `ops/common/math.cuh` | `tanh.approx` is sm_75; Volta uses `1 / (1 + exp(-x))` |
| Decode GEMVs | `ops/linear/common/stream_gemv.cuh` (+ `stream_gemv_decode.cuh`, `Q4StreamFormat` / `Q5StreamFormat` / `Q6StreamFormat` / `Q8StreamFormat` in the row-split storage headers) | every Q4/Q5 T = 1 GEMV, routed from `launch_q4_a16_gemv`, `launch_q5_a16_gemv`, the Q4 SwiGLU pair and the Q5 residual linear; the Q6 head from `launch_q6_gemv_t1`; and T = 2..8 (speculative verification) from the Q4/Q5 `linear_add`, Q4 SwiGLU, Q6 head and Q8 sliced-K launchers. See "Decode" below. See "Decode" below |
| Decode routes for the fused input projections and Q4 `linear_add` | `q4_q5_gdn_input_plan.cpp`, `q4_q5_attn_input_plan.cpp`, `q4_linear_add.cu` | T = 1..6 (input projections) and T = 1 (`linear_add`) take their GEMV/SIMT schedules instead of the emulated small-T `mma` |

The mma emulations accumulate in FP32 (INT32 for the integer shapes) exactly as the instruction
does, so results match the hardware up to summation order. `mma_f16_acc16` rounds once at the end
instead of after every step, which is more accurate than the hardware.

## Performance: what to expect and why

The emulated `mma` is a correctness baseline. Per m16n8k16 it costs about 32 shuffles and 64 FMAs
per lane (about 45-60 SM clocks) against 2 HMMA instructions of real tensor-core work, so a
kernel that spends its time in `mma` runs at roughly 6-9 TFLOP/s on a V100 instead of the card's
~100. Where that matters:

- **Decode at T = 1** is bandwidth-bound and runs no emulated `mma`; see "Decode" below.
- **Small T (2-16, MTP / DFlash verification)** is where the 3090 route tables pick `mma`
  schedules. These are tuned for sm_86 and are the first thing to re-measure; several formats
  also have `*_simt` schedules that may win on Volta.
- **Prefill** is compute-bound. Its wide GEMMs run on HMMA through cuBLAS (see "Prefill"); the
  prompt attention, GDN chunked recurrence and the sub-129-token tails still run emulated `mma`.
- `cp.async` is a synchronous copy, so the multi-stage global-to-shared pipelines lose their
  overlap. Occupancy hides some of it.

### Decode

Measured on a V100-PCIE-32GB (OCuLink Gen3 x4, 200 W cap, 879 GB/s sustained read from
`tools/hbm_bandwidth_probe.cu`), Qwen3.8-27B, greedy, INT8 KV, 2026-11-16:

| Build | decode tok/s |
|---|---|
| port as first built | 16.0 |
| + decode routes for the fused input projections and Q4 `linear_add` | 30.8 |
| + streaming GEMV for every Q4/Q5 T = 1 linear | 40.5 |
| + decode attention: halved split tiers, CUDA-core QK and PV at one row tile | 41.9 |
| + streaming GEMV for the Q6 LM head | 42.1 |

Speculative decoding (same prompt, 200 greedy tokens, `--lm-head-draft`):

| Build | plain | MTP3 (`--draft-tokens 3`) | DFlash2 (`--draft-tokens 7`) |
|---|---|---|---|
| T = 1 work above only | 42.1 | 26.0 | 36.7 |
| + streaming GEMV for T = 2..8 and the Q8 drafter | 41.4-42 | 69.8 | 46.4 |
| + verify widths (input projections, tensor-core GEMV, chunked attention) | 42.2 | 76.1 | 67.6 |

Acceptance on that prompt: MTP3 3.3 tokens per round, DFlash2 5.2. Without `--lm-head-draft` MTP3
reads 77.4 (same acceptance): on this card the optimized proposal head is not faster.

Across three kinds of text (200 greedy tokens each; code / technical prose / fiction), MTP3 remains
the best setting: 76.1 / 68.0 / 48.2 tok/s at 76 / 65 / 36% acceptance, against 66.7 / 58.5 / 38.0
for MTP4, 68.8 / 54.4 / 32.7 for MTP5 and 65.6 / 41.1 / 27.4 for DFlash2 K=7. MTP2 edges it on
fiction (50.7). At a 16K-token context (bench corpus) plain decode is 38.7 tok/s.

The model's weight floor at 879 GB/s is 17.4 ms per token (`tools/perf_model.py`), about 57 tok/s.

**Where the first two steps came from.** The port's Volta decode routes lived in `q4_dispatch.cpp`,
but the 27B calls the fused ops (`gdn_input_proj`, `attn_input_proj`, `linear_add`) whose plans
kept their sm_86 routes: emulated small-T `mma` at T = 1, compute-bound at ~6.5 TFLOP/s because a
T = 1 tile still pays for 16 padded columns. Those were ~43 of 67 ms per token. Each plan now has a
`NINFER_SM70` table that sends decode to the GEMV/SIMT schedule it already had.

**Why the GEMVs needed their own kernel.** With `mma` gone, the Q4/Q5 GEMVs ran at 35-61% of the
read rate. They are instruction-bound on Volta, not memory-bound: every weight went through an
integer-to-float or half-to-float conversion, which sm_70 issues at a quarter of the FP32 rate
(about 7 SASS instructions per weight in the half2 decode), and their latency hiding relied on
`cp.async`, which is synchronous here. `stream_gemv.cuh` keeps the same Output / Epilogue contract
and replaces only the inner loop:

- the activation is staged once per persistent CTA as fp32, pre-scaled per k parity, so a stored
  field becomes an fp32 operand with ONE `LOP3` under a magic exponent (`stream_gemv_decode.cuh`
  has the arithmetic) -- about 2.5 instructions per Q4 weight, none of them conversions;
- each lane keeps the next step's 16-byte weight vectors in flight across row boundaries;
- shared-memory chunks are swizzled so the `LDS.128` reads are conflict-free.

It is plain CUDA and compiles everywhere; `kStreamGemvDefault` turns it on for sm_70 only, and
`NINFER_STREAM_GEMV=0/1` overrides that at runtime, so the same binary A/B's it on any card. Per
shape, in the model (Nsight Compute, native clocks): the SwiGLU pair at 88% of the read rate, the
K = 17408 down projections at 77-80%, the 12288 x 5120 Q5 projection at 79%, and the K = 5120/6144
shapes under 25 MB at 63-72%. Those small shapes are near the ~80% a load-only kernel reaches at
that size: launch ramp, staging and tail dominate them. A sweep of warps per CTA and slices in
flight (1-32 warps, 1-2 slices) moved no shape by more than 2%.

Greedy text matches the previous build for ~150 tokens, then may take a different but equally
valid continuation: the streaming GEMV sums in a different order (relative error ~1e-6 against
fp64 on random weights). The op suites' fp32/fp64 oracles pass.

**Speculative verification (T = 2..8).** MTP3 verifies four columns per round and DFlash2 eight,
and those widths went to small-T `mma` routes (`q4_ksplit_mma`, `q5_small_t_mma`, the Q6 head's
`q6_small_t_mma`, and the MTP drafter's Q8 `q8_a16_sliced_k_mma`), emulated and compute-bound
here: a T = 4 verify cost ~5x a decode step, so both modes ran slower than plain decoding. The
streaming GEMV now takes 1..8 tokens:

- formats decode a packed word into eight fp32 operands once (`decode_word`) and every token adds
  one FFMA per weight, so the decode is shared across the verify width;
- when Cols x K of x' does not fit in shared memory (K = 17408 at T >= 2), K is staged in chunks
  and each warp keeps its rows' partial sums in shared memory between chunks (`Chunked`, a
  compile-time property, so the T = 1 kernel carries none of it);
- from T = 2, x' reads from shared memory bound the kernel (one per token per weight), so a plain
  GEMV streams two rows per warp (r and r + rows / 2) against each read: 20-36% faster at T = 4
  and 30-35% at T = 8; four rows spill registers and lose;
- `Q8StreamFormat` (8-bit fields fill bits 15..22, odd weights under a 16.0 magic) serves the
  drafter through the identity-row `launch_q8_a16_sliced_k_mma`;
- the T = 1 kernel is held to 64 registers (`kStreamGemvMinBlocks`): unconstrained, the Q6 decode
  took 75 and lost a CTA per SM.

Hooks for the verify widths: Q4 `linear_add` (T <= 8), the Q4 SwiGLU small-T launcher, Q5
`linear_add` (split2 and small-T launchers), the Q6 T = 2 and small-T launchers, and the Q8
sliced-K launcher. T = 4 costs about 1.6-2x a T = 1 GEMV on the CUDA cores (FMA- and
shared-memory-bound); tensor cores would flatten that, and are the same work as a Volta prefill
path.

MTP3's output matched plain decoding before this change and can now differ at near-ties: the
verify GEMVs sum in a different order from the T = 1 ones (chunked K, paired rows). DFlash2
already differed from plain on this card before any of this work (T = 1 and T = 8 took different
kernels even with the streaming GEMV off); speculation is exact with respect to the target's own
verify-width logits.

**Decode attention** (`causal_attention_small_t_i8_tiled`) went from 81.5 to 31.7 us per layer at
~430 keys. Its QK ran emulated `mma_s8` in one producer warp per CTA, a ~2,300-shuffle serial
chain per 32-key tile with seven warps waiting at the barrier; PV ran emulated `mma_f16`. Two
changes, both inside the shared kernel:

- `CausalSmallTSplitTiers` (`small_t.cuh`) is now the one table of keys-per-split tiers that the
  device policy and the host launch capacity both read; sm_70 halves every tier, so more CTAs each
  carry a shorter serial chain (81.5 -> 53.2 us).
- `SimtDecode` (`small_t_i8.cuh`, one row tile only): the whole CTA computes the int8 QK dots with
  `dp4a` into the consumed K tile, the producer reads its fragment positions from there, and PV
  accumulates on the CUDA cores into the same fragment positions (53.2 -> 31.7 us). Everything
  else in the kernel is shared. Greedy output is byte-identical.
- Register-staged tiles (`StagedTiles`, SimtDecode at two or more tokens, INT8 G64 and rk8v4): sm_70
  has no `cp.async`, so `issue_kv_tile`'s copies were a load and a store each and every warp stalled
  on the next tile's loads right before PV. The next tile's codes and scales are now loaded into
  registers at the top of the iteration and stored where `issue_kv_tile` stored them, so they are in
  flight across QK, softmax and the V dequant; the shared tiles hold the same bytes at the same
  barriers, and the outputs are byte-identical. Op level, 27B geometry, cached entry, CUDA Graph, warm,
  ABBA (`ninfer_causal_softmax_attention_bench`, 2026-11-22): T = 2 (the chunk every MTP verify runs
  as) 13.5-19% faster in INT8 (4K-128K keys; 128K: 2.60 -> 2.10 ms a call) and 14-22% in rk8v4, T = 4
  (two chunks) 14-21% and 14-25%. At T = 1 the staged registers raised the spills of a kernel already
  at its 128-register cap (4 -> 56 bytes) and it ran 4-10% slower, so T = 1 keeps the old path. End to
  end, Qwen3.8-27B, MTP3, INT8 KV, `ninfer-serve` with the `run-v100.sh` flags, 256 greedy tokens on
  exact-hit repeats of a checkpointed prompt, ABBA at a 200 W power limit (2026-11-23): 46.75 -> 47.88
  tok/s at 40K context (+2.4%) and 23.46 -> 25.02 at 162K (+6.7%); 4K (57.83 -> 57.92), the 12 short
  chats (63.6 both) and plain `ninfer_bench` 4K decode (41.5 both) are unchanged, and all 24 responses
  are md5-identical. The sm_86 SASS of the kernel is identical with and without the change, so the
  RTX 3090 is unaffected by construction (not measured on a 3090).

**Small-T attention on the tensor cores (`NINFER_SM70_VERIFY_ATTN_HMMA`, on by default in an
`sm_70` build).** With the verify GEMVs flat (`--verify-a8`), long-context rounds were attention:
at 42.7K tokens an MTP3 round spent 18.6 of its 46.5 ms in verify attention (nsys), because the
`SimtDecode` path above serves one row tile (two tokens on the 27B), so a four-column verify ran as
two two-token launches per layer, each re-reading the whole INT8 cache and doing PV as FP32 FMAs
on the CUDA cores. `small_t_i8_hmma.cuh` serves 1..8 columns of the 27B geometry over the INT8 G64
cache in one launch on `mma.sync.m8n8k4` (FP16 in, FP32 accumulate), with the fragment maps
measured for `qsa/attention_wide_hmma.cu`:

- One CTA per (KV head, split), the same partials (FP32 accumulator, natural-log m and l) and the
  same reducer as the general kernel; the split capacity drops to 20 per KV head
  (`kVoltaVerifyHmmaSplits`), one wave of one CTA per SM. Groups of warps (four, or two from six
  columns, where a fourth spilled) take 16-key tiles in turn, double-buffered in shared memory, and
  merge in shared memory at the end. Holding the next tile in registers instead took the kernel to
  255 registers with spills and ran ~1.8x slower.
- QK takes the Hadamard-rotated query in FP16 (per row, scaled by a power of two so its absmax
  lands in [2^7, 2^8)) against the key codes widened exactly to FP16, summed per 64-group in FP32
  and scaled by the group scale in FP32. PV folds the value scale into the probability,
  A = FP16(p * vscale * 2^u), against the exactly widened value codes; u is set from the first tile
  and moved only when a tile's largest scale x 2^u would leave [2^0, 2^15), and the rescale of the
  running output is skipped when no row's maximum moved, which is most tiles.
- QK on `dp4a` (the query quantized to INT8 per 64-group, as the CUDA-core path does) was built in
  the same structure and measured no faster (T = 4 at 164K keys 2.81 vs 2.80 ms, T = 8 5.77 vs 5.49
  in the one-stage form), so the FP16 query, which keeps more bits, is the one kept.
- Routing (`causal_sm70_verify_hmma`, `launch.h`): five to eight columns at every key count; three
  and four (two CUDA-core chunks) from 1,024 keys, two from 6,144 and one from 20,480, where the
  CUDA-core kernel's lower fixed cost stops winning (table below). In the served model a four-column
  call at a few hundred keys took 74 us against the two chunks' 60 (nsys, a 300-token code reply,
  about 0.25 ms of a 30.5 ms MTP3 round), more than the op bench shows there, so the four-column
  threshold sits at 1K, where the op bench has 53 against 69.
  The route, the split capacity, the graph launch shape (route bit 19) and the launcher all read
  the envelope's maximum visible keys, so a captured graph re-captures when the decode envelope
  crosses a threshold. `NINFER_SM70_VERIFY_ATTN_HMMA=0` restores the previous routes exactly.
  Outside `sm_70` builds the kernel is not compiled and the switch is always off, so the RTX 3090
  path is unchanged by construction.

Op level, 27B geometry, INT8 KV, appending entry, CUDA Graph, cold L2, us per layer call (median of
20, best of two interleaved rounds; `ninfer_causal_softmax_attention_bench`, 200 W, 2026-11-24),
previous route / tensor cores; "-" where the route keeps the CUDA-core kernel:

| Columns | 1K keys | 4K | 16K | 40K | 160K |
|---|---:|---:|---:|---:|---:|
| 1 (plain decode, MTP drafter) | 28 / - | 57 / - | 141 / - | 313 / 325 | 1,909 / **1,146** |
| 2 | 36 / - | 77 / - | 194 / **156** | 437 / **326** | 2,720 / **1,146** |
| 3 | 63 / **53** | 143 / **89** | 340 / **189** | 757 / **431** | 4,637 / **1,613** |
| 4 (MTP3 verify) | 69 / **54** | 154 / **89** | 387 / **189** | 872 / **427** | 5,435 / **1,596** |
| 6 | 100 / **55** | 230 / **92** | 579 / **233** | 1,309 / **516** | 8,149 / **1,914** |
| 8 (DFlash2 K7 verify) | 132 / **60** | 305 / **97** | 773 / **237** | 1,744 / **520** | 10,867 / **1,927** |

A crossover sweep at one and two columns set the thresholds: one column runs 216 / 251 us at 24K
keys and 486 / 782 at 64K on the two kernels, with 40,960 the one measured count where the CUDA-core
kernel's split tier fits better (325 / 313); two columns cross between 4K (80 / 77) and 6K
(89 / 92). The previous route's cost grows with the column count because each two-token chunk
re-reads the cache; the tensor-core kernel reads it once, so its cost barely depends on the width.

Accuracy, the decode-path KLD gate (`ninfer-perplexity --kld-write` / `--kld-base`) at long
context: two 32,768-token chunks of PG-19 text (books 01 and 02), each prefilled to 16,384 tokens
and then decoded, so every scored row attends over 16-32K keys; INT8 KV, FP16 GDN state; base = the
FP32 linears (`--no-verify-a8`) with the previous attention route at T = 1:

| Run | KLD mean | p99 | same top-1 | PPL |
|---|---:|---:|---:|---:|
| previous route, T = 4 | 6.21e-4 | 3.20e-3 | 98.90% | 7.9720 |
| tensor cores, T = 4 | 6.16e-4 | 3.19e-3 | 98.83% | 7.9712 |
| previous route, T = 8 | 6.35e-4 | 3.20e-3 | 98.91% | 7.9737 |
| tensor cores, T = 8 | 6.18e-4 | 3.23e-3 | 98.86% | 7.9728 |
| tensor cores, T = 1 (from 20,480 keys) | 5.27e-4 | 2.63e-3 | 99.00% | 7.9733 |
| `dp4a` QK variant, T = 4 | 6.21e-4 | 3.33e-3 | 98.89% | 7.9708 |
| previous route, T = 4, `--verify-a8` (the default) | 1.01e-3 | 5.20e-3 | 98.52% | 7.9756 |
| tensor cores, T = 4, `--verify-a8` (the default) | 1.01e-3 | 5.09e-3 | 98.43% | 7.9767 |

(base PPL 7.9720.) The kernel is as far from the T = 1 base as the previous route's own verify
reordering is, at every width. The T = 1 row is its distance from the CUDA-core kernel at one
column, which rounds the query to INT8 where this kernel keeps it in FP16. Greedy text therefore
parts from master's at near-ties.

End to end, `ninfer-serve` with the router's flags, 256 greedy tokens (`ignore_eos`) on exact-hit
repeats of a PG-19 prompt, 200 W, against master `96b14bca` (with #609's split fix; MTP3 ABBA,
three repeats a leg, the legs of each side within 0.05 tok/s; DFlash2 one leg each; plain from
the same build pair), tok/s and ms per speculative round (generation time over rounds; a round
emits its accepted drafts plus one):

| Context | MTP3 | master | DFlash2 K7 | master | plain | master |
|---:|---:|---:|---:|---:|---:|---:|
| 4,215 | 84.6 (31.4 ms) | 75.9 (32.3) | 85.4 (40.9) | 62.3 (44.0) | 43.7 | 43.7 |
| 42,663 | 69.5 (36.3) | 53.0 (50.1) | 50.4 (48.2) | 37.3 (75.9) | 36.1 | 34.6 |
| 172,487 | 56.7 (52.3) | 26.4 (109.6) | 34.9 (71.0) | 14.0 (184.6) | 24.2 | 20.7 |

nsys at 42,663 tokens, MTP3: verify attention went from 18.6 ms of a 46.5 ms round (3,504 chunk
launches of the CUDA-core kernel for 1,749 verify calls) to 5.8 ms (364 us a call). The greedy
text differs from master's at every context of this prompt (near-ties; at 172K both summarise
the same section in different words), and acceptance moved with it, so at 4K most of
the tok/s gain is acceptance: the round itself is 3% (MTP3) and 7% (DFlash2) cheaper.

Short prompts (the nine-prompt set, 300-7,230 tokens generated, same flags, ABBA): an MTP3 round
costs the same as master's on the short replies (30.50 against 30.50 ms on `code`, 30.51-30.60
against 30.49-30.58 on the others) and 2.8% less on the 7K-token `struct` reply (31.65 against
32.55 ms, 126.3 against 122.8 tok/s). `code` and `struct` produce master's text token for token;
the other seven replies differ after a few dozen tokens, and their tok/s moves with acceptance
both ways (`story` 71.9 against 77.0, `fr` 78.9 against 80.8 on 176/367 against 179/360 accepted).
DFlash2 K7 (eight-column verifies, which take the tensor cores at every length): `struct` 189.2
against 177.4 tok/s with identical text; the others differ in text and range from `think` 115.2
against 125.6 to `code` 142.6 against 136.6. Plain decode below 20K keys runs the same kernel as master.

What is left in decode, by size: the GEMVs' gap to the read rate (~3.5 ms), the per-kernel
overhead of ~690 launches per token (~1.2 ms of gaps plus ramp/tail), and the GDN recurrent update
(~0.5 ms, 32% of its byte rate). The Q6 LM head (993 MB) runs on the streaming GEMV at 86% of the
read rate in the model (1477 -> 1309 us); its decode is the most ALU-heavy of the three formats,
so back-to-back microbenchmarks that hold the 200 W power cap understate it.

**Verify widths.** A speculative round verifies T = K + 1 columns, and there the SIMT streaming
GEMV, one FFMA per weight per token, stops being bound by the weight stream: at T = 4 an MTP3
round spent ~26 ms of ~50 in it against a ~19 ms read. Two changes:

- The attention and GDN input projections ran T = 2..6 on SIMT tiles at 40-60% of the stream and
  T = 7..8 on the emulated sliced-K mma; they now take the streaming GEMV for T = 2..8 (11.4 ->
  ~6 ms per MTP3 round).
- `ops/linear/common/stream_hmma.cuh` runs the same function on the tensor cores: each lane of
  `mma.m8n8k4` streams one weight row, quadpairs share the activation, the format decodes a
  packed word straight into exact FP16 integer pairs, and group scales are applied in FP32 per
  32-weight slice. Lane pairs read each 32-byte sector whole and swap halves (half-sector reads
  from 32 rows at once cost 2-3x the bytes in DRAM). It sits behind `try_launch_stream_gemv`, so
  every Output / Epilogue works unchanged, and takes Q4 rows of K <= 6144 at T >= 5, or T = 4 on
  the SwiGLU-sized matrices; its cost barely grows with T, the SIMT kernel's does (table in
  `try_launch_stream_hmma_rows`). `NINFER_STREAM_HMMA=0` turns it off.
- Q5's form (its high plane spread into the FP16 magic with one multiply per four weights) loses
  at T = 4 (6144 x 5120: 66 SIMT vs 75 us; 5120 x 17408: 150 vs 266 us) and wins from T = 6 at
  K <= 6144 and at T = 8 on the down projection (117 -> 96, 103 -> 80, 259 -> 230 us at T = 8).
  So it can only help the two-lane verify (T = 8). There, Qwen3.8-27B goes 78.6 -> 79.5-80.8 tok/s
  in total, but a lane's T = 8 verify then sums differently from its T = 4 one. The greedy outputs
  of a prompt alone and in a pair part sooner: 243 -> 69 shared tokens of 256 on one of two
  prompts, a near tie (logprob -0.80 vs -0.73). It is opt-in, `NINFER_STREAM_HMMA_Q5=1`.

`ninfer_bench -pg 2048,256`, bench corpus, optimized proposal head (tok/s):

| | MTP3 | MTP5 | MTP7 | DFlash2 K=7 |
|---|---|---|---|---|
| before | 78.3 | | | |
| + input projections on the streaming GEMV | 82.3 | 75.6 | 72.5 | 72.3 |
| + tensor-core verify GEMV | 86.3 | 85.9 | 85.4 | 84.2 |
| + int8 verify GEMV (`--verify-a8`; 2026-11-24, 200 W, master then 88.0 and 93.2) | 118.9 | | | 141.9 |

**Two lanes (`--max-concurrency 2`).** Two lanes widen what one lane runs on the streaming
kernels into calls the shape tables still sent to emulated Ampere mma: the MTP proposal head (Q4
131072 x 5120 under `--lm-head-draft`) at T = 2 on `q4_ksplit_mma`, 2.8 ms a call; the Q6 LM head
at T = 8 on `q6_small_t_mma`, 5.7 ms a call; and verify attention for B = 2, T = 4 on the 4-token
mma tile. Now:

- every Q4 and Q5 linear of 2..64 tokens tries the streaming GEMV (its tile GEMM past eight
  tokens) before the tables (`ops/linear/common/stream_tokens_sm70.h`): the proposal head 2.8 ->
  0.40 ms a call;
- the K-chunked streaming GEMV runs further waves of CTAs when one wave's partial sums would not
  fit in shared memory (the 248320-row head at T = 8 needed ~97 KB a CTA), so the Q6 head takes it
  at T = 8: 5.7 -> 3.7 ms a call;
- INT8-family verify attention chunks batched rows two tokens at a time, as one lane does, onto
  the CUDA-core `SimtDecode` path.

Qwen3.8-27B, MTP3, two 256-token greedy requests at once: 70.9 -> 79.2 tok/s in total (one lane
alone runs 65-71). One lane's output is bit-identical. The changes also make a request's tokens
depend less on whether a second lane is running: the two prompts matched their one-lane output for
the first 42 and 123 of 256 tokens before, 243 and 256 after.

### Decode: Qwen3.8-Flash-Next

Same card at 250 W, Flash-Next IQ2_XS (`qwen4exp-iq2xs-mtp-vision.ninfer`), INT8 KV, 8K context,
warm expert cache (18.7K frames, 26 GiB), 200 greedy tokens, 2026-11-17:

| Build | plain | MTP3 |
|---|---|---|
| port as first built | 44.7 | |
| + compact routing kernels and one-column router projection | 48.3 | |
| + `ggml_mmvq`: one decode site, lane-parallel Q8_1 cast | 52.2 | 61.8 |
| + `--gdn-head-map` | 53.3 | 63.1 |

A decode step is ~900 launches of ~40 different kernels, more code than an SM's instruction caches
hold, so every kernel starts cold. In a standalone bench, evicting only the instruction caches
before a dense IQ4_XS GEMV added ~9 us to its ~30 us (`imc_miss`/`no_instruction` stalls), and a
kernel that one warp runs straight through pays for every instruction it fetches: the fully
unrolled routing top-k (57 KiB of code) took ~27 us in the model against ~10 us warm. On this card
code size is time. The changes above are bit-identical and shared with the 3090:

- `route_column` rolls the top-k rounds (lane k keeps pick k); 57 KiB -> 8 KiB.
- `projection_fp32` has a one-column instance for the decode router (37 KiB -> 8 KiB).
- `ggml_mmvq` decodes at one site instead of two, and casts x to Q8_1 four lanes per group
  (`quantize_q8_1_part`) instead of one thread unrolling all 32 values (30 KiB -> 16 KiB). The
  same-format pair kernel runs one copy of the body.

Tried without a gain in the model: the HC mixers' K1a at three CTAs per SM (it spilled; nine rows
a CTA fixed the second wave instead, below) or with a one-column instance (~3 us per call in
isolation, nothing measurable in the model).

The routed experts' gate/up: the IQ2 codebook in shared memory helps the register kernel
(`gate_up_registers_kernel`, the tuned profile's) and is kept: 101 -> 84 us a call under MTP3.
These did not help:
- the codebook in shared memory for the staged kernel (`gate_up_kernel`);
- that kernel's slice staging issuing every load before its stores (the `cp.async` emulation is a
  synchronous copy);
- in the register kernel:
  - the call's precast activations copied to shared memory per block (no change);
  - three blocks per SM (10% faster in isolation, 1.5 tok/s slower in the model);
  - each warp copying its gate and up rows to shared memory with coalesced 4-byte loads, the next
    item's rows loaded while the current one computes (84 -> 98 us a call). Lanes 0-7 decode
    sub-blocks of the same 82-byte block, so the kernel's small global loads mostly hit L1. The
    copy adds stores, warp syncs and byte reads from shared memory, and 7 KiB of code.

Gate/up is bound by instruction issue in the decode and the dp4a products, not by its loads.
`--cpu-verify-misses` (verification misses served on the CPU) cost 1.5 tok/s.
The bit-identical MoE options
(`--moe-precast-input`, `--moe-gate-up-registers`, `--moe-fused-route`) are neutral here.

The HC mixer pair (`hyper_connection_mix_fused.cu`) was sized for the 3090's 82 SMs: its down
kernel's 164 CTAs ran as a second wave of four on the V100's 80. On sm_70 it takes 9 rows a CTA
(144 CTAs, one wave). The sums of squares stay on the first 256 threads, so their order and the bits
are unchanged. The per-column work was the larger cost at verify widths:
- the columns' slices are now staged together, with one barrier pair;
- the `Rn` pass reads `w` once per four features instead of a dependent load per element;
- the shared `Rn` layout splits each 8-value chunk into two planes (the lanes' `float4` reads had a
  two-way bank conflict);
- the up kernel prefetches the collapse's residual values across lanes instead of loading them on
  lane 0 per column.

Bench pair at T = 1 / 4: 43.3 / 75.4 -> 41.2 / 60.9 us, bit-identical at T = 1-16. MTP3 decode
72.4 -> 74.0 tok/s.

The up kernel (`hc_up_collapse`) then grew ~3.2 us per verified column, against ~2.4 for the down
kernel. Removing parts of it in the bench showed where (none of these variants is exact): the
float4 reads of m from shared memory (~1 us a column: lane l's chunk is 32 bytes from lane l + 1's,
a two-way bank conflict), and the per-column tail, four warp butterflies and lane 0's four sigmoids
with an IEEE division each, issued while the other 31 lanes wait. On sm_70 (`kUpScatter`) m is
stored in two planes as `Rn` is; the four butterflies are one reduce-scatter (at offset 16 each half
of the warp keeps two streams, at 8 each quarter keeps one), which adds the same pairs of values, so
each stream's sum has the butterfly's bits; and lanes 0, 8, 16 and 24 take the four sigmoids at once
before lane 0 collapses them in the same order. Bitwise identical to the previous kernel at
T = 1-16 with and without injects (checked against a copy of it). Bench up kernel at T = 1 / 4 / 8:
16.6 / 26.6 / 39.2 -> 16.0 / 25.0 / 33.4 us; in the model, 27.2 -> 24.3 us a call over the code
request's 110 calls a round (-0.32 ms a round). Flash-Next decode, ABBA at 200 W (router flags, tok/s
story / code / Chinese / French / thinking): 80.7 / 123.2 / 78.9 / 71.7 / 100.0 -> 81.3 / 124.7 /
79.6 / 72.1 / 101.0, same text.

What did not help, all exact:
- two W_down rows per warp in the down kernel (half the `Rn` reads per product, half as many CTAs
  re-staging the residual, one CTA per SM): 4-14% faster in the bench, but 22.9 -> 26.4 us a call
  in the model. Every one-CTA-per-SM shape measured lost in the model (8 columns a pass: 39 us);
- the up kernel's weights decoded once instead of per column, or two columns a step: both spill
  under the 64-register bound that two 512-thread CTAs per SM leave.

Two other levers were measured and left alone. The draft head's 106,056 IQ4_XS rows are ~144 MB a
step and take ~190 us, about 760 GB/s, already at the memory bound. And capping the window by
measured verify cost would not pay on code: from the per-position acceptance of the code request,
the rounds that offer a fifth to seventh draft accept most of them, and dropping those positions
loses about as many tokens as the verify time it saves.

MTP rounds with `--mtp-joint-acceptance` drafted `--draft-tokens` (3) steps every round, ~1.2 ms a
step here, then verified the prefix the joint rule kept: over 96 rounds of the bench story, position
2 was offered 46 times and position 3 only 10. `--mtp-staged-drafts` drafts one step at a time and
stops once the joint product of the drafts so far is below the threshold (no later draft could then
be verified, since the rule breaks at the first such draft). Every draft and probability is the full
chain's (step 0 is the one-step draft graph, each later step a single-step continuation graph), the
window is the same, and so is the output; a round pays a host sync per extra step. MTP3
story/code/zh/fr 76.0/95.3/69.0/79.1 -> 78.1/95.7/71.5/80.9 tok/s, tokens per round unchanged
(2.14). `run-v100.sh` turns it on for Flash-Next. Plain `--draft-tokens 2` gained on zh/fr (+2) but
lost 6.6 on code; staged drafting keeps the third draft for the rounds that use it.

Re-measured on master `61b3e032c` (2026-11-23, 200 W, the router's flags: the `run-v100.sh` profile
plus the refusal-vector graft and `--presence-penalty 1.0`; 300 greedy tokens, warm second request, best of
five runs): story / code / Chinese / French / thinking 80.1 / 119.1 / 78.6 / 88.7 / 97.1 tok/s at
1.94 / 3.31 / 1.90 / 2.18 / 2.62 tokens per round. Conditions and ranges: [tesla-v100.md](../tesla-v100.md#qwen38-flash-next).

Then `run-v100.sh` moved to seven drafts with the reduced draft head (`--mtp-draft-vocab`, the Basalt
list fetched at install): 80.5 / 123.3 / 78.8 / 71.8 / 100.3 at 1.92 / 3.93 / 1.88 / 1.78 / 3.05 tokens per
round, the same text. The head is a `gather_columns` copy of the listed IQ4_XS rows (one BF16 view of
the row bytes) made at load; the drafter's projection and argmax run over it and
`proposal_remap_token_ids` maps the winners back. It is off on sm_86 (`auto` checks the compute
capability). Lapilli's lead on code and thinking was not its MTP weights or algorithm: its
self-distilled MTP dense, converted and run here, was slower (code 115 against 119), and the experts
are byte-identical; its UD-IQ2_XS target (IQ4_NL down projections in 26 layers) generates different
text, which the drafter predicts better.

**The draft vocabulary from the tokenizer (2026-11-25).** `--mtp-draft-vocab auto` on sm_70 now
builds the list at load with `models::qwen4_exp::tokenizer_draft_vocab` (program.cpp) from the
artifact's tokenizer: ids `[0, 65536)`, the ids in `[65536, 120000)` that `draft_vocab_cjk_bytes`
accepts (any byte E3-E9, a whole full-width or Hangul character, or continuation bytes only: a CJK
character split across tokens), and the added tokens; sorted, distinct, padded to an even count for the
BF16 view of the rows. 90,068 ids against the Basalt list's 106,056. The Basalt list keeps every id
below 32,768 and, above it, mostly CJK tokens, which is why French lost 0.4 tokens a round with it; the tokenizer's ids follow merge order, so a contiguous prefix covers every
script's common tokens, and CJK needs the extra band because its characters merge late. Router flags,
ABBA against master: code 126.6 -> 129.6 (3.90 -> 4.05 tokens a round), French 72.7 -> 77.2 (1.78 ->
1.94), the same text. `tools/flash_next/draft_vocab.py` writes the same list from `tokenizer.json`
(checked equal to the server's, id for id); `tokenizer` selects it on any card, a path reads a file.

**Staged GGML verify GEMV (2026-11-24).** Flash-Next's dense linears (GDN and attention input and
output projections, shared expert) are GGML blocks on `ggml_mmvq`, which was already a `dp4a` GEMV
on Q8_1 activations: #600's int8 idea was there, and the verify columns still cost about 1 ms a
round each. Profiled on the code request, the main stream's `ggml_mmvq` kernels took 6.3 ms of a
round at about four columns (none of it overlapped), about three times the weight stream. A harness
(L2-cold weights, chained calls) split the IQ4_XS 2560 x 6144 output projection's 36.6 us at T = 4:
without the activation cast it took 22.2 us, the cast alone 16.9. Each of the 240 persistent CTAs
casts all T x K inputs (two IEEE divisions per 32-value group), so the cast grows with T and repeats
240 times. On sm_70 `ggml_q81_stage_kernel` now casts x once into the call's workspace in the
shared-memory layout, and `ggml_mmvq_staged_kernel` (`mmvq_body`'s `Direct` form) reads it through
L1 instead of staging it; the output projections (K = 6144) run one 512-thread CTA per SM, two rows a
warp. Same cast, same per-row order: bit-identical, checked at T = 1-8 with and without the residual
for IQ4_XS, IQ3_S and Q6_K. Harness at T = 2 / 4 / 8 (us, plain -> staged): IQ4_XS output
24.8 / 36.7 / 51.9 -> 20.4 / 24.0 / 30.7, IQ3_S 28.0 / 39.7 / 55.2 -> 23.2 / 26.3 / 33.9, Q6_K
46.2 / 56.7 / 80.5 -> 36.2 / 35.7 / 44.9; the K = 2560 attention query 29.7 / 38.0 / 55.5 ->
30.0 / 35.2 / 50.8. Flash-Next decode, ABBA at 200 W against `NINFER_GGML_STAGED=0` (router flags,
story / code / Chinese / French / thinking): 81.2 / 124.6 / 79.5 / 72.2 / 100.9 -> 81.9 / 126.7 /
80.4 / 72.8 / 103.4, master's text in every run.

What did not pay, all exact: two or four rows a warp sharing each activation load (spills, up to 3x
slower); batching the cast's loads ahead of the arithmetic (slower); one CTA per SM without the stage
kernel (the output projection 38.6 -> 27.7 us at T = 4, but the K = 2560 and 640 shapes lost); the
staged route at K = 640 (the shared expert's down projection: 8.1 -> 10.2 us at T = 4, so it keeps
the plain kernel). The pair kernels (GDN qkv + z, shared-expert gate + up) keep the plain route:
`linear_pair` takes no workspace, and the stage gained them only 1-3 us.

Working with the 3.3 GB/s eGPU link (Flash-Next, measured 2026-11-17):
- In steady decode the link costs almost nothing: the expert cache hits 99.6-100% and
  `stage_kernel` is ~5 us a call. The link matters for prefill and for refilling the cache.
- A warm start restores half the cache's frames (`kSeedShare`). The first wide prefill's give-back
  then refilled every free frame, ~12 GiB, about 4 s here, and the next request waited it out (a
  19-token prompt took 4.5 s). On a link below half the 27.5 GB/s reference, the Program now issues
  that fill in the background right after the warm start (`fill_free_frames_on_slow_link`). Prompts
  of 342 / 939 tokens prefill in 1.3 / 3.7 s instead of 3.3 / 5.4 s, the first requests decode at
  94-99% hits instead of 80-87%, and the 2K bench prefills at 333 tok/s instead of 301.
- A missed expert costs ~450 us to copy (1.5 MB). Computing it on the CPU costs less while it sees
  few tokens, as in short prompts. `--prefill-cpu-all-columns N` gives the CPU every miss it can
  take in prefill calls of at most N tokens; `run-v100.sh` sets 1024. A short prompt's TTFT drops
  from ~300 to ~250 ms, while 2K prompts keep share 0.4 (share 1 there: 333 -> 326 tok/s, and
  1,852 tokens 4.95 -> 5.52 s).
- Longer calls hit a different limit: a cold expert routed more than 8 tokens used to be too wide
  for a CPU job, so it was copied (`stage_kernel`) inside its layer, on the critical path (740 ms
  of a 3.9K-token prompt). `--prefill-cpu-max-columns N` (1..64) lets the CPU take a miss of up to
  N columns as ceil(N / 8) request slots of the same expert (`cpu_plan_kernel` publishes them,
  `cpu_wait_kernel` places each slot's columns; the bits are the same in any grouping). Calls that
  give the CPU every miss (`--prefill-cpu-all-columns`) keep 8 columns, or the CPU becomes their
  bottleneck (308 tokens: +10%). With 64 the best share rises to 0.5-0.6; `run-v100.sh` sets
  `--prefill-cpu-max-columns 64 --prefill-cpu-share 0.5`. Prompt times, the same reply bit for
  bit (2026-11-18, two runs each):

  | Tokens | 8 columns, share 0.4 | 64 columns, share 0.5 |
  |---|---|---|
  | 308 | 0.85-0.87 s | 0.85 s |
  | 939 | 1.87-1.88 s | 1.87-1.88 s |
  | 1,979 | 3.77-3.78 s | 3.36 s |
  | 2,573 | 4.61-4.62 s | 4.20-4.21 s |
  | 3,641 | 5.17-5.18 s | 4.84-4.85 s |

  What remains of the 3.9K-token prompt (5.05-5.18 s): 2.55 s of expert copies (6.2 GB) overlapped
  with compute, 360 ms of in-layer staging, and an 838 ms idle start waiting on the n-gram read. That
  read is 46K random 4K reads on a drive that tops out at ~43K IOPS, and the link is PCIe gen3 x4
  (3.3 GB/s), so both are hardware limits here. Shrinking the stream's ring (1,024 frames) to keep
  more experts cached measured no better (ring 512: 5.17 s; 320: 5.04 s with 1,529 misses left
  to staging).
- A prompt's first call reads its n-gram rows from the volume before anything else could run
  (later calls' rows are prefetched during the previous call). `run_call` now uploads the other
  inputs, starts the expert stream and queues the layers before the PLE layer first, then reads
  the rows while the device and the link work (`Forward::run_front` / `run_back`, the same output).
  3.9K tokens 5.15-5.25 -> 5.07-5.08 s; 939 tokens 1.87-2.05 -> 1.83-1.84 s; 2,149 tokens 3.46-3.48
  -> 3.38 s. The read itself (46K random 4 KiB reads, 0.85 s) is the drive's limit: the YMTC
  PC300-256GB (gen4 x4, no I/O scheduler) plateaus at ~43K random reads/s at any queue depth or
  span (~115 us at depth 1).
- `--ngram-ram-mib N|auto` (off by default in the server; `run-v100.sh` sets `auto`, and
  `NINFER_NGRAM_RAM_MIB=0` turns it off there) holds a
  prefix of the volume's blocks in pageable RAM, loaded by a background thread in 8 MiB sequential
  reads (2.7 GB/s) that pause while a request reads rows. Rows hash uniformly over the blocks, so a
  held fraction f serves ~f of a cold prompt's rows. The engine fits it last
  (`fit_ngram_ram`): after the experts, the context cache's Host tier, every planned allocation
  and the reserve, with 2 GiB more left free. On this 59 GiB host `auto` holds 13.1 of 27.1 GiB
  (48% of the blocks); the context cache keeps its 4 GiB and ~6 GiB stays available with no new
  swapping. A 3.9K-token prompt reads 27K blocks instead of 46K (0.50 s instead of 0.86): 4.97-5.05
  -> 4.67-4.79 s; 344 tokens 0.90-0.92 -> 0.84 s, 939 tokens 1.82-2.00 -> 1.68-1.71 s, 3,726 tokens
  4.78-4.80 -> 4.61 s; the same reply. The request log line counts rows served from RAM.
- In-layer staging (`stage_kernel`) is link time, not latency to hide (2026-11-18, nothing kept).
  On a 3.9K-token prompt (4.6-4.7 s) 87 of 480 stage launches carry ~370 ms: cold experts routed
  more than 64 tokens, too wide for a CPU job. Pinned H2D on this link is 3.34 GB/s by DMA or by SM
  copies (4-80 CTAs) alike, and the two share it: the expert stream's DMA runs at 2.73 GB/s alone
  and 1.48 GB/s while a large stage copy runs. Tried, each the same or slower:
  - the copy on a high-priority side stream while the pass's resident jobs compute, its staged jobs
    after: the stage left the compute stream (377 -> 18 ms), but the split doubled the gate/up
    launches (+162 ms) and the stream waits grew: 4.66-4.71 against 4.63-4.70 s;
  - the same with 128 double-buffered slots, pass p + 1's copy under pass p: 4.63-4.72 against
    4.63-4.64 s;
  - the stream's ring as a FIFO in layer order (about nine layers of lookahead in the same 1024
    frames): 5.04-5.07 against 4.73-4.74 s, as the link never idles when a layer stages;
  - the next layer's stream copies held until this layer's staging is enqueued: 5.50-5.54 against
    4.65-4.66 s (too little lookahead);
  - CPU jobs wider than 64 columns: 128 -> 5.15-5.21 s, 256 -> 5.64 s (the CPU becomes the limit).
  The two-layer parity ring with 64-column CPU jobs at share 0.5 is the balance of stream lookahead
  against staging contention. Fewer bytes over the link (a higher expert-cache hit rate) is what
  would shorten it.
- The stream ring is sized to what the prompt streams (2026-11-18). A wide prompt lent the expert
  cache's frames for its workspace and a ring of 2 x 512 frames, evicting that many cached experts
  for the prompt, but a layer streams at most ~190 experts (its misses less the CPU share's cold
  set). `ring_need` counts each layer's non-resident experts before the lend, takes away the cold
  set of the widest call's share, and adds a margin for the experts the lend itself evicts (their
  mean per layer, three standard deviations and 8); a layer still needing more than its half stages
  the rest, as before. `select_stream_cold` now takes the layer's whole miss count as its capacity,
  so the cold set no longer depends on the ring's size. On a 3.9K-token prompt the lend fell from
  2,210 to 1,666 frames and streamed experts from 4,216 to 3,944, with no expert left to staging:
  3,798 tokens 4.60-4.63 -> 4.39 s, 2,573 tokens 4.02-4.05 -> 3.77-3.79 s, 2,179 tokens 3.82-3.85 ->
  3.54 s, 1,313 tokens 3.40-3.41 -> 2.90 s (the same replies). An 8,068-token prompt (two calls)
  prefills in the same time (8.65-9.16 against 9.07-9.08 s), and decode after it is 56.6 tok/s against
  50-51, because the give-back reloads fewer displaced experts. The workspace is the other half of the
  lend: its bound is 1,709 MiB at 3,935 columns (Forward 1,588, tensor-core staging 108, shared expert
  13), but the arena's measured peak is 835 MiB. Tightening it needs each op's scratch audited,
  because an arena that is too small fails the call.
- The prefill lend's workspace is the call's replayed peak (2026-11-18). `Forward::workspace_bytes`
  is a configuration-only bound for the plan: the largest block's scratch plus 462 MiB of linear
  casts and 256 MiB of slack, but a block's mixer tensors stay live through its MoE, so the arena's
  shape is a sum along the layer, not a max. `Forward::call_workspace_bytes` replays the call's
  allocations on a bump allocator with the arena's 256-byte alignment, from the Forward's own
  weights and options: the residual; per layer the PLE scope, the inject, the mix, the GDN or
  attention tensors in order, then the MoE (mix, the side stream's carve and shared-expert tensors,
  routing, dispatch, the routed outputs and the experts' workspace); the drafter's prefill cells
  (`kv_only`, so up to the KV append); the final mix. Each Op's own scratch is its capacity query
  (linear, which includes the sm_70 tensor-core staging; hyper-connection mix; GDN; QSA select and
  attention) taken at the offset where it runs. Calls of at most 16 columns (the int8 mixers and the
  fused read) keep the old bound, and the result never exceeds it. The forward real test checks
  each mode (serial, the shared-expert side stream, `--gdn-head-map`, `--moe-precast-input`, tensor
  cores on and off) at 1,495-1,500 columns: the estimate equals the arena's peak or exceeds it by at
  most 1.5 KB. `program.cpp` logs a diagnostic once if a lent call ever peaks above its estimate.
  At 3,935 columns the workspace lend fell from 1,186 to 582 frames (835 MiB peak); at 939 columns
  from 365 to 146. Prompts (HEAD | this, the same replies): 3,935 tokens 4.74-4.75 | 4.62-4.65 s;
  3,798 4.11 | 3.88-3.90 s; 2,573 3.62-3.64 | 3.40-3.41 s; 2,212 3.39 | 3.19-3.20 s; 1,414
  2.84-2.85 | 2.78-2.79 s; 939 1.73 | 1.69-1.71 s; an 8,063-token prompt (two calls) 8.93 | 8.92-8.96 s.
- Flash-Next serves its full 262,144-token context from `run-v100.sh` (ac760a09; run.sh's default
  is 32,768). The INT8 KV pool backs 496 MiB at startup and grows into expert frames as a
  conversation fills it (3.6 GiB at 256K), so the expert cache starts at 18,108 frames against
  18,330 at 8K. Decode does not slow with context (2026-11-18, one request each, 256 tokens, MTP3):
  10,670 tokens 68.8 tok/s at 29.4 ms per verification round; 38,240 72.2 at 27.3; 79,228 76.8 at
  27.0; 148,129 77.9 at 27.3; 221,567 78.8 at 28.2; decode hit rates 97.9-99.3%. QSA reads a fixed
  2,048-token budget per query, and its indexer's pass over the pooled keys (a quarter of the
  context, 128-dim) stays small. tok/s follows MTP acceptance, which follows the content (1.97-2.65
  tokens per round), not the context. A 221,567-token prompt prefills in 245 s (903 tok/s; 10,670
  tokens 811 tok/s). Peak 68 C at 215 W through the whole run.
- Wide prefill calls are 8,192 tokens on a slow link (897d725a; Infernix's are 4,096, which a fast
  link, at or above half the 27.5 GB/s reference, keeps: an RTX 3090 has not been measured at 8,192,
  so the width is picked from the link measured at startup and logged with it). A call streams nearly every
  non-resident expert whatever its width, so a 4K call paid the link for ~4,000 experts per 4,096
  tokens; at 8K the same bytes cover twice the tokens (a 38K-token prompt streamed 20,805 experts
  instead of 38,300). `kMaxCpuCallColumns` follows (`cpu_plan_kernel`'s two `short` column arrays
  are 32 KiB of static shared memory at 8,192; 16,384 would need dynamic shared memory). An 8K call
  is compute-bound: in a 16K-token prompt's profile the main stream ran kernels 12.7 of 15.9 s and
  the link was busy 5.9 s (gate/up 4.4 s, the tensor-core linears' casts and epilogues ~2.5 s,
  down 1.3 s, in-layer staging 1.3 s). The CPU share stays 0.5 (0.3 / 0.4 measured no better).
- On a slow link a prompt's give-back no longer refills the frames it lent (897d725a). The refill
  loaded every displaced expert in one burst, 2.5 GB after an 8K-token prompt (~0.75 s of the
  link), and the decode rounds that followed slowed by about that much: an 8,063-token prompt then
  128 tokens decoded at 57 tok/s with the refill, 67 with it capped to 1,086 frames and 83 without
  (the same hit rate). Over 512 tokens and the requests after it, no refill decoded faster at an
  equal or higher hit rate (decode promotes the experts it routes into the free frames), and the
  next long prompt prefilled in the same time. The rule is the slow-link one (`kSlowLinkShare` of
  the 27.5 GB/s reference); a fast link refills as before, and `lend()`'s undo always does.
  Together, ABAB against 079c2fa2 (the same replies): 8,009 tokens 8.98-9.00 -> 8.23-8.24
  s and the decode after it 72.7-72.9 -> 81.7-81.9 tok/s; 9,823 tokens 9.87-9.93 -> 9.60-9.67 s;
  10,667 12.96 -> 12.41-12.44 s; 12,958 13.47-13.49 -> 12.70-12.75 s; 38,237 41.4-41.5 -> 38.4 s
  (37.5 s with the 8K calls alone: without the refill a prompt right after another one starts with
  fewer experts cached); 365-3,798 tokens unchanged (one call either way).

Serving on a V100: `scripts/run-v100.sh <model>` is run.sh with the V100's defaults (the sm_70
server from build-v100/, the memory-lock fallback, a GPU check). For Flash-Next it keeps run.sh's
`tuned` profile, which measured best here: MTP3 decode 68.9 tok/s and a 2K-token prefill 293 tok/s
against 63 and 251 with only `--spec mtp --draft-tokens 3 --gdn-head-map`; leaving out any one of
its flags was neutral or slower (MTP joint acceptance -4 tok/s, the shared-expert stream -1.7, CPU
prefill share 0 -> 226 tok/s prefill, 0.6 -> 285). Through the launcher at its 32K context with
vision: 307 tok/s prefill, 71 tok/s MTP3 decode. For the 27B it picks MTP3 with INT8 KV.

Never profile the Flash-Next server with `ncu` kernel replay: it backs device memory up to host
RAM (~21 GB beside the 33 GB of pinned experts) and the OOM killer ends the server. Profile the
kernels in a standalone bench or a test binary instead; `ncu`'s cold caches match the model's.

### Decode: Qwen3.6-35B-A3B

The 35B-A3B (40 layers, 30 GDN and 10 attention; 256 experts, top-8, intermediate 512) decodes
plain at 6.8 ms a step on master (6.4 after the change below), and an MTP3 round (three draft steps, a T = 4 verify) took 17.5 ms on master.
nsys over 276 story rounds (2026-11-25, 200 W) put the verify's Q6 output head first: 4.09 ms on
`q6_a16_sliced_k_mma` (the table's T = 2-32 routes are `mma.sync`, emulated here), then the Q8
mixer projections on the A16 streaming GEMV (~4.1 ms, 89 calls), the routed experts' small-T path
(`d3_path_tiled` 3.4 ms, `d4_token` 2.2 ms, FP32 SIMT dequant), the Q4 draft head (3 x 0.30 ms) and
attention (~1.0 ms). The head now tries `try_launch_q6_stream` first, as the 27B's does
(`q6/shapes/n248320_k2048.cpp`); `linear_bench`, 200 W, us:

| Tokens | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| master route | 893 | 4,537 | 4,095 | 4,073 |
| streaming GEMV, A16 (`NINFER_STREAM_DP4A=0`) | 505 | 599 | 834 | 1,319 |
| streaming GEMV, `dp4a` (the `--verify-a8` default) | 482 | 497 | 512 | 554 |

Serve ABBA and KLD are in [tesla-v100.md](../tesla-v100.md#qwen36-35b-a3b): MTP3 +25-27% on every
prompt, plain +5.4%. `NINFER_SM70_Q6_35_STREAM=0` keeps master's head.

Profiling this model under nsys needs a workaround: a cooperative `grid.sync` kernel (the GDN gating
projection's split schedules) faults with `cudaErrorInvalidPc` under nsys on sm_70, master included,
so profiles were taken with those schedules disabled in a local build and that kernel's time
corrected by its op bench (14 us a call at T = 1-4).

### Prefill: Qwen3.6-35B-A3B

A 4K-token prompt took 5.5 s (765 tok/s), of which the routed experts' prefill GEMMs (Q4 gate/up,
Q5/Q6 down; `sparse_moe_prefill_kernels.cu`) were about 2.4 s: they decode into shared memory as
BF16 and run Ampere's `mma.sync`, emulated here. `sparse_moe_prefill_hmma_sm70.cu` takes the same
route-job list (one job per nonempty column tile of one expert), packed activations and outputs,
and per 64-wide K step dequantises the 64-row weight tile to FP16 in shared memory (code x group
scale, one rounding) and multiplies with WMMA m16n16k16, FP16 in, FP32 accumulation; the
activations are staged as FP16 without scaling, exact for BF16 values in FP16's normal range. Gate
and up rows of one intermediate row sit in one tile so the SwiGLU epilogue pairs them in shared
memory. Prefill 765 -> 1,160 tok/s at 4.2K, 710 -> 1,000 at 13.4K; prefill-scored PPL (PG-19, 8,192
context) 6.174 -> 6.173. `NINFER_SM70_MOE_PREFILL_HMMA=0` keeps the old kernels. The shared expert
(Q8) still runs its `mma.sync` kernels.

### Prefill

The wide prefill GEMMs run on the tensor cores through cuBLAS, by default. The cuBLAS prefill
route (`ops/linear_swiglu/q4cublas/w4_cublas_prefill.cu`, `--prefill-cublas`) is an opt-in quality
trade on the 3090 because it materialises the weight as int8 with one scale per row and quantises
the activations; Volta has no integer tensor cores, so on sm_70 the same route (`kFp16Route`)
materialises the weight as FP16 -- code times its exact group scale, one rounding -- carries the
activations as FP16 under a per-token power-of-two scale, and runs `cublasGemmEx` FP16 x FP16 ->
FP32 on HMMA. Activation preparation, the GEMM call and the workspace switch on that constant;
token tiling, the parents and destinations, and the SwiGLU / store / residual epilogues are shared
(they read FP32 where the int8 route reads int32, with a row scale of 1).

That makes it more accurate than the routes it replaces, not less:

| | relative L2 vs fp64 (op test) | perplexity, 2,292 tokens (ctx 1024) |
|---|---|---|
| A16 reference / default integer routes | 0.007-0.020 | 6.009389 |
| FP16 cuBLAS route | 0.0011-0.0020 | 6.009804 (+0.007%) |

so `kPrefillCublasDefault` (`include/ninfer/types.h`) turns it on in sm_70 builds and
`--no-prefill-cublas` turns it off; every other target keeps the int8 route opt-in. Qwen3.8-27B,
402-token prompt: 80 -> 623 tok/s (the int8 route measured 488 here, its GEMM on `dp4a`). After the
change the cuBLAS GEMM is about half of prefill time, the FP16 weight materialisation ~16% (it is
weight-sized, so a larger `--prefill-chunk` amortises it), and the GDN chunked recurrence ~9%
(emulated TF32/BF16 `mma`) -- which sm_70 no longer takes: `chunked::kPrefillMinTokens` sends every
prompt through the sequential FP32 recurrence, +3.6-4% prefill at 1k-4k tokens with identical
greedy output and perplexity 6.00650 vs 6.00980 (an idea from geoffwatts/ninfer-v100, measured
here).

**Chunk size and the narrow tail.** `--prefill-chunk` past the default 1,024 is worth 1-4% here
(Qwen3.8-27B, 2k and 4k prompts, chunks 1,024 / 2,048 / 4,096: 796-802 and 748-774 tok/s). What
cost far more was a narrow segment: the route admits a call from `kCublasPrefillMinTokens` (129,
which decode/verify widths must stay below), and anything narrower takes the integer/A16 routes,
emulated here at ~80 tok/s. Steps also split at context-cache captures (just before the closing
template tokens) and rewrite frontiers, so 1,150 tokens ran as 1,024 + 122 + 4 and paid ~1.3 s for
the 122. With the route on, `prefill_tail_floor` (planning/startup.h, set to the gate) makes
`advance_prefill` shorten a step so the segment before the next frontier is at least the gate:
1,017 + 129 + 4, 449 -> 755 tok/s on that prompt, same output.

**Short prompts.** The wide gate exists for decode and verify rounds (up to 8 lanes x 16 columns =
128 through the same Ops and policies), not for prefill. A prefill-phase call therefore takes
`ops::prefill_phase_policy(policy)`: `AllowPrefillCublas` becomes `AllowPrefillCublasNarrow`,
whose gate is `kCublasPrefillNarrowMinTokens` -- 65 on sm_70, the wide gate elsewhere. Below it
the streaming GEMV takes 1..64 tokens of the Q4/Q5 weights on sm_70 (1..32 for formats without the
tile GEMM; `kStreamGemvMaxTokens`). Past eight it runs whole passes of 32 tokens and a 16 or 9..16
pass on the tensor cores (`ops/linear/common/stream_tile_gemm.cuh`): a CTA dequantises a tile of
weight rows to FP16 in shared memory once per 64-wide K step -- the cuBLAS route's `q x scale`
rounding, ~8.6e-4 relative L2 against the exact streaming GEMV -- and WMMA multiplies it against
every token of the pass; for matrices of few rows four warp groups split K inside the CTA and add
in a fixed order. At most eight tokens remain for one pass of eight. Against passes of eight, per
Qwen3.8-27B matrix: 1.25-2x at 12 tokens, 1.5-2.4x at 16, 2.2-3.6x at 32. `ninfer_bench`:
pp16 138 -> 236, pp32 143 -> 356, pp48 196 (cuBLAS) -> 306, pp64 262 (cuBLAS) -> 371 tok/s; the
cuBLAS route still wins from ~100 tokens (pp96 320, pp128 416). Server time to first token on a
20-token message 175 -> 160 ms, 44-50-token messages 211-246 ms. The tiles reach only 25-35% of
the weight stream (shared memory issue and barriers bind, not DRAM: deeper prefetch, padding and
occupancy were tried without gain), and the server splits short prompts at their checkpoint
frontiers (12 + 7 + 1 tokens for a 20-token chat message), each call one read of the weights.
Matrices of 16,384 or more rows (the SwiGLU gate/up) take a second form of the same GEMM that
decodes the weights straight into `mma.m8n8k4` A-fragment registers (each lane owns two weight
rows, lane pairs read whole code sectors, the activation is staged per K group as the B operand):
SwiGLU at 16 tokens 328 -> 260 us, at 32 464 -> 413 us; bit-identical to the WMMA tiles on the
measured shapes. Shorter matrices keep the WMMA tiles, which split K across warp groups and win
there (both kernels are in `stream_tile_gemm.cuh`).
`NINFER_STREAM_TILE_GEMM=0` turns the tiles off (passes of eight) for A/B measurement.

**Merged checkpoint splits.** With prefix reuse on, a chat prompt's checkpoint (its recovery
frontier, before the assistant opener) and the opener's rewrite execution frontiers cut the prompt
into separate prefill calls, and on sm_70 every call costs a full read of the weights whatever its
width (~70 ms for the 27B at 9-12 tokens): a 20-token message ran as 11 + 5 + 4 tokens. On sm_70 the
prompt's last call now runs through its pending capture's frontier
(`ProgramImpl::merge_capture_split`, `transactions/capture.cpp`): each GDN layer runs its
convolution and recurrence over the columns before the frontier into the capture's reserved state
slot and on from it (`TextContext::set_prefill_state_split`) -- the stateful steps the split calls
ran -- while the projections and attention run once over the whole call; the hidden state of the
frontier's last column goes to that slot's continuation hidden. After the first token is committed
the capture publishes the written slot as the checkpoint (`StateImageStore::freeze_written_destination`)
instead of freezing the lane's state and forking. It applies to text prompts whose last call holds
the request's last capture group with a Device destination reserved (not lossy, sliced or DFlash);
anything else splits as before, and `NINFER_MERGE_CAPTURE_SPLIT=0` turns it off. Server, MTP3:
20-token message TTFT ~158 -> ~104 ms, 44 / 50 tokens 215 / 244 -> 179 / 198 ms, follow-up turns
prefill in ~105 ms; greedy replies, cache hits and shared-prefix reuse unchanged. Before the
streaming GEMV, wherever the narrow route's scratch did not fit (the bench, the server), 9..32
tokens ran the emulated small-T mma kernels: `ninfer_bench -p 32` 93 -> 139 tok/s. (A narrow
cuBLAS call needs its scratch planned: a bench whose largest prompt is under 129 tokens plans none,
so 65..128 there fall to the emulated MMA schedules -- 55 tok/s at 65 -- where the server, whose
chunk reaches the wide gate, takes cuBLAS.) The text layers pass the phase to `ffn`, `project_add`,
`attention_projection` and `gdn_projection` (a defaulted `prefill` flag); decode, verify, MTP and
draft calls are unchanged. A narrow call takes the route only if its scratch fits in the
workspace left (`w4_cublas_prefill_fits` / `w4_cublas_projection_fits`), which the chunk-width
plan covers whenever the chunk reaches the wide gate. Qwen3.8-27B: 54-token prompt 49 -> 140
tok/s, 125-token prompt 86 -> 268 tok/s, same outputs; decode unchanged.

**Prompt attention.** The INT8 prompt kernel (`causal_attention_prompt_i8_fa2`) runs its `mma` on
the emulation, and after the GEMMs moved it was ~30% of a 4k-token prefill. On sm_70 a
single-sequence INT8 G64 call of 17 or more columns (and `causal_softmax_attention_cached` over
the same) instead takes llama.cpp's FP16 tensor-core flash attention
(`third_party/llama_cpp_fattn`, MIT, pinned at 62bf73d25), driven by
`ops/launcher/gqa_attention_volta_flash.cu` -- both adapted from geoffwatts/ninfer-v100
(Apache-2.0). The launcher appends the call's K/V to the paged cache, dequantizes the visible keys
into contiguous FP16 once per layer, and runs Q in blocks of `kVoltaFlashQBlockTokens` under a
causal mask. Qwen3.8-27B: 1,150 / 2,186 / 4,392-token prompts 783 -> 868, 834 -> 959, 778 ->
1,020 tok/s, same greedy output; perplexity 6.00650 -> 6.00707.

The route starts at 17 columns, one past the widest verify width (`kMaximumVerifyTokens`, so the
graphed verify and draft widths never take it); at first it started at 64, which left short
prompts over a long history on the emulated INT8 kernel. That kernel's cost grows with the history
and the flash route's barely does (`ninfer_causal_softmax_attention_bench`, append, 24 heads, one
layer, W = 17..63): no history 203-220 -> 80-111 us, 1K keys 2.05-2.25 ms -> 132-179 us, 4K
7.5 ms -> 231-366 us, 16K 29.5 ms -> 0.65-1.16 ms, 37K 66.3-66.7 ms -> 1.3-2.5 ms. A 24-35-token
follow-up question to a 37K-token conversation (Qwen3.8-27B, MTP3, INT8 KV) went from a TTFT of
1.15-1.28 s to 0.11-0.24 s; greedy answers matched in four of five follow-ups (the fifth diverged
at its third token, a near tie: -1.38 against -1.47), and the matched tokens' logprobs moved by at
most 0.068.

Two costs come with it, both measured in the op suite. The kernel accumulates P x V in FP16 (as
llama.cpp does on every tensor-core GPU), so its error grows with the key count: relative L2
3.0e-3 up to 5K keys, 4.0e-3 at 12K, 7.0e-3 at 33K, against 1.75e-3 for the INT8 kernel
(`kAttentionVoltaFlashCriterion` in `tests/ops/softmax_attention/causal_cache.cpp` holds it under
the FP8 KV profile's 1.2e-2). And its scratch holds the visible keys in FP16 plus the mask, about
6 KB per key (roughly 50 MB at 8K, 200 MB at 32K, 800 MB at 128K), which the workspace query
reports, so it comes out of the memory planned for the KV cache. Chunking the keys with an FP32
merge of the partial results would remove both; the kernel would need to export its row maxima
and sums.

Thermals: sustained prefill holds the card at its 200 W cap and took a passively cooled V100 from
40 to 75 C in about a minute; measure in short runs without forced airflow.

#### Flash-Next: the tensor-core route (`NINFER_SM70_TENSOR_CORES=1`)

Qwen3.8-Flash-Next's prefill linears are BF16 (hyper-connection, PLE, router, indexer, MTP
drafter) and GGML (attention and GDN projections, shared expert), and on sm_70 both run emulated
Ampere MMA (BF16 `mma`, int8 `mma` / `dp4a` over a Q8_1 activation cast). With
`NINFER_SM70_TENSOR_CORES=1` in the server's environment, every such linear of at least 64 tokens
on a weight of at most 33M elements goes through `ops/linear/common/sm70_tensor_core.h` instead:
the weight row converted to FP16 under its own power of two (exact for BF16; one FP16 rounding of
code x scale for GGML), the token's BF16 activations converted exactly under theirs, cuBLAS
`cublasGemmEx` FP16 x FP16 -> FP32 on HMMA, and both powers of two undone before the BF16 output
rounding (plus the residual for LinearAdd). Like the dense route above it drops the integer
route's activation cast, so it is more accurate per op, not less (relative L2 against FP64 over the
BF16 activations, `ninfer_linear_ggml_test` with the variable set, which then holds the wide calls
to that oracle):

| | GGML linears (every format and shape, T=300) | BF16 linear (HC down shape) |
|---|---|---|
| default route | 3.8e-3 from the Q8_1 cast alone, before its BF16 rounding | 1.66e-3 |
| tensor-core route | 1.67e-3 (the BF16 output rounding) | 1.66e-3 |

It is not bit-identical (FP32 summation order, and no cast), so the server keeps it off unless the
variable is set; `scripts/run-v100.sh` sets it for Flash-Next (`NINFER_SM70_TENSOR_CORES=0` there
turns it off). Measured on
the 3,935-token prompt: 6.62 -> 6.16 s (594 -> 639 tok/s, ABAB); the compute stream's kernel time
drops ~490 ms, and the rest is absorbed by waits on the expert stream (the 3.3 GB/s link) and the
0.82 s n-gram volume read the first call makes before the device starts. Model-level checks (no
perplexity tool serves Flash-Next yet): a long-context retrieval set (24 prompts of 2.4-4.3K tokens
with one planted fact each) scores 24/24 both ways; greedy 128-token replies to 12 document prompts
diverge after a median ~300 characters into equivalent text, as any reordering does. Decode is
untouched (its calls are narrower than 64 tokens) except that the MTP drafter's catch-up over a
prompt of 64+ tokens takes the route, so its drafts (never the accepted tokens) can differ. 
The same variable moves prompt attention onto the tensor cores: the QSA wide route (one CTA per
column and KV head, the unsplit prefill form) runs `ops/qsa/attention_wide_hmma.cu` instead of
`attention_wide.cu`, written in native `mma.sync.m8n8k4` fragments where the original's m16n8k16 is
emulated on sm_70. QK takes the INT8 key codes widened exactly with the per-group scales in FP32 as
before; PV folds the value scale into the probability (one FP16 rounding of `p * vscale` per tile,
under a per-tile power of two). `ninfer_qsa_test` with the variable set: relative L2 against FP64
1.660-1.680e-3 on the wide cases against 1.676-1.699e-3 for the original (the decode and split
cases do not take it). The 12 attention calls of the 3,935-token prompt: 715 -> 172 ms; the prompt
6.40 -> 5.97 s (6.90 s with the variable unset).

The route's side passes are bandwidth-bound passes over the activations and the FP32 product.
`tokens_to_f16` reads 16 bytes at a time and keeps up to 64 values per thread in registers between
the absmax and the conversion. `finish` runs a 2D grid (four rows of one token per thread, `float4`
loads), where it used to do a 64-bit divide and modulo per element. Both do the same arithmetic as
before, so outputs are bit-identical (Flash-Next greedy text and vision outputs unchanged). Over four
5,230-token prompts (nsys), they take 802 -> 696 ms and 859 -> 749 ms, and the prompt time goes
5,131 -> 5,061 ms (-1.4%). Removing the passes would take an epilogue fused into the GEMM, which
cuBLAS does not offer for FP16 in and BF16 out with per-row and per-token scales.

The BF16 sliced-K kernel (`bf16_a16_sliced_k_mma.cuh`) takes the same sm_70 FFMA branch as the
tile kernels (`bf16_mma_common.cuh`): each lane reads its own fragment rows and tokens and runs the
emulated m16n8k16's FMAs in its order, so it too is bit-identical. Flash-Next's row-tail
instance goes 392 -> 254 us per call, the others 47 -> 41 and 74 -> 52 us; this is a few ms per
request.

Every staging buffer, cuBLAS's workspace included (set per call), comes from the call's
`WorkspaceArena`, so the shared expert's side stream and the device stream never share scratch.
`linear_workspace_capacity_bytes` (BF16) and the GGML capacity queries include
`sm70_tensor_core_workspace_bytes` while the route is on; Flash-Next adds
`Forward::tensor_core_workspace_bytes` (a bound over its widest linears, ~110 MB at 512+ columns)
to a wide call's lent arena, which comes out of expert frames only for that call, and to the shared
expert's carve. The static workspace is not grown, so decode keeps its expert frames: a static call
(a prompt or tail under 512 tokens) takes the route when its arena has room and otherwise keeps
the default kernels, which the server reports once on stderr.

### Vision attention

The vision tower's attention (27 blocks, 16 heads of size 72, full attention within each image)
ran `packed_attention_flash_kernel`, BF16 `mma.sync` and `ldmatrix` on the emulation: 0.51 s of a
1-megapixel image and 7.6 s of a 4-megapixel one (283 ms a block), growing with the square of the
patch count. On sm_70 the equal-length form of `packed_softmax_attention` with a workspace (the one
`vision.cpp` calls) now runs the vendored llama.cpp FP16 flash kernel instead
(`ops/launcher/packed_attention_volta_flash.cu`):

- head size 72 is zero-padded to 80, the nearest instantiated size (exact);
- Q, K and V are staged unscaled (BF16 converts to FP16 exactly) and the 1/sqrt(72) scale is
  applied to the FP32 Q.K sums. The kernel's own `scale` rounds Q x scale to FP16, which moved
  outputs of sharp attention rows by up to 3%. A small ninfer patch in the vendored kernel makes a
  nonzero `logit_softcap` without `use_logit_softcap` an FP32 KQ scale; the text route passes 0;
- P x V accumulates in FP16 on Volta, so a segment longer than 1,024 keys is split along the keys
  across CTAs and merged in FP32 by the stream-K fixup: over a 16K-patch image the relative L2
  error is 1.9e-3 (1.8e-3 for the BF16 kernel, 4.3e-3 unsplit);
- a Q, K or V value past FP16's range sets a device flag, and the BF16 kernel then reruns the call
  (it is always launched and returns at once unless the flag is set);
- the staging (FP32 Q and output, FP16 K and V, the split partials) goes through heads in passes of
  at most 64 MiB and comes from the QKV projection's scratch, which the vision workspace plan sizes
  for it.

`ninfer_packed_softmax_attention_bench`, one segment (BF16 kernel -> FP16 flash): 1,024 patches
1.55 -> 0.36 ms, 4,096 patches 19.5 -> 4.3 ms, 16,384 patches 283 -> 58 ms. Prompt time of an image
request (encode and prefill, thinking off):

| | 1 MP (1,051 tokens) | 4 MP (4,123 tokens) |
|---|---|---|
| Qwen3.8-27B | 2.40 -> 2.01 s | 14.46 -> 8.40 s |
| Flash-Next | 3.42 -> 3.14 s | 12.79 -> 6.67 s |

#### Vision linears on the tensor cores (Qwen3.8-27B)

The 27B stores its vision blocks' linears as group-quantized Q4/Q5/Q6 (qkv 3456x1152, fc1
4304x1152, attention out 1152x1152, fc2 1152x4304), and those calls of hundreds to thousands of
patches ran the emulated BF16 `mma.sync` tile GEMMs. With `NINFER_SM70_TENSOR_CORES=1`, an A16 call
of 64+ tokens on a Q4/Q5/Q6/Q8 weight of at most 8M elements decodes the weight once per call to
per-row power-of-two FP16 (`ops/linear/common/rowsplit_tensor_core_sm70.cu`) and takes the same
cuBLAS FP16 route as BF16 (FP32 accumulation, per-token power-of-two activations). Products stay
exact, so only the summation order changes. The size bound keeps the text model's linears (all
larger) and the vision mergers on their kernels: the mergers' staging would push the 27B's
`--vision-residency overlay` encode window past what it can evict. The route admits K % 8 (it was
K % 32), which lets fc2's K = 4304 in. On Flash-Next that moves the BF16 vision fc2 onto the route,
the only change there.

`cublasCreate` cannot run inside a stream capture. If the route has not yet created its handle when
a capture reaches it, it declines (`sm70_tensor_core_can_launch`), and the call keeps the default
kernels.

Image prompt time (encode and prefill), measured on top of the FP16 vision attention route (#517):

| | 1 MP | 4 MP |
|---|---|---|
| Qwen3.8-27B | 2.01 -> 1.36 s | 8.40 -> 5.87 s |
| Flash-Next (fc2 only) | 3.17 -> 3.02 s | 6.67 -> 6.14 s |

Flash-Next's vision tower against its FP64 oracle (`ninfer_qwen4_exp_vision_real_test`) has
normwise error 2.4-7.1e-2 across six images, against 2.7-7.6e-2 before. It is lower on five of the
six, all at the test's BF16-storage estimate. `ninfer_linear_q{4,5,6,8}_a16_tensor_cores_test`
reruns the Q format conformance tests with the variable set.

### The routed experts' wide gate/up

`wide_gate_up_kernel` (the wide route, one CTA per 32 intermediates and job) stays on dp4a: each
32-value sub-block's integer sum is exact, so it is bit-identical to the CPU engine. Measured on
`ninfer_offloaded_moe_wide_test --bench` with `clock64` per warp and phase, a full 64-column tile
spends ~51% of a warp's time in the dp4a-and-scaling loop (at ~80% of the FMA pipe's peak with both
of a scheduler's warps busy), the rest in the column fetch, fold and decode; the IQ2 decode alone is
~7% (stubbed out: 1,287 -> 1,201 ms). With the dp4a work free (stubbed), the kernel drops only 22%,
so FP16 tensor cores could not gain more than that. The loss was in narrow tiles: a warp's math is
latency-bound, so a 16-column tile cost ~77% of a full one (its two busy warps ran a full share
while six waited at each leaf's barrier), and half the bench's tiles have 32 columns or fewer. The
kernel now picks its warp layout from the tile's width (`kLayout`: 2x16, 2x8 or 1x8 row-by-column
tiles per warp), as three kernels, because one kernel holding all three bodies spills. The bench's
gate/up time is 12-17% lower (IQ2_S, Flash-Next's format, 1,703 -> 1,421 ms). The 3,935-token
prompt's gate/up time drops 1,468 -> 1,387 ms, and its tiles are mostly full. Prompt time is
unchanged at 3.6-3.9K tokens: the compute stream waits on the expert uploads (4.3 s of copies over
the 3.3 GB/s link). Prompt time improves ~6% at ~940 tokens (2.02 -> 1.87 s, two runs each).

### Why the shared kernels have no HMMA path

The PTX `mma.m8n8k4` computes four independent 8x8x4 products per warp, one per octet (lanes
4o..4o+3 and 16+4o..16+4o+3). Expressing a m16n8k16 with it needs the operands in the octet
layout. With the m16n8k16 fragment layout the shared kernels hold, that costs about 24 shuffles
per mma (so it is shuffle-bound at roughly the same rate as the emulation), or a zero-shuffle
scheme that uses a quarter of the tensor work. So the kernels that matter get Volta-only versions
written in native `m8n8k4` fragments instead: the prefill GEMMs and Q4 linears above, prompt
attention (`*_volta_flash.cu`), the wide QSA attention (`qsa/attention_wide_hmma.cu`) and the
27B's small-T decode and verify attention (`causal_cache/small_t_i8_hmma.cuh`, "Decode").

## Compile-time facts (CUDA 12.8.1, `-arch=sm_70`)

- CUDA 12.8 is the newest toolkit that compiles for Volta (13.0 dropped it); nvcc warns that
  offline compilation for sm_70 is deprecated.
- A full `ninfer_ops` build is 419 translation units and about 15-20 minutes on 8 cores. One nvcc
  job peaks near 2 GiB; do not build with more jobs than RAM / 2 GiB.
- The container the build was checked in cannot run the result (no GPU). `ptxas` accepted every
  kernel, so register caps, shared-memory limits and instruction support are all checked; what is
  not checked is anything that needs execution.
- Register counts differ from the `sm_86` build, because the emulated MMA needs more of them. A
  residency figure measured with `cuobjdump -res-usage` on `sm_86` does not hold here, and a
  cooperative grid sized from it fails to launch. The Qwen3.6-35B-A3B GDN gating projection's
  split-8/4/2 kernels are 74 registers on `sm_86` (three CTAs per SM) and 84-87 on `sm_70` (two),
  so every prompt of 2,700-3,900 tokens, or with a chunk of that size, stopped the server with
  `cudaErrorCooperativeLaunchTooLarge` (2026-11-25). The launch now also asks
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and splits the token range to the smaller of the
  two figures; `ninfer_gdn_gating_proj_test` covers the widths at each split's top.

## What is verified

- `ninfer_ops`, `ninfer` and `ninfer-serve` build and link for `CMAKE_CUDA_ARCHITECTURES=70`.
- The new S8-k16 and S4-k64 emulations were run on the host against naive references (32 threads
  standing in for the lanes of a warp, running the real header text); see
  `tests/ops/test_volta_emulation.cu` for the on-device version, which has not been run.
- The earlier emulations (`ldmatrix`, BF16/FP16/TF32 m16n8k16, S8 k32) passed on a real V100 when
  the port was first written (the PR #300 checkpoint).

Verified on a real V100-PCIE-32GB (2026-11-16): `tests/ops/test_volta_emulation.cu` passes every
emulated instruction; the op suites for the linears, input projections, `linear_add` and SwiGLU
pass; Qwen3.8-27B generates correct text (decode numbers under "Decode" above).

## Validating on a V100

`scripts/v100_validate.sh` is the first thing to run on a rented card (the box needs CUDA 12.8
devel, 48 GiB or more of RAM and 100 GB of disk, for example `nvidia/cuda:12.8.1-devel-ubuntu24.04`):

```bash
git clone -b v100-port https://github.com/ashalliants/ninfer-3090 && cd ninfer-3090
bash scripts/v100_validate.sh --tests                      # emulation test, build, op suites
MODEL_27B=/workspace/models/qwen3_8_27b.ninfer bash scripts/v100_validate.sh --models
```

It stops at the first failing stage. The emulation test runs first (seconds); if an emulated
instruction is wrong, nothing after it means anything. Then read the generated text per model:
sensible text means the kernels are right, and the log carries decode and prefill speed.

Models the port is meant to run: Qwen3.8-27B, Ternary Bonsai 2 27B, Qwen3.6-35B-A3B and
Qwen3.8-Flash-Next. A 32 GB V100 holds the 27B (19 GiB) with room for KV; the Bonsai 27B is 8.9 GiB.

**Flat-cost verify on int8 activations (`--verify-a8`, on by default in an `sm_70` build).** The
SIMT streaming GEMV pays one FFMA per weight per token and the tensor-core one above only covers
some Q4 shapes, so a round's GEMV time still grew with its width: an MTP3 round cost ~43 ms and a
DFlash2 K = 7 round ~80 ms against 23.3 ms for a plain step. `ops/linear/common/stream_dp4a.cuh`
serves every Q4 / Q5 / Q6 streaming call of 1..8 tokens instead (the input projections, `linear_add`,
SwiGLU, plain linears and the Q6 LM head):

- each CTA quantises its staged activation to int8, one FP32 scale per 32 values and token (absmax /
  127, llama.cpp's q8_1), and stores the slice's sum beside it;
- the format turns a packed word of eight weights into two words of four biased unsigned bytes
  (`dp4a_word`: 3 instructions for Q4, more for Q5's high plane and Q6's pairs), and each token
  then costs one `dp4a` per four weights. The bias times the slice sum seeds each chain, so the
  integer dot product is exact; the group scale and the activation scale apply in FP32 per slice;
- about 0.4 + 0.45 T instructions per weight instead of 1.5 + 1.25 T, so the kernel stays at the
  weight stream through T = 8; past 8 tokens the tile GEMM serves as before;
- every token's sum is formed in the same order at any width: the K chunk is a function of K only,
  lanes own fixed slices, the warp reduces a chunk, and chunks add in order. A token's result is
  bit-for-bit the same at T = 1 as in an eight-token verify. End to end that does not make
  speculative output equal plain decoding's: verify attention and the GDN recurrence still differ
  by width, as before. On six greedy prompts (300-1,200 tokens) MTP3 and plain decoding gave the
  same text on 0 of 6 (1 with DFlash2) against 2 of 6 on master, and parted after 56-1,522
  characters; MTP3 and DFlash2 gave the same text on 1 of 6 (master: 1). `--no-verify-a8` gave
  master's MTP3 text on all six.

Microbenchmark, cold L2, us per call at T = 1 / 4 / 8 (`ninfer_linear_bench --sweep 1:8:1`, 200 W):

| Shape | SIMT (before) | int8 `dp4a` |
|---|---|---|
| Q4 34816 x 5120 (gate/up) | 122 / 172 / 305 | 121 / 131 / 144 |
| Q4 6144 x 5120 | 29 / 47 / 83 | 29 / 34 / 44 |
| Q4 5120 x 17408 (down) | - / 108 / 185 | - / 86 / 112 |
| Q5 5120 x 6144 | - / 61 / 95 | - / 39 / 47 |
| Q6 248320 x 5120 (head) | 1,228 / 2,086 / 3,772 | 1,195 / 1,260 / 1,310 |

(- : the bench's T = 1 call of that shape does not take the streaming route.)

Accuracy, the decode-path KLD gate (`ninfer-perplexity --kld-write` / `--kld-base`, 17 chunks of 512
tokens from four held-out domains, INT8 KV, FP16 GDN state; base = the FP32 kernels at T = 1):

| Run | KLD mean | p99 | same top-1 | PPL |
|---|---:|---:|---:|---:|
| FP32 kernels, T = 1 (base against itself) | 0 | 4e-5 | 100% | 3.6102 |
| FP32 kernels, T = 8 (master's own verify reordering) | 3.8e-4 | 3.0e-3 | 99.42% | 3.6116 |
| int8, T = 1 | 7.2e-4 | 6.1e-3 | 98.87% | 3.6134 |
| int8, T = 4 | 7.2e-4 | 6.2e-3 | 98.89% | 3.6156 |
| int8, T = 8 | 7.8e-4 | 5.8e-3 | 99.19% | 3.6106 |

That is about twice the FP32 reordering floor and a fifth of the Inkling FP16 tensor-core prefill
(about 4e-3), so it is the default. `--no-verify-a8` (or `NINFER_STREAM_DP4A=0`) restores the FP32
kernels; `NINFER_STREAM_DP4A=1` forces the route for op tests and A/B runs. The choice is per CUDA
stream (`ops/linear/common/stream_a8.h`): a Qwen3.5-family model instance registers its ranks'
streams with its own `verify_a8` for its lifetime (`runtime/engine/model_instance.cpp`), so two
engines in one server keep their own settings, and a launch on any unregistered stream (other
models, the op suites) keeps the A16 kernels. The op suites' `_stream_a8` variants run with the
route forced, at a widened allowance (`with_stream_a8_allowance` in `tests/ops/op_tester.h`).

An FP16 tensor-core form (`mma.m8n8k4`, exact FP16 weight integers, the activation scaled per token
into [2^14, 2^15), FP32 accumulation per 64-weight group) was built first and measured slower than
the int8 form at every width up to 8 (Q4 34816 x 5120: 152 / 179 / 182 us; 6144 x 5120: 34 / 54 /
57 us). Turning the codes into FP16 pairs and applying the group scales costs about two
instructions per weight however few tokens share them, against `dp4a`'s 0.4 + 0.45 T, and at one
16-warp CTA per SM the kernel was latency-bound (25% occupancy, long-scoreboard stalls, 53-68%
of DRAM bandwidth in `ncu`). 32-warp CTAs, two CTAs per SM and a three-deep load ring all
measured slower. Lapilli's own FP16 variant of its `dense_mmvq` beats its `dp4a` form only from five
columns on NVFP4 weights, and not on its 8-bit ones until 15 columns. It was not kept.

**The DFlash2 drafter on int8 activations (`NINFER_DRAFT_A8`, on by default in an `sm_70` build).**
With the verify GEMVs flat, the 27B's DFlash2 K = 7 round spent ~14 ms of its ~80 ms (master
`63762a40`) in the block drafter itself, which runs every linear at the block width of eight
columns: per layer (x5) the Q8 SwiGLU gate/up on `q8_a16_sliced_k_mma`, an `mma.sync` kernel written
for `sm_80` and emulated on Volta (1.43 ms, about 6x its weight stream), and the Q8 attention input,
output and down projections on the SIMT streaming GEMV (one FFMA per weight per token); then the Q4
proposal head's top-k on `q4_ksplit_mma`, also emulated (3.2 ms). The target verifies every draft,
so the drafter's numbers decide which tokens are proposed, not which are emitted, and may trade
exactness for speed where the target's linears may not.

`propose_dflash2_batch` holds an `ops::ScopedDraftA8` (`ops/linear/common/draft_a8.h`, a
thread-local flag read at launch time, so captured graphs keep it) and the four drafter Ops route
to `stream_dp4a.cuh`'s kernel inside it (`ops/linear/common/stream_draft_dp4a.cuh`): the Q8 attention
input (`q8_dflash2_attn_input.cu`), the Q8 SwiGLU pairs (`q8_dflash2_linear_swiglu.cu`, gate and up
rows as one item), the Q8 conv-add projections (`q8_dynamic_grouped_conv_add_materialized.cu`) and
the Q4 proposal head (`linear_topk/q4.cu`, writing the K-split producers' keys so the same merge
reduces them). The Q8 codes are signed int8 already, so `Q8StreamFormat::dp4a_word` is two byte
permutes with no bias. Q8's form is marked `kDp4aDrafterOnly`: the verify route (`--verify-a8`) does
not take it, so the MTP layer, the vision tower, Inkling-2-Mini and every other Q8 linear keep their
kernels. `NINFER_DRAFT_A8=0` turns the route off, and the drafter then runs exactly master's kernels.
Outside `sm_70` builds the kernel is not compiled and the scope changes nothing.

Microbenchmark, cold L2, us per call (`ninfer_benches` with `NINFER_DRAFT_A8=0` / default, 200 W).
The FP16 tensor-core column is a standalone `mma.m8n8k4` GEMV over the same row-split codes (FP16
weights times the group scale through one HFMA2 per pair, FP32 accumulation, eight tokens, best of
four launch shapes), without the SwiGLU pairing or the top-k epilogue, so it is a lower bound for
that form:

| Drafter Op (shape, width) | before | int8 `dp4a` | FP16 `mma.m8n8k4` |
|---|---:|---:|---:|
| SwiGLU gate/up, Q8 34816 x 5120, T = 8 | 1,440 | **247** | 329 |
| SwiGLU gate/up, T = 4 | 1,513 | **234** | 329 |
| attention input, Q8 6144 x 5120, T = 8 | 105 | **60** | 98 |
| output projection + conv add, Q8 5120 x 4096, W = 8 | 71 | **44** | 58 |
| down projection + conv add, Q8 5120 x 17408, W = 8 | 230 | **156** | 209 |
| proposal head top-k, Q4 131072 x 5120, 7 columns | 3,338 | **598** | 1,678 |

`dp4a` wins every shape. For Q8 it needs no weight decode at all; the tensor-core form still pays
the codes' conversion to FP16 (one permute and one HFMA2 per two weights) and, at eight columns, is
latency-bound rather than at the weight stream (329 us is 576 GB/s against `dp4a`'s 769). Per
round that is about 9.4 ms less drafter time at K = 7 (5 x 1.34 ms in the layers, 2.7 ms in the
head). Two-lane rounds (16 columns) run the route as two passes and still gain: at T = 16 the
SwiGLU takes 489 against 1,422 us, the attention input 119 against 214, the down and output
projections 306 / 84 against 466 / 140, and the 14-column head 1,174 against 5,942. Wider calls
(the top-k past 16 columns, the attention input past 16 tokens) keep their old kernels, measured
equal. The code request below confirms the per-round saving end to end (the same 371 drafts in ~53 rounds, 0.48 s
less, 9.1 ms a round).

Serve legs, ABBA in one session (`NINFER_DRAFT_A8=0` / on / on / `=0`), the router's flags, INT8 KV,
`--ngram-draft-tokens 15`, 200 W, no thermal throttling, tok/s:

| Prompt | DFlash2 K7 before | DFlash2 K7 | K7 adaptive 3 before | K7 adaptive 3 |
|---|---:|---:|---:|---:|
| code | 110.5-111.3 | **135.5-135.7** | 110.4-110.6 | 135.4-135.6 |
| think | 102.3-103.0 | **125.0-125.2** | 102.3-102.4 | 125.1-125.2 |
| story | 52.5-52.8 | 60.6-60.7 | 55.1-56.5 | 60.7 |
| Chinese | 32.4-32.6 | 39.7 | 39.7 | 51.2 |
| French | 55.6-56.0 | 68.3 | 56.4 | 72.9-73.0 |
| conversation (T 0.6) | 75.0-75.3 | 85.7 | 64.2-73.6 | 81.3-81.4 |
| structured JSON | 146.3-146.6 | **176.9-177.0** | 146.3-146.4 | 176.6 |
| prose (T 0.6) | 56.5 | 69.5 | 59.3-60.0 | 77.3-77.4 |
| conversation, greedy | 71.7-71.8 | 90.4 | 75.2-76.1 | 91.3 |

Acceptance is unchanged on most prompts: code 246/371, Chinese 118/1,248, French 194/726, thinking
242/399 and structured 6,318/6,392 drafts accepted at K = 7 with and without the route, and the
greedy text of those five prompts is byte-identical. Story (187/775 before, 180/816 after) and the
greedy conversation part from the text without the route, after 127 and 2,408 characters ("a man
who had spent" against "the lighthouse keeper, who had spent"). That is the target's verify
arithmetic, not the drafter's: a token's logits depend on how many columns its round carries
(attention and the GDN recurrence by width, as above), so a different accept pattern can flip a
near-tie. The drafter route is not needed to see it: without it, K = 7 and K = 7 with
`--adaptive-draft-tokens 3` (the same kernels, a different window policy) give the same text on 2
of 6 greedy prompts, and plain decoding matches K = 7 on 1 of 6. With the window policy fixed, the
route keeps the text on 4 of 6 at K = 7 and 2 of 6 adaptive (code and thinking identical in every
mode). MTP3 does not use the route and is unchanged (the same text on all nine prompts, the same
speed within 0.1 tok/s).

K = 8 does not pay on this card: nine verify columns pass the int8 verify GEMV's eight, and one
leg with the route ran 33-116 tok/s (code 77.4, structured 115.7); with `--adaptive-draft-tokens 3`
it ran code 113.0, structured 117.8, story 70.8, Chinese 53.7, French 74.8, prose 76.2. Per prompt
the fastest setting measured is DFlash2 K = 7 for code (135.7), thinking (125.2) and structured
output (177.0), K = 7 adaptive for the long greedy conversation (91.3), and MTP3 for story,
Chinese, French and sampled prose and conversation (76.9-91.5).

**Verifies past eight columns** (2026-11-25). A DFlash2 round of 9-32 columns (two lanes at K = 7,
or K = 8-15) ran the target's Q6 output head, 248,320 x 5,120, on `q6_small_t_mma` (Ampere
`mma.sync`, emulated): 9.86-9.98 ms at 9-16 tokens and 16.5-16.9 ms at 17-32, against 1.31 ms for
the eight-token `dp4a` GEMV. `try_launch_q6_stream` now takes up to 32 tokens on the int8-activation
verify route, as `dp4a` passes of eight over the same weights (`linear_bench`, cold, 200 W):

| Tokens | 9 | 12 | 16 | 17 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| before, us | 9,862 | 9,814-9,865 | 9,869-9,982 | 16,504-16,582 | 16,518-16,523 | 16,916-16,919 |
| passes of eight, us | 2,494-2,495 | 2,573-2,575 | 2,629-2,640 | 3,822-3,829 | 3,958-3,967 | 5,302-5,304 |

A token's `dp4a` sum does not depend on the pass width (stream_dp4a.cuh), so every column of a wide
verify gets the logits the head computes for that token alone. Decode-path KLD at a 16-token window
(two 32K-token PG-19 chunks, against the FP32 base): 7.04e-4 and 98.79% the same top-1 (PPL 7.9720)
against master's 6.24e-4 and 98.90% (7.9707); the same text at one, four and eight tokens measures
6.16e-4, 5.27e-4 and 6.18e-4. Serve legs, ABBA against master `3c6a616d`, the router's flags,
`--ngram-draft-tokens 15`, 200 W, no thermal throttling:

| | before | after |
|---|---:|---:|
| DFlash2 K7, two requests at once (story + code, 256 tokens each), total tok/s | 89.3-90.9 | **94.6-96.0** |
| DFlash2 K15, code | 92.4-92.7 | **101.8** |
| DFlash2 K15, structured JSON | 164.8-164.9 | **180.8** |
| DFlash2 K15, Chinese / French | 20.8-20.9 / 33.5 | 22.9 / 36.8 |
| DFlash2 K15, story / thinking / prose (T 0.6) | 31.6 / 59.5 / 37.3 | 35.1 / 68.3 / 39.7 |

Code, Chinese, French and structured output keep their text at K = 15 (9.6-9.9% faster); story,
thinking and the conversations part at near-ties, since the head's arithmetic at 16 columns changed.
K = 7 is still the faster window on one request (structured 191.2-191.9 against 181.3-181.4 in the
same session). `NINFER_SM70_Q6_STREAM_WIDE=0` keeps the old head.

A 16-column form of the `dp4a` kernel itself (one pass instead of the tile GEMM for the Q4/Q5 linears)
was built and measured first and lost everywhere but the head: 123 registers a thread (no spills),
one CTA per SM, and K = 5,120 split into two chunks because sixteen tokens' int8 stage does not fit
96 KB in one. At 16 tokens Q4 34,816 x 5,120 took 316-354 us against the tile GEMM's 261-280,
Q5 5,120 x 17,408 217-240 against 204-213, Q4 4,096 x 5,120 77 against 57-58; on the head it took
2.48 ms, no better than two exact passes of eight.

## Next, in order

1. Run the full op suites (`scripts/v100_validate.sh --tests`); the linear, input-projection,
   `linear_add`, SwiGLU, GDN, MTP-round and attention suites have run.
2. Measure small-T on the other models (MTP / DFlash verification on the 27B is done: see "Decode"), and re-pick the sm_86 route tables where an `mma`
   route loses to a SIMT one on Volta (`docs/maintainer/linear-tuning.md` has the method).
3. Verify widths past 8 tokens: the int8 streaming GEMV (above) holds 1..8 tokens at the weight
   stream, so MTP up to seven drafts and DFlash2 up to K = 7 verify at about a plain step's GEMV
   cost. Past eight the Q6 head now runs `dp4a` passes of eight, but the Q4/Q5 linears still take
   the tile GEMM (1.2-1.9x the eight-token cost), and a 16-column `dp4a` kernel lost to it
   ("Verifies past eight columns" above): a form that keeps sixteen tokens' int8 stage in one
   K-chunk (smaller per-slice metadata) and fewer registers would be the next attempt.
4. Kernel count: a plain decode step is ~700 kernels, and each short GEMV loses ~8 us to fill and
   drain between dependent kernels (Volta has no programmatic dependent launch), ~1-2 ms a token.
   Fusing the norms and small element-wise steps into their neighbours is the remaining decode
   headroom beyond the GEMVs themselves.
5. Key-chunked flash attention (see "Prompt attention"), so long prompts keep INT8-kernel
   accuracy and bounded scratch.
6. Calibrate `context_cost_defaults.cpp` for the card (`ninfer_context_cost_bench`); until then the
   generic transfer and prefill costs apply.
