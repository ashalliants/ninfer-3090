# Tesla V100 (sm_70) port

Status: **runs on a V100**: decode is tuned (see "Decode"); prefill's wide GEMMs run on the
tensor cores through cuBLAS (see "Prefill"), its attention and GDN recurrence still on emulated `mma`.

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

Acceptance on that prompt: MTP3 3.3 tokens per round, DFlash2 5.2.

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

What is left in decode, by size: the GEMVs' gap to the read rate (~3.5 ms), the per-kernel
overhead of ~690 launches per token (~1.2 ms of gaps plus ramp/tail), and the GDN recurrent update
(~0.5 ms, 32% of its byte rate). The Q6 LM head (993 MB) runs on the streaming GEMV at 86% of the
read rate in the model (1477 -> 1309 us); its decode is the most ALU-heavy of the three formats,
so back-to-back microbenchmarks that hold the 200 W power cap understate it.

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
(emulated TF32/BF16 `mma`).

**Chunk size and the narrow tail.** `--prefill-chunk` past the default 1,024 is worth 1-4% here
(Qwen3.8-27B, 2k and 4k prompts, chunks 1,024 / 2,048 / 4,096: 796-802 and 748-774 tok/s). What
cost far more was a narrow segment: the route admits a call from `kCublasPrefillMinTokens` (129,
which decode/verify widths must stay below), and anything narrower takes the integer/A16 routes,
emulated here at ~80 tok/s. Steps also split at context-cache captures (just before the closing
template tokens) and rewrite frontiers, so 1,150 tokens ran as 1,024 + 122 + 4 and paid ~1.3 s for
the 122. With the route on, `prefill_tail_floor` (planning/startup.h, set to the gate) makes
`advance_prefill` shorten a step so the segment before the next frontier is at least the gate:
1,017 + 129 + 4, 449 -> 755 tok/s on that prompt, same output. Prompts shorter than the gate
still prefill on the narrow routes.

Thermals: sustained prefill holds the card at its 200 W cap and took a passively cooled V100 from
40 to 75 C in about a minute; measure in short runs without forced airflow.

### Why there is no HMMA path yet

The PTX `mma.m8n8k4` computes four independent 8x8x4 products per warp, one per octet (lanes
4o..4o+3 and 16+4o..16+4o+3). Expressing a m16n8k16 with it needs the operands in the octet
layout. With the m16n8k16 fragment layout the kernels already hold, that costs about 24 shuffles
per mma (so it is shuffle-bound at roughly the same rate as the emulation), or a zero-shuffle
scheme that uses a quarter of the tensor work. The real fix is for the kernels that matter (the
prefill GEMMs and attention) to hold Volta-layout fragments and use `wmma` or `m8n8k4` directly.
That is a kernel rewrite, and it cannot be validated without hardware.

## Compile-time facts (CUDA 12.8.1, `-arch=sm_70`)

- CUDA 12.8 is the newest toolkit that compiles for Volta (13.0 dropped it); nvcc warns that
  offline compilation for sm_70 is deprecated.
- A full `ninfer_ops` build is 419 translation units and about 15-20 minutes on 8 cores. One nvcc
  job peaks near 2 GiB; do not build with more jobs than RAM / 2 GiB.
- The container the build was checked in cannot run the result (no GPU). `ptxas` accepted every
  kernel, so register caps, shared-memory limits and instruction support are all checked; what is
  not checked is anything that needs execution.

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

## Next, in order

1. Run the full op suites (`scripts/v100_validate.sh --tests`); only the linear, input-projection,
   `linear_add` and SwiGLU suites have run so far.
2. Measure small-T on the other models (MTP / DFlash verification on the 27B is done: see "Decode"), and re-pick the sm_86 route tables where an `mma`
   route loses to a SIMT one on Volta (`docs/maintainer/linear-tuning.md` has the method).
3. The rest of prefill on tensor cores: prompt attention (FP16), the GDN chunked recurrence, and
   the narrow (< 129 token) tails, which still take the emulated small-T and A8 routes.
4. Calibrate `context_cost_defaults.cpp` for the card (`ninfer_context_cost_bench`); until then the
   generic transfer and prefill costs apply.
