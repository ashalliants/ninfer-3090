# Quality trades: int4 vocabulary head, FP16 GDN state, integer-activation MLP decode

Three speed-for-quality trades, all **implemented, measured, and wired to CLI flags**:
`--lm-head-q4`, `--gdn-state-fp16` and `--mlp-a8-decode` on `ninfer`, `ninfer-serve`, and
`ninfer-perplexity`. All default off. The [`sm_86` findings](../performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
explain why they were the next levers: a decode round is bandwidth-bound at C1 and tensor-rate bound
at C8, and both trades buy bytes.

**Verdict, ahead of the detail below:** `--gdn-state-fp16` is a clean win — free within measurement
noise on quality, a modest real C8 speedup, and it halves the host state image. Keep it enabled
whenever a serving profile can spare the output-fidelity risk (which measures as none so far).
`--lm-head-q4` is a narrower win — a real ~3% C8 gain, no measurable C1 gain on real text, at a
quality cost (+0.69% perplexity, visibly different greedy output within the first ~50 tokens) larger
than every KV format in [the KV table](../performance.md#choosing-a-kv-format-rtx-3090-qwen38-27b).
It is kept as an opt-in flag because the evidence supports it winning on C8, but it should not be
recommended as a default-on suggestion the way `--gdn-state-fp16` could be.

The reference stack this fork is chased against (syv-ai/qwen38-27b-rtx3090) runs both — it lists a
"calibrated int4 lm_head" and "fp16 state" in its optimized configuration.

## What each one does

### `--lm-head-q4` — int4 vocabulary head

The 27B's output head is `W8G32_F16S`, 248320 x 5120: **1.27 GB read for every decode step and
every verify round**, about 1.5 ms of a 25.7 ms C1 round and 2.7 ms of a 58 ms C8 round.
`ops::requantize_w8g32_to_q4g64_in_place` (`src/ops/linear/q4/q4_requantize.{h,cu}`) rewrites it to
`Q4G64_F16S` during weight binding, halving that read.

- Each 64-k group takes the fp16 scale, out of 25 clipping ratios of `absmax / 7` in [0.70, 1.18],
  that minimises the group's squared reconstruction error. Plain `absmax / 7` is one candidate, so
  the result is never worse than round-to-nearest.
- The conversion is in place: the Q4 code and scale planes are each exactly half their W8
  counterparts, written over the start of the W8 planes in ascending row chunks staged through a
  temporary. No extra VRAM, and the freed upper halves are simply left unused.
- `q4_dispatch` gained `n = 248320` routes, and `launch_q4_small_t_rows` serves any K = 5120 matrix
  whose rows are a whole number of CTAs (rows travel in the store, not the geometry).

**Refused** with overlay vision (an evicted head is restored from the artifact as W8 bytes, which
would then be read as Q4) and with DFlash/DFlash2 (its `linear_topk` reads the W8 head directly).
Both are checked at the call site in `src/targets/qwen3_6_27b/impl/load/bindings.cpp`.

### `--gdn-state-fp16` — FP16 recurrent state

The GDN recurrent state is 48 layers x 48 heads x 128 x 128 FP32 = **147 MiB per slot**, read and
written every round. At C8 that is ~3.6 GB of a 58 ms round; it is also the size of the host state
image that `--host-state-slots` pins.

The recurrence still computes in FP32 — FP16 only rounds what is *stored* between steps.
`LinearAttentionStatePoolSpec::recurrent_dtype` carries the choice; the four recurrent kernels
(direct, batch update, record, replay fold) are templated on the storage type and their launchers
dispatch on the tensor's dtype; the chunked prefill kernels keep their FP32 interface and the
wrapper stages FP16 through an FP32 workspace around them.

## Measured 2026-09-12, this box

Windows, RTX 3090, `sm_86`, 315 W cap, CUDA 12.8, int8 KV, Qwen3.8-27B, MTP3 with the optimized
draft head unless noted. This box's ceilings are **854.2 GB/s** and **67.6 TFLOPS BF16** — lower
than the rented Linux 3090 the small-T kernel numbers in `docs/performance.md` used (892.8 GB/s,
81.8 TFLOPS at 350 W), so absolute tok/s here is not comparable to that doc; the ratios below are.

Four arms off one build — base, `q4head`, `fp16state`, `both` — toggled by CLI flag (measured before
they were wired, via the env vars this doc used to describe; behaviour is identical either way,
since the flag only changes how `StartupFeatures` gets set).

**Perplexity** (`ninfer-perplexity --quick`, `ninfer-ppl-1m-v1`, context/stride 4096/2048, 261,167
tokens):

| arm | overall PPL | vs base |
|---|---:|---:|
| base | 4.342425 | — |
| `q4head` | 4.372320 | **+0.688%** |
| `fp16state` | 4.342315 | −0.003% (noise) |
| `both` | 4.372435 | +0.691% |

`q4head`'s cost is real and larger than every KV format in the KV table (nvfp4 KV is +0.36%, the
largest there). `fp16state`'s perplexity is unchanged, as expected — see the caveat below on why
that alone doesn't clear it.

**GDN state drift** (`ninfer_gdn_state_fp16_test`, synthetic 4096-token decode, FP16 vs FP32 state
through the real Op): width 1 (decode) worst relative error **0.339%**, quarterly means flat at
~0.31-0.32%; width 4 (MTP3) worst **0.219%**, quarterly means flat at ~0.19-0.21%. Both comfortably
under the 2% fail threshold and not trending upward — this is the evidence perplexity can't give.

**Q4 head requantization** (`ninfer_q4_requantize_test`): per-group error never worse than plain
`absmax/7` rounding, and the routed Q4 head matches an fp64 oracle at T=1..32. **OK.**

**Greedy output divergence** (one real prompt, 300 tokens, greedy, int8 KV, through `ninfer`):
`base` and `fp16state` are **byte-identical**. `q4head` first diverges from `base` at character
~210 — an early wording choice ("the cache evicts the least recently used item" vs "the cache
maintains a fixed maximum size and evicts..."), consistent with the perplexity cost showing up
immediately rather than only at long range. `q4head` and `both` differ from each other by two
characters over 300 tokens (a near-tied argmax flip on top of `q4head`'s already-altered
distribution) — `fp16state`'s effect is not perfectly inert once stacked on a coarser head, even
though it's inert alone.

**`ninfer_bench`, synthetic corpus (`bench/fixtures/bench_corpus.ids`), `-r 3`:** plain (T=1, no
speculation) decode tok/s moved base 44.65→45.16 (2 interleaved reps) vs `q4head` 45.27→46.47 —
consistently faster by +1.4%/+2.9%, close to the naive bandwidth-share prediction. **MTP3 decode
told a dramatically different story that turned out to be a measurement artifact**: base 71.05 tok/s
at 31.1% acceptance vs `q4head` **100.19 tok/s at 53.8% acceptance** (+41%). This is the exact
pitfall TODO.md already documents for this fixture (98.4% repeated bigrams, DFlash reports 100%
acceptance on it at every draft count) — quantizing the verification head apparently makes it agree
with the draft head *more* often on this repetitive text, which is a statement about the fixture,
not the model. **Do not quote the 41% figure for anything.** `fp16state` on the same fixture: 72.40
tok/s / 30.2% acceptance, i.e. flat.

**`tools/bench/run_chat_decode.py`, real prompts (syv-ai's eight `bench/prompts_real.jsonl`,
thinking-off, greedy, 256 max tokens, 2 interleaved reps) — this is the authoritative speed number,
since it doesn't share the synthetic corpus's repetition:**

| C | arm | decode tok/s (rep0, rep1) | mean | vs base | accept% |
|---|---|---|---:|---:|---:|
| 1 | base | 110.33, 105.70 | 108.02 | — | 62.9% |
| 1 | `q4head` | 110.13, 106.19 | 108.16 | +0.1% (noise) | 60.9% |
| 1 | `fp16state` | 107.16, 103.89 | 105.53 | −2.3% (noise) | 60.3% |
| 1 | `both` | 110.18, 108.19 | 109.19 | +1.1% (noise) | 60.6% |
| 8 | base | 420.09, 425.35 | 422.72 | — | 61.4% |
| 8 | `q4head` | 435.99, 436.36 | 436.18 | **+3.2%** | 61.1% |
| 8 | `fp16state` | 437.50, 425.06 | 431.28 | **+2.0%** | 61.3% |
| 8 | `both` | 451.57, 447.69 | 449.63 | **+6.4%** | 61.8% |

C1's rep0→rep1 drop (all four arms slower by 3-5% in rep1) is the same between-process spread
TODO.md already names; none of the C1 deltas clear that noise floor. C8's gains do — acceptance
stays in a tight 60.6-61.8% band across all four arms there, confirming the MTP3-bench-corpus jump
above was fixture-specific, not a real acceptance effect from either trade.

**Why perplexity alone doesn't clear `fp16state`.** It scores through prefill in 1024-token chunks,
so the state is only rounded at three chunk boundaries per 4096-token window, where decode rounds it
every round. The drift test and the divergence check exist because of this gap, and both came back
clean.

## Remaining loose end

Freeing the W8 head's now-unused upper halves after `--lm-head-q4` requantizes it in place is a
further ~670 MB of VRAM for KV, and is not done.

## `--mlp-a8-decode` -- integer-activation MLP gate_up at decode widths

Added after the other two, and measured differently because perplexity cannot see it. The kernel,
its per-width sweep and the five variants that were measured and rejected live in the header of
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`; the README carries the user-facing table. The short
version:

- The Op is 4-6% faster than the BF16 small-T kernel from sixteen columns up, and slower at eight,
  so the route is admitted for 16..32 columns only.
- End to end at C8 with MTP3 it is **+1.28%** over four interleaved repetitions (431.5 -> 437.0
  tok/s), which is what gate_up's share of a round predicts.
- Quality evidence is the FP64 oracle bound, 0.0080-0.0371 relative L2 across 2..32 columns against
  the 0.04 allowance, plus the fact that output demonstrably changes at C8.

**Why perplexity is silent on it, and what to use instead.** `CausalScoring` forces a 1024-token
prefill chunk, and this route covers 16..32 columns, so scoring never reaches it -- the flag is
accepted by `ninfer-perplexity` and changes nothing there. That is a genuine gap in the evidence
rather than a clean bill of health: the trade is only exercised by cohort decode, so the honest
checks are the oracle bound above and a greedy-divergence comparison at C8, both of which this
branch has. A stronger number would need a scorer that can run at cohort widths.

**The finding that outlived the kernel.** `docs/performance.md` called C8 tensor-rate bound, which
is why int8 looked like a 4.7x lever. `ncu` says otherwise: the kernel sits at 74% L1/TEX against
42% DRAM and 47% SM, so operand movement through L1 and shared memory is the limit and the int8
MMA's arithmetic advantage is mostly unspendable at this shape. The largest single win in the whole
exercise was not the instruction swap but deleting a redundant copy of the staged activation slab.
That reading applies to the other small-T kernels too -- they share the shape and the staging
pattern -- and is the reason a cp.async ring lost at every width here.
