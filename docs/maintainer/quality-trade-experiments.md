# Quality trades: int4 vocabulary head and FP16 GDN state

Two speed-for-quality trades, both **implemented, compiling, tested-by-construction and never
measured**. They exist on branch `perf/quality-trades` behind environment variables so that one
build can be A/B'd against itself; neither is wired to a CLI flag, and neither should be until the
numbers below exist. The [`sm_86` findings](../performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
explain why they are the next levers: a decode round is bandwidth-bound at C1 and tensor-rate bound
at C8, and both trades buy bytes.

The reference stack this fork is chased against (syv-ai/qwen38-27b-rtx3090) runs both — it lists a
"calibrated int4 lm_head" and "fp16 state" in its optimized configuration.

## What each one does

### `NINFER_LM_HEAD_Q4=1` — int4 vocabulary head

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

### `NINFER_GDN_STATE_FP16=1` — FP16 recurrent state

The GDN recurrent state is 48 layers x 48 heads x 128 x 128 FP32 = **147 MiB per slot**, read and
written every round. At C8 that is ~3.6 GB of a 58 ms round; it is also the size of the host state
image that `--host-state-slots` pins.

The recurrence still computes in FP32 — FP16 only rounds what is *stored* between steps.
`LinearAttentionStatePoolSpec::recurrent_dtype` carries the choice; the four recurrent kernels
(direct, batch update, record, replay fold) are templated on the storage type and their launchers
dispatch on the tensor's dtype; the chunked prefill kernels keep their FP32 interface and the
wrapper stages FP16 through an FP32 workspace around them.

## How to measure, and the trap in measuring it

Run four arms off one build — base, `q4head`, `fp16state`, `both` — interleaved in one sitting,
because between-process spread on this card is 3-5% (see TODO.md, "This card is power-capped").

| signal | what it covers |
|---|---|
| `ninfer_bench` plain and MTP3, `-r 3` | round cost, both trades |
| the eight thinking-off prompts at C1 and C8 | end-to-end decode, both trades |
| `ninfer-perplexity --quick` | quality of the **head** directly |
| `ninfer_gdn_state_fp16_test [width]` | FP16-vs-FP32 state drift over 4096 decode steps |
| greedy output divergence vs base | quality proxy for both |
| `ninfer_q4_requantize_test` | per-group error no worse than `absmax/7`, and the routed Q4 head against an fp64 oracle at T=1..32 |

**Perplexity barely sees the FP16 state.** It scores through prefill in 1024-token chunks, so the
state is only rounded at three chunk boundaries per 4096-token window, where decode rounds it every
round. A flat perplexity is therefore *not* evidence that FP16 state is safe — that is what the
drift test and the divergence check are for. If a stronger number is wanted, teach the scorer a
smaller prefill chunk (the engine forces 1024 for `CausalScoring` in `engine.cpp`) and compare FP16
against FP32 at the same chunk size.

## If they win

Wire them to CLI flags rather than environment variables (`--lm-head-q4`, `--gdn-state-fp16`),
plumbed through `SpeculativeOptions`-style option fields into `StartupFeatures`, with `serve`, the
CLI and `ninfer-perplexity` all able to set them, and record the measured numbers here. Leave the
defaults alone: both change output, and this fork's rule is that a quality trade is opt-in with its
evidence written down.

Freeing the W8 head's now-unused upper halves is a further ~670 MB of VRAM for KV, and is not done.
