# Single-GPU serving performance

> **Read the hardware label before the numbers.** This index and the per-model pages it links to
> carry measurements from two different GPUs, and they must not be read against each other:
>
> | section | hardware | measured by |
> |---|---|---|
> | [RTX 3090 (`sm_86`) findings](#rtx-3090-sm_86-findings-this-fork) below | RTX 3090, `sm_86`, CUDA 12.8 | this fork |
> | [Vision residency on RTX 3090](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86) | RTX 3090, `sm_86` | this fork |
> | Every per-model page under `performance/` | **RTX 5090, `sm_120a`, CUDA 13.1** | upstream |
>
> The upstream campaign is kept because it is the only corpus-scale evidence published for these
> artifact profiles, and all of its tested revisions are reachable in this fork's history. It is
> **not** a statement about how this fork performs on an RTX 3090. The two parts answer different
> questions: upstream's tables characterise the artifact profiles, and this fork's measurements
> characterise `sm_86` kernel behaviour — where upstream's inherited route boundaries were wrong by
> 12–41%.

## RTX 3090 (`sm_86`) findings — this fork

**DFlash2 measured on this fork (RTX 3090, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`--draft-tokens 7`, greedy, 96 new tokens).** Text: 20.0% acceptance, 2.38 tok/round.
Vision (`--vision`, the committed `image_chart` fixture): 85.7% acceptance, 7.00 tok/round,
and byte-identical output to the non-speculative vision run. These are single-prompt smoke
numbers, not a campaign; the [Vision residency](performance/qwen3.8-27b.md#vision-residency-on-rtx-3090-groupwise-int-sm_86)
and per-model pages remain the measured corpus results.

**Speculative decoding is not bit-identical to non-speculative decoding here, it is not required
to be, and every configuration is nonetheless deterministic in itself.** Measured 2026-09-09 by
hashing the generated text of 23 configurations x 3 repetitions
(`scripts/sweeps/dflash2-draft-tokens-realtext.ps1`, `content_sha256`):

- **every configuration reproduced its own output exactly**, 23 of 23, three runs each;
- the width-1 greedy path produced a hash matched by no speculative configuration;
- **DFlash2 and MTP do not match each other**, and DFlash2's output varies with the draft count —
  eight distinct outputs across k = 1..12.

Verification evaluates k+1 columns in one pass while plain decode evaluates one, so the reductions
run in a different order and a near-tie argmax can flip; a different draft count is a different
width and so a different order again. That is why bit-identity to greedy is not a target here: it
would require computing the accepted column with the width-1 kernel on every round, which is the
work speculation exists to avoid. It costs nothing in quality — swapping an MMA tile for a
different reduction order leaves perplexity bit-identical to twelve significant figures — so what
is guaranteed is per-configuration determinism, not cross-configuration equality.

*An earlier version of this section claimed DFlash2 and MTP "produce the same output as each other".
That was inferred from one prompt and is wrong; the hashes above are the measurement.*

**Acceptance on realistic text, which the committed corpus cannot measure.** 27B DFlash2, INT8 KV,
greedy, 256 generated tokens of the model's own prose, medians of three. `bench/fixtures/bench_corpus.ids`
reports a flat 100% at every draft count because it is a curated bank tiled to length; these are the
numbers to quote instead:

| `--draft-tokens` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acceptance | 80.9% | 67.6% | 51.7% | 49.1% | 42.1% | 33.1% | 28.2% | 31.0% | 23.2% | 19.5% |
| tok/round | 1.81 | 2.35 | 2.55 | 2.95 | 3.07 | 2.97 | 2.93 | 3.45 | 3.27 | 3.27 |
| decode tok/s | 47.3 | 55.2 | **58.6** | 58.3 | 57.2 | 51.0 | 50.5 | 47.5 | 39.7 | 39.3 |

Acceptance falls steadily as the window widens, from 80.9% at k=1 to 19.5% at k=12, with one
uptick against the trend at k=7 -> 8 (28.2% -> 31.0%); tokens-per-round plateaus near 3.0 from
k=4 — which is the real reason to recommend four rather than more. MTP3 reaches 58.3 tok/s at 56.0%
acceptance, and MTP3 with `--lm-head-draft` is the fastest configuration measured at 62.9 tok/s.

**Route boundaries are measured here, not inherited.** Two maintainer benches time every
schedule of an Op at the same column count, cold, so a boundary can be chosen from data rather
than from whichever table happened to be live:
`bench/ops/q4_q5_attn_input_schedule_bench.cu` and
`bench/ops/q4_linear_swiglu_schedule_bench.cu`. Both take an explicit per-schedule domain,
because a kernel handed more columns than it supports either corrupts memory (q4_q5
parent_split_fixed past 12) or silently does one column's work at a constant cost and appears to
win everywhere (q4 swiglu gemv_pair, registered for a single column). Cold is the production
regime for these: the Q4/Q5 attention projection alone streams ~35 MB of weights per call
against the 3090's 6 MB of L2.

The Q4/Q5 attention-input table had been dead code since the catch-up -- `resolve_plan` was a
hardcoded chain of upstream's sm_120 boundaries -- and routing through the measured table
exposed a `switch` fallthrough that ran a second kernel over the first. Public-Op cost at T=16
fell 502.8 -> 226.3 us and at T=32 500.7 -> 217.1 us. Q4 SwiGLU's SmallTTiled bound returned to
24, where the tiled kernel and the 40-wide pair tile actually cross (6% at T=25, 14% at T=32).

Concurrent decode extents are what the sm_86 kernel routes are selected for. A decode round covers
`concurrency x (draft window + 1)` token columns, and above the single-token point the cost of a
route is set by its padded tile width rather than by the live column count. Selecting the narrowest
tile that still covers each extent is worth 40% at C4 and C8; C1, whose four columns already sit on
the exact-T routes, is unchanged.

**The upstream DFlash2 corpus numbers are still not reproduced here.** The upstream catch-up
brought the DFlash2 speculative backend (`--spec dflash2 --draft-tokens 7`, Qwen3.8-27B only), and
upstream publishes DFlash2 corpus numbers for their own hardware. Those are deliberately not
added to the findings in this section: only single-prompt smoke numbers exist on a 3090 so
far, and pasting another architecture's corpus results beside them would read as agreement that
has not been measured. DFlash2 rows will be added here once measured on a 3090.

### Handing prefill GEMMs to cuBLAS (`--prefill-cublas`, opt-in)

**Result.** Prompt processing is 43-83% faster again, for +0.156% perplexity. The weights are
materialised as int8 with one scale per row and the GEMM is handed to cuBLAS, which runs this card's
shapes about twice as fast as this fork's own integer mainloop can. Measured on one RTX 3090,
Qwen3.8-27B groupwise-int, `--kv-dtype int8`, both arms on the same build and card in one session:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| shipped integer-activation routes | 1,666.6 | 1,634.4 | 1,503.9 | 1,240.8 |
| **`--prefill-cublas --prefill-chunk 4096`** | **2,379.8** | **2,988.7** | **2,609.0** | **1,904.1** |
| change | +42.8% | +82.9% | +73.5% | +53.5% |

Decode is untouched (45.1 against 45.7 tok/s, the route is gated to calls of at least 129 tokens,
above the widest 128-column verify round), and the workspace grows from 155.6 MiB to 661.2 MiB.

**It is off by default because it is a quality trade, not a free win.** cuBLAS reduces over the
whole of K, so it cannot see a scale per 64 columns: the weight carries one scale per row and the
activations one per token. Perplexity on `ninfer-ppl-1m-v1` (quick preset, kv int8) against 4.343155
for the integer route:

| | perplexity | change |
|---|---:|---:|
| MLP and out_proj on the route | 4.346284 | +0.072% |
| plus the attention and GDN input projections | 4.349944 | +0.156% |

The second step is `--no-prefill-cublas-projections` to decline. For scale, this fork has accepted a
trade at +0.082% and rejected one at +0.69%.

**Two things had to be true for the quality to land there.** The weight side is cheap because int8
buys four more bits per code while only losing per-group scale granularity: measured on the real
artifact it costs 9.7e-3 to 1.1e-2 relative L2 (the median per projection kind; the worst single
tensor is 1.5e-2), against the 9e-3 to 2.0e-2 this fork already accepts from per-group activation
quantisation (`tools/w4_row_scale_error.cpp`, using the route's own row scale with channel
equalisation off). An earlier figure of 8e-3 to 1.0e-2 came from a clipping search the route cannot
perform, and was the best case rather than the route. The activation side was
*not* cheap until the channels were equalised -- activation outliers concentrate in a few input
channels, so a token's absmax is set by those and everything else is quantised against far too
large a step. Folding a per-channel scale out of the activations and into the weights is exact
(`X[j,t]/s[j]` with `W[i,j]*s[j]` leaves the product unchanged) and took the cost from +0.474% to
+0.072%.

**Chunk size is the knob.** The dequantise pass is weight-sized and the GEMM token-sized, so the
route wants a large `--prefill-chunk` and the two settings belong together. The trade-off table,
including what each chunk costs a concurrent decode lane, is in
`src/ops/linear_swiglu/q4cublas/w4_cublas_prefill.h`.

### Integer activations for every registered prefill projection

**Result.** Prompt processing is 22-29% faster than the previous state at every context length:
the attention and GDN projections now reach the s8 tensor cores that only the MLP pair used, and
the schedule they share stopped re-streaming the weight matrix once per 128 tokens. Measured on one RTX 3090 at 315 W, Qwen3.8-27B groupwise-int, `--kv-dtype int8`,
`ninfer_bench -r 3 --warmup 1`, all three arms on the same build and card:

| prefill tok/s | 1k | 4k | 16k | 51k |
|---|---:|---:|---:|---:|
| `--no-prefill-a8` (every projection A16) | 1,008.5 | 995.4 | 946.9 | 836.4 |
| previous state (integer MLP only, 128-token tile) | 1,303.2 | 1,277.5 | 1,195.1 | 1,022.9 |
| every registered projection, 128-token tile | 1,529.8 | 1,493.3 | 1,382.3 | 1,153.3 |
| **every registered projection, widened tile** | **1,684.9** | **1,648.5** | **1,521.8** | **1,242.7** |
| change against the previous state | +29.3% | +29.0% | +27.3% | +21.5% |

Three routes were added, in the order their Nsight share justified — a 4,096-token prefill spends
749 ms of 3,280 ms in the GDN input projection, 346 ms in the two output projections and 209 ms in
the attention input projection:

| route | shape | per-Op, T=1024 | end to end, pp4096 |
|---|---|---|---|
| attention o_proj and GDN out_proj | Q5 [5120,6144] LinearAdd | 1,097.7 → 827.4 us | +3.0% |
| GDN input projection | Q4 [4096,5120] + Q5 [12288,5120] | — | +10.1% |
| attention input projection | Q4 + Q5 [7168,5120] | — | +1.6% |

They share one schedule (`src/ops/common/rowsplit_a8_mma.cuh`): the 128x128 tile the MLP routes
measured, with the codec and the output mapping as template parameters. Both input projections are
two parents over one activation, so each quantises its input once for all its launches — O(K*T)
against the GEMMs' O(N*K*T).

**Quality.** Quick corpus, same build: 4.342982 with `--no-prefill-a8` against **4.343155** with
every integer route on, **+0.004%** — inside run-to-run noise, and inside the +0.05% this fork
requires before a lossy route is default-on. Relative L2 against the FP64 oracle is 0.010-0.020
across every destination range, against the 0.04 allowance A8 activation compute is held to.
Decode is untouched (45.8 against 45.6 tok/s at tg128): the routes admit only full 128-column
prefill tiles, so decode and partial chunks stay on A16.

**The token tile, and why it was hiding.** Ablating the probe at the gate_up shape, T=512, gives
the decomposition that explains the whole schedule -- each row removes one cost and keeps the MMA
count identical:

| | us | |
|---|---:|---|
| complete kernel | 2,068 | |
| minus the activation and weight scale reads | 1,918 | the rescale's loads are ~7% |
| minus the per-group rescale entirely | 1,856 | the rescale is ~10% |
| minus the A-fragment shared reads | 1,423 | **assembling A is ~21%** |
| minus the B-fragment reads as well | 1,424 | B is free: one 128-bit load per n-tile |
| minus the MMAs, keeping every load | 1,377 | |
| minus the MMAs *and* every shared read | 1,378 | **streaming alone is 67% of the kernel** |

So these kernels were never compute-bound, and what generates the traffic is the token tile: a
block covering BN tokens re-streams the entire weight matrix once per column block, eight times
over at the production chunk of 1,024. Widening it is worth more than everything else attempted:

| tile (rows x tokens) | us at T=1024 | TOP/s |
|---|---:|---:|
| 128 x 128 | 4,265 | 85.6 |
| 128 x 256 | 3,702 | 98.6 |
| **64 x 512** | **3,060** | **119.3** |

A grid swizzle to let L2 serve the repeated passes measured within 0.5% of nothing: the passes have
to be removed, not cached. gate_up is the exception that proves the rule -- its gate/up pairing
fixes the row tile at 128 and so caps it at 256 tokens, where the tile it gains does not pay for
the operand assembly it would lose, so it keeps its own kernel.

**What is still missing, measured against a tuned kernel rather than against a microbenchmark.**
`tools/int8_gemm_reference_probe.cu` runs cuBLAS's own int8 GEMM (IMMA, TN, s32 accumulate) over
these shapes on this card. It solves a strictly easier problem -- 8-bit weights, no unpack, no
per-group scale, no fused epilogue -- and it reads *twice* the weight bytes our int4 codes do, so
it is an upper bound rather than a like-for-like rival. At T=1024:

| shape | cuBLAS int8 | this fork's W4A8 | ratio |
|---|---:|---:|---:|
| gate_up 34816x5120 | 237.5 TOP/s | 95.5 | 2.5x |
| down 5120x17408 | 176.1 TOP/s | 100.1 | 1.8x |
| out_proj 5120x6144 | 176.7 TOP/s | 93.3 | 1.9x |

That reading also corrects the ablation above: cuBLAS finishes the entire gate_up GEMM in 1,537 us,
less than the 1,711 us our schedule spends on streaming *alone* at the same shape and tile. The
streaming is not a hardware floor, it is our streaming -- a 2-stage pipeline over 16 warps with the
A fragments assembled from per-lane 2-byte loads. Everything cheap has now been tried against it:
pipeline depth 2 -> 4 is worth 3.9%, interleaving the B loads with the MMAs -1.9%, occupancy 6%,
grid swizzle 0.5%, operand layout ~17% (and that one needs a repack this engine does not allow).
What is left is not a knob but a mainloop: register-level double buffering, `ldmatrix`, and a
swizzled shared layout, which is what CUTLASS-class kernels -- and the Marlin kernel the vLLM
stacks use -- are built out of.

**What did not pay, so nobody re-walks it.** `TODO.md`'s long-standing explanation for these
kernels running at ~30% of the INT8 ceiling — 124 registers holding the SM to 16 of 48 warps — is
not what costs the time. `tools/w4a8_rowsplit_probe.cu` measures the alternatives at the gate_up
shape, T=512:

- **Doubling occupancy buys 6%.** A 64x128 tile compiles to 55 registers and genuinely runs 2
  blocks per SM (32 of 48 warps, confirmed with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`):
  1,890 → 1,785 us, against the 40% `ncu` estimated.
- **Smaller tiles past that lose badly**: 128x64 at 63 registers measures 2,844 us, 1.5x worse.
- **Layout is worth nothing without a repack.** Streaming the weights with `cp.async` in their
  RowSplit order, packed nibbles in shared and the scales as an async ring — everything the
  fragment-order probe won except permuting the weights — measures 1,726 us against production's
  1,701. `tools/w4a8_real_weight_probe.cu` puts the fully repacked layout at 1,566 us with
  per-group scales, so the whole layout lever is ~8%, and a permuted copy of the 27B's two MLP
  matrices is ~9.7 GB on a 24 GB card. The 1,415 us headline needs per-token activation scales,
  whose relative L2 reaches 12.9% on outlier-heavy inputs against 0.9-2.0% per group.

### Small-T tensor-core kernels for verify and cohort decode

**Result.** Qwen3.8-27B MTP3 decode is 1.5x faster at C1 and 1.7x at C8 than v0.9.1, with
perplexity bit-identical. Measured before/after on one rented RTX 3090 in one session: Linux,
CUDA 12.8, 350 W, INT8 KV, MTP3 with the optimized draft head, CUDA Graphs, greedy unless noted.

| workload | v0.9.1 | small-T kernels |
|---|---:|---:|
| thinking-off chat, C1 (their 8 prompts, C × 1000 / mean TPOT) | 74.6 tok/s | **113.2 tok/s** |
| thinking-off chat, C1, temperature 0.7 | 72.6 tok/s | **112.8 tok/s** |
| thinking-off chat, C8 | 268.6 tok/s | **460.4 tok/s** |
| reasoning cohort C1 / C2 / C4 / C8 decode | 63 / 89 / 137 / 221 | **93 / 168 / 271 / 376** |
| `ninfer_bench` plain / MTP3 | 39.98 / 53.96 | **47.15 / 85.29** |
| perplexity, quick corpus | 4.342425 | 4.342425 |

The thinking-off prompts are
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)'s
`bench/prompts_real.jsonl`. That project serves the same model on the same card through patched
vLLM, and reports 111-124 tok/s at C1 and 407.3 at C8 with the same metric. Their card is capped
at 250 W and they report 5-8% run-to-run spread, so C1 is parity and C8 a lead, measured on
different machines.

**Why the round cost too much.** An MTP3 verify is four token columns and a C8 cohort round is 32.
The profile attributing an MTP3 round against a plain decode step
(`scripts/sweeps/decode-step-profile.sh 27b-decode 27b-decode-mtp3`) had the round at 38.2 ms
against a 24.8 ms step, about 1.5x. vLLM/Marlin pays 1.14x for the same round on this card. The
extra time sat in four families: Q4 gate_up, the Q5 residual projections, and the GDN and
attention input projections. At four columns they ran either SIMT split kernels, whose cost grows
with every column, or a Q4 small-T MMA that was instruction-bound. The round is now 25.7 ms
against a 22.3 ms step, 1.15x.

**What changed**, cold single-kernel medians in us, from the schedule benches
(`bench/ops/*_schedule_bench.cu`):

| Op (27B shape) | T | v0.9.1 route | now |
|---|---:|---:|---:|
| Q4 gate_up, 34816 × 5120 | 4 | 200.7 | **120.8** |
| | 32 | 444.4 (c40 tile) | **~205** |
| Q5 down, 5120 × 17408 | 4 | 99.3 (split2) | **82.9** |
| | 32 | 289.8 (best MMA tile) | **144.4** |
| Q5 out, 5120 × 6144 | 4 | 38.9 (split2) | **34.8** |
| | 32 | 105.5 (best MMA tile) | **55.3** |
| GDN input, Q4 4096 + Q5 12288 rows | 4 | 100.4 (independent) | **78.8** |
| | 32 | 263.2 (grouped c32) | **153.6** |
| attention input, Q4 7168 + Q5 7168 rows | 4 | 90.1 (parent split) | **68.6** |
| | 32 | 191.5 (r32/c32) | **135.2** |

- **Bank conflicts.** The Q4 small-T MMA staged its code rows 256 B apart, an 8-way shared-memory
  bank conflict on every A load. Padding the rows was worth 9.6% at C1 on its own.
- **Permuted k order.** The MMA's k slots may be any permutation of a group, provided A and B agree.
  Giving each lane sixteen contiguous k makes its A operand one 64-bit code load, decoded to bf16
  by a magic-bias `byte_perm` with no int-to-float, and its B operand two 128-bit loads with no
  `ldmatrix`.
- **A Q5 counterpart** (`q5_small_t_mma.cuh`) folds the high-bit plane into the same decode. It
  replaces split2 from T=3 and the MMA tiles to 32 columns, for the residual projections, GDN
  value/z and attention gate/value.
- **Wide extents share activation slabs** (`ops/common/small_t_layout.cuh`). At 32 columns and one
  16-row tile per CTA, gate_up re-read ~713 MB of staged activations per call against 89 MB of
  weights. Two to four row tiles per CTA now share each staged slab, and at 24 columns each warp
  feeds one B fragment to two tiles.
- **The fused GDN conv projection is gone.** Batch-1 widths 1-3 and 5-6 of the GDN conv
  forms ran a fused SIMT projection that lost at every width: 95.2 against 80.9 us at T=1, and
  162.8 against 85.0 at T=5. At T=5 that was 4.8 ms of a four-draft-token round. Removing it made
  plain decode 5% faster, and four or five draft tokens stopped costing more than they return.

**Measured and rejected**, so nobody re-runs them:

| idea | result |
|---|---|
| split-K across CTAs for the 5120-row Q5 shapes | 0-2.5% at narrow T; not worth a workspace |
| magic-bias decode alone, before the k permutation | neutral in situ |
| 2- and 3-stage cp.async rings for the narrow Q4 tile | slower; occupancy is the latency hiding |
| deeper rings through dynamic shared memory (to 4 stages, 99 KB) | slower at every shape: occupancy loss outweighs the hidden latency |
| two tiles per warp at 16 and 32 columns | slower (kept only at 24, where it wins 14%) |
| `cp.async ... L2::256B` prefetch hint on weight loads | neutral |
| a 40-65K-row prefix of the frequency-sorted draft head | acceptance 62.7% -> 56.5% at 40,960 rows; net 4% slower |
| drafting with the full output head | +2% acceptance for 4 ms more per round |
| programmatic dependent launch to hide kernel ramp | needs sm_90; compiled out on `sm_86` |

What is left at C1 is mostly ramp-up and drain at the ~500 kernel boundaries of a round (the
Q5 residual kernels reach 68-82% of their streaming floor, gate_up 89%).

**What is left at C8 is not tensor-core rate, though this page said so for a cycle.** The claim was
that gate_up at T=32 runs at about 68% of the card's measured bf16 MMA peak -- 205 us against a
139 us tensor floor -- which reads as a kernel most of the way to saturating its tensor cores.
Counters disagree. `ncu` on the T=32 gate_up kernel, Windows RTX 3090 at 315 W:

| | BF16 small-T | int8 small-T |
|---|---:|---:|
| tensor pipe active | **38.3%** | 21.0% |
| L1/TEX throughput | 36.4% | 73.5% |
| DRAM throughput | 39.4% | 42.7% |
| warp occupancy | 30.9% | 45.4% |

Nothing is near saturation in the BF16 column: tensor, L1 and DRAM all sit within three points of
each other around 38%, at 31% occupancy, which is the signature of a latency-bound kernel rather
than one limited by any unit. The 68% figure compares against a computed floor, not against a
measured pipe, and the two do not agree.

That distinction decided a real experiment. An int8 tensor-core route for this Op was built on the
strength of the old reading, since s8 MMA is about 4.7x the bf16 rate on this hardware. It
delivered 4-6%, not 4.7x, and the right column above says why: it did exactly what a denser MMA
should, halving tensor-pipe pressure from 38.3% to 21.0%, but that pipe was never the constraint,
and the cost of getting there landed on L1 at 73.5%. See
[the quality-trade notes](maintainer/quality-trade-experiments.md) and
`src/ops/linear/q4/q4_small_t_mma_i8.cuh`. Work aimed at C8 should target operand movement and
occupancy; a wider tile that puts more work in flight is the lever the counters actually point at.

**Reproduce.** Build both trees with `-DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON` and run
each harness against both `ninfer-serve` binaries on one card, interleaved in one sitting:
`tools/bench/run_qwen38_replayssm_cohort_sweep.py` for the cohort, and for the thinking-off numbers

```bash
python tools/bench/run_chat_decode.py --model models/qwen3_8_27b.ninfer \
  --prompts prompts_real.jsonl --concurrency 1 --reps 2 --out chat-decode \
  --arm base=/path/to/old/ninfer-serve --arm new=./build/apps/ninfer-serve
```

with the eight prompts of syv-ai/qwen38-27b-rtx3090's `bench/prompts_real.jsonl` (that file is
theirs and is not vendored here). `--concurrency 8` gives the C8 row. Both report decode as
C × 1000 / mean TPOT from the server's own request log.

### Integer-activation MLP at decode (`--mlp-a8-decode`, opt-in)

Qwen3.8-27B on `sm_86`, off by default. The MLP gate_up projection already runs its full prefill
tiles through the s8 tensor cores; this flag extends that to the widths a cohort round decodes at,
quantising activations to s8 with one scale per (token, 64-k group) and feeding
`mma.m16n8k32.s8.s8.s32`. It buys a little speed and costs a little fidelity.

**The Op, paired against the BF16 small-T kernel inside one sitting** (the card drifts several
percent between sittings, so only the pairing is meaningful), cold, median of 15:

| columns | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| run 1 | +10.7% | -5.8% | -3.4% | -4.8% | -4.8% | -5.0% | -4.5% |
| run 2 | +7.5% | +4.3% | -0.7% | -5.9% | -6.1% | -4.0% | -6.4% |

It wins from sixteen columns up and loses at eight, so the route is admitted for 16..32 columns
only and every narrower width stays on the BF16 kernel. A cohort round reaches those widths through
concurrency: eight lanes verifying four MTP columns each is thirty-two.

**End to end it is worth about a percent.** Eight concurrent thinking-off chat requests, MTP3,
INT8 KV, greedy, four interleaved repetitions: 431.5 tok/s by default against **437.0 tok/s**
(+1.28%) with the flag. That is the expected size: gate_up is roughly a quarter of a C8 round, so
four to six percent off it arrives as one percent overall. The flag won three of the four paired
repetitions and tied the fourth, against a spread of about 1.7% within the unflagged arm alone.

**What it costs.** Output changes -- this is a lossy trade, not a free one. Against an FP64 oracle
the Op measures 0.0080 to 0.0371 relative L2 across 2..32 columns, inside the 0.04 allowance the
integer-activation path is held to everywhere else in the tree. Perplexity cannot see this trade at
all: the route is admitted only in the verify phase, and scoring runs the prefill phase, so
`ninfer-perplexity` reports the same score with and without the flag. Judge it on the oracle bound
and on your own outputs. It does nothing for single-stream use, where one request decodes one
column per step, far below the sixteen the route needs.

### Earlier baselines (v0.9.1, a different RTX 3090 host)

Measured before the small-T kernels and before the cuBLAS prefill route, on a different host from
the sections above; compare within a table, not across sections. Qwen3.8-27B, INT8 KV, ReplaySSM,
MTP3, CUDA Graphs.

**Long-output cohort.** Every request generated 1,024 tokens from a 29-34-token prompt with an
8,192-token per-request context, so each sequence reached roughly 1,053-1,058 tokens. This is a
long-output/decode benchmark, not an 8K-prompt test. C1 used an 8,192-token shared KV pool; C2-C8
used 16,384 tokens so every requested output could be admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **77.84 tok/s** | **78.71 tok/s** | 71.27% | 133 ms | 19,475 MiB |
| C2 | 2,048 tokens | **94.75 tok/s** | **96.04 tok/s** | 62.17% | 225 ms | 19,919 MiB |
| C4 | 4,096 tokens | **136.43 tok/s** | **139.91 tok/s** | 66.15% | 420 ms | 20,247 MiB |
| C8 | 8,192 tokens | **240.34 tok/s** | **250.26 tok/s** | 69.74% | 866 ms | 20,903 MiB |

**Prompt processing.** 4,362 fresh input tokens per request, 512-token prefill chunks, prefix reuse
disabled, 16 generated tokens so the run measures prefill rather than decode.

| Cohort | Total fresh input | Aggregate prefill | Active-prefill speed | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| C1 | 4,362 tokens | **861.51 tok/s** | 893.98 tok/s | 4,893 ms | 19,114 MiB |
| C2 | 8,724 tokens | **853.86 tok/s** | 883.95 tok/s | 7,478 ms | 19,697 MiB |
| C4 | 17,448 tokens | **847.26 tok/s** | 874.49 tok/s | 12,692 ms | 20,894 MiB |
| C8 | 34,896 tokens | **844.10 tok/s** | 870.94 tok/s | 23,028 ms | 23,207 MiB |

`Aggregate prefill` is total fresh input divided by the complete request-wave time, the user-facing
number. NInfer processes one long prefill at a time: cohort batching accelerates decode but does not
multiply prompt ingestion, so C1-C8 stay near 844-862 input tok/s while queued requests raise mean
TTFT. `Active-prefill speed` excludes queue waiting.

### Recommended configurations (RTX 3090, Qwen3.8-27B)

Every flag below is measured elsewhere in this file or in
[quality-trade-experiments.md](maintainer/quality-trade-experiments.md). Nothing here is a guess.

**Fastest at one stream, when context beyond ~130K is not needed:**

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --prefill-cublas --prefill-chunk 4096 --kv-dtype rk8v4 --embedding-q4 --gdn-state-fp16
```

Prefill about 1.7x and decode about 1.39x against the previous defaults. Costs +0.156% perplexity
from the cuBLAS route, +0.083% from rk8v4, and nothing measurable from the other two.

**Longest context, still fast:**

```
--spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --prefill-chunk 2048 --kv-dtype rk8v4 --embedding-q4 --lm-head-q6 --gdn-state-fp16
```

**200,000 tokens of context, verified by loading it.** MTP rather than DFlash2 because DFlash2's
draft weights and its refusal of `--lm-head-q6` cost about 65K tokens between them -- the same
configuration on DFlash2 loads at 130K and fails at 150K (needing 4.79 GB against 4.46 free) and at
200K (6.09 GB against 4.45). Chunk 2048 rather than 4096 because the larger chunk costs ~300 MiB of
workspace for its last 4% of prefill, and at this context that is the binding constraint.

**Why these flags and not others:**

| flag | what it buys | what it costs |
|---|---|---|
| `--kv-dtype rk8v4` | 23% less KV than int8 — 26,112 B/token against 33,792 | +0.083% perplexity, ~1% decode |
| `--embedding-q4` | +24.3K tokens of context | nothing measurable (−0.062%, inside noise) |
| `--lm-head-q6` | +12.9K tokens of context | +0.012%; **incompatible with DFlash/DFlash2** |
| `--gdn-state-fp16` | halves the host state image, modest C8 gain | nothing measurable |
| `--prefill-cublas` | prefill 1.63x–1.83x | +0.156% perplexity, ~500 MiB workspace |
| `--lm-head-q4` | +24.3K tokens | **+0.69%** — larger than every KV format; not recommended |
| `--kv-dtype nvfp4` | 45% less KV than int8 | +0.36% perplexity, 12% slower decode |

**With vision**, add `--vision --vision-residency overlay`: the overlay residency keeps the tower in
host memory and borrows device memory per image, which is what preserves the context budget above.
Note that `--lm-head-q4` was silently skipped under overlay vision before 2026-09-14; `--lm-head-q6`
and `--embedding-q4` transcode in the materializer and are captured from the final bytes, so they
compose with it correctly.

### Choosing a speculative backend by concurrency (RTX 3090, Qwen3.8-27B)

**DFlash2 at K=7 is faster than MTP3 wherever it fits, and it stops fitting at C8.** Aggregate
decode tok/s through the serving route, thinking off, greedy, decode time taken from the server's
request log rather than a wall clock:

| C | MTP3 | DFlash2 K=7 | change | tokens/round, MTP3 / DFlash2 |
|---:|---:|---:|---:|---|
| 1 | 135.0 | **187.1** | +38.6% | 3.51 / 5.57 |
| 2 | 238.2 | **313.5** | +31.6% | 3.51 / 5.82 |
| 4 | 387.4 | **406.2** | +4.9% | 3.54 / 5.63 |
| 8 | **522.8** | does not fit | — | 3.53 / — |

The lead shrinks with concurrency because speculation pays by making a multi-column round cost about
one sweep of the weights, and batching already amortises that sweep across lanes, so the baseline
catches up. Acceptance itself does not degrade: tokens per round holds flat at every level.

What stops it at C8 is memory. The DFlash2 artifact carries the draft model — 18.3 GiB of weights
against 16.7 — and the runtime reservation then needs 4.64 GB where 3.78 GB remains. A 4096-token KV
still needs 4.24 GB; only 2048 fits, which is too little for eight streams to do useful work.

**K is workload-dependent.** Swept on realistic generation, mean of a reasoning, a code and a
summarisation prompt: K = 3/4/5/6/7/8/9/10/12 gives 128.0/146.0/159.0/169.6/**172.3**/163.7/163.3/
160.2/159.2 tok/s. The peak is 7, matching the published shape `E = (1 - a^(K+1))/(1 - a)` and the
5–8 band reported for EAGLE-class drafters. But the mean hides real spread — the reasoning prompt
keeps improving to K=12 (204.5) while summarisation collapses there (119.6) — so a deployment
serving one kind of work should sweep its own K. A synthetic corpus is no guide: `bench_corpus.ids`
picks K=15, which loses on real generation.

**DFlash2 also forfeits `--lm-head-q4/q6`**, which DFlash's candidate top-k cannot read
(`src/models/qwen3_5/load/storage_trades.cpp`). Between the extra weights and the lost head trade it
costs about 65K tokens of context on this card -- 130K against 200K, both verified by loading -- so
the choice between it and MTP3 is a speed/context trade, not a free upgrade.

### Choosing the draft count (RTX 3090, Qwen3.8-27B)

The MTP draft count is a trade on what the output looks like. Each round verifies K+1 columns and
runs K draft-head steps whether or not the drafts survive, so a larger K pays only where the head
keeps guessing right. Measured on the RTX 3090, Qwen3.8-27B, `rk4v4` KV, `--lm-head-draft
--lm-head-q6 --embedding-q4 --gdn-state-fp16`, greedy, 512 generated tokens, one stream (decode
tok/s; tokens emitted per round in parentheses):

| `--draft-tokens` | edit code (copies the prompt) | write new code | explain a concept | short story |
|---:|---:|---:|---:|---:|
| (none) | 47.3 | 47.5 | 47.5 | 47.6 |
| **3** | 141.9 (3.9) | 118.2 (3.3) | 102.1 (2.8) | 77.1 (2.1) |
| 5 | 177.7 (5.7) | 124.8 (4.0) | 95.6 (3.1) | 75.1 (2.4) |
| 7 | 226.0 (7.4) | 130.6 (4.3) | 103.9 (3.4) | 67.7 (2.2) |
| 9 | 226.4 (8.8) | 129.9 (5.0) | 91.3 (3.5) | 56.3 (2.2) |
| 11 | 244.4 (10.4) | | | |
| 15 | 262.6 (12.5) | 114.7 (5.4) | 75.1 (3.5) | 47.6 (2.2) |

Three stays the default: it is best or within 2% on prose, and larger counts lose up to 38% there.
When the output mostly reproduces the input -- refactoring, renaming, applying an edit and
returning the whole file -- the draft head predicts it almost perfectly and every extra position is
nearly free, so 11 to 15 is up to 1.85x faster than three. For a coding assistant that mostly
writes new code, seven is about 10% faster than three. The former MTP-only context lookup added
nothing on top of MTP there: the head already copies (K=15 with and without `--lookup-ngram 8`:
262.4 and 262.6 tok/s, the same tokens per round). It verified at the same width as the head, so
it could never widen a round; it has been removed in favour of
[n-gram copy drafting](#n-gram-copy-drafting-rtx-3090-qwen38-27b), which verifies a copy wider than
the neural window. For comparison, DFlash2 at its default seven is still faster at one
stream on the same runs (268.5 tok/s editing code, 115.7 explaining); MTP is the backend that fits
the full context and a second lane. Draft counts of eight and above add a second CUDA Graph
topology class on the 27B (its wide verify moves between the prompt and chunked attention routes),
which reserves about 64 MiB more per lane.

DFlash2 accepts every draft count from 1 through 15. Seven is the checkpoint recommendation and the
best mean on this card; the sweep behind it is in
[Choosing a speculative backend by concurrency](#choosing-a-speculative-backend-by-concurrency-rtx-3090-qwen38-27b).
DFlash2's `--lm-head-draft` is within noise of unset at every count and can be left off.

### N-gram copy drafting (RTX 3090, Qwen3.8-27B)

`--ngram-draft-tokens 15` beside DFlash2 K=7 (draft head, `rk4v4`). A copy round verifies 16
columns; at one lane it skips the drafter, at two lanes the drafter still proposes for the lane
that does not copy. It pays only while a 16-column round costs well under twice an 8-column DFlash2
round. Measured with the round bench (`ninfer_qwen3_5_dflash_round_bench --copy-width 16`): every
timed copy round accepts all of its drafts; GPU ms per round, median (copy) or mean (neural) of 30,
corpus prompt:

| context | lanes | DFlash2 round, 8 columns | copy round, 16 columns | ratio |
|---|---:|---:|---:|---:|
| 2,048 | 1 | 28.8 ms | 34.9 | 1.22 |
| 32,768 | 1 | 31.5 ms | 39.5 | 1.26 |
| 65,536 | 1 | 33.9 ms | 44.6 | 1.32 |
| 131,072 | 1 | 39.2 ms | 55.2 | 1.41 |
| 2,048 | 2, one copying | 38.4 ms | 55.3 | 1.44 |
| 32,768 | 2, one copying | 43.3 ms | 63.7 | 1.47 |
| 65,536 | 2, one copying | 48.4 ms | 72.1 | 1.49 |
| 2,048 | 2, both copying | 38.4 ms | 56.9 | 1.48 |
| 32,768 | 2, both copying | 43.3 ms | 66.4 | 1.53 |
| 2,048 | 3, one copying | 48.9 ms | 118.9 | 2.43 |
| 2,048 | 4, one copying | 58.7 ms | 138.8 | 2.36 |

(The 131,072 row is from the one-lane measurement before two lanes were built; the three- and
four-lane rows are from before copy rounds were limited to two lanes, and the bench now rejects
`--copy-width` above `--batch 2`.) A full copy round
yields 16 tokens for 1.2-1.4 DFlash2 rounds, which yield about 7.4 on copyable text. Beside a lane
that does not copy it costs 1.44-1.49 rounds, so the other lane decodes **about a third slower in
those rounds**; a two-lane round therefore copies only when the longest copy offers at least 12
drafts (12 + 1 + p >= 1.47 (7.4 + p) for a free lane's p = 2-3 tokens per round). From three lanes
the 48-64-column round costs 2.4x and returns less than it costs, so copy rounds run at one and two
lanes only and the copy family captures no larger batch. Wider windows need the verification Ops
extended past 16 columns; a final-prefill-chunk proxy for them (same route resolver, prefill GDN
instead of the record path, so only indicative) costs 1.9-7.1x an 8-token chunk at 32 tokens and
4.9-8.2x at 64, against 1.3-1.5x at 16, because those widths fall to the prompt attention route.
Windows of 31 and 63 are not worth building on this card on that evidence.

**Copy misses.** A request whose copy round commits fewer than 4 of its copied tokens doubles the
match it needs before it copies again (12, then 24, then 36 tokens); each round that copies well
lowers it one step. Reverting a diff is where this matters: the file and both sides of the diff
offer near-identical text, and without the backoff its copies were accepted 46% of the time.

**Agent replay, one lane** (`tools/bench/agent_replay`: 4 sessions, 112-114 turns of Read/Edit/
explain traffic growing to 39-117K tokens, greedy, 131,072-token KV; two runs per arm, alternating,
each arm's runs within 0.2% of each other, and one run of the build before the backoff in the same
session, which reproduced that build's own runs within 0.2%; decode tok/s over committed tokens):

| category | off | on, before the backoff | on | change | share of output copied |
|---|---:|---:|---:|---:|---:|
| rewrite a file | 187.4 | 332.8 | 330.5 | +76% | 87% |
| apply a change | 198.5 | 341.4 | 342.7 | +73% | 90% |
| edit a function | 230.2 | 352.5 | 352.1 | +53% | 85% |
| revert a diff | 212.8 | 204.9 (-3.7%) | 212.5 | **-0.1%** | 9% |
| explore | 163.2 | 193.7 | 199.9 | +22% | 45% |
| explain | 74.9 | 72.2 (-3.6%) | 71.3 | **-4.8%** | 1% |
| all turns | 174.6 | 240.5 | 248.0 | +42% | 72% |

Whole replay wall time drops from 786 s to 677 s. Copies were accepted 95% of the time (117,476 of
124,280 copied tokens). The backoff took reverting a diff from 243 copy rounds per run at 46%
acceptance to 88 at 51% and back to the speed it has without copies; the file-returning categories
moved by -0.7% to +0.4%. Explaining is the one category still slower, and not by its round cost:
its rounds take 32.90 ms against 32.76 ms off (+0.4%, with 5 copy rounds per run), while its answers
differ between the arms (6,179 against 6,837 tokens) and the "on" answers draft worse (2.34 against
2.45 tokens per round). It is not a like-for-like pair.

**Agent replay, two lanes** (the same replay with two sessions in flight, `--max-concurrency 2`,
147,456-token KV; two runs per arm, alternating; which turns share rounds depends on timing, so runs
of an arm differ by up to 20% per category):

| category | off | `--ngram-draft-tokens 15` | change |
|---|---:|---:|---:|
| rewrite a file | 108.9 (109.3 / 108.4) | 179.2 (159.7 / 198.8) | +65% |
| apply a change | 103.3 (107.4 / 99.2) | 164.4 (176.8 / 152.0) | +59% |
| edit a function | 152.1 (157.7 / 146.5) | 177.9 (182.0 / 173.7) | +17% |
| revert a diff | 110.5 (114.8 / 106.1) | 89.3 (88.3 / 90.3) | **-19%** |
| explore | 95.1 (85.8 / 104.3) | 81.9 (82.3 / 81.4) | **-14%** |
| explain | 40.2 (40.0 / 40.5) | 40.1 (40.7 / 39.4) | -0.4% |
| all turns, per request | 99.2 | 124.0 | +25% |
| both lanes together (output tok/s over wall time) | 117.8 / 120.1 | 139.4 / 134.8 | +15% |

Wall time drops from 674-677 s to 593-604 s. The losses are the other lane's: a turn that copies
little shares its rounds with a partner that copies, and those rounds verify 16 columns for both. A
diff revert's rounds take 78 ms against 61-65 off at the same tokens per round (6.95 against 6.97),
an exploring turn's 76-85 ms against 52-60. Explaining turns, which mostly overlapped each other, are
unchanged. Two lanes traded the latency of the lane that does not copy for throughput: the batch
gained 15%, a copying request up to 65%, and a request beside it lost up to a fifth.

**Partner balance (2026-10-12).** A two-lane round now copies only when every row without a copy
can afford the widening. Each request keeps a balance in thousandths of a two-lane neural round:

- +30 for every two-lane round it decodes in;
- -470 (r - 1, r = 1.47 from the table above) for a copy round in which it had no copy;
- for a copy round in which it copied, the rounds that copy saved it,
  committed / (K + 1) - r. K + 1 = 8 is the most a neural round commits, so the credit never
  overstates the saving.

A request that never copies is widened at most about once in 16 of its rounds, about 3% of its
decode-round time. One whose own copies have saved it rounds can absorb its partner's. Only
committed rounds move a balance, never timing. The constants are measured at DFlash2 K = 7 only, so
two lanes copy at K = 7 and no other `NINFER_DRAFT_TOKENS`: at any other window a two-lane batch
runs the neural round (batch one still copies) and the copy family is captured at batch one only.
Infernix has no partner rule for its Qwen3.5 lookup
drafting: any copy widens the whole batch.

Same replay, `--max-concurrency 2`, 147,456-token KV. Three runs off, three with the balance and two of
the build without it, interleaved in one session.

Two measures:

- **Engine rate:** committed tokens over the request's own decode-round time, the request log's
  `engine_timing.decode` device wait plus host time. It leaves out the partner's prefill chunks and
  queueing, which dominate short turns and made the wall-clock rate swing by up to 20% between runs
  above.
- **Wall rate:** output over first-to-last-token time, as in the table above.

| category | off | with the balance | change, engine (run range); wall | without the balance (2 runs), engine; wall |
|---|---:|---:|---:|---:|
| rewrite a file | 150.4 | 224.0 | +49% (+46..+52); +53% | +60%; +50% |
| apply a change | 145.6 | 214.2 | +47% (+42..+53); +20% | +60%; +52% |
| edit a function | 179.9 | 231.3 | +29% (+21..+36); +26% | +38%; +37% |
| revert a diff | 157.3 | 155.7 | **-1.0%** (-5..+4); +16% | **-14.7%** (-21..-8); -0.7% |
| explore | 118.1 | 128.5 | +8.8% (-1..+19); -0.5% | +2.0%; -11% |
| explain | 57.8 | 57.7 | **-0.1%** (-5..+4); -1.0% | **-4.9%** (-10..-0); -6.6% |
| all turns | 135.3 | 170.7 | +26% (+22..+28); +22% | +27%; +23% |
| both lanes together (output / wall) | 125.9 (123.0-127.9) | 144.6 (142.6-146.5) | **+14.9%** | 144.1 (+14.5%) |

Engine rates in tok/s, means of the runs; a run range compares each run with each run of the other
arm. Wall time drops from 641-664 s to 562-574 s (568 s twice without the balance).

- **Throughput.** The balance keeps the batch gain: +14.9%, against +14.5% without it in the
  same session.
- **Partners.** It removes the partner losses: turns that copy little are within 1% of off where
  they were 5-15% slower.
- **Cost to the copying side.** It copies 15% fewer rounds (3,465 against 4,100 per run) and 60% of
  the output against 71%. Writing back a file gains 49% instead of 60%.
- **Determinism.** Greedy outputs are not comparable across two-lane replay runs. Even two runs of
  the off arm differ in 47 of 112 turns, because which rows share a round changes the batch
  arithmetic. Output identity is checked by the copy-beside-free-lane real test instead.

The bench corpus, which never copies, decodes the same with and without the flag at two lanes:
86.81 against 86.74 batch tok/s (+0.08%, 8 interleaved runs each, ranges 86.60-87.47 and
86.41-87.67). The build without the balance gave 86.75 against 86.56 in the same session.

**Without copies.** The bench corpus (`ninfer_bench -n 512`, two alternating rounds) never copies.
At one lane DFlash2 decodes 63.40 and 63.29 tok/s with the flag on against 63.96 and 63.66 off
(-0.7%); the one-lane build before two lanes measured 63.30 and 63.18 in the same rounds. At two
lanes the bench corpus decodes 87.55 and 87.57 tok/s per batch with the flag on against 85.98 and
86.42 off (+1.6%, identical rounds and acceptance).

*Where that cost comes from: not the n-gram work; leading hypothesis, buffer placement.* The
evidence below argues against the lookup, the copy graphs and any extra GPU work, and points at where
the persistent buffers land within the arena. Placement is not isolated: no single buffer has been
shown to carry the cost. The cost also comes and goes between sessions. The evidence (2026-10-12, same bench, `ninfer_bench -n 512 -r 3`,
decode tok/s):

- **Same work.** In a node-level Nsight Systems trace each round runs the same 791 or 786
  kernels with the same grid, block and shared-memory configuration, the same one or two copies
  (1,312 or 1,316 bytes) and two memsets, and nothing is launched between rounds that the off run
  does not launch. Kernel time per round is +171 µs (+0.6%) and the gaps +11 µs. It sits in the
  weight-streaming linears and is not uniform: the MLP gate/up Q4 linear +1.1% (+85 µs per round),
  the 320-CTA Q5 linears +1.2% (+56 µs), the GDN q/k projection +2.1%, while the 12288x5120 Q5 linear
  is 0.7% faster.
- **Not the lookup.** Two diagnostic builds, both slower than off by the same amount as on (one
  session, ABBA order):
  - flag on with the per-round lookup skipped: 60.14 / 59.85;
  - flag off with the copy family configured anyway: 59.95 / 59.93;
  - against off at 60.65 / 60.28 and on at 60.43 / 60.13.
- **Placement inside the persistent arena.** In a 4x4 Latin square (each arm once per position).
  The first three of its 16 runs were taken right after another job and ran at 51.8-59.2. The
  figures below leave those three out and cover the other 13; the all-16 means are not given
  because the three samples' individual values and arms are not recorded here:
  - off 60.27;
  - on -0.8%;
  - copy family without the flag -0.9%;
  - **flag off with the persistent buffers moved by 18 MiB of padding, the copy family's
    growth, -1.0%.**
  - Moving whole allocations instead (64 or 170 MiB allocated before the model, or 64 MiB before
    every arena) leaves off at its speed and on still about 1% slower. That points to the
    buffers' offsets within the arena rather than their absolute addresses. The copy family's ReplaySSM records, pending features
    and verify logits are allocated ahead of the target's hidden states, so they move them.
  - These are ~1% differences over a handful of runs per arm. The padding run shows that moving
    the offsets can cost ~1%, not that the flag's cost is that move. Confirming it needs the padding bisected buffer by buffer in a
    session where the on/off gap is present, with the cost following one buffer's offset and
    vanishing when only that buffer is put back.
- **It comes and goes.** Later the same day, in two interleaved 4x4 squares (8 runs per arm), the
  unchanged build decoded 60.31 on against 60.28 off (+0.05%; ranges 60.18-60.55 and 60.13-60.65).
  The two-lane +1.6% did not recur either. Two interleaved squares (8 runs per arm) gave 86.75
  batch tok/s on against 86.56 off (+0.2%; ranges 86.52-87.10 and 85.81-87.10).

Rejected: allocating every buffer sized by the round width after the fixed-size ones, with the
target's hidden states first, so a copy family leaves the neural round's buffers where they are.
In the session where the unchanged build showed no difference it measured on +0.05% and off -0.08%
against the unchanged off (one 58.9 warm-up outlier dropped; the figures exclude it, and the arm it
belongs to is not recorded here, so they cannot be recomputed with it). Nothing could show it removed a cost that was absent there, and it moves
the off configuration's buffers too, so it is not merged. The one-lane launchers keep the flag on.
The effect is within the noise of a few runs. The padding result suggests it is not specific to copy
rounds and that any change to the persistent layout may move it; that is not shown beyond the one
padding and allocation experiments above.

Rounds now catch the draft context up at their own width: at two
lanes a configuration that never copies cost 43.68 ms per round at 32K against 43.50 off (three
alternating pairs) when the catch-up ran at the copy width, and costs 43.50 now.

**Memory.** At the Windows DFlash2 launcher profile (188,416 tokens, one lane) the planner reserves
170 MiB more with the flag on (5,616,887,552 against 5,438,487,040 bytes, the same as the one-lane
build), and it starts with 991 MiB of device memory free. At two lanes (65,536 tokens) it reserves
340 MiB more, of which about 100 MiB is actually used after startup (2.61 against 2.71 GiB free). The two-lane launcher start on the Windows desktop (0.9 GiB in use by the desktop) fits a 137,216-token shared KV with the flag against 156,672 without it (-12%).
The `tuned` launchers pass the flag with DFlash2 at one and two lanes, and keep it off at three or
more unless `NINFER_NGRAM=on` (see [launcher profiles](maintainer/launcher-profiles.md)).

### Choosing a KV format (RTX 3090, Qwen3.8-27B)

All seven SM86 KV formats, measured on Qwen3.8-27B.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.342517 | — | 31.93 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.342425 | −0.0021% | **33.63 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.344724 | +0.0508% | 30.34 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346413 | +0.0897% | 33.17 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347258 | +0.1092% | 28.90 tok/s |
| `nvfp4` | 18,432 | 36.00 MiB | 4.352201 | +0.2229% | 29.86 tok/s |
| `rk4v4` | **17,920** | **35.00 MiB** | 4.352432 | +0.214% vs `int8` | ≈ `rk8v4` |

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens. Decode is 128 timed steps on top of a 32,768-token prefill, no
speculation, mean of two independent runs; attention re-reads the whole cache each step, so a
format's cost only shows at depth. These are the September 2026 re-measurements after the fix for
a data race in the quantized attention kernels, which had corrupted every prefill output column
except the last and inflated the perplexity of `fp8`, `nvfp4` and `k8v4` (greedy generation was
unaffected). An earlier revision of this table carried the pre-fix scores, including `nvfp4` at
+0.36%.
The `rk4v4` row comes from a later session and build (2026-09-23), where `int8` measured 4.343155,
`rk8v4` 4.347943 and `nvfp4` 4.353589, and `rk4v4` decoded within ±1% of `rk8v4` at 4K-32K; its
figures are stated against those rather than this table's. See the README's
[`rk4v4` section](../README.md#lloyd-max-4-bit-keys-rk4v4).
See [the README](../README.md#choosing-a-kv-format) for the fuller writeup and recommendations.

**Not built: `rk2v4-e8` (2-bit E8 root keys).** The sibling RTX 4090 forks ship an `rk2v4-e8` mode:
keys as one E8 root index, a 4-bit log radius and a 4-bit axis correction per 8 dimensions, about
216 B per head and token here against `rk4v4`'s 280. Simulated on this build by passing `int8`'s
rotated keys through that fork's encoder and decoder (`absmax/7` G64 scale) and storing the decoded
keys as INT8, with values left at INT8, the same quick protocol measured **4.476942 against
4.343155 (+3.08%)**, with every domain between +2.2% and +3.9% (2026-09-29). That is the keys
alone, about fourteen times `rk4v4`'s whole +0.214%, and it agrees with the rule `rk4v4` found that
any 3-bit key coding costs +0.7% or more. `rk4v4` already reaches the model's native 262,144-token
context on the 27B, so the extra headroom does not pay for the quality.

### Context store read and write speed (RTX 3090 host, Windows 11, C: NVMe)

The [context store](serving.md#context-store) reads a stored conversation back into the Host tier
(hydration) when that is cheaper than prefilling it, priced with a read rate the server learns. The
drive and the store's own path were measured on 2026-10-13, on the shared box with other work idle
on the disk:

| What | Write | Read |
|---|---:|---:|
| Drive, unbuffered sequential, 32 MiB requests, 8 GiB file (two runs) | 1.84 / 0.92 GB/s | 5.56 / 5.47 GB/s |
| Store, 2 GiB image, one thread per image (before) | 0.98 GB/s | cold 0.45-0.48, warm 0.58-0.70 GB/s |
| Store, 2 GiB image, chunks read and verified on up to 8 threads (now) | 0.98-0.99 GB/s | cold 1.27-1.44, warm 1.67-2.27 GB/s |
| Server, read plus import into the Host tier, as learned by the cost model | | 0.53-1.15 GB/s |

The drive figures come from a small Win32 probe (`FILE_FLAG_NO_BUFFERING`, `WRITE_THROUGH`); the
second write run is slower, most likely the drive's write cache filling. The store figures are
`ninfer_context_store_bench put|load` (`bench/context_cost/context_store_bench.cpp`, a benchmarks
build): "cold" is a copy of the store made with unbuffered I/O (`xcopy /J`), so none of it is in the
file cache; "warm" reads the same directory again. A single reader was bound by hashing and page
faults at about a tenth of the drive's rate; reading the 32 MiB chunks in parallel tripled it. The
learned rate (`ninfer_context_store_read_bytes_per_second`) adds the import into pinned Host memory
and the server's other work, so it sits below the store's own.

With the read priced, a short conversation is read back when that is cheaper than prefilling it
instead of only when the store holds at least 4,096 more tokens than the cache. Restart probe on
the 27B (`tuned` DFlash2 profile at four lanes, 1 GiB Host tier, six one-turn conversations stored,
the server restarted on a fresh unbuffered copy of the store, then each conversation continued;
ABBA, two runs each):

| Conversation | Before (first token) | Now (first token) |
|---:|---:|---:|
| 4,097 tokens | 1.55 / 1.56 s, prefilled | 0.42 / 0.45 s, read back |
| 6,188 tokens | 1.25 / 1.40 s, read back | 0.94 / 0.94 s, read back |
| 16,076 tokens | 1.71 / 1.71 s, read back | 1.06 / 1.07 s, read back |

The other three (1.5K, 2.6K and 10K tokens) were restored at start-up in every run and are equal.

### Context store as a cache tier: write-on-evict and shared prefixes (RTX 3090, Qwen3.8-27B)

Hosted rotation, 2026-10-13: 24 tenants on three shared system prompts of about 2.5K tokens, each
visit appending a 1-3K-token tool result (conversations restart at 30K tokens), one client per
lane, 120 requests, greedy, thinking off, 64 output tokens, store on the C: NVMe, every arm from an
empty store. Same build with and without write-on-evict (ABCCBA with the shared-prefix build as C;
two runs each):

| C4, `tuned` DFlash2 flags, 2 GiB Host tier | hit rate | re-prefilled tokens | TTFT p50 / p95 / p99 | wall time |
|---|---:|---:|---|---:|
| store with idle writes only | 0.320 / 0.318 | 869K / 873K | 6.89 / 34.5 / 42.5 s ; 7.29 / 32.0 / 41.5 s | 479 / 463 s |
| + write-on-evict | 0.737 / 0.711 | 336K / 370K | 4.70 / 9.4 / 10.1 s ; 4.21 / 13.6 / 18.0 s | 235 / 248 s |
| + shared prefixes | 0.737 / 0.737 | 336K / 336K | 4.92 / 9.0 / 10.9 s ; 4.57 / 8.9 / 10.8 s | 242 / 237 s |


Write-on-evict is not free: copying each evicted conversation out (about 330 MB here) is a step of
the Engine worker, 37-38 s per C4 run of about 240 s, and the store wrote 530-560 GB per hour.
Hydrations (91-92 per run, none failed) waited about 1.1 s each for the read. Per-request prefill
and decode medians were unchanged (2,470-2,500 and 106-112 tok/s).

Warm start after a crash (each arm seeds its own store with 48 requests, is killed, restarts and
serves 24 new tenants on the same system prompts; ABBA): stored shared prefixes raised the hit
rate from 0.418 / 0.418 to 0.478 / 0.478 and lowered mean TTFT from 2.87 / 2.73 s to 2.09 / 2.39 s
(p95 6.26 / 6.26 s to 4.80 / 4.83 s). Writing them while busy cost nothing measurable (seed-phase
mean TTFT 3.16 / 3.16 s against 3.15 / 3.36 s).

#### Other Host sizes and eight lanes

Same workload and method, after the two write-on-evict fixes (write only when the release takes the
resume point; do not re-export a shared prefix the store holds). ABBA, two runs per build, `master`
(base) against this series; C8 uses the `mtp` launcher profile (the DFlash2 profile does not start at
eight lanes on this card). Each cell is base / series, first run then second run of each build:

| Configuration | hit rate | TTFT p50 (s) | TTFT p95 (s) | mean TTFT (s) | wall time (s) |
|---|---|---|---|---|---|
| C4, 2 GiB Host | 0.286 / 0.308 vs 0.724 / 0.705 | 6.73 / 7.20 vs 4.63 / 4.39 | 33.6 / 29.6 vs 10.1 / 12.4 | 11.8 / 10.6 vs 5.2 / 5.2 | 500 / 485 vs 248 / 251 |
| C4, 8 GiB Host | 0.710 / 0.701 vs 0.735 / 0.724 | 2.91 / 3.15 vs 3.42 / 3.29 | 8.9 / 10.0 vs 8.3 / 9.7 | 3.8 / 4.2 vs 4.0 / 4.2 | 197 / 208 vs 205 / 211 |
| C8, 2 GiB Host, MTP | 0.591 / 0.634 vs 0.733 / 0.733 | 6.10 / 5.11 vs 5.08 / 5.24 | 23.2 / 19.3 vs 13.4 / 14.0 | 8.4 / 7.3 vs 5.9 / 6.2 | 259 / 240 vs 224 / 217 |
| C8, 8 GiB Host, MTP | 0.737 vs 0.737 (all 4 runs) | 4.08 / 3.88 vs 4.26 / 4.49 | 22.7 / 23.4 vs 15.9 / 20.5 | 9.2 / 7.0 vs 6.3 / 7.1 | 272 / 209 vs 199 / 221 |

What this shows:

- The store tier pays when the Host tier is smaller than the working set (2 GiB at both lane counts):
  the hit rate rises by 0.14 to 0.42 and tail latency falls by half or more.
- When the Host tier already holds the working set (8 GiB), the hit rate is equal or 0.02 to 0.03
  higher, and TTFT and wall time are within run-to-run noise (the same base build ranged from 209 to
  272 s wall at C8). Write-on-evict is neutral there, not a gain, and the store still writes
  340-460 GB per hour.
- Per-request prefill and decode medians are unchanged in every configuration (C4 2,400-2,490 and
  98-107 tok/s; C8 2,220-2,385 and 63-74 tok/s).

Shared-prefix filter, C8 / 8 GiB / MTP, ABCCBA (A base, B series before the filter, C the final series;
the 8 GiB runs of the first two rows above are B). Before the filter the Engine wrote 77-83 "shared"
images per run, nearly all of them entries the cache publishes inside one conversation's history.
Only shared entries with repeat demand are now written:

| Build | shared writes | store writes | GB/h written | hit rate | mean TTFT (s) | wall time (s) |
|---|---:|---:|---:|---|---|---|
| A: base | 0 / 0 | 0 / 0 | 0 / 0 | 0.737 / 0.737 | 9.21 / 6.97 | 272 / 209 |
| B: before the filter | 77 / 77 | 115 / 112 | 382 / 333 | 0.737 / 0.737 | 6.30 / 7.12 | 199 / 221 |
| C: with the filter | 3 / 3 | 41 / 36 | 245 / 198 | 0.737 / 0.737 | 6.36 / 7.24 | 196 / 219 |

The filter removes about 40% of the bytes written without changing the hit rate or latency.

## Published coverage

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded measurements; a model/backend being supported does not
mean every workload or concurrency has a published measurement.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

Each cell links to the relevant result section. "Not published" describes measurement coverage,
not product support. C is configured request concurrency; K is the number of draft tokens.

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. Qwen3.8 measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |

These rates use different time boundaries. Server TTFT is an internal phase sum; external
streaming TTFT has its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. See the [35B termination and anomalies](performance/qwen3.6-35b-a3b.md#termination-and-anomalies)
and [Qwen3.8 completion outcomes](performance/qwen3.8-27b.md#completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
