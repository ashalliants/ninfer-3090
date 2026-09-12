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

**Speculative decoding is not bit-identical to non-speculative decoding here, and that is not
new.** On the text prompt, DFlash2 and MTP produce the *same* output as each other and both
differ from the width-1 greedy path a hundred tokens in (`数学上是未定义的` vs `不确定的`).
Verification evaluates k+1 columns in one pass while plain decode evaluates one, so the
reductions run in a different order and a near-tie argmax can flip. MTP shows it identically,
so it is a property of the shared verification path rather than anything DFlash2 introduced.

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

### Choosing a KV format (RTX 3090, Qwen3.8-27B)

All six SM86 KV formats, measured on Qwen3.8-27B.

| KV profile | Bytes/token | KV at 2,048 tokens | Perplexity | vs `bf16` | Decode at 32K depth |
|---|---:|---:|---:|---:|---:|
| `bf16` | 65,536 | 128.00 MiB | 4.343225 | — | 32.50 tok/s |
| `int8` | 33,792 | 66.00 MiB | 4.343263 | +0.0009% | **33.86 tok/s** |
| `fp8` | 33,024 | 64.50 MiB | 4.347181 | +0.0911% | 30.13 tok/s |
| `rk8v4` | 26,112 | 51.00 MiB | 4.346811 | +0.0826% | 33.54 tok/s |
| `k8v4` | 25,728 | 50.25 MiB | 4.347596 | +0.1006% | 28.61 tok/s |
| `nvfp4` | **18,432** | **36.00 MiB** | 4.358924 | +0.3615% | 29.62 tok/s |

Perplexity is `ninfer-perplexity` on the fixed `ninfer-ppl-1m-v1` corpus, `--quick`, context/stride
4096/2048, 261,167 scored tokens. Decode is 128 timed steps on top of a 32,768-token prefill, no
speculation; attention re-reads the whole cache each step, so a format's cost only shows at depth.
See [the README](../README.md#choosing-a-kv-format) for the fuller writeup and recommendations.

## Published coverage

Published measurements use one NVIDIA GeForce RTX 5090 through NInfer's public HTTP serving route.
Choose a model below for its detailed results, run conditions, output limitations, and reproduction
commands. These are recorded historical measurements; a model/backend being supported does not
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
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | Not published |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8 and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. The Qwen3.8 NVFP4 saturation reports retain configuration and
values but no tested Git revision; the model page records that provenance limitation.

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
and [Qwen3.8 DFlash2 outcomes](performance/qwen3.8-27b.md#dflash2-completion-outcomes).

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](README.md#model-artifacts), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
