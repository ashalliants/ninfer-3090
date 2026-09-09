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
