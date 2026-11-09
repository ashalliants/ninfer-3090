# Performance model (no GPU needed)

Three small Python tools, standard library only, so they run on a Mac or any machine without CUDA
and need no artifact file. They predict how fast NInfer should run on the RTX 3090, point at where the
time goes, and say whether a change is worth a rented card.

| Tool | Answers | Models |
|---|---|---|
| [`tools/perf_model.py`](../../tools/perf_model.py) | Where do a decoded token's bytes go, what is each what-if worth, how fast should this be? | Qwen3.5 Dense and MoE (27B, 35B-A3B, any `config.json`) |
| [`tools/kernel_bandwidth.py`](../../tools/kernel_bandwidth.py) | Which kernel families run well below the card's bandwidth, and what would fixing them save? | the same, joined to an `nsys` capture |
| [`tools/flash_next_perf_model.py`](../../tools/flash_next_perf_model.py) | What is the expert-cache hit rate worth, what should MTP give, do two lanes beat MTP? | Qwen3.8-Flash-Next (`qwen4exp`) |

They cannot replace the card. Bank conflicts, occupancy, register spills, memory/compute overlap and
launch gaps are invisible to them and only show up as the gap between a prediction and a run. **When
a model and a measurement disagree, the measurement is right and the disagreement is the finding.**
They also cannot say whether a kernel change helped: that is an A/B on the card
([`tools/bench/run_interleaved_ab.py`](../../tools/bench/run_interleaved_ab.py)). What they can say is
whether a measured gain is plausible against the ceiling.

```bash
python tools/perf_model.py --model 27b --recipe qwen3_8_27b --levers
python tools/perf_model.py --model 35b-a3b --recipe qwen3_6_35b_a3b --spec-k 3 --acceptance 0.6
python tools/perf_model.py --model 27b --recipe qwen3_8_27b --format output_head=q4 --tok-s 49.0
python tools/kernel_bandwidth.py profiles/sweeps/prof_27b-decode_cuda_gpu_kern_sum.csv --steps 128
python tools/flash_next_perf_model.py --hit-rate 0.972 --mtp-k 3 --lanes 2 --levers
python -I tools/test_perf_model.py          # and test_kernel_bandwidth.py, test_flash_next_perf_model.py
```

The three tests are ctest `ninfer_perf_model_test`, `ninfer_kernel_bandwidth_test` and
`ninfer_flash_next_perf_model_test`, all `cpu`.

## Finding what to tune

1. **Rank the levers.** `perf_model.py --levers` runs a fixed menu (head to q4, FP16 GDN state, MLP /
   experts / attention+GDN / every matrix to q4, and 25% / 50% / all of the per-layer kernel overhead
   removed) and ranks them by predicted plain-decode gain. Each row carries what the repository
   already knows: the q4 head and FP16 state cite their measured perplexity and speed in
   [quality trades](quality-trade-experiments.md); the rest say the quality cost is **not measured**.
   A `bytes` lever is arithmetic about bandwidth, never about quality. A `kernel` lever costs no quality
   and needs engineering.
2. **Size the gap.** Give `--tok-s` a measured plain-decode rate and it prints the milliseconds per
   token the bytes do not explain. On the 27B that is 2.5 ms of 20.4 ms; on the 35B-A3B 1.8 ms of 4.9 ms.
3. **Find the kernels.** Run `scripts/sweeps/decode-step-profile.sh 27b-decode` on the card, then
   `kernel_bandwidth.py` on its `..._cuda_gpu_kern_sum.csv`. It reports, per kernel family, ms per
   step, launches per step, GB/s, percent of the card's sustained read rate and the time that bringing
   the family to 90% would save. Kernel names carry the weight *format* (`q4_`, `q5_`...) but not the
   role, and fused ops read several roles, so the join is by format; routed experts, attention (against
   the KV bytes) and GDN recurrence (against the state traffic) are their own families. A family
   reading *faster* than the card can is flagged: it means a wrong `--steps`, model or recipe. A
   kernel fused over two stored formats (`q4_q5_...`) gets its own row with no bytes, and a printed
   caveat says the q4 and q5 rows then read high. Pass
   `--columns K+1` for a speculative round. **This tool has only been run on synthetic CSVs in nsys's
   layout, never on a real capture** (none is in the repository): the first real run may show kernels
   the rules put in `other`, and the biggest are printed so `classify` can be extended.

For Flash-Next, `flash_next_perf_model.py --levers` ranks cache, kernel and model levers for the mode
you name (plain, `--mtp-k`, `--lanes`).

## Qwen3.5 model: how it works and how well it matches

| Part | Method | Trust |
|---|---|---|
| Decode bytes per token | Every dense matrix, one embedding row, 8 of 256 routed experts per layer, GDN state in and out, KV to the current depth. Formats come from the recipe. | Exact arithmetic. Matches [`decode_byte_accounting.py`](../../tools/decode_byte_accounting.py)'s artifact-header count within 0.02%. |
| Decode speed | `bytes / 854.2 GB/s + layers x 0.0425 ms` | One fitted constant, two points. Consistent with the runs below, not proven for a third model. |
| Speculative decode | `tokens per round = 1 + acceptance x k`; round = step at `k+1` columns x 1.15 | Acceptance is an input. MoE verify uses the expected expert union under independent uniform routing: right for unrelated sequences, an overestimate for the columns of one sequence (Flash-Next measured 47% new experts per column, uniform predicts 98%). |
| Prefill | `steps x fixed cost + 2 x active parameters x tokens / (316 INT8 TOPS x efficiency) + attention` | Two constants per family, fitted on two prompt lengths each. |

Acceptance is accepted drafts over drafted tokens, the definition in
[methodology](../performance/methodology.md). "Fitted" points set a constant; "held out" did not.

| Quantity | Model | Measured | Status |
|---|---:|---:|---|
| 27B weight bytes per token (Q8 head) | 15.741 GB | 15.743 GB | independent |
| 35B-A3B weight bytes per token | 2.522 GB | 2.522 GB | independent |
| 35B-A3B routed experts, resident | 18.556 GB | 18.556 GB | independent |
| 27B plain decode | 48.6 tok/s | 49.0 | fitted |
| 35B-A3B plain decode | 208.3 tok/s | 204.5 | fitted |
| 27B prefill 512 / 16,384 | 2,057 / 2,892 tok/s | 2,057 / 2,893 | fitted |
| 27B prefill 4,096 | 3,146 tok/s | 3,143 | held out |
| 35B-A3B prefill 512 / 16,384 | 5,484 / 7,762 tok/s | 5,479 / 7,749 | fitted |
| 35B-A3B prefill 4,096 | 8,531 tok/s | 8,468 | held out |
| 27B MTP3 at 55.6% acceptance | 112.7 tok/s | 107.1 | held out (+5%) |
| 27B prefill, 131k prompt | 1,650 tok/s | 1,644 (Engine's fitted table) | independent |
| q4 head, plain decode | +1.8% | +1.4% to +2.9% | held out |

The Engine prices prefill with its own fitted table (`context_cost_defaults.cpp`, read by
[`prefill_time_model.py`](../../tools/bench/prefill_time_model.py)). The model's 27B prefill agrees with
it within 2% from 512 to 131,072 tokens, and its 103 ms fixed cost per step is close to the table's
107.8 ms; the two were derived by different routes.

### Reading the decode table

```text
  bytes read                          GB   share     ms  tok/s if free
  mlp.down                         3.431   22.5%   4.02           60.3
  output_head                      0.993    6.5%   1.16           51.5
```

`tok/s if free` is the throughput if that row cost no bytes at all: the most any optimisation of it
can buy. A row whose "free" number barely moves the total is not worth a kernel project.

## Flash-Next model

Flash-Next does not run at the bandwidth floor: its step is a chain of small latency-bound kernels plus
PCIe traffic when an expert is not in the VRAM cache. So the model is built from the measured pieces in
[flash-next-plan.md](flash-next-plan.md) and the lanes spike
([#284](https://github.com/ashalliants/ninfer-3090/pull/284)), not from bytes:

- **Anchor:** the 8K plain step, 12.5 ms at a 97.2% decode hit rate; verification of W columns costs
  17.0 / 20.0 / 23.8 ms at W = 2 / 3 / 4 (PR 13); a drafter step 0.88 ms.
- **A miss** is an expert frame (1,510,400 B) over the effective PCIe rate (24.2 GB/s): 62 us. The hit
  rate then moves the step by `misses x 62 us`, so **one point of hit rate is 4.8 misses per token,
  about 0.3 ms, about 2.4% of a plain step**.
- **Independent rows share almost no experts.** Two lanes touch 19.8 distinct experts per layer by the
  uniform union (the spike measured 19.6 from real traces); the columns of one MTP verification share
  more. Each lane also reads and writes its own recurrent state.

| Quantity | Model | Measured | Status |
|---|---:|---:|---|
| Plain decode, 8K (97.2% hit) | 80.0 tok/s | 79.9 | anchor |
| Plain decode, 1K (98.0%) / 32K (96.6%) | 81.6 / 78.9 | 81.2 / 79.0 | held out |
| Plain decode, cold sessions (93.3%) | 73.2 | 74.5 | held out |
| Distinct experts per layer at two lanes | 19.8 | 19.6 | held out |
| Two plain lanes, aggregate | 102.8 (warm), 91.9 (cold) | 95-104 | held out |
| MTP at 1 / 2 / 3 drafts, 8K | 101.8 / 115.8 / 111.6 | 102.1 / 112.0 / 111.9 | composition of PR 13's own pieces, not independent |
| MTP at 4 / 5 drafts | 108.1 / 101.4 | 96.0 / 90.0 | **12-13% fast**: verification cost per column rises |

**What it says about lanes (#284, #286, #291).** One MTP3 lane at the measured 8K acceptance (65%)
decodes at about 112 tok/s; two plain lanes at about 103. They tie at **57% acceptance for MTP3 (62%
for MTP2)**. Two lanes win decode throughput only where MTP accepts less than that, which fits the
measured 1.07x on the cold burst, where most of the gain was queued time to first token (PR 16), not
decode speed. The reason the lanes lose is
the cost of the second row: 4.2 ms of extra routed experts (unrelated streams share none) against 2.0
ms for an MTP column, which also yields only 0.65-0.76 of a token. The gate of 1.15x of one MTP lane
would need the second row to cost under about 3 ms; the model says it costs about 7.

Not modelled: Flash-Next prefill, vision, the context cache, 128K contexts, and any C above two (the
spike measured a hit-rate collapse there that this model does not reproduce).

## Keeping it honest

- The Qwen3.5 recipes mirror `tools/convert/official_recipes.py`. Change them together; the test
  fails if a recipe name disappears but cannot see a changed format. `--recipe` defaults to the
  model's family (`qwen3_8_27b` for dense, `qwen3_6_35b_a3b` for an MoE) and a recipe for the other
  family is refused, as `official_recipes.py` does.
- Numeric flags are validated at the command line (positive and finite where a zero would divide or
  silently fall back); `--bandwidth-gb-s` or `--int8-tops` mark the report as a custom card.
- Calibration constants sit in one dataclass per tool, each with the measurement it came from. After a
  new campaign refit them and update the tables above in the same change.
- KV format: `--kv int8` is 33/32 bytes per element (it reproduces the 10,560 B/token the 35B-A3B
  roofline run passed in). For `rk4v4` and others pass the exact value with `--kv-bytes-per-token`.
- Not modelled in `perf_model.py`: norms and biases (<0.1% of bytes), vision, MTP and DFlash weights,
  multi-lane batching, the draft head, chunk sizes other than 4,096 (accepted, extrapolated).
- Other cards: `--bandwidth-gb-s` and `--int8-tops` rescale the ceilings but the fitted constants are
  the 3090's. Treat a prediction for another card as a ceiling, not a forecast.
- `perf_model.py` is the shared base: `resolve`, `add_model_arguments`, `tokens_per_round` and
  `bytes_by_group` are what the other two tools import.
