# Performance model (no GPU needed)

[`tools/perf_model.py`](../../tools/perf_model.py) predicts how fast NInfer should run on the RTX 3090
from a model's shape and the recipe that quantises it. It is pure Python standard library, so it runs
on a Mac or any machine without CUDA, and it needs no artifact file.

Use it to:

- see **where a decoded token's bytes go** and what removing any one of them could buy, before
  deciding what to tune or which quantisation experiment is worth a rented card;
- **size a what-if** (`--format output_head=q4`, a different KV format, FP16 GDN state, speculation at
  a given acceptance) in seconds;
- turn a measurement into **the part bytes cannot explain** (`--tok-s`), which is the headroom left
  for kernel work.

It cannot replace the card. Bank conflicts, occupancy, register spills, how well memory and compute
overlap, and launch gaps are all invisible to it; they only show up as the gap between the model and a
run. When the model and a measurement disagree, the measurement is right and the disagreement is the
finding.

```bash
python tools/perf_model.py --model 27b --recipe qwen3_8_27b --depth 0 --depth 32768
python tools/perf_model.py --model 35b-a3b --recipe qwen3_6_35b_a3b --spec-k 3 --acceptance 0.6
python tools/perf_model.py --model 27b --recipe qwen3_8_27b --format output_head=q4
python tools/perf_model.py --config /path/to/config.json --recipe qwen3_8_27b --tok-s 49.0 --json
python -I tools/test_perf_model.py        # also registered as ctest ninfer_perf_model_test
```

## What it computes

| Part | Method | Trust |
|---|---|---|
| Decode bytes per token | Every dense matrix, one embedding row, 8 of 256 routed experts per layer, the GDN state in and out, and the KV cache to the current depth. Formats and bits per weight come from the recipe. | Exact arithmetic. Matches the artifact-header count of [`decode_byte_accounting.py`](../../tools/decode_byte_accounting.py) within 0.02%. |
| Decode speed | `bytes / 854.2 GB/s + layers x 0.0425 ms` | One fitted constant, two points. Consistent with the runs below, not proven for a third model. |
| Speculative decode | `tokens per round = 1 + acceptance x k`; round time = step at `k+1` columns x 1.15 | A composition of measured pieces. Acceptance is an input: no CPU model predicts it. MoE verify uses uniform expert routing, an upper bound on the experts read. |
| Prefill | `steps x fixed cost + 2 x active parameters x tokens / (316 INT8 TOPS x efficiency) + attention` | Two constants per family, fitted on two prompt lengths each. |

Acceptance is accepted drafts over drafted tokens, the definition in
[methodology](../performance/methodology.md).

## How well it reproduces what was measured

Everything here is an RTX 3090 row already in [`docs/performance.md`](../performance.md) or a number a
tool printed for a real artifact. "Fitted" points set a constant; "held out" points did not.

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

The Engine prices prefill with its own fitted table (`context_cost_defaults.cpp`, shown by
[`prefill_time_model.py`](../../tools/bench/prefill_time_model.py)). The model's 27B prefill agrees with
it within 2% from 512 to 131,072 tokens, and its fixed cost per step (103 ms) is close to the table's
107.8 ms; the two were derived by different routes.

## Reading the decode table

```text
  bytes read                          GB   share     ms  tok/s if free
  mlp.down                         3.431   22.5%   4.02           60.3
  output_head                      0.993    6.5%   1.16           51.5
```

`tok/s if free` is the throughput if that row cost no bytes at all: the most any optimisation of
it can buy. A row whose "free" number barely moves the total is not worth a kernel project.
`--tok-s` then prints how far the measurement sits above the bandwidth floor. On the 27B that is about
2.5 ms of a 20.4 ms token (88% of the floor), and on the 35B-A3B 1.8 ms of 4.9 ms (63%): those gaps are
launch and scheduling cost, and the MoE has far more of it relative to its bytes.

## Keeping it honest

- The recipes in `perf_model.py` mirror `tools/convert/official_recipes.py`. Change them together; the
  test fails if a recipe name disappears, but it cannot see a changed format.
- The calibration constants are in one dataclass, each with the measurement it came from. After a new
  campaign on the card, refit them and update the table above in the same change.
- KV format: `--kv int8` is 33/32 bytes per element (it reproduces the 10,560 B/token the 35B-A3B
  roofline run passed in). For `rk4v4` and other formats pass the exact value from `ninfer_bench`
  with `--kv-bytes-per-token`.
- Not modelled: norms and biases (<0.1% of bytes), vision, the MTP and DFlash weights, multi-lane
  batching, the draft head, and chunk sizes other than 4,096 (accepted, but extrapolated).
- Other cards: `--bandwidth-gb-s` and `--int8-tops` rescale the ceilings, but the fitted constants are
  the 3090's. Treat a prediction for another card as a ceiling, not a forecast.
