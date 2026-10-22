# Decode during a long prefill: what is built, what is not, and what decides the rest

Status: design note. It was written without a GPU. The options it describes are implemented in separate PRs (#340-#344 and #352); their GPU tests and measurements are recorded there and in the handoff note of #353, not in this note. Numbers are
from `docs/serving.md`, `docs/performance.md` and the source comments they cite, and say so. Delete or
fold into `docs/serving.md` once the open question below is settled.

## The problem

One worker cycle runs a decode round for every lane that is generating and then **one** prefill
step for one lane (`engine_core.h`, `worker_loop`). A prefill step is a whole `--prefill-chunk`. So a
stream that decodes next to a prefill gets one round per step, and its gap is the step's duration:

| `--prefill-chunk` | one step (27B, cuBLAS prefill, `w4_cublas_prefill.h`) | what a stream sees |
|---|---|---|
| 512 | ~265 ms | a token every ~0.3 s |
| 1024 | ~460 ms | |
| 2048 | ~775 ms | |
| 4096 | ~1,490 ms | 0.75 tok/s, 1.6 s gaps (previous engine, `docs/serving.md`) |

Steps get longer, not shorter, deep into a long prompt, because attention grows with the prefix. A
100,000-token prompt at chunk 4096 is about 25 steps and about a minute of this.

A step cannot simply be made small, because the cuBLAS route dequantises the whole weight to int8
once per step: 1.22x the integer route's speed at chunk 512, 1.59x at 4096. The question is how to give
streams short gaps without giving the long prompt up to that.

## What is built

Each is a flag, off by default, with its own counters:

- **`--prefill-stall-budget-ms N`**: while another lane decodes, a prefill step is cut so the cost
  model predicts it to take at most N ms (multiples of 128 tokens, never below 512). A lone prefill is
  untouched. This shortens the gap and slows the prompt.
- **`--prefill-decode-share P`**: after a prefill step, decode rounds run until they have used P percent
  of the GPU time of that step before the next one starts. This does not shorten the longest gap. It
  turns one round per step into a burst of rounds per step.
- **`--prefill-layer-slices N`**: while another lane decodes, a text prefill chunk runs as N slices of
  consecutive layers and the neighbour's decode round runs between slices, so its gap is a slice
  (about a step / N) while the chunk keeps its size. The option the next sections argue for (PR #352; see
  "Decode rounds between the layers of a prefill step" below).
- **`--yield-long-prefill-ms N`**: with every lane busy, a short request waiting at the head of the
  queue pauses a long prefill (at least N ms left) at a step boundary instead of waiting out its
  minute. The paused prefill keeps its progress as a snapshot and resumes after.
- **`--progress-anchor-tokens N --progress-anchors-shared`**: a long prompt publishes a shared checkpoint every N tokens, so a
  request cancelled part-way leaves its work for the retry.
- **`ninfer_prefill_backlog_seconds`**: predicted prefill work resident and queued, which is what a
  caller waits behind.

The first two are the "smaller chunks, or capping prefill's share per round" that the prefill/decode
split assessment (branch `docs/prefill-decode-split-assessment`, analysis only) asked to be tried before
any redesign.

## Options not taken, and the one built without a GPU

### Fusing the decode rows into the prefill step

The idea: run the decode lanes' tokens through the same layer pass as the prefill chunk, so a step
produces a token for every stream and the separate decode round disappears.

It would not shorten the gap. A fused step takes as long as the chunk it contains, and each stream
gets one token per step. That is the cadence of "one decode round per step" today. What fusion saves is
the decode round itself, which is tens of milliseconds (`docs/serving.md`), against a 1.5 s step: about
3% at chunk 4096 and 10-15% at the 300-500 ms steps the stall budget picks. The only way fusion shortens
the gap is by making the step small, which is what the stall budget does without the rewrite.

What the rewrite would cost, from reading the code:

- The ops differ per layer kind. Prefill attention is the single-sequence causal prompt kernel over
  `T` new tokens with its KV split schedule; decode attention is the paged-KV kernel over each lane's
  own length. Prefill GDN is a chunked scan over `T` tokens with a state source and destination slot;
  decode GDN is a per-token step that writes ReplaySSM records. A mixed pass needs both inside each
  layer: a second kernel launch per layer sharing the GEMMs, not a single fused kernel.
- The GEMMs are the part that would be shared. Decode rows add to `M`, which a compute-bound prefill
  GEMM absorbs (64 rows on a 4096-token step, eight lanes at DFlash2's eight rows each, is 1.6%). But under `--prefill-cublas` those rows would go through
  the int8-with-row-scale route that decode deliberately avoids (its exact group-64 path; the A8 decode
  route is a separate opt-in, `--mlp-a8-decode`). Decode numerics would change with it.
- Decode rounds are CUDA graphs with fixed shapes (`graphs.cpp`); a prefill step of varying `T` is not.
  A fused step would be eager, and every decode round during a prefill would lose its graph.
- Both phases run in the Program's one workspace (`WorkspacePlan`, `work_.reset()` per step), so they
  cannot be live together without a second arena.

### Overlapping a prefill step with a decode round on two streams

Rejected by measurement, not by argument: the split assessment (see above) records an earlier measurement on this
card (not kept in the repo) in which overlapping one prefill with one decode "gave no stall win and cost
11-32% of prefill". It would also need the second workspace above.

### Decode rounds between the layers of a prefill step

This is the option that shortens gaps while keeping a large step (so the dequantise pass stays
amortised), and it **is built** as `--prefill-layer-slices`. It was written without a GPU for this note; its tests and
measurements are recorded with the implementation (PR #352). How it works and what it rests on:

- A chunk runs as N Program calls. Between slices the residual stream (`T x hidden`, 40 MB at 4096 on
  the 27B) is parked in `prefill_hidden`, a persistent buffer only the chunk's tail writes, so no
  second arena is needed (a second arena was what I expected this to cost). A later slice rebuilds
  the workspace-resident inputs of the layers (positions, rope positions, the scoped control tensors)
  because a decode round reuses the workspace, and copies the residual back.
- The invariants it needs, each checked by reading and some by the fake-model Engine tests: only
  one lane mid-chunk (one parking buffer), so it goes first for prefill; no pause of a lane that is
  mid-chunk (its in-place GDN state is half advanced); replays and forced-token control batches, the
  other writers of `prefill_hidden`, wait for it; a refused yield is retried when the chunk ends.
- Limits: text-only chunks, no DFlash tap (its per-chunk state would have to cross the boundary), no
  pipeline split, so the default one-lane DFlash2 profile never slices and the two-lane MTP profile is the
  one it is for.
- **What it assumes, and where that is checked:** that nothing but the residual and the roots is live in
  the arena across layers. The checks belong to the implementation and are recorded there: the GPU test
  `sliced_prefill_matches_whole` (sliced hidden states, KV and state equal the whole-chunk prefill; a
  part-way abort leaks nothing; a lane part-way through a chunk cannot be paused), served-output equality,
  the fake-model scheduling tests for ordering and lifecycle, and the timings. This note does not repeat them.

Its cost is the decode round it adds at each boundary (tens of ms, while the neighbour lasts) and the
two residual copies per boundary (under a millisecond), against a prompt that runs at full chunk speed.

## What would decide it

All on the 3090, with a long prompt and streams that are decoding, using the two-lane profile
(`NINFER_SPEC=mtp`, the one with a second lane; the default one-lane DFlash2 profile never slices).
[`tools/bench/run_prefill_stall.py`](../../tools/bench/run_prefill_stall.py) runs each of them against a
live server (its README section gives the commands) and reports the stream's gaps in three windows, the
long prompt's time and rate, the short request's first-event time and the retry's time.

0. **First, correctness of the sliced path:** `pwsh tools/run-tests.ps1 -GpuOnly -Regex native_transactions`.
   It must pass before any number from `--prefill-layer-slices` means anything.
1. **Baseline.** A 100,000-token uncached prompt submitted while one stream decodes. Record the stream's
   inter-token latency (p50, p99, max) and tokens per second during the prompt, and the prompt's total
   prefill time. Take it at `--prefill-chunk` 1024, 2048 and 4096.
2. **Stall budget.** The same with `--prefill-stall-budget-ms` at 250, 500 and 1000 (chunk 4096). Read
   `ninfer_paced_prefill_steps_total` to confirm it bound.
3. **Decode share.** The same with `--prefill-decode-share` at 25 and 50, alone and with a budget of 500.
4. **Yield.** A 100,000-token prompt prefilling at one lane, a 200-token request arriving 10 s in, at
   `--max-concurrency 1` and 2. Record the short request's time to first token with and without
   `--yield-long-prefill-ms`, and the long prompt's finish time.
5. **Anchors.** Cancel a 100,000-token prompt at 60% and resubmit it. Record the retry's prefill time.

The acceptance questions are: does a stream reach an acceptable inter-token latency while a 100,000-token
prompt prefills (the assessment's symptom was 4-6 tok/s against ~55 uncontended), and what did the
prompt's own speed cost? Compare them with `--prefill-layer-slices` (2, 4 and 8, chunk 4096): the claim is a gap of about a step
divided by N at full prompt speed. If a budget around 500 ms with a decode share already reaches the gap
you need, slicing is not needed; if the only settings that reach it make the prompt 3x slower, slicing
is the option that should.
