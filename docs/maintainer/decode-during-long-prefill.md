# Decode during a long prefill: what is built, what is not, and what decides the rest

Status: design note. Nothing here was measured; the host this was written on has no GPU. Numbers are
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
- **`--yield-long-prefill-ms N`**: with every lane busy, a short request waiting at the head of the
  queue pauses a long prefill (at least N ms left) at a step boundary instead of waiting out its
  minute. The paused prefill keeps its progress as a snapshot and resumes after.
- **`--prefill-anchor-tokens N`**: a long prompt publishes a shared checkpoint every N tokens, so a
  request cancelled part-way leaves its work for the retry.
- **`ninfer_prefill_backlog_seconds`**: predicted prefill work resident and queued, which is what a
  caller waits behind.

The first two are the "smaller chunks, or capping prefill's share per round" that the prefill/decode
split assessment (branch `docs/prefill-decode-split-assessment`, analysis only) asked to be tried before
any redesign.

## What is not built, and why

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

This is the option that would give streams short gaps while keeping a large step (so the dequantise
pass stays amortised): make a prefill step resumable at layer boundaries and run a decode round every few
layers. The gap becomes the step time divided by the number of slices, at the cost of the decode round
repeated each slice.

It needs more than it looks:

- The step's residual stream (`T x hidden`, 40 MB at 4096 on the 27B) and every per-step tensor in
  `work_` must survive the decode rounds in between, so decode needs its own workspace arena. The size
  is the decode phase's peak (`ordinary_round`/`mtp_round`/`dflash_round`, which `WorkspacePlan` already
  sizes), and it has to be found in device memory that the KV pool and context length currently use.
- `TextContext` holds step state in members that RAII scopes set and restore (`active_cache_positions_`,
  `active_rope_positions_`, `active_causal_attention_envelope_`) and that a decode round on the same
  object would overwrite. They would have to be saved per slice.
- The step's tail (final norm, head, sampling, capture splits, the MTP and DFlash feature taps that
  read layer outputs) runs after the layer loop and must see the same state it does now.
- GDN's state fork (`state_source`/`state_destination`) is settled after a step; a decode round between
  slices must not touch a lane that is mid-step, which the lane model already guarantees only because the
  two never interleave.

I would not start this without the measurements below, because stall budget plus decode share may already
be enough.

## What would decide it

All on the 3090, with a long prompt and streams that are decoding, using the two-lane profile
(`NINFER_SPEC=mtp`, the one with a second lane) and DFlash2 with `--max-concurrency 2`.
[`tools/bench/run_prefill_stall.py`](../../tools/bench/run_prefill_stall.py) runs each of them against a
live server (its README section gives the commands) and reports the stream's gaps in three windows, the
long prompt's time and rate, the short request's first-event time and the retry's time.

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
prompt's own speed cost? If a budget around 500 ms with a decode share reaches it, layer slicing is not
needed. If the only settings that reach it make the prompt 3x slower, layer slicing is the next option
and this note is its starting point.
