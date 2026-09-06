# Pipeline parallelism: plan of record

Goal, in the user's terms: run int8 KV with several large concurrent sessions, which a single
24 GB 3090 cannot do. This document is the design and the running status.

## Why a layer split rather than a tensor split

Measured, not assumed (see `multi-gpu-status.md` for the numbers and the hardware runs):

- A cross-GPU hop without NVLink costs ~13 us and is **latency-bound**, flat from 4 KiB to 40 KiB,
  which covers both models' per-token activation. Bandwidth only matters above ~320 KiB.
- Cost is therefore set by **crossings per token**. A layer split crosses **once**. A tensor split
  crosses twice per layer -- 128x more for the 27B.
- Peer access is not required at all: `cudaMemcpyPeer` stages through host memory, confirmed on a
  rented bridgeless 2x 3090.

**A layer split does not make one request faster.** Layers are sequential, so rank 1 is idle while
rank 0 works. Its win is capacity, and throughput only with concurrency >= 2 and micro-batching.
An NVLink bridge is close to irrelevant to it -- one 13 us hop per token against a ~25 ms decode
step is 0.05%. The bridge is worth something to a *tensor* split, which is the other, harder port
and the one that improves single-request latency.

## What it buys

35B-A3B, int8 KV at ~10 KiB/token:

| | single 3090 | two 3090s, layer split |
|---|---|---|
| weights resident | ~21 GB | ~10.5 GB per card |
| free for KV | ~2.6 GB | ~12.5 GB per card, ~25 GB total |
| total KV tokens | ~262k | **~2.5M** |
| e.g. concurrent sessions | 2 x 128k | **16 x 150k** |

That is the goal, and it comes from the memory split rather than from interconnect speed.

## Why this is cheaper than the earlier assessment feared

`multi-gpu-status.md` called the artifact layer "the piece that decides whether the rest is
tractable", because their tensor split needs **per-rank arenas inside one artifact** for row-split
weights. A layer split does not: every layer lives wholly on one device, so:

- `Binder` builds a `MaterializationPlan` from whatever objects the caller requests
  (`bindings.cpp:116` loops layers). Two binders filtered by rank produce two plans.
- `materialize(reader, plan, DeviceContext&, ...)` already takes a device. Call it once per rank
  under `ScopedDeviceRank`.
- `PagedKVCache(DeviceSpan backing, layout)` is parameterised by layer count, so one cache per
  rank on its own device backing.

The artifact layer needs **no redesign**. That is the difference that makes this worth doing now.

## The seam

`text_context_impl.h:997`:

```cpp
template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    for (int layer = 0; layer < kCfg.n_layers; ++layer) { ... }   // x is the residual stream
}
```

`x` is `{hidden, T}` in BF16 and is the only value crossing layers. The split is: run
`[0, boundary)` on rank 0, copy `x` to rank 1, run `[boundary, n_layers)` there.

## Work items

Each is verifiable on one GPU with rank count 1 as a no-op, except where noted.

1. **Layer/rank mapping.** A single type owning boundary choice and global-layer -> (rank, local
   index) translation. Everything else consumes it.
2. **Rank-partitioned bindings.** Filter the layer loop in both targets'  `bindings.cpp` by rank;
   embedding on rank 0, final norm + output head on rank 1.
3. **Per-rank materialization.** Two plans, two `materialize()` calls under `ScopedDeviceRank`.
4. **Per-rank workspace arenas.**
5. **Per-rank KV and GDN state pools**, with global->local layer remapping at the call sites.
6. **`run_layers` split** plus the boundary transfer, and the final hidden coming back for
   sampling.
7. **Per-device memory accounting** so `--kv-capacity auto` solves both cards.
8. **CUDA graphs.** Capture is per-device; expect to disable across the boundary initially and
   requalify, since the single-card baseline captures a whole MTP round.
9. **Validate on rented dual 3090** (needs two GPUs).

## Boundary choice

Not necessarily `n_layers / 2`. The two halves should be balanced by **resident bytes**, not layer
count, because MoE and attention layers differ in size, and because whatever is left over on each
card becomes KV. The mapping type should take the per-layer byte costs and solve for the split
that equalises free memory. Start with an even layer count, make it byte-aware once the plumbing
works.

## Status

- [x] 1. Layer/rank mapping -- `core/pipeline_split.h`, `PipelineSplit` + `RankOwnership`, tested
      in `tests/test_pipeline_split.cpp`.
- [~] 2. Rank-partitioned bindings -- done for **35B-A3B** only. `bind_artifact` takes a
      `RankOwnership` defaulting to the whole model, and binds unowned layers `ValidateOnly`.
      **27B is not done**: it has two separate layer-binding paths (`bind_groupwise_text_layers`
      and `bind_nvfp4_text_layers`), so it is left until the plumbing is proven on the 35B. Item 3
      must reject a pipeline split on the 27B loudly rather than let both ranks materialize the
      whole model.
- [ ] 3. Per-rank materialization -- two plans, two `materialize()` calls under `ScopedDeviceRank`,
      driven from `registry.cpp:105-137`, which is single-device throughout today.
- [ ] 4. Per-rank workspace
- [ ] 5. Per-rank KV / GDN state
- [ ] 6. run_layers split
- [ ] 7. Memory accounting -- `resolve_kv_capacity` and `current_free_device_bytes()` both assume
      one device; this is the item that actually delivers the KV capacity goal.
- [ ] 8. CUDA graphs
- [ ] 9. Hardware validation
