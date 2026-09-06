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

## What it buys -- measured on hardware, not projected

Run on a rented 2x RTX 3090 box (2026-09-06), `--devices 0,1`, int8 KV:

```
loading weights | 9.82 GiB   ->  weights ready | 1.9s | 5.15 GiB/s
loading weights | 9.78 GiB   ->  weights ready | 1.9s | 5.18 GiB/s

rank 0 (cuda device 0): layers [0,20), weights 10052 MiB, free 13805 MiB
rank 1 (cuda device 1): layers [20,40), weights 10011 MiB, free 13847 MiB
```

Single-device control on the same box: one card loads all 19.6 GiB and serves, leaving 3.37 GiB.

| | single 3090 | two 3090s, layer split |
|---|---|---|
| weights resident | 19.6 GiB | **9.82 + 9.78 GiB** |
| free after weights | ~3.4 GiB | **13.5 + 13.5 = 27.0 GiB** |
| KV tokens at ~10 KiB/token | ~262k | **~2.8M** |

That is roughly **10x the KV**, and it comes from the memory split rather than from interconnect
speed. The two halves came out within 41 MiB of each other, so the even layer split really is
byte-balanced for this model -- as expected, since each rank gets five of the ten full-attention
layers (layers 3,7,11,15,19 against 23,27,31,35,39).

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
- [ ] 8. CUDA graphs. **This is the item with a genuine unknown.** A CUDA graph is captured on one
      device's stream, so a cross-device schedule cannot be one graph. Our decode path captures a
      whole MTP round; their fork disabled capture for the cross-device schedule and never
      quantified the loss. Either the round splits into a per-rank graph either side of the
      boundary, or capture is off for split decode and the cost has to be measured before anyone
      calls this a win.
- [x] 9. Hardware validation **of the load half**. Confirmed on a rented 2x RTX 3090 (numbers
      above): both ranks materialize on their own device, the layer split is byte-balanced to
      within 41 MiB, and the combined KV headroom is 27.0 GiB against 3.4 GiB on one card. The
      single-device control on the same box still loads 19.6 GiB and serves normally. Execution
      remains unvalidated because it does not exist yet.

### Measured surface of item 6

Counted rather than guessed, since it decides whether this is a day or a week:

| site | count |
|---|---|
| `ctx_.stream` in `text_context_impl.h` | 18 |
| `work_` in `text_context_impl.h` | 76 |
| stream references in `program_impl.h` | 58 |

`TextContext` is built on the stack per schedule from `state.execution.{device, model, work}`
(`decode_impl.h:24`, `mtp_impl.h:25`, `mtp_impl.h:85`, `dflash_impl.h:377`), all of which are
single-device today. Item 6 is therefore the large one, and items 4 and 5 are its prerequisites.

## Implementation guide for items 4-6

Written after reading the execution path, so the remaining work is mechanical rather than
exploratory. The ordering matters: 4 and 5 are prerequisites of 6.

### The resources that must become per-rank

`Program` (`program.h:650-670`) holds a single set of device-resident members. For a split, these
need one instance per rank:

| member | why |
|---|---|
| `workspace_storage` (`DeviceArena`) | scratch must live on the device running the layer |
| `work` (`WorkspaceArena`) | same |
| `decoder` (`DecoderState`, holds `PagedKVCache`) | KV for a layer must sit with that layer |
| `state_images` / `host_state_images` | GDN recurrent state is per layer |
| `kv_arena`, `text_kv_pages`, `text_kv_addresses` | KV paging is per device |

`PagedKVCache(DeviceSpan backing, layout)` already takes a layer count, so each rank constructs one
sized to `split.rank_layers(rank)`. Every call site indexing KV by layer must switch from the
global layer index to `split.placement(layer).local` -- that reindexing is the single most likely
source of silent corruption in this whole change, and is why `PipelineSplit` returns both.

### Threading the rank through execution

`ExecutionCore` (`schedule.h:31`) is the natural carrier. It currently holds `DeviceContext&`,
`WorkspaceArena&` and one `LoadedModelData&`; it should hold the per-rank workspace vector and the
split, with a helper returning the right workspace and stream for a layer.

`TextContext` is built on the stack per schedule (`decode_impl.h:24`, `mtp_impl.h:25`,
`mtp_impl.h:85`, `dflash_impl.h:377`), which is convenient: it means the rank can be chosen at
construction rather than threaded through every method. The 18 `ctx_.stream` uses inside
`text_context_impl.h` become the active rank's stream, and `run_layers` gains, at each boundary
`split.crosses_after(layer)`:

1. record an event on the producing rank's stream,
2. make the consuming rank's stream wait on it,
3. `cudaMemcpyPeerAsync` the residual stream `x` (hidden x T, BF16) to the consuming rank's buffer,
4. continue with the consuming rank's workspace and weights.

`cudaMemcpyPeerAsync` is correct with or without peer access -- it stages through host memory when
peer access is unavailable, confirmed on hardware.

### The trap

`x` is a workspace tensor. Each rank has its own workspace, so the destination is *not* the same
allocation -- the copy is between two arenas, and the consuming rank's `run_layers` must continue
from its own buffer rather than the pointer it was handed. Getting this wrong yields a program
that runs, reads another device's memory through a stale pointer, and produces plausible-looking
rubbish rather than crashing.

## A useful milestone short of execution

Items 1-3 plus 7 give a build that **loads the model split across two cards and reports the KV
headroom on each**. That demonstrates the capacity claim -- the entire point of the exercise --
and is verifiable on rented hardware before any of the execution rewrite exists. Worth reaching
and validating first, precisely because it de-risks the expensive part.
