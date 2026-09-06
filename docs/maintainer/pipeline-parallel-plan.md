# Multi-GPU expert offload

Goal, in the user's terms: run int8 KV with several large concurrent sessions, which a single 24 GB
3090 cannot do. This is the design, the reasoning behind it, and the status.

## What it does

With `--devices 0,1`, the expert / MLP block of each layer is materialized on the second GPU.
Rank 0 keeps everything else -- embeddings, attention, GDN, norms, the head -- and therefore keeps
the KV cache, the GDN recurrent state, round state and the whole context-cache machinery. During a
forward pass the residual stream crosses to rank 1 for each layer's mlp tail and comes straight
back.

Every byte rank 0 sheds becomes KV, which is the entire point.

## Why the expert block, and not whole layers

The first plan was a layer split: layers `[0,k)` on one card, `[k,N)` on the other. Reading the
execution path showed that is the expensive way to get what we want.

KV lives with attention, so the question is not "how do we divide the model" but "how much can we
get **off** the card that serves attention". Three things are pinned to rank 0 regardless:

- **attention** needs its KV cache,
- **GDN** needs its recurrent state,
- **the head** writes round state and the persistent prefill-hidden buffer, which callers of
  `run_layers` reach directly (`rmsnorm` into `prefill_hidden_`, then `io_.logits` / `io_.token`).

Moving whole layers drags KV pools, GDN state pools, decoder state, state images and the host
offload / context cache along with them: 33 `decoder->` sites, 43 `state_images->` sites and ~198
KV-paging sites, plus the planner.

The expert block is pinned to nothing, and it is **~88% of the weights** -- 17.3 GiB of 19.6 on the
35B-A3B (465 MB per layer x 40), and a similar share of the 27B's dense MLP. Offloading only that
leaves every one of those subsystems untouched.

| | KV room on rank 0 | tokens at ~10 KiB | surface |
|---|---|---|---|
| single card today | ~3.4 GiB | ~262k | -- |
| offload 20 layers' experts | ~12.6 GiB | ~1.3M | small |
| **offload all experts (default)** | **~21.2 GiB** | **~2.2M** | small |
| full layer split | ~26 GiB | ~2.7M | 274+ sites |

83% of the benefit for a fraction of the risk.

## What it is not

**It does not make anything faster.** The two cards work in sequence, not in parallel: rank 0 idles
during each mlp tail and rank 1 idles the rest of the time. Total weight bytes read per token are
unchanged, so decode speed is roughly what one card gives, minus the crossing overhead. The win is
capacity.

**NVLink is nearly irrelevant to it.** A staged hop costs ~13 us at activation sizes (measured;
latency-bound, flat from 4 KiB to 40 KiB), against a decode step of tens of milliseconds. A bridge
would save a few microseconds per crossing. Bandwidth only starts to matter above ~320 KiB, i.e.
at prefill batch sizes, where the crossings are larger and the cost is worth measuring separately.

Peer access is not required: `cudaMemcpyPeerAsync` stages through host memory when it is
unavailable, confirmed on a rented bridgeless 2x 3090.

## How it works

- **`PipelineSplit` / `RankOwnership`** (`core/pipeline_split.h`) answer which rank holds a layer's
  expert block, and pin everything else to rank 0. A single-rank split is the identity mapping,
  which keeps every consumer a no-op on one GPU. `experts_on_last` is the default; an empty rank 0
  is the point of it, not a bug.
- **`DeviceArena` carries one backing per rank** and switches between them, so one workspace serves
  both cards and all 76 existing `work_` call sites are unchanged. `Scope` records the rank it was
  taken on -- a scope opened on one rank legitimately outlives a switch to another, and restoring a
  foreign bump pointer would hand out overlapping scratch. `reset()` clears every rank, since it
  marks a round boundary.
- **`Program`** allocates a workspace per rank, each while its own device is current.
- **`run_mlp_tail`** crosses the residual stream to the card holding the experts, runs the whole
  tail there, and copies the result back into the caller's rank-0 buffer.
  `post_attention_norm` is bound *with* the expert block so the offloaded card does rmsnorm and the
  expert matmuls back to back -- one crossing out and one back, rather than two of each.
- **Ordering** uses per-rank streams and fences, never a host sync: record on the producer, wait on
  the consumer, issue the copy on the consumer's stream.
- **Loading** binds every rank against the same artifact, with the tensors a rank does not own
  placed `ValidateOnly`: the file is validated in full on each rank while only that rank's bytes
  are uploaded. `materialize()` already takes a `DeviceContext`, so each rank runs under
  `ScopedDeviceRank`.

## The trap

The residual stream is workspace memory and each rank has its own workspace, so a crossing is
between two arenas. The offloaded rank must run against *its* buffer, and the result must land back
in the caller's. Getting that wrong yields a program that runs, reads another device's memory
through a stale pointer, and emits plausible rubbish rather than crashing.

That is why the hardware test compares **greedy text against the single-GPU reference** rather than
checking that it starts. Clean startup and sensible memory figures are both compatible with a build
that is silently reading the wrong device.

## Status

- [x] Layer/rank mapping, with tests (`tests/test_pipeline_split.cpp`).
- [x] Per-rank arena switching, with tests (`tests/test_arena_ranks.cpp`).
- [x] Rank-partitioned bindings and per-rank materialization, both targets.
- [x] Per-rank workspaces.
- [x] Cross-device mlp tail execution.
- [x] **27B**, all three of its layer-binding paths (groupwise, NVFP4, and the Qwen3.8 NVFP4/FP8
      mix).
- [x] `--devices N,M` on the CLI, so the split can be checked by output rather than by inspection.
- [ ] Hardware equivalence run: greedy text from the split against the single-GPU reference.
- [ ] KV capacity measured with `--kv-capacity auto` in both modes.
- [ ] CUDA graphs across the boundary. Capture is per-device, so a cross-device schedule cannot be
      one graph. Whether decode capture survives the offload, and what it costs if not, is
      unmeasured.
- [ ] Concurrency and context-cache behaviour under the split. These live entirely on rank 0 so
      they should be unaffected, but "should be" is not "measured".

## Earlier hardware runs

A dual 3090 box (bridgeless, `nvidia-smi topo -m` reporting `PHB`) confirmed the pieces this is
built on:

```
peer 0->1=0
enable_peer 0->1=peer access is not supported between these two devices
memcpy_peer_1MiB=OK
device_context=OK size=2 model_parallel=1 peer_access=0
```

Peer access is unavailable on a consumer pair and irrelevant: the copy works anyway. `DeviceContext`
constructs across both cards with distinct per-rank streams and fences.
