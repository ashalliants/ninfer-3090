# Multi-GPU: what exists, what does not, and what it would take

Status as of this branch. Written after evaluating
[devon-caron/ninfer-dual-3090-nvlink](https://github.com/devon-caron/ninfer-dual-3090-nvlink),
which implements a working dual-3090 graph split on top of an older NInfer base.

**Nothing in this branch executes a model across two GPUs.** What it adds is the device layer that
any multi-GPU execution needs, plus a per-device correctness fix that was a latent bug in the
single-GPU tree. Both halves are verifiable on one card and are verified below.

## What their fork does

Two commits on a v0.6.0-era base, authored through an agent (`add codex changes to support dual
gpu graph mode`). The approach is **partial tensor parallelism**, not pipeline parallelism:

- Every main-model MLP has its packed Q4 gate/up and Q5 down weights physically split across the
  two cards, 8,192 intermediate channels on the primary and 9,216 on the secondary. The primary
  sends the normalized hidden activation over NVLink, the secondary returns one hidden-size
  partial, and the primary adds both to the residual.
- Attention projections, embeddings, norms and the output head stay on the primary. There is no
  full model replica on the secondary.
- The paged KV cache is split **by attention layer**: the primary owns full-attention layers 0-4,
  the secondary owns 5-15 plus the optional MTP KV. Q/K/V and block-table selectors cross the link
  per layer, and the attention result comes back.
- CUDA Graph capture is disabled for the cross-device schedule.
- Restricted to the dense 27B groupwise Q4/Q5 target. The 35B-A3B and NVFP4 paths are excluded.

Their reported numbers, on their machine: prefill +41.3% at 512 tokens and +41.1% at 2,048;
decode +17.5% at 256 tokens; C8 HTTP aggregate 127.05 -> 145.97 tok/s (+14.9%).

## Assessment

The design is reasonable and the documentation is unusually careful — it states its limits, warns
that CUDA indices do not match `nvidia-smi` ordering, and separates pre- and post-sharding
measurements instead of quietly mixing them. Three things are worth flagging before anyone treats
the numbers as a target:

**The decode number is the one to scrutinise.** Decode is memory-bandwidth-bound, and splitting
MLP weights halves the per-card weight read for the largest tensor in the model. If the split were
free, decode should gain far more than 17.5%. It does not, which says per-layer synchronisation
over the link is eating most of the win — two crossings per split MLP per token, serialised, at
batch 1. That is a structural cost of this design at low batch, not a tuning miss.

**Disabling CUDA Graphs is a real and unquantified cost.** Our decode path captures a whole MTP
round at load time. Their README says graph capture is off for the cross-device schedule but does
not say whether the single-card baseline in their table kept graphs on. If it did, the +17.5% is a
net of two opposing effects and the sharding gain is larger than it looks while the graph loss is
hidden inside it. If it did not, the baseline is understated. Either way the comparison needs
restating before it can guide a decision.

**It targets the model we do not recommend.** Our tuned launchers default to Qwen3.6-35B-A3B; the
graph split explicitly excludes it and the NVFP4 path. Applied to this fork as-is, it would do
nothing for the configuration we tell people to run.

## Why this branch does not contain their execution path

Their base and ours diverged by 280 commits against their 2. The overlap is not incidental:

| file | our change since the common base | theirs |
|---|---|---|
| `program_impl.h` | **+10,996 / -756** (file is now 12,432 lines) | +307 / -83 |
| `text_context_impl.h` | +180 / -128 | +240 / -25 |
| `materializer.cpp` | +110 / -30 | +322 / -13 |
| `binder.cpp` | +58 / -3 | +105 / -1 |

A trial merge produced 38 conflicted files and roughly 90 conflict hunks, 26 of them inside
`program_impl.h` alone. The conflicts are not textual. Two examples:

- Their materializer replaces `storage_data(handle)` with `device_data(handle, rank)` throughout.
  Our fork has since built an eviction-pool and pinned-host layer on that same API for vision
  overlay residency. The merged API needs per-rank arenas *and* the eviction machinery, which is a
  redesign of the artifact layer rather than a conflict resolution.
- Their `program_impl.h` changes target an execution path that our 130-commit upstream catch-up has
  substantially rewritten — MTP, dflash, NVFP4/K8V4/FP8 KV, vision overlay.

Resolving that faithfully is a multi-day port, and **none of it is verifiable on a single-GPU
machine**. Shipping an unverifiable cross-device execution path that merely compiles would be worse
than shipping nothing: it would look finished. So this branch stops at the layer that can be
proven correct here, and the port is left as a scoped piece of work below.

## What this branch does contain

### 1. Per-device kernel attribute configuration (a real bug, fixed)

`cudaFuncSetAttribute` is scoped to the current device context. Nineteen call sites across our
kernels cached its result in a function-local `static`:

```cpp
static const cudaError_t attr = cudaFuncSetAttribute(kernel, ..., kDynamicBytes);
CUDA_CHECK(attr);
```

The first device to reach that line records the result; every later device skips the call. On one
GPU this is correct. On two, the second device launches a kernel whose dynamic shared-memory
ceiling was never raised — a launch failure, or worse, silently wrong output. All nineteen now use
`configure_cuda_device_once`, a device-keyed cache. On a single GPU the map holds exactly one entry
and behaviour is unchanged, which is verified below.

This fix stands on its own merits regardless of whether the execution path is ever ported.

### 2. Multi-endpoint `DeviceContext`

Adapted from their design, reconciled with ours. `DeviceContext` now holds one or two endpoints,
each with its own compute stream, transfer stream, vision stream and fence, with `device`/`stream`
/`props` as aliases for the active rank. `activate_rank` and the `ScopedDeviceRank` guard switch
between them without leaving a thread bound to the wrong device.

Construction validates, **before any weight is uploaded**: device ids exist, are distinct, number
one or two, report matching compute capability, and have bidirectional peer access. Peer access is
enabled on both directions.

Our `vision_stream` (absent from their base) is carried per endpoint, and our `transfer_stream`
naming is kept over their `load_stream` — 92 call sites against 9.

### 3. `--devices N,M`

On `ninfer-serve`, mutually exclusive with `--device`, plumbed through `EngineOptions`. Single-entry
lists are accepted and are equivalent to `--device`.

### 4. `ninfer-multi-gpu-probe`

Their standalone enumeration probe, now a build target at `build-*/tools/`. It links nothing from
NInfer on purpose: it answers "which CUDA indices are the cards, and can they reach each other"
before the loader is involved. A display GPU commonly takes `cuda=0` even when `nvidia-smi` lists
it elsewhere, so `--devices` indices must be read from a CUDA-level enumeration.

## Verification performed

On one RTX 3090, which is what was available:

- `ctest` 110/110.
- Generated text is **byte-identical** to the published v0.8.1 release binary across three prompts
  at temperature 0 with a fixed seed — the sharpest available check that the kernel-attribute
  change is a no-op on one device.
- `--devices 0,1` on this box fails before loading with
  `CUDA device 1 does not exist: 1 device is visible`.
- `--device 0 --devices 0,1` is rejected as mutually exclusive.
- `--devices` rejects three entries, duplicates, empty strings and trailing commas.
- `--devices 0` starts and serves normally, matching `--device 0`.
- The probe reports this machine's single card and its capability correctly.

**Not verified, because it cannot be here:** anything involving two devices. The peer-access and
matching-capability branches have never executed. On a real two-GPU box the first thing to check is
that `DeviceContext` construction succeeds and that `--devices` still produces correct single-rank
output, since execution remains entirely on rank 0.

## What a port would take

Roughly in dependency order, and each step needs two GPUs to validate:

1. **Artifact layer.** Reconcile per-rank device arenas with the eviction-pool and pinned-host
   layer. `device_data(handle, rank)` alongside `storage_data(handle)`, `device_arena(rank)`, and
   `shard_row_split_across_devices` for the row-split weights. This is the piece that decides
   whether the rest is tractable.
2. **Memory accounting.** Per-device reservation curves so `--kv-capacity auto` can solve both
   cards, plus the secondary fields in `MemorySummary`.
3. **MLP shard kernels.** Their Q4 swiglu and Q5 linear-add shard variants.
4. **Execution schedule.** The cross-device MLP dispatch and reduction in the current
   `program_impl.h`, which is where their 307-line change no longer applies.
5. **KV by layer**, if the layer split is wanted; it is separable from the MLP split and was a
   later commit for them too.
6. **Requalify decode with CUDA Graphs on the single-card baseline**, so the comparison means
   something.

A cheaper alternative worth considering first: for a 24 GB card pair, running **two independent
single-GPU engines behind one router** gives near-linear aggregate throughput for concurrent
requests with none of this complexity. It does not help single-request latency or let one model
exceed one card's memory, which is what the graph split buys. Which of those matters should decide
whether the port is worth doing at all.

## Does any of this need NVLink?

No. **Confirmed on real hardware**, not inferred: a rented 2x RTX 3090 box (vast.ai instance
50039003, `nvidia-smi topo -m` reporting `PHB` between the cards, i.e. PCIe host bridge, no
bridge) runs the probe as:

```
devices=2
cuda=0 bus=0000:04:00.0 name=NVIDIA GeForce RTX 3090 sm=86 vram_mib=24124
cuda=1 bus=0000:09:00.0 name=NVIDIA GeForce RTX 3090 sm=86 vram_mib=24124
peer 0->1=0
peer 1->0=0
enable_peer 0->1=peer access is not supported between these two devices
memcpy_peer_1MiB=OK
```

`cudaDeviceCanAccessPeer` returns 0 and `cudaDeviceEnablePeerAccess` fails outright -- **and
`cudaMemcpyPeer` succeeds anyway**, staging through host memory. That last line is the whole
finding: the capability query and the ability to move data are different questions, and the code
below was conflating them.

The assumption that NVLink was required was wrong in two places in this tree -- both now fixed.

`cudaDeviceCanAccessPeer` returning 0 does **not** mean the cards cannot exchange data.
`cudaMemcpyPeer` works between any two devices; without peer access CUDA stages through host
memory. NVIDIA disables P2P over PCIe on GeForce, so every 2x RTX 3090 without a physical bridge
reports 0 -- and of 19 rentable 2x 3090 offers surveyed on vast.ai on 2026-09-06, **none** had a
bridge. Treating peer access as mandatory therefore excluded the entire consumer dual-GPU market
for no reason. `DeviceContext` now records it as a capability (`peer_access()`) and starts anyway.

### What the fallback costs

Measured on this fork's reference 3090, PCIe 4.0 x16, staged round trip (D2H + H2D), after
forcing the link out of its idle gen1 state:

| transfer | staged | note |
|---|---|---|
| 4 KiB | 13.6 us | 35B-A3B activation/token (hidden 2048, bf16) |
| 10 KiB | 13.1 us | 27B activation/token (hidden 5120, bf16) |
| 40 KiB | 13.4 us | |
| 5 MiB | 409 us | 12.8 GB/s |
| 64 MiB | 5.12 ms | 13.1 GB/s |

The hop is **latency-bound, not bandwidth-bound**, at every size an activation actually is: flat
~13 us from 4 KiB to 40 KiB. Bandwidth only starts to matter above ~320 KiB, i.e. at prefill batch
sizes. So the cost of a schedule is set by **crossings per token**, not by how wide the link is:

| scheme | crossings/token (27B, 64 layers) | link cost/token | share of a ~45 ms decode step |
|---|---|---|---|
| pipeline (split by layer) | 1 | ~13 us | 0.03% |
| tensor split (their design) | 128 | ~1.7 ms | ~4% |

Two caveats on those figures. The 13 us is a single-stream same-device D2H+H2D and omits
cross-device event synchronisation, so a real staged hop is likely 20-40 us; and NVLink would cost
a couple of us, so the *delta* a bridge buys a tensor split is on the order of 3-9% of a decode
step. Their measured decode gain was +17.5% **with** a bridge, so the no-bridge version of the same
design should still be net positive -- just less good. Prefill is bandwidth-bound rather than
latency-bound and loses more, but starts from a much larger +41%.

### Which implies pipeline parallelism, not tensor parallelism

This inverts the priority in the port plan above. A layer split crosses **once per token** instead
of twice per layer -- 128x fewer crossings for the 27B -- which makes it almost perfectly suited to
a machine with no bridge, and it is the scheme nobody has built. It also captures the memory win,
which is the larger prize and needs no interconnect speed at all: splitting the 35B-A3B's ~21 GB of
weights across two 24 GB cards frees roughly 12.5 GB per card, taking total KV from ~2.6 GB to
~25 GB. At 10 KiB/token that is ~262k tokens to several million -- an order of magnitude more
context or concurrency.

What it does not do is speed up a single request: the layers are sequential, so rank 1 idles while
rank 0 works. The throughput gain needs concurrency >= 2 and micro-batching. That is the trade
against the tensor split, which does help single-request latency and is the harder port.
