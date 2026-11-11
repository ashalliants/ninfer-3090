# DeepSeek-V4.1-Flash on the RTX 3090: scoping and plan

Scoping done on a Mac with no GPU and no weights (2026-11-11). Everything below comes from the model's
configs, safetensors headers, reference code and published quantisations, plus a handful of range
requests (one real expert, one router, three Engram rows). Nothing has run on a card. Numbers marked
**model** are the output of `tools/dsv4_perf_model.py` and carry that model's error bars (section 4).

Target: **DeepSeek-V4.1-Flash-1017** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`; its config and tensor
layout are identical to the earlier `DeepSeek-V4.1-Flash`). The older V4-Flash (284B, 256 experts) is a
different, much smaller model and is out of scope.

## 1. Verdict

- **It fits, as an SSD-tiered MoE, and it is slow on 64 GB of RAM.** The routed experts are 544 B
  parameters (15,360 experts). Even the best 2-bit build is 132 GB of experts. VRAM + 64 GB RAM holds
  about 54% of them; the rest must stream from NVMe on every miss.
- **Decode on the target box (3090, 64 GB, 3.6 GB/s NVMe): about 4.5-6 tok/s** (model, static hot set,
  no speculation). Strata, an engine that runs this model on consumer cards (llama.cpp b11870+ does too), measures 4.7 tok/s
  on an RTX 5070 with the same RAM and drive. More RAM is the lever: with 128 GB almost nothing is on
  the SSD and the model says about 14 tok/s, PCIe-bound.
- **Prefill is bound by streaming the model once per chunk**, not by compute: about 190 tok/s at 4K
  chunks and 380 at 8K (model; Strata measures 121 and 231 on the 5070). Bigger chunks are almost free.
- **Concurrency does not multiply throughput here the way it did for Flash-Next.** Experts are not
  shared between independent requests, so aggregate decode only moves from 5.5 to 6.0 tok/s at 8 lanes
  (model). It is still worth building for, because a lane costs almost no memory (section 6) and
  prefill, shared-prefix and multi-user traffic gain from it. It must not be designed out.
- **My first estimate in this conversation was wrong in two ways**, both corrected here: it used the
  284B V4-Flash instead of the 544B V4.1, and it borrowed Flash-Next's expert skew. V4.1's measured
  routing is far flatter (top 10% of experts carry 32% of traffic, not 85%+), so the SSD matters much
  more than that estimate allowed.
- **Quality is not the blocker.** ISTA-DASLab's GSQ-RCO 2-bit builds report 100.0% (IQ2_XXS) and 99.8%
  (IQ2_XS) of the base model's task average (their own evaluation; Strata measures perplexity +15.5% and
  +7.3% against the original). They are standard GGUF, from the group whose IQ2_XS Flash-Next already ships.

## 2. The model

From `config.json`, the safetensors headers and `inference/model.py` of the model repo.

| | V4.1-Flash |
|---|---|
| Layers / hidden / vocab | 40 / 5,120 / 129,280 |
| Routed experts | 384 per layer, 6 per token, SwiGLU 5,120 -> 2,304 -> 5,120, clamp 10 |
| Shared experts | 1 per layer (same shape) |
| Router | sqrt-softplus scores, an aux-loss-free bias picks the experts but does not weight them, top-6 normalised, x1.5 |
| Attention | 64 query heads, head dim 512, ONE shared KV head, low-rank Q (1,280) and grouped low-rank output (8 x 1,024); 128-token sliding window plus up to 512 compressed positions chosen by an indexer; compressed KV is only built on layers 2, 8, 14, 20 and shared down the stack |
| Residual stream | 4 hyper-connection copies per token (`hc_mult` 4, Sinkhorn mixing); each sublayer's mix is computed by the one before it |
| Engram | layers 1 and 14: 8 heads x (4-1) n-gram hashes = 24 rows per token per layer, from a 384 M-row x 256-wide table; hashes depend only on token ids |
| Speculation | 3 MTP layers and a DSpark block drafter (block 5, 128 experts) ship in the checkpoint |
| Vision | a 32-layer ViT plus aligner |
| Context | 1,048,576 (YaRN) |

### Bytes (`tools/dsv4/layout.py`, `tools/dsv4/gguf_sizes.py`; checked against the real headers)

| | Native checkpoint | GSQ-RCO IQ2_XS | GSQ-RCO IQ2_XXS | GSQ-RCO IQ1_M |
|---|---:|---:|---:|---:|
| Routed experts (all) | 288.8 GB (FP4 + e8m0 scale) | 149.1 GB | 132.1 GB | 114.6 GB |
| One expert | 18.80 MB | 9.71 MB | 8.60 MB | 7.46 MB |
| Dense path read every token | 8.4 GB (FP8, BF16 head) | 5.55 GB | 5.55 GB | 5.55 GB |
| Backbone | 307 GB | 155.2 GB | 138.2 GB | 120.7 GB |
| Engram tables | 202.8 GB (FP8) | 110.6 GB (IQ4_NL) | same | same |

The sizes computed here match the published backbones to 0.03%. The GSQ-RCO builds assign a type
**per layer** (expert size runs 7.74-10.23 MB across layers in the IQ2_XXS build), so an expert record's
size is not a model constant. Per-token traffic: 240 expert reads (40 layers x 6), 2.1 GB at IQ2_XXS.

Engram rows are 144 B at IQ4_NL, 48 per token, random reads from a 110 GB file: about 7 KB of data,
48 page-sized reads, known from the token ids before the layer runs.

## 3. What fits where (3090, 64 GB RAM, IQ2_XXS)

| Tier | Capacity after reserves | Experts held | Share |
|---|---|---:|---:|
| VRAM | 24 GiB - 1.8 reserve - 5.75 GB dense = 16.9 GiB | 2,100 | 13.7% |
| RAM | 64 - 14 reserve = 50 GB | 6,242 | 40.6% |
| SSD | about 61 GB of experts, plus 110 GB Engram | 7,018 | 45.7% |

The 14 GB RAM reserve (OS, the embedding table, staging, the Engram row cache, pinned buffers) is an
assumption; a tighter one moves experts from SSD to RAM. Disk needed: about 250 GB.

## 4. Performance model

`tools/dsv4_perf_model.py` (tests: `tools/test_dsv4_perf_model.py`). A token picks expert i of a layer
with probability q_i; the caches hold the most popular experts; a round of B lanes touches an expert
with probability 1 - (1 - q_i)^B; time is the dense path plus VRAM hits at 29% of the card (Flash-Next's
measured narrow-route efficiency) plus RAM misses over PCIe (24.2 GB/s, measured) plus SSD misses.

**The routing data is real.** Strata ships `data/dsv41/profile-2.00.bin`: how often the router chose
each of the 384 experts in each layer over 1.78 M calibration tokens (MIT; `tools/dsv4/profile_to_curve.py`
turns it into `tools/dsv4/data/routing_rank_curve.json`). The 2.00 and 2.25 profiles agree, so it is a
property of the model. The top 5 / 10 / 20 / 44% of experts per layer carry 21 / 32 / 48 / 73% of
routing. Zipf slope is about 0.5. A static cache of the 13.7% most popular experts therefore serves
38% of reads and the SSD sees 19%.

Decode, one lane (model, 3.6 GB/s NVMe, 64 GB RAM, 3090):

| Build | tok/s | SSD share of reads | Notes |
|---|---:|---:|---|
| GSQ IQ2_XS | 4.3 | 52% of experts, 24% of reads | |
| GSQ IQ2_XXS (recommended) | 5.6 | 46% of experts, 19% of reads | |
| GSQ IQ1_M | 7.9 | 37% of experts, 14% of reads | 98.7% task recovery |
| IQ2_XXS, 7 GB/s NVMe | 7.9 | | a gen4 drive |
| IQ2_XXS, 128 GB RAM | 14.2 | none | PCIe-bound on RAM-tier experts |
| IQ2_XXS, half of every miss hidden behind compute | 10.2 | | the biggest single lever (section 9) |

Lanes at IQ2_XXS, 3.6 GB/s: 1 / 2 / 4 / 8 lanes give 5.5 / 5.7 / 5.9 / 6.0 tok/s in aggregate.

Anchors, and why to trust the model only to about 30-40%:

- **Prefill vs Strata's RTX 5070 (12 GB, 64 GB, 3.6 GB/s):** model 160 / 320 tok/s at 4K / 8K against
  published 121 / 231. Optimistic by about a third; the 4K -> 8K doubling matches.
- **Decode vs the same box** (12 GB VRAM): model gives 4.2 tok/s against published 4.7. Strata's engine
  computes RAM-tier experts on the CPU instead of staging them, and uses speculation, so this is a
  coincidence of two different mechanisms, not a fit.
- **Flash-Next's routed-expert time** is reproduced within 25% (tested).
- **Assumptions to measure first:** dense-path efficiency 60% of the card, 0.4 ms per extra lane, a
  14 GB RAM reserve, SSD rate on this box.
- **Static hot set.** Real LFRU caches also exploit locality inside a prompt, so hit rates should be
  at least these. By how much is unknown until a per-prompt routing trace exists.

## 5. What Flash-Next gives us, and what is new

Flash-Next (`src/models/qwen4_exp`) was built from the same ideas (hyper-connections, an Engram-style
n-gram volume, an MoE with an expert cache), so most of the structure transfers.

| Piece | Reuse | Where |
|---|---|---|
| VRAM expert cache: frames in a VMM arena, per-layer frame tables, LFRU, warm start, grow/shrink control law | Reuse; the policy needs a record stride that varies by layer | `program/expert_residency.*`, `program/expert_cache/*` |
| Narrow T=1 expert route, staging slots, zero-copy misses, CPU miss channel | Reuse as the model for a V4.1 kernel; the IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S decoders already exist | `src/ops/offloaded_sparse_moe` |
| Partial RAM residency (RAM slots for experts no VRAM frame holds, write-backs) | The RAM tier, as is | `program/ram_slots.*`, `memory_plan.*` |
| Prefill expert streaming into lent frames, 4,096-token chunks | Reuse; V4.1 wants bigger chunks | `execution/expert_stream` |
| N-gram volume reader and row prefetch | Reuse the mechanism; row format and hash differ | `program/ngram_volume.*`, `ngram_rows.h` |
| Hyper-connection ops | Reuse after checking mix/Sinkhorn details against `hc_split_sinkhorn` | `src/ops/hyper_connection` |
| MTP drafter, DFlash, chained verify | Later milestone | `program/draft_policy.h`, `docs/maintainer/dflash.md` |
| NVMe checkpoint tier | Pattern for an SSD reader | `program/nvme_tier.*` |
| **SSD expert tier** | **New.** `memory_plan.h` records that it was not ported from Infernix (which has one) | - |
| MLA-style attention: one KV head of dim 512, low-rank grouped output, window + indexer-chosen compressed positions, attention sinks | **New.** The largest kernel work | `src/ops/softmax_attention`, `qsa` are the nearest |
| FP4 / GGML decode of per-layer mixed types, Q5_K and Q8_0 dense | Q5_K / Q8_0 decoders: check; the rest exist | `src/ops/linear` |
| Sqrt-softplus router with bias and the clamped SwiGLU | **New**, small | `tools/dsv4/reference.py` is the oracle |
| Vision, DSpark | Out of scope for the first milestones | |

Open-source references that already run this model, MIT or Apache: **Strata** (`src/dsv41`, CUDA/HIP/
Metal kernels for these exact GGML types, ported from Project Maya's `maya-ds4`) and **llama.cpp b11870+**
(V4.1 support, mmap'd Engram). DeepSeek's own `inference/` (PyTorch + TileLang) is the maths reference.
Per this repo's convention, adapted code carries a header naming its source and licence.

## 6. Concurrency: what a lane costs and what it does not share

A lane's state is small: a 128-row window KV (40 layers x 128 x 512 x 2 B = 5.2 MB), the compressed
KV on four layers (about 2 KB per context token, so 260 MB at 128K and about 2 GB at 1M, before any
quantisation), and 40 KB of hyper-connection residual. A lane is cheap in memory; what it is not cheap
in is **experts**: each lane brings its own 240 reads per token and independent requests share few.

Design rules so that the lane count is never a corner:

1. **The Program owns per-lane state in slots**, as Flash-Next PR 16 does, with no `lanes == 1`
   assumption in scheduling, graph capture or the expert path.
2. **The expert path is grouped by expert across the round's rows**: build the union of
   (layer, expert) the round needs, read each once, apply it to every row that selected it (Strata's
   `moe_win` does this for a verify window). This is also what makes MTP verification and prefill work.
3. **Tier placement is invisible to results**, the existing Flash-Next invariant ("expert outputs
   are placement-invariant"). Cache state changes speed, never bits.
4. **Admission is working-set aware.** The scheduler can estimate a request's expert cost (its recent
   routing) and the marginal read cost of adding a lane, and refuse or defer a lane that would push
   the SSD tier past its budget. Requests with a shared system prompt or domain overlap far more
   than the independent draws modelled here; that is the upside to measure.
5. **Prefill chunks are the amortiser**: a chunk pays the model's streaming cost once for all its
   tokens, so prefill of several requests should batch into one chunk where the Engine allows.

## 7. Design rules that keep us out of corners

- **Record format is a plugin.** An expert record is (layer, expert) -> bytes of three matrices with a
  per-layer type. Cache frames, RAM slots and SSD extents are sized by the largest layer's record, as
  Flash-Next does for its mixed IQ banks. No code may assume one quantisation.
- **Three tiers behind one interface**: `fetch(layer, expert) -> ticket` completing in VRAM, with
  placement decided by a policy object. VRAM-only and RAM-only (Flash-Next today) are special cases.
- **Dims come from the artifact config, never constants.** The same code should load a 384-expert and
  a 192-expert (ISTA Coder) variant.
- **SSD reads are first-class and asynchronous**: io_uring on Linux, overlapped I/O on Windows, queue
  depth above one, expert-sized aligned reads. The Linux AIO fix already merged is the base.
- **Engram is a tier-three tensor** with prefetch from token ids, one volume shared by all builds.
- **The artifact keeps the GGUF layout recognisable** so ISTA builds convert without re-quantising:
  a conversion is a repack of existing tensors, with no numeric change.
- **Nothing in the engine core knows the model's name.** Add a `dsv4` architecture next to
  `qwen4_exp`, selected from the artifact, as `runtime::artifact_is_qwen4_exp` does.

## 8. Milestones (each merges to master on its own)

| # | Delivers | Needs |
|---|---|---|
| 0 | **This PR**: the scoping tools, the CPU reference and golden vectors, the perf model, this plan | Mac |
| 1 | `dsv4` config and tensor-name map from the GGUF metadata; per-layer expert record layout and a repack of an ISTA GGUF into the artifact, with a tiny synthetic round-trip test | Mac, real GGUF shard headers |
| 2 | Host-only three-tier memory plan (VRAM / RAM / SSD ledgers) and an SSD extent reader with tests; refuses nothing a 64 GB box can run | Mac (CUDA-free build) |
| 3 | CPU reference forward of one block (router, hyper-connections, attention, MoE) in C++ or NumPy against `tools/dsv4/reference.py`, so GPU kernels have an oracle | Mac |
| 4 | Dense path on the card: attention, shared experts, head; layer-by-layer parity against the reference | 3090 or Vast |
| 5 | MoE with VRAM + RAM tiers (full-RAM first, as Flash-Next M0), then the SSD tier | 3090, GPU lock |
| 6 | Prefill streaming at large chunks; Engram volume and prefetch | 3090 |
| 7 | Lanes, then MTP / DSpark speculation, then vision | 3090 |

Milestones 1-3 are doable without a GPU; 4 onward need the card (or a Vast box; see the Vast memory
notes) and the GPU lock protocol in `CLAUDE.md`.

## 9. Open questions that only real data answers

1. **Per-prompt routing locality**: how much better than the global curve is a dynamic cache on one
   conversation? Record a trace with the first working forward and feed `tools/expert_cache_sim.py` (#442).
2. **Dense-path efficiency** on the 3090 for Q5_K / Q8_0 GEMVs and the 64-head, 512-dim attention
   (the model assumes 60% of the card; at 40% a 180 ms token becomes 184 ms, so this barely matters while the SSD dominates).
3. **Whether SSD reads should be computed on the CPU** (Strata's `cpu_part`) or staged to the card.
   On this box the 5950X and the PCIe link are comparable in bandwidth; measure both.
4. **How much of a miss can be hidden.** The model's `--overlap 0.5` (half of every RAM/SSD
   transfer overlapped with GPU work) takes 5.6 to 10.2 tok/s. Routing is only known at each layer,
   so hiding needs prediction (the previous token's experts, or the router run early on a cheap
   proxy) or other lanes' work to overlap with. This is the first thing to prototype after the
   baseline, and it is where lanes could earn their keep.
5. **Whether top-5 routing is acceptable** (Strata's Coder build does it; it cuts expert reads 17%).
6. **Speculation under SSD pressure**: a W-column verify touches the union of W tokens' experts,
   so it only pays when acceptance is high; Strata reports +2-13% on 12 GB cards.
7. **A 128 GB RAM box** changes every number here. Worth deciding what the product target is.

## 10. Reproducing the numbers

```sh
python tools/dsv4/fetch_headers.py deepseek-ai/DeepSeek-V4.1-Flash-1017 scratch/v41   # 10 MB, no weights
python tools/dsv4/layout.py scratch/v41 --tiers 16.9 50 --bpw 1.944
python tools/dsv4/gguf_sizes.py allocation.txt --check-gb 138.24        # ISTA's rco-allocation list
python tools/dsv4_perf_model.py --ssd-gb-s 3.6 [--sweep] [--format gsq_iq1_m]
python -I tools/dsv4/test_dsv4.py && python -I tools/test_dsv4_perf_model.py
```

Sources: [config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) and the tech report in
the base repo; [ISTA-DASLab GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF);
[Strata](https://github.com/Niko1221/Strata) `docs/DEEPSEEK41.md` and `data/dsv41/`; llama.cpp b11870.
