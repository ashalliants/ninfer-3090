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
| Hyper-connection ops | **Not reusable** (checked): Flash-Next's mixer is a gated low-rank form (SiLU, sigmoid gates, an inject row), V4's is a projection of the flattened stream followed by sigmoid pre/post and a Sinkhorn-normalised 4 x 4 comb. New Ops `hc_sinkhorn_mixes`, `hc_collapse`, `hc_expand_mix` (`include/ninfer/ops/hc_sinkhorn.h`) beside them; `inject` and `expand` do not apply | `src/ops/hyper_connection/hc_sinkhorn.cu` |
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
|   | *1a, done:* `tools/convert/dsv4.py` reads the GGUF metadata into the text config and maps all 1,043 tensors of ISTA's IQ2_XXS build (checked against its real headers, `tests/fixtures/dsv4/ista_iq2_xxs_header.json`), refusing anything unknown, missing, misplaced or mis-shaped. *1b, done:* the expert record formats: GSQ-RCO picks a type per layer and this build mixes eight (gate/up, down) pairs (IQ1_M, IQ2_XXS, IQ2_XS gate/up with IQ1_M, IQ2_XXS or IQ2_XS down), now registered (the other two builds need further pairs and IQ1_S). *1c, done:* `tools/convert/dsv4_model.py` and the `dsv4_gguf` recipe repack a GGUF exactly (block tensors keep their blocks, each layer's experts become one record bank in its own pair), tested by converting a tiny synthetic GGUF and comparing every object with the GGUF byte for byte. *Engram volume, done:* `tools/convert/dsv4_engram.py` writes the table (768,022,850 rows of 256 IQ4_NL values, 110 GB) as an n-gram volume beside the artifact, one row per 4 KiB read with the rows' GGUF bytes exactly, and the artifact's text config records its geometry and id under `engram_table`; opt-in with `--ngram-out` (write) or `--ngram-reuse` (bind an existing volume, checked on 512 sampled rows), so the default conversion stays a 132 GB artifact. *Tokenizer, done:* the runtime's `Tokenizer` (`src/models/qwen3_5/frontend/tokenizer.cpp`) now supports the V4 pipeline (no normalizer; Splits for runs of up to three digits, CJK and kana runs, and the word regex; byte level) beside Qwen's, matching the Hugging Face library on 1,159 pre-tokenizer cases and 208 full encodes of the real `tokenizer.json`; the converter requires the checkpoint's `tokenizer.json` (`--resource tokenizer.json=PATH`, because the GGUF does not say which of its control tokens are special: 49 are not) and checks vocab, merges, added tokens and pipeline against the GGUF. | |
| 2 | Host-only three-tier memory plan (VRAM / RAM / SSD ledgers) and an SSD extent reader with tests; refuses nothing a 64 GB box can run | Mac (CUDA-free build) |
| 3 | CPU reference forward of one block (router, hyper-connections, attention, MoE) in C++ or NumPy against `tools/dsv4/reference.py`, so GPU kernels have an oracle | Mac |
|   | *3a, done:* `tools/dsv4/hyper_connection.py`, the NumPy hyper-connection oracle (mixes, Sinkhorn, pre, post and the two-sublayer chain of a block), checked against the model repo's own formulas in PyTorch to float32 rounding, with a golden fixture. *3b, done:* `tools/dsv4/attention.py`, the NumPy attention oracle (window and compressed KV, the compressor, the indexer with its two-level candidate blocks, sinks, grouped output, YaRN), matching the model repo's own `Attention`/`Compressor`/`Indexer` classes run on CPU (`tools/dsv4/model_golden.py`) to about 1e-6 over a 7-layer stack. *3c, done:* `tools/dsv4/moe.py`, the NumPy MoE layer (router chain, routed experts with their separate swiglu clamps, shared expert), matching the model repo's own `MoE` class on CPU. *3d, done:* `tools/dsv4/engram_ref.py`, the NumPy Engram (hash ids from the compressed vocabulary with the XOR chain over look-backs, the dequantised row lookup, the gated residual write), hash ids exactly equal and outputs within float32 rounding of the repo's `EngramLayout`/`NgramHashState`/`Engram`. *3e, done:* `tools/dsv4/model_ref.py`, the whole forward pass (embedding, the hyper-connection stream, Engram writes before their layers, blocks, final collapse, norm, head) composed from the pieces above, matching the repo's own `Block`s, embedding, norm and head on a 7-layer model to 1.5e-6 in the logits, so milestone 3 is complete. *4a, done:* the hyper-connection Ops (`hc_sinkhorn_mixes`, `hc_collapse`, `hc_expand_mix`) against FP64 oracles of the formulas at V4's geometry (S = 4, H = 5120; one block per column for the mixes, unmeasured). Kernel tests for milestone 4 can compare against any of 3a to 3e, layer by layer. Decision: the oracle is a mathematically faithful float reference, not a bit-exact one: the model repo's attention simulates FP8/FP4 activation quantisation in its KV caches and uses `tilelang` kernels, which kernel tests should compare against with tolerances rather than bit for bit. | |
| 4 | Dense path on the card: attention, shared experts, head; layer-by-layer parity against the reference | 3090 or Vast |
|   | *4c, done:* `gathered_sink_attention`, V4's attention core: 64 query heads over one shared KV head, each query reading its own row list from a KV table (its window plus the indexer's compressed positions) with the per-head sink in the softmax denominator only, keys and values the same rows; D in {64, 128, 256, 512}. Against an FP64 oracle at one BF16 rounding (half a ulp plus a small FP32 allowance), at V4's geometry (D = 512, H = 64, K = 640) and edge cases (invalid and out-of-range entries, empty lists, large scores, extreme sinks). Decode (T <= 4) splits each head's list across 8 warps (37 us per layer at K = 640 on the 3090, 137 us before the split); wider calls run one warp per head (about 11.5 us per token per layer: CUDA cores, no tensor-core path yet). Not yet: the compressor, the indexer's score and top-k, RoPE with the layer's YaRN frequencies, the grouped output projection. | |
|   | *4d, done:* `rope_pairs`, V4's rotary embedding: adjacent element pairs of the last `rotary_dim` elements of each head rotated by `position * inv_freq[i]` (angle in FP64, so positions past a million stay exact), a caller-supplied frequency table (YaRN where a layer compresses, plain elsewhere), forward or inverse (the attention output's rotation is removed before the output projection). Against an FP64 oracle at one BF16 rounding, on V4's query and kv geometries, positions to 1,048,575, both directions. | |
|   | *4e, done:* `softmax_pool`, the pooling step of V4's KV compressor: each channel pools every group of `ratio` consecutive tokens with a softmax over the group (BF16 in and out, FP32 arithmetic); the projections and the norm around it are the existing linear and rmsnorm Ops. V4.1-Flash's `compress_ratios` are 0 (window only), 2 (compress) and 1 (a plain projection, no pooling), so ratio 2 is the case that matters; ratios to 256 are served. Against an FP64 oracle at one BF16 rounding, including 70,000 groups (past grid.y). | |
|   | *4f, done:* the indexer's selection, in `position_selection.h`: `weighted_relu_dot_scores` (score[t, s] = scale * sum over the 32 index heads of weight * relu(q . k), -inf past each query's reachable length; FP32, within a few FP32 ulps of the FP64 oracle) and `top_k_positions` (the exact k best reachable positions, ties to the lower position, written ascending with an offset and -1 padded: a four-pass radix select and an ordered compaction; bit-exact against a stable-sort oracle on heavily tied scores, 300,000 positions, k to 4,096). On the 3090: decode over a 32k-token context (16,384 compressed positions) 21 us scores + 57 us top-512 per index layer; a 512-token prefill chunk over 16k tokens 2.9 ms scores (CUDA cores, about 6 TMAC/s; the tensor-core version is the obvious follow-up) + 0.14 ms top-512. Not yet: the second-level candidate-block selection of the later index layers (layer 20 keeps the best 2048 blocks of 8 positions, which only matters past 32k tokens of context; the caller can already apply a mask by writing -inf into the scores). | |
|   | *4g, done:* V4.1-Flash's dense GGML projections on the existing `linear` and `linear_add` Ops (no new kernels: the registry and the compiled K instances grew): Q8_0 `q_a` (5120 -> 1280) and `kv` (5120 -> 512); Q5_K `q_b` (1280 -> 32,768), the grouped output projection's `a` (4096 -> 8192, and one group as 4096 -> 1024) and `b` (8192 -> 5120, LinearAdd), and the shared expert (5120 -> 2304 twice, 2304 -> 5120 LinearAdd); K = 1280, 2304, 4096, 5120 and 8192 are new (a registered shape with an uncompiled K is now a compile error). All pass the suite's GGML criteria on the 3090 on both routes and the int8 tensor-core decode route bit for bit against its emulation. Decode (T = 1) per layer: q_a 13.8 us, kv 11.8, q_b 37.4 (about 790 GB/s), out_a 31.1, out_b 39.1, shared gate/up 16.5 each and down 16.5: about 183 us, 7.3 ms over 40 layers (the weights are 4.6 GB, 5 ms at the card's 936 GB/s). The existing Flash-Next kernels are SASS-identical on sm_86 and sm_70. Not yet: the vocabulary head and token embedding at K = 5120, the BF16 projections (compressor, indexer, router), the Engram's Q5_K `engram_kv` (6144 -> 25,600). | |
|   | *4h, done:* V4.1-Flash's BF16 projections (the router [384, 5120], the compressor's kv and gate [512, 5120], the indexer's weights [32, 5120], query [4096, 1280] and key [128, 512]) on `projection_fp32` (FP32 out; `cast_fp32_to_bf16` where BF16 is wanted), whose K limit grows from 3,072 to 8,192: K up to 5,120 at decode widths (1 to 8 columns) takes the narrow one-row-per-warp kernel instantiated at its larger size (the router: 10 us a call against 43 us on the wide mapping, which serves every wider call and K above 5,120; same bits as the K <= 3,072 mapping, so a column does not depend on T), checked against the FP64 bound at N = 384, 512 + 512, 32, 4096 (K = 1280) and K = 8192. At T = 512 the router takes 0.21 ms a layer. The Flash-Next kernels are SASS-identical on sm_86 and sm_70. | |
|   | *4i, done:* `clamped_swiglu` (the shared expert's SwiGLU with V4's limit: gate capped above, up both sides; `silu_mul` has none) and `weighted_slot_sum` (a token's routed expert outputs weighted by the router and summed in slot order, plus the shared expert's output; `moe_combine` gates the shared term by a sigmoid and is specific to Qwen), each at one BF16 rounding against an FP64 oracle. With the router (4b), `moe_experts_narrow` and the Q5_K shared-expert projections (4g) the MoE half of a block is now made of Ops. | |
|   | *5a, done:* `src/models/dsv4/`: the text config (per-layer compress ratio, kv and index sources, the nearest preceding kv source each layer reads, Engram layers, the two-level candidate selection constants) and the parameter binding of every GGUF tensor in the formats it was stored in, expert banks pinned or left in the artifact; `ninfer_dsv4_load_test` binds the real IQ2_XXS artifact (dense weights 6.16 GB on the device, experts 132 GB). Not yet: the forward pass, the state, the Program, the registry and Engine. | |
|   | *5d, done:* `src/models/dsv4/execution/attention.{h,cpp}`: the attention sublayer of a block as library code over bound weight views and persistent per-sequence state (`AttentionState`: per kv owner one KV table of the compressed rows plus each member layer's raw rows, the index keys, the compressor's pending partial group, the shared top-k and the candidate blocks), so a prefill chunk of any length and a decode step run the same function (`attention_layer`; transient storage from the caller's arena, no host sync; `bind_attention_views` takes the Program's bound parameters). New small exact Ops in `attention_rows.h`: reachable lengths, the per-query row lists, and the candidate-block selection (`block_max_scores`, `mask_scores_outside_blocks`) of the layer-20 two-level top-k. `ninfer_dsv4_attention_block_test` on real exported layers 0 (window), 2 (ratio 2, kv and index source) and 20 and 21 (ratio 1, a kv/index source and a layer reading its KV): whole sequence against the NumPy oracle at 0.9-1.1% relative L2, odd chunkings (1, 7, 128, 129, remainder, then 20 decode steps) and chunks of 3 within 0.2% of the single chunk with bit-identical compressed rows and index keys, a graph-captured decode step bit-identical to the eager one. Not yet: the Program, a batched (several sequences) call, a graph that replays across positions (a chunk's `start` is a kernel argument today), the candidate mask beyond a unit check (it bites only past 16,384 compressed positions). | |
|   | *5e, done:* `src/models/dsv4/execution/forward.{h,cpp}`: the whole forward pass for one sequence (embedding into the four hyper-connection streams, forty blocks wrapping the attention and FFN sublayers in their mixes with the previous sublayer's `pre`, final collapse, norm, Q5_K head), chunked prefill and one-token decode from the same code. On the real IQ2_XXS artifact with the experts read from the artifact file, 'The capital of France is' continues ' Paris. ... The City of' (greedy); logits are the same whatever the chunking (relative L2 0). Cold-cache prefill of 6 tokens 42 to 72 s and decode about 7.5 s per token: I/O bound (about 280 MB/s from the drive, 24 GB read for 10 tokens). Not yet: Engram (the artifact lacks its token map and multipliers), the Program and Engine, decode graphs, expert tiers and prefetch. | |
|   | *5b, Engram injection, library done:* `src/models/dsv4/execution/engram.{h,cpp}`: the n-gram hash (`EngramHasher`, a per-sequence `History` of the previous three compressed ids, so prefill chunks, decode steps and copied speculative branches hash alike; row ids exactly the oracle's), the row reader (Flash-Next's `NgramVolume` over the Engram volume, one read call per staged chunk into a pinned buffer, no device wait), and the device sequence per Engram layer (copy of the staged rows, `ple_embed`, `engram_kv` through Linear, new Op `engram_gate_inject`: the gate and the write into the four streams). New shape: Q5_K `engram_kv` 6144 -> 25600 in the GGML registry (K = 6144 was compiled already). Tested by `ninfer_dsv4_engram_test` (hash golden from the oracle, a synthetic volume, and layer 1's real weights and rows with `tools/dsv4/export_engram_fixture.py`). Not yet: the artifact does not carry `engram.token_map` and `ple.layer_multipliers` (the converter drops them; the hasher takes them as `EngramHashParams`), no Program or prefetch, no dead-token (image) mask. *5b, Engram wired, done:* `python3 -m tools.convert.dsv4_engram_volume GGUF DIR` writes the volume (112,350,777,344 bytes, about 44 minutes from the HDD to a SATA SSD), the GGUF's token map and a sidecar `engram.json` (volume id and geometry, multipliers, bucket counts, checksums) and validates it; because the converted artifact records no volume and lacks the token map and multipliers, the runtime binds that folder (`--engram-dir DIR` or `NINFER_DSV4_ENGRAM_DIR`, checked against the artifact's config by `load_engram_sidecar`) and `Forward` stages and applies the Engram before layers 1 and 14 (`engram_applied()` true; the hash history persists across chunks and decode). On 'The capital of France is' the top logit stays ' Paris' (20.36 without, 20.91 with) but the rest of the distribution and the continuation change; chunked and whole prefill still agree exactly, warm decode stays about 41 ms per token. | |
|   | *5f, done (tiers):* `FileExpertSource` now has a device tier: a pool of VRAM frames (`device_cache_bytes`, 12 GiB is 1259 frames of the 9.3 MB stride) the Op reads in place, over the pinned RAM LRU, over the file. The policy is `expert_tiers.h` (host-tested, `ninfer_dsv4_expert_tiers_test`): frequency with halving every 16384 calls decides who holds a frame, recency breaks ties, nothing a call routed is evicted by it, so one prefill chunk does not flush the hot set (at most 16 promotions per call). Promotion copies a RAM slot to its frame on a non-blocking copy stream and is usable from the next call (the copy's event is waited on before that call's Op; a RAM slot being copied is not an eviction victim); the call that promotes still reads the record from RAM. The missing records of a call are read in file order on `read_threads` threads, and the ids' copy starts right after the router (`ExpertSource::begin`) so the shared expert runs while the reads happen. Outputs are the same bits as before (`ninfer_dsv4_ffn_block_test` asserts it for the tiered source at T=8, T=1 and T=512). Measured on the real artifact (6-token prompt, 8 GB RAM tier, H: hard disk): decode of the same text again with the caches warm 405 to 42 ms/token (88% of routed experts from frames, the rest from RAM, no file reads), decode after a cold prefill 8.3 to 6.9 s/token (the hard disk, 33% file reads), cold prefill unchanged (about 75 s for 9 GB read at about 125 MB/s). Parallel reads: on the hard disk 8 threads read random 8 MB records 4x slower (35 against 140 MB/s, seek thrash), so the default is 1; on a SATA SSD 8 to 16 threads gave 1.5x (160-210 to 280-330 MB/s); the NVMe box is the target and is not measured. Not done: graph capture of the file-backed path, the Program's own tiers. (The RAM-tier policy and O_DIRECT are in 5g.) |
|   | *5g, done (tier policy, VRAM sizing, read bench):* RAM and VRAM tiers share one policy object (`expert_tiers.h`, `TierPolicyOptions`). The RAM tier is no longer plain LRU: default `Lfuda` (use count plus the priority of the last victim, so a hot record outlives a scan of once-used ones and an old hot set still fades), and `exclusive`: a record that has a VRAM frame is the first RAM victim, so RAM and VRAM hold distinct records. `NINFER_DSV4_TIER_POLICY` (lru, lfu, segmented, lfuda) and `NINFER_DSV4_TIER_EXCLUSIVE` select others on the same build. The VRAM tier is sized after the forward exists (`FileExpertSource::set_device_cache_bytes`, `dsv4_expert_vram_bytes`: free device memory less a 2 GiB headroom, cap 16 GiB; the decode graphs took 27 MiB) instead of 12 GiB less a 5 GiB reserve taken before the forward's allocations; `NINFER_DSV4_EXPERT_VRAM_MIB` still overrides it. Evaluation: `NINFER_DSV4_EXPERT_TRACE=FILE` records the (layer, experts) of every call, `ninfer_dsv4_expert_trace_record` runs prompts (`NINFER_DSV4_PROMPTS`) to make one, and `ninfer_dsv4_expert_trace_replay_test` with `NINFER_DSV4_TRACE=FILE` replays it through the real policy code at any tier sizes. Six different 16-28 token prompts with 20 greedy tokens each (traces of 16 KB each, Engram on, 4 GiB RAM + 12 GiB VRAM = 505 + 1515 records): file reads (misses) fall from 45.1% of expert uses (LRU, inclusive) to 42.7% (Lfuda, exclusive); segmented 42.2%, pure LFU 53.6% (a record used once in a prefill chunk is a bad victim only when everything else is used twice: LFU without aging loses), Belady bound 35.3%. Exclusivity is most of the gain; the choice among LRU / segmented / Lfuda is within 0.5 points on these cold-heavy traces (the prompts share little), so Lfuda is kept for its resistance to prefill sweeps in longer sessions. The knobs (aging period, promotion margin, 4 to unlimited promotions per call) are flat. Real artifact, `ninfer_dsv4_forward_test`, 4 GiB RAM, H: disk: warm decode (the same prompt for the fifth time) 32.5 -> 22.3 ms/token with VRAM hits 91.3% -> 99.8% (old: 12 GiB tier = 1259 frames, LRU inclusive; new: sized from free memory = 1506 frames; the new policy alone at 12 GiB gives 32.8 ms, so the gain here is the larger tier), and a pass that evicts (pass 2 of the same test) read 0.55 GB instead of 2.75 GB with the new policy at the same tier size. Cold reads (`tools/dsv4/read_bench.cpp`, random 9.3 MB reads, WSL2 9p mounts with msize 64 KiB, no NVMe here): K: SATA SSD 152 MB/s with one reader, 205 / 221 / 234 MB/s at 2 / 4 / 8 threads, 250 MB/s at 16 with O_DIRECT or WILLNEED, flat after that: O_DIRECT, fadvise RANDOM/WILLNEED and 0.5-2 MiB chunks across threads buy nothing over buffered parallel preads (the 9p link is the ceiling), mmap is 5-10x slower; H: hard disk 110-150 MB/s with one reader (sorted batch of 96: 149), 26-37 MB/s with 2 to 8 (seek thrash), O_DIRECT 123 vs buffered 110. So `read_threads` stays 1 by default and 8 is right for an SSD; an NVMe on a native file system is not measured. |
| 5 | MoE with VRAM + RAM tiers (full-RAM first, as Flash-Next M0), then the SSD tier | 3090, GPU lock |
|   | *Expert kernel, narrow route, done:* `moe_experts_narrow` serves DeepSeek-V4.1-Flash's [E, 5120, 2304] experts on the card (the eight K-quant record pairs of the IQ2_XXS build, the clamped SwiGLU), the existing narrow kernels instantiated at the new shape (`Dims<5120, 2304>`; the Qwen kernels are unchanged, SASS-identical); no CPU-served misses, wide (prefill) route or register gate/up variant yet. Against an FP64 oracle at normwise 1.7e-3 on all eight pairs, bit-identical across placements. | |
| 6 | Prefill streaming at large chunks; Engram volume and prefetch | 3090 |
| 7 | Lanes, then MTP / DSpark speculation, then vision | 3090 |
|   | *7a, prefix reuse, done:* the single-lane Program keeps the sequence the forward's state covers (`held_`) after a request ends, plus up to four `ForwardCheckpoint`s (`Forward::capture/restore`, `AttentionState::capture/restore`): a checkpoint is only each layer's carried raw window rows (127 x 512 BF16) and each compressor's pending partial group, **5.0 MiB** on the device, because the compressed rows and index keys are append-only and stay valid while nothing runs from an earlier position (any run from position s drops the checkpoints past s). The rule is `program/prefix_reuse.h` (host-tested, `ninfer_dsv4_prefix_reuse_test`): a prompt starting with the whole held sequence continues the live state; otherwise the deepest checkpoint inside the common prefix is restored; the prompt's last token always runs. Every prompt's prefill stops one token short of its end to checkpoint there (one decode-sized extra step), which serves a retry and the thinking-mode follow-up (its template turns the last prompt token `<think>` into `</think>` and drops the reasoning, so the prompts agree to the token before it; `ninfer_dsv4_frontend_real_test` pins both shapes). To the Engine it is a root source with `reused_tokens` and no `CheckpointHandle` (there is no context cache, store, capture or sharing), reported as a prefix hit; `allow_prefix_reuse` off runs from position 0 and keeps nothing. Exact: a restore plus the rest of a sequence gives bit-identical logits to the same chunks from scratch (`ninfer_dsv4_forward_test`, 137-position checkpoint over a 200-token sequence, after a different continuation overwrote the state). Measured through the Engine on the real artifact (H: hard disk, 4 GiB RAM tier, 474-token first message): second turn (492 tokens) prefilled 17 tokens, time to first token **55 s against 317 s** with reuse off (caches already warm for the second; the cold first turn took 373 s); the first prompt sent again prefilled 1 token, 0.66 s; the answers equal those from scratch (`ninfer_dsv4_engine_real_test`). Held: one sequence, so an interleaved second conversation evicts the first (the lane count is milestone 7c). |
|   | *7b, speculation: assessed, not implemented.* The artifact has no MTP layers (`tools/convert/dsv4.py`: they are not in the files) and no drafter block, so only prompt-lookup / n-gram drafting is possible. `Forward::run` takes a chunk, so verifying k drafts costs one call, but it would also need per-position logits (the head over k columns, about 0.45 GB of weights read once) and a rollback of the state on a rejection (now cheap: `Forward::capture/restore`, 5 MiB). The deciding number is what a k-token call costs against k one-token steps, measured with `NINFER_DSV4_SPEC_BENCH=1` in `ninfer_dsv4_forward_test` on the model's own greedy continuation (best case, every draft accepted; 3090, 4 GiB RAM tier, H: disk). **Resident regime** (9-token sequence, every routed expert in VRAM or RAM, 0 file reads, a warm decode step 26.7 ms): k=2 40 ms per call, k=3 39, k=4 46, k=6 50, k=8 71, i.e. 20, 15, 11, 12, 9 ms per token. A step is dominated by the 40 per-layer host syncs and launch latency, not bytes, so a chunk costs little more than a step. **Disk-bound regime** (32-token sequence, the working set exceeds the tiers, 4.7% file reads even on the fifth pass): 468 ms per token at k=1, 462 (k=2), 422 (k=3), 467 (k=4), 381 (k=6), 395 (k=8): the union of experts across tokens barely overlaps, every token still needs its own 6 x 40 records and the hard disk is the cost, so a chunk saves nothing but the dense weights' one read. Break-even in the resident regime at k=4 is 46/26.7 = 1.7 tokens per call (0.7 accepted drafts of 3 on average); prompt-lookup on free chat prose accepts well under that, on code or quoted text it can exceed it, and in the disk-bound regime nothing helps. Decision: not implemented now, because the configuration this model ships in (a hard disk or a RAM tier that holds a fraction of the experts) is the disk-bound one; revisit when an NVMe box or enough RAM keeps the decode working set resident, with the pieces above (per-position head, `restore` on rejection) and an acceptance measurement on real code and chat traces first. The k=1 row is also the decode graph's 26.7 ms; k > 4 does not scale, as the routed-expert work then grows with the union. |
|   | *7c, a second lane: assessed, not implemented.* A lane's own state is small (`AttentionState` 0.09 GB at 2048 context, 5 MiB per checkpoint, the compressed rows about 2 KB per token), so the memory is no obstacle; two other things are. (1) The `Forward` is one object: one workspace arena, one logits buffer, one set of captured decode graphs, one Engram hash history; a second lane means a second `AttentionState` plus either a second `Forward` over the shared `ExpertSource` or swapping the state in and out of one (the checkpoint machinery could carry a lane's window rows, but the compressed rows are per lane and not part of a checkpoint). (2) Serialised lanes share the expert tiers, and requests with different content route to different experts: with the working set already larger than the RAM and VRAM tiers (the disk-bound regime above, 468 ms per token), alternating two requests halves each one's hit rate and roughly doubles the reads per token; total throughput does not rise and each request's latency doubles. Real batching (the union of both rows' experts read once per layer) does not help either when the rows share few experts, and needs the round's rows in one `Forward::run`, per-row positions in the attention and a batch-aware decode graph. The Engine does queue the second request, which is the right behaviour for one disk. Decision: not implemented; a second lane pays only where the expert working set of two conversations fits the tiers (NVMe or large RAM), and is then best done as batching of the decode rows with the expert union (`ExpertSource` already unions across a chunk's tokens), together with 7b's per-position logits. |

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
