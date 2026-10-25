# Basalt learnings: design

Temporary working document for the draft PR that implements issue #383. Delete it, together with
its plan in `docs/superpowers/plans/`, when the PR is split into its final stack or abandoned
(`AGENTS.md`: temporary plans live only while the work is active). The findings below are the
issue's text; the decisions that follow are the ones agreed for this PR.

## Decisions for this PR

- **One draft PR, one commit per item.** Each commit stands alone so it can be dropped, reverted or
  cherry-picked into a stack of focused PRs once the GPU numbers are in (`PR_POLICY.md`).
- **Every speed or numerics change is behind an option that defaults to off.** The GPU session
  builds once and A/Bs each option against the baseline inside one GPU-lock hold. Bit-exact
  refactors that need no A/B (for example a keep-alive interval) may change the default, and say so.
- **The decode-path quality gate (cross-model item A) lands first**, so every numerics-changing
  item (int8 HC above all) is judged with the same tool on the first GPU run.
- **Big items go last:** partial RAM residency with the IQ3 expert record formats, the two-stage
  27B head, and the int8 tensor-core dense GEMV.
- **Excluded:** drafter self-distillation (cross-model item E) needs GPU training; splitting the
  long docs (serving item "process checks", third bullet) is churn that conflicts with concurrent
  work. Both stay in the issue.
- **No GPU on the build host.** Everything is compiled here (sm_86, CUDA 12.8) and verified by
  CPU-only tests wherever the logic can be separated from the device: host policies, codecs,
  quantizers, planners and exact CPU emulations of kernel arithmetic against the FP64 or exact
  oracle that `docs/maintainer/op-development.md` requires. GPU tests are written against the same
  oracles and labelled `gpu`; they run later. Nothing here may be described as measured or
  verified on a GPU.
- **A GPU verification script** (`tools/bench/basalt_gpu_verify.ps1`, plus a Linux twin if the box
  is Linux) runs the gpu tests of the touched areas, the quality gate per numerics option, and an
  interleaved A/B per speed option, under the GPU lock.

## Findings (issue #383)

### Summary

[Basalt](https://github.com/jesdga95/basalt) (MIT) is a fork of Strata that runs only Qwen3.8-Flash-Next and only on Blackwell (sm_120). It has an optional second GPU tier, an OpenAI/Anthropic server and a [community benchmark site](https://jesdga95.github.io/basalt/). Its [packs are on Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt). It is moving fast: 219 commits since 2026-10-18.

This issue records a read of Basalt against our code (2026-10-25) so the findings aren't lost. The method:
- We read Basalt's docs and code, and our matching code, side by side.
- The claims that matter most were checked in both trees.
- Nothing was built or run, and no GPU was used.

**Every gain below is an estimate** from code reading or from Basalt's RTX 5090 numbers. None has been measured on a 3090.

### How far ahead they are, and why

- **Community figures.** A 24 GB RTX PRO 4000 Blackwell at 145 W, with less memory bandwidth than a 3090, decodes the larger IQ3_S at 187 tok/s. A 16 GB 5060 Ti gets 112 tok/s. Our 3090 does about 80 tok/s plain and 96-112 tok/s with MTP on IQ2_XS.
- **These are not like for like.** Their community figure is a cold first request with an 8-token draft ceiling, mixed with a structured-output prompt that drafts about 2x better than prose. The fairer yardstick is their prose-only number: 341 tok/s IQ3_S on a 5090 with a 5060 Ti peer.
- **Most of the lead is technique**, and most of it works on sm_86.
- **They don't run 8-token windows.** `--spec 7` is a ceiling; an adaptive rule (below) trims windows to about 2.2 drafts. Each draft costs them about 21% of a plain step against about 34% for us. That is flatter, but not radically.
- **Their routed-expert kernels are not at the byte floor either.** They reach about 52% of bandwidth at T=3; we reach about 28-30%. A persistent MoE kernel lost for them too, as it did in our PR 11.

### Flash-Next, ranked by estimated gain for the effort

### 1. Int8 copies of the hyper-connection matrices, activations kept fp32

**Basalt:**
- Quantizes `down`, `inject` and `up` once at load to int8 with one fp16 scale per 32 inputs: `s = f16(absmax/127)`, codes clamped to ±127 (`src/kernels/cuda/hc_fused.cu:1204`, `hc8_quant_kernel`).
- The down and inject reads convert the codes to float exactly and run fp32 `fmaf` against fp32 activations. The activation is never quantized (`hc_fused.cu:486-530`).
- `up` splits the fp32 activation into an fp16 hi/lo pair: two m16n8k16 mma with fp32 accumulate, flat for T=1..8 (`hc_fused.cu:664-718`).
- A simpler three-kernel version that uses no TMA is in `hc8_gr.cu`; that is the portable template.

**Ours:** BF16 weights, about 50% of bandwidth, 2.8 ms per token (`src/ops/hyper_connection/hyper_connection_mix_fused.cu`). We dropped Infernix's Q8 weight form (`include/ninfer/ops/hyper_connection.h:5`).

**Estimate:**
- Bytes drop to 0.54x (13.2 MB → about 7.1 MB per mixer call), saving about 1.3-1.8 ms per token (+11-17% plain decode).
- Dropping the BF16 copies would return about 0.6 GB, which is about 400 more expert frames.
- Low to medium effort.

**Risk:** Basalt shows that quantizing the hc *activations* costs about 1% in the mixed stream (`docs/precision/quality.md`). It has no same-engine A/B of int8 against BF16 *weights*, so this needs our own KLD gate (cross-model item A).

### 2. MTP for constrained (JSON and tool) requests

**Ours:** any constrained row turns MTP off (`src/models/qwen4_exp/program/program.cpp:3902`).

**Basalt:** checks the grammar on the host after each verify window. At the first refused row it redraws that row with the mask added as a bias, then ends the window there. A window the grammar accepts costs nothing extra (`src/program/generate.cpp:8976-8995`, `src/core/verify.cpp:4342-4372`).

**What to adopt:**
- This is exact for greedy rows: use it there.
- For sampled rows, use our per-position masks. Basalt reuses the same Philox counter for the redraw, which biases sampled redraws, and its draw-then-check is not our mask-then-top-k semantics.

**Estimate:** about 81 → about 110 tok/s on greedy constrained Flash-Next requests.

### 3. Adaptive draft length by joint acceptance

**Basalt:** `include/basalt/core/draft_cum.hpp` (about 40 lines).
- Draft j stays in the window while `prod_{i<=j} acc(p_i) >= th`.
- `acc()` is a fixed table mapping draft probability to acceptance: 0.12 below p 0.3, rising to 0.96 at p 0.99 and above.
- `th` starts at 0.35. `DraftCumAdapt` moves it toward 0.25 when observed acceptance beats the table, with a decay of 0.97 per window.
- The device evaluates the same products so the drafter stops early (`verify_kernels.cu:824-827`, `src/core/mtp.cpp:1390-1453`).

**Ours:** `choose_draft_length` (`program.cpp:3881-3896`) picks K before drafting from per-position EMA acceptance, with no per-token signal. PR 13 found it no better than fixed 2 drafts.

**Estimate:** +3-8% on Flash-Next, +1-3% on 27B/35B MTP3. Bit-exact, since only what is proposed changes.

### 4. A cheaper verify-window MoE

- **Shared expert beside the routed experts.** Basalt runs it on a side stream at high priority, forked and joined by events (`src/core/verify.cpp:274-283, 2446-2580`). Ours runs after the routed experts (`src/models/qwen4_exp/execution/forward.cpp:581-590`). Estimate about 0.25-0.5 ms per token. Bit-exact, since combine order is unchanged.
- **Quantize the FFN input to q8_1 once, upstream.** Ours re-quantizes BF16 x inside every gate/up CTA (`src/ops/offloaded_sparse_moe/cuda/moe_layer.cu:514-525`).
- **Gate/up kernel shape** (`gu2.cuh`, `dec_moe.cuh:464-590`):
  - Two rows per warp, with weights loaded straight into registers and no shared-memory staging.
  - The third pass of an 80-sub-block row is balanced across half-warps. Ours idles lanes 16-31 there (`moe_layer.cu:449`).
  - A co-resident grid of `(SMs - SMs/8) x 2` blocks.
  - Ours grows shared memory with NC (about 29 KB at NC=1, about 50 KB at NC=8), so occupancy drops exactly when verify needs it. Estimate 10-25% of gate/up at W ≥ 2.
- **Plan the window's expert union inside the gate/up launch** (the "glue" path, `dec_moe.cuh:452-545`). Basalt runs about 5 launches per layer; we run 8-11 (router, route, dispatch, stage, gate_up, down, shared x3, combine). Estimate about 0.2-0.4 ms per token.
- **CPU leg for misses during verify.**
  - Basalt never streams expert weights over PCIe in decode (`include/basalt/kernels/dec_moe.hpp:1-11`). A route kernel publishes CPU entries to a mapped ring; a host thread computes them while the GPU runs its resident entries; combine waits on a flag.
  - Our CPU route exists (`cpu_plan_kernel`, `cpu_wait_kernel`, `miss_service`) but is not wired in decode. At T=1 PR 11 showed it cannot win.
  - At W ≥ 2 the GPU's expert work per layer is about 130-170 us, so one or two CPU misses at 67-84 us could hide behind it.
  - Our CPU engine is bitwise equal to the GPU (PR 7). Basalt's isn't.

### 5. GDN: drop the head-tiling copies

**Basalt:** `dec_gdn_rec.cu` fuses conv1d+SiLU, the q/k L2 norm, the alpha/beta GEMV, the recurrence, the gated norm and the q8_1 output. Value head h reads key head `h % h_k` through a head map, so tiled q/k are never materialized.

**Ours:** about 11 nodes per GDN layer, including 6 `cudaMemcpy2DAsync` (`forward.cpp:71-77, 427-428`), which make the 216 copy nodes at 0.25 ms. `src/models/qwen3_5/execution/gdn.cpp:110-180` already has fused ops to build on.

**Estimate:** 0.4-0.7 ms per token.

**sm_86:** Basalt uses 4-block clusters and DSMEM, so on sm_86 it becomes one block per value head.

### 6. Int8 tensor-core GEMV for the dense layers (`dense_tc`)

**Basalt:** `src/kernels/cuda/dense_tc.cu`.
- 16-row by 8-column tiles, one `mma.m16n8k32.s8` per 32-value block, with an exact int32 dot per block and a fixed summation order, so a column is bitwise independent of T.
- Weights are staged with 16-byte `cp.async`.
- IQ3_S is decoded from shared memory with a branchless sign trick (`:96-101`).
- Formats: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0, Q8_0.

**Ours:** IQ3_S runs at about 42% of bandwidth, with about 9 sub-word global loads per 32 values (`ggml_blocks.cuh:190-213`). T=8 costs 30.9 us against 20.2 us at T=1.

**Estimate:** 0.3-0.5 ms per token, plus a flatter verify.

**sm_86:** the instructions exist on sm_80, but Basalt's code is wrapped in an sm_90+ guard that we would remove.

### 7. Hide the n-gram row read

- **Keep-alive:** a read every 40 ms from table open; ours is 100 ms (`program.cpp:686`). The drive sleeps after about 100 ms idle. One-line change.
- **Draft-row prefetch:** prefetch the draft tokens' n-gram rows into a row cache while drafting (`mtp.cpp:1395-1403`).
- **Flag wait:** one graph per round, with a kernel spinning on a host-mapped flag (`verify_kernels.cu:1075`), instead of our graph split around a synchronous read (`program.cpp:2722-2731`). This needs testing under WDDM.

### 8. Bigger quants through partial RAM residency

**Ours:** every expert is pinned in RAM, or startup is refused (`src/models/qwen4_exp/memory_plan.h:4`).

**Basalt:** the RAM copy holds only the experts not in VRAM, filled hottest first (`expert_source.hpp:105-125`).

**What this enables:**

| Pack | Experts | KLD vs Q8_0 | Fit |
|---|---|---|---|
| IQ3_XXS | 40 GiB | 0.24 | 64 GB box |
| IQ3_S | 46.8 GiB | 0.15 | 64 GB box, probably |
| IQ2_XS (ours) | — | — | 32 GB box becomes viable |

It also needs the IQ3_XXS/IQ3_S/IQ4_NL expert record formats.

**Effort:** high. Evictions need a VRAM-to-RAM copy before a frame is reused. WDDM caps pinned registration at about `RAM/2 - 8 GiB`.

### 9. CPU share for short prefill chunks

**Basalt:** below 6,144 tokens per chunk, non-resident experts with at most 8 rows go to the CPU pool: round(0.6 x the layer's streamed experts), fewest rows first, concurrently with GPU staging (`src/prefill/prefill.cpp:2155-2180`).

**Ours:** 1K prefill runs at half our 8K rate (1,260 against 2,452 tok/s) because staging dominates. Our CPU engine is bit-exact, so the share costs no quality.

**Estimate:** +10-25% on prompts of 2K tokens or less.

### 10. Small items

- **Ship a default expert state** so a fresh install doesn't start cold. Basalt ships a routed-frequency profile in the pack (`tools/convert/model_sections.py:540-562`).
- **Keep only hot embedding rows in VRAM.** Basalt keeps the first 8,192 ids in VRAM and the rest in mapped host memory. That frees about 330 MB, about 220 frames.
- **Async GDN commit with a finisher kernel** (`verify_kernels.cu:358-450`), about 0.1-0.25 ms per MTP round.
- **Fold the previous inject, the router and top-10, and the q8_1 output into one hc read** (`hc_fused.cu:970`). Do this after item 1. About 0.3-0.5 ms.
- **Don't pin CPU expert workers.** Basalt removed pinning after a preempted pinned worker stalled the whole join (commit d5273cb); ours pin (`expert_team.cpp:48-58`).

If the estimates hold, items 1 and 4-7 together target roughly 2.5-4 ms of our ~12.5 ms Flash-Next token.

### Across all three models

- **A. Decode-path quality gate (KLD and top-1).** This is the highest-value process item.
  - `ninfer-perplexity` scores only prefill (`apps/perplexity/main.cpp:79`), so changes to decode kernels, MTP/DFlash verify and the KV-write paths can't be seen by it.
  - Basalt prefills half of each 512-token wikitext chunk and scores the second half through decode, against a llama.cpp Q8_0 `.kld` base (`docs/precision/quality.md`).
  - Basalt found about 0.01-0.02 KLD of noise in the reference itself, so compare builds with each other.
- **B. Two-stage verify head (27B first).**
  - Every row is scored from a 2-bit copy: 4 levels and an fp16 scale per 32 weights, the scale raised by `sqrt(|w|^2 / <w, w_hat>)`, on int8 mma.
  - Every row within 4.0 of a lower bound on each column's 64th-best logit is rescored with the exact kernel's bits (`src/kernels/cuda/head_proxy.cuh`, `q6k_head.cu`).
  - Results: greedy unchanged, top-64 recall 99.97%, KLD +0.0017.
  - For the 27B: about −1 ms of a 25.7 ms C1 round (about 4%), for about 400 MB of VRAM. Unlike `--lm-head-q4`, the top rows stay exact.
- **C. Relaxed drafting (opt-in, off by default).** Under sampling, keep a draft whenever the target gives it probability ≥ P. Never for greedy, `logit_bias`, grammar, special tokens or the inside of tool calls.
  - Basalt measures +4-10% on sampled decode at P = 0.5, at 0.03-0.04 total-variation distance per token (`docs/precision/features.md`).
  - Ours accepts with p(draft) (`include/ninfer/ops/speculative_round.h:95-127`).
- **D. Draft-head context rows (27B/35B).** Add the head rows of the prompt's and the generated text's own tokens to the reduced draft vocabulary (`mtp.cpp:1310-1362`). This targets exactly the acceptance we lost when we tried a 40,960-row proposal head.
- **E. Drafter self-distillation.** Basalt ships a "self_distilled" MTP dense layer for its low-bit packs, but no training code or figures. Research item.

### Serving and process

- **A randomised soak harness.** Basalt's `tools/serve_soak.py` mixes OpenAI and Anthropic streaming, cancels, near-limit and over-long prompts and images, while sampling `/slots`, VRAM and RSS. We have nothing like it.
- **Bench rows stamped with the hardware state.** Read the enforced power limit through NVML at the start and end of a run. Add a frozen `--community` preset with a JSON row format and a CI format check.
- **Security.**
  - Validate the Host header and check Origin when there is no key (DNS rebinding).
  - Warn when listening beyond loopback without `--api-key`.
  - Compare keys in constant time. Ours uses `==` (`src/serve/http_server.cpp:216, 418`).
- **A `/slots` endpoint.** Per-lane state, progress, context use, tok/s and drafts. `docs/serving.md` says it is not served.
- **A small local dashboard page** fed by our existing Prometheus `/metrics`.
- **`--config` file.** CLI > file > defaults, unknown keys refused, the API key refused in the file, one startup line naming where each setting came from. It would make the launcher profiles data instead of duplicated `run.sh` / `run.bat` logic.
- **Cache-affinity overtake in admission.** A waiting request may jump the queue only to resume a lane that holds its prefix, and each older waiter can be passed at most 4 times.
- **Process checks:**
  - A ratcheting check on direct `fprintf(stderr` use.
  - A greedy-identity tool between two builds.
  - Splitting very long docs (`docs/serving.md` is 2,449 lines).

### Don't copy

- **Their full-layer fused kernel.** It was 11% slower (324.6 → 288.2 tok/s) and was retired (`docs/unified-kernel.md`). The causes were cross-CTA dependency latency, 1.5 MB of SASS stalling on instruction fetch, and a standalone harness that predicted 3-10x the gains the engine showed.
- **`nvcc --split-compile`.** About 4x the register spills, −9%.
- **A persistent MoE kernel, or int8-MMA gate/up.** The first matches our own PR 11 result; Basalt keeps the second off by default.
- **Blackwell-only features:** TMA, PDL/griddepcontrol, clusters/DSMEM, FP4 mma.
- **Their "no .bat" and "one fast path, no tuning flags" rules.**

### Where we're ahead

- Prometheus metrics.
- `/v1/load` and the multi-model router.
- The Responses API and `structured_outputs`.
- A paged-KV context cache with snapshots and a disk store.
- The interleaved A/B tooling.
- A C++ in-process server rather than Python over a subprocess.
- An FP64 oracle rule for every Op.
