# Inkling-2-Mini (`inkling2`, looped) model

Status: **text MVP, correctness first**. `src/models/inkling2` runs the text model on the GPU with
naive FP32 kernels and matches the llama.cpp golden ("Engine (text MVP, correctness first)" and
"Validation" below); the DSpark drafter and speculative decoding (greedy and sampled) are in ("DSpark drafter" below, measured on
a physical Tesla V100); so is the vision tower and image prompts ("Vision tower" below). The public Engine and
`ninfer-serve` run it with its chat template, sampling and output decoder ("Serving" below). This page is the mathematical authority for the work, plus the GGUF
mapping, reuse map and V100 plan. The
converter's config and tensor map are `tools/convert/inkling2.py`, the NumPy reference is
`tools/inkling2/reference.py`, and the C++ op oracles are `tests/ops/inkling2_reference.h`.

Thinking Machines' Inkling-2-Mini (`Inkling2LoopForConditionalGeneration`, `model_type`
`inkling2_loop`, released 2026-11-20) is a 24-layer MoE decoder with three passes over its middle 12
layers. Total parameters are 26.2B, with 3.4B active per layer pass (5.4B per token with three
passes). It takes image input through a 23-layer ViT and ships a 3-layer DSpark block drafter.

## Sources

Line numbers below refer to these exact revisions. Everything was read, nothing was executed.

| Tag | Source | What it is authoritative for |
|---|---|---|
| **[L]** | `unslothai/llama.cpp` @ `8aa2f23f6bbe0b2d23eb3ff676f318f96111d591` (PR #390 head: upstream Inkling 2 ggml-org#32217, looped layers #389 and DSpark ggml-org#32218), `src/models/inkling2.cpp` | the **loop** (the only published implementation), GGUF names and keys, everything below |
| **[Lg]** | same revision, `src/llama-graph.cpp` (`build_moe_ffn`, `build_attn` with relative positions), `src/llama-kv-cache.cpp:2010` (`set_input_relative_pos`), `src/llama-hparams.cpp:366-383` (`loop_layer`, `loop_visit`), `src/llama-model.cpp:1690-1712, 2031-2037` | routing, SITU, relative-position gather, visit mapping |
| **[Lc]** | same revision, `conversion/inkling2.py` | HF -> GGUF renames, w13 de-interleave, NVFP4 repack, tap flips |
| **[Ld]** | same revision, `src/models/dspark.cpp:946-1125` | DSpark drafter |
| **[H]** | `huggingface/transformers` @ `4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8`, `src/transformers/models/inkling2/modeling_inkling2.py` | every non-loop op; has **no** loop |
| **[Q]** | `thinkingmachines/Inkling-2-Mini` `config.json`, `hf_quant_config.json`, `tokenizer.json` | dimensions, stored precisions, tokenizer |
| **[Lv]** | same revision, `tools/mtmd/models/inkling2v.cpp`, `tools/mtmd/mtmd-graph.cpp` (`build_vit`, `build_rope_2d`, `build_patch_merge_permute`, `resize_position_embeddings`), `tools/mtmd/mtmd-image.cpp` and `mtmd-model.cpp` (`inkling2v`) | the vision tower and its preprocessing |
| **[V]** | `vllm-project/vllm` `recipes/models/thinkingmachines/Inkling-2-Mini.yaml` | prose confirmation: "12 middle layers run three passes per token, each with its own KV cache and gated messages from the previous pass" |

sglang's `inkling2.py` and the HF code have no loop. vLLM's loop implementation is only in its
`inkling-2-mini` Docker image. **The loop math therefore has one source, [L], which is an open,
unmerged PR.** Phase 2 must check it against a second implementation (see "Validation").

[L] and [H] agree on every non-loop op checked: clip, relative logits, convolutions, sinks, attention
residuals, routing, SwiGLU softcaps, final softcap and vocabulary trim. The older Unsloth pin
`26a2a424` (`inkling2` branch) is also consistent: it folds the convolution residual and scale into
the taps, which is algebraically the same thing.

## Shape

| Quantity | Value |
|---|---|
| hidden H | 2304 |
| physical layers / layer visits | 24 / 48 (layers 6..17 three times) |
| attention: query heads / KV heads / head dim | 24 / 8 / 128 (GQA 3), identical in local and global layers |
| local (sliding) layers | all except physical 5, 11, 17, 23; window 1024 |
| global visits | physical 5, 11 (x3), 17 (x3), 23 = **8 global KV caches**; 40 sliding |
| relative bias | `d_rel` 16 per head; extent 1024 (global) and window 1024 (local) |
| MoE | 128 routed experts, top 8, no shared expert, intermediate I 1152 |
| short convolution | kernel 4 (3 taps of history) on K, V, attention output and MoE output of every layer |
| attention-residual block | 6 layers (boundaries at physical 0, 6, 12, 18) |
| loop | core [6, 18), 3 passes, controller rank 64 |
| vocabulary | 201024 padded rows, 200058 real tokens (`unpadded_vocab_size`) |
| position encoding | **none** (no RoPE). Positions enter only through the relative bias and log scaling |
| eps | RMSNorm 1e-6 everywhere, including the unweighted norms |

## Layer visits

A *visit* `v` in `0..47` is one application of a physical layer `p(v)` ([Lg] `loop_layer`):

```text
p(v) = v                      for v < 6
p(v) = 6 + (v - 6) mod 12     for 6 <= v < 42      pass(v) = (v - 6) div 12
p(v) = v - 24                 for v >= 42
```

Weights belong to the physical layer. **Every visit has its own KV cache and its own convolution
state** ([L] `load_arch_tensors` and the hparams indexed by visit; [V]). The local/global pattern
and head counts are per visit in the GGUF (`sliding_window_pattern` has 48 entries).

## Forward, in order

Notation: `x` is one token's FP32 column; `u(x) = x / sqrt(mean(x^2) + 1e-6)` is the unweighted RMSNorm;
`n_g(x) = g ⊙ u(x)` is the weighted one. All matrices are `y = W x`.

### 1. Embedding ([L] 298-303, [H] `Inkling2NormedEmbedding`)

```text
cur = n_{embed_norm}(E[token])          # embedded image features (multimodal rows) skip this norm
blocks = []                             # completed residual blocks (the "depth memory")
```

### 2. Log scaling of positions ([L] 293-296, [H] FullAttention.forward)

```text
tau(pos) = 1 + 0.1 * ln(max(1, (pos + 1) / 8192))      # FP32; tau(1M-1) = 1.485
```

It multiplies q and r of **global** layers only. Local layers do not scale ([H] SlidingAttention
has no tau).

### 3. Per visit `v` (physical `p`, pass `k`)

**3a. Loop pass bookkeeping** ([L] 307-351). Only for core visits (`6 <= p < 18`):

```text
if p == 6 and k == 0:  prelude_blocks = blocks; prelude_cur = cur       # before layer 6 pushes
if p == 6 and k  > 0:  bank      = blocks[len(prelude_blocks)+1 :]      # the previous pass's core block outputs
                       bank_last = cur                                  # the previous pass's final output
                       blocks = prelude_blocks; cur = prelude_cur       # every pass restarts from the prelude
if k > 0:
   j = p - 6
   (m_attn, m_mlp) = MIX([bank..., bank_last], Qloop[2j], Qloop[2j+1])         # loop_res_query rows
   (m_k,    m_v  ) = MIX([bank..., bank_last], Qkv  [2j], Qkv  [2j+1])         # loop_kv_res_query rows
```

With 12 core layers and blocks of 6, `bank` is the one block pushed inside the previous pass (the
output entering physical 12), and `bank_last` is that pass's output after physical 17. Each MIX
therefore runs over 2 states.

**3b. Attention inputs by depth attention** ([L] 365-374, [H] `Inkling2BlockAttentionResidual`)

```text
MIX(S, q_1..q_n):  for each query q_i (a learned FP32 vector of H, the same at every position):
   s_j  = u(S_j)                                  # each source normalized, no gamma
   a_ij = softmax_j( <q_i, s_j> / sqrt(H) )
   out_i = sum_j a_ij s_j                         # values are the NORMALIZED sources (attn_res_normalize_values)
```

MIX is permutation invariant over its sources ([L] concatenates them in reverse order).

```text
if p == 0:   xq = xk = xv = cur                                          (no attn_res_query on layer 0)
local p:     xq = xk = xv = MIX(blocks + [cur], A_p)                     A_p: 1 query
global p:    (xq, xk, xv) = MIX(blocks + [cur], A_p[0], A_p[1], A_p[2])  A_p: 3 queries
```

**3c. Loop messages enter before the input norms** ([L] 376-382, `control` 227-237). For pass `k > 0`:

```text
CTRL_m(c, msg) = (1 - tanh(F_m h)) ⊙ c + tanh(G_m h) ⊙ msg,   h = D_m [u(c); u(msg)]
     D_m: [64, 2H],  G_m, F_m: [H, 64],  m in {attn, k, v, mlp}
xq = CTRL_attn(xq, m_attn);  xk = CTRL_k(xk, m_k);  xv = CTRL_v(xv, m_v)
```

`u` normalizes `c` and `msg` separately. Note `1 - tanh(.)` lies in `[0, 2]`, so it is not a
convex gate.

**3d. Input norms** ([L] 384-395)

```text
local:  xq, xk, xv = n_{attn_norm}(each)     # one norm weight; with messages the three inputs differ
global: xq = n_{q_inp_norm}(xq); xk = n_{k_inp_norm}(xk); xv = n_{v_inp_norm}(xv)
```

**3e. Projections and short convolutions** ([L] 399-404, 239-283; [H] 236-340)

```text
q = Wq xq            [24*128]       r = Wr xq [24*16]       g = Wgate xq [24*128]      t = Wsink xq [24]
k~ = Wk xk           [8*128]        v~ = Wv xv [8*128]
local:   k = SCONV(k~; wk),                       v = SCONV(v~; wv)
global:  k = DSCONV(k~; wk, Wkd xk),              v = DSCONV(v~; wv, Wvd xv)    Wkd, Wvd: [8*4, H]

SCONV(x; w)_t[c]     = ( x_t[c] + sum_{l=0..3} w[c, 3-l] x_{t-l}[c] ) / sqrt(4 + 1)
DSCONV(x; w, d)_t[c] = ( x_t[c] + sum_{l=0..3} (w[c, 3-l] + d_t[head(c)*4 + l]) x_{t-l}[c] ) / sqrt(2*4 + 1)
```

In the GGUF, tap 3 is the current token (`ggml_ssm_conv`). The HF `static_w[lag, c]` is
flipped into this order by [Lc]. `head(c) = c div 128`. `x_{t-l}` for `l > t` comes from the
visit's convolution state, which is zero at sequence start. The state holds the last 3
**pre-convolution** inputs of each of the four streams (K, V, attention output, MoE output).

**3f. Clips, value norm, scaling** ([L] 406-414; [H] 125-152)

```text
CLIP(y; thr) = y * min(1, thr / (||y||_2 + 1e-6))             per head vector
q_h = CLIP(sq_h ⊙ q_h; 16*sqrt(2))     sq: q_clip scale [24,128]
k_h = CLIP(k_h; 16*sqrt(2))            no scale
v_h = u(v_h)                           unweighted RMSNorm per head (use_value_norm_no_gamma)
r_h = CLIP(sr_h ⊙ r_h; 8)              sr: rel_clip scale [24,16]
global only:  q_h *= tau(pos);  r_h *= tau(pos)
```

The KV cache stores these final `k` and `v`. They are bounded (`|k| <= 22.63`, `rms(v) = 1`), which
makes FP16 and INT8 KV safe.

**3g. Attention with relative bias and a learned sink** ([L] 421-446, [Lg] `set_input_relative_pos`,
`build_attn`; [H] 154-177, 355-387)

```text
d = pos_query - pos_key
bias_h(d) = sum_c r_h[c] P[c, d]    if 0 <= d < 1024 else 0      P: [16, 1024] per layer (rel_bias)
s_hj = <q_h, k_{kv(h), j}> / sqrt(128) + bias_h(d_j) + mask_j     local mask: d >= 1024 excluded; causal
o_h  = sum_j exp(s_hj) v_j / ( exp(t_h) + sum_j exp(s_hj) )       sink t_h = (Wsink xq)_h, a zero-value logit
a    = Wo ( o ⊙ sigmoid(g) )
a    = SCONV(a; w_attn_post)
```

The relative bias is not scaled by `1/sqrt(128)`. For the local layers the extent equals the window,
so every visible key gets a bias. [L] implements the sink as `table - t` with a zero sink logit,
which is the same thing.

**3h. Residual and block boundary** ([L] 451-459; [H] decoder layers)

```text
if p mod 6 == 0:  blocks.append(cur_in);  ffn_inp = a           # a new residual block starts
else:             ffn_inp = cur_in + a
```

Here `cur_in` is `cur` at the start of the visit, after 3a.

**3i. MoE input** ([L] 461-467)

```text
z = MIX(blocks + [ffn_inp], M_p)                     M_p: ffn_res_query, 1 query
if k > 0:  z = CTRL_mlp(z, m_mlp)
z = n_{ffn_norm}(z)
```

**3j. Routed MoE** ([Lg] `build_moe_ffn`: SIGMOID gating, `norm_w`, `w_scale`, SITU; [H] 629-703)

```text
L = W_router z                     FP32, [128, H]
s = sigmoid(L)
S = top8(s + b)                    b: exp_probs_b (score-correction bias), selection only
w_e = 8 * s_e / (sum_{e' in S} s_e' + 1e-20)                for e in S
[g; u] = Wgu_e z                   Wgu_e: [2*1152, H], gate rows first (GGUF; [Lc] de-interleaves w13)
h_e = 4 tanh(g/4) * sigmoid(g) * 25 tanh(u/25)              softcapped SwiGLU ("SITU"): sigmoid sees the raw g
y   = sum_{e in S} w_e Wd_e h_e    FP32 accumulation
m   = SCONV(y; w_mlp_post)
cur = ffn_inp + m
```

For NVFP4 experts, `Wgu_e` and `Wd_e` are the GGML blocks times a per-expert F32 multiplier
(`ffn_gate_up_exps.scale`, `ffn_down_exps.scale`), applied to the product ([Lg]
`build_lora_mm_id(..., up_exps_s)`, [Lg] `llama-model.cpp:1942`).

In device memory the engine holds each expert record's blocks in a tiled order. At load,
`ops::nvfp4_tile_records` permutes the 36-byte blocks of each matrix so that one step of a decode warp is one
contiguous slab: 4 (gate, up) or 8 (down) rows, each run's next 4 x P blocks (`nvfp4_tiled_block`). Before, a step
read 8 separate 144- or 288-byte spans. The arithmetic does not change: golden byte-equal (781 files, 0 differ).
On the V100 the op bench (T 1, top 8) goes from 75.2 to 72.2 us and decode at a 1,024 context from 65.61 to
66.09 tok/s (ABBA).

### 4. Output ([L] 494-520, [H] 1293-1294 and 1371-1373)

```text
x = n_{output_norm}( MIX(blocks + [cur], output_res_query) )
logits = 30 * tanh( (W_head x) / 30 )      FP32
logits = logits[0 : 200058]                rows 200058..201023 are padding, never sampled
```

At the end, `blocks` holds 4 states: the embedding, the prelude output (pushed at physical 6), the
third pass's block entering physical 12, and the third pass's output pushed at physical 18.

## State per sequence

| State | Per visit | Total |
|---|---|---|
| global KV (FP16 K and V) | 8 heads x 128 x 2 x 2 B = 4 KiB/token | 8 visits: **32 KiB/token** |
| local KV (window 1024, ring) | 4 MiB | 40 visits: 160 MiB, fixed |
| convolution state (FP32, 3 taps) | (1024 + 1024 + 2304 + 2304) x 3 x 4 B = 78 KiB | 48 visits: 3.7 MiB |
| DSpark context KV (1 global, 2 local layers) | | ~4 KiB/token + 8 MiB |

There is no recurrent state other than the convolutions, so Snapshot/Replay works like the
Qwen3.5 GDN conv state, only smaller.

## Engine (text MVP, correctness first)

`src/models/inkling2` binds a converted artifact (`inkling2_gguf` recipe, UD-Q8_K_XL) and runs the
48-visit schedule for one sequence. It is the correctness baseline the V100 kernels of phase 4 are
measured against. The public Engine and `ninfer-serve` run it one request at a time by default (see "Serving"); `--max-concurrency N`
(N > 1, opt-in) batches the decode of up to 8 requests (see "Batched decode").
The DSpark drafter runs speculative decoding on greedy and sampled requests (see "DSpark drafter").

| Piece | Where | What it does now |
|---|---|---|
| Config, binding | `config.{h,cpp}`, `model.h`, `load.cpp` | `inkling2_text` config, every parameter by name, all device resident (22.9 GB, 8.6 s on the V100) |
| Program | `program/text_program.{h,cpp}` | `TextProgram::forward(tokens)`: the visit order, loop restarts and messages of "Forward, in order", any chunk size up to `max_chunk` (the engine's prefill chunk), positions up to `max_context` (8192); optional per-visit trace hook with llama.cpp's names; optional taps (`tap_visits`) and `keep(n)`, which cuts a forward of up to `max_rollback` tokens back to its first n |
| Drafter | `program/dspark_program.{h,cpp}`, `ops/dspark_fp32.h` | `DSparkProgram::extend(taps, n)` writes the context K/V of committed positions; `draft(anchor)` runs the block and the Markov chain (W_head W_o precomputed at start and stored as Q8; see "Compact drafter") |
| Dense projections | `ops/projection_fp32.h` | BF16 activations x BF16 weights, FP32 out; the head too |
| Element-wise and norms | `ops/fp32_rows.h` | FP32 RMSNorm (weighted or unit), add, sigmoid gate, tanh softcap, BF16 embedding gather |
| MIX, CTRL | `ops/depth_attention.h` | depth attention over at most 8 sources and 4 queries; the controller's combine `(1 - tanh f) c + tanh(g) m` (D, G, F are `projection_fp32`) |
| Short convolutions | `ops/short_conv.h` | static and per-head dynamic taps, FP32 history per visit and stream |
| Attention | `ops/relative_bias_attention.h` | per-head L2 clips and log scale, V unit norm, relative-bias sink attention over an FP32 KV ring per visit (local capacity window - 1 + max_chunk, global max_context); calls of at most 8 columns split their keys across blocks ([Split decode attention](#split-decode-attention)) |
| MoE | `ops/moe_situ.h` | sigmoid top-8 route with bias steering; SITU experts from `ggml_rec_nvfp4_nvfp4` records (layers 0-19) or the BF16 banks (20-23), with the per-expert multipliers |
| Vision | `program/vision_program.{h,cpp}`, `ops/vit_fp32.h` | `prepare_image` (pad, normalize, resize above 4096 tokens) and `VisionEncoder::encode`; `TextProgram::forward(tokens, embeddings)` puts an image's rows in place of its placeholder tokens (see "Vision tower") |
| Frontend | `frontend/frontend.{h,cpp}` | the chat template and tokenizer, images to prepared pixels, and the output decoder (see "Serving") |
| Engine core | `runtime/engine/inkling2_core.{h,cpp}` | one worker thread; by default requests one at a time from position 0; with `--max-concurrency N` > 1 a batched decode over N slots (`worker_batched`); host sampling (see "Serving") |
| Tokenizer | `qwen3_5/frontend/tokenizer.{h,cpp}` `Pipeline::Inkling2` | the word regex as a hand-written matcher, no normalizer, BPE with `ignore_merges` |

All Ops are naive FP32 kernels qualified against `tests/ops/inkling2_reference.h`. The KV rings are
FP32 (8 KiB per token per global visit, twice the FP16 figure above); the KV codec is phase 4.
Q8_0 experts (UD-Q4_K_XL) are not bound yet. Decode on the V100 runs at 20.6 tokens/s
(2026-11-20, 15 greedy steps at contexts 6 to 20), against a weight-streaming floor of about 155 tokens/s
(5.78 GB per token at the nominal 900 GB/s): phase 4 starts from there. On master `dad1623b` (2026-11-22) plain
greedy decode through `ninfer-serve` is 78.5-78.8 tokens/s (see "Measured" under "Serving").

## DSpark drafter ([Ld], `dspark-Inkling-2-Mini-Q8_0.gguf`, `general.architecture = dflash`)

Three dense layers (H 2304, 24/8 heads of 128, SiLU MLP of 9216). Each layer has query and key norms
and RoPE with theta 1e6 over the full head. Layers 0 and 2 are local (window 1024), layer 1 is
global. Attention within the block is non-causal (`dflash.attention.causal = false`). The block
size is 8, made of one anchor plus 7 drafts. The drafter borrows the target's `token_embd`,
`token_embd_norm` and `output`.

- **Taps:** `target_layers = [5, 23, 41, 48]` count **visits**. Tap `v < 48` is `cur` at the input
  of visit `v`: the block-local residual, not a MIX output, taken after the pass restart in 3a.
  Visit 23 is physical 11 in pass 1, and visit 41 is physical 17 in pass 2. Tap 48 is the final
  normed hidden state (step 4 before the head).
- **Context:** each draft layer `i` takes `c = n_{dflare_norm_i}( sum_t softmax(F[i,:])_t tap_t )`,
  where `F = dspark_fuse [4 x 4]`. Row 3 builds the RNN's `f_rnn` with `dspark_norm`. The context
  K is `rope(n(Wk_ctx c))` and the context V is `Wv_ctx c`, written into the draft KV per position.
- **Block forward:** the anchor is `n_{embed_norm}(E[a])`, and the 7 masks are `mask_embd`. Each
  layer is pre-norm attention (q, k: head RMSNorm then RoPE) plus a SiLU MLP. After a final norm,
  `base_i = W_head x_i` (not softcapped yet).
  The block attention (`ops::dspark_block_attention_fp32`) sees the whole context on the global
  layer. It splits the keys into at most 64 runs of 32-key tiles, each folded by an online softmax
  that one block computes for 16 of a KV head's (3 heads x W) queries, and then combines the runs.
  The keys are not bounded by shared memory: an earlier single-block form kept every score there
  and refused contexts past 23,040 keys.
- **Markov RNN head** ([Ld] 946-968, 1100-1110). The state starts at 0 and `prev` starts at the
  anchor. For `i = 1..7`:
  ```text
  j = Wj [state; W1[prev]; x_i; f_rnn(anchor)] + bj        Wj: [3*256, 2*256 + 2*2304], W1: [256, 201024] rows
  state = c + sigmoid(j0) ⊙ (state - c),  c = tanh(j1)
  logits_i = 30 tanh( (base_i + W_head (Wo tanh(j2) + bo)) / 30 )    Wo: [2304, 256]
  prev = argmax logits_i[0:200058], lowest id on ties
  ```
  This chain is sequential: 7 dependent steps, each reading `W_head`. Precompute
  `W_head Wo` ([201024 x 256]) and `W_head bo` once. That is exact in real arithmetic and turns
  each step's head read from H = 2304 columns into 256.

**Status (2026-11-21, run on a physical Tesla V100-PCIE-32GB, sm_70, FP32 kernels; see [tesla-v100.md](../tesla-v100.md)).** `ninfer_inkling2_dspark_test` checks that greedy speculative tokens
equal plain greedy tokens on the four golden prompts (64 tokens each), and they do. Drafts accepted per block of 7
match CPU llama.cpp at the same revision (`llama-speculative-simple --spec-type draft-dspark`):

| prompt | ninfer | llama.cpp |
|---|---|---|
| p0 "The capital of France is" | 15.7% | 10.1% |
| p1 Fibonacci docstring | 48.6% | 48.1% |
| p3 French narrative | 1.6% | 1.8% |

p2 is the chat prompt (13.9% in ninfer). The two greedy paths differ slightly, so the rates do too. On these
prompts, drafting all 7, speculative decoding is slower than plain decoding (36 against 74 tokens/s). A round
accepts about one draft. Verifying 8 tokens costs about 3 decode steps (43 ms), and drafting takes 9 ms.
Drafting fewer (`DSparkProgram::draft(anchor, n)`, llama.cpp PR #394's n_max) and the split Q8_0 projections
bring a round of n=2 to 4.4 ms drafting and 20.4 ms verifying (T=3), and the serving loop gates rounds on
measured cost; see "Serving".

**Compact drafter (2026-11-21).** The drafter changes only which tokens are proposed; verify runs the exact
target, so greedy output is identical whatever the drafter computes. It therefore reads its per-round weights in
compact, lossy forms:

- `P = W_head W_o` as Q8 (int8 codes with an FP32 scale per vocabulary row and 32 of R: 58 MB instead of 206 MB of
  FP32 per Markov step; `ops::dspark_markov_quantize`, `ops::dspark_markov_logits`, which also take BF16 and Q4);
- the base logits through a Q8 copy of the text head (`ops::quantize_q8_g32` at start, 492 MB of VRAM; 926 MB of BF16
  read per round becomes 492 MB);
- the block's projections with K up to 3072 (all but the MLP's down) as BF16 activations into
  `ops::projection_q8_fp32` over the artifact's Q8_0 values, and the joint (BF16) through `ops::projection_fp32`.

Measured (`ninfer_inkling2_dspark_test` at its defaults, 3 drafts and 128 tokens on the four golden prompts; `NINFER_INKLING2_SPEC_DRAFTS=all` drafts the whole block; V100):

| Drafter | draft ms a round | drafts accepted a round (p0 / p1 / p2 / p3) | speculative tok/s |
|---|---|---|---|
| FP32 P, BF16 head, split Q8_0 kernels | 4.82 | 1.11 / 2.53 / 1.15 / 0.12 | 70.1 |
| joint through projection_fp32 | 4.63 | same | 70.5 |
| P as BF16 / Q8 / Q4 | 4.41 / 4.33 / 4.38 | same (Q4: p3 0.13) | 71.1 / 71.3 / 71.5 |
| Q8 head copy | 4.55 | same | 70.8 |
| block projections through projection_q8_fp32 | 4.61 | same | 70.9 |
| **all (default): Q8 P, Q8 head, BF16 joint, Q8 block** | **3.64** | **same** | **73.1** |
| all, and the block truncated to the n + 1 positions drafted | 3.12 | 1.08 / 2.34 / 1.15 / 0.11 | 73.8 |

Serving (`ninfer-serve --spec dspark`, 12 greedy chat requests of up to 300 tokens, ABBA, `predicted_per_second`;
before = the exact drafter): 79.7 against 82.8 tok/s (+3.9%), 50.7% and 49.9% of drafts accepted, and the same text
on every request. A serve run's acceptance moves by a point or two from run to run on the same binary, and so does a
single request's, by up to a third. Over the same 12 requests, each form alone
(BF16 joint, Q8 or Q4 P, Q8 head, Q8 block) accepted 45.2-45.9% against 44.6% and 46.5% for two runs of the exact
drafter.

Q4 P is no faster than Q8 (the kernel is no longer bound by its bytes). Truncating the block is not used: it changes
the drafts' attention context and costs acceptance (p1, code, drops 7%).

### Draft floor and adaptive width

Two serving options change how many drafts a round verifies, never the output (verify still licenses every token).
Both are on by default under `ninfer-serve`/`ninfer-cli --spec dspark`; the Engine options default to off
(`SpeculativeOptions::draft_min_p`, `dspark_adaptive`).

- **`--draft-min-p P`** (default 0.5; 0 turns it off): `DSparkProgram::draft(anchor, n, pick, uniforms, &probs)` also
  returns the drafter's probability of each draft (`ops::dspark_draft_pick`), and the round keeps the drafts before
  the first one under P. For a greedy pick that is the argmax (the same token as `argmax_fp32`) and its softmax
  probability at temperature 1; for a sampled pick, the coupled draw and its weight over the mass the request's
  temperature, top-k, min-p and top-p keep (`DraftPick`). P only stops drafting: verify is unchanged either way.
  A round whose first draft is already under P runs as a plain step (a fallback step; its time, which includes the
  draft, is not used for either cost).
- **`--dspark-adaptive on|off`** (default on): `DraftWidth` (`inkling2_core.cpp`) picks each round's draft count
  from 1 to `--draft-tokens`. It keeps, per position i, an EMA of the chance that draft i is accepted once drafts
  0 .. i - 1 were, so a round of w drafts is expected to commit `1 + sum_{i<w} prod_{j<=i} s_j` tokens, and an EMA of
  each width's measured round time (an untimed width is priced on the line through the timed ones). The width with
  the most expected tokens per second wins against a plain step's one token per plain step time; every 8th round runs
  a neighbouring width so its cost and survival stay measured. The old gate's probes (8 .. 64 plain steps) remain
  for a request whose rounds stop paying. The idea is ninfer-all's `--adaptive-mtp` (67a92907) and its DSpark
  draft floor; the controller here is smaller (one request at a time, no batch).

Serving, 12 greedy chat requests of up to 300 tokens (code, lists, YAML, a haiku, SQL, a
translation, a bash script, a table, a limerick, a recipe, an explanation), `predicted_per_second` averaged over the
requests, one server per leg, the output md5 the same on every leg and configuration (V100, 2026-11-21):

| `--spec dspark` and | tok/s | drafts accepted |
|---|---|---|
| fixed 3 (before; ABBA A legs) | 81.83 / 82.02 | 50.1% / 50.3% of 2,095 / 2,098 |
| **`--dspark-adaptive on --draft-min-p 0.5` (default; ABBA B legs)** | **88.86 / 88.73 (+8.4%)** | 60.7% / 61.0% of 2,904 / 2,867 |

(An earlier ABBA of the same pair: 81.96 / 82.00 against 88.63 / 88.63, +8.1%.)

On master `dad1623b` (2026-11-22, after the expert-major MoE, the fused per-entry experts and the latency-hidden
decode attention), the same 12 requests, ABBA against a server without `--spec`, the output md5 the same on every leg:

| Server | tok/s | drafts accepted |
|---|---|---|
| plain | 78.69 / 78.45 | |
| **`--spec dspark` (defaults)** | **93.67 / 93.36 (+19.2%)** | 60.6% / 60.7% of 2,947 / 2,915 |

Two more legs of each on the adapter server (no adapter selected) gave 78.83 / 78.78 plain and 94.13 / 94.35 with
DSpark. By prompt type (two prompts each, 300 tokens, ABBA; plain 78.2-78.6 on all of them): maths 114.3-120.7
tok/s (75-77% of drafts accepted), code 105.5-115.4 (68-77%), lists 92.3-99.8 (56-65%), prose 81.2-88.2 (53-57%).

Single legs while choosing (the same requests; a leg moves by about 1 tok/s): fixed 3 81.4, min-p 0.1 81.7,
0.3 83.6, 0.5 86.8, 0.7 87.2; adaptive alone 85.4 and 86.1; adaptive with min-p 0.3 87.0, 0.5 88.4; `--draft-tokens 4`
and 5 with a fixed width lose (81.9 at min-p 0.5, 84.0 at 0.6), and with the adaptive width 5 ties 3 (88.8 at min-p
0.5) while 4 does not help (84.0). Per request the default gains on 11 of 12 (+3.7% to +20.5%) and ties the YAML
conversion (-0.9%), where fixed rounds already accept 72%. Most of the gain is rounds that the fixed gate turned
down: the adaptive width drafts 35% more tokens and accepts more of them.

ninfer's DFlash ([dflash.md](dflash.md)) already has the masked-block, non-causal, context-K/V shape
with separate context roles. What is new: per-layer softmax tap fusion instead of one feature
projection, taps counted in visits, the Markov RNN chain instead of per-column argmax, and global
attention in a draft layer.

### Sampled requests

A sampled request (temperature above 0, with any top-k, top-p, min-p, presence and frequency penalty) speculates
too, and its tokens are exactly the ones plain sampling draws under the same seed, bit for bit. That needs no
rejection step, because the host sampler (`sample()` in `inkling2_core.cpp`) already makes each token a
deterministic function of the target's processed logits at that position and the request's next uniform: it draws
one `std::uniform_real_distribution` value (one `generate_canonical` of its `mt19937_64`) per sampled token, in
order, and inverts the CDF over the candidates sorted by logit. A round's columns are verified one at a time
through that same loop: column c's token is sampled from column c's logits with the next uniform, and the round
continues only while that token is the draft the next column was computed from. So the uniforms are consumed by
emitted tokens only, never by drafts, and the logits of each column are the plain decode step's (the verify
forward computes each column as a decode step does; `ninfer_inkling2_engine_dspark_sampled_test` checks the
tokens). Forced tokens (a thinking budget's close) draw nothing, in both paths.

The drafts are coupled to those draws (the idea of ninfer-all's Gumbel-max coupled drafts, 2a743592, here with the
inverse CDF the sampler already uses, so the default sampling keeps its tokens): before a round the core copies the
generator and reads the next n uniforms, and the drafter draws each draft from its own logits with the request's
temperature, top-k, top-p and min-p and the uniform of that position (`ops::dspark_draft_pick`: the top-k by a radix
select, sorted, then the host sampler's arithmetic in FP64). Where the drafter's distribution matches the target's,
its draft is the token the target is about to sample, so it is accepted at the rate a greedy draft is; penalties are
not applied to the drafts (they only cost acceptance). `--draft-min-p` reads the drafter's probability of its draw
under those processors.

Serving, the same 12 chat requests at temperature 0.8 and top-p 0.95, each with its own seed, ABBA against a server
without `--spec`, `predicted_per_second` averaged over the requests (V100, 2026-11-21). The output md5 is the same
for plain and DSpark at every temperature, on every leg:

| temperature | plain | `--spec dspark` (defaults) | drafts accepted |
|---|---|---|---|
| 0.8 (ABBA) | 69.75 / 69.70 | **80.68 / 80.32 (+15.5%)** | 58.7% / 59.2% |
| 0.8, a fixed 3 drafts (one leg) | | 76.08 (+9.1%) | 50.4% |
| 0.6 (one leg each) | 69.70 | 80.78 (+15.9%) | 57.8% |
| 1.0 (one leg each) | 69.91 | 79.61 (+13.9%) | 61.7% |
| 0.8, master `dad1623b` (ABBA, 2026-11-22) | 74.92 / 74.90 | **83.47 / 83.37 (+11.4%)** | 57.6% / 58.3% |

Plain sampled decode is slower than plain greedy (about 70 against 72 tok/s; 74.9 against 78.7 on `dad1623b`): each token downloads its column's
logits and sorts on the host, and so does each verified column.

## Vision tower (`mmproj-BF16.gguf`, `clip`, projector `inkling2v`)

Sources: llama.cpp at the [L] revision, `tools/mtmd/models/inkling2v.cpp`, `mtmd_graph::build_vit`, and the
`inkling2v` preprocessor and loader in `tools/mtmd/mtmd-image.cpp` and `mtmd-model.cpp` ([Lv]). Also [H]
`Inkling2VisionModel` with its embeddings, rotary embedding and merger, at lines 858-1450. The two agree on every
step below. The tower is a ViT: 23 layers, width D 1024, 16 heads of 64, an MLP of 4096 with exact (erf) GELU,
and LayerNorm with eps 1e-5. All its weights take 0.6 GiB as BF16 matrices plus F32 vectors.

1. **Preprocess.** Pad the image at the bottom and right to multiples of 40 (patch 20 times merge 2).
   - The padding holds -1 before normalization. Every value then becomes `(v - 127.5) / 127.5` in FP32, so the
     padding ends up at -1.0078.
   - An image is resized only when its padded area is above 4096 merged tokens (6.55 MP). Its size then comes
     from llama.cpp's `calc_size_preserved_ratio`. llama.cpp resizes bicubically; ninfer resizes bilinearly.
2. **Patches.** The patches are 20 x 20 in raster order. Each is projected by the F32 kernel
   `v.patch_embd [1024, 3, 20, 20]`, with no bias. llama.cpp folded the checkpoint's two temporal slices into this
   one kernel, which is why it is F32.
3. **Positions and CLS.**
   - The learned table is a 32 x 32 grid. It is resized to the image's gw x gh patch grid bilinearly, with
     half-pixel centres and clamped edges (ggml `GGML_SCALE_MODE_BILINEAR`; [H] `align_corners=False`), and added.
   - The CLS token `class_embd + class_position_embd` goes in front.
   - Then comes `pre_ln`.
4. **Each layer:**
   ```text
   y = LN1(h);  q, k, v = W y + b
   q, k = rope2d(q, k)
   h += ls1 * (Wo attn(q, k, v) + bo)
   h += ls2 * (W2 gelu(W1 LN2(h) + b1) + b2)
   ```
   - `rope2d` works per head of 64. Dims 0..31 rotate by the token's x position and dims 32..63 by its y position.
     Each rotation is ggml's normal (pair-interleaved) RoPE over 32 dims, with theta 10000: pair i turns by
     `p * 10000^(-2i/32)`.
   - Patch (px, py) sits at (px + 1, py + 1) and CLS at (0, 0).
   - Attention is full (non-causal) with scale 1/8.
   - There is no post-norm.
5. **Merge and project.**
   - Drop CLS. Each 2 x 2 square becomes one token of 4096: value `e + D dx + 2 D dy` is patch
     (2X + dx, 2Y + dy).
   - Tokens are in raster order of the merged grid.
   - Then `mm.1` (4096 x 4096), GELU, and `mm.2` (2304 x 4096). Neither has a bias.
6. **Into the text model.** Each image token's embedding replaces the input row of one placeholder token.
   - The placeholder is `<|unused_200054|>`, [H] `image_token_id`.
   - The embedding norm does **not** apply to these rows: [H] scatters the image rows after `Inkling2NormedEmbedding`,
     and [L] normalizes token rows only.
   - Each image token takes one position.
   - The chat template writes `<|content_image|><|unused_200054|><|end_message|>`, and the processor expands the
     placeholder to the image's tokens. llama.cpp's mtmd writes the same block, with `<|content_image|>` and then the
     embeddings.
   - The GGUF records no RoPE base and no token limit. Both come from llama.cpp's loader (10000; 1 to 4096 tokens),
     and the converter writes them into the component config.

**Engine (FP32 kernels, `include/ninfer/ops/vit_fp32.h`, `program/vision_program.{h,cpp}`).**
- `prepare_image` does step 1 on the host.
- `VisionEncoder::encode` runs steps 2-5 with `vit_linear_fp32`, a 64 x 64-tiled GEMM over BF16 or F32 words that
  sums in K order, so each column is exact whatever T is.
- `TextProgram::forward(tokens, embeddings)` takes the image rows for step 6, and a chunk may split them.

**Status (2026-11-21, V100).** `ninfer_inkling2_vision_test` checks two images: llama.cpp's `test-1.jpeg`, decoded
to PNG (640 x 488, 208 tokens), and a synthetic 300 x 170 one (40 tokens).
- Embeddings: compared with an FP64 reference of the tower (`tools/inkling2/vision_reference.py`).
- The reference itself is within 0.6% mean (3.8% worst) relative error per token of llama.cpp's CPU embeddings,
  which pass BF16 activations to its matmuls. Swapping the RoPE to split-half gives errors of order 1, so this
  comparison does tell the variants apart.
- Prefill of an image prompt in chunks of 50 gives the same bits as one whole forward.

| image | tokens | encode | vs FP64, worst / mean token | vs llama.cpp, worst / mean | greedy = llama.cpp |
|---|---|---|---|---|---|
| test-1 640 x 488 | 208 | 0.091 s | 8.1e-5 / 6.8e-6 | 3.7e-2 / 6.7e-3 | 48 of 48 tokens |
| synthetic 300 x 170 | 40 | 0.034 s | 1.5e-5 / 4.1e-6 | 2.2e-2 / 6.1e-3 | 48 of 48 tokens |

The greedy continuations are llama.cpp's `llama-mtmd-cli` outputs, tokenized again. The prompt is
`<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_image|>`, then the
image, then `<|end_message|><|message_user|><|content_text|>Describe this image in one paragraph.<|end_message|>`. The
model reads the newspaper's front page: "The New York Times ... July 20, 1969 ...".

## Serving

`ninfer-serve model.ninfer [--vision]` serves an Inkling-2 artifact through the public Engine. The OpenAI,
Anthropic and Responses routes, streaming and usage are the server's own. The Engine picks the model's own
core when `models::inkling2::is_inkling2` accepts the artifact (`runtime/engine/inkling2_core.h`). The core
does not use the Qwen3.5 frontend, Program contract or scheduler.

**Prompt** (`models/inkling2/frontend`).
- The artifact's `chat_template.jinja` renders the conversation in the Hugging Face chat format.
  - Content is a string, or a list of text and image parts.
  - Tool calls carry parsed arguments, and tools are given as parsed JSON.
  - `chat_template_kwargs` are template variables, and `reasoning_effort` maps to the template's names.
  - Thinking off is effort `none`, which the template renders as "Thinking effort level: 0".
- The template renders no generation prompt; the model writes its own `<|message_model|>`.
- The Inkling-2 tokenizer encodes the text. The template's literal spans (the conversation's own strings) are
  never parsed as control tokens, so user text that spells `<|end_message|>` stays text.
- Each image's `<|unused_200054|>` becomes `PreparedImage::tokens()` copies, and the image is recorded at
  that column.
- Images are decoded by `media::decode` (FFmpeg) and prepared by `prepare_image`. They need `--vision` and
  an artifact converted with `--mmproj`.
- Refused requests:
  - continuing a final assistant message;
  - grafts and LoRA adapters other than the two opt-in forms of [Adapters](#adapters-opt-in-uncensoring)
    (a `'+'` combination of grafts too);
  - video;
  - output constraints (grammar, JSON schema) and `tool_choice` required or named;
  - logprobs;
  - a prompt that leaves no room for one output token (token counting still reports its length);
  - a stop token id outside the vocabulary.
- Tool calls are free generation: the default `tool_constraints` (`basic`) is not enforced, since the route has
  no grammar engine (output constraints are refused for the same reason). A call's name and arguments are what
  the model wrote; malformed JSON stays content. `parallel_tool_calls: false` ends the request at the first
  complete call (finish reason stop token, as the model's end token would).
- A server without `--vision` binds and uploads no vision tower, and one without `--spec dspark` no drafter.

**Output.** Control tokens switch the channel, as `tokenizer_config.json`'s `response_template` describes:

| Token | Effect |
|---|---|
| `<|content_thinking|>` | reasoning |
| `<|content_text|>`, `<|content_tool_error|>`, `<|content_invoke_tool_text|>` | content |
| `<|content_invoke_tool_json|>` | a tool call: `{"name", "args"}`, parsed at `<|end_message|>`; malformed JSON is kept as content |
| `<|message_model|>` | opens a header; its text, such as a tool's name, is not output |
| `<|content_model_end_sampling|>` | the default stop token |

- Text is published in whole UTF-8 characters.
- Text that could still become a caller's stop string is held back.
- A thinking budget closes the reasoning by injecting `<|end_message|><|message_model|><|content_text|>`.
  The budget applies only where the model's 2-token turn opening, the 3-token close and one answer token fit in the output limit.

**Execution** (`Inkling2Core`).
- One worker thread takes requests in submission order. Each request:
  - resets the `TextProgram`;
  - encodes its images, copying each encoding out, since a chunk can need the end of one image and the
    start of the next;
  - prefills in chunks of the startup's width ([Tensor-core prefill](#tensor-core-prefill)), one ending where
    prefix reuse keeps its snapshot;
  - decodes with the decode graph.
- Sampling runs on the host from the last column's FP32 logits (`unpadded_vocab_size`). It applies, in
  order: presence and frequency penalties, temperature, top-k (at most 20, as everywhere in the Engine),
  min-p, then top-p. Temperature 0 is argmax, lowest id first. A seed repeats its output.
- The defaults follow `generation_config.json`: temperature 1, top-p 1. Top-k is 20.
- `max_pending_requests` bounds the queue, and `pending_timeout` refuses a request that waited too long.
- `--max-concurrency N` (N > 1) decodes up to N requests together ([Batched decode](#batched-decode)). There is no context
  store or suspend.
- Prefix reuse (on unless `context_cache.enabled` is off or a request sets `allow_prefix_reuse` false): after the
  last whole multiple of the prefill chunk that leaves a position to compute, the core copies the state a forward
  from there needs into pinned host memory (`TextProgram::save_async`: every sliding visit's last 1023 K/V columns
  and every visit's convolution histories, 163 MiB; the global caches are indexed by position and stay valid). See
  "Snapshots in host memory" for how the copy runs beside the rest of the prompt. A next prompt whose first positions
  are the same (token ids, and images by a hash of their pixels) restores it and prefills only the rest, in the same
  chunks a prefill from 0 runs: every forward cut falls on a multiple of the chunk (`inkling2_prefill_plan.h`), so the
  resumed forwards are the cold ones call for call and, as the tensor-core routes dispatch on a call's width, bit for
  bit. Reuse is therefore chunk-grained (2048 positions at chunk 2048, the chunk the engine picks up to 262,144 with DSpark and vision; `--prefill-chunk 256` for finer reuse). With `--spec dspark` the drafter's sliding rings and anchor context are kept with it (`DSparkProgram::save_async`; of each ring only the positions its attention reads, at most the window); every request runs the drafter, so it resumes only from a snapshot whose drafter state is at the same position. Images wholly in the reused prefix are not encoded. The
  result reports `reused_prompt_tokens`. One snapshot is kept; a prompt that does not resume from it replaces it (or clears it, when no longer than one chunk).
- `--spec dspark` (up to 3 drafts, `--draft-min-p 0.5` and `--dspark-adaptive on` by default; see "Draft floor and
  adaptive width") speculates on every request, greedy and sampled (`Inkling2Core::run`):
  - A round drafts n tokens after the anchor, forwards `[anchor, drafts]`, and consumes each column
    whose input is the token the previous column chose; the first that differs ends the round, and
    `keep()` cuts the target back. Every token still goes through the plain loop, so the output is
    plain decoding's (greedy, or sampled under the request's seed).
  - A round drafts at most the output budget left after the anchor minus one (n drafts yield n + 1 tokens), so a
    round near `max_tokens` drafts fewer and none runs when one token is left; `drafted_tokens` counts what was drafted.
  - A round runs only while its cost (EMA) is below (1 + accepted drafts EMA) plain steps; plain steps
    hold their taps on the device for one deferred drafter extend, and a probe round follows 8, 16, ..
    64 plain steps.
  - Greedy requests without penalties take their tokens from a device argmax, one per column.
  - Sampled requests speculate too, and sample exactly what plain sampling does under the same seed; see
    "Sampled requests".
  - Measured (V100, 2026-11-21, `ninfer_inkling2_engine_dspark_test`, five chat prompts of 256 tokens, fixed widths): plain 65.0,
    n=1 73.3, n=2 80.6, n=3 83.3, n=4 81.9 tokens/s.

**Measured** on the V100 through `ninfer-serve`, master `dad1623b`, 2026-11-22 (250 W limit, peak draw 236 W,
PCIe Gen3 x4 eGPU, FP16 KV and tensor-core prefill; `gpu_guard.sh` saw no throttle or power capping). Best clean
run first, then the range over repeated or ABBA legs:

| What | Best | Range and conditions |
|---|---|---|
| Load (engine ready) | 9.2 s | 9.2-14.2 s over 23 starts; weights read at 1.7-2.5 GiB/s |
| Decode, `--spec dspark` | 120.7 tokens/s | a maths prompt, greedy, 300 tokens; 81.2-120.7 by prompt type (see "Draft floor and adaptive width") |
| Decode, `--spec dspark`, 12 chats | 94.35 tokens/s | 93.36-94.35 over four legs (greedy, 300 tokens) |
| Decode, `--spec dspark`, temperature 0.8 | 83.47 tokens/s | 83.37-83.47 (see "Sampled requests") |
| Decode, plain | 78.83 tokens/s | 78.45-78.83 on the 12 chats; 74.90-74.92 sampled at 0.8 |
| Prefill (`--max-context 131072`, chunk 2048) | 1,930 tokens/s at 7,979 tokens, TTFT 4.1 s | 1,578-1,629 at 1,098 (TTFT 0.67-0.70 s); 1,895-1,896 at 9,798 (5.2 s); 1,244 at 100,107 (80.5 s). With DSpark and vision on: 1,558-1,621, 1,900, 1,866-1,867 and 1,232 |
| Prefill (`--max-context 262144 --spec dspark --vision`, both adapters loaded, the default chunk 2048; 2026-11-22, branch `perf/inkling2-host-snapshots`) | 1,868 tokens/s at 7,979 tokens, TTFT 4.27 s | 1,865-1,868 at 7,979; 1,566-1,671 at 1,098 (0.66-0.70 s, the first request after startup the slowest); 1,613 at 34,701 (21.5 s); 1,219 at 100,107 (82.1 s). Before the snapshots moved to the host the engine picked chunk 256 here: 699-703, 782-786, 739-743 and 629-632 (see "Context with DSpark and vision") |
| Second turn of a chat (prefix reuse, same server) | TTFT 1.10 s at 7,977 tokens, 6,144 reused | against 4.26 s cold; 1.51 s against 21.4 s at 34,699 tokens (32,768 reused); the output equals the cold request's. The restore copies 198 MiB host to device in 62 ms |
| Decode after a long prompt (128 tokens, summarize) | 91.2 tokens/s, DSpark at 7,979 | DSpark 84.9-91.2 at 7,979, 88.0-89.8 at 9,798, 47.4-47.8 at 100,107; plain 71.5, 70.7 and 48.1 |
| Batched aggregate | 198.2 tokens/s at `--max-concurrency 8` | see "Batched decode" |
| Image prompt, llama.cpp's `test-1.jpeg` (640 x 488), 230 tokens | TTFT 375 ms (vision and prefill) | 375-398 ms over three requests at 262,144 (382-425 at 131,072); decode 100-105 tokens/s with DSpark |

As first published (2026-11-21): load 9.1 s, decode 71.3
tokens/s through the Engine, and the same image at 234 tokens in 0.088 s of vision and 0.58 s of prefill.

- `ninfer_inkling2_frontend_test` checks that a text turn and the image turn tokenize to llama.cpp's own
  prompt ids.
- Through the Engine, greedy on the image matches llama.cpp's continuation for 16 of 48 tokens. The same
  text Program matches it for 48 of 48 when it gets the vision test's pixels (`test1.rgb`). The difference
  is the JPEG decoder:
  - FFmpeg's pixels differ from `test1.rgb` by 0.73 levels on average and 14 at most;
  - the two continuations part ways after 16 tokens, and both describe the same front page.
- `curl` against `ninfer-serve`:
  - a greedy chat returns `"content": "Paris"`, with the reasoning in `reasoning_content`;
  - streaming sends reasoning deltas, then content;
  - the image request describes "MEN WALK ON MOON".

## Adapters (opt-in uncensoring)

Two published Inkling-2-Mini uncensoring adapters load at startup and apply only to the requests that select them
(`models/inkling2/program/adapters.h`). Nothing changes for any other request: a server without them, or a request
that selects none, launches exactly the kernels it launched before, and its output is bit for bit the same (the FP32
golden's 781 dump files are byte-identical, see below).

| | OpenUncensored LoRA | satellitedown per-visit ablation |
|---|---|---|
| Source | [OpenUncensored/Inkling-2-Mini-Uncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored), its PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`, 865 KB) | [satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine), `inkling-2-mini-abliterated.safetensors` (885 KB); reference code `abliteration/inkling_ref.py` (`Ablation`) and `ablate.py` |
| Licence | Apache-2.0 | Apache-2.0 |
| Flag | `--lora NAME=DIR` | `--graft NAME=FILE` |
| Request | `"lora": "NAME"` (`--default-lora`) | `"graft": "NAME"` (`--default-graft`) |
| Form | PEFT LoRA, r 2, alpha 2 (scale 1), F32, on `model.llm.layers.N.attn.wo_ud` for N = 4..23 | F32 `attn_dirs`, `mlp_dirs` [48, 2304], `attn_w`, `mlp_w` [48]; metadata `params.scope` `per_visit` |
| Applied | `a_in += scale * B (A gated)` after the attention output projection, FP32, every visit of the layer (all three loop passes of a core layer) | `y <- y - w_v r_v (r_v . y)` on the post-attention and post-MLP short-convolution outputs at visit v, FP32, before the residual add |

**The LoRA.** `wo_ud` is the artifact's `text/layers/N/attention/output`, BF16 [2304, 3072] (rows out, columns
in), the same bytes as the Hugging Face checkpoint's `model.llm.layers.N.attn.wo_ud.weight` (checked on rows 7 and
1500 of layers 4, 11 and 23). In the program it is `project(gated, ow, orows)`, whose result `a_in` is the input of
the post-attention convolution (3g). `low_rank_update_fp32` adds `scale B (A gated)` to `a_in` in FP32, reading
`gated` (BF16, the projection's own input). The adapter is not merged: a merged BF16 copy of `wo_ud` would cost 14 MB
per layer and drop the FP32 update to BF16. The publisher's own merged checkpoint is `BF16(base + B A)` (a research
script checked their baked `wo_ud` rows against that bit for bit), so the unmerged update differs from it only by that
BF16 rounding of the weights. `ninfer_inkling2_adapters_test` checks rows 7 and 1500 of layer 4's effective `wo_ud`
(`projection_fp32` and the update over the identity) against `base + scale B A` in FP64: 4.2e-8 of the row's largest value at worst,
and rounded to BF16 the same in 3072 of 3072 values of each row. Only `wo_ud` targets load; any other module, DoRA,
rank or alpha patterns, LoRA bias, `modules_to_save` and the other PEFT variants are refused with the reason, as is
a rank above 16.

**The ablation graft.** The reference hooks `post_attention_convolution` and `post_mlp_convolution`, whose outputs
are a visit's only writes into the attention-residual blocks, and keys the 48 directions on the visit index (layers
0-5, three passes of 6-17, 18-23: `TextConfig::physical`). In the program those outputs are 3h's `a` and 3j's MoE
convolution output:
- **Attention (3h).** Inside a block, the convolution writes `a`, `direction_ablate_fp32` removes the direction in
  place and writes `ffn_inp = cur + a'`. At a block's first layer (`p mod 6 == 0`) the convolution writes `ffn_inp`
  itself (`ffn_inp = a`, the residual restarts), and the ablation runs on `ffn_inp` in place; the block bank keeps
  `cur`, as before.
- **MLP (3j).** The convolution (with the expert combine) writes into `z`, dead once the experts ran, and the
  ablation writes `next = ffn_inp + y'`.
- A visit whose weight is 0 keeps the fused residual add. The convolution histories hold the convolutions' inputs,
  which the ablation does not touch, so rollback (`keep()`) and snapshots are unchanged.
- The loader takes F32 tensors of exactly this layout and `params.scope` `per_visit`, and normalizes each direction
  to unit length as the reference's `Ablation.set` does. It refuses a GGUF file (a control vector or KV graft is not
  an Inkling-2 graft), another scope, another shape or a non-finite value. The published file's weights are nonzero
  at 44 of the 48 attention visits and at every MLP visit.
- `ninfer_inkling2_adapters_test` ablates one visit's write at a time (weight 1 there) and compares the program's
  `ffn_inp` or `l_out` with the FP64 ablation of the base run's write: inside a block, at a block's first layer and
  for an MLP write, within 7e-8 relative; the visit before is the base run's bit for bit. Both Ops are also held to
  FP64 on random operands (`ninfer_inkling2_adapters_ops_test`, ranks 1, 2 and 16, strided columns), and with both
  adapters on the decode graph gives the eager forward's logits bit for bit.

**Selection and caches.** `Inkling2Core` loads the adapters before the weights (an unsupported one refuses to start),
the frontend resolves the request's names (`PreparedPrompt::lora`, `graft`; one graft per request, no `'+'`), and
`TextProgram::set_adapters` selects them for the request's forwards. A change of selection drops the captured
decode graphs, so each selection's decode step captures its own; the base model's graph is the one it always was.
Prefix reuse keys the snapshot on the selection (a digest of each adapter's tensors), as the dense models key their
cached prefixes on the LoRA digest: a prefix computed under one adapter, or none, never serves another.

**DSpark.** The drafter runs without the adapters: it only proposes, and the target, adapted, verifies every draft,
so greedy output is plain greedy decoding's with the same adapters. `ninfer_inkling2_dspark_test` with
`NINFER_INKLING2_LORA` or `NINFER_INKLING2_ABLATION` checks it (the same tokens on every golden prompt); the
acceptance rate can be lower than the base model's.

**A/B** (V100, `ninfer-serve` greedy). The refusals are counted on the first 20 prompts of `mlabonne/harmful_behaviors`
`test`, the set satellitedown reports, with `reasoning_effort` `none` and up to 1,024 tokens (the model still reasons
briefly first). The answer is judged by satellitedown's refusal markers on its first 300 characters, so an answer that
complies only partly or deflects counts as complied; a few adapted answers do. The KLD is each adapter's next-token
distribution against the base model's over 2,048 teacher-forced tokens of mixed text (prose, C++, Python and Q&A),
in the FP32 program (`ninfer_inkling2_adapters_test` with `NINFER_INKLING2_KLD_TEXT`). Every reply with
`--spec dspark` was the plain run's, byte for byte:

| | Base | `"lora": "ou"` | `"graft": "sd"` |
|---|---|---|---|
| Harmful prompts refused | 20/20 | 0/20 | 0/20 |
| KLD from base, mean (max) | 0 | 0.0142 (1.18) | 0.0171 (0.78) |
| Top-1 agreement with base | 1 | 0.969 | 0.962 |
| Math (4, low effort) | 4/4 | 4/4 | 4/4 |
| Python tasks (2, tests run) | 2/2 | 2/2 | 2/2 |
| Output tok/s, plain (`--spec dspark`), 2026-11-21 | 71.7 (77.1) | 68.9 (68.1) | 68.9 (67.8) |
| Output tok/s, plain (`--spec dspark`), master `dad1623b` | 78.83 / 78.78 (94.13 / 94.35) | 75.43 / 75.39 (91.65 / 90.84) | 75.15 / 75.15 (92.71 / 91.56) |

## Dense 8-bit re-encoding

**Lossy.** `--dense-q8 head|all` (`EngineOptions::dense_q8`, `ProgramLimits::dense`; opt-in, off by default: the
default runs the BF16 weights exactly) changes the target's logits (KLD below), so greedy output can differ from the
default's. It re-encodes BF16
dense weights at load with Q8_0's values (`ops::quantize_q8_g32`: per 32 values of a row d = amax / 127 as FP16 and
int8 codes, ggml's `quantize_row_q8_0_ref` rounding; 8.5 bits a weight; the artifact is unchanged).
`ops::projection_q8_fp32` runs them with BF16 activations and FP32 sums (oracle `ninfer_projection_q8_test`). `head`
re-encodes the LM head (926 MB of BF16 read per token becomes 492 MB); `all` also re-encodes every attention
projection (q, rel, gate, sink, k, v, the dynamic-tap projections and the output; 3.52 GB a token becomes 1.87 GB).
The experts are NVFP4 already, and the controllers, router and convolutions stay as they are.

Measured on the V100 (2026-11-21). KLD(BF16 || Q8) per position is teacher-forced over 3072 tokens of English
prose, Markdown and C++ (`ninfer_inkling2_golden_test` with `NINFER_INKLING2_DENSE_KLD`). "Program" is the golden
test's 127-step decode (three ABBA rounds). "Serve" is the mean over four greedy chat requests of 300 tokens through
`ninfer-serve` (ABBA, `predicted_per_second`).

| Form | KLD mean | median | p99 | max | argmax agrees | program tok/s | serve tok/s | serve, `--spec dspark` |
|---|---|---|---|---|---|---|---|---|
| BF16 (default) | 0 | 0 | 0 | 0 | 100% | 69.9 | 64.7 | 75.5 |
| `head` | 8.8e-6 | 3.9e-6 | 5.3e-5 | 1.2e-4 | 99.77% | 72.3 (+3.4%) | 66.7 (+3.1%) | |
| `all` | 6.3e-3 | 1.6e-3 | 7.6e-2 | 0.32 | 96.2% | 77.2 (+10.4%) | 71.0 (+9.7%) | 82.4 (+9.1%) |

Re-encoding any one attention group alone (q-side, k/v or output) already gives a KLD mean of 5-6e-3 (the loop
visits compound it). With `all`, one golden prompt's argmax differs from llama.cpp's. The Q8 kernels reach
550-850 GB/s against BF16's 610-870 GB/s, because a call costs about 5 us whatever its size, so `all` gains less
than its byte count suggests (1.35 ms of the 1.65 GB saved per token).

## Split decode attention

On by default; `--no-split-attention` (`EngineOptions::split_attention`, `ProgramLimits::split_attention`) turns it
off. A forward of at most `ops::kSplitMaxColumns` (8) tokens (decode, DSpark verification) gives each attention
column's visible keys to blocks of 256, one CUDA block each, with the sink in the first. The last of a column's blocks
to finish (an atomic counter per column and head) merges the blocks' maxima, denominators and sums in order, each
weighted by e^{m_z - max m}. It applies only once a column can see more than `ops::kSplitMinKeys` (512) keys: below
that the extra blocks cost more than they save. Decode replays one CUDA graph for positions up to 512 (the single
block) and another past them (split; its grid is sized for the whole ring, and the blocks past a column's keys return
at once). The single-block kernel walks
every key of a column in one block (then of 128 threads), so a decode step ran 24 blocks per visit on 80 SMs and
was latency-bound. At 16K context, 70% of the step was attention.

It is FP32 all the way, with no quantisation. Only the order of the softmax sums changes, so a column that sees more
than 256 keys gets other bits than the single-block kernel. A column of at most 256 keys gets the same bits: the
golden test is byte-identical (781 dumped tensors, 0 differ). Prefill chunks (more than 8 columns) keep the single
block. `--no-split-attention` reproduces the earlier builds exactly.

Inside a block (single or split) the kernel is bound by instruction issue and load latency, not bytes: a
256-key block moves 128 KB of FP16 keys and values (256 KB from FP32 rings). Since 2026-11-22 a block has 8 warps. Each warp issues the loads of four score steps
before their arithmetic. The value fold runs as four chains: key j below n & ~3 goes to chain j % 4, and the
remaining keys go to chain 0, each chain in key order. Warp w runs chain w % 4 over its lanes' dimensions, with
one load of two values per key (4 bytes from an FP16 ring, 8 from FP32) and the next 16 keys' loads in flight. The chains meet in shared memory as
(0 + 1) + (2 + 3). Only threads below 128 sum the softmax's exponentials, with the old stride, and the other warps
add +0. So every FMA, sum and shuffle tree is the 128-thread kernel's, and the bits are unchanged: the golden dumps
(FP32 and FP16 rings) and the decode and 4-column logits at contexts 300 to 8,192 are byte-identical. The kernel
was 63 us a call at 256 keys and is now 24 us.

Measured on the V100 (2026-11-21), `ninfer_inkling2_golden_test` with `NINFER_INKLING2_DECODE_BENCH`
(`NINFER_INKLING2_SINGLE_BLOCK=1` for the single block). The text is
30,840 tokens of docs and C++, prefilled to the context in chunks of 256, then 64 single-token graph steps (one
warm-up step first). Each context is A B B A in one hold.

| Context | Decode, single block (tok/s) | Decode, split (tok/s) | Speedup |
|---|---|---|---|
| 1,024 | 47.1 | 67.7 | 1.44x |
| 16,384 | 25.1 | 61.0 | 2.43x |
| 30,775 | 17.4 | 56.0 | 3.22x |

KLD(single block || split) is teacher-forced over the same text (`NINFER_INKLING2_ATTENTION_KLD`), through
8-column calls in three ranges. These are
the numbers of an FP32 reordering on this model. Near-tied MoE routing turns last-bit differences into flipped
experts, and the loop compounds them. For comparison, the FP32 program against the FP64 golden oracle reaches a KLD
max of 1.1e-2.

| Positions | KLD mean | KLD median | KLD p99 | KLD max | Argmax agrees |
|---|---|---|---|---|---|
| [0, 1024) | 1.66e-3 | 1.57e-4 | 2.31e-2 | 5.91e-2 | 97.95% |
| [1024, 4096), first 1,024 | 3.58e-3 | 1.27e-3 | 3.23e-2 | 2.04e-1 | 97.75% |
| [16384, 30840), first 1,024 | 2.45e-3 | 4.69e-4 | 2.33e-2 | 1.37e-1 | 98.63% |

The split parts take `relative_bias_attention_split_bytes`: 8 columns × 24 heads × (d + 2) floats per 256 keys of
the longest ring. That is about 25 MB of workspace at 65,536 context.

## FP16 KV rings

**Lossy, on by default.** The target's and the drafter's KV rings hold FP16 (`EngineOptions::kv_rings`,
`ProgramLimits::kv_fp16`); `--kv-dtype fp32` keeps the exact FP32 program, for reproduction (it also resolves an unset `--prefill-math` to
`fp32`; an explicit `--prefill-math tc` beside it runs the tensor cores and is not exact). With FP16 rings the
head norms' append rounds each normalized K and V value to nearest FP16, and the attention widens the values back to
FP32 as it reads them. Scores, softmax and sums stay FP32. A position costs 32 KiB across the 8 global visits
instead of 64 KiB, so about twice the context fits. Besides `fp16` and `fp32`, Inkling-2 takes only the opt-in
low-bit global rings `int8`, `rk8v4` and `rk4v4` ([Low-bit global KV rings](#low-bit-global-kv-rings-opt-in)) and refuses every other form.

The values suit FP16: the head norms bound them (K is clipped per head, V has unit norm). Over the 30,840-token
prompt below (1.5e9 values each of K and V), max |K| is 21.7 and max |V| 11.3, none overflows, and 60,305 K and
102,465 V values (under 0.01%) fall in FP16's subnormal range, 31 and 54 of them to zero. The mean relative
rounding is 1.76e-4, against BF16's 1.41e-3 (also printed by `NINFER_INKLING2_KV_FP16_KLD`).

It is not a speed option. Decode attention is latency-bound, not bandwidth-bound, so halving the bytes changes
little.

Measured on the V100 (2026-11-21), `ninfer_inkling2_golden_test`, split attention on, A B B A in one hold per
context. The numbers are the mean of the two runs of each form.

| Context | Prefill FP32 / FP16 (tok/s) | Decode FP32 / FP16 (tok/s) |
|---|---|---|
| 1,024 | 370.6 / 377.7 | 67.80 / 68.11 |
| 16,384 | 288.3 / 309.6 | 61.07 / 62.68 |

KLD(FP32 rings || FP16 rings) is teacher-forced over 30,840 tokens of docs and C++ in chunks of 256
(`NINFER_INKLING2_KV_FP16_KLD`). It is the same size as an FP32 reordering of the attention sums on this model
([Split decode attention](#split-decode-attention)): near-tied MoE routing turns any small change into flipped
experts.

| Positions | KLD mean | KLD median | KLD p99 | KLD max | Argmax agrees |
|---|---|---|---|---|---|
| [0, 1024) | 2.70e-3 | 8.52e-4 | 2.38e-2 | 6.44e-2 | 97.36% |
| [1024, 4096) | 3.65e-3 | 1.41e-3 | 3.24e-2 | 3.98e-1 | 97.07% |
| [4096, 16384) | 3.47e-3 | 1.27e-3 | 3.14e-2 | 4.50e-1 | 97.78% |
| [16384, 30840) | 3.49e-3 | 1.33e-3 | 3.14e-2 | 1.91e-1 | 97.36% |

## Low-bit global KV rings (opt-in)

**Lossy, opt-in, off by default.** `--kv-dtype int8 | rk8v4 | rk4v4` stores the 8 global visits' K and V rings in 8
or 4 bits; the 40 sliding rings stay FP16 and the FP16 default is untouched (same bytes: golden, layout hashes and
DSpark tests as before). At the full 262,144 with `--spec dspark --vision` the freed memory lets the engine keep the
drafter's **Q8 head copy and its whole-context window** (FP16 there falls back to the target's BF16 head and a
16,384-position window), and long-context decode runs faster:

| `--kv-dtype` | K / V | Global rings at 262,144 | Frees | KLD mean / p99 vs FP16 | Top-1 | Decode 100K / 200K (plain) |
|---|---|---:|---:|---|---:|---|
| `fp16` (default) | FP16 / FP16 | 8192 MiB | | (FP32 floor: 3.46e-3 / 3.06e-2) | (97.50%) | 47.80 / 34.81 tok/s |
| `int8` | rotated INT8 / INT8 | 4224 MiB | 3968 MiB | 3.66e-3 / 3.22e-2 (1.06x floor) | 97.62% | 52.92 / 40.72 (+10.7% / +17.0%) |
| `rk8v4` | rotated INT8 / int4 | 3264 MiB | 4928 MiB | 6.82e-3 / 5.68e-2 (2.0x) | 96.62% | |
| `rk4v4` | rotated Lloyd-Max 4-bit / int4 | 2240 MiB | 5952 MiB | 9.71e-3 / 7.92e-2 (2.8x) | 95.85% | **53.24 / 41.05 (+11.4% / +17.9%)** |

`int8` sits at the noise floor of an FP32 reordering (the same KLD as FP16 against FP32); `rk4v4` frees the most and
decodes fastest at about 3x the floor. Only one sequence at a time: `--max-concurrency` above 1 is refused with them.

**Layout.** A global ring is a U8 tensor `[kv_heads * row_bytes, capacity]`; a position's row holds every head's codes,
then every head's FP16 scales. K (L2-clipped per head, no RoPE) is rotated by a normalized 128-point Hadamard and coded
per 64 dimensions: Lloyd-Max 4-bit with a least-squares scale (68 bytes a head) or INT8 absmax/127 (132 bytes). V
(unit RMS) is not rotated: int4 absmax/7 per 32 (72 bytes) or INT8 per 64 (132 bytes). The q clip job rotates the
queries by the same Hadamard, so scores need no inverse. The head-norm append codes each new position
(`ops::kLowbitKeys4/8`, `ops::kLowbitValues4/8`); decode and verification read the codes in
`attention_lowbit_group_kernel` (split blocks of 256 keys; a block takes the 3 query heads of a GQA group so each key
and value is loaded and expanded once; scores, softmax and sums FP32), and the tensor-core prefill expands them to FP16
as it stages a tile. The relative bias and sinks are as in FP16.

**Accuracy.** KLD(FP16 rings || low-bit) teacher-forced over the 30,840-token docs-and-C++ prompt in chunks of 256
(`NINFER_INKLING2_KVQ_KLD`, FP32 math, both programs in lockstep); flat over [0,1K), [1K,4K), [4K,16K), [16K,31K).
Greedy output is not token-identical to FP16 (0 of 12 300-token chats match, `int8` included: at this floor a
near-tied expert choice flips within a few hundred tokens), but answers hold: the maths prompt gives the same 14:43, the code prompt the same
algorithm and complexity, and the needle at 100K and 200K (depths 0.2, 0.8; 0.3, 0.7) is found 4 of 4 with each.

Variants measured with the simulator (`NINFER_INKLING2_KVQ`, which codes and decodes into the FP16 rings; its
`k4v4` 1.01e-2 matches the real rings' 9.71e-3) and not shipped:

| Variant | KLD mean | p99 | Top-1 |
|---|---:|---:|---:|
| K FP16, V int4 | 6.85e-3 | 5.71e-2 | 96.66% |
| K 4-bit, V FP16 | 8.83e-3 | 7.23e-2 | 96.05% |
| K 4-bit, V INT8 | 8.87e-3 | 7.25e-2 | 95.81% |
| K 4-bit, V Lloyd 4-bit | 9.65e-3 | 7.43e-2 | 95.79% |
| `rk4v4` on the 2 most tolerant visits (g6, g7) | 1.25e-3 | 1.08e-2 | 98.78% |
| on 4 (g4-g7) | 2.56e-3 | 1.95e-2 | 97.97% |
| on 6 (g2-g7) | 5.17e-3 | 3.95e-2 | 96.86% |
| g2-g7 at 4 bits, g0-g1 INT8 | 6.66e-3 | 5.26e-2 | 96.51% |
| g4-g7 at 4 bits, g0-g3 INT8 (real rings) | 5.45e-3 | 4.19e-2 | 96.98% |
| `rk4v4`, last 4,096 positions FP16 | 5.52e-3 | 5.21e-2 | 97.06% |
| `rk4v4` on the sliding rings too | 2.69e-2 | 2.11e-1 | 92.97% |

The early global visits are the sensitive ones (`rk4v4` alone on g0 ... g7: 6.9e-3, 4.4e-3, 2.7e-3, 1.8e-3, 8.9e-4,
7.8e-4, 6.0e-4, 6.8e-4). A mix saves memory in proportion to the visits it codes (744 MiB a visit at 4 bits, 496 at
8), so none beats `int8` (at the floor, 3968 MiB) on accuracy per byte. The FP16 recent window only helps while the
window is a large part of the context (7.06e-3 over [16K, 31K)), and the sliding rings are far too sensitive.

**Speed.** `ninfer_inkling2_golden_test` decode bench (`NINFER_INKLING2_DECODE_BENCH`, plain program decode, TC
prefill chunk 2048), FP16 / `int8` / `rk4v4` as A B C C B A in one hold per context, 2026-11-22, the two legs
equal to 0.1%:

| Context | Decode FP16 / int8 / rk4v4 (tok/s) | Prefill FP16 / int8 / rk4v4 (tok/s) |
|---|---|---|
| 8,192 | 72.43 / 72.94 / 73.05 | 1888 / 1873 / 1870 |
| 102,400 | 47.80 / 52.92 / 53.24 | 1221 / 1156 / 1152 |
| 204,800 | 34.81 / 40.72 / 41.05 | 880 / 812 / 809 |

Prefill is 5-8% slower past 100K (the flash kernel expands the codes as it stages each tile). A per-head lowbit
kernel (one block per query head, `NINFER_LOWBIT_PER_HEAD=1`) gives only +1-4% decode: the gain comes from reading a
key once for its GQA group.

Through `ninfer-serve` with the production flags (`--vision --spec dspark --draft-tokens 3 --max-context 262144`,
both adapters loaded), one server per form:

| | FP16 | `int8` | `rk4v4` |
|---|---|---|---|
| Fit at 262,144 | chunk 2048, BF16 head, window 16,384, 302 MiB free | chunk 2048, **Q8 head, whole window**, 2840 MiB free | chunk 2048, **Q8 head, whole window**, 4824 MiB free |
| 12 chats, DSpark (mean tok/s) | 89.35 | 92.83 | **93.35** |
| Maths / code, DSpark (tok/s) | 112.6 / 112.1 | 119.7 / 118.2 | 119.6 / 112.9 |
| Decode after 8K / 100K / 216K prompt, DSpark | 80.8 / 46.7 / 32.4 | 88.4 / 49.9 / 38.8 | 90.1 / 49.9 / 39.3 |
| Prefill 8K / 100K / 216K (tok/s) | 1859 / 1207 / 846 | 1845 / 1144 / 781 | 1846 / 1141 / 779 |

## Tensor-core prefill

**Lossy, on by default.** Prefill forwards of 64 or more columns run on the FP16 tensor cores with FP32
accumulation (`EngineOptions::prefill_math`, `ProgramLimits::prefill_tc`). `--prefill-math fp32` keeps the exact
FP32 prefill. Decode and DSpark's verify (at most 17 columns) are the FP32 program either way.

- Dense projections: `projection_tc_fp32`. Each BF16 weight row is converted to FP16 under a power of two per row
  (its absmax in [2^14, 2^15)), and each activation column under its own. cuBLAS `GemmEx` multiplies them with FP32
  accumulation, and the two scales are divided out. A BF16 value converts exactly unless its scaled magnitude falls
  under FP16's normal range: 1 of the 4.7e9 dense values does (`NINFER_INKLING2_TC_COUNT`).
- Experts: `moe_situ_experts_tc`. The entries are grouped by expert, and one kernel per matrix runs every expert. A
  CTA takes 128 rows of one expert by up to 64 of its entries. It decodes the NVFP4 blocks into shared memory and
  multiplies on WMMA 16x16x16.
  - Every NVFP4 value is exact in FP16 unscaled: at most 6 significant bits, with magnitudes in [2^-10, 2688].
    That makes 0 of 2.0e10 inexact.
  - Inputs are FP16 under a power of two per entry.
  - The gate/up kernel applies SiTU and stores the activation in FP16 under one power of two for the call; the down
    kernel scatters each entry's column.
  - Layers 20-23's BF16 banks take per-expert cuBLAS calls.
- Attention: one pass of flash attention over the FP16 rings. A block takes 32 columns by the 3 heads of a KV group,
  so each staged key and value serves all three heads.
  - Q and the softmax weights are FP16, K and V as stored, and the sums FP32 on WMMA.
  - The online softmax rescales the accumulators in registers through the fragment's row map.
  - The relative bias is a WMMA product too, over FP16 hi + lo parts of r and the table, which keeps about FP32's
    absolute accuracy.

Bit-identical changes made along the way also speed up the FP32 prefill (`test_wide_bits` checks them against one
call per column):

- the router scores take 4 columns by 16 experts per block;
- the loop controllers form z once per column, then one thread per output;
- the short convolutions split the columns into segments;
- a forward holds only the visit outputs still referred to (the blocks, the loop's prelude and bank, the current
  one) in a pool of slots, and releases each visit's own temporaries when it returns, instead of keeping every
  visit's (six hidden-wide FP32 columns each, 2.65 MB of workspace per chunk token over the 48 visits). Only the
  addresses change: the golden test stays byte-identical.

Quality is measured with `NINFER_INKLING2_TC_KLD` on the FP16 rings: tensor cores in chunks of 1024 against the
FP32 prefill in chunks of 256. Each row covers the logits after the prompt and those of the 64 teacher-forced decode
steps that follow. The control changes only the order of the FP32 attention's sums (the tiled FP32 kernel in place of
the flash one, through a measuring switch since removed; over FP32 rings `NINFER_INKLING2_TC_PARTS=attention` still
takes that kernel), so it is the reordering noise floor. `NINFER_INKLING2_TC_PARTS` (any of dense, experts, attention)
selects the parts for the other rows.

| Route | 2K: KLD mean / max | 2K: argmax | 16K: KLD mean / max | 16K: argmax |
|---|---|---|---|---|
| All three | 2.63e-3 / 1.58e-2 | 65/65 | 1.92e-3 / 3.42e-2 | 64/65 |
| Control: FP32, reordered | 3.06e-3 / 2.39e-2 | 65/65 | 1.96e-3 / 4.05e-2 | 64/65 |
| Dense only | 2.46e-3 / 1.57e-2 | 64/65 | 1.21e-3 / 3.22e-2 | 65/65 |
| Experts only | 2.96e-3 / 2.30e-2 | 64/65 | 1.46e-3 / 2.00e-2 | 64/65 |
| Attention only | 2.58e-3 / 1.56e-2 | 64/65 | 1.64e-3 / 3.27e-2 | 65/65 |

Speed on the V100 (2026-11-21), FP16 rings, `NINFER_INKLING2_PREFILL_BENCH`:

| Context | FP32, chunk 256 | TC, chunk 256 | TC, 512 | TC, 1024 | TC, 2048 |
|---|---|---|---|---|---|
| 2,048 | 416 | 864 | 1,285 | 1,731 | 2,096 |
| 10,240 | 382 | 831 | 1,227 | 1,633 | 1,968 |
| 16,384 | - | - | - | 1,575 | 1,895 |

(tok/s.) ninfer-all's tensor-core prefill on the same card is about 2,124 tok/s at 2K and 1,589 at 10K.

The engine picks the widest chunk of 2048, 1024, 512 and 256 whose programs (with the drafter and the vision tower
when they are on) fit beside `--max-context`, leaving 256 MiB free. It logs the choice and the free memory at
startup. `--prefill-chunk N` sets the chunk instead. At 262,144 with DSpark and vision chunk 2048 leaves 302 MiB
free, the prefix-reuse snapshots being in host memory (see "Context with DSpark and vision").

## GGUF mapping (Unsloth `unsloth/Inkling-2-Mini-GGUF`)

Every UD quant has 734 tensors and the same keys. Read from 20 MB header range reads
(2026-11-20). The physical blocks are `blk.0..23`, and the per-visit keys have 48 entries.

| GGUF tensor (per `blk.N` unless global) | Role | UD-Q8_K_XL | UD-Q4_K_XL | UD-Q4_K_M | UD-IQ4_XS |
|---|---|---|---|---|---|
| `token_embd` [201024, 2304] | E | BF16 | Q8_0 | Q8_0 | Q8_0 |
| `token_embd_norm` | embed_norm g | F32 | F32 | F32 | F32 |
| `output` [201024, 2304] | W_head | BF16 | Q8_0 | Q8_0 | Q6_K |
| `output_norm`, `output_res_query` [2304] | final norm, final MIX query | F32 | F32 | F32 | F32 |
| `loop_res_query`, `loop_kv_res_query` [24, 2304] | Qloop, Qkv | F32 | F32 | F32 | F32 |
| `attn_q` [3072,2304], `attn_gate` [3072,2304], `attn_output` [2304,3072] | Wq, Wgate, Wo | BF16 | Q8_0 | Q8_0 | Q8_0/Q6_K/Q5_K |
| `attn_k`, `attn_v` [1024, 2304] | Wk, Wv | BF16 | Q8_0 | Q8_0 | Q8_0/Q6_K |
| `attn_rel` [384, 2304] | Wr | BF16 | Q8_0 | Q8_0 | Q8_0/Q6_K |
| `attn_sink_proj` [24, 2304] | Wsink | BF16 | BF16 | BF16 | BF16 |
| `attn_k_conv_proj`, `attn_v_conv_proj` [32, 2304] (global only) | Wkd, Wvd | BF16 | BF16 | BF16 | BF16 |
| `attn_q_clip` [24,128], `attn_rel_clip` [24,16] | sq, sr | F32 | F32 | F32 | F32 |
| `attn_rel_bias` [1024, 16] | P^T | F32 | F32 | F32 | F32 |
| `attn_k_conv`, `attn_v_conv` [1024, 4]; `attn_post_conv`, `ffn_post_conv` [2304, 4] | w taps | F32 | F32 | F32 | F32 |
| `attn_norm` (local) or `attn_{q,k,v}_inp_norm` (global); `ffn_norm` | norm g | F32 | F32 | F32 | F32 |
| `attn_res_query` [3 or 1, 2304] (N > 0), `ffn_res_query` [2304] | A_p, M_p | F32 | F32 | F32 | F32 |
| `ffn_gate_inp` [128, 2304] | W_router | F32 | F32 | F32 | F32 |
| `exp_probs_b.bias` [128] | b | F32 | F32 | F32 | F32 |
| `ffn_gate_up_exps` [128, 2304, 2304], layers 0-19 | Wgu | NVFP4 | NVFP4 | NVFP4 | NVFP4 |
| `ffn_down_exps` [128, 2304, 1152], layers 0-19 | Wd | NVFP4 | NVFP4 | NVFP4 | NVFP4 |
| `ffn_{gate_up,down}_exps.scale` [128], layers 0-19 | per-expert multipliers | F32 | F32 | F32 | F32 |
| `ffn_gate_up_exps`, layers 20-23 | Wgu | BF16 | Q8_0 | Q5_K | Q5_K/Q4_K/IQ4_XS |
| `ffn_down_exps`, layers 20-23 | Wd | BF16 | Q8_0 | Q8_0 | Q5_0/IQ4_NL |
| `loop_{attn,k,v,mlp}_down` [64, 4608], `_gain`, `_forget` [2304, 64], layers 6-17 | D, G, F | BF16 | F32 | F32 | F32 |

Shapes are row-major (the GGUF `ne` reversed). Sizes: Q8_K_XL 21.29 GiB, Q4_K_XL 16.41,
Q4_K_M 15.46, IQ4_XS 14.64. The 10.68 GiB of NVFP4 experts are **byte-identical in every variant**:
they are the checkpoint's own NVFP4 weights, repacked bit-exact ([Lc] `_nvfp4_expert_rows`). The
original checkpoint ([Q] `hf_quant_config.json`) stores the dense weights in BF16, the routers and
queries in FP32, layers 0-19 experts in NVFP4 and layers 20-23 experts in MXFP8. **UD-Q8_K_XL is
therefore the exact original model**: its BF16 experts are the MXFP8 values, exact in BF16 ([Lc]
`mxfp8_to_f32`). The 4-bit variants differ from it only in the dense weights and the last 4 expert
layers. Unsloth's table gives the cost: Q4_K_XL has mean KLD 0.0107 against Q8_K_XL, with the same
top token 98.0% of the time.

GGML NVFP4 (type 40, 36 bytes per 64 values) is `{u8 d[4]; u8 qs[32]}`. Sub-block `s` covers 16 values
with scale `d[s]` (UE4M3). Value `j < 8` of the sub-block is the low nibble of `qs[8s + j]`, and
value `j + 8` the high nibble. The value is `kvalues_mxfp4[nibble] * ue4m3(d[s])`, where
`kvalues_mxfp4 = {0,1,2,3,4,6,8,12, -0,-1,...,-12}` (twice E2M1) and `ue4m3` is E4M3 times 0.5,
with the byte values 0x00 and 0x7F both meaning 0. This is ggml's `dequantize_row_nvfp4`, and it
equals E2M1 x E4M3 exactly. The doubled codes are small integers, so A8 `dp4a` arithmetic applies
directly.

The original model quantized expert **activations** too (NVFP4 per token with E4M3 block scales,
MXFP8 for layers 20-23). Neither llama.cpp nor this plan reproduces that: activations are A16 or A8,
which is more precise than training. (Ambiguity A6.)

## Which GGUF, and disk

All numbers come from the header fixtures (`tools/convert/inkling2.py` `decode_bytes_per_token`,
`kv_bytes_per_token`). Decode bytes count the stored weights one token streams: every core-layer tensor
three times, 8 of 128 experts, the loop controllers on passes 2 and 3, and the head.

| GGUF | File | Decode bytes/token | VRAM left for KV (32 GB V100) | Global-KV context (FP16) | Fidelity |
|---|---|---|---|---|---|
| UD-Q8_K_XL | 21.29 GiB | 5.78 GB | ~7.3 GiB | ~210k | the checkpoint itself (dense BF16, L20-23 experts BF16 = MXFP8 exactly) |
| UD-Q8_K_XL, L20-23 experts re-encoded MXFP8 | ~17.6 GiB | ~5.5 GB | ~11 GiB | ~320k | identical (lossless re-encode) |
| UD-Q4_K_XL | 16.41 GiB | 4.01 GB | ~12.2 GiB | ~355k | KLD 0.0107, 98.0% same top token (Unsloth) |

The VRAM column subtracts about 0.5 GiB of CUDA context, 0.6 GiB of vision tower, 0.36 GiB of DSpark
drafter, 1.5 GiB of workspace and 0.16 GiB of sliding-window rings. Global KV is 32 KiB per token (8 global
visits; the loop's two extra passes over physical 11 and 17 are in that count) plus ~4 KiB for the drafter's
global layer. INT8 KV halves it. All options hold 128k with room to spare; none holds 1M.

Decode on V100 is not pure bandwidth: about 1,200 dependent kernels per token cost 6-10 ms before fusion.
Q8_K_XL streams 1.44x the bytes of Q4_K_XL, so the expected decode cost is 10-20% slower, not 44%
(unmeasured estimate: Q4 ~65-90 tok/s, Q8 ~58-75 tok/s after phase 4).

**Recommendation: download UD-Q8_K_XL as the single source.** It fits with ~210k context of headroom
(more after the lossless MXFP8 re-encode), it is the exact model, so the engine's quality reference and
the llama.cpp golden are the same weights, and a faster Q8-dense artifact can still be derived from it with
ninfer's own encoders. Q4_K_XL is the fallback if 210k context is not enough before the re-encode lands.

**Disk.** The GGUF is 22.87 GB. Put it in `/dev/shm` (30 GB tmpfs, 59 GB RAM). The converter streams
(peak RSS ~0.6 GiB, as for `dsv4_gguf`) and writes an artifact of ~22.9 GB (BF16 kept) or ~18.9 GB (MXFP8
re-encode) to `~/models`. Root has ~8.8 GB free, so the 18 GB Qwen3.8-27B artifact must be deleted first
(~27 GB free; 4-8 GB margin). Then delete the GGUF from `/dev/shm`. Reading the GGUF in place is not an
option: v3 `.ninfer` is the only artifact the engine loads, and a GGUF on tmpfs would keep 23 GB of RAM
pinned while serving.

## Ambiguities and decisions

| # | Question | Resolution |
|---|---|---|
| A1 | Loop semantics have one source ([L], an unmerged PR) | Implement [L] exactly. Phase 2 gate: compare logits against a CPU build of `unslothai/llama.cpp` at that SHA on the same GGUF (KLD and top-1), before any kernel work is trusted |
| A2 | Do the passes share KV? | No. Each visit has its own cache ([L], [V]; `is_varloop` false, and the converter refuses varloop) |
| A3 | Which hidden state is a DSpark tap | `cur` at a visit's input, after the pass restart; tap 48 is the final normed state ([L] 363 and 497 `t_layer_inp`, [Lc] DSpark comment). Single source |
| A4 | Rows 200058..201023 of embed/head | Padding. Never sampled ([L] trims the logits); a converter may drop them |
| A5 | `ue4m3` byte 0x7F (NaN) | ggml decodes it as 0. The repack refuses such scales ([Lc]), so none occur. The converter checks |
| A6 | Activation quantization of experts in training | Not reproduced (A8/A16 instead). Quality reference is UD-Q8_K_XL under llama.cpp, which does the same |
| A7 | `1 - tanh(F h)` in the controller | As written in [L] (`ggml_scale_bias(forget, -1, 1)`). Not a typo candidate: the forget term is unconstrained in sign |
| A8 | Sink handling with flash attention in [L] | [L] passes a zero sink and folds `-t` into the bias table. Mathematically identical to the HF concat-then-drop. Kernels may use either |
| A9 | Local layer relative extent | The window (1024). The GGUF has `attn_rel_bias` [1024, 16] on every layer |

## Reuse map

| Piece | Reuse | New |
|---|---|---|
| Converter GGUF reader, artifact writer, record packing | `tools/convert/sources/gguf.py`, `pack_expert_records` | `inkling2_gguf` recipe (`tools/convert/inkling2.py`), `ggml_nvfp4` format and records `ggml_rec_nvfp4_nvfp4`, `ggml_rec_q8_0_q8_0` (Q4_K_XL), BF16 expert bank (Q8_K_XL) |
| Dense projections | Linear GGML Q8_0 (Q4_K_XL) or BF16 (Q8_K_XL) routes and the sm_70 streaming GEMV | register the hidden-2304 problems: [3072,2304], [1024,2304], [2304,3072], [384,2304], small [24/32,2304]; head [201024,2304] Q8_0/BF16 |
| Router | `projection_fp32` (FP32 weights) | the [128, 2304] problem; sigmoid + bias top-8 + normalize x8 route kernel (DSV4 has sqrt-softplus; this is a third gating) |
| Experts | offloaded_sparse_moe's canonical A8 dp4a arithmetic and the GGML block decoders (`src/ops/linear/ggml`) | a **resident** MoE expert Op at [128, 2304, 1152] (the existing one is fixed at 2560x640 and organized around offload). NVFP4 sub-block decode to int8 codes plus a per-16 scale; per-expert output multiplier; SITU activation; BF16 records for Q8_K_XL layers 20-23 |
| Attention | sm_70 prompt flash attention (vendored llama.cpp `fattn`), the decode attention kernels, sinks (`gathered_sink_attention`) | an additive per-head **relative bias** that is a gather from a per-query 1024-entry table, plus a per-query sink; log scale on global layers; no RoPE |
| Short convolutions | `causal_conv1d_silu` (Qwen3.5 GDN) state handling and Replay | 4 streams per layer, no SiLU, residual and `1/sqrt(K+1)` scale; dynamic per-head taps on global K/V |
| Depth attention (MIX), loop controller, clips | none | small fused element-wise and reduction Ops (MIX over at most 5 sources; CTRL is three skinny GEMVs at rank 64) |
| KV cache | paged KV, the SWA ring (Flash-Next local attention), INT8/K8V4 codecs | 48 caches (8 global and 40 local) indexed by visit |
| Program | qwen4_exp program structure (Model/Program split, CUDA Graph capture, MTP round plumbing) | an explicit 48-visit schedule. Weights are bound per physical layer; state is per visit |
| Tokenizer | `src/models/qwen3_5/frontend/tokenizer.cpp` byte-level BPE engine | the o200k-style Split regex as a hand-written matcher (like the Qwen and DSV4 ones), and `ignore_merges` (whole-piece vocabulary hits). The Python pipeline check is `tools/convert/inkling2_tokenizer.py` |
| Chat template | Jinja engine | `chat_template.jinja` (thinking-effort control, `<|message_*|>` roles, tool calls as `<|content_invoke_tool_json|>`); the reasoning and tool parsers follow `tokenizer_config.json` `response_template` |
| DSpark | DFlash context materializer and verify round | tap fusion, the Markov RNN chain, a global draft layer, and visit taps |
| Vision | Qwen3.5 vision plumbing (Vision contract, frontend media) | a different ViT (axial RoPE, patch 20, LayerNorm, merger) |

## Plan

Sizes are rough engineer-days for a person or agent who knows the codebase, including tests.

| Phase | Content | Size |
|---|---|---|
| 0 (done) | this spec; header fixtures of three GGUFs and the drafter; converter config, visit schedule and tensor map (`tools/convert/inkling2.py`); NumPy reference with the GGML NVFP4 decoder and checkpoint repack (`tools/inkling2/reference.py`); C++ FP64 op oracles (`tests/ops/inkling2_reference.h`, checked by the host-only `ninfer_inkling2_reference_test`); the tokenizer contract, checked token by token against the GGUF (`tools/convert/inkling2_tokenizer.py`). **Not** in this phase: the `ggml_nvfp4` artifact format and the converter (phase 2), any engine code | -- |
| 1. Golden (done) | CPU llama.cpp (Unsloth SHA above) on UD-Q8_K_XL: logits and per-visit activations for four prompts; the NumPy reference on the real weights against them (see "Validation") | -- |
| 2. Converter | `inkling2_gguf` recipe writes the artifact: C++ format registry for `ggml_nvfp4` and the new records, per-expert scales, padded vocab, tokenizer and chat resources. Reader test on the real file | 2-3 |
| 3. Text MVP, correctness first (done, see "Engine") | Model, config and binding. Program with the 48-visit schedule. New Ops with naive FP32 kernels qualified against the oracles: rel-bias sink attention (decode and prefill), conv streams, MIX, CTRL, clips, sigmoid router, resident NVFP4/Q8_0/BF16 MoE. Linear problem registrations. Tokenizer matcher. Greedy text equals the golden. Done except Q8_0 experts and the Linear registrations (the MVP uses `projection_fp32`), which move to phase 4 | 8-12 |
| 4. V100 decode performance (in progress; done so far: decode graphs, split and latency-hidden attention, FP16 rings, fused per-entry and expert-major experts, batched decode; 78.8 tok/s plain serve) | Fused decode kernels (MIX, CTRL, conv and norms), expert GEMV for NVFP4 via dp4a, global attention on FP16 KV, CUDA Graph per decode width; measure against the bandwidth floor | 5-8 |
| 5. Prefill (done: [Tensor-core prefill](#tensor-core-prefill), 1,930 tok/s at 8K) | cuBLAS HMMA for the dense and expert GEMMs (dequantize tiles to FP16), flash prefill with the relative bias and sink (extend the vendored `fattn` like [L]'s banded op) | 4-6 |
| 6. DSpark (done, greedy and sampled, adaptive width) | Drafter binding, tap capture at visits 5/23/41/48, Markov chain with the precomputed `W_head Wo`, verify of the full block (8) with `keep()` | 4-6 |
| 7. Vision (done, correctness first) | mmproj conversion and the ViT (see "Vision tower") | 4-6 |
| 8. Chat/serving (MVP done) | template, reasoning and tool parsers, effort control (see "Serving") | 2-3 |

The MVP (phases 1-3) is about 2.5 weeks. Everything after it is optional ordering.

## V100 (sm_70) specifics and risks

- **No BF16 arithmetic:** BF16 weights (Q8_K_XL dense, layers 20-23 experts) are widened in
  registers for GEMV, and dequantized to FP16 tiles for cuBLAS HMMA in prefill. BF16 to FP16 is
  exact for the observed weight magnitudes only if none are below 2^-24 or above 65504. The
  converter can verify this per tensor and fall back to FP32 GEMM for an outlier tensor.
- **Kernel count dominates decode:** 48 visits times about 25 small kernels is about 1200 launches
  per token. At the 5-8 us fill/drain of a dependent short kernel (volta-port.md "Next" #4), that
  is 6-10 ms, comparable to the weight stream. Fusing MIX, CTRL, the clips, the norms and the conv
  streams is the main decode lever.
- **Loop weight reuse does not hit L2:** the core layers' dense weights (about 28 MB per visit at
  Q8_0, 55 MB at BF16) are streamed three times. V100 L2 is 6 MB.
- **Relative-bias attention:** the vendored llama.cpp sm_70 flash kernel has no per-query bias table.
  [L] wrote CUDA for this (`fattn-banded`, `fattn-mma-f16` changes), which is a template, but it
  needs the Volta `fattn` path. The decode attention is simple (one query; gather 1024 bias values).
- **FP16 ranges:** `|q k| / sqrt(128) <= 22.63^2 * 1.485 / 11.3 = 67`. The bias is bounded by the
  clip at 8 x 1.485 x max|P|. The sink is unbounded but enters as a logit. Max-subtracted softmax in
  FP32 is required. K and V fit FP16 and INT8.
- **Single-source loop math (A1):** all loop work is gated on the CPU golden.
- **NVFP4 dp4a decode:** E2M1 to int8 is a 16-entry LUT (`__byte_perm`), and the E4M3 scale to FP32 is
  a 256-entry table or bit manipulation. Volta has no FP8 hardware, which this does not need.
- **Conversion space:** see "Which GGUF, and disk" above.

## Validation

1. NumPy reference against the C++ oracles: `ninfer_inkling2_reference_test` (host only) checks
   `tests/ops/inkling2_reference.h` against `tests/fixtures/inkling2/golden.json`.
2. CPU llama.cpp at [L]'s SHA as the end-to-end golden. `tools/inkling2/llama_golden.cpp` runs a GGUF
   and writes every position's logits plus `attn_out`, `ffn_inp` and `l_out` per layer visit and
   `result_norm`, through the scheduler's eval callback. Feeding the prompts whole or in chunks of 5
   gives bit-identical logits. `tests/fixtures/inkling2/llama_golden_topk.json` keeps the top 20
   logits of four prompts on UD-Q8_K_XL (`unsloth/Inkling-2-Mini-GGUF@29eac40c`), for the engine.
3. `tools/inkling2/compare_golden.py` runs the NumPy reference on the real weights (decoded lazily
   by `tools/inkling2/gguf_weights.py` from the GGUF, or by `tools/inkling2/artifact_weights.py`
   from the converted artifact, which holds the same bytes) against that golden. Result on
   UD-Q8_K_XL, 2026-11-20:

   | Prompt | Tokens | Per-visit relative error (all 48 visits) | Logit KLD per position | Top-1 |
   |---|---|---|---|---|
   | "The capital of France is" | 5 | 3e-3 to 6.4e-2, cosine >= 0.9979 | 8e-6 to 7e-4 | 5/5 |
   | chat turn ("why the sky is blue") | 13 (also in chunks of 4) | mostly 1e-2; single tokens to 1e-1 | up to 2.6e-2 | 13/13 |

   The larger per-token errors are not position dependent: they start in one layer visit's MoE for
   one token, where the router's 8th and 9th scores are nearly tied (visit 16, token 6: a margin of
   0.0015), so llama.cpp's activation rounding (BF16 and Q8 blocks on the CPU path) picks another
   expert than the FP64 reference. The reference therefore agrees with [L] to rounding, loop
   included. This shows the spec transcribes [L] faithfully. It does not resolve A1: both are still
   one source's reading of the loop. The outputs are sensible (" Paris"; a French continuation;
   `<|content_thinking|>` opening a chat answer), which is the evidence available that [L] is right.
4. Each new Op qualified against its oracle on the real shapes (op-development.md):
   `ninfer_inkling2_fp32_ops_test` (`tests/ops/test_inkling2_fp32_ops.cpp`) runs every entry of the
   five families above against the FP64 oracles of `tests/ops/inkling2_reference.h` at H 2304,
   24/8 heads of 128, d_rel 16, 128 experts of 1152, top 8, with the stateful ones (convolution
   history, KV ring) over several calls, a ring that wraps and single-token decode steps. On the
   V100 the largest error is 2.0e-7 of `1 + |ref|` (bound 2e-6), and 6.7e-7 of the output's
   largest magnitude for the NVFP4 and BF16 experts (bound 1e-5).
5. Engine against the golden: `ninfer_inkling2_golden_test` (`tests/models/inkling2/test_golden.cpp`,
   with `NINFER_TEST_INKLING2_ARTIFACT` and `NINFER_INKLING2_GOLDEN`). The acceptance bar is KLD
   at most 0.002 against llama.cpp on the same GGUF, measured on prompts without near-tied
   routing, and the same top-1 token at every position of the golden prompts. Result on the
   V100, 2026-11-20:

   | Prompt | Tokens | Per-visit relative error, worst | KLD mean / max | Top-1 |
   |---|---|---|---|---|
   | p0 "The capital of France is" | 5 | 5.5e-2 | 3.4e-4 / 8.2e-4 | 5/5 |
   | p1 Python docstring | 20 | 7.2e-2 | 1.8e-4 / 1.5e-3 | 20/20 |
   | p2 chat turn | 13 | 8.8e-2 | 5.7e-3 / **2.1e-2** | 13/13 |
   | p3 French | 20 | 6.4e-2 | 4.8e-4 / 2.4e-3 | 20/20 |

   Chunks of 3 give the whole run's logits bit for bit. Greedy from p0: " Paris.\n\nA. Noun
   clause\n\nB. Adjective clause".

   p2 is above the bar, and p1 and p3 at their worst position too. Whether that is llama.cpp's
   rounding or an engine error was settled without a new golden, by comparing the engine with the
   FP64 reference directly: the test's `NINFER_INKLING2_DUMP` writes the engine's logits, traces
   and chosen experts per visit (`route_ids-v`), and `compare_golden.py --routes` makes the
   reference use those experts. On p2:

   | Comparison | KLD max | Per-visit relative error, worst | result_norm |
   |---|---|---|---|
   | engine vs llama.cpp | 2.1e-2 | 8.8e-2 | 7.6e-2 |
   | FP64 reference vs llama.cpp (as item 3) | 2.6e-2 | 6.7e-2 | 6.1e-2 |
   | FP64 reference with the engine's experts vs llama.cpp | 2.1e-2 | 8.9e-2 | 7.6e-2 |
   | engine vs FP64 reference, each routing itself | 1.1e-2 | 5.9e-2 | 4.8e-2 |
   | **engine vs FP64 reference, same experts** | **3.4e-5** | **4.3e-3** | **2.3e-3** |

   With the same experts the engine is the FP64 reference to FP32 and BF16-activation rounding,
   loop included; the whole gap is expert selection at near-tied router scores, where each of the
   three implementations' rounding picks differently. The same holds on every prompt: with
   the engine's experts the FP64 reference is within KLD 2.6e-6 (p0), 3.8e-6 (p1), 3.4e-5 (p2) and
   9.6e-6 (p3) of the engine at every position, every visit within 1e-2 relative (4.7e-3 except
   one attention output of p0, visit 21, at 9.7e-3) and `result_norm` within 2.3e-3. So the engine has no
   error the golden can show, and the 0.002 bar cannot be met on these prompts by any of the three
   against another: it needs prompts without near ties, or a statistic over many tokens
   (perplexity) rather than a per-position maximum.

   Where the engine and the FP64 reference part ways (`compare_golden.py` with an engine golden
   reports every (visit, token) whose expert set differs, with the reference's margin of its 8th
   over its 9th steered score): p0 11 of 240 decisions, p1 91 of 960, p2 42 of 624, p3 5 of 960.
   The first difference in visit order, the seed of the rest, is at a margin of 7.9e-6 (p0, visit
   3), 1.4e-4 (p1, visit 1), 1.8e-4 (p2, visit 12) and 5.5e-4 (p3, visit 34): within what FP32
   accumulation and BF16 activations move a router score. Later differences also follow from earlier
   ones, through attention and the convolutions.

## Context with DSpark and vision

At startup the engine picks the widest prefill chunk whose programs fit beside `--max-context` with 256 MiB to spare
(`kStartupSpare`). With `--spec dspark` it tries, for each chunk, the drafter at its best and then gives up what costs
it least, never the output:

1. the drafter's Q8 copy of the text head (492 MB); without it the base logits read the target's BF16 head
   (`DSparkProgram`'s `head_copy`), a slower draft;
2. with `--dspark-window auto` (the default), the whole context for the last 32768, then 16384 positions of the
   drafter's global layer (`SpeculativeOptions::dspark_window`; its FP16 ring then holds that many positions at
   4 KiB each instead of `--max-context`). `--dspark-window 0` keeps the whole context, `N` fixes a window
   of at least the drafter's sliding window.

The startup log's `prefill |` note says what was chosen (chunk, workspace, drafter head, drafter window).

Other memory the stack frees or reserves:

- The drafter's layer matrices are bound in pinned host memory: the drafter only reads them once, into its own split
  Q8 device copies, so a device copy of the artifact's was unused.
- The ViT workspace (an image of `max_image_tokens`) is reserved at startup inside the text workspace
  (`ProgramLimits::min_workspace_bytes`, `VisionEncoder::workspace_bytes`); images are encoded before the prompt's
  prefill, so the two never overlap. It used to be allocated on the first image, which could fail at an edge
  context. Its activations share one region (patches, q/k/v, FFN and the merger reuse it in turn).
- `--vision-residency overlay` keeps the 0.64 GB tower in pinned host memory and copies it into the workspace for
  each image (one transfer over the tower's range). The embeddings are bit-identical to resident ones. It saves
  device memory only when the workspace is already larger than the tower plus the ViT's activations (a chunk of 512
  or more), and costs each image request about 0.2-0.25 s: on the V100, 0.091 against 0.283 s for a 640 x 488 image
  (208 image tokens) and 0.035 against 0.271 s for 300 x 170 (40).

Measured on the V100 (2026-11-21, `ninfer-serve` startup, FP16 rings, tensor-core prefill):

| Configuration | Largest `--max-context` | Chosen at that context |
|---|---|---|
| neither DSpark nor vision | 307,200 | chunk 256 (262,144: chunk 1024) |
| DSpark and vision, before this stack's drafter and vision changes | 217,088 | chunk 256 |
| DSpark and vision, `--dspark-window 0` | 238,592 | chunk 256, the target's head |
| DSpark and vision, `--dspark-window 32768` | 260,096 | chunk 256, the target's head |
| DSpark and vision, `auto` (default) | 262,144 (265,216 fails) | chunk 256, the target's head, window 16384 |
| same, the snapshots in host memory (2026-11-22) | 262,144 | chunk 2048, the target's head, window 16384; up to 222,720 the drafter's Q8 head copy and the whole context |
| same, `--prefill-chunk 512 --vision-residency overlay` | 262,144 | chunk 512, under the 256 MiB spare |

Before the layout below (2026-11-22), `--prefill-chunk 512` without the overlay did not start at 262,144 with DSpark
and vision. With it every chunk did, though none left the 256 MiB spare beside the snapshots on the device, so `auto`
picked 256 there; with the snapshots in host memory it picks 2048:

| `--max-context 262144 --spec dspark --vision`, both adapters | Free after startup | Workspace |
|---|---|---|
| master before this layout, auto (chunk 256) | about 78 MiB (nvidia-smi: 94 MiB more used than now) | 420 MiB (the ViT's), and the drafter's own |
| auto (chunk 256) | 172 MiB | 420 MiB |
| `--prefill-chunk 512` | 152 MiB | 420 MiB |
| `--prefill-chunk 1024` | 134 MiB | 420 MiB |
| `--prefill-chunk 2048` | 64 MiB (46 MiB after the first requests' decode graphs) | 454 MiB, 416 at its peak |
| the snapshots in host memory, auto (chunk 2048) | 302 MiB | 454 MiB, 416 at its peak |

(cudaMemGetInfo after the programs and, before the last row, the device snapshots. Below about 24 MiB the first
request's decode graph cannot be instantiated: the spare is not idle.)

At 262,144 the drafter keeps the target's BF16 head and the 16384-position window: its Q8 head copy (470 MiB) would
need 424 MiB more than the spare leaves, the whole-context window (BF16 head) about 910 MiB more, and both about
1,380 MiB more. Each 2,048 positions of context cost 72 MiB with the whole-context drafter, so with both the largest
context is 222,720 (258 MiB free; 217,088 when the snapshots were on the device). DSpark decodes about 4% faster
with both (94.3 tokens/s over the 12 chats at 217,088 against 90.0-90.6 at 262,144).

### Prefill memory at 262,144

Where the device memory goes with DSpark and vision at chunk 2048 (drafter window 16384, the target's head):

| What | MiB |
|---|---:|
| CUDA context, before the weights | 586 |
| weights (text, vision tower, drafter's device part) | 22,516 |
| global KV rings, 8 visits x 262,144 x 4 KiB | 8,192 |
| local KV rings, 40 visits x (1023 + 319) x 4 KiB | 210 |
| text workspace (forward temporaries; the ViT's activations; the drafter's workspace, lent) | 454 |
| DSpark taps (4 x FP32 [2304, 2048]), rollback state, convolution histories | 92 |
| drafter (rings, split Q8 layers, Q8 Markov head) | 396 |
| held taps | 16 |
| free (cudaMemGetInfo after startup) | 302 |

The prefix-reuse snapshots (text 163 MiB, drafter 73 at window 16384) are in pinned host memory, not here; on the
device they took 238 MiB, which left 64 MiB free at chunk 2048.

The text workspace at chunk 2048 was 1,744 MiB at 16,384 context (more at 262,144: the decode's split-attention
parts were added to the prefill's), and the drafter's own workspace 145 MiB more. What changed, each a layout or
lifetime change (the layout-hash harness below compares every byte of the logits, taps, decode and verify against
master at chunks 256, 577, 600, 1024 and 2048):

- The workspace is the largest of one visit's stages, not their sum: `TextProgram::visit` releases the loop
  messages, mixed inputs and controller outputs after the q/k/v projections, the projections after the attention
  output, and the routing's temporaries before the experts (`forward_bytes`). Decode's split-attention parts are
  counted for the narrow forwards only.
- A local visit attends a chunk of 320 or more in pieces of 256 columns (`kLocalAttentionSlice`), each appending
  its keys first, so its ring holds 1023 + 319 positions instead of 1023 + chunk. The flash attention's tiles take
  keys by absolute position, so a piece computes every column as the whole call did (0.4% slower at chunk 1024).
- The drafter borrows the top of the text workspace (`TextProgram::lent()`, `ProgramLimits::lend_bytes`): it runs
  only between forwards, and no forward's results reach that far.
- The tensor-core experts keep one FP16 input per column (NVFP4: every entry of a column reads its row) and, for the
  BF16 banks, convert one expert's entries at a time; each entry gets the same values.

`NINFER_INKLING2_LAYOUT_HASH=<ids>` (`ninfer_inkling2_golden_test`, with `NINFER_INKLING2_LAYOUT_CHUNK`, `_TOKENS`,
`_CONTEXT` and `NINFER_INKLING2_KV=fp32`) prints those hashes and the workspace peak against its capacity.

The window changes only the drafts. Over two prompts of about 101,850 tokens (256 greedy tokens each, the output
the same for every window), drafts accepted of drafts made:

| Window | Continue the text | Summarize it |
|---|---|---|
| whole context | 9 / 21 | 115 / 159 (45.7 tok/s) |
| 32768 | 28 / 75 | 12 / 27 (41.0 tok/s) |
| 8192 | 48 / 80 | 12 / 27 (41.1 tok/s) |

A task that reads far back loses most of its accepted drafts, so `auto` windows the drafter only when the whole
context does not fit.

### Snapshots in host memory

The prefix-reuse snapshot (one request slot) is in pinned host memory: 163 MiB for the text Program and, with
DSpark, the drafter's 73 MiB at window 16384 (236 MiB at 262,144). Freeing the 238 MiB the two took on the device is
what lets `auto` pick chunk 2048 at 262,144. The prefill saves it at its usual cut, and the copy runs on its own
stream (`Inkling2Core::copy_stream_`) beside the rest of the prompt and the first decode steps:

- `TextProgram::save_async` stages the convolution histories, which every forward rewrites, on the device in stream
  order (one copy kernel, about 4 MB), then copies them and each sliding visit's ring window to the host visit by
  visit, an event after each sliding visit. The source stays still until the copy has read it: a later forward that
  would write ring slots the copy still reads (a forward reaching `copy_first + capacity`, so the rest of a prompt
  of 320 or more positions past the cut, or decode about 320 positions on) waits for that visit's event in each
  visit, a one-column forward for the whole copy. `DSparkProgram::save_async` stages the anchor context, which
  `draft()` rewrites, and copies the rings' visible positions; `extend()` waits for it.
- The copy that the next forward needs first goes first: the text rings when the rest of the prompt reaches them,
  else the drafter's, which the prompt's last extend needs.
- A snapshot is valid only once its copy's event completes: `restore()`, `save()`, `reset()` and a batched forward
  wait for it on the stream, so a request that resumes while the copy runs (a 1-token request, then at once its
  next turn) waits and never reads a half-copied snapshot. The layout-hash harness resumes from a `save_async`
  issued after the first chunk and raced by the chunks after it, and `ninfer_inkling2_dspark_test` races the
  drafter's and the target's copies against the next draft and forward; both equal the cold run.
- The copy is one paced copy kernel (`gather_copies_paced`, one block of 128 threads at about 2 GB/s). On this
  card's link (device-to-host writes at about 3.3 GB/s) neither faster way hides: a copy-engine copy holds the
  forward's own small device-to-host copies (the tensor-core experts' routing, the logits) behind it, and an
  unpaced copy kernel slows the forward's kernels beside it for about the copy's own time. A restore is on the
  critical path and uses the copy engine (198 MiB in 62 ms at a 6,144-position cut).
- `release_stream()` waits for a copy in flight before the programs and their pinned memory are freed, so unloading
  the model (the router's swap between models destroys the engine) returns the pinned memory with the rest.
- Startup sizes the pinned memory (N x 163 MiB with `--max-concurrency N`, where the drafter is off; plus the
  drafter's with N = 1) and refuses to start when the host cannot pin it with 1 GiB of RAM to spare or
  `RLIMIT_MEMLOCK` is lower, naming the sizes. With DSpark a refusal moves the fit to a narrower drafter window first.
  Batched decode keeps the synchronous `save()`.

Measured through `ninfer-serve` in the production configuration (262,144, DSpark, vision, both adapters, chunk
2048; TTFT is the server's `prompt_ms`, three requests each, ABBA against the same build with `--no-prefix-reuse`):

| Prompt | Without saving | Saving (this layout) | Saving, copy engine or unpaced kernel |
|---|---:|---:|---:|
| 7,979 (cut at 6,144, the rest reaches the rings) | 4,264-4,266 ms | 4,272-4,277 ms | 4,335-4,342 ms |
| 6,422 (the rest does not reach them) | 3,572-3,577 ms | 3,580-3,585 ms | 3,648-3,651 ms |
| 6,138 (10 positions past the cut) | 3,226-3,228 ms | 3,234-3,236 ms | - |
| 1,098 (no cut) | 657-701 ms | 657-701 ms | |

The save costs a prompt 8 ms (0.2%); on the device it cost about nothing, but the 238 MiB it freed buys chunk
2048 over 256, 2.4 times the prefill speed at 8K. The first 64 tokens decode as before (DSpark 81-89 tokens/s,
by prompt, either way), and the 12 chats decode at 78.87-78.89 plain and 90.00-90.11 with DSpark against
78.92-78.95 and 90.03-90.56 on the device-snapshot build in the same configuration.

## Batched decode

`--max-concurrency N` (2 to 8; default 1) runs up to N requests at once. Each request gets its own slot in the text
Program, and every decode step is one forward with one column per decoding request
(`TextProgram::forward_batch`, `Inkling2Core::worker_batched`). The default (N = 1) is unchanged: one request at a
time on the single-sequence decode graph, with DSpark when configured.

Exactness: each request's tokens and logits are bit-identical to its solo run.

- Every decode op except the attention and the short convolutions is already column-independent: the projections
  (per column at every width below the tensor cores' 64), the experts (the per-entry route, and the expert-major one
  described below, both bit-exact with T = 1), depth mix, loop controllers, norms, routing and the head.
- Experts of a call with several columns (`moe_situ_experts`, CUDA cores):
  - One column runs per entry: a block takes one entry's rows.
  - NVFP4 calls of up to 16 entries (T = 1 decode, B = 2) run gate_up and down in one launch. Blocks take items by
    an atomic ticket, every gate item before any down item. A down block issues its weight loads, waits for its
    entry's gate blocks on a counter, then reads the activations. The arithmetic is the two kernels', so the bits
    are unchanged. This removes the gap between the two launches (B = 1 serve +0.7%).
  - Calls of fewer than 64 entries (up to 7 columns at top 8, so batched decode up to B = 7 and DSpark verify) also
    run per entry. Entries of one expert run at the same time, so its second read mostly hits the L2, and the
    per-entry kernels' occupancy (64 registers, four blocks an SM) beats the expert-major kernels at these widths.
  - From 64 entries the call runs expert-major. Block (g, s) takes row group g of the s-th distinct expert in order
    of first appearance, lists that expert's entries itself (no grouping launch), and stages up to 4 of them at a
    time. A tile computes exactly its entries, with the per-entry kernel's arithmetic for each.
  - This replaced the grouped kernels, which computed 4 columns per tile whatever the expert's count and held one
    block an SM (166-198 registers).
- The stateful ops take `ops::ColumnSequences`: a device table of each column's position and slot, and the byte
  stride between slots.
  - `short_conv_fp32_batch` reads each column's window from that column's own history and writes only that history
    back.
  - `head_norms_fp32` takes the log scale from the column's position and appends at that position in the column's
    ring.
  - `relative_bias_sink_attention` reads the column's rings and decides per column whether to split its keys. It
    splits exactly when a one-column call at that position would (more than `kSplitMinKeys` visible keys); otherwise
    block z = 0 runs the single block's loop and store. So a batch can mix columns of 100 and 100,000 keys and each
    keeps its solo bits.
- B is at most 8 (`kMaxBatchColumns`, the split attention's column limit), far below the tensor-core width, so decode
  never takes a tensor-core route.
- Tested by `ninfer_inkling2_batch_test`, which compares every logit byte for byte against each prompt's solo run.
  Requests arrive staggered at 2, 4 and 8 in flight, the prompt lengths straddle the split threshold and the sliding
  window, and slots are reused.
- `ninfer_inkling2_engine_batch_test` checks the same through the public Engine: concurrent requests (greedy, a seeded
  sample, a thinking budget, a stop string, streaming and cancellation) give their one-at-a-time outputs.
- `ninfer_inkling2_fp32_ops_test` (`test_column_sequences`) checks the three ops' column mode against one-column
  calls, with no artifact needed.

State: all N slots are laid out identically in one allocation, with slot s at `s * slot_stride` from slot 0. A
captured graph (one per batch width B, captured on first use) therefore reads the slots, positions and tokens from
the step table, and serves every combination of slots. Prefill stays one request at a time. A new request prefills one chunk
(eager, in its own slot) between decode steps, round robin among those still prefilling. The rollback buffers and
taps are a single copy, but they are unused, because with N > 1 there is no drafter (`--spec dspark` is accepted and
noted in the startup log, and every request decodes plain). `--lora` and `--graft` are refused with N > 1: the adapters
are one Program-wide selection set per request, which a batched forward cannot apply to one column alone. Prefix reuse keeps one snapshot per slot. A prompt
resumes only from the snapshot of the slot it is given, since the global caches below it are that slot's own. The
scheduler gives a prompt the free slot whose snapshot it extends (the longest such snapshot), or else an empty slot,
or else the slot released longest ago.

Memory: `--max-context` stays the per-request ceiling, as on the other routes. Each slot holds its own rings:

| Per slot | Bytes (FP16 rings) |
|---|---|
| global KV, 8 visits | 32 KiB x context |
| local KV, 40 visits (ring of 1023 + chunk, at most 1023 + 319) | 200 MiB at chunk 256, 210 MiB at a chunk of 320 or more |
| convolution histories | 3.7 MiB |
| prefix-reuse snapshot (when on) | 163 MiB of pinned host memory, none on the device |

At startup the core first tries the widest prefill chunk at N x `--max-context`. If no chunk fits, it takes the
narrowest chunk and lowers the context per slot to the largest multiple of 256 that fits beside the 256 MiB spare,
found by halving, never below `fit_context_floor` (the startup fails when the floor does not fit). The Engine's `max_context`, the startup log's `batched decode |` note, and the server's limits and
`/v1/models` follow it. `kv_capacity` reports N x the context per slot.


Measured on the V100 (32 GB, sm_70), Q8 artifact, FP16 rings. The `dad1623b` columns are master `dad1623b`
(2026-11-22, two runs each, the output of every request the same at every N); the columns marked "at the PR" are this
section's earlier measurement:

| N (`--max-concurrency`) | aggregate tok/s (serve, at the PR) | aggregate tok/s (serve, `dad1623b`) | per request tok/s (`dad1623b`) | step bench, aggregate (at the PR) |
|---|---|---|---|---|
| 1 | 71.3 | 77.1 / 77.1 | 79.1 | 67.7 |
| 2 | 113.5 | 121.7 / 121.8 | 63.0 | 113.6 |
| 4 | 169.0 | 169.1 / 170.5 | 45.6 | 176.5 |
| 8 | 169.8 | **196.8 / 198.2** | 26.4 | 198.6 |

- Serve: `ninfer-serve --max-concurrency N` at 8,192 context, N concurrent greedy chat requests of 256 tokens with
  different prompts. Aggregate is all completion tokens over the wall time, prefill included; per request is the
  server's `predicted_per_second`. Each request's output is identical at every N.
- Step bench: `NINFER_INKLING2_BATCH_BENCH=1 ninfer_inkling2_batch_test`, B full columns at about 1,000 positions,
  one distinct token per column.
- B = 1 is unchanged. ABBA against the base build gave plain 72.0 vs 72.0 tok/s and DSpark the same within noise.
- From B = 4 to B = 8 the experts dominate. Eight diverse columns route to up to 8 x 8 distinct experts, and every
  distinct expert's bytes must be read each step (op bench of the routes against that floor:
  `NINFER_INKLING2_OP_SWEEP=1 ninfer_inkling2_fp32_ops_test`). The 8-column projections and the attention take
  most of the rest of the step.
- Context per slot when `--max-context 262144` is asked with N = 8: 27,392 positions (8 x 27,392 KV rows, chunk 256),
  reported by `/v1/models`. Lower N or a smaller `--max-context` keeps more context per slot. With `--vision` and
  the snapshots in host memory (2026-11-22): 29,696 at N = 8 and 66,048 at N = 4 (24,576 and 60,928 while they
  were on the device), with 8 x 163 and 4 x 163 MiB pinned.
