# Inkling-2-Mini (`inkling2`, looped) model

Status: **design and phase-1 tooling only**. Nothing in the engine runs this model yet. This page is
the mathematical authority for the work, plus the GGUF mapping, reuse map and V100 plan. The
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

ninfer's DFlash ([dflash.md](dflash.md)) already has the masked-block, non-causal, context-K/V shape
with separate context roles. What is new: per-layer softmax tap fusion instead of one feature
projection, taps counted in visits, the Markov RNN chain instead of per-column argmax, and global
attention in a draft layer.

## Vision (`mmproj-*.gguf`, `clip`, projector `inkling2v`)

A ViT: 23 layers, width 1024, 16 heads, MLP 4096 with exact GELU, LayerNorm eps 1e-5, axial 2-D RoPE
with theta 10000, native patch 20, temporal patch 2. Patch embedding is linear (HWC). A 2x2
adapter merge, then `proj_in` / `proj_out` up to H 2304 (projection hidden 4096). Mean and std are
0.5. The tower is about 0.6 GiB in F16/BF16. Details are deferred to the vision phase.

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
| 0 (done) | this spec; header fixtures of three GGUFs and the drafter; converter config, visit schedule and tensor map (`tools/convert/inkling2.py`); NumPy reference with the GGML NVFP4 decoder and checkpoint repack (`tools/inkling2/reference.py`); C++ FP64 op oracles (`tests/ops/inkling2_reference.h`, host-only, not yet in CMake); tokenizer contract check. **Not** done: the `ggml_nvfp4` artifact format (Python and C++ registries must land together), the artifact writer, any engine code | -- |
| 1. Golden (done) | CPU llama.cpp (Unsloth SHA above) on UD-Q8_K_XL: logits and per-visit activations for four prompts; the NumPy reference on the real weights against them (see "Validation") | -- |
| 2. Converter | `inkling2_gguf` recipe writes the artifact: C++ format registry for `ggml_nvfp4` and the new records, per-expert scales, padded vocab, tokenizer and chat resources. Reader test on the real file | 2-3 |
| 3. Text MVP, correctness first | Model, config and binding. Program with the 48-visit schedule. New Ops with naive FP32 kernels qualified against the oracles: rel-bias sink attention (decode and prefill), conv streams, MIX, CTRL, clips, sigmoid router, resident NVFP4/Q8_0/BF16 MoE. Linear problem registrations. Tokenizer matcher. Greedy text equals the golden | 8-12 |
| 4. V100 decode performance | Fused decode kernels (MIX, CTRL, conv and norms), expert GEMV for NVFP4 via dp4a, global attention on FP16 KV, CUDA Graph per decode width; measure against the bandwidth floor | 5-8 |
| 5. Prefill | cuBLAS HMMA for the dense and expert GEMMs (dequantize tiles to FP16), flash prefill with the relative bias and sink (extend the vendored `fattn` like [L]'s banded op) | 4-6 |
| 6. DSpark | Drafter binding, tap capture at visits 5/23/41/48, Markov chain with the precomputed `W_head Wo`, verify widths 2..8 | 4-6 |
| 7. Vision | mmproj conversion and the ViT | 4-6 |
| 8. Chat/serving | template, reasoning and tool parsers, effort control | 2-3 |

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
   by `tools/inkling2/gguf_weights.py`) against that golden. Result on UD-Q8_K_XL, 2026-11-20:

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
4. Each new Op qualified against its oracle on the real shapes (op-development.md).
5. Engine greedy text and perplexity against the golden. The acceptance bar is KLD at most
   0.002 against llama.cpp on the same GGUF, measured on prompts without near-tied routing, and
   the same top-1 token at every position of the golden prompts.
