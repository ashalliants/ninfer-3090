# Qwen3.8-Flash-Next (`qwen4exp`) model

The mathematics, instance config and logical parameters of the qwen4exp model as NInfer runs it
from the ISTA DASLab GSQ-RCO IQ2_XS GGUF. The composition is lifted from Infernix (`a3edb450`,
`src/models/qwen4_exp`, Apache-2.0); the weights stay the GGUF's GGML blocks. Status and the staged
plan are in [flash-next-plan.md](flash-next-plan.md); formats in [tensor-formats.md](tensor-formats.md).

Milestone M0: the whole model runs through `ninfer_qwen4_exp_forward_real_test` (one sequence,
teacher-forced or decoding). There is no Program, Engine, frontend or serving yet (PR 9).

## Shape

| Quantity | Value |
|---|---|
| Hidden H, hyper-connection streams S, residual width S*H | 2560, 4, 10240 |
| Blocks | 48: GDN, except QSA at layers 3, 7, ..., 47 |
| GDN | 16 key heads, 48 value heads, head 128, conv kernel 4, output gate (see conventions) |
| QSA | 24 query heads, 2 KV heads, head 256, rotary 64 (interleaved MRoPE 11/11/10), indexer 4 x 128, blocks of 4, budget 2048 tokens |
| MoE | 512 routed experts (intermediate 640), top 10, renormalized, one shared expert (640) behind a sigmoid score |
| PLE | one injection before layer 1 (zero-based): 16 n-gram rows of 160 values (bigram heads 0-7, trigram 8-15), conv kernel 4, dilation 3 |
| Vocabulary | 248,320; the n-gram EOS is 248044 |

## Forward

Per call of T consecutive positions of one sequence (`execution/forward.cpp`):

1. `x0 = embed(ids)` (GGML IQ4_XS rows decoded to BF16); `R = expand(x0)` into S streams.
2. For each block: the PLE injection (its layer), `xa, inject = mix(attn_hc, R)`,
   `y = GDN(xa)` or `QSA(xa)`, `R += inject * y`, `xm, inject = mix(mlp_hc, R)`, `y = MoE(xm)`,
   `R += inject * y` (each update rounded once to BF16).
3. `x = mix(final_mixer, R)`; FP32 logits of the IQ4_XS head with the A8 activation cast.

The mixer, injection and PLE are the `hyper_connection` and `ple` Ops; their formulas are in their
headers. GGML-format projections run through `linear` with the A8 (Q8_1) activation cast; BF16
projections are A16. The router and shared-expert score are `projection_fp32` rows (FP32 logits);
the experts are `offloaded_sparse_moe` (canonical A8 arithmetic).

**GDN.** `qkv = W_qkv x` (C = 10240 rows: q, k, v), `z = W_z x`, `[a; b] = W_ab x`;
`g = decay * softplus(a + dt_bias)`, `beta = sigmoid(b)` (`gdn_gating_decay`; `decay` is the
stored `-exp(A_log)`); the causal SiLU convolution splits into q [128, 16], k [128, 16], v
[128, 48]; value head h reads key head `h mod 16` (tiled order), so q and k are tiled to 48 heads
and the recurrence runs with one key head per value head; `o = GatedDeltaNet(...)` (normalized
q/k, scale 1/sqrt(128)); `y = W_out (norm(o) * w * act(z))` per value head.

**QSA.** `W_q` rows interleave per head: 256 query rows, then 256 gate rows. q and k are
RMS-normalized with `1 + gamma`, rotated (64 of 256 dims), K/V appended to the INT8-G64 paged
cache; the indexer's BF16 rows give 4 index queries and one raw index key per token; complete
blocks of 4 raw keys are pooled, normalized and rotated; each query selects at most 512 blocks
(`sum_h relu(<q_h, k_b>) / sqrt(128)`, ties to the lower block) plus its open tail;
`out = attention(q, selected keys)`, `y = W_o (out * sigmoid(gate))`.

**MoE.** `logits = [W_router; w_score] x` in FP32; softmax over 512, top 10 (lower id on ties),
renormalized; each expert is `A8(x)`, gate/up rowdots, `SiLU(gate) * up` in FP32, `A8(h)`, the down
rowdot rounded to BF16; `y = sum_i w_i expert_i + sigmoid(score) * shared(x)` with the shared
expert `W_d A8(SiLU(W_g A8(x)) * W_u A8(x))`.

## Logical parameters

Parameters mirror the GGUF tensors (`tools/convert/qwen4_exp.py`), so most hold the GGUF bytes
unchanged. Per layer `L` under `text/layers/L/`:

| Parameter | Shape | Stored as |
|---|---|---|
| `attn_hc/{norm,down,inject,up}`, `mlp_hc/...` | [S*H], [320, S*H], [S, S*H], [S*H, 320] | norm FP32 multiplier; down and inject one BF16 parent [324, S*H] |
| `gdn/qkv`, `gdn/z`, `gdn/output` | [10240, H], [6144, H], [H, 6144] | GGML (IQ4_XS, IQ3_S, ...) |
| `gdn/a_projection`, `gdn/b_projection` | [48, H] each | one BF16 parent [96, H] |
| `gdn/a`, `gdn/dt_bias` | [48] | FP32; `gdn/a` is `-exp(A_log)` |
| `gdn/convolution` | [4, 10240] | BF16, tap-major (the GGUF's [C, K] transposed) |
| `gdn/norm` | [128] | BF16 plain multiplier |
| `attention/{query_gate,key,value,output}` | [12288, H], [512, H], [512, H], [H, 6144] | GGML |
| `attention/{query,key}_norm`, `indexer/{query,key}_norm` | [256], [128] | BF16 gamma (unit offset) |
| `indexer/query`, `indexer/key` | [512, H], [128, H] | BF16 |
| `moe/router`, `moe/shared_score` | [512, H], [1, H] | BF16 |
| `moe/shared/{gate,up,down}` | [640, H], [640, H], [H, 640] | GGML |
| `moe/experts` | [512, H, 640] | `ggml_expert_record_v1`, pinned host memory |
| `ple/{key,value}` (layer 1) | [S*H, 2560], [H, 2560] | one BF16 parent [12800, 2560] |
| `ple/{key,query,conv}_norm`, `ple/convolution` | [S*H], [S*H, 4] | FP32 (the convolution is the GGUF's F16 widened, channel-major) |

Globals: `text/token_embedding` and `text/output_head` [248320, H] (GGML IQ4_XS),
`text/final_mixer/{norm,down,up}`. The n-gram table is the separate volume (`<artifact>.ngram`).

## Stored conventions

See [flash-next-plan.md](flash-next-plan.md#export-conventions-of-this-gguf) for the evidence of each.

- Every norm except `ssm_norm` is the GGUF's `fl32(1 + gamma)` of a BF16 `gamma`. The
  hyper-connection and PLE norms keep the stored F32 multiplier; the attention and indexer norms
  are stored as `gamma` (the converter forms `v - 1` exactly and refuses any value without an exact
  BF16 gamma that reproduces `v`), which the attention Ops add 1 to in FP32: the same gain.
- `gdn/norm` and `gdn/convolution` are F32 words that are BF16 words; they are stored as BF16 after
  the same refusal check.
- GDN value heads are in tiled order (value head `h` reads key head `h mod 16`) in every GDN
  tensor, including the K axis of `gdn/output`, which a block format cannot permute; the GDN output
  gate is a sigmoid; the PLE and GDN convolution taps are in the GGUF's order (tap `j` reads
  `t - (K-1-j) * dilation`); `attention/query_gate` interleaves each head's query and gate rows.
  Each was settled in PR 8 by running the model with the alternative: every alternative is far
  worse against Strata (the plan's convention table).
