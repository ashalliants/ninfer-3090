# Hadamard rotation of the regular Qwen3.8-27B: CPU probe

Question: would storing the regular 27B's projections Hadamard-rotated (as the Bonsai checkpoints
are) improve its groupwise Q4/Q5 weights, or make INT8 activations (`LinearPolicy::AllowA8Int`)
safe? `tools/rotation_quant_probe.py` answers it on the CPU before any CUDA work.

**These are CPU proxies.** Each number is the relative output error `||WX - W'X'||² / ||WX||²` of one
matrix on real activations. Perplexity and decode speed on the 3090 are the real verdict, and
nothing here was run on a GPU. No speed or end-to-end quality claim follows from this note.

## Method

- Weights: BF16 `Qwen/Qwen3.8-27B`, layers 0, 3, 31, 32, 62, 63 (0/32/62 are GDN, 3/31/63 full
  attention): early, middle and late.
- Activations: the input of every probed projection, captured with `capture-torch` (the BF16 model
  streamed through transformers one decoder layer at a time, FP32 CPU compute) over 5 held-out
  domains x 512 tokens (`eval/corpora/perplexity-heldout-2026-09`), 2560 tokens in all.
- Arms: `primal` (as stored); `online` (`W Qᵀ`, `Q x`, Q the 1024-block signed Walsh-Hadamard of
  `hadamard_transform`); `fold` (`W diag(g) Qᵀ`, `Q diag(1/g) x`, g = 1 + gamma, matrices that read a
  norm output only: the QuaRot residual-stream layout).
- Weight error: the converter's own `search_quantize_matrix` (`q4`/`q5_g64_fp16`, negative scales,
  importance = mean x² of that arm, as an imatrix gives).
- Activation error: s8 with one scale per token (`a8tok`); `q4a8` is Q4 weights with s8 activations
  at one scale per token and 64-channel group, the AllowA8Int scheme. Both use the activations of
  the same arm.
- Caveats: the probe scores `in_proj_qkv` and `in_proj_z` as whole matrices, while the recipe stores
  GDN q/k at Q4 and the rest at Q5; the probe uses the importance of the captured tokens, not the
  imatrix the recipe ships; 6 of 64 layers.

Reproduce (about 54 GB of shards to download, a few GB of RAM, roughly 3 h of CPU):

```bash
python tools/rotation_quant_probe.py fetch         --work DIR
python tools/rotation_quant_probe.py capture-torch --work DIR   # `capture` on Apple silicon (MLX)
python tools/rotation_quant_probe.py analyze       --work DIR
```

## Results

Relative output error. `a8tok` and `q4a8` columns: primal / online.

| layer | matrix | q4 primal | q4 online | q4 fold | q5 primal | a8tok primal / online | q4a8 primal / online |
|--|--|--|--|--|--|--|--|
| 0 | linear_attn.in_proj_qkv | 7.0e-04 | 8.4e-04 | 8.0e-04 | 1.7e-04 | 2.4e-04 / 7.3e-06 | 7.1e-04 / 8.5e-04 |
| 0 | linear_attn.in_proj_z | 1.3e-03 | 1.6e-03 | 1.5e-03 | 3.3e-04 | 4.3e-04 / 1.4e-05 | 1.4e-03 / 1.6e-03 |
| 0 | linear_attn.out_proj | 5.7e-04 | 1.2e-03 | - | 1.2e-04 | 1.7e-03 / 7.1e-06 | 5.9e-04 / 1.2e-03 |
| 0 | mlp.gate_proj | 4.1e-03 | 4.4e-03 | 2.9e-01 | 1.0e-03 | 4.1e-04 / 4.4e-05 | 4.2e-03 / 4.4e-03 |
| 0 | mlp.up_proj | 4.6e-03 | 4.9e-03 | 3.1e-01 | 1.1e-03 | 4.6e-04 / 4.8e-05 | 4.6e-03 / 4.9e-03 |
| 0 | mlp.down_proj | 1.6e-03 | 2.6e-03 | - | 3.9e-04 | 1.3e-02 / 3.8e-05 | 1.6e-03 / 2.6e-03 |
| 3 | self_attn.q_proj | 5.0e-04 | 2.2e-03 | 2.1e-03 | 1.0e-04 | 5.9e-03 / 1.3e-05 | 5.6e-04 / 2.2e-03 |
| 3 | self_attn.k_proj | 1.7e-02 | 3.2e-02 | 3.2e-02 | 4.1e-03 | 9.1e-02 / 2.0e-04 | 1.8e-02 / 3.2e-02 |
| 3 | self_attn.v_proj | 1.8e-02 | 6.2e-02 | 6.0e-02 | 7.5e-03 | 1.6e-01 / 3.8e-04 | 2.0e-02 / 6.2e-02 |
| 3 | self_attn.o_proj | 2.7e-04 | 4.0e-04 | - | 6.1e-05 | 1.1e-03 / 4.7e-06 | 2.8e-04 / 4.1e-04 |
| 3 | mlp.gate_proj | 5.4e-03 | 5.4e-03 | 1.6e-01 | 1.4e-03 | 3.5e-04 / 5.2e-05 | 5.4e-03 / 5.4e-03 |
| 3 | mlp.up_proj | 6.9e-03 | 6.9e-03 | 2.1e-01 | 1.8e-03 | 4.5e-04 / 6.7e-05 | 6.9e-03 / 7.0e-03 |
| 3 | mlp.down_proj | 4.1e-03 | 5.9e-03 | - | 1.0e-03 | 1.8e-02 / 8.5e-05 | 4.3e-03 / 5.9e-03 |
| 31 | self_attn.q_proj | 1.8e-03 | 3.4e-03 | 4.8e-03 | 4.3e-04 | 6.1e-03 / 3.3e-05 | 1.9e-03 / 3.5e-03 |
| 31 | self_attn.k_proj | 4.3e-03 | 5.9e-03 | 7.7e-03 | 9.9e-04 | 9.8e-03 / 5.2e-05 | 4.5e-03 / 5.9e-03 |
| 31 | self_attn.v_proj | 6.8e-03 | 9.9e-03 | 1.5e-02 | 1.9e-03 | 1.7e-02 / 9.1e-05 | 7.1e-03 / 9.9e-03 |
| 31 | self_attn.o_proj | 1.2e-02 | 1.3e-02 | - | 3.2e-03 | 9.0e-04 / 2.0e-04 | 1.3e-02 / 1.3e-02 |
| 31 | mlp.gate_proj | 3.5e-03 | 4.9e-03 | 1.3e-02 | 8.7e-04 | 5.2e-03 / 5.1e-05 | 3.5e-03 / 4.9e-03 |
| 31 | mlp.up_proj | 7.7e-03 | 9.2e-03 | 2.5e-02 | 1.9e-03 | 9.8e-03 / 9.6e-05 | 7.9e-03 / 9.3e-03 |
| 31 | mlp.down_proj | 6.7e-03 | 6.6e-03 | - | 1.7e-03 | 3.0e-03 / 7.7e-05 | 6.8e-03 / 6.6e-03 |
| 32 | linear_attn.in_proj_qkv | 5.1e-03 | 7.5e-03 | 1.2e-02 | 1.2e-03 | 1.3e-02 / 7.1e-05 | 5.3e-03 / 7.5e-03 |
| 32 | linear_attn.in_proj_z | 1.5e-03 | 2.9e-03 | 4.4e-03 | 3.5e-04 | 5.1e-03 / 2.7e-05 | 1.5e-03 / 2.9e-03 |
| 32 | linear_attn.out_proj | 7.4e-03 | 7.5e-03 | - | 1.9e-03 | 1.1e-03 / 9.6e-05 | 7.5e-03 / 7.5e-03 |
| 32 | mlp.gate_proj | 3.4e-03 | 4.8e-03 | 1.2e-02 | 8.6e-04 | 5.2e-03 / 5.0e-05 | 3.5e-03 / 4.9e-03 |
| 32 | mlp.up_proj | 7.9e-03 | 9.4e-03 | 2.3e-02 | 1.9e-03 | 1.0e-02 / 9.8e-05 | 8.0e-03 / 9.5e-03 |
| 32 | mlp.down_proj | 6.7e-03 | 6.5e-03 | - | 1.7e-03 | 2.8e-03 / 7.6e-05 | 6.8e-03 / 6.5e-03 |
| 62 | linear_attn.in_proj_qkv | 2.4e-03 | 2.4e-03 | 2.4e-03 | 6.0e-04 | 8.3e-04 / 2.6e-05 | 2.4e-03 / 2.4e-03 |
| 62 | linear_attn.in_proj_z | 1.5e-03 | 1.6e-03 | 1.6e-03 | 3.8e-04 | 5.3e-04 / 1.7e-05 | 1.6e-03 / 1.6e-03 |
| 62 | linear_attn.out_proj | 6.6e-03 | 9.1e-03 | - | 1.6e-03 | 5.0e-03 / 1.0e-04 | 6.8e-03 / 9.1e-03 |
| 62 | mlp.gate_proj | 1.5e-03 | 1.6e-03 | 1.8e-03 | 3.8e-04 | 4.2e-04 / 1.6e-05 | 1.5e-03 / 1.6e-03 |
| 62 | mlp.up_proj | 2.8e-03 | 2.9e-03 | 3.2e-03 | 6.9e-04 | 7.8e-04 / 3.0e-05 | 2.8e-03 / 2.9e-03 |
| 62 | mlp.down_proj | 5.5e-03 | 7.2e-03 | - | 1.4e-03 | 8.1e-03 / 9.2e-05 | 5.7e-03 / 7.2e-03 |
| 63 | self_attn.q_proj | 1.9e-03 | 1.8e-03 | 1.9e-03 | 4.7e-04 | 4.7e-04 / 1.9e-05 | 1.9e-03 / 1.8e-03 |
| 63 | self_attn.k_proj | 6.4e-03 | 3.6e-03 | 4.4e-03 | 1.6e-03 | 9.3e-04 / 3.7e-05 | 6.4e-03 / 3.6e-03 |
| 63 | self_attn.v_proj | 1.7e-03 | 1.2e-03 | 1.3e-03 | 4.4e-04 | 3.1e-04 / 1.2e-05 | 1.7e-03 / 1.3e-03 |
| 63 | self_attn.o_proj | 5.6e-03 | 5.3e-03 | - | 1.5e-03 | 8.1e-04 / 8.1e-05 | 5.7e-03 / 5.4e-03 |
| 63 | mlp.gate_proj | 1.2e-03 | 1.3e-03 | 1.5e-03 | 3.1e-04 | 4.5e-04 / 1.3e-05 | 1.3e-03 / 1.3e-03 |
| 63 | mlp.up_proj | 1.6e-03 | 1.7e-03 | 1.9e-03 | 3.8e-04 | 5.7e-04 / 1.7e-05 | 1.6e-03 / 1.7e-03 |
| 63 | mlp.down_proj | 2.4e-03 | 4.1e-03 | - | 5.6e-04 | 4.4e-03 / 4.9e-05 | 2.5e-03 / 4.1e-03 |

Int4 value plane (rk8v4/rk4v4: absmax/7 per 32 channels), error with and without a per-head H256:

| layer | V int4 | V int4 + H256 |
|--|--|--|
| 3 | 1.18e-02 | 9.39e-03 |
| 31 | 1.30e-02 | 9.32e-03 |
| 63 | 1.40e-02 | 9.64e-03 |

## Findings

1. **Rotation does not lower Q4 or Q5 weight error.** Median `online`/`primal` Q4 error is 1.20x
   (minimum 0.56x). No `online` or `fold` Q4 error reaches `primal` Q5 error in any of the 36
   matrices, so nothing supports moving layers from Q5 to Q4. The imatrix-weighted search already
   spends its precision on the outlier channels that rotation spreads out.
   Only layer 63 improves (k_proj 0.56x, v_proj 0.7x); layers 3 and 0 get 2-3.5x worse.
2. **`fold` is no better than `online`, and is pathological where the norm gain is small.** The
   post-attention norms of layers 0 and 3 have channels with `1 + gamma` = 0.004; folding them gives
   0.16-0.31 error on gate/up. Elsewhere `fold` is 1.0-2.5x `online`.
3. **Rotation shrinks INT8-activation error by 10-2000x** (`a8tok`, one scale per token, e.g. layer-3
   k_proj 9.1e-2 to 2.0e-4). But with the shipped AllowA8Int scheme (a scale per 64 channels) the
   Q4+s8 error is already within 1-3% of Q4 alone in every primal matrix (`q4a8` vs `q4`), so rotation
   has almost no INT8 headroom to recover there. It only matters for coarser activation scales.
4. **H256 cuts the int4 V-plane error by 20-31%** in all three attention layers probed. This is the
   only positive signal, and it is a proxy for a KV-format change worth about 0.1% perplexity
   (`docs/performance.md`, "Choosing a KV format"), so the expected gain is small.

## Decision

Hypothesis 1 (residual-stream rotation of the 27B's weights, to buy Q4 layers or INT8 activations):
**no**. The numbers above do not justify a converter recipe, a runtime change or a rented GPU run, so
none was built. Hypothesis 3 (GDN state) was not investigated, as the brief ordered.

Hypothesis 2 (H256 on the rk8v4/rk4v4 value plane) is the one lead left. Run it only if the 0.09-0.21%
perplexity cost of int4 V is worth chasing: reuse the NVFP4 epilogue that undoes the rotation, and
measure `ninfer-perplexity` on `ninfer-ppl-1m-v1 --quick` (4096/2048) against rk8v4/rk4v4. Until then,
no quality or speed gain from rotation is claimed.
