# Ternary Bonsai 2 27B on the RTX 3090

[Performance index](../performance.md) · [Model card](../../model-cards/Ternary-Bonsai-2-27B-NInfer-3090/README.md)

Everything on this page was measured by this fork on **one RTX 3090 (`sm_86`), CUDA 12.8, Windows**,
unlike most pages under `performance/`, which carry upstream's RTX 5090 numbers. They are
single-card, few-prompt measurements, run on 2026-10-15 with greedy decoding and `--kv-dtype int8`
unless stated. The artifact is `bonsai2_27b.ninfer` (8.87 GiB, SHA-256 `97512677...6d4634`).

## Summary

| | Ternary Bonsai 2 27B | Qwen3.8-27B (this fork's 6-bit artifact) |
|---|---:|---:|
| Weights on the device | 6.70 GiB (7.28 with MTP) | 14.6 GiB (17.0 with DFlash2) |
| Plain decode | 88 tok/s | 48 tok/s |
| DFlash2, 7 drafts, one coding prompt | 297 tok/s | 195 tok/s |
| Prefill, 1k / 4k / 16k | 2,853 / 2,776 / 2,576 tok/s | 1,717 / 1,687 / 1,615 tok/s |
| `--prefill-cublas --prefill-chunk 4096`, 1k / 4k / 16k | 2,791 / 2,447 / 2,294 tok/s | 2,421 / 3,093 / 2,869 tok/s |
| Quick-corpus perplexity | 5.627 | n/a (a different model) |

The ternary model decodes 1.8 times faster plain on 2.2 times fewer weight bytes, so decode here is
bound by weight traffic and the ternary dequantisation is cheap.
`--prefill-cublas` is slower than the default on the ternary artifact, where it adds 43-83% to the
6-bit one.

## Speculative decoding

DFlash2 with 7 drafts on four prompts (code, networking, arithmetic, an email; 300 tokens, thinking
off), through `ninfer --prompt`:

| Variant | Mean decode | Mean acceptance length | Per prompt (tok/s) |
|---|---:|---:|---|
| Q4 adapter (shipped) | **249.9 tok/s** | 3.97 | 307, 195, 325, 173 |
| Q8 adapter | 239.1 tok/s | 4.00 | 297, 185, 309, 167 |
| Q4 adapter, `--lm-head-draft` | 249.4 tok/s | 3.97 | 307, 193, 324, 174 |
| Q8 adapter, `--lm-head-draft` | 237.8 tok/s | 4.00 | 296, 184, 306, 166 |

- The drafter is a large share of each draft step's bytes against a 2.125-bit target, so its
  feature, output and MLP projections are Q4: **+4.5% decode** for a 0.8% lower acceptance length.
  The gain held on every prompt.
- The ternary proposal head (`--lm-head-draft`) is lossless (acceptance is identical) but did not
  change decode speed here, so it stays optional.
- `ninfer_bench`'s corpus is random token ids, which a drafter cannot predict (acceptance about
  15%); use real prompts for speculative measurements.

## Where prefill goes

`nsys` over a 4,096-token prefill (1,024-token chunks, 1.51 s of kernels):

| Kernels | Share |
|---|---:|
| The four `t2_prefill_i8_kernel` shapes (gate/up, down, QKV/Z splits, outputs) | 81% |
| Hadamard producers (`silu_mul_hadamard`, `rmsnorm_hadamard`, `gated_rmsnorm_hadamard`) | ~4% |
| GDN chunk recurrence | 3.4% |
| Causal attention | 3.1% |
| Activation quantisation | ~2.7% |

`tools/tensor_core_rate_probe.cu` measures this card's int8 `m16n8k32` peak at 316 TOPS (1,695 MHz).
The ternary GEMMs sustain about 160 TOPS, roughly half, so the only large lever left is the kernel
itself. The integer route is 2.7 times faster than `--no-prefill-a8`;
`NINFER_T2_A8_TILE=off` costs about 30%; `NINFER_T2_A8_RASTER=rows|cols` does not beat the default.

### Int4 activations, emulated and rejected

The card's int4 peak is 637 TOPS, twice int8, and a ternary weight fits int4 exactly. Quantising the
activations to 4 bits is the quality risk, so it was emulated before any kernel was written: the
prefill quantiser clamped to +-7 levels, running through the existing int8 GEMM.

| Activations | Quick-corpus perplexity | Change |
|---|---:|---:|
| int8 (shipped) | 5.627 | |
| int4, scale = absmax / 7 | 5.788 | +2.9% |
| int4, scale = 0.75 absmax / 7 | 5.816 | +3.4% |

This fork has accepted a perplexity cost of +0.08% for a speed gain and rejected +0.69%, so the
2x GEMM that int4 would offer is not taken. Clipping the scale made it worse, so outliers matter.

## Reproduction

```text
ninfer_bench --weights models\bonsai2_27b.ninfer -p 1024,4096,16384 -n 16 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 20480
ninfer models\bonsai2_27b.ninfer --prompt "..." --max-new 300 --greedy --no-thinking --spec dflash2 --draft-tokens 7
ninfer-perplexity models\bonsai2_27b.ninfer --corpus eval\corpora\perplexity-1m\manifest.json --quick --kv-dtype int8
```

Run GPU work under the lock described in `CLAUDE.md`. The Q8-adapter twin used for the drafter
comparison was converted with the same recipe and an override that assigns the adapter's Q4
projections `q8_g32_fp16`.
