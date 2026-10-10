# Qwen3.8-Flash-Next on the RTX 3090: working plan

Temporary. This file tracks the staged work to run Qwen3.8-Flash-Next (GGUF architecture
`qwen4exp`) from the ISTA DASLab GSQ-RCO IQ2_XS GGUF. Delete it when the last milestone lands and
its stable facts have moved to their references (a `qwen4-exp-model.md`, the format and layout
references, `engine-architecture.md`).

## Status

| PR | Scope | Status |
|---|---|---|
| 0 | Strata baselines: decode, MTP, prefill, teacher-forced log-probs on frozen token ids | not started |
| 1 | GGUF reader, nine exact GGML block formats, `ggml_blocks_v1` / `ggml_rows_page4k_v1`, `qwen4_exp` name map, tokenizer synthesis, `--subset dev` | done (Python only) |
| 2 | C++ registration and materialization of the GGML formats and layouts; host exact decoder | done |
| 3 | Dense GGML linears: `linear`/`linear_add` for IQ4_XS, IQ3_S, Q6_K, IQ4_NL, Q8_0, Q2_0 at the model's 19 dense problems, Q8_1 activation profile | done (perf gates below) |
| 6 | Ops from Infernix: `hyper_connection` (mix, inject, expand), `ple` (IQ4_NL `ple_embed`, gate, stateful convolution, commit), `projection_fp32` (BF16 router; IQ4_XS head on PR 3's decode route), `rows` | done (HC perf gate missed, below) |
| 4-9 | QSA, PLE frontend and stream residency, model skeleton, MoE on GPU, MoE on CPU | not started |
| 10 | Whole model at 32K, quality gate against Strata | not started |
| 11-14 | Residency policy and miss split, 128K, MTP, vision | not started |

The C++ loader reads and places every `ggml_*` object. Linear and LinearAdd execute the six dense
formats (PR 3); the expert formats IQ2_S, IQ2_XXS and IQ1_M wait for the MoE Op (PR 8).

## PR 3 measurements

RTX 3090, CUDA 12.8, cold L2. Decode is IQ4_XS [10240, 2560] (13.93 MB); wide compares the Q8_1
cast plus int8 MMA with dequantize-to-FP16 plus cuBLAS (`cublasGemmEx`, FP32 compute).

| Workload | Result |
|---|---|
| Decode T = 1, kernel (nsys) | 20.2 us = 74% of the 936 GB/s spec; a plain read of the same bytes takes 17.2-17.7 us (79-81%) |
| Decode T = 1, public Op in a graph (events) | 22.5 us = 66% |
| Decode vs Strata `native_mmvq`, same bytes (nsys) | T = 1: 20.2 vs 21.1 us (+2.1 us Strata quantize); T = 8: 30.9 vs 62.9 us |
| Wide, T = 512 / 4096, ours over dequant + cuBLAS | 1.02-2.02x for five formats; Q6_K 0.96x / 0.77x |

The 80% decode gate is not met; see the PR 3 description for the tuning that was tried.

## PR 6 measurements

RTX 3090, CUDA 12.8, L2 flushed before each graph replay. "Chained" is 8 calls per graph on
distinct weight copies (the per-call cost inside a decode graph).

| Workload | Result |
|---|---|
| `hyper_connection_mix`, T = 1, 13.2 MB BF16 (block mixer / final mixer) | chained 28.2 / 26.8 us = 50% / 52% of 936 GB/s; single call 30.7 / 28.7 us |
| Same bytes, loads only (a probe of a rejected one-kernel variant, chained) | 17.3 us = 81% |
| `hyper_connection_mix`, fused route T = 2 / 4 / 8 / 16; composed route T = 17 / 64 | 35 / 49 / 76 / 122 us; 34 / 66 us |
| `projection_fp32` IQ4_XS head [248320, 2560], T = 1 / 2 / 8 | 392 / 396-400 / 583 us = 92% / 91% / 63% |
| `projection_fp32` BF16 router + shared gate [513, 2560], T = 1 / 8 / 9 | 10.2 / 14.3 / 26.6 us |

The 80% HC gate is not met: the fused mixer is latency-bound on sm_86, and above T = 2 it is
slower than the composed route. See the PR 6 description.

## Files ported from Infernix

Infernix (Apache-2.0) at commit `a3edb450`. Each file carries its own adaptation notice.

| Ours | Infernix |
|---|---|
| `include/ninfer/ops/hyper_connection.h` | `include/infernix/ops/hyper_connection.h` |
| `src/ops/hyper_connection/hyper_connection.cu` | `src/ops/hyper_connection/hyper_connection.cu` |
| `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` |
| `include/ninfer/ops/ple.h`, `src/ops/ple/ple.cu` | `include/infernix/ops/ple.h`, `src/ops/ple/ple.cu` |
| `include/ninfer/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | `include/infernix/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` |
| `include/ninfer/ops/rows.h`, `src/ops/rows/rows.cu` | `include/infernix/ops/rows.h`, `src/ops/rows/rows.cu` |
| `tests/ops/test_hyper_connection.cpp`, `test_projection_fp32.cpp`, `test_rows.cpp` | `tests/ops/test_hyper_connection.cpp`, `tests/ops/linear/test_projection_fp32.cpp`, `tests/ops/test_rows.cpp` |

## Artifact decisions taken in PR 1

- Logical parameters mirror the GGUF tensors one to one, named under `text/`; shapes are the
  reversed GGUF dimensions, so every object holds the GGUF bytes unchanged. The exceptions: routed
  experts are split into per-expert `moe/experts/E/{gate,up,down}` parameters, and
  `ffn_gate_inp_shexp` becomes `moe/shared_score` of shape `[1, 2560]`. The map is the table in
  [`tools/convert/qwen4_exp.py`](../../tools/convert/qwen4_exp.py); unknown, missing or misplaced
  GGUF tensors are errors.
- Each layer's routed experts are two objects: `gate|up` of shape `[512 * 1280, 2560]` with expert
  `e`'s 640 gate rows then its 640 up rows, and `down` of shape `[512 * 2560, 640]`. These are the
  `[512, 1280, 2560]` and `[512, 2560, 640]` banks with identical bytes; the 2-D form is what the
  existing packing groups and per-expert bindings produce, as for Qwen3.5 MoE.
- The n-gram embedding table (`text/ple/table`) uses `ggml_rows_page4k_v1`: 7,111,146 pages,
  29,127,254,016 bytes. It is written in 16 MiB chunks of whole pages.
- GGML-format projections record `AllowA8` (ggml's Q8_1 activation path); BF16 projections keep
  `A16Only`.
- The development subset is a builder option (`--subset dev`), not a second recipe, because a
  recipe cannot drop parameters or shorten the n-gram table. It is 2,930,421,760 bytes, not the
  ~1.2 GB first estimated: the token embedding and output head alone are 675 MB and the three
  expert banks 2.04 GB.

## Export conventions of this GGUF

Settled by evidence in PR 1 (`tests/convert/test_qwen4_exp_real.py` reproduces the data checks):

| Convention | Finding | Evidence |
|---|---|---|
| Unit-offset norms | Every F32 norm except `ssm_norm` stores `1 + gamma` of a BF16 `gamma`: `v - 1` is an exact BF16 word for 100% of values in all ten families, while only 8-67% of the stored `v` are (the BF16 checkpoint plus 1.0 in FP32). All 4,608 `ssm_norm` values are exact BF16 words, so it is the checkpoint's own multiplier. The model multiplies by the stored vector; nothing is subtracted. | data check; llama.cpp `conversion/qwen.py` (+1 on `norm.weight` except `linear_attn.norm`) and `conversion/qwen4exp.py` (+1 on PLE and indexer norms); Strata `gr.hpp`, `ple.hpp` and `gdn.hpp` apply `rms(x) * w` |
| `ssm_a` | `-exp(A_log)`; all 1,728 values are negative | data check; llama.cpp `qwen.py` |
| PLE hash constants | The literal multipliers equal splitmix64 with seed 1234, PLE index 0 and the 248,320-row vocabulary; the head sizes are the 16 consecutive primes from 20,000,000; the table rows are their sum rounded up to 128 | data check against the Infernix derivation |
| `attention/query_gate` | per head, 256 query rows then 256 gate rows | llama.cpp graph, Strata `layer.cpp`, Infernix mapping |

Still inferred, to be confirmed by the PR 7 FP64 block reference:

- **GDN value-head order is tiled** (value head `h` pairs with key head `h % 16`) in `gdn/qkv`'s
  value rows, `gdn/z`, `gdn/a_projection`, `gdn/b_projection`, `gdn/a`, `gdn/dt_bias`,
  `gdn/convolution` and the K axis of `gdn/output`. The basis is provenance: the file's names and
  conventions match llama.cpp's `Qwen4ExpTextModel`, which inherits the grouped-to-tiled reorder,
  and Strata, which runs this file, pairs heads by modulo with no runtime permutation. No HF
  checkpoint is local to compare against.
- **Decision:** the converter keeps the tiled order. `gdn/output`'s K axis cannot be permuted
  exactly in a block format, so the qwen4exp GDN path adopts the tiled order throughout rather than
  the converter undoing it for the other tensors only.

## Tokenizer

The synthesized `tokenizer.json` equals the Qwen3.8-27B artifact's in vocabulary (248,044 entries),
merges (247,587), pre-tokenizer, normalizer, decoder and the ids and contents of all 33 added
tokens. The `special` flag differs on six tokens (`<|fim_prefix|>`, `<|fim_middle|>`,
`<|fim_suffix|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`): Hugging Face has them non-special,
and llama.cpp's converter stores every `<|...|>` added token as a control token, so the GGUF cannot
say. `--resource tokenizer.json=PATH` keeps the Hugging Face flags after checking the file against
the GGUF. Whether Flash-Next's own Hugging Face tokenizer equals Qwen3.8-27B's is assumed, not
checked. The GGUF's chat template is used as is; it differs from the NInfer Qwen3.8-27B template.

## PR 1 measurements

Host: Ryzen 9 5950X, Samsung 980 PRO (C:), Python 3.12.7, numpy 2.5.3, torch 2.11 (CPU).

| Workload | Result |
|---|---|
| `--subset dev` conversion, GGUF on C: to C: | 2.93 GB in 20 s on the first run, 11 s with the GGUF ranges cached; peak working set 0.69 GiB (0.48 GiB of it is the interpreter with torch loaded) |
| Full n-gram table paging, output discarded | 28.80 GB read in 43 s (0.67 GB/s, CPU-bound in Python); peak working set 0.57 GiB |
| GGUF header parse (73 keys, 248,320-token vocabulary, 1,224 tensors) | 0.7 s |

The full-model conversion has not been run; it needs ~68.4 GB free on C: (see the spec's disk plan).
