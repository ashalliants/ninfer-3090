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
| 2b | Infernix-aligned artifact: `ggml_expert_record_v1` expert banks and three record formats (Python and C++), the n-gram table as an Infernix-format volume with IQ4_NL rows, Infernix's text config keys; replaces PR 1's two-object banks and `ggml_rows_page4k_v1` | done |
| 3 | Dense GGML linears: `linear`/`linear_add` for IQ4_XS, IQ3_S, Q6_K, IQ4_NL, Q8_0, Q2_0 at the model's 19 dense problems, Q8_1 activation profile | done (perf gates below) |
| 6 | Ops from Infernix: `hyper_connection` (mix, inject, expand), `ple` (IQ4_NL `ple_embed`, gate, stateful convolution, commit), `projection_fp32` (BF16 router; IQ4_XS head on PR 3's decode route), `rows` | done (HC perf gate missed, below) |
| 7a | `offloaded_sparse_moe` contract, routing, dispatch, combine, canonical A8 arithmetic, CPU expert engine (scalar, AVX2, worker team) | done |
| 7b | `offloaded_sparse_moe` GPU narrow route (device frames, staging, zero-copy), CPU miss channel and service | done (perf gate missed, see below) |
| 8-9 | Model skeleton, Program and Engine integration | not started |
| 10 | Whole model at 32K, quality gate against Strata | not started |
| 11-14 | Residency policy and miss split, 128K, MTP, vision | not started |

The C++ loader reads and places every `ggml_*` and `ggml_rec_*` object. Linear and LinearAdd execute
the six dense formats (PR 3); offloaded_sparse_moe executes the three expert record formats (PR 7).

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

## PR 7 measurements

RTX 3090, CUDA 12.8, 5950X with DDR4-2133; random records with finite scales (record bytes are
fixed per format); 10 experts each routed every column; cold L2; medians. Bit-exactness (CPU scalar
= AVX2 = team at 1-16 workers = GPU in device frames, staged and zero-copy, and CPU-served) holds on
synthetic and on the subset artifact's real IQ2_S, IQ2_XXS and IQ1_M banks.

| Workload | IQ2_S | IQ2_XXS | IQ1_M | Gate |
|---|---|---|---|---|
| Narrow route T = 1, resident, graph replay | 53.2 us, 284 GB/s (30%) | 46.1 us, 283 GB/s (30%) | 45.1 us, 261 GB/s (28%) | >= 70% of 936 GB/s: **missed** |
| Same, kernels (nsys median): gate/up + down | 35.4 + 16.9 us | 33.4 + 16.9 us | 28.8 + 16.9 us | |
| Plain read of the same bytes | 22.5 us (670 GB/s) | 19.5 us | 17.4 us | |
| Strata native grouped kernels, same records (quantize + grouped) | 61.4 us | 53.2 us | 52.2 us | |
| Canonical / native, T = 1, 2, 4, 8 | 0.90, 0.99, 0.88, 0.90 | 0.92, 1.02, 0.89, 0.76 | 0.90, 0.93, 0.81, 0.71 | K7: within 5% of native, **not triggered** |
| Staged misses T = 1 (10 misses), effective PCIe | 24.2 GB/s | 24.2 GB/s | 15.8-23.8 GB/s (one outlier run) | >= 18 GB/s: met (IQ1_M once below) |
| One cold CPU expert T = 1, 16 workers (sweep 1-16 in the PR) | 72.6 us, 20.8 GB/s | 88.9 us, 14.7 GB/s | 122.5 us, 9.6 GB/s | recorded |

The narrow route stays at the speed of Strata's native MMVQ grouped kernels on the same records:
both run two dependent kernels (gate/up, then down) over 15 MB, and neither reaches the read
roofline at T = 1. A sweep of CTA geometry and grids in shared memory moved T = 1 by at most ~15%;
reaching 70% needs a different structure (one persistent two-phase kernel), left open.

## Artifact decisions

Taken in PR 1 and revised in PR 2b, which aligned the artifact with Infernix (the model code and
expert cache are lifted from it in PRs 7-9).

- Logical parameters mirror the GGUF tensors one to one, named under `text/`; shapes are the
  reversed GGUF dimensions, so every object holds the GGUF bytes unchanged. The exceptions: a
  layer's three routed-expert tensors are one expert bank, and `ffn_gate_inp_shexp` becomes
  `moe/shared_score` of shape `[1, 2560]`. The map is the table in
  [`tools/convert/qwen4_exp.py`](../../tools/convert/qwen4_exp.py); unknown, missing or misplaced
  GGUF tensors are errors.
- **Expert banks (PR 2b).** Each layer's routed experts are one parameter
  `text/layers/L/moe/experts` of shape `[512, 2560, 640]` in `ggml_expert_record_v1`: one record
  per expert holding its gate, up and down blocks, in one of three record formats
  (`ggml_rec_iq2_s_q2_0`, `ggml_rec_iq2_xxs_q2_0`, `ggml_rec_iq1_m_q2_0`). Infernix's frame pool,
  residency table, staging and CPU jobs treat an expert as one opaque record with one
  `record_stride`; packing records on the host at load would be runtime repacking, so the converter
  writes them. Records are 1,510,400 / 1,305,600 / 1,177,600 bytes, back to back: the stride is
  rounded to 256 bytes, not Infernix's 4 KiB, because only the deferred SSD expert tier reads
  records in place with direct I/O, and 4 KiB rounding would cost 1.5% (0.5 GiB) of host RAM. This
  replaces PR 1's two objects per layer (gate|up interleaved, and down).
- **n-gram volume (PR 2b).** The PLE n-gram table is not an artifact object. It is Infernix's
  `NINFERNG` volume, version 2 with a row-format field (`ggml_iq4_nl`): 45 rows of 90 bytes per
  4 KiB block, exactly PR 1's page geometry, after a 4 KiB header; 29,127,258,112 bytes. The
  artifact's `ngram_table` records the geometry and the volume id; `--ngram-out` and
  `--ngram-reuse` follow Infernix, including its 512-row reuse check. This replaces
  `ggml_rows_page4k_v1`, which no longer exists.
- **Text config (PR 2b).** The keys are exactly Infernix `parse_config`'s (`hc_lowrank`,
  `indexer_n_heads`, `indexer_budget`, one-based `ple_layer_ids`, `ple_embed_dim` = 16 heads x 160,
  `eos_token_id` = the PLE EOS 248044, and so on). The GGUF's literal hash tables are replaced by
  Infernix's derivation parameters (seed 1234, base 20,000,000, padding 128), which the converter
  proves equal to the literals; a GGUF with other literals is refused. Facts the GGUF does not
  record are fixed by the architecture: `output_gate_type` sigmoid (llama.cpp's and Strata's GDN
  both apply a sigmoid gate), `norm_topk_prob` true, one indexer key head (checked against
  `indexer.k_proj`), and `split_ngram_parts` 1 (the GGUF holds the table as one tensor; Infernix
  parses the key and does not use it). The GGUF's `ple.image_token_id` is not carried: Infernix has
  no such key and vision is PR 14.
- GGML-format projections and the expert banks record `AllowA8` (ggml's Q8_1 activation path);
  BF16 projections keep `A16Only`.
- The development subset is a builder option (`--subset dev`), not a second recipe, because a
  recipe cannot drop parameters or shorten the n-gram table. It is 2,928,979,712 bytes plus a
  413,696-byte volume of 4,500 rows: the token embedding and output head alone are 675 MB and the
  three expert banks 2.04 GB.

## Code adapted from Infernix

Infernix is Apache-2.0; each adapted file carries the notice the spec's licensing rule fixes.

| NInfer file | Infernix source (`a3edb450`) | Adaptation |
|---|---|---|
| `tools/artifact/ngram_volume.py` | `tools/convert/qwen4_exp.py` (`ngram_geometry`, `read_ngram_volume_id`, `write_ngram_volume`) | Version 2 header with a row-format field; streams GGML rows from any source |
| `tools/convert/qwen4_exp.py` (config keys, `layer_multipliers`, `head_tables`, expert bank parameter, volume binding) | `src/models/qwen4_exp/config.cpp`, `tools/flash_next/ngram.py`, `tools/convert/qwen4_exp.py` (`ExpertBankSource`, `import_expert_bank`), `tools/convert/__main__.py` | Sourced from GGUF metadata; GGML expert records instead of NVFP4 banks |
| `include/ninfer/ops/offloaded_sparse_moe.h` | `include/infernix/ops/offloaded_sparse_moe.h` | GGML record formats, no per-expert scales; SSD tier, fetch channel, landing, streamed records, overlap/fork streams, L2 warming and wide route not carried |
| `src/ops/common/canonical_math.h` | `src/ops/common/canonical_math.h` (`0cf68068`) | IEEE helpers, BF16, exp and SiLU only |
| `src/ops/offloaded_sparse_moe/cpu/expert_team.cpp` | `src/ops/offloaded_sparse_moe/cpu/expert_team.{h,cpp}` | GGML jobs; units of 32 intermediates; no A16, AVX-VNNI or AVX-512 |
| `src/ops/offloaded_sparse_moe/cpu/miss_service.cpp` | `src/ops/offloaded_sparse_moe/cpu/miss_service.{h,cpp}` | No tiered requests; fixes a startup race (the first request could be taken as already answered) |
| `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` | `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu` (`398cf2cc`) | Route, dispatch, staging, CPU plan/wait and combine kept; narrow kernels rewritten for GGML sub-blocks |
| `include/ninfer/ops/hyper_connection.h` | `include/infernix/ops/hyper_connection.h` | see PR 6 |
| `src/ops/hyper_connection/hyper_connection.cu` | `src/ops/hyper_connection/hyper_connection.cu` | see PR 6 |
| `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | `src/ops/hyper_connection/hyper_connection_mix_fused.{h,cu}` | see PR 6 |
| `include/ninfer/ops/ple.h`, `src/ops/ple/ple.cu` | `include/infernix/ops/ple.h`, `src/ops/ple/ple.cu` | see PR 6 |
| `include/ninfer/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | `include/infernix/ops/projection_fp32.h`, `src/ops/projection_fp32/projection_fp32.cu` | see PR 6 |
| `include/ninfer/ops/rows.h`, `src/ops/rows/rows.cu` | `include/infernix/ops/rows.h`, `src/ops/rows/rows.cu` | see PR 6 |
| `tests/ops/test_hyper_connection.cpp`, `test_projection_fp32.cpp`, `test_rows.cpp` | `tests/ops/test_hyper_connection.cpp`, `tests/ops/linear/test_projection_fp32.cpp`, `tests/ops/test_rows.cpp` | see PR 6 |

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

## Measurements

Host: Ryzen 9 5950X, Samsung 980 PRO (C:), Python 3.12.7, numpy 2.5.3, torch 2.11 (CPU).

| Workload | Result |
|---|---|
| `--subset dev` conversion, GGUF on C: to C: | 2.93 GB in 20 s on the first run, 11 s with the GGUF ranges cached; peak working set 0.69 GiB (0.48 GiB of it is the interpreter with torch loaded) |
| Full n-gram table paging, output discarded | 28.80 GB read in 43 s (0.67 GB/s, CPU-bound in Python); peak working set 0.57 GiB |
| GGUF header parse (73 keys, 248,320-token vocabulary, 1,224 tensors) | 0.7 s |

PR 2b, same host, GGUF on C: to C:, peak working set from `psutil` after the run:

| Workload | Result |
|---|---|
| `--subset dev` conversion with its 4,500-row volume | 2.93 GB artifact and 0.41 MB volume in 20.0 s; peak working set 0.62 GiB |
| Whole n-gram volume through the writer, output discarded | 320,001,536 rows, 7,111,146 blocks, 29,127,258,112 bytes in 51.4 s (0.56 GB/s read); peak working set 0.60 GiB |

The full-model conversion has not been run; it needs ~68.4 GB free on C: (see the spec's disk plan).
