---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model:
  - nvidia/Qwen3.8-27B-NVFP4-QAD
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - qad
  - multimodal
  - conversational
  - speculative-decoding
  - cuda
  - tesla-v100
---

# Qwen3.8-27B NVFP4 QAD for NInfer on the Tesla V100

[![GitHub: ashalliants/ninfer-3090](https://img.shields.io/badge/GitHub-ashalliants%2Fninfer--3090-2a78d6?logo=github)](https://github.com/ashalliants/ninfer-3090)
[![Stars](https://img.shields.io/github/stars/ashalliants/ninfer-3090?style=flat&color=2a78d6)](https://github.com/ashalliants/ninfer-3090/stargazers)
[![Release](https://img.shields.io/github/v/release/ashalliants/ninfer-3090?label=release&color=2a78d6)](https://github.com/ashalliants/ninfer-3090/releases/latest)
[![License](https://img.shields.io/github/license/ashalliants/ninfer-3090?color=2a78d6)](https://github.com/ashalliants/ninfer-3090/blob/master/LICENSE)

> **Maintained at [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090)**: the engine, converter and builds this artifact is made and tested with. Releases, issues and updates live there, and a star helps others find it.

This model card is the version-controlled source for
[WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100).

The repository holds [NVIDIA's Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD)
converted to NInfer's native `.ninfer` container by
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), with the vision tower, the MTP
speculative head and (since revision 2026-11-27) the DFlash2 drafter in the same file. It is not a Transformers checkpoint, Safetensors distribution or GGUF
file, and it needs this project's executables (a build from master after 2026-11-26).

## Which Qwen3.8-27B should I use?

Two builds of the same model, so you can pick:

| | [`qwen38-27b`](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090) (the default) | `qwen38-27b-qad` (this one) |
|---|---|---|
| Weights | This project's Q4/Q5 conversion | NVIDIA's quantization-aware-distilled NVFP4/FP8 |
| Accuracy | Close to full precision | Closer: NVIDIA trained it against the BF16 model, and it scores lower perplexity than the default on every domain measured |
| Decode on a V100 (MTP3, served) | 110 tok/s code, 77 story | 90 tok/s code, 59 story |
| Speculation | MTP and DFlash2 | MTP (default) and DFlash2 |
| File | 17.68 GiB | 18.99 GiB |

Choose this one when answer quality matters more than speed. Choose the default for the fastest decode.

## The model

NVIDIA produced this checkpoint with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
by quantization-aware distillation: the BF16 Qwen3.8-27B supplies token-level targets, and the
quantized weights are trained so their NVFP4 values match it. NVIDIA's own evaluation (BF16 against
NVFP4 QAD, on vLLM/SGLang, not on NInfer):

| Benchmark | BF16 | NVFP4 QAD |
|---|---:|---:|
| GPQA Diamond | 88.19 | 87.94 |
| IFBench | 79.25 | 79.04 |
| MMMU Pro | 75.78 | 75.61 |
| SciCode | 48.34 | 48.82 |
| Terminal-Bench 2.1 | 77.25 | 76.69 |
| AA-LCR | 78.63 | 79.13 |

## What the conversion does

Recipe `qwen3_8_27b_nvfp4_qad` ([weight conversion](https://github.com/ashalliants/ninfer-3090/blob/master/docs/weight-conversion.md)):

- NVFP4 weights (the MLP, `lm_head` and most GDN projections): codes and scales copied word for word.
- FP8 weights (attention, and the rest of GDN): codes copied word for word, with the per-matrix scale
  rounded to BF16 per row (a gain within 2^-8).
- The token embedding: Q8, generated from the checkpoint's BF16.
- Vision: from the checkpoint. MTP: NVIDIA's export keeps 11 of the head's 15 tensors, so the head
  comes from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)'s last shard.
- DFlash2 (added 2026-11-27): the drafter of
  [WarlaxZ/Qwen3.8-27B-NInfer-3090](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NInfer-3090), its 66
  tensors copied byte for byte with `tools/artifact/add_component.py`. It was trained against the BF16
  model, not this checkpoint, so it accepts fewer drafts here than on the Q4/Q5 build (see below).
  Everything else in the file is byte-identical to the first revision, and MTP output is unchanged.

Source: `nvidia/Qwen3.8-27B-NVFP4-QAD` revision `cd46ab575a5a687f0bc3e54251016fc2cc1a6aef`.

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `qwen3_8_27b_nvfp4_qad.ninfer` (the model, with vision, MTP and DFlash2) | 20,386,365,952 | `7c62491a31ce950f8ad2053861f03843dd1c75064028ab2e072607b6a8dadcb1` |

The first revision (commit `2cd93ef5`, without DFlash2) was 18,159,548,164 bytes, SHA-256
`c79ff2d1f9b3c0f729327d073bfd0b9dbf085c5c300a53d51f260a0fd634de0d`.

## Running it

A Tesla V100 has no FP4 or FP8 hardware. NInfer widens these weights exactly in software and runs them
on the same streaming and tensor-core routes as its own Q4/Q5 formats. Build it for the V100 from source
on Linux ([V100 guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/tesla-v100.md)), then:

```bash
git clone https://github.com/ashalliants/ninfer-3090 && cd ninfer-3090
scripts/v100_build.sh ninfer-serve
scripts/download-model.sh qwen38-27b-qad          # this repository, pinned and verified
NINFER_API_KEY=<a secret> scripts/run-v100.sh qwen38-27b-qad
```

That serves MTP3 with the draft head, INT8 KV and two lanes sharing the full 262,144-token context,
with images (vision in overlay residency), on port 8080. `NINFER_SPEC=dflash2` serves DFlash2 K=7 with
n-gram copies instead, same lanes and context: faster on code and JSON, slower on prose. It is an OpenAI-compatible endpoint, and the
Anthropic Messages API is served too.

**On an RTX 3090:** `download-model.bat qwen38-27b-qad` / `run.bat qwen38-27b-qad` (or the `.sh`
launchers) fetch and serve it, but it has not been measured on the 3090 yet.

## Performance (one Tesla V100-PCIE-32GB, 2026-11-26, 200 W)

Served (`ninfer-serve`, two lanes, INT8 KV, overlay vision, greedy, thinking off, 400-token
answers, one request at a time; 2026-11-27, two runs per column agreeing within 1 tok/s):

| Prompt | This model, MTP3 | This model, DFlash2 K=7 + n-gram | Default `qwen38-27b`, DFlash2 (one run) |
|---|---:|---:|---:|
| Code | 90.1-91.1 tok/s | **116.0-116.4** | 139.1 |
| JSON | 93.2-94.1 | **114.6-114.8** | 151.3 |
| Explanation | **66.5-67.1** | 63.6-63.7 | 84.3 |
| Story | **58.5-59.0** | 49.1-49.3 | 63.4 |
| French essay | **58.0-58.4** | 46.7-46.9 | 59.7 |

DFlash2's drafter was trained against the BF16 model and drafts this checkpoint less well than the
default build, so it wins on predictable text and loses on prose. MTP3 stays the default.

One request through `ninfer`, no speculation, BF16 KV:

| | This model | Default `qwen38-27b` |
|---|---:|---:|
| Prefill, 3,754-token prompt | 745.8 tok/s | 741.9 tok/s |
| Prefill, 22-token chat prompt | 203.6 tok/s | 206.0 tok/s |
| Decode | 40.6 tok/s | 42.2 tok/s |

Method and conditions: [V100 guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/tesla-v100.md#qwen38-27b-nvidias-nvfp4-qad-checkpoint).

## Accuracy

Perplexity on NInfer's held-out corpus (quick set, 4,096 context, stride 2,048, BF16 KV, Tesla V100),
lower is better:

| Domain | This model | Default `qwen38-27b` |
|---|---:|---:|
| wikipedia_en | **5.520** | 5.547 |
| wikipedia_zh | **8.961** | 9.287 |
| arxiv | **6.401** | 6.425 |

The corpus's other three domains were not run.

## License and credits

The model is governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
NVIDIA's terms for `nvidia/Qwen3.8-27B-NVFP4-QAD`. The base model, Qwen3.8-27B by Alibaba Cloud, is
licensed under the Apache License 2.0 (`LICENSE` in this repository). The files here are a conversion
of NVIDIA's checkpoint into the `.ninfer` format; the MTP head comes from Qwen/Qwen3.8-27B, and the
DFlash2 companion weights (Apache License 2.0) from the Q4/Q5 `qwen38-27b` build.

Model by the [Qwen team](https://huggingface.co/Qwen); quantization-aware distillation by
[NVIDIA](https://huggingface.co/nvidia); this `.ninfer` conversion and the V100 engine by the
[NInfer-3090](https://github.com/ashalliants/ninfer-3090) project.

---

**Running a 3090 or a V100?** [NInfer-3090 on GitHub](https://github.com/ashalliants/ninfer-3090) has
every model this project supports, measured figures for each card, and the launchers to serve them.
Issues and PRs are welcome, and a ⭐ helps others find it.
