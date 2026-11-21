---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
tags:
  - ninfer
  - inkling-2
  - moe
  - multimodal
  - conversational
  - speculative-decoding
  - cuda
  - tesla-v100
  - gguf-derived
---

# Inkling-2-Mini for NInfer on the Tesla V100

[![GitHub: ashalliants/ninfer-3090](https://img.shields.io/badge/GitHub-ashalliants%2Fninfer--3090-2a78d6?logo=github)](https://github.com/ashalliants/ninfer-3090)
[![Stars](https://img.shields.io/github/stars/ashalliants/ninfer-3090?style=flat&color=2a78d6)](https://github.com/ashalliants/ninfer-3090/stargazers)
[![Release](https://img.shields.io/github/v/release/ashalliants/ninfer-3090?label=release&color=2a78d6)](https://github.com/ashalliants/ninfer-3090/releases/latest)
[![License](https://img.shields.io/github/license/ashalliants/ninfer-3090?color=2a78d6)](https://github.com/ashalliants/ninfer-3090/blob/master/LICENSE)

> **Maintained at [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090)**: the engine, converter and builds this artifact is made and tested with. Releases, issues and updates live there, and a star helps others find it.

This model card is the version-controlled source for
[WarlaxZ/Inkling-2-Mini-NInfer-V100](https://huggingface.co/WarlaxZ/Inkling-2-Mini-NInfer-V100).

The repository holds [Inkling-2-Mini](https://thinkingmachines.ai/model-card/inkling-2-mini/), Thinking
Machines' looped mixture-of-experts model, for NInfer
([ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090)). It is Unsloth's
[UD-Q8_K_XL GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) rewritten into NInfer's `.ninfer`
container without re-quantizing anything, plus the model's own DSpark speculative drafter and its vision
tower, in one file. It is not a Transformers checkpoint, Safetensors distribution or GGUF file, and it
needs NInfer's executables. It runs on one **Tesla V100 32 GB** (`sm_70`, Linux, built from source). It
takes text and images; the model has no video input.

## Get NInfer

This artifact runs only on **[NInfer-3090](https://github.com/ashalliants/ninfer-3090)**, a
from-scratch C++/CUDA inference engine built for and tuned on exactly two cards: the **RTX 3090** and
the **Tesla V100 32 GB**. On those cards it serves Qwen3.8-27B at up to 140 tok/s on a 3090, the
512-expert Qwen3.8-Flash-Next with 262K context on a V100, and now Inkling-2-Mini. It provides drop-in
OpenAI and Anthropic APIs, so Claude Code, Codex, Cursor, Cline and other tools can point at your own GPU.

- **[Source, build guides and docs](https://github.com/ashalliants/ninfer-3090)**. The
  [V100 guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/tesla-v100.md) covers this model's build.
- **[Releases](https://github.com/ashalliants/ninfer-3090/releases/latest)**: prebuilt RTX 3090 archives for Windows and Linux.
- **[Issues](https://github.com/ashalliants/ninfer-3090/issues)**: bug reports, model requests and questions.
- **[Star the repo](https://github.com/ashalliants/ninfer-3090/stargazers)** if this is useful. It's
  how other 3090 and V100 owners find it.

## The model

- 26.2B parameters in total; 3.4B active per layer pass, 5.4B per token (the middle 12 layers run three times); hidden size 2304; vocabulary of 200,058 tokens.
- 24 decoder layers, of which layers 6-17 run three times per token (48 layer visits), each visit with
  its own KV and convolution state, steered by a rank-64 loop controller, with depth-attention residuals
  over 6-layer blocks.
- No RoPE: a learned relative position bias (extent 1024) with per-head attention sinks, L2-clipped
  q, k and r, and 4-tap short convolutions.
- 128 routed experts, top-8 with sigmoid routing and softcapped SwiGLU. Layers 0-19 keep their experts
  in NVFP4 and layers 20-23 in BF16, as the Q8_K_XL GGUF stores them.
- A logit softcap of 30.

## What the conversion does

Every value is stored exactly as the sources hold it; nothing is rounded again.

- **Text weights** (from `Inkling-2-Mini-UD-Q8_K_XL.gguf`): GGML blocks keep their bytes, BF16 and FP32
  keep their words. Each layer's routed experts become one bank of per-expert records
  (`ggml_rec_nvfp4_nvfp4` for the NVFP4 layers), so an expert is one contiguous byte range; unpacking
  the records gives back the GGUF tensors exactly.
- **DSpark drafter** (component `dspark`, from `dspark-Inkling-2-Mini-Q8_0.gguf`): the model's own 3-layer
  block drafter, trained with it (block size 8). Its projections keep their Q8_0 blocks; the Markov head
  stays BF16. It borrows the target's embeddings and LM head.
- **Vision tower** (component `vision`, from `mmproj-BF16.gguf`): the `inkling2v` encoder, stored as BF16.
- **Tokenizer and chat template** come from the GGUF and are checked token by token against it.

An independent checker compared every stored object against the three GGUFs: all 1,190 tensor objects
are byte-exact.

## Files

| File | Bytes | SHA-256 | Read |
|---|---:|---|---|
| `inkling2-mini-q8.ninfer` (the model, with DSpark and vision) | 23,914,098,432 | `2167c0c0ff2ebb5d9dcf7ea5c8289ce13239bdc5104aac053af79a3b8cf009c1` | once, at load |
| `inkling2-mini-q8.ninfer.conversion.json` (conversion report) | 357,006 | `d9a417ed09afa9d69541255d46d758e192454cb0a4144fc31dc462b474777732` | not needed to run |

Artifact id `36220e7183a840f7b32ccda590f1fcb9`, components `text`, `dspark` and `vision`, 1,194 objects.
The conversion report's local paths are reduced to file names.

Sources (Unsloth's `unsloth/Inkling-2-Mini-GGUF`, files as published on 2026-11-20):

| Source | Bytes | SHA-256 |
|---|---:|---|
| `Inkling-2-Mini-UD-Q8_K_XL.gguf` | 22,867,451,616 | `dfb04bf37bee457f98800989dedf93ec442ede8636dbea9433992f1e527e9a41` |
| `dspark-Inkling-2-Mini-Q8_0.gguf` | 402,278,560 | `2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773` |
| `mmproj-BF16.gguf` | 641,805,760 | `2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7` |

## Running it

Get [NInfer-3090](https://github.com/ashalliants/ninfer-3090) and build it for the V100 from source on Linux ([V100 guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/tesla-v100.md),
`scripts/v100_build.sh`), then:

```bash
git clone https://github.com/ashalliants/ninfer-3090 && cd ninfer-3090
BUILD_DIR=$HOME/build-v100 scripts/v100_build.sh ninfer ninfer-serve
scripts/download-model.sh inkling2-mini                         # this repository, pinned and verified
~/build-v100/apps/ninfer-serve models/inkling2-mini-q8.ninfer --vision --spec dspark   # DSpark, 3 drafts by default
```

`ninfer-serve` serves the OpenAI chat, Anthropic messages and Responses routes, with streaming, tool
calls (`parallel_tool_calls` honoured), reasoning and thinking budgets, stop strings, and images with
`--vision`. DSpark runs for greedy requests and is switched off automatically on text where its drafts
are not accepted; sampled requests decode plainly. Greedy output with DSpark is identical to plain decode.

Prefix reuse is on by default: one device snapshot is kept, a prompt that starts with the same tokens
and images resumes from it, and the result reports `reused_prompt_tokens`.

Not supported yet: more than one request at a time (requests queue), the context store and host cache
tiers (the server refuses those options), JSON-schema or grammar output, and logprobs.

## Performance (one Tesla V100-PCIE-32GB, 2026-11-21)

Measured on a V100 with a 250 W limit over a PCIe Gen3 x4 eGPU link.

| | |
|---|---:|
| **Load** | 9.1 s |
| **Decode, plain** | 62-73 tok/s (the spread is GPU clock and thermal state between sessions) |
| **Decode, DSpark (3 drafts)** | 83 tok/s on average over five chat prompts, 1.28x plain |
| **Prefill** | ~377 tok/s at a 1K-token prompt |
| **Image prompt** (234 tokens) | 0.09 s of vision, 0.58 s of prefill |

DSpark by prompt (256 tokens each, plain 65 tok/s): code 121, arithmetic 106, C explanation 77,
story 70, list 67 tok/s. Drafts per round against speed: 1 draft 73.3, 2 drafts 80.6, **3 drafts 83.3**,
4 drafts 81.9 tok/s.

## Accuracy

Checked against CPU llama.cpp built from Unsloth's Inkling-2 pull request (`unslothai/llama.cpp` #390 at
`8aa2f23f`):

- **Text:** the greedy token matches at 58 of 58 checked positions across four prompts. The largest
  per-position KL divergence, 2.2e-2, comes from near-tied expert choices: with the experts forced to
  match, it is 3.4e-5.
- **Images:** greedy output equals llama.cpp's for 48 of 48 tokens on both test images when given the
  same pixels. JPEG decoding through the server (FFmpeg) differs from the reference pixels by about
  0.7 levels on average, so served answers can diverge from llama.cpp after a few tokens while describing
  the same image.
- **DSpark:** greedy speculative tokens equal plain greedy tokens on every tested prompt, and its
  acceptance matches llama.cpp's `draft-dspark` at the same revision.

## License and credits

Apache 2.0, as the base model. Model by [Thinking Machines](https://thinkingmachines.ai/); GGUF
conversion, DSpark drafter files and the llama.cpp reference implementation by
[Unsloth](https://huggingface.co/unsloth); this `.ninfer` conversion and the V100 engine by the
[NInfer-3090](https://github.com/ashalliants/ninfer-3090) project.

---

**Running a 3090 or a V100?** [NInfer-3090 on GitHub](https://github.com/ashalliants/ninfer-3090) has
every model this project supports, measured figures for each card, and the launchers to serve them.
Issues and PRs are welcome, and a ⭐ helps others find it.
