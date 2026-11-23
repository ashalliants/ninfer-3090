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
needs NInfer's executables. It runs on one **Tesla V100 32 GB** (`sm_70`, Linux, built from source) or one
**RTX 3090** (24 GB, `sm_86`). It takes text and images; the model has no video input.

## Why this build

- **Exact.** Converted from Unsloth's UD-Q8_K_XL (their KLD reference quant) without re-quantizing anything, and
  DSpark speculation never changes the answer: greedy output equals plain greedy decode, and sampled requests draw
  the same tokens plain sampling draws under the same seed. Unsloth's DSpark notes for llama.cpp say its greedy
  output with the drafter "is not guaranteed to be bit identical to a run without it".
- **Fast on a V100.** 78.8 tok/s plain, 93-94 tok/s with DSpark over 12 chats (120.7 on maths), 1,930 tok/s prefill
  at 8K (1,868 with DSpark and images at the full 262,144-token context). The other published V100 port (Csavoldi's, onto Lapilli)
  reports 69 tok/s without a drafter and 850 tok/s prompt reading, on a different machine and prompts
  ([report](https://github.com/chris83-dev/lapilli-v100/blob/f9b7dd093a9ff018bc68f177f81a1d27d3ffec91/docs/community/inkling2-mini.md)).

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
~/build-v100/apps/ninfer-serve models/inkling2-mini-q8.ninfer --vision --spec dspark   # DSpark, up to 3 drafts a round
```

`ninfer-serve` serves the OpenAI chat, Anthropic messages and Responses routes, with streaming, tool
calls (`parallel_tool_calls` honoured), reasoning and thinking budgets, stop strings, and images with
`--vision`. The full 262,144-token context fits with DSpark and vision on.

- **DSpark** (`--spec dspark`) speculates on greedy and sampled requests. By default each round drafts 1 to 3
  tokens, picked from the request's recent acceptance (`--dspark-adaptive on`), and drops drafts the drafter gives
  under 0.5 probability (`--draft-min-p 0.5`). Greedy output equals plain greedy decode, and a sampled request
  draws the tokens plain sampling draws under the same seed. Both held on every DSpark request measured below.
- **Several requests at once:** `--max-concurrency N` (2 to 8) decodes up to N requests together. Each request's
  tokens are its one-at-a-time tokens. At N > 1 requests decode without DSpark, and `--lora`/`--graft` are refused.
- **Prefix reuse** is on by default. A prompt that starts with the same tokens and images resumes from a snapshot
  kept in pinned host memory (one per request slot: 236 MiB with DSpark, 163 MiB per slot at `--max-concurrency N`),
  and the result reports `reused_prompt_tokens`. A chat's second turn at 8K answers in 1.10 s instead of 4.26 s.
- **Opt-in uncensoring adapters:** `--lora NAME=DIR` loads the
  [OpenUncensored](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored) LoRA and `--graft NAME=FILE`
  the [satellitedown](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine) per-visit ablation.
  Only a request that names one (`"lora"` or `"graft"`) runs with it. Any other request, and any server that
  loads none, runs the base model unchanged.
- **Numerics:** the KV cache is FP16 and prompts of 64 or more tokens prefill on the FP16 tensor cores with FP32
  accumulation, by default. `--kv-dtype fp32` runs the exact FP32 program and prefill instead (slower; see
  [Accuracy](#accuracy)).

Not supported yet: the context store and host cache tiers (the server refuses those options), JSON-schema or
grammar output, and logprobs.

**On an RTX 3090**, build NInfer for the 3090 (the [release archives](https://github.com/ashalliants/ninfer-3090/releases/latest)
or the Linux build guide) and serve with a context that fits beside the desktop:

```bash
ninfer-serve models/inkling2-mini-q8.ninfer --vision --max-context 65536
```

Layers 20-23 keep their experts in BF16 in this file (7.6 GiB); they are the checkpoint's MXFP8 values, so on a 24 GB
card the engine holds them as MXFP8 (3.9 GiB), re-encoded at load and checked exact: every output is the same, bit
for bit (`--mxfp8-experts auto`, the default; `off` keeps BF16). `--fit-context MIN` starts at the largest context that
fits. On Windows, `scripts\download-model.bat inkling2-mini` downloads and verifies this file.

## Performance (one Tesla V100-PCIE-32GB, 2026-11-22)

Measured through `ninfer-serve` built from NInfer master `dad1623b`, on a V100 at its 250 W power limit (peak
draw 236 W) over a PCIe Gen3 x4 eGPU link (OCuLink, 3.3 GB/s), with the default FP16 KV and tensor-core prefill, at the server's default 8,192-token context unless a row says
otherwise. `gpu_guard.sh` reported no thermal throttle and no power capping in any run used here. Each row gives the best
clean run first, then the typical range across repeated or ABBA legs.

| | Best measured | Conditions | Typical range |
|---|---:|---|---|
| **Decode, DSpark** | **120.7 tok/s** | maths prompt, greedy, 300 tokens, `--spec dspark` | 114.3-120.7 on maths, 105.5-115.4 on code, 81.2-99.8 on prose and lists |
| **Decode, DSpark, 12 chats** | **94.4 tok/s** | 12 chat requests of 300 tokens, greedy, mean of each request's tok/s | 93.4-94.4 over four legs, 60.5-60.7% of drafts accepted; plain 78.5-78.8 |
| **Decode, DSpark, sampled** | **83.5 tok/s** | the same 12 chats at temperature 0.8, top-p 0.95, seeded | 83.4-83.5; plain sampled 74.9 |
| **Decode, plain** | **78.8 tok/s** | the 12 chats, greedy | 78.2-78.8 |
| **Batched decode, aggregate** | **198.2 tok/s** | `--max-concurrency 8`, 8 greedy chats of 256 tokens at once, prefill included | 196.8-198.2 at N = 8 (26.4 tok/s each); 170.5 at N = 4, 121.8 at N = 2 |
| **Prefill** | **1,930 tok/s** | 7,979-token prompt, TTFT 4.1 s, `--max-context 131072` (chunk 2048, picked by the engine) | 1,629 at 1,098 tokens (TTFT 0.67 s), 1,896 at 9,798 (5.2 s), 1,244 at 100,107 (80.5 s); with DSpark and vision on, 1-4% lower (1,900 at 7,979) |
| **Prefill at 262,144** | **1,868 tok/s** | 7,979-token prompt, TTFT 4.27 s, `--max-context 262144 --spec dspark --vision` with both adapters loaded, the default chunk 2048 (2026-11-22, branch `perf/inkling2-host-snapshots`) | 1,566-1,671 at 1,098 (0.66-0.70 s), 1,613 at 34,701 (21.5 s), 1,219 at 100,107 (82.1 s). Before the prefix-reuse snapshot moved to host memory the engine picked chunk 256 here: 699-703, 782-786, 739-743 and 629-632 tok/s |
| **Second turn of a chat** | **TTFT 1.10 s** | 7,977-token prompt, 6,144 positions resumed from the first turn's snapshot, same server | 4.26 s cold; 1.51 s against 21.4 s at 34,699 tokens; the same output as cold |
| **Decode at long context** | **91.2 tok/s** | DSpark, 128 tokens after a 7,979-token prompt (summarize it) | DSpark 84.9-91.2 at 8K and 88.0-89.8 at 10K; plain 71.5 at 8K and 48.1 at 100K (DSpark 47.4-48.0 there) |
| **Image prompt** | **TTFT 375 ms** | 230-token prompt (llama.cpp's 640 x 488 `test-1.jpeg`, correctly described), vision encode and prefill, `--vision --spec dspark` at 262,144 | 375-398 ms (382-425 at 131,072); then 100-105 tok/s decode |
| **Load** | **9.2 s** | engine ready, plain server | 9.2-14.2 s over 23 starts; weights read at 1.7-2.5 GiB/s |
| **Context** | **262,144 tokens** | with DSpark and vision on | |
| **Decode at 200K, low-bit KV** (opt-in, lossy) | **41.1 tok/s** | plain program decode at 204,800 context, `--kv-dtype rk4v4` (2026-11-22, branch `exp/inkling2-lowbit-kv`, A B C C B A against FP16 and `int8`) | FP16 34.8; `int8` 40.7. At 100K: 53.2 (`int8` 52.9) against 47.8. Prefill 5-8% slower past 100K |

DSpark per prompt type (greedy, 300 tokens, two prompts each, ABBA against plain in one session; the output equals
plain decode on every request):

| Prompt type | DSpark, best | DSpark, range | Drafts accepted | Plain |
|---|---:|---:|---:|---:|
| Maths (arithmetic, equations) | **120.7 tok/s** | 114.3-120.7 | 75-77% | 78.2-78.5 |
| Code (Python, C) | **115.4 tok/s** | 105.5-115.4 | 68-77% | 78.4-78.5 |
| List | **99.8 tok/s** | 92.3-99.8 | 56-65% | 78.4-78.6 |
| Prose (story, description) | **88.2 tok/s** | 81.2-88.2 | 53-57% | 78.5-78.6 |

Over the 12-chat set, DSpark is 1.19x plain greedy (93.4-94.4 against 78.5-78.8 tok/s) and 1.11x plain sampled at
temperature 0.8 (83.4-83.5 against 74.9). At 100K context on a summary, few rounds pay (10-54 drafts over 128
tokens), so DSpark mostly takes plain steps and decodes at plain speed.

**Batched decode** (`--max-concurrency N`, plain, outputs identical to one-at-a-time):

| N | Aggregate tok/s | Per request tok/s |
|---:|---:|---:|
| 1 | 77.1 | 79.1 |
| 2 | 121.8 | 63.0 |
| 4 | 170.5 | 45.6 |
| 8 | **198.2** | 26.4 |

**Adapters** make decode 4-5% slower for the requests that select one (12 chats, greedy, ABBA in one server):
none 78.8, `"lora"` 75.4, `"graft"` 75.2 tok/s. With DSpark: none 94.1-94.4, `"lora"` 90.8-91.7, `"graft"`
91.6-92.7 tok/s, and the output equals the adapter's plain output. The adapters take the refusals on the first 20
prompts of `mlabonne/harmful_behaviors` from 20/20 to 0/20, at a KLD from the base model of 0.014 (LoRA) and 0.017
(graft).

## Performance (one RTX 3090, 2026-11-23)

Measured through `ninfer-serve` under WSL on a 3090 at a 315 W power limit (PCIe Gen4 x16; power capping, no thermal
slowdown during the runs), layers 20-23's experts held as MXFP8. 12 greedy chats of 300 tokens, each figure the range
over two runs:

| | |
|---|---|
| Decode, plain | **104 tok/s** (103.3-103.9 mean over the 12 chats, at 64K and 131K context); 100.6 with the experts in BF16 |
| Decode, DSpark | 103.5 mean over the 12 chats, 115.5 on a code prompt; 73-74 against plain's 98 on a short reply after a 1.2K-token prompt |
| Prefill (65,536 context, vision on, chunk 2048) | **2,410 tok/s** at 8.4K tokens (TTFT 3.5 s); 1,784 at 1.2K (0.65 s) |
| Largest context | 137,216 plain; 118,784 with DSpark; 106,496 with vision; 89,088 with both (`--fit-context`; a busier desktop leaves less). With `--kv-dtype rk4v4` (lossy, opt-in, one request at a time): the full 262,144, plain or with DSpark and vision |
| Long prompts (every token computed) | FP16 rings at 131,072: 1,650 tok/s at 8.4K, 1,440 at 36K (TTFT 25 s), 1,090 at 112K (1 m 43 s), then decode at 58.7 tok/s. `rk4v4` at 262,144: 1,424 at 112K (78 s, decode 69.5), 1,030 at 200K (3 m 13 s, decode 56.0), 887 at 254K (4 m 46 s, decode 49.9) |
| Several requests (`--max-concurrency`, 8,192 context) | **282 tok/s** aggregate at 8, 228 at 4, 153 at 2; each request's tokens the same as alone |
| Load | 75 s from a SATA SSD (300 MiB/s), the MXFP8 re-encode included |

**Natively on Windows** (same card and limit, NInfer master `311ad169a`, 2026-11-23) the output is the same, word for
word, and the figures are close:

| | Windows | WSL |
|---|---|---|
| Decode, plain (12 chats) | 101.4-101.8 tok/s | 103.3-103.9 |
| Prefill (65,536 context, vision on) | **2,421 tok/s** at 8.4K (TTFT 3.5 s); 2,416 at 1.2K (0.48 s) | 2,410; 1,784 |
| Long prompts at 131,072 | 1,640 tok/s at 8.4K, 1,449 at 36K, 1,077 at 112K (1 m 44 s) | 1,650, 1,440, 1,090 |
| Largest context (`--fit-context`) | 128,000-136,192 plain; 113,664-116,736 DSpark; 101,376-105,472 vision; 80,896-87,040 both | 137,216; 118,784; 106,496; 89,088 |
| Several requests (8,192 context) | 277 tok/s at 8, 224 at 4, 150 at 2 | 282, 228, 153 |

The context range is what the desktop left free across the session's starts.

DSpark gains less on the 3090 than on the V100 because plain decode is already fast; leave it off unless the work is
mostly code.

## Accuracy

Checked against CPU llama.cpp built from Unsloth's Inkling-2 pull request (`unslothai/llama.cpp` #390 at
`8aa2f23f`), with the exact FP32 program (FP32 KV and FP32 prefill, now `--kv-dtype fp32`):

- **Text:** the greedy token matches at 58 of 58 checked positions across four prompts. The largest
  per-position KL divergence, 2.2e-2, comes from near-tied expert choices: with the experts forced to
  match, it is 3.4e-5.
- **Images:** greedy output equals llama.cpp's for 48 of 48 tokens on both test images when given the
  same pixels. JPEG decoding through the server (FFmpeg) differs from the reference pixels by about
  0.7 levels on average, so served answers can diverge from llama.cpp after a few tokens while describing
  the same image.
- **DSpark:** greedy speculative tokens equal plain greedy tokens on every tested prompt, and its
  acceptance matches llama.cpp's `draft-dspark` at the same revision. Sampled requests give the same tokens as
  plain sampling under the same seed (all 12 chats at temperature 0.8, every leg above).
- **The defaults** (FP16 KV, tensor-core prefill) are not bit-exact with the FP32 program. Teacher-forced over
  30,840 tokens, FP16 KV gives a KLD mean of 2.7e-3 to 3.7e-3 per position range, with the argmax agreeing at
  97.1-97.8% of positions. The tensor-core prefill's KLD mean at a 2K prompt is 2.6e-3. Reordering the FP32
  attention sums alone gives 3.1e-3. Both come from near-tied MoE routing, where a last-bit change flips an expert.
- **Batched decode:** each request's tokens and logits are bit-identical to its one-at-a-time run (tested at 2, 4
  and 8 in flight).


**Low-bit KV rings (opt-in, lossy).** `--kv-dtype int8 | rk8v4 | rk4v4` codes the 8 global KV rings in 8 or 4 bits
(the sliding rings and the default stay FP16). Teacher-forced KLD from the FP16 rings over 30,840 tokens: `int8`
3.66e-3 (p99 3.22e-2, top-1 97.62%), at the FP32-vs-FP16 noise floor of 3.46e-3; `rk8v4` 6.82e-3; `rk4v4` 9.71e-3
(p99 7.92e-2, top-1 95.85%). Greedy text diverges from FP16 within a few hundred tokens, but the maths and code
answers hold and needles at 100K and 200K are found. At 262,144 with DSpark and vision they free 3,968 (`int8`) to
5,952 MiB (`rk4v4`), so DSpark keeps its Q8 head and whole-context window: 12 chats 92.8-93.4 tok/s against 89.4.
One request at a time. See the [spec](https://github.com/ashalliants/ninfer-3090/blob/master/docs/maintainer/inkling2-model.md#low-bit-global-kv-rings-opt-in).
## License and credits

Apache 2.0, as the base model. Model by [Thinking Machines](https://thinkingmachines.ai/); GGUF
conversion, DSpark drafter files and the llama.cpp reference implementation by
[Unsloth](https://huggingface.co/unsloth); this `.ninfer` conversion and the V100 engine by the
[NInfer-3090](https://github.com/ashalliants/ninfer-3090) project.

---

**Running a 3090 or a V100?** [NInfer-3090 on GitHub](https://github.com/ashalliants/ninfer-3090) has
every model this project supports, measured figures for each card, and the launchers to serve them.
Issues and PRs are welcome, and a ⭐ helps others find it.
