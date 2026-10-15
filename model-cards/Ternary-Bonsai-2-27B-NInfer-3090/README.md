---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: prism-ml/Ternary-Bonsai-2-27B-gguf
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - ternary
  - bonsai
  - multimodal
  - conversational
  - cuda
  - rtx-3090
---

# Ternary Bonsai 2 27B for NInfer on the RTX 3090

This model card is the version-controlled source for
[WarlaxZ/Ternary-Bonsai-2-27B-NInfer-3090](https://huggingface.co/WarlaxZ/Ternary-Bonsai-2-27B-NInfer-3090).

Created using Bonsai by Prism ML.

The repository contains Prism ML's
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), a ternary
(-1, 0, +1 with an FP16 scale per 128 weights, 2.125 bits per weight) build of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted to the native `.ninfer` artifact
format by the RTX 3090 fork of NInfer,
[ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090). It is not a Transformers
checkpoint, Safetensors distribution or GGUF file, and it needs this fork's executables: older
builds refuse it at startup because they do not know the `t2_g128_fp16` weight format.

## What is in it

- **Text tower**: Prism ML's PQ2_0 GGUF, imported without rounding. Every text projection except the
  GDN A/B controls, the output head and the token table is stored as `t2_g128_fp16`: the ternary
  codes and per-128 scales are Prism's own. The weights are Hadamard-rotated in the checkpoint, so
  the runtime rotates the matching activations (fused into the ops that produce them). The recipe
  undoes llama.cpp's exporter conventions (GDN value-head order, `1 + w` norms, `ssm_a`).
- **Vision**: the Qwen3.8-27B vision tower and frontend resources, which share the geometry.
- **MTP head** and **DFlash2 adapter**: the heads ProCreations trained against Bonsai 2 itself
  ([MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP),
  [DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)). The adapter's
  feature, output and MLP projections are stored as Q4: against a 2.125-bit target the drafter is a
  large share of every draft step's bytes. On four prompts this decoded 4.5% faster than a Q8
  adapter, with acceptance length 3.97 against 4.00.
- **Proposal head**: the 131,072 most frequent tokens' rows of the ternary output head, in their own
  encoding. It is lossless and loaded only with `--lm-head-draft`. On this card it measured no
  difference in decode speed, so it is optional.

## Artifact

| Field | Value |
|---|---|
| Filename | `bonsai2_27b.ninfer` |
| Size | 9,520,051,456 bytes (8.87 GiB) |
| SHA-256 | `975126774d7e1f9ffa36b654c0a6fc0db42c773ba4dd5f1196a9ddc6a46d4634` |
| Container version | 3 |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `bonsai2-27b` |
| Components | text, vision, MTP, DFlash2, proposal head |
| Stored objects | 1,193 (1,187 tensors and 6 resources) |

```bash
printf '%s  %s\n' \
  '975126774d7e1f9ffa36b654c0a6fc0db42c773ba4dd5f1196a9ddc6a46d4634' \
  'bonsai2_27b.ninfer' | sha256sum --check
```

The repository also holds `bonsai2_27b.ninfer.conversion.json`, the converter's report (every
object, its sources, format and method).

## Measurements

One RTX 3090 (24 GB), CUDA 12.8, Windows, greedy decoding, `--kv-dtype int8` unless stated. These
are single-card, few-prompt measurements: treat them as indicative.

**Quality.** `ninfer-perplexity --quick --kv-dtype int8` on `perplexity-1m`: **5.627**. The
reference implementation this port follows (pxzleo/ninfer-all) reports 5.631 for the same artifact
and protocol.

**Prefill**, `ninfer_bench`, 1,024-token chunks:

| Prompt | 1k | 4k | 16k |
|---|---:|---:|---:|
| tok/s | 2,853 | 2,776 | 2,576 |

**Decode**, one short coding prompt, 300 tokens, thinking off:

| Mode | tok/s |
|---|---:|
| Plain | 88 |
| MTP, 3 drafts | 199 |
| DFlash2, 7 drafts | 297 (4.89 tokens per round) |

Four varied prompts (code, networking, arithmetic, an email) with DFlash2 and 7 drafts averaged
249.9 tok/s, from 173 to 325 depending on how predictable the text was. For comparison, this
fork's 6-bit Qwen3.8-27B on the coding prompt decodes 48 tok/s plain and 195 with DFlash2.

**Memory.** Weights are 6.70 GiB on the device without drafter heads and 7.28 GiB with MTP, against
14.6 GiB for the fork's 6-bit Qwen3.8-27B.

**Tests.** Against this artifact the fork's real-engine tests pass for prefix reuse, grammar,
scoring, vision overlay, DFlash2, n-gram drafting and agent continuation.

## Run it

Build [ashalliants/ninfer-3090](https://github.com/ashalliants/ninfer-3090), then:

```text
scripts\download-model.bat bonsai2-27b
```

or on Linux `scripts/download-model.sh bonsai2-27b`. Direct CLI use:

```bash
./build/apps/ninfer models/bonsai2_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7
```

## Provenance

| Field | Value |
|---|---|
| Ternary text tower | `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `b072e1d3b35a0a630cece372c2127528e0994386`, `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| Vision and resources | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| MTP head | `ProCreations/Ternary-Bonsai-2-27B-MTP` @ `efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51` |
| DFlash2 adapter | `ProCreations/Ternary-Bonsai-2-27B-DFlash2` @ `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b` |
| Recipe | `bonsai2_27b_ternary` in `tools/convert/ternary.py` |

[`artifact-manifest.json`](artifact-manifest.json) has the object inventory, the converter command
and every source hash. The ternary import and the Hadamard-rotated runtime follow the work of
[pxzleo/ninfer-all](https://github.com/pxzleo/ninfer-all) and its upstream
[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all).

## Limits

- Prefill is dominated by the ternary integer GEMMs (81% of a 4k prompt), which sustain roughly half
  of this card's measured int8 tensor-core peak, so a faster kernel could still help. Activations
  are quantised to int8; an int4 emulation raised perplexity by about 2.9% and is not used.
- `--prefill-cublas` is slower than the default on this artifact.
- NInfer executes on one CUDA device, or splits layers across up to eight GPUs on Linux. It does
  not do CPU offload, and it does not execute generated tool calls.

## License

Apache License 2.0, matching [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (copyright Prism
ML, Inc.) and the ProCreations MTP and DFlash2 heads.
