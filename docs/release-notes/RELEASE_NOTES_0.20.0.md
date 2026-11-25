# NInfer-3090 v0.20.0

**A Tesla V100 now decodes Qwen3.8-27B at 136 tokens per second on code (was 111), serves Qwen3.6-35B-A3B from the V100 launcher at 254 tokens per second, reads a 4,000-token prompt of the 35B in about a second, and every card gets a long-context crash fix and a server that starts in 15 seconds instead of two and a half minutes.** If you run a 3090 and nothing else, no default moved and no flag was removed; the visible changes for you are the start-up time and the fixes below.

## What this means for you

- **You have a Tesla V100 32 GB.** Download `ninfer-v100-linux-x64-0.20.0-rtx3090.tar.gz` and run `./run-v100.sh qwen38-27b`, `qwen38-flash-next` or the new `qwen36-35b-a3b`. The 27B, Flash-Next and the 35B all got faster (tables below). One default moved on the V100: verification now runs on 8-bit activations (see "Breaking changes").
- **You run a 27B with very long conversations (about 168K tokens and up), on any card.** A kernel wrote past the end of a shared-memory table once a request's cache grew beyond about 168K tokens, which could end the server with an illegal-address error. That is fixed on both the RTX 3090 and the V100.
- **You run a server that restarts often, or loads the 35B on a 3090.** Opening a model file took 19 seconds, eight times per start. A start of Qwen3.6-35B-A3B now takes 15 seconds from "starting engine" to "engine ready", where it took 2 minutes 15 seconds.
- **You run an always-on server near its memory limit.** With the Host context cache full, the engine could spend minutes choosing what to evict while every running request waited (decode at 0 tokens per second, GPU nearly idle). It now lets running requests decode between eviction steps.
- **Your clients send a system message that is not the first message** (many do, on `/v1/messages`). A model file converted with the stock Qwen template used to answer `System message must be at the beginning.`. The server now notices that template and uses NInfer's bundled one. `--chat-template FILE` still wins.
- **You do nothing and nothing changes** on an RTX 3090 for a model already running well: no default moved, and your model files still load.

## Breaking changes

Nothing was removed or renamed. A diff of every `ninfer-serve` and `ninfer` flag between v0.19.0 and this release shows two flags added (`--verify-a8` and `--no-verify-a8`) and none removed, and no `NINFER_*` launcher variable was dropped.

- **Tesla V100 only: verification now runs on 8-bit activations by default (`--verify-a8`).** This is what makes the 27B's MTP3 decode 110 tokens per second on code (up from 82) and DFlash2 reach 136. It is lossy at the noise floor: against the full-precision kernels the decode-path KL divergence is 7.2e-4 to 7.8e-4 and 98.9-99.2% of tokens keep the same top choice. Pass `--no-verify-a8` for the old behaviour (the MTP3 output was then byte-identical to the previous release's on all six text prompts tested). The RTX 3090 does not compile the kernel and is unchanged.
- **`--mtp-draft-vocab` has a new value, `tokenizer`**, and on a V100 `auto` now means `tokenizer`: the draft list is derived from the model's own tokenizer instead of read from `<model>.draft-vocab`. The output is unchanged (drafts are always verified by the full model).
- The request log's server-options record gains a `verify_a8` field. A tool that parses it with a fixed list of fields should tolerate that.

## Qwen3.8-27B on the V100

Measured on one V100-PCIE-32GB at 200 W, no thermal throttling sampled, 2026-11-24, with `run-v100.sh` defaults:

| What | Before | Now |
|---|---:|---:|
| Code, DFlash2 K7 | 111 tok/s | **136** |
| Thinking, DFlash2 K7 | 102 | 125 |
| Structured JSON, DFlash2 K7 | 146 | 177 |
| Code, MTP3 | 82 | **110** |
| Story, Chinese, French, MTP3 | about 57 | 77-81 |
| Two requests at once, MTP3 (total) | 74 | 122 |
| MTP3 at 43K tokens of context | 53.0 | **69.5** |
| MTP3 at 172K tokens of context | 26.4 | **56.7** |

Long conversations gain most: attention over a long cache now runs on the V100's tensor cores instead of re-reading the whole cache for every token (#608). The cost of the 8-bit activations is described under "Breaking changes".

## Qwen3.6-35B-A3B on the V100

`./run-v100.sh qwen36-35b-a3b` now serves the tuned profile with resident vision and two lanes. Before, the launcher had no entry for it and fell through to a profile whose vision mode the server refuses on this card. Requesting `NINFER_VISION_RESIDENCY=overlay` is now refused before the server starts, with a message that says what to set instead.

Measured at 200 W (2026-11-25):

| What | Before this series | Now |
|---|---:|---:|
| Reading a 4.2K-token prompt (prefill) | 765 tok/s | **3,984** |
| Reading a 40K-token prompt | 1,098 | **3,118** |
| Code, MTP3 + draft head | | **253.7 tok/s** |
| Thinking, structured JSON | | 265, 303 |
| Story, Chinese, French, conversation | | 172.6-192.7 |
| Structured JSON, DFlash K7 | 293 | **319** |
| Two requests at once (total) | | 197-200 |

The "before" figures are the baselines in the PRs that changed them (#616, #621, #618); decode before this series has no single baseline, so none is claimed. Context was tested to 65,536 tokens on this model, not further. Two V100-only bugs were fixed along the way: a prompt of 2.7K-3.9K tokens aborted the server with `cudaErrorCooperativeLaunchTooLarge`, and a verification of 9-16 tokens produced NaN logits (DFlash or MTP with 8-15 drafts accepted nothing).

## Qwen3.8-Flash-Next and Inkling-2-Mini on the V100

| What | Before | Now |
|---|---:|---:|
| Flash-Next, code (MTP7 + draft vocabulary) | 123 tok/s | **129.6** |
| Flash-Next, French | 72 | 77 |
| Flash-Next, thinking | 100 | 103-106 |
| Inkling-2-Mini, 8K prompt, default chunk | 2,069 tok/s | **2,170** |
| Inkling-2-Mini, 8K prompt, `--prefill-chunk 8192` | 2,687 | 2,727 |

Flash-Next decode stays flat from 10K to 221K tokens. These are V100 figures only; no RTX 3090 figure for these models changed.

## Also fixed

- Long prompts, 27B, any card: a kernel overran a shared-memory table above about 168K keys (see above; compute-sanitizer reported 1,614 errors on master and 0 with the fix).
- A busy server with the Host cache full no longer shows decode at 0 tokens per second while evicting.
- Servers no longer take two silent minutes between "starting engine" and "loading weights" on large model files.
- `/v1/messages` with a system message that is not first no longer fails on stock-template model files; the "only text system blocks" error now names the block type it received.
- `run.bat` and `run.sh` explain the `runtime reservation requires ... available for runtime capacity` failure (an explicit `NINFER_CONTEXT` turns `--fit-context` off).
- Inkling-2-Mini's prefill on the V100 is 3-9% faster at the default chunk.

## Under the hood

For profiling on a V100 under `nsys`, `NINFER_NO_COOPERATIVE_KERNELS=1` turns off the cooperative launches `nsys` cannot trace. It slows the kernels and is for measurement only. `NINFER_SM70_VERIFY_ATTN_HMMA=0`, `NINFER_GGML_STAGED=0` and `NINFER_INKLING2_BANK_EXPERTS=0` restore the previous V100 routes, for comparison. The 35B's experts and projections, the 27B's drafter and Flash-Next's verify now run on the V100's FP16 tensor cores or on `dp4a` instead of emulated Ampere instructions. The README leads with new Tesla V100 comparison charts; per-model figures and every source are in `docs/tesla-v100.md`.

## Verification

Tests: this release builds only the three products (`ninfer`, `ninfer-serve`, `ninfer_bench`); the full test suite and the real-model tests were not run for the release itself. The changes in it were tested in their own PRs.

What was checked, on the build machine (Windows 11, and Ubuntu 24.04 under WSL), before publishing:

- **Windows zip, Linux RTX 3090 archive:** VERIFICATION_PENDING
- **Linux Tesla V100 archive: built, not run on a V100 before this release was published.** VERIFICATION_PENDING The V100 figures above come from source builds of master on a V100, not from this archive. If `./run-v100.sh qwen38-27b` from the archive does not start on your card, please open an issue.
- The Linux archives' `ldd` lists were VERIFICATION_PENDING.

## Install and upgrade

- Windows: unzip `ninfer-rtx3090-windows-x64-0.20.0-rtx3090.zip`, then `.\run.bat qwen38-27b`.
- Linux RTX 3090: `ninfer-rtx3090-linux-x64-0.20.0-rtx3090.tar.gz`, then `./run.sh qwen38-27b`.
- Linux Tesla V100: `ninfer-v100-linux-x64-0.20.0-rtx3090.tar.gz`, then `./run-v100.sh qwen38-27b`. It does not start on an RTX 3090, and the 3090 archives do not start on a V100.
- Upgrading: replace the folder. Your `models/` are unchanged; nothing needs converting.

Checksums are in the three `SHA256SUMS-v0.20.0-*.txt` files.
