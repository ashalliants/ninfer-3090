# NInfer-3090 v0.19.0

**The Tesla V100 32 GB is now a first-class card with its own download, and this release adds Inkling-2-Mini (a new 26B model that runs on both the V100 and the RTX 3090) and a much faster Qwen3.8-Flash-Next on the V100.** If you run a 3090 and only the Qwen3.8-27B profile, nothing you use changes: no default moved and no flag was removed.

## What this means for you

- **You have a Tesla V100 32 GB.** Until now you built NInfer from source. Download `ninfer-v100-linux-x64-0.19.0-rtx3090.tar.gz` instead, run `./run-v100.sh qwen38-27b`, and you are serving. It runs Qwen3.8-27B, Qwen3.8-Flash-Next and Inkling-2-Mini.
- **You run an RTX 3090 and want a second model.** Inkling-2-Mini now runs on the 3090, on Windows and on Linux, with images and a 137,216-token context.
- **You drive coding agents from a V100 box.** Flash-Next decodes code at 123 tokens per second (the 27B on the same card does 76-86). See "Qwen3.8-Flash-Next on the V100".
- **You serve fine-tuned models.** One server can now hold the base Qwen3.8-27B and up to eight LoRA adapters and pick one per request.
- **You run an always-on server on a shared or rented GPU.** The 27B can now keep its cache in a smaller footprint (per-layer KV mixes, below), and the dashboard shows medians, means and peaks, so you can see whether a slow stretch was a blip.
- **You do nothing and nothing changes.** No default for Qwen3.8-27B or Qwen3.6-35B-A3B moved, and your existing model files still load.

## Breaking changes

Nothing was removed or renamed. A diff of every `ninfer-serve` / `ninfer` flag between v0.18.0 and this release shows 16 flags added and none removed, and no `NINFER_*` launcher variable was dropped. One behaviour is stricter:

- **Flash-Next-only options on a Qwen3.5, 3.6 or 3.8 dense model are now an error.** Before, `ninfer-serve` accepted options such as `--ngram-ram-mib` or `--hot-embedding-rows` for a dense model and silently ignored them. It now refuses to start and names each flag that was set. No shipped profile or launcher sets any of them, so this only affects a hand-written command line. `--vram-headroom-mib` is still valid with `--fit-context`.
- The request log and `GET /v1/models` shape are unchanged, except that the log's server-options record now has a `default_lora` field and `kv_cache` can read as a per-layer spec (for example `rk4v4x12,int8x4`) when you use one. A tool that parses the log with a fixed list of fields should tolerate both.

## The Tesla V100 archive

- **New download:** `ninfer-v100-linux-x64-0.19.0-rtx3090.tar.gz`, with `SHA256SUMS-v0.19.0-linux-v100.txt`. Linux only; the V100 is built for `sm_70` and the archive holds that code and nothing else. **It does not start on an RTX 3090**, and the 3090 archives do not start on a V100. Pick the one for your card.
- It carries `run-v100.sh`, which is the V100's measured launcher (MTP3 with INT8 KV for the 27B; the full Flash-Next flag set), the downloader, the server, the CLI and the benchmark tool.
- Requirements: glibc 2.38+, the CUDA 12.8 runtime libraries including cuBLAS, FFmpeg 6, libcurl and libzstd. CUDA 12.8 is the newest toolkit that still targets Volta. Flash-Next wants about 59 GiB of RAM.
- `run-v100.sh` listens on every network interface so your other machines can reach it. **Set `NINFER_API_KEY` before you start it**; it warns when you have not.
- **This archive was built and dependency-checked, but was not run on a V100 before this release was published.** See "Verification".

## Qwen3.8-Flash-Next on the V100

Measured on one V100-PCIE-32GB over an OCuLink PCIe Gen3 x4 link, 59 GiB host, with `run-v100.sh` defaults (2026-11-23, 200 W, warm cache):

| What you type | Decode |
|---|---:|
| Code | **123 tok/s** |
| Thinking | 100 tok/s |
| Story and Chinese | 79-81 tok/s |
| French | 72 tok/s |
| Plain decode, no speculation (250 W) | 53 tok/s |

Decode does not slow down with context: it stays flat from 10K to 221K tokens. A 3.8K-token prompt prefills in 3.8 s (about 990 tok/s at 250 W). This comes from a reduced draft vocabulary for the speculative head and seven drafts per round, plus the earlier CPU-assisted verification. Cost: French is 14% slower with seven drafts than with three; `NINFER_DRAFT_TOKENS=3` restores the old behaviour. These are V100 figures only.

Flash-Next now has a second build to choose between: Unsloth's UD-IQ3_XXS (`qwen38-flash-next-iq3`), which reads about 7% better by perplexity than the 2-bit build at roughly half the speed, text only. And on the V100 a Flash-Next server keeps its context cache on disk across restarts, so a restart no longer makes a long conversation prefill from scratch (112K tokens took about 5 minutes).

The refusal-steering vector for Flash-Next can now be applied through `--graft`, per request or always on.

## Inkling-2-Mini (new model)

Thinking Machines' Inkling-2-Mini is a 26B-total looped mixture of experts with 128 experts (5.4B active per token), with images and its own DSpark speculative drafter. The file NInfer serves is Unsloth's UD-Q8_K_XL converted without changing a single tensor byte. With DSpark on, greedy output equals plain greedy decode, and a sampled request draws the tokens plain sampling would draw under the same seed.

| Card | Context | Prefill | Decode, one stream | All lanes |
|---|---:|---:|---:|---:|
| Tesla V100 32 GB | 262,144 with DSpark and images | 1,930 tok/s at 8K | 78.8 plain; **93-94** with DSpark (12 chats); 120.7 on maths | 198 tok/s at 8 requests |
| RTX 3090 (WSL) | 137,216 plain; 89,088 with DSpark and images | 2,421 tok/s at 8.4K | **104** plain (101.8 on Windows); 115.5 best with DSpark | 282 tok/s at 8 requests |

V100: 2026-11-22, master `dad1623b`, 250 W. RTX 3090: 2026-11-23, 315 W limit. On the 3090, the full 262,144 tokens with images and DSpark needs the opt-in `--kv-dtype rk4v4` (lossy). The 3090 holds the experts of layers 20-23 in an exact 8-bit (MXFP8) re-encoding so it fits; that is the default there. Up to eight requests decode in one batch (`--max-concurrency 8`), bit-identical to one at a time; DSpark applies when one request is running.

**Where it is rough:** there is no launcher profile for it yet, so you start `ninfer-serve` on the artifact yourself (the V100 archive README shows the command); download it with `download-model.sh inkling2-mini` (22.3 GiB).

## Also new for Qwen3.8-27B and the dense models

- **LoRA adapters per request.** `ninfer-serve model.ninfer --lora pi=/path/to/peft-dir`, then `"lora": "pi"` in the request (Chat Completions, Responses and Anthropic Messages). One server holds the base model and up to eight adapters in the same decode rounds. Unknown names return `400 unknown_lora`. Validated on a V100 with the real 27B and the `harness-hub/Qwen3.8-27B-pi-adapted` adapter. PEFT directories only; GGUF LoRAs are not loaded.
- **Per-layer KV mixes.** The 27B's 16 attention layers are not equally sensitive to a lossy cache: the last four carry half of `rk4v4`'s quality penalty and the first four carry none. `--kv-dtype rk4v4x12,int8x4` halves that penalty for 22% more bytes than `rk4v4` (+1.69 millinats against +3.56 for `rk4v4`, measured on a V100 over 391K tokens). `rk4v4x4,int8x12` matches `int8` within noise in 12% fewer bytes. Opt-in and lossy; no default changed.
- **Control vectors** (GLP and llama.cpp format) on the dense models.
- **Log probabilities** on `/v1/chat/completions` (`logprobs: true`, `top_logprobs` 0 to 20, every generated token, non-streaming) and `/v1/responses`. Useful for classification and multiple-choice scoring.
- **Dashboard:** under each tile, the median, mean and max over the chart window; `run.sh` and `run.bat` print a ready-to-open dashboard link that carries your key in the part of the address the server never sees.
- **Responses API** accepts tool names the engine cannot carry directly (they are mapped and mapped back), and honours `include=[reasoning.encrypted_content]`.
- Flash-Next launchers: `NINFER_REASONING_LOOP`, `NINFER_PRESENCE_PENALTY` and `NINFER_MIN_P` now apply to it too (off unless you set them), and it starts without MTP (`NINFER_SPEC=none` or a plain artifact).

## Also fixed

- Starting the Inkling-2 server under WSL's memory-lock limit no longer fails; a startup that would spill is refused with a clear message instead of running slowly, and `--fit-context` is honoured with one lane.
- DeepSeek-style thinking prompts start in reasoning.
- `qwen4_exp` refuses `--engram-dir` for a model that has no Engram instead of ignoring it.
- The `download-model.bat` and the Windows build now cover Inkling-2.
- Steadier prefill tokens-per-second on the dashboard in the middle of a request.

## Under the hood

Many Volta (`sm_70`) routes moved off emulated Ampere instructions onto the V100's FP16 tensor cores (vision, two-lane verification, short prompts over long histories). The V100 decode attention stages the next K/V tile in registers, with the same output. A large body of DeepSeek-V4.1-Flash groundwork (converter, tokenizer, kernels and a first Engine path) merged; it is **not a supported model in this release** and there is no download for it.

## Verification

Tests: this release builds only the three products (`ninfer`, `ninfer-serve`, `ninfer_bench`); the full test suite and the real-model tests were not run for the release itself. The changes in it were each tested in their own PRs.

(Filled in after the smoke tests; see the release page.)

## Install and upgrade

- Windows: unzip `ninfer-rtx3090-windows-x64-0.19.0-rtx3090.zip`, then `.\run.bat qwen38-27b`.
- Linux RTX 3090: `ninfer-rtx3090-linux-x64-0.19.0-rtx3090.tar.gz`, then `./run.sh qwen38-27b`.
- Linux Tesla V100: `ninfer-v100-linux-x64-0.19.0-rtx3090.tar.gz`, then `./run-v100.sh qwen38-27b`.
- Upgrading: replace the folder. Your `models/` are unchanged; nothing needs converting.

Checksums are in the three `SHA256SUMS-v0.19.0-*.txt` files.
