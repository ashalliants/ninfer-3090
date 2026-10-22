# NInfer-3090 v0.16.1

**A long prompt no longer has to freeze the streams next to it: five new opt-in controls cut the stall from about 3 seconds to about 0.4 seconds or let a short request jump the queue, constrained JSON output stops coming back corrupted, and one server can now hold several models. Nothing changes unless you turn something on, with one exception: JSON output layout (below).**

About 35 merged pull requests since v0.16.0 (#291 to #359). This release ships a **Linux x86-64 archive only**; there is no Windows build for it. Windows users stay on v0.16.0 (or build from source); nothing in this release removes a flag, a launcher variable or a metric.

## Behaviour that changed

- **Constrained JSON output (`response_format` `json_object` / `json_schema`) is laid out the way the model writes it, not compacted.** v0.16.0 forced compact text, which pushed the model off its normal tokens and produced replies that parsed as JSON but were wrong: empty keys, punctuation as values, scalars wrapped in arrays. Measured on the 27B: before, `{"city":", ","country":"France",...}`; after, correct, indented JSON. The same values come back, with spaces and newlines between tokens (runs of whitespace are capped at 32 characters). If you diff or hash the raw text, expect a different layout. `json_object` with `"schema":{"type":"object"}` (what llama.cpp-style clients send) is now accepted (#335, fixes #331).

No other defaults moved. There are no removed flags, launcher variables or `/metrics` series since v0.16.0. API parsing only got more permissive (#317, #335).

## What this means for you

- **You run agent fleets next to long prompts** and a 100K-token paste stalls every other stream for seconds: pick `--prefill-layer-slices 8` (smallest gap) or `--prefill-stall-budget-ms 500`, and add `--yield-long-prefill-ms` if short requests queue behind the long one. Numbers below.
- **Your clients cancel and retry long prompts** (timeouts, dropped connections): `--progress-anchor-tokens N --progress-anchors-shared` keeps the work done so far, so the retry restarts near where the first attempt stopped (57.8 s down to 24.5 s on a 108K-token prompt).
- **You share the card with a game or another program**: `--model-suspend` hands the GPU memory back while idle and takes it again on the next request.
- **You want several models behind one address**: `--models-dir DIR` or `--models-preset FILE` (router mode).
- **You scrape `/metrics`**: GPU utilization, memory, temperature, power and cumulative energy are now exported.
- **You do nothing**: your server behaves as before, apart from the JSON layout above.

## Long prompts next to other streams (all opt-in, default off)

A prompt that is not in any cache is read in chunks. While one chunk runs, a neighbouring stream gets no tokens, and a short request behind it waits for the whole prompt. Measured on the release commit's code, on one RTX 3090, Qwen3.8-27B (`--prefill-cublas --kv-dtype int8 --max-concurrency 2 --prefill-chunk 4096`), a stream at 48 tokens/s next to one 108K-token uncached prompt, a fresh server for each row:

| setting | neighbour's longest wait for a token | the 108K prompt takes |
|---|---|---|
| none (as before) | 3.01 s | 57.7 s |
| `--prefill-stall-budget-ms 500` | 0.52 s | 73.6 s |
| `--prefill-layer-slices 8` | **0.39 s** | **61.2 s** |
| `--prefill-stall-budget-ms 500 --prefill-decode-share 50` | 0.52 s, and the stream runs at its normal pace in between | 146.9 s |

- **`--prefill-layer-slices N`** (#352) runs each chunk as N groups of layers and lets the neighbour decode between groups: a 7.7x shorter stall for 6% more prompt time. Text prompts only; not with media, DFlash or a multi-GPU split.
- **`--prefill-stall-budget-ms N`** (#341) shrinks the chunk while a neighbour decodes. Simple, but it slows the prompt by 28%.
- **`--prefill-decode-share P`** (#342) gives the neighbour P percent of the prefill's time as extra decode rounds. It makes the prompt take 2.5x as long at 50%; use it only when the stream must not pause at all.
- **`--yield-long-prefill-ms N`** (#343) pauses a long prompt when a short request is waiting for a lane. A short request sent 10 s into the 108K prompt got its first token after **1.05 s instead of 47.9 s**; the long prompt finished about 1 s later (58.7 s against 57.7 s).
- **`--progress-anchor-tokens N`** (#338) with **`--progress-anchors-shared`** (#340): a cancelled prompt leaves its work behind. Cancel at 35 s and resubmit: the retry took **24.5 s instead of 57.8 s**, with 73,728 tokens reused. Without `--progress-anchors-shared` the anchors are private and a cancel drops them.
- **`--admission-fit-check`** (#337) queues a prompt that cannot fit beside the requests already running, instead of starting it and having it paused. Two 50K-token prompts into a 70K-token pool: one preemption fewer, the first prompt finished in 20.5 s instead of 27.3 s, the pair in the same 40.8 s.
- **`ninfer_prefill_backlog_seconds`** and the `prefill_backlog_seconds` field of `/v1/load` (#344) report how much prefill work is queued ahead of a new request (24.9 s with one 50K prompt running and another queued).

Output is unchanged: greedy text of a 56K-token prompt beside a stream and a short request was identical with and without each option and with all of them together. `tools/bench/run_prefill_stall.py` (#345) reproduces these measurements on your own card; `docs/maintainer/long-prefill-handoff.md` has the method and the time model (#353).

## One server, several models; giving the GPU back

- **Router mode** (#316): `ninfer-serve --models-dir DIR` or `--models-preset FILE.ini` serves every model in the folder on one address. Models load on first request; at `--models-max` the least recently used idle model sleeps or unloads. Measured with two copies of the 27B: first request 6.0 s, later swaps about 4.0 s. Multi-GPU router mode was not exercised.
- **Model suspend** (#315): `--model-suspend` returns the model's GPU memory while idle (15.4 GiB on the 27B in about 0.15 s) and takes it back on the next request (3.3 s from the artifact, 1.9 s from a host copy). `--sleep-idle-seconds N` does it automatically.
- **GPU metrics** (#348, #349): `ninfer_gpu_utilization_percent`, `ninfer_gpu_memory_*`, `ninfer_gpu_temperature_celsius`, `ninfer_gpu_power_watts` and `ninfer_gpu_energy_joules_total` (NVML, loaded at run time; a missing driver never stops startup).

## Cached conversations

- **Peer stores** (#329): `--context-store-peer DIR` lets two engines on one box share the conversations they finish, by hard link; one directory has one writer, enforced by a lock. Hard links need both directories on the same volume.
- **Compression** (#328): `--context-store-compress` writes zstd chunks. On a real 3-conversation workload the store was 7.9% smaller and reading it back was 25 to 30% slower. Off by default.
- **`prompt_cache_key` on Chat Completions** (#317) keeps a resumable point for tool-calling loops, as Responses and Anthropic requests already did; keys up to 256 bytes are accepted (was 64 on Chat).

## Models

- **Ternary Bonsai 2 27B** (#323): `scripts/download-model.sh bonsai2-27b`. A 2.125-bit ternary build of Qwen3.8-27B: 6.7 GiB of weights instead of 14.6, 88 tokens/s plain and 297 tokens/s with DFlash2 on the 3090 (this fork's 6-bit 27B: 48 and 195 on the same prompt), prefill about 2.8K tokens/s.
- **Qwen3.8-Flash-Next two lanes** (#291, Windows preview): `--max-concurrency 2` is available but opt-in; it missed its throughput target (1.07x against 1.15x) and shipped for the 6% shorter makespan on bursts.

## Also fixed

- A paused request keeps its snapshot when the Host arena is fragmented, instead of replaying its whole prompt (#334). This was written without a GPU and its test is a CPU test; a before/after on a fragmented arena has not been run.
- A shared-prefix export failing on one candidate no longer drops the others (#308); overlay vision on a multi-GPU split works (#314).
- Linux builds: `ninfer_qwen4_exp_memory_plan_test`, the chunked-reservation compile and the model-registry test link all build on GCC 13 now (#339, #354, #357, #359).

## Not verified in this release

- Windows: no Windows archive was built or run for this release.
- Pipeline stages and router mode on two physical GPUs, and model suspend with a cold page cache.
- A before/after for #334 (fragmented Host arena) and for the context-store changes on Flash-Next.
- The full GPU test suite on the release commit. On a 3090 these passed: the CPU suite (all tests), `ninfer_qwen3_5_native_transactions_test` (including the new two-lane sliced-prefill test and the admission-fit test on real KV pools), `ninfer_qwen3_5_ngram_real_test`, `_context_store_ngram_real_test` and `_grammar_real_test`. The slow op tests and the other real-model tests were not run.
- One known timing flake: `ninfer_qwen4_exp_ngram_volume_test` (a 20 ms check) can fail under load.

## Verification

PLACEHOLDER-VERIFICATION

## Install and upgrade

Linux: unpack `ninfer-rtx3090-linux-x64-0.16.1-rtx3090.tar.gz` next to your `models` folder and run `./run.sh <profile>`; `docs/release-archive-linux.md` lists the glibc 2.38, CUDA 12.8, cuBLAS, FFmpeg 6, libcurl and libzstd requirements. Check the SHA256SUMS file. Upgrading from v0.16.0 needs no changes to a working launch script.
