# NInfer-3090 v0.18.0

**Long agent sessions with many screenshots no longer fail with `media_budget_exceeded`, Claude Code's deferred tools work against `/v1/messages`, and a keyless server now refuses requests from a web page on a foreign origin. A launch script that worked on v0.17.2 still works: no flag or launcher variable was removed.**

Changes: #381, #394, #438 to #444, #446 to #448, #451, #452. This release ships a Linux x86-64 archive and a Windows x64 archive. **The Windows archive was built on a machine with no NVIDIA GPU and has not been run** (see "Verification").

## Upgrade notes

- **Nothing was removed or renamed.** The `--flag` lists of `ninfer-serve` and `ninfer` at v0.17.2 and at this release were compared: every v0.17.2 flag is still accepted, and the only differences are additions. The launcher scripts keep every `NINFER_*` variable they read before.
- **Changed behaviour you can hit without touching anything:**
  - **Host and Origin checks (new, on by default for a server without `--api-key`).** The server answers requests whose `Host` is an IP address, its own name, or a name you add with `--allowed-host NAME` (launchers: `NINFER_ALLOWED_HOST=NAME`, a plain host name, no wildcard). It refuses a request with a foreign `http(s)` `Origin` and any `Sec-Fetch-Site: cross-site`. This stops a web page you visit from reaching your server through DNS rebinding. **If you reach the server by a name such as `gpubox.lan` or through a reverse proxy, add that name** or the request is refused. Curl, SDKs, Claude Code and browser extensions are unaffected. With `--api-key` set the Host check is skipped. API keys are now compared in constant time.
  - **`GET /slots` is served again, in llama.cpp's array shape** (v0.17.2 did not serve it). The waiting queue is on `GET /v1/load`.
  - **Over-budget images are shrunk instead of rejected** (below). A request that used to get `400 media_budget_exceeded` now succeeds.
  - **The n-gram keep-alive period dropped from 100 ms to 40 ms** (Flash-Next n-gram volume). CPU expert workers are no longer pinned to cores. Neither changes output.
- **Added:** `GET /dashboard` (a live page), `--config FILE.json` (the launcher profiles are now such files), `--allowed-host`, and the opt-in options listed under "Opt-in options". New `/metrics` series: `ninfer_gpu_power_limit_watts`, `ninfer_admission_overtakes_total` and `ninfer_spec_decode_relaxed_kept_tokens_total`. No existing series was renamed.
- **Request log:** schema version 32 (was 30). It adds `preparation.degraded_media_items` and `preparation.dropped_media_items`, and with `--config` the `effective_argv` and `setting_sources` of the start-up record.

## What this means for you

- **You drive a coding agent that sends screenshots every turn:** a long session used to die once the sum of all earlier images passed the vision budget. Now the oldest images shrink first and the newest stays sharp (#452).
- **You use Claude Code (or any client) with deferred tools against `/v1/messages`:** `defer_loading` and `tool_reference` blocks no longer return 400 (#448).
- **You expose the server on a LAN or behind a proxy:** add your host name with `--allowed-host` / `NINFER_ALLOWED_HOST`. Otherwise you do nothing.
- **You use MTP with prompts of tens of thousands of tokens and images:** a prefill failure `scatter: expected src [D,V], indices [V], dst [D,T]` (reported on v0.17.2 with prompts of about 59k tokens, failing again on retry) is fixed (#447).
- **You run the tests or develop:** four fixes to flaky or root-sensitive tests (#440, #446).
- **You use none of this:** speed and output are the same as v0.17.2 with every new option off. #394 compared the 27B's greedy output against master for DFlash2, MTP and no speculation (24 prompts each, all identical).

## Long screenshot sessions (#452)

When a request's images exceed the vision budget (the processor cap, less the prompt's text tokens and 1,024 tokens of headroom), the server now:

1. shrinks the **oldest** images to a 64-token ceiling (about 256x256), oldest first, and stops at 60% of the budget so later turns have room;
2. if that is still too much, replaces the oldest images with the text `[image omitted: too many images for the context]`.

The newest image stays full size unless nothing else is left. Decisions depend only on image order and size, so appending one image keeps the earlier decisions and the cached prefix stays reusable. A shrunk image costs one extra decode and resize, then it is cached. The counts appear only in the request log (`media_shrunk=`, `media_dropped=`), not in an API response.

- Videos are never shrunk this way. `count_tokens` does not know the context size, so for an over-budget request it is an upper bound.
- If the text plus the omission notes alone exceed the context, the request fails with `context_length_exceeded`.
- **Not measured on a card:** the PR's new tests are CPU tests; a long screenshot session replay (or the `vision-envelope-degrade` bench case) was not run before this release.

## Deferred tools for Claude Code (#448)

`defer_loading: true` tools are parsed but kept out of the system-prompt tool block. A `tool_reference` inside a `tool_result` writes the named tool's definition into that result and makes the tool callable from then on, in full at the first reference only. The system prompt is untouched, so discovering a tool appends to the prefix instead of invalidating the cache. Errors match Anthropic's API (`tool_reference_not_found`, `all_tools_deferred`, `deferred_tool_cache_control`, `deferred_tool_not_loaded`).

- Top-level `tool_reference` blocks and Anthropic's server-side `tool_search_tool_*` are still rejected. OpenAI tool search on `/v1/responses` is still unsupported; its rejections now say so and what to do instead.
- **Not measured:** whether Qwen reliably calls a tool whose schema sits inside a tool result. If it proves unreliable, the fallback is re-rendering discovered tools into the system block, at the price of a cache miss per discovery.

## Server safety and visibility (#394)

- **Host and Origin checks**, as above. Costs and limits: on a wildcard bind, start-up does one DNS lookup for the machine's name (a broken resolver delays start-up by its timeout; a loopback bind never does it). Without ports on either side the server cannot tell plain HTTP from a TLS proxy, so `https://127.0.0.1` and `http://127.0.0.1` count as one origin. On a keyed server a rebound page can still read `GET /health` and the static dashboard, as before.
- **`/dashboard`** shows lanes, queue, throughput and memory from `/metrics`, `/v1/load` and `/slots`.
- **`--config FILE.json`** reads options from a file; the command line wins. The start-up log says which setting came from where.
- **`--admission-affinity` (opt-in):** a waiter that resumes a prefix held on the GPU may overtake the queue; overtakes are counted.

## Opt-in options

All are **off by default**. Turn one on only if it matches your situation.

- **`--numeric-tripwire`** (`NINFER_NUMERIC_TRIPWIRE=on`, #443): logs one line naming the first NaN, Inf or overflow in decode and carries on. Measured on an RTX 3090 (Vast, 285 W cap): MTP 68.90 tok/s off and on, DFlash2 64.07 off and 64.04 on, token ids identical. It found nothing on a clean answer. Limits: no prefill probe, one prompt, one request at a time; Flash-Next and multi-GPU are rejected.
- **`--adaptive-draft-tokens 3`** (#444, with `--spec dflash2` and `--draft-tokens` above 3): switches to a 3-token draft window while recent acceptance is low. **It missed its own target:** +2.9% decode on a low-acceptance prompt at one request at a time (target was +5%), within 0.5% either way elsewhere, no effect at 2 or 4 concurrent requests. **It can change greedy output** when it switches width. Leave it off unless you serve mostly low-acceptance text at one lane.
- **`--relaxed-drafting`** (#394, 27B): accepts more drafts under sampling; DFlash2 +2.3% and MTP +3.8% median at temperature 0.7, and it **changes the output distribution**.
- **`--draft-context-rows`** (#394, 27B): refused above one lane (a second concurrent request changed the first's output).
- **`--mtp-rows-verify`** (#451, needs `--mtp-joint-acceptance`): batched MTP verification for two or more requests. **Do not enable it.** With it on, two-request decode measured 61 tok/s aggregate against 120 with it off, because one row's acceptance collapses; the cause is not found. Output is still correct.
- **Flash-Next speed options** (#394): `--mtp-constrained`, `--mtp-joint-acceptance`, `--shared-expert-stream`, `--moe-precast-input`, `--moe-gate-up-registers`, `--moe-fused-route`, `--cpu-verify-misses`, `--gdn-head-map`, `--ggml-decode-mma`, `--ngram-prefetch`, `--ngram-flag-wait`, `--partial-ram-residency`, `--prefill-cpu-share`, `--hot-embedding-rows`, `--gdn-async-commit`, `--hc-fused-read`, and numerics option `--hc-int8`. Per-option numbers and what was verified are in `docs/serving.md` and `docs/maintainer/flash-next-plan.md`. The 16 Flash-Next "everything off equals master" checks could not run on the test box (its memory lock limit blocks both builds), so that comparison is **unverified**.
- **Formats:** IQ2_XS, IQ3_XXS, Q4_K and Q5_K GGML blocks and the GSQ-RCO IQ3 packs load. A fresh Flash-Next install can start from a shipped default expert state.
- A Hadamard rotation probe found no benefit for the 27B's Q4/Q5 weights (#381); nothing ships from it.

## Also fixed

- A prefill failure with MTP, `--vision-residency overlay` and an image starting right after a chunk boundary (#447).
- `ninfer_context_store_test` no longer races its start-up refresh (#440).
- Test fixes for fake-engine lane release, a directory read as a file, and permission checks that root bypasses (#446).
- The Vast release scripts refuse a rented card whose power limit the host lowered (#441). Maintainers only.

## Under the hood

New maintainer tools: a GPU-free performance model with `nsys` calibration (#438, #439), an expert-cache replay simulator (#442), and a GPU verification toolkit (`tools/bench/basalt_gpu_verify.*`, `greedy_identity.py`, `serve_soak.py`).

## Verification

- **Linux archive:** built on a rented RTX 3090 (CUDA 12.8.1, Ubuntu 24.04) from the release commit. VERIFICATION_LINUX
- **Windows archive:** built on a machine with no NVIDIA GPU. Its executables were checked for missing DLLs but **not run**; the start-up check with only Windows on `PATH` and a chat completion on the card were not done. The zip's hash is in `SHA256SUMS-v0.18.0-windows.txt`.
- **Tests not run for this release:** the full GPU suite and all real-model tests; the release only builds the three products. The PRs above carry their own test results, and several (#448, #452) were tested on a machine that could not link the CUDA tests.

## Install

Unpack the archive and run `run.bat` (Windows) or `./run.sh` (Linux) with a profile name, as in the README. Linux requirements (glibc 2.38+, CUDA 12.8 runtime with cuBLAS, FFmpeg 6) are in `docs/release-archive-linux.md`. Checksums are in the `SHA256SUMS-*` files.
