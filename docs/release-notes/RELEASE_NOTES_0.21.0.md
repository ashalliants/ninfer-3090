# NInfer-3090 v0.21.0

**Four long agent sessions now run at the same time on one RTX 3090 (total decode 98.7 -> 114.5 tokens per second, the slowest session's wait for its first word 41 s -> 13 s), requests that offer tools on Qwen3.8-Flash-Next now decode with speculation (Tesla V100: 44 -> 70 tokens per second on an explanation, 46 -> 115 on code), and NVIDIA's NVFP4 QAD build of Qwen3.8-27B, which is closer to full precision, is a second 27B you can run.** If you run one chat at a time on a 3090 with the default launcher, your speed is unchanged and your model files still load; what moved is listed under "Breaking changes".

## What this means for you

- **You run several coding agents against one 3090 (`NINFER_CONCURRENCY=3` or more, `NINFER_SPEC=mtp`).** The launcher now sizes one shared memory pool for all the sessions instead of reserving one context per session. In the test, four sessions of 75-84K tokens all ran at once; before, only two of the four did and the others queued. Each request is capped at 131,072 tokens in this mode (see "Breaking changes").
- **You run Qwen3.8-Flash-Next with tools (coding agents, chat clients).** Any request that offered tools used to decode with no speculation at all. It now speculates. Output is byte-identical; only the speed changes. `--no-mtp-constrained` turns it off.
- **You want the most accurate 27B.** `./run.sh qwen38-27b-qad` (`run qwen38-27b-qad` on Windows) serves NVIDIA's NVFP4 QAD checkpoint. It decodes slower than the default 27B. It has been measured on a Tesla V100 only, not on an RTX 3090, so treat it as a preview there.
- **You run a router that switches between models, or run near the memory limit.** An unload now returns freed memory to the system and saves the model's conversations first, and a model whose conversation store cannot fit is decided before its weights are loaded, so a switch loads once instead of twice.
- **You run Inkling-2 and need JSON.** Inkling-2 now serves `response_format` JSON, JSON schema, grammar, choice and regex constraints. It used to refuse them.
- **You have a Tesla V100.** The NVFP4 QAD 27B runs there (details below), and the V100 server binary is about 70% smaller (448 MB, was 1,473 MB), so its archive is far under GitHub's size limit and `ninfer_bench` is back in it.
- **You do nothing and nothing changes** if you run one chat at a time on a 3090 with a model already working: no flag was removed and no launcher variable was dropped.

## Breaking changes

A diff of every `ninfer-serve` and `ninfer` flag between v0.20.0 and this release shows one flag added (`--no-mtp-constrained`) and none removed. No `NINFER_*` launcher variable was dropped. These defaults and behaviours moved:

- **Flash-Next with `--spec mtp` now speculates on constrained requests by default** (JSON schema, grammar, and any request that offers tools; #646). Before, this needed `--mtp-constrained`, which is still accepted. Use `--no-mtp-constrained` for the old behaviour. It uses about 1 MiB more memory.
- **`qwen38-27b` profiles keep only the first 8,192 rows of the token embedding in video memory** (every profile except `none`; #642). The other 644 MiB is read from pinned system RAM, so the server now pins about 644 MiB more host RAM. The output is bit-for-bit the same and no speed change was measured (DFlash2 on 8 chat prompts 129.4-129.8 vs 129.8-129.9 tok/s; a 98K-token prompt 91.5-91.7 vs 91.7-91.8). The 622 MiB freed goes to the KV cache. `NINFER_HOT_EMBEDDING_ROWS=0` keeps the whole table in video memory, and `NINFER_EMBEDDING_Q4` / `NINFER_EMBEDDING_Q6` turn it off. The new `qwen38-27b-qad` profile is not affected: its embedding is Q8.
- **`NINFER_SPEC=mtp` with `NINFER_CONCURRENCY` of 3 or more now caps each request at 131,072 tokens and uses a shared pool** (`--kv-capacity auto --device-state-slots 0`, no `--fit-context`; #642). Before, the cap was the launcher's longer default. One and two lanes, and DFlash2 at any lane count, are unchanged. `NINFER_CONTEXT` sets the cap yourself.
- **The Windows `run.bat` menu has a fifth entry** (Qwen3.8-27B, NVIDIA NVFP4 QAD), so a script that answers the prompt must now choose from 1-5.
- **The dashboard now remembers the API key in the browser's `localStorage`** (it used `sessionStorage`; #640). It survives closing the browser, is stored per host and port, and anyone with access to that browser profile can read it. The Forget button clears it.
- **An idle server no longer writes throughput lines** to the log or the request log (#626); they appeared every 5 seconds before.
- **Release builds no longer compile with `-lineinfo`, and the V100 build carries no PTX** (#633). This changes nothing about speed (the machine code is identical) but a profiler's source view needs `-DNINFER_CUDA_LINEINFO=ON`.
- **Linux archives ship `ninfer` as a link to `ninfer-serve`** (#628): started under the name `ninfer`, the server runs as the command-line tool. A copy of the folder made with a tool that does not preserve symbolic links will lose that link. Windows still ships a separate `ninfer.exe`.
- The Windows launcher's `--fit-context` default for DFlash2 is unchanged at 188,416 tokens. It was re-measured on 2026-11-26 with the desktop holding 1.5 GiB: it starts with 1.61 GiB free; 196,608 started once in two tries, so it was not raised.

## Four long agent sessions on one RTX 3090

RTX 3090, Windows 11 with the desktop holding 1.4-1.9 GiB, 315 W, MTP3, `rk4v4` KV cache, overlay vision, 75-84K-token sessions where about 95% of each follow-up is already cached, 1,700 tokens out each, two runs:

| Launcher at `NINFER_CONCURRENCY=4` | KV pool | Sessions running at once | Worst wait for first word | Whole batch | Total decode |
|---|---:|---|---:|---:|---:|
| v0.20.0 | 212,992 | 2 of 4 | 41.0 s | 68.9 s | 98.7 tok/s |
| This release | 330,240 | **4 of 4** | **13.4 / 13.8 s** | **59.4 / 59.4 s** | **114.5 / 114.4 tok/s** |

A second change speeds this exact case: with two or more sequences in one verification, the attention kernels now pick a split count that fills the 3090's 164-CTA waves. Four sessions past 79,360 keys went from 114.6 / 114.1 to **119.9 / 119.5 tok/s** (batch 59.4 -> 56.8 s, worst wait unchanged at 13.4 s; #643). On its own the kernel takes 19-20% less time at 4x80K keys, 6.6% at 131K and within noise on one sequence.

Cost and caveat: do not set an explicit `--kv-capacity` close to the card's limit. At 335,872 tokens (200 MiB free) Windows paged video memory to RAM and decode fell to 1-5 tokens per second, which is why the launcher keeps the automatic pool's margin. An image request through overlay vision was not exercised with the 8,192-row embedding on; startup and text requests were. The `none` profile (no speculation) leaves the rows off, because without them overlay vision's evictable tail is below one image-encode window.

## Single-session decode on the RTX 3090

Qwen3.8-27B, DFlash2 with the launcher's settings (`rk4v4`, n-gram copies), Windows, 315 W: a **98K-token prompt decodes at 92.9-93.0 tok/s against 90.6-91.0 before (+2.3%)** in an A/B in one session with identical output; measured alone afterwards it is 93.8-94.0 tok/s, which is the figure the README, `docs/performance.md` and the model card carry. Short-context code is unchanged (189.5-189.6 vs 189.2-189.6 tok/s). On its own the INT8-family verify attention kernel takes 11-16% less time at 98K-131K keys (#641).

## Flash-Next with tools, on the V100

Tesla V100, 200 W, greedy, thinking off, the router's Flash-Next flags (7 drafts, 32,768 context), same output byte for byte (#646):

| Request | Before | Now |
|---|---:|---:|
| 7 tools offered, explanation | 43.0-44.7 | **70.3-70.6** |
| 7 tools offered, code | 45.4-46.8 | **114.3-115.5** |
| 7 tools offered, story | 45.5-46.0 | **72.2-72.4** |
| JSON schema | 21.4-21.8 | **112.8-114.3** |

These are V100 figures; Flash-Next was not re-measured on an RTX 3090 for this change. Qwen3.5-family models (27B, 35B) already speculated under grammar masks.

## NVIDIA's NVFP4 QAD build of Qwen3.8-27B

`qwen38-27b-qad` is [NVIDIA's quantization-aware-distilled checkpoint](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD) of Qwen3.8-27B (19 GB; NVIDIA reports scores within about 0.3 points of the 16-bit model on GPQA Diamond and IFBench). The converter now reads NVIDIA ModelOpt's NVFP4 and FP8 tensors directly (#635, #636); the artifact is a download (`download-model.sh qwen38-27b-qad`), with vision, MTP3 and, from the 2026-11-27 revision, a DFlash2 drafter (`NINFER_SPEC=dflash2`).

Tesla V100, 200 W, greedy, thinking off, 400-token answers, one request at a time (#639, #647):

| Prompt | QAD, MTP3 | QAD, DFlash2 K=7 | Q4/Q5 27B, DFlash2 |
|---|---:|---:|---:|
| Code | 90.1-91.1 | **116.0-116.4** | 139.1 |
| JSON | 93.2-94.1 | **114.6-114.8** | 151.3 |
| Explanation | **66.5-67.1** | 63.6-63.7 | 84.3 |
| Story | **58.5-59.0** | 49.1-49.3 | 63.4 |
| French | **58.0-58.4** | 46.7-46.9 | 59.7 |

MTP3 is the default because DFlash2 is faster only on predictable text. Reading a 3,754-token prompt runs at 745.8 tok/s, matching the Q4/Q5 build (741.9). Perplexity is lower than the Q4/Q5 build on the three domains measured (wikipedia_en 5.520 vs 5.547, wikipedia_zh 8.961 vs 9.287, arxiv 6.401 vs 6.425); the other three domains were not run. Costs: decode is about 4% slower in plain generation and 17-24% slower than the default 27B under DFlash2 on code and JSON, and the artifact is 1.3 GiB larger. **It was measured on a Tesla V100 only.** The RTX 3090 has the model and launcher entries, but no RTX 3090 download, launch or speed run was done before release.

## Routers: unloading, memory and model switches

- **An unload returns memory to the system** and saves the unloading model's conversations first (#630). A router that had served Qwen3.6-35B-A3B kept 1.4-1.5 GiB of heap with nothing loaded; it keeps about 1 GiB less now (#633, from not retaining debug data of every loaded kernel image), and the unload trims freed heap (a CPU test: 260 MiB held, 12 MiB after the trim). Production figures for the combined effect were not available at release.
- **A model switch loads once** (#631). A switch to a larger model used to load 53 GiB of weights, fail at "finalizing engine" because the context store needed a cache that had no room, and load again 80 seconds later. The store's fit is now decided before the weights load; when RAM is short the model runs without the context store, saves stay on disk, and the log says so.

## Inkling-2 constraints

`response_format` JSON object and schema, and `structured_outputs` grammar, choice and regex now work on Inkling-2 (#629). The model's reasoning stays free text and the answer follows the grammar. Under a constraint no tool call is possible even with tools active; this differs from the Qwen models' JSON-plus-tools mode. DSpark decoding is unchanged.

## Context cache under pressure

With a full Host cache, the server prices which saved states to evict once per request instead of once per candidate (#644), and a one-off prompt's end-of-prompt save competes only with unproven entries, which age (#645). Measured on a CPU fixture: a tier full of long one-off prompts used to keep none of the later shorter ones forever and now keeps the first after 497 attempts. On the router's 35B labelling run, victim choice had cost about 18% of the engine thread with the GPU idle. No end-to-end tokens-per-second figure is claimed.

## Also fixed

- NVFP4 and FP8 models failed to plan a normal 1,024-token prefill chunk on a V100 and on sm_80-89 cards ("nvfp4 linear_swiglu A16 is registered only through T=16"); they now serve any width (#636).
- The NVFP4 QAD 27B on a Tesla V100 decoded a short prompt at 8.6 tok/s and read a 3,754-token prompt at about 25 tok/s; now 40.6 and 745.8 (#637).
- `run.bat` and `run.sh` explain a failed automatic KV pool as missing GPU memory instead of a raw error.
- The router's startup log gains a line when the context store is off for a run, with the reason.
- `--ram-headroom-mib` help now says "the system", not "Windows".

## The Tesla V100 archive

- `ninfer-v100-linux-x64-0.21.0-rtx3090.tar.gz` with `SHA256SUMS-v0.21.0-linux-v100.txt`, Linux only, `sm_70` only: it does not start on an RTX 3090, and the 3090 archives do not start on a V100.
- **`ninfer_bench` is back in it.** v0.20.0 dropped it for size; with the 448 MB server (no `-lineinfo`, no PTX) and `ninfer` as a link, the tarball is __V100_SIZE__ (v0.20.0: 1,498,182,285 bytes, v0.19.0: 2,125,147,854).
- `run-v100.sh` serves `qwen38-27b-qad` with INT8 KV, and `NINFER_SPEC=dflash2` on it.

## Under the hood

New in this release: the NVFP4 decode on the int8 `dp4a` streaming kernels and an FP16 cuBLAS prefill route for NVFP4/FP8 below sm_89 (V100), exact bit-level FP8/NVFP4 widening, a conflict-free shared-memory tile for the INT8-family verify attention (a 2-way bank conflict was ~30% of that kernel's shared-memory wavefronts), wave-aware split counts for multi-row verify, and `tools/artifact/add_component.py` (copies a component, such as a drafter, from one artifact into a copy of another). Per-model figures and sources are in `docs/performance.md` and `docs/tesla-v100.md`.

## Verification

Tests: this release builds only the three products (`ninfer`, `ninfer-serve`, `ninfer_bench`); the full test suite and the real-model tests were not run for the release itself. The changes in it were tested in their own PRs; #637 and #639 record RTX 3090 runs as not run.

__VERIFICATION__

## Install and upgrade

- Windows: unzip `ninfer-rtx3090-windows-x64-0.21.0-rtx3090.zip`, then `.\run.bat qwen38-27b`.
- Linux RTX 3090: `ninfer-rtx3090-linux-x64-0.21.0-rtx3090.tar.gz`, then `./run.sh qwen38-27b`.
- Linux Tesla V100: `ninfer-v100-linux-x64-0.21.0-rtx3090.tar.gz`, then `./run-v100.sh qwen38-27b`. It does not start on an RTX 3090, and the 3090 archives do not start on a V100.
- Upgrading: replace the folder. Your `models/` are unchanged; nothing needs converting. The QAD 27B is a new download (`download-model qwen38-27b-qad`).

Checksums are in the three `SHA256SUMS-v0.21.0-*.txt` files.
