# NInfer-3090 v0.17.1

**A server whose Host memory is full of cached history no longer freezes for 10 to 27 minutes on a long prompt, a long conversation stays warm instead of being thrown away when a short request arrives, and every `ninfer-serve` setting can now be set from the launcher with an environment variable. No default changed and nothing was removed.**

Changes: #401, #403, #404, #405. Also #400 (release tooling). This release ships a Linux x86-64 archive and a Windows x64 archive. **The Windows archive was built on a machine with no NVIDIA GPU and has not been run** (see "Windows archive: built, not run").

## Upgrade notes

- **Nothing was removed or renamed.** Every flag, launcher variable, endpoint and `/metrics` series of v0.17.0 is still there, so a launch script that worked on v0.17.0 works unchanged. **No default changed.** No flag or endpoint was added to `ninfer-serve`; the new variables are read by `run.sh` and `run.bat` only.
- The two cache fixes change which cached history is kept when memory is tight (see below). With plenty of free memory nothing differs.

## What this means for you

- **You run long agent sessions with the Host cache full:** a cold prompt of about 130K tokens could sit for 10 to 27 minutes making no progress while `/health` still answered 200. That stall is gone (#404).
- **You keep one or two very long conversations going next to short requests (a shared system prompt, small side tasks):** the long conversation's cached history used to be dropped when a short request reused the shared system prompt, so its next turn read the whole prompt again (85 to 95 seconds at 130-140K tokens on a 3090). It stays cached now (#401).
- **You drive the server through `run.sh` or `run.bat`:** any `ninfer-serve` setting that has no launcher variable of its own can be set as `NINFER_<FLAG>`, for example `NINFER_PREFILL_STALL_BUDGET_MS=50 ./run.sh qwen38-27b` (#403).
- **You use none of this:** nothing changes for you.

## No more minutes-long stall when the Host cache is full (#404, #405; issue #402)

Reported on a 3090 running a long agent session. When the Host tier was full, the engine priced every possible eviction against every cached checkpoint, and the price grew with the square of the number of checkpoints being removed. A cold 130K-token prompt on the 27B made no progress for 10 to 27 minutes; a profile put about 95% of the engine thread's time in that one function.

- **#404:** the pricing is now close to linear. The answer it gives is the same (a test compares it with a copy of the old algorithm on 3,000 random cache layouts and matches exactly). With a full Host tier of 64 sessions of 64 checkpoints each and one action releasing 16 session tails, it does 512 comparisons where the old code did about 1.0e9. Timed on a Mac, one call went from about 380 ms to 0.4 ms. The same code was copied in `qwen4_exp`; both models now share one version.
- **#405:** when the Host's free memory was split into pieces too small to hold a page or a saved state, a move from the card to the Host was still planned against the total free bytes, failed, and was planned again on the next pass. The issue counted about 3,000 failed moves in one run. Only memory that can really take the move now counts. With an unfragmented Host, nothing differs.

Not measured: a replay of the reporter's stall on a card. The speed-up above is the function's cost in a test, not a measured end-to-end request.

## A long conversation stays cached beside short requests (#401; issue #397)

The cache protects recently reused history, up to 3/4 of each resource. On the 27B, a dialogue of about 139K tokens holds about 2,200 of the 2,560 card KV pages, above that limit. A short request that reused the shared system-prompt checkpoint made that checkpoint the newest protected entry and the dialogue the oldest, so the dialogue lost its protection and its deepest Host checkpoint was released. With two state slots the limit was one slot, so two dialogues could never both stay protected.

The 3/4 limit now applies to Host bytes only, because only the Host tier loses work when an entry goes; moving a checkpoint from the card to the Host loses nothing. The ratio, the ranking and the eviction order are unchanged. The reporter measured 0 of 149 cold switches in a replay with this change and a second one together; it was not measured with this change alone. Not measured: whether pressure on the card now moves protected dialogues to the Host more often under real traffic. That is the intended trade.

## Every serve setting from the launcher (#403)

`run.sh` and `run.bat` only had variables for about a dozen settings. A setting with no variable of its own is now read from `NINFER_<FLAG>`: the flag upper-cased, dashes to underscores (`--prefill-stall-budget-ms` is `NINFER_PREFILL_STALL_BUDGET_MS`).

- Value flags are passed as `--flag value` when the variable is set. Switch flags take `on` or `off` (`NINFER_CORS=on`); any other value stops the launcher with exit code 2 and a message.
- Unset passes nothing, so every profile's arguments are unchanged. It applies to all profiles.
- Covered: sampling, auth and logging, prefill scheduling, media, response store, context store and S3 tuning, suspend and idle, weight transcodes, `--host-pause-reserve-mib` and the `--lossy-prefill-*` settings. `run.bat` also passes the Flash-Next-only flags. Left out: settings a profile sets itself, the repeatable `--graft` and `--context-store-peer`, and router mode (`--models-*`).
- No default moved. The newer settings stay opt-in because each one is a trade-off (for example `--prefill-order shortest` and the stall budget help a neighbour stream at the cost of prompt speed, and lossy prefill means the model does not see part of the prompt).
- `docs/maintainer/launcher-profiles.md` and the README describe it.

## Also changed

- **Release tooling:** `scripts/vast/vast-up.sh` now rents the Vast offer that costs least for the whole job (GPU hours plus the host's per-GB download and upload prices) instead of the lowest price per hour (#400).

## Not verified in this release

- **Windows:** built and dependency-checked only; nothing was started or served on Windows. In particular the `NINFER_<FLAG>` handling in `run.bat` (#403) has **not been run**; it was written without a Windows machine.
- **The cache fixes (#401, #404, #405) were not exercised under the conditions they fix:** no full Host tier, no 130K-token dialogue and no fragmented Host was reproduced on the card. Their tests passed (see Verification), but the 10-27 minute stall and the cold re-reads were not replayed, so the real-traffic effect is unmeasured.
- Only five test programs were run on the release commit, not the full CPU or GPU suite.
- The Linux library list was not diffed against v0.17.0's. No CMake or dependency file changed since v0.17.0, and `ldd` reports nothing unresolved.

## Verification

Checked on 2026-10-29 on a rented RTX 3090 (24 GB, Ubuntu 24.04, CUDA 12.8.1, GCC 13), against the release commit's code, before publishing:

- **Build (Linux):** a clean `git archive` of the release commit, `-DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=ON`, targets `ninfer`, `ninfer-serve`, `ninfer_bench`; `ninfer-serve --version` reports `0.17.1-rtx3090`.
- **Linux archive:** the outer and inner checksums match; the packaged `ninfer-serve --version` and `--help` run, and `./run.sh --help` runs from a fresh unpack; `scripts/check-linux-scripts.sh` passes; `ldd` reports no unresolved library.
- **Launcher smoke test:** the unpacked archive's own `./run.sh qwen38-27b` served Qwen3.8-27B with the default profile and answered a chat completion ("Hello there, how are you?").
- **Tests that exercise the changed code, on the card:** `ninfer_recovery_loss_test`, `ninfer_resource_manager_test`, `ninfer_host_context_arena_test` and `ninfer_qwen4_exp_context_cache_test` pass. `ninfer_qwen3_5_native_transactions_test` passes with the Qwen3.8-27B artifact (without one it skips).
- **Not run:** the rest of the test suite, and any serving check beyond the one chat completion (streaming, tools, the context store, lossy prefill and the new flags were checked for v0.17.0 and were not repeated).

## Windows archive: built, not run

`ninfer-rtx3090-windows-x64-0.17.1-rtx3090.zip` was compiled with CUDA 12.8.1 and Visual Studio 2022 on a Windows machine that has no NVIDIA GPU, so the programs could not be started there. It is the same source as the Linux archive. What was checked: the zip matches the SHA-256 in `SHA256SUMS-v0.17.1-windows.txt`, and every DLL the programs and libraries import is either shipped in the zip, a Windows system DLL or an NVIDIA driver DLL. Starting the programs, `run.bat` and a chat completion on an RTX 3090 have **not** been tested. If it fails to start for you, please report it; v0.16.0 remains the last Windows archive that was run on a card.

## Install and upgrade

Linux: unpack `ninfer-rtx3090-linux-x64-0.17.1-rtx3090.tar.gz` next to your `models` folder and run `./run.sh <profile>`; `docs/release-archive-linux.md` lists the glibc 2.38, CUDA 12.8, cuBLAS, FFmpeg 6, libcurl and libzstd requirements. Windows: unpack the zip and run `run.bat <profile>`. Check the SHA256SUMS file for your platform. Upgrading from v0.17.0 needs no change to a working launch script.
