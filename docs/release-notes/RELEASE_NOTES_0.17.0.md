# NInfer-3090 v0.17.0

**A busy server stops stalling itself: the context store no longer rewrites the same conversations in a loop (that was cutting single-stream decode from about 85 to 21-27 tokens/s on a production box), stays inside its disk cap, and can be told to save everything before a box goes. Three new opt-in settings let short prompts get ahead of long ones, keep room for paused requests, and, experimentally, read very long prompts about three times faster by skipping the parts that matter least. No default changed.**

Server changes: #384 to #387, #389, #391 to #393, #395, #396. Also #378 (re-measured figures), #380 and #388 (release tooling), #390 (README). This release ships a Linux x86-64 archive and a Windows x64 archive. **The Windows archive was built on a machine with no NVIDIA GPU and has not been run** (see "Windows archive: built, not run").

## Upgrade notes

- **Nothing was removed or renamed.** Every flag, launcher variable, endpoint and `/metrics` series of v0.16.2 is still there, so a launch script that worked on v0.16.2 works unchanged. **No default changed**: every new setting below is off until you turn it on.
- **Added flags:** `--prefill-order rotate|shortest`, `--host-pause-reserve-mib N`, `--lossy-prefill-min-tokens N` with its tuning knobs (`--lossy-prefill-*`), and `--request-log-lossy-detail`. **Added endpoint:** `POST /v1/context-store/flush`. `ninfer-serve --help` and `docs/serving.md` describe each.
- **If you use `--context-store`, it behaves differently in these ways:**
  - A conversation the store removed on its own (for space or age) is written again only once it changes (its next turn). Before, it was written straight back and pushed out another one. With a store too small for all your conversations, one that was let go is lost when it leaves memory, unless it has changed since.
  - The store now keeps at least 1 GiB of the disk free and refuses a write it has no room for (`context store volume is full: ...`), evicting the least recently used sessions first.
  - With a bucket, files waiting to upload now count against `--context-store-max-gib`. When the directory is full, the oldest waiting upload is given up and that conversation is uploaded at its next write. Up to 64 uploads can wait (was 16).
  - Progress anchors (`--progress-anchors-shared`) are no longer written to the store.
- **If you use `--admission-fit-check`:** it now also counts the part of each running request's prompt that is not read yet, so two long prompts arriving together are queued one behind the other instead of both starting. Expect `ninfer_admission_fit_deferrals_total` to rise and pauses (`ninfer_preemptions_total`) to fall.
- **`ninfer_host_context_geometry_failures_total` counts much less.** It used to count every size tried while placing pages; it now counts one per placement that failed. A dashboard alert on its rate will go quiet. That is expected.
- The archive now contains `THIRD_PARTY_NOTICES.txt`, the license notice for the code the summary strategy ports from hermes-vcc-engine (Apache-2.0).

## What this means for you

- **You run a context store with little disk or a small `--context-store-max-gib`:** the rewrite loop is gone, and so is the decode slowdown it caused. On the production box where it was found (two engines, a 4 GiB store each, 8.5 hours of live traffic), 85-89% of all store writes rewrote a session that had not changed, and single-stream decode ran at 21-27 tokens/s in the windows where only those writes happened, against 81-88 without them. Replaying that box's write log under the new rules gives 87-90% fewer writes. Nothing to configure.
- **You run several boxes behind one bucket and replace boxes:** call `POST /v1/context-store/flush` when you start draining a box. It saves every conversation and uploads what the bucket does not have yet, while the server keeps serving, then answers `{"written": N, "unsaved": 0, "uploaded": true, "seconds": ...}`. With that in place you can set a long `--context-store-s3-upload-delay-seconds` and send far less upload traffic.
- **You run coding agents that send long prompts and short ones at the same time:** try `--prefill-order shortest`. A short request that already has a lane gets past a long prompt still being read, and two long prompts run one after the other instead of both finishing late.
- **Your Host memory fills with cached history and `--yield-long-prefill-ms` stopped doing anything:** set `--host-pause-reserve-mib` to keep room for paused requests (size below).
- **Your agents send cold prompts of 30K tokens and more, and you can accept that the model does not see all of it:** `--lossy-prefill-min-tokens 30000` is the experimental path. Read its section first.
- **You use none of the above:** nothing changes for you.

## Context store fixes (#389, #391 to #393, #396)

All found on one production box running v0.16.2 (two engines on a 36 GB disk, a 4 GiB store each, an R2 bucket and a peer link).

- **Rewrite loop (#389).** Any session the store removed (which happens on every turn, as a new turn replaces the old one) made the engine forget which conversations were already stored, so it wrote them all again, which evicted more, which started it again. One 110K-token session was written 763 times in two hours, and each engine handed the store 8-11 TiB in 8.5 hours. Now a conversation is written once per state. New counters: `ninfer_context_store_idle_writes_total` and `_idle_export_seconds_total`.
- **Store over its cap and a full disk (#391).** One engine's store held 10.0 GB against a 4 GiB cap and the disk reached 91%; writes then failed. The cap now counts files waiting for upload, the store keeps 1 GiB of the disk free, and "cannot write" errors now say why (`No space left on device`).
- **Good sessions deleted as "corrupt" (#391).** Reading a session while a write evicted its files counted it as damaged and deleted it. A session being read is now protected until the read ends, and a missing file is fetched from the bucket again. The peer link also stopped trying (and failing) to link sessions the other engine only knew from the bucket: 5,184 failures on that box.
- **Bucket lookups added about 3.4 s to request start and almost never found anything (#392).** A request now reads only its own conversation's entries from the bucket, and connections to the bucket are reused instead of paying a new TLS handshake per call. A download that cannot finish before the request's wait ends stops there instead of holding up the reads queued behind it. The latency gain on a real bucket has not been measured.
- **Copying sessions in and out is faster (#393, #396).** Each save and each read used a fresh buffer of the session's size; most of that time was page faults. Copying 1.6 GB took about 690 ms into a fresh buffer against about 90 ms into one already in use (Linux RTX 3090 box). The engine now reuses the last buffer, at most 8 GiB, and frees it after 30 s without use. The effect on decode and request start on live traffic has not been measured.
- **Why things fail is now visible:** `ninfer_context_store_hydration_failures_by_reason_total{reason}`, `_corrupt_by_reason_total{reason}`, `_remote_download_failures_by_reason_total{reason}`, `_lookup_hits_total`, `_remote_uploads_dropped_total`, plus one warning line per failed hydration.

## Flush before a box goes (#395)

`POST /v1/context-store/flush`, optional body `{"budget_seconds": N}` (1-3600, default 600). It writes every conversation the store does not hold as it is now, one a second between steps so requests carry on, then uploads everything waiting for the bucket without its delay. It answers when nothing is left or the budget is spent. Stop routing new turns to the box first, and call it again if `unsaved` is not 0 or `uploaded` is false. Without `--context-store` it answers 409 `context_store_off`. The reason it exists: on the production box, uploading every turn 30 s after it happened sent about 150 GB per engine in 8.5 hours, about $1.50 of bandwidth against $2.80 of rent.

## Long prompts and short ones (#385 to #387, from issue #332)

- **`--prefill-order shortest`** (default `rotate`, unchanged). Reading a prompt (prefill) already uses the whole card, so two prompts read in turns finish no sooner than one after the other; taking turns only makes the first one late. In the #332 runs (two 44K-token prompts, the second sent 10 s after the first) taking turns gave 40.1 s and 43.4 s; running them one after the other gave 27.6 s and 42.7 s. With `shortest` each request gets a deadline (when it arrived plus its predicted reading time) and the earliest deadline goes first, so short prompts overtake and a long prompt is never passed over for good. It orders only requests that already have a lane: at `--max-concurrency 1` a short request still waits in the queue. The `shortest` timings themselves come from CPU tests with a simulated model; they were not measured on a card.
- **`--host-pause-reserve-mib N`** (default 0, off). Keeps N MiB of the Host budget that cached history can never fill, so pausing a request saves its progress without evicting anything, and `--yield-long-prefill-ms` keeps working when the cache is full (on #332 it fired 0 times with 11.7 of 12.8 GB of the Host holding history). Size it for one paused request: its share of the KV pool (the startup `capacity` line) plus one saved state. It takes that room away from cached history, which can lower cache hits. `ninfer_host_pause_reserve_used_bytes` shows what it holds. Qwen3.5-family models only; Flash-Next refuses it.
- **Fixed:** a yield could lose its snapshot when the Host had room for the saved state but not for the KV, and then re-read the whole prompt from the start. A yield that does not fit is no longer started.

## Faster long prompts by skipping some of them (experimental, #384)

Off by default, and **lossy: when it runs, the model does not see part of the prompt.** With `--lossy-prefill-min-tokens N`, a cold prompt (no cached conversation) whose part after the system messages and tools is longer than N tokens is first scanned by a quick pass over the model's first layers. That pass scores every piece, and only the best-scoring 30% (plus the start, the most recent 4,096 tokens, and every role and tool-call marker) is read. The system messages and tools are always read in full. Later turns of the conversation reuse the thinned result like any cached conversation.

Measured 2026-10-25 on one RTX 3090 with Qwen3.8-27B (`--kv-dtype rk4v4 --max-context 131072 --prefill-cublas --prefill-chunk 4096`), one cold prompt of 116,865 tokens:

| | reading the prompt | KV kept | needle at 32K/64K/100K | cold `Edit` turns from agent logs |
|---|---:|---:|---|---|
| full prefill | 70.2 s | all | 22/22 each | 25 of 26 |
| lossy, defaults | **21.8 s** | about 30% | 22/22 each | 26 of 26 |

Caveats: the quality checks are small. One `Edit` turn is 3.8 points, the needle test scores 22/22 everywhere so it does not tell settings apart, and a larger run is advised before you depend on it. When on, it reserves about 2 KB of VRAM per token of `--max-context`. It works on text-only Qwen3.5-family dense models on one GPU; the server refuses to start with it on MoE or pipeline-split models. The alternative `--lossy-prefill-strategy summary` (replace the middle of a long conversation with a deterministic summary) lost one `Edit` turn on the same set. The int4 scout (`--lossy-prefill-scout-a4`) failed its quality gate and stays opt-in. Watch it with `ninfer_lossy_prefill_total{outcome}` and the request log's `lossy_prefill` object. The design and full sweep are in `docs/maintainer/lossy-long-prefill.md`.

## Also changed

- **Fixed:** `--admission-fit-check` let two long prompts arriving together both start, then paused one (#385).
- **Fixed:** `ninfer_host_context_geometry_failures_total` climbed by tens to hundreds per request while every placement succeeded (#385).
- **Docs:** the fit calculator and headline figures were re-measured on a stock RTX 3090, and Bonsai 2 27B and Qwen3.6-27B were added to the calculator (#378). The README opens with a plain-language summary (#390). Release notes moved to `docs/release-notes/` (#388); archives still carry them at their root.

## Not verified in this release

VERIFICATION-PENDING

## Verification

VERIFICATION-PENDING

## Windows archive: built, not run

`ninfer-rtx3090-windows-x64-0.17.0-rtx3090.zip` was compiled with CUDA 12.8.1 and Visual Studio 2022 on a Windows machine that has no NVIDIA GPU, so the programs could not be started there. It is the same source as the Linux archive. What was checked: the zip matches the SHA-256 in `SHA256SUMS-v0.17.0-windows.txt`, and every DLL the programs and libraries import is either shipped in the zip, a Windows system DLL or an NVIDIA driver DLL. Starting the programs, `run.bat` and a chat completion on an RTX 3090 have **not** been tested. If it fails to start for you, please report it; v0.16.0 remains the last Windows archive that was run on a card.

## Install and upgrade

Linux: unpack `ninfer-rtx3090-linux-x64-0.17.0-rtx3090.tar.gz` next to your `models` folder and run `./run.sh <profile>`; `docs/release-archive-linux.md` lists the glibc 2.38, CUDA 12.8, cuBLAS, FFmpeg 6, libcurl and libzstd requirements. Windows: unpack the zip and run `run.bat <profile>`. Check the SHA256SUMS file for your platform. Upgrading from v0.16.2 needs no change to a working launch script; read "Upgrade notes" if you use `--context-store` or `--admission-fit-check`.
