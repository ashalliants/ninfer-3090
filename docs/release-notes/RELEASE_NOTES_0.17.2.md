# NInfer-3090 v0.17.2

**A request that makes the model pick a token it cannot say no longer takes the whole server down, a follow-up to a `/v1/responses` call now finds the model that answered it when you run several models, and clients that ask for `thinking.display: "omitted"` work instead of getting a 400. No flag, endpoint or default was removed or changed.**

Changes: #408, #409, #411, #412, #416, #418, #419, #420, #422, #423, #425, #435. Also #413, #414, #415, #417, #421, #424 (tests and docs). This release ships a Linux x86-64 archive and a Windows x64 archive. **The Windows archive was built on a machine with no NVIDIA GPU and has not been run** (see "Windows archive: built, not run").

## Upgrade notes

- **Nothing was removed or renamed.** Every `ninfer-serve` flag, launcher variable and endpoint of v0.17.1 is still there (the flag lists of the two versions were compared and are identical), so a launch script that worked on v0.17.1 works unchanged. **No default changed.**
- **Added:** `/v1/messages` and its token-count endpoint now accept `thinking.display: "omitted"` (they answered 400 `thinking_display_not_supported` before). Three new `/metrics` series for the context store: `ninfer_context_store_logical_bytes`, `ninfer_context_store_shared_bytes` and `ninfer_context_store_remote_only_dropped_total`. No existing series was renamed.
- **One value changed meaning:** with `--context-store-peer`, `ninfer_context_store_used_bytes` (what `--context-store-max-gib` is enforced against) is now this engine's share of the physical size, so a chunk shared by hard link counts once between the engines instead of in full for each. If you size a box from that gauge, see "Context store" below.
- `run.bat` now prints nothing on stderr while reading variables (it used to print `FINDSTR: Cannot open >nul` 84 times per start).

## What this means for you

- **You run agent traffic on MTP or DFlash speculation:** a draft token outside the model's vocabulary used to raise inside the engine, fail every request in flight and stop the process (one reporter saw 7 engine exits in 27 hours). Now only the request that hit it ends, with HTTP 500 `numerical_failure`, and the rest carry on (#412).
- **You serve several models behind the router (`--models-*`):** `GET`/`DELETE /v1/responses/{id}`, `.../input_items`, `.../cancel` and a new response with `previous_response_id` went to whichever model was used last, so they returned not-found whenever another model had served in between. They now go to the model that owns the response, loading it again if it was unloaded (#425).
- **Your client sends `thinking.display: "omitted"`** (some clients and gateways do by default): it works. Replayed empty thinking blocks are restored from a bounded in-memory store, so the prompt still matches the cache and tool loops keep their cache hits (#408).
- **You use the context store with a bucket or with `--context-store-peer`:** the directory now stays inside `--context-store-max-gib`, startup restore stops at its time limit instead of hanging on a slow bucket, and retired sessions cannot be republished (#411, #416, #418, #420, #422).
- **You run on a full Host cache or use pause/resume:** demotions and pauses now choose victims by whether the freed memory forms a run big enough to use, not by total bytes (#409, #419, #423).
- **You launch from Windows with `run.bat`:** the stderr noise is gone and the double-quote guard on environment variables works (#435).
- **You use none of this:** nothing changes for you.

## A bad token ends one request, not the server (#412; issue #410)

A sampled or licensed token outside the public vocabulary threw from the Qwen 3.5 collect paths. The worker's catch-all then failed every request and latched the engine (exit 2). Reported upstream (Neroued/ninfer#426), most likely with MTP or DFlash drafts.

- The token is replaced with 0, the row is marked, and that one request is released through the same path a grammar dead end uses: HTTP 500, code `numerical_failure`, message "the model selected a token outside its vocabulary after N generated tokens". Other rows and the engine carry on.
- Forced tokens and replay still throw, since they are not sampled.
- **Not measured:** there is no fault-injection hook, so this was not reproduced on a card. A request that hits the new 500 under `--spec none` and `--spec mtp:3` next to a healthy one has not been observed.

## Several models behind the router (#425; issue #336)

Each routed model keeps its own Responses store and a response id carries no routing information. The router now keeps a bounded id-to-model map (capacity `--response-store-max-records` times the number of models). It reads the new id from the first 64 KiB of a successful `POST /v1/responses` reply, including streamed ones, before the bytes reach the client, so an immediate follow-up finds the owner.

- A follow-up that names no model goes to the owner. A model the client names is honoured even if wrong (that model answers not-found). A successful `DELETE` forgets the id.
- An id that fell out of the map routes as before and the model answers not-found. The map lives in the router process, so it starts empty after a router restart.
- Tested with fake per-model Responses stores on a Mac; the real HTTP routes of a loaded model were not exercised in the PR.

## `thinking.display: "omitted"` (#408)

Responses carry thinking blocks with empty `thinking` plus the signature; streams send the `signature_delta` but no `thinking_delta`; reasoning tokens still count in `output_tokens`. The hidden reasoning is kept in a bounded, process-local LRU store (sized by the response-store limits) keyed by the signature, and put back into the prompt when the client replays the empty block.

- After a restart or eviction the reasoning is dropped like a closed turn: output stays correct, only cache reuse is lost from that turn.
- `thinking.type=disabled` with a `display` is still a 400. Not included: persisting the store across restarts, encrypted `redacted_thinking`, `between_tools`.
- **Not measured on a card:** that turn 2 of a tool loop with `omitted` reads the same cached tokens as with `summarized`.

## Host memory: placement and pause (#409, #419, #423; issues #304, #406, #398)

The Host arena never compacts, so total free bytes is not the same as room for a page or a state image.

- **#419:** demotion victims are chosen until a simulation of the free runs can place the destination, not until a byte total is reached. On a fragmented Host the old choice could free enough bytes and still leave every run too short, and the demotion failed again.
- **#409:** when a state demotion has no free run for its image, only whole Host state images count as room (all images are the same size, so each leaves a fitting run).
- **#423:** a pause snapshot's fit is decided by replaying the allocations on a model of the arena (checked against the real arena on 4,000 random layouts), so mixed KV strides are no longer refused by a bound that rounded every run down to the larger stride. Victims may now come from inside the pause reserve.
- With an unfragmented Host, or a model without the query (qwen4_exp), nothing differs. **None of this was run on a card**; the PRs were tested on a Mac with CPU tests only.

## Context store (#411, #416, #418, #420, #422)

- **#411:** with `--context-store-peer`, a chunk is charged as `ceil(size / st_nlink)`, so engines sharing it pay once between them. Measured on a live 2x3090 box before the change: 42.7 GB per engine at a 40 GiB cap, 50.6 GB physical. After a peer evicts its link this engine's share rises until its next put, maintain or metrics scrape.
- **#416:** bucket manifests that no local chunk backs may take at most a quarter of `--context-store-max-gib` (oldest dropped first, registered again when there is room). Before, they were counted but never evicted, so a large bucket grew the directory past the cap. New metric `ninfer_context_store_remote_only_dropped_total`.
- **#418:** every bucket request now ends at `--context-store-restore-seconds`, not only the start of new ones, including a slow trickle that never trips the stall timeout.
- **#420:** retiring a session writes a tombstone per version, so an engine still holding it cannot put the old manifest back. Cost: one extra HEAD per registered manifest and per upload. Limits are listed in the PR and tracked in #428.
- **#422:** which peer supplied a chunk is tracked per chunk, and an image imported from a peer is marked on disk so it stays the peer's to upload after a restart.
- All covered by tests with in-memory and loopback buckets; **no real S3 bucket and no card were used**.

## Also fixed

- **`run.bat` printed `FINDSTR: Cannot open >nul` 84 times per start** and its quote guard never fired (#435; thanks to Pawls for the report and the fix). Written without a Windows machine; verified by the reporter in cmd.
- Credit for the state-image demotion fix (#409) goes to kyhoavuong (#413).
- Tests only: route-level tests for the router relay and the suspend/residency API (#421, #424), fault injection for the shared-prefix scan (#415), a seam for the packager's import check (#417), a follow-up to #409 (#414).

## Not verified in this release

- **Windows:** built and dependency-checked only; nothing was started or served on Windows. The `run.bat` fix (#435) has not been run by us.
- **None of the fixes above was exercised on a card under the conditions it addresses** (no out-of-vocabulary token, no fragmented Host, no router with several loaded models, no real bucket). Their PR tests passed, mostly on a Mac.
- Only a few test programs were run on the release commit, not the full CPU or GPU suite.
- The Linux library list was not diffed against v0.17.1's; no CMake or dependency file changed.

## Verification

To be filled in after the checks.

## Windows archive: built, not run

`ninfer-rtx3090-windows-x64-0.17.2-rtx3090.zip` was compiled with CUDA 12.8.1 and Visual Studio 2022 on a Windows machine that has no NVIDIA GPU, so the programs could not be started there. It is the same source as the Linux archive. What was checked: the zip matches the SHA-256 in `SHA256SUMS-v0.17.2-windows.txt`, and every DLL the programs and libraries import is either shipped in the zip, a Windows system DLL or an NVIDIA driver DLL. Starting the programs, `run.bat` and a chat completion on an RTX 3090 have **not** been tested. If it fails to start for you, please report it; v0.16.0 remains the last Windows archive that was run on a card.

## Install and upgrade

Linux: unpack `ninfer-rtx3090-linux-x64-0.17.2-rtx3090.tar.gz` next to your `models` folder and run `./run.sh <profile>`; `docs/release-archive-linux.md` lists the glibc 2.38, CUDA 12.8, cuBLAS, FFmpeg 6, libcurl and libzstd requirements. Windows: unpack the zip and run `run.bat <profile>`. Check the SHA256SUMS file for your platform. Upgrading from v0.17.1 needs no change to a working launch script.
