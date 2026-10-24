# NInfer-3090 v0.16.2

**If you share conversations between machines through a context store bucket (S3 or R2), they now actually resume on the other machine, and the bucket traffic that was costing more than the GPU rental drops sharply. If you do not use `--context-store-s3-endpoint`, nothing about how this release serves changes.**

Four pull requests change the server (#366, #373, #374, #363) plus build, CI and documentation work (#361 to #365, #371, #376, #377). This release ships a Linux x86-64 archive and a Windows x64 archive. **The Windows archive was built on a machine with no NVIDIA GPU or driver and has never been run** (see "Windows archive: built, not run"); v0.16.0 remains the last Windows archive that was run on a card.

## Upgrade notes

- **No flag, launcher variable or `/metrics` series was removed or renamed since v0.16.1.** A launch script that worked on v0.16.1 works unchanged.
- **One flag was added:** `--context-store-s3-upload-delay-seconds N` (default 30). It needs `--context-store-s3-endpoint`, like the other bucket flags; giving it without one is rejected at start-up.
- **Sessions stored in a bucket by v0.16.0 or v0.16.1 are not reused after upgrading.** Stored sessions are now matched on the artifact's own id instead of its file's download time, so the key they are filed under changes once. The first request of each conversation after the upgrade prefills in full; later turns resume as before. Old objects are never read again. They expire under the bucket lifecycle rule, so if you have not set the 7-day expiration rule on the prefix (`docs/serving.md`), add it, or the old objects stay.
- **Sessions written to a bucket are uploaded 30 seconds late, by default.** A newer turn of the same conversation replaces one still waiting, so a burst of turns is sent once. Shutdown uploads immediately. Set `--context-store-s3-upload-delay-seconds 0` for the old behaviour.
- **Start-up no longer downloads sessions from the bucket.** It restores only what is already on the disk; a bucket session is downloaded when a request asks for it. A fresh box therefore starts faster and cold, and the first request of a conversation that moved here pays for the download.
- **Prompt grafts:** sessions that start from a configured graft still match by the graft file's size and modification time, so they pass between machines only when the graft file is the same copy.

## What this means for you

- **You run several boxes behind one bucket** (rented GPUs, drained and replaced boxes): a conversation that moves to another box now resumes from its stored session instead of prefilling from scratch. Before this release it never did, because two machines that downloaded the same model file gave it different timestamps and so different keys. Reported on a fleet of RTX 3090 boxes: 0% cached tokens on prompts of 78K to 100K tokens that had moved boxes.
- **You pay for bucket traffic:** see the numbers below. Nothing to configure; the defaults changed.
- **You use `json_schema` with `minLength` / `maxLength`:** string limits are handled by a smaller, shared grammar (#366).
- **You build from source with CUDA 13 or MSVC:** three source changes let the tree compile there (#363). CUDA 12.8 behaviour is unchanged.
- **You use one box and no bucket, or a local-disk context store:** nothing changes for you.

## Bucket traffic (#374)

Measured on one vast.ai RTX 3090 box on v0.16.1 with Cloudflare R2, over 6.5 hours and 212 text requests: about **186 GB downloaded** and **51 GB uploaded**, which cost more than the rental. The causes and what changed:

- **Earlier turns of a conversation stayed in the bucket and were downloaded again on every refresh**, more each turn (0, 1, 3, 6, 10, 15 manifests over six turns of one conversation). Now a turn's upload deletes the turns it supersedes, so the bucket holds one session per conversation.
- **Every turn was uploaded at once**, though the next turn replaced it seconds later. Now it waits (the 30 seconds above).
- **A session's index (the per-token header) was re-sent in full every turn**, 2 to 6 MB. Now it is stored in pieces and a turn re-sends about 0.5 MB whatever the context length; a session's manifest is a few KB.
- **Saved recurrent states moved every turn, so unchanged ones were uploaded again.** Each is now stored on its own, so one that did not change is never sent again.
- **Sessions that share an opening (an agent's system and tools prompt) store that opening once**, in the bucket and on the disk, and a box that holds it downloads only a conversation's own part. Shared openings are evicted last.
- **Bucket keys sit under a folder named by a hash of the model and settings**, so a box lists only sessions it could use.
- **Downloads were invisible in the counters.** `ninfer_context_store_remote_manifest_download_bytes_total` and `_remote_manifest_downloads_total` now count them. Also new: `_lookups_total`, `_lookup_seconds_total`, `_remote_retired_total`, `_remote_uploads_skipped_total` and `_remote_fetch_bytes_per_second`.

These are the causes found in the logs of that one box; **the reduction in bandwidth has not been measured on a real bucket for this release**, only reproduced as CPU tests (for example, six turns of one conversation: no manifest re-downloads, against 0/1/3/6/10/15 before). Expect the quadratic growth to be gone; do not expect a particular percentage.

Known limit (#375): deleting a superseded session from the bucket is best effort. If another engine re-publishes an old one, or a crash loses a pending delete, a stale manifest can come back; the conversation's next turn removes it again, at the cost of a few KB of download per occurrence.

## Also changed

- **Length-bounded JSON strings in constrained output** (`minLength` / `maxLength`) compile to counted repetitions instead of one rule per length, so the rule count and mask size stop growing with the limit (#366, from upstream). Output is unchanged; the schema test ran about 3x faster in the PR's small sample.
- **Build:** the tree compiles with CUDA 13 and MSVC (#363). CI now builds and runs 38 CPU tests without `nvcc` on every pull request (#376).
- **Docs:** the fit calculator is shown in the README and served on GitHub Pages (#377), with model-card charts and a llama.cpp comparison (#361, #362), README badges and a demo (#364, #365).

## Not verified in this release

- **Windows:** built and dependency-checked only; nothing was started or served on Windows.
- **The context-store changes (#373, #374) were not run against a real bucket, and not on a GPU.** Their tests are CPU tests (an in-memory bucket and a GPU-free fake model). The two-box check in #373 (second engine's `remote_downloads_total` and `hydrations_total` rise on a copy of the artifact with a different file time) has not been run, and `qwen3_5` checkpoint export (`export_checkpoints`) was compiled but not run on a GPU.
- #363 was written without a CUDA 13 compiler; it is known to compile only on CUDA 12.8 (this release's toolkit).
- The full GPU test suite on the release commit.

## Verification

VERIFICATION_PLACEHOLDER

## Windows archive: built, not run

`ninfer-rtx3090-windows-x64-0.16.2-rtx3090.zip` was compiled with CUDA 12.8.1 and Visual Studio 2022 on a Windows machine that has no NVIDIA GPU, so the programs could not be started there. WINDOWS_CHECKS_PLACEHOLDER If it fails to start for you, please report it; v0.16.0 remains the last Windows archive that was run on a card.

## Install and upgrade

Linux: unpack `ninfer-rtx3090-linux-x64-0.16.2-rtx3090.tar.gz` next to your `models` folder and run `./run.sh <profile>`; `docs/release-archive-linux.md` lists the glibc 2.38, CUDA 12.8, cuBLAS, FFmpeg 6, libcurl and libzstd requirements. Windows: unpack the zip and run `run.bat <profile>`. Check the SHA256SUMS file for your platform. Upgrading from v0.16.1 needs no changes to a working launch script (see "Upgrade notes" for what a bucket does after the upgrade).
