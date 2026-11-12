# Cutting a release

How a `vX.Y.Z-rtx3090` release is built, packaged, verified and published from this repository. The
steps are short; the gotchas below are the reason to read the page, because each one cost real time
or very nearly shipped a broken archive.

## The flow

1. **Prepare.** Branch `release/vX.Y.Z` from `master`. Set `VERSION` to the full tag (for example
   `0.11.0-rtx3090`), write `docs/release-notes/RELEASE_NOTES_X.Y.Z.md`, refresh the README highlights, open a PR and
   merge it. Everything version-specific is derived from `VERSION` by the packagers; there is nothing
   to copy and edit.
2. **Build both platforms from the same commit, after `VERSION` is set.** The binaries embed
   `VERSION` (`--version`, `GET /health`, the `X-NInfer-Version` header), so build from the release
   branch tip, not from a commit before the `VERSION` bump. A binary built from the pre-merge
   release-branch commit reports `<VERSION>+<commit>`; the tag lands on the merge commit, so that
   suffix is expected there, but a build from `master` before the bump would report the previous
   release. Check `ninfer-serve --version` against the archive name in the smoke test.
   - Windows: the VS 2022 BuildTools environment and an existing Ninja tree (see `AGENTS.md`,
     "Windows build environment"). Turn `NINFER_BUILD_BENCHMARKS` on: the archive ships
     `ninfer_bench.exe`. Build only `ninfer ninfer-serve ninfer_bench`.
   - Linux: under WSL, from a **fresh** `git archive` of the commit, targets `ninfer ninfer-serve
     ninfer_bench`, `-DCMAKE_CUDA_ARCHITECTURES=86`, `-DBUILD_TESTING=OFF` (the release only needs the
     products). See the WSL notes below.
3. **Package.** `scripts/package-release.ps1` (Windows zip) and `scripts/package-release.sh` (Linux
   tar.gz), or `build.ps1 -Package` / `build.sh --package`. Each writes the archive and
   `SHA256SUMS-vX.Y.Z-{windows,linux}.txt` under `dist/`.
4. **Smoke-test on the real machine** (below), not just the unit tests.
5. **Publish.** Merge the release PR, tag the merge commit (annotated), push the tag, then
   `gh release create vX.Y.Z-rtx3090 <zip> <tar.gz> <both SHA256SUMS files> -R ashalliants/ninfer-3090
   --notes-file docs/release-notes/RELEASE_NOTES_X.Y.Z.md --title "..." --latest`.

### A default expert state (model releases)

The binary archives ship no model files. When a model release should warm-start a fresh
Flash-Next install, it ships `<model>.expert-state.default` beside the model. Produce it from a
server's saved state, after a representative session on that exact artifact:

```
build/tools/ninfer-qwen4-exp-expert-state-export <model>.expert-state <model>.expert-state.default <model>.ninfer
```

The file is tied to the artifact id (`artifact_id=<id>` in place of the saved state's path, size and
time), so it must be regenerated whenever the artifact is converted again (every conversion writes
a fresh id; a default for another id is ignored). See [A default expert state](../serving.md#a-default-expert-state).

## Gotchas

**Always pass `-R ashalliants/ninfer-3090` to `gh`.** With no `-R` it resolves to the `upstream` remote
(`Don-Chad/ninfer-3090`) and silently reads or writes the wrong fork. The same applies to the babysit
helper (also set `GH_REPO`).

**Windows executables import cuBLAS, and the archive must carry it.** The cuBLAS prefill route makes
every executable import `cublas64_12.dll` (which imports `cublasLt64_12.dll`). A machine without the
CUDA Toolkit then fails to start any of them (`0xC0000135`), even when `--prefill-cublas` is never used.
The developer machine hides this because the Toolkit is on its `PATH`. `package-release.ps1` copies both
DLLs from `%CUDA_PATH%\bin` and **refuses to produce an archive whose executables do not start with only
Windows on `PATH`**, so this cannot regress silently. Do not weaken that check.

**Redistributing NVIDIA's DLLs has conditions.** The CUDA Toolkit EULA (Attachment A) lists the cuBLAS
DLLs as redistributable, including versioned names such as `cublas64_12.dll`. The grant requires that our
application has material functionality of its own, that the DLLs are used only by it, and that our
distribution terms are consistent with the EULA and protect NVIDIA's rights. The packager therefore ships
the toolkit's `EULA.txt` as `NVIDIA-CUDA-EULA.txt`, and the archive README names the two DLLs as NVIDIA's,
outside the project's Apache-2.0 licence. Re-read the EULA of the toolkit that builds a release.

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The Linux archive does not ship `libcublas` (the
Lt library alone is about 500 MB), so `docs/release-archive-linux.md` states the requirements: glibc 2.38
or newer (built on Ubuntu 24.04), the CUDA 12.8 runtime libraries including cuBLAS, FFmpeg 6, libcurl and libzstd.
Check them with `ldd` on the new binaries and update the guide if they change.

**Asset sizes.** Each executable is about 460-490 MB (every kernel is linked in). The Windows zip is about
1.5 GB with the cuBLAS DLLs and the Linux tar.gz about 1 GB. GitHub rejects release assets over 2 GiB, so
watch the Windows zip if more DLLs are added.

**Smoke-test the launcher default on the real card.** A busy desktop holding 2-3 GiB of the 3090 refuses
the default DFlash2 profile's full context. `run.{sh,bat}` therefore pass `--fit-context 32768`, so the
server starts at the largest context that fits and logs `context lowered to ...`, and the launcher retries
with a smaller Host context budget if pinning it fails; see `docs/maintainer/launcher-profiles.md`.
Start the launcher with nothing overridden and confirm it serves a request.

### Building on WSL

- **Use a fresh export**, not an old snapshot: `git archive` the commit to a new directory, so stale files
  cannot leak in. To add later documentation or script changes, extract only those paths. Extracting a
  whole `git archive` over a build tree stamps every source file with the commit time and forces a full
  rebuild.
- **`ninja: fatal: posix_spawn: Resource temporarily unavailable`** means the session's cgroup hit its
  process cap, not that `-j` is too high. Look at `/sys/fs/cgroup/init.scope/pids.{max,current}` with shell
  built-ins (a failing `fork` breaks `ps`). When leftover processes from other tooling fill it (they did:
  about thirty `pnpm`/`node` processes with 67 threads each), do not kill them; run the build in its own
  scope instead: `wsl.exe -u root -e systemd-run --scope --quiet --uid=<you> --gid=<you> -p
  TasksMax=infinity -- bash -lc '...'`.
- **Background long jobs with the harness, not the shell.** `wsl.exe -e bash -lc "cmd & disown"` does not
  survive; run the whole `wsl.exe` invocation as a background task and read its log.
- **A model on `/mnt/c` loads at about 200 MiB/s**, so a 19 GB artifact takes minutes per attempt. Copy it
  into the WSL filesystem for repeated runs.
- `BUILD_TESTING` is `OFF` by default; `ctest` then reports "No tests were found". Turn it on to run the
  suite, which the release itself does not need on Linux.

### Windows notes

- Reconfiguring the existing Ninja tree is the one thing that breaks it; a second tree configured against
  a worktree (with the vcpkg installed directory reused read-only and `-DVCPKG_MANIFEST_INSTALL=OFF`) is
  safe. A full ops build takes about ten minutes; incremental rebuilds under a minute.
- Extract the finished zip to a temporary folder and run each executable with only Windows on `PATH`
  before uploading, even though the packager now does the same check.
- `cmd` batch files: a trailing `^` on a `rem` line continues the comment onto the next line, an unquoted
  `>>` in `set /a` is a redirect, and `Tee-Object` writes UTF-16 (`findstr` cannot search it).

## Building the release without the RTX 3090 PC

Compiling needs the CUDA toolkit, not a card, so a release can be cut with the 3090 PC off. Written
for being away from it (a handoff for a fresh agent is in `release-away-from-3090.md`); the Windows half was run end to end for v0.16.1 (below); the Vast half has not been run yet, so expect a snag or two there
and record them here.

- **Linux archive: a rented Vast 3090.** Run from `scripts/vast/` (all four scripts below use a state file,
  `~/.cache/ninfer-release/instance_id`, so they can only touch the instance they rented, never another box
  on the same account that may be serving traffic):
  1. `./vast-up.sh [repo-dir] [hours]` rents the verified 3090 with at least 16 effective cores that costs least for the whole job (GPU hours plus the host's per-GB download and upload prices; `NINFER_VAST_EST_HOURS`, `NINFER_VAST_DOWN_GB` and `NINFER_VAST_UP_GB` set the estimate, and it prints the candidates it will try), narrows the search with Vast's `gpu_max_power` (350 W at the default 100%; its docs do not say whether that is the enforced limit, so each candidate prints it), checks the booted card's power limit and, if a host has capped it below `NINFER_VAST_MIN_POWER_PCT` of the card's default (100 by default; 0 turns the check off, which is fine for a build that times nothing), destroys that host and tries the next, up to `NINFER_VAST_MAX_HOSTS` (4), installs the
     build dependencies, arms an on-box kill switch (default 3 h) and downloads the 27B model for the smoke
     test. About 5 minutes. Bandwidth can cost more than the GPU: the v0.17.0 box was $0.85, of which $0.53 was 14 GB of transfer. A host stuck in "loading", or any failure after the rental, destroys the instance; run it again. The ssh key is `NINFER_VAST_KEY`, else `~/.ssh/id_rsa_ash`, else `~/.ssh/id_ed25519`, always with `IdentityAgent=none`. The model downloader sent to the box is the one in the release commit.
  2. `./vast-release-linux.sh <repo>` (the release commit checked out and clean) syncs the commit, builds
     `ninfer ninfer-serve ninfer_bench` from a `git archive` of it, runs `package-release.sh`, checks
     `--version`, `ldd` and the checksums, serves one chat completion with the default profile on the real
     card, and pulls the archive into `dist/`. The build is detached on the instance and polled.
  3. `./vast-destroy.sh` when the archive is back. Do not leave it running.

  The instance image is Ubuntu 24.04 with CUDA 12.8.1, the baseline `docs/release-archive-linux.md`
  documents. v0.16.2 was cut this way by hand (same steps, before these helpers existed): cold build 5 min
  on 24 effective cores, `nproc` there reports 192, so the build caps `-j` at 32. v0.17.0 ran the helpers
  end to end (a Quebec host, $0.26/h, 22 cores: cold build about 15 minutes, a rerun about 5 with ccache).
  What it hit:
  - **The smoke test's server outlived the script** until the fix in v0.17.0: `run.sh` runs `ninfer-serve`
    in a pipeline, so it is a grandchild of the launcher, and a second `vast-release-linux.sh` on the same
    box failed with "not enough free GPU memory". The script now starts the launcher under `setsid` and
    stops its process group. If a run still leaves a `ninfer-serve` on the card, stop it by PID before rerunning.
  - **Another model for the smoke test:** copy the scripts elsewhere, change `download-model.sh qwen38-27b`
    in `vast-up.sh`, and put `NINFER_MODEL=/root/models/<artifact>` before `./run.sh` in
    `vast-release-linux.sh`. Bonsai 2 27B (8.9 GiB) serves under the `qwen38-27b` profile.
    `NINFER_VAST_STATE=<dir>` keeps the copy's state file apart. To avoid a region, filter the offers on
    `geolocation` in `vast-up.sh`.
  - Every commit after the build needs a rebuild on the box (the script refuses a remote HEAD that differs).
    A rerun is about 5 minutes on ccache, so fix first, then rerun, rather than waiting for the first run.
  - **v0.17.2 ran the helpers cleanly** (an Ontario host, 16 cores, about $0.29 estimated): `vast-up.sh` about 5
    minutes, then `vast-release-linux.sh` about 15 minutes to a pulled-back archive. Run `vast-destroy.sh` the
    moment the archive is in `dist/` and delete any `send_later` backstop.
  - **v0.18.0 (a Spain host, $0.37/h, 24 cores): `vast-up.sh` about 5 minutes, a warm rerun of
    `vast-release-linux.sh` about 4.** Traps it hit:
    - **A stale or revoked API key fails late and confusingly.** `vast-up.sh` printed a Python
      `JSONDecodeError` and then "No suitable offer"; the real message (`401 Invalid user key`) was a few
      lines above. Run `vastai show user` first. A throwaway key from an earlier release was dead.
    - **Killing `vast-release-linux.sh` does not stop the build on the box.** The build is detached there. A
      second run then started a second packager on the same tree and failed with `tar: ... File removed
      before we read it`. If you must restart, `./vast-dev-ssh.sh 'pgrep -fa "ninja|cmake|package"'` until
      nothing is running, then rerun. Better: do not start the Linux run until the commit is final (see the
      Windows trap below), so you never need to kill it.
    - `vast-up.sh` rents and destroys hosts until one has a full-power card, so `vastai show instances`
      may briefly list two `ninfer-release` boxes. Only the one in `~/.cache/ninfer-release/instance_id`
      is yours; other labels (for example `nevermetered:`) are live services, never touch them.
    - The script's log is on your machine only; the box's log goes with the box. Read the smoke test's
      reply out of `/root/release-linux.log` (`./vast-dev-ssh.sh`) **before** `vast-destroy.sh` if the notes
      should quote it.
  - **Editing the notes after the Linux archive exists does not need another box.** The tarball embeds
    `RELEASE_NOTES_X.Y.Z.md` and an inner `SHA256SUMS.txt`. Extract it to a scratch directory, copy the final
    notes in, regenerate the inner sums exactly as `package-release.sh` does (`find . -maxdepth 1 -type f !
    -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z`, then `sha256sum`), re-pack with
    `tar -C "$scratch" --owner=root --group=root -czf "$archive" "$product_name"` (the original entries are
    root-owned; `$scratch` is the directory you extracted into, `$product_name` the single top-level
    directory it contains, `ninfer-rtx3090-linux-x64-<tag>`, and `$archive` the tarball to replace, in
    `dist/`, which is where the checksum step looks; this mirrors the `tar -C "$dist_root"` line in
    `package-release.sh`) and rewrite
    `SHA256SUMS-vX.Y.Z-linux.txt`. Check the inner sums, the outer sum and `diff` the embedded notes. Re-packing
    1 GB takes a couple of minutes, so run it in the background.
- **Windows archive: any x64 Windows machine, no NVIDIA GPU needed.** `scripts/release-windows.ps1`
  installs the toolchain (VS 2022 Build Tools, CMake, Ninja, CUDA 12.8.1 without the driver, vcpkg at the
  manifest baseline, ccache), builds the three products and runs the packager.
  `scripts/release-windows-remote.sh <ssh-host>` drives it over SSH and pulls the zip back. The first run installs the toolchain and so needs an elevated
  session (an SSH login as an Administrators user is one); later runs can set `NINFER_SKIP_BOOTSTRAP=1` to build and
  package without elevation. The first
  run builds ffmpeg through vcpkg and compiles every kernel cold, so it is slow; later runs are
  incremental.
- **What the Windows box cannot verify.** With no driver there is no `nvcuda.dll`, which every
  executable imports, so the executables cannot start there. `package-release.ps1` therefore checks the
  import tables with `dumpbin /dependents` (every DLL imported must be shipped, a Windows DLL, or an
  NVIDIA driver DLL) instead of starting them. It says so, and uses the real start-up check whenever a
  driver is present (`NINFER_PACKAGE_STATIC_DEPENDENCY_CHECK=1` forces the static check, to test it).
  Nothing is served on the card, so the release notes must say the Windows archive was built and
  dependency-checked but not run, and the first chance to run it should be taken.
- **Using CUDA 12.8.1 on both platforms** keeps one cuBLAS to reason about. Do not move the Windows build
  to a newer toolkit for speed without measuring it on the card.

### Windows without a card, from WSL: `scripts/release-windows-wsl.sh`

On the GPU-less Windows machine whose WSL you are in (the GMKtec box), with the toolchain installed once
(below), one command builds, packages and verifies the zip and copies it to `dist/`:
`scripts/release-windows-wsl.sh` (committed HEAD only). It sends the commit as a git bundle (nothing to
push), checks it out in `C:\ninfer-release\src` through `powershell.exe`, and runs
`release-windows.ps1 -SkipBootstrap`. v0.16.2 did the same steps by hand: about 25 minutes with a warm
ccache, then 5 minutes for each repackage. Traps:
- **Run git on `C:\ninfer-release\src` through `powershell.exe`, never from WSL.** From WSL over `/mnt/c`
  `git status` took over two minutes (and hit the tool timeout) and listed files as modified that were
  not.
- **`powershell.exe` is not on the WSL `PATH`**; the script uses
  `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe`.
- **Package twice.** The zip embeds the notes, so a notes edit changes its hash; rerunning the script is
  incremental and removes the old `dist/*<version>*` first.
- The first run of `release-windows.ps1` without the notes in the checkout compiles everything and then
  fails at packaging with "Missing docs/release-notes/RELEASE_NOTES_X.Y.Z.md": commit the notes before building, or just rerun.
- **This build is the only MSVC compile of `master`.** CI builds Linux only, so a Windows-only compile
  error surfaces here first. v0.17.0 found `src/serve/serve_options.cpp` failing with `C1061: blocks nested
  too deeply`: MSVC limits an if/else-if chain to about 128 levels, and the option parser had passed it.
  New flags with a shared prefix go in their own chain ahead of the main one, as `--context-store*` and
  `--lossy-prefill-*` do.
- **Run the script by absolute path from the repository root.** A background shell that drifted into `dist/`
  failed instantly with `scripts/release-windows-wsl.sh: No such file or directory` (exit 127), which looks
  like a completed task.
- **The notes inside the Windows zip have CRLF line endings** (the checkout on `C:` converts them). When
  comparing them with the committed file use `diff --strip-trailing-cr`; the Linux copy matches byte for byte.
- **Build the Windows zip first, then start the Linux run.** The Windows build is the only MSVC compile, so
  it is where a release-blocking error shows up, and fixing one changes the commit that Linux must be built
  from. v0.18.0 hit `C1061: blocks nested too deeply` again in `serve_options.cpp` (the main option chain,
  not the context-store or lossy-prefill chains that were already split) after #394 added about 30 flags;
  the fix was to split the chain in two with a `handled_first_half` flag and `continue`. The Linux box had
  already been building, so it was rebuilt. Every new batch of flags can trip this again: if the release
  adds serve options, check the Windows compile before renting anything, or at least before the Linux build
  finishes.
- **The disk check is a hard stop at 20 GB free on C:.** The old zips in `C:\ninfer-release\src\dist`
  (22 GB at v0.18.0, five releases of zips plus their unpacked folders) are safe to delete once those
  releases are published, and need the user's say-so. `cuda.7z` and `cuda_12.8.1_windows.exe` in
  `C:\ninfer-release` (6 GB) are only used by a fresh bootstrap and can also go. A failed attempt leaves
  a partly built tree, so a rerun is incremental.
- **A copy still in progress looks like a small file.** The script prints its result before `dist/` has
  the zip: a `ls -l` taken then showed 580 MB (it ends at 1.6 GiB) and `unzip` said "End-of-central-directory
  signature not found". Wait for the background task to complete before checking `dist/`.
- **A repackage rebuilds binaries that say `+<commit>`.** After a notes-only commit the Windows binaries
  report the newer commit; that is expected (the tag is on the merge commit anyway).
- A `-SkipBootstrap` run needs 20 GB free on C: (the full bootstrap needs 50 GB). The GMKtec box had 29 GB
  in 2026-10; old releases' zips in `C:\ninfer-release\src\dist` take about 1.6 GB each once published.

### Windows without a card: what the v0.16.1 run needed

The v0.16.1 Windows zip (1.5 GB, built in about 35 minutes cold, including ffmpeg through vcpkg) came from
`scripts/release-windows.ps1 -SkipBootstrap` on the GPU-less GMKtec box, driven from a WSL session on that
same machine (WSL can run `powershell.exe`, so no SSH was needed). To do it again:

1. **Checkout.** `release-windows.ps1` wants a clean checkout of the commit to build. Clone to
   `C:\ninfer-release\src` and check out the release tag. Until the tooling is on `master`, merge it on top
   locally and confirm the product directories did not change:
   `git diff --stat <tag> HEAD -- src include apps CMakeLists.txt` must print nothing.
2. **Toolchain, once, elevated.** Needs admin and nothing else does. Install VS 2022 Build Tools (C++ workload and
   the Windows 11 SDK), CMake and Ninja with `winget`, and put CUDA 12.8 in
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`. Run the rest as a normal user.
3. **CUDA without a card.** The NVIDIA installer's silent mode exits after about two seconds with exit code
   -522190823 and installs nothing when there is no NVIDIA GPU (its log stops after "No driver installed for
   the display device"). `Expand-CudaToolkit` in `release-windows.ps1` is the fallback: the installer is a
   7z archive, so it unpacks the compiler and library packages with the standalone `7zr.exe` and copies them
   into the toolkit root (nvcc, cudart, cuBLAS, CCCL, NVRTC, NVTX, NVML headers, profiler API). It runs
   automatically after a failed install. It needs admin because it writes under Program Files; if you
   unpack by hand instead, stage the tree elsewhere and `robocopy /E` it in from an elevated shell. This
   toolkit has no Visual Studio CUDA integration, which the Ninja build does not use.
4. **Build.** From a shell that has the new machine `PATH` (a shell opened before the installs does not see
   CMake or Ninja; read `Machine` and `User` from `[Environment]::GetEnvironmentVariable('Path', ...)`):
   `powershell -File scripts\release-windows.ps1 -SkipBootstrap`. It checks `STATUS.txt` for `DONE`, runs
   the packager's import-table check and prints the zip hash.
5. **Publish.** Upload the zip and its `SHA256SUMS-vX.Y.Z-windows.txt` to the release with
   `gh release upload ... -R ashalliants/ninfer-3090`, and say in the notes that it was built and
   dependency-checked, not run.

Snags: the `7zr` fallback was tested by itself against the real installer, but the installer-fails-then-fallback path
inside `Install-Cuda` was exercised by hand (the unpack and copy), not by a fresh machine run. The bootstrap script's last
log line can fail with "file in use" because the transcript still holds the log; it is harmless.

One-time OpenSSH setup on the Windows box (elevated PowerShell), so `release-windows-remote.sh` can drive it:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic ; Start-Service sshd
New-ItemProperty -Path HKLM:\SOFTWARE\OpenSSH -Name DefaultShell -PropertyType String -Force `
  -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
# Administrators' keys live in one file with a locked-down ACL, not in ~\.ssh:
Set-Content -Path C:\ProgramData\ssh\administrators_authorized_keys -Value '<your public key>'
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F'
powercfg /change standby-timeout-ac 0 ; powercfg /change hibernate-timeout-ac 0
```

Do not forward port 22 to the internet. Reach the box over a private network such as Tailscale.

## Writing the notes when a release adds many options

v0.18.0 added about 30 opt-in options from one PR (#394), and the first draft listed them by name with a
pointer to the docs. That hides the part a reader needs: **which ones are slower**. Read the PR's speed
table and put every measured regression in the notes next to the option, not just the wins (here
`--ngram-prefetch` -16.6%, `--cpu-verify-misses` -16.7%, `--hc-int8` -26.4%, `--ggml-decode-mma` -5.5%
end to end). `AGENTS.md` requires adverse results in the notes. Also check the breaking-change audit
against the Host/Origin-style defaults that change behaviour for working setups (a name-based client
now needs `--allowed-host`), which no flag diff shows.

Where a PR body and `docs/maintainer/flash-next-plan.md` disagree (the plan's "Verification status" still
said nothing had run on a GPU after #394 had run it), trust the PR body's dated results and fix the doc.

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Both archives: every file matches its inner `SHA256SUMS.txt`, and the outer checksum file verifies.
- Windows: all three executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on the real 3090.
