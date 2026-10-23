# Cutting a release

How a `vX.Y.Z-rtx3090` release is built, packaged, verified and published from this repository. The
steps are short; the gotchas below are the reason to read the page, because each one cost real time
or very nearly shipped a broken archive.

## The flow

1. **Prepare.** Branch `release/vX.Y.Z` from `master`. Set `VERSION` to the full tag (for example
   `0.11.0-rtx3090`), write `RELEASE_NOTES_X.Y.Z.md`, refresh the README highlights, open a PR and
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
   --notes-file RELEASE_NOTES_X.Y.Z.md --title "..." --latest`.

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

- **Linux archive: a rented Vast 3090.** `scripts/vast/vast-release-linux.sh <repo>` (copy it into the `vast-ninfer` scripts
  directory, beside `vast-dev-launch.sh`) syncs the committed release commit to the dev instance,
  builds `ninfer ninfer-serve ninfer_bench` from a `git archive` of it, runs `package-release.sh`,
  checks `--version`, `ldd` and the checksums, serves one chat completion with the default profile on
  the real card, and pulls the archive back into `dist/`. The instance image is Ubuntu 24.04 with CUDA
  12.8.1, the baseline `docs/release-archive-linux.md` documents.
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

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Both archives: every file matches its inner `SHA256SUMS.txt`, and the outer checksum file verifies.
- Windows: all three executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on the real 3090.
