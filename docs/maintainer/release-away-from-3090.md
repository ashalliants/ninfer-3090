# Cutting a release while away from the RTX 3090 PC (handoff for a fresh agent)

This page is self-contained. Hand it to an agent with no context and it has what it needs to set up
the machinery and, later, cut a release. `release-process.md` has the normal flow and its gotchas;
this covers only what changes when the only machine to hand is a Mac.

## Situation

- The maintainer is away for about a month with only a Mac (no GPU). The 3090 PC is off and
  unreachable. A release is still due.
- A GPU is not needed to **compile** CUDA, only to run it. So:
  - **Linux archive:** build, package and smoke-test on a rented Vast.ai RTX 3090.
  - **Windows archive:** build and package on a GPU-less x64 Windows mini PC (GMKtec K8 Plus, Ryzen 7
    8845HS, Radeon iGPU, Windows 11 Pro) that the maintainer can reach.
- **CUDA 12.8.1 on both platforms.** It is what the Vast image `vastai/base-image:cuda-12.8.1-auto`
  carries, `AGENTS.md` and `build.ps1` pin 12.8, and the Windows zip ships `cublas64_12.dll` from the
  toolkit. Do not move to a newer toolkit for speed without measuring it on the card.

## The one thing that cannot be done: run the Windows binaries

Every executable imports `nvcuda.dll` (the driver API, via `CUDA::cuda_driver`, for the VMM pools). With
no NVIDIA driver on the build box the loader stops them before `main`, so:

- `package-release.ps1` cannot do its normal check (start each exe with only Windows on `PATH`). When
  `nvcuda.dll` is absent it checks import tables instead, with `dumpbin /dependents`: every DLL imported
  by any exe or DLL in the archive must be shipped in it, be a Windows DLL, or be an NVIDIA driver DLL.
  It prints a warning saying the binaries were not run. With a driver present the original check runs.
  `NINFER_PACKAGE_STATIC_DEPENDENCY_CHECK=1` forces the static check (use it to test it on a 3090 PC).
  The maintainer approved this change; never weaken the real check beyond it.
- No real-card smoke test is possible for Windows. **The release notes' Verification section must say
  the Windows archive was built and dependency-checked but not run**, and the final report must list
  the Windows stripped-`PATH` start and card smoke test as unverified. Run them when the 3090 PC is back.

## What this PR added

| File | Purpose |
|---|---|
| `scripts/release-windows.ps1` | Runs **on the Windows box**. Installs VS 2022 Build Tools (C++ workload), CMake, Ninja, CUDA 12.8.1 components (no driver), vcpkg at the `vcpkg.json` baseline and ccache; builds `ninfer ninfer-serve ninfer_bench` from the checked-out commit via `build.ps1 -Package`; prints archive hashes. Re-runnable. Writes `C:\ninfer-release\release-windows.log` and `STATUS.txt` (last line `DONE` or `FAILED`). |
| `scripts/release-windows-remote.sh` | Runs **on the Mac**. Checks out a pushed commit on the Windows box over SSH, runs the script above, copies the zip and checksum file into `./dist`, verifies them. |
| `scripts/vast/vast-release-linux.sh` | Runs **on the Mac**. Copy it next to `vast-dev-launch.sh` (the maintainer's `~/empty/vast-ninfer/`; it calls `vast-dev-sync.sh` and `vast-dev-ssh.sh` there). Builds, packages, checks and smoke-tests the Linux archive on the tracked Vast instance, then pulls it back into `<repo>/dist`. Detached on the instance and polled, so a dropped connection is harmless. |
| `scripts/package-release.ps1` | Static dependency fallback described above. |
| `scripts/build.ps1` | Uses `VCPKG_ROOT` when set; `-Target` takes several values; `-Package` builds only the three release products. |

**None of the PowerShell or the Vast script has been run.** The author had only a Mac with no
PowerShell. Expect to fix a snag or two on the first run, and record each in `release-process.md`.
The likeliest failure is the silent CUDA install with an explicit component list (reported to hang);
`release-windows.ps1` times out after 40 minutes and says to run the installer by hand with the driver
unticked, then re-run with `-SkipBootstrap`.

## Setup, in order

### 1. Windows box (the maintainer does this once, in an elevated PowerShell)

```powershell
winget install Git.Git
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Set-Service sshd -StartupType Automatic ; Start-Service sshd
New-ItemProperty -Path HKLM:\SOFTWARE\OpenSSH -Name DefaultShell -PropertyType String -Force `
  -Value 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
Set-Content -Path C:\ProgramData\ssh\administrators_authorized_keys -Value '<the Mac public key>'
icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant 'Administrators:F' /grant 'SYSTEM:F'
powercfg /change standby-timeout-ac 0 ; powercfg /change hibernate-timeout-ac 0
```

Reach it over a private network such as Tailscale. **Never forward port 22 to the internet.** Then add a
`Host` alias for it to `~/.ssh/config` on the Mac and confirm `ssh <alias> hostname` works. Note the Mac's
`~/.ssh/config` routes `Host *` through the 1Password agent; add `IdentityFile` and `IdentitiesOnly yes`
to the new alias if the right key is not offered.

### 2. Vast (already scripted by the maintainer)

`vast-dev-launch.sh` rents the instance and builds the fork; it needs the Vast account to have credit
(`vastai show user --raw | jq .balance`) and a 2FA session (`vastai tfa login --method-type totp -c <CODE>`,
which only the maintainer can run). Quirks and fixes are in the maintainer's `vast-dev-box-workflow`
notes: use the key `~/.ssh/id_rsa_ash` with `IdentityAgent=none`, never fall back to `$VASTAI_API_KEY`, run
long builds under `nohup`, and prefer offers with 48 GB or more of RAM.

### 3. Dry run, well before the release

Pick any pushed commit (for example this branch) and run both halves, to find the snags while there is time
and to warm the caches (the first Windows run builds ffmpeg through vcpkg and compiles every kernel cold):

```bash
scripts/release-windows-remote.sh <ssh-alias> <commit>     # long; safe to re-run after a dropped connection
~/empty/vast-ninfer/vast-release-linux.sh <repo-dir>       # needs VERSION and RELEASE_NOTES_X.Y.Z.md committed
```

The packagers refuse without `RELEASE_NOTES_<version>.md` for the `VERSION` file, so a dry run needs a
throwaway release-notes file committed in the dry-run commit.

## Cutting the release

Follow the `/release` skill and `release-process.md` for the version choice, the breaking-changes audit,
the notes and publishing, with these substitutions for step 4 (build) and step 6 (verify):

1. Commit `VERSION` first, then the notes (as the doc says). Push the release branch to the fork.
2. Build both archives **from that same pushed commit**, in parallel:
   `scripts/release-windows-remote.sh <ssh-alias>` and `vast-release-linux.sh <repo-dir>`.
3. `Verification` in the notes: Linux built and served a chat completion on a real 3090 (Vast); Windows
   built and dependency-checked only. Paste the `ldd` list the Vast script saves as
   `dist/release-ldd-linux.txt` and diff it against the previous release's.
4. Publish exactly as in the doc (`gh ... -R ashalliants/ninfer-3090`; confirm with the maintainer first).
5. Destroy the Vast instance (`vast-destroy.sh`) after pulling everything. Its idle watchdog destroys it
   anyway, and it is the only copy of its logs.

## Checks the agent should do before trusting any of this

- `ssh <alias> hostname` and `ssh <alias> "winget --version"` from the Mac.
- On the Windows box: `nvcc --version` reports 12.8, `where cl` is not a VS 2026 compiler (the build script
  imports VS 2022's environment itself), `dumpbin` is found.
- After the first Windows build: the `dependency check:` line in the log, and `--version` read from the
  built exe is impossible there, so compare `git rev-parse HEAD` with the commit the script printed.
- Do not weaken any packager check to get green. If one fails, find out why.
