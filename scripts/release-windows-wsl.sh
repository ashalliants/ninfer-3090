#!/usr/bin/env bash
# Build the Windows release archive from a WSL shell on the same Windows machine, with no SSH: WSL can run
# powershell.exe. The machine needs no NVIDIA GPU, only the toolchain release-windows.ps1 installs (do that
# once, elevated: `release-windows.ps1` without -SkipBootstrap; docs/maintainer/release-process.md).
#
#   scripts/release-windows-wsl.sh [repo-dir]       (default: the repository this script is in)
#
# Builds the COMMITTED HEAD of repo-dir (VERSION bumped and RELEASE_NOTES_X.Y.Z.md committed: the packager
# refuses without them). The commit travels as a git bundle, so nothing needs to be pushed. The working copy
# is C:\ninfer-release\src and is reused, so a second run is incremental. Git on that copy runs through
# powershell.exe: git on /mnt/c from WSL takes minutes and reports phantom changes.
# Copies the zip and checksum file into <repo-dir>/dist and verifies them.
set -euo pipefail

repo="$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")/..}" && pwd)"
ps=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
work=/mnt/c/ninfer-release
[[ -x "$ps" ]] || { echo "powershell.exe not found at $ps: not WSL on Windows?" >&2; exit 1; }
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] || { echo "Tracked changes in $repo: commit them first." >&2; exit 1; }
commit="$(git -C "$repo" rev-parse HEAD)"
version="$(tr -d '\r\n' < "$repo/VERSION")"
[[ -f "$repo/RELEASE_NOTES_${version%%-*}.md" ]] || { echo "Missing RELEASE_NOTES_${version%%-*}.md in the commit." >&2; exit 1; }
mkdir -p "$work"

git -C "$repo" bundle create "$work/release.bundle" HEAD >/dev/null 2>&1
cat > "$work/release-wsl-run.ps1" <<PS
\$ErrorActionPreference = 'Stop'
if (-not (Test-Path C:\ninfer-release\src\.git)) { git clone -q C:\ninfer-release\release.bundle C:\ninfer-release\src }
Set-Location C:\ninfer-release\src
git fetch -q C:\ninfer-release\release.bundle HEAD
git checkout -q -f --detach FETCH_HEAD
if ((git rev-parse HEAD) -ne '$commit') { throw 'checkout is not the release commit' }
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist\\*$version*
\$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
\$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\release-windows.ps1 -SkipBootstrap
exit \$LASTEXITCODE
PS

echo "==> Building $commit ($version) on Windows; log: C:\\ninfer-release\\release-windows.log"
"$ps" -NoProfile -ExecutionPolicy Bypass -File 'C:\ninfer-release\release-wsl-run.ps1' | tee "$work/release-wsl.out"
[[ "$(tail -n1 "$work/STATUS.txt" | tr -d '\r\n')" == DONE ]] || { echo "Windows build did not finish (STATUS.txt)." >&2; exit 1; }

mkdir -p "$repo/dist"
cp "$work/src/dist/ninfer-rtx3090-windows-x64-$version.zip" "$work/src/dist/SHA256SUMS-v${version%%-*}-windows.txt" "$repo/dist/"
( cd "$repo/dist" && sha256sum -c "SHA256SUMS-v${version%%-*}-windows.txt" )
echo "Windows archive built from $commit. Not run: no driver here (the packager checked import tables instead)."
