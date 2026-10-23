#!/usr/bin/env bash
# Drive the Windows release build on a remote GPU-less Windows box over SSH, from a Mac or Linux
# machine, then pull the zip and its checksum file back into ./dist.
#
#   scripts/release-windows-remote.sh <ssh-host> [commit]
#
#   ssh-host   a Host alias from ~/.ssh/config (or user@address) for the Windows machine. See
#              docs/maintainer/release-process.md, "Building the Windows archive without a GPU",
#              for the one-time OpenSSH setup on the Windows side.
#   commit     defaults to HEAD. It must already be on a remote (the Windows box fetches it from
#              GitHub), and the working tree must be clean.
#
# The work is scripts/release-windows.ps1 on the remote side. It is safe to re-run: after a dropped
# connection just run this again and the build continues from the build tree and caches. The first
# run installs the toolchain and builds ffmpeg through vcpkg, so allow a long time.
set -euo pipefail

host="${1:?usage: release-windows-remote.sh <ssh-host> [commit]}"
repo_root="$(git rev-parse --show-toplevel)"
commit="$(git -C "$repo_root" rev-parse "${2:-HEAD}")"
remote_url="${NINFER_RELEASE_REPO_URL:-https://github.com/ashalliants/ninfer-3090.git}"
remote_dir='C:/ninfer-release/src'

if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to release from a dirty tree: commit or stash first." >&2
  exit 1
fi
# The Windows box fetches from $remote_url, so the commit must be on the remote that points at it (any
# name: origin, fork, ...), not on a remote that merely has the same branch.
remote_name="$(git -C "$repo_root" remote -v | awk -v u="${remote_url%.git}" '$3 == "(fetch)" { x = $2; sub(/\.git$/, "", x); if (x == u) { print $1; exit } }')"
if [[ -z "$remote_name" ]]; then
  echo "No git remote points at $remote_url; add one, or set NINFER_RELEASE_REPO_URL to a URL one of your remotes uses." >&2
  exit 1
fi
git -C "$repo_root" fetch --quiet "$remote_name" 2>/dev/null || true
if [[ -z "$(git -C "$repo_root" branch -r --contains "$commit" 2>/dev/null | grep -E "^\s*${remote_name}/")" ]]; then
  echo "Commit ${commit:0:12} is not on the $remote_name remote ($remote_url); the Windows box fetches from there. Push it first." >&2
  exit 1
fi

ssh_opts=(-o ServerAliveInterval=30 -o ServerAliveCountMax=20)
run_ps() { ssh "${ssh_opts[@]}" "$host" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command - ; }

echo "==> Checking out ${commit:0:12} on $host"
run_ps <<EOF
\$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path C:/ninfer-release | Out-Null
if (-not (Test-Path '$remote_dir/.git')) { git clone --quiet '$remote_url' '$remote_dir'; if (\$LASTEXITCODE) { throw 'clone failed' } }
git -C '$remote_dir' fetch --quiet origin; if (\$LASTEXITCODE) { throw 'fetch failed' }
git -C '$remote_dir' checkout --quiet --force --detach $commit; if (\$LASTEXITCODE) { throw 'checkout failed' }
git -C '$remote_dir' clean --quiet -fd -e dist
git -C '$remote_dir' log --oneline -1
EOF

echo "==> Running release-windows.ps1 (this is the long step; log is on the box at C:\\ninfer-release\\release-windows.log)"
ssh "${ssh_opts[@]}" "$host" powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
  -File "$remote_dir/scripts/release-windows.ps1"

echo "==> Pulling the archive back to $repo_root/dist"
mkdir -p "$repo_root/dist"
names="$(ssh "${ssh_opts[@]}" "$host" powershell -NoProfile -NonInteractive -Command \
  "Get-ChildItem '$remote_dir/dist' -File | Where-Object { \$_.Name -like '*windows*' } | ForEach-Object Name" | tr -d '\r')"
[[ -n "$names" ]] || { echo "No Windows archive found in $remote_dir/dist on $host." >&2; exit 1; }
while IFS= read -r name; do
  scp "${ssh_opts[@]}" "$host:$remote_dir/dist/$name" "$repo_root/dist/$name"
done <<< "$names"

( cd "$repo_root/dist" && for sums in SHA256SUMS-v*-windows.txt; do shasum -a 256 -c "$sums"; done )
ls -l "$repo_root/dist"
echo "Release commit: $commit"
echo "REMINDER: the Windows executables were NOT run (no NVIDIA driver on the build box). Say so in the release notes."
