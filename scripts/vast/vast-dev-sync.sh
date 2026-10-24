#!/usr/bin/env bash
# Put the committed HEAD of <repo-dir> on the release instance as a git checkout at /root/ninfer-3090.
# Uncommitted edits are not sent: the release is the commit.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${1:?usage: vast-dev-sync.sh <repo-dir>}" && pwd)"
bundle="$(mktemp -d)/repo.bundle"
git -C "$repo" bundle create "$bundle" HEAD >/dev/null 2>&1
"$root/vast-dev-ssh.sh" 'rm -rf /root/ninfer-3090 /root/repo.bundle && cat > /root/repo.bundle' < "$bundle"
"$root/vast-dev-ssh.sh" 'git clone -q /root/repo.bundle /root/ninfer-3090 && git -C /root/ninfer-3090 checkout -q --detach $(git -C /root/ninfer-3090 rev-parse HEAD)'
rm -rf "$(dirname "$bundle")"
