#!/usr/bin/env bash
# Bash counterpart of scripts/release-windows.ps1, which builds the Windows release archive and has to
# run on Windows (it installs and drives the MSVC and CUDA toolchain). From Linux or macOS the way to
# get that archive is to drive a Windows machine over SSH, which is what this does:
#
#   scripts/release-windows.sh <ssh-host> [commit]
#
# See scripts/release-windows-remote.sh and docs/maintainer/release-process.md.
set -euo pipefail
exec "$(dirname -- "${BASH_SOURCE[0]}")/release-windows-remote.sh" "$@"
