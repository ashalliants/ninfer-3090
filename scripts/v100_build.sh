#!/usr/bin/env bash
# Configure and build NInfer for Tesla V100 (sm_70) on any Linux host, with ccache when it is
# installed (scripts/setup-ccache.sh installs and configures it; the build works without it).
# Usage: scripts/v100_build.sh [ninja targets...]   (default: ninfer ninfer-serve)
#   NINFER_BUILD_TESTS=1   also configure the tests (target ninfer_tests builds all of them)
#   BUILD_DIR              build tree (default $HOME/build-v100, where scripts/run-v100.sh looks)
#   JOBS                   parallel compile jobs (default: min(cores, RAM GiB / 2); one nvcc job
#                          peaks at 1.5-2 GiB, and an OOM-killed container cannot be restarted)
# CUDA 12.8 is the newest toolkit that still compiles for Volta (13.0 dropped sm_70); it warns that
# offline compilation for sm_70 is deprecated, and works.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$HOME/build-v100}"

# ccache, when installed, with the user's own cache and configuration (CCACHE_DIR and the size
# come from scripts/setup-ccache.sh's settings, or ccache's per-user default).
launcher=()
if command -v ccache >/dev/null; then
  # nvcc output is only cacheable when the compiler hash is stable and sloppiness covers
  # the timestamps nvcc embeds in temporary file names.
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime}"
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
  launcher=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
else
  echo "ccache not found: building without it (scripts/setup-ccache.sh installs it)." >&2
fi

testing=OFF
[ "${NINFER_BUILD_TESTS:-0}" = "1" ] && testing=ON

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  ${launcher[@]+"${launcher[@]}"} \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING="$testing"

if [ -z "${JOBS:-}" ]; then
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  JOBS=$(( ram_gb / 2 )); [ "$JOBS" -lt 2 ] && JOBS=2
  [ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
fi

if [ "$#" -eq 0 ]; then set -- ninfer ninfer-serve; fi
ninja -C "$BUILD" -j "$JOBS" "$@"
if [ "${#launcher[@]}" -gt 0 ]; then ccache -s | head -12; fi
