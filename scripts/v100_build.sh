#!/usr/bin/env bash
# Configure and build NInfer for Tesla V100 (sm_70) with ccache on a rented box.
# Usage: scripts/v100_build.sh [ninja targets...]   (default: ninfer ninfer-serve)
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-/workspace/build-v100}"

command -v ccache >/dev/null || { apt-get update -qq && apt-get install -y -qq ccache; }
export CCACHE_DIR="${CCACHE_DIR:-/workspace/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
# nvcc output is only cacheable when the compiler hash is stable and sloppiness covers
# the timestamps nvcc embeds in temporary file names.
export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime}"
export CCACHE_COMPILERCHECK=content

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF

if [ "$#" -eq 0 ]; then set -- ninfer ninfer-serve; fi
ninja -C "$BUILD" "$@"
ccache -s | head -12
