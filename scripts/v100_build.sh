#!/usr/bin/env bash
# Configure and build NInfer for Tesla V100 (sm_70) on any Linux host, with ccache when it is
# installed (scripts/setup-ccache.sh installs and configures it; the build works without it).
# Usage: scripts/v100_build.sh [ninja targets...]   (default: ninfer ninfer-serve)
#   NINFER_BUILD_TESTS=1   also configure the tests (target ninfer_tests builds all of them)
#   NINFER_BUILD_BENCHMARKS=1  also configure the benchmarks; asking for the ninfer_bench target does it too
#   BUILD_DIR              build tree (default build-v100/ inside this checkout, where scripts/run-v100.sh
#                          looks first; each worktree gets its own tree, which ccache sharing needs: a
#                          tree that is not directly inside its checkout misses the cache, and gets a note)
#   JOBS                   parallel compile jobs (default: min(cores, RAM GiB / 2); one nvcc job
#                          peaks at 1.5-2 GiB, and an OOM-killed container cannot be restarted)
# CUDA 12.8 is the newest toolkit that still compiles for Volta (13.0 dropped sm_70); it warns that
# offline compilation for sm_70 is deprecated, and works.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$SRC/build-v100}"
# shellcheck source=scripts/lib/build-dir-note.sh
. "$SRC/scripts/lib/build-dir-note.sh"
note_build_dir_outside_checkout "$BUILD" "$SRC" BUILD_DIR build-v100

# ccache, when installed. cmake/CompilerCache.cmake sets the compiler launchers itself, with the
# settings that let worktrees share entries (CCACHE_BASEDIR, CCACHE_NOHASHDIR, and ignoring the
# per-tree -Xcompiler=-ffile-prefix-map option). Passing a bare `ccache` launcher here bypassed
# those, so every worktree's CUDA objects missed. A tree configured by the old script keeps the
# cached bare launcher; the -U options below clear it.
have_ccache=0
if command -v ccache >/dev/null; then
  have_ccache=1
  # nvcc output is only cacheable when the compiler hash is stable and sloppiness covers
  # the timestamps nvcc embeds in temporary file names.
  export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-time_macros,include_file_mtime,include_file_ctime}"
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
else
  echo "ccache not found: building without it (scripts/setup-ccache.sh installs it)." >&2
fi

testing=OFF
[ "${NINFER_BUILD_TESTS:-0}" = "1" ] && testing=ON

# The benchmarks are off by default, so a fresh tree has no ninfer_bench target. Only turn them ON here,
# never OFF: a tree configured with them keeps them.
bench_flag=()
if [ "${NINFER_BUILD_BENCHMARKS:-0}" = "1" ]; then bench_flag=(-DNINFER_BUILD_BENCHMARKS=ON); fi
for target in "$@"; do
  if [ "$target" = ninfer_bench ]; then bench_flag=(-DNINFER_BUILD_BENCHMARKS=ON); fi
done

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -U CMAKE_C_COMPILER_LAUNCHER -U CMAKE_CXX_COMPILER_LAUNCHER -U CMAKE_CUDA_COMPILER_LAUNCHER \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING="$testing" \
  ${bench_flag[@]+"${bench_flag[@]}"}

if [ -z "${JOBS:-}" ]; then
  ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  JOBS=$(( ram_gb / 2 )); [ "$JOBS" -lt 2 ] && JOBS=2
  [ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
fi

if [ "$#" -eq 0 ]; then set -- ninfer ninfer-serve; fi
ninja -C "$BUILD" -j "$JOBS" "$@"
if [ "$have_ccache" = 1 ]; then ccache -s | head -12; fi
