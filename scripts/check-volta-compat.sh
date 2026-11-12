#!/usr/bin/env bash
# Keeps the Tesla V100 (sm_70) build from rotting. Volta has no cp.async, ldmatrix, m16n8kK mma.sync,
# __reduce_*_sync or tanh.approx, and the nvcc cross-compile that proves a kernel is clean takes
# 20 minutes, so this catches the usual way it breaks in seconds: a kernel that writes the Ampere
# instruction as inline asm (or calls the intrinsic) instead of going through the helpers that have
# a Volta form (docs/maintainer/volta-port.md):
#
#   asm "mma.sync..."        -> ops/common/mma.cuh       (mma_bf16, mma_f16, mma_s8, mma_s4_m16n8k64, ...)
#   asm "cp.async..."        -> ops/common/memory.cuh    (cp_async, cp_async_mem, cp_commit_mem, ...)
#   asm "ldmatrix..."        -> ops/common/mma.cuh       (ldmatrix_x2, ldmatrix_x4, and the _t forms)
#   __reduce_{max,...}_sync  -> ops/common/warp.cuh      (masked_reduce_max, ...)
#   asm "tanh.approx..."     -> ops/common/math.cuh      (sigmoid_approx)
#
# The helpers themselves and the TMA routes (Hopper and Blackwell only; the build never instantiates
# them for sm_8x or sm_70) are the only places allowed to name the instructions.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

allowed='^src/ops/common/(mma|memory|volta_compat|warp|math)\.cuh:|^src/ops/linear/(fp8/fp8_a8_tma_mma|bf16/bf16_a16_tma_mma|nvfp4/nvfp4_a4_tma)\.cuh:|^src/ops/common/mbarrier\.cuh:'
instruction='"(mma\.sync\.aligned|cp\.async\.(ca|cg)\.shared|cp\.async\.(commit_group|wait_group)|ldmatrix\.sync|tanh\.approx\.f32)|[^_a-z]__reduce_(max|min|add|or|and|xor)_sync\('

hits="$(grep -rnE "$instruction" src --include='*.cu' --include='*.cuh' --include='*.h' | grep -vE "$allowed" || true)"

if [ -n "$hits" ]; then
  echo "Ampere-only instructions outside the helpers that emulate them on sm_70:" >&2
  printf '%s\n' "$hits" >&2
  echo >&2
  echo "Use the helper named in the header comment of scripts/check-volta-compat.sh." >&2
  exit 1
fi
echo "volta-compat: no raw Ampere-only instructions outside the shared helpers"
