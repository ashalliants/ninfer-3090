#!/usr/bin/env bash
# First run on a Tesla V100 (sm_70): is the Volta port right, and how fast is it?
#
# Nothing in the port could be run when it was written (it was cross-compiled with nvcc and ptxas
# only), so this is the check that turns "compiles" into "works". It runs on the V100 box, from a
# checkout of the v100-port branch, and stops at the first stage that fails:
#
#   1. device     nvidia-smi reports compute capability 7.0
#   2. emulation  the standalone numerical test of every emulated instruction (seconds)
#   3. build      scripts/v100_build.sh ninfer (everything, tests included, with --tests)
#   4. tests      the op correctness suites against their FP32/FP64 oracles (--tests)
#   5. models     greedy generation per model, so the output can be read, with decode and prefill
#                 speed (give a model with MODEL_27B, MODEL_BONSAI, MODEL_35B, MODEL_FLASH)
#
#   bash scripts/v100_validate.sh --tests
#   MODEL_27B=/workspace/models/qwen3_8_27b.ninfer bash scripts/v100_validate.sh --models
#
# Output goes to $OUT (default /workspace/v100-out): one log per stage and summary.txt. Copy it
# off the box before destroying it; the instance is the only copy.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-/workspace/v100-out}"
BUILD="${BUILD_DIR:-/workspace/build-v100}"
mkdir -p "$OUT"
RUN_TESTS=0
RUN_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --tests) RUN_TESTS=1 ;;
    --models) RUN_MODELS=1 ;;
    *) echo "unknown argument: $arg (expected --tests or --models)" >&2; exit 2 ;;
  esac
done
FAILED=()
note() { echo "$*" | tee -a "$OUT/summary.txt"; }

note "=== v100 validate $(date -u +%FT%TZ)  commit=$(git -C "$SRC" rev-parse --short HEAD) ==="

# 1. device
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
name="$(nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader 2>/dev/null | head -1)"
note "gpu: $name (compute capability ${cc:-unknown})"
if [ "$cc" != "7.0" ]; then note "FAILED: this script is for a Tesla V100 (7.0)"; exit 1; fi

# 2. the emulation test, built on its own so it runs before the long build
note "--- emulation test ---"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
"$NVCC" -arch=sm_70 -std=c++20 -O2 -Wno-deprecated-gpu-targets -I "$SRC/src" \
  "$SRC/tests/ops/test_volta_emulation.cu" -o "$OUT/volta_emulation_test" > "$OUT/emulation-build.log" 2>&1 \
  || { note "FAILED: emulation test did not build (see $OUT/emulation-build.log)"; exit 1; }
"$OUT/volta_emulation_test" | tee "$OUT/emulation.log" | tee -a "$OUT/summary.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { note "FAILED: an emulated instruction is wrong; every kernel result below is suspect"; exit 1; }

# 3. build
note "--- build ---"
targets=(ninfer)
# CTest also registers standalone test executables beside the ninfer_tests bundle, so build everything.
if [ "$RUN_TESTS" -eq 1 ]; then export NINFER_BUILD_TESTS=1; targets=(all); fi
BUILD_DIR="$BUILD" bash "$SRC/scripts/v100_build.sh" "${targets[@]}" > "$OUT/build.log" 2>&1 \
  || { note "FAILED: build (see $OUT/build.log)"; tail -20 "$OUT/build.log"; exit 1; }
note "built"

# 4. the op suites. They need no model. -j1: one process on the card at a time.
if [ "$RUN_TESTS" -eq 1 ]; then
  note "--- tests ---"
  ctest --test-dir "$BUILD" -j1 --output-on-failure -E 'softmax_attention|kv_materialize' \
    > "$OUT/tests.log" 2>&1
  rc=$?
  tail -15 "$OUT/tests.log" | tee -a "$OUT/summary.txt"
  if [ "$rc" -ne 0 ]; then FAILED+=("ctest (see $OUT/tests.log)"); fi
  note "(the two slowest suites, softmax_attention and kv_materialize, are skipped here: they are the"
  note " ones that exercise the emulated mma hardest. Run them next: ctest -R 'softmax_attention|kv_materialize')"
fi

# 5. models
if [ "$RUN_MODELS" -eq 1 ]; then
  note "--- models ---"
  prompt="List the first eight prime numbers, then explain briefly why 1 is not prime."
  ran=0
  for entry in "27b:${MODEL_27B:-}" "bonsai:${MODEL_BONSAI:-}" "35b-a3b:${MODEL_35B:-}" "flash-next:${MODEL_FLASH:-}"; do
    label="${entry%%:*}"; model="${entry#*:}"
    [ -n "$model" ] || continue
    ran=1
    note "[$label] $model"
    "$BUILD/apps/ninfer" "$model" --prompt "$prompt" --max-new 200 --greedy --kv-dtype int8 \
      --no-thinking --max-context 8192 --kv-capacity 8192 --device 0 \
      > "$OUT/$label-output.txt" 2> "$OUT/$label-stderr.txt"
    rc=$?
    note "  exit $rc"
    grep -E "gpu weights used|free after startup|decode speed|prefill speed|error" "$OUT/$label-stderr.txt" | head | tee -a "$OUT/summary.txt"
    head -c 600 "$OUT/$label-output.txt" | tee -a "$OUT/summary.txt"; echo | tee -a "$OUT/summary.txt"
    if [ "$rc" -ne 0 ]; then FAILED+=("$label generation (exit $rc)"); fi
  done
  [ "$ran" -eq 1 ] || note "no model given: set MODEL_27B, MODEL_BONSAI, MODEL_35B or MODEL_FLASH"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  note "=== FAILED: ${#FAILED[@]} stage(s) ==="
  printf '  - %s\n' "${FAILED[@]}" | tee -a "$OUT/summary.txt"
  exit 1
fi
note "=== DONE. Read the generated text above: sensible text means the kernels are right. Results are in $OUT ==="
