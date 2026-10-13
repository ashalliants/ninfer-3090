# Splitting heavy CUDA translation units

Temporary plan (2026-10-13). Delete it when the series lands or is abandoned.

Goal: bound the compile cost of any single CUDA translation unit (TU), so a cold build parallelises
across all cores instead of waiting on one single-threaded `cicc` + `ptxas` run, and so ccache hits
at a useful granularity. Every PR in this series is a pure refactor: no behaviour, dispatch or
numerics change.

## Why

[Build system](build-system.md) already says a full build is "10+ minutes, dominated by a few
multi-minute CUDA translation units". ccache is per TU and all-or-nothing, so a TU that instantiates
a large template cross-product is both the critical path of a cold build and a full recompile on any
edit to it or to a header it includes.

The motivating case was a Tesla V100 (sm_70) port on an older base, where a cold build took hours of
summed CPU time and single `ptxas` runs exceeded 30 minutes. `causal_cache/small_t.cu` alone emitted
about 2.4M lines of PTX for sm_86 there. **Those timings are from the old base and are not
re-measured on current master**, which is why every PR in this series reports a master-vs-branch
measurement (see "Testing a split PR").

## What is established

- Current master already uses thin per-shape and per-dtype TUs (`linear/*/shapes/*.cu`,
  `softmax_attention/.../small_t_{fp8,nvfp4,k8v4}.cu`, `prompt_*.cu`). Extend that pattern.
- Source lists are explicit manifests (`sources.cmake`); no globbing, no generated registries, no
  unity builds (`build-system.md`). `ninfer_ops` is RDC: a split must keep language, arch, RDC mode,
  device-link ownership and numerical options identical.
- Source or header size does **not** predict cost (a static header-closure scan is flat: the top 25
  of 257 ops TUs are 16% of the total). Template **instantiation count** does.
- The `linear/*/shapes/*.cu` TUs are already thin (about 11-17 launch sites each). Leave them.

## Ranking by estimated kernel instantiations (read from dispatch code; not compiler output)

| TU | Est. instantiations | Notes |
|---|---|---|
| `softmax_attention/dense/causal_cache/small_t.cu` | ~784 | INT8-family small-T; 839-line kernel; **PR 1** |
| `small_t_{fp8,nvfp4,k8v4}.cu` | ~128 each | same shape as above, 635-670-line kernels |
| `bf16/launch.cu`, `dense/context/launch.cu`, `sliding_window/launch.cu` | ~64-68 | |
| `prompt_i8_fa2.cu` | ~60 | 840-line FA2 kernel x metadata x packed x config |
| `linear_add/q8/q8_linear_add_gemm_mma.cu` | ~33 | one schedule per launcher |
| `sparse_moe/{decode,prefill}/*_kernels.cu` | ~25-30 each | 871 / 1399 lines |
| `linear_topk/{q4,q8,fp8}_m64.cu` | ~12-25 each | |

## Design: how to split one TU

Follow the thin-slice precedent (`linear/bf16/shapes/*.cu` export a function pointer, a `.cpp`
dispatch TU holds the table).

1. Pick split axes from the cross-product that already select a distinct kernel template, so each
   slice is similar in cost (for `small_t.cu`: head geometry x query-row tile).
2. Move the slice's instantiations into its own `.cu` exporting one plain function, declared in a
   private header that does **not** include the kernel headers.
3. Keep policy (route/split decisions, host-only code) and any cheap shared kernels in the dispatch
   TU, which calls slices through function-pointer tables and never instantiates the heavy kernels.
4. Move code verbatim. Do not change launch bounds, tile constants, template arguments or dispatch
   order. Each kernel template instance must be instantiated in exactly one TU, and its
   `cudaFuncSetAttribute` / `configure_cuda_device_once` call moves with it.

## Testing a split PR (do this for every PR in the series)

**Always compare against master, with the compiler cache OFF, on a cold build.** A warm ccache hides
exactly the cost this series removes. On the CUDA machine (same CUDA toolkit, same arch, same `-j`):

```
# 1) baseline: master
git switch master && git pull
cmake -S . -B build-master -G Ninja -DNINFER_COMPILER_CACHE=OFF   # plus your usual flags
time cmake --build build-master -j
python3 -I scripts/ninja_tu_times.py build-master --top 20

# 2) the PR branch, fresh build dir, cache still OFF
git switch <pr-branch>
cmake -S . -B build-branch -G Ninja -DNINFER_COMPILER_CACHE=OFF
time cmake --build build-branch -j
python3 -I scripts/ninja_tu_times.py build-branch --top 20

# 3) prove nothing changed: same kernels, same register/shared usage
scripts/compare_kernel_symbols.sh build-master build-branch

# 4) incremental cost: touch one slice, rebuild with the cache ON in a warm tree
touch <one slice .cu> && time cmake --build build-branch -j
```

Record, for master and branch: total cold wall time, the longest single TU (from
`ninja_tu_times.py`), and the "touch one slice" time. Then run the affected tests/model smoke run.
Wall times inflate under contention; for a clean single-TU number run
`ninja -C <dir> <object>` alone.

## PR plan (stand-alone PRs, not stacked)

1. `small_t.cu` -> ten slice TUs + dispatch (this branch). Adds `scripts/ninja_tu_times.py` and
   `scripts/compare_kernel_symbols.sh`.
2. `small_t_{fp8,nvfp4,k8v4}.cu`: same pattern, slice by (geometry, tile).
3. Then only the TUs the measurements above still show as the long pole (`prompt_i8_fa2.cu`,
   `bf16/launch.cu`, context / sliding-window, `q8_linear_add_gemm_mma.cu`, sparse-MoE).

Stop when the longest TU is within about 2x of the median heavy TU.

## Out of scope

- Dropping instantiated variants or a geometry/dtype on current targets (a behaviour decision).
- Whole-program (non-RDC) compilation: measured neutral for speed in `build-system.md`.
- Making the Volta (sm_70) port build faster by compiling fewer variants (its own change).
