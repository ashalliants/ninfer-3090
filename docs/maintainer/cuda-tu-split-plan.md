# Splitting heavy CUDA translation units

Temporary plan (2026-10-13). Delete it when the work lands or is abandoned.

Goal: bound the compile cost of any single CUDA translation unit (TU), so a cold build parallelises
across all cores instead of waiting on one single-threaded `cicc` + `ptxas` run, and so ccache hits
at a useful granularity. This is a pure refactor: no behaviour, dispatch or numerics change.

## Why

[Build system](build-system.md) already says a full build is "10+ minutes, dominated by a few
multi-minute CUDA translation units", and that kernel instantiation work, not header parsing, is
the cost. ccache is per TU and all-or-nothing, so one TU that instantiates a large template
cross-product is both the critical path of a cold build and a full recompile on any edit to it or to
a header it includes.

The motivating case came from porting to a Tesla V100 (sm_70) on a stale base (see the V100 draft PR,
`ashalliants/ninfer-3090#300`): there, a cold `ninfer_ops` build took hours of summed CPU time, 24 TUs
took more than 10 minutes each (the worst about an hour), and several single `ptxas` runs exceeded
30 minutes while 23 cores sat idle. A related measurement: `causal_cache/small_t.cu` on the old
base emitted about **2.4M lines of PTX for sm_86**, 3 min 15 s in `cicc` alone, before `ptxas`.
That base predates the shape-TU work now on master, so **re-measure on current master before
believing any number above** (step 0).

## What is established (read this first)

- Current master already splits some families into per-shape and per-dtype TUs:
  `linear/{fp8,q4,...}/shapes/n*_k*.cu`, `softmax_attention/dense/causal_cache/{small_t,small_t_fp8,
  small_t_nvfp4,small_t_k8v4,prompt,prompt_i8_fa2,prompt_fp8,...}.cu`. **Extend that pattern; do not
  invent a new one.**
- Source lists are explicit manifests (`sources.cmake`); no globbing, no generated registries.
- `build-system.md` forbids unity builds and says splitting a manifest does not reduce instantiation
  work. Splitting a *TU* does, but only if the instantiations actually move.
- `ninfer_ops` is RDC. A split must keep language, arch, RDC mode, device-link ownership and
  numerical compile options identical (`build-system.md`, "CUDA compilation boundaries").
- A static header-closure scan of the 257 ops TUs on master (sum of project-header line counts per
  TU) is **flat**: the top 25 TUs are only 16% of the total and none stands out. Source size does
  not predict cost; **template instantiation count does**. Do not rank by line count.
- Header dependency depth is not the problem (deepest TU pulls about 22 project headers). The shared
  leaf headers (`ops/common/memory.cuh`, `math.cuh`, `mma.cuh`) are reached by about 112 of 131 ops
  TUs on the old base, so editing one rebuilds most of `ninfer_ops` by design. Leave that alone.

## Step 0: measure (no code change; one small PR)

Cheapest honest data: a cold ccache-disabled build on a machine with nvcc (no GPU needed to compile;
any many-core CUDA box works), then:

```
python3 -I scripts/ninja_tu_times.py <build-dir> --top 40
```

(`scripts/ninja_tu_times.py` is added on this branch; it ranks objects from `.ninja_log`.) Wall times
are inflated by contention, so for the top 10 re-time each object alone:
`ninja -C <build-dir> <object>`. For the top 5 also record PTX size and kernel count:
`nvcc <same flags> -ptx` then `wc -l` and `grep -c '^.visible .entry'`. Record arch, nvcc version, core
count and the table in the PR. **The ranking decides which TUs to split; do not split on a hunch.**

## Candidate TUs (from reading source, unmeasured on current master)

Verify each against the step-0 ranking before touching it.

1. `src/ops/softmax_attention/dense/causal_cache/small_t.cu`: the BF16 and INT8 split-KV small-T
   launcher. It dispatches `TokenTile` 1..6 x `MultiBatch` x `Masked` x `CacheInput` x
   `Geometry` (27B group 6 and 35B-A3B group 8) x {BF16 kernel, INT8 tiled kernel with per-tile
   `WarpsPerCta/MinBlocksPerSm/KeyBlock/DynamicArena` variants} in one TU. All of it is reachable from
   one `switch` on `invocation.width`, which is why one file holds the whole cross-product.
2. `.../causal_cache/prompt.cu` and `int8/launch.cu`, `.../dense/context/launch.cu`,
   `.../sliding_window/launch.cu`: same pattern (many kernel templates behind a launcher).
3. The `*_mma.cu` / `*_splitk.cu` GEMM families (q4/q5/q6/w8, `linear_add`, `linear_pair`,
   `linear_swiglu`, `gdn_input_proj`, `attn_input_proj`): per-schedule instantiation tables in one TU.
   Whether they are still monolithic on current master is unknown; check the ranking.

## Design: how to split one TU

Follow the shape-TU precedent (`linear/fp8/shapes/n5120_k6144.cu`: a small TU that defines the
instances for one slice and exports one launch function).

1. Pick the split axes from the cross-product so each slice is similar in cost and independent:
   typically `(Geometry, KV dtype, TokenTile)`. Prefer axes that already select a distinct kernel
   template, because that is where instantiation weight is.
2. Move each slice's explicit instantiations and its small launch wrapper into its own `.cu`
   (`small_t_g6_i8_t5.cu` style names; keep them next to the family, listed in its `sources.cmake`).
3. Keep one thin dispatch TU (`small_t.cu` becomes this) that holds the route policy
   (`causal_small_t_split_count`, window tiers, the `switch`) and calls the slice launchers through
   plain function declarations in a private header. It must not include the kernel `.cuh` files, which
   is what makes it cheap and keeps the heavy headers out of every TU except their slice.
4. Do not change launch bounds, tile constants, template arguments or the order of dispatch checks.
   The diff of generated SASS/PTX per kernel entry should be empty (see Verification).
5. Do not add a unity build, a generated registry, recursive globs, or a second dispatch path.

Risks to check, not assume:
- `static const cudaError_t attr = cudaFuncSetAttribute(...)` function-local statics (per kernel
  instantiation, in `small_t.cu`) must stay one-per-instantiation; moving code must not duplicate an
  instantiation across two TUs (ODR and RDC link duplicates). Each kernel template instance should
  be instantiated in exactly one TU.
- RDC device link: a kernel instantiated in a slice TU must still be reachable from the dispatch TU
  through the host launcher only; no cross-TU `__device__` symbol dependencies are introduced.
- Anonymous-namespace helpers currently shared inside the big TU need a header or a duplicate.
  Prefer a small private header over duplication when they are non-trivial.

## Verification (match `build-system.md` and `op-development.md`)

- Source ownership: compare configured target/sources and compile flags before and after
  (`ninja -t commands`), including arch, RDC, `-lineinfo`, numerical options.
- Equivalence: per kernel entry, the symbol set and resource usage (`cuobjdump --dump-resource-usage`)
  must match before/after for the same arch. Any difference needs an explanation.
- Behaviour: the affected op tests (see `tests/` for the softmax-attention / linear families)
  and a short real-model smoke run on a GPU. No new tests are needed for a pure move.
- Build-time claims need measurement at the claimed scope, with a number for: cold build wall time
  with ccache off, and "edit one slice, rebuild" time. Report the critical-path TU before and after.
- AGENTS.md applies: no backward-compat shims, no unrelated cleanup, no speculative generality.

## PR plan (per `PR_POLICY.md` / the repo's PR size and stacking rules)

Each PR merges on its own and leaves main working. Not stacked.

- **PR A: measurement tooling and the data.** `scripts/ninja_tu_times.py` plus a table of the
  step-0 results in the PR description (not in the tree). Small, mergeable alone.
- **PR B: split the top-ranked TU** (expected `causal_cache/small_t.cu`). One TU family per PR so
  the equivalence check stays reviewable.
- **PR C..: next-ranked TUs**, only while the measured critical path justifies it. Stop when the cold
  build's longest TU is within roughly 2x of the median heavy TU; further splitting has diminishing
  returns and adds manifest churn.

Never more than 3 open PRs for this work at once; if a PR is conflicting, fold it into its
neighbour instead of stacking.

## Out of scope

- Reducing the number of instantiated variants, or dropping a geometry or dtype, on current targets
  (that changes behaviour; a separate performance decision).
- Whole-program compilation (non-RDC); it was measured as neutral for speed in `build-system.md`.
- Making the Volta (sm_70) port build faster by compiling fewer variants. That port may want a
  Volta-only variant cut as a stopgap, but it is its own change in `ashalliants/ninfer-3090#300`.
