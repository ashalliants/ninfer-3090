# TODO

State as of 2026-09-07, branch `sync/neroued-catchup-20260906`, open as **PR #16**. Full suite is
**121/121 passing**, tree builds clean on sm_86, DFlash2 works on the real 27B including with
`--vision`.

Since the PR was opened, all three route tables the merge changed have been re-measured on sm_86
and retuned (12-41% faster at the widths that moved), a second switch fallthrough was found and
fixed, and every KV plane cast now derives from `d256_kv_cache_profile`.

Everything below is what is *not* done. Ordered by what blocks what.

---

## 1. Blocking — the work isn't shared yet

- [x] ~~Push the branch.~~ Pushed.
- [x] ~~Open the PR.~~ **PR #16** — "Upstream catch-up: DFlash2, measured sm_86 route boundaries,
      one KV plane declaration". Route retunes have landed on the branch since it was opened.
- [x] ~~Decide whether PR #15 lands before or after this catch-up.~~ **PR #15 goes first — it is
      safe.** Trial-merged with `git merge-tree --write-tree`: 25 files are touched by both branches
      but only **four conflict**, and none of them is in this branch's route tables, KV plane
      typing, or the causal small-T revert. All seven `causal_cache/*.cu` files auto-merge cleanly.
      Once #15 is on `master`, this branch merges `master` and resolves:
      - `apps/cli/options.cpp` (1 hunk, 7 lines) — usage string. Take the union:
        `[--device N] [--devices N,M]` *and* `--spec mtp|dflash|dflash2`.
      - `src/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu` (1 hunk, 12 lines) — take **#15's**
        version. It opts into >48 KB dynamic shared memory at runtime where ours only had a
        `static_assert`. Confirm the sm_86 schedule stays under the ~99 KB opt-in ceiling.
      - `src/targets/qwen3_6_27b/impl/load/bindings.cpp` (3 hunks, 36 lines) — take **#15's**: it
        reorders `bind_weight`'s defaults to `(evict_rank, placement)` and threads `core_placement`.
        All three hunks are the same change; apply it at every call site consistently.
      - `tests/targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp` (1 hunk, 18 lines) — take **#15's**
        richer failure message, but keep whichever label matches the enclosing block (the two sides
        name different branches of the same test).
- [ ] PR #12 is a `DO NOT MERGE` draft recording the prefill/decode overlap negative result. Close
      it or leave it as the record — it should not merge either way.

## 2. Correctness and coverage gaps

- [ ] **Port upstream's DFlash2 attention sweep.** `run_softmax_attention_dflash2_tests` currently
      returns 77 with a message saying it is not ported. Driving upstream's `run_dflash2_cases`
      through this fork's causal-cache fixture **segfaults** — it uses shapes the fixture does not
      model, and because it is a memory fault rather than a throw, no try/catch reports it. Each
      shape needs validating against this fixture's geometry and storage overloads first.
      *Note the trap: rk8v4 is only constructible through the `CachePlan` overload of `make_cache`;
      the `KvCacheStorage` overload has no packed-int4 branch and silently builds an unpacked INT8
      value plane that the kernel then reads as packed and runs off the end of.*
- [ ] **`ninfer_qwen3_8_27b_dflash2_real_test` cannot run on a 24 GB card.** With
      `NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS` set it gets past the skip and then fails: *"requested
      Engine runtime reservation requires 6320667392 bytes, but only 4785954304 bytes are available
      for runtime capacity"*. The test's reservation is fixed. Either parameterise it or document
      it as a >24 GB test.
- [ ] **Six other real-model tests skip** for want of artifacts/env vars: `27b_prefix_real`,
      `27b_score_real`, `27b_load_plan`, `35b_a3b_real`, `35b_a3b_dflash_real`,
      `35b_a3b_dflash_load_plan`. See `ninfer-3090-35b-real-tests-environmental` in memory — some
      of these are known to fail environmentally on this host rather than because of a defect.

## 3. Combinations asked for but never exercised

- [ ] **`--vision-residency overlay` + DFlash2.** Planned in the test matrix, never actually run.
      The overlay path borrows device memory per image from the evictable text-weight tail, and
      DFlash2 binds its own weight bundle, so the interaction with the eviction ladder is exactly
      the sort of thing that will not show up until someone tries it.
- [ ] **DFlash2 + multi-GPU expert offload.** This host has one 3090; the offload path degenerates
      to the single-rank identity mapping, so nothing meaningful was tested. Needs the two-card
      box (vast) or a second local GPU.
- [ ] **`--spec dflash` (v1) on 35B-A3B.** Refused by the 27B artifact ("selected masked draft
      backend is not supported by this target") because v1 is a 35B backend — correct behaviour,
      but it means v1 itself is untested this cycle. Needs the 35B DFlash artifact.

## 4. Performance work left on the table

**Every route table the catch-up merge changed has now been measured on sm_86 and retuned.**
There were exactly three (`git diff master...HEAD` over the route arrays): W8 attention-input
DFlash2, W8 SwiGLU DFlash2, and W8 linear-pair k=5120. All three were wrong here, by 12-41%.
The other seven tables the merge left alone. The general lesson is in
`ninfer-3090-route-tables-can-be-dead-code`: **after a catch-up, diff the route arrays first and
re-measure every one that moved** — a table is tuning for the GPU it was tuned on, and nothing
about it fails on a different one.

- [ ] **Three schedules are now routed at no width**: `DFlash2MmaR16C64K128` (W8 attention input,
      never won anywhere), `DFlash2MmaR32C64K128` (W8 SwiGLU, ties `R64C64K128` at 33..44), and
      `MmaResidualR64C32` (Q5 linear-add). All are deliberately kept — deleting an upstream
      schedule costs merge effort for no measured gain — but a schedule nothing selects is what let
      both switch fallthroughs hide. Consider a test that *lists* unrouted schedules per Op, so the
      set is visible and deliberate rather than accidental.
- [ ] **`w8_pair` k=2048 table (37 routes) is unmeasured on sm_86.** The merge did not touch it, so
      it is not a new regression, but it is the largest route table in the tree and sits on the 35B
      DFlash path. `bench/ops/w8_pair_schedule_bench.cu` is the starting point but this is **more
      than a second `sweep_for_k(2048, ...)` call**, which is what a first look suggested:
      - `w8_pair_execute_schedule` calls `require_dflash_row_views` whenever `k == 2048`, so the two
        weights cannot be standalone 1024-row matrices as they are for k=5120. They must be row
        views into a 6144-row parent taken at rows 4096 and 5120. `PairFixture` in
        `tests/ops/linear_pair/linear_pair_test_common.cpp` already builds exactly that and is the
        thing to copy.
      - The schedule list needs the split-K and concat families, and those are the ones that assume
        the k=2048 geometry. A kernel run outside the shape it was written for faults rather than
        throwing, and the sweep driver can only catch throws — add them a few at a time.
- [ ] **q4 SwiGLU `Materialized` boundaries are unmeasured on sm_86**: routes `{49,128}`,
      `{257,384}`, `{513,640}`. `bench/ops/q4_linear_swiglu_schedule_bench.cu` deliberately excludes
      Materialized because it needs a workspace and has a different launch signature. Extending the
      bench to cover it would close the last unmeasured boundaries in that table.
- [ ] **Two q4_q5 schedules never win at any measured width**: `grouped_r32_c64_s4` (also reachable
      as `PairR32C64S4`) and `pair_r32_c64_s3`. They still occupy enum entries and switch cases.
      Either find the shape where they win, or delete them — dead schedules are what let the
      fallthrough bug hide.
- [x] ~~Audit other Ops for the dead-table pattern.~~ Done: all ten `*_resolve_plan` files were
      checked and only `attn_input_proj/q4_q5` had a table nothing read. Both GDN resolvers, both
      W8 attention/SwiGLU resolvers and `w8_pair` iterate theirs. The two switch fallthroughs found
      along the way are fixed and `-Werror=implicit-fallthrough` now guards non-MSVC builds
      (MSVC has no equivalent warning, so CI carries the guard for this host).
- [ ] **The causal small-T subsystem is still reverted** (12 files under
      `src/ops/softmax_attention/dense/causal_cache/`), but the INT8-family corruption is now
      **root-caused and fixed** on branch `investigate/small-t-upstream` (commit `c044640b`).

      The revert commit guessed at "a softmax denominator over the wrong split count". That was
      wrong. Upstream's `small_t_i8.cuh` stages V for an **f16** PV MMA — `dst` is `__half*`, read
      by `ldmatrix` and consumed by `mma_f16` — but fills it using
      `kv_cache_int8_dequant_i8x8_from` / `_int4_dequant_i4x8_from`, both of which pack with
      `pack_bf16x2`. Both types are sixteen bits, so it compiles, runs, and reinterprets every
      value. Both affected storages (int8-g64 and rk8v4) go through exactly those two helpers,
      which is precisely why the symptom was "the INT8 family" and nothing else. It reproduced at
      3.1x/4.0x/7.9x/11.1x with a varying ratio — **including at `keys=1`**, where there is one
      split, one key and no denominator arithmetic at all, which is what disproves the split
      theory.

      The rest of the tree already had the convention right and unambiguous: every kernel staging V
      for `mma_f16` uses an explicit `_f16x8`/`_f16x16` loader, and `_bf16x8` appears only on the K
      tile feeding `mma_bf16`. `prompt_i8.cuh` even carries its own local
      `causal_prompt_i8_dequant_f16x8`, because the shared codec had no f16 variant for the INT8
      codings — that gap is what let the wrong helper look right. The fix adds
      `kv_cache_int8_dequant_f16x8_from` / `kv_cache_int4_dequant_f16x8_from` beside their bf16
      counterparts. `ninfer_softmax_attention_test` then reports **zero criterion failures across
      179 cases**.

      Two things still block adopting upstream's version and dropping the revert:
      - [ ] **A separate segfault** on the last case, `rk8v4 T=1 keys=16385` (`run_a1_case`,
            `causal_cache.cpp:3070`). It is a host SIGSEGV, it survives the dequant fix, and the
            equivalent FP8 case at the same shape (line 3027) passes. `compute-sanitizer memcheck`
            is the tool; host and device split counts were checked and do agree for that shape, so
            it is not the split-count mismatch either.
      - [ ] **Benchmark upstream's small-T against this fork's on sm_86.** Upstream's rework also
            extends small-T to T=8 for QHeads=24 and adds a batch-size grid clamp. The revert is
            only worth undoing if the rework is actually faster here — it was tuned on sm_120, and
            every other thing tuned there has been wrong on this card.
      - [ ] Once adopted, dedupe `prompt_i8.cuh`'s local f16 dequant helpers against the shared
            codec ones.

## 5. Test-criterion calibration

- [ ] **Audit `gross_relative_to_max_reference` across the other Op tests.** The W8 linear-pair
      A16 criterion was set at 3.8e-3, which is *below the floor its own output dtype can
      represent*: BF16 has seven stored mantissa bits, so one ULP is 3.9e-3 to 7.8e-3 of the value
      and correct rounding alone costs up to half of that. The bound therefore required the single
      worst element in the tensor to round the way the FP32 oracle does. Over 330 sampled cases the
      distribution was bimodal — bulk at 0.33-0.87 of the limit, then two outliers at 0.9997 and
      1.0059 — so it was a coin flip, and the 0.9997 sample predates the route re-measurement.
      Raised to 4.5e-3 with the reasoning recorded at the constant. **Any other reduction criterion
      whose gross limit is under ~4e-3 against a BF16 output has the same latent flake**, and it
      will surface as "your kernel change broke accuracy" the next time a route boundary moves.
      `NINFER_OP_REPORT_STATS=1` prints `gross_ratio` per case, which is how to check cheaply.
      The relative-L2 field is the bound that actually constrains a kernel and should not be
      touched — those 330 cases all sit at 0.45-0.69 of it.

## 6. Measurement debt

- [ ] **DFlash2 corpus numbers on sm_86.** Only single-prompt smoke numbers are recorded (text
      20.0% / 2.38 tok-per-round, vision 85.7% / 7.00). `docs/performance.md` deliberately does not
      reproduce upstream's tables because they are sm_120.
- [ ] **Speculative decoding is not bit-identical to greedy**, and it is not clear that it should
      be. DFlash2 and MTP produce byte-identical output *to each other* and both diverge from the
      width-1 greedy path about a hundred tokens into the text fixture. Verification evaluates k+1
      columns in one pass where plain decode evaluates one, so reductions run in a different order
      and a near-tie argmax flips. MTP reproduces it exactly, so it predates this merge — but
      nobody has decided whether that is acceptable or worth pinning down.

## 7. Hygiene

- [x] ~~`plane_types.h` wiring is partial.~~ Extended to every plane-cast site that should have it,
      and the earlier "~76 sites across 12 files" estimate was wrong: most `__half*`/`__nv_bfloat16*`
      appearances under `softmax_attention/` are shared-memory arena partitioning or BF16 activation
      inputs, not cache planes. The actual set is **ten** sites in eight files — the producer
      (`kv_cache/append/{launch,k8v4_launch,nvfp4_launch}.cu`, `context_kv_materialize`) and the
      consumer (`causal_cache/{prompt,small_t}_{fp8,k8v4,nvfp4}.cu`) halves of each storage — and
      both halves now derive from `d256_kv_cache_profile`, so a producer/consumer disagreement is a
      compile error. `assert_kv_scale_planes` / `assert_kv_planes` were added to cover the scale
      planes, which needed it more than the code planes: across the six storages the value scale is
      FP16, a raw E4M3 byte, or FP16-again, and `Fp8KeyNvfp4Value` mixes two within one cache.
      **One deliberate exception**, commented at the site: the INT8 family in `causal_cache/prompt.cu`
      and `small_t.cu` keeps `std::int8_t*` for both codings because a single kernel serves int8-g64
      and rk8v4 and the packed-int4 path re-casts internally where it unpacks. Substituting
      `KvValueCodeT<...>` there would change behaviour, not tidy it.
- [ ] **Clean up the extra worktrees**: `C:/ninfer-fork/baseline-master` (created to get a
      pre-merge baseline; its purpose is served) and `C:/ninfer-fork/wt-readme`.
- [ ] **Untracked clutter in the repo root**: `config.bat`, `config_exit.txt`, `repro/`,
      `scripts/download-ornith-1.5-35b-a3b.bat`. Decide keep-and-commit or delete.
      `build_merge.bat` *is* committed and carries the toolchain pinning this host needs.

---

## Build note, because it cost hours

Never run two builds against `build-ninja` at once. Concurrent ninja instances produce
"Permission denied" on object files, `cmake --build` reporting success with the executable never
relinked, and phantom hangs. **Always compare the test binary's mtime against the sources before
believing a test result.** And a build that looks hung is almost always just slow: `nvcc` idles at
~0.1s CPU while its `cicc` child does the work, and `small_t_fp8.cu` legitimately burns 170+
seconds in `cicc`. Check `cicc`, not `nvcc`. Full details in the
`ninfer-3090-windows-build-recipe` memory.
