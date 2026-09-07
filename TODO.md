# TODO

State as of 2026-09-07, branch `sync/neroued-catchup-20260906` (96 commits ahead of `master`:
86 from upstream, 10 ours). Full suite is **121/121 passing**, tree builds clean on sm_86, DFlash2
works on the real 27B including with `--vision`.

Everything below is what is *not* done. Ordered by what blocks what.

---

## 1. Blocking — the work isn't shared yet

- [ ] **Push the branch.** `sync/neroued-catchup-20260906` has never been pushed; there is no
      remote tracking branch and no PR. 96 commits currently exist only on this machine.
- [ ] **Open the PR.** Use `gh pr create --repo ashalliants/ninfer-3090 --base master` — `gh`
      picks the fork parent as base otherwise and fails misleadingly.
- [ ] **PR #15 (multi-GPU expert offload) is still open** on `feat/dual-gpu-graph-mode`. Decide
      whether it lands before or after this catch-up; they touch different subsystems but both
      touch the engine layer.
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

- [ ] **q4 SwiGLU `Materialized` boundaries are unmeasured on sm_86**: routes `{49,128}`,
      `{257,384}`, `{513,640}`. `bench/ops/q4_linear_swiglu_schedule_bench.cu` deliberately excludes
      Materialized because it needs a workspace and has a different launch signature. Extending the
      bench to cover it would close the last unmeasured boundaries in that table.
- [ ] **Two q4_q5 schedules never win at any measured width**: `grouped_r32_c64_s4` (also reachable
      as `PairR32C64S4`) and `pair_r32_c64_s3`. They still occupy enum entries and switch cases.
      Either find the shape where they win, or delete them — dead schedules are what let the
      fallthrough bug hide.
- [ ] **Audit other Ops for the dead-table pattern.** Only `attn_input_proj/q4_q5` and
      `linear_swiglu/q4` were checked. In q4_q5 the tuned `kRoutes` array was dead code beside a
      live hardcoded if-chain carrying upstream's sm_120 boundaries — nothing read the table except
      a `static_assert`. Grep every `*_resolve_plan` and confirm it iterates its table. See
      `ninfer-3090-route-tables-can-be-dead-code` in memory.
- [ ] **The causal small-T subsystem is reverted to this fork's pre-merge version** (12 files under
      `src/ops/softmax_attention/dense/causal_cache/`). Upstream's rework of it produces
      garbage-magnitude output for the whole int8 family on sm_86 — both geometries, T=1..16,
      int8-g64 and rk8v4 alike, roughly 3-8x the reference with a varying ratio, which is what a
      softmax denominator taken over the wrong split count looks like. Root-causing that would
      recover upstream's improvements instead of carrying a revert forward into every future merge.

## 5. Measurement debt

- [ ] **DFlash2 corpus numbers on sm_86.** Only single-prompt smoke numbers are recorded (text
      20.0% / 2.38 tok-per-round, vision 85.7% / 7.00). `docs/performance.md` deliberately does not
      reproduce upstream's tables because they are sm_120.
- [ ] **Speculative decoding is not bit-identical to greedy**, and it is not clear that it should
      be. DFlash2 and MTP produce byte-identical output *to each other* and both diverge from the
      width-1 greedy path about a hundred tokens into the text fixture. Verification evaluates k+1
      columns in one pass where plain decode evaluates one, so reductions run in a different order
      and a near-tie argmax flips. MTP reproduces it exactly, so it predates this merge — but
      nobody has decided whether that is acceptable or worth pinning down.

## 6. Hygiene

- [ ] **`plane_types.h` wiring is partial.** Two files derive their KV plane cast types from the
      profile (`kv_cache/append/launch.cu`, `context_kv_materialize/materialize.cu`); there are
      ~76 KV plane cast sites across 12 files. The Op-level dtype validation is the real guard and
      that is now correct everywhere, but extending the aliases would make a wrong cast impossible
      rather than merely detected.
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
