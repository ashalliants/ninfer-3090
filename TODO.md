# TODO

State as of 2026-09-08. **PR #16 is approved and merged to `master`**, and released as
**v0.9.0-rtx3090** (Windows + Linux). Full suite **123/123**, plus three real-model tests passing
on an idle GPU. The branch carried the upstream catch-up, PR #15's multi-GPU expert offload, four
measured route retunes, two switch-fallthrough fixes, profile-derived KV plane typing, and the
adoption of upstream's causal small-T.

What landed, with numbers:

| change | effect |
|---|---|
| Four route tables re-measured on sm_86 | **12–41% faster** at the widths that moved |
| Upstream causal small-T adopted (12-file revert gone) | **13–25% faster** decode, int8 |
| Two switch fallthroughs fixed | each was running a second kernel over the first |
| DFlash2 unblocked on this fork | `--spec dflash2` previously died at startup |
| KV plane typing derived from the profile | producer/consumer disagreement is now a compile error |

**Eight review rounds** on PR #16, seven of eight findings genuine (one disputed with evidence and
withdrawn). The recurring lesson, worth keeping: **fix the class, not the flagged line** — chasing
the class is what found the `kv_cache_append` contract lines and the `AGENTS.md` wrong-target claim
that nobody had flagged.

Everything below is what is *not* done. Ordered by what blocks what.

---

## 0. Next up, in this order

1. **Dedupe `prompt_i8.cuh`'s local f16 dequant helpers** (section 7). ~15 minutes, and it closes
   the exact gap that caused the small-T bug: the shared codec had no f16 variant for the INT8
   codings, so upstream's kernel reached for the bf16 one. Removing the local copies removes the
   trap.
2. **Measure the `w8_pair` k=2048 table** (section 4). The best remaining pure-speed bet — 37
   routes, the largest table in the tree, on the 35B DFlash path, never measured on sm_86.
3. **Audit the gross-error limits** (section 5). Cheap insurance: any criterion under ~4e-3 against
   a BF16 output will read as an accuracy regression the next time a route boundary moves.

Then, in no fixed order: the two items descoped from PR #16 under review (sections 4a and 5a), and
the DFlash2 attention sweep (section 2), which is now narrowed to a single reproducible case.

**Keep `investigate/small-t-upstream` until the next catch-up.** It is merged, but it is also the
clean, self-contained record of how upstream's small-T was adopted and what had to be fixed to make
it work here (`19c7617c` and its parents). The next merge from `neroued/master` will touch the same
subsystem.

---

## 1. Shipping — done

- [x] ~~Push the branch, open the PR, land PR #15 first.~~ All done. PR #15 merged to `master`
      first (four mechanical conflicts, resolved), then this branch merged `master`.
- [x] ~~**PR #16**~~ — **approved and merged.** CodePulse: *"No action needed — ship it."* Eight
      review rounds; the resolutions are recorded in the commit messages, which is where to look
      if any of them needs revisiting.
- [x] ~~Cut a release.~~ **v0.9.0-rtx3090**, Windows + Linux archives with SHA256SUMS.
- [ ] PR #12 is a `DO NOT MERGE` draft recording the prefill/decode overlap negative result. Close
      it or leave it as the record — it should not merge either way. **Still the only open PR
      housekeeping item.**

## 2. Correctness and coverage gaps

- [ ] **Port upstream's DFlash2 attention sweep.** `run_softmax_attention_dflash2_tests` currently
      returns 77 with a message saying it is not ported. Driving upstream's `run_dflash2_cases`
      through this fork's causal-cache fixture **segfaults** — it uses shapes the fixture does not
      model, and because it is a memory fault rather than a throw, no try/catch reports it. Each
      shape needs validating against this fixture's geometry and storage overloads first.
      **Narrowed 2026-09-07:** it fails on the *first* case of the sweep — BF16, `width=2`,
      `batch=1`, `base=0` — not on some exotic later shape, and it fails before reaching any
      attention call (neither `causal_attention_resolve_route` nor
      `causal_softmax_attention_workspace_capacity_bytes` is entered). The symptom varies run to run
      ("vector too long", "bad allocation", "invalid naive Softmax Attention geometry", a raw access
      violation), and `compute-sanitizer` reports **0 device errors**, so it is host-side. Start
      from that one case rather than the whole sweep; `cdbX64.exe` is available for it (see
      `windows-cdb-debugger-available` in memory).
      *Note the trap: rk8v4 is only constructible through the `CachePlan` overload of `make_cache`;
      the `KvCacheStorage` overload has no packed-int4 branch and silently builds an unpacked INT8
      value plane that the kernel then reads as packed and runs off the end of.*
- [x] ~~`ninfer_qwen3_8_27b_dflash2_real_test` cannot run on a 24 GB card.~~ **It can — it just
      defaults to a maximum configuration.** The test already takes argv
      (`k graph optimized batch kv vision state_slots`) but `add_test` passes none, so it runs
      k=15, **batch=8** (18,432-token KV) and 3 state slots, wanting 6.32 GB. Run it scaled and it
      passes end to end: `ninfer_qwen3_8_27b_dflash2_real_test.exe 7 1 1 2 int8 0 1` →
      `ok K=7 B=2 graph=1 optimized=1 accepted=20/20`.
      **Now wired into ctest.** `ninfer_add_test` gained a `TEST_ARGS` option and the test is
      registered with `TEST_ARGS 7 1 1 2 int8 0 1`, so it runs that configuration by default and
      passes; the executable still accepts any other configuration when run by hand.

### Running the real-model tests on this box (verified 2026-09-07, idle GPU)

```
NINFER_QWEN3_8_27B_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b.ninfer
NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b_dflash2.ninfer
NINFER_QWEN3_6_35B_A3B_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer
NINFER_REAL_TEST_MAX_CONTEXT=8192      # only needed for the 35B
```

| test | with an idle GPU |
|---|---|
| `27b_prefix_real` | **passes** (10.4 s) |
| `27b_dflash2_real` | **passes** (23.6 s) — wired via `TEST_ARGS` |
| `35b_a3b_real` | **passes** with `NINFER_REAL_TEST_MAX_CONTEXT=8192` |
| the other four | skip — need a Qwen3.6 27B artifact, or a DFlash-carrying 35B artifact |

**Free VRAM is the whole story, and it is not marginal — it is decisive.** With the desktop
busy (~4.4 GiB free) `27b_prefix_real` skips; idle (~21.9 GiB free) it passes. For the 35B the
runtime reservation the maximum-configuration exercise asks for scales with the ceiling:
262,144 → 4.15 GB, 32,768 → 1.48 GB, 16,384 → 1.29 GB, 8,192 → fits. Available runtime capacity
on an idle box is ~1.02 GiB once the 20.8 GiB of weights are resident.

The default stays 262,144 deliberately: on a machine that can hold it, that is the configuration
worth pinning. The env var is for boxes that cannot, and the test now *skips* rather than fails
when it genuinely will not fit.

- [ ] **The real-model tests fail on their *maximum* configurations, not on this box's capability.**
      Measured 2026-09-07 via the CLI, which is the honest way to size them:
      | model | configuration | result |
      |---|---|---|
      | 27B | 131,072 ctx, int8 | **works** — 4.44 GiB reservation, 449 MiB spare |
      | 27B | 32,768 ctx, int8, `--vision` | **works** — 2.01 GiB reservation, 2.54 GiB spare |
      | 35B-A3B | 32,768 ctx, int8 | **works** — 513.7 MiB reservation, 726 MiB spare |

      So there is plenty of room for meaningful end-to-end coverage; the tests simply pin
      262,144-context/vision/batch-8 layouts. `NINFER_REAL_TEST_MAX_CONTEXT` now lowers the ceiling
      the 35B test's `exercise_maximum_configuration` asks for (unset, the pinned 256K layout is
      exercised exactly as before).
      **The 35B's remaining blocker is its *base* engine, not the maximum one:** 20.8 GiB of weights
      plus a 472 MB runtime reservation against ~21.4 GiB free leaves it **~93 MB short**, because
      the desktop is holding ~2.4 GB (Chrome, Steam, Ferdium, Docker…). Closing a couple of GPU
      clients is enough to run it. Worth deciding whether the base config should also honour
      `NINFER_REAL_TEST_MAX_CONTEXT`, or whether the test should skip with a clear message when the
      reservation genuinely will not fit rather than failing.
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
      box (vast) or a second local GPU. **More interesting now that PR #15 has landed on `master`**
      — the offload path is no longer hypothetical, and this branch has merged it.
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
- [x] ~~**The causal small-T subsystem is reverted**~~ **Done — upstream's small-T is adopted and
      the revert is gone.** Root cause of the INT8 corruption: upstream's `small_t_i8.cuh` stages V
      for an **f16** PV MMA (`dst` is `__half*`, read by `ldmatrix`, consumed by `mma_f16`) but
      filled it with `kv_cache_int8_dequant_i8x8_from` / `_int4_dequant_i4x8_from`, both of which
      pack with `pack_bf16x2`. Both types are sixteen bits, so it compiled, ran, and reinterpreted
      every value. Both affected storages (int8-g64, rk8v4) go through exactly those two helpers,
      which is why the symptom was "the INT8 family" and nothing else. Fixed by adding
      `kv_cache_int8_dequant_f16x8_from` / `kv_cache_int4_dequant_f16x8_from` next to their bf16
      counterparts — the shared codec had never had f16 variants for the INT8 codings, which is
      exactly why the wrong helper looked right (`prompt_i8.cuh` had worked around it with local
      copies; **deduping those against the shared codec is the one follow-up left**).

      The revert commit had guessed "a softmax denominator over the wrong split count". That was
      wrong, and `keys=1` disproves it: one split, one key, no denominator arithmetic, still 7.9x.

      **Measured payoff on sm_86** (int8, cached decode, cold, B=1, d256-h24-kv4, median us):

      | context | W=1 | W=2 | W=4 | W=6 |
      |---|---|---|---|---|
      | 512  | 23.6 -> 18.4 | 25.6 -> 21.5 | 29.7 -> 24.6 | 36.9 -> 27.6 |
      | 2048 | 31.7 -> 24.6 | 35.8 -> 27.6 | 43.0 -> 33.8 | 68.6 -> 52.2 |
      | 8192 | 68.6 -> 52.2 | 77.8 -> 59.4 | 95.2 -> 82.9 | 112.6 -> 84.0 |

      13-25% faster at every shape measured, on the decode path, for the default KV dtype.

      **The "second blocker" was self-inflicted and is worth remembering.** Restoring upstream's
      *test* file along with the kernels dropped two fork-only fixes (the DFlash2-sweep skip from
      `b91ee27c`, and the rk8v4 fixture fix). The result crashed with a different symptom almost
      every run — "vector too long", "bad allocation", "invalid naive Softmax Attention geometry",
      a raw access violation — which reads exactly like memory corruption in the new kernels and is
      not. Two things cut through it: `compute-sanitizer` reported **0 device errors** twice, and
      `cdbX64.exe` (from the Store WinDbg package, in `%LOCALAPPDATA%\Microsoft\WindowsApps`) showed
      the fault was a C++ exception, not a memory fault. **When porting a subsystem, take the
      kernels and keep this fork's tests.**


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

## 4a. Documents that still speak for the wrong GPU

- [ ] **Audit `docs/performance.md` provenance.** `AGENTS.md` claimed the implementation targets
      `sm_120a` on an RTX 5090 — corrected, since this fork is `sm_86`/RTX 3090 and the same file's
      own build section already said so. Chasing that turned up a bigger question this branch has
      not answered: `docs/performance.md` presents campaigns measured **on an RTX 5090 with CUDA
      13.1** (lines ~63, ~80, ~131, ~190). Some of that is certainly upstream's, and the PR
      description claims this fork "keeps this fork's sm_86 measurements" — those two statements
      cannot both be fully true.

      Deliberately **not** rewritten under review: sorting which tables are ours and which are
      inherited needs the provenance of each campaign, not a search-and-replace, and getting it
      wrong would replace one misleading claim with another. The bounded fix — label each campaign
      with the hardware it ran on, the way `qwen3.8-27b-dflash2.md` now separates 上游证据 from
      本 fork 证据 — is the shape to aim for.

      `README.md` is already correct here: its "## Upstream" section contrasts upstream's
      RTX 5090/`sm_120a` target with this fork's SM86 layer, so it needs no change.

## 5a. Speculative greedy finiteness guard — done

- [x] ~~Thread a per-row finiteness signal into `speculative_accept_sparse_warp_greedy_kernel`.~~
      **Done, and it was both smaller and larger than this entry assumed.**

      *Smaller:* no signature change across the launcher boundary was needed.
      `speculative_accept_sparse_drafts_launch` already receives the verify logits and already
      computes `physical_rows`; the raw-greedy branch simply ignored them. Only the kernel
      signatures lacked the parameters.

      *Larger:* this entry claimed "every other commit path now refuses to license a token chosen
      from a non-finite column". That was wrong. **Four** greedy routes were passing a hardcoded
      `true`, not one:
        - sparse raw-greedy (the route this entry described);
        - dense multiblock `speculative_sampling_group_finalize_kernel<false>`, no penalties;
        - dense penalized greedy, in the same kernel's general path;
        - sparse penalized greedy, in `speculative_sparse_warp_accept`'s greedy branch.

      All four now test the raw verify logit of the token they are about to license, which is the
      signal the single-block dense kernel has always used. Penalties change the score that selects
      the terminal but not the logit behind it, and a diverged pass makes the whole column NaN
      before any penalty applies.

      Regression cases in `tests/ops/test_speculative_round.cpp` cover every route with poisoned
      and clean rows mixed in one batch. Verified adversarially: reverting the guard produces 60
      failures. The last two routes were found by CodePulse review on PR #18.

      *Also found while adding coverage for the dense multiblock routes:* `sampling_adjusted_logit`
      applied presence/frequency penalty subtraction to non-finite raw logits unconditionally.
      CUDA's NaN canonicalization on that subtraction produces a different bit pattern than an
      untouched NaN, which the total-order sort key (`score_id_order_key`) ranks as strictly higher
      -- so a poisoned verify column's already-drafted (penalized) token could spuriously win the
      greedy top-1 selection and get accepted, moving the terminal to a later, clean column and
      evading the finiteness guard entirely. `sampling_adjusted_logit` now returns a non-finite raw
      value unperturbed, so every non-finite entry in a poisoned column stays bit-identical and the
      guard sees whichever one the id tie-break selects.

## 5c. Greedy acceptance validates the whole committed span — done

- [x] ~~Every greedy route tested `sampling_selected_logit_is_finite` only at the divergence
      column, never at a column accepted as a match.~~ **Fixed.** If a diverged forward pass gave a
      column an all-NaN row whose argmax happened to equal that column's draft, the round accepted
      it as a match, folded it into `licensed_tokens` with no finiteness check, and then committed
      on whichever later, clean column became the terminal.

      Raised by CodePulse on PR #18 against 5a's new sparse code, and confirmed to be the
      pre-existing shape of `speculative_accept_greedy_drafts_kernel` on `master` — 5a made the
      other routes match that pattern rather than introducing it. Initially descoped as "touches
      code PR #18 never modified"; that was the wrong call, because the fix is cheap.

      All five greedy sites now validate the committed span `[0, accepted_count]`:
      the two sparse warp routes with a single `__ballot_sync` (one lane per column, and
      `a <= extent <= k <= 15` always fits a warp), the single-block greedy+penalties path by
      accumulating into a shared flag inside the column loop it already runs, and both dense
      multiblock routes through the shared `speculative_commit_span_is_finite` helper.

      Verified adversarially: reverting only the sparse warp path to terminal-only produces **40
      failures, all 40 in the new poisoned-matched-column cases**, with every terminal-column case
      still passing.
## 5b. Reproducibility

- [ ] **fp8, k8v4 and nvfp4 causal attention are not run-to-run deterministic.** Running
      `ninfer_softmax_attention_test` twice from the *same binary* produces ~36 differing
      `OP_ERROR_STATS` lines, always in those three storage families (plus a couple of bf16
      geometry lines); int8-g64 and rk8v4 are byte-identical across runs. All of it stays well
      inside tolerance, so nothing fails — but it means **those cases cannot be used for
      exact-match regression checks**, and it cost a real detour: after a change that touched only
      the INT8 prompt loader, 34 stat lines moved and looked like collateral damage until a
      same-binary control run showed the same 36 lines moving on their own.
      Worth understanding rather than assuming: a split reduction whose order varies, or an atomic
      accumulation, would both explain it. Until then, diff *only the storage family you changed*
      when using these stats to prove a change is neutral.

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
- [ ] **Dedupe `prompt_i8.cuh`'s local f16 dequant helpers.** It defines its own
      `causal_prompt_i8_dequant_f16x8` / `causal_prompt_i4_dequant_f16x8` because the shared codec
      had no f16 variants for the INT8 codings. It does now
      (`kv_cache_int8_dequant_f16x8_from` / `kv_cache_int4_dequant_f16x8_from`), so the locals are
      redundant. That missing pair is exactly what made upstream's small-T reach for the bf16
      helper and silently reinterpret every value, so collapsing them removes the trap rather than
      just tidying.
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

Two more that cost real time on 2026-09-07:

- **Never truncate a build pipeline.** `cmake --build ... | Select-Object -First N` (or `| head`)
  returns while `ninja` keeps running detached, and the next build collides with it —
  `ninja: error: opening deps log: Permission denied`, i.e. self-inflicted concurrent-ninja
  corruption. Redirect the whole build to a file and grep the file afterwards. Before assuming the
  build directory is free: `Get-Process ninja,cmake,cicc`.
- **Buffered stdout lies about where a crash happened.** The last flushed line is *not* the last
  line executed; a truncated trailing line named the wrong function twice during the small-T
  investigation. Add explicit `<< std::flush` markers around candidate regions instead. Note also
  that `cdb` does not capture the debuggee's **stderr** — put diagnostics on stdout.

## Debugging note

A scriptable console debugger is installed: `%LOCALAPPDATA%\Microsoft\WindowsApps\cdbX64.exe`
(from the Store WinDbg package — there is no plain `cdb.exe`, and the Windows Kits `Debuggers\x64`
directory holds only DLLs). `compute-sanitizer` sees device memory only, so
`ERROR SUMMARY: 0 errors` on a process that still dies is positive evidence of a **host-side**
fault — switch to cdb at that point. See `windows-cdb-debugger-available` in memory for the
invocations.
