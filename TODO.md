# TODO

State as of 2026-09-08, after a clearing pass that closed fourteen items.

Released: **v0.9.0-rtx3090** (Windows + Linux). Full suite **125/125** on this box, now including
upstream's DFlash2 attention sweep, which had been skipped since the catch-up merge.

**Every route table in the tree has now been measured on sm_86.** That was the largest outstanding
body of work and it is finished. What remains is correctness, coverage needing hardware this box
does not have, and measurement debt.

In flight: #21 (this file), #26 (503 during startup), #27 (q4 SwiGLU retune), #29 (housekeeping),
#30 (DFlash2 sweep).

---

## 0. Next up, in this order

1. **The `T=112` graph-replay failure** (§1.1). The only open item that could be a correctness
   defect in *released* code. Four hypotheses are already ruled out — read that entry first, it
   will save a day.
2. **The two test-criterion outliers** (§4). Small, the last loose ends from the §5 audit, and
   neither needs hardware this box lacks.
3. **The real-model maximum-configuration decision** (§1.2). A judgement call more than a task.

Everything else is blocked on hardware/artifacts (§2) or is measurement debt (§3).

**Keep `investigate/small-t-upstream` until the next catch-up.** It is merged, but it is the clean
record of how upstream's small-T was adopted and what had to be fixed (`19c7617c` and its parents).
The next merge from `neroued/master` will touch the same subsystem.

---

## 1. Correctness and coverage

### 1.1 `attn_input_proj` grossly wrong at `W8 DFlash2 A16 T=112 graph phase=1`

- [ ] Not a tolerance miss — `actual=34` against `reference=-65.9`, `actual=4.09` against `67.15`,
      on q, k *and* value. Seen in **2 of 6 full-suite runs** (`ctest -j2`), always that exact case
      and always the graph-replay phase; passes 3/3 in isolation.

      Two reasons this outranks everything else. **`T=112` sits in the `{97,128}` →
      `R32C64K128` band retuned in PR #16**, which shipped in v0.9.0. And the criterion involved
      was *loosened* by the §5 floor and still fails loudly, which rules out tolerance — the values
      are ~10x off.

      **Four hypotheses ruled out. Do not repeat these** — 69 targeted iterations across five
      isolation experiments, of which only the full-suite row reproduced:

      | experiment | iterations | reproduced |
      |---|---|---|
      | full `ctest -j2` | 6 (across two builds) | **2** |
      | `-R` subset of 11 related tests, `-j2` | 6 | 0 |
      | full attn test vs one long-lived partner | 14 | 0 |
      | `--dflash2-only` vs one long-lived partner | 40 | 0 |
      | attn test alone | 3 | 0 |

      - **Not tolerance.** One failure was on the §5 loosened-floor build, and the values are ~10x
        off — two orders of magnitude past any plausible bound.
      - **Not memory pressure**, the original assumption. On a 24,576 MiB card the idle baseline is
        1,782 MiB and the partner tests peak at 2,135 MiB: **353 MiB of added load, leaving
        21.9 GiB free at the peak.** Nothing is near a capacity limit.
      - **Not single-partner compute contention**, across 54 iterations that provably covered the
        case (`NINFER_OP_REPORT_STATS=1` confirms `--dflash2-only` runs all three of q/k/value at
        `T=112 graph phase=1`).
      - **Not modest process churn** — 11 tests under `-j2`, six times, stayed clean.

      So the trigger is something only the *full* 125-test run supplies: cumulative allocator state
      across many processes, a specific predecessor test, or duration. **The next step is not
      another contention harness** — it is to catch a failing full-suite run with instrumentation
      already attached (dump the failing tile's inputs, or loop the suite overnight capturing
      `NINFER_OP_REPORT_STATS=1` for this Op only).

      *Method note, because it wasted two attempts:* the first harnesses shelled out to
      `ninfer_softmax_attention_nvfp4_test.exe` / `_k8v4_test.exe`, which **do not exist** — those
      are ctest entries sharing one binary with `--nvfp4-only` / `--k8v4-only`. `Start-Process`
      failed silently, no load ever ran, and the "clean" results were meaningless. Resolve binaries
      with `Resolve-Path`, assert the partner is alive, and take argument lists from
      `build-ninja/tests/CTestTestfile.cmake`.

### 1.2 Real-model tests pin maximum configurations, not this box's capability

- [ ] Measured via the CLI, which is the honest way to size them:

      | model | configuration | result |
      |---|---|---|
      | 27B | 131,072 ctx, int8 | **works** — 4.44 GiB reservation, 449 MiB spare |
      | 27B | 32,768 ctx, int8, `--vision` | **works** — 2.01 GiB reservation, 2.54 GiB spare |
      | 35B-A3B | 32,768 ctx, int8 | **works** — 513.7 MiB reservation, 726 MiB spare |

      There is room for meaningful end-to-end coverage; the tests simply pin
      262,144-context/vision/batch-8 layouts. `NINFER_REAL_TEST_MAX_CONTEXT` lowers the ceiling the
      35B's `exercise_maximum_configuration` asks for (unset, the pinned 256K layout runs as
      before).

      **The 35B's remaining blocker is its *base* engine, not the maximum one:** 20.8 GiB of
      weights plus a 472 MB runtime reservation against ~21.4 GiB free leaves it **~93 MB short**
      when the desktop holds ~2.4 GB. Closing a couple of GPU clients is enough. Decide whether the
      base config should also honour the env var, or skip with a clear message when the reservation
      genuinely will not fit.

- [ ] **Six real-model tests skip** for want of artifacts/env vars: `27b_prefix_real`,
      `27b_score_real`, `27b_load_plan`, `35b_a3b_real`, `35b_a3b_dflash_real`,
      `35b_a3b_dflash_load_plan`. Some fail environmentally on this host rather than from a defect
      — see `ninfer-3090-35b-real-tests-environmental` in memory.

#### Running the real-model tests here (verified 2026-09-07, idle GPU)

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

**Free VRAM is decisive, not marginal.** Desktop busy (~4.4 GiB free): `27b_prefix_real` skips.
Idle (~21.9 GiB free): it passes. The 35B's runtime reservation scales with the ceiling —
262,144 → 4.15 GB, 32,768 → 1.48 GB, 16,384 → 1.29 GB, 8,192 → fits — against ~1.02 GiB available
once the 20.8 GiB of weights are resident.

### 1.3 `docs/config-calculator.html` undercounts startup memory for speculative configurations

- [ ] Its "Startup memory" total only adds the extra resident weights measured for each
      speculation option; it reuses the no-speculation CUDA graph allowance
      (`DATA.graphBytes`, 12 MiB) for every mode and adds no extra KV pages for either backend.
      The real engine (`src/targets/qwen3_6/impl/runtime/layouts_impl.h`) sizes the graph
      allowance per speculative backend and draft window — up to ~86 MiB/lane for MTP, ~96
      MiB/lane for DFlash under `NINFER_SM8X_COMPAT`, against the calculator's flat 12 MiB — and
      `persistent_layout`'s `mtp_extra_pages` reserves additional paged-KV pages for MTP's
      buffered draft tokens that the calculator never adds either. Closing this properly needs
      either a full port of `mtp_graph_profiles`/`dflash_graph_profiles`/
      `graph_topology_allowance` into the page's JS (nontrivial, capacity- and
      draft-window-dependent piecewise logic) or fresh `ninfer_bench` measurements of the actual
      per-mode graph allowance and extra KV pages — both out of scope for a docs PR. Descoped with
      a prominent in-page caveat (the "What this does not model" section and the speculation
      hint) rather than patched inline; the no-speculation case is unaffected and is the one this
      page's own engine cross-check validates.

---

## 2. Genuinely blocked, and what by

This section previously read "blocked on hardware or artifacts" and lumped three items together.
That was wrong on one of them and imprecise on another — **only one needs hardware this box does
not have.** Check before assuming an entry here is unreachable.

### Needs a second GPU — one item

- [ ] **DFlash2 + multi-GPU expert offload.** `nvidia-smi` reports exactly one device here, so the
      offload path degenerates to the single-rank identity mapping and there is nothing to
      exercise. Needs the two-card box or a second local GPU. More interesting now that PR #15 has
      landed and the offload path is no longer hypothetical.

### Needs an artifact we do not have — not a hardware limit

- [ ] **`--spec dflash` (v1) on 35B-A3B.** Refused by the 27B artifact ("selected masked draft
      backend is not supported by this target") because v1 is a 35B backend — correct behaviour.
      The local `qwen3_6_35b_a3b.ninfer` (20.84 GB, revision `c8b8c1c0`) carries **no DFlash
      bundle**: `ninfer_qwen3_6_35b_a3b_dflash_load_plan_test` skips with "this artifact carries no
      DFlash bundle". So this needs a DFlash-carrying 35B artifact, not a bigger card. Worth
      checking whether `neroued/Qwen3.6-35B-A3B-NInfer` publishes one at another revision before
      treating it as out of reach.
The six skipping real-model tests in §1.2 are mostly this same shape — four of them want a
Qwen3.6 27B artifact, a different model family from the `qwen3_8_27b` held locally. Tracked there
rather than duplicated here.

---

## 3. Measurement debt

- [ ] **DFlash2 corpus numbers on sm_86.** Only single-prompt smoke numbers exist (text 20.0% /
      2.38 tok-per-round, vision 85.7% / 7.00). `docs/performance.md` deliberately does not
      reproduce upstream's tables because they are sm_120 — see the provenance banner there.
- [ ] **Speculative decoding is not bit-identical to greedy**, and it is not clear it should be.
      DFlash2 and MTP produce byte-identical output *to each other* and both diverge from the
      width-1 greedy path about a hundred tokens into the text fixture. Verification evaluates k+1
      columns in one pass where plain decode evaluates one, so reductions run in a different order
      and a near-tie argmax flips. MTP reproduces it exactly, so it predates the merge — but nobody
      has decided whether that is acceptable or worth pinning down.
- [ ] **q4 SwiGLU `{513,640}` is still on `Materialized`, unmeasured either way.** The one band
      where Materialized and the c128 tile could not be separated: Materialized won T=576 in three
      of four runs, but c128 there ranged 4147–4959 µs (**19.6% spread**) and the margins outside
      the single outlying run were ±2%. Changing it would be fitting noise. Re-measure if the noise
      floor on this Op improves — it is markedly noisier than the other route tables, which is
      itself worth understanding.

---

## 4. Test-criterion calibration

The §5 audit floored every BF16 gross bound at two rounding steps (#20). Two criteria were
deliberately left out because the BF16 floor argument does not reach them, and both still want
doing:

- [ ] **`sparse_moe` sits at 0.92 of its limit** with `gross_relative_to_max_reference = 0.0`, so
      its bound is pure `gross_absolute` and the floor cannot apply. It also carries the **largest
      observed BF16 error in the tree — 3.64 rounding steps**, against 0.09–1.53 everywhere else.
      Both facts want explaining before the bound is touched: either that kernel is genuinely less
      accurate than every other BF16 Op, or its criterion measures something different.
- [ ] **`gated_delta_net`'s state criterion sits at 0.87** and compares an **FP32** output, so dtype
      rounding is not its floor; the error arrives from BF16 inputs propagating. Needs its own
      derivation rather than the BF16 one.

---

## 5. Reproducibility

- [ ] **fp8, k8v4 and nvfp4 causal attention are not run-to-run deterministic.** Running
      `ninfer_softmax_attention_test` twice from the *same binary* produces ~36 differing
      `OP_ERROR_STATS` lines, always in those three storage families (plus a couple of bf16 geometry
      lines); int8-g64 and rk8v4 are byte-identical across runs. All of it stays well inside
      tolerance, so nothing fails — but those cases **cannot be used for exact-match regression
      checks**, and it cost a real detour: after a change touching only the INT8 prompt loader, 34
      stat lines moved and looked like collateral damage until a same-binary control run showed the
      same 36 lines moving on their own. A split reduction whose order varies, or an atomic
      accumulation, would both explain it. Until then, diff *only the storage family you changed*.

---

## 6. Operational

- [ ] **The pinned host-KV default is 8 GiB regardless of host RAM.** #25 made the failure
      *diagnosable* — the CUDA "out of memory" text now names the size, says it is system RAM not
      VRAM, and names `--host-kv-mib` and `--no-prefix-reuse`. It did not change the sizing, which
      is a policy question needing its own measurement: what "available" means differs by OS, and
      shrinking it silently would regress prefix reuse for people who have the memory.
- [ ] **Binaries embed their build directory.** `/home/ash/ninfer-rel/src/...` appears 200 times in
      the Linux binaries and `C:\ninfer-fork\ninfer-3090\...` about 466 times in the Windows ones,
      via `__FILE__` and nvcc source paths. Pre-existing (v0.8.1 embedded `/mnt/c/ninfer-fork/...`
      405 times) and not a secret — the file names are already public and no Windows binary
      contains a `C:\Users\...` path. `-ffile-prefix-map=` plus `--compiler-options` for nvcc would
      rewrite them to relative paths, which also makes assertion messages more readable. Cosmetic;
      do it with a release build, not on its own.

---

## 7. Closed this cycle, and what it taught

Kept because the reasoning is what stops the same investigation being repeated.

| # | item | the useful part |
|---|---|---|
| #17 | release scripts could not cut a release | `--package` configured `NINFER_BUILD_BENCHMARKS=OFF` while every packager requires `bench/ninfer_bench`, so it failed *after* the whole tree had built |
| #18 | greedy finiteness guard | **five** routes, not the one recorded; and span-wide, not terminal-only — a matched column whose logits are all NaN matched by accident, and a terminal-only guard licensed it |
| #19 | `docs/performance.md` provenance | all ten "tested revisions" are upstream commits; every hardware line says RTX 5090 except the vision section, so "every figure here is measured on sm_86" was false |
| #20 | BF16 gross-error floor | the bound and the error were the same quantity — kernels are accurate to ~1 rounding step, so a bound of that size measures BF16, not the kernel. `2.0 * kBf16UnitRoundoff` was already the house convention in five files |
| #22 | `w8_pair` k=2048, **up to 52.8%** | under `NINFER_SM8X_COMPAT` all twelve `DualSplitKMedium` schedules are **one kernel**, so eight routes could never win. A live table whose *distinctions* are dead |
| #23 | unrouted schedules visible | pins the set **by name**, not by count — a change stranding one schedule while un-stranding another keeps the count identical |
| #24 | shipped launchers | one shipped `HOST=0.0.0.0` (unauthenticated, LAN-wide), one hardcoded an absolute path from this machine, one could never find its own server binary in the release layout |
| #25 | pinned-host diagnostics | `cudaMallocHost` failing says "out of memory" and points entirely at the GPU; it is **system RAM** |
| #26 | 503 during startup | `bind()` before the Engine is deliberate (fast port-clash failure) but left a 10 s window accepting TCP with nothing answering — a `tcpSocket` probe called that ready |
| #27 | q4 SwiGLU Materialized, **up to 23%** | upstream alternated Materialized/c128 three times across one contiguous range; the winner does not flip back and forth, and it did not |
| #28 | q4_q5 "never wins" | the old claim covered T≤208 only; extended to 4096 it holds, and `pair_c64` is a *slower twin* of `mixed_r32_c64_s3`, never ahead |
| #29 | repo housekeeping | worktrees removed, dead files deleted, `repro/` ignored rather than binned, PR #12 closed as the record |
| #30 | DFlash2 attention sweep | the fixture sized the cache table by the **batch**, but `table_rows` are indices *into* the table; B=1 addressing row 7 indexed a one-element vector, unchecked |
| — | §7 `prompt_i8` dedupe | already landed with the small-T adoption; the entry was simply stale |
| — | `--vision-residency overlay` + DFlash2 | **was never blocked** — it runs on this one 3090 and always could have. See below |

### `--vision-residency overlay` + DFlash2 — verified working, 2026-09-08

Listed for weeks as needing hardware. It does not. On this single 3090, with
`qwen3_8_27b_dflash2.ninfer`, `--vision --vision-residency overlay --spec dflash2 --draft-tokens 7`:

- starts cleanly — 18.0 GiB of weights, runtime 1.03 GiB, **2.30 GiB still free**, ready in 9.0 s;
- answers four *different* images in sequence (2.4–2.7 s each), describing each correctly and
  reading its embedded label with the index incrementing 00 → 01 → 02 → 03;
- the server is still healthy afterwards and the log carries no `ERROR`, `FATAL` or eviction line.

The worry in the old entry — overlay borrows device memory per image from the evictable
text-weight tail while DFlash2 holds its own weight bundle, so the eviction ladder is untested —
is exactly what the four-image sequence exercises, because the borrow-and-release has to happen
more than once. It holds.

**The lesson is about the list, not the feature**: "never run" had drifted into "cannot be run".
Try it before writing it off; this took one command.

Two recurring lessons worth carrying forward:

**Fix the class, not the flagged line.** #18 was reported as one route and was five. #24 was
reported as one launcher and was four problems across six. Chasing the class is also what found the
`kv_cache_append` contract lines and the `AGENTS.md` wrong-target claim that nobody had flagged.

**A route table can be live and still wrong.** `q4_q5` had a tuned table nobody read beside a
hardcoded chain. `w8_pair` k=2048 had a table that *was* read, but whose distinctions did not exist
on this hardware. So `grep resolve_plan` is necessary and not sufficient — also grep the launchers
for `NINFER_SM8X_COMPAT`, and confirm in the sweep, where identical schedules print *identical*
times.

---

## Build notes, because they cost hours

**Never run two builds against `build-ninja` at once.** Concurrent ninja produces "Permission
denied" on object files, `cmake --build` reporting success with the executable never relinked, and
phantom hangs. **Always compare the test binary's mtime against the sources before believing a test
result.** A build that looks hung is usually just slow: `nvcc` idles at ~0.1 s CPU while its `cicc`
child works, and `small_t_fp8.cu` legitimately burns 170+ s in `cicc`. Check `cicc`, not `nvcc`.

**A running test binary breaks the next link**, with the same `LNK1104: cannot open file` signature
as concurrent ninja but a different cause. This bit three times in one session, twice from a
background `ctest` still holding a binary while the next build started — including once from a
background job that session had started itself and then rebased underneath. Two habits:

- Before building: `Get-Process ninja,cmake,cicc,ninfer_*` — the `ninfer_*` half is the one people
  forget.
- **Never leave a background build or ctest running while switching branches or rebasing.** Two
  tests "failed" that way and hung as processes; they passed in isolation and the suite was clean.

**Check the build's exit code separately from the test's.** The stale-binary trap is silent: the
build fails, the test runs the *previous* binary, and reports a pass. Every "verified" claim in this
file was checked that way after being caught out by it.

**Never truncate a build pipeline.** `cmake --build ... | Select-Object -First N` returns while
`ninja` keeps running detached, and the next build collides with it. Redirect the whole build to a
file and grep the file.

**Buffered stdout lies about where a crash happened.** The last flushed line is not the last line
executed. Add explicit `<< std::flush` markers around candidate regions. `cdb` does not capture the
debuggee's **stderr** — put diagnostics on stdout.

## Debugging note

A scriptable console debugger is installed: `%LOCALAPPDATA%\Microsoft\WindowsApps\cdbX64.exe` (from
the Store WinDbg package — there is no plain `cdb.exe`, and the Windows Kits `Debuggers\x64`
directory holds only DLLs). `compute-sanitizer` sees device memory only, so `ERROR SUMMARY: 0
errors` on a process that still dies is positive evidence of a **host-side** fault — switch to cdb
at that point. The DFlash2 sweep (#30) is the worked example: zero device errors, and the fault was
a `std::vector` index out of range in the host reference. See `windows-cdb-debugger-available` in
memory for the invocations.
