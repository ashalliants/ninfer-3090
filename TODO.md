# TODO

State as of 2026-09-08, after a clearing pass that closed fourteen items and a KV-format
measurement pass that produced §2b.

Released: **v0.9.0-rtx3090** (Windows + Linux). Full suite **125/125** on this box, now including
upstream's DFlash2 attention sweep, which had been skipped since the catch-up merge.

**Every route table in the tree has now been measured on sm_86.** That was the largest outstanding
body of work and it is finished. What remains is correctness, coverage needing hardware this box
does not have, and measurement debt.

---

## Start here if you are new to this box

Written as a handoff. Everything below is state you cannot recover by reading the code or the git
log, and getting it wrong costs hours.

### The models on disk, and which is which

`C:\Ninefer-3090\models\` — 111 GiB total, and only ~38 GiB free on `C:`, so **check free space
before downloading anything**. Nothing here is in git; `models/` is gitignored.

| file | bytes | what it is |
|---|---:|---|
| `qwen3_8_27b.ninfer` | 18,210,531,328 | Qwen3.**8**-27B dense. The default 27B for benchmarks. |
| `qwen3_8_27b_dflash2.ninfer` | 20,437,336,576 | Same model **with the DFlash2 bundle**. Only artifact that can run `--spec dflash2`. |
| `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 | Qwen3.6-35B-A3B MoE at revision `560f227e`, **with DFlash**. The recommended one. |
| `qwen3_6_35b_a3b_v1_no_dflash.ninfer` | 22,373,184,256 | The superseded `c8b8c1c0` revision. **Renamed from a `.pre-dflash.bak` suffix** — the loader rejects anything not ending `.ninfer`, which is why it could not be benchmarked until it was renamed. Kept only to re-run the DFlash-residency comparison; **20.84 GiB reclaimable** if that is not needed again. |
| `qwen3_6_27b.ninfer` | 17,495,365,888 | Qwen3.**6**-27B at `faaa0c14`. A **different model family** from `qwen3_8_27b` — having one does not satisfy tests wanting the other. |
| `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 | NVFP4-*weight* variant of the above. Note this is a weight profile, unrelated to `--kv-dtype nvfp4`. |

The `qwen3_8_` versus `qwen3_6_` prefix is the single easiest thing to get wrong here, and the
failure mode is a test skipping rather than erroring.

### Environment for the real-model tests

```
NINFER_QWEN3_8_27B_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b.ninfer
NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS=C:\Ninefer-3090\models\qwen3_8_27b_dflash2.ninfer
NINFER_QWEN3_6_35B_A3B_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_35b_a3b.ninfer
NINFER_QWEN3_6_27B_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_27b.ninfer
NINFER_REAL_TEST_MAX_CONTEXT=8192      # only needed for the 35B
```

**Free VRAM decides whether these pass or skip**, not correctness — see §1.2. Close GPU clients
first and check `nvidia-smi`.

### Recently landed, and what is still owed on it

- **#32 — `docs/config-calculator.html` — merged, but not finished.** Three of the six confirmed
  defects were fixed before it landed and **two were not**; the file itself says so, carrying a
  `KNOWN GAP (see TODO.md)` comment where speculative memory should be modelled. See §2b. So the
  page is live on master, README and `docs/cli.md` link to it, and **it still undercounts any
  speculative configuration by roughly 170 MiB.** Closing that is the top non-correctness item in
  §0, and the multiplier it needs is already measured in §2b. Until then, treat its speculative
  rows as optimistic.
- A worktree at `.claude/worktrees/eager-baking-cascade` exists and has been used by a second agent
  working the same branches. **Check `git worktree list` before assuming a branch is free**, and
  `git fetch` before pushing: concurrent work on `fix/artifact-download-revisions` was duplicated
  once this cycle because of exactly that.

### Windows tooling hides Linux breakage, and did

`scripts/run-qwen36-35b-a3b-c1-maxctx.sh` and `scripts/run-qwen38-c1-maxctx.sh` — the two launchers
README recommends as *the* Linux entry point, both shipped in the Linux release archive — had CRLF
committed and **would not parse on Linux at all**:

```
run-qwen36-35b-a3b-c1-maxctx.sh: line 84: syntax error near unexpected token `$'in\r''
```

Inside a `case` block the stray CR joins the `in` token. Those two files are the only ones with a
`case`, which is why only they were fatal while the tree looked healthy. Fixed in #34, with the
policy pinned in `.gitattributes` and a guard added to `check-linux-scripts.sh`.

**The lesson is the part worth keeping.** Every tool on this box reported the tree clean: Git Bash
strips CR on read so `bash -n` passed, `check-linux-scripts.sh` passed, and Git Bash's `grep`
translates CR away so searching for them returned nothing — a loop using it called all 26 shell
scripts clean while two were 141 and 114 CRLF pairs deep. `core.autocrlf=input` was already set and
did not help, because the blobs predate it. It was visible only from **real Linux bash under WSL**,
or by reading bytes with .NET. When something is claimed to work on Linux and has only been checked
here, run it under `wsl -e bash` before believing it.

### Measurement scripts

`scripts/sweeps/` holds the PowerShell that produced every number in §2b, README's KV table and the
calculator. They are committed because §2b asks for re-measurement and reconstructing them is an
hour of work. Read `scripts/sweeps/README.md` first — it records which of them answer which
question, and which one produced a wrong answer and why.

---

## 0. Next up, in this order

1. **The `T=112` graph-replay failure** (§1.1). The only open item that could be a correctness
   defect in *released* code. Four hypotheses are already ruled out — read that entry first, it
   will save a day.
2. **Decode runs at 66% of the card's memory bandwidth** (§2c). The largest single number in this
   file, it applies to every configuration rather than one feature, and no performance work here
   has ever been measured against an absolute ceiling. Profile one decode step first.
3. **DFlash2 currently costs 20% on the 27B** (§2c). A shipped v0.9.0 feature that is a net loss
   on text, and the cheapest possible first test -- sweep `--draft-tokens` below 7 -- has never
   been run.
4. **The calculator's speculative-memory gap** (§2b). Now shipped on master and linked from
   README, so it is live: every speculative configuration it reports is roughly 170 MiB optimistic.
   The multiplier that fixes it is already measured — implementing, not investigating.
5. **Re-run the six real-model tests** (§1.2, §2). Their artifact blockers are gone as of #31 and
   nobody has looked since; at least one is expected to fail rather than skip.
6. **The two test-criterion outliers** (§4). Small, the last loose ends from the §5 audit, and
   neither needs hardware this box lacks.
7. **The real-model maximum-configuration decision** (§1.2). A judgement call more than a task.

Only one open item now needs hardware this box lacks (§2). The rest is measurement debt (§3),
calibration (§4) or policy calls (§6).

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

      **The artifact half of this is now gone** (2026-09-08): the Qwen3.6 27B artifact and a
      DFlash-carrying 35B are both local and pinned, per §2. Nobody has re-run the six since. Do
      that before treating any of them as blocked, and expect at least one real failure rather than
      a skip — `27b_score_real` was last seen reporting *a repeated score window inherited prior
      State/KV*, which is a defect the missing artifact was masking.

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

This section once read "blocked on hardware or artifacts" and lumped three items together. Two of
those three were not blocked at all, and both have since been closed by going and looking. **Exactly
one open item needs hardware this box does not have.** Check before assuming an entry here is
unreachable — this section has a poor record of being right about that.

### Needs a second GPU — one item, and it is the only one

- [ ] **DFlash2 + multi-GPU expert offload.** `nvidia-smi` reports exactly one device here, so the
      offload path degenerates to the single-rank identity mapping and there is nothing to
      exercise. Needs the two-card box or a second local GPU. More interesting now that PR #15 has
      landed and the offload path is no longer hypothetical.

### Needs an artifact we do not have — nothing is left here

Both entries are closed. The suggestion one of them carried — "check whether
`neroued/Qwen3.6-35B-A3B-NInfer` publishes one at another revision before treating it as out of
reach" — was the right instinct and paid off: it did. Kept rather than deleted because the pattern
recurs, and because the follow-on work below only exists now that they are gone.

- [x] **`--spec dflash` (v1) on 35B-A3B** — resolved 2026-09-08. Revision `560f227e` (now `main`)
      carries the DFlash bundle; `c8b8c1c0` predates it. `scripts/download-qwen36-35b-a3b.{sh,bat}`
      and `flake.nix` are pinned to it as of #31. Loading with `--spec dflash --draft-tokens 3`
      reports 21,448,523,264 bytes of weights against 21,038,469,632 with `--spec` unset — exactly
      the 410,053,632-byte on-disk difference between the two revisions, so **carrying DFlash costs
      nothing resident unless it is selected**, and the published 24 GB profiles still apply.
- [x] **The Qwen3.6 27B artifact** — downloaded and pinned at `faaa0c14` (#31), so the four tests
      in §1.2 that wanted it are no longer artifact-blocked.

### Unblocked, and now owed work

- [ ] **Re-run the six real-model tests now that both artifacts exist** (§1.2). Their skip reasons
      are gone; what actually passes on this box is unknown. `27b_score_real` in particular was
      last seen failing with *a repeated score window inherited prior State/KV* — a genuine defect
      the missing artifact was hiding, not an environmental skip.

---

## 2b. `docs/config-calculator.html` is advertised as authoritative and is not yet correct

Added by #32, **which merged with two of its six confirmed defects still open**, one of them only partly. README and
`docs/cli.md` link to the page, so those two are live rather than theoretical. Read the checkboxes
rather than assuming a merged PR means a finished page; the file itself agrees, carrying a
`KNOWN GAP (see TODO.md)` comment where speculative memory should be modelled.

Closed before it landed: KV page rounding (with regression tests), sub-4,096 decode (now labelled a lower bound held at the
shallowest measurement rather than claimed as interpolated), and per-row weight-artifact labelling.
The doc contradictions were fixed separately in #33. The evidence for each is kept below so nobody
has to re-derive it.

- [ ] **Speculative modes undercount memory by roughly 170 MiB, plus a per-token term.** The page
      models speculation as a weights delta only. Measured on the 27B at `--max-ctx 40960`, INT8,
      `none` against `mtp3 + --lm-head-draft`:

      | component | none | mtp3+head | delta | modelled? |
      |---|---:|---:|---:|---|
      | weights | 17,093,490,688 | 17,901,798,400 | +771 MiB | yes |
      | sequence | 1,550,561,536 | 1,646,461,184 | +91.5 MiB | **no** |
      | CUDA graph allowance | 12,582,912 | 90,177,536 | +74 MiB | **no** |
      | workspace | 159,981,568 | 159,981,568 | 0 | n/a |

      `graphBytes` is hardcoded at 12,582,912, which is only right with speculation off.

      **The speculative KV term is per-token, and it is one constant.** Measured across all six
      formats on the 27B, MTP3 + draft head raises KV bytes/token by the same **6.26%** every time:

      | format | none | mtp3+head | ratio |
      |---|---:|---:|---:|
      | `bf16` | 65,536 | 69,638.4 | 1.0626 |
      | `int8` | 33,792 | 35,907.3 | 1.0626 |
      | `fp8` | 33,024 | 35,091.2 | 1.0626 |
      | `rk8v4` | 26,112 | 27,746.6 | 1.0626 |
      | `k8v4` | 25,728 | 27,338.5 | 1.0626 |
      | `nvfp4` | 18,432 | 19,585.8 | 1.0626 |

      So the fix is a per-token multiplier on the cache plus a per-mode constant for the graph
      allowance and sequence delta, not a flat offset. Only MTP3 + draft head was swept; DFlash and
      DFlash2 need the same treatment before their rows can be trusted. Raw CSVs come from
      `scripts/sweeps/kv-decode-with-speculation.ps1`.

- [x] **KV is allocated in 64-token pages; the page charges exact tokens.** Fixed on #32. `KV page groups
      4096 / 4096` at a 262,144 context is 64 tokens per group. A context just past a page boundary
      reserves a whole further page, so both the memory figure and the largest-context result
      should round to a page. Every context measured so far happens to be page-aligned, which is
      why this never showed up in the validation against the engine's own refusal message.

- [x] **Sub-4,096 decode is reported at the wrong depth.** Fixed on #32, by labelling rather than by measuring — it now reads "lower bound, held at the 4,096-token measurement". Measuring 1,024 and 2,048 would still be better and is cheap. `decodeAtDepth` returns the 4,096-token
      measurement for every context from 256 up, and the UI then labels it `interpolated` when no
      interpolation happened. The honest fix is to measure: 1,024 and 2,048 are cheap, and short
      contexts are exactly where a casual user starts.

- [x] **The model selector does not name the artifact each row was measured against.** Fixed on #32. `27b` means
      `qwen3_8_27b.ninfer` specifically; the runtime has separate load plans and device capacities
      per weight profile, so `weightsBytes` is not transferable to, say, the NVFP4-weight variant.
      Either add the weight-profile dimension or label each row with its artifact.

- [ ] **The regression tests exist but nothing runs them.** #32 landed
      `docs/config-calculator.test.mjs`, and it is good: it extracts the page's `<script>` into a
      `vm` sandbox against a DOM stub — keeping the one-file, no-build property rather than
      restructuring the page into modules — and pins page rounding, the `decodeAtDepth` boundaries,
      and a golden case asserting the 262,144-token INT8 figure still equals the engine's own
      9,197,389,568-byte refusal. It passes (`node docs/config-calculator.test.mjs`).

      What is missing is a runner. Nothing in CI, CTest or `scripts/` invokes it, so it will catch
      a regression only if someone remembers to run it by hand. Wire it in. It needs `node`, which
      no other test here does, so it probably belongs in the GitHub workflow rather than CTest.
      It also does not cover the speculative path — which is the half still known to be wrong — so
      extend it alongside that fix rather than after.

- [x] **Active docs still contradict the six-format claim.** Fixed in #33, across four places. #32 fixed README's `Current limits`
      and `docs/perplexity.md`, but README's *opening* summary still says the FP8 E4M3 KV profile
      "is not" admitted on SM86, and `docs/cli.md` and `docs/serving.md` were not touched. All six
      formats are measured and working; the Blackwell restriction applies to FP8/NVFP4 *weights and
      activations*, not KV storage. Fix them together so the claim is consistent everywhere.

---

## 2c. Performance left on the table

Nothing here was in this file before 2026-09-08, which is itself the point: the KV sweeps were run
to document formats, and these fell out of the data on the way. Ordered by size of the prize.

**Read the first entry before any of the others.** The rest of this section is specific kernels and
features. The first one says the whole decode path has room in it, and no work here has ever been
framed against a roofline — every performance change in this repository so far has been *relative*
("this schedule beats that one"), never *absolute* ("this is N% of what the card can do"). That is
the gap most likely to be hiding real speed.

- [ ] **Dense decode runs at 65-66% of the card's memory bandwidth, and nobody has ever checked.**
      Decode is memory-bound: each token streams the resident weights once plus the KV it attends
      over. Against the 3090's 936.2 GB/s, from this cycle's own measurements (single lane, **no
      speculation**, so tokens and weight-read rounds are the same thing):

      | model | KV | depth | tok/s | achieved | % of peak |
      |---|---|---:|---:|---:|---:|
      | 27B dense | `int8` | 4,096 | 36.15 | 623 GB/s | **66.5%** |
      | 27B dense | `int8` | 16,384 | 35.11 | 620 GB/s | **66.2%** |
      | 27B dense | `int8` | 32,768 | 33.86 | 616 GB/s | **65.8%** |
      | 27B dense | `rk8v4` | 32,768 | 33.54 | 602 GB/s | **64.3%** |

      Flat across depth and across KV format, which says it is a property of the decode path rather
      than of anything the KV work touched. A well-tuned memory-bound decode generally reaches
      75-85% of peak. Closing even half that gap is roughly **36 → 41 tok/s on the 27B**, which is
      larger than everything else in this section combined and applies to every configuration
      rather than one feature.

      This is a headline number, not a diagnosis: it says the room exists, not where it is. Next
      step is a profile of one decode step (`--profile-measured` exists on `ninfer_bench`) to find
      where the stall is — occupancy, L2 behaviour, or a launch gap between the per-layer kernels.

- [ ] **Nobody knows where the MoE sits at all, because the accounting does not exist.** Applying
      the same arithmetic to the 35B returns *391% of peak*, which is not a result, it is a proof
      that the formula does not apply: an A3B MoE activates a fraction of its 21.04 GB per token,
      so it never reads resident weights the way the dense model does. The bytes actually touched
      per token — shared attention weights plus whichever experts route — have never been counted,
      so there is no denominator, and therefore no way to say whether the 35B's 174 tok/s is close
      to its ceiling or half of it. **Do that accounting before optimising anything on the MoE
      path**; it is arithmetic over the artifact's layer inventory, not a measurement.

- [ ] **Eight lanes buy 3.2x, not 8x, and the reason is unestablished.** README's own cohort table
      has C1 decode at 78.71 tok/s against C8 at 250.26. Batched decode amortises the weight read
      across the whole cohort — the same bytes serve every lane in a round — so a weight-bound
      decode should scale much closer to linearly until compute takes over. Somewhere between one
      lane and eight, something other than weight bandwidth becomes the limit, and no document says
      what. Worth knowing: C8 is the profile the 27B release recommends for multi-user serving.

      Careful with those particular numbers, though: they are end-to-end with MTP3 enabled, so
      tokens per round is roughly `1 + 3 x acceptance` rather than 1, and dividing them into a
      bandwidth figure gives nonsense (144% of peak at C1, doing it naively). The comparison needs
      re-measuring without speculation before it can be reasoned about — `ninfer_bench` can do that
      directly and the cohort table cannot.

- [ ] **Prefill has no roofline either.** Measured this cycle at 1,254 tok/s on the 27B and 5,715
      on the 35B at a 4,096-token prompt, falling to 1,087 and 4,302 by 32,768 — a 13% and 25%
      decay that nothing explains or has looked at. Prefill is compute-bound rather than
      bandwidth-bound, so the denominator is FLOPs and the accounting differs from decode; it has
      not been done. Note the route tables *are* all measured (that work is finished), but measured
      against each other, not against the hardware's ceiling.

- [ ] **Vision has essentially one performance number in the entire repository.** One acceptance
      figure for DFlash2 on the committed image fixture, and nothing about encode throughput,
      how it scales with resolution, or what the overlay residency costs in time rather than in
      bytes. It is an advertised feature of both models and it is unmeasured.

- [ ] **DFlash2 makes the 27B *slower*, and no document says so.** Measured on one artifact
      (`qwen3_8_27b_dflash2.ninfer`), 8,192-token context, INT8 KV, tg128, three repetitions:

      | config | tok/s | vs no speculation |
      |---|---:|---:|
      | none | 35.44 ± 1.42 | — |
      | `--spec dflash2 --draft-tokens 7` | 27.57 ± 0.06 | **−22.2%** |
      | ...` --lm-head-draft` | 28.33 ± 0.02 | **−20.1%** |
      | `--spec mtp --draft-tokens 3` | 41.28 ± 0.10 | +16.5% |
      | ...` --lm-head-draft` | 43.99 ± 0.10 | +24.1% |

      The deviations rule out noise. `docs/performance.md` already records the *acceptance* —
      20.0% on text, 2.38 tok/round — but never states the throughput consequence, so a reader
      reasonably assumes a headline v0.9.0 feature is a win. On text it is a 20% penalty, and it is
      36% behind MTP3 with the draft head on the same file.

      The arithmetic explains it and points at the fix. Seven drafts at 20% acceptance yields ~2.38
      tokens per round, but each round verifies eight columns instead of one. **Nobody has tried
      any other draft count** — the sweep used 7 throughout, because `docs/cli.md` calls seven "the
      checkpoint recommendation". Sweeping `--draft-tokens 1..7` is cheap and is the first thing to
      do; there may simply be a crossover below 7 where DFlash2 turns positive on text.

      Note the same document records **vision** DFlash2 at 85.7% acceptance and 7.00 tok/round,
      which is close to ideal. So the backend is not broken — text drafting specifically is. If no
      draft count makes text positive, say so in the docs and scope the feature to vision rather
      than leaving it as an apparently-free option.

- [ ] **The three KV formats with the worst decode falloff are exactly the three that are not
      run-to-run deterministic.** Falloff from a 4,096- to a 32,768-token cache, no speculation:

      | format | 27B | 35B | deterministic (§5) |
      |---|---:|---:|---|
      | `int8` | −6.3% | −10.9% | yes |
      | `rk8v4` | −6.2% | −9.5% | yes |
      | `bf16` | −11.5% | −14.9% | mostly |
      | `fp8` | −14.6% | −21.0% | **no** |
      | `nvfp4` | −15.7% | −21.1% | **no** |
      | `k8v4` | −17.9% | −25.2% | **no** |

      `rk8v4` is the control that kills the obvious explanation: it packs 4-bit values exactly as
      `k8v4` does, and it has the flattest curve of all six. So this is not the cost of unpacking.
      **If `nvfp4` decoded on `rk8v4`'s curve it would be the outright best format** — 45% smaller
      than INT8 with no speed penalty — rather than the compromise it currently is.

      Read the code rather than inferring from the table, and two things came back. One is a
      likely cause; the other kills the tidy theory that produced this entry.

      **Cause, and a one-line experiment.** fp8, k8v4 and nvfp4 are the only formats calling
      `causal_small_t_quantized_active_splits` (`small_t.cuh:122-130`), which at decode past 8,198
      keys asks for `SmallTMaximumSplits` (85). But the *host* capacity function
      (`small_t.cu:49-53`) grants that bump to `Fp8E4M3Row256` **only**, and the device then clamps
      to whatever the host allocated (`small_t.cuh:129`). At a 32,768-token decode the default tier
      is `div_up(32768, 480) = 69`, so:

      | format | device policy asks | host grants | actually runs |
      |---|---:|---:|---:|
      | `fp8` | 85 | 85 | 85 |
      | `nvfp4` | 85 | 69 | **69** |
      | `k8v4` | 85 | 69 | **69** |

      nvfp4 and k8v4 run 19% fewer splits than the code intends, at exactly the depths where they
      measure worst, and nothing reports the discrepancy. Extending the condition at
      `small_t.cu:50` to `Nvfp4Group16` and `Fp8KeyNvfp4Value` and re-running
      `scripts/sweeps/kv-decode-vs-depth.ps1` is a cheap test of how much of the falloff that is.
      Note it cannot be the whole story: fp8 *does* get 85 splits and still falls off 21% on the
      35B, so it is paying something else — plausibly `KeyBlock` pinned to 32 rather than the int8
      path's 64, forced by sm_86's shared-memory budget because the fp8 kernel keeps dequantized
      BF16 copies of K *and* V (`small_t_fp8.cu:26-29`).

      **What this entry originally claimed, wrongly.** It read the §5 non-determinism and this
      falloff as one root cause — "a split reduction whose order varies, or an atomic
      accumulation". There are **no atomics anywhere** in `src/ops/softmax_attention/`, and both
      reducers accumulate over an identical ordered `for (split = 0; split < active_splits; ++split)`
      loop, which is deterministic given the same split count. So the correlation across six
      formats is real and still wants explaining, but the mechanism proposed for it is not the one.
      Treat §5 as open on its own terms.

      One thing found on the way that is worth its own look: the fp8 partial kernel returns early
      for `split >= active_split_count` **without writing neutral values**
      (`small_t_fp8.cuh:148`), so untouched splits keep whatever the workspace arena last held.
      That is safe only while the partial kernel and the reducer compute the same
      `active_split_count`. They do today — both clamp to the same launch capacity — but it is an
      invariant held by coincidence of two separate call sites rather than by construction, and the
      host/device asymmetry above shows those sites already disagree about intent.

- [ ] **The sm_86 fallback constants were chosen to fit, not measured.** `NINFER_SM8X_COMPAT`
      guards eleven sites, and most encode a real hardware limit rather than a missing feature —
      sm_86's 49,152-byte static shared-memory cap forces `KWarps` from 8 to 4 in
      `w8_config.h:59`, halves the K tile in `w8_linear_swiglu_gemm_mma.cu:120`, drops
      `kLastExactCols` from 48 to 32 in `w8_linear_add_gemm_splitk.cu:19`, and so on. The comments
      are honest that these are what fits. None of them record a measurement showing the chosen
      value is the *best* one that fits, and the alternatives were never swept on this hardware.
      Lower confidence than the two above, but it is untouched ground across every W8 Op.

- [ ] **`w8_pair` medium discards its schedule entirely on sm_86.**
      `w8_pair_gemm_splitk.cu:133` is `(void)schedule` under `NINFER_SM8X_COMPAT`, so every
      schedule runs the same chunked loop. PR #22 already removed the twelve now-identical route
      entries this made redundant, confirmed by identical 24.576 µs timings, so the *table* is
      honest. What is unknown is whether a genuinely sm_86-specific tiling would beat the generic
      chunking — nobody has written one to find out.

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

- [ ] **The speculative decode sweep measures acceptance, not depth, and cannot be read like the
      non-speculative one.** `scripts/sweeps/kv-decode-with-speculation.ps1` produced 35B INT8 at
      212.19 tok/s on a 4,096-token cache and **316.83 tok/s on a 32,768-token one** — faster
      deeper, which is not physical. Acceptance went 58.3% → 100% between those two points, and on
      the fixed `bench_corpus.ids` a draft can simply be right every time on a repetitive stretch.
      Several rows sit at exactly 100%. So those tok/s figures describe what the corpus does to the
      draft, not what depth does to attention, and none of them are in README or the calculator for
      that reason. Speculative throughput needs a corpus with realistic diversity, or many more
      repetitions, before it means anything. The memory columns from the same run **are** sound —
      they are read at load and do not depend on the corpus.

- [ ] **The published perplexity figures for fp8, k8v4 and nvfp4 are single runs of a
      non-deterministic path.** README's six-format table and the calculator both carry
      `fp8 4.347181` and `k8v4 4.347596`, a gap of 0.0096%, and those two formats are named in the
      entry directly below as not run-to-run deterministic. 261,167 scored tokens should average
      most of that away, but nobody has checked, so **the fp8-versus-k8v4 ordering may not be
      real** — and it is currently the stated reason for preferring one over the other. One repeat
      run of each settles it; if the spread is comparable to the gap, say so beside the numbers
      rather than ranking them. `bf16`, `int8` and `rk8v4` are unaffected: the same entry records
      those as byte-identical across runs.

- [ ] **fp8 and k8v4 are dominated on all three axes and nothing says so structurally.** Each is
      beaten by `rk8v4` on size, decode-at-depth and perplexity simultaneously (README's table).
      That is a documentation statement today; decide whether it should be more — de-emphasised in
      `--help`, or left fully available on the grounds that upstream parity is worth more than
      steering. Not a defect either way, but the measurement is in and the decision is not.

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
- [ ] **The unpinned downloaders cannot verify anything they fetch.** #31 gave the pinned
      downloaders revision-scoped staging plus size and SHA-256 verification. `download-qwen38-27b`
      (both the script and the flake entry) and `flake.nix`'s `download-qwen36-35b-v2` deliberately
      track upstream `main`, so they have no expected size or hash to check against: a truncated or
      corrupt 17 GB download is accepted silently and, because `main` can move between runs, they
      cannot safely resume either. Pinning `qwen3_8_27b` would close it at the cost of freezing
      people to a revision that goes stale — a policy call, not a bug. Whatever is decided, the
      asymmetry should be stated where people choose a downloader.

- [ ] **`package-release-rtx4090-early1.ps1` has no Linux counterpart.** `check-linux-scripts.sh`
      requires every `.bat`/`.ps1` to have one, and #31 exempted this file by name so the check
      could run at all. Either write the counterpart or leave the exemption, but it is a list that
      will rot silently if release packaging grows another Windows-only script.

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

## Measuring memory and capacity, because two of these were dead ends

**`ninfer_bench` cannot tell you the automatic-sizing context.** Running it with `-n 1` and no
`-p` and letting `--kv-capacity` default resolves `max_context` to **3**: auto-sizing follows the
*workload*, not the card. It looks like a plausible answer in the CSV and is nothing of the sort.
Use the serving path instead — `apps/ninfer --prompt hi --max-new 1 --kv-capacity auto` prints
`KV capacity` in its summary — and note the answer moves with free VRAM, so it is a property of
the box at that moment, not of the format.

**Never derive a per-token cost from a single context length.** `sequence_capacity_bytes` minus
`kv_payload_bytes` divided by one context looks exactly like 4,063.5 B/token on the 27B. Measured
at 8K/16K/32K/64K it resolves to `166,438,656 + 0.0625 × ctx` — a *fixed* 158.7 MiB block plus a
sixteenth of a byte per token. The single-point reading understates memory by 127 MiB at 8K and
overstates it by 349 MiB at 131K. Two points distinguish the shapes; four confirm it.

**`sequence_capacity_bytes` contains `kv_payload_bytes`.** They are not additive. Summing the two
columns double-counts the whole cache.

**The engine will check your arithmetic for you.** Ask for a context that does not fit and the
refusal names the exact requirement: `minimum Engine runtime reservation requires 9197389568 bytes
in addition to 1073741824 bytes of automatic headroom`. That figure is
`kv_per_token × ctx + fixed sequence block + 0.0625 × ctx + workspace + graph allowance`, to the
byte, and `--kv-capacity auto` holds back a further 1 GiB on top. Cheaper than any probe, and it
is the number the runtime actually applies.

---

## Guards that exit early can silently disable everything after them

`scripts/check-linux-scripts.sh` had been failing on master for some time, at a counterpart rule
near the top, so none of its downloader coverage below had been running — which is how the curl
fixture in it went stale unnoticed. A check that exits non-zero *is* visible in CI, but only as
"the check failed", and the failing line was unrelated to the part that mattered. When a guard
script grows sections, either make it accumulate failures and report them all at the end, or be
suspicious when the first failure is something cosmetic: everything downstream is then untested,
not passing. Same shape as the dead route table in the note above — the thing that looked live
was not running.

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
