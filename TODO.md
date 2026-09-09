# TODO

State as of 2026-09-09, after a correctness pass that closed nine items and found three defects
nobody had filed.

Released: **v0.9.0-rtx3090** (Windows + Linux). Full suite **126/126** on this box.

**The headline is that three of the six KV formats were quietly wrong.** fp8, nvfp4 and k8v4
attention was missing a block barrier between `cp_wait<0>()` and a whole-tile shared-memory read,
and it corrupted every prefill output column except the last — by up to 26 in logprob. Greedy
decode reads only the last column, so generation stayed byte-identical and nothing noticed. Every
published perplexity figure for those three formats was measured through it. See §5.

**The decode roofline now has a denominator**, and it is not the one this file used for a cycle.
The card sustains 854 GB/s on reads, not its advertised 936; and a token reads far less than
`weights_capacity_bytes`. On that accounting the dense 27B sits at ~66-70% of achievable and the
**35B MoE at ~51%, flat across depth** — the larger prize, on the recommended model. See §2c.

Every route table in the tree has been measured on sm_86; that work is finished. What remains is
profiling, measurement debt, and coverage needing hardware this box does not have.

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
NINFER_QWEN3_6_27B_NVFP4_WEIGHTS=C:\Ninefer-3090\models\qwen3_6_27b_nvfp4.ninfer
NINFER_REAL_TEST_MAX_CONTEXT=8192      # only needed for the 35B
```

`NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` was missing from this block for a cycle, and it is the only
thing that was keeping `27b_load_plan` skipping — the artifact has been on the disk all along.

**Free VRAM used to make these fail outright, not skip.** Skip (exit 77) here is reserved for a
missing artifact env var; a VRAM shortfall instead threw during `Engine` construction and failed
the test. The specific cause was the pinned host-KV allocation being charged against the card (§6,
#45), and with that clamped the six real-model tests pass on an idle box and under `ctest -j2`.
That closes the one source of VRAM-caused failure this fix addresses -- it does not mean free VRAM
no longer affects these tests at all: the model-weights allocation itself can still fail under real
memory pressure from another process. Still worth closing GPU clients before a *measurement* run,
where free VRAM also changes the automatic sizing.

### Recently landed, and what is still owed on it

- **#32 — `docs/config-calculator.html` — merged, but not finished.** Three of the six confirmed
  defects were fixed before it landed and **two were not**; the file itself says so, carrying a
  `KNOWN GAP (see TODO.md)` comment where speculative memory should be modelled. See §2b. So the
  page is live on master, README and `docs/cli.md` link to it, and **it still undercounts any
  speculative configuration by roughly 170 MiB.** Until then, treat its speculative rows as
  optimistic.
- **A second agent works in this checkout.** On 2026-09-09 a *locked* worktree at
  `.claude/worktrees/moonlit-scribbling-sundae` was on `sync/neroued-catchup-20260908` with build
  logs minutes old. **Run `git worktree list` before assuming a branch is free**, `git fetch`
  before pushing, and do not delete a worktree you did not create even when it is holding 46 GB.
- **Do not merge a stack of dependent PRs in one pass.** Seven were opened stacked, each based on
  the one below; merging them bottom-up with `--delete-branch` deleted each base out from under
  the PR above it, and GitHub *closed* three rather than retargeting them. Only the first reached
  master. The recovery is to retarget each branch at `master` and merge one at a time, which works
  because a branch that is a prefix of the chain diffs to exactly its own commit once its
  predecessors have landed.

### Disk, because it stopped a build dead

`C:` filled to **zero bytes free** mid-session and a build failed with
`fatal error C1085: Cannot write compiler generated file ... No space left on device`. Three
things were holding it: the other agent's worktree (46 GB), `build-linux` (60 GB), and
`scripts/models/qwen3_8_27b.ninfer` — a **19 GB byte-identical duplicate** of
`qwen3_8_27b_dflash2.ninfer`, verified by SHA-256, sitting under the downloaders' default output
path and *misnamed*, so anything trusting that path would silently load the DFlash2 artifact.
The duplicate is deleted. Two WSL crash dumps in `%TEMP%\wsl-crashes` were another 12 GB.

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

1. **Profile one decode step** (§2c). The MoE runs at ~51% of the card's *achievable* read
   bandwidth and is flat across depth; the dense path at ~66-70%. Both denominators now exist and
   neither path has ever been profiled. `scripts/sweeps/decode-step-profile.ps1` captures exactly
   one measured repetition under nsys and reports GPU-busy against window-wall first, because a
   launch gap is not recoverable by kernel tuning and rules itself in or out in one run.
2. **Re-measure perplexity for fp8, k8v4 and nvfp4** (§5). Those three numbers were scored
   through the attention race #49 fixed. They are not "single runs that probably averaged out";
   they are measurements of a broken kernel, and the format ranking built on them means nothing
   until they are redone. Include int8 as a control.
3. **The KV decode falloff still wants an explanation** (§2c). #49 removed the tidy theory
   without moving the numbers: fp8/k8v4/nvfp4 still fall off two to three times as fast as
   int8/rk8v4.
4. **DFlash2 costs 20% on the 27B and no draft count below 7 has ever been tried** (§2c).
   `scripts/sweeps/dflash2-draft-tokens.ps1` sweeps 1..12 with and without the draft head.
5. **The calculator's speculative-memory gap** (§2b). Live on master and linked from README, so
   every speculative configuration it reports is roughly 170 MiB optimistic.
   `scripts/sweeps/speculative-memory-terms.ps1` reads every term the page needs in about ten
   minutes; this is implementing, not investigating.
6. **The two test-criterion outliers** (§4). Small, and neither needs hardware this box lacks.

Only one open item now needs hardware this box does not have (§2). The rest is measurement debt
(§3), calibration (§4) or policy calls (§6).

**Keep `investigate/small-t-upstream` until the next catch-up.** It is merged, but it is the clean
record of how upstream's small-T was adopted and what had to be fixed (`19c7617c` and its
parents). The next merge from `neroued/master` will touch the same subsystem.

---

## 1. Correctness and coverage

### 1.1 `attn_input_proj` grossly wrong at `W8 DFlash2 A16 T=112 graph phase=1` — closed

- [x] **A race in the test harness, not in any kernel.** Closed by #44, 2026-09-09.

      `DeviceBuffer::copy_from_host` uses `cudaMemcpy`, and for a host-to-device copy out of
      *pageable* memory CUDA documents that the call returns once the source has been staged
      for DMA, "but the DMA to final destination may not have completed". That trailing DMA
      completes on the legacy stream, and every stream `DeviceContext` owns is created with
      `cudaStreamNonBlocking`, which is exempt from the legacy stream's implicit ordering. So
      "stage the inputs, then launch on the engine stream" is a race, and it is the sequence
      every Op test's setup uses.

      A faithful replica of exactly what the failing case runs — a 1,146,880-byte pageable
      H2D, three `cudaMemsetAsync` on a non-blocking stream, then a captured graph launched on
      it — read stale bytes in **146 of 200 iterations**, up to 17.8% of the buffer, on an
      otherwise idle GPU. With `cudaStreamSynchronize(nullptr)` after the copy: 0 of 200.
      Copies at or below ~256 KiB never raced; every size above it did.

      Every property of the reported failure follows: only the graph-replay phase, because it
      is the only phase that re-uploads its activation; q, k *and* value at once, because they
      share one input; never in isolation, because an idle box lands the DMA in time; and the
      input verifying clean afterwards, because by then the DMA has long finished.

      `tests/test_device_buffer_visibility.cu` is the reproducer kept as a regression test. It
      reads stale data in roughly 90 of every 100 iterations against an unsynchronized
      `DeviceBuffer`.

      **The part worth keeping.** The window is narrow enough that instrumentation closes it.
      Two probes were built and both were useless for opposite reasons: a D2H read-back on the
      same stream, ordered ahead of the kernel, made the race vanish entirely (the copy engine
      serializes the in-flight H2D behind it), and an early draft of the regression test
      cleared its counter with `DeviceBuffer::fill` between the copy and the stream work —
      one synchronous runtime call in the gap, and 0 races out of 90 where there should have
      been 84. When measuring a race, check that your instrument still shows it on a case you
      know is broken.

### 1.2 Real-model tests — all six now run and pass

- [x] **Re-run on 2026-09-09, idle GPU, serially and under `ctest -j2`.** Full suite **126/126**.

      | test | before | now |
      |---|---|---|
      | `27b_prefix_real` | failed (host-KV pin) | **passes** |
      | `27b_score_real` | failed (`a repeated score window inherited prior State/KV`) | **passes** |
      | `27b_dflash2_real` | failed (host-KV pin) | **passes** |
      | `27b_load_plan` | skipped | **passes** with `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` set |
      | `35b_a3b_real` | failed (host-KV pin) | **passes** |
      | `35b_a3b_dflash_real` | failed (host-KV pin) | **passes** |

      Three separate defects were behind those, and none of them was the model:

      1. **The pinned host KV cache could not be reserved beside any shipped model** (#45, and
         see §6). On Windows a pinned host allocation is charged against VRAM, so the 8 GiB
         default failed with the card full and tens of GiB of system RAM free.
      2. **`score_real` was a real correctness defect** — fp8/nvfp4/k8v4 attention corrupting
         every prefill output column but the last (#49, and see §5).
      3. **`load_plan` crashed silently** rather than reporting, because it had no top-level
         catch (#46), and the environment variable naming its second artifact was missing from
         the handoff block above.

      `NINFER_QWEN3_6_27B_NVFP4_WEIGHTS` is the one environment variable the handoff block at
      the top of this file was missing; the artifact has been on disk all along.

      **`27b_load_plan` now pins a hardware bound rather than skipping.** Its NVFP4 case plans
      a 1,024-token prefill chunk, and on sm_86 that cannot exist: NVFP4 weights need A4
      execution, which is sm_100a/sm_120a only, so the policy degrades to `A16Only`, and NVFP4
      A16 `linear_swiglu` is registered through T=16 and no further. The test asserts the
      refusal and its message, so a wider registration would fail it rather than pass silently.

- [x] **The maximum-configuration judgement call, resolved.** The tests pin layouts sized for a
      card this box does not have, and `NINFER_REAL_TEST_MAX_CONTEXT` already lowers the ceiling
      the 35B's `exercise_maximum_configuration` asks for. With the host-KV fix in place the
      *base* engine now fits without closing anything, so the remaining question — whether the
      base config should also honour the env var — is moot: it starts. Left as it is.

### 1.3 `docs/config-calculator.html` undercounts startup memory for speculative configurations — closed

- [x] Closed 2026-09-09 by the same work as §2b's first entry, which has the measured terms, the
      engine cross-checks and the scale of what was missing. The short version: the page reused the
      12 MiB no-speculation CUDA graph allowance for every mode and added no extra cache for any of
      them, and the entry's own guess at the fix — "a full port of `mtp_graph_profiles` /
      `dflash_graph_profiles` / `graph_topology_allowance` into the page's JS, nontrivial and
      piecewise" — turned out not to be needed. Measuring the engine's reported terms at three
      contexts per mode gave five constants per mode, one of which is a two-step function of
      context, and those reproduce the engine to 0.044% on MTP3 and exactly on DFlash-3.

      Worth keeping: the entry descoped this as "out of scope for a docs PR" and was right that
      porting the layout logic would have been. It was wrong that the alternative needed "fresh
      `ninfer_bench` measurements of the actual per-mode graph allowance" as though that were the
      expensive path — the whole matrix is ten minutes of load-and-report, because the engine
      already prints every term.

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

### Unblocked, and now owed work — closed

- [x] **Re-run the six real-model tests now that both artifacts exist** (§1.2). Done 2026-09-09.
      All six run and pass. `27b_score_real` was indeed hiding a genuine defect rather than an
      environmental skip, and it was worse than the message suggested: not State/KV inheritance
      but a data race corrupting every prefill output column but the last, in three of the six KV
      formats (#49, §5). The other four were failing on the pinned host-KV allocation (#45, §6),
      and `27b_load_plan` needed one environment variable that the handoff block had never listed.

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

- [x] **Speculative modes undercount memory by roughly 170 MiB, plus a per-token term.** Closed
      2026-09-09. Every term is now measured per mode by
      `scripts/sweeps/speculative-memory-terms.ps1`, and the undercount was far larger than 170 MiB
      at anything but the mildest mode and depth.

      | model | spec | graph | seq fixed | seq/token | kv ratio | kv extra/token |
      |---|---|---:|---:|---:|---:|---:|
      | 27B | none | 12 MiB | 158.7 MiB | 1/16 | 1 | 0 |
      | 27B | mtp3 (+head) | 86 MiB | 167.6 MiB | 1/8 | 1.06299 | 0 |
      | 27B | dflash2-7 (+head) | **288/384 MiB** | 266.2 MiB | 1/16 | 1 | 0 |
      | 35B | none | 12 MiB | 67.3 MiB | 1/16 | 1 | 0 |
      | 35B | mtp3 (+head) | 86 MiB | 72.6 MiB | 1/8 | 1.10078 | 0 |
      | 35B | dflash-3 (+head) | 160 MiB | 184.2 MiB | 1/8 | 1 | **4,096 B** |

      **The "one constant multiplier" in the old entry only ever fitted MTP.** MTP buffers extra
      draft tokens *in the KV format*, so its cost is multiplicative and identical across formats
      — +6.2988% per token on the 27B, +10.078% on the 35B, both confirmed against nvfp4 to within
      page rounding. DFlash v1 instead reserves a fixed scratch region per token, measured at
      **exactly 4,096 bytes** and format-independent: 32 MiB at 8,192 tokens, 64 MiB at 16,384,
      128 MiB at 32,768. Read as a ratio that is +38.8% on `int8` and +71.1% on `nvfp4`, which is
      what made it look format-dependent. DFlash2 costs nothing per token and pays entirely in a
      graph allowance that is itself a step in context: 288 MiB to 8,192 tokens, 384 MiB beyond.

      **Cross-checked against the engine's own refusal arithmetic, which the old entry noted had
      only ever been done for `--spec` unset:**

      | configuration | engine | page | error |
      |---|---:|---:|---:|
      | 27B int8 262,144, mtp3+head | 9,838,038,272 | 9,842,380,571 | +0.044% |
      | 35B int8 262,144, dflash-3 | 4,312,377,600 | 4,312,377,600 | **exact** |

      For scale, the model the page used before this: 611 MiB short on the first, and **1,289 MiB
      short (−31%)** on the second. "Roughly 170 MiB" was measured on MTP3 at a 40,960 context —
      the mildest combination of mode and depth in the matrix.

      User-visible effect: the page reported 262,144 tokens as the largest fitting context for
      *every* speculation mode on the 35B. DFlash-3 actually tops out at 240,192.

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

- [x] **The regression tests exist but nothing runs them.** Closed. #47 put
      `docs/config-calculator.test.mjs` in a GitHub workflow — it needs `node`, which no other test
      here does, so CTest was the wrong home for it. It now also covers the speculative path, which
      the old entry correctly said it did not: three golden cases against the engine's own refusal
      figures (no speculation, MTP3+head, DFlash-3), a check that DFlash's extra KV is a fixed
      4,096 bytes per token rather than a ratio, a check that DFlash2's graph allowance steps with
      context, and an invariant that no speculative mode may cost less than no speculation in
      either fixed or per-token memory — which is precisely the failure the old model had.

      `docs/config-calculator.render.test.mjs` is new and covers what none of that did: `render()`
      itself, over all 360 model x speculation x KV x context combinations, plus an assertion that
      the reported largest-fitting context actually responds to the speculation mode. Most of the
      page's code lives in `render()`, and a wrong property name there shows up only as a blank
      page in a browser.

- [x] **Active docs still contradict the six-format claim.** Fixed in #33, across four places. #32 fixed README's `Current limits`
      and `docs/perplexity.md`, but README's *opening* summary still says the FP8 E4M3 KV profile
      "is not" admitted on SM86, and `docs/cli.md` and `docs/serving.md` were not touched. All six
      formats are measured and working; the Blackwell restriction applies to FP8/NVFP4 *weights and
      activations*, not KV storage. Fix them together so the claim is consistent everywhere.

---

## 2c. Performance left on the table

Ordered by size of the prize. The first three entries were rewritten on 2026-09-09, when the
roofline finally acquired a denominator; read them before the rest.

- [x] **Dense decode runs at 65-66% of the card's memory bandwidth.** Closed by #50 and #51 — the
      observation was right that headroom exists, and wrong in both of its terms.

      *936.2 GB/s is not a ceiling anything reaches.* It is the advertised 384-bit GDDR6X at
      19.5 Gbps. `tools/hbm_bandwidth_probe.cu` was already in the tree and had never been
      pointed at this card. 4 GiB working set (683x L2), best of five: read 854.2 GB/s (91.3% of
      advertised), memset-write 863.3, kernel write 824.1, D2D copy 813.2. Decode streams
      weights and KV *in*, so 854.2 GB/s is its ceiling.

      *Resident is not read.* `weights_capacity_bytes` counts the vision tower, MTP and DFlash
      when unselected, the draft head with speculation off, and all but one row of the token
      embedding — 2.47 GB of the 27B that never moves per token.
      `tools/decode_byte_accounting.py` counts what a token actually touches, from the
      artifact's own object directory: **15.743 GB** on the 27B.

      | model | kv | depth | tok/s | achieved | % achievable |
      |---|---|---:|---:|---:|---:|
      | 27B dense | int8 | 4,096 | 37.44 | 595 GB/s | 69.6% |
      | 27B dense | int8 | 32,768 | 33.56 | 565 GB/s | 66.2% |

      So the dense path sits around two-thirds of what the card can deliver, not 66% of an
      unreachable number. Real headroom, roughly thirteen points, and still unprofiled.

- [x] **Nobody knows where the MoE sits at all, because the accounting does not exist.** Closed by
      #50. It exists now, and it changes which entry in this section matters most.

      An A3B MoE reads 1.943 GB of dense text weight per token plus 8 of 256 experts per layer —
      0.580 GB out of 18.556 GB of experts resident — for **2.522 GB per token**. Against the
      854.2 GB/s read ceiling:

      | model | kv | depth | tok/s | achieved | % achievable |
      |---|---|---:|---:|---:|---:|
      | 35B-A3B | int8 | 4,096 | 169.55 | 435 GB/s | 50.9% |
      | 35B-A3B | int8 | 16,384 | 163.39 | 440 GB/s | 51.6% |
      | 35B-A3B | int8 | 32,768 | 153.62 | 441 GB/s | 51.6% |

      **Flat to within a point across an eightfold change in cache depth**, which is a far
      cleaner signal than the dense model's gentle decay: whatever limits the MoE is not the
      cache and not the depth.

      This reverses the ordering this section opened with. The dense path has about thirteen
      points of headroom; **the MoE has about thirty**, on the model the release recommends. A
      gather of 8 scattered expert blocks per layer is the obvious suspect and is where the
      profile should start. `scripts/sweeps/decode-step-profile.ps1` (#51) is the tooling.

- [x] **Profile one decode step.** Done 2026-09-09 (#53), and it says the shortfall is inside the
      kernels rather than between them — on both models, but for different reasons.

      | workload | launches | GPU busy | wall | idle |
      |---|---:|---:|---:|---:|
      | 27B dense decode | 83,623 | 3,336 ms | 3,500 ms | 4.7% |
      | 35B-A3B decode | 57,770 | 717 ms | 849 ms | **15.5%** |
      | 27B dense prefill | 87,110 | 6,512 ms | 6,675 ms | 2.4% |
      | 35B-A3B prefill | 60,715 | 1,362 ms | 1,481 ms | 8.1% |

      Dividing the read set into *busy* time alone: 27B 601 GB/s (70.7% of achievable), 35B
      450 GB/s (52.7%). So closing every launch gap buys 4.7% on the dense path and **15.5%** on
      the MoE, and the rest is kernel efficiency.

      *These are the corrected figures.* #53 reported 4.5% and 12.6%, computed by summing each
      kernel's duration rather than taking the union of their `[start, end)` intervals. Decode
      launches across streams and concurrent kernels overlap, so summing double-counts the overlap
      and understates idle — found by CodePulse review on #51 and fixed in #60. The dense path
      barely moved (1.0% of overlap); the MoE's idle went 12.6% to 15.5%, which makes its launch
      overhead a fifth of its total shortfall rather than a quarter of it. Any future change to
      this script's arithmetic should be checked against synthetic overlapping intervals, which is
      how #60 verified the merge-sweep.

      **On the MoE the shortfall has an address.** Bytes attributed from the artifact inventory,
      instance counts fixing the mapping (129 rounds, 40 text layers, 8 of 256 experts):

      | kernel | bytes | time | achieved | % achievable |
      |---|---:|---:|---:|---:|
      | `sparse_moe_d3` (routed gate_up) | 8.91 MB | 23.4 µs | 381 GB/s | **44.6%** |
      | `sparse_moe_d4` (routed down) | 5.58 MB | 16.4 µs | 340 GB/s | **39.9%** |
      | `w8_k2048_decode` (gdn qkv_z) | 26.74 MB | 38.2 µs | 700 GB/s | 81.9% |
      | `q6_rowsplit_gemm_simt` (output head) | 397.31 MB | 595.6 µs | 667 GB/s | 78.1% |

      The expert kernels reach 40-45% of what the card can read while *contiguous* weight kernels
      on the same step in the same model reach 78-82%. The four `sparse_moe` stages are 38% of
      decode busy time. That is the gather of eight scattered expert blocks per layer, measured
      rather than suspected, and it is the single largest identified opportunity in this file.

      The 27B is a different shape: 86% of its busy time is four GEMV kernels that *are* the weight
      streaming, so its remaining thirty points are inside those.

      *Two instrument bugs on the way, both of which produced confident nonsense.* `-pg 4096,128`
      captures prefill and decode together and prefill dominates — the 27B's top kernel came back
      as 256 launches of `q4a8_swiglu`, which is 4 prefill chunks x 64 layers. Then decode reported
      **99.3% GPU idle** from 807 launches over 128 steps of a 64-layer model, about six kernels
      per step, because nsys records a CUDA graph replay as one opaque entity unless given
      `--cuda-graph-trace=node`. Both reasons are written into the script.

- [ ] **The routed prefill gate/up pipeline-depth threshold is upstream's RTX 5090 number,
      unmeasured on this card.** `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:314-315`
      (`kGateUpDeepJobsNum`/`Den`, currently 7/4) picks between a 2-stage ("Spread") and a 6-stage
      ("Packed") pipeline for the narrow routed gate/up kernel based on jobs-per-expert crossing
      this ratio. The comment above it is explicit that the crossing point was measured on
      upstream's server, with a fixture that walks the ratio continuously and disagrees with the
      operator microbenchmark by about 6x depending on L2 warmth. That crossing depends on L2 size,
      and an RTX 5090 has 16x the L2 of this box's RTX 3090 (96 MB vs. 6 MB) — there is no reason to
      expect 7/4 is where *this* card's Spread/Packed tradeoff actually flips. Wrong constant,
      wrong pipeline depth selected, not wrong output: `kExpertStages`/`kGateUpNarrowStages` still
      produce correct results either way, so this is a performance risk, not a correctness one.
      Re-measuring needs the same round-robin fixture the comment describes, built and run on this
      RTX 3090; no sweep script here currently drives sparse MoE prefill directly
      (`decode-step-profile.ps1` and the `scripts/sweeps/*.ps1` sweeps are all decode-only). Do not
      guess a replacement constant without that measurement.

- [ ] **Eight lanes buy 3.6x, not 8x, because no kernel amortises a weight read across 2-10 rows.**
      Measured 2026-09-09 on the 27B, int8 KV, no speculation, `--decode-tokens 512
      --max-context 8192`, via `tools/bench/run_serve_concurrency.py --suite decode-saturation`:

      | C | tok/s | vs C1 | batch | ms/round |
      |---|---|---|---|---|
      | 1 | 36.8 | 1.00x | 1.00 | 27.20 |
      | 2 | 61.1 | 1.66x | 2.00 | 32.73 |
      | 4 | 101.0 | 2.75x | 4.00 | 39.61 |
      | 8 | 132.9 | 3.62x | 8.00 | 60.18 |

      Batching itself works: `batch` is exactly C at every point and `row_rounds = C x rounds`, so
      one round serves the whole cohort and the 15.9 GiB weight read is amortised across it as
      intended. The loss is entirely in what a round costs. Against the 20.0 ms weight-bandwidth
      floor (15.9 GiB at the measured 854.2 GB/s), a C1 round spends 7.2 ms on everything that is
      not the weight read, and each added row costs 4.7 ms — 65% of that whole budget.

      **What it is not.** nsys at node-level graph tracing over a clean 150-round steady window
      (`profiles/conc-prof`, overhead negligible: 27.45 ms/round against 27.20 unprofiled) puts the
      GPU 95.2% busy at C1 and 97.6% busy at C8, so launch gaps explain none of it. The cohort is
      ~850 tokens deep on int8 KV, far too little traffic to matter, and host time stayed at
      0.3-0.5%.

      **What it is.** C1 and C8 run almost disjoint kernel sets. At C1, 89% of the round is GEMV
      specialisations (`q5_rowsplit_gemv`, `q4_linear_swiglu_gemv_pair`). At C8 those drop to zero
      instances and the round redistributes:

      | kernel | C1 ms/round | C8 ms/round |
      |---|---|---|
      | `q4_small_t_mma` | 0.00 | 15.45 |
      | `rowsplit_grouped_mma` | 0.00 | 13.78 |
      | `q5_rowsplit_gemm_simt_split2` (2 variants) | 0.00 | 17.10 |
      | `q5_rowsplit_gemv` (4 variants) | 13.61 | 0.09 |
      | `q4_linear_swiglu_gemv_pair` | 7.59 | 0.05 |
      | **all kernels** | **26.18** | **56.29** |

      36% of the C8 round runs in SIMT kernels that use no tensor cores and had zero instances at
      C1. That looks like a routing bug and **it is not one** — checked, so nobody re-checks it.
      Every dominant family was swept against its own schedule bench at T=1..16 and in each case
      the routed schedule is the fastest kernel that exists at T=8: `q5_linear_add` picks
      `split2_exact` (74.8 us at k=6144) over `mma_r64_c16` (101.4); `q4_q5_attn_input` picks
      `parent_split_fixed` (169.0) over `grouped_r32_c32_s4` (202.8); `q4_swiglu` picks
      `small_t_tiled`. The `{{2, 10}, Split2ExactResidual}` band in
      `src/ops/linear_add/q5/q5_linear_add_plan.cpp` carries no measured comment while
      every band from 11 upward does, but it turns out to be right anyway.

      The gap is kernel coverage, not dispatch. In the T=2..10 band the choice is between two bad
      shapes, and the crossover at 11 is simply where they cross:

      - `split2_exact` **grows with T** — 41.0 / 56.3 / 74.8 / 93.2 us at T=4/6/8/10 (k=6144),
        about +8.7 us per row. It barely amortises the weight read at all.
      - `mma_r64_c16` is **flat** — 101.4 us from T=4 all the way to T=16 — but flat at 4x the
        25.3 us that weight (21.6 MB at 854.2 GB/s) should cost to stream once. The T=1 GEMV
        reaches 41.0 us, or 62% of that floor.

      So the prize is a narrow-extent MMA kernel that keeps `mma_r64_c16`'s flatness at the GEMV's
      fraction of bandwidth. Perfect amortisation would hold the C8 round at C1's 26.18 ms and give
      305 tok/s instead of 132.9; the realistic share of that is whatever closes the 4x. Worth it:
      C8 is the profile the 27B release recommends for multi-user serving, and all three dominant
      families sit at 1.9-2.5x their T=1 cost for 8x the rows.

      Two notes on how this entry used to read. The numbers it quoted (C1 78.71 vs C8 250.26 tok/s)
      came from README's cohort table, which is end-to-end **with MTP3 enabled** — tokens per round
      is roughly `1 + 3 x acceptance` rather than 1, so dividing them into a bandwidth figure gives
      nonsense (144% of peak at C1, done naively). And it named two tools that cannot do this:
      `ninfer_bench` has no concurrency option at all, and `NINFER_OP_REPORT_STATS` is the Op-test
      *accuracy* reporter, not a timing one. The serving harness plus nsys is the path, and reaching
      it needed the Windows shutdown fix in #70 — the campaign aborted on a throughput/`request_done`
      reconciliation that was correct and unmeetable.

- [x] **Prefill has no roofline either.** It has one now (2026-09-09), and the naive version of it
      is a trap worth documenting.

      Prefill is compute-bound: a 1,024-token chunk does 2 x 25.02e9 x 1024 = 51.2 TFLOP of GEMM
      against 15.74 GB of weight reads, an arithmetic intensity above 3,000 FLOP/byte. So the
      denominator is MMA throughput, and `tools/tensor_core_rate_probe.cu` — already in the tree,
      never pointed at this card — measures what it actually is:

      | MMA shape | measured |
      |---|---:|
      | BF16 m16n8k16, f32 accumulate | **67.6 TFLOPS** |
      | FP16 m16n8k16, f16 accumulate | 148.6 TFLOPS |
      | INT8 m16n8k32 | **314.8 TOPS** |
      | INT8 m16n8k16 | 302.4 TOPS |
      | INT4 m16n8k64 | 601.3 TOPS |

      **There is no single denominator, and using one gives whatever answer you want.** The 27B's
      prefill mixes paths: the MLP GEMMs quantize activations to INT8 (`q4a8_swiglu_kernel`,
      `q5a8_add_kernel`, fed by `quantize_activations`) while the GDN projections stay BF16
      (`rowsplit_grouped_mma_kernel`, `q5_rowsplit_gemm_mma_kernel`). Time splits 54/46 between
      them. Taking `2 x params` over the whole model and dividing by one ceiling gives **96.0% of
      peak if you assume BF16** and **20.6% if you assume INT8** — for the same measurement. The
      first number is very nearly what this entry was going to claim.

      Per kernel, against the ceiling each one actually uses (from #53's capture, 4,096-token
      prefill, parameter counts from the artifact):

      | kernel | GFLOP/token | path | achieved | % of that ceiling |
      |---|---:|---|---:|---:|
      | `q4a8_swiglu` (mlp gate_up) | 22.82 | int8 | 97.4 T/s | **30.9%** |
      | `q5a8_add` (mlp down) | 11.41 | int8 | 89.4 T/s | **28.4%** |
      | `rowsplit_grouped_mma` (gdn value_z, query_key) | 8.05 | bf16 | 34.7 T/s | 51.4% |
      | `q5_rowsplit_gemm_mma` (gdn output) | 3.02 | bf16 | 36.8 T/s | 54.4% |

      So prefill is *not* near its ceiling. The MLP GEMMs, which are 68% of the FLOPs, run at
      about **30% of the card's INT8 tensor-core rate**, and the BF16 projections at about 52% of
      the BF16 rate. A well-tuned large GEMM on Ampere usually reaches 60-80% of a pure-MMA
      microbenchmark, so there is real room here — plausibly more than in decode.

      One caveat kept deliberately: these chunks are 1,024 tokens, which is skinny for a GEMM whose
      other dimensions are 34,816 x 5,120, and the percentage would look different at a larger
      prefill chunk. Sweeping `--prefill-chunk` against this ceiling is the obvious next step and
      has not been done.

- [ ] **Vision has essentially one performance number in the entire repository.** One acceptance
      figure for DFlash2 on the committed image fixture, and nothing about encode throughput,
      how it scales with resolution, or what the overlay residency costs in time rather than in
      bytes. It is an advertised feature of both models and it is unmeasured.

- [x] **DFlash2 makes the 27B *slower*, and no document says so.** Wrong, and closed 2026-09-09.
      On text DFlash2 is a **win at every draft count from 1 to 12**. The entry's own premise came
      from a benchmark fixture that cannot measure drafting at all.

      Measured through the serving path on the model's own generated prose — 27B DFlash2 artifact,
      INT8 KV, greedy, 256 generated tokens, mean of three runs, spread ≤0.2 tok/s except one row:

      | `--draft-tokens` | decode | vs none |
      |---:|---:|---:|
      | (none) | 37.7 | — |
      | 1 | 49.7 | +31.7% |
      | 2 | 56.4 | +49.5% |
      | 3 | 58.4 | +54.9% |
      | **4** | **59.1** | **+56.5%** |
      | 5 | 57.2 | +51.5% |
      | 6 | 48.4 | +28.3% |
      | 7 | 48.2 | +27.7% |
      | 8 | 47.6 | +26.2% |
      | 10 | 42.4 | +12.4% |
      | 12 | 40.6 | +7.6% |

      Three things follow.

      **The shipped recommendation is the wrong draft count.** `docs/cli.md` said seven; four is
      **18.5% faster**, and `docs/cli.md` now says four and carries this table. The entry guessed
      there might be "a crossover below 7 where DFlash2 turns positive" — there is a crossover at
      four, but DFlash2 was never negative to begin with.

      **MTP is still ahead, by far less than seven implied.** MTP3 with the draft head reaches
      62.4 tok/s on the same measurement, so the gap is 5% at DFlash2's best count rather than the
      36% this entry recorded.

      **`--lm-head-draft` does nothing for DFlash2.** Within noise of unset at every count.

      There is a clean cliff between five and six, 57.2 to 48.4, that looks like a block-geometry
      boundary rather than acceptance — worth a look if anyone wants the last few percent.

      *Why the old number was wrong, which matters more than the number.* It was measured on
      `bench/fixtures/bench_corpus.ids`, which holds 65,536 tokens drawn from **682 distinct ids**
      with **98.4% of bigrams repeated**, because it is a curated bank tiled to length. Both the
      corpus manifest and the generator's docstring asserted that "repetition fills length only and
      does not bias throughput" — true for prefill and plain decode, and false for anything that
      drafts. Swept through `ninfer_bench` this corpus reports **100% acceptance at every draft
      count from 1 to 12**, with decode rising monotonically to 159 tok/s because each round emits
      k+1 free tokens. Both claims are now corrected in place, and
      `scripts/sweeps/dflash2-draft-tokens-realtext.ps1` is the sweep that does not use it.

- [ ] **The three KV formats with the worst decode falloff were exactly the three that were not
      run-to-run deterministic — and that turned out to be a coincidence.** #49 fixed the
      non-determinism completely and the falloff did not move. Re-measured after it, and after the
      split-tier change in #52:

      | model | int8 | rk8v4 | fp8 | k8v4 | nvfp4 |
      |---|---:|---:|---:|---:|---:|
      | 27B dense | −10.4% | −7.9% | −12.5% | −15.6% | −18.0% |
      | 35B-A3B | −9.4% | −7.2% | −21.3% | −26.0% | −21.4% |

      The original table, kept because the ratios are what matter and they are unchanged:

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

      **That experiment was run (#52) and it refutes the hypothesis.** Extending the grant to
      `Nvfp4Group16` and `Fp8KeyNvfp4Value` made them *slower* on the 35B -- -3.1%/-2.5%/-2.3% at
      4,096/16,384/32,768 against an fp8 control that moved 0.3% -- and removing the bump from fp8
      as well gained 0.4%/0.5%/1.2%. More splits at depth is worse for every quantized format, the
      host's default tier was the better number, and the special case is now gone from both sides.
      The falloff is unchanged by it and still wants an explanation.
      Note it cannot be the whole story: fp8 *does* get 85 splits and still falls off 21% on the
      35B, so it is paying something else — plausibly `KeyBlock` pinned to 32 rather than the int8
      path's 64, forced by sm_86's shared-memory budget because the fp8 kernel keeps dequantized
      BF16 copies of K *and* V (`small_t_fp8.cu:26-29`).

      **What this entry originally claimed, wrongly — now settled.** It read the §5
      non-determinism and this falloff as one root cause — "a split reduction whose order varies, or an atomic
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

### Found by this cycle's profiling, and not previously on this list

The two entries below are the largest identified speed opportunities in the repository. Both come
out of #53's profile and #50's byte accounting, both have a measured gap against a measured
ceiling, and neither has had any optimisation attempted.

- [ ] **The MoE expert gather runs at 40-45% of the card's read bandwidth while contiguous weight
      kernels on the same decode step reach 78-82%.** This is the single biggest number left in
      this file. Measured (#53), bytes attributed from the artifact inventory:

      | kernel | bytes | time | achieved | % of achievable |
      |---|---:|---:|---:|---:|
      | `sparse_moe_d3_nine_warp` (routed gate_up, 8/256) | 8.91 MB | 23.4 µs | 381 GB/s | **44.6%** |
      | `sparse_moe_d4_nine_warp` (routed down, 8/256) | 5.58 MB | 16.4 µs | 340 GB/s | **39.9%** |
      | `w8_k2048_decode` (gdn qkv_z, contiguous) | 26.74 MB | 38.2 µs | 700 GB/s | 81.9% |
      | `q6_rowsplit_gemm_simt` (output head, contiguous) | 397.31 MB | 595.6 µs | 667 GB/s | 78.1% |

      The four `sparse_moe` stages are **38% of decode busy time** on the 35B, and that model sits
      at ~51% of achievable overall against the dense 27B's ~66-70%. A further **15.5%** of its
      decode window is GPU idle between kernels (corrected in #60 from the 12.6% first reported),
      which its mean kernel duration of 12.3 µs against the 27B's 40.3 µs explains: the MoE
      launches three times as many, three times shorter, so per-launch overhead lands three times
      as hard. Eight scattered expert blocks
      per layer per token is the shape; whether the cost is address divergence, L2 behaviour, or
      too little work per CTA to cover the latency is unknown — `ncu` would say, and is installed
      but has no `ncu.exe` at the expected path (see the tooling entry in §3).

      Closing even half the gap between the expert kernels and the contiguous ones is worth
      roughly 15-20% on the recommended model. Nothing here has been tried.

- [ ] **Prefill's MLP GEMMs run at ~30% of the card's INT8 tensor-core rate**, and the obvious
      excuse for that has been measured and ruled out.

      Measured (#53 plus `tools/tensor_core_rate_probe.cu`): `q4a8_swiglu` reaches 97.4 T/s and
      `q5a8_add` 89.4 T/s against a measured **314.8 TOPS** INT8 ceiling, while the BF16 GDN
      projections reach 51-54% of the measured **67.6 TFLOPS** BF16 ceiling. A well-tuned large
      GEMM on Ampere usually reaches 60-80% of a pure-MMA microbenchmark.

      **It is not a skinny-tile artifact.** The worry was that 1,024-token chunks against GEMM
      dimensions of 34,816 x 5,120 are too narrow to fill the MMA pipeline, so the percentage would
      improve with a bigger chunk. Swept on the 27B, int8 KV, an 8,192-token prompt, three
      repetitions each:

      | `--prefill-chunk` | prefill tok/s | vs 1,024 |
      |---:|---:|---:|
      | 256 | 1,139.7 | −6.9% |
      | 512 | 1,192.1 | −2.7% |
      | 1,024 (default) | 1,224.5 | — |
      | 2,048 | 1,234.9 | +0.8% |
      | 4,096 | 1,237.0 | +1.0% |

      Throughput **plateaus by 2,048**, and quadrupling the chunk from the default buys 1.0%.
      Nothing there closes a gap from 30% to 60-80%, so the shortfall is in the kernels and they
      are the target. The MLP pair is 68% of prefill FLOPs, which makes this the largest
      compute-side opportunity in the file.

      Two small things fall out. The default chunk of 1,024 is 1.0% off the plateau, so 2,048 is
      free throughput *if* the extra workspace is affordable — worth checking against the memory
      model in `docs/config-calculator.html` before changing a default. And 256 costs 6.9%, which
      is worth knowing for anyone tempted to shrink the chunk to save memory.

- [ ] **DFlash2 has a cliff between five and six draft tokens**, 57.2 to 48.4 tok/s (#65), a 15%
      drop where acceptance is still rising. Seven through twelve continue to degrade. It looks
      like a block-geometry boundary — seven forms block length eight, per `docs/cli.md` — so five
      may be sitting just under a tile edge that six crosses. Worth a look if the last few percent
      matter; four is already the recommended count and is on the good side of it.

- [ ] **`w8_pair` medium discards its schedule entirely on sm_86.**
      `w8_pair_gemm_splitk.cu:133` is `(void)schedule` under `NINFER_SM8X_COMPAT`, so every
      schedule runs the same chunked loop. PR #22 already removed the twelve now-identical route
      entries this made redundant, confirmed by identical 24.576 µs timings, so the *table* is
      honest. What is unknown is whether a genuinely sm_86-specific tiling would beat the generic
      chunking — nobody has written one to find out.

---

## 3. Measurement debt

- [ ] **Every route boundary in the repository was decided on cold-flush margins, which overstate
      the win.** `bench/ops/schedule_sweep.cuh` and its siblings call `measure_cold_launch`, which
      flushes 256 MiB through L2 before every repetition. That is the right default — these
      projections stream tens of MB against 6 MB of L2 and production is usually cold — but it is
      not the same as in situ, and the gap is not small. Measured on the q4 SwiGLU `{513,640}`
      decision (§5): the bench put c128 7.5% ahead of Materialized at T=576, while an nsys profile
      of the same schedules inside a real 600-token chunk put it 2.4% ahead, with **both** schedules
      slower per call in situ than in the bench (c128 4193 vs 3625 µs, Materialized 4295 vs 3927).
      The flush penalises Materialized's second weight pass more than a real prefill does.

      The sign was right in that case, and no boundary is known to be wrong. But every route
      comment in `src/ops/*/`*`_plan.cpp` quotes cold-flush microseconds, several of them at
      margins under 5%, and those margins are not speedups. Worth doing: take the boundaries whose
      measured margin is under ~5% — the `{2,10}`/`{11,16}` q5 crossover at T=10 (93.2 vs 101.4,
      8%) and the q4 SwiGLU `{2,24}`/`{25,40}` crossover at T=25 (464.9 vs 436.2, 6%) are the
      obvious two — and check each in situ with a profile rather than the bench. The method is in
      §5's entry; it is one nsys run per boundary.

      Do not "fix" the flush. Cold is the honest default for a first pass and warm numbers pick
      different winners, which is documented at the top of `schedule_sweep.cuh`. The point is that
      a narrow cold-flush margin is a reason to profile, not a decision.

- [ ] **The 315 W power cap has never been measured, and it bounds every number in this file.**
      `nvidia-smi` reports the limit at 315 W against a 350 W default (400 W maximum) and throttle
      reason `SwPowerCap` continuously through every sweep; the SM clock swings 1,665-1,755 MHz
      with temperature while memory holds at 9,501 MHz. It looks deliberate, so it has been left
      alone — and changing it needs an elevated shell, which this session did not have.

      Two separate questions, both open. **What does the cap cost?** Decode is memory-bound and the
      memory clock is not throttling, so it may be little — but `nvidia-smi -pl 350` and a re-run
      of `kv-decode-vs-depth.ps1` would say, and 10% of TDP is worth knowing about on a project
      whose goal is riding the bandwidth ceiling. **Should measurement runs lock clocks?**
      `nvidia-smi -lgc <clock>` would collapse the 3-5% between-process spread that made the 27B
      unreadable this cycle. Both need someone at an elevated prompt.

- [ ] **`compute-sanitizer` cannot launch this repository's test binaries.** It reports "Target
      application doesn't exist or is not a valid executable" for `ninfer_*_test.exe` — absolute
      path, `.bat` wrapper and direct `.exe` all tried — while launching a small standalone CUDA
      executable from the same shell without complaint. The test binaries are large (575 MB for
      `ninfer_qwen3_6_27b_score_real_test.exe`) and statically link everything, which is the
      obvious suspect but is unconfirmed.

      This cost real time: `initcheck` was the right tool for the fp8 corruption in #49 and could
      not be pointed at it, so the diagnosis came from a `--cuda-graph-trace` timeline and a
      hand-built reproducer instead. `ncu.exe` is also missing from the Nsight Compute 2025.1.0
      install directory, so per-kernel occupancy and L2 counters — exactly what the MoE expert
      gather in §2c needs next — are currently unavailable too. Worth half an hour to fix before
      the next kernel investigation, not during one.

- [ ] **The perplexity harness drifts 0.019% from the published figures and nobody knows why.**
      Re-measuring all six KV formats (#63) moved the three formats #49 does not touch by
      -0.009% to -0.019%: `int8` 4.343263 to 4.342425, `bf16` 4.343225 to 4.342517, `rk8v4`
      4.346811 to 4.346413. Same corpus, same window, same 261,167 scored tokens, and those code
      paths are byte-identical run to run. So something else differs between whenever the published
      numbers were taken and now — a build change, or a harness detail.

      It is small and it is systematic, which is exactly why it should be named: it is the floor on
      every perplexity comparison in this repository, and `k8v4`'s -0.008% "improvement" from #49
      sits underneath it and was reported as no change for that reason. Bisecting it would make
      future quality claims sharper by an order of magnitude.

- [ ] **`27b_load_plan` skips its DFlash2 binding matrix** for want of the *old* Qwen3.8 artifacts
      (`NINFER_QWEN3_8_27B_OLD_WEIGHTS`, `NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS`) and the NVFP4
      DFlash2 artifact. Everything else in that test now runs and passes (#46). This is the last
      real coverage gap in the real-model suite and it is an artifact problem, not a defect —
      check whether upstream still publishes those revisions before treating it as out of reach,
      which is the move that closed both §2 artifact entries last cycle.

- [ ] **DFlash2 corpus numbers on sm_86 — re-scoped, because the obvious way to get them is void.**
      Only single-prompt smoke numbers exist (text 20.0% / 2.38 tok-per-round, vision 85.7% /
      7.00). `docs/performance.md` deliberately does not reproduce upstream's tables because they
      are sm_120 — see the provenance banner there.

      **Do not produce these with `ninfer_bench` on the committed corpus.** #65 established that
      `bench/fixtures/bench_corpus.ids` is 65,536 tokens over 682 distinct ids with 98.4% of
      bigrams repeated, and DFlash2 reports *exactly 100% acceptance at every draft count from 1
      to 12* on it. Any acceptance or tok-per-round figure from that path is a statement about the
      fixture. The text throughput question is answered (#65, via the serving path on generated
      prose), but **acceptance and tokens-per-round are still unmeasured on realistic text**, and
      that is what this entry now wants. Either bake a diverse corpus with
      `make_bench_corpus.py --source-text` — `eval/corpora/perplexity-1m/` is real wikitext and
      pg19 prose sitting right there, though it needs a local HF tokenizer that this box does not
      have — or extend the real-text sweep to report acceptance. Vision acceptance is unaffected by
      any of this: it is measured on the committed image fixture, not the token corpus.
- [ ] **Speculative decoding is not bit-identical to greedy**, and it is not clear it should be.
      DFlash2 and MTP produce byte-identical output *to each other* and both diverge from the
      width-1 greedy path about a hundred tokens into the text fixture. Verification evaluates k+1
      columns in one pass where plain decode evaluates one, so reductions run in a different order
      and a near-tie argmax flips. MTP reproduces it exactly, so it predates the merge — but nobody
      has decided whether that is acceptable or worth pinning down.
- [x] **q4 SwiGLU `{513,640}` settled: it goes to c128, and the alternation is gone.** The band
      stayed on `Materialized` because the measurement could not separate the two — over four runs
      Materialized won T=576 three times, while c128 there ranged 4147–4959 µs (19.6% spread) and
      the margins outside the outlying run were ±2%.

      The blocker was the instrument, not the card. `ColdTiming` had carried `min_us` and `p95_us`
      all along and the sweep harness threw both away, printing only the median, so the spread that
      made the decision impossible was never visible in the table. Added `--spread` to
      `bench/ops/schedule_sweep.cuh` and to the q4 SwiGLU bench (which has its own arg loop), then
      re-measured at 31–51 repetitions on an idle card. c128 wins all three widths in four
      independent runs — 1.3–2.5% at 513, 1.2–7.5% at 576, 7.3–11.2% at 640 — and its *fastest*
      sample beats Materialized's fastest everywhere, by 7–9% at 640 with no overlap at all. Sign
      consistent 12 times out of 12. `{49,512}`, `{513,640}` and `{641,∞}` collapse into one
      `{49,∞}` c128 route and the table drops from seven entries to five.

      **Two things worth more than the route change itself.**

      *The bench overstates margins.* Confirmed in situ with nsys on a single 600-token chunk:
      Materialized costs 269.73 ms in `q4_rowsplit_gemm_mma` plus 5.16 ms in
      `silu_and_mul_dim0_split` across 64 layers = 274.89 ms, against 268.32 ms for the c128 pair
      kernel. c128 still wins, but by **2.4%, not 7.5%** — and both are slower per call in situ
      than in the bench (4193 vs 3625 µs for c128; 4295 vs 3927 for Materialized). The bench
      flushes L2 before every repetition and a real prefill does not arrive with a cold cache, so
      the flush penalises Materialized's second pass more than production does. **This applies to
      every band in every one of these route tables**, all of which were decided on cold-flush
      numbers alone. Nothing is known to be mis-routed because of it — the sign was right here —
      but no margin in those comments should be quoted as a speedup.

      *End-to-end it is invisible, and that is expected.* `pp600` measured 886.0–888.8 tok/s with
      c128 against 888.5–892.1 on Materialized: inside the run-to-run spread, because 6.6 ms of a
      ~660 ms prefill sits under the ±3% that unrelated kernels move between runs.

      One trap on the way: `--prefill-chunk 640` looks like the obvious way to exercise this band
      and does not touch it. `q4a8_swiglu` claims any width with `tokens >= 128 && tokens % 128 ==
      0`, and `--prefill-chunk` is required to be a multiple of 128, so **every full prefill chunk
      goes to the integer-activation kernel and never reaches this table at all.** The `{49,∞}`
      route is reached only by decode widths and by a prompt's ragged tail chunk. That is why
      `{513,640}` was simultaneously close and inconsequential for so long. The first end-to-end
      comparison here was run at chunk 640 and measured nothing, twice, before the profile showed
      `q4a8_swiglu_kernel` where `q4_linear_swiglu` was assumed to be.

---

## 4. Test-criterion calibration — closed

The §5 audit floored every BF16 gross bound at two rounding steps (#20). Two criteria were left
out because the BF16 floor argument was thought not to reach them. Measured on 2026-09-09 with
`NINFER_OP_REPORT_STATS=1` over the full case matrix, it reaches both — and neither was quite the
shape the entries described.

- [x] **`sparse_moe` sits at 0.92 of its limit** with `gross_relative_to_max_reference = 0.0`.
      Closed. The Op is not uniformly inaccurate; its gross error **steps at a route boundary**:

      | codec | T | max_abs | max_reference | BF16 steps |
      |---|---|---:|---:|---:|
      | q4+q5 | 1..46 | 4.85e-4 | 0.1778 | 0.70 |
      | q4+q5 | 47..4097 | 2.36e-3 | 0.1778 | **3.40** |
      | q4+q6 | 1..46 | 4.88e-4 | 0.1766 | 0.71 |
      | q4+q6 | 47..768 | 2.51e-3 | 0.1766 | **3.64** |
      | w8+w8 | 1..19 | 9.18e-4 | 0.2718 | 0.87 |
      | w8+w8 | 20..768 | 3.69e-3 | 0.2718 | **3.47** |

      So "the largest observed BF16 error in the tree — 3.64 steps against 0.09–1.53 everywhere
      else" is **one route above a T boundary**, and below that boundary this Op sits at 0.70–0.87
      like everything else. The question was "either that kernel is genuinely less accurate than
      every other BF16 Op, or its criterion measures something different"; the answer is that the
      Op has two routes with a 5x accuracy spread and the criterion was sized for the worse one.
      A different accumulation order over more terms is a property of the algorithm, not a defect.

      The real fragility was the bound's *shape*, not its size. `gross_absolute` alone at 4.0e-3
      does not scale with the data, so the same relative accuracy on a case whose max_reference is
      0.5 rather than 0.27 would produce ~6.8e-3 and fail spuriously. Six rounding steps of
      max_reference covers the measured 3.64 with headroom and still bounds a genuinely wrong
      element hundreds of times more tightly. Worst case now sits at **0.36** of its limit.
      `relative_l2` is untouched at 1.2e-2 against a measured 1.118e-2 — it is the criterion that
      constrains accuracy, and #20's convention leaves it alone.

- [x] **`gated_delta_net`'s state criterion sits at 0.87** and compares an FP32 output. Closed,
      and the reason it was excluded turns out not to matter: the error's *source* is BF16 anyway.

      | path | max_abs | max_reference | BF16 steps |
      |---|---:|---:|---:|
      | decode / small-T / batch update | 1.2e-8 .. 3.5e-8 | ~0.09 | **0.00** |
      | exact chunk / chunk-tail / two-chunk | 4.6e-4 .. 8.1e-4 | ~0.23 | 0.53–0.88 |

      The non-chunked paths are exact to eight decimal places. Everything above 1e-4 comes from the
      chunked recurrence, where the carried state crosses a BF16 intermediate at each chunk
      boundary — and just under one rounding step of the state's own magnitude is exactly what one
      such round-trip costs. The FP32 output dtype was the wrong thing to reason from.

      That makes the hand-picked 3.9e-3 the same mistake #20 was written to fix: it is about 1.0
      rounding step against an observed 0.88, so the bound and the error were the same quantity
      with 12% between them. It now uses `kBf16GrossRelativeFloor` like every other criterion the
      audit touched, and the worst case sits at **0.44**.

      `relative_l2` stays at 2.7e-3 against a measured 2.582e-3 — 0.92–0.96, which is tight, but
      loosening it would stop the chunked recurrence being checked at all.
---

## 5. Reproducibility

- [x] **fp8, k8v4 and nvfp4 causal attention are not run-to-run deterministic.** Closed by #49,
      2026-09-09, and it was not a tolerance curiosity — it was a data race corrupting output.

      Five quantized kernels called `dequant_k_tile()` immediately after `cp_wait<0>()` with no
      `__syncthreads()` between them. `cp_wait<0>()` retires only the *calling thread's*
      cp.async group, while `dequant_k_tile()` has every thread walk the whole tile, so each
      thread reads bytes another thread issued. The prologue in the same files pairs the two
      correctly and the lambda's own comment states the requirement; only the steady-state loop
      omitted it, and it omitted it in exactly the three storage families named here. bf16 and
      int8 pair them everywhere and were always byte-identical.

      What it cost, on the 27B through `EnginePurpose::CausalScoring`: `score_tokens` called
      twice on the same window disagreed on **all 1024** logprobs — median 0.2, mean 3.2, worst
      ~26. A token at -0.0013 in one run was -22.61 in the next: p about 1 becoming p about
      1e-10. Total logprob over a 200-token window ranged from -352 to -588 across five calls in
      one process, and again across processes. int8 and bf16 returned the identical checksum
      every time.

      **Why it hid for so long.** Greedy decode reads only the last column's hidden state.
      Prefill's other columns — the ones scoring and perplexity consume — are discarded there,
      so generation looked perfect and stayed byte-identical while the same kernels were
      producing garbage for every other position.

      After the fix, `ninfer_softmax_attention_test` run three times from one binary is
      byte-identical (15 differing `OP_ERROR_STATS` lines before), and `27b_score_real` reports
      `max_overlap_error=0` for fp8, nvfp4, k8v4 and int8 alike.

      *Ruled out on the way, so it is not re-investigated:* the `split >= active_split_count`
      early return in the small_t partial kernels skips `write_neutral()`, which §2c flags as
      safe only by coincidence. Writing the merge identity there changed nothing — 15 differing
      lines before and after. It is a latent hazard, not this defect.

- [x] **The published perplexity figures for fp8, k8v4 and nvfp4 were measured through the race
      above.** Re-measured 2026-09-09, twice each, with int8 and bf16 as controls whose code paths
      #49 does not touch.

      | format | published | re-measured | delta |
      |---|---:|---:|---:|
      | `int8` (control) | 4.343263 | 4.342425 | −0.019% |
      | `bf16` (control) | 4.343225 | 4.342517 | −0.016% |
      | `rk8v4` (control) | 4.346811 | 4.346413 | −0.009% |
      | `fp8` | 4.347181 | **4.344724** | **−0.057%** |
      | `k8v4` | 4.347596 | 4.347258 | −0.008% |
      | `nvfp4` | 4.358924 | **4.352201** | **−0.154%** |

      **All four re-measured formats are now bit-identical between passes** — `fp8` returned
      4.344723843631465 twice over 261,167 scored tokens. That is #49's determinism claim proven at
      the level that matters, not just at the Op-test level.

      The three formats #49 does not touch all drifted −0.009% to −0.019%, which is a harness or
      build offset against whenever the published numbers were taken, and is the floor on any claim
      from this comparison. `fp8` moved three times that and `nvfp4` eight to seventeen times it:
      real. `k8v4` at −0.008% sits *inside* the control band, so it did not measurably improve
      despite being one of the patched kernels.

      README, `docs/config-calculator.html` and `docs/rtx-3090-windows.md` all carry the new
      figures.

- [x] **fp8 and k8v4 are dominated on all three axes and nothing says so structurally.** Settled
      2026-09-09, and the premise was half wrong once the numbers were redone.

      **`fp8` is no longer dominated on all three axes.** Its re-measured perplexity, 4.344724,
      *beats* `rk8v4`'s 4.346413 — a reproducible 0.039%, and both come from the same run so the
      control drift cancels. It is still larger (33,024 B/token against 26,112) and still slower at
      depth (30.34 against 33.17 tok/s at 32K), so what it now offers is a genuine trade: 26% more
      KV memory and 8.5% of decode speed for 0.039% better quality. A bad trade for almost anyone,
      but a trade rather than a strict loss, and README says exactly that instead of "no niche".

      **`k8v4` is dominated, and comfortably.** 1.5% smaller than `rk8v4`, for 13% less decode
      speed at depth, the worst falloff of any format on the 35B (−25.9%), and slightly worse
      perplexity. No configuration makes that 1.5% worth having.

      Left fully available in `--help` either way: upstream parity is worth more than steering, and
      the README table now states the trade precisely enough that nobody needs steering.

- [x] **The speculative decode sweep measures acceptance, not depth, and cannot be read like the
      non-speculative one.** Cause found and named, 2026-09-09. The entry blamed the corpus for
      being "repetitive on a stretch" and prescribed "a corpus with realistic diversity, or many
      more repetitions". More repetitions would not have helped: the fixture is *structurally*
      unable to measure acceptance.

      `bench/fixtures/bench_corpus.ids` is 65,536 tokens drawn from **682 distinct token ids**,
      with **98.4% of its bigrams repeated**, because it is a curated bank rotated and tiled to
      length. A draft head predicts that perfectly. Swept through `ninfer_bench`, DFlash2 reports
      **exactly 100% acceptance at every draft count from 1 through 12**, and decode climbs
      monotonically from 37.6 to 159.7 tok/s because each round emits k+1 tokens for free. Not
      several rows at 100% — all of them.

      Worse, both the corpus manifest and `tools/bench/make_bench_corpus.py` asserted in writing
      that "repetition fills length only and does not bias throughput". That is correct for
      prefill and plain decode, which are token-count and bandwidth bound, and wrong for every
      speculative measurement. Both are corrected in place, and they were the reason this was read
      as a sampling problem rather than a fixture that cannot answer the question.

      The replacement is `scripts/sweeps/dflash2-draft-tokens-realtext.ps1`: the serving path on
      the model's own generated prose, greedy so each configuration produces identical text and the
      comparison is speed on the same output. That is what produced the DFlash2 result in §2c.
      Anything speculative measured through `ninfer_bench` on the committed corpus should be
      treated as void. The memory columns from such runs remain sound — they are read at load and
      do not depend on content.

---

## 6. Operational

- [x] **The pinned host-KV default is 8 GiB regardless of host RAM.** Closed by #45, 2026-09-09,
      and the framing was wrong: host RAM was never the constraint.

      On Windows/WDDM a pinned host allocation is mapped into the GPU's address space and
      charged against the card. Measured on this 24,576 MiB 3090, allocating N MiB on the device
      and then finding the largest pin that succeeds:

      | device resident | VRAM free | largest pin |
      |---:|---:|---:|
      | 15,360 MiB | 7,972 MiB | 8,192 MiB |
      | 17,408 MiB | 5,924 MiB | 6,656 MiB |
      | 19,456 MiB | 3,876 MiB | 3,840 MiB |
      | 21,504 MiB | 1,828 MiB | 2,816 MiB |
      | 22,528 MiB |   804 MiB | 1,536 MiB |

      Resident-device plus pinned-host lands within a few hundred MiB of the card's capacity
      every time. The failure is `cudaErrorAlreadyMapped`, not out-of-memory, and #25's
      diagnostic read it as "this is system RAM, not VRAM" — exactly backwards, which is what
      sent the investigation the wrong way for an hour.

      **Backing off does not work, and this is the part to remember.** One failed
      `cudaMallocHost` poisons every later one in the process. With 2,852 MiB free, 1,024 MiB
      succeeded twice; then a deliberate 8,192 MiB failure made 1,024, 256 and even **64 MiB**
      fail with the same error, and `cudaGetLastError` did not clear it. A halving retry loop was
      written and abandoned on that evidence. The size is now clamped before the first attempt,
      to free VRAM less 1 GiB and then halved, Windows only.

      **Still owed, and its own entry at the end of this section.**

- [x] **The unpinned downloaders cannot verify anything they fetch.** Closed by #48, 2026-09-09.
      `download-qwen38-27b.{sh,bat}` had in fact already pinned `18dfc887` in their URL — they
      simply verified nothing and resumed onto the final path. They now stage under a
      revision-scoped name and check size and SHA-256, matching `download-qwen36-27b`.
      `flake.nix` had been tracking `main` for the same model, so `nix run` and the shell script
      could fetch different artifacts; it is pinned to match. The local artifact every published
      27B number was measured against hashes to the pinned revision, so the pin also records
      which bytes those numbers describe. One downloader stays unverifiable by design —
      `download-qwen36-35b-v2` tracks upstream `main`, which is the whole point of it — and
      README now says so where people choose.

- [x] **`package-release-rtx4090-early1.ps1` has no Linux counterpart.** Closed by #47. The
      counterpart is written and the `windows_only` exemption list is deleted rather than left
      empty. The same PR made that loop accumulate its misses instead of exiting at the first,
      and added an executable-bit check read from the git index rather than the filesystem —
      which immediately found four scripts committed at 644, including
      `scripts/package-release-v090.sh`, the current release's own Linux packager.

- [x] **Binaries embed their build directory.** Half fixed, and the other half measured as not
      fixable. Closed 2026-09-09.

      `-ffile-prefix-map=${PROJECT_SOURCE_DIR}=.` is now set for C, C++ and (via `-Xcompiler`) the
      host half of CUDA translation units on GCC/Clang, which covers the Linux binaries and their
      ~200 occurrences of `/home/ash/ninfer-rel/src/...`. Verified under real Linux g++ on this
      box: `/tmp/ftest/sub/a.cpp` becomes `./sub/a.cpp` and the absolute prefix leaves `strings`
      entirely.

      **MSVC has no working equivalent and that is measured, not assumed.** It takes `__FILE__`
      from the path as written on the command line and CMake writes absolute ones; the usual
      suggestion, the undocumented `/d1trimfile:`, had *no effect* on `__FILE__` when tested
      against 14.44.35207 with an absolute source path. The Windows binaries keep their ~466
      occurrences. Device-side `__FILE__` from nvcc's own frontend is not covered either -- there
      is no documented flag for it.

- [ ] **The shipped `-maxctx` launchers ask for a host-KV pin they cannot have.**
      `run-qwen38-c1-maxctx` and `run-qwen36-35b-a3b-c1-maxctx`, both `.bat` and `.sh`, pass
      `--host-kv-mib 8192` explicitly, and both deliberately fill the card with KV. Per the
      measured table above, at that residency the largest pin that succeeds on Windows is a small
      fraction of 8 GiB — they could never have had it, before or after #45. Since #45 the request
      is clamped rather than fatal, so the launchers do work; they are asking for something
      impossible and the flag reads as though it were doing something.

      Decide what they should say. A realistic figure, or drop the flag and take the clamped
      default, or `--no-prefix-reuse` if prefix reuse is not worth any VRAM at maximum context —
      which is a real question at these residencies and has not been measured either. Whichever
      way, the four launchers should agree with each other and with README's launcher table.

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
| #37 | master did not compile | a lambda introduced in #30's review commit could not see `order`; nothing had rebuilt that TU, so every "125/125" since was measured against a binary the tree could no longer produce |
| #44 | the `T=112` graph-replay failure | not a kernel. `cudaMemcpy` out of pageable memory returns before the DMA lands, and the DMA rides the legacy stream that `cudaStreamNonBlocking` is exempt from |
| #45 | the pinned host-KV default | on WDDM a pinned host allocation is charged against **VRAM**; #25's diagnostic asserted the opposite. And a failed `cudaMallocHost` poisons every later one, so back-off is impossible |
| #46 | four real-model tests died as `0xc0000409` | no top-level catch, so `what()` never printed. `e06d7363` in a debugger is a C++ throw, not corruption |
| #47 | the Linux guard did not guard | an exemption list, a loop that exited at its first complaint, and an `-x` check that cannot see a mode-644 file on Windows or WSL — which had let the current release's own Linux packager sit at 644 |
| #48 | the 27B downloaders | already pinned, verified nothing; `flake.nix` disagreed with the shell scripts about which revision to fetch |
| #49 | fp8/nvfp4/k8v4 non-determinism | a missing `__syncthreads()` between `cp_wait<0>()` and a whole-tile shared-memory read. Corrupted every prefill output column but the last, which is why generation looked perfect |
| #52 | `SmallTMaximumSplits` | section 2c proposed extending the bump to nvfp4 and k8v4; measured, it made them 2.3-3.1% slower, and removing it from fp8 too gained 0.4-1.2%. The host was right and the device policy was wrong |
| #53 | where the MoE's bandwidth goes | the expert gather runs at 40-45% of achievable while contiguous weight kernels on the same step reach 78-82% |
| #50, #51 | the decode roofline | both terms were wrong: the ceiling is measured at 854 GB/s not 936, and the numerator is the read set not the resident set. Gave the MoE its first denominator |
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

**A correlation is not a mechanism, and this file said so twice before it mattered.** §2c noted
that the three formats with the worst decode falloff were exactly the three that were not
run-to-run deterministic, and wisely added "treat §5 as open on its own terms". #49 fixed the
non-determinism completely and the falloff did not move at all. Two real defects sharing a
population is not one defect.

**Check that your instrument still fires on a case you know is broken.** Two probes were built
for the `T=112` race and both were useless in opposite directions. A D2H read-back on the same
stream, ordered ahead of the kernel, made the race vanish entirely — the copy engine serialises
the in-flight H2D behind it. An early draft of the regression test cleared its counter with
`DeviceBuffer::fill` between the copy and the stream work, and that one synchronous runtime call
in the gap took 84 races out of 90 down to zero. Both looked like clean results.

**The bug you can see is the one that does not matter.** fp8 attention was corrupting every
prefill output column except the last, by up to 26 in logprob, for as long as anyone has been
measuring perplexity with it — and greedy generation stayed byte-identical throughout, because
greedy reads only the last column. `27b_score_real` was the only thing in the tree that looked,
and it had been skipping for want of an artifact that was sitting on the disk.

**"Resident" and "read" are different numbers and only one of them is a denominator.** Dividing
throughput into `weights_capacity_bytes` overstated the dense path by six points and returned
418% of peak on the MoE. The MoE figure had been in this file for a cycle, correctly labelled as
a non-result, and the fix was arithmetic over the artifact rather than any measurement.

Two from earlier cycles, still true:

**Fix the class, not the flagged line.** #18 was reported as one route and was five. #24 was
reported as one launcher and was four problems across six. Chasing the class is also what found the
`kv_cache_append` contract lines and the `AGENTS.md` wrong-target claim that nobody had flagged.

**A route table can be live and still wrong.** `q4_q5` had a tuned table nobody read beside a
hardcoded chain. `w8_pair` k=2048 had a table that *was* read, but whose distinctions did not exist
on this hardware. So `grep resolve_plan` is necessary and not sufficient — also grep the launchers
for `NINFER_SM8X_COMPAT`, and confirm in the sweep, where identical schedules print *identical*
times.

---

---

## This card is power-capped, which sets the floor on every measurement here

`nvidia-smi` reports the power limit as **315 W against a 350 W default** (400 W maximum), and
reports throttle reason `0x4` — `SwPowerCap` — continuously through every sweep. The SM clock
swings **1,665–1,755 MHz** as a result while the memory clock stays fixed at 9,501 MHz. It looks
deliberate rather than accidental, so it has been left alone; changing it needs an elevated shell
anyway.

**This is why within-run stddev lies.** `ninfer_bench -r 3` reports ±0.02–0.5 tok/s, which looks
like a tight measurement. The spread *between processes* on the 27B is 3–5%: int8 at a
4,096-token depth measured 37.44 tok/s in one run and 35.45 in another, on identical code, an hour
apart. The clock drifts with temperature across runs and no amount of repetition inside one
process sees it.

Three consequences worth internalising before quoting any number in this file:

- **Every performance comparison needs a control** — a configuration whose code path the change
  does not touch, measured in the same run. The `SmallTMaximumSplits` work (#52) is the worked
  example: the 35B's fp8 control held to 0.3% and made a 2.5% effect readable, while the 27B's
  control moved ±3% and made that model's numbers worthless. Without the control, six numbers
  looked like a result and three of them were noise.
- **Do not compare across sessions.** Interleave the variants you are comparing inside one sitting,
  or accept a 5% floor.
- **The 35B is the quieter instrument.** Its controls repeatedly held to ≤0.3% where the 27B's
  moved 3–5%, so a small effect should be measured there first.

Also worth someone's attention: 315 W is 90% of this card's default TDP. Decode is memory-bound and
the memory clock is not throttling, so the cost may be small — but it has never been measured.
`nvidia-smi -pl 350` and a re-run of `kv-decode-vs-depth.ps1` would answer it, and
`nvidia-smi -lgc <clock>` would collapse the 3–5% spread for measurement runs. Both need
elevation.

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

## Editing this file with a script, because I deleted two sections doing it

`s.index('

---', from)` looked like a safe way to find the end of an entry. There is no `---`
between §5 and §6, so it matched the one after §7 and silently removed both sections. It shipped in
#65 and was only noticed when a later edit could not find `## 6. Operational`; recovering it meant
`git show 17423c11:TODO.md`.

If you script an edit to this file, anchor the *end* of a replacement on the next thing you can
name — a heading, or the next `- [ ]` — never on a separator, and check `grep -c '^## '` before and
after. The same applies to `cmake --build --target A B`, which builds only `A` here and produced a
"verified" test result measured against a stale binary in the same session.

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
