# TODO

State as of 2026-09-09, at the end of a profiling pass that closed six more items and refuted five
of its own hypotheses. **34 items closed, 17 open.** If you are picking this up on different
hardware, read "Handing this off to another machine" below before anything else.

Released: **v0.9.0-rtx3090** (Windows + Linux). Full suite **126/126** on this box.

**One speedup shipped this pass, and it was small and cheap.** The GDN input projection was routing
everything from 7 to 32 columns through a 32-wide MMA tile; adding C8 and C16 tiles is worth
**+5.7% on the C8 serving profile** (3 of 3 runs, arms not overlapping) and +5.3% on DFlash2 at four
draft tokens. That is the only behaviour change; everything else this pass produced was measurement.

**The five biggest opportunities are all now located rather than suspected**, and three of them
turned out to be the opposite of what this file assumed:

- The **MoE expert gather** is latency-bound, not divergence-bound. 31.1 of 32 threads per warp are
  active, so there is no divergence to fix; and neither tiling nor batching helps, because the
  kernel is 1.17 machine-fulls of work and an MoE with per-token routing does not amortise its
  routed weight read across a cohort — 8 tokens touch ~57 distinct experts, not 8. The 35B's ~51%
  of achievable is what top-8-of-256 costs, not a bug.
- The **KV decode falloff** is 3.21x the per-key cost in the decode attention kernel — 5.89 ns
  against int8's 1.83 ns — and it belongs to the three formats that stage dequantized tiles in
  shared memory. `KeyBlock` was not the difference.
- **Prefill's MLP GEMMs** at 31% of INT8 peak are not limited by memory (compute-bound over their
  own memory floor by 4.3x), tile shape, or dequantization (1-3% of the call). What is left needs
  counters.
- **Eight lanes** buy 3.83x, and the gap is a narrow-extent kernel that reaches 24% of its
  weight-streaming floor. A narrower tile was built for it and lost.
- The **315 W power cap costs nothing** — +35 W buys +3% SM clock and 0% throughput, because memory
  clock never leaves 9,501 MHz at either limit.

Two pieces of tooling were believed broken and were not: `compute-sanitizer` (two copies installed,
and the older one reports `0 errors` without executing the binary) and `ncu` (a permission, not a
missing file). Both now work; see §3.

Every route table in the tree has been measured on sm_86 and every one of them has a schedule
bench. What remains is four kernel problems, some measurement debt, and two items needing hardware
or artifacts this box does not have.

---

## Handing this off to another machine

Written 2026-09-09 for an agent picking this up on different hardware. The section after this one
("Start here if you are new to this box") is about *this* box; read this one first, because it says
which of that still applies.

### What every number in this file is, and is not

All of it is one machine: **RTX 3090, `sm_86`, 24,576 MiB, CUDA 12.8, MSVC 14.44.35207, Windows 11,
board power capped at 315 W.** Three measured ceilings are used throughout and none of them is a
datasheet figure:

| ceiling | value | note |
|---|---:|---|
| achievable read bandwidth | **854.2 GB/s** | not the advertised 936.1; every roofline here divides by this |
| INT8 MMA | **314.8 TOPS** | `tools/tensor_core_rate_probe.cu` |
| BF16 MMA, f32 accumulate | **67.6 TFLOPS** | same probe |

**Re-measure those three first on new hardware.** Every percentage in §2c is a ratio against them,
and carrying them across cards would repeat exactly the mistake this fork found in upstream's
inherited tables.

**The 315 W cap does not need reproducing.** Measured both ways: at 350 W decode is 37.15 tok/s
against 37.16-37.55 at 315 W, and prefill 1,228.5 against 1,204-1,255 — so +35 W buys +3% SM clock
and 0% throughput. Memory clock never leaves 9,501 MHz at either limit, which is why. See §3.

### What definitely does not transfer

**Every route table in the tree was measured on `sm_86` and is expected to be wrong elsewhere.**
That is not a caveat, it is the main finding of the last two cycles: upstream's tables were
inherited from `sm_120` and were wrong here by up to 52.8%. The tables, each with its own schedule
bench under `bench/ops/`:

| route table | bench | boundaries |
|---|---|---|
| `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp` | `q4_linear_swiglu_schedule_bench` | 1 / 2..24 / 25..40 / 41..48 / 49..∞ |
| `src/ops/linear_add/q5/q5_linear_add_plan.cpp` | `q5_linear_add_schedule_bench` | 1 / 2..10 / 11..16 / 17..24 / 25.. / 104.. / 128..∞ |
| `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp` | `q4_q5_attn_input_schedule_bench` | see file |
| `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp` | `q4_q5_gdn_input_schedule_bench` | 1..6 / 7..8 / 9..16 / 17..32 / 33..64 / 65..∞ |
| `src/ops/linear_pair/w8/…` (`w8_pair`) | `w8_pair_schedule_bench` | see file |
| DFlash2 w8 | `w8_dflash2_schedule_bench` | see file |

Every one of those benches takes `--tokens`, `--repeat`, `--warmup` and **`--spread`**. Use
`--spread`: it prints `median min..p95` per cell, and a boundary is only decidable when the winner's
p95 clears the runner-up's min. A boundary in §5 went unresolved for a cycle because the harness
printed medians only.

Two rules learned the hard way about those tables, both with worked numbers in the files:

- **Narrowing a tile helps when padding is the cost and hurts when bandwidth is.** Adding C8/C16 to
  the GDN input projection won 9-11% and shipped; the identical change to `q5_linear_add` lost 6-32%,
  because that kernel is already at 24% of its weight-streaming floor and a narrower tile removes no
  work while halving the warps hiding the read.
- **Cold-flush margins overstate the win.** These benches flush 256 MiB through L2 per repetition,
  which is the right default, but a margin measured that way is not a speedup — one 7.5% bench
  margin was 2.4% in situ. §3 has the entry; confirm anything under ~5% with a profile.

### What must be re-established before any of this is reproducible

1. **Model artifacts.** Paths, sizes and which-is-which are in the next section. They are ~18-23 GB
   each and are not in the repo. The real-model tests take them by environment variable; the
   sweeps take `NINFER_MODEL_DIR`.
2. **Build environment.** Windows needs the VS 2022 BuildTools `vcvars64.bat` shell; the recipe is
   in the next section. `cmake --build --target A B` **silently builds only A** on this generator —
   pass one `--target` per name or you will measure a stale binary. That mistake produced a
   "verified" result against a binary four hours older than its source.
3. **GPU performance counters, if you want `ncu`.** On Windows they are administrator-only and
   `ncu` fails with `ERR_NVGPUCTRPERM`. Running from an elevated shell satisfies it per-run with
   nothing persistent and no reboot; `scripts/sweeps/admin-profile.ps1` refuses to start unelevated
   and collects everything that needs it in one pass. The persistent alternative is
   `HKLM\SOFTWARE\NVIDIA Corporation\Global\NvTweak\RmProfilingAdminOnly = 0` plus a reboot.
4. **Clock stability.** See below — this is the single biggest threat to reproducing any A/B here.

### Measurement hygiene, which is where the time actually goes

**This card drifts 3-5% between processes, and that is larger than most effects in this file.** Two
comparisons this cycle came out with opposite-signed drift (+2.9% then −3.8%) on code paths that had
not changed. Consequences, all of them load-bearing:

- **Interleave A/B arms within each repetition**, not all of one then all of the other. Build both
  binaries, alternate them inside the loop, and report the *paired* median. `scripts/` has no
  committed harness for this; the pattern is three lines of shell and it is not optional.
- **Keep a control** — a configuration whose code path did not change between the two arms — and
  normalise against it. In the GDN tile A/B, draft counts 4 and 5 stay on an unchanged route and
  calibrated the drift; without them the result was unreadable.
- **`nvidia-smi -lgc <clock>`** would remove this entirely and needs an elevated shell. It is the
  cheapest measurement improvement available and it is still not done — §3 carries it.
- Two identical-looking runs differing by <3% have measured nothing.

Three more that each cost hours here:

- **`nsys` needs `--cuda-graph-trace node`.** Decode replays a captured CUDA graph and the default
  records the replay as one opaque entity, which reported 99.3% GPU idle. The tell was 807 launches
  over 128 steps of a 64-layer model.
- **`ninfer_bench` has no concurrency option.** Multi-lane numbers come from
  `tools/bench/run_serve_concurrency.py`. An earlier entry claimed otherwise and sent the work the
  wrong way.
- **Never measure speculation on `bench/fixtures/bench_corpus.ids`.** It is 65,536 tokens over 682
  distinct ids with 98.4% of bigrams repeated, and DFlash2 reports *exactly 100% acceptance at every
  draft count from 1 to 12* on it. Any acceptance or tokens-per-round figure from that path is a
  statement about the fixture. Use the serving path on generated prose; `scripts/sweeps/dflash2-draft-tokens-realtext.ps1`
  is the pattern.

And one reassurance, because it saves a whole class of worry: **MMA tile choice does not perturb
perplexity.** Routing width 1024 in `q5_linear_add` from `C128` to `C64` moved the score rate
568.9 → 535.8 tok/s, so a different kernel demonstrably ran, and perplexity came back bit-identical
to twelve figures. Route changes are not a quality risk.

### Tooling traps specific to Windows, all hit this cycle

- **`compute-sanitizer` has two installed copies and one lies.** The 2024.1.0 copy under
  `CUDA/v12.4/` prints its banner and `ERROR SUMMARY: 0 errors`, exits 0, and **never executes the
  binary** — a clean report having checked nothing. The v12.8 copy works, and is what
  `compute-sanitizer` on `PATH` resolves to. Never call it by an absolute v12.4 path.
- **`-k 'regex:a|b'` in PowerShell** is parsed as a pipeline before `ncu` sees it. One
  `--kernel-name` pattern per invocation.
- **`apps/ninfer` writes its logs UTF-16LE.** `grep` finds nothing in them; decode first.
- **`Get-FileHash` is unavailable** under the `powershell` on this box, which silently breaks the
  hashing step of `dflash2-draft-tokens-realtext.ps1` (non-fatal, one error per iteration).
- **Copying a built `.exe` out of `build-ninja/apps/`** breaks DLL resolution: exit 127 with an
  empty log, which reads exactly like a model failure. Keep A/B binaries in that directory under
  different names.
- **Heredocs through this agent harness mangle `\n` and `\v`** inside Python strings. A `\v` in a
  path became a literal vertical tab in committed Markdown. Write files with a file-write tool
  rather than a heredoc when the content contains backslash escapes.

### Work in flight that does not transfer

- **Two `ncu` profiles are outstanding on this box** and a new machine simply re-runs
  `scripts/sweeps/admin-profile.ps1 -SkipPower` there: section 2 (the contiguous-kernel baseline the
  MoE's 40-45% figures are compared against, which failed first time on the PowerShell `|` bug) and
  section 3 (the prefill MLP GEMMs, where memory, tile shape and dequantization are all already
  ruled out and only issue rate / shared-memory feeding / occupancy remain). §2c has both.
- **The perplexity drift bisect is scoped and unstarted**: 392 commits, ~nine steps at 20-30 minutes
  each, about four hours of exclusive GPU time. It is a repository question rather than a hardware
  one, but the baseline figures it bisects against were measured here. §3 has the constraints,
  including that three candidates are already eliminated.
- **`ninfer_qwen3_6_27b_score_real_test`** and its siblings need `NINFER_*_WEIGHTS` set or they skip.
  A skip is not a pass, and under `compute-sanitizer` a skip surfaces as
  *"Target application terminated before first instrumented API call"*.

### The seventeen open items, and the next concrete action for each

Ordered by expected value, not by section.

| item | § | next action | needs |
|---|---|---|---|
| MoE expert gather is latency-bound | 2c | split the shared W8 path out of the 9-warp block, as `sparse_moe_d3_path_tiled_kernel` already does for multi-token; then confirm the 2.2x over-fetch on d4 with `dram__bytes_read.sum` | kernel work; `ncu` for the confirm |
| Prefill MLP GEMMs at ~30% of INT8 peak | 2c | run `admin-profile.ps1` section 3 and read issue rate vs shared-memory feeding vs occupancy | one elevated run |
| Eight lanes buy 3.6x (3.83x since the GDN tiles) | 2c | the 4x between `mma_r64_c16` and its weight-streaming floor is the target; **not** by narrowing the tile, which was tried and lost | kernel work |
| KV decode falloff, 3.21x per-key on fp8 | 2c | attack the per-key dequant-into-shared cost; `rk8v4` proves the floor is reachable without a shared arena | kernel work |
| DFlash2 5→6 cliff | 2c/3 | remainder after the GDN tiles is the grouped kernel sitting at 27% of its own weight-streaming floor | same kernel as the KV item |
| Routed prefill pipeline-depth threshold 7/4 | 2c | sweep the constant through the product on this card — the method the source comment endorses over the operator fixture | GPU time only |
| `w8_pair` medium discards its schedule on sm_86 | 3 | write one genuinely sm_86-specific tiling and bench it against the generic chunked loop | kernel work |
| sm_86 fallback constants chosen to fit | 2c | sweep the alternatives at the eleven `NINFER_SM8X_COMPAT` sites | GPU time only |
| Cold-flush margins overstate wins | 3 | profile the narrow boundaries in situ; the method is in §5's q4 SwiGLU entry | GPU time only |
| Perplexity drift 0.019% | 3 | the scoped four-hour bisect | exclusive GPU time |
| Speculative decoding not bit-identical to greedy | 3 | decide whether it should be; the divergence is a reduction-order effect in k+1-column verification and MTP reproduces it, so it predates DFlash2 | judgement, not measurement |
| DFlash2 corpus acceptance on real text | 3 | bake a diverse corpus with `make_bench_corpus.py --source-text`, or extend the real-text sweep to report acceptance | a local HF tokenizer, which this box lacks |
| `27b_load_plan` DFlash2 binding matrix | 3 | needs the *old* Qwen3.8 artifacts and the NVFP4 DFlash2 artifact | artifacts nobody has |
| DFlash2 + multi-GPU expert offload | 2 | genuinely blocked | a second GPU |
| No committed interleaved-A/B harness | 3 | ~40 lines in `tools/bench/`; the pattern is written out in the item | nothing |
| `Get-FileHash` missing breaks a sweep's hash column | 3 | swap for `certutil -hashfile` | nothing |
| Host memory pressure invalidates 35B runs | 3 | give the other sweeps the 24 GiB guard `moe-prefill-pipeline-depth.ps1` has | nothing |

Two of those seventeen are hard-blocked on things no amount of work here provides (a second GPU, and
artifacts that no longer exist). One is a judgement call rather than a measurement. The remaining
fourteen are all actionable — three of them need no GPU at all, and four are kernel work on two
closely related problems: narrow-extent weight streaming in `q5_linear_add`/`gdn_input`, and
dequant-into-shared in the quantized attention kernels.

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

Every item in the previous version of this list is now closed. What follows is what a fresh agent
should do first, and the ordering is by expected value rather than by section. The full
seventeen-item table with what each one needs is in "Handing this off to another machine" above.

1. **Run `scripts/sweeps/admin-profile.ps1 -SkipPower` from an elevated shell.** Two `ncu` profiles
   are outstanding and both are cheap: the prefill MLP GEMMs (§2c — memory, tile shape and
   dequantization are all already ruled out, so only issue rate, shared-memory feeding and
   occupancy remain, and those need counters) and the contiguous-kernel baseline the MoE gather's
   40-45% figures are compared against. This is minutes of GPU time and it unblocks the largest
   compute-side item in the file.
2. **Lock clocks for measurement: `nvidia-smi -lgc 1500`** (§3, elevated). The card drifts 3-5%
   between processes, which is larger than most effects here — it forced every A/B this cycle to
   interleave its two binaries within each repetition and carry an unchanged control. This is the
   cheapest improvement to every future measurement in this repository and it is one line.
3. **Two related kernel problems, and they are the real prizes** (§2c). Both are narrow-extent
   weight streaming and both have a measured target:
   - `q5_linear_add`'s `mma_r64_c16` sits at **24%** of what its weight costs to stream once, flat
     from T=4 to T=16. Closing that is most of the eight-lane gap. **Not** by narrowing the tile —
     that was built and lost 6-32%.
   - the quantized attention kernels pay **3.21x** int8's per-key cost (5.89 ns against 1.83 ns),
     which is the whole KV decode falloff. `rk8v4` reaches the flattest curve of all six formats
     with no shared arena at all, so the floor is demonstrably reachable.
4. **Split the shared W8 path out of the MoE's 9-warp block** (§2c). The source already names this
   — `sparse_moe_d3_path_tiled_kernel` exists for it and says so — and it is what the 44.6%/58.6%
   "no eligible warp" reading measures. Do not chase launch geometry or batching first: both were
   measured this cycle and neither helps, because the kernel is 1.17 machine-fulls of work and an
   MoE does not amortise its routed weight read across a cohort.
5. **Sweep the routed prefill pipeline-depth constant through the product** (§2c). `7/4` is
   upstream's RTX 5090 number and an RTX 5090 has 16x this card's L2. Sweeping the constant is the
   method the source comment endorses over the operator fixture, which disagrees with the server by
   6x. GPU time only, no new code.
6. **The perplexity drift bisect** (§3), when four hours of exclusive GPU time are available. Three
   candidates are already eliminated and the current value is exactly reproducible, so the
   remaining work is mechanical.

Two of the seventeen open items are hard-blocked — one on a second GPU (§2), one on artifacts that
no longer exist (§3) — and one is a judgement call about whether speculative decoding should be
bit-identical to greedy rather than a measurement (§3). Everything else is actionable, and three of
the newest items (§3: the A/B harness, a hash call, and a memory guard) need no GPU at all.

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

      Those C8 figures predate the narrow GDN tiles of §3, which lifted C8 to **140.8 tok/s, 3.83x**
      without moving C1/2/4 (widths 1, 2 and 4 stay on the independent route). Everything below is
      still measured against the 3.62x profile, and the gap it describes is unchanged in kind.

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

      **The narrow-extent MMA kernel was built and it is worse. Measured 2026-09-09.** Added an
      `R64C8` to `q5_linear_add` — `BlockCols` 8 with `WarpCols` 8, so one warp column, 4 warps and
      128 threads against C16's 8 and 256 — on exactly the reasoning above. It loses to `c16` at
      every width tested (us, medians of 21-31, k=6144):

      | T | 2 | 4 | 6 | 8 | 10 | 16 |
      |---|---:|---:|---:|---:|---:|---:|
      | `c8` | 131.1 | 116.7 | 115.7 | 108.5 | 133.1 | 126.0 |
      | `c16` | 114.7 | 103.4 | 102.4 | 102.4 | 103.4 | 95.2 |
      | `split2_exact` | 36.9 | 44.0 | 56.3 | 76.8 | 96.3 | — |

      6-32% worse than `c16`, and still far behind `split2_exact` below 11. Same story at k=17408.
      Reverted; the route table is unchanged and the schedule is not registered. The numbers are
      recorded in `src/ops/linear_add/q5/q5_linear_add_plan.cpp` beside the table so it is not
      retried.

      **Why it fails here and works elsewhere is the transferable part.** The same change succeeds
      on the GDN input projection — C8 and C16 beat C32 by 9-11% there, and that shipped. The
      difference is what dominates each kernel. The GDN projection's cost is padded MMA work, so
      cutting `BN` cuts real work. `q5_linear_add` is already weight-read-bound: `c16` sits at 24%
      of the 25.3 µs its weight costs to stream once, so cutting `BN` removes no work and halves
      the warps available to hide the read. **Narrowing a tile helps when padding is the cost and
      hurts when bandwidth is** — and this Op is the second kind.

      So the 4x between `c16` and the weight-streaming floor is real and still worth having, but it
      is not a tile-width problem and the prize stated above is not reachable that way. What
      `rk8v4` proves in §2c's KV entry applies here too: something reaches a much larger fraction of
      the floor at narrow extents, so the ceiling is not the obstacle — but the mechanism is not
      tile geometry.

      **Part of this is now attributed.** §3's DFlash2 cliff entry chased the same kernel from the
      other direction and landed on `rowsplit_grouped_mma_kernel`, which is 13.78 ms of the 56.29 ms
      C8 round here. It is the GDN input projection, and at C8 it runs the `{{7, 32}}` route from
      `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp` — a **32-wide** MMA tile whose cost is
      set by padded width, so eight live columns pay for thirty-two. The per-instance cost confirms
      it: 0.300 ms at width 8 against 0.311 ms at width 7, near-identical where a live-token-scaled
      kernel would differ by an eighth. So a `GroupedMixedMmaR64C8`/`R64C16` is wanted by both
      entries, and neither the wider direct route nor a routing change gets it — that was measured
      and refused, see §3.

      **The 35B MoE is now measured on the same curve, and it scales worse: 3.22x at eight lanes.**
      164.4 / 273.2 / 400.0 / 529.7 tok/s at C1/2/4/8, same harness and settings. That is the
      opposite of what "the MoE is work-starved at one token, so a cohort should fill the machine"
      predicts, and the reason is worth carrying here: **a dense model amortises its weight read
      across the cohort, an MoE with per-token routing does not.** Each token brings its own top-8
      of 256, so the expected distinct routed experts per round goes 8.0 / 15.8 / 30.5 / **57.4** —
      7.18x the routed weight bytes for 8x the tokens. Only the shared expert and the dense layers
      amortise at all. §3's MoE entry has the full working.

      So the 3.6x here and the 3.22x there have different causes and want different fixes, which is
      worth knowing before anyone treats "batched decode should be nearly free" as a general rule.
      On the dense 27B it nearly should be, and the gap is the narrow-extent kernels below. On the
      MoE it should not be, and no kernel work changes that.

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

- [x] **Vision now has numbers. Closed 2026-09-09**, and the headline is that the encoder is not
      the expensive part. `scripts/sweeps/vision-encode-throughput.ps1`; 27B, int8 KV, one image per
      request, two measured repetitions per point. `apps/ninfer` reports the Vision stage separately
      from text prefill, so encode cost is read directly rather than subtracted out.

      | side | image tokens | encode (resident) | tok/ms | encode (overlay) | overlay penalty | text prefill | prefill / encode |
      |---:|---:|---:|---:|---:|---:|---:|---:|
      | 224 | 85 | 15.1 ms | 5.65 | 36.5 ms | +143% | 206 ms | 13.7x |
      | 448 | 217 | 27.2 ms | 7.98 | 50.3 ms | +85% | 316 ms | 11.6x |
      | 672 | 462 | 56.0 ms | **8.26** | 85.0 ms | +52% | 568 ms | 10.2x |
      | 896 | 805 | 112.0 ms | 7.19 | 128.0 ms | +14% | 964 ms | 8.6x |
      | 1120 | 1246 | 181.5 ms | 6.87 | 228.0 ms | +26% | 1,100 ms | 6.1x |
      | 1344 | 1785 | 300.0 ms | 5.95 | 342.0 ms | +14% | 1,700 ms | 5.7x |
      | 1568 | 2422 | 475.0 ms | 5.10 | 513.0 ms | +8% | 2,100 ms | 4.4x |

      **Prefilling the image tokens costs 4.4-13.7x what encoding them does.** That is the number
      to know before optimising anything here: at 1024px a single image is ~173 ms of encode against
      ~945 ms of text prefill, so the Vision tower is ~15% of time-to-first-token and the other 85%
      is the ordinary prefill path already covered by §2c's prefill entries. Vision needs no special
      optimisation attention until that changes.

      **Encode throughput peaks at 672px and falls 38% by 1568px** — 8.26 down to 5.10 tok/ms.
      Against the 672px peak, encode time is superlinear in tokens: 1.01x at 672, 1.21x at 1120,
      1.40x at 1344, **1.64x at 1568**. That is the shape attention inside the tower predicts, and
      it means very large images are charged twice — more tokens, each more expensive. Small images
      are also inefficient (1.48x at 224px) but for the opposite reason: 85 tokens does not fill the
      machine.

      **Overlay residency costs a roughly fixed ~30 ms per request, not a proportional one, and
      saves 250.7 MiB.** Runtime reservation is 848.0 MiB resident against 597.3 MiB overlay. The
      penalty is +143% at 224px and **+8% at 1568px** because the tower crosses PCIe once per
      request whatever the image size. So overlay is close to free on large images and expensive on
      small ones — the opposite of the intuition that a bigger image costs more to overlay. The
      release launchers default to overlay, which these numbers support for the image sizes anyone
      actually sends.

      **Encode is exactly linear in image count**, and the overlay penalty is paid once per request
      rather than once per image. Overlay, 1024px fixtures:

      | images | tokens | encode | tok/ms | vs 1 image |
      |---:|---:|---:|---:|---:|
      | 1 | 1,045 | 173.0 ms | 6.04 | 1.00x |
      | 2 | 2,071 | 352.0 ms | 5.88 | **2.03x** |
      | 4 | 4,123 | 691.5 ms | 5.96 | **4.00x** |

      So batching images into one request is neither better nor worse per token than sending them
      separately, except that it amortises the one-off overlay crossing. Nothing here needs a
      per-image cache.

      One measurement trap the script documents: **resolutions are not free-form.** The
      preprocessor snaps to a multiple of the merged patch size, so nearby resolutions can produce
      identical token counts and a naive sweep shows plateaus that look like measurement error. The
      sides above are multiples of 112 for that reason, and the token count is printed beside every
      row so a plateau is visible rather than mysterious.

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

- [ ] **The three KV formats with the worst decode falloff are exactly the three that stage
      dequantized tiles in shared memory — and the falloff is 3.21x the per-key cost.** Located
      2026-09-09; the mechanism is below. #49 fixed the non-determinism that first drew attention to
      the same three formats, and the falloff did not move — but they are siblings rather than a
      coincidence, because the missing barrier sat around exactly that shared staging. Re-measured after it, and after the
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

      **Located and quantified 2026-09-09, and the sub-hypothesis above is wrong.** nsys with
      node-level graph tracing on the 27B, `-pg P,128` at P = 4,096 and 32,768, filtered to the
      decode attention kernels (`small_t`, 4,225 launches in both runs so per-launch figures are
      directly comparable) and away from the prefill `prompt` kernels that otherwise dominate a
      `-pg` capture:

      | format | 4,096 keys | 32,768 keys | growth for 8x the depth |
      |---|---:|---:|---:|
      | `int8` | 62.65 µs/launch | 115.24 µs/launch | **1.84x** |
      | `fp8` | 87.06 µs/launch | 255.96 µs/launch | **2.94x** |

      So the falloff is in the decode attention kernel, it is not subtle, and it separates into two
      distinct components:

      - a **fixed** part — fp8 already costs 1.39x int8 at 4,096 keys, +24.4 µs per launch before
        depth enters;
      - a **per-key** part — marginal cost per additional key is **1.83 ns for int8 against 5.89 ns
        for fp8, a 3.21x ratio**. This is the falloff. It is per-key work, which is what
        dequantizing every key block into shared memory looks like, and it is why the gap widens
        with depth (1.39x at 4K keys, 2.22x at 32K).

      Both formats scale far better than the 8x a serial-in-keys kernel would give, because
      split-K absorbs depth — so this is not a parallelism failure, it is a cost-per-key one.

      **`KeyBlock` is not the difference.** This entry guessed fp8 was "paying something else —
      plausibly `KeyBlock` pinned to 32 rather than the int8 path's 64". Read the dispatch:
      `small_t.cu`'s final `else` branch, which is the one decode takes (`TokenTile <= 3`), launches
      `<8, 2, 32, false>` — **`KeyBlock = 32` for int8 too**. The 64-wide case appears only under
      `TokenTile >= 6`, which is a prefill width. There is no 32-against-64 asymmetry at decode to
      explain anything.

      **What does separate the six formats is `DynamicArena`.** int8, rk8v4 and bf16 all decode with
      `DynamicArena = false` and zero dynamic shared memory. fp8, k8v4 and nvfp4 all decode with
      `DynamicArena = true`, staging dequantized tiles in a shared arena:

      | format | dynamic shared per block | blocks/SM that allows | 27B falloff | 35B falloff |
      |---|---:|---:|---:|---:|
      | `int8` / `rk8v4` / `bf16` | **0** | unbounded by shared | −10.4 / −7.9 / — | −9.4 / −7.2 / — |
      | `nvfp4` | 24 KiB (`3 x 32 x 256`) | 4 | −18.0% | −21.4% |
      | `k8v4` | 44 KiB (`11 x 32 x 256 / 2`) | 2 | −15.6% | −26.0% |
      | `fp8` | 48 KiB (`6 x 32 x 256`) | 2 | −12.5% | −21.3% |

      That is the **same three-way split as §5's non-determinism**, and it means the correlation this
      entry opened by calling "a coincidence" is not one. Both symptoms come from the same
      architectural choice — dequantize-into-shared — because #49's missing barrier sat between
      `cp_wait<0>()` and `dequant_k_tile()`, i.e. around exactly that staging. The non-determinism
      was not causally upstream of the falloff (fixing it moved nothing), but the two are siblings
      rather than coincidence, which is a more useful thing to know.

      **Occupancy alone does not order them, so it is necessary and not sufficient.** nvfp4 uses
      half fp8's dynamic bytes and gets twice the blocks per SM, yet has the *worst* 27B falloff of
      the six. So block count is not the whole mechanism; the per-key dequantization work is the
      part that matches, and that wants confirming with `ncu` on the three quantized kernels — now
      possible, see the closed tooling entry in §3 and `scripts/sweeps/admin-profile.ps1`.

      What this is worth: **if `nvfp4` decoded on `rk8v4`'s curve it would be the outright best
      format** — 45% smaller than INT8 with no speed penalty — rather than the compromise it is.
      The target is the 3.21x per-key ratio, and `rk8v4` proves it is achievable, because it packs
      4-bit values exactly as `k8v4` does and yet has the flattest curve of all six without a
      shared arena at all.

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

- [ ] **The MoE expert gather is latency-bound, not divergence-bound or bandwidth-bound.** Still
      the biggest number left in this file, but it now has a cause. `ncu` counters collected
      2026-09-09 via `scripts/sweeps/admin-profile.ps1` (elevated; the counters are
      administrator-only on Windows and that was the whole blocker), 35B, int8 KV, decode, clocks
      locked at 1,500 MHz, `--graph-profiling node`:

      | | `d3_nine_warp` (gate_up) | `d4_nine_warp` (down) |
      |---|---|---|
      | duration | 28.45 µs | 25.18 µs |
      | memory throughput | 425.9 GB/s (**49.9%** of achievable) | 487.5 GB/s (**57.1%**) |
      | DRAM throughput | 46.79% | 53.56% |
      | compute (SM) throughput | 35.69% | 36.55% |
      | **avg. active threads per warp** | **31.11 / 32** | **29.06 / 32** |
      | L1/TEX hit rate | 31.67% | 87.58% |
      | L2 hit rate | 5.89% | 15.96% |
      | **schedulers with no eligible warp** | **44.57%** | **58.63%** |
      | eligible warps per scheduler | 1.62 (of 8.28 active) | 1.13 (of 10.25 active) |
      | achieved / theoretical occupancy | 60.16% / 93.75% | 78.91% / 93.75% |
      | **block limit: registers** | **5** | **5** |
      | block limit: shared memory | 12 | 7 |
      | registers per thread | 36 | 40 |
      | waves per SM | **1.25** | 5.00 |

      This entry listed three candidate causes. The counters settle all three.

      **Address divergence — ruled out.** 31.11 and 29.06 active threads per warp out of 32. The
      8-of-256 expert selection happens at *block* granularity, so lanes within a warp still walk
      one expert's weights contiguously. There is no lane divergence to fix, and that was the
      leading hypothesis.

      **L2 behaviour — real, inherent, and not a bug.** 5.89% and 15.96% hit rates are close to
      zero reuse, but an expert's weights are read once per token and there is nothing to hit. No
      tuning recovers this; it is what top-8-of-256 costs.

      **Too little work in flight — confirmed, and this is the answer.** Neither DRAM (46.79/53.56%)
      nor SM (35.69/36.55%) is anywhere near saturated, which is the textbook signature. The
      schedulers say it directly: **no warp is eligible to issue 44.6% of cycles on d3 and 58.6% on
      d4**, with only 1.1-1.6 eligible warps per scheduler against 8-10 resident. Warps are there
      and stalled, not absent.

      **The binding constraint is the 9-warp block, and it is structural.** `Block Limit Warps` is
      **also 5** — the same as `Block Limit Registers` — so registers are tied with the warp limit,
      not binding beyond it, and cutting them buys nothing on its own. sm_86 allows 48 warps per
      SM; a 9-warp block gives `48 / 9 = 5` blocks and 45 resident warps, which is exactly the
      93.75% theoretical the counters report.

      Nine warps is not a tuning choice. In both kernels warps 0..7 each take one of the top-8
      routed experts through `RoutedCodec`, and **warp 8 takes the shared expert** through
      `W8Codec` — `kTopK + 1`. Eight warps would have nowhere to put the shared expert. So the
      6.25% of theoretical occupancy lost to 45-of-48 warps is the price of top-8-plus-shared, and
      it is the small part anyway.

      **What actually costs d3 is the wave tail, and the arithmetic matches to within 2 points.**
      512 blocks (one per `kIntermediate` column) over 82 SMs at 5 blocks each is 410 slots: one
      full wave of 410, then a second wave only 102 blocks deep, i.e. **25% full**. Average
      occupancy across the two waves is `(410 + 102) / (2 x 410) = 62.4%` against a measured
      **60.16%**. d3's occupancy shortfall is that tail and essentially nothing else. d4 launches
      2,048 blocks (5.00 waves), where a tail of the same absolute size is a fifth as costly, and it
      duly reaches 78.91%.

      **And there is a load imbalance inside the block that the source already names.** The shared
      W8 path is heavier than a routed Q4 path, but the block cannot retire until all nine warps
      finish, so eight completed routed warps sit holding registers and warp slots while warp 8
      drains. That is the 44.6% / 58.6% "no eligible warp" directly, and it is not a guess:
      `sparse_moe_d3_path_tiled_kernel` in the same file exists for this reason and says so —
      *"Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
      heavier shared W8 path from holding eight completed routed warps resident."* That kernel is
      written for the multi-token path and takes a `tokens` argument; single-token decode does not
      use it.

      So the order to try things, revised — and the first item is a refutation of the obvious fix.

      1. **Launch geometry does not help, and no block shape does.** d3's 512 blocks over 82 SMs at
         5 blocks each is 410 slots, which reads like a fixable tiling problem. It is not: total
         work is 512 columns x 9 warp-paths = **4,608 warp-units against the card's 3,936 warp
         slots**, so the kernel is only **1.17 machine-fulls of work** and every partitioning gives
         two waves. Efficiency `N / (slots x ceil(N/slots))` comes out at 62.4% for 9-warp blocks,
         58.5% for 3-warp, 58.5% for 1-warp. There is no scheduling trick that makes a 1.17-full
         kernel efficient. **d3 is not badly written; it is too small for this card at one token.**

      2. **Batching does not rescue it either — measured, and this refuted a prediction.** If the
         gather were merely work-starved, a cohort should fill the machine and the MoE should scale
         *better* with concurrency than a dense model. Measured 2026-09-09, mtp0, int8 KV, 512
         decode tokens, both models through `run_serve_concurrency.py`:

         | C | 35B MoE tok/s | vs C1 | 27B dense tok/s | vs C1 |
         |---|---|---|---|---|
         | 1 | 164.4 | 1.00x | 36.8 | 1.00x |
         | 2 | 273.2 | 1.66x | 61.1 | 1.66x |
         | 4 | 400.0 | 2.43x | 101.0 | 2.75x |
         | 8 | 529.7 | **3.22x** | 132.9 | **3.62x** |

         The MoE scales *worse*, and the reason is structural rather than a kernel defect. **A dense
         model amortises its weight read across the cohort — the same bytes serve every lane — but
         an MoE with per-token routing does not.** Each token brings its own top-8 of 256, so the
         expected number of distinct routed experts a round must read grows almost linearly with
         the cohort: 8.0 at C1, 15.8 at C2, 30.5 at C4, **57.4 at C8** — `7.18x` the routed weight
         bytes for `8x` the tokens. Only the shared expert and the dense layers amortise. The
         marginal cost per added row bears it out: 1.29 ms/row on a 6.08 ms base for the MoE (21%
         of base) against 4.71 ms/row on 27.20 ms for the dense model (17%).

         This is worth stating plainly because it is the opposite of the usual intuition about
         batching, and it means **the 35B's ~51% of achievable is not a bug to be fixed by
         batching or by tiling.** It is what top-8-of-256 routing costs on a card whose per-layer
         MoE work is a single machine-full.

      3. **Split the shared path out of the block.** This one still stands, and the source already
         names it: the shared W8 path is heavier than a routed Q4 path, the block cannot retire
         until all nine warps finish, and eight completed routed warps sit holding warp slots while
         warp 8 drains — which is the 44.6% / 58.6% "no eligible warp" reading directly.
         `sparse_moe_d3_path_tiled_kernel` in the same file exists for exactly this and says so:
         *"Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep
         the heavier shared W8 path from holding eight completed routed warps resident."* It is
         written for the multi-token path and takes a `tokens` argument; single-token decode does
         not use it. This does not add work to the machine, so item 1 does not apply — it removes a
         serialisation inside a block that is already resident.

      4. **Over-fetch.** ncu measured 12.12 MB of DRAM traffic on d3 and 12.27 MB on d4 where #50's
         artifact inventory attributes **8.91 MB** and **5.58 MB** of useful weight bytes — 1.36x
         and **2.20x**. If real, d4 moves over twice the bytes it needs and that is worth more than
         anything else here. Confirm with `dram__bytes_read.sum` before optimising against it,
         since the attribution and the ncu run are separate measurements.

      Registers are *not* on that list, which is the correction: an earlier draft of this entry
      claimed `Block Limit Registers = 5` was the constraint and that 32 registers per thread would
      give 7 blocks per SM. At 288 threads it would give 7 by the register rule, but the warp rule
      caps it at 5 regardless, so the change would measure as exactly nothing. Read both limits
      before believing either.

      What is left after all that is items 3 and 4 — a block-level serialisation and a possible 2x
      over-fetch — not the 15-20% "close half the gap to the contiguous kernels" this entry used to
      promise. That framing compared an 8-of-256 gather against kernels that stream contiguous
      weights, and items 1 and 2 are why that comparison was never going to close.

      Not yet collected: the contiguous-kernel reference from the same elevated run.
      `admin-profile.ps1` passed `-k 'regex:a|b'` and PowerShell parsed the `|` as a pipe, so that
      half produced only an error. Fixed in the script; re-run it to get the local baseline these
      percentages are compared against.

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

      **Two more explanations ruled out 2026-09-09, both analytically, which narrows this to the MMA
      pipeline itself.** Working from the chunk-640 profile — `q4a8_swiglu` takes 1,212.20 ms over
      512 calls, so 2.368 ms per call at the 27B's `34,816 x 5,120` MLP shape with 640 columns:

      | | |
      |---|---:|
      | work | 228.2 GOP |
      | weights read | 94.7 MB (4-bit codes plus one FP16 scale per 64) |
      | activations | 47.8 MB |
      | achieved | **96.4 TOPS = 31%** of the 314.8 TOPS ceiling |
      | compute floor at the ceiling | 0.725 ms |
      | memory floor at 854.2 GB/s | 0.167 ms |

      **Not memory.** The kernel is compute-bound over its own memory floor by **4.3x** at this
      shape, so no amount of bandwidth work touches it. Worth stating because "30% of peak" invites
      the assumption that something is starving.

      **Not dequantization.** Unpacking every one of the 178M 4-bit codes costs 0.020 ms at 2 int
      ops per code, 0.040 ms at 4, and 0.080 ms at 8 — **1%, 2% and 3% of the 2.368 ms call**,
      against the 3090's 17.8 T int-op/s CUDA-core rate. Even a pessimistic unpack cannot account
      for a 69% shortfall. This was the natural next hypothesis after the tile-shape one and it is
      also wrong.

      So the gap is inside the MMA pipeline: issue rate, shared-memory feeding, or occupancy, and
      those three need counters to separate. `scripts/sweeps/admin-profile.ps1` now collects them
      (section 3 of that script, `ComputeWorkloadAnalysis` and `InstructionStats` alongside the
      usual occupancy and scheduler sections) — it needs one elevated run, since GPU performance
      counters are administrator-only on Windows.

      Two small things fall out. The default chunk of 1,024 is 1.0% off the plateau, so 2,048 is
      free throughput *if* the extra workspace is affordable — worth checking against the memory
      model in `docs/config-calculator.html` before changing a default. And 256 costs 6.9%, which
      is worth knowing for anyone tempted to shrink the chunk to save memory.

- [ ] **DFlash2's cliff between five and six draft tokens is a GDN input-projection route
      boundary, and the fix is a narrower MMA tile.** Found 2026-09-09. Not the block geometry this
      entry guessed at.

      The cliff reproduces exactly — 27B DFlash2 artifact, greedy, `--max-new 256` on a prose
      prompt, four repetitions each within ±0.1 tok/s. Converting to round cost is what makes it
      readable, since acceptance and rate move together:

      | k | tok/s | tok/round | ms/round | step |
      |---|---|---|---|---|
      | 3 | 59.1 | 2.55 | 43.1 | — |
      | 4 | 59.5 | 2.95 | 49.6 | +6.4 |
      | 5 | 59.2 | 3.07 | 51.9 | +2.3 |
      | 6 | 50.0 | 2.97 | 59.4 | **+7.5** |
      | 7 | 48.2 | 2.93 | 60.8 | +1.4 |
      | 8 | 47.1 | 3.45 | 73.2 | +12.5 |

      **Accepted tokens per round is flat at ~3.0 from k=4 onward** (2.95, 3.07, 2.97, 2.93, 3.45),
      so nothing past four draft tokens pays for itself — every column beyond it is cost. That is
      the real justification for the recommendation of four, which `docs/cli.md` already gives.

      The 5→6 step itself is a schedule switch, confirmed by nsys per-round diff (83 rounds at k=5,
      86 at k=6, both deterministic): kernel time per round goes 58.22 → 66.21 ms, +7.99, matching
      the +7.5 from the throughput arithmetic. `q4_rowsplit_gemm_simt` (5.17 ms) and
      `q5_rowsplit_gemm_simt_split4` (6.43) vanish and `rowsplit_grouped_mma_kernel` (15.82)
      appears, on ~51 instances per round — the 27B's GDN layer count, not its 17 attention layers.
      That is `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp`:

      ```
      {{1, 6},  IndependentDirectFixed},
      {{7, 32}, GroupedMixedMmaR64C32},
      ```

      Verification width is k+1, so k=5 is the last count inside the direct route and k=6 is the
      first to cross into a **32-wide** MMA tile whose cost, as that file's own comment says, is set
      by padded width rather than live tokens. Width 7 pays for 32 columns.

      **The obvious fix does not work, and that is the useful part.** Both `launch_q4` and
      `launch_q5` accept T up to 15 with a dedicated R8C8 route for 5..15, so extending the direct
      band looks free. Measured `{{1, 15}}` / `{{16, 32}}` end to end and normalised against k=4/5
      (unchanged in both tables, and they calibrate a ~3% between-process drift): **−3.6% at width
      7, +1.3% at width 8, −23.8% at width 9, −13.2% at width 11.** The `{1,6}` boundary is right.
      Width 8 is the one place the direct path is competitive, which is the R8C8 tile fitting
      exactly; 9 needs two passes and collapses. Recorded in that file so it is not retried.

      **What to build instead: a `GroupedMixedMmaR64C8` or `R64C16`.** The grouped tile wins at
      width 7 *despite* padding 7 columns into 32 — so the MMA path beats the SIMT direct path by
      more than 4.6x of wasted width, and the prize is keeping that efficiency without the dead
      columns. Two independent measurements want the same kernel: this cliff, and §2c's 8-lane
      entry, where a C8 decode cohort spends **13.8 ms of a 56.3 ms round in this exact kernel at
      width 8** (0.300 ms per instance at width 8 against 0.311 at width 7 — near-identical, which
      is what padded-width-dominated cost looks like). Fixing it pays twice.

      One tooling gap was behind all of this: this Op had no schedule bench, which is why its
      boundary carried no measurement for so long. Written now, and it produced the two narrow
      tiles in the entry below — which take back a third of this cliff (k=6 goes from 47.9 to 51.2
      tok/s, +5.3% paired, 6/6 runs). The cliff is smaller but still there: the remaining gap is
      that the grouped kernel runs at 27% of its own weight-streaming floor at any width, and no
      tile choice changes that.

- [x] **The GDN input projection now has a schedule bench, and it bought two narrower tiles.**
      Closed 2026-09-09. This was the one route table here with no schedule bench, which is why its
      boundary carried no measurement while being implicated in two separate slowdowns.

      `bench/ops/q4_q5_gdn_input_schedule_bench.cu` plus a
      `q4_q5_gdn_input_execute_schedule` seam (the pattern `q4_linear_swiglu` and `w8_pair` already
      use: the real dispatch minus its plan-matches-problem check, so every schedule can be timed
      at every width). First result: **every existing boundary in that table is correct** — 6/7,
      32/33 and 64/65 all sit exactly where the curves cross.

      Second result: the table was missing two tiles. `GemmCfg` takes the tile width as `BN`, so
      `R64C8` and `R64C16` are instantiations, not new kernels. Measured, cold, medians of 31 with
      `--spread`, each winner's p95 below the runner-up's min:

      | T | independent | c8 | c16 | c32 |
      |---|---|---|---|---|
      | 6 | **188.4** | 240.6 | 247.8 | 290.8 |
      | 7 | 330.8 | **238.6** | 242.7 | 267.3 |
      | 8 | 319.5 | **236.5** | 243.7 | 267.3 |
      | 9 | 486.4 | 504.8 | **243.7** | 267.3 |
      | 16 | — | 506.9 | **242.7** | 267.3 |
      | 17 | — | 750.6 | 495.6 | **269.3** |
      | 32 | — | 999.4 | 491.5 | **264.2** |

      So c8 wins 7..8 by 10.7-11.5% and c16 wins 9..16 by 8.8-9.2%, each collapsing one column past
      its own width because a second pass costs a whole extra weight read. The table is now
      `{1,6} independent / {7,8} c8 / {9,16} c16 / {17,32} c32 / {33,64} c64 / {65,∞} c128`, and
      `tests/ops/test_gdn_input_proj.cpp` gained widths 8, 9 and 17 so every new boundary is
      exercised on both sides.

      End to end on DFlash2, interleaving the two binaries within each repetition so thermal drift
      lands on both arms — which it had to, because the card moved 3-5% between processes and that
      is larger than the effect:

      | k | width | base | c8/c16 | paired median | positive |
      |---|---|---|---|---|---|
      | 6 | 7 | 47.9 | 51.2 | **+5.3%** | 6/6 pairs |
      | 8 | 9 | 44.4 | 45.2 | **+3.2%** | 4/6 pairs |

      **Do not read the 11% as evidence that padding was the main cost.** Dropping `BN` from 32 to
      8 removes 75% of the padded MMA work and buys 11%, because this Op streams ~55 MB of weights
      (4,096x5,120 q4 plus 12,288x5,120 q5) whose 64.4 µs at 854.2 GB/s no tile choice changes. c8
      at 236.5 µs is **27% of that floor**. The padded work was a minor term all along, and the
      remaining 3.7x is what is actually worth chasing — the same conclusion §2c's 8-lane entry
      reaches about `mma_r64_c16`, from a different Op.

      **The C8 serving cohort — the other workload this kernel dominates, and the recommended
      multi-user profile — gains 5.7%.** Measured after the fact, same interleaving, 27B/mtp0/int8
      at `--concurrency 8 --decode-tokens 512`:

      | rep | base | c8 tile | |
      |---|---|---|---|
      | 1 | 133.8 | 140.8 | +5.2% |
      | 2 | 131.6 | 139.1 | +5.7% |
      | 3 | 131.0 | 142.8 | +9.0% |

      Positive 3 of 3 and **the two arms do not overlap at all** — base tops out at 133.8, tiles
      bottom out at 139.1. That is better than the "low single digits" this entry first predicted
      from `~25% of the round x ~11% of the kernel`, so something else improved alongside it; the
      prediction was a lower bound rather than an estimate.

      §2c's eight-lane curve moves with it: C1/2/4 are untouched (widths 1, 2 and 4 all stay on the
      independent route) so the curve is now **36.8 / 61.1 / 101.0 / 140.8 tok/s, or 3.83x at eight
      lanes** against the 3.62x measured before these tiles existed.

- [ ] **`w8_pair` medium discards its schedule entirely on sm_86.**
      `w8_pair_gemm_splitk.cu:133` is `(void)schedule` under `NINFER_SM8X_COMPAT`, so every
      schedule runs the same chunked loop. PR #22 already removed the twelve now-identical route
      entries this made redundant, confirmed by identical 24.576 µs timings, so the *table* is
      honest. What is unknown is whether a genuinely sm_86-specific tiling would beat the generic
      chunking — nobody has written one to find out.

---

## 3. Measurement debt

- [ ] **There is no committed harness for an interleaved A/B, and every comparison in this file
      needed one.** The card drifts 3-5% between processes, so measuring all of arm A then all of
      arm B is not a comparison — two runs this cycle came out with opposite-signed drift (+2.9%
      then −3.8%) on code paths that had not changed. The working pattern, hand-rolled three
      separate times this cycle:

      1. build both binaries and place them **inside `build-ninja/apps/`** under different names,
         because an executable copied elsewhere fails DLL resolution with exit 127 and an empty
         log, which reads exactly like a model failure;
      2. alternate the two inside one repetition loop, not one arm then the other;
      3. report the **paired** median — the median of per-repetition ratios — and how many pairs
         were positive, not the ratio of the two medians;
      4. keep a control configuration whose code path is identical in both arms, and normalise
         against it. In the GDN tile A/B, draft counts 4 and 5 stay on an unchanged route and are
         what made the result readable.

      Worth ~40 lines in `tools/bench/` taking two executable paths and a command template. It
      would also stop the next person rediscovering that the ratio of two medians and the median of
      ratios disagree here by more than the effects being measured. `nvidia-smi -lgc` (§3) reduces
      the need for this but does not remove it, and needs an elevated shell.

- [ ] **`scripts/sweeps/dflash2-draft-tokens-realtext.ps1` calls `Get-FileHash`, which does not
      exist under this box's `powershell`.** It errors once per iteration — 26 iterations, 26 error
      blocks in the output — and the `content_sha256` column comes out empty. Non-fatal, and the
      throughput numbers it prints are unaffected, but the hash column is the entire mechanism by
      which that script lets a reader check whether two configurations produced identical text,
      which its own header says is the point. Replace it with `certutil -hashfile <path> SHA256`, or
      call `[System.Security.Cryptography.SHA256]::Create()` directly, either of which works here.

- [ ] **Host memory pressure can silently invalidate any 35B measurement on this box, and did.**
      `vmmemWSL` holds up to **27 GiB of the 64 GiB** of host RAM while WSL is running, and the 35B
      artifact is 22.8 GB — so loading it contends directly with WSL's footprint, the machine pages,
      and timings become noise. A first attempt at the pipeline-depth sweep was abandoned for
      exactly this. `wsl --shutdown` reclaims it.

      `scripts/sweeps/moe-prefill-pipeline-depth.ps1` now refuses to start below 24 GiB free, and
      **every other sweep here should grow the same guard** — none of them currently checks, and a
      paging run produces plausible-looking numbers rather than an error. Related: this box's C:
      drive sits at 98% full (45 GB free) and hit *zero* bytes earlier in the cycle, which failed a
      build with `C1085 ... No space left on device`. Two artifacts were byte-identical duplicates
      (19 GB recovered) and WSL crash dumps held another 12 GB.

      **Reading today's numbers in light of this:** the paired interleaved A/Bs are robust to it,
      because both arms met the same conditions — the GDN tile results (+5.7% C8 serving, 3 of 3
      non-overlapping; +5.3% DFlash2 k=6, 6 of 6 pairs) and the route-boundary bench sweeps are
      safe. Single-shot end-to-end figures taken during the same window are the ones to treat as
      provisional and re-run under the guard.

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

- [x] **The 315 W power cap costs nothing measurable. Closed 2026-09-09.** Measured both ways with
      `scripts/sweeps/power-and-clocks.ps1`, and then at 350 W from an elevated shell:

      | | 315 W (cap) | 350 W (default) |
      |---|---|---|
      | decode `tg1024` | 37.16-37.55 tok/s | **37.15 tok/s** |
      | prefill `pp8192` | 1,204-1,255 tok/s | **1,228.5 tok/s** |
      | decode SM clock | 1,500-1,515 MHz median | 1,545 MHz median |
      | prefill SM clock | 1,635-1,650 MHz median | 1,680 MHz median |
      | memory clock | 9,501 MHz, every sample | 9,501 MHz, every sample |
      | board power | 314 W median | 342-349 W median |
      | `sw_power_cap` active | 161-168 of 162-168 | **161 of 163** |
      | thermal throttle | 0 samples | 0 samples |

      **+35 W (+11% of budget) buys +3% SM clock and 0% throughput, in both phases.** Decode is
      identical to within 0.4%; prefill is inside its own run-to-run spread. So the cap can stay
      where it is, and every number in this file stands without an asterisk.

      Why it costs nothing is the useful part. The memory clock **never leaves 9,501 MHz** — the
      full 19 Gbps spec — at either power limit, in any sample of any run. Decode is
      bandwidth-bound, so SM clock is not what it is waiting for; prefill is the phase where SM
      clock could matter and it is the phase the cap squeezes least (1,635 vs 1,680 MHz, 2.8%).
      Nothing was ever thermally throttled either, at any temperature reached, up to 80 °C.

      Note the cap is still *binding* at 350 W — `sw_power_cap` is active in 161 of 163 busy
      samples there too — so this is not "the cap stopped mattering", it is "this workload does not
      convert board power into throughput". Lifting to the 400 W maximum would not change that
      conclusion for decode, which cannot go faster than its memory clock allows.

      The compute ratios were self-cancelling as predicted: prefill's "~30% of INT8 MMA peak"
      divides a capped measurement by a ceiling probe run under the same cap.

      Also corrects the figures this entry originally carried. It claimed the SM clock "swings
      1,665-1,755 MHz"; steady state is 1,500-1,515 on decode and 1,635-1,650 on prefill at 315 W.
      Individual samples do reach 1,905-1,935 MHz, but only in the first sample or two before the
      cap clamps a cold card — the first draft of the sampling script reported
      `Measure-Object -Average` under a column labelled "median", which is exactly how a boost
      spike gets quoted as a steady clock, so it now takes a real median and says why.

      **Still worth doing, and not blocked on anything: `nvidia-smi -lgc` for measurement runs.**
      The 3-5% between-process spread is now the largest source of noise in every end-to-end
      comparison here. It forced the DFlash2 tile A/B to interleave the two binaries within each
      repetition rather than run one build then the other, and it made an earlier comparison need a
      control column to be readable at all. That is a measurement-hygiene fix, not a performance
      one, and it is independent of the power limit.

- [x] **`compute-sanitizer` and `ncu` both work. Closed 2026-09-09 — this entry was wrong about
      both, and the way it was wrong is worth keeping.**

      **`compute-sanitizer`: there are two copies installed and one of them lies.**

      | copy | version | result on `ninfer_gdn_input_proj_test.exe` |
      |---|---|---|
      | `CUDA\v12.4\compute-sanitizer\` | 2024.1.0 | **exit 0, "ERROR SUMMARY: 0 errors", and the test never ran** |
      | `CUDA\v12.8\compute-sanitizer\` | current | runs to completion, real test output, 0 errors |

      The v12.4 copy produces two lines of output — the banner and a clean error summary — and exits
      0 without executing a single line of the binary. **That is a false pass, which is worse than
      the failure this entry described**: a sanitizer run that reports success having checked
      nothing. The v12.8 copy runs everything. `compute-sanitizer` on `PATH` resolves to
      `CUDA\v12.8\bin\compute-sanitizer.bat`, which is the good one, so an unqualified invocation is
      fine — but an absolute path to the v12.4 directory is not, and that is presumably how this
      went wrong.

      Verified working on v12.8: `memcheck` and `initcheck` both run `ninfer_gdn_input_proj_test`
      and `ninfer_linear_swiglu_fp8_test` to completion, 0 errors. **`initcheck` also runs clean on
      `ninfer_softmax_attention_test`, which is the Op #49 fixed** — the exact investigation this
      entry says had to be done by hand instead.

      Size is not the problem either. The 598 MB
      `ninfer_qwen3_6_27b_score_real_test.exe` this entry named launches fine; it skips because
      `NINFER_QWEN3_6_27B_WEIGHTS` is unset, and compute-sanitizer then says *"Target application
      terminated before first instrumented API call"* — which is a consequence of the test making
      no CUDA calls, not a launch failure. That message is easy to read as one.

      **`ncu`: present, and the blocker was a permission, not a missing file.** The entry says
      "`ncu.exe` is also missing from the Nsight Compute 2025.1.0 install directory". True as
      written and misleading: the launcher is `ncu.bat` at the top of that directory, with the
      binary under `target\windows-desktop-win7-x64\`. It runs. What actually stopped it is
      `ERR_NVGPUCTRPERM` — **GPU performance counters are administrator-only by default on
      Windows** — which is a permission, fixable per-run by an elevated shell with nothing
      persistent and no reboot. `scripts\sweeps\admin-profile.ps1` exists for that and produced
      §2c's MoE counters on the first try.

      The lesson for the next investigation: when a tool appears not to work, check whether a
      second copy is shadowing it and whether the failure is a permission. Neither of the two
      things this entry called broken was broken.

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

      **Three candidates eliminated 2026-09-09, and the current value is exactly reproducible.**
      `int8` re-runs to **4.342425** at HEAD — identical to twelve significant figures, on 261,167
      scored tokens, in 465 s. So the drift is a real, repeatable difference between two builds and
      not measurement noise.

      **Not the artifact.** `qwen3_8_27b.ninfer` has an mtime of 29 August 14:06, and the old
      figures were published in `838c8b5d` on 29 August. The weights have not been reconverted
      since before the old measurement, which was the tidiest available explanation for a small
      systematic shift across every format at once.

      **Not the scoring set.** The token count is 261,167 in both, so the harness is windowing and
      selecting exactly the same tokens. Whatever changed, changed arithmetic rather than what is
      being averaged.

      **Not kernel or tile selection — and this one is worth keeping.** The natural theory was that
      route-table changes (the upstream catch-up rewrote several bands) select a different MMA tile,
      whose reduction runs in a different order, moving the score in the fifth decimal. Tested
      directly: routed width 1024 in `q5_linear_add` from `MmaResidualR64C128` to `R64C64`, which
      the harness exercises on every one of its four 1,024-wide prefill chunks, and re-ran. The
      score rate moved 568.9 to 535.8 tok/s, so a genuinely different kernel ran — and perplexity
      came back **bit-identical at 4.342425**. Two different MMA tiles over the same weights produce
      the same score to twelve figures. **Tile geometry does not perturb perplexity at all**, so no
      route change anywhere in this repository can be the cause of this drift, and route changes
      need not be treated as a quality risk.

      What is left is a genuine numerics change in some kernel between 29 August and 9 September.
      The range is **392 commits**, which includes the whole upstream catch-up, so a bisect is
      about nine steps at roughly 20-30 minutes each — one build plus one 8-minute
      `ninfer-perplexity` run — call it four hours of exclusive GPU time. Worth scheduling as a
      block rather than squeezing between other work. Note the three shifts are *not* uniform
      (−0.0193%, −0.0163%, −0.0092% for `int8`, `bf16`, `rk8v4`), so whatever it is does not simply
      offset every score by a constant.

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

- [x] **The shipped `-maxctx` launchers ask for a host-KV pin they cannot have. Closed 2026-09-09 —
      by documenting it, because the behaviour is right and the flag is not.**

      All four launchers pass `--host-kv-mib 8192` and they already agree with each other, so the
      "make them agree" half of this entry was moot. What they do *not* do is agree across
      platforms, and nothing said so. The clamp is `#if defined(_WIN32)` only:

      | launcher | platform | free after startup | pinned host KV |
      |---|---|---:|---:|
      | `run-qwen38-c1-maxctx.sh` | Linux | — | **8,192 MiB**, honoured in full |
      | `run-qwen36-35b-a3b-c1-maxctx.sh` | Linux | — | **8,192 MiB**, honoured in full |
      | `run-qwen38-c1-maxctx.bat` | Windows | 1.59 GiB | **302 MiB** |
      | `run-qwen36-35b-a3b-c1-maxctx.bat` | Windows | 184-344 MiB | **0 — none at all** |

      `clamp_host_kv_reservation_bytes` takes `(free VRAM − 1 GiB) / 2`, so the 35B maxctx profile
      falls entirely under the 1 GiB floor and the flag is a **complete no-op** there. On the 27B it
      delivers 3.7% of what it asks for.

      **The three options this entry offered were all based on a wrong premise, which is why none of
      them was right.** It assumed the pin competes with context — "both deliberately fill the card
      with KV". It does not. The clamp reads `cudaMemGetInfo` *after* the KV cache is allocated
      (`program_impl.h:1030`), so the pin takes from the slack that is left over and **costs no
      context at all**. A realistic figure would be wrong on the next machine, dropping the flag
      changes nothing because the default is also 8192, and `--no-prefix-reuse` would trade away
      something that is currently free.

      So the behaviour needs no change and the flag needs no correction — it is right on Linux and
      harmless on Windows. What was wrong is that a reader had no way to know any of that. Each of
      the four launchers now carries the measured table above and states plainly, on the Windows
      side, not to read "8192" as a description of the machine; and on the Linux side, that the same
      flag really does pin 8 GiB of host RAM there and why the platforms differ from identical
      arguments.

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
