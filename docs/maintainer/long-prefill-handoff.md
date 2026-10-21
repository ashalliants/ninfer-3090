# Long uncached prefill: handoff for the first session that has a 3090

Written 2026-10-21 by a session with **no GPU**. Everything below was built, reasoned and CPU-tested
without running a kernel. This page is for an agent (or person) with no prior context who is about to
sit in front of the RTX 3090 and finish the job. Read it top to bottom once, then work the plan in
"What to do, in order".

## The problem

Someone sends a ~100K-token prompt that is not in any cache. On Qwen3.5-27B (dense, int8) that is about
55 s of prefill; the 35B-A3B MoE is roughly 2.5x faster. While it runs:

- **Other streams stall.** The worker loop does, per cycle: control work, one decode round, then ONE prefill
  step (a whole `--prefill-chunk`, default 4096 tokens). At 100K depth one chunk takes about 3.1 s, so a
  neighbouring stream gets one decode round per ~3 s.
- **The queue is blocked.** A fresh request never preempts a running prefill, so a short request behind it waits
  the full ~55 s at one lane (C1).
- **A cancelled prompt wastes its work**: only a `SharedPrefix` checkpoint survives an abandon.

Cached traffic is fine. Only this uncached case hurts. The fix is scheduling, not faster kernels, plus a
look at whether long-context attention is a larger share than it should be.

Cost model (fitted on sm_86, docs/performance.md): `step = chunk_ns + token_ns*tokens + attention_pair_ns*pairs`,
`pairs = prefix*tokens + tokens*(tokens+1)/2`. 27B: 107.76 ms + 284.5 us/token + 4.54 ns/pair. 35B-A3B:
44.9 ms + ~104 us/token + 2.01 ns/pair. At 100K attention is ~42% of the prompt's time. Per-step and
per-token terms are collinear at one chunk size, so only `base` and `attention` are identifiable from a
single-chunk sweep.

## Repo rules you must follow

Read `AGENTS.md`, `CLAUDE.md`, `PR_POLICY.md`, `.github/pull_request_template.md`.

- **One RTX 3090 shared by many sessions.** Anything touching CUDA (server, bench, ncu/nsys, GPU tests) holds the lock
  `%TEMP%\ninfer-gpu.lock`: an open file handle with `FileShare.None`, never create/delete the file. Use
  `. tools/gpu-lock.ps1; Invoke-WithGpuLock { ... }` inside ONE PowerShell invocation. Poll every 0.5 s.
  Details in `CLAUDE.md`.
- **Tests:** `pwsh tools/run-tests.ps1 -CpuOnly` (seconds, no lock) and
  `pwsh tools/run-tests.ps1 -GpuOnly -Regex '<area>'`. Never a bare `ctest`, never the whole GPU suite
  unless asked. Say in the PR which tests ran and why the rest did not.
- **Risky features are opt-in, default off.** Report honestly: label anything unrun, do not call a prediction a
  measurement. No unrelated cleanup.
- Windows build recipe (`build-ninja`, VS2022 vcvars) is in `AGENTS.md`.
- Linux/WSL with no card can build and run the CPU tests (`libcuda` stub needed); that is how all of this was checked.

## The PR stack (all Draft, against the user's fork `ashalliants/ninfer-3090`)

Linear order: #341 -> #343 -> #344 -> #342 -> #351 -> #352. Others hang off master.

| PR | Branch | What | New flag / artefact | Status |
|----|--------|------|---------------------|--------|
| #339 | fix/qwen4-exp-memory-plan-test-linux | Linux build of a test | - | merged |
| #340 | feat/prefill-progress-anchors | `SharedPrefix` checkpoints along a long prefill so a cancelled prompt's work survives; rolling retirement | `--prefill-anchor-tokens N` | **overlaps merged #338, see below** |
| #341 | feat/prefill-stall-budget | `PrefillPacing::step_cap`: shrink a prefill step so a neighbouring decode waits at most N ms | `--prefill-stall-budget-ms` | CPU-tested only |
| #343 | feat/yield-long-prefill | pause a long prefill for a short queue head (`choose_yield_victim`, ratio 4, max 4 yields) | `--yield-long-prefill-ms` | CPU-tested only |
| #344 | feat/prefill-backlog-metric | `ninfer_prefill_backlog_seconds` gauge + `/v1/load` field | - | CPU-tested only |
| #342 | feat/prefill-decode-share | decode rounds get a % of prefill time (debt cap 10 s) | `--prefill-decode-share` | CPU-tested only |
| #351 | test/engine-fake-harness | GPU-free fake-model `EngineCore<Instance>` harness, 16 scheduling scenarios on the real worker loop | tests/runtime/fake_engine.h | passes on CPU |
| #352 | feat/prefill-layer-slices | residual parked in `prefill_hidden`, decode rounds between layer groups of one chunk | `--prefill-layer-slices` | **GPU path never run** |
| #345 | bench/prefill-stall-probe | `tools/bench/run_prefill_stall.py`: what a long prompt does to a stream next to it | - | the experiment runner |
| #346 | docs/decode-during-long-prefill | design note, why fusing decode into the prefill step was NOT built | - | doc |
| this PR | feat/long-prefill-handoff | `tools/bench/prefill_time_model.py`, `run_prefill_depth_sweep.py` and tests, this page | - | CPU tests pass |

`PrefillScheduleOptions` (include/ninfer/types.h): `stall_budget_ms, yield_remaining_ms, decode_share_percent,
layer_slices`. Flash-Next accepts and ignores the options (layer slices refused).

### #340 vs merged #338 (decide first)

Master now has `--progress-anchor-tokens` (#338): private `LongAnchor` checkpoints every N tokens
(`src/runtime/contract/progress_anchors.h`, `request_plan.cpp`). #340 does the same job with `SharedPrefix` anchors
under another flag. Do not ship both. Recommended: rebase #340 down to a delta on #338 (also publish the anchors
as `SharedPrefix` so a *cancelled* prompt's work survives, keep rolling retirement), or close it if #338 covers the
use case. This was not reconciled; the rest of the stack was written against pre-#338 master and needs a rebase
onto current master (expect conflicts in `engine_core.h`, `request_plan.cpp`, `types.h`, `serve_options.cpp`).

## What is verified and what is not

Verified on CPU: the pacing maths (`test_prefill_pacing.cpp`, `test_prefill_yield.cpp`); the engine logic on the real
worker loop against a fake model (`tests/runtime/test_engine_scheduling.cpp`, mutation-checked); option parsing,
metrics, load report; the time model against numbers in docs/performance.md and against C++ `step_cap`.

**Never run on a GPU:** everything above as real serving; `TextContext::prefill_impl` layer slicing
(`run_stage_layers(..., first, last)`, `PrefillSliceState`) was compiled and reviewed only. The test
`sliced_prefill_matches_whole` in `tests/models/qwen3_5/test_native_transactions.cpp` (also `capped_prefill_steps`,
`progress_anchors`) compiles but has not run. Forced-token control batches write `prefill_hidden`, so the engine
holds control batches while a sliced chunk is in flight; check that hold is enough.

## What to do, in order

1. **Rebase and build.** Reconcile #340/#338, rebase the stack onto master, build (`build-ninja`), run
   `tools/run-tests.ps1 -CpuOnly`.
2. **Correctness gate for slicing (#352):** under the lock,
   `pwsh tools/run-tests.ps1 -GpuOnly -Regex 'qwen3_5_native_transactions'`. `sliced_prefill_matches_whole` must pass
   bit-exact (sliced hidden states and KV/GDN state equal to whole-chunk prefill). If it fails, fix or drop #352;
   do not enable the flag.
3. **Run the model, get predictions:** `python tools/bench/prefill_time_model.py --model 27b --prompt 100000 --chunk 4096`.
   Reference output (chunk 4096, 100K): baseline 55.0 s / worst neighbour gap 3108 ms; stall budget 500 ms ->
   73.1 s / 544 ms; layer slices 8 -> 62.9 s / 428 ms; slices 4 -> 58.4 s / 811 ms; decode share 50% -> 108 s;
   yield wait 48.1 s -> ~1.6 s.
4. **Measure the baseline stall:** `tools/bench/run_prefill_stall.py` (#345): a short streaming request next to a
   100K uncached prompt; record the neighbour's worst inter-token gap and the prompt's total time, flags off. Compare
   with step 3's baseline row.
5. **Each option on its own, same probe:** `--prefill-stall-budget-ms 500`, `--prefill-layer-slices 8`,
   `--prefill-decode-share 50`, `--yield-long-prefill-ms <N>` (for the yield, add a short request arriving at ~50%).
   Compare measured vs the model. A disagreement over ~15% means the model or the implementation is wrong; find out which.
6. **Attention study:** `python tools/bench/run_prefill_depth_sweep.py --weights <27B> --lengths 4096,16384,32768,65536,100000
   --chunk 4096 --model 27b -- <ninfer_bench flags>` (`--from-json` re-analyses a saved run;
   `--print-attention-commands` prints the ncu/nsys commands). It reports where the fitted `attention_pair_ns` is wrong
   with depth. Candidates, in rough order of expected win:
   - chunk 8192 (fewer steps, fewer weight passes) vs the longer stalls it causes;
   - refit the FA2 plan constants at 4096 columns (they were fitted only for 17-2048 columns);
   - check GDN and memory-bound shares at depth.
7. **Promote or not.** A flag becomes a default only if, on your measurements, it cuts the neighbour's worst gap or
   the short request's wait by a large factor AND costs under ~25% on the long prompt's total time, with no
   correctness test regressing. Otherwise leave opt-in and record the numbers in `docs/serving.md`. Ask the user for the decision.
8. **Report** in each PR: commands, numbers, what you did not run. Flip Draft to ready only with GPU results.

## Risk list (look here first if something is off)

- `Request::slice_in_flight` and `Scheduler::next_prefill` / `youngest_victim` (`src/runtime/engine/scheduler.h`): a
  half-sliced request must not be preempted, evicted or have its lane taken.
- `engine_core.h` `run_prefill_step`, `yield_long_prefill_for_head`, `restore_held_for_yield`: pause/restore with
  snapshots interacts with the Host arena fragmentation fix (#334, merged) and the admission fit check (#337, open).
- Each sliced call must report its own timing; the engine sums `slice_elapsed_ns` (a double count was fixed once).
- Cancellation `abandon`s private checkpoints; only `SharedPrefix` survives.
- Cost-model feedback in `prefill_pacing.h` (correction EMA): check it converges on real timings and does not oscillate.
- Flaky tests seen: `ninfer_qwen4_exp_ngram_volume_test` (timing), `ninfer_context_store_test` (once).

## Files in this PR

- `tools/bench/prefill_time_model.py`, `test_prefill_time_model.py` (CPU tests `ninfer_prefill_time_model_test`)
- `tools/bench/run_prefill_depth_sweep.py`, `test_prefill_depth_sweep.py` (`ninfer_prefill_depth_sweep_test`)
- `tests/cmake/BenchSupportTests.cmake`, `TestGpuClasses.cmake` (both tests are `cpu`)
- this page
