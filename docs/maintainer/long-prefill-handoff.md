# Long uncached prefill: handoff for the first session that has a 3090

Written 2026-10-21 by a session with **no GPU**; the first GPU session (2026-10-22, rented RTX 3090) ran the plan and its results are in "Results of the first GPU session" at the end. Everything below was built, reasoned and CPU-tested
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
| #340 | feat/prefill-progress-anchors | `SharedPrefix` checkpoints along a long prefill so a cancelled prompt's work survives; rolling retirement | `--progress-anchors-shared` (needs #338's `--progress-anchor-tokens`) | rebased as a delta on #338 |
| #341 | feat/prefill-stall-budget | `PrefillPacing::step_cap`: shrink a prefill step so a neighbouring decode waits at most N ms | `--prefill-stall-budget-ms` | CPU-tested only |
| #343 | feat/yield-long-prefill | pause a long prefill for a short queue head (`choose_yield_victim`, ratio 4, max 4 yields) | `--yield-long-prefill-ms` | CPU-tested only |
| #344 | feat/prefill-backlog-metric | `ninfer_prefill_backlog_seconds` gauge + `/v1/load` field | - | CPU-tested only |
| #342 | feat/prefill-decode-share | decode rounds get a % of prefill time (debt cap 10 s) | `--prefill-decode-share` | CPU-tested only |
| #351 | test/engine-fake-harness | GPU-free fake-model `EngineCore<Instance>` harness, 16 scheduling scenarios on the real worker loop | tests/runtime/fake_engine.h | passes on CPU |
| #352 | feat/prefill-layer-slices | residual parked in `prefill_hidden`, decode rounds between layer groups of one chunk | `--prefill-layer-slices` | GPU path run 2026-10-22, see Results |
| #345 | bench/prefill-stall-probe | `tools/bench/run_prefill_stall.py`: what a long prompt does to a stream next to it | - | the experiment runner |
| #346 | docs/decode-during-long-prefill | design note, why fusing decode into the prefill step was NOT built | - | doc |
| this PR | feat/long-prefill-handoff | `tools/bench/prefill_time_model.py`, `run_prefill_depth_sweep.py` and tests, this page | - | CPU tests pass |

`PrefillScheduleOptions` (include/ninfer/types.h): `stall_budget_ms, yield_remaining_ms, decode_share_percent,
layer_slices`. Flash-Next accepts and ignores the options (layer slices refused).

### #340 and #338

Master has `--progress-anchor-tokens` (#338): private `LongAnchor` checkpoints every N tokens. #340 is now only a
small delta on it: `--progress-anchors-shared` publishes the same anchors as `SharedPrefix`, so a *cancelled*
request's work survives for the retry, and the engine keeps a request's two newest (`trim_progress_anchors`,
`ResourceManager::retire_unreused_shared`). The stack was rebased onto master of 2026-10-21 (after #348) and builds
with the CPU tests green on Linux/GCC 13 (only `ninfer_qwen4_exp_ngram_volume_test`, a timing flake, fails).
If the rebased branches are not yet on the remotes, the old pre-#338 commits are what the PRs show: force-push them.

### Master fixes the stack needs

#354 fixes two things that kept master from passing on Linux (a `reserve_in_chunks` compile error from #334 and a
self-contradicting `ninfer_serve_options_test` entry from #338). Merge it first.

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

1. **Build.** Merge #354, make sure the stack is rebased on master, build (`build-ninja`), run
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
   `--print-attention-commands` prints the raw attention benchmark commands (run them under ncu/nsys)). It reports where the fitted `attention_pair_ns` is wrong
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

## Results of the first GPU session

2026-10-22, a rented RTX 3090 (24 GB, Linux, CUDA 12.8), Qwen3.8-27B (`qwen3_8_27b.ninfer`), `--prefill-cublas
--kv-dtype int8 --max-concurrency 2 --prefill-chunk 4096`, the rebased stack. One run per row, a fresh server each.
`run_prefill_stall.py`: a stream at 48 events/s, then one 108K-token uncached prompt.

| setting | worst neighbour gap | 108K prompt | model said (100K) |
|---|---|---|---|
| none | 2930 ms | 56.1 s | 3108 ms / 55.0 s |
| `--prefill-stall-budget-ms 500` | 528 ms | 72.1 s | 544 ms / 73.1 s |
| `--prefill-layer-slices 8` | 402 ms | 60.8 s | 428 ms / 62.9 s |
| `--prefill-stall-budget-ms 500 --prefill-decode-share 50` | 538 ms (p50 20 ms) | 138.5 s | - |

- The model agrees with the measurements to 3-6%. (The default `bench/fixtures/bench_corpus.ids` holds 65,536 token ids,
  so `ninfer_bench` refuses `pp100000`: tile it into a longer file and pass `-- --corpus FILE`. Tiled text is fine for prefill timing.) `run_prefill_depth_sweep.py` (4K to 100K, chunk 4096): measured
  1.27 s / 5.5 s / 12.2 s / 28.5 s / 50.4 s for 4K / 16K / 32K / 64K / 100K tokens; fitted scales 0.97 for the
  per-step and per-token cost and 0.89 for attention (the Engine's pair cost is slightly pessimistic at depth).
  Attention is 40% of the 100K prompt.
- **Layer slices are the best of the options for the neighbour**: a 7.3x smaller worst gap than baseline (2930 to
  402 ms) for 8% more prompt time, against 5.5x smaller (528 ms) for 28% more with the stall budget. `sliced_prefill_matches_whole` passes bit-exact on the
  GPU (it did not before: see below), and greedy output of a 56K prompt, a stream and a short request is
  token-identical with every flag (and with all four together).
- **Yield**: a short request sent 10 s into the 108K prompt got its first token after 0.77 s with
  `--yield-long-prefill-ms 5000`, against 46.5 s with no flag; the long prompt took 57.2 s instead of 56.3 s.
- **Shared progress anchors (#340)**: cancel the prompt at 35 s and resubmit: the retry took 24.5 s with
  `--progress-anchor-tokens 8192 --progress-anchors-shared` (73,728 tokens reused), against 57.8 s with no anchors
  and 58.1 s with private anchors.
- **Admission fit check (#337)**: two 50K-token prompts into a 70K-token pool: without the flag one preemption and the
  first prompt took 27.3 s; with it one deferral, no preemption, the first prompt took 20.5 s and the pair finished
  in the same 40.8 s.
- **Backlog gauge (#344)**: `/v1/load` `prefill_backlog_seconds` read 0 idle and 24.9 s with one 50K prompt running
  and a second queued.

What broke and was fixed: three GPU test fixtures written without a card passed a hard-coded page count of 2 where
the explicit capacity needs 8 at 64 tokens a page (`progress_anchor_planning`, `capped_prefill_steps`,
`sliced_prefill_matches_whole`), and the last two asked for a capacity above what `max_concurrency 1` allows. The
engine code was right. `ninfer_model_registry_test` did not link on Linux (spdlog include path).

Not run: the combined stall budget + slices + share + yield under a real mixed load (only for output equality),
chunk sizes other than 4096, `--prefill-layer-slices` with DFlash or media (refused by design), and the second
card cases. The decode share costs 2.5x on the prompt at 50%: leave it off unless the neighbour must see no gap at all.
