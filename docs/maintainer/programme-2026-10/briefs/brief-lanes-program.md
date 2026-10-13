# Task F: Flash-Next Program with two lanes (slice 2 of multi-lane)

Read `brief-common.md` first. You are a careful, methodical implementer: work in small verified
steps, build and run the tests for each step, and keep a short progress log in your PR description.

## Starting point (CHANGED: neither PR is merged; the user told us to base on the cache branch)
- **Base**: branch `origin/feat/qwen4exp-context-cache` (PR #285, CHANGES_REQUESTED by the CodePulse
  bot with three blocking items; the original agent is fixing them on that branch with new commits,
  no force-push). Create your worktree from its CURRENT tip (`git fetch origin` first) and fetch
  again before every significant step; merge/rebase the fix commits as they land (they touch
  `program.cpp` near lines 925, 1568, 1856: prefill execution-boundary frontiers, vision pricing
  in source pricing, capture slab-ownership leak). Your PR targets `feat/qwen4exp-context-cache`
  (one level of stacking). Read its design note `docs/maintainer/flash-next-context-cache.md`
  fully; handles, keys and the "mirror" are per lane by design; the Host slab pool is shared.
- **Another stack member**: Task E works on branch `feat/qwen4exp-context-store`, also on top of
  that branch (`--context-store` export/import and an NVMe tier). It will touch `program.cpp` too:
  keep your changes in `program.cpp` as local as you can and rebase carefully.
- **Batched forward** (PR #286 draft, branch `origin/feat/qwen4exp-batched-forward`, 3 files:
  `src/models/qwen4_exp/execution/forward.{h,cpp}` and `tests/models/qwen4_exp/test_forward_real.cpp`):
  it does not overlap #285. **Fold its commits into your branch** (`git cherry-pick` them) so
  your PR carries the forward changes with it; the user will close #286 in favour of yours.
  It gives `execution::Forward::run_rows` (up to 8 slots, plain decode) with new
  `ForwardRowsState` / `ForwardRowsKV` / `ForwardRowsBatch`; prompts prefill one slot at a time
  through `select_row`; every row is bit-identical to decoding it alone (tested B = 1..4).
- **Spike results**: `docs/maintainer/flash-next-lanes-spike.md` (on branch
  `origin/spike/qwen4exp-lanes`, PR #284; read with `git show origin/spike/qwen4exp-lanes:docs/maintainer/flash-next-lanes-spike.md`),
  especially section 4's table (copy / adapt / new) and section 5's slice definition and
  acceptance gate. Measured so far: an extra row costs 2.2-2.5 ms non-expert; predicted C = 2
  aggregate ~1.10-1.16x one-lane MTP; this is borderline, so the **server-level C = 2 test is the
  real decision**.

## Goal
`--max-concurrency 2` works for Qwen3.8-Flash-Next: two requests decode in the same rounds.

Scope:
1. **Lanes in the Program** (`src/models/qwen4_exp/program/program.{h,cpp}`): replace the single
   `Lane` with an array of up to 2 (refuse > 2 with a message pointing at the spike). Mirror the
   Qwen3.5 Program's lane bookkeeping (`src/models/qwen3_5/program/decode.cpp`
   `decode_ordinary_batch`, `storage/sequence.cpp`) for handles, epochs, phases, per-row pending
   tokens, commits, per-row sampling configs/penalties/grammar masks.
2. **Memory**: per-lane state slots and KV extents from the elastic VMM KV pool, keeping PR 12's
   order (release expert frames first, then map; growth only at a round boundary with nothing in
   flight; never race the VRAM monitor). `plan_device`, the startup memory report and the RAM ledger
   must account per lane (spike section 1 has the numbers: ~116 MB state, ~14.7 KB/token KV).
   Remove the `max_concurrency != 1` refusals in `program.cpp` and `qwen4_exp_instance.cpp` only
   for 2 lanes.
3. **Rounds**: with two active lanes run one plain batched decode round (`run_rows`); with one
   active lane run today's MTP path unchanged (a lone request must stay within 2% of today's
   speed). Decode graph pair per B (1 and 2). Prefill: one lane at a time; a lane's prefill calls
   are interleaved with the other lane's decode rounds by a simple policy (start with: decoding
   lane waits out the other's prefill call, i.e. at call granularity, so waits are bounded by one
   4,096-token call ~1.7 s; document the number you measure).
4. **Expert cache**: promotion budget over the round's tokens; frame leases for a wide prefill
   while another lane decodes must not break (spike section 4, "What could explode").
5. **Context cache at 2 lanes**: capture/restore per lane, the shared Host pool, resume == cold
   still holds, cancellation of one lane does not disturb the other.
6. **Launcher**: `NINFER_CONCURRENCY` for `qwen38-flash-next` (default stays 1 unless the gates
   below pass cleanly, then say in the PR that the default was changed and why); docs updated
   (`docs/serving.md`, README Flash-Next section, plan, `docs/maintainer/flash-next-lanes-spike.md`
   is on another branch: put your results in `flash-next-plan.md` instead).

## Acceptance gates (report every number; ABBA, same sitting, GPU lock)
1. Greedy ids with 2 lanes equal each request run alone at C = 1 (plain decode in shared rounds),
   many steps, prompts of different lengths, one deep in the QSA budget; with the cache on and
   a resume in one lane.
2. **Aggregate decode >= 1.15x MTP C = 1** on the spike's 8-request workload (tools in
   `tools/flash_next/lanes_spike/`, branch `spike/qwen4exp-lanes`; copy what you need into your
   branch with attribution if it is not on master). Also report the queue metrics (makespan, mean
   completion, mean time to first token) against C = 1.
3. 4K / 32K / 128K alternation at C = 2 with no crash (the cuMemSetAccess NOT_READY class); the
   VRAM monitor must stay sane.
4. A lone request at `--max-concurrency 2` within 2% of `--max-concurrency 1` MTP decode speed and
   cold prefill speed.
5. Existing tests: Flash-Next engine/vision/cache real tests, the CPU tests, `ctest` for
   qwen4_exp. The full suite once at the end (known environment failures: jsonschema x2,
   `ninfer_artifact_writer_interop_test`).

If gate 2 fails but gates 1, 3, 4 pass and C = 2 is still faster than C = 1 in aggregate and for
queued TTFT, ship it **opt-in** (default stays 1) and disclose the numbers; do not hide a miss and
do not turn the default on. If correctness or a crash gate fails, stop and report.

## Process
- Own worktree from the cache branch tip (see Starting point). Do not touch other agents' worktrees.
  One PR with base `feat/qwen4exp-context-cache` (split into at most 3 only at natural seams, per
  PR_POLICY; if #285 has merged by the time you open the PR, retarget to master).
- If you are stuck for more than a couple of attempts on one problem (a crash, a numerics mismatch),
  write down what you tried and the evidence and stop that line instead of guessing; report it.
