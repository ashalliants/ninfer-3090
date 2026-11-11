# Batched MTP verification across lanes: design notes

Branch `feat/mtp-multi-row-lanes`, off `feat/basalt-learnings`. Written 2026-11-11 after a session
measured Flash-Next's two-lane mode falling back to fully plain decoding the moment a second
request joins a round (`program.cpp`'s `decode()`, `sequences.size() > 1`), confirmed as a
deliberate design choice (`flash-next-plan.md` PR 16), not a bug -- but one worth lifting, since a
naive sequential-per-row MTP round would cost *more* than today's plain path by this session's own
profiled per-round costs (verify at W=3 ~20 ms vs plain ~12.5 ms; see `docs/performance.md`'s
"MTP and lanes" section). A real fix needs the GPU forward pass itself batched across rows, not
just removing the fallback.

**Status: design only, not implemented.** This document exists so the implementation (by whoever
picks it up, including a future session) does not have to re-derive the kernel survey below, which
is the single most valuable and time-consuming part of this investigation: establishing that the
feature needs **no new CUDA kernels**, only new host-side orchestration.

**Scoped to N rows, not 2.** `kMaximumLanes = 2` is today's hard cap on Flash-Next concurrency, set
independently in `qwen4_exp_instance.cpp`'s refusal ("three or more lanes lose to two on one RTX
3090") and the `std::array<T, kMaximumLanes>` sizing throughout `program.cpp`/`program.h`. That cap
is a *separate* piece of work (raising it is its own decision, re-measuring whether more lanes
still wins on one card) from *this* feature, which is "however many lanes are active, MTP should
speculate on all of them instead of falling back to plain the moment there is more than one." The
new structs and functions below should take the active row count as a runtime size (a `std::span`
or count, the same way `ForwardRowsBatch::slots` already does for the plain path, which already
scales to `kMaxForwardRows = 8`), not a literal `2`, so that raising `kMaximumLanes` later needs no
changes here -- only to the places that gate concurrency. The real ceiling on N is not
`kMaximumLanes` but the dense Ops' column-invariance bound: `N * W <= kMaxForwardRows (8)`, where W
is the draft-plus-anchor width (`plan_.drafts + 1`). At W=4 (3 drafts) that's N<=2 today; a smaller
W (fewer drafts) admits a larger N before this bound bites. GDN's own kernels are more generous
(`gated_delta_net_replay_record`/`causal_conv1d_silu_snapshot` independently support B up to 8), so
they are never the limiting factor; attention's per-row loop (below) is O(N) in cost but not capped
by any kernel limit at all. Worked numbers below use N=2 because that's what's measurable on this
engine today, not because the design assumes it.

## The kernel survey (why no new CUDA math is needed)

Every low-level primitive a batched 2-row x W-column MTP verification needs already exists,
already handles a `rows`/`B` dimension natively, and is already used elsewhere in this codebase for
either the plain multi-row case or the single-row verify case. Only attention has no fused multi-row
kernel, and that is fine -- it needs a loop, not new math.

| Stage | Existing primitive | Already supports |
|---|---|---|
| Dense Linears, rmsnorm, rope, MoE | the usual column-invariant Ops (`project`, `rmsnorm`, `ops::rope`, `moe`) | column-invariant at <= `kMaxForwardRows` (8) columns regardless of which (row, draft-position) a column belongs to -- 2 rows x 4 columns (anchor + 3 drafts) fits exactly |
| GDN recurrence | `ops::gated_delta_net_replay_record` (`include/ninfer/ops/gated_delta_net.h`) | **natively** `q/k [128,Hq,T,B]`, `initial_state_slots: I32[B]` reading from arbitrary absolute slots in a shared state pool, `B=1..8`, `T=2..16` -- exactly the `[W, 2]` shape needed, and it can read directly from the *same* shared pool (`rows_->state.gdn_recurrent`) the existing plain `decode_rows()` path already uses, via `batch.slot_ids` |
| GDN convolution | `ops::causal_conv1d_silu_snapshot` (`include/ninfer/ops/causal_conv1d_silu.h`) | **natively** `x/out [C,W,B]`, `B=2..8` accepts `W=1..16`, with `initial_state_slots` (read) and a *separate* `snapshot_base_slots` (write) -- the write target can be scratch, giving the same non-destructive "replay" semantics the single-row verify path already relies on, just for B rows |
| GDN commit fold | `ops::GdnReplayFoldPlan::execute(std::span<const GdnReplayFoldRow>, stream)` (`include/ninfer/ops/gdn_replay.h`) | **explicitly documented** to take multiple rows at once: "The active row count is `rows.size()` and must be in `[1, records.spec.record_capacity]`" -- each `GdnReplayFoldRow` carries its own `source_state_slot`, `destination_state_slot` and `commit_columns`, so two lanes with *different* accepted-draft counts commit in one call |
| GDN replay records storage | `GdnReplayRecords::layer(layer, rows)` (`src/core/gdn_replay_records.h`) | the `rows` parameter already exists (today always called with `rows=1`); `record_capacity` is a planned dimension, not a hardcoded 1 |
| Attention / QSA | `qsa_pool_keys`, `qsa_select_blocks`, `qsa_attention`, `qsa_commit_tails` (`include/ninfer/ops/qsa.h`) | **no batched multi-row form exists.** Each is single-sequence (`qsa_commit_tails` says so explicitly: "one sequence"). `attention_rows()` (`forward.cpp`) already copes with this today for the *plain* B-rows-1-column case by **looping over rows and calling the single-row Ops once per row** (see below) -- the same pattern generalizes to W columns per row |
| Commit dispatch | `program.cpp`'s `commit()` | **already branches on `rows.size() > 1`** and calls `commit_rows()` -- but `commit_rows()` as written requires `d.accepted_tokens == 1` for every row (the plain-decode assumption) and would need a new speculative sibling, not a reused call |
| Multi-row speculative result | `PendingBatch` (`program.h`) | **already has** `rows_: array<SequenceHandle, kMaximumLanes>`, `counts_: array<int32_t, kMaximumLanes>` ("a verification round licenses each row's accepted drafts"), `speculative_: bool`, `row_stride()` -- built for exactly this, unused by anything today |

The attention loop precedent, confirmed by reading `Forward::attention_rows()` directly
(`forward.cpp` ~line 1097): it does **not** use a fused kernel for block selection/paged attention
at all. It batches only the dense projections (query/key/value/index, rmsnorm, rope) across all B
rows' single columns, then runs a plain `for (b = 0; b < B; ++b)` loop calling the exact same
single-row `qsa_pool_keys` / `qsa_select_blocks` / `qsa_attention` the single-sequence path uses,
sliced to that row's column and block table. A batched-verify attention path is the same loop, just
slicing each row's **W** columns instead of 1, and reusing the single-row *verify* attention code
(which already handles W>1 for one row) inside each iteration.

## What's confirmed safe to reuse unchanged

- `ops::speculative_accept_greedy_drafts` (the acceptance kernel): single-row signature, cheap
  relative to the forward pass. Call it twice, once per row's W-column slice of the combined
  logits output. No new kernel needed; this was never the bottleneck.
- `mtp_draft()` (the drafter step): ~0.88 ms per call, trivial next to a ~20 ms verify round. Keep
  it sequential per row (`select_row` between calls), reusing the existing single-row code
  unchanged. Batching the drafter too is a possible later optimization, not a correctness
  requirement.
- `qsa_commit_tails`: single-sequence only, call it twice at commit time (once per row), exactly as
  the forward-pass attention loop above does for the forward side.

## What's still open (where this document stops, and implementation starts)

1. **New structs** (`forward.h`): `ForwardRowsVerify` (an N-row `ForwardVerify`: `GdnReplayRecords`
   with `record_capacity` planned for `kMaximumLanes` (today 2, so this grows for free if that cap
   rises later), `qsa_keys`/`qsa_tails` scratch and `conv_scratch` sized per lane,
   `initial_slots: I32[N]` holding the active lanes' real slot indices) and
   `ForwardRowsVerifyBatch` (an N-row, W-column-per-row version of `ForwardRowsBatch`:
   `ids`/`positions`/`rope_positions`/`ngram_rows` etc. sized `[N*W]` instead of `[B]`, row-major
   with each row's W columns contiguous, `slots`/`slot_ids` dynamically sized like
   `ForwardRowsBatch::slots` already is).
2. **New Forward methods** (`forward.cpp`): `run_rows_verify_front/back`, a new
   `layer_rows_verify(layer, batch)` parallel to `layer_rows`, with `gdn_rows_verify` (dense
   batching + `gated_delta_net_replay_record` + `causal_conv1d_silu_snapshot`, both called once
   with B=N) and `attention_rows_verify` (the per-row W-column loop described above, `for (b = 0;
   b < N; ++b)`).
3. **Memory planning** (wherever `plan_.spec.*` and `spec_backing_` are sized today): a new region
   for the N-row verify scratch above, sized by `kMaximumLanes` (not a literal row count) so it
   already has headroom if that cap rises. This is the part this document could not finish safely:
   the existing per-lane scratch (`spec_base() + cur_ * stride`-style addressing) is tempting to
   alias into "lane i = row i" without a new allocation, but doing that without fully confirming
   the lanes' private regions are laid out with a stride a combined `[.., N]` view can legally
   assume is a real risk of silent memory aliasing/corruption. The safe path is a **new, dedicated
   allocation**, planned like any other region in this codebase, not a reinterpret of existing
   per-lane scratch.
4. **New graphs** (`program.cpp`'s `Graphs`/`LaneGraphs` structs): graphs indexed by `(N, width)`
   rather than `width` alone for this new path -- in practice today that's just `N=2`, since
   `kMaximumLanes` caps it there, but the indexing itself should not assume N=2 specifically.
5. **New program-level functions**: `try_speculate_rows(sequences, budgets, masks)` (the N-row
   sibling of `try_speculate`, taking `sequences.size()` as N rather than assuming 2, gated so it
   only attempts the batched path when *every* active row independently passes `try_speculate`'s
   existing eligibility checks -- budget, not constrained without `--mtp-constrained`, not
   CPU-verify-failed -- falling back to today's plain `decode_rows()` entirely, for all rows,
   otherwise, which is simple and conservative); a `verify_rows()` building the combined batch,
   running the new Forward calls once, then calling the unchanged single-row accept Op N times
   (once per row's W-column logits slice) and combining the N results into one `PendingBatch`
   using its already-existing multi-row fields (themselves sized by `kMaximumLanes`, so already
   N-general); a `commit_speculative_rows()` (the N-row sibling of `commit_verified()`, using
   `GdnReplayFoldPlan::execute()` with N `GdnReplayFoldRow` entries and calling `qsa_commit_tails`
   N times).
6. **`decode()`'s dispatch**: replace the unconditional `if (sequences.size() > 1) return
   decode_rows(...)` with "try `try_speculate_rows`, fall back to `decode_rows` if it declines."
7. **Testing**: a new exactness test mirroring the project's own convention
   (`ninfer_qwen4_exp_lanes_real_test --exact --mtp`): run the batched 2-row path and, separately,
   each row's own single-lane MTP path on an identical prompt pair, and assert the outputs are
   **bitwise identical**. This is the only way to catch an indexing/stride mistake in the new code
   -- there is no CPU reference implementation of any of these kernels to check against on paper,
   and the failure mode of a subtle bug here is silently wrong generation, not a crash.
8. **This needs a CUDA-capable GPU for every step past (1)**: this Mac has no `nvcc`, so none of
   this compiles here, and correctness can only be established by actually running the exactness
   test on real hardware (there is no CPU-side way to validate GDN/QSA kernel numerics).

## Cost model sanity check (why this is worth doing, if done right)

Using this session's own profiled numbers (`flash-next-plan.md` PR 13: a plain round ~12.5 ms, a
verify round at W=3 drafts ~20 ms) and this session's measured median acceptance (~2.3 tokens per
verify round):

- **Today (plain, batched):** 2 rows x 1 token, ~13-14 ms (batching overhead is cheap) => ~0.148
  tok/ms.
- **Naive sequential MTP** (each row's own full verify round, back to back, no GPU batching): 2
  rows x ~2.3 tokens, 2 x 20 ms = 40 ms => ~0.115 tok/ms -- *worse* than today.
- **Genuinely batched verify** (one combined forward call servicing both rows' W columns at once,
  the design above): if the combined call costs close to a single row's ~20 ms (plausible, given
  dense ops batch for free and GDN's replay kernel is natively B-capable; attention's per-row loop
  is the one part that does not batch, but it is a small fraction of a verify round's cost), 2 rows
  x ~2.3 tokens for ~20-24 ms => ~0.19-0.23 tok/ms -- genuinely better than today, and the only one
  of the three that is.

This is why "just remove the fallback" (the naive option) is not a fix -- only the fully batched
design is worth building.
