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

## A non-obvious gap found while scoping the implementation (read this before writing `gdn_rows_verify`)

`ops::causal_conv1d_silu_snapshot`'s `initial_state_slots` (read) and `snapshot_base_slots` (write)
are **both indices into the same `conv_states` tensor's Slots dimension** -- not "a live slot to
read, write anywhere disjoint." The real per-lane live state pool (`rows_->state.gdn_conv[l]`,
shape `[C, K, slots]`) has exactly `slots = kMaximumLanes` (or however many lanes the server is
configured for) -- there is no spare room in it for the non-destructive snapshot's write targets.
Pointing `causal_conv1d_silu_snapshot` at that tensor directly, the way `gated_delta_net_replay_record`
safely can at the GDN *recurrence* state (which has no analogous read/write-same-tensor
constraint, confirmed from its own doc comment: separate `ssm_states` read-only input, separate
`key_record`/`value_record`/`gate_record` writes), would either throw (if `snapshot_base_slots`
validation catches an out-of-range index) or, worse, silently alias a *different* lane's real live
window -- exactly the "silent memory corruption" risk flagged above, now with a concrete mechanism.

The single-row verify path sidesteps this entirely by using a **different** Op,
`causal_conv1d_silu_split` (the "distinct-state form"), whose `conv_state_in`/`conv_state_out` are
simply two independent tensors with no shared slot-indexing scheme -- it reads the live state
directly and writes the advanced (discarded) window to scratch, no staging needed. That path is
not a template for the batched one; the Ops are different enough that copying its pattern naively
would be wrong.

The fix is an explicit staging step `gdn_rows_verify` must do and the single-row path never needs:
copy each active row's live convolution window (`rows_->state.gdn_conv[l].slice(2, slot, 1)`) into
a **dedicated scratch tensor the new code owns**, sized `[C, K, 2 * lanes]` (not `lanes`) -- the
first `lanes` slots as `initial_state_slots` targets (the just-copied windows), the next `lanes` as
`snapshot_base_slots` targets (the op's write destinations) -- and run
`causal_conv1d_silu_snapshot` entirely within that scratch tensor, never passing
`rows_->state.gdn_conv` to it directly. The copy itself is cheap (a handful of small
device-to-device memcpys, same cost class as the single-row path's existing tail/conv scratch
copies), but it is easy to miss, since nothing in the single-row code suggests it is needed.

## Implementation progress (branch `feat/mtp-multi-row-lanes`, commits after this doc)

Items 1-3 below are done, built, and verified on a real Linux GPU box (compiles clean, and the
existing `ninfer_qwen4_exp_program_plan_test` -- which checks this exact memory layout -- still
passes, confirming the new region is additive and does not disturb the single-row plan). Nothing
exercises any of it yet; it is all reachable-but-uncalled code until the pieces below it land.

1. `ForwardRowsVerify`/`ForwardRowsVerifyBatch` and `Forward::run_rows_verify_front/back` in
   `forward.h`/`forward.cpp`. Two real simplifications found while writing this that are worth
   keeping, not reverting to the originally-designed-more-complex version:
   - **No `causal_conv1d_silu_snapshot`, no scratch-staging dance.** The GDN convolution step is
     cheap relative to the recurrence it sits next to, so `gdn_rows_verify` just loops per row
     calling `causal_conv1d_silu_split` (the single-row "distinct state" form, which takes any
     disjoint input/output with no shared slot-indexing scheme) directly against the real live
     window -- no staging, no extra scratch, removing an entire class of risk for near-zero
     performance cost. The batched GDN *recurrence* (`gated_delta_net_replay_record`) is still
     genuinely batched across rows; that is the part that actually matters for throughput.
   - **`qsa_keys` is `std::vector<Tensor>` (one per row), not one 4D tensor.** The forward pass
     wants per-(layer, row) access and the commit wants one row's whole `[Di, W, layers]` slab at
     once (`qsa_commit_tails` takes every layer in one call); no single axis order makes both
     slicing patterns contiguous at once, so this sidesteps the question entirely by giving each
     row its own tensor, shaped exactly like the single-row case's `qsa_keys`.
2. `SpecLayout::plan()` extended with a second, independent `record_capacity = lanes` region
   (`rows_gdn`/`rows_ple`/`rows_qsa_keys`/`rows_tails`), appended via the same `take()` builder
   after the existing single-row fields.
3. `rows_verify_view(W)`, paralleling `verify_view(W)`, lazily building `ForwardRowsVerify` from
   the new region (a new `rows_records_` member, parallel to `records_`).
4. **Found while scoping the next piece (IO staging), not yet acted on:** the existing
   `IoLayout`/`plan_.io.ids/positions/rope/block_rope/logit_columns` fields are already sized for
   `columns` (the widest call anywhere -- prefill chunks, easily thousands) wide, far more than
   `B * W` (<= 8) ever needs, so **no new IoLayout region is needed** for those fields: a new
   `write_rows_verify_inputs`-style function can write directly into the existing `io_host_`/
   `io_device_` scratch at `b * W`-offset column ranges, for each row's own lane (`lanes_[lane]`,
   not `L()`, since this needs several lanes' own history/rope without switching `cur_`). Separately,
   `plan_.io.rows_block`/`rows_rope` (already `lanes`-sized, built for `decode_rows()`'s one
   triple-per-row need) can likely be reused as-is for this path's `block_start_rope` too --
   genuinely the same shape and purpose, not just a coincidence.
5. **Decision: no CUDA graph capture in the first version.** The single-row verify path's
   `replay()`/`launch_round()` graph machinery is a real optimization but an orthogonal one -- the
   throughput win this feature chases comes from batching the forward pass across rows, not from
   graph replay. `verify_rows()` should call `run_rows_verify_front/back` eagerly (every round,
   no capture), exactly matching how every new path in this codebase already starts (eager on the
   first round in the single-row case too). Graph capture for this path is a clean, separable
   follow-up once correctness is established, not a blocker for a first working version.

## Finished, built, and tested -- but shipped off by default (2026-11-11, same session)

Items 1-5 above landed: `try_speculate_rows`/`verify_rows`/`commit_speculative_rows`, the IO staging
function, and `decode()`'s dispatch (try the batched path, fall back to `decode_rows()` if it
declines). `ninfer_qwen4_exp_lanes_real_test --exact --mtp 3 --generate 96 --cache --max-context
16384` passes bitwise-identical against the single-lane reference with this path genuinely
exercised (the test now forces `mtp_joint_acceptance`/`mtp_rows_verify` on itself -- without both,
a round with more than one row never reaches `try_speculate_rows` at all, so the test would prove
nothing about this code; this was initially missed and the test silently validated only the plain
fallback for several iterations of this session).

Seven real bugs surfaced getting there, six found and fixed, the seventh found but not yet
root-caused. All but the last were found by firing two real HTTP requests at a running server at
the same moment (`scripts/run.sh ... NINFER_CONCURRENCY=2`, then two `curl`s backgrounded in one
shell line) -- the standalone test's "pairs" can stagger two lanes across separate rounds without
forcing them simultaneous, so it reliably missed every one of these:

1. **Joint-acceptance trim applied before staging, not just at accept.** `verify_rows()` was
   trimming each row's drafted tokens to the joint-acceptance window's `kept` count *before*
   staging them into the round, but staging needs every row's *full* drafted width (`W = limit + 1`
   uniformly) to build one shared batch -- a `kept < limit` row read past its own history.
   Fix: stage the full draft, apply the trim only at accept time as a narrower span.
2. **`ple_inputs` passed 3D to a function needing flat 2D.** `ple_gate`/`ple_conv_inject` want
   `[C, T]` (T = B x W); the per-row commit fold wants `[C, W, B]`. Fix: a `.view()` at the forward
   call site, not a new buffer.
3. **A non-contiguous `rope_positions` slice for QSA's per-row calls.** `batch.rope_positions` is
   axis-major (`[T, 3]`, 3 planes of T words); slicing the fast axis to a `W`-wide sub-range is not
   itself a contiguous `[W, 3]` tensor (qsa's Ops require one). Fix: gather it into a small per-row
   scratch tensor first (`attention_rows_verify`); `mtp_catch_up_row_rows` below sidesteps the same
   hazard differently, by *restaging* its own RoPE words from the lane's `LaneRope` instead of
   slicing the round's, matching how the single-row `commit_verified` already does it.
4. **The accept step used the round's shared ceiling, not the row's own trimmed count.**
   `speculative_accept_greedy_drafts` requires `target`/`logits` sized exactly `[K + 1, *]` for
   *this* row's own `K`; a full-`W`-wide slice mismatched whenever a row's own count was smaller.
5. **Missing `decode_ns` accumulation** (stats-only): `verify_rows()` never added to
   `lane.decode_ns`, so a concurrent request reported an impossible decode rate. Not a correctness
   bug, but it hid real throughput numbers during measurement.
6. **No drafter catch-up after a rows-verify commit -- severe, not low-cost as scoped.** The first
   version of `commit_speculative_rows()` deliberately skipped catching the drafter up (reasoning:
   "a stale drafter only costs future acceptance, never correctness"). True in principle -- the
   exactness test still passed -- but the *magnitude* was far worse than guessed: acceptance fell
   to single digits almost immediately, because nothing ever re-synced `L().mtp_cells`/
   `saved_residual()` to the real committed history, and the single-cell `mtp_write_pending()`
   (built for being exactly one token behind) silently did the wrong thing when several rounds
   behind instead of catching up properly. Fixed with `mtp_catch_up_row_rows(b, S, k, W)`, the
   per-row generalization of the single-row `mtp_catch_up(S, k)`: it reads row `b`'s residuals from
   the round's own `rows_residuals` export (a new `SpecLayout` field, `rows_verify_batch` wiring
   `residual_out` into it) rather than the single-row `residuals` region, and restages its own RoPE
   words for the same contiguity reason as bug 3. A second, genuine race surfaced inside this fix
   itself: unlike `mtp_catch_up_row`'s single cell (one word at a *unique* per-row offset, so two
   rows' host writes never overlap), the multi-cell version reuses one shared scratch window
   (`w_mtp_ids`, `io.rope`) at a fixed offset for every row. `upload_pinned` is a GPU kernel that
   reads that pinned host memory asynchronously (not a blocking `cudaMemcpyAsync`), so without an
   explicit `device_.synchronize()` between rows, the second row's host writes could race ahead of
   the first row's kernel actually reading them -- caught the same way, by the HTTP concurrent test
   showing one of two simultaneous lanes' acceptance corrupted while the other (served first) was
   fine. Fixed by synchronizing at the end of `mtp_catch_up_row_rows`, once per row per round.
7. **Open: once two rows genuinely speculate together, one row's acceptance stays near zero,
   every round, from the very first round on -- confirmed not caused by bug 6's fix.** Diagnosed
   with temporary instrumentation (not kept) tracing drafts, the main model's own greedy target,
   and the accept outcome per row per round: the *drafter's own* proposals for the affected row
   look healthy early on (high self-reported confidence, matching baseline), but the round's
   accept step only ever licenses 1 token for that row, repeatedly, which then starves that row's
   `joint_threshold` of real observations (`L().joint_offered` only updates when `kept > 0`) and
   the acceptance window it offers next round collapses too -- self-reinforcing, but the root cause
   is upstream of the threshold, in whichever row consistently loses. It is *not* which physical
   lane: re-running with no prior single-lane warm-up on either request still reproduces it, and
   the row that loses is whichever one lands at a nonzero `b` in `verify_rows`'s own per-row loop,
   not a property of the request's content (ruled out: identical asymmetry across different prompt
   pairs, different physical lanes, with and without warm-up). Every addressing expression that
   scales with `b` (`rows_residuals`/`io.positions`/`io.rows_block` offsets) was re-derived and
   checked against the project's own "anchor" convention (`state_tokens` is the anchor's index;
   `history.size() == state_tokens + 1` is the invariant `decode()` itself asserts) and matches the
   single-row convention exactly -- so this is likely in the rows-batched *forward* (bugs 2/3's own
   area: PLE flattening or the QSA RoPE gather) rather than in catch-up, in a way that skews a
   nonzero row's verify logits without (so far) flipping the final *committed* token away from what
   a single lane would have produced -- which is why it never showed up as an exactness failure.
   **Correctness is unaffected** (every draft is independently verified against the real model
   regardless of how it was proposed, and the exactness test keeps passing); the cost is entirely
   in that row's speed. Measured end-to-end (production `tuned` profile's HTTP route, 4 paired
   ~3.6-4.1k-token-prompt reps, 150-token completions, `--mtp-rows-verify` forced on for the
   measurement): aggregate 2-lane decode of 52-62 tok/s per pair -- *worse* than the existing
   plain-round fallback it would replace (~115-127 tok/s aggregate on the same box, same reps, flag
   off). **Decision: ship the implementation, but gate it behind a new flag,
   `--mtp-rows-verify` (`ProgramOptions::mtp_rows_verify`/`FlashNextOptions::mtp_rows_verify`,
   needs `--mtp-joint-acceptance`), default **off**, not enabled in any shipped profile.** With the
   flag off (today's behavior, unchanged), `try_speculate_rows` declines immediately and every
   multi-row round takes the proven plain fallback -- solo and 2-lane-plain numbers in
   `docs/performance.md` are unaffected by this branch. The flag exists so the already-landed
   implementation and its own exactness coverage stay exercised and reviewable, and so a follow-up
   session has a known, isolated place to pick this up (tracing the per-row forward call's own
   logits against a single-row reference for the losing row specifically, column by column, is the
   next step -- not yet done here) without re-deriving any of the above.

## Bug 7 deep dive, no GPU available (2026-11-12, follow-up session)

A follow-up session re-read every line of the rows-batched path by hand (no build, no GPU, no
Vast box -- this Mac has no `nvcc`) specifically hunting an off-by-row-index, aliasing, or wrong
stride for row `b != 0`, since bug 7's own symptom ("whichever row lands at a nonzero `b` in
`verify_rows`'s own per-row loop loses, independent of physical lane or prompt content") points
squarely at something tied to **loop position**, not lane identity. Nothing new was found --
every one of the following re-derivations matches the single-row convention or an already-proven
precedent, confirming (not just repeating) the prior session's own re-derivation:

- `GdnReplayRecordLayer`'s `conv`/`key`/`value`/`gate` tensors are `[*, width, rows]` (`rows`
  outermost, `width` next, channels fastest) -- flattening `width` and `rows` into one `T` axis
  (`records.conv.view({C, T})`, `gdn_rows_verify`, `forward.cpp:1296`) therefore produces exactly
  `t = row * W + w`, matching the `b * W + w` column convention used everywhere else. Checked
  against `src/core/gdn_replay_records.h:38-42`.
- `ops::ple_conv_inject`'s own documented contract (`include/ninfer/ops/ple.h:84-97`): "`T` columns
  form `sequences` runs of `W = T / sequences` consecutive columns" -- `sequences` is
  `source_slots.numel()` (`= B`, not `T`), so the op's own row-major run convention already matches
  `rows_verify_batch`'s `b * W` layout without the call site needing to do anything special. Not a
  bug; the earlier concern that `ple_conv_inject` might want one slot per *column* rather than per
  *row* is unfounded.
- `mtp_catch_up_row_rows`'s every `b`-dependent address (`rows_residuals + 2*width*base_column`,
  `io.positions + 4*base_column`, `io.rows_block + 12*b`, the restaged `io.rope` at offset 0) was
  re-derived term by term against `mtp_catch_up`'s single-row original and against
  `commit_speculative_rows`'s own per-row staging just above its call site
  (`program.cpp:5272-5280`) -- all consistent, no transposition, no stale `S`/`k`/`W` capture.
- `stage_rows_verify_inputs`'s `stage_rope`/`stage_rope_value` calls (`program.cpp:4979-4982`) write
  into the shared `[T, 3]` axis-major RoPE buffer at `columns = total`, `column = b * W` -- matches
  exactly what `attention_rows_verify`'s `cudaMemcpy2DAsync` gather
  (`forward.cpp:1401-1405`) expects (`spitch = T * 4`, offset `4 * row * W`).
- The `pooled`/`tail` mutation pattern inside `attention_rows_verify`'s per-row loop (writing
  directly into the *live* `kv.pooled.at(index)` during verify, same as the single-row
  `attention_verify` at `forward.cpp:802-808`) is **not new**: the existing, long-proven plain
  `decode_rows()` path's own `attention_rows()` does the identical thing per row
  (`forward.cpp:1155-1160`) and shows no such asymmetry there. This rules out "mutating live pooled
  state out of commit order" as the differentiator, since the exact same pattern already works
  symmetrically for 2 rows elsewhere in this codebase; whatever is wrong is specific to the
  *verify*-only additions (W > 1 columns per row, drafting, joint acceptance, catch-up), not to the
  general per-row attention loop shape.
- `qsa_pool_keys`/`qsa_select_blocks`/`qsa_attention`'s own doc comments
  (`include/ninfer/ops/qsa.h:90,167-168,213-214`) state each is a pure, stream-ordered function of
  its declared inputs and an explicit `WorkspaceArena` -- "no host reads of device values", no
  hidden call-order-dependent internal state. If taken at face value (no reason found to doubt it),
  this rules out an internal-kernel-state explanation for an effect that depends on *which call in
  the loop* a row is, rather than on any value the row's own tensors carry.
- `kv.max_visible_keys` is a fixed per-slot sizing constant set once at slot creation
  (`program.cpp:1065`), identical in both lanes' configuration; not round- or order-dependent.

**What this rules in, not just out:** every address a human can check by reading the code is
correct, and the primitives involved are documented as pure functions of their declared arguments.
Combined with the prior session's finding that the exactness test (which *would* catch a logits
divergence, since the committed token is literally `target[0]`, the model's own argmax from the
real forward pass) keeps passing, the forward pass's *output* for row `b = 1` is, as far as static
reading can establish, computing the textbook-correct greedy target. If a subtle numeric skew still
exists, it cannot be a wrong address or a stale buffer by this document's reading -- it would have
to be something no file read can show: an actual kernel defect reachable only at `B > 1, W > 1`
that doesn't flip the argmax (so exactness survives) but does shift the distribution enough to
change which of the *drafter's own* tokens matches it less often, or a real behavioral consequence
of non-greedy sampling (flagged by the prior session as unconfirmed) actually explaining the whole
thing. Static review cannot distinguish between these; both predict "acceptance for row b=1 looks
bad while the exactness test stays green."

**Two concrete, cheap next steps for whoever has GPU access again** (not attempted here --
this Mac cannot build or run either):

1. **Force `--greedy` and re-run the same two-curl HTTP concurrent test** that originally exposed
   bug 7. If the asymmetry disappears or sharply narrows under greedy decoding, the "non-greedy
   sampling" explanation from the prior session (never fully confirmed) is very likely the whole
   story, and bug 7 may not be a kernel bug at all -- it would just mean `joint_threshold`'s
   calibration reacting badly to legitimate sampling variance in one row, which is a tuning
   question, not a correctness one. This is a single flag, no code change, and should be tried
   *first* since it could retire the bug entirely without touching any kernel.
2. **If the asymmetry survives `--greedy`**, add one targeted diagnostic: in `verify_rows()`, right
   after `run_rows_verify_back`, `cudaMemcpy` the top-5 logit values and indices at column `b * W`
   for `b = 1` to host and print them, then immediately run the *single-row* `attention_verify`
   path (not the batched one) for that same lane's identical history/anchor in the same process
   (e.g. a debug-only branch that also runs the old single-lane verify path for comparison, never
   committing its result) and diff the two top-5 lists. Identical top-5s at identical ranks would
   finally rule out the forward pass with actual evidence rather than code reading; a divergence
   would localize the bug to a specific layer by bisecting which layer's `layer_rows_verify` output
   first differs from the single-row equivalent, column by column -- the next step named in the
   previous session's write-up, still not done.

**This remains unresolved as of the above re-derivation.** No fix was proposed at that point:
without a GPU to run either experiment above, a patch would have been a guess dressed up as a fix,
worse than leaving the flag off.

## Bug 7 found: the logits scratch was too small for a batched round (2026-11-12, independent re-review)

An independent review (a fresh agent, told nothing beyond the symptom and pointed at the same
files) found the actual root cause, confirmed by directly reading the allocation math rather than
trusting the report: **`logits32_`/`logits16_` are sized for `max(max_width, lanes)` columns
(`program.cpp`'s `plan_layout` and the `Program` constructor), but a batched rows-verify round
writes `B * W` columns into them (`verify_rows`'s `Tensor wide32(logits32_.data, FP32, {vocab_, B *
W})`).** With `--mtp 3` (`max_width = 4`) and 2 lanes, the old sizing allocated 4 columns; a 2-row
round needs 8. Row 0's 4 columns (columns 0-3) fit inside the real allocation and are always
correct. Row 1's 4 columns (columns 4-7) silently overrun into whatever is allocated right after --
`lane_logits_`, `sampled_`, `sample_position_`, `configs_`, and the start of `token_counts_` -- no
crash, since it's all one buffer, just corrupted logits for row 1, every round, from the first one.
This matches every observed symptom exactly: deterministic by loop position (not lane, not
content), present from round 1, the drafter's own proposals look healthy (it never touches this
buffer), and `target[0]` for row 1 -- read out of corrupted memory -- rarely matches a healthy
draft. It also explains why the exactness test missed it: that test checks the two requests
*overlapped in time*, not that they *decoded in the same batched round* (`rows_rounds_` is counted
but never asserted on), so it is plausible no run of it ever actually exercised a genuine B=2 round
in the first place.

**Two further, smaller bugs surfaced in the same review**, both affecting every row equally (so
neither explains the asymmetry, but both are real correctness gaps in the rows-verify path):

- **The per-row QSA tail copy that `attention_rows_verify`'s own comment claims happens was never
  written.** The single-row verify path explicitly copies the live tail into scratch before every
  round (`verify()`'s `front` lambda, "the verification pools from copies of the live tails");
  nothing in `stage_rows_verify_inputs`/`verify_rows` did the per-row equivalent into `rows_tails`,
  so `qsa_pool_keys` read whatever stale bytes were last there.
- **`stage_rows_verify_inputs`'s block-start RoPE staging used the wrong layout.** It called
  `stage_rope_value(value, span, columns=B, column=b)`, which writes the axis-major pattern `word[a
  * B + b]` (matching the big `[T, 3]` RoPE buffer's own convention) -- but `io.rows_block` is `[3,
  B]`, 3 *contiguous* words per row (the `rows_block + 12 * b` convention every reader, including
  `mtp_catch_up_row_rows` and `attention_rows_verify`, relies on). With 2 rows this interleaved the
  two rows' block-start words: row 0 read `(row0, row1, row0)`, row 1 read `(row1, row0, row1)`.
  `decode_rows()`'s own staging of the same buffer (`row_block[3 * b + a] = block[a]`) was the
  correct precedent, a few hundred lines above the broken call.

**All three fixed in this commit** (`logit_columns_for()` sizes the logits scratch for `B * W` when
`--mtp-rows-verify` is on, plus a runtime capacity check in `verify_rows`; a per-row
`cudaMemcpy2DAsync` into `rows_tails` added to `verify_rows`'s drafting loop; the block-start RoPE
write corrected to match `decode_rows()`'s layout). **None of this has been built or run** -- this
Mac has no CUDA toolchain and no GPU was rented to verify it (per instruction). The logic was
checked by hand against the existing, working single-row and plain-rows precedents each fix
mirrors, and the capacity arithmetic was independently re-derived (not just trusted from the
review), but **this is still unverified until someone runs
`ninfer_qwen4_exp_lanes_real_test --exact --mtp 3 --generate 96 --cache --max-context 16384` and
the two-curl HTTP concurrent test on real hardware.** If both pass and the acceptance asymmetry is
gone, `--mtp-rows-verify` is probably ready to re-measure for the default-on decision; if not, this
section's fixes are the first thing to re-check, not the forward pass math that two separate
reviews have now independently re-confirmed as correct.

## What's still open (where this document stops, and implementation starts)

Items equivalent to the original 1-4 here are now **done** -- see "Implementation progress" above
for what actually landed (in places, simpler than originally planned here: no `conv_scratch`/
`initial_slots` staging dance, `qsa_keys` as a vector not a single tensor, no new `IoLayout` region
needed, and no CUDA graph capture in the first version). What remains:

1. **The IO staging function**: a `write_rows_verify_inputs(rows, width)`-style function, writing
   each active row's W-column ids/positions/RoPE/n-gram-rows into the existing `io_host_`/
   `io_device_` scratch at `row * W`-offset column ranges (confirmed large enough: sized for
   `columns`, the widest call anywhere), reading each row's own `lanes_[lane]` state directly
   (not `L()`, since several lanes' own history/RoPE are needed without switching `cur_`), and each
   row's `block_start_rope` via the existing `plan_.io.rows_block` (already `lanes`-sized, built
   for `decode_rows()`'s identical one-triple-per-row need).
2. **New program-level functions**: `try_speculate_rows(sequences, budgets, masks)` (the N-row
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
3. **`decode()`'s dispatch**: replace the unconditional `if (sequences.size() > 1) return
   decode_rows(...)` with "try `try_speculate_rows`, fall back to `decode_rows` if it declines."
4. **Testing**: a new exactness test mirroring the project's own convention
   (`ninfer_qwen4_exp_lanes_real_test --exact --mtp`): run the batched 2-row path and, separately,
   each row's own single-lane MTP path on an identical prompt pair, and assert the outputs are
   **bitwise identical**. This is the only way to catch an indexing/stride mistake in the new code
   -- there is no CPU reference implementation of any of these kernels to check against on paper,
   and the failure mode of a subtle bug here is silently wrong generation, not a crash.
5. **This needs a CUDA-capable GPU for every step past (1)**: this Mac has no `nvcc`, so none of
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
