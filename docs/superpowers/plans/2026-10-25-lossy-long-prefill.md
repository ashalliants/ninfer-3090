# Lossy Long-Prompt Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cold Qwen3.5-27B prompt whose uncached part exceeds `--lossy-prefill-min-tokens` is scouted by the model's first layers, thinned to the most-attended ~20% of tokens, and prefilled on those only, with later turns reusing the result.

**Architecture:** Three prefill phases inside the Qwen3.5 Program (Exact up to the last shared marker, Scout over the rest through layers `< L` plus the filter layer's Q/K, Main over the kept tokens with their original RoPE positions). The kept prompt replaces the sequence's prompt, so ledger = KV rows and every existing invariant holds. A per-lineage gap map lets follow-up prompts be rewritten onto the kept identity before ordinary prefix matching. Int4 scout GEMMs come last, behind their own gate.

**Tech Stack:** C++20, CUDA 12.8 (sm_86), CMake/Ninja, the repo's test bundle (`tests/ninfer_tests`), Python 3 for bench tools.

**Spec:** `docs/maintainer/lossy-long-prefill.md` (read it first; this plan argues from it).

## Global Constraints

- Engine option off by default: `--lossy-prefill-min-tokens 0` means off. Production launcher profile sets `30000`.
- Trigger only when the reuse source is none or a SharedPrefix checkpoint (never a private conversation record), the prompt has no Vision items, the Program is Qwen3.5 dense, and the model is not pipeline-split.
- The product prompt (up to the last shared marker) is prefilled exactly; the threshold applies to the part after it.
- `prompt_tokens` / `cached_tokens` in usage keep their current meaning (original lengths); kept counts only in the request log and counters.
- `max_context` and RoPE capacity apply to the original prompt length.
- Production KV dtype is `rk4v4`: every GPU correctness and quality run uses `--kv-dtype rk4v4` (int8 runs are extra).
- Starting defaults before the sweep: `keep 0.2`, `filter_layer 15`, `block 64`, `recent 4096`, `head 256`, `query_rows 64`, `pool_window 33`, `skip_fraction 0.8`.
- Acceptance (fixed now, not changed after results): needle accuracy ≥ 95% of baseline at 32K/64K/100K; agent-replay `Edit` exact-match within 2 points of baseline; 100K cold prefill ≤ 40% of baseline time. Final go/no-go is the user's, on the PR's numbers.
- Int4 scout gate: kept-block Jaccard vs int8 scout ≥ 0.9; quality gate unchanged within noise; scout GEMMs ≥ 1.4x faster.
- Repo rules (`AGENTS.md`, `CLAUDE.md`, `PR_POLICY.md`): opt-in risky features, honest labels ("compiled, not run" for anything unrun on a GPU), tests via `tools/run-tests.ps1` (Windows) or `tests/ninfer_tests <name>` with `CUDA_VISIBLE_DEVICES=-1` for cpu tests (Linux), never the whole GPU suite unasked, GPU lock on the shared Windows 3090.
- This WSL host has no GPU. GPU tasks run on a rented Vast RTX 3090 (`~/vast/`, memory note "Vast 3090 box workflow"). **Ask the user before renting** and state the expected hours and cost.
- New `cpu` tests go on `tests/cmake/TestGpuClasses.cmake`; ones that link no kernel also on `tests/cmake/HostOnlyTests.cmake`.
- Commit messages: conventional prefixes (`feat`, `test`, `perf`, `docs`, ...), ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A prompt that matches a gap map only partly** (diverges inside `U`, or is a shorter prefix of the original) must be treated as a cold prompt, never rewritten. Test: `GapMapRewrite.PartialMatchIsNotRewritten` (Task 3).
2. **An exact request must never bind a selected checkpoint, and a rewritten prompt must never bind an exact one past the gap.** Test: `LossyFlow.ExactRequestNeverBindsSelectedState` (Task 7).
3. **Cancel or preemption during the scout** must leave the sequence's GDN state, its pages below `C`, and the page pool exactly as before the scout. Test: `ScoutAbortLeavesNoTrace` (Task 6) and fake-engine `LossyCancelDuringScout` (Task 8).
4. **A prompt just over the threshold whose forced keeps exceed `skip_fraction`** falls back to a full main pass, and its result is identical to a non-lossy prefill. Test: `Selection.SkipRuleKeepsEverything` (Task 2) and `LossyKeepAllMatchesOrdinaryPrefill` (Task 7).
5. **Rewrite and typed rewrite checkpoints (`TurnClosure`, `ResponseReplay`) whose frontier falls in a dropped span** are dropped, not mapped to a neighbour. Test: `CompactPrompt.RewriteCheckpointInDroppedSpanIsDropped` (Task 3).

---

### Task 1: Options and flags

**Files:**
- Modify: `include/ninfer/types.h` (beside `PrefillScheduleOptions`, ~:278; field in `EngineOptions` ~:404-511)
- Modify: `src/serve/serve_options.h`, `src/serve/serve_options.cpp` (usage ~:151/181, help ~:223-245, handlers beside `--prefill-stall-budget-ms` ~:486, validation ~:985)
- Modify: `src/serve/generation_service.cpp` (`make_engine_options` ~:318)
- Modify: `src/models/qwen4_exp/.../qwen4_exp_instance.cpp:200` (refuse the flag)
- Test: `tests/test_serve_options.cpp`

**Interfaces:**
- Produces:
```cpp
struct LossyPrefillOptions {
    std::uint32_t min_tokens     = 0;     // 0: off
    float keep_fraction          = 0.2F;  // (0, 1]
    std::uint32_t filter_layer   = 15;    // must be a full-attention layer
    std::uint32_t block_tokens   = 64;    // multiple of 16, >= 16
    std::uint32_t recent_tokens  = 4096;
    std::uint32_t head_tokens    = 256;
    std::uint32_t query_rows     = 64;    // 1..256
    std::uint32_t pool_window    = 33;    // odd
    float skip_fraction          = 0.8F;  // (0, 1]
    bool scout_a4                = false;
    [[nodiscard]] bool enabled() const noexcept { return min_tokens > 0; }
};
// EngineOptions::lossy_prefill
```
- Flags: `--lossy-prefill-min-tokens`, `--lossy-prefill-keep`, `--lossy-prefill-filter-layer`, `--lossy-prefill-block`, `--lossy-prefill-recent`, `--lossy-prefill-head`, `--lossy-prefill-query-rows`, `--lossy-prefill-pool-window`, `--lossy-prefill-skip`, `--lossy-prefill-scout-a4`.

- [ ] **Step 1: Write failing tests** in `tests/test_serve_options.cpp`: `lossy_prefill_defaults_off` (no flags → `enabled() == false`, fields equal the defaults above); `lossy_prefill_parses_all_flags` (each flag sets its field); `lossy_prefill_rejects_invalid` (keep `0`, keep `1.5`, block `24`, pool window `32`, query rows `0` each fail validation with a message naming the flag).
- [ ] **Step 2: Run** `CUDA_VISIBLE_DEVICES=-1 tests/ninfer_tests ninfer_serve_options_test` → FAIL (unknown flag).
- [ ] **Step 3: Implement** the struct, the flags, help text, validation, copy into `EngineOptions`. Filter-layer validity against `layer_types` is checked at Program startup (Task 7), not here. Qwen4-exp refuses any non-default `lossy_prefill` at startup like the other Qwen3.5-only flags.
- [ ] **Step 4: Run** the test → PASS; run the full `-CpuOnly` set → no regressions.
- [ ] **Step 5: Commit** `feat(serve): lossy prefill options (off by default)`.

### Task 2: Token selection

**Files:**
- Create: `src/models/qwen3_5/program/lossy/selection.h`, `selection.cpp`
- Modify: `src/models/qwen3_5/program_sources.cmake`
- Test: `tests/models/qwen3_5/test_lossy_selection.cpp`, register in `tests/models/qwen3_5/tests.cmake` as `ninfer_qwen3_5_lossy_selection_test` (cpu; host-only if it links no kernel)

**Interfaces:**
- Consumes: `LossyPrefillOptions` (Task 1).
- Produces:
```cpp
namespace ninfer::models::qwen3_5::detail {
struct SelectionInput {
    std::span<const float> scores;          // one per token of U, attention mass
    std::span<const std::uint8_t> forced;   // one per token of U; nonzero = always keep
};
struct SelectionResult {
    std::vector<std::uint32_t> kept;        // offsets into U, ascending, unique
    bool skipped = false;                   // skip rule fired: kept == every offset
};
[[nodiscard]] SelectionResult select_tokens(const SelectionInput&, const LossyPrefillOptions&);
}
```
Algorithm (fixed by the spec): mark forced = `forced[i]` ∪ first `head_tokens` ∪ last `recent_tokens`; smooth scores with a centred max-pool of width `pool_window`; block score = sum of smoothed scores over each `block_tokens` block (last block may be short); budget = `ceil(keep_fraction * |U|)`; if forced count > `skip_fraction * |U|` → keep all, `skipped = true`; else keep all forced tokens, then whole blocks in descending block score (ties: lower index first) until the budget is reached (the last block may overshoot by < one block).

- [ ] **Step 1: Write failing tests:** `Selection.KeepsHeadRecentAndForced` (|U|=10,000, all scores 0 except none; forced at offsets {5000, 5001}; result contains [0,256), [5904,10000), 5000, 5001); `Selection.PicksHighestBlocks` (|U|=8192, keep 0.5, recent 0, head 0, scores 1.0 in block 3 and 7 only → blocks 3 and 7 kept first); `Selection.BudgetRespected` (kept.size() ≤ budget + block_tokens - 1 and ≥ budget); `Selection.SkipRuleKeepsEverything` (|U|=5,000 with recent 4096 + head 256 → skipped, kept = all 5,000); `Selection.Deterministic` (same input twice → identical output); `Selection.OutputSortedUnique`.
- [ ] **Step 2: Run** `CUDA_VISIBLE_DEVICES=-1 tests/ninfer_tests ninfer_qwen3_5_lossy_selection_test` → FAIL (not built).
- [ ] **Step 3: Implement** `select_tokens` in `selection.cpp`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(qwen3_5): lossy prefill token selection`.

### Task 3: Prompt compaction and gap maps

**Files:**
- Create: `src/models/qwen3_5/program/lossy/gap_map.h`, `gap_map.cpp`
- Test: `tests/models/qwen3_5/test_lossy_gap_map.cpp` → `ninfer_qwen3_5_lossy_gap_map_test` (cpu)

**Interfaces:**
- Consumes: `PreparedPromptData` (`frontend/prepared_prompt.h`), `SelectionResult` (Task 2).
- Produces:
```cpp
struct GapMap {
    std::uint32_t begin = 0;                   // M: first original index of U
    std::uint32_t original_end = 0;            // M + |U|
    std::array<std::uint64_t, 2> digest{};     // digest of original token ids and positions [0, original_end)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kept_runs; // (offset in U, length), ascending
    [[nodiscard]] std::uint32_t kept_tokens() const noexcept;
    [[nodiscard]] std::uint32_t dropped_tokens() const noexcept { return original_end - begin - kept_tokens(); }
};
using GapMapSet = std::vector<std::shared_ptr<const GapMap>>; // lineage order

[[nodiscard]] GapMap make_gap_map(const PreparedPromptData& original, std::uint32_t begin,
                                  std::span<const std::uint32_t> kept_offsets);
// Kept prompt: [0, begin) unchanged, then kept tokens of U, then [original_end, size) unchanged.
// token_ids, token_types and all three position axes are subset; positions keep original values;
// rope_delta += dropped; identity.rewrite_checkpoint and rewrite_execution_frontiers are mapped
// through the index translation and dropped when they fall in a dropped span;
// context_cache.opportunities likewise; ngram_index reset; external_prefix_tokens must be <= begin.
[[nodiscard]] PreparedPromptData compact_prompt(const PreparedPromptData& original, const GapMap&);
// nullopt unless prompt has at least original_end tokens and its digest over [0, original_end)
// equals map.digest; otherwise compact_prompt(prompt, map).
[[nodiscard]] std::optional<PreparedPromptData> apply_gap_map(const PreparedPromptData& prompt,
                                                              const GapMap& map);
```
Digest: reuse `PrefixShortlistDigests` (`program/prefix_identity.h`) over the original prompt so the value equals `PrefixShortlistDigests::at(original_end)` for that prompt.

- [ ] **Step 1: Write failing tests:** `CompactPrompt.SubsetsTokensAndKeepsPositions` (20 tokens, begin 4, keep U offsets {0,1,5,6,7}: ids/types equal the subset, `positions` axis 0 = {0,1,2,3,4,5,9,10,11,20-…}, `rope_delta` grows by dropped); `CompactPrompt.TailAfterUIsUnchanged`; `CompactPrompt.RewriteCheckpointMapped` (frontier in a kept span maps to the compacted index); `CompactPrompt.RewriteCheckpointInDroppedSpanIsDropped`; `CompactPrompt.ResultIdentityDiffersFromOriginal` (`ResidentPrefixIdentity` of the result does not `matches` the original prompt at its full length); `GapMapRewrite.FollowUpPromptIsRewritten` (original + 7 new tokens → rewritten = compacted original + the 7 tokens, with positions original_end..original_end+6); `GapMapRewrite.PartialMatchIsNotRewritten` (differs at one token inside U → nullopt; shorter than original_end → nullopt); `GapMapRewrite.TwoMapsApplyInOrder`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** in `gap_map.cpp`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(qwen3_5): kept-prompt compaction and gap maps`.

### Task 4: Explicit RoPE positions for text prefill (GPU)

**Files:**
- Modify: `src/models/qwen3_5/execution/text.h:257` (`TextPrefill`), `text.cpp:1530-1546` (`prefill_impl`)
- Modify: `src/models/qwen3_5/program/prefill.cpp` (`prefill_text_chunk` call ~:670)
- Modify: `src/models/qwen3_5/program/transactions/replay.cpp:17-43` (replay uses the identity's stored positions)
- Test: `tests/models/qwen3_5/test_native_transactions.cpp`

**Interfaces:**
- Produces: `TextPrefill::rope_positions` — `std::span<const std::int32_t>` over the whole prompt's axis-0 positions; empty means slot + `rope_delta` (today). When non-empty, `prefill_impl` copies the chunk's slice into `roots.rope_positions` (1 axis) instead of `offset_i32_positions`. Layer slices stay allowed (text-only).
- `prefill.cpp` passes it when the prompt's positions are not `index + rope_delta` for every index ≥ the chunk base (precompute once per staged prompt: `bool regular_text_positions`).

- [ ] **Step 1: Write failing GPU tests:** `explicit_positions_equal_rope_delta_path` (a 3,000-token text prompt with explicit positions `i + 17` gives bit-identical last-token logits, KV and GDN state to the same prompt with `rope_delta = 17` and no explicit positions); `explicit_positions_with_layer_slices` (same with `layer_slices = 4`, bit-identical); `replay_preserves_gapped_positions` (prefill a gapped prompt, pause with snapshot dropped so it replays, resume: greedy 32 tokens identical to the unpaused run).
- [ ] **Step 2: Compile** on this host (`cmake --build ~/build/<dir> --target ninfer_tests`) → builds; GPU run deferred.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: On the GPU box** (`--kv-dtype rk4v4` and `int8`): `NINFER_TEST_ARTIFACT=<27B> tests/ninfer_tests ninfer_qwen3_5_native_transactions_test` → PASS, including the existing `sliced_prefill_matches_whole`.
- [ ] **Step 5: Commit** `feat(qwen3_5): explicit RoPE positions in text prefill and replay`.

### Task 5: Attention key-mass op (GPU)

**Files:**
- Create: `include/ninfer/ops/attention_key_mass.h`, `src/ops/attention_key_mass/{attention_key_mass.cu, sources.cmake}`
- Modify: `src/ops/CMakeLists.txt` (or `basic_sources.cmake`, as `op-development.md` directs)
- Test: `tests/ops/test_attention_key_mass.cpp` → `ninfer_attention_key_mass_test` (gpu)

**Interfaces:**
- Produces:
```cpp
namespace ninfer::ops {
// q: BF16 [head_dim, q_heads, rows]; k: BF16 [head_dim, kv_heads, keys]; q_heads % kv_heads == 0.
// out: F32 [keys], overwritten: out[j] = sum_rows sum_heads softmax_j(scale * q·k_j).
// Causality is not applied (every query row is after every key). Two passes: row max/sum, then mass.
void attention_key_mass(const Tensor& q, const Tensor& k, float scale, Tensor& out,
                        WorkspaceArena& work, cudaStream_t stream);
}
```
- [ ] **Step 1: Write failing test** against a CPU double-precision reference: shapes (256, 24, 64) × (256, 4, N) for N ∈ {17, 4096, 100003}; relative error ≤ 1e-3 per element and sum(out) = rows × q_heads within 1e-3 relative; plus a timing print for N = 100,000 (expect < 20 ms).
- [ ] **Step 2: Compile here; run on GPU** → FAIL then implement.
- [ ] **Step 3: Implement** (CUDA cores are fine: ~40 GFLOP per pass at 100K; tensor cores optional).
- [ ] **Step 4: GPU run** → PASS.
- [ ] **Step 5: Commit** `feat(ops): attention key mass for prompt token scoring`.

### Task 6: Scout chunk in TextContext (GPU)

**Files:**
- Modify: `src/models/qwen3_5/execution/text.h`, `text.cpp` (new entry beside `prefill_chunk`, ~:1788)
- Modify: `src/models/qwen3_5/program/program_impl.h` (scout scratch members)
- Test: `tests/models/qwen3_5/test_native_transactions.cpp`

**Interfaces:**
- Consumes: `run_stage_layers(stage, x, ph, tap, first, last)`, `attention_key_mass` (Task 5), `TextPrefill::rope_positions` (Task 4).
- Produces:
```cpp
struct ScoutScratch {
    Tensor filter_keys;    // BF16 [head_dim, kv_heads, capacity], rows [0, |U|)
    Tensor last_queries;   // BF16 [head_dim, q_heads, query_rows]
    Tensor scores;         // F32 [capacity]
};
// Runs layers [0, filter_layer) over ids[begin, begin+len) of the full prompt (KV for those layers is
// written to the sequence's own pages at slots begin.., GDN states advance in place), then the filter
// layer's input norm + Q/K projection + q/k norm + RoPE only, writing K rows to filter_keys at
// [begin - u_begin, +len). If the chunk ends the prompt, also writes the last query_rows Q rows
// and runs attention_key_mass into scores[0, |U|).
PrefillChunkResult TextContext::scout_chunk(const TextPrefill& prompt, std::uint32_t u_begin,
                                            std::uint32_t begin, std::uint32_t len,
                                            std::uint32_t filter_layer, ScoutScratch& scratch);
```
Before the first scout chunk the Program copies the sequence's GDN conv and recurrent states for layers `< filter_layer` into a scratch slot (sized once at startup when `lossy_prefill.enabled()`), and copies them back when the scout ends or aborts. No DFlash tap, no MTP alignment, no LM head in scout chunks.

- [ ] **Step 1: Write failing GPU tests:** `scout_leaves_no_trace` (bind at a 2,048-token shared prefix, scout 12,288 tokens, restore: GDN states and KV rows `< 2048` bit-identical to before); `ScoutAbortLeavesNoTrace` (abort after the second scout chunk: same check, page pool free count back to its value before the scout); `scout_scores_chunk_invariant` (scores at prefill chunk 1024 vs 4096 agree within 1e-3 relative per token); `scout_finds_needle` (a 32K haystack prompt with a passkey sentence at 50% depth and a question at the end: the passkey's tokens are in the top 5% of scores).
- [ ] **Step 2: Compile here.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: GPU run** (`rk4v4`) → PASS.
- [ ] **Step 5: Commit** `feat(qwen3_5): scout prefill chunk with filter-layer key mass`.

### Task 7: Lossy phases in the Program (GPU)

**Files:**
- Modify: `src/models/qwen3_5/program/prefill.cpp` (`advance_prefill` :517-864: Exact → Scout → Main phases)
- Modify: `src/models/qwen3_5/program/program.h`, `program_impl.h` (`SequenceState`: `LossyPhase`, `GapMapSet gap_maps`; `CheckpointState`: `GapMapSet gap_maps`; a weak registry `std::vector<std::weak_ptr<const GapMap>>`)
- Modify: `src/models/qwen3_5/program/planning/request_plan.cpp` (`plan_request` applies live gap maps, longest `original_end` first; summary gains `reported_prompt_tokens`)
- Modify: `src/models/qwen3_5/program/transactions/binding.cpp:493-498`, `capture.cpp` (gap maps inherited from source / copied into captures)
- Modify: `src/models/qwen3_5/program/planning/startup.cpp` (validate `filter_layer` is `FullAttention`; size scout scratch)
- Test: `tests/models/qwen3_5/test_native_transactions.cpp`, `tests/models/qwen3_5/test_engine_prefix_real.cpp`

**Interfaces:**
- Consumes: Tasks 2-6.
- Produces:
  - `RequestBasePlan` / summary: `std::uint32_t reported_prompt_tokens` (original length; equals `prompt_tokens` for non-lossy), `std::optional<LossyPlan> lossy` where `struct LossyPlan { std::uint32_t exact_end; }` (`M`, the last SharedStablePrefix opportunity frontier ≥ reuse frontier, else the reuse frontier).
  - `Program::enable_lossy(SequenceHandle)` — called by the engine after `start_binding` when the trigger holds; refuses (returns false) for multimodal, split or non-dense.
  - `PrefillProgress` gains `std::uint32_t scouted_tokens` (scout work this step) and `std::optional<std::uint32_t> compacted_prompt_tokens` (set once, on the step that ends the scout: the new `prompt_tokens`).
  - `start_pause` returns false while the sequence is in the Scout phase (so `require_snapshot` yields are refused); `abort` during Scout restores state (Task 6).
- Phase rules: Exact = ordinary prefill to `M` (captures and shared publication as today). Scout = `scout_chunk` over `[M, end)`. At scout end: `select_tokens` with `forced` = special/template token types and ids from the tokenizer's special set; `make_gap_map`; `compact_prompt`; swap into the staged prompt; release pages above the new frontier; then Main = ordinary prefill from `M` of the kept tokens. If `skipped`, Main prefills everything (no gap map recorded). Scout steps honour `max_tokens` (stall budget) but are never layer-sliced: a scout step runs `L`/64 of the model, about a quarter of a normal step's stall already (deviation from the spec's "layer slices apply"; flag it in the PR).

- [ ] **Step 1: Write failing GPU tests:** `LossyKeepAllMatchesOrdinaryPrefill` (keep 1.0: greedy 64 tokens, KV and GDN state bit-identical to the flag off); `lossy_releases_pages_above_kept_frontier` (40K prompt, keep 0.2: resident pages after prefill = `kv_pages_for_frontier(M + kept + 1)`); `lossy_follow_up_turn_reuses` (in `test_engine_prefix_real.cpp`: 40K cold prompt with the flag on, then prompt + reply + new user message: second request's reused tokens ≥ the first prompt's kept count and no scout runs); `LossyFlow.ExactRequestNeverBindsSelectedState` (same 40K prompt with the flag off on the same Program after a lossy one: it does not bind the selected checkpoint); `lossy_reports_original_prompt_tokens`.
- [ ] **Step 2: Compile here.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: GPU run** (`rk4v4`) → PASS; also `-GpuOnly -Regex 'qwen3_5_(native_transactions|prefix_real|agent_continuation)'`.
- [ ] **Step 5: Commit** `feat(qwen3_5): lossy prefill phases, gap-map lineage and prompt rewrite`.

### Task 8: Engine trigger, accounting and observability

**Files:**
- Modify: `src/runtime/engine/engine_core.h` (`try_admit_one` :2396 after source choice and binding; `resolve_prefill_progress` :1689; `run_prefill_step` :1741; yield / preemption paths)
- Modify: `src/runtime/engine/engine.cpp` (usage uses `reported_prompt_tokens`)
- Modify: metrics registration (beside `ninfer_paced_prefill_steps_total`), request log writer (`--request-log-jsonl`), backlog predictor (`prefill_work` for a lossy plan = scout cost over `[M,end)` with layers `< L` + main cost over `keep_fraction` of it)
- Modify: `tests/runtime/fake_engine.h` (fake Instance supports `enable_lossy`, `scouted_tokens`, `compacted_prompt_tokens`)
- Test: `tests/runtime/test_engine_scheduling.cpp`

**Interfaces:**
- Consumes: Task 7's Program API.
- Produces: counters `ninfer_lossy_prefill_total{outcome="completed|skipped|aborted"}`, `ninfer_lossy_prefill_scouted_tokens_total`, `ninfer_lossy_prefill_kept_tokens_total`, `ninfer_lossy_prefill_alias_hits_total`; request-log object `lossy_prefill: {uncached_tokens, kept_tokens, filter_layer, scout_ms, main_ms}`.
- Trigger (in `try_admit_one`): enabled, source is none or SharedPrefix, `prompt_tokens - lossy.exact_end > min_tokens`, then `enable_lossy`.

- [ ] **Step 1: Write failing fake-engine tests:** `LossyTriggersOnlyWhenCold` (private-record source → not enabled; shared source with 31K after marker and threshold 30K → enabled; 29K → not); `LossyPromptLengthChangeAccepted` (progress with `compacted_prompt_tokens` updates the request and finishes without the frontier `throw`s); `LossyCancelDuringScout` (cancel mid-scout: request finishes cancelled, fake reports restore, no pages leaked); `LossyYieldRefusedDuringScout` (a short head waits until Main starts, then yields as today); `LossyUsageReportsOriginal`; `LossyCountersAndRequestLog`.
- [ ] **Step 2: Run** `CUDA_VISIBLE_DEVICES=-1 tests/ninfer_tests ninfer_engine_scheduling_test` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS; full `-CpuOnly` set → no regressions.
- [ ] **Step 5: Commit** `feat(engine): lossy prefill trigger, accounting and metrics`.

### Task 9: Speculative decoding after a lossy prefill (GPU)

**Files:**
- Test: `tests/models/qwen3_5/test_engine_dflash2_real.cpp`, the MTP two-lane real test it sits beside
- Modify only if the tests fail: `program/decode.cpp` (:475 MTP, :771 DFlash positions)

- [ ] **Step 1: Write tests:** `dflash2_after_lossy_matches_plain_decode` and `mtp_after_lossy_matches_plain_decode` (40K prompt, keep 0.2, greedy 128 tokens with speculation equal to the same lossy prefill without speculation).
- [ ] **Step 2: GPU run** → expect PASS; fix positions if not.
- [ ] **Step 3: Commit** `test(qwen3_5): speculative decoding after lossy prefill`.

### Task 10: Quality and speed sweep harness

**Files:**
- Create: `tools/bench/run_lossy_prefill_sweep.py`, `tools/bench/test_lossy_prefill_sweep.py` (cpu, registered in `tests/cmake/BenchSupportTests.cmake`)
- Modify: `tools/bench/README.md`

**Interfaces:**
- Produces: `python tools/bench/run_lossy_prefill_sweep.py --server-cmd "<ninfer-serve ...>" --grid filter_layer=7,11,15,23,31 keep=0.1,0.2,0.3,0.5 block=32,64,128 --workloads needle,agent_replay,greedy_agreement,speed --out <dir>`; for each grid point it starts a fresh server with the flags, runs the workloads, and writes `results.jsonl` plus a `summary.md` table with each acceptance check marked pass/fail against the Global Constraints numbers. `--from-json` re-summarises.
- Workloads: needle = `eval` evalscope needle_haystack at 32K/64K/100K × 5 depths (config derived from `eval/configs/qwen3_6_35b_needle_haystack.yaml`); agent_replay = `python -m tools.bench.agent_replay run --min-context 30000 --max-context 120000`, metrics: tool-call validity and `Edit` `old_string`-in-file rate; greedy_agreement = first-divergence token index vs baseline on 20 long prompts; speed = `run_prefill_stall.py` 108K prompt (total, scout, main from the request log; neighbour worst gap with `--prefill-layer-slices 8`).
- The spec's teacher-forced KL is **not** in v1: `Program::causal_score` does not go through admission, so it would need its own lossy path. Greedy agreement replaces it. (Flag this deviation in the PR.)

- [ ] **Step 1: Write failing CPU tests** for the summariser: given a fixture `results.jsonl`, `summary.md` marks needle 0.96× baseline as pass and 0.94× as fail, `Edit` −2.5 points as fail, and time 0.39× as pass.
- [ ] **Step 2: Run** `CUDA_VISIBLE_DEVICES=-1 tests/ninfer_tests ninfer_lossy_prefill_sweep_test` (or `python -m pytest tools/bench/test_lossy_prefill_sweep.py`) → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `bench: lossy prefill quality and speed sweep`.

### Task 11: Run the sweep, choose defaults (GPU)

**Files:**
- Modify: `include/ninfer/types.h` (defaults), the production launcher profile (sets `--lossy-prefill-min-tokens 30000`; find it via `docs/maintainer/launcher-profiles.md`), `docs/serving.md` (new section), `docs/maintainer/lossy-long-prefill.md` (results section)

- [ ] **Step 1:** On the GPU box, baseline (flag off) then the grid, all `--kv-dtype rk4v4`, Qwen3.8-27B, `--max-concurrency 2`. Record commands and raw outputs.
- [ ] **Step 2:** Pick the fastest grid point that passes all three acceptance checks; if none passes, record the best and stop here for the user's go/no-go (do not lower the bar).
- [ ] **Step 3:** Set defaults, launcher profile, docs with the measured table (marked measured, date, box).
- [ ] **Step 4: Commit** `perf(qwen3_5): lossy prefill defaults from the 2026-10 sweep`.

### Task 12: Int4 Q4_G64 GEMM (GPU)

**Files:**
- Create: `include/ninfer/ops/q4_a4_linear.h`, `src/ops/linear/q4/q4_a4_mma.cu`, `src/ops/common/mma.cuh` (add `mma_s4_m16n8k64`)
- Test: `tests/ops/linear/test_q4_a4_linear.cpp` → `ninfer_q4_a4_linear_test` (gpu)

**Interfaces:**
- Produces:
```cpp
namespace ninfer::ops {
// x: F16/BF16 [K, T] -> xq: packed s4 [K/2, T] bytes, xs: F16 [K/64, T] (symmetric, per token per 64).
void quantize_a4_g64(const Tensor& x, Tensor& xq, Tensor& xs, cudaStream_t stream);
// w: Q4_G64_FP16 [N, K] (existing object); out: same dtype as x [N, T]. mma.sync.m16n8k64.s4.s4.s32
// per 64-wide group, group scales applied in FP32.
void q4_a4_linear(const Tensor& xq, const Tensor& xs, const QuantizedWeight& w, Tensor& out,
                  WorkspaceArena& work, cudaStream_t stream);
}
```
- [ ] **Step 1: Write failing test:** shapes gate_up 34816×5120 and down 5120×17408 (when Q4), T ∈ {512, 4096}; error vs a CPU reference of the same int4-quantized inputs ≤ 1e-2 relative (exactness of the integer path), and vs the A8 route's output cosine ≥ 0.995; timing print vs the existing A8 route at T = 4096 (target ≥ 1.4x).
- [ ] **Step 2-4:** compile here, implement, GPU run → PASS.
- [ ] **Step 5: Commit** `feat(ops): int4-activation Q4_G64 GEMM for sm86`.

### Task 13: Int4 scout and its gate (GPU)

**Files:**
- Modify: `src/models/qwen3_5/execution/text.cpp` (scout chunk routes Q4 projections through `q4_a4_linear` when `lossy_prefill.scout_a4`), `src/ops/wrapper/linear_swiglu.cpp` / `linear_add.cpp` (new policy value for the scout only)
- Modify: `tools/bench/run_lossy_prefill_sweep.py` (`--workloads jaccard`: kept-block Jaccard between `scout_a4` on/off from the request log's kept runs, add `kept_runs` to the request log behind `--request-log-lossy-detail`)

- [ ] **Step 1: Write failing GPU test:** `scout_a4_kept_overlap` (32K and 100K haystack prompts: kept-block Jaccard a4 vs int8 ≥ 0.9; the passkey is kept in both).
- [ ] **Step 2-3:** implement; GPU run → PASS.
- [ ] **Step 4:** sweep the chosen default with `--lossy-prefill-scout-a4`; record the gate (Jaccard, quality, speed). Enable by default only if all three pass; else leave opt-in and record why.
- [ ] **Step 5: Commit** `perf(qwen3_5): int4 scout GEMMs`.

### Task 14: Pull request

- [ ] **Step 1:** `-CpuOnly` on Linux; on the GPU box the GPU tests touched (`qwen3_5_(native_transactions|prefix_real|agent_continuation|dflash2_real)`, `attention_key_mass`, `q4_a4_linear`). List what ran and why the rest did not.
- [ ] **Step 2:** Fold stable facts into `docs/serving.md`, `engine-architecture.md`, `resource-scheduling-and-context-cache.md`; delete `docs/maintainer/lossy-long-prefill.md` and this plan only after the user's go/no-go (keep them in the PR until then).
- [ ] **Step 3:** Push `feat/lossy-long-prefill`, open the PR against `master` of `ashalliants/ninfer-3090` with `.github/pull_request_template.md` filled: measured tables, deviations (teacher-forced KL dropped, scout uses full pages, scout steps not layer-sliced), what was not run.
