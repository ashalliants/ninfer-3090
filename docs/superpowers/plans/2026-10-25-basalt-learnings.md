# Basalt learnings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement every adoptable item of issue #383 in one draft PR. Each item is one commit
behind an off-by-default option, compiled for sm_86, and verified by CPU-only tests now and by
GPU tests and A/Bs later.

**Architecture:**
- Each item lands as a self-contained commit: code, tests, docs and its option.
- The decode-path quality gate comes first so later numerics changes have a judge.
- Device work keeps a host-callable core (quantizers, planners, exact emulations of kernel
  arithmetic). CPU tests pin that core against the FP64 or exact oracle that
  `docs/maintainer/op-development.md` requires; GPU tests qualify the kernels against the same
  oracle and are compiled here but not run.
- A final script runs every GPU test and A/B on the GPU box.

**Tech Stack:**
- C++20, CUDA 12.8 (sm_86), CMake/Ninja.
- Tests: `ninfer_add_test` with the cpu and host-only label lists.
- Python 3.11 tooling; PowerShell and bash for the GPU-box scripts.

**Spec:** `docs/superpowers/specs/2026-10-25-basalt-learnings-design.md`: issue #383 plus the agreed
decisions. Item numbers below are the spec's. F = Flash-Next, X = cross-model, S = serving.

## Global Constraints

- Branch `feat/basalt-learnings` off `origin/master` (0312a1d7); one draft PR. One commit per task,
  Conventional Commit subject (`perf(qwen4exp): ...`, `feat(serve): ...`), body ending with the
  session's `Co-Authored-By` trailer.
- **Options:**
  - Every speed or numerics change sits behind an option that defaults to off.
  - Exceptions, stated in the commit: deletions with no behaviour change (Task 6), and the n-gram
    keep-alive default (Task 5).
  - Off must leave behaviour and bits exactly as on master.
- **Flash-Next options:**
  - A field in `FlashNextOptions` (`include/ninfer/types.h:340-372`), copied into `ProgramOptions`
    in `src/runtime/engine/qwen4_exp_instance.cpp:312-345`, with a field in
    `src/models/qwen4_exp/program/program.h:338-387`.
  - A serve flag parsed in `src/serve/serve_options.cpp` (pattern at `:655-656`) with usage text
    (`:177`, `:252-254`) and a case in `tests/test_serve_options.cpp`.
  - A row in the Flash-Next option table at `docs/serving.md:166-174`.
  - Options that reach `Forward` go through `ForwardExperts` (`forward.h:51-68`) or a new
    `ForwardOptions` struct passed to the `Forward` ctor (`forward.h:240-246`, built at
    `program.cpp:922-923`); Task 7 creates `ForwardOptions`.
- **Cross-model options:**
  - A field in `EngineOptions` (types.h) or, when per request, in `SamplingOverrides` /
    `ResolvedSamplingParameters` / `ops::SamplingConfig` (`sampling.h:44-54`).
  - Request keys are an external contract: update `tests/test_openai_schema.cpp` (and the
    Anthropic schema test) and `docs/serving.md` in the same commit.
- **Hardware and dependencies:**
  - sm_86 only.
  - No TMA, `cp.async.bulk`, mbarrier `expect_tx`, PDL/`griddepcontrol`, thread-block clusters or
    DSMEM, and no FP4/FP8 mma.
  - Use `cp.async` (16-byte), `mma.sync` s8/f16 and dp4a.
- **Numerics:**
  - Every new floating-point Op has a naive FP32/FP64 oracle; codecs have an exact oracle
    (op-development.md §3, §6).
  - "Bit-identical" claims are tested bitwise, against the existing route, in a `gpu` test.
- **Tests:**
  - A new test is `gpu` by default.
  - CPU tests go on `NINFER_CPU_ONLY_TESTS` (`tests/cmake/TestGpuClasses.cmake`). Those that link
    only `ninfer_core` (no `ninfer_ops`/model/engine/serve) also go on `NINFER_HOST_ONLY_TESTS`
    (`tests/cmake/HostOnlyTests.cmake`).
  - Python self-tests are `add_test(... ${Python3_EXECUTABLE} -I <script>)` on both lists
    (pattern `tests/cmake/BenchSupportTests.cmake:24-39`).
- **Basalt-derived code** (MIT, commit `8d73a39`):
  - Add `third_party/basalt/LICENSE` (Task 1).
  - Head each adapted file with `// Adapted from Basalt 8d73a39 <path> (MIT, third_party/basalt/LICENSE).`
- **Docs:**
  - Docs change in the same commit as the behaviour they describe (AGENTS.md "Change consistency").
  - Flash-Next measurement notes go in `docs/maintainer/flash-next-plan.md`; precision-changing
    items also note the numerics change there.
  - Nothing may claim a GPU measurement.
- **Build and test commands:**
  - Full tree: `source ~/ninfer-env.sh && cmake --build ~/build/ninfer-basalt -j8` (configured
    sm_86, tests and apps on).
  - Host-only tree: `~/build/ninfer-basalt-host` (Task 1 configures it with
    `-DNINFER_HOST_ONLY=ON -DBUILD_TESTING=ON`).
  - Run a CPU test with `cd ~/build/ninfer-basalt && CUDA_VISIBLE_DEVICES=-1 ctest -L cpu -R '^<name>$' --output-on-failure`.
  - A GPU test step is "compiles": `cmake --build ~/build/ninfer-basalt -j8 --target ninfer_tests`
    succeeds and `ctest -N -R '^<name>$'` lists it.
- Python: Python 3.11, and `python3.11 -m py_compile` on every changed script.
- **Unsupported configurations are refused.** When an option cannot run in some configuration
  (two lanes, vision, MTP off, another model), `Qwen4ExpInstance` (or the Qwen3.5 startup check)
  refuses it at startup with a message naming the flag, and the owning task adds a CPU test for
  the refusal. An option that does work there says so in its doc row.
- **Every new device allocation is planned.** Buffers, side-stream workspaces, channels and proxy
  heads are each a named term of `ProgramDevicePlan::fixed_bytes()` (Flash-Next) or of the
  Qwen3.5 startup plan, with a CPU test asserting the term.
- Windows (MSVC) cannot be built here. Keep code portable: no POSIX outside existing seams, and
  string literals under 16 KB.

## Review Focus

1. **An option turned on in a configuration it does not support** (two lanes, vision, the MTP
   drafter's own mixers, constrained rows). The expected behaviour is a startup refusal naming the
   option, or correct output; never silently wrong output. Owning tasks add a refusal test to
   `tests/test_serve_options.cpp` or the instance's option check.
2. **New device allocations missing from the VRAM plan** (int8 HC copies, the proxy head, hot
   embedding rows, side-stream workspace, the CPU-leg channels). If missed, expert-frame sizing
   over-commits VRAM and WDDM pages device memory. Owning tasks assert their bytes appear in
   `ProgramDevicePlan::fixed_bytes()` (or the Qwen3.5 startup plan) in a CPU test.
3. **Tool calls and constrained JSON under MTP and relaxed drafting.** Arguments must stay valid
   JSON and names must be declared tools. Relaxed acceptance never applies to special/added tokens,
   inside `<tool_call>`, with a grammar, or for greedy requests. Tasks 8 and 23 test the
   eligibility table.
4. **A server bound to `0.0.0.0` or a LAN name.** Requests with `Host: <lan-ip>:<port>` or the
   machine's hostname must keep working after the Host check, and `--cors` users must keep
   working. Task 25 tests these cases.
5. **`--config` with a boolean set true in the file and the user trying to undo it on the command
   line.** The expected behaviour is a clear error naming the key, not a silent keep. Unknown keys
   name the file and the key. Task 28 tests both.

---

## Phase 1: quality gate (X-A)

### Task 1: KLD base reader and row metrics (host-only)

**Files:**
- Create: `third_party/basalt/LICENSE` (copy of `/tmp/basalt-src/basalt/LICENSE`)
- Modify: `apps/perplexity/evaluation.h`, `apps/perplexity/evaluation.cpp`
- Test: `tests/test_perplexity_evaluation.cpp` (already cpu and host-only, `tests/cmake/ProductTests.cmake:13-19`)

**Interfaces:**
- Produces:
  - `struct KldBase { std::uint32_t n_ctx; std::int32_t n_vocab; std::int32_t n_chunk; std::vector<std::int32_t> tokens; std::filesystem::path path; }`
  - `KldBase read_kld_base(const std::filesystem::path&)`: header and tokens only; throws `std::runtime_error` naming the file on a bad magic, a truncated file or a size mismatch.
  - `std::vector<double> kld_base_logprobs(const KldBase&, int chunk, int row)`: decodes one row (`scale*q + min_lp`, `n_vocab` values) from the file. The row offset is `20 + n_chunk*n_ctx*4 + (chunk*rows + row)*row_u16*2`, where `rows = n_ctx-1-n_ctx/2` and `row_u16 = 2*((n_vocab+1)/2)+4`.
  - `struct ScoredRow { double nll, nll_base, kld; bool same_top1; double dlogp_top1; }`
  - `ScoredRow score_row(std::span<const float> test_logits, std::span<const double> base_logprobs, std::int32_t target, std::int32_t valid_vocab)`: FP64 log-softmax over `[0, valid_vocab)`. KLD(base‖test) skips terms with base log-prob ≤ -16. Top-1 ties go to the lower id.
  - `struct KldSummary { double ppl, ppl_err, kld_mean, kld_median, kld_p99, kld_max, same_top1; std::size_t rows; }`
  - `KldSummary summarize(std::span<const ScoredRow>)`: percentiles by nearest rank.
  - `std::vector<std::pair<int,int>> decode_windows(int first, int end, int width)`: half-open windows covering `[first, end)` in order, the last one partial.

- [ ] **Step 1: Write failing tests** in `test_perplexity_evaluation.cpp`:
  - `kld_round_trip_odd_vocab`: write a synthetic file with n_vocab=5, n_ctx=8, n_chunk=2. `read_kld_base` returns the same tokens, and each `kld_base_logprobs` row equals the encoded values within `scale/2`.
  - `kld_rejects_bad_files`: bad magic, a file cut 1 byte short, and n_vocab ≤ 0 all throw.
  - `score_row_identical_is_zero`: test logits equal the base log-probs gives `kld < 1e-12` and `same_top1`.
  - `score_row_hand_case`: a 4-vocab case with values worked out by hand to 1e-12; one base term at -17 is skipped.
  - `summarize_percentiles`: KLDs 1..100 give median 50, p99 99 and max 100; PPL = exp(mean nll).
  - `decode_windows_cover`: for `first=4, end=15` and width 1..16, the windows tile `[4,15)` exactly once, in order.
- [ ] **Step 2:** configure the host-only tree (`cmake -S . -B ~/build/ninfer-basalt-host -G Ninja -DNINFER_HOST_ONLY=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release`), build, and run `ctest -R '^ninfer_perplexity_evaluation_test$'`. Expected: the new cases fail to compile.
- [ ] **Step 3:** implement the functions above in `evaluation.cpp`. Read rows with positional reads (`std::ifstream` seek), not a whole-file load: a 100-chunk base is several GB.
- [ ] **Step 4:** run the test in both trees. Expected: PASS.
- [ ] **Step 5:** commit `feat(perplexity): read llama.cpp KLD bases and score rows on the host`.

### Task 2: Decode-path scoring for Qwen3.5 and `ninfer-perplexity --kld-base`

**Files:**
- Modify:
  - `include/ninfer/engine.h:83` (new overload)
  - `src/runtime/engine/engine.cpp:749-775`
  - `src/runtime/engine/causal_score_core.h:114-136`
  - `src/models/qwen3_5/program/program.h:454`, `program.cpp:150`, `program_impl.cpp:543-697`
  - `src/models/qwen3_5/program/planning/startup.cpp:203-224, 528-549`
  - `apps/perplexity/main.cpp:76-80, 276-292`
  - `docs/perplexity.md`
- Test: `tests/models/qwen3_5/test_engine_score_real.cpp` (gpu; skips without `NINFER_TEST_ARTIFACT`); `tests/test_perplexity_options.cpp` if the CLI parser is separable, otherwise a case in the existing app test.

**Interfaces:**
- Consumes: Task 1 (`read_kld_base`, `kld_base_logprobs`, `score_row`, `summarize`, `decode_windows`).
- Produces:
  - `struct ScoringOptions { enum class Phase { Prefill, Decode } phase = Phase::Prefill; int window = 1; }` in `types.h`, with window in 1..16.
  - `class ScoreRowSink { public: virtual void row(std::int64_t position, std::span<const float> logits) = 0; }`
  - `Engine::score_tokens(tokens, first_target, const ScoringOptions&, ScoreRowSink*)`.
  - `EngineOptions::scoring_decode_window` (0 = no decode scoring planned). It plans replay records and a Verify workspace at that width, and relaxes the guard at `program_impl.cpp:548` for CUDA graphs and the speculative backend only.

- [ ] **Step 1:** write GPU tests in `test_engine_score_real.cpp`:
  - `decode_phase_matches_prefill`: T=1 and T=4 logprobs are within the target_logprobs criterion of prefill scoring on the same tokens.
  - `decode_windows_agree`: T=1 against T=4 for every row.
  - `row_sink_sees_every_target`: each position in `[first_target, n)` exactly once.
- [ ] **Step 2:** build and confirm the tests compile (GPU test step).
- [ ] **Step 3:** implement the decode phase in `ProgramImpl::causal_score`:
  - Prefill `[0, first_target)` as today.
  - Then teacher-force `decode_windows(first_target, n, T)`:
    - T=1: `ordinary_decode_batch`.
    - T≥2: `target_verify_batch`, then `GdnReplayFoldPlan::execute{commit_columns=valid}` and `commit_frontier`.
  - Copy each window's BF16 logits to the host as FP32 and call the sink.
- [ ] **Step 4:** add to `ninfer-perplexity`:
  - Flags `--kld-base FILE`, `--chunks N`, `--decode-window T` (default 3, like Basalt's `--quality-t`).
  - Read the base and check that each chunk's prompt half equals the base tokens.
  - Score the second half of each `n_ctx` chunk through the sink, then print `summarize()` and write per-row TSV to `--output`.
  - Use the public token count as `valid_vocab` and take n_ctx from the file. Rows past the public count are excluded, because llama.cpp may pad to 248,320.
  - Update the usage text at `main.cpp:76-80`: verify-only routes are no longer inert with `--decode-window`.
- [ ] **Step 5:** build `ninfer-perplexity` and `ninfer_tests`. Run `ninfer-perplexity --help` (no GPU needed). Expected: the new flags are listed.
- [ ] **Step 6:** document in `docs/perplexity.md`: the method (prompt half prefilled, second half decoded), the reference noise of 0.01-0.02 KLD found by Basalt, and comparing builds against each other.
- [ ] **Step 7:** commit `feat(perplexity): score through the decode path against a KLD base`.

### Task 3: Flash-Next KLD gate in the forward harness

Flash-Next has no CausalScoring (`qwen4_exp_instance.cpp:126`). Its teacher-forced harness,
`ninfer_qwen4_exp_forward_real_test` (PR 8), already scores texts.

**Files:**
- Modify: `tests/models/qwen4_exp/test_forward_real.cpp`, `tests/models/qwen4_exp/tests.cmake` (link the evaluation sources from Task 1), `docs/maintainer/flash-next-plan.md` (how to run it)

**Interfaces:**
- Consumes: Task 1.
- Produces: `--kld-base FILE --chunks N --window T` modes of the forward harness.
  - It prefills each chunk's first half as one call.
  - It then decodes the second half T columns at a time with the forced reference tokens, through the same single-sequence decode/verify path the Program uses.
  - It prints `summarize()` and writes per-row TSV.
  - Later tasks' options are passed as harness flags (`--hc-int8`, `--gdn-head-map`, ...); each task adds its own flag.

- [ ] **Step 1:** add the mode. Reuse `score_row` and `summarize`, with no new math.
- [ ] **Step 2:** build. Expected: the test compiles. It is a gpu test that skips without arguments.
- [ ] **Step 3:** commit `test(qwen4exp): score the decode path against a KLD base in the forward harness`.

## Phase 2: Flash-Next host-side and small items

### Task 4: Pure draft-length and joint-acceptance policies (F3, host part)

**Files:**
- Create: `src/models/qwen4_exp/program/draft_policy.h` (header-only, no CUDA)
- Modify: `src/models/qwen4_exp/program/program.cpp:3881-3896` (call the extracted function)
- Test: `tests/models/qwen4_exp/test_draft_policy.cpp`, new, cpu and host-only (links nothing but the header)

**Interfaces:**
- Produces:
  - `int ema_draft_length(std::span<const double> accept, int limit, double width_cost)`: today's `choose_draft_length` body, unchanged.
  - `float joint_acceptance_table(float p)`: Basalt `draft_cum_acc` bins 0.30/0.50/0.70/0.80/0.90/0.95/0.99 → 0.12/0.26/0.43/0.51/0.59/0.66/0.79/0.96.
  - `int joint_acceptance_length(std::span<const float> probs, float threshold)`: the largest n with `prod_{i<n} table(p_i) >= threshold`.
  - `class JointAcceptanceThreshold { float base = 0.35f, low = 0.25f; double obs, exp; float current() const; void observe(std::span<const float> probs, int accepted); }`, with decay 0.97 and ramp 0.9..1.1, as Basalt's `DraftCumAdapt`.

- [ ] **Step 1:** write the tests:
  - `ema_matches_master`: the extracted function on recorded EMA vectors equals hand-computed master results.
  - `joint_table_bins`: every bin edge, including p=0.99 → 0.96 and p=0.2999 → 0.12.
  - `joint_length_stops`: probs {0.99, 0.99, 0.6} at th 0.35 gives 2.
  - `joint_threshold_adapts`: accepting more than the table predicts moves `current()` toward 0.25 and never below; accepting less returns toward 0.35.
- [ ] **Step 2:** run. Expected: FAIL (header missing).
- [ ] **Step 3:** implement the header (Basalt attribution). Replace the body of `choose_draft_length` with a call to `ema_draft_length`.
- [ ] **Step 4:** run the test (host tree and full tree) and `ninfer_qwen4_exp_*` cpu tests. Expected: PASS.
- [ ] **Step 5:** commit `refactor(qwen4exp): draft-length policies as pure functions with a joint-acceptance rule`.

### Task 5: N-gram keep-alive interval option, default 40 ms (F7a)

**Files:**
- Modify: `program.cpp:686`; the FlashNextOptions plumbing (Global Constraints)
- Test: `tests/models/qwen4_exp/test_ngram_volume.cpp:208-220`; `tests/test_serve_options.cpp`

**Interfaces:**
- Produces: `FlashNextOptions::ngram_keepalive_ms = 40` and `--ngram-keepalive-ms N` (0 disables).

- [ ] **Step 1:** add a test that a keep-alive of 40 ms issues a read within 3× the interval of idleness and none before half of it, using the counters at `:208-220`. The existing 20 ms check (`keep-alive reads every 20 ms of idleness`) already flakes under ctest on this WSL host: keep the new windows generous and leave the old check alone. Add a serve-options test that the flag parses and that the default is 40.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** plumb the option. Pass `std::chrono::milliseconds(options_.ngram_keepalive_ms)` at `:686`.
- [ ] **Step 4:** run the tests. Expected: PASS. Note in the commit that the default changed (I/O only, no numerics).
- [ ] **Step 5:** commit `perf(qwen4exp): keep the n-gram drive awake every 40 ms`.

### Task 6: Unpinned CPU expert workers (F10e)

**Files:**
- Modify:
  - `src/ops/offloaded_sparse_moe/cpu/expert_team.cpp:14-22, 32-35, 48-60, 201-206`
  - `src/ops/offloaded_sparse_moe/cpu/miss_service.cpp:84, 89`
  - `src/ops/offloaded_sparse_moe/cpu/expert_cpu.h:38-40`
  - `include/ninfer/ops/offloaded_sparse_moe.h:174, 335`
- Test: `tests/ops/test_offloaded_moe_cpu.cpp` (cpu)

- [ ] **Step 1:** delete `Options::cpus` (both structs) and `pin_current_thread`, with its includes. No caller sets them (recon).
- [ ] **Step 2:** build. Run `ninfer_offloaded_moe_cpu_test`. Expected: PASS (team = scalar at 1-16 workers).
- [ ] **Step 3:** commit `refactor(moe): stop pinning CPU expert workers to cores` (deletion, no option; Basalt d5273cb found that a pinned worker preempted by another task stalls the join).

### Task 7: Shared expert on a side stream (F4a)

**Files:**
- Modify:
  - `src/models/qwen4_exp/execution/forward.h:240-246` (new `ForwardOptions`), `forward.cpp:531-591, 1040-1051`
  - `program.cpp:922-923`
  - option plumbing
- Test: `tests/models/qwen4_exp/test_forward_real.cpp` (gpu): `--shared-expert-stream` logits bit-identical to off

**Interfaces:**
- Produces:
  - `struct ForwardOptions { bool shared_expert_stream = false; }`. Later tasks add their own fields here (`gdn_head_map`, `hc_int8`, `hc_fused_read`, `hot_embedding_rows`, the `moe_*` switches), each in its own commit.
  - `FlashNextOptions::shared_expert_stream`, `--shared-expert-stream`.
- Design:
  - Forward owns a `cudaStreamCreateWithPriority` stream (greatest priority) and two events per use.
  - Record an event on `s`, have the side stream wait on it, then run the shared gate/up/silu/down on the side stream from a carved `WorkspaceArena` of its own (the bump arena must not be shared across streams). Record the join and have `s` wait on it before `moe_combine`. All of this is capture-safe.
  - Also in the MTP block.

- [ ] **Step 1:** add the GPU identity check to the forward harness (`--shared-expert-stream`, compare logits bitwise with a run without it).
- [ ] **Step 2:** implement. Build. Expected: compiles.
- [ ] **Step 3:** serve-options test for the flag (cpu). Run it. Expected: PASS.
- [ ] **Step 4:** commit `perf(qwen4exp): run the shared expert beside the routed experts (opt-in)`.

### Task 8: MTP for constrained requests (F2)

**Files:**
- Modify:
  - `program.cpp:153-160` (`make_speculative` carries `constraint_failed`)
  - `:583, 723, 746` (mask staging sized `mask_words*(drafts+1)`)
  - `:3900-3913` (guard)
  - `:3946-3972` (fill masks after the front replay, install them in `install_round_config`)
- Create: `src/models/qwen4_exp/program/constrained_window.h` (pure)
- Test: `tests/models/qwen4_exp/test_constrained_window.cpp` (cpu, host-only); a GPU case in `tests/models/qwen4_exp/test_engine_real.cpp`

**Interfaces:**
- Consumes: `runtime::TokenMaskProvider::fill(row, drafts, words) -> dead bits` (`src/runtime/contract/execution.h:11-20`); the Qwen3.5 precedent at `qwen3_5/program/grammar_masks.cpp:24-41` and `transactions/commit.cpp:45-48`.
- Produces:
  - `int constrained_accept_limit(std::uint32_t dead_bits, int count)`: the number of leading positions not dead, so later drafts are dropped as rejections.
  - `FlashNextOptions::mtp_constrained`, `--mtp-constrained`.
- Design:
  - Design (b) from recon: fill the K+1 per-position masks from `drafts_` on the host while the front replay runs.
  - Masked configs make the accept op take a masked argmax or sample per column, which is exact for greedy and sampled rows.
  - Dead bits become `constraint_failed`.

- [ ] **Step 1:** write host tests:
  - `limit_all_live`: dead bits 0 with count 4 gives 4.
  - `limit_first_dead`: dead bits 0b0100 with count 4 gives 2.
  - `limit_masks_high_bits`: bits above count are ignored.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the header and the Program changes behind the option. When the option is off the guard at `:3902` is unchanged.
- [ ] **Step 4:** write the GPU test `mtp_constrained_matches_plain`: greedy ids for a JSON-schema request with `--mtp-constrained` equal plain constrained decoding. It uses the Engine's mask provider with a simple schema. Build. Expected: compiles.
- [ ] **Step 5:** run the host tests and serve-options tests. Expected: PASS.
- [ ] **Step 6:** commit `feat(qwen4exp): keep MTP on for constrained requests with per-draft masks (opt-in)`.

### Task 9: Draft probability Op and joint-acceptance draft length (F3, device part)

**Files:**
- Create:
  - `include/ninfer/ops/argmax_probability.h`
  - `src/ops/kernel/argmax_probability.cu` (or beside `argmax` in its directory)
- Modify:
  - `src/models/qwen4_exp/execution/forward.cpp` ~1060-1072 (drafter head writes the probability next to `spec.w_drafts`)
  - `forward.h:142-158` (`MtpCall::probs`)
  - `program.cpp:3842-3874, 3881-3896`
  - option plumbing
- Test: `tests/ops/test_argmax_probability.cpp` (gpu, FP64 oracle); `tests/models/qwen4_exp/test_draft_policy.cpp` (from Task 4)

**Interfaces:**
- Consumes: Task 4 (`joint_acceptance_length`, `JointAcceptanceThreshold`).
- Produces:
  - `void ops::argmax_probability(const Tensor& logits /*F32 [V,T]*/, Tensor& ids /*I32 [T]*/, Tensor& probs /*F32 [T]*/, cudaStream_t)`. It computes `p = 1/Σ exp(l - lmax)`; ties go to the lower id, matching `ops::argmax`.
  - `FlashNextOptions::mtp_joint_acceptance`, `--mtp-joint-acceptance`.
- Design:
  - Phase A: draft the full ceiling K in the existing per-step-count graph.
  - After the sync at `:3869`, read the probabilities and truncate the verify window to `joint_acceptance_length(probs, threshold.current())`, at least 1 when drafting.
  - Then call `threshold.observe(...)` after commit.
  - Early stop inside drafting is out of scope; record that in flash-next-plan.md.

- [ ] **Step 1:** write `test_argmax_probability.cpp` (gpu): ids equal the FP64 argmax and probs are within 1e-6 relative at V=248,320 and T=1..8, including ties.
- [ ] **Step 2:** implement the Op and the wiring. Build. Expected: compiles.
- [ ] **Step 3:** run the Task 4 host tests and the serve-options tests. Expected: PASS.
- [ ] **Step 4:** commit `feat(qwen4exp): trim MTP windows by joint draft acceptance (opt-in)`.

### Task 10: N-gram prefetch for the anchor column during drafting (F7b)

**Files:**
- Modify:
  - `program.cpp:3866-3869` (warm the cache between the draft replay and its sync)
  - `:2756-2779` (extract the row-id computation)
  - option plumbing
- Create: `src/models/qwen4_exp/program/ngram_rows.h` (pure)
- Test: `tests/models/qwen4_exp/test_ngram_volume.cpp` (cpu)

**Interfaces:**
- Produces:
  - `std::vector<std::uint64_t> ngram_window_rows(const NgramHash&, std::span<const std::int32_t> history, std::span<const std::int32_t> drafts, std::int64_t begin, int width)`: the rows `upload_rows` reads today.
  - `FlashNextOptions::ngram_prefetch`, `--ngram-prefetch`.

- [ ] **Step 1:** write tests:
  - `window_rows_match_inline`: rows equal what the inline code produced on fixed histories, using `NgramHash::row_ids`.
  - `prefetch_then_cached`: on a synthetic volume, `read_rows` of the anchor rows, then `cached()` is true for each.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement. The prefetch runs on the Program thread, because `read_rows` is not reentrant (`ngram_volume.h:42`).
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `perf(qwen4exp): read the anchor column's n-gram rows while drafting (opt-in)`.

### Task 11: Default expert state shipped beside the model (F10a)

**Files:**
- Modify:
  - `program.cpp:4223-4241` (warm start falls back to the default file)
  - `src/models/qwen4_exp/program/expert_cache/expert_state.{h,cpp}` (identity `artifact_id`)
  - `qwen4_exp_instance.cpp:336-344`
  - option plumbing
  - `docs/serving.md`
- Test: `tests/models/qwen4_exp/test_expert_cache.cpp:532` area (cpu)

**Interfaces:**
- Produces:
  - `FlashNextOptions::default_expert_state_path` (empty uses `<artifact>.expert-state.default`), `--default-expert-state FILE`.
  - `SavedState load_saved_state(path, Identity)` accepts the identity `artifact_id=<id>` for default files.
  - `ninfer_qwen4_exp_expert_state_export` (a tiny CLI in `tools/` or a mode of an existing app) rewrites a session state with the artifact identity, for the release.
- Behaviour: the primary file wins. The default is read only when the primary is missing or its identity does not match. A default is never written.

- [ ] **Step 1:** write tests:
  - `default_used_when_primary_missing`
  - `primary_wins`
  - `default_identity_is_artifact_id`
  - `export_round_trip`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `feat(qwen4exp): warm-start a fresh install from a shipped expert state`.

### Task 12: Hot embedding rows in VRAM, the rest in host memory (F10b)

**Files:**
- Modify:
  - `include/ninfer/ops/embedding.h:29` (two-plane variant)
  - the embedding kernel file
  - `src/models/qwen4_exp/load.cpp:321`; `plan_load` flag in `model.h`
  - `memory_plan.h:32` (`other_pinned`)
  - `ProgramDevicePlan` (`program.h:318-336`)
  - `forward.cpp:272, 629, 982`
  - option plumbing
- Test: `tests/ops/test_embedding.cpp` (gpu, exact oracle); `tests/models/qwen4_exp/test_memory_plan.cpp` (cpu)

**Interfaces:**
- Produces:
  - `void ops::embedding_split(const Tensor& ids, const Weight& hot /*device rows [0,H)*/, const Weight& cold /*host-mapped, all rows*/, std::int32_t hot_rows, Tensor& out, cudaStream_t)`: the same decode, bitwise, as `ops::embedding`.
  - `FlashNextOptions::hot_embedding_rows` (0 = off), `--hot-embedding-rows N`. Basalt uses 8192.
  - The memory plan moves `rows*row_bytes` from `LoadPlan::device_bytes` to `ProgramDevicePlan` for hot rows, and the whole table to `HostMemoryDemand::other_pinned`.

- [ ] **Step 1:** write the CPU test `hot_embedding_ledgers`: with N=8192 the device plan shrinks by `(V-N)*row_bytes` and host pinned demand grows by `V*row_bytes`; with N=0 nothing changes. Write the GPU test `embedding_split_matches_embedding`: ids across the boundary give bitwise-equal outputs.
- [ ] **Step 2:** run the CPU test. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS. The GPU test compiles.
- [ ] **Step 5:** commit `perf(qwen4exp): keep only hot embedding rows in VRAM (opt-in)`.

## Phase 3: Flash-Next kernels

### Task 13: GDN with a tiled key-head map (F5)

**Files:**
- Modify:
  - `include/ninfer/ops/gated_delta_net.h:31-33` (head-map enum)
  - `src/ops/linear_attention/gated_delta_net/common.cuh:97-116` (`head_map`), `recurrent.cuh:182, 444`
  - replay `replay.cpp:107-108, 163`, `recurrent.cu:272` (fold geometry 16×48)
  - `src/models/qwen4_exp/execution/forward.cpp:69-78, 424-442, 730-736`
  - `program.cpp:441-446` (`records_spec`)
  - option plumbing
- Test: `tests/ops/gdn_ref.h:33` (tiled mode) with `tests/ops/test_gdn_head_map.cpp` (new, cpu, host-only); `tests/ops/test_gated_delta_net*.cpp` (gpu)

**Interfaces:**
- Produces:
  - `enum class GdnHeadMap : std::uint8_t { Blocked, Tiled }`, with `qk_head(h, Hv, Hqk, map)`: Blocked is `h / (Hv/Hqk)`, Tiled is `h % Hqk`.
  - A `GdnHeadMap map = GdnHeadMap::Blocked` field on the `gated_delta_net`, `gated_delta_net_replay_record` and `gated_delta_net_batch_update` argument structs.
  - `ForwardOptions::gdn_head_map`, `FlashNextOptions::gdn_head_map`, `--gdn-head-map`.

- [ ] **Step 1:** write `test_gdn_head_map.cpp`. `tiled_oracle_equals_expanded` checks that the FP64 oracle on 16-head q/k with the Tiled map equals, bitwise, the oracle on q/k expanded to 48 heads by tiling with the Blocked identity map. Use 3 steps of state at head 128.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the oracle mode, then the kernels, replay geometry and forward, behind the option. Skip `tile_heads` when it is on.
- [ ] **Step 4:** run the CPU test. Expected: PASS. Add GPU cases (Tiled map against expanded Blocked, bitwise outputs and state). Expected: compiles.
- [ ] **Step 5:** add the forward-harness flag `--gdn-head-map` with a logits identity check against off.
- [ ] **Step 6:** commit `perf(qwen4exp): read tiled GDN key heads through a head map instead of copies (opt-in)`.

### Task 14: Quantize the FFN input once per MoE call (F4b)

**Files:**
- Modify:
  - `include/ninfer/ops/offloaded_sparse_moe.h:251-276, 296` (`MoeExpertSource::precast_input`, workspace bytes)
  - `src/ops/offloaded_sparse_moe/cuda/moe_layer.cu:496-555, 938-963, 1049-1122`
  - `forward.cpp:558-567`
  - option plumbing
- Test: `tests/ops/test_offloaded_moe_layer.cpp` (gpu)

**Interfaces:**
- Produces:
  - `MoeExpertSource::precast_input` (bool). With it set, `moe_experts` runs one A8 cast of `x` into the workspace (`[T][2560]` i8 + `[T][80]` f32, `canon::a8_quantize_bf16`), and `gate_up_kernel` reads the codes instead of calling `quantize_columns_bf16`.
  - `moe_experts_workspace_bytes` grows by `T*(2560+320)` when it is set.
  - `FlashNextOptions::moe_precast_input`, `--moe-precast-input`.

- [ ] **Step 1:** add GPU cases `precast_bitwise`: GPU outputs with precast equal the CPU engine bitwise at T ∈ {1,2,7,8,16}, every placement.
- [ ] **Step 2:** implement. Build. Expected: compiles.
- [ ] **Step 3:** run `ninfer_offloaded_moe_cpu_test`. Expected: PASS (unchanged).
- [ ] **Step 4:** commit `perf(moe): cast the FFN input to A8 once per call (opt-in)`.

### Task 15: Register-resident gate/up kernel with a balanced tail (F4c)

**Files:**
- Modify: `moe_layer.cu:439-555, 626-686` (new kernel variant and its launcher), `MoeExpertSource::gate_up_variant`, option plumbing
- Test: `tests/ops/test_offloaded_moe_layer.cpp` (gpu); `tests/ops/test_offloaded_moe_cpu.cpp` (cpu, the grid helper)

**Interfaces:**
- Consumes: Task 14 (reads precast codes; requires `precast_input`).
- Produces:
  - `enum class GateUpVariant : std::uint8_t { Staged, Registers }`.
  - `int coresident_blocks(int sms, int blocks_per_sm)`, which returns `(sms - sms/8) * blocks_per_sm`; a pure helper in a host-visible header.
  - `FlashNextOptions::moe_gate_up_registers`, `--moe-gate-up-registers`. Turning it on without `--moe-precast-input` is refused at startup.
- Design:
  - Two rows per warp, weights loaded straight to registers, no shared-memory weight staging.
  - The third pass of an 80-sub-block row: lanes 0-15 take row A at sub-block 64+l; lanes 16-31 take row B at 64+(l-16) and hand `(c·d, P)` to lane l-16 with `__shfl_down_sync(…,16)` before its fma. Each row's lane order and fma sequence are therefore unchanged.
  - A co-resident grid loops over (row block × job) items, sized with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

- [ ] **Step 1:** write the CPU test `coresident_blocks_values`: 82 SMs at 2 blocks gives 144; 1 SM gives 2. Write GPU cases `registers_bitwise` (same matrix as Task 14).
- [ ] **Step 2:** run the CPU test. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run the CPU test. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(moe): gate/up from registers with a balanced last pass (opt-in)`.

### Task 16: One-kernel deterministic route and dispatch for T ≤ 8 (F4d, narrowed)

Recon found that planning inside gate/up cannot coexist with the staging pass that decode needs.
The narrower change fuses `route_kernel` and `dispatch_small_kernel` into one launch, with a
deterministic order of entries within each expert.

**Files:**
- Create: `src/ops/offloaded_sparse_moe/dispatch_plan.h` (host-callable)
- Modify:
  - `moe_layer.cu:227-398` (fused kernel)
  - `include/ninfer/ops/offloaded_sparse_moe.h:204-226` (document the deterministic order when the fused path is used)
  - `forward.cpp:542-557`
  - option plumbing
- Test: `tests/ops/test_moe_dispatch_plan.cpp` (cpu, host-only); `test_offloaded_moe_layer.cpp` (gpu: device dispatch arrays equal `plan_dispatch`)

**Interfaces:**
- Produces:
  - `struct DispatchPlan { std::vector<std::int32_t> counts, offsets, jobs, entries; }`.
  - `DispatchPlan plan_dispatch(std::span<const std::int32_t> ids /*k*T, column-major*/, int experts)`:
    - jobs are ascending expert ids;
    - entries within a job are in ascending entry index;
    - out-of-range ids are dropped.
  - `FlashNextOptions::moe_fused_route`, `--moe-fused-route`.

- [ ] **Step 1:** write host tests:
  - `plan_counts_offsets`
  - `plan_jobs_ascending`
  - `plan_entries_ascending`
  - `plan_drops_out_of_range`
  - `plan_t8_k10` (80 entries, repeated experts)
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the header and the fused kernel (T ≤ 8, one CTA).
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(moe): route and dispatch in one deterministic launch for decode (opt-in)`.

### Task 17: CPU leg for misses in verify windows (F4e)

**Files:**
- Create: `src/ops/offloaded_sparse_moe/cpu_selection.h` (host-callable mirror of `cpu_plan_kernel`'s rule)
- Modify:
  - `program.cpp` (own a `CpuMissService` built from `parameters_.model.expert_banks()`, created before the Forward and destroyed after `device_.synchronize()`; a mapped error word checked once per round)
  - `forward.h:51-68` (`ForwardExperts::cpu` channels per layer)
  - `forward.cpp:558-592` (set `source.cpu` and `source.error` when `T >= 2`; with `wait_for_cpu=false`, run the shared expert, then `moe_experts_cpu_wait` before combine)
  - option plumbing
  - `docs/maintainer/flash-next-plan.md`
- Test: `tests/ops/test_moe_cpu_selection.cpp` (cpu, host-only); `test_offloaded_moe_layer.cpp` (gpu: T ∈ {2,3,8,16} with the CPU leg, bitwise against the CPU engine)

**Interfaces:**
- Produces:
  - `std::vector<int> select_cpu_jobs(std::span<const std::int32_t> job_experts, std::span<const std::int32_t> job_columns, std::span<const std::uint8_t> resident, int cap, int divisor, int max_job_columns)`: the fewest-column jobs first, ties in job order; `want = min(cap, divisor > 0 ? M - M/divisor : M)`.
  - `FlashNextOptions::cpu_verify_misses`, `--cpu-verify-misses`.
  - Two-lane plain decode also reaches T=2. The option applies only to verify calls, so `Forward::moe` receives `bool verify` from its callers.

- [ ] **Step 1:** write host tests:
  - `fewest_columns_first`
  - `ties_in_job_order`
  - `width_cap_excludes`
  - `divisor_zero_takes_all`
  - `divisor_three`
  - `cap_limits`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement. Add a GPU case asserting that the `MissRequest` contents equal `select_cpu_jobs`.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(qwen4exp): compute verify-window misses on the CPU beside the GPU (opt-in)`.

### Task 18: Async GDN commit (F10c)

**Files:**
- Modify: `program.cpp:3701, 4111-4130` (fold on a side stream; the next round's front replay waits on its event), option plumbing
- Test: `tests/models/qwen4_exp/test_engine_real.cpp` (gpu: MTP greedy ids with `--gdn-async-commit` equal off)

**Interfaces:**
- Produces: `FlashNextOptions::gdn_async_commit`, `--gdn-async-commit`.
- Constraint: terminal rows (`k < LC`, checks at `:4092-4095`) still fold before the Engine sees the round, so the async path applies only to rows that continue.

- [ ] **Step 1:** write the GPU identity test.
- [ ] **Step 2:** implement. Build. Expected: compiles. Serve-options test PASS.
- [ ] **Step 3:** commit `perf(qwen4exp): fold the GDN commit on a side stream (opt-in)`.

### Task 19: Int8 hyper-connection copies with fp32 activations (F1)

**Files:**
- Create:
  - `src/ops/hyper_connection/hc_int8_codec.h` (`NINFER_HD`, host and device)
  - `src/ops/hyper_connection/hyper_connection_mix_int8.cu`
- Modify:
  - `include/ninfer/ops/hyper_connection.h:62-70` (int8 overload; contract text)
  - `src/models/qwen4_exp/program/program.{h,cpp}` (`ProgramDevicePlan::hc_int8_bytes`, a Program-owned `DeviceBuffer` filled at construction by a quantize kernel from the BF16 weights)
  - `forward.cpp:339-346` (`Forward::mix` selects the int8 weights)
  - option plumbing
  - `docs/maintainer/flash-next-plan.md` (the ported-files table says "no Q8 weight form"; update it, and record the numerics change)
- Test: `tests/ops/test_hc_int8_codec.cpp` (cpu, host-only); `tests/ops/test_hyper_connection.cpp:77-130` (gpu; the FP64 oracle takes dequantized weights); `test_memory_plan` (cpu, plan bytes)

**Interfaces:**
- Produces:
  - `struct HcInt8Group { std::int8_t q[32]; __half scale; }` semantics:
    - `void hc_int8_quantize_group(const bf16* w, std::int8_t* q, __half* s)`: `s = f16(absmax/127)`; `q = clamp(rint(w/float(s)), -127, 127)`; a zero group gives `s = 0` and `q = 0`.
    - `float hc_int8_dequant(std::int8_t q, __half s)`.
    - `void ops::hc_int8_quantize(const Weight& bf16_rows, DeviceSpan codes, DeviceSpan scales, cudaStream_t)`.
  - `hyper_connection_mix(residual, norm, const HcInt8Weights& down, const HcInt8Weights& up, ...)`:
    - down/inject: exact int8→float conversion, fp32 `fmaf` against fp32 activations, no activation quantization;
    - up: int8→fp16 exact, the fp32 activation split into fp16 hi + lo, two `mma.sync.m16n8k16` with fp32 accumulate;
    - T ≤ 16 (the fused limit); the composed route T > 16 dequantizes to BF16.
  - `ForwardOptions::hc_int8`, `FlashNextOptions::hc_int8`, `--hc-int8`.
  - The BF16 copies are kept in this commit (VRAM cost is about 0.68 GB, so about 450 fewer frames); the plan bytes say so.

- [ ] **Step 1:** write CPU tests:
  - `codec_bounds`: |q| ≤ 127 and round-trip error ≤ s/2 on random and extreme groups.
  - `codec_zero_group`
  - `codec_subnormal_and_inf`: inf input is refused by an assert, the subnormal absmax rounds to the fp16 subnormal.
  - `codec_matches_basalt_rule`: 1,000 random groups against an independent re-implementation of `f16(absmax/127)`.
  - `hc_int8_plan_bytes`: the plan includes `97 mixers × (rows*cols + rows*cols/32*2)` bytes.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the codec, the quantize kernel, the mix kernels (cp.async-staged, no TMA) and the wiring. Extend `test_hyper_connection.cpp` with int8 cases against the FP64 oracle on dequantized weights, at T = 1, 2, 4, 8, 16, 17 and real shapes.
- [ ] **Step 4:** run the CPU tests. Expected: PASS. The GPU tests compile. Add the forward-harness flag `--hc-int8` for Task 3's KLD gate.
- [ ] **Step 5:** commit `perf(qwen4exp): int8 hyper-connection weights with fp32 activations (opt-in, numerics change)`.

### Task 20: Fused hyper-connection read: previous inject, router top-10 and A8 output (F10d)

**Files:**
- Create: `src/ops/hyper_connection/hc_fused_read.cu`
- Modify: `include/ninfer/ops/hyper_connection.h` (new Op `hc_fused_read`), `forward.cpp:321-329, 536-541`, option plumbing
- Test: `tests/ops/test_hyper_connection.cpp` (gpu: against the composed FP64 oracle; router ids and weights equal `moe_route` on the oracle logits)

**Interfaces:**
- Consumes: Task 19 (`HcInt8Weights`); requires `--hc-int8`, refused otherwise.
- Produces:
  - `void ops::hc_fused_read(residual, prev_bo, prev_inject, norm, down, up, router /*BF16 [513,2560]*/, x_out, inject_out, routing_out /*MoeRouting*/, a8_codes, a8_scales, ws, stream)`.
  - The work items are claimed from global counters, with waits only on earlier items, so the kernel cannot deadlock without co-residency.
  - `ForwardOptions::hc_fused_read`, `--hc-fused-read`.

- [ ] **Step 1:** write GPU tests against the composed path (inject, then mix, then `projection_fp32` router, then `moe_route`, then the A8 cast): equal routes, x within the HC criterion, A8 codes bitwise against `canon::a8_quantize_f32` of x.
- [ ] **Step 2:** implement. Build. Expected: compiles.
- [ ] **Step 3:** commit `perf(qwen4exp): one hyper-connection read with inject, router and A8 output (opt-in)`.

### Task 21: In-graph n-gram flag wait (F7c)

**Files:**
- Modify:
  - `program.cpp:1413-1420, 3274-3289` (one graph per round)
  - `forward.cpp:253-258`
  - a `wait_flag_ge` kernel beside the PLE ops
  - option plumbing
- Test: `tests/models/qwen4_exp/test_engine_real.cpp` (gpu: greedy ids with `--ngram-flag-wait` equal off)

**Interfaces:**
- Produces:
  - `void ops::wait_flag_ge(const volatile std::uint32_t* mapped, std::uint32_t value, std::uint32_t* error, cudaStream_t)`: it spins with `__nanosleep` and traps after the heartbeat timeout (`kHeartbeatTimeoutNs`).
  - `FlashNextOptions::ngram_flag_wait`, `--ngram-flag-wait`.
- Behaviour: the host gathers rows into the pinned buffer and raises the flag. The commit notes that this is untested under WDDM.

- [ ] **Step 1:** write the GPU identity test.
- [ ] **Step 2:** implement. Build. Expected: compiles.
- [ ] **Step 3:** commit `perf(qwen4exp): wait for n-gram rows inside the round graph (opt-in)`.

### Task 22: Prefill CPU share for short chunks (F9)

**Files:**
- Create: `src/ops/offloaded_sparse_moe/cpu_share.h` (host-callable)
- Modify:
  - `include/ninfer/ops/offloaded_sparse_moe.h:237-249` (`MoeCpuChannel::share_num/share_den` replace the unused `pcie_divisor`)
  - `moe_layer.cu:708-838` (`want` uses the share for prefill calls)
  - `src/models/qwen4_exp/execution/expert_stream.cpp:56-92` (in chunks below `kStreamMinColumns`, the CPU's experts are left out of the stream table so they are not staged)
  - `program.cpp`
  - option plumbing
- Test: `tests/ops/test_moe_cpu_selection.cpp` (cpu, host-only, extends Task 17); `test_offloaded_moe_wide.cpp` (gpu: wide route with the CPU channel, bitwise)

**Interfaces:**
- Consumes: Task 17 (`CpuMissService` ownership, `select_cpu_jobs`).
- Produces:
  - `int cpu_share_take(int streamed_experts, int candidates, int num, int den)`, which returns `min(candidates, (2*num*n + den) / (2*den))`.
  - `std::vector<int> select_prefill_cpu_jobs(job_experts, job_rows, resident, num, den, max_rows /*8*/)`: candidates have rows ≤ 8, sorted by (rows, expert id).
  - `FlashNextOptions::prefill_cpu_share` (0 = off; Basalt's 0.6 is stored as num 3, den 5), `--prefill-cpu-share F`.

- [ ] **Step 1:** write host tests:
  - `share_matches_llround`: for n ≤ 1024 at 0.6, equal to `llround(0.6*n)` (no half cases).
  - `prefill_rows_cap`
  - `prefill_sorted_by_rows_then_id`
  - `share_zero_takes_none`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(qwen4exp): compute low-row prefill experts on the CPU beside staging (opt-in)`.

## Phase 4: cross-model speculation

### Task 23: Relaxed drafting (X-C)

**Files:**
- Create: `src/ops/common/relaxed_accept.h` (`NINFER_HD`)
- Modify:
  - `include/ninfer/ops/sampling.h:44-54` (`SamplingConfig::relaxed_threshold`, `added_token_floor`)
  - `include/ninfer/ops/speculative_round.h:91-186` (contract)
  - `src/ops/kernel/speculative_round.cuh:265-277, 504-513, 846-855, 910-919`
  - `include/ninfer/types.h:540-551` (`SamplingOverrides::relaxed_drafting`, resolved field), `:1214` (`SpeculativeStats::relaxed_kept`)
  - `src/runtime/contract/sampling.h:9`
  - `src/models/qwen3_5/program/planning/request_plan.cpp:29-40`; `decode.cpp:486, 782` (zeroed for constrained rows or rows inside a tool call)
  - Flash-Next `program.cpp:3965-3972`
  - `src/serve/openai_chat_request.cpp` and the Anthropic request parser (key `relaxed_drafting`: number or `false`)
  - `src/serve/serve_options.cpp` (`--relaxed-drafting [P]`, engine default; bare = 0.5)
  - `docs/serving.md`
- Test: `tests/ops/test_relaxed_accept.cpp` (cpu, host-only); `tests/runtime/test_sampling_defaults.cpp` (cpu); `tests/test_openai_schema.cpp` and the Anthropic schema test (cpu); `tests/ops/test_speculative_round.cpp` (gpu: P=0 bitwise unchanged on all four paths; forced keeps)

**Interfaces:**
- Produces:
  - `bool relaxed_accept(float pd, float qd, float u, float threshold, bool eligible)`: the exact rule, or `eligible && threshold > 0 && pd >= threshold`.
  - `bool relaxed_position_eligible(bool row_ok, bool in_call_at_start, std::span<const std::int32_t> drafts_through_i, std::int32_t added_token_floor)`: false once any draft so far is ≥ the floor.
  - `bool ledger_in_tool_call(std::span<const std::int32_t> ids, std::int32_t open_id, std::int32_t close_id)`.
  - The added-token floor is computed once at startup from the tokenizer. Startup asserts the added ids are contiguous from the floor.

- [ ] **Step 1:** write host tests:
  - `p0_is_exact_rule`: dense one-hot and sparse q on a grid of (pd, qd, u).
  - `keeps_at_threshold`
  - `special_draft_never_relaxes`
  - `after_special_never_relaxes`
  - `in_call_ledger`: open without close is true; close after open is false.
  - `resolve_override`: a request value wins; `false` gives 0; absent uses the engine default.
  - schema tests: `relaxed_drafting` accepts 0.3, `false` and `true` (gives 0.5), and rejects -1 and 2 with the documented error code.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the header, then plumbing, then kernels, behind threshold 0 = off. Greedy rows never reach the relaxed paths, `logit_bias` is already refused (`openai_chat_request.cpp:104-117`), and constrained rows are zeroed.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `feat(spec): relaxed drafting under sampling (opt-in, changes the output distribution)`.

### Task 24: Draft-head context rows (X-D)

**Files:**
- Create:
  - `src/models/qwen3_5/program/context_rows.{h,cpp}` (host)
  - `include/ninfer/ops/linear_rows.h`
  - `src/ops/linear_rows/linear_rows.cu`
- Modify:
  - `src/models/qwen3_5/execution/text.cpp:563-590` (MTP `proposal_argmax` merges the proposal argmax with the context-row argmax)
  - `draft.cpp:359-372` (DFlash2 top-16 merge)
  - the program round setup (upload the row ids per round, graph-stable capacity)
  - `EngineOptions::draft_context_rows` (0 = off)
  - `--draft-context-rows N`
  - docs
- Test: `tests/models/qwen3_5/test_context_rows.cpp` (cpu); `tests/ops/test_linear_rows.cpp` (gpu, FP64 oracle with masked ids)

**Interfaces:**
- Produces:
  - `class ContextRowSet { ContextRowSet(std::span<const std::int32_t> subset_ids, std::int32_t vocab, int capacity); void note(std::span<const std::int32_t> ids); void add_lane(int lane, ...); void release_lane(int lane); std::span<const std::int32_t> slots() const; /* -1 = empty */ }`.
    - Ids in the subset are ignored and ids are deduplicated.
    - It is capped at capacity; new ids are dropped once full.
    - It is a refcounted union across lanes, with a stable slot order (first noted first).
  - `void ops::linear_rows(const Tensor& hidden /*[H,T]*/, const Weight& head, const Tensor& ids /*I32 [C]*/, Tensor& out /*F32 [C,T]*/, cudaStream_t)`: a row of id -1 gives -inf. It supports Q8_G32_FP16, Q6_G64 and Q4_G64 row-split heads.

- [ ] **Step 1:** write CPU tests:
  - `subset_ignored`
  - `dedup`
  - `capacity_cap`
  - `stable_order`
  - `lane_refcounts`: releasing one lane keeps rows another lane holds.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the set, the Op and the merges. The GPU oracle test covers every head format at C=512, T=1..4.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(spec): add the prompt's own tokens to the reduced draft head (opt-in)`.

## Phase 5: serving and process

### Task 25: Host-header check, Origin check, exposure warning, constant-time key compare (S3)

**Files:**
- Create: `src/serve/http_security.{h,cpp}` (no Engine dependency)
- Modify:
  - `src/serve/http_server.cpp:189-216, 388-418`
  - `src/serve/router_server.cpp:345-362` (same checks), `:692-712` (strip `Origin` when relaying to inner servers)
  - `apps/serve/main.cpp:249, 353` and `operational_log` (warning)
  - `serve_options.{h,cpp}` (`--allowed-host NAME`, repeatable)
  - `docs/serving.md`
- Test: `tests/test_http_security.cpp` (cpu, host-only); `tests/test_http_error_handler.cpp:52-59`; `tests/test_serve_options.cpp`

**Interfaces:**
- Produces:
  - `bool constant_time_equal(std::string_view, std::string_view)`
  - `struct HostPolicy { std::vector<std::string> names; }`
  - `HostPolicy make_host_policy(std::string_view bind_host, int port, std::span<const std::string> allowed, std::string_view machine_name)`: `localhost`, `127.0.0.1`, `[::1]`, the bind host when it is not a wildcard, the machine name when bound to a wildcard, and each `--allowed-host`, with and without `:port`.
  - `bool host_allowed(const HostPolicy&, std::string_view host_header)`: case-insensitive; a missing header is allowed (HTTP/1.0 tools).
  - `bool origin_allowed(const HostPolicy&, std::string_view origin, bool has_key, bool cors)`: true when there is a key or `cors` is set, or the origin is empty or its host is in the policy.
  - `bool exposed_without_key(std::string_view bind_host, bool has_key)`
- Decisions (mine, stated in the commit):
  - The Host check always applies.
  - The Origin check applies only without a key and without `--cors`.
  - A rejected request gets 403 with the OpenAI error shape.

- [ ] **Step 1:** write host tests:
  - `ct_equal_cases`: equal, different, different lengths, empty.
  - `host_loopback_ok`
  - `host_rebind_rejected`: `evil.example:8080` is rejected.
  - `host_lan_ip_ok_when_bound`: bound to 192.168.1.5, `Host: 192.168.1.5:8080` passes.
  - `host_wildcard_machine_name`
  - `host_allowed_flag`
  - `origin_cases`: key set, cors set, and a foreign origin without a key.
  - `exposure_warning_cases`: `0.0.0.0` without a key warns; `127.0.0.1` does not.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement and wire both servers; the three `==` compares become `constant_time_equal`.
- [ ] **Step 4:** run. Expected: PASS (host-only tree and full tree).
- [ ] **Step 5:** commit `fix(serve): reject DNS-rebinding hosts and compare keys in constant time`.

### Task 26: `/slots` endpoint (S4)

**Files:**
- Modify:
  - `include/ninfer/types.h:1449` (`RuntimeStats::lanes`, `RuntimeStats::waiting`)
  - `src/runtime/engine/engine_metrics.inl:157-221`
  - `src/serve/http_server.cpp:483-507`
  - `router_server.cpp:402`
  - `docs/serving.md:216, 287`
- Create: `src/serve/slots_report.{h,cpp}`
- Test: `tests/runtime/test_engine_scheduling.cpp` (cpu, fake engine); `tests/test_slots_report.cpp` (cpu)

**Interfaces:**
- Produces:
  - `struct RuntimeLaneStats { enum class State : std::uint8_t { Idle, Prefill, Decoding } state; std::uint64_t request_id; std::int64_t prompt_tokens, computed_prompt_tokens, reused_prompt_tokens, generated_tokens, context_tokens; double decode_tokens_per_second; std::uint64_t drafted_tokens, accepted_tokens; }`
  - `struct RuntimeWaitingStats { std::uint64_t request_id; std::int64_t prompt_tokens; double waited_seconds; }`
  - `std::array<RuntimeLaneStats, kMaximumConcurrency> lanes; std::uint8_t lane_count; std::vector<RuntimeWaitingStats> waiting;`
  - `std::string render_slots_json(const RuntimeStats&)`: `{"slots":[{"id","state","prompt_tokens","prompt_done","reused","generated","n_ctx_used","tokens_per_second","drafted","accepted"}], "queue":[{"id","prompt_tokens","waited_s"}]}`

- [ ] **Step 1:** write tests:
  - `lanes_report_prefill_then_decode`: the Rig with two requests; states and progress after each step.
  - `idle_lanes`
  - `render_slots_shape`: exact JSON for a fixed `RuntimeStats`.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement (route `GET /slots`, auth as other GET routes).
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `feat(serve): /slots reports each lane and the waiting queue`.

### Task 27: Local dashboard page (S5)

**Files:**
- Create:
  - `src/serve/dashboard/dashboard.html` (one file, inline JS and CSS, no external fetches)
  - `cmake/EmbedFile.cmake` (`file(READ ... HEX)` into a generated `.cpp` with a `char[]`)
  - `src/serve/dashboard_page.{h,cpp}`
- Modify: `src/serve/CMakeLists.txt`, `http_server.cpp:388-418` (exempt `GET /dashboard` from auth; the page's JS sends a key the user types), `:483` (route), `router_server.cpp:402`, `docs/serving.md`
- Test: `tests/test_dashboard_page.cpp` (cpu)

**Interfaces:**
- Produces: `std::string_view dashboard_page()`, served as `text/html; charset=utf-8`.
- The page polls `/metrics` (Prometheus text, parsed in JS), `/v1/load` and `/slots` every 1 s.
- It shows tok/s, prefill and decode charts, draft acceptance, context use, lanes, GPU power, VRAM and temperature.

- [ ] **Step 1:** write `page_embedded`: non-empty, starts with `<!doctype html>`, and contains `/slots` and `/metrics`.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `feat(serve): a local dashboard page`.

### Task 28: `--config` file with setting sources (S6)

**Files:**
- Create: `src/serve/serve_config_file.{h,cpp}`
- Modify:
  - `src/serve/serve_options.{h,cpp}:393-441, 862` (expand `--config` before parsing, then record sources)
  - `apps/serve/main.cpp` (startup line)
  - `src/serve/request_log.cpp:873`
  - `docs/serving.md`
- Test: `tests/test_serve_config_file.cpp` (cpu, host-only if the parser links only `ninfer_core`); `tests/test_serve_options.cpp`

**Interfaces:**
- Produces:
  - `struct ConfigWords { std::vector<std::string> words; std::vector<std::string> keys; }`
  - `ConfigWords read_serve_config(const std::filesystem::path&)`:
    - a JSON object of flag names without dashes;
    - strings and numbers become `--key value`, `true` becomes `--key`, `false` emits nothing, arrays repeat the flag;
    - nested objects are refused, as are `api-key` and `config`.
  - `ServeOptions::setting_sources` (`std::map<std::string, std::string>`: flag → `cli` | `<file>` | `default`).
- Behaviour:
  - Precedence is CLI > file > defaults.
  - An unknown key fails with `<file>: unknown key "<key>"`.
  - A key set to `true` in the file that the CLI tries to negate (`--no-<key>` exists) is resolved by the CLI. When no negation flag exists, it fails with `<file>: "<key>" is set and has no command-line override`.
  - In router mode the file is expanded before `model_arguments`.

- [ ] **Step 1:** write tests:
  - `config_strings_numbers_bools_arrays`
  - `config_refuses_api_key`
  - `config_unknown_key_names_file`
  - `cli_overrides_file`
  - `bool_without_negation_errors`
  - `sources_recorded`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `feat(serve): --config reads options from a JSON file`.

### Task 29: Launcher profiles as config files (S6, launchers)

**Files:**
- Create: `scripts/profiles/<model>-<profile>[-<spec>].json`, one per static flag set of `scripts/run.sh:145-263` and `scripts/run.bat:233-330`
- Modify:
  - `scripts/run.sh`, `scripts/run.bat` (pass `--config <file>`; dynamic choices such as the n-gram default by lane count, `--fit-context` and the host-memory retry stay in the scripts)
  - `scripts/check-linux-scripts.sh:99-130` (assert the config file and the remaining flags)
  - `scripts/package-release.sh:59`, `scripts/package-release.ps1:94` (ship `scripts/profiles/`)
  - `docs/maintainer/launcher-profiles.md`
- Test: `bash scripts/check-linux-scripts.sh` (Linux); `tests/test_serve_config_file.cpp` case `shipped_profiles_parse` (every file under `scripts/profiles/` parses and names only known flags)

- [ ] **Step 1:** add `shipped_profiles_parse`. Run. Expected: FAIL (no files).
- [ ] **Step 2:** write the profile files from the scripts' current flags (no value changes), and change the scripts.
- [ ] **Step 3:** run `bash scripts/check-linux-scripts.sh` and the test. Expected: PASS. `run.bat` is untested here; the commit says so.
- [ ] **Step 4:** commit `refactor(scripts): launcher profiles as --config files`.

### Task 30: Cache-affinity overtake in admission (S7)

**Files:**
- Modify:
  - `src/runtime/engine/scheduler.h:161-220`
  - `src/runtime/engine/request_record.h:204` (`overtakes`)
  - `src/runtime/engine/engine_core.h:2396-2620` (affinity predicate: a source with `reused_tokens > 0`, `take_over || session_hint`, and no host-to-device transfers; evaluated only when no admission decision is pending)
  - `EngineOptions::admission_affinity`, `--admission-affinity`
  - `docs/serving.md:2232-2234`, `docs/maintainer/engine-architecture.md:84, 248, 261`
- Test: `tests/runtime/test_scheduler.cpp` (cpu, host-only)

**Interfaces:**
- Produces:
  - `template <class Pred> std::optional<std::size_t> Scheduler::affinity_candidate(const Fresh&, Pred affine) const`: the oldest affine waiter behind the head whose every older waiter has `overtakes < kMaxAffinityOvertakes` (4).
  - `void Scheduler::overtook(std::size_t index)`: charges every older waiter `overtakes += 1`.

- [ ] **Step 1:** write host tests:
  - `affine_jumps`
  - `non_affine_waits`
  - `older_passed_at_most_four`
  - `charged_only_on_admit`
  - `bypass_allowance_unchanged`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `feat(engine): let a waiter that resumes a held prefix overtake (opt-in)`.

### Task 31: GPU power limit in bench rows, a frozen preset and a row checker (S2)

**Files:**
- Modify:
  - `src/core/nvml.{h,cpp}:42-60, 121` (bind `nvmlDeviceGetEnforcedPowerLimit`; `GpuSample::power_limit_watts`)
  - `src/serve/metrics.cpp:233-256` (`ninfer_gpu_power_limit_watts`)
  - `bench/inference/ninfer_bench.cpp:240-278` (sample before and after; warn when it changed)
  - `bench/inference/ninfer_bench_support.{h,cpp}` (`BenchEnvironment::power_limit_watts_{start,end}`, `--preset community-1`, one-line JSON row, `kSchemaVersion = 20`)
  - `tools/bench/run_ninfer_bench_matrix.py:43`
  - `bench/README.md`
- Create:
  - `tools/bench/check_bench_rows.py`
  - `bench/community/rows.json`
  - `tests/cmake/BenchSupportTests.cmake` entry
- Test:
  - `tests/test_gpu_telemetry.cpp` (cpu, host-only: fake symbol returns the limit)
  - `tests/test_ninfer_bench_support.cpp` (cpu: preset refuses other measurement flags; the row has the fields)
  - `check_bench_rows.py --self-test` (cpu, host-only)

**Interfaces:**
- Produces: preset `community-1`:
  - The 27B fixtures already in `bench/fixtures/` at fixed settings: one cold run each of a 16K prefill, a 64K prefill and decode over the chat prompts.
  - Row fields: `submitter, date, commit, preset, gpus[{name, vram_gib, power_limit_w, pcie}], ram_gib, cpu, os, model, prefill_16k_tps, prefill_64k_tps, decode_tps`.

- [ ] **Step 1:** write the tests above. Run. Expected: FAIL.
- [ ] **Step 2:** implement.
- [ ] **Step 3:** run. Expected: PASS.
- [ ] **Step 4:** commit `feat(bench): stamp the GPU power limit and add a frozen community preset`.

### Task 32: Randomised soak harness (S1)

**Files:**
- Create: `tools/bench/serve_soak.py`, `tools/bench/test_serve_soak.py`, a registration in `tests/cmake/BenchSupportTests.cmake` (both lists), a section in `tools/bench/README.md`

**Interfaces:**
- Produces:
  - `serve_soak.py --url URL --minutes M [--api-key K] [--seed S] [--images DIR] --out report.json`.
  - It mixes OpenAI chat streaming, Anthropic messages streaming, cancels mid-stream, near-limit prompts and over-long prompts (expect a 400 `context_length_exceeded`), oversized bodies (expect 413) and images.
  - It samples `/v1/load`, `/metrics`, `/slots` and the server's RSS (by `--pid`) every 2 s.
  - It flags a stream with no event for 60 s as stalled.
  - The exit code is non-zero on any unexpected status, stall, or an RSS growth above `--rss-growth-mib` (default 512) over the last half.
- Built on `tools/streaming_http/client.py` and `sse.py`.

- [ ] **Step 1:** write `test_serve_soak.py`: a fake `ThreadingHTTPServer` (pattern `tools/bench/test_prefill_stall.py`) with a stalling route, a 400 route and growing RSS. Assert each is flagged.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run `ctest -R serve_soak` and `py_compile`. Expected: PASS.
- [ ] **Step 5:** commit `test(serve): a randomised soak harness`.

### Task 33: Stderr ratchet and greedy-identity tool (S8)

**Files:**
- Create:
  - `tools/check_stderr_logging.py`
  - `tools/stderr_logging.allow` (current counts per file, from `git grep -c` on `fprintf(stderr` and `std::cerr` under `src/`)
  - `tools/bench/greedy_identity.py`, `tools/bench/test_greedy_identity.py`
  - registrations (both lists)
- Modify: `docs/maintainer/logging.md`

**Interfaces:**
- Produces:
  - `check_stderr_logging.py [--update]` fails when a file's count rises or a new file appears. `--update` lowers the allowlist.
  - `greedy_identity.py --a EXE --b EXE --model M [--prompts bench/fixtures/chat_prompts.jsonl] [--tokens 256]`:
    - runs `ninfer --greedy --print-token-ids` for both builds;
    - prints `GREEDY IDENTICAL` or the first differing prompt and token index;
    - reuses `run_interleaved_ab.py`'s build-directory preflight.

- [ ] **Step 1:** write the self-tests: a temporary tree where the count goes up fails and down passes; fake binaries print fixed id lines, giving identical and differing cases.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS.
- [ ] **Step 5:** commit `chore(tools): ratchet direct stderr writes and compare greedy output between builds`.

## Phase 6: big items

### Task 34: IQ2_XS and IQ3_XXS GGML block formats (F8, part 1)

**Files:**
- Modify:
  - `src/core/weight.h:13, 47-69` (QTypes `GGML_IQ2_XS` = ggml type 17, 74 B; `GGML_IQ3_XXS` = type 18, 98 B)
  - `src/artifact/formats.cpp:12-34`
  - `third_party/ggml/ggml-common-tables.h` and `tools/artifact/codecs/ggml_tables.py` (`iq2xs_grid` 512×u64, `iq3xxs_grid` 256×u32, from `tools/artifact/gen_ggml_tables.py` run against a llama.cpp checkout at the pinned tag, fetched into a scratch directory outside the repo)
  - `tools/artifact/formats.py:95-147, 251-266`
  - `tools/artifact/codecs/ggml_blocks.py:196`
  - `tools/artifact/gen_ggml_fixtures.py:28-38` and the new `tests/fixtures/ggml/{iq2_xs,iq3_xxs}.{blocks,f32}`
  - `tests/ops/ggml_blocks_decode.h:223-250`
  - `src/ops/linear/ggml/ggml_blocks.cuh` (dense decode for both)
  - `docs/maintainer/tensor-formats.md`
- Test: `tests/ops/test_ggml_blocks_decode.cpp:40-46` (cpu, host-only); `tests/artifact/test_ggml_codecs.py`; `tests/ops/linear/test_ggml.cpp` (gpu)

- [ ] **Step 1:** add both formats to the test lists and the codec sizes. Run. Expected: FAIL.
- [ ] **Step 2:** generate tables and fixtures with ggml. Implement the host and Python decoders.
- [ ] **Step 3:** run `ninfer_ggml_blocks_decode_test` and the codec tests. Expected: PASS (exact against the ggml goldens).
- [ ] **Step 4:** commit `feat(formats): IQ2_XS and IQ3_XXS GGML blocks`.

### Task 35: Expert records for the IQ3 packs, CPU engine and GPU routes (F8, part 2)

**Files:** follow the recon checklist, steps 1-14 (record formats in `weight.h`/`formats.cpp`/`formats.py`, CPU decoders and AVX2 `Fmt<F>`, `ExpertFormat<F>`, `down_kernel<D,NC>` and the `launch_pass` pair switch, wide `run_pass`/`kOneMultiplier`/`wide_down_kernel<D>`, header "Supported domain", `canonical_ggml.h:18` bound, `wide_expert.h:18-21`, `tensor-formats.md` §3.6, `storage-layouts.md` §7, tests and bench lists).

**Interfaces:**
- Produces:
  - Record formats `ggml_rec_<gateup>_<down>` for every (gate/up, down) pair found in the GSQ-RCO IQ3_XXS and IQ3_S GGUFs. Read the pairs from the GGUF headers; download only the header ranges with `hf download --include` or an HTTP range request into a scratch directory.
  - Gate/up ∈ {IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS}; down ∈ {Q2_0, IQ4_NL}.
  - Canonical decompositions per the recon table (IQ3_XXS c=d/4, m=2ls+1; IQ3_S c=d, m=1+2·nibble; IQ2_XS c=d/8, per-half m; IQ4_XS c=d, m=ls-32; IQ4_NL c=d, m=1).

- [ ] **Step 1:** extend `tests/ops/offloaded_moe_fixtures.h:35, 92-105` and `test_offloaded_moe_cpu.cpp:75, 164` with the new formats, and add `extreme_magnitude_records`: IQ4_XS m=-32, |w|=127, q=±127; IQ3_XXS grid 62. These pin |P| < 2^24 exactness and that AVX2 does not saturate. Run. Expected: FAIL.
- [ ] **Step 2:** implement the CPU engine. Run the CPU tests. Expected: PASS (scalar against FP64, AVX2 = scalar, team = scalar).
- [ ] **Step 3:** implement the device decoders and the narrow and wide routes. The GPU cases in `test_offloaded_moe_layer.cpp` and `_wide.cpp` (bitwise against the CPU engine) compile.
- [ ] **Step 4:** add the converter: `tools/convert/qwen4_exp.py` accepts the IQ3 GGUFs, plus a fixture in `tests/convert/test_qwen4_exp.py:273-276` with mixed downs per layer. Run the Python tests. Expected: PASS.
- [ ] **Step 5:** commit `feat(qwen4exp): expert records for the GSQ-RCO IQ3_XXS and IQ3_S packs`.

### Task 36: Partial RAM residency (F8, part 3)

**Files:**
- Modify:
  - `include/ninfer/ops/offloaded_sparse_moe.h:251-273` (`MoeExpertSource::host_slots`: a per-expert host address table replacing `base + e*stride` when set)
  - `moe_layer.cu` stage and zero-copy reads
  - `execution/expert_stream.*` (`Bank` takes the table)
  - `program/expert_cache/expert_cache.h:142, 185-205` (`Command::WriteBack`)
  - `expert_residency.*`
  - `memory_plan.{h,cpp}` (mode `Partial`: RAM holds every expert not in VRAM, plus swap spares)
  - `load.cpp:140-146` (banks read from the file into RAM slots instead of pinned wholesale)
  - `qwen4_exp_instance.cpp:254-288`
  - option plumbing
  - `docs/serving.md` (RAM needs)
- Create: `src/models/qwen4_exp/program/ram_slots.{h,cpp}`
- Test: `tests/models/qwen4_exp/test_ram_slots.cpp` (cpu); `test_memory_plan.cpp` (cpu); `test_expert_cache.cpp` (cpu); `test_offloaded_moe_layer.cpp` (gpu: `host_slots` reads)

**Interfaces:**
- Produces:
  - `class RamSlots { RamSlots(std::size_t slots, std::size_t stride); std::optional<std::uint32_t> acquire(ExpertKey); void release(ExpertKey); std::uint64_t address(ExpertKey) const; }`
  - `struct SwapPlan { std::vector<ExpertKey> to_ram, to_vram; }`
  - `SwapPlan plan_swap(const Residency&, std::span<const ExpertKey> evict, std::span<const ExpertKey> admit)`
  - Invariant: every expert is in exactly one tier or in flight, and a frame is reused only after its write-back event.
  - `FlashNextOptions::partial_ram_residency`, `--partial-ram-residency`. Off keeps the whole-pin mode and its refusal unchanged.
  - The memory plan in Partial mode: `host = (E_total - vram_frames + spares) * stride`, where `spares = 2 * max experts per round`. The WDDM registration cap is `RAM/2 - 8 GiB`, and the plan refuses with the largest workable `--vram-headroom-mib` or cache size.

- [ ] **Step 1:** write CPU tests:
  - `slots_acquire_release`
  - `slots_exhaustion`
  - `swap_plan_invariant`: random evict/admit sequences, with every key in exactly one place.
  - `writeback_before_reuse`: the controller refuses to reuse a frame with a pending write-back.
  - `partial_plan_bytes`
  - `partial_plan_wddm_cap_refusal`
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `feat(qwen4exp): keep only non-VRAM experts in RAM (opt-in)`.

### Task 37: Int8 tensor-core GEMV for GGML dense linears at T ≤ 8 (F6)

**Files:**
- Create:
  - `src/ops/linear/ggml/ggml_mma_decode.cuh`
  - `src/ops/linear/ggml/ggml_mma_emulation.h` (host)
- Modify:
  - `src/ops/linear/ggml/ggml_dispatch.cpp:57-75` (private route choice inside `run<F>`, gated by `LinearPolicy::ggml_decode_mma`)
  - `include/ninfer/ops/linear.h:126-162` (contract)
  - Flash-Next option plumbing (`--ggml-decode-mma`, passed in the Forward's `LinearPolicy`)
- Test: `tests/ops/test_ggml_mma_emulation.cpp` (cpu, host-only); `tests/ops/linear/test_ggml.cpp` (gpu)

**Interfaces:**
- Produces:
  - `void ggml_mma_emulate(QType, const std::byte* row, const std::int8_t* x_codes, const float* x_scales, int K, float* out)`:
    - an exact int32 sum per 32-value block (2×16 for Q6_K);
    - fp32 scale terms in the fixed order the kernel uses (`canonical_ggml.h`: `fma(c*d, float(P), partial)`, 32 lane partials, butterfly).
  - The kernel: 16-row × 8-column tiles with `mma.sync.m16n8k32.s8`; weights staged by 16-byte `cp.async` into an S=3 ring; IQ3_S decoded from shared memory. Formats: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0, Q8_0.

- [ ] **Step 1:** write CPU tests:
  - `emulation_within_fp64` for every registered Flash-Next shape and format: within `ReductionCriterion` (`test_ggml.cpp:57-62`).
  - `emulation_column_invariant`: T=1..8 bitwise per column.
  - `int32_range_safe`: |P| < 2^22 on extreme blocks.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement the emulation, then the kernel and route. The GPU tests check bitwise against the emulation and FP64 bounds at T=1..9, plus `linear_add` and `linear_pair` parity.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests compile.
- [ ] **Step 5:** commit `perf(linear): int8 tensor-core decode GEMV for GGML formats (opt-in)`.

### Task 38: Two-stage head for Qwen3.8-27B (X-B)

**Files:**
- Create:
  - `src/ops/two_stage_head/proxy_codec.h` (`NINFER_HD`)
  - `include/ninfer/ops/two_stage_head.h`
  - `src/ops/two_stage_head/two_stage_head.cu`
- Modify:
  - `src/models/qwen3_5/load/storage_trades.cpp:19-30` and `load/bindings.h:42` (build the proxy as a companion device object at load from the row-split Q8 head; bindings learn companions)
  - `src/models/qwen3_5/execution/text.cpp:707, 767-768` (verify and ordinary heads)
  - startup VRAM plan
  - `EngineOptions::lm_head_two_stage`, `--lm-head-two-stage`
  - `docs/maintainer/quality-trade-experiments.md`
- Test: `tests/ops/test_two_stage_head_codec.cpp` (cpu, host-only); `tests/ops/test_two_stage_head.cpp` (gpu)

**Interfaces:**
- Produces:
  - `void fit_group_2bit(const float* w /*32*/, std::uint8_t* levels /*32 values 0..3*/, __half* scale)`: Basalt's 6-candidate scale search, then the scale × `sqrt(|w|²/<w,ŵ>)`.
  - `void pack_proxy_rows(...)`
  - `std::int32_t proxy_dot_exact(...)`, emulating the int8 mma.
  - `float threshold_from_block_maxima(std::span<const float> block_max, int k /*64*/, float bins_per_unit /*8*/, float margin /*4.0*/)`
  - `void ops::two_stage_head(const Tensor& hidden, const Weight& head, const ProxyHead& proxy, std::int32_t valid_rows, Tensor& logits /*BF16 [V,T]*/, Workspace, cudaStream_t)`: all rows get the approximate value; rows ≥ the column threshold are rescored by the exact route, within the Q8 A16 FP64 criterion.
- Exclusions: rows with penalties or a grammar mask take the exact head (decided per round from the row's config).

- [ ] **Step 1:** write CPU tests:
  - `unshrink_scale_ge_ls`
  - `inner_product_identity`: `<w, ŵ·s'> = |w|²` within fp16 rounding.
  - `pack_round_trip`
  - `proxy_dot_equals_fp64_of_dequant`
  - `threshold_le_kth_best` on random and adversarial maxima.
  - `recall_report`: top-64 recall on a synthetic Gaussian head is printed, not asserted beyond ≥ 99%.
- [ ] **Step 2:** run. Expected: FAIL.
- [ ] **Step 3:** implement.
- [ ] **Step 4:** run. Expected: PASS. The GPU tests (argmax and top-16 agreement with the dense head at T=1..16 × B=1..8; rescored rows against FP64) compile.
- [ ] **Step 5:** commit `perf(qwen3_5): two-stage output head with exact top rows (opt-in)`.

## Phase 7: GPU verification script and the draft PR

### Task 39: GPU verification scripts, docs pass, draft PR

**Files:**
- Create: `tools/bench/basalt_gpu_verify.ps1` (Windows, under `tools/gpu-lock.ps1`), `tools/bench/basalt_gpu_verify.sh` (Linux box), `tools/bench/basalt_options.json` (one entry per option: flag, model, kind `identity|quality|speed`, tests regex, A/B workload)
- Modify: `docs/maintainer/flash-next-plan.md` (a "Basalt items" section listing every option and its verification status: "compiled, CPU-tested, GPU-unverified")

**Interfaces:**
- Consumes: every option and test name above.
- Produces: the scripts run, in order:
  1. `pwsh tools/run-tests.ps1 -GpuOnly -Regex '<touched areas>'`, which takes the lock per test.
  2. For `identity` options: the forward harness or engine test with the option on against off.
  3. For `quality` options (`--hc-int8`, `--relaxed-drafting`, `--lm-head-two-stage`, `--ggml-decode-mma`): the KLD gate (Tasks 2/3), 20 chunks then 100 chunks, against a base built once with llama.cpp.
  4. For `speed` options: `run_interleaved_ab.py` ABBA, n=12, inside one lock hold per option.
  5. A combined row with every winning option on.
  - Results go to `profiles/bench/basalt-<date>/`.

- [ ] **Step 1:** write the scripts, plus a `--dry-run` that prints the plan. Run `bash tools/bench/basalt_gpu_verify.sh --dry-run`. Expected: every option in `basalt_options.json` is listed, and every regex matches `ctest -N` names.
- [ ] **Step 2:** run the full CPU suite in both trees: `ctest -L cpu -j8` (full tree) and `ctest -j8` (host-only tree). Expected: all PASS, apart from the known `ninfer_qwen4_exp_ngram_volume_test` timing flake (memory note), which is reported if seen.
- [ ] **Step 3:** run `bash scripts/check-linux-scripts.sh`, `python3.11 -m py_compile` on every changed script, and `git diff --check origin/master`.
- [ ] **Step 4:** commit `chore(bench): scripts that verify the Basalt options on a GPU`.
- [ ] **Step 5:** push `feat/basalt-learnings`. This uses SSH only if `.github/workflows/` changed (it should not). Open a **draft** PR following `.github/pull_request_template.md`, linking #383. Say which tests ran (CPU only) and that every GPU test and measurement is pending. The body ends with the session's PR attribution line.
