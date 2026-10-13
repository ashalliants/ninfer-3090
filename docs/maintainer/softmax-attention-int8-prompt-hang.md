# softmax_attention int8-prompt test: hang or slow? (investigation note)

Status: findings complete 2026-10-13, no fix applied. Delete this file once the test is fixed or the question is closed.

## Question

A full `ctest` run sat in `ninfer_softmax_attention_int8_prompt_test`
(`ninfer_tests ninfer_softmax_attention_test --int8-prompt-only`, `tests/ops/tests.cmake`) for 49
minutes with about 2,900 CPU-seconds on one core and the GPU at 0% before it was killed. AGENTS.md
lists `ninfer_softmax_attention_test` at about 184 s. Hang, or just slow?

## Short answer so far

Slow, not hung. The test is a host-side FP64 oracle sweep and takes tens of minutes on one core.
No evidence of a kernel hang. The AGENTS.md 184 s figure (2026-10-06) predates this test, which was
added with the FA2-style INT8 prompt kernel (d8a85a66, 2026-10-10) and is not part of the no-argument
`ninfer_softmax_attention_test` run.

## Evidence

Run 1 (unmodified test, RTX 3090, GPU lock held, 15 min cap, origin/master at 02e6d98b plus a
one-line local fix for an unrelated qwen4_exp build break that #290 later fixed):

- Process used 1.00 core the whole time (cpu seconds == wall seconds at every 20 s sample),
  working set 2.8-3.2 GB, GPU utilisation 0-48% in short bursts, about 37 W idle power with a
  single 112 W burst. It was still running at 900 s and was killed by PID at the cap. That matches
  the reported "one core, GPU 0%" signature.
- Output is fully buffered when redirected, so the unmodified test shows nothing until it ends.

Run 2 (same binary plus a local timing print per case, not committed): per-case wall time, all in
the test's host code (`run_a1_case`/`run_a3_case` in `tests/ops/softmax_attention/causal_cache.cpp`):

| case (tokens, history) | oracle rows | seconds |
|---|---|---|
| 65 / 263 | all | 1-4 |
| 200 / 4,100 | 13 | 3.2 |
| 3,584 / 8,192 | 6 | 4.0 |
| 17 / 33,000 | all (17) | 31.5 |

Cost model: the oracle (`naive_dense_softmax_attention`, `oracle.h`) runs FP64 on one thread,
`rows x q_heads x visible_keys x head_dim x 2` element reads through `cache_value`. At a 33K history
that is about 30 s per case. `run_int8_prompt_cases` runs about ten such cases per (storage,
geometry) pair, for 3 storages x 2 geometries, plus the 4K-40K cases and `make_cache` over up to 40K
positions each time.

## Reproduce

```powershell
# VS 2022 BuildTools env (AGENTS.md), own worktree and build-ninja, build target ninfer_tests
# hold %TEMP%\ninfer-gpu.lock (CLAUDE.md), then:
build-ninja\tests\ninfer_tests.exe ninfer_softmax_attention_test --int8-prompt-only
```

## Recommendation (draft)

Not a kernel bug. Options: shrink the 33K-history cases (oracle rows sampled for every case, fewer
repeats at 33K), run the oracle with `worker_limit` > 1 (the oracle already takes `worker_limit`),
and set an explicit `TIMEOUT` plus a note in AGENTS.md next to the other slow tests. See the second run below.

## Second run (per-case timing, 900 s cap, same hold of the GPU lock)

Local-only instrumentation: a destructor timer in `run_a1_case`/`run_a3_case` printing to stderr.
In 900 s the test completed 150 cases whose own wall times sum to 881 s (98% of the run is inside
cases, so there is no hidden stall between them):

- 29 cases over 10 s account for 695 s. Every one is a 33,000-token-history launch (17-1,100 query
  columns, 17-37 s each; the 24-head geometry is the slow one) or an 8K-history verify-width case
  (6-11 s).
- Only `int8-g64` was reached: both geometries of its prompt cases finished, and it was inside its
  "verify-width route boundaries" block (about 130 small cases, 0.05-11 s each) when the cap hit.
  `rk8v4` and `rk4v4` had not started. Each storage repeats the same work, so the whole test is
  roughly 3x this: 45-60 minutes on one core. Not measured to completion (cap was 15 min as asked).
- No case showed a time out of line with its shape; the time scales with
  `rows x q_heads x visible_keys`. GPU memory moved between cases (1.3 to 2.5 GB), GPU work is in
  short bursts: the kernel launches return normally.

Conclusion: slow by construction, not hung. The other agent's 49 min / 2,900 CPU-s kill was
probably within a few minutes of finishing or at most 10-20 minutes short of it.

## CI

`.github/workflows` contains only `host-checks.yml` and `codepulse.yml`; no CUDA test runs in CI, so
this test only ever runs on a developer's GPU host with `ctest`.

## Recommendation

1. Make the oracle cheaper before touching the kernel: the 33K cases dominate and are repeated
   (5 widths x 2 bases, 6 plan-boundary cases, an envelope case, a cached-entry case, a graph case)
   for 3 storages x 2 geometries. Keep one or two 33K cases per storage and move the plan-boundary
   scan to a 4K-8K history (the split boundary is a function of width and visible keys; check that
   it still splits there), or run the oracle with `worker_limit` = the core count (the oracle
   already accepts it) so the same cases take a few minutes.
2. Give the test an explicit `TIMEOUT` and add it to the slow-test list in AGENTS.md (it is not in
   the "three op tests" list there, which predates it).
3. Print progress per storage with `std::endl` so a redirected run is not silent for an hour.
