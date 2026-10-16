# Task D: a real disk tier for the context cache (engine-wide, benefits every model)

Read `brief-common.md` first (its Flash-Next specifics do not apply to you: you work on the shared
engine and the **Qwen3.8-27B** default profiles; keep its process rules: own worktree, GPU lock, PIDs
only, no force pushes, PRs against `ashalliants/ninfer-3090`).

## Why
The user hosts Qwen3.8-27B for many people. Today the pinned Host tier is the only place a cache
entry lives "warm"; the local context store (`--context-store`, `src/runtime/` ContextStore, chunks,
dedup, S3 copy) is an idle backup read back only on a miss or at startup, and Host eviction does not
know a disk copy exists. A study (`cache-study.md` in this folder; read sections 0, 3, 4, 5.4-5.5 and
6 first) ranked what to do. Another agent (Task M) is producing baselines for hosted workloads in
`docs/maintainer/cache-provider-baseline.md`: wait for/consult its numbers if they exist when you
reach the measurement steps, and reuse its generator and its NVMe speed script.

## Scope: three ideas from the study, plus one measurement
0. **Measure the drives** (if Task M's script exists, use it): sequential read/write of the C: NVMe
   (unbuffered) and the store's own end-to-end hydrate throughput cold (the study's 0.57 GB/s was
   from a warm file cache). Record in `docs/performance.md`.
1. **Disk read cost in the cost model** (study idea 2): replace the fixed
   `kMinimumHydrationGain = 4096` (`src/runtime/engine/engine_core.h` around line 2887) with a term
   in `ContextMachineCostModel` for disk read time of the actual image bytes (learn the rate online
   from observed hydrations with a sane default, or a flag; your call, justify it). Hydration then
   happens when restore time < recompute time for *that* model, not at a magic 4,096 tokens.
2. **Disk-aware Host eviction + write-on-evict** (idea 1): a victim whose current fingerprint is
   already in the store is priced as a disk-restore loss, not a full recompute; and a victim that is
   not in the store may be written to it as part of eviction (the commit `9e893281` dropped "writing
   a session at eviction time": read its message and the code first to learn **why**; keep that
   lesson). Constraints: the Engine worker must not stall (see `docs/serving.md` ~lines 455-470
   about the export step blocking arrivals); eviction must either wait for the write or keep the Host
   copy pinned until written; failures to write must degrade to today's behaviour.
3. **Persist shared prefixes and restore the highest-value set** (idea 3): SharedPrefix entries
   (system/tool prompts) survive restarts and agent subagents start warm; the store restores by
   value (density), not only recency, when it cannot restore everything within the Host budget.

Do the ideas in that order, each as its own PR if it merges alone (PR_POLICY: at most 3 open PRs
for this feature, one level of stacking, review fixes go on the PR's branch). Everything is opt-in
through the existing `--context-store` flags unless measurement says the default profile gains;
without a store configured, behaviour and numbers must be unchanged.

## Gates (measure each, report all numbers; a miss is disclosed, not hidden)
- **Hosted workload** (Task M's generator, or `tools/bench/run_agent_rotation.py` + `bench/agent-replay`):
  27B launcher defaults at C4 and C8, Host tier small (2 GiB) and 8 GiB, store on a local folder on C:.
  ABBA against the same build without your change in the same sitting, under the GPU lock.
  Report hit rate, tokens re-prefilled, TTFT p50/p95/p99, decode tok/s, bytes written per hour.
- **No-store regression**: decode and prefill tok/s and TTFT unchanged within noise with the store
  off (the default).
- **Exactness**: a request restored from the store gives the same greedy ids as one served from
  Host (existing store tests + a real-model check).
- **Flash-Next**: the store plumbing is model-agnostic; another agent is implementing Flash-Next's
  `export_checkpoints`/`import_checkpoints` on it. Do not add anything Qwen3.5-specific to the
  store side; keep the Program-facing contract stable and mention any change in the PR.
- Run the existing store/engine test subsets (`ninfer_context_store*`, engine context tests; the
  full 159-test suite once at the end under the lock) and add tests for the new policy
  (CPU-only where possible).

## Disclosures to put in each PR
Disk write volume per hour under the workload (GB/day estimate) and the SSD-wear implication; any
case where the policy loses to today's (e.g. a short prompt that is cheaper to recompute); every
number that is an estimate rather than a measurement.
