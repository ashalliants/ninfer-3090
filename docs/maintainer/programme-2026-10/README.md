# Programme 2026-10: Flash-Next cache and lanes, and engine-wide cache work

Living hand-over document. Started 2026-10-13, refreshed 2026-10-14. If you are an agent picking this up cold, read this
file, then `briefs/brief-common.md` (process rules and hazards on this machine), then the brief for
the strand you are continuing. Everything here was produced by Claude Code sessions working for the
repo owner; every number below is from a PR description or a document in this repo unless marked.

## Why

- The owner hosts **Qwen3.8-27B as a provider for many people**: concurrency and cache hit rate
  matter more than single-stream peak tok/s. SSD wear is not a concern (hosted instances are the
  provider's drives): write eagerly. Gateway is a separate repo (`ninfer-gateway`).
- Qwen3.8-Flash-Next (a 512-expert MoE, IQ2_XS, Windows preview) had no prefix/KV cache and served
  one request at a time.
- A study of Infernix's hybrid prefix cache (`cache-study-infernix.md` here) produced nine ideas
  that could help any model.

## Decisions taken (and why)

1. **Flash-Next cache = implement the existing Engine contract in the qwen4_exp Program**, not a
   second resource-manager mode from Infernix's index. The Engine, ResourceManager and Scheduler are
   unchanged; no second cache architecture. (The study recommended the Infernix index; overruled
   for engine-risk reasons. Revisit only if the hooks route proves unworkable.)
2. **Checkpoint bytes sit behind a `CheckpointTier`** (`src/models/qwen4_exp/program/context_cache.h`):
   RAM is built (#285), persistence through `--context-store` and an NVMe tier are the next steps,
   so a 0 GiB RAM budget can become disk-only reuse. Today 0 GiB turns the cache off.
3. **Prefill splits at capture points always** (cache on or off), so resumes equal cold runs.
   Chat outputs of Flash-Next therefore differ from before #285; cost +6% on a short first turn,
   ~2% at 28K. Reversible; the preview is not released.
4. **Multi-lane Flash-Next**: C = 2 only. The spike (`spike/qwen4exp-lanes`, PR #284) predicts +10-16%
   aggregate and ~30% shorter mean time-to-first-token for queued requests; C > 2 is NO-GO (expert
   hit rate 87-90%). The first slice (batched forward, PR #286) measured 2.2-2.5 ms non-expert
   cost per extra row against the spike's 1.4 ms. PR #291 folds #286 in (superseding it) and the
   server-level test has run: aggregate decode **1.07x, below the 1.15x gate**; mean TTFT is 19%
   lower. Ships opt-in (default 1), see the strand table.
5. **No extraction of a tiered-cache component from the Qwen3.5 stack** (about 14k lines of mature
   code, no measured hit-rate gap to win back). Make the existing stack disk-aware with small
   changes instead.
6. **Self-set gates are not vetoes**: a real, regression-free win that misses our own threshold ships
   with disclosure. Regressions on the default 27B/35B profiles do not ship.

## Measured so far

| What | Result | Source |
|---|---|---|
| Flash-Next cache (#285), 28K prompt | cold 11.7 s; exact repeat 0.10 s; next turn 0.95 s; restore ~21 ms for 500 MiB; decode unchanged (~102 tok/s) | PR #285 |
| Flash-Next agent rotation | continuing turns 5.8 s -> 1.2 s | PR #285 |
| 27B, C4, 2 GiB Host tier, write-on-evict vs #287 | hit 0.32 -> 0.71-0.74; re-prefilled tokens 870K -> 336-370K; TTFT p50 6.9-7.3 -> 4.2-4.7 s; p95 32-35 -> 9-14 s; costs: ~16% engine-worker export stall, ~550 GB/h written | disk-tier branch (see PR) |
| 27B shared prefixes, warm start after a crash | hit 0.418 -> 0.478; mean TTFT 2.8 -> 2.1-2.4 s; neutral in steady state | same |
| Flash-Next batched forward, per extra row | 2.2-2.5 ms non-expert; predicted C = 2 aggregate 1.157x same-sitting MTP (1.10x vs spike's 92.6) | PR #286 |
| Lane cost on the 3090 | 116 MB state + 14.7 KB/token KV per lane; 395 expert frames per 32K lane | PR #284 |
| Two lanes, server burst of 8 conversations x 1,024 tokens (C1 vs C2, MTP) | tokens/makespan 71.4, 74.4 -> 77.5, 78.2 (**1.07x**, gate 1.15x missed); makespan 0.94x; mean TTFT 50.7, 48.7 -> 40.4, 39.9 s (0.81x); mean completion flat; a pair decodes 47-52 tok/s per stream | PR #291 |
| Two lanes, other gates | exactness passed (overlapped and staggered pairs, cache, cancel); 4K/32K/128K alternation passed; existing tests passed; lone-request speed (within 2%) inconclusive (drift in the sitting) | PR #291 |
| Flash-Next store restart, 24K prompt (n = 1) | TTFT 10.79 s cold -> 0.211 s (RAM tier) / 0.445 s (NVMe only); 7K: 3.27 -> 0.142 / 0.281 s | PR #296 |
| Flash-Next store and NVMe, server ABBA | no cold-prefill or decode regression vs master (cold 24K ~10.6 s; decode ~93-96 tok/s); NVMe-only repeat costs ~80 ms more than RAM at 24K | PR #296 |
| Hosted 27B, C8, 2 GiB Host, sustainable, store on vs off (one run each) | hit 0.732 -> 0.876; TTFT p50 4.07 -> 1.99 s; p95 16.6 -> 8.0 s; ~93 GB/h written | PR #294 |
| Hosted 27B, C8, 8 GiB Host, saturated, store on | 0 writes in 12 minutes (the idle-only writer never runs under saturation): no effect (hit 0.610 vs 0.603) | PR #294 |
| Hosted 27B, Host tier size, below saturation | 8 GiB or more: hit 0.886 vs ideal 0.940, TTFT p50 ~1.1 s; 2 GiB: hit 0.732, p50 4.1 s. Above ~2,100 tok/s prefill, queue wait dominates (p50 60-106 s) | PR #294 |
| Disk-aware eviction + write-on-evict at other sizes (agent replay, ABBA) | 2 GiB (C4, C8): hit +0.14 to +0.42, p95 TTFT halved or better; 8 GiB (C4, C8): neutral within noise but still writes 340-460 GB/h; shared-prefix filter cut shared writes 77 -> 3 per run | PR #295 |

## Strands, branches and state

Open PRs and branches move; check `gh pr list -R ashalliants/ninfer-3090`.

| Strand | PR / branch | State (2026-10-14) | Next step |
|---|---|---|---|
| Flash-Next context cache | merged: #285, #290 | done | none |
| Engine disk tier 1: learned disk read cost | merged: #287 | done | none |
| Hosted-27B baseline and ranking | #294 `bench/provider-baseline`, ready for review | done: attribution table and ranking written; the store at 2 GiB sustainable helps (hit 0.73 -> 0.88, p50 4.07 -> 1.99 s) but under C8 saturation the idle-only writer writes nothing | owner review; then follow the ranking below |
| Engine disk tier 2+3: disk-aware eviction, write-on-evict, persisted shared prefixes | #295 `feat/context-disk-tier-evict`, draft | all queued measurements done (see table). Wins at 2 GiB; neutral at 8 GiB but still writes 340-460 GB/h | **owner decision**: write-on-evict default at large Host tiers (on, off, or size-gated); then un-draft |
| Flash-Next `--context-store` + NVMe tier | #296 `feat/qwen4exp-context-store`, ready for review | all real-model gates pass; restart 24K TTFT 10.79 s -> 0.21 s (RAM) / 0.45 s (NVMe only), no cold regression. The NVMe tier stays opt-in (benefit beyond the RAM tier not measured) | review. Check `ninfer_openai_responses_test` (crashes 0xc0000409; unchecked on clean master, the branch does not touch that code) |
| Flash-Next two lanes | #291 `feat/qwen4exp-lanes`, draft; supersedes #286 | exactness and alternation gates pass; gate 2 (aggregate decode >= 1.15x) missed at 1.07x; gate 4 (lone request within 2%) inconclusive. Ships opt-in (`NINFER_CONCURRENCY=2`), launcher default 1 | settle gate 4 with more alternating runs; then review. Close #286 as superseded |
| Flash-Next lanes feasibility spike | #284 `spike/qwen4exp-lanes`, draft | unchanged (GO-WITH-CONDITIONS at C = 2, NO-GO above); keeps the route traces and replay tools | leave open as the reference for #291 |
| Op-test slowness | #297 closed; fixed by #298 (merged) | `ninfer_softmax_attention_int8_prompt_test` was slow, not hung: 45-60 min -> about 9 min on 12 threads | none |
| Test tooling | #299 merged | CPU tests run without the GPU lock; GPU tests take it one at a time | none |
| Host demotion failures (27B) | no PR yet; separate session | per that session's notes (instrumented builds, not in a PR): failures are allocator geometry, not shortage; about 93% of failed allocations are KV runs, but each failed state-image demotion loses a whole conversation; the engine stall is victim-selection CPU cost. No fix shipped. #294 independently measures 30-291 state and 4-172 KV reservation failures per run at 2-8 GiB | wait for that session's PR; it is the first step of queue item 4 below |

## Queue: ideas not started (from the study, ranked there; re-rank with the baseline)

Ranking from the baseline (#294), which supersedes the study's order: **idea 7 first step**
(stop the page-by-page demotion retry, count and log reservation failures), then **ideas 1 + 2**
(write at eviction time plus a disk cost term, the largest ceiling), then **idea 9** (history
points on disk). Ideas 3, 4, 5, 6 and 8 measured small; 4 only matters if clients fan out.

1. Write-through at release instead of demote/export on pressure: removes the ~16% engine-worker
   stall write-on-evict introduced (copy-engine discipline on WDDM matters).
2. Divergence-tap single-flight: concurrent subagents whose prompts diverge away from the planned
   markers prefill the shared part once.
3. Blocked-head Host-to-Device prefetch for the queue head (TTFT -3..-6% in Infernix's pressure
   replay; ceiling ~14% at 120K).
4. Slab Host pool (fixed-size slabs, no allocator geometry failures): the baseline now shows
   allocator failures whenever the Host tier is full (none at 16 GiB); first count and log them
   (idea 7 first step), then decide. The Flash-Next cache already uses slabs.
5. Free VRAM as Device block cache (helps the 35B-A3B; little on 27B at 3090).
6. Dense flexible taps on a cheap disk-backed tier (P1 was rejected earlier because it was Host-bound).
7. Persisting shared prefixes across restarts is in #295; partial restore by value
   (density) belongs with it. Its first step (demotion retry and failure counters) leads the ranking.
8. Gateway: session-affine routing multiplies every hit-rate gain when several engines serve the
   same tenants (see the baseline PR's gateway note).

Not recommended (rejected with evidence earlier): GDSF eviction, skip-echo capture, history points
at small Host tiers.

## Rules that bit people on this machine

- One RTX 3090, a lock file `%TEMP%\ninfer-gpu.lock`: open with `FileShare.None`, hold it inside one
  PowerShell invocation, poll every 0.5 s, never create or delete the file (see `CLAUDE.md`).
- Flash-Next pins ~33 GiB of experts in RAM: only one Flash-Next process machine-wide.
- A worktree needs its own `build-ninja`; build named targets only; `AGENTS.md` has the recipe.
- Full ctest run takes ~13 minutes: use `ctest -E` while iterating. Since #298 the int8 prompt
  oracle takes about 9 minutes, and since #299 CPU tests need no GPU lock. Known environment
  failures: `jsonschema` module missing (two oracle tests), `ninfer_artifact_writer_interop_test`;
  `ninfer_openai_responses_test` crashes (0xc0000409) on the #291 and #296 branches and is not yet
  checked on clean master.
- PR policy: at most 3 open PRs per feature, stack one level at most, review fixes go on the PR's
  branch. The owner merges.

## Scratch artefacts

Briefs for each strand are in `briefs/` (they reference scratchpad paths of the original session,
which may not exist; the repo is authoritative). The raw route traces and replay tools of the spike
are on `spike/qwen4exp-lanes` under `tools/flash_next/lanes_spike/`.
