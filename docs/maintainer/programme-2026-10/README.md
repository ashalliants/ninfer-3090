# Programme 2026-10: Flash-Next cache and lanes, and engine-wide cache work

Living hand-over document. Started 2026-10-13. If you are an agent picking this up cold, read this
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
   hit rate 87-90%). The first slice (batched forward, PR #286, draft) measured 2.2-2.5 ms non-expert
   cost per extra row against the spike's 1.4 ms: the gate (>= 1.15x) is borderline and the
   server-level C = 2 test decides. Ship opt-in (default 1) if it only just passes.
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

## Strands, branches and state

Open PRs and branches move; check `gh pr list -R ashalliants/ninfer-3090`.

| Strand | Branch | State at time of writing | Next step |
|---|---|---|---|
| Flash-Next context cache | merged: #285, #290 | done | none |
| Engine disk tier 1: learned disk read cost | merged: #287 | done | none |
| Engine disk tier 2+3: write-on-evict, persisted shared prefixes | `feat/context-disk-tier-evict` | measured at C4/2 GiB; C4/8 GiB, C8 MTP and no-store regression runs queued | finish runs, open PR |
| Flash-Next `--context-store` + NVMe tier | `feat/qwen4exp-context-store` | code committed, real-model gates pending | gates, PR |
| Flash-Next two lanes | `feat/qwen4exp-lanes` (folds in #286) | rebased on master, builds, exactness run queued | exactness, C = 2 gate, alternation test, PR |
| Hosted-27B baseline and ranking | `bench/provider-baseline` | generator and runs in progress | attribution table, recommendation |
| Op-test hang check | `investigate/softmax-attention-int8-prompt-hang` | `ninfer_softmax_attention_int8_prompt_test` ran 49 minutes with the GPU idle under the full suite | classify hang vs slow |

## Queue: ideas not started (from the study, ranked there; re-rank with the baseline)

1. Write-through at release instead of demote/export on pressure: removes the ~16% engine-worker
   stall write-on-evict introduced (copy-engine discipline on WDDM matters).
2. Divergence-tap single-flight: concurrent subagents whose prompts diverge away from the planned
   markers prefill the shared part once.
3. Blocked-head Host-to-Device prefetch for the queue head (TTFT -3..-6% in Infernix's pressure
   replay; ceiling ~14% at 120K).
4. Slab Host pool (fixed-size slabs, no allocator geometry failures): only if the baseline shows
   allocator failures (the Flash-Next cache already uses slabs).
5. Free VRAM as Device block cache (helps the 35B-A3B; little on 27B at 3090).
6. Dense flexible taps on a cheap disk-backed tier (P1 was rejected earlier because it was Host-bound).
7. Persisting shared prefixes across restarts is in the disk-tier PR; partial restore by value
   (density) belongs with it.
8. Gateway: session-affine routing multiplies every hit-rate gain when several engines serve the
   same tenants (see the baseline PR's gateway note).

Not recommended (rejected with evidence earlier): GDSF eviction, skip-echo capture, history points
at small Host tiers.

## Rules that bit people on this machine

- One RTX 3090, a lock file `%TEMP%\ninfer-gpu.lock`: open with `FileShare.None`, hold it inside one
  PowerShell invocation, poll every 0.5 s, never create or delete the file (see `CLAUDE.md`).
- Flash-Next pins ~33 GiB of experts in RAM: only one Flash-Next process machine-wide.
- A worktree needs its own `build-ninja`; build named targets only; `AGENTS.md` has the recipe.
- Full ctest run takes ~13 minutes but three op tests are very slow: use `ctest -E` while iterating.
  Known environment failures: `jsonschema` module missing (two oracle tests),
  `ninfer_artifact_writer_interop_test`.
- PR policy: at most 3 open PRs per feature, stack one level at most, review fixes go on the PR's
  branch. The owner merges.

## Scratch artefacts

Briefs for each strand are in `briefs/` (they reference scratchpad paths of the original session,
which may not exist; the repo is authoritative). The raw route traces and replay tools of the spike
are on `spike/qwen4exp-lanes` under `tools/flash_next/lanes_spike/`.
