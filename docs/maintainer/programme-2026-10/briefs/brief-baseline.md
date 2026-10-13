# Task M: baseline and diagnosis of the hosted multi-tenant 27B workload (measure first)

Read `brief-common.md` first (its Flash-Next specifics do not apply to you: you run the **Qwen3.8-27B**
server, `qwen3_8_27b.ninfer` in `C:\ninfer-fork\ninfer-3090\models\`, via `scripts\run.bat qwen38-27b`
or the profiles in `scripts\run.bat`; keep the other rules: GPU lock, PIDs only, worktree of your own,
no destructive actions).

## Why
The user hosts Qwen3.8-27B as a provider for many people. Cache hit rate, TTFT under many tenants
and concurrency are the priority. A study of Infernix's cache
(`cache-study.md` in this folder; read sections 0, 3, 4 and 6) produced nine candidate ideas
(disk-aware eviction, disk cost term, persisted shared prefixes, divergence-tap single-flight,
blocked-head prefetch, write-through at release, slab Host pool, free VRAM as block cache, dense
taps on a disk tier). Step 5 of an earlier programme rejected some "obvious" cache ideas on
measurement (history points, skip-echo capture, GDSF). So before anything is built, we need data
that says which problems actually cost TTFT and hit rate on a hosted workload.

## What to do (no engine behaviour changes; tooling and docs only)
1. **A hosted-workload generator** on top of what exists (`tools/bench/run_agent_rotation.py`,
   `bench/agent-replay`, `tools/bench/run_serve_concurrency.py`, `run_serve_ttft*.py`,
   `tools/bench/ttft/`). It should model: N tenants (16-64 conversations) sharing a few system/tool
   prompts, each growing turn by turn with appended tool results and the occasional edit/branch of
   history, prompt sizes 4K-100K, arrivals Poisson or bursty, lanes 4 and 8, thinking on and off,
   MTP3 (`NINFER_SPEC`) as the launcher default. If a good generator exists, reuse it and just
   parameterise; do not rewrite what works. Document exactly what each preset means.
2. **Baselines on the default 27B profiles** (C4 and C8, the launcher defaults; Host tier 2, 8 and
   16 GiB via `--host-context-mib`; context store off and on to a local folder on C:). For each:
   hit rate split into private-continuation / shared-prefix / none, tokens re-prefilled and **why**
   (never captured, evicted from Host, diverged, sibling still prefilling, store miss), TTFT
   p50/p95/p99 and queue wait, aggregate decode tok/s, lane utilisation, Host allocator failures and
   demotion stalls (look at the metrics the server exports: `GET /metrics`, `/stats`; read
   `docs/serving.md` and `docs/performance.md`), restore times, bytes written to the store per hour.
3. **Attribute the loss.** Using 2, estimate for each of the nine ideas how much TTFT / re-prefilled
   tokens it could recover on this workload (ceiling analysis: e.g. replay the request trace through
   an idealised cache, or count tokens whose only miss reason matches the idea). Say what you
   measured and what is a ceiling. This sets the order of work.
4. **Answer these open questions** from code, docs, git history and measurement:
   - What did step 5's "P4" cover and why was it NO-GO (search `git log`, PR descriptions via
     `gh -R ashalliants/ninfer-3090 pr list --state all --search P4`, docs)?
   - Does a Host allocator geometry failure show up in logs/metrics, and how often under the
     workload (the #254 workaround left "Host alloc failure on demotion" unfixed)?
   - Time spent in admission planning / source inspection per request at C8 with a large cache.
   - Raw sequential read and write speed of the C: NVMe drive (unbuffered, 1-8 GiB file in
     `%TEMP%`, deleted afterwards; also a 4-16 MiB block size sweep). This number feeds the disk
     tier work (Task D runs after you; give it a small reusable script under `tools/bench/`).
   - How the separate gateway repo (`C:\ninfer-fork\ninfer-gateway`, read only) routes requests
     across engines, and whether session-affine routing would raise hit rates when more than one
     engine serves the same tenants. A paragraph and a recommendation; do not change that repo.
5. **Deliverables:** `docs/maintainer/cache-provider-baseline.md` with the tables above, the tooling
   under `tools/bench/` (documented in its README), and a ranked recommendation of the nine ideas
   with expected gain. Open one PR "docs/tools" (branch `bench/provider-baseline`, base master).

## Constraints
- 27B server runs take the GPU lock for their whole lifetime. Keep each hold to about 45 minutes,
  release between phases, keep ABBA sets inside one hold. Other agents queue behind you (a
  Flash-Next cache agent and a batched-forward agent also need the card); never hold it idle.
- RAM: a 27B server needs far less than Flash-Next, but check free RAM before each start.
- Disk: `Get-PSDrive C` before and after; delete scratch artefacts; context-store folders live in
  `%TEMP%\ninfer-baseline\` and are removed at the end.
- Report numbers with n, medians and ranges; mark ceilings as ceilings.
