# Context cache study: Infernix hybrid prefix cache vs ninfer-3090

Date 2026-10-13. Read-only study, nothing built or run.

- **Ours**: worktree `C:\ninfer-fork\ninfer-3090\.claude\worktrees\linked-baking-piglet` (branch `spike/qwen4exp-lanes`, base master `04901788`). Paths below are relative to it.
- **Infernix**: clone at scratchpad `infernix-ref/repo`, HEAD `bbc2771c` (2026-10-12). Paths are prefixed `IX:`.

Evidence tags:
- **[V]** verified in code.
- **[D]** stated in a design doc or user doc in that tree, not checked in code.
- **[M]** a measurement recorded in a doc, commit or memory note. It was not re-run here.
- **[N]** taken from the session memory notes (`infernix-adoption-plan.md`), not from the repo.
- **[I]** my inference or arithmetic.

---

## 0. Short answer

1. Infernix's cache is a vLLM/SGLang-style design, built on two kinds of entry:
   - a content-addressed radix tree of 64-token KV blocks;
   - sparse recurrent-state snapshots attached to tree nodes.

   It has two physical tiers: VRAM, plus one pinned Host slab pool. There is also a persistence file that is written at an orderly stop and read at startup. It has no online disk tier and no remote tier.

   Policy lives in a model-agnostic `PrefixCacheIndex`:
   - Device LRU, evicting tail blocks first;
   - Host eviction in this order: superseded snapshots, then dead KV, then GDSF;
   - queue holds.

   Each model has a thin "binding" that moves the bytes the index decides on. The Engine sees about 5 calls.
2. Our cache is upstream NInfer's continuation/checkpoint engine, plus our fork's additions: host auto-sizing, the context store, S3, live-branch priority, single-flight and layered restore. On persistence and tiering it is **richer** than Infernix: an incremental, deduplicated, crash-safe disk store with online hydration and an S3 copy.

   On the Device and Host tiers Infernix is **simpler and more robust**:
   - fixed-slab Host pool, so no allocator-geometry failures;
   - write-through instead of demote-on-pressure;
   - block-level dedup across sessions;
   - prefetch for the blocked head;
   - free VRAM used as block cache.

   Step 5 already measured that we have **no hit-rate gap** on the rotation workload [N].
3. Tiering: no component in our tree owns cross-tier placement.
   - Device↔Host is decided by `ResourceManager` together with the Program's physical actions.
   - Host→Disk is an idle *backup copy* made by `EngineCore`.
   - Disk→Host happens only on a miss (hydration) or at startup.
   - Disk↔S3 is handled by `ContextStore`.
   - Host eviction does not know that a disk copy exists.

   Infernix has the clean seam that was asked about (`PrefixCacheIndex` + `PrefixIndexBackend` + `HybridResourceManager`), but only for two tiers.

   **Recommendation:**
   - Do *not* extract a tiered component out of the mature Qwen3.5 cache.
   - Make the existing stack disk-aware. This is about 3 small changes.
   - Build Flash-Next's cache on Infernix's generic index, with a third (disk) residency added there. That index then *becomes* the tiered component, behind a ~5-call Engine seam.
   - Decide about moving 27B/35B onto it only after measuring.
4. Flash-Next disk tier. A 30K-token checkpoint is ≈553 MB: 437 MB of KV plus a 116 MB image.
   - Through today's store read path (≈0.57 GB/s measured end to end, from the page cache) it restores in ≈1.0 s.
   - Through a streaming NVMe read (assumed 2–3 GB/s, **not measured in the repo**) it restores in ≈0.2–0.3 s.
   - Re-prefilling at 2,350 tok/s takes 12.8 s.
   - Break-even is only ≈110–510 reused tokens. Disk restores never stop paying on latency for long prompts.

   The real costs are elsewhere:
   - RAM: today's path materialises whole images in RAM and imports them into the pinned Host tier.
   - Worker stalls during export.
   - SSD write volume.

---

## 1. How Infernix's cache works

Main source: `IX:docs/maintainer/hybrid-prefix-cache-spec.md` (the spec). Legacy (their own NInfer-derived checkpoint cache) was removed on 2026-10-09 (spec:5-8, 1366-1371).

### 1.1 Data model [D, V]

**Block tree.** Block *b* covers tokens `[64b, 64b+64)`. Its key is `(parent node, the 64 exact tokens, media extra key)`.

- A chained `xxh3` `lookup_hash` only selects a child. A hit always compares tokens and extra exactly (spec:266-287).
- The Frontend computes the hashes off the worker, in `PreparedPrompt::block_hashes`.
- A node stores its Device page id(s), a Host slab, `active_refs`, `restore_pins`, `live_snapshots_below` and its LRU/dead-list links (spec:294-320).
- A node is inserted only when every enabled pool has committed all 64 positions (spec:318, 770-790).

**Snapshots** attach to the anchor node of the last full block below frontier F. Each holds:
- an optional private tail page for `F % 64 ≠ 0`;
- the state image, in a Device snapshot slot and/or in Host slabs;
- a kind: Tap, Endpoint, Boundary or OutputBoundary;
- a `superseded` flag;
- GDSF fields (spec:322-353).

A snapshot is valid only while every node on its path, its tail and its image have at least one resident copy. Losing a node's last copy deletes its subtree (spec:349-351, 992-998).

**Code.** `IX:src/runtime/prefix_cache/prefix_index.h:210-390` (`PrefixCacheIndex`: match, choose, supersede, holds, insert, fills, evictions, slots, host images, persistence helpers) [V].
- The model-agnostic **backend seam** is `PrefixIndexBackend` (`prefix_index.h:86-97`): `release_device_block`, `release_snapshot` and `drop_snapshot_device_image` [V].
- The index owns slab allocation itself (`take_free_slabs`/`allocate_slabs`, `prefix_index.h:498-504`) [V].

### 1.2 Memory tiers [D]

- **Device KV pool.** Same layout as before. With the cache on, the `C·L` upper clamp is removed and `--kv-capacity auto` turns free VRAM into Device block cache (spec:357-364, 1384-1387). This costs 0.2–0.35% decode [M] (spec:1385-1387).
- **Device snapshot slots.** `D = C+1` with a Host tier (spec:365-368).
- **Host slab pool.**
  - One pinned allocation, in chunks of at most 4 GiB.
  - A slab is one page record rounded to 4 KiB.
  - A state image spans `ceil(image/slab)` non-contiguous slabs.
  - The free list is O(1). There is no fragmentation and no KV-vs-state split (spec:369-381).
  - Default 8 GiB, or 4 GiB for Flash-Next (spec:39-41, 1313, 1594).
- **Persistence file** (`--prefix-cache-file`, opt-in; spec:383-427).
  - **Save** happens only on an orderly stop. It covers Host-resident retained snapshots plus their Host paths, written to a tmp file and renamed. A fatal error saves nothing.
  - **Load** happens at startup, only when the fingerprint and geometry match:
    - fingerprint: artifact path/size/mtime, prefill signature, KV, spec backend, RoPE, binary id;
    - geometry: the Host geometry.
  - If the tier is smaller than the file, snapshots are restored greedily by GDSF density (spec:1505-1506).
  - There is **no** online read, **no** periodic or idle write, and no remote copy.
- **No SSD/disk tier for KV or state.** I searched the docs. Flash-Next's design mentions SSD only for *experts* and the n-gram volume, not for the prefix cache (`IX:docs/maintainer/qwen3_8-flash-next-design.md` §19.3.7). [V by search]

### 1.3 Admission [D, V]

1. **Match.** Walk the prompt's block hashes from the root and compare exactly. Collect snapshots on the matched path whose tail tokens match, with `F ≤ n−1`, so at least one token is always prefilled (spec:435-447).
2. **Choose.** Pick among: the root, the deepest valid snapshot, and the deepest fully Device-resident snapshot.

   `cost = restore_bytes / h2d_bw + per_batch + prefill(base=F, n−F)`

   The cost uses the Engine's calibrated `context_cost` coefficients (spec:452-464). The model is `IX:src/runtime/prefix_cache/cost.h:12-53` [V]. For offloaded-expert models it adds a saturating per-call cost `call_route_fraction` (cost.h:22-31) [V].
3. **Capacity quote.**

   `need_pages = host-only path blocks + tail COW + first unit growth ≤ free + evictable`

   On a shortfall, fall back to the next candidate and finally the root (spec:468-481). This is O(prompt blocks), with no search: "p99 < 1 ms" was the target (spec:1266).
4. **Admit.** This is the Engine's ordinary binding transaction:
   - pin the path;
   - evict the Device LRU to make room;
   - open one restore batch on a dedicated restore stream for Host-only blocks, tail and image;
   - prepare page-prefix forks (spec:517-551).
5. **Layer-pipelined restore.** The prelude goes first, then each model layer's KV/state in forward order, with one event per layer. The lane's first prefill pass waits per layer, so admission never waits for the copies (spec:553-592).

   Measured on a 5090 at PCIe5 x8 (24–27 GB/s) [M] (spec:1396-1407):
   - ~11.4K revisits: 60.0→45.7 ms;
   - 90K: 301→268 ms.

### 1.4 Snapshot policy and tap planning [D, V]

`plan_taps` (`IX:src/runtime/prefix_cache/tap_planner.h:11-77` [V]; spec:625-683) lists candidates in this priority order:

| Priority | Candidate | Placement |
|---|---|---|
| 1 | client breakpoints / `cache_control` | exact |
| 2 | generation opener | exact |
| 3 | end of tools / leading system block | exact |
| 4 | prompt tail `n−1` | flexible |
| 5 | geometric ladder `n − G·2^k`, snapped to message boundaries | flexible |

Placement rules:
- **Exact** taps split a prefill chunk, at about 15 ms per split on 27B.
- **Flexible** taps snapshot at the next chunk boundary at no compute cost: one D2D copy of the image, about 0.25 ms (spec:642-646, 1372-1375).

Budget: at most 8 new taps with a Host tier and 2 without. Proximity and cluster rules mean chat traffic gets about one split per request (spec:647-671).

Endpoints are snapshotted at finish or a consistent abort. The partial last page moves in without a copy (spec:792-809).

### 1.5 Block publication and write-through [D]

- Blocks join the tree at every commit, covering prefill, decode and cancelled requests (spec:770-790).
- A recomputed block adopts the existing node, so it never holds two pages (spec:757-768).
- Write-through D2H of a lane's Device-only nodes runs in one batch **when the lane releases them** (spec:779-781).
- As a result, Device eviction never copies. Dropping a backed copy is pure bookkeeping (spec:75, 896-900).

### 1.6 Eviction and cost model [D, V]

**Device LRU** (spec:889-900):
- Unpinned cached pages are released tail-first, in reverse block order.
- Host-backed entries go before unbacked ones.

**Device snapshot slots** are taken in this order (spec:902-920):
1. a free slot;
2. a superseded snapshot's slot;
3. the lowest-GDSF Device-only snapshot, only if the new one is worth as much;
4. the least-recently-hit Host-backed snapshot.

**Host slabs** are freed in this order (spec:922-990):
1. Dead-KV sweep: nodes with no live snapshot below them, no refs and no pins, leaves first.
2. The oldest superseded snapshot.
3. The minimum-`H` retained snapshot, scored by GDSF:

   `H = L + (1+hits)·C/Z`, where
   - `C = prefill(base = nearest retained ancestor) − restore`;
   - `Z = image + tail + exclusive path bytes`.

An image write is refused if it would evict something worth more.

**Supersession** (spec:924-944). When a lineage publishes deeper, the snapshot it resumed from becomes evict-first, together with the "passed endpoints". Boundary snapshots are superseded only while the tree below them is a single chain.

**Queue holds** (spec:1009-1048). The queue acts as eviction lookahead:
- held snapshots are never Host victims;
- held Device entries move to per-rank LRU lists.

This came from a Flash-Next production log: 84–98% of each prompt was cached when the request was first considered, but only 9–21% at admission [M] (spec:1017-1019).

### 1.7 Single-flight, prefetch and other engine-side features [D, V]

**Single-flight** (`hybrid_await_sibling`, spec:1128-1148, 1392-1395):
- A head that shares a prefix with a lane still prefilling waits for that lane's snapshot **at the divergence**.
- An exact tap at `floor64(d)` is *added to the sibling's plan*.
- Measured [M]: a 4×13.9K burst went from 11.0% to 74.9% cached, mean TTFT 3.52→1.48 s.

**Blocked-head prefetch** (spec:594-618; engine side `IX:src/runtime/engine/context_cache/hybrid_resource_manager.h:42-57` [V]):
- While the FIFO head waits, up to 256 Host-only path blocks are copied into spare Device cache.
- Measured [M] (spec:1450): admission Host restores −22/−25%, TTFT without queue wait −3/−6%.

**Engine seam** (`hybrid_resource_manager.h:20-119` [V]): `candidates`, `reclaim`, `prefetch_blocked_head`, `hold_queue` and `populate_runtime_stats`. On the Program side, about 10 `hybrid_*` calls (spec:815-838).

Not built (spec:58-61, 1151-1161):
- zero-split GDN tap;
- KV page transfer kernel;
- output-boundary snapshots;
- KV-write elision;
- backfill. Backfill was tried and reverted as a latency trade-off, not a win (spec:1464-1472).

### 1.8 Flash-Next (Qwen4Exp) binding [D, V]

Spec §17 (spec:1582-1605) and the design `IX:docs/maintainer/qwen3_8-flash-next-design.md:3700-3900`.

- It reuses the generic index, tap planner and Frontend keys unchanged. The binding is new, about 3.0k lines in `IX:src/models/qwen4_exp/program/prefix/` [V by wc].
- **Snapshots are Host-born.** They are copied straight from the lane to Host slabs, one decoder layer at a time, with no Device snapshot slots. The Host tier defaults to 4 GiB and must be positive (spec:1594-1595; `prefix_cache.h:143-199` [V]).
- **Sizes** [D] (design:3758-3760):
  - state image 115,673,088 B (110.3 MiB);
  - INT8 KV incl. MTP 14,560 B/token;
  - 30K tokens = 437 MB of KV.

  Restoring 30K at ~26 GB/s takes ≈21 ms, against 45–75 s of re-prefill in their estimate, which predates their prefill work.
- **MTP rules MR1–MR9** handle the drafter's cells, with per-node `mtp_next` (design:3815-3829).
- **Lane-resident reuse.** The next turn of the conversation still in the lane skips the image restore (design:3810-3813).
- **Preemption** publishes the lane's blocks and state, then resumes without replay (spec:1604).
- Supersession differs from §9.3: a request's own ladder, boundary and last taps stay retained. This measured +1.9 points cached and −8.7% average TTFT on a Dense8 agent replay [M] (spec:1580).

---

## 2. Our cache, briefly

Authority: `docs/maintainer/resource-scheduling-and-context-cache.md` (Chinese, upstream). Code: `src/runtime/engine/context_cache/`, `src/runtime/engine/context_store/`, `src/models/qwen3_5/program/{storage,transactions,planning}`.

**Units** (doc:9-15, 120-131) [D]:
- *continuation owners*, each with an endpoint E;
- a typed input recovery point R (`ResponseReplay`/`TurnClosure`);
- up to 4 private long anchors;
- *public shared entries* at up to 4 marker frontiers (doc:185-201).

Independently computed states with equal tokens are **not** spliced (doc:182-183; `resource_manager.h:162-163` [V]).

**Index.** `src/runtime/engine/context_cache/prefix_index.h:11-44` [V] is a compressed trie over *checkpoint* digests at checkpoint frontiers, not over KV blocks.

**Policy/physics split.** The header says "Policy owns bounded logical continuations. Native owns checkpoint contents and every byte" (`resource_manager.h:20-22` [V]). Sources are ranked by `transfer + prefill` cost, then bytes, then depth, then session hint (`resource_manager.h:87-175` [V]).

**Retention and reclaim** (doc:313-410) [D]:
- ordinary vs reuse tiers;
- a 256-entry demand summary (repeated-compute evidence);
- the reuse tier has a 3/4-of-pool soft target;
- reclaim order: superseded-only first, then ordinary content by marginal recovery loss per useful freed unit, then reuse content by oldest demand.

Demotion copies Device→Host under pressure, before it releases (doc:368-370). Host allocation can fail on "allocator geometry" even when the byte quote fits (doc:360-361, 409-410).

**Tiers and defaults** (doc:65-80) [D]:
- The Main KV pool upper bound is `C × ceil(max_context/64)`, so it is not grown into free VRAM.
- Extra Device state slots default to C.
- The pinned Host tier is `8 GiB + 8 × StateImage`, or `--auto-host-cache`.

**Fork additions in tree** [V/D]:

| Addition | Where | Status |
|---|---|---|
| Single-flight | `engine_core.h:1834-1901`, doc:281-288 | waits only for a sibling's *already planned* SharedPrefix frontier ≥1024 tokens deeper than the best source [V] |
| Live-branch priority | doc:318-323 | PR #254 [N] |
| Layered Host restore | doc:499-517 | PR #259. Uses a gather kernel, because GeForce under WDDM has one copy engine [D] |
| Context store + S3 | `context_store.h`, `engine_core.h:2840-3099`, `docs/serving.md:382-500` | see §3 below |

---

## 3. Feature comparison

| # | Mechanism | Infernix | Ours | Verdict, and what it buys |
|---|---|---|---|---|
| 1 | Unit of reuse / identity | 64-token content blocks in a radix tree + sparse snapshots. Exact token compare, hash only for lookup (spec:266-287) | Owners with E/R/≤4 anchors + ≤4 shared entries; digest trie over checkpoints (`prefix_index.h:11-44`) | **Infernix simpler**. Hit rate: step 5 found no gap on rotation, where append turns already get 100% reuse [N]. The gap shows only where reuse points must be dense |
| 2 | KV sharing across sessions | Unconditional: identical blocks share one physical page (spec:70) | Only via shared entries or forks of one history; no splicing of independently computed KV (doc:182-183) | **Infernix better** on memory with many sessions sharing an unmarked preamble. Small at our C ≤ 2–4 [I] |
| 3 | Where state is captured | Exact taps at semantic points (split) + flexible taps at chunk boundaries (no split) + log2 ladder, ≤8 per request (spec:625-683) | Exact semantic frontiers only (E, R, anchors, markers); chunk ends never capture (doc:207-216) | **Mixed.** Ladder/history points (our P1) were **rejected** at 8 GiB because they were Host-bound [N]. A cheaper tier (slabs or disk) would change that |
| 4 | Admission choice | root / deepest / deepest Device-resident; cost = restore + prefill; O(path) (spec:452-481) | All matching points + root, same cost formula, plus tie-breaks (`resource_manager.h:134-161`) | **Equivalent** in ranking. Our planning time is unmeasured here |
| 5 | Device tier size | Free VRAM becomes block cache (`--kv-capacity auto`, no `C·L` clamp) (spec:357-364) | Pool ≤ `C·ceil(L/64)` (doc:73-76) | **Infernix better** where VRAM is free (35B-A3B). Little on 27B@3090, where the 188K profile barely starts [N] |
| 6 | Host tier allocator | Fixed slabs, O(1), no fragmentation, KV and state share the pool (spec:369-381) | Extent arena + state pool; geometry can fail (doc:360-361, 409-410; `host_kv_store.h:51-53`) | **Infernix better**: robustness. Our #254 "Host alloc failure on demotion" was worked around, not fixed [N] |
| 7 | Device→Host movement | Async write-through at lane release; Device eviction is free (spec:75, 779-781) | Demote on pressure: D2H before release (doc:368-370) | **Infernix better** under pressure. A copy leaves the critical path [I]. Watch the single copy engine on WDDM (doc:502-504) |
| 8 | Host eviction policy | Superseded → dead KV → GDSF; image writes refused if worth less (spec:922-990) | Superseded-only → ordinary by loss/byte → reuse by demand age; demand-evidence admission for new captures (doc:351-410) | **Comparable.** GDSF was judged NO-GO in step 5 [N]. Dead-KV has no analogue in ours, because we keep no KV without a checkpoint [I] |
| 9 | Supersession | Lineage supersedes its resume point + passed endpoints; Boundary only while unshared (spec:924-944) | Abandoned branch ≥256 shared tokens → superseded; live branch inherits priority (doc:318-323, #254) | **Equivalent intent.** Ours has a known misjudgement risk for unrelated conversations sharing ≥256 tokens [N] |
| 10 | Waiting requests' sources | Queue holds: eviction order from queue rank (spec:1009-1048) | Waiting references protect a retained source (doc:290-297) | **Roughly equivalent** [I]. Ours protects per waiting request; Infernix orders by rank |
| 11 | Single-flight | Waits at the **divergence**: adds an exact tap to the sibling (spec:1128-1148) | Waits only for a planned SharedPrefix frontier ≥1024 deeper (`engine_core.h:1865-1876`) | **Infernix better** when siblings diverge away from a marker. Ours covers the common case (system/tools markers) |
| 12 | Restore overlap | Restore stream, per-layer events for KV **and** state; copy engines (spec:553-592) | Gather kernel for Main KV only; initial binding, single rank, non-Replay (doc:499-517) | **Equivalent.** Ours is adapted to one copy engine. P6 measured −7.6% p50 TTFT [N] |
| 13 | Blocked-head prefetch | Yes (spec:594-618) | No (no prefetch in `engine_core.h`) [V by search] | **Infernix only.** Small ceiling: restore was 14.4% of TTFT at 120K [N] |
| 14 | Lane-resident reuse | Flash-Next skips the image restore when the lane still holds it (design:3810-3813) | Moving an exclusive E is equivalent on Qwen3.5 (doc:239-241) | **Equivalent** on 27B. Needed for Flash-Next |
| 15 | Preemption recovery | Replay only on Qwen3.5; Flash-Next publishes to the cache | Snapshot **or** Replay (doc:458-481) | **Ours more capable** |
| 16 | Persistence | One file at orderly stop and startup; partial load by GDSF density; no crash safety (spec:383-427) | Chunked, content-deduplicated store: idle background writes (≤30 s loss on crash), startup restore, **online hydration on miss**, TTL/LRU, foreign-binding ageing (`serving.md:382-464`) | **Ours much better** |
| 17 | Remote tier | None | S3-compatible bucket; warm start for a new box (`serving.md:467-500`; `context_store.h:57-66`) | **Ours only** |
| 18 | What persistence saves | Every retained Host snapshot, including shared Boundary ones | Private continuations only; "Public shared prefixes are not stored" (`serving.md:389`) | **Infernix better**: shared system/tools prefixes survive a restart |
| 19 | Code size | ≈2.9k index + 3.2k Qwen3.5 binding + 3.0k Flash-Next binding | ≈2.7k `context_cache` + 1.7k store + 11.8k Qwen3.5 storage/transactions/planning (not all cache-specific) [V wc] | **Infernix is about 2× smaller** |
| 20 | Flash-Next | Full cache (§1.8) | None. Every hook is a stub (`qwen4_exp/program/program.h:79, 415, 432, 436`); `--context-store` is refused (`qwen4_exp_instance.cpp:111`) [V] | **Infernix only** |

### Already evaluated in step 5 [N]

| Item | Status | Reference / result |
|---|---|---|
| Single-flight | ported | PR #255 |
| Live-branch priority | ported | PR #254 |
| Restore overlap | ported | PR #259 |
| Context store under DFlash2 | ported | PR #227 |
| History points / ladder (P1) | rejected | edits ≥ ideal−128 only 15/23; append hit 0.266→0.245 at 2 GiB |
| Skip-echo capture (P2) | rejected | append hit 0.678→0.490 |
| GDSF | no-go | — |
| P4 | no-go | contents not recorded in my notes; see Q4 |

### Not yet considered in step 5

All of these apply to any model:
- blocked-head prefetch;
- divergence-tap single-flight;
- free VRAM used as Device block cache;
- slab Host pool;
- write-through at release;
- content-deduplicated blocks;
- persisting shared prefixes;
- partial store restore by value (GDSF density) rather than recency;
- the disk-aware eviction ideas in §6.

---

## 4. Ranked list of adoptable ideas

The "Benefit" column is the expected benefit and the "Risk" column the cost or risk. Ranked by benefit ÷ (effort × risk), for any model.

| Rank | Idea | Benefit | Risk |
|---|---|---|---|
| 1 | **Disk-aware Host eviction + write-on-evict.** Re-add "writing a session at eviction time", which was dropped in `9e893281` (commit message). Let reclaim price a victim whose current fingerprint is in the store as a disk-restore loss, not a full recompute | Evicted conversations come back in ~1 s instead of a full re-prefill. Lets the Host tier stay small (Flash-Next) | Small, in EngineCore + ResourceManager. Risks: eviction must wait for the write or keep the Host copy pinned until written; the Engine-worker export step already blocks arrivals (`serving.md:461-464`) |
| 2 | **Disk read cost in the cost model.** Replace the fixed `kMinimumHydrationGain = 4096` (`engine_core.h:2887-2890`) with a disk-read term in `ContextMachineCostModel` | Correct hydration choices per model: Flash-Next break-even is ~0.1–0.5K tokens, not 4K. Feeds idea 1 | Low. Needs one disk-rate measurement per machine (Q1) |
| 3 | **Persist shared prefixes and store the largest-value set** (Infernix §5.5 density restore) | System/tools prefixes survive restarts; agent subagents start warm | Low–medium. The store is model-agnostic, so this is mostly export/import of SharedPrefix entries |
| 4 | **Divergence-tap single-flight** (spec:1128-1148) | Concurrent subagents with prompts that diverge away from markers prefill once | Medium: adds an exact capture to a running sibling's plan. Only at C ≥ 2 |
| 5 | **Blocked-head Host→Device prefetch** (spec:594-618) | TTFT −3…−6% in Infernix's pressure replay [M]. Ceiling ~14% of TTFT at 120K [N] | Medium: the one copy engine contends with the head's own inputs. Would need our gather kernel or idle-only copies |
| 6 | **Write-through at release** instead of demote-on-pressure (spec:779-781) | Pressure eviction becomes bookkeeping; the D2H leaves the critical path | Medium–high: changes demotion in `transactions/reclaim.cpp` (1.2k lines); copy-engine discipline on WDDM |
| 7 | **Slab Host pool** (spec:369-381) | Removes allocator-geometry failures (doc:360-361) and makes capacity arithmetic exact | High: core storage (`host_kv_store.h`, `state_store.h`). Worth it only if geometry failures show up in logs |
| 8 | **Free VRAM as Device block cache** (spec:357-364) | Helps 35B-A3B. Little on 27B@3090 | Medium. Check against step 5's P4 NO-GO (Q4) |
| 9 | **Dense flexible taps (ladder) on a cheap tier.** P1 was rejected because it was Host-bound; with disk-backed points it might pass | Edits deep in history resume closer | Medium. Only after ideas 1–2 exist |

Not recommended: GDSF and skip-echo capture (rejected with evidence), backfill (Infernix reverted it), zero-split GDN tap (Infernix never needed it).

---

## 5. Tiering and the "single tiered cache" question

### 5.1 Tiers today

| Tier | Ours | Infernix |
|---|---|---|
| VRAM KV | Main/backend paged pools ≤ `C·L`; checkpoint KV shares pages with histories (doc:73-76) | Pool incl. free VRAM; cached blocks (spec:357-364) |
| VRAM state | C lane images + C extra slots (doc:70) | C lanes + D snapshot slots; Flash-Next D = 0 (spec:365-368, 1594) |
| Pinned Host | Extent arena + state pool, 8 GiB + 8 images or auto (doc:78-92) | Slab pool, 8 GiB; Flash-Next 4 GiB (spec:369-381) |
| Local disk | `--context-store`: online, deduplicated, crash-safe (`context_store.h:25-43`) | Shutdown/startup file only (spec:383-427) |
| Remote | S3 (`context_store.h:57-66`) | none |
| OS file cache | Implicit: warm store reads come from RAM (`serving.md:415-417`) | n/a |

### 5.2 Who decides placement and movement

**Ours** [V/D]:

| Movement | Decided by | Mechanism |
|---|---|---|
| Device↔Host | `ResourceManager::reclaim` / `host_victims` / `begin_reclaim` (`resource_manager.h:528-583`) picks victims | The Program enumerates "finite physical actions" (`plan_reclaim`, `plan_releases`, `start_demote`; `program.h:500-507`) |
| Host→Device restore | chosen at bind | `inspect_source` / `start_binding` |
| Host→Disk | `EngineCore` idle persist (`engine_core.h:2840-2885`), deepest idle owner first | A *copy*. The Host content is not freed and eviction does not know the disk has it (`resource_manager.h` never mentions the store except `adopt_restored`, lines 753-806) [V by grep] |
| Disk→Host | `EngineCore::start_hydration` (`engine_core.h:2902-2945`), when the store beats the cache by ≥4096 tokens, and at startup | `import_for_request` makes Host room under OptionalWrite rights (`engine_core.h:2992-3031`) |
| Disk↔S3 | `ContextStore` | Upload on put, list/fetch on load |

**Infernix** [V]: one index decides Device and Host. Bindings move bytes. Persistence is a stop/start bulk copy.

### 5.3 The interface between engine and cache

**Ours.** `EngineCore<Instance>` → `ResourceManager<Model>` → Program.
- `EngineCore` calls 49 distinct Program methods (counted by grep in `engine_core.h` + `resource_manager.h`). About 25 of them are checkpoint- or cache-specific: `checkpoint_*`, `plan_reclaim`, `plan_releases`, `start_demote`, `start_capture`, `prepare_capture`, `export/import_checkpoints`, and so on.
- Handles are model types (`CheckpointHandle`, `SourceCandidate` with Move/Fork/carry flags; `program.h:142-192`).
- The Flash-Next Program stubs **27** of these (count of stub returns in `qwen4_exp/program/program.h`) [V].

**Infernix.** Engine → `HybridResourceManager` (5 calls) → Program `hybrid_*` (≈10) → `PrefixCacheIndex` (generic) + model binding via `PrefixIndexBackend` (3 callbacks) [V].

### 5.4 Is there a clean seam for one tiered cache component?

**In Infernix: yes, for two tiers.** `PrefixCacheIndex` already owns residency per node/snapshot, Device LRU, Host GDSF, supersession and holds. The model binding only copies bytes and owns leases.

Adding a third residency (`disk`) to the index would cover the "best reusable state for this prompt / keep this state" contract:
- node/snapshot `disk_state`;
- a `restore_seconds` term per tier in `CacheCostModel`;
- demote-to-disk as a Host eviction step before deletion;
- persistence that becomes online instead of stop/start.

That is the cleanest place in either codebase to build a tiered cache [I].

**In ours: no clean seam today** [I, from the evidence above]. Three things are in the way:
- The logical unit (continuation owners with typed points, Move/Fork/carry, demand evidence) is entangled with physical facts that only the Program knows: shared page ownership between histories, extent geometry, StateImage replicas. The ResourceManager receives *quotes* of physical actions and makes no residency decisions of its own (doc:338-345; `program.h:226-239`).
- Disk sits in a third place (EngineCore) with a different unit (whole session images).
- A component owning placement across all four tiers would have to absorb the ResourceManager's policy, the Program's demotion planning and the store's write/hydrate logic. That is effectively a rewrite of ~14k lines of mature code (§3 row 19), with upstream merge pain on every future Neroued catch-up (155 conflicts on one merge [N]).

### 5.5 Is extraction worthwhile? Assessment

**What it would own.** One residency map, keyed by (prefix identity → state + KV ranges), recording which tiers hold a copy. Plus:
- one cost model: prefill / H2D / disk read / S3 fetch;
- one eviction ladder: VRAM → pinned → disk → S3 → gone, by age × value;
- promotion on admission, prefetch for queued requests, write-behind.

**What it would buy** [I]:
- Disk becomes a real demotion tier rather than a backup copy.
- Flash-Next and future models plug in by implementing a byte-mover, not 25 hooks.
- One place for metrics.

**What it would not fix:**
- the recurrent-state rule: state exists only where a snapshot was taken;
- tap placement and chunk-split costs;
- per-model exactness: MTP cells, DFlash rings, Vision spans;
- the single copy engine on GeForce/WDDM;
- Flash-Next's RAM budget;
- Engine scheduling, preemption and replay;
- forward-pass integration of layered restores.

**Cost and risk.**
- For Qwen3.5, a rewrite on the scale of Infernix's own replacement: ~6k new lines plus about two weeks of production-log tuning, with three rounds of eviction fixes (spec §16.3–16.7).
- Step 5 measured no hit-rate gap to recover on our rotation workload [N].
- High regression risk on the default 27B/35B profiles.

**Verdict.** Not worth extracting from the current Qwen3.5 cache now. Instead:

1. **Now, in the current stack (small):** ideas 1–3 of §4. These give disk-aware eviction, write-on-evict, a disk term in the cost model and persisted shared prefixes. They make disk a de facto tier without a new component.
2. **Flash-Next cache: build on Infernix's generic index instead of our 25 hooks** (see §6.4). It gives us the clean two-tier seam for free. It also matches the code our `qwen4_exp` port came from (Infernix's Flash-Next binding with MR1–MR9 already exists, ~3k lines). Integration cost: a second resource-manager mode in our `EngineCore` (Infernix's `hybrid_resource_manager.h` is 121 lines) plus the `hybrid_*` Program surface.
3. **Then add the disk residency inside that index.** This is the actual tiered component. Back it with our `ContextStore` (chunks, dedup, S3), which is already model-agnostic (`context_store.h:35-38`).
4. **Only then decide on Qwen3.5.** Run an A/B of the hybrid index against our cache on the agent replay with an 8 GiB Host tier. Migrate only if it wins clearly; Infernix's 0.658 geo-mean TTFT ratio was measured against NInfer's *older* catalog cache, not our current engine (spec:1513-1533) [D].

Note: `brief-cache.md` in the scratchpad asks an agent to build Flash-Next reuse on *our* hooks (`plan_reclaim`, `checkpoint_footprint`, …). Path 2 above conflicts with that brief; see Q5.

---

## 6. Flash-Next disk tier: estimate

### 6.1 Inputs

| Quantity | Value | Source |
|---|---|---|
| State image | 115.7 MB (110.3 MiB) | IX design:3759 [D]; task statement |
| INT8 KV incl. MTP | 14,560 B/token (~15 KB) | IX design:3758 [D]; task statement |
| Re-prefill | 2,350 tok/s (NVMe n-gram volume: 2,345 / 2,422 / 2,372 at 4K / 32K / 128K) | `docs/serving.md:127-131` [M] |
| H2D on this 3090 | ≈24.9 GB/s (D2H ≈23.1) | `context_cost_defaults.cpp:104-115`: `ns_per_byte_q32` 172,624,952 / 2³² [V]; the direction order follows the enum, `resources.h:77-81` [I] |
| Store read, end to end | 856 MB image in ~1.5 s from a **warm file cache** (TTFT 1.6 s vs 5.1 s prefill, 27B 7.8K) | `docs/serving.md:415-417` [M]. ≈0.57 GB/s effective. Includes two xxh64 passes per 32 MiB chunk (`context_store.cpp:199-200`), `ifstream` reads (`:809`), assembly into a `std::vector` (`context_store.h:159`) and import into the Host tier |
| Raw NVMe sequential read | **not measured in the repo**. Assumed 2–3 GB/s for a PCIe 3/4 NVMe with large reads | [I], unverified. The n-gram volume measurements are random 4K reads (`serving.md:122-135`), so they don't measure this |

### 6.2 Restore vs re-prefill

Bytes = 115.7 MB + 14,560·n. Restore assumes the image is read through the store (A) or a streaming reader (B). H2D overlaps with the first pass in a layered restore.

| Prompt n | Bytes | Re-prefill (2,350 tok/s) | (A) today's path, 0.57 GB/s | (B) streaming NVMe, 2.5 GB/s [I] | Pinned RAM only (H2D) |
|---:|---:|---:|---:|---:|---:|
| 8K | 235 MB | 3.5 s | 0.41 s | 0.09 s | 9 ms |
| 30K | 553 MB | 12.8 s | 0.97 s | 0.22 s | 22 ms |
| 64K | 1.07 GB | 27.9 s | 1.88 s | 0.43 s | 43 ms |
| 128K | 2.02 GB | 55.8 s | 3.55 s | 0.81 s | 81 ms |

On top of a disk restore, add the suffix prefill and one H2D. For (A) the H2D is serial today: hydration lands in the Host tier first and binds after.

### 6.3 Break-even and when disk-only stops paying

**Break-even** is the reused prompt length n where `n / 2350 = image / B + n·14,560 / B`. Prefill costs 425.5 µs/token. A disk restore costs 5.8 µs/token at 2.5 GB/s and 25.5 µs/token at 0.57 GB/s.

| Disk rate | Break-even n |
|---|---:|
| 0.57 GB/s (today) | ≈ 510 tokens |
| 2.5 GB/s | ≈ 110 tokens |
| ~0.45 GB/s (SATA, cold) | ≈ 650 tokens |

Flash-Next prefill also has a large fixed cost per call (expert staging; the first 4K call is the slowest, `serving.md:114-116`). The true break-even is lower still [I].

**It never stops paying on latency for long prompts.** Per token, a disk restore is 17–70× cheaper than prefill, and prefill attention grows superlinearly. The `kMinimumHydrationGain = 4096` gate (`engine_core.h:2890`) is conservative for Flash-Next but harmless.

It stops paying, or costs too much, in these cases:

1. **RAM, not time** [V/I].
   - `ContextStore::load` returns the whole image as a `std::vector` (`context_store.h:159`), and `export_checkpoints` builds a whole `CheckpointImage::bytes` (`program.h:405-419`). The writer queue holds 2 images (`serving.md:461`). Transient RAM is ≈0.55 GB per 30K and ≈2 GB per 128K image, ×2.
   - Hydration then needs equal **pinned Host-tier room**, because it imports there (`engine_core.h:2992-3010`; `serving.md:385` "needs a Host context cache").
   - So "disk-only" is not possible on the current path. It needs a streaming reader: disk → small pinned staging ring (e.g. 256 MiB) → device, layer by layer, reusing our layered restore idea (doc:499-517) or Infernix's per-layer restore tickets (`IX:.../qwen4_exp/program/prefix/prefix_cache.h:201-221`).
2. **Worker stalls.** Taking an image is one Engine-worker step, so arrivals wait for the D2H/H2H copy (`serving.md:461-464`): ≈25 ms of D2H per 553 MB plus memcpy [I]. Writes happen only after 30 s idle, so busy agent loops are not taxed.
3. **SSD writes** [I].
   - Each persisted conversation update writes the changed 32 MiB chunks: the full 116 MB image per recovery point plus the new KV tail. An image can also hold *several* points (E, R, anchors); the 27B 7.8K image was 856 MB.
   - At one 150 MB write every 20 s that is ~650 GB/day.
   - The idle-only policy and chunk dedup keep the real figure far lower, but write-on-evict (idea 1) would raise it.
   - Mitigation: persist only the endpoint/opener point for Flash-Next, and make writes depend on the eviction value.
4. **No resident copy for the active conversation.** At C = 1 the next turn of the live conversation should resume from the lane (lane-resident reuse, IX design:3810-3813), never from disk. Disk is for interleaved conversations (subagents) and restarts.
5. **Short reuse or early divergence.** Below ≈0.5K reused tokens, or when the prompt diverges before the only stored point (re-rendered thinking without an opener tap), disk adds latency and saves nothing.

### 6.4 What already exists to reuse

- **`ContextStore`.** Model-agnostic byte images with regions, keys and a binding: chunking, dedup, crash safety, TTL/LRU, S3 (`context_store.h:25-43, 47-67`) [V]. Reusable as is.
- **EngineCore persist/hydrate machinery.** Idle writer, startup restore, hydration on miss with a deadline (`engine_core.h:2840-3099`) [V]. It needs Program `export_checkpoints`, `import_checkpoints`, `checkpoint_image_host_bytes` and ResourceManager owners. Qwen4Exp stubs those today: `program.h:485-489` declares them; the instance refuses `--context-store` at `qwen4_exp_instance.cpp:111`.
- **Layered restore pattern.** Our gather kernel (doc:499-517) and Infernix's per-layer restore events for Flash-Next (prefix_cache.h:212-221).
- **Infernix's Flash-Next prefix binding and persistence.** `IX:src/models/qwen4_exp/program/prefix/{prefix_cache,prefix_program,state_image,call_plan,persist}.*`, ≈3.0k lines, Apache-2.0 (keep attribution). Its `persist.cpp` already serialises Flash-Next blocks and images (format `NINFQ4PC`, spec:1603); a store adapter could use the same layout.
- **RAM ledger** (`plan_host_memory`, cited in `brief-cache.md`). That is where a Host-tier budget and a staging ring must be accounted for.

**Suggested Flash-Next shape** [I]:
- lane-resident reuse for the live conversation;
- a small pinned Host tier (1–4 GiB ≈ 2–7 sessions at 30K) inside the RAM ledger;
- the disk store as the main capacity tier, with write-on-evict and streaming hydration;
- the cost model deciding whether to hydrate (≈0.2–1 s) or prefill (12.8 s at 30K).

---

## 7. Open questions

1. **Raw NVMe sequential read rate on this box (C:).** Measure: unbuffered, 32 MiB reads. Also measure how much of the 1.5 s per 856 MB is hashing, assembly or import rather than I/O. This decides whether a streaming reader is worth building (≈0.2 s vs ≈1 s at 30K).
2. **How often Host eviction drops conversations that the store holds.** Do agent-replay logs show re-prefills that hydration could have served if eviction were disk-aware? Is there a counter for "evicted while persisted"?
3. **Planning time per admission on our engine.** Infernix's reason for replacing Legacy was a 250 ms planning allowance (spec:107-113). Our current engine was never measured here.
4. **What P4 was in step 5 and why it was NO-GO.** It may already cover idea 8 (free VRAM as block cache).
5. **Is the Task-A agent (`brief-cache.md`, branch `feat/qwen4exp-context-cache`) already building Flash-Next reuse on our hooks?** If so, §5.5 step 2 (port Infernix's index + binding instead) needs a decision before more is built.
6. **Allocator-geometry failures in our Host tier.** Do they show up in production logs (the #254 workaround)? This decides whether the slab pool (idea 7) is worth its risk.
7. **SSD endurance budget the user accepts** for write-on-evict on a desktop NVMe.
