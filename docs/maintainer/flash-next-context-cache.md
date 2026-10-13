# Qwen3.8-Flash-Next context cache

Status: in progress on `feat/qwen4exp-context-cache` (design note; becomes the authority for the
Flash-Next cache once merged, beside [the model reference](qwen4-exp-model.md)).

The Qwen3.8-27B/35B Programs implement NInfer's context cache contract (capture frontiers, private
continuations, shared prefixes, Host tier, reclaim), driven by the shared `EngineCore` and
`ResourceManager` ([design](resource-scheduling-and-context-cache.md)). The Flash-Next Program
stubbed every hook, so each request prefilled its whole prompt. This change implements the same
contract in the Flash-Next Program. The Engine, ResourceManager and Scheduler are unchanged.

## What a checkpoint is

A Flash-Next checkpoint is the lane's complete execution position at one token frontier F, held
only in pinned Host memory:

| Part | Size (INT8 KV, MTP on) | Source |
|---|---|---|
| KV pages `[0, ceil(F/64))`: K/V codes, scales, pooled index keys of the 12 QSA layers and the MTP drafter's layer | 931,840 B per 64-token page (14,560 B/token) | the lane's elastic KV pool, plane-major |
| State image: 36 GDN conv + recurrent states, PLE convolution history and slot word, 12 QSA index-key tails, the drafter's tail | ~110 MiB | `state_backing_`, one contiguous block |
| The drafter's saved residual (`residual(F - 1)`) | 20 KiB | speculation buffer |
| BF16 logits of position F - 1, when F is the end of a prefill call | ~0.5 MiB | the last call's logits |
| Identity: tokens `[0, F)`, three-axis positions, media items, rolling digests; `mtp_cells`, `history[F]` | host vectors | the request |

Nothing lives in VRAM: the expert cache keeps every frame it had. A restore copies Host to the
lane's own KV pool and state, so the source checkpoint is never written.

## Host tier

- One pinned arena allocated at startup, cut into fixed slabs of one page record (the 65 plane
  slices of one page, 4 KiB aligned). A KV page is one slab; a state image is
  `ceil(image / slab)` slabs. Slabs never fragment.
- Its size is part of the startup RAM ledger (`plan_host_memory`): pinned with lock overhead,
  beside the experts. Default 4 GiB (Infernix's Flash-Next default), `--host-context-mib N`
  overrides, `--host-context-mib 0` turns the cache off. If the ledger cannot fit the request it
  shrinks to what fits above the reserve, and says so; below 512 MiB the cache is off.
  `--auto-host-cache` is refused for this model (the ledger sizes the tier).
- The RAM tier copies with `gather_copies` (a kernel reading or writing pinned Host memory, one
  launch per restore or capture), so the single WDDM copy engine stays free for the round's inputs
  and the expert stream.

## Storage tiers

Capture and restore never name the memory a checkpoint lives in. They describe their copies as
`SlabCopy` entries (a slab index and offset, a device region, rows and pitches) and hand them to a
`CheckpointTier` (`program/context_cache.h`); `HostCheckpointStore` keeps the slab bookkeeping
(records, shared pages, release quotes) independent of the tier. `HostTier` (pinned RAM, gather
kernel) is the only tier built.

Order of the tiers, each a separate step:

1. **RAM** (this change). Sized by the RAM ledger beside the experts.
2. **Persistence on NVMe through the Engine's context store** (`--context-store`): implement
   `export_checkpoints` / `checkpoint_image_host_bytes` / `import_checkpoints` for this Program, with
   the Qwen3.5 Program's image container (magic and version, a configuration section, KV pages once
   in first-reached order so successive images of a conversation share bytes in the
   content-addressed store, the state images, exact identities, a point table, a trailing checksum)
   and a Flash-Next configuration section (page record and state image layout, drafter on or off,
   vision). The Engine's store plumbing (idle and shutdown writes, start-up and on-demand hydration,
   S3 copies) is then reused unchanged. It imports into the RAM tier, so it needs a nonzero one.
3. **An NVMe tier for little or no RAM**: a `CheckpointTier` whose slabs are offsets in a file on
   local NVMe, moved through a small pinned bounce buffer (tens to a few hundred MiB from the
   ledger). A 32K-token checkpoint (567 MiB) then restores in roughly 0.2-0.3 s at NVMe speed
   instead of 13 s of prefill. Its byte layout is the same slab layout, so step 2's images can be
   written from it without a second format.

**A 0 GiB (or too small) Host budget** degrades, deliberately, to *prefix reuse off*: startup does
not refuse; one line says the ledger had no room (or that `--host-context-mib 0` asked for none),
requests prefill their whole prompts, and everything else runs as before. With step 3 the same case
becomes disk-only reuse.

## Sharing

Full KV pages are immutable once every layer's 64 positions are final (main KV below
`state_tokens`, drafter cells below `mtp_cells`), so checkpoints of one history share them by
reference count:

- The lane keeps a *mirror*: the Host page of each leading full page its device KV equals (after a
  restore, the source's pages; after a capture, the pages it copied).
- A capture copies only pages past the mirror, plus the partial tail page, which belongs to that
  checkpoint alone. Repeated turns of one conversation therefore store each page once.
- Host bytes freed by deleting a set of checkpoints are the slabs no other holder references
  (`host_bytes_released`), which is what the ResourceManager's eviction quotes.

## Contract mapping (mirrors the Qwen3.5 Program)

| Hook | Flash-Next |
|---|---|
| `plan_request` | capture groups exactly as Qwen3.5: InputReplay at the prompt's recovery frontier (or its end), LongAnchor and SharedPrefix at the frontend's opportunities; vision items are never split; `publish_continuation` when the cache is on and the prompt is reusable |
| `inspect_source` | exact match (digests, then tokens, positions and media); remaining work from F; one Host-to-Device transfer requirement of the restored bytes; carried private points and retirements as Qwen3.5 |
| `start_binding` / `poll_context` | backs the request's extent, pins the source, restores pages and state, sets `state_tokens = F`, the mirror and the drafter's position (below); a consumed source is released after the restore |
| prefill | a call stops at the next capture frontier; reaching a mid-prompt one returns `capture_ready`; the prompt-end group is offered at the Begin commit |
| `prepare_capture` / `start_capture` / `skip_capture` | reserves slabs (the Engine's capture admission evicts for it), copies, publishes the checkpoint in the next `poll_context` |
| `finish` | the endpoint checkpoint E at `state_tokens` (Continuation role), when the Host tier has room |
| `plan_releases`, `plan_reclaim` | one release action per allowed checkpoint (Host bytes only); no demotion (nothing is on the device) |
| `physical_usage` | Host capacity, occupancy and peak; one state slot, the lane's KV pages |
| `checkpoint_recovery_loss`, `checkpoint_recovery_frontier` | Qwen3.5's lineage arithmetic over Flash-Next identities |

Pausing stays refused (one lane reserves its whole extent at admission), so there is no Snapshot or
Replay. Keys, handles and the mirror are per lane, so more lanes add no new ownership rule.

## The MTP drafter

Following Infernix's rules (Flash-Next design §19.3.1 MR1-MR6): cell c pairs `residual(c)` with
token `c + 1`; at a settled point `mtp_cells` is `F` or `F - 1` and the saved residual is
`residual(F - 1)`. A checkpoint records `mtp_cells` and `history[F]`. A resume at F keeps
`mtp_cells = F` only when the new prompt's token at F equals the recorded one; otherwise
`mtp_cells = F - 1`, the first call prepends cell F - 1 with the right token, and the mirror stops
before the page holding F - 1 (that page is rewritten).

## Exactness

A resumed request's greedy ids must equal the same request run cold. Two effects can break it and
are measured, not assumed:

1. Prefill call boundaries. Cold prefill runs 4,096-token calls from 0; a resume runs them from F.
   Whether the forward is invariant to where calls start (the GDN route is chunked) decides whether
   capture frontiers must be rounded to a page (stage 0 probe below).
2. The endpoint E carries decode-computed state (one-column and verification rounds); a later
   prompt resumed from E is computed differently from a cold prefill of the same tokens. This is
   inherent to continuing from a generated turn (Qwen3.5 has the same property) and is reported.

An exact repeat of a prompt whose checkpoint is at the prompt end samples from the stored logits,
so it reproduces the cold first token and state exactly.

## Stages

0. Baseline build at master; probe call-boundary invariance (cold ids with prefill split at
   aligned and unaligned positions).
1. Host checkpoint registry (CPU only): slab pool, shared pages, footprint and release quotes,
   lineage loss; unit tests in `tests/models/qwen4_exp/`.
2. Program integration: plans, prefill splits, capture, endpoint, restore, reclaim hooks, RAM
   ledger and startup report, Engine wiring and option validation.
3. Real-model gates (GPU lock): resume == cold (repeat, extension), MTP on and off, image prefix,
   eviction under a tiny budget, cancel during capture, failure after capture, restart cycles.
4. Serving smoke and measurements (TTFT repeat / extend / agent rotation, capture cost, decode
   and cold prefill regression), ABBA against the same build with the cache off.
5. Defaults (launcher opt-out `NINFER_PREFIX_CACHE=off`), documentation.
6. `--context-store` only if 0-5 are green.
