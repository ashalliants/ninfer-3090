# Qwen3.8-Flash-Next context cache

The design authority for the Flash-Next context cache, beside [the model reference](qwen4-exp-model.md).

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

A resumed request's greedy ids must equal the same request run cold.

- **Where prefill calls start changes the numerics.** On master 04901788, the same 300-token prompt
  gives four different 48-token greedy outputs with `--prefill-chunk` 1024, 100, 64 and 96. So the
  capture groups' frontiers inside a prompt split its prefill calls *in every configuration*, cache
  on or off (`BasePlanImpl::splits`, as Qwen3.5 splits at its rewrite frontiers): a prompt's output
  does not depend on the cache, and a resume at one of those frontiers runs exactly the calls a cold
  run of the longer prompt runs. Measured before this rule: a 28K-token chat prompt generated 18
  tokens with the cache on and 25 with it off (both greedy, different call partitions).
- **Short calls take the static route** (fewer than 512 columns, even while the wide route's lease is
  held): the wide route streams every non-resident expert of each layer, about 0.8 s per call,
  which made a split cost as much as a 4K call. Greedy ids of a 4,101-token prompt (a 5-token tail
  call) are identical to master's.
- **Groups below 512 tokens are not planned** (a 111 MiB image and a split for less than 0.2 s of
  prefill).
- **Exact repeats** sample from the stored logits of the prompt's last position: same first token,
  same state.
- **Endpoints** carry decode-computed state (one-column and verification rounds); a later prompt
  resumed from one is computed differently from a cold prefill of the same tokens, inherent to
  continuing a generated turn (Qwen3.5 has the same property). In the real test it matched the cold
  run for 32 tokens (with and without MTP); it is reported, not asserted.

The real tests (`ninfer_qwen4_exp_engine_real_test --cache [--mtp 3]`, `--cache --evict`,
`ninfer_qwen4_exp_vision_engine_real_test --cache`) assert ids equal to cold for the exact hit, the
resume at a call boundary, the cold runs with captures, an image in the cached prefix, eviction
under a 512 MiB tier, and a request after a cancelled one.

## Measurements

RTX 3090, Windows 11, CUDA 12.8, `qwen4exp-iq2xs-mtp-vision.ninfer` with the launcher's flags
(`--spec mtp --draft-tokens 3 --vision`, 32K context, n-gram volume on NVMe), 2026-10-13. Each server
starts with an empty expert cache (`--no-expert-state`); arms in ABBA order in one sitting; "off" is
the same build with `--host-context-mib 0`; one request per cell per arm. The chat requests are
`temperature 0`, thinking off.

| Request (TTFT) | off | on |
|---|---:|---:|
| 28,011-token document, cold | 11.72 s (other off arm: 18.03 s, an outlier) | 11.71 / 11.70 s |
| the same request again | 11.28 s (17.49) | 0.102 / 0.101 s, 28,002 reused |
| + its reply + a ~1K-token user turn (29,036 tokens) | 11.91 s (13.32) | 0.953 / 0.966 s, 28,028 reused |

Against master in an earlier sitting (master, off, on, on, off, master; before the splits were made
unconditional, so "off" did not split there): cold 11.32 / 11.47 s master, 11.38 / 11.45 off,
11.62 / 11.61 on.

Agent rotation (`tools/bench/run_agent_rotation.py --agents 3 --turn-words 1500 --target-tokens
22000 --edit-every 0`, 27 requests per arm, prompts 2.5K-22.7K tokens, same earlier sitting): mean
TTFT of the 24 continuing turns 5.82 / 5.84 s off and 1.20 / 1.21 s on (82% of their prompt tokens
reused); the 3 first turns 1.10 / 1.10 s off and 1.17 / 1.17 s on (the split, below).

Decode, 512 greedy tokens after a 27-token prompt: six requests per arm, the five after the first
102.4 / 102.2 tok/s median off and 102.2 / 102.4 on (ranges 101.7-102.6 and 101.0-102.7); in the earlier sitting master 72.1 / 72.1,
off 71.9 / 71.7, on 74.1 / 73.6 (one request each, a cold expert cache). One sitting measured
on 62.0 / 64.8 against off 71.5 / 72.9 on that cell (and 0.24 s against 0.18 s of its prefill);
it did not repeat in the next two sittings and is reported, not explained.

Restore: 500 MiB (a 28K-token checkpoint) in 20.5-21.2 ms. Capture: 21-22 ms for the prompt's
checkpoint (every page), 5-7 ms for a later one (its image and new pages). The first capture of a
process measured 650-680 ms in the real test: it waits for the prefill's expert refill already
queued on the stream (decode would wait for it too).

Costs: chat prompts split prefill at the typed recovery frontier (just before the assistant
header), a few-token static call: +0.07 s on a 2.5K-token first turn (+6%) and +0.2-0.3 s on
28K (+2%), in every configuration. The 4 GiB Host tier pins in about 1 s at startup.

## Status

Done: the RAM tier, capture/restore/reclaim hooks, the launcher default (`NINFER_PREFIX_CACHE=off`
opts out), real-model gates. Not built: `--context-store` and the NVMe tier (see Storage tiers),
more than one lane (the handles, keys and mirror are per lane already).
