# Qwen3.8-Flash-Next context cache

The design authority for the Flash-Next context cache, beside [the model reference](qwen4-exp-model.md).

The Qwen3.8-27B/35B Programs implement NInfer's context cache contract (capture frontiers, private
continuations, shared prefixes, Host tier, reclaim), driven by the shared `EngineCore` and
`ResourceManager` ([design](resource-scheduling-and-context-cache.md)). The Flash-Next Program
stubbed every hook, so each request prefilled its whole prompt. This change implements the same
contract in the Flash-Next Program. The Engine, ResourceManager and Scheduler are unchanged.

## What a checkpoint is

A Flash-Next checkpoint is the lane's complete execution position at one token frontier F, held
in pinned Host memory or in the NVMe tier's file (Storage tiers):

| Part | Size (INT8 KV, MTP on) | Source |
|---|---|---|
| KV pages `[0, ceil(F/64))`: K/V codes, scales, pooled index keys of the 12 QSA layers and the MTP drafter's layer | 931,840 B per 64-token page (14,560 B/token) | the lane's elastic KV pool, plane-major |
| State image: 36 GDN conv + recurrent states, PLE convolution history and slot word, 12 QSA index-key tails, the drafter's tail | ~110 MiB | `state_backing_`, one contiguous block |
| The drafter's saved residual (`residual(F - 1)`) | 20 KiB | speculation buffer |
| BF16 logits of position F - 1, when F is the end of a prefill call | ~0.5 MiB | the last call's logits |
| Identity: tokens `[0, F)`, three-axis positions, media items, rolling digests; `mtp_cells`, `history[F]` | host vectors | the request |

Nothing lives in VRAM: the expert cache keeps every frame it had. A restore copies Host (or disk,
through Host) to the lane's own KV pool and state, so the source checkpoint is never written.

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
(records, shared pages, release quotes) independent of the tier, and tells the tier when a slab's
last reference goes. A tier also moves whole slabs to and from host memory (`read`, `write`), which
is what checkpoint images use. Two tiers are built:

1. **RAM** (`HostTier`): one pinned allocation of the Host budget, copied by the gather kernel. This
   is the tier when no NVMe tier is configured.
2. **NVMe** (`NvmeTier`, `program/nvme_tier.h`, `--context-nvme-mib N`): every slab is an offset in a
   scratch file (`core/scratch_file.h`: unbuffered overlapped I/O, sparse, created at startup in
   `--context-nvme-dir` or beside the model, deleted on close and by Windows when the process dies).
   The Host budget becomes its RAM cache of *frames*:
   - a capture gathers into frames and a writer thread writes them to the file at once
     (write-through; SSD wear is not a concern). A frame is evictable as soon as its write lands, so
     RAM never holds the only copy for long; the least recently used clean frame goes first;
   - a capture larger than half the frames (or with no frames) gathers into a pinned bounce buffer
     (128 MiB from the RAM ledger, two halves) and writes the file directly, one half's write
     overlapping the next half's gather;
   - a restore gathers the slabs RAM holds from their frames and reads the rest through the bounce
     halves (8 requests of 4 MiB in flight), one half's reads overlapping the other's copy to the
     device;
   - images import through the file and keep what fits in frames (a hydrated session then restores
     from RAM).
   The Engine sees the file's size as the Host capacity (`host_bytes`), so the ResourceManager evicts
   only when the file is full. The file is capped at half the volume's free space at startup.

**A 0 GiB (or too small) Host budget** without an NVMe tier degrades, deliberately, to *prefix
reuse off*: startup does not refuse; one line says the ledger had no room (or that
`--host-context-mib 0` asked for none), requests prefill their whole prompts, and everything else
runs as before. With an NVMe tier the same case is *disk-only reuse*: no frames, every capture and
restore goes through the bounce buffer.

**Failures degrade to misses.** A capture whose bytes the tier cannot store (an I/O error, a full
disk) is skipped (`start_capture` returns false; the Engine skips the group). A frame whose write
failed stays resident (`stuck`) with the only good copy until its slab is freed. A restore the tier
cannot read leaves the lane at the root: the request prefills its whole prompt (its begin summary
reports 0 reused tokens). An import that cannot be stored returns nullopt (a miss) and keeps nothing.

## Persistence (`--context-store`)

`export_checkpoints`, `checkpoint_image_host_bytes` and `import_checkpoints`
(`program/checkpoint_image.h`) write and read the Qwen3.5 Program's image container
(`qwen3_5/program/storage/checkpoint_image_codec.h`, shared by both Programs): magic, a version of
its own (`0x464E0001`, so neither Program's parser accepts the other's image), the model binding,
then a Flash-Next configuration section (the five plane sizes per attention layer, the state block,
the drafter's residual and the logits sizes, vision, the identity tag), the session, each exact
identity once, a point table (role, frontier, full pages, the drafter's cells and next token, the
logits flag, rope delta, page indices, key), then the payload: every KV page once in the order the
points (deepest first) first reach them, every state image once, and a trailing checksum. Pages and
state images are stored *compact* (planes back to back, none of a slab's alignment padding; logits
never captured as zeros), so an image does not depend on where its slabs were and successive images
of one conversation share their leading pages byte for byte in the content-addressed store. A
shared prefix is exported alone, without a session (the rule the shared engine uses).

Import checks the binding, the configuration and the checksum before it takes a slab, then the
table bounds and each point's key against its identity, and the drafter's position (cells at the
frontier or one behind; none without the drafter); any mismatch is `std::invalid_argument`, which
the Engine counts as a foreign or damaged image. It imports into whichever tier is configured, so
with an NVMe tier a 0 GiB Host budget still restores stored sessions.

The Engine's store plumbing is reused: idle and shutdown writes, start-up restore, request-time
hydration and S3 copies (`engine.cpp` now attaches the store to either generation core).

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
- A capture's slabs are owned by a `SlabRefs` from the moment they leave the lane's reservation
  until a published record holds them, and a capture group is inserted all or nothing
  (`insert_group`), so a failed capture (prompt or endpoint) never strands Host capacity.

## Contract mapping (mirrors the Qwen3.5 Program)

| Hook | Flash-Next |
|---|---|
| `plan_request` | capture groups exactly as Qwen3.5: InputReplay at the prompt's recovery frontier (or its end), LongAnchor and SharedPrefix at the frontend's opportunities; vision items are never split; `publish_continuation` when the cache is on and the prompt is reusable |
| `inspect_source` | exact match (digests, then tokens, positions and media); remaining work from F, including the images it encodes again (any image with a token at or after F); one Host-to-Device transfer requirement of the restored bytes; carried private points and retirements as Qwen3.5 |
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
  tokens with the cache on and 25 with it off (both greedy, different call partitions). Calls also
  end at the typed rewrite's *execution frontiers* (`rewrite_execution_frontiers`, as the Qwen3.5
  Program's prefill does), which the prefix identity already compares: a resumed request and its
  cold run then decompose the recurrent state at an earlier turn's rewrite identically.
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

### NVMe tier and `--context-store` (2026-10-13, same machine and flags)

Server sessions, one sitting, ABBA in the order master, RAM tier, RAM + NVMe tier, NVMe only | NVMe
only, RAM + NVMe, RAM, master (master is the build before this branch; `--context-nvme-mib 16384`;
`tools/bench/flash_next_store/bench.py`). A 24,144-token and a 7,283-token document; the repeat is an
exact hit, so its TTFT is the restore plus the fixed ~0.08 s of a one-token request.

| | master | RAM | RAM + NVMe | NVMe only (2nd arm) |
|---|---:|---:|---:|---:|
| cold 24K TTFT, s | 10.63 / 10.61 | 10.64 / 10.33 | 10.76 / 10.33 | 10.55 |
| cold 7K TTFT, s | 3.33 / 3.32 | 3.33 / 3.24 | 3.38 / 3.26 | 3.31 |
| repeat 24K TTFT, median of 5, s | 0.118 / 0.123 | 0.118 / 0.118 | 0.108 / 0.124 | 0.198 |
| repeat 7K TTFT, median of 5, s | 0.097 / 0.075 | 0.082 / 0.092 | 0.076 / 0.078 | 0.127 |
| decode, 512 tokens, median of 3, tok/s | 92.7 / 93.8 | 93.9 / 96.5 | 93.7 / 96.5 | 94.8 |

The first NVMe-only arm ran while something else used the card (cold 24K 14.5 s, decode 80.7 tok/s,
repeat 0.253 s) and is not used; it is in the raw data. The tier costs nothing measurable on cold
prefill or decode. With RAM frames in front, a hit is served from RAM exactly as before; with no RAM
tier a 24K-token restore costs about 80 ms more (7K: 45 ms) than from RAM. Micro-benchmark
(`ninfer_qwen4_exp_nvme_tier_test --bench`): file only, 511 MiB restore 96.5 ms (5.6 GB/s), capture
263 ms; with RAM frames, restore 21.4 ms.

Restart (first conversation, server stopped with CTRL_BREAK, started again on the same store, n = 1
per cell, the repeats after it n = 3): the 24K-token request, cold 10.79 s, after the restart 0.211 s
with the RAM tier and 0.445 s with `--host-context-mib 0 --context-nvme-mib 16384`; the 7K-token
one 3.27 s cold, 0.142 s / 0.281 s after. Start-up imports one session (283-289 MiB image) in
0.17-0.77 s.

Not measured: hit rate or TTFT under a working set larger than the RAM tier (the eviction gate with
a 1 GiB tier proves correctness, not benefit), and a store on a different volume than the model.

## Status

Done: the RAM tier, capture/restore/reclaim hooks, the launcher default (`NINFER_PREFIX_CACHE=off`
opts out), real-model gates; the NVMe tier and `--context-store` (Storage tiers, Persistence).
The launcher keeps the NVMe tier opt-in (`NINFER_CONTEXT_NVME_MIB`): it costs nothing measurable, but
it only pays once the working set exceeds the RAM tier, which was not measured, and it writes a
scratch file beside the model (capped at half the volume's free space). Revisit with a hit-rate run.
Not built: more than one lane (the handles, keys and mirror are per lane already); retrying a failed
frame write (a frame whose write failed stays in RAM until its slab is freed).
