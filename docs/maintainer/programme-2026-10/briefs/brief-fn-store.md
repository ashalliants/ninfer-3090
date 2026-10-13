# Task E: Flash-Next persistence (`--context-store`) and an NVMe checkpoint tier

Read `brief-common.md` first (own worktree, GPU lock, one Flash-Next process machine-wide, PIDs
only, PRs against `ashalliants/ninfer-3090`).

## Where things are
PR #285 (`feat/qwen4exp-context-cache`, open, mergeable, CI green) added an in-memory prefix/KV
cache for Flash-Next: checkpoints in a pinned Host slab pool behind a narrow `CheckpointTier`
(`src/models/qwen4_exp/program/context_cache.h`), design note
`docs/maintainer/flash-next-context-cache.md` (read it all; its "Storage tiers" section is your
spec: steps 2 and 3). Measured there: 28K-token prompt 11.7 s cold, 0.10 s exact repeat, 0.95 s
next turn; restore of ~500 MiB checkpoint 21 ms from RAM; the Host budget defaults to 4 GiB and
**0 GiB turns the cache off**. Not built: `--context-store` (still refused in
`qwen4_exp_instance.cpp`) and any non-RAM tier.

Create your worktree **from that branch** (`git worktree add <path> -b feat/qwen4exp-context-store
origin/feat/qwen4exp-context-cache`) and open your PR with base `feat/qwen4exp-context-cache`
(one level of stacking is allowed; say in the PR that it must merge after #285, and retarget to
master if #285 has merged by the time you finish).

Another agent (Task D, branches `feat/context-disk-tier*`) is changing the shared engine's store
side: disk term in the cost model, disk-aware Host eviction with write-on-evict, persisted shared
prefixes. Keep your store-side changes model-agnostic and minimal; do not edit those engine
policies; rebase on master when its PRs land.

## Goal
1. **Persistence**: implement `export_checkpoints`, `checkpoint_image_host_bytes` and
   `import_checkpoints` for the Flash-Next Program using the Qwen3.5 Program's image container
   (see `src/models/qwen3_5/program/storage/checkpoint_image.cpp`: magic/version, configuration
   section, KV pages once in first-reached order so successive images share bytes in the
   content-addressed store, state images, exact identities, point table, trailing checksum) plus a
   Flash-Next configuration section (page record and state image layout, drafter on/off, vision).
   Remove the `--context-store` refusal for this model. The Engine's plumbing (idle and shutdown
   writes, start-up and on-demand hydration, S3 copies) is reused unchanged. Restart survival is the
   headline: after a restart a returning conversation resumes in ~1 s instead of re-prefilling.
2. **NVMe tier** (`CheckpointTier` implementation): slabs are offsets in a file on local NVMe
   (default beside the model or under a `--context-nvme-dir`-style flag following the existing
   store flag naming), moved through a small pinned bounce buffer taken from the RAM ledger (tens to
   a few hundred MiB). Use unbuffered/overlapped I/O so a 500 MiB restore approaches NVMe speed
   (Task M measures the drive: look for its script under `tools/bench/` and for the numbers in
   `docs/maintainer/cache-provider-baseline.md` if they exist; otherwise measure yourself). Policy:
   RAM tier first, spill/demote to NVMe when RAM is full or absent, so a 0 GiB Host budget becomes
   "disk-only reuse" instead of "off". SSD wear is NOT a concern (user decision): write eagerly.
3. Update `docs/serving.md`, the README Flash-Next section, and the design note.

## Gates
- Exactness: a request restored from the store/NVMe gives the same greedy ids as one restored
  from RAM, with and without MTP, with an image in the prefix.
- Restart: stop and start the server, send the same conversation: measure TTFT (cold re-prefill vs
  restored), including the RAM-budget-0 / NVMe-only case.
- Restore latency by tier (RAM / NVMe / store hydrate) at ~8K and ~30K tokens, n >= 5, ABBA, and
  decode/cold-prefill tok/s unchanged versus #285 in the same sitting.
- Failure paths: truncated/corrupted image, wrong artifact identity, store folder missing, disk
  full, kill -9 during a write (existing store tests show the pattern); each degrades to
  "cache miss", never a crash or wrong output.
- The full ctest once at the end under the lock (3 environment failures are known: jsonschema
  module x2, `ninfer_artifact_writer_interop_test`).

Report as in brief-common.md. Say what is estimated, what was not tested, and what is worse.
