# Sharing a context store between engines on one box

Status: design for review, not built. This is a temporary plan: when the change lands, the stable
parts move into `docs/serving.md` ("Context store") and this file is deleted.

## Problem

A box with two GPUs runs two independent engines, one per GPU (measured 1.9x faster than one engine
split across both, because a split engine does not pipeline prefill). A gateway sends each turn of a
conversation to whichever engine is free, so turn N+1 often lands on the engine that did not serve
turn N and prefills the whole conversation again (about 1 minute at 72k tokens).

Engines can share through S3 today (`--context-store-s3-*`), but the bucket is listed every 60 s and
the image is uploaded and downloaded in full: 35 s at best, 1-2 minutes typically. The goal is that
engine B reuses what engine A finished within seconds, through the local disk.

## Why one directory cannot simply be shared

`ContextStore` assumes it is the only process in its directory (`context_store.h`: "Serializes
everything that changes files"; the mutexes are in-process). Pointing two engines at one `DIR` breaks
in ways that are silent:

1. **Start-up scan deletes the peer's files.** `scan()` empties `tmp/` (the peer's in-flight writes)
   and deletes every chunk no manifest *this process read* names (the peer's chunks written since).
2. **Refcounts are per process.** A removes its manifest, drops the last reference it knows of and
   deletes a chunk that B's manifest, written after A's scan, also names. B's image then fails its
   read and is removed as corrupt (a miss, but the image is gone).
3. **The index is read once.** A running engine never re-reads `manifests/`, so it never learns of
   the peer's images.
4. **Byte accounting double counts or misses** what the other process wrote, so `max_bytes` and
   eviction act on a wrong total.

Every one of these ends in a miss or a lost image, never in wrong output, because every chunk is
verified by hash when read and a manifest carries its own checksum. The failure that matters is that
sharing does not work at all, and that the lost images are not noticed.

## Considered

| Option | Verdict |
|---|---|
| **One shared chunk pool with a cross-process lock** (flock / lock files around put, load, scan, GC) | Needs refcounts that survive process death (a crashed holder leaks references or, worse, drops them), or a mark-and-sweep GC that stops both engines while it runs. A lock held across a 1.4 GB write stalls the peer's request path. Large, and its failure modes are the dangerous kind. |
| **Lease-based GC** (chunks carry an expiry, readers refresh it) | Chunk deletion becomes time-based, so a slow reader can lose a chunk it is reading. Wrong trade for a store whose bytes are the product. |
| **Per-engine directories, S3 between them** (today) | Works, but is 35 s to 2 min, costs egress and upload bandwidth, and is the thing being replaced for the same-box case. Stays as the cross-box path. |
| **Per-engine directories, local peers imported by hard link** | **Chosen.** No shared mutable state and no cross-process locks. |

## Design: owned directories, read-only peers

**Each directory has exactly one writer.** Nothing about the store's own logic changes: its
refcounts, GC, scan and accounting stay correct because the premise (one process) becomes true and
enforced. A new option lets an engine also *read* other engines' directories:

```
ninfer-serve --context-store D0 --context-store-peer D1 ...      # engine on GPU 0
ninfer-serve --context-store D1 --context-store-peer D0 ...      # engine on GPU 1
```

### 1. Owner lock (new, small)

At construction the store takes an exclusive, non-blocking lock on `DIR/owner.lock` and keeps it for
its lifetime (`LockFileEx` / a share-none `CreateFile` on Windows, `flock(LOCK_EX|LOCK_NB)` on
Linux). The OS drops it when the process exits, so a crash cannot leave it stuck. A second owner
fails at start-up with "context store DIR is in use by another process", instead of corrupting the
directory. Today two engines on one directory is exactly the failure above; this turns it into an
immediate error. Peers never take or read the lock.

### 2. Importing a peer's image

A background thread (`peer_thread_`, started after construction, interval `--context-store-peer-poll-ms`,
default 250) lists `PEER/manifests/*.manifest`. Listing is names only; an id already in the index, or
in the skip memo below, costs one hash lookup. For a new id:

1. Read the manifest and `parse_manifest()` it (checksum, structure, `info.id == name`). A bad file
   is skipped and remembered.
2. For every chunk it names, **hard-link** `PEER/chunks/xx/<hash>.chunk` to our own
   `chunks/xx/<hash>.chunk`, taking `io_mutex_` so our own GC cannot remove a same-named chunk
   between our existence check and the link. A chunk we already hold is reused. A hard link is
   independent of the source from then on: the peer deleting, evicting or superseding the image does
   not touch our copy, and our deleting ours does not touch theirs. No bytes are copied and the disk
   holds the data once.
3. If any link fails because the source is gone (the peer evicted it in the meantime), remove the
   links this import created, leave the index alone and remember `(id, manifest mtime)`. That is a
   miss. A manifest the peer rewrites later has a new mtime and is tried again.
4. Otherwise write our own copy of the manifest atomically (so our restart scan sees it),
   `add_references()`, and register the entry, exactly as `refresh_remote()` registers an image
   from the bucket. From here the image is an ordinary local image: `list()` offers it, the engine's
   cost model prices it with the measured read rate, `load()` reads and verifies every chunk, and
   eviction, TTL and `max_bytes` treat it like any other. **`engine_core.h` does not change**:
   `start_hydration()` already asks `list()` on admission.

Nothing is imported that the store would immediately evict: an image older than the TTL, or one that
does not fit in `max_bytes` after counting only its chunks we do not already hold, is skipped and
remembered (otherwise an over-budget directory would link, evict and re-link every poll).

Importing happens off the engine's worker thread and holds `io_mutex_` for a few milliseconds per
image (a 1.4 GB image is about 45 links). `put()` and `load()` wait that long at most.

### 3. When an image becomes visible

An image is visible to the peer once the owner has *written* it, and the engine writes in three
cases (idle for `--context-store-idle-seconds`, default 30 s; on eviction; at shutdown). So **with
the default 30 s, a peer cannot see a turn before it is 30 s old**, whatever the import latency. A
gateway that sends the next turn to the other engine within 30 s still prefills cold. Use
`--context-store-idle-seconds 1` or `2` on shared boxes. The cost is more writes (only the changed
chunks, since consecutive images of a conversation share most of theirs; measured in the PR) and the
rule that background writes are skipped while a request is waiting stays, so a saturated engine still
writes late. Whether that is a problem in the replay is part of the measurement, not assumed.

### 4. Failure modes

| Event | Result |
|---|---|
| Peer evicts / supersedes / erases an image while we import it | Link fails: skipped, a miss. Or the links succeeded first: we own an intact copy. |
| Peer evicts it after we imported | Nothing: our hard links keep the bytes. |
| Peer crashes mid-write | Its `tmp/` leftovers are not in `manifests/` and are never read. Its manifest is renamed last, so a visible manifest names chunks that were already renamed into place. |
| Peer restarts and its scan removes orphan chunks | Only its own links; ours are separate directory entries. |
| Both engines write the same endpoint (same id) | Each writes its own directory. Ids are the endpoint key hash, so both are valid images of the same state. |
| Peer manifest is damaged or from another binding | Checksum rejects the first; the engine's existing binding check in `start_hydration()` ignores the second. |
| A chunk is damaged on disk | Unchanged: `load()` verifies every chunk by hash; with no remote it is removed and is a miss. |
| Peer is on another volume | Hard links fail: the image is never imported (a miss), counted in a metric, documented. **No copy fallback** (a copy of a 1.4 GB image under `io_mutex_` would stall the engine's own writes). |
| Both directories configured as each other's peer (the normal case) | Each imports the other's images; ids it already holds are skipped, so nothing ping-pongs. |
| Windows: peer's `remove` of a manifest we have open | Our read is a few milliseconds; if their remove fails it leaves a stale-but-valid manifest that their next start-up scan adopts. Harmless, and it can only produce an extra cache entry. |

The guarantee stays "extra misses, never wrong output": an imported image is read through the same
`load()` that verifies every chunk and the manifest checksum, and the engine validates the image's own
binding and trailer on restore.

### 5. Why not look up by prompt hash on a miss

The id is `endpoint digest + frontier`, but the engine cannot enumerate frontiers it does not know,
and the poll already makes the image visible within one interval. A synchronous probe on the
admission path would put directory I/O and manifest parsing on the engine worker for the benefit of
a few hundred milliseconds. Revisit only if measurement shows the poll interval matters.

## Interface

- `ContextStoreOptions::peers` (`std::vector<std::filesystem::path>`), `peer_poll` (default 250 ms).
- `ninfer-serve --context-store-peer DIR` (repeatable), `--context-store-peer-poll-ms N`. Requires
  `--context-store`.
- Stats and metrics: `peer_images` (imported), `peer_import_bytes_linked`, `peer_import_skipped`
  (gone, damaged, over budget), `peer_link_failures`.
- No change to `ContextStore::put/load/list/find/erase` signatures.

## Tests (CPU-only, `ninfer_context_store_test`)

- Owner lock: a second store on the same directory throws; after the first is destroyed it opens.
- Import: A puts, B (peer A) imports within a poll interval, `load()` returns identical bytes, the
  chunk files share storage (same file index), no bytes copied.
- A evicts or erases the image after B imported it: B still loads it.
- A erases the image while B imports it (loop, many iterations): B ends with a full image or none,
  never a partial one.
- Both engines put the same id concurrently: both load a valid image.
- B restarts: its scan keeps imported images (it has its own manifest copy) and A's files are untouched.
- A restarts and scans while B imports: B's imports are unaffected.
- Damaged peer manifest and damaged peer chunk: skipped or a miss on load; the peer's files are not
  modified or removed.
- Over-budget and expired images are skipped and not retried every poll.
- Different volume: skipped and counted (only if a second volume is available in the test host; else
  covered by forcing the link to fail).

## Measurement plan

On the one 3090, two `ninfer-serve` processes with a small config if VRAM allows (otherwise one
engine run twice, restarted between turns, which exercises the same import and read path but not
concurrent writers; stated as such). Engine A serves a ~70k-token conversation; engine B receives
the next turn. Time to first token: shared store vs a cold prefill vs the S3 path (local MinIO or the
`MemoryObjectStore` equivalent is not a fair S3 stand-in, so the S3 number is stated with its
conditions or not claimed).

## Not in this change

- Sharing through S3 gets no faster (60 s listing, full download).
- No gateway-facing API (prefetch, pin, flush). With peers, "flush" is `--context-store-idle-seconds`.
- No cross-volume copy, and no more than local directories as peers.
