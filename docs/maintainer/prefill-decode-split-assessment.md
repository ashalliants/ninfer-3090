# Assessment: a prefill engine plus a decode engine on a 2-GPU box

Status: analysis only, nothing built. A temporary note: delete it when the question is settled.

Question: would one engine that only prefills and one that only decodes (one per GPU) beat two
general engines (one per GPU) on long agent conversations (median prompt 72.5k tokens, 63k cached,
median completion 361 tokens)? What would have to change?

Evidence is from the code at `63bab74fc` unless marked **(brief)**, which means a number taken from
the request that commissioned this note and not measured here, or **(inferred)**.

## What the engine has and lacks

| Needed for a split | State today | Evidence |
|---|---|---|
| A prefill-only request | **Missing as prefill-only.** `Engine::submit` accepts `requested_output_tokens == 0`, but only as an immediate result: it returns `OutputLimit` at once, with no prefill and nothing retained. So a zero-token request does not warm the cache. The smallest request that prefills and retains the conversation is one output token, which costs one decode step. | `src/runtime/engine/engine.cpp:846` |
| "Resume from this stored image" request | **Missing as a request, present as a mechanism.** Hydration is not requested: on admission the engine matches the prompt's own prefix digests against the store (`start_hydration`) and reads the deepest image that is cheaper than prefilling. A decode engine that receives the full prompt after the prefill engine has stored its image would hydrate it with no new request type. | `engine_core.h:3113-3169`, `engine.cpp:552-600` |
| Image export and import between engines | **Present through S3 only in this checkout; the directory path is external.** Stored images are byte images of KV pages and recurrent states; the Host tier import is what hydration uses. This checkout exchanges them through S3 (35 s to 2 min). Exchange through directories (`--context-store-peer`) is PR #329, not merged here: its measurement (a second engine resumed a 46-55k-token conversation in 1.8-2.7 s against 20-28 s cold) comes from that PR and is not reproducible from this tree. | `docs/serving.md` (S3 options; no peer section yet) |
| Writing the image right after prefill | **Missing, and starved exactly when it matters.** Writes happen when a session has been idle for `--context-store-idle-seconds` (default 30), on eviction, and at shutdown. `persist_idle_session` returns without writing while any request is pending, prefilling, replaying or materializing, and writes at most one session a second. During a burst the prefill engine always has a request pending, so it would almost never write. | `engine_core.h:3027-3087` |
| A signal the gateway can wait on ("image is durable") | **Missing.** The gateway has nothing to call to prefetch, pin or flush (brief; consistent with `docs/serving.md` "There is nothing for a supervisor or gateway to call"). | |

## Cost of one hand-off

Sizes from the brief: about 18 KB/token of KV with `--kv-dtype rk4v4`, about 150 MB of recurrent
state per recovery point (about 80 MB with `--gdn-state-fp16`).

- Prefill engine to decode engine, for a 10k-token delta: the **new** KV is about 180 MB plus one
  recurrent state, about 260-330 MB. The older KV is already on the decode engine if the same
  decode engine serves the conversation again, because chunks are content-addressed and unchanged.
- Decode engine back to the prefill engine: the reply's tokens (a few hundred, about 6 MB) plus the
  endpoint state (80-150 MB), if the next turn's prefill is to run on the prefill engine with a
  warm cache. This is the leg that cannot be avoided: the reply has to reach the engine that
  prefills the next turn. (Alternative: let the decode engine prefill its own next delta, which is
  no longer a split.)
- Through a local store, the cost is the write plus the read at roughly 1 GB/s each (the cost model's
  starting rate for reads; the measured rates are in `docs/performance.md`), about 0.3-0.6 s each way for those sizes **(inferred)**, on
  top of the peer poll interval. That is small against 7-10 s of prefill, and large against the
  small turns of a quiet conversation. A direct device-to-device path does not exist: RTX 3090s
  have no NVLink here, so a P2P copy would stage through pinned host memory across two processes
  (shared memory or IPC handles), which is a larger build than the disk path.

## Does a dedicated decode GPU sit idle?

The brief says prefill takes about twice as long as decode on this traffic, so the decode GPU would
be about half idle. The medians in the brief do not support "twice": the median uncached prefill is
72.5k - 63k = about 9.5k tokens, 7-8 s at 1.2-1.4k tok/s, and the median completion of 361 tokens at
the 55 tok/s uncontended rate is about 6.6 s. By medians the ratio is about 1.1-1.3 **(inferred)**.
It would be nearer 2 only if the mean is dominated by cold starts (a cold 72k conversation is about
a minute of prefill). I cannot say which from the data I have; the request log has `prompt_tokens`,
`cached_tokens` and `completion_tokens` per request, so the ratio should be computed from that log
before this is decided. Whether a 1:1 split beats two general engines on throughput is **unmeasured**.
Total GPU work is the same, so no gain is guaranteed, and the split adds the transfers above. But
removing prefill/decode contention on each card could raise aggregate throughput even at equal
total work, and an unbalanced split would leave one GPU underused. The trade-off is scheduling
contention against transfer overhead and idle capacity; it needs a measurement on the request log
and a prototype before it is decided.

## What a split would fix

Streaming speed during bursts: 4-6 tok/s against about 55 uncontended **(brief)**. This is the real
prize, because prefill and decode on one engine compete for the GPU. It is also fixable more cheaply:
the engine's prefill chunk size (`--prefill-chunk`) and scheduling decide how much a prefill delays
decode rounds, and an earlier measurement on this card (not recorded in the repo) found that overlapping
one prefill with one decode gave no stall win and cost 11-32% of prefill, so the contention is real
and not removed by simple overlap. A split removes it by construction.

## What would have to change

1. **Engine, prefill-only:** make `max_tokens = 0` (or a flag) prefill, retain the closure and finish
   without a decode round. Today the zero-token path returns before any prefill. Small (admission path).
2. **Engine, write-through:** a request flag that queues the finished session for the store
   immediately and reports when it is durable, bypassing the idle gate and the "pending request"
   gate. Medium: the writer is a single thread with a bounded queue and an eviction-priority rule.
3. **Store, peer import:** PR #329. Without it nothing here works inside seconds.
4. **Engine, hydration on demand:** today a hydration is attempted once per request
   (`store_probed`) at admission, against a snapshot of the index. The decode engine must see the
   image before the request arrives, so the gateway has to wait for item 2's durability signal and
   the peer poll interval, or the engine needs a retry path. Small to medium.
5. **Gateway:** two-phase routing (prefill request, wait for durable, decode request), plus the
   return leg and failure fallbacks (a lost image is a full prefill on the decode engine, which is
   correct but slow). Medium; not in this repository.
6. **Metrics and capacity:** the decode engine needs a Host tier large enough to hold images it
   imports (`--host-context-mib`), and each image is up to 1.4 GB at 72k tokens.

Rough size **(inferred, not estimated by building anything)**: items 1 and 4 are days; item 2 is
about a week with its tests and a measurement; item 3 is the open design; item 5 is the gateway
team's. Total on the order of 2-4 weeks including measurement, with the result uncertain until the
real prefill:decode work ratio is known.

## Recommendation

Do not build it yet. First (a) land the peer store (#329; its measurements are of two general engines
sharing it, not of a split),
(b) compute the prefill:decode work ratio from the production request log, and (c) try the cheap
mitigation for burst decode speed (smaller prefill chunks, or capping prefill's share per round)
against the 4-6 tok/s symptom. Revisit the split only if (c) cannot reach acceptable streaming speed.
