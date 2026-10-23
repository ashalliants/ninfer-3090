# Flash-Next with several lanes on one RTX 3090: spike results

Spike, 2026-10-13, branch `spike/qwen4exp-lanes`. Question: can Qwen3.8-Flash-Next serve 2-4
requests at once (`--max-concurrency` 2-4) on one RTX 3090 without falling over, and what would the
user get for it? Today it serves one request at a time; `plan_layout` and the forward refuse more.

Nothing here changes the default path. The branch carries two pieces of throw-away instrumentation
(a route trace behind an environment variable, one startup log line), a CPU replay tool, an Op bench
mode and the scripts that produced the numbers (section 6).

## Verdict

| | Verdict | Why |
|---|---|---|
| **C = 2** | **GO-WITH-CONDITIONS** | It starts and fits. Predicted aggregate throughput is ~109 tok/s, against 92.6 tok/s for today's single MTP stream on the same requests (+18%). Each stream drops to ~55 tok/s, against ~93 alone. For a burst of 8 queued agent requests, the predicted gains are: last request done ~12% sooner, mean completion ~9% sooner, mean time to first token ~28% sooner. Worth building only if queued-agent latency matters more to the user than single-stream speed, and only in the form below. |
| **C > 2** | **NO-GO** | C = 3 and C = 4 add only +14% and +17% aggregate over C = 2 (124 and 127 tok/s). Each stream falls to 41 and 32 tok/s, the expert hit rate drops, and at 64K contexts C = 4 gives *less* than C = 3. Nearly all the gain comes from the second lane. |

**Conditions for C = 2:**

- Plain batched decode only when two rows decode, and MTP when one row decodes (Infernix's speculation gate). A lone request then keeps today's MTP speed, less ~1% for the smaller cache.
- MTP inside two-row rounds is not worth building first. It is predicted at ~102-106 tok/s, no better than plain C = 2.
- A prefill policy for the other lane (section 4).
- An acceptance gate measured on this workload: real C = 2 aggregate at least 1.15x MTP C = 1, and greedy ids equal to C = 1.

The step-time model behind these figures is an estimate (section 3). The replayed hit rates and every
C = 1 number are measured.

**What the user would get, in plain terms.** Today a second agent's request waits for the first to
finish. On these requests that wait is about 11-16 s each, so a burst of 8 waits 45 s on average
before its first token. With two lanes it starts within its own prefill time. Each answer then
streams at roughly half speed while the other lane is busy. Total work done per minute rises by
about a fifth. Answers do not get faster: a single request is never faster than today.

## 1. Resource cost per lane

Measured from the device plan at startup (the spike's `spike plan bytes` line) for the two
launcher configurations. Both use 32K context, `--kv-capacity 32768`, INT8 KV, the desktop attached
(1,243 MiB in use before loading) and the PCIe link measured at 26.8 GB/s:

| Allocation (bytes) | Plain | MTP (ceiling 3) + vision | Per lane? |
|---|---:|---:|---|
| KV at 32K (13 or 12 QSA layers, INT8-G64 + pooled index keys, 2 MiB plane alignment) | 480,247,808 | 520,093,696 | yes: its extent |
| Recurrent state: 36 GDN (conv BF16 61,440 + recurrent FP32 3,145,728 each), PLE conv (span 9), 12 QSA tails | 115,651,840 | 115,652,608 | yes |
| MTP verification records, tail copies, drafter residuals | 0 | 6,841,600 | yes |
| Workspace (sized by the 1,024-token prefill chunk) | 667,304,192 | 699,086,592 | no: one prefill at a time; a B-row decode needs far less |
| Miss staging (64 slots), residency tables and route log | 104,628,224 | 104,628,224 | no |
| Per-call inputs, logits, sampling | 8,499,088 | 13,085,840 | mostly no (B logits columns) |
| CUDA graph reserve | 256 MiB | 256 MiB | no: one graph pair per batch size fits the same 256 MiB floor (`internal_reserve`) |
| **One expert frame** | **1,510,400** | **1,510,400** | the IQ2_S record, the widest of the three formats |
| Expert frames at startup | 12,258 (17.24 GiB) | 11,614 (16.34 GiB) | |
| Free VRAM after startup | 358 MiB | 310 MiB | |

- The frame is 1.44 MiB, not Infernix's 2.64 MiB: that figure is Infernix's own expert recipe. So a
  lane costs more *frames* here than Infernix's "~130 frames at 16K" suggests.
- KV per token is 14,656 B plain and 15,872 B with MTP, both including the plane alignment. The raw
  figures are 13,440 B and 14,560 B, the latter Infernix's "14.6 KB".
- The recurrent state matches Infernix's "113 MB".

**Frames lost per lane** (KV extent + state [+ verification memory] over one frame):

| Context per lane | Plain | MTP |
|---|---:|---:|
| 32K | 395 frames (3.2%) | 426 (3.7%) |
| 64K | 687 (5.6%, estimated KV ~921 MB) | not computed |

| Expert frames | C = 1 | C = 2 | C = 3 | C = 4 |
|---|---:|---:|---:|---:|
| Plain, 32K each | 12,258 | 11,863 | 11,468 | 11,073 (-9.7%) |
| MTP, 32K each | 11,614 | 11,188 | 10,762 | 10,336 (-11%) |
| Plain, 64K each | 11,966 | 11,279 | 10,592 | 9,905 (-17%) |

These are arithmetic on measured startup sizes, not a C > 1 startup (C > 1 cannot start today).

**Does it start inside 24 GB with the desktop attached?** Yes, by construction. The fixed
allocations grow by ~116-123 MiB per lane. The KV comes out of the frames, of which there are over
11,000. The startup sizing then gives the frames whatever is left.

- Even 4 lanes at 64K take ~4.1 GB of KV and state, which leaves ~9,900 frames.
- The VRAM control law kept working at C = 1 in this session: it trimmed the plain run's frames from
  12,258 to 11,863 under desktop load.

**Host RAM** is not a per-lane problem. The pinned per-lane buffers (per-call inputs and the n-gram
row staging) are megabytes. That is an estimate from the IO layout, not measured. The 33.9 GiB of
pinned experts are shared.

RAM is already the binding constraint machine-wide, though. At 07:23 only 39.3 GiB was free (other
sessions' processes), and the C = 1 server refused to start: "0.6 GiB short". Every run in this spike
started with `--ram-headroom-mib 1200` instead of the default 2 GiB reserve. More lanes do not
change that.

## 2. Expert hit rate with interleaved streams (the key unknown)

**Method.**

1. A spike build appends every round's routes to a file (`NINFER_ROUTE_TRACE`, in
   `ExpertResidency::after_round`). The record holds every layer's top-10 ids per column, the
   promotion budget, and the Program's own hit and routed counts.
2. One cold `ninfer-serve` session (plain decode, `--no-expert-state`, 12,258 frames) served 8
   different conversations one after another. Each generated 1,024 greedy tokens, except the tool
   call, which stopped at 74:

   | Conversation | Prompt tokens |
   |---|---:|
   | Python code generation | 109 |
   | C++ code review with the file inline | 7,106 |
   | Literary story | 101 |
   | Tool-calling agent turn with tool results | 3,235 |
   | Operator summary of `docs/serving.md` | 10,478 |
   | Probability word problem | 185 |
   | Hotel SQL schema | 112 |
   | Chinese explanation of MoE | 99 |

3. `tools/flash_next/lanes_spike/lanes_replay.cpp` replays the traces through the Program's own
   policy object (`expert_cache::CacheController`: LFRU, frame pool, budgeted admission). It mirrors
   the Program's decode promotion budget (one per layer per token while filling, then every 4th
   token), prefill calls that promote nothing, and the give-back refill. A hit means READY in a frame
   at the round's start, as the Program counts it.

**Validation at C = 1.** The replay of the same session in the same order, with the recorded
budgets, reproduces the server's own decode hit rate per request to within 0.4 points. The only
exception is the 73-round tool turn, at 1.1 points:

| Request | Server's decode hit | Replay |
|---|---:|---:|
| code generation | 92.30% | 92.66% |
| code review | 94.57% | 94.70% |
| story | 91.92% | 91.99% |
| tool turn (73 rounds) | 73.68% | 74.74% |
| summary | 95.11% | 95.10% |
| math | 89.23% | 89.18% |
| SQL | 94.32% | 94.44% |
| Chinese | 92.36% | 92.32% |

These cold-session hit rates (89-95%) are below the 97-98% the plan records. Those runs were
warm-started and repeated the same prompts; here every request changes topic, starting from an empty
cache. Default serving warm-starts from the saved state, so absolute hit rates in use sit between
the two. The comparison across C below starts every C from the same cold state.

**Interleaving model.** C lanes serve the list once. Each lane takes the next unserved
conversation when its own ends, so no text repeats. A lane's prefill calls run as rounds of their
own (the others wait). Every decode round routes the union of the C lanes' next tokens per layer and
promotes for C tokens. Only rounds in which all C lanes decode are measured, after 200 warm-up
rounds. Each figure is the median of 8 rotations of the list.

| Scenario | C | Frames | Decode hit, median [min-max] | Distinct experts per layer | Misses per round | Misses per token | Layer calls with a miss / with >2 | Promotions per round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plain, 32K per lane | 1 | 12,258 | 93.33% [93.22-93.53] | 10.00 | 32.0 | 32.0 | 37% / 7% | 5.0 |
| | 2 | 11,863 | 93.25% [92.27-94.03] | 19.63 | 63.6 | 31.8 | 61% / 18% | 14.9 |
| | 3 | 11,468 | 91.95% [90.87-93.03] | 28.82 | 111.4 | 37.1 | 80% / 37% | 28.9 |
| | 4 | 11,073 | 89.82% [88.27-92.16] | 37.70 | 184.2 | 46.0 | 94% / 64% | 44.9 |
| same frames as C = 1 | 2 | 12,258 | 93.80% | 19.63 | 58.4 | 29.2 | 58% / 16% | 14.1 |
| | 3 | 12,258 | 93.31% | 28.82 | 92.7 | 30.9 | 75% / 30% | 26.9 |
| | 4 | 12,258 | 92.30% | 37.70 | 139.4 | 34.9 | 88% / 48% | 42.2 |
| plain, 64K per lane | 1 | 11,966 | 93.02% | 10.00 | 33.5 | 33.5 | 39% / 8% | 5.1 |
| | 2 | 11,279 | 92.33% | 19.63 | 72.3 | 36.2 | 65% / 21% | 15.9 |
| | 3 | 10,592 | 90.17% | 28.82 | 136.0 | 45.3 | 86% / 47% | 30.9 |
| | 4 | 9,905 | 86.85% [85.03-89.63] | 37.70 | 238.0 | 59.5 | 97% / 77% | 46.4 |

- **Unrelated streams barely share experts:** 19.6 distinct per layer at C = 2 against 2 x 10. That
  matches Infernix's 19.9 on an RTX 5090. Each further lane adds ~9 experts per layer to read
  (9.6, 9.2, 8.9).
- **The cache holds up at C = 2:** the hit rate per routed expert is unchanged, and misses per
  token are equal (31.8 against 32.0).
- **The cache degrades at C = 3-4:** misses per token rise 16% and 44% at 32K, and 35% and 78% at
  64K. The "same frames" rows show that about half of the C = 4 loss is the frames the lanes' KV
  takes, and half is interleaving.
- **The tool turn ran at 68-74% decode hit alone,** because its prompt's routes displace the decode
  set. A lane admitted beside decoding lanes will see that too.

**PCIe implied.** A miss is one 1,510,400 B record, 56 µs at 26.8 GB/s. Every miss in a round of
more than one column is staged: the zero-copy rule applies only at T = 1.

| | C = 1 | C = 2 | C = 3 | C = 4 |
|---|---:|---:|---:|---:|
| Misses, ms of link per round | 1.8 | 3.6 | 6.3 | 10.4 |
| Promotions, ms of copy engine per round (overlapped, same link and host DRAM) | 0.3 | 0.8 | 1.6 | 2.5 |

At C = 4 that is ~346 MB per ~31 ms round: ~41% of the link, on a DDR4-2133 host that also serves
the n-gram rows.

## 3. Step time, throughput and queue wait

**Measured at C = 1 on these 8 requests**, the same cold sessions without the trace, 7 full-length
requests each. Tok/s is the server's own decode figure:

| Configuration | Decode tok/s, median [min-max] | Makespan of the 8 requests in sequence |
|---|---:|---:|
| MTP, ceiling 3 (the launcher's default) | **92.6** [81.4-95.7]; 57-76% of drafts accepted | 97.7 s |
| Plain | 74.5 [70.2-77.2] | 112.7 s |

This workload is below the 109-113 tok/s MTP figure of the recorded runs. Those runs were
warm-cache with repeated prompts; here the sessions are cold, and every request is a new topic in
thinking mode.

**Round-time model for C plain lanes** (`tools/flash_next/lanes_spike/step_model.py`):

```
round(C) = a + n * (C - 1)                                   C = 1 fit; non-expert work per column
         + 48 * (3.67 us * (D_C - D_1) + 4.3 us * (C - 1))   distinct experts and columns per layer
         + b * M_1 + 55.8 us * (M_C - M_1)                   misses (staged when C > 1)
```

- **a = 11.82 ms, b = 43.6 µs per miss.** Least squares over the 7 measured plain requests, decode
  ms per token against the misses per round implied by their server-reported hit rate. Largest
  residual 0.16 ms.
- **3.67 µs per distinct expert, 4.3 µs per column, 55.8 µs per staged miss.** A pooled fit of the
  new Op bench mode `ninfer_offloaded_moe_bench --mode lanes`. It times one decode layer call of C
  columns, D distinct experts and m misses, for all three record formats, under a CUDA graph with
  cold L2 (median of 30; 114 points; RMS residual 24 µs). One layer call at C = 1, D = 10 costs
  47-50 µs; at C = 2, D = 20, 92-103 µs; at C = 4, D = 40, 163-185 µs.
- **n = 1.4 ms per extra column** for everything that is not routed experts: dense linears, HC
  mixers, GDN, QSA, norms, head. It comes from PR 13's nsys cost model: a verification column costs
  +3.4 ms, of which 2.0 ms is routed experts. The band uses n = 1.0-1.8 ms, plus up to 0.6 ms of
  exposed n-gram row read per extra column at the high end.
- **D_C and M_C** are the replay medians from section 2.

| Plain, 32K per lane | Round ms, mid [low-high] | Aggregate tok/s, mid [range] | Per stream tok/s | vs MTP C = 1 (92.6) |
|---|---:|---:|---:|---:|
| C = 1 | 13.2 | 76 (measured 74.5) | 76 | 0.82x |
| C = 2 | 18.3 [17.9-19.3] | **109** [104-112] | 54.7 | **1.18x** aggregate; 0.59x per stream |
| C = 3 | 24.2 [23.4-26.2] | 124 [115-128] | 41.4 | 1.34x; 0.45x |
| C = 4 | 31.4 [30.2-34.4] | 127 [116-132] | 31.8 | 1.37x; 0.34x |
| 64K per lane, C = 2 / 3 / 4 | 18.8 / 25.5 / 34.4 | 107 / 118 / **116** | 53 / 39 / 29 | |

**Cross-checks** (none is a C > 1 measurement on this card, which needs the feature itself):

- The model gives ~21.2 ms for a single-sequence verification of W = 3 on the replayed W = 3 routes
  (22.7 distinct experts and 78 misses per round). PR 13 measured 20.0 ms under nsys at a higher
  hit rate. This check is partly circular, since n comes from the same PR 13 cost model.
- Two Infernix figures on an RTX 5090, on different texts, imply a warm plain C = 2 round of ~1.25x
  its C = 1 round: 162.3 tok/s for the pair against ~102 tok/s plain alone at 8K. The model gives
  1.39x here, where misses cost more relative to the round (26.8 GB/s link, DDR4 host).

**MTP inside two-row rounds** (estimate). A proxy replays each lane's next 3 tokens per round and
advances 2, giving 43.7 distinct experts per layer and 181 misses per round. Adding two drafter
steps (0.88 ms each, unbatched) gives ~37.5-39 ms per ~4 tokens: ~102-106 tok/s, no better than
plain C = 2. At C = 3-4 the drafted rounds also exceed 8 columns, where dense GGML linears leave the
column-invariant route. Infernix reached the same conclusion and drafts only in one-row rounds.

**Queue wait** (`queue_sim.py`). An agent burst: all 8 requests queued at t = 0 and served FIFO.
Each lane's prefill runs alone, at the measured plain prefill time (0.35-4.47 s). Decode rounds use
the model's round times for however many lanes are decoding. The model's C = 1 plain run comes out
5% optimistic against the measured 112.7 s, so model rows are also given scaled by 112.7 / 107.0.

| | Makespan | Mean completion | Mean time to first token |
|---|---:|---:|---:|
| C = 1 MTP (measured) | 97.7 s | 56.1 s | 45.3 s |
| C = 1 plain (measured) | 112.7 s | 63.5 s | 50.9 s |
| C = 2 plain (model; scaled) | 81.5; ~86 s (**-12%** vs MTP) [80.3-84.7 unscaled] | 48.2; ~51 s (**-9%**) | 31.0; ~33 s (**-28%**) |
| C = 3 plain (model; scaled) | 74.9; ~79 s | 47.8; ~50 s | 26.1; ~27 s |
| C = 4 plain (model; scaled) | 68.9; ~73 s (-26%) [66.8-74.1] | 48.5; ~51 s (-9%) | 19.6; ~21 s (-54%) |

Two lanes do not make the first answer faster. The first full-length answer finishes at 22.5 s
under C = 2, against 13.4 s with MTP alone, because it shares rounds. Mean completion barely moves
past C = 2: the extra lanes trade one request's speed for another's start. The gain is time to
first token, which is what a queued agent waits for, and a modest throughput gain.

## 4. What would have to change, and what could break

The port runs one sequence end to end:

- `ForwardState` holds one sequence's GDN, PLE and QSA tail state.
- `ForwardKV` holds one identity block table over the whole pool.
- `ForwardBatch` holds consecutive positions of that sequence.
- The Program holds one `Lane`.

Infernix's original is multi-row throughout:

- its `ForwardBatch` carries `slots` and `table_rows`;
- its state tensors have a slot axis;
- its QSA Ops take `block_tables` and `tail_slots`.

The port narrowed each of those to one. The shared `EngineCore` already schedules multi-lane
Programs (Qwen3.5), so the scheduler side exists.

| Piece | Today | What C > 1 needs | Source |
|---|---|---|---|
| Lane table, phases, handles, epochs; `decode(rows)` | one `Lane`; lane 0 only; "decodes one row" | an array of lanes; B-row decode; per-row pending tokens and commits | **copy**: the Qwen3.5 Program's lane bookkeeping (`decode_ordinary_batch`) |
| Sampling | one `SamplingConfig`, one token-count array | per-row configs, penalties, grammar masks | **copy** (Qwen3.5) |
| `max_concurrency` refusal | `plan_layout`, `qwen4_exp_instance.cpp` | lift; plan per lane (`graph_bound(lanes, ...)` already takes lanes) | **copy** |
| Recurrent state | one sequence | a slot axis, zeroed at binding | **adapt**: Infernix `ForwardState` slots |
| GDN decode and its causal convolution | single-sequence calls | per-row state slots | **adapt**: NInfer already has the B-slot one-token `gated_delta_net` update Qwen3.5 uses; the tiled 16-to-48 key-head layout is this model's |
| PLE convolution and slot | one history | per-row history and slot | **adapt**: Infernix's `ple` Op with slots (NInfer's is single-sequence) |
| QSA index query, key pooling, block selection, attention, tail commit | single-sequence Ops (`ops/qsa.h`) | per-row block tables, tails, positions | **adapt** Infernix's batched QSA Ops as new NInfer Op variants, each qualified against an FP64 oracle per `op-development.md`. This is the largest single piece |
| KV pool | identity block table over the elastic VMM pool; whole extent backed at binding | per-lane page allocation; per-row block tables; growth at one lane's admission while others hold KV | **adapt** Infernix's lane-aware `grow_kv`; new for this Program |
| n-gram rows | one history; the next prefill call's rows read during the current one | each column hashes its own lane's last tokens; C row reads per round | **adapt** (host side) |
| MoE, HC mixer, dense GGML linears | take T columns already | nothing; routes are column-invariant up to 8 columns, so plain C <= 8 keeps C = 1's greedy output | none |
| Expert cache | promotion budget per advanced token | budget over C tokens (the replay used exactly this); leases for one lane's wide prefill while others decode | **new** policy decisions |
| CUDA graphs | front/back decode pair for B = 1; verification graphs per width | a decode pair per B | **adapt** |
| MTP drafter, verification records, catch-up | one lane | per-lane drafter tail, saved residual and records; draft only when one row decodes, drafter catch-up at W = 1 otherwise | **adapt** Infernix's speculation gate; after plain C = 2 works |
| Vision lease; frame lending for wide prefill | one lane's prompt | one lane's leases while others decode | **new** (Qwen3.5 has no expert cache to copy from) |

**What could explode:**

- **The `cuMemSetAccess` NOT_READY crash class and WDDM overcommit (PR 12).** Every admission
  grows the KV pool by first releasing frames. With lanes, that growth happens while other lanes
  hold live KV. It must still run only at a round boundary with nothing in flight, keep PR 12's
  take-first order, and never race the VRAM monitor's `cudaMemGetInfo`. A lane admitted while
  another lane's wide prefill holds a lease has fewer frames to take from (a lease is not
  resizable). This is the one failure mode with a crash history; it needs the 4K/32K/128K
  alternation test at C = 2.
- **OOM at startup.** Not a risk at C = 2-4 (section 1).
- **CUDA graph shapes.** One captured decode pair per B. With MTP in multi-row rounds, B x W > 8
  columns sends dense GGML linears to the MMA tiles, which reduce in another order: greedy output
  could then differ from C = 1 at near-ties (Infernix design 11.3). Plain C <= 8 is unaffected.
- **Expert-cache thrash.** Measured in section 2: none at C = 2. Real at C = 4 and 64K (86.9% hit,
  238 misses per round).
- **A lane's prefill stalls the others.**
  - A prompt of 4,096 tokens or more runs wide calls (~1.7 s each). Measured here: 3.10 s for the
    7,106-token prompt and 4.47 s for the 10,478-token one.
  - Each wide prompt lends 2,252 frames, 18% of the pool (measured: "prefill lends 2252 expert
    frames ... a ring of 1024").
  - Nothing is promoted while that lease is out.
  - Decoding lanes either wait out the whole prompt, or run beside it with 18% fewer frames.
  - A policy is needed: a share of rounds per chunk, or narrower calls while lanes decode (Infernix
    S8 measures exactly this). The queue model above assumed decoding lanes wait.
- **Host DRAM and the link.** At C = 4, ~41% of the link is staging and promotion traffic, on a
  DDR4-2133 host that also feeds the n-gram rows. Infernix found a host-DRAM floor binding plain
  C = 2 in its fill phase. Unmeasured here.

## 5. Smallest viable first slice

If the user wants C = 2 for queued agents:

1. **The forward over B rows**, plain decode only:
   - state slots, per-row block tables and per-row n-gram rows;
   - batched QSA and PLE Op variants, qualified against their oracles;
   - `test_forward_real`: B = 2 decode of two unrelated prompts with greedy ids equal to each
     decoded alone.

   This is the only slice whose cost is uncertain: it is where most of the new code is.
2. **The Program at `--max-concurrency 2`:**
   - lane array, per-lane KV extents through the elastic pool (PR 12's order), and one decode graph
     pair per B;
   - MTP only in one-row rounds;
   - decoding lanes wait out another lane's prefill calls at first, with the policy left for later;
   - C > 2 refused.

   **Acceptance gate:** on this 8-request workload, aggregate decode ≥ 1.15x MTP C = 1 (92.6
   tok/s); greedy ids equal C = 1; the 4K/32K/128K alternation with no crash; a lone request within
   2% of today's MTP speed.

Rough size: two to three PRs, ~2,500-3,500 hand-written lines plus tests. This is an estimate from
the table above, not a design.

If the gate is missed (for example, real non-expert column cost well above 1.4 ms, or a host-DRAM
floor), stop at the forward PR. That is the point where the model's main assumption is first
measured.

## 6. Measured, estimated and how to reproduce

| Claim | Status |
|---|---|
| Per-lane bytes, frame size, startup frames, free VRAM | measured (startup logs, both launcher configurations) |
| Frames at C = 2-4 | arithmetic on measured bytes; 64K KV estimated |
| Host RAM per lane | estimated from the IO layout |
| C = 1 hit rates; replay validity | measured; replay within 0.4 points of the server per request (one short request 1.1) |
| C = 2-4 hit rates, misses, distinct experts | replayed real routes through the real policy; the interleaving (prefill alone, lockstep rounds) is a model |
| Layer-call cost vs columns, distinct experts, misses | measured (Op bench, cold L2): an Op result, not end to end |
| C = 1 decode, MTP and plain, on the 8 requests | measured |
| Round time, aggregate and per-stream tok/s at C > 1 | **estimated** (model; n = 1.4 ms per column from PR 13's nsys) |
| Queue wait, makespan | **estimated** (model applied to measured prefill times and token counts) |
| MTP in multi-row rounds | **estimated** from a proxy |

**On the spike branch, not for merge:**

- **`ExpertResidency::after_round`:** appends the round's routes to `NINFER_ROUTE_TRACE` when that
  variable is set. Nothing changes when it is unset: the decode tok/s of the traced and untraced
  plain sessions agree within their frame-count difference.
- **`report_startup`:** one extra `spike plan bytes` line.

**Tooling:**

- `bench/ops/offloaded_moe_bench.cu` `--mode lanes`.
- `tools/flash_next/lanes_spike/`: `lanes_replay.cpp` (CPU; links only `expert_cache.cpp`),
  `client.py` (the 8 conversations), `sweep.py`, `fit_bench.py`, `step_model.py`, `queue_sim.py`.

**Reproduce:**

1. Build `ninfer-serve` from this branch.
2. Start it with `--no-expert-state --max-context 32768 --kv-capacity 32768 --kv-dtype int8
   --prefill-chunk 1024` and `NINFER_ROUTE_TRACE` set. Hold the GPU lock.
3. `python tools/flash_next/lanes_spike/client.py <port> <trace path> <out dir> <repo root>`.
4. Build the replay with the `cl` line in its header.
5. `python sweep.py <lanes_replay.exe> <out dir>`.
6. `python step_model.py <the server's stderr log> <sweep csv>`.

All runs: RTX 3090 (WDDM, desktop attached), CUDA 12.8, MSVC 14.44, Ryzen 9 5950X, DDR4-2133,
artifact `qwen4exp-iq2xs-mtp-vision.ninfer` with its n-gram volume on NVMe. Every GPU run held the
shared GPU lock.
