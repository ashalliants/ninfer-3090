# Context cache on a hosted multi-tenant 27B: baseline and loss attribution

Measured 2026-10-13 on one RTX 3090 with Qwen3.8-27B at master `04901788`. This is a baseline: the
engine was not changed. It answers which cache problems actually cost TTFT and hit rate when the
27B is hosted for many people, so that the nine candidate ideas from the Infernix cache study can be
put in order. The tooling is `tools/bench/run_hosted_workload.py`, `tools/bench/hosted_analysis.py`
and `tools/bench/disk_seq_io.py`, described in `tools/bench/README.md`.

## Summary

- **Below saturation, a Host tier of 8 GiB or more already reuses everything it can.** With 16
  tenants and conversations of 5-50K tokens, every append after warm-up resumed from its own
  conversation at 8 and at 16 GiB (65/65 requests each). The two sizes gave identical reuse (hit
  0.886 against an ideal of 0.940) and the same TTFT (p50 1.05 / 1.10 s). The whole remaining gap
  is edits and branches whose resume point was superseded. At 2 GiB, 25 of 61 appends lost their
  conversation (hit 0.732, TTFT p50 4.1 s).
- **Above saturation, eviction while waiting is the largest loss.** With 32 tenants offering more
  prefill than the card can do (about 2,100 tok/s), queue wait dominates TTFT: p50 60-106 s,
  p95 118-150 s. Conversations are evicted 30-65 s (median) after their last turn, while their
  next turn is still queued. They then re-prefill 540-830K tokens per 10-minute run, 37-51% of
  all prompt tokens the GPU computed. An LRU the size of the Host tier alone would still have held about half of
  those tokens.
- **Host allocator failures are real, frequent and can stall the engine for seconds.** A
  measurement build counted 30-291 failed Host state reservations and 4-172 failed KV extent
  reservations per run whenever the Host tier was full (2 and 8 GiB), and none at 16 GiB. In one
  8 GiB run a single admission retried a failing KV demotion page by page for about 14 s. Each
  attempt took 1.5-4.0 s on the Engine worker, with the GPU idle and a request waiting. No log line
  or metric shows any of this today.
- **The context store works and recovers restarts, and it already repairs a small Host tier at
  sustainable load, but it writes nothing under saturation.** After a crash, 16 of 16
  conversations resumed from the store (TTFT p50 1.07 s, max 3.0 s). With a 2 GiB Host at
  sustainable load it raised the post-warm-up hit from 0.732 to 0.876 (ideal 0.941), cut TTFT p50
  from 4.07 to 1.99 s and p95 from 16.6 to 8.0 s, and cut tokens prefilled over the run by 33%
  (892K to 595K), for about 90-110 GB of new chunks per hour. Under saturation (C8, 8 GiB) it
  wrote 0 sessions in 12 minutes and changed nothing (hit 0.610 against 0.603, TTFT p50 81.9
  against 83.3 s), because the background writer runs only while no request waits and no lane
  prefills (`persist_idle_session`, `engine_core.h:2821`), and a saturated server is never in that
  state.
- **Lanes are not the limit.** C4 and C8 gave the same reuse and almost the same TTFT. Under
  saturation a request waited with a lane free 70-89% of the time: prefill compute is the limit, and
  C8's extra lanes mostly idle (mean busy 0.36).
- **Ranked recommendation** (details below): 1) make Host demotion failures cheap and visible, then
  decide on the slab pool (idea 7); 2) disk-aware eviction with write-on-evict, plus store writes
  that are not blocked by a busy server (ideas 1 and 2); 3) disk-backed history points for edits
  (idea 9); 4) divergence-tap single-flight (idea 4), which matters only when clients fan out over
  a fresh document: then it is worth 20% of all prefill, and without fan-out nothing. The remaining
  ideas measured small on this workload.

## How it was measured

- **Box.** One RTX 3090 (24 GB) beside a desktop holding 1.0-1.2 GiB, and 64 GB of RAM. Other
  agents' jobs waited on the GPU lock, so every run had the card to itself; CPU and disk were
  shared. Windows 11, CUDA 12.8, master `04901788`, Release.
- **Server.** The launcher's `tuned` 27B flag set with `NINFER_SPEC=mtp`, started directly so that
  the lanes and the Host size could vary:
  `--max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 --spec mtp --draft-tokens 3
  --lm-head-draft --gdn-state-fp16 --prefill-cublas --prefill-chunk 2048 --max-pending-requests 16
  --pending-timeout-ms 600000 --vision --vision-residency overlay --fit-context 32768 --min-p 0.03
  --presence-penalty 0.5`, plus `--max-concurrency C`, `--host-context-mib H`,
  `--request-log-jsonl` and, where stated, `--context-store DIR`. No prompt graft was installed.
  Every run started at the full 262,144-token pool, with the default device state slots (C
  extra). VRAM free at start was 539-562 MiB at C8 and 1.19 GiB at C4.
- **Workloads** (`tools/bench/README.md` has the exact presets). Products are shared system and
  tool prompts of 1,608, 4,563 and 9,090 tokens. Opening documents are 2-40K tokens and appended
  tool output is 0.3-4K tokens per turn. Of the turns, 5% are edits and 5% are branches. Replies
  are at most 160 tokens (the model answers in about 30). Temperature 0, the same seed for every
  run, so runs of one preset send the same requests until their timing diverges.
  - *Sustainable*: `hosted-16` with a 120 s mean think time, 840 s. Statistics are after a 300 s
    warm-up, so the first prefill of every conversation is left out.
  - *Saturated*: `hosted-32` with a 40 s mean think time, 600 s plus drain. Statistics cover the
    whole run, which never reaches a steady state.
- **Ideal reuse.** For every request the client records what a cache that kept every completed
  reusable point forever could have reused: the same lineage's previous prompt plus reply (prompt
  only with thinking on), or the product prompt for a new conversation. On the smoke run every
  append's `cached_tokens` was exactly ideal − 1, which validates the estimate.
- **Hit classes** (256-token tolerance): `private` reused its own conversation; `shared` reused only
  the product prompt; `none` reused under 64 tokens.
- **Loss causes**, per request: an edit or branch below its ideal is `diverged`; a new conversation
  below its product prompt is `shared_miss`; an append below its ideal is `evicted_*`. The
  `evicted_*` split replays the trace through an LRU over whole conversations of
  `tokens × 18,000 B + 74.5 MiB` (rk4v4 per `docs/serving.md`, one FP16 StateImage):
  - `evicted_capacity`: an LRU of Host + the 262K Device pool would also have dropped it;
  - `evicted_host_capacity`: only a Host-sized LRU would have dropped it;
  - `evicted_policy`: even a Host-sized LRU would still hold it.

  This is a ceiling model. The engine also keeps E and R points, shared entries and pause
  snapshots in the same bytes; with 2 or 4 StateImages per conversation the policy share shrinks
  but stays (see Saturated).
- **Measurement build.** Runs named `s-*` and `b-*`, and the saturated store run `c8-h8-store`, used a build of the same commit with a
  measurement-only patch (appendix; not in this PR). It writes one stderr line per `try_admit_one`
  with its wall time, and one per Host demotion that failed to reserve its destination or had to
  split a KV batch. Nothing else differs. The other saturated runs used the unpatched build.
- **Sample size.** One run per configuration, about 70 requests after warm-up for sustainable runs
  and 118-162 requests for saturated ones. Differences of a few percent in TTFT between two runs are
  within what one run cannot resolve; the reuse figures are exact counts.

## Results

### Sustainable load (16 tenants, 120 s think)

After the 300 s warm-up:

| run | n | hit (cached / prompt) | ideal | private / shared / none | TTFT p50 / p95 / p99 s | queue p50 / p95 s | lanes busy |
|---|---:|---:|---:|---|---|---|---:|
| C8, Host 2 GiB | 69 | 0.732 | 0.941 | 36 / 31 / 2 | 4.07 / 16.6 / 25.4 | 0.05 / 3.50 | 0.13 |
| C8, Host 8 GiB | 71 | 0.886 | 0.940 | 65 / 6 / 0 | 1.05 / 12.7 / 19.9 | 0.04 / 3.21 | 0.09 |
| C8, Host 16 GiB | 71 | 0.886 | 0.940 | 65 / 6 / 0 | 1.10 / 11.4 / 19.3 | 0.02 / 4.71 | 0.08 |
| C4, Host 8 GiB | 70 | 0.882 | 0.939 | 63 / 7 / 0 | 1.22 / 10.9 / 19.7 | 0.04 / 5.63 | 0.14 |
| C4, Host 2 GiB | 67 | 0.750 | 0.940 | 43 / 22 / 2 | 4.32 / 19.4 / 25.1 | 0.04 / 8.84 | 0.21 |

Tokens an ideal cache would have saved, by cause (requests in brackets):

| run | diverged | shared_miss | evicted_capacity | evicted_host_capacity | evicted_policy |
|---|---:|---:|---:|---:|---:|
| C8, 2 GiB | 138,004 (8) | 0 | 121,672 (12) | 123,749 (11) | 13,423 (2) |
| C8, 8 GiB | 106,125 (6) | 0 | 0 | 0 | 0 |
| C8, 16 GiB | 106,121 (6) | 0 | 0 | 0 | 0 |
| C4, 8 GiB | 109,856 (7) | 0 | 0 | 0 | 0 |
| C4, 2 GiB | 160,261 (7) | 0 | 124,205 (12) | 50,769 (4) | 12,658 (1) |

- **TTFT by outcome** (C8, 8 GiB): a private hit p50 0.93 s, p95 5.1 s (n 65); a request that
  reused only the product prompt p50 7.2 s, p95 20.5 s (n 6). At 2 GiB: private 1.12 / 7.9 s (n 36),
  product-only 7.3 / 21.3 s (n 31). The p95/p99 tail is these re-prefills plus whatever queued
  behind them.
- **Restores are cheap.** The H2D part of binding was p50 7-23 ms and p95 45-122 ms per restored
  request (75-230 MiB). Context transfers of every kind took 2.7-3.6 s per 840 s run.
- **Edits and branches** are the only loss left at 8 GiB or more. Their resume point is an earlier
  turn's endpoint, which the newer turn superseded and reclaim drops first. With little pressure
  (the smoke run) all 16 edits and 14 branches hit. Under these runs' pressure 1-2 of 8 did.
- **Including warm-up** (all requests): hit 0.698 / 0.814 / 0.814 / 0.811 / 0.717 and TTFT p50
  4.2 / 1.4 / 1.4 / 1.5 / 4.2 s, p99 53 / 49 / 49 / 42 / 42 s for C8-2, C8-8, C8-16, C4-8 and
  C4-2 GiB. The tail is the first
  request of each conversation, prefilling up to 50K tokens behind the others.
- **Throughput.** Prefill ran at 2,091-2,141 tok/s while busy. The prompt traffic averaged
  660-1,040 tok/s, so the GPU idled roughly 50-70% of the time. Decode was 4.1-4.4 tok/s of wall
  time, because replies are short. `hosted_analysis.py` reports both.

### Saturated load (32 tenants, 40 s think)

Whole run. The first three rows used the unpatched build; the store row used the measurement build:

| run | n | hit | ideal | private / shared / none | TTFT p50 / p95 / p99 s | queue p50 / p95 s | lanes busy | 429s |
|---|---:|---:|---:|---|---|---|---:|---:|
| C4, Host 8 GiB | 162 | 0.620 | 0.807 | 85 / 72 / 5 | 84.7 / 178 / 221 | 59.7 / 118 | 0.57 | 2,347 |
| C8, Host 8 GiB | 150 | 0.603 | 0.793 | 66 / 75 / 9 | 83.3 / 183 / 229 | 69.3 / 150 | 0.36 | 1,052 |
| C8, Host 2 GiB | 118 | 0.407 | 0.743 | 23 / 91 / 4 | 142 / 194 / 247 | 106 / 150 | 0.68 | 1,045 |
| C8, Host 8 GiB, store | 152 | 0.610 | 0.798 | 62 / 84 / 6 | 81.9 / 178 / 223 | 75.5 / 150 | 0.34 | 901 |

| run | diverged | evicted_capacity | evicted_host_capacity | evicted_policy | total evicted |
|---|---:|---:|---:|---:|---:|
| C4, 8 GiB | 170,894 (9) | 163,516 (10) | 169,690 (9) | 246,860 (17) | 580,066 (36) |
| C8, 8 GiB | 163,374 (8) | 78,085 (6) | 98,987 (10) | 359,673 (28) | 536,745 (44) |
| C8, 2 GiB | 95,926 (5) | 271,148 (15) | 338,991 (19) | 217,790 (24) | 827,929 (58) |
| C8, 8 GiB, store | 121,311 (7) | 129,605 (12) | 111,154 (8) | 346,761 (31) | 587,520 (51) |

- **Prefill-bound.** Prefill ran 689-734 s of each 740-770 s run at 2,124-2,224 tok/s. Per run,
  1.46-1.63M tokens were computed: openings (about 570K), evicted appends (540-830K, 37-51% of the
  total) and edits (96-171K).
- **Eviction while queued.** Evicted appends lost a median 8.5-12.9K tokens, and their source was
  33-65 s old (median) when they were submitted. The ingress bound (C + 16) turned away
  1,045-2,347 attempts with `429`; the client retried every second, as a gateway would.
- **Policy share is robust to the model.** With 2 or 4 StateImages per conversation in the LRU,
  `evicted_policy` at C8 8 GiB is still 333K or 323K tokens (27 or 25 requests).
- **The pools are full.** Device Main KV occupancy averaged 0.90 of 262,144 tokens in every run,
  saturated or not (retained history fills it by design). Host peaked at the full budget in every
  run at 2 and 8 GiB, and at 13.0 of 16 GiB.
- **Host-active Engine time** was 0.66-0.67 of wall under saturation, against 0.22-0.34 at
  sustainable load (`/v1/load` `host.active_seconds`). Not attributed further here.

### Lanes: C4 against C8

At sustainable load C4 and C8 at 8 GiB are the same within one run's resolution: hit 0.882 against
0.886, TTFT p50 1.22 against 1.05 s, queue p95 5.6 against 3.2 s. Under saturation, the
`/v1/load` polls show a request waiting with a lane free 81% of the time at C4 and 89% at C8 (70%
at C8 with 2 GiB). Mean lane occupancy was 0.57 at C4 against 0.36 at C8, and the hit rates are
0.620 against 0.603. Prefill compute is the limit; more lanes only add decode rows, and decode is
a small share of this workload. With long replies the balance shifts: with thinking on (below)
lanes were busy 0.28 of the time at C8.

### Context store on, and a restart

C8, 8 GiB, sustainable load, 600 s with the store on. At the end the client idled 60 s, the server
was killed (TerminateProcess, so a crash with no shutdown flush), restarted with the same store,
and each of the 16 open conversations sent one more turn, 2 s apart.

| | value |
|---|---|
| hit, same 600 s window: store on / off | 0.796 / 0.794 (n 91 each) |
| TTFT p50 / p95 / p99, same window: on / off | 1.60 / 22.9 / 49.9 s against 1.45 / 21.8 / 50.2 s |
| sessions written, new bytes, deduplicated bytes | 65; 19.0 GB; 21.1 GB |
| new bytes per hour (plain information) | 110 GB/h (105,000 MiB/h) |
| one image write (sampled from the log) | 447-764 MiB in 0.38-0.73 s |
| store at the end | 30 sessions, 11.5 GiB; 35 superseded images removed |
| hydrations during the run | 2, 65,563 tokens, 5.59 s of waiting (2.8 s each) |
| start-up restore after the crash | 12 sessions, 7.82 GiB in 15.4 s (0.54 GB/s) |
| resumes after the crash | 16/16 private; 6 more hydrations (147,250 tokens, 6.29 s) |
| resume TTFT p50 / p95 / max | 1.07 / 2.56 / 3.03 s; 95.1% of prompt tokens reused |

- The store **served an edit**. A 46.6K-token edit whose resume point the cache had dropped was
  hydrated from the store's older image: TTFT 7.5 s, against 19.5 s for the same request without
  the store. In effect it is already a disk-backed history point for edits (idea 9).
- The 0.15 s higher TTFT p50 with the store on is within one run's resolution; it was not
  investigated. The documented cost is that taking an image is one Engine-worker step that
  arrivals wait for.
- **Admission with the store** had one call over 1 s (1.17 s) and 14 over 100 ms (5.2 s in total),
  against 6 over 1 s without the store in the 840 s run. The counts are too small to say whether
  the store changes this.

### Context store at a 2 GiB Host, and under saturation

Two further runs with the store on and no restart.

| | C8, 2 GiB, sustainable, store | same, no store | C8, 8 GiB, saturated, store | same, no store |
|---|---:|---:|---:|---:|
| requests after warm-up (sustainable) / whole run (saturated) | 69 | 69 | 152 | 150 |
| hit / ideal | 0.876 / 0.941 | 0.732 / 0.941 | 0.610 / 0.798 | 0.603 / 0.793 |
| private / shared / none | 53 / 14 / 2 | 36 / 31 / 2 | 62 / 84 / 6 | 66 / 75 / 9 |
| TTFT p50 / p95 / p99 s | 1.99 / 8.0 / 10.6 | 4.07 / 16.6 / 25.4 | 81.9 / 178 / 223 | 83.3 / 183 / 229 |
| evicted tokens (requests) | 64,896 (9) | 258,844 (25) | 587,520 (51) | 536,745 (44) |
| diverged tokens (requests) | 56,192 (7) | 138,004 (8) | 121,311 (7) | 163,374 (8) |
| prompt tokens prefilled, whole run | 595,351 | 891,677 | 1,475,875 | 1,462,518 |
| store writes / new bytes / reused bytes | 85 / 21.8 GB / 31.7 GB | n/a | 0 / 0 / 0 | n/a |
| hydrations / tokens / seconds waiting | 14 / 302,004 / 14.8 | n/a | 0 | n/a |

- **At 2 GiB and sustainable load the store recovers about three quarters of the eviction loss.**
  Evicted tokens fell from 258,844 to 64,896, and private hits rose from 36 to 53 of 69. Prefill
  over the whole run fell by 296K tokens (33%, about 140 s of GPU time). The writes were about
  88,800 MiB per hour (93 GB/h; 21.8 GB in 841 s), each image taking 0.16-0.61 s. The 14
  hydrations cost 14.8 s of waiting in total (1.06 s each, 21K tokens per second of restored
  context), one hydration failed (`hydration_failures_total` 1; not traced, the request was
  served by prefill), and the store held 28 images, 12.0 GiB, at the end. The 9 evicted requests
  that remain were not traced; the plausible causes are conversations evicted before the 30 s idle
  threshold and hydrations under the 4,096-token gate, and neither was checked.
- **At 8 GiB under saturation the store never wrote.** `ninfer_context_store_writes_total` stayed
  at 0 for the whole run, so there was nothing to hydrate (`hydrations_total` 0). The only
  difference from the no-store run, hit 0.610 against 0.603 and TTFT p50 81.9 against 83.3 s, is
  within what one run resolves. Evicted conversations were evicted before they were ever written,
  because they are evicted 30-65 s after their last turn, while their next turn queues, and the
  writer waits for an empty queue and idle lanes. A store that only writes in idle moments cannot
  recover saturation losses; the write has to happen at or before eviction (idea 1).
- **Allocator failures were not reduced by the store**: 250 state and 57 KV failures at 2 GiB
  (291 and 172 without), and 727 and 294 under saturation (the highest of any run except the
  bursty one; there is no matching unpatched count).

### Thinking on

C8, 8 GiB, sustainable load, `--thinking on` (640 output tokens, `thinking_budget` 384, the
client sends back only the answer), 600 s, whole run: hit 0.786 against an ideal of 0.797. All 58
appends resumed privately from the input checkpoint before the previous reply (cached = ideal − 7).
Edits lost only 3.5K of 110K ideal tokens, and branches 16K of 41K. TTFT p50 1.37 s, p95 44.6 s:
the p95 is warm-up openings queued behind long replies. Decode was 49 tok/s of wall, with lanes
busy 0.28. The thinking-on path has no cache problem of its own on this workload.

### Bursty arrivals with sub-agent fan-out

`bursty-32` with 16 tenants, C8, 8 GiB, 720 s, measurement build. Every 45 s a random group of 8
tenants is released at once. 15% of turns are a fan-out: 3 sub-agent requests sent together, each
the conversation plus one new 4-16K document plus its own task. Fan-out made the run saturated:
148 requests, hit 0.646 against an ideal of 0.781, TTFT p50 / p95 / p99 30.8 / 189 / 251 s, queue
p50 18.5 s, 67 `429`s, lanes busy 0.47.

- **No sibling ever reused the shared document.** Across 18 fan-out groups (54 requests), every
  sibling's `cached_tokens` stopped at or below the conversation's own endpoint. None reached the
  document that its siblings were prefilling at the same moment. The document ends at no chat
  marker, so today's single-flight (it waits only for a planned SharedPrefix frontier at a marker)
  does not apply. **Sibling ceiling: 300,884 tokens** in this run, the document prefilled again by
  the second and third sibling of each group. That is 20% of the 1.50M tokens the run computed,
  about 140 s of prefill.
- **Admission source waits are large under bursts.** `generation.admission.source_wait_seconds`
  summed to 2,454 s over 148 requests (fan-out 1,315 s, appends 930 s; 66 requests waited over 1 s,
  the longest 136 s). That is part of their queue wait, which summed to about 6,000 s. At
  sustainable load the same counter summed to 32-51 s per run (at most 15 s for one request). These
  are the engine's deliberate waits for a sibling's planned frontier or a retained source. That they
  then often reused nothing beyond the endpoint suggests the waits did not pay off here. This was
  not traced further.
- The parent endpoint itself was also lost for many siblings (`evicted_policy` 537K tokens in 42
  requests). Allocator failures were the highest of any run: 434 state and 181 KV reservation
  failures, 4,719 splits, and 25 admissions over 100 ms (8.8 s in total).

### Host allocator failures and demotion stalls

Nothing in the shipped server reports these: there is no log line and no metric. A failed Host
demotion reservation returns `false` from `start_demote`
(`src/models/qwen3_5/program/transactions/reclaim.cpp`). The resource manager then tries
superseded-only Host releases, and after that the next ranked action, which can delete a live
source (`resource_manager.h`, the #254 workaround). The measurement build counted:

| run | state reservation failed | KV extent reservation failed | KV batch split | admissions > 100 ms (sum) | > 1 s |
|---|---:|---:|---:|---|---:|
| C8, 2 GiB | 291 | 172 | 1,677 | 12 (2.35 s) | 0 |
| C8, 8 GiB | 134 | 14 | 356 | 9 (15.07 s) | 6 |
| C8, 16 GiB | 0 | 0 | 0 | 2 (0.30 s) | 0 |
| C4, 8 GiB | 30 | 4 | 284 | 5 (0.80 s) | 0 |
| C8, 8 GiB, store | 37 | 18 | 543 | 14 (5.16 s) | 1 |
| C8, 8 GiB, thinking | 4 | 1 | 32 | 1 (0.16 s) | 0 |
| C8, 8 GiB, bursty fan-out | 434 | 181 | 4,719 | 25 (8.82 s) | 0 |
| C4, 2 GiB | 267 | 56 | 613 | 8 (1.46 s) | 0 |
| C8, 2 GiB, store | 250 | 57 | 916 | 8 (2.30 s) | 1 |
| C8, 8 GiB, saturated, store | 727 | 294 | 7,887 | 10 (1.69 s) | 0 |

**The stall.** In the C8 8 GiB run, from 14:20:20 to 14:20:34, the Engine reported `running 0 |
waiting 1 | host 80-86%`, and the stderr shows this sequence repeated:
`demote_split kv count=1 of=6 … of=2`, then `demote_fail kv missing=6 offset=5`, then
`admit … us=1470631`. The same pattern followed with `missing=5`, `4`, `3`, `2` and `1`, each
attempt taking 1.5-4.0 s. Every attempt halves the batch down to one page, reserves pages one at a
time, fails on the last one, unwinds, and returns to admission. One waiting request (req 107) paid
about 14 s, and the GPU was idle the whole time. This is the "Host alloc failure on demotion" that
#254 worked around: it happens only when the Host tier is full, and it is expensive, not just
frequent.

### Admission planning time

`try_admit_one` wall time per call (measurement build). Calls that admitted, restored or paused
something:

| run | n | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| C8, 2 GiB | 306 | 2.8 ms | 84 ms | 207 ms | 416 ms |
| C8, 8 GiB | 336 | 3.7 ms | 30 ms | 2,257 ms | 4,042 ms |
| C8, 16 GiB | 306 | 3.7 ms | 13 ms | 34 ms | 159 ms |
| C4, 8 GiB | 366 | 4.7 ms | 27 ms | 127 ms | 192 ms |
| C8, 8 GiB, thinking | 237 | 3.9 ms | 15 ms | 52 ms | 157 ms |

Calls that did nothing (another 440-880 per run) took 0-1 µs at p50 and at most 1.3 ms at p99.
This time is the whole admission step, including `inspect_source`, reclaim and `start_binding`. It
runs on the Engine worker, so every lane waits for it. Ordinary admissions cost 3-5 ms, and with
Host room the p99 stays under 130 ms. The large values are the demotion failures above, not
planning: with a 16 GiB Host the p99 is 34 ms. Infernix's reason for replacing its old cache was a
250 ms planning allowance. Ours plans in milliseconds on this workload, and the outliers are
allocator failures.

### Raw disk speed of the C: NVMe

`tools/bench/disk_seq_io.py --dir %TEMP% --size-gib 4 --block-mib 4 8 16 32 --repeats 2
--buffered`, plus one 8 GiB pass at 16 MiB, unbuffered (`FILE_FLAG_NO_BUFFERING`, write-through),
queue depth 1. Other agents were building on the same drive at times.

| block | write GB/s (2 passes) | read GB/s (2 passes) | buffered read, from disk / again |
|---|---|---|---|
| 4 MiB | 0.83, 0.64 | 3.12, 2.43 | 1.98-3.23 / 1.98-3.51 |
| 8 MiB | 0.93, 1.02 | 5.08, 4.80 | 1.54-3.18 / 1.68-2.58 |
| 16 MiB | 1.00, 1.11 | 4.78, 5.19 | 1.36-2.81 / 2.02-3.13 |
| 32 MiB | 0.88, 1.51 | 4.52, 5.32 | 1.20-1.22 / 1.13-1.28 |
| 16 MiB, 8 GiB file | 1.07 | 4.95 | 1.29 / 3.65 |

- **Reads**: about 4.9 GB/s unbuffered at 8 MiB blocks or more (2.4-3.1 at 4 MiB). That is 8-9×
  the 0.54-0.57 GB/s the store's read path achieves end to end (start-up restore above; the 856 MB
  / 1.5 s figure in `docs/serving.md`).
- **Writes**: 0.6-1.5 GB/s write-through at queue depth 1. That covers the store's measured write
  rate (110 GB/h, about 0.03 GB/s) many times over.
- **Buffered reads** through Python were 1.2-3.7 GB/s and noisy, so they are not a ceiling for
  anything. The point is that the unbuffered path is not slower.

## Attribution: what each idea could recover

Prefill on this workload runs at about 2,100 tok/s, so 1,000 re-prefilled tokens cost about 0.48 s
of GPU time and of every queued request's TTFT. All ideas are measured against the runs above;
"ceiling" means the most the idea could recover if it worked perfectly.

| # | idea | what it addresses here | measured ceiling on this workload |
|---|---|---|---|
| 7 | slab Host pool (no geometry failures) | demotion reservation failures, page-by-page retries, worker stalls | 30-291 state and 4-172 KV failures per run at ≤ 8 GiB; up to 15 s of GPU-idle Engine stall per 840 s run, concentrated on single requests (14 s on one). Upper bound on top: the `evicted_policy` losses under saturation (218-360K tokens per run), which failed demotions can cause by deleting live sources. Not proven to be the cause |
| 1 | disk-aware eviction + write-on-evict | conversations dropped from the Host tier | every `evicted_*` token: 540-830K tokens per saturated 10-minute run (37-51% of computed prefill) and 259K post-warm-up at sustainable 2 GiB (25 requests, about 125 s of prefill). Restoring a median 8.5K-token conversation (about 0.3 GB with two states) takes 0.06 s at the drive's 4.9 GB/s, or 0.55 s through today's store path, against 4.0 s of prefill. Measured: today's idle-write store already recovers 75% of the 2 GiB sustainable loss (194K of 259K tokens, hit 0.732 → 0.876), but writes nothing under saturation, where the loss is largest, so the missing part is writing at eviction, not the disk tier itself. The cost of write-on-evict was measured on the disk-tier branch as about 16% of Engine-worker time and about 550 GB/h (agent-replay workload, C4 2 GiB, `tools/bench/disk_tier/results/RESULTS.txt` on the disk-tier branch, PR #295: hit 0.32 to 0.74, 550 GB/h; not re-measured on this hosted workload) |
| 2 | disk read term in the cost model | the fixed 4,096-token hydration gate | 5-8 evicted appends per run lost less than 4,096 tokens (at most about 30K tokens, about 15 s per run). An enabler for 1 and 9 more than a gain |
| 9 | dense taps / history points on a cheap tier | edits and branches whose resume point was superseded | 106-110K tokens per 840 s run at 8-16 GiB (6-7 of about 70 requests): hit 0.886 → 0.940, and those requests' TTFT p50 7.2 s → about 1 s. 96-171K tokens per saturated run. Scales linearly with the edit and branch rate, which is this workload's assumption (5% + 5%). The store already recovered one such edit |
| 4 | divergence-tap single-flight | siblings prefilling the same non-marker prefix at once | 300,884 tokens (36 sibling requests) in the bursty fan-out run: 20% of everything it computed, about 140 s of prefill. Zero in the Poisson runs, which have no fan-out: there the shared product prompts sit at markers, and `shared_miss` was 0. Depends entirely on how often clients fan out over a fresh document |
| 3 | persist shared prefixes | the product prompts after a restart | 15,261 tokens per restart (all three products), about 7 s of prefill once. `shared_miss` was 0 in every steady-state run: the existing shared entries and single-flight already cover live traffic |
| 5 | blocked-head Host→Device prefetch | restore time of a queued request | H2D restore p50 7-23 ms and p95 45-122 ms against a TTFT p50 of about 1 s: at most 2-10% of a private hit's TTFT, and nothing under saturation, where the queue is 60-106 s |
| 6 | write-through at release | D2H demotion on the critical path | all context transfers took 2.7-3.6 s per 840 s run (all directions), at most 0.4% of wall time. The demotion stalls are reservation failures, not copies, which idea 7 covers |
| 8 | free VRAM as Device block cache | Device room beyond the C × L pool | 539-562 MiB of VRAM free at C8 start, about 30K tokens (+12% of the pool); 1.19 GiB (about 66K tokens) at C4. `evicted_host_capacity` (124K tokens at 2 GiB, none at 8 GiB) is the most it could address |

## Ranked recommendation

Ordered by the measured ceiling on this workload, then by the effort and risk given in the cache
study:

1. **Idea 7: make Host demotion failures cheap and visible first, then decide on the slab pool.**
   The step-one fix is small and needs no new allocator: stop retrying a failing KV demotion page
   by page inside one admission; give up on that victim after the first failed reservation and move
   on, so that one victim cannot hold the Engine worker. Also count the failures and log them (a
   `ninfer_context_host_reservation_failures_total{resource}` series). Then re-measure: if the
   failures still cause live-source deletions (the `evicted_policy` share under saturation), the
   slab pool is justified. Expected gain: the multi-second stalls, up to 15 s per 840 s run at
   8 GiB, and the TTFT tail they create; possibly part of the 218-360K `evicted_policy` tokens under
   saturation.
2. **Ideas 1 and 2: disk-aware eviction with write-on-evict, and a disk term in the cost model.**
   The largest token ceiling (540-830K tokens per saturated 10-minute run, 259K at 2 GiB
   sustainable), and the drive reads 4.9 GB/s. Today's idle-write store recovers 75% of the sustainable 2 GiB loss and none of the saturated one (it never writes there), so the work is to write at eviction time, not to build a new tier. Write volume is not counted as a
   cost (the user's decision for hosted instances).
3. **Idea 9: history points on disk.** About 6% of prompt tokens on this workload (5% edits,
   5% branches), and the TTFT of those requests drops from about 7 s to about 1 s. The store
   already hydrates one kind of these; extending it to keep superseded endpoints (with idea 2's
   cost term) is the cheap form. P1, the Host-only form, was rejected for Host pressure, which a
   disk tier removes.
4. **Idea 4: divergence-tap single-flight**, if clients fan out. It recovered nothing in the Poisson
   runs, but 300,884 tokens (20% of computed prefill) in the fan-out run, where no sibling reused
   the shared document. Before building it, check why the existing source waits (2,454 s summed in
   that run) did not lead to reuse.
5. **Idea 3: persist shared prefixes.** About 7 s of prefill per restart; cheap, but small.
6. **Idea 5: blocked-head prefetch.** At most a few tens of milliseconds per restored request.
7. **Idea 6: write-through at release.** Under 0.4% of wall time in copies; its robustness benefit
   is better obtained from item 1.
8. **Idea 8: free VRAM as block cache.** About 30K tokens of room on a 27B on a 3090.

Not cache work, but bigger than any of the above on a saturated server: prefill throughput and
admission control. Once the offered prefill exceeds about 2,100 tok/s, queue wait (p50 60-106 s)
dominates TTFT, and waiting conversations get evicted, which adds more prefill.

## Open questions answered

### What step 5's P4 was, and why it was NO-GO

P4 was **flexible ladder anchors on long prefills**. During a prefill of at least 2·G new tokens
(G = 2 × the prefill chunk), it would capture LongAnchor points at the chunk-grid frontiers nearest
n − G·2^k: no extra split, at most 3 per request, sharing P1's slots and eviction class. The aim
was to serve a later request that diverges deep inside a long prefilled span without falling back
to the root or the endpoint. Its cost was about 80 MiB of Host per anchor, a D2D copy and a capture
transaction per anchor.

Its gate was mid-history divergence of at least 2·G on at least 10% of requests. Agent replay
measured 0/108 requests at C1 (2·G = 16,384) and 0/106 at C2 (2·G = 8,192), so it was never built:
no branch, no PR. The long-document case that might have exercised it was not built either. The
source is the step-5 specification and PR0 results in the step-5 session's scratchpad
(`step5-spec.md:387-417`, `s5p0/results.md:103-112`). Nothing about P4 was committed or put in a PR
description, and `gh pr list --search P4` finds nothing.

So P4 was the Host-only form of idea 9, not idea 8. Idea 8 was never evaluated in step 5. This
workload's edits and branches do diverge mid-history (6-10% of requests), unlike agent replay,
which is why idea 9 ranks third here.

The other step-5 items:

| item | outcome |
|---|---|
| P1 history points | Rejected: edits within 128 tokens of ideal 15/23 against a 90% gate; append hit 0.266 → 0.245 at 2 GiB |
| P2 skip-echo capture | Rejected: append hit 0.678 → 0.490 on the edit rotation |
| P3 | The context store under DFlash/DFlash2, PR #227 |
| P5 sibling single-flight | Shipped, PR #255: a 4-way 12.7K burst went 10.2 → 5.4 s |
| P6 restore overlap | Shipped, PR #259: TTFT p50 −7.6% |
| GDSF | NO-GO from a victim trace: victims already followed recency |
| #254 live-branch priority | Shipped |

### Does a Host allocator geometry failure show up in logs or metrics, and how often?

No. There is no log line at any level and no counter. `context_cache.captures.aborted` in the
JSONL throughput event counts only captures that started and then failed, and it is not on
`/metrics`. Counted with the measurement patch, failures occur on every run whose Host tier fills
(table above): up to 463 failures and 1,677 splits per 840 s at 2 GiB, and 148 failures with
multi-second stalls at 8 GiB. None occurred at 16 GiB.

### Time in admission planning per request at C8 with a large cache

Admission (`try_admit_one`, all of planning, source inspection, reclaim and binding start) takes
p50 3.7 ms at C8 with an 8 or 16 GiB Host. p99 is 34 ms at 16 GiB, and 2.3 s at 8 GiB because of
the demotion failures. The table is above. No shipped metric separates it: it sits inside
`queue_wait_seconds` and `ninfer_host_work_seconds_total{phase="engine_boundary"}`.

### Raw sequential speed of the C: NVMe

About 4.9 GB/s read and 1.0-1.2 GB/s write-through at 8-32 MiB blocks, with the cache bypassed;
2.4-3.1 GB/s read at 4 MiB. The table is above. The reusable script is
`tools/bench/disk_seq_io.py`.

### How the gateway routes, and whether session affinity would help

The gateway (`C:\ninfer-fork\ninfer-gateway`, read only) has one global fair queue
(`apps/gateway/src/scheduler/scheduler.ts`). It picks a backend in this order:
1. an engine a request must use (a Responses `previous_response_id` is pinned to its engine);
2. preferred engines, waiting up to an affinity budget if a preferred engine is busy;
3. otherwise the least loaded (`inUse/slots`, then KV reserved).

It learns load by polling each engine's `/v1/load` every 2 s.

**Session affinity already exists** (`apps/gateway/src/services/affinity.ts`):
- a per-conversation key `c:<user>:<fingerprint>`, where the fingerprint hashes the model, tools,
  leading system messages and the first conversational message;
- a weaker per-user key;
- entries live 15 minutes after last use;
- a busy warm engine is waited for between 3 and 30 s, scaled by prompt ÷ an assumed 1,500 tok/s
  prefill.

**Would affinity raise hit rates?** Yes, decisively, and it is already on. Without it, with E
engines a conversation's next turn would land on its warm engine only about 1/E of the time. On
the sustainable runs almost all reuse is the conversation's own. A turn sent to the cold engine
would still reuse its product prompt, about a fifth of a typical prompt. So at E = 2 the hit rate
would fall from 0.81-0.89 to roughly 0.5 (arithmetic from the measured split, not run).
Recommendations:
- **Keep conversation affinity.** Product-level affinity is unnecessary: shared prompts are small
  (1.6-9K tokens) and each engine caches them once.
- **Set `prefillTokensPerSecond` to about 2,100** for the 27B on a 3090 (measured here), not the
  1,500 default.
- **Decide what a saturated engine should do with its sticky traffic.** At a 60-106 s queue, the
  30 s affinity cap spills, and the turn re-prefills cold elsewhere. That is right when another
  engine is idle, and wrong when all are saturated.
- **Report eviction to the gateway.** The gateway cannot see that an engine has already evicted a
  conversation; under saturation, conversations were evicted 30-65 s after their last turn. A cheap
  engine-side hint, for example a per-request "reused N tokens" signal the gateway already gets in
  usage, would let it drop stale affinity after a miss.

Nothing in the gateway was changed.

## Not measured, and caveats

- One run per configuration. TTFT differences of a few tenths of a second between runs are not
  resolved; the reuse counts are exact for each run.
- The workload is synthetic. Word-filler text gives the MTP drafter little to accept; the
  edit and branch rates (5% + 5%), think times and sizes are assumptions, and the ceilings scale
  with them. Replies are short (about 30 tokens), so decode-heavy hosted traffic is covered only by
  the thinking-on run.
- The `evicted_*` split rests on an LRU model with an assumed 18 KB/token and 74.5 MiB per
  conversation; the policy share was checked at 2 and 4 StateImages per conversation, not against
  the engine's own victim choices.
- Whether failed demotions cause the `evicted_policy` losses was not proven: the saturated runs used
  the unpatched build, and the patch counts failures without tracing what reclaim did next.
- Not run: C4 under saturation at 2 or 16 GiB, C8 under saturation at 16 GiB, `--auto-host-cache`,
  DFlash2 (the launcher default at C1-C2; this baseline used the MTP set because it was asked for
  C4/C8), and S3.
- The disk numbers come from a shared drive at queue depth 1; a deeper queue may read faster.

## Appendix: measurement patch used for the `s-*` and `b-*` runs

Applied to `04901788` in a separate worktree, built as `ninfer-serve`, never committed. It only
adds `fprintf(stderr)` lines:

```cpp
// src/runtime/engine/engine_core.h: wrap the existing function
bool try_admit_one(bool restoring) {
    const auto instr_begin = Clock::now();
    const bool instr_result = try_admit_one_impl(restoring);   // the original body
    const auto instr_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              Clock::now() - instr_begin).count();
    std::fprintf(stderr, "[instr] admit result=%d restoring=%d us=%lld\n",
                 instr_result ? 1 : 0, restoring ? 1 : 0, static_cast<long long>(instr_us));
    return instr_result;
}

// src/models/qwen3_5/program/transactions/reclaim.cpp, start_demote
if (!transfer) { std::fprintf(stderr, "[instr] demote_fail state\n"); return false; }
...
if (!destination) {
    std::fprintf(stderr, "[instr] demote_fail kv missing=%zu offset=%zu\n", missing.size(), offset);
    return false;
}
if (count != missing.size() - offset) {
    std::fprintf(stderr, "[instr] demote_split kv count=%zu of=%zu\n", count, missing.size() - offset);
}
```
