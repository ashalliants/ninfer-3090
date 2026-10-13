# Task B: spike - can Flash-Next run several lanes on one RTX 3090, and is it worth building?

Read `brief-common.md` (same folder) first.

## Question
Today `run.bat qwen38-flash-next` serves **one request at a time** (`--max-concurrency 1`; the
Program and `qwen4_exp_instance.cpp:61` refuse more; the forward pass is single-sequence).
Infernix's version of the same model supports `--max-concurrency 1-8`, but its docs say the main
gain is shorter queue wait because "requests that decode together share each round's dense weight
reads, but their routed experts barely overlap", and each lane costs 113 MB of state plus 14.6
KB/token of KV out of the expert cache. Before anyone builds batched decode on the 3090, answer:
**does running 2-4 lanes work here without things falling over, and what do we get for it?**

## What to deliver (a results document, not a feature)
Write `docs/maintainer/flash-next-lanes-spike.md` on a branch `spike/qwen4exp-lanes` (docs and any
small measurement tooling only; do NOT implement batched decode or change the Program's behaviour
on the default path; if you need a patched Program for a measurement, keep it on the spike branch
and say so). Open it as a **draft** PR so the numbers are reviewable. It must answer, with
measured numbers wherever possible and clearly marked estimates where not:

1. **Resource cost per lane on this 3090**: VRAM (KV at 32K and 64K, GDN/PLE/QSA state, verify
   records, workspaces, graph memory) from `memory_plan.cpp` and the real startup report; the
   resulting expert-frame loss at C=2/3/4; host RAM cost (anything per-lane in pinned memory?).
   Does it still start inside 24 GB with the desktop attached and ~40 GiB free RAM?
2. **Expert hit rate under interleaved streams.** This is the key unknown: one stream has a
   97-98% decode hit rate with ~11-12K of 24,576 experts resident. With 2-4 independent
   conversations decoding in the same rounds the working set widens. Measure it with real routes:
   capture route traces (the forward's `route_log`, `ForwardExperts::route_log`,
   `program.cpp` expert-cache promotion; see `tests/models/qwen4_exp/test_expert_cache.cpp` and
   `test_forward_real.cpp` for harness patterns) from several *different* real conversations
   (code, prose, tool-call-heavy), then replay them through the real LFRU residency policy
   (`program/expert_cache`, `expert_residency.*`) interleaved as C=1,2,3,4 at the frame counts
   each C leaves (answer 1). Report hit rate, misses per round and the PCIe time they imply at
   the measured link rate (~26 GB/s; frame = 2.64 MiB... verify the record size).
3. **Step-time model**: using the plan's nsys breakdown (`flash-next-plan.md`, decode section)
   and your own measurement where you can get one, predict a round's time at C=2/3/4 (dense
   linears amortise across lanes; HC mixer, GDN, QSA, MoE route scale with columns; experts per
   the answer above; MTP verify columns cost +3.4 ms each of which 2.0 ms is experts) and from it
   aggregate tok/s, per-stream tok/s and queue wait for a second request. A direct measurement
   beats a model: if the forward can run B>1 columns of different sequences cheaply in a test
   harness (the MoE and linear ops take T columns), time that; if not, say what it would take.
4. **Correctness risks**: what in the single-sequence forward, QSA ops (`commit_tail_kernel`
   "one sequence"), GDN state, PLE n-gram rows, the MTP drafter/verification records and the
   elastic VMM KV pool must change; which are mechanical copies of the Qwen3.5 batched paths
   (`src/models/qwen3_5/program/decode.cpp`) and which are new. Sort into: copy, adapt, new.
   List what could "explode": OOM, WDDM overcommit, the cuMemSetAccess NOT_READY crash class
   (plan PR 12), CUDA-graph shapes, expert-cache thrash.
5. **Verdict**: GO / NO-GO / GO-WITH-CONDITIONS for C=2 and for C>2, with the predicted gain and
   the smallest viable first slice. Say what the user gets (throughput, latency for queued
   agents) honestly; a no-go with a clear reason is a good outcome.

## Constraints
- GPU time is shared and RAM allows one Flash-Next process machine-wide: batch your GPU work into
  as few lock holds as you can, and prefer CPU-side replay/analysis of captured traces.
- Another agent is implementing the prefix cache in the same files (`program.cpp`); avoid editing
  them on your branch except for throw-away instrumentation you clearly label and do not intend to
  merge.
- Do not start a PR for a feature. The verdict goes to the user, who decides.
