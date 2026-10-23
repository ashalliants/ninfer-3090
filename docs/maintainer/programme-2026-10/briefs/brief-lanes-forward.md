# Task C: Flash-Next forward over B rows (slice 1 of multi-lane), with a stop-loss gate

Read `brief-common.md` (same folder) first. Then read the spike results, which are your design
input: `C:\ninfer-fork\ninfer-3090\.claude\worktrees\linked-baking-piglet\docs\maintainer\flash-next-lanes-spike.md`
(branch `spike/qwen4exp-lanes`, draft PR #284; sections 3, 4 and 5 matter most; its tooling is in
`tools/flash_next/lanes_spike/` and `ninfer_offloaded_moe_bench --mode lanes`).

## Why
The user wants Flash-Next to serve 2 requests at once. The spike found C = 2 plausible: predicted
~109 tok/s total against 92.6 for one MTP stream (+18%), +28% shorter mean wait for queued agents,
C > 2 not worth it. The expert hit rate holds at C = 2 (93.3%), so the open risk is the one number
the spike could only estimate: the **non-expert cost of an extra row in a batched decode round**
(estimated 1.4 ms per extra column from single-sequence profiling). This task builds the piece that
measures it, and is the stop-loss for the whole feature.

## Scope: the forward pass only
Make `execution::Forward` (`src/models/qwen4_exp/execution/forward.{h,cpp}`) able to decode B
independent sequences in one call (plain decode, one token per row, B in 1..4; B = 1 must stay
byte-identical to today). Nothing in the Program, the Engine or the launcher changes: the
`--max-concurrency 1` refusal stays, and the default serving path must be bit-for-bit unchanged.
This is a self-contained module with its own tests that can merge to master on its own.

What the spike's section 4 table says you need (copy / adapt / new):
- Per-row state slots: add new batched types (for example `ForwardStateBatch`, `ForwardKVBatch`
  with per-row block tables) **next to** the existing single-sequence `ForwardState` / `ForwardKV`
  instead of changing them. Another agent is adding checkpoint capture/restore to the Program on
  top of the single-sequence structs; do not break their layout or semantics.
- GDN decode + causal conv with per-row slots (NInfer's B-slot one-token `gated_delta_net` update
  from the Qwen3.5 path exists; the model's tiled 16-to-48 key-head layout is the Flash-Next part).
- PLE convolution with per-row history/slot (Infernix `ple` Op has slots; NInfer's is single-row).
- QSA: index query, key pooling, block selection, attention, tail commit with per-row block tables,
  tails and positions. Adapt Infernix's batched QSA Ops (`infernix-ref` clone, Apache-2.0, keep
  attribution) as **new** NInfer Op variants; do not change the existing single-sequence ops'
  behaviour. Each new op needs its FP64/CPU oracle test per `docs/maintainer/op-development.md`.
- n-gram rows per row (host side), MoE/HC mixer/dense linears already take T columns: B rows
  decode as T = B columns with per-row routing; check the route log/promotion budget handles B
  tokens per round (the spike replay used a budget over C tokens).
- Plain decode only. No MTP verification, no prefill batching, no vision.

## Gates (report each with numbers)
1. **Correctness**: B = 2, 3 and 4 decode of unrelated prompts gives greedy token ids equal to
   each prompt decoded alone through the existing path (extend `test_forward_real`; the real
   model is available; routes are column-invariant up to 8 columns per the spike). Several steps
   (>= 64 tokens) and prompts of different lengths, including one that crosses a QSA block
   boundary (blocks of 4) and one deep into the budget (> 2,048 tokens of selection).
2. **B = 1 unchanged**: ids identical, and decode step time within noise of master in the same
   sitting.
3. **The number the spike needs**: time a full decode round (all 48 layers, real expert cache
   state warmed by a prompt) for B = 1, 2, 3, 4 with the real kernels, ABBA with B = 1, report the
   median per-round time and the **non-expert cost per extra row** (subtract the miss cost using
   the measured 43.6 us per miss from the spike). Then plug the measured values into the spike's
   queue/throughput model (`tools/flash_next/lanes_spike/`) and report predicted aggregate tok/s at
   C = 2 against the one-lane MTP baseline measured in the same sitting.
4. **Stop-loss**: if predicted C = 2 aggregate is below 1.15x one-lane MTP (the spike's gate), or
   the extra-row cost is more than about 2.5 ms, stop: finish and document the measurement, open
   the PR as a draft "measurement only" if code is not worth merging, and say so clearly in the
   report. Do not push on into Program work. A clean negative result is a good outcome.

## Deliverables
Branch `feat/qwen4exp-batched-forward`, one PR against master (body: the three measured numbers,
the oracle results, what is unverified, what the next slice is). Update
`docs/maintainer/flash-next-lanes-spike.md`'s section 5 only through your PR (add a short
"measured" note), not by editing the spike branch.

## Coordination
- Another agent works on `feat/qwen4exp-context-cache` touching `program/program.{h,cpp}` and
  `qwen4_exp_instance.cpp`. Stay out of those files (and out of `forward.h`'s existing
  single-sequence structs). If you must touch a shared file, keep the diff minimal and mention it.
- GPU lock discipline as in brief-common.md; at most one Flash-Next process machine-wide (the
  other agent also needs the GPU: keep your holds short and batched). Ops tests are CUDA tests too
  and need the lock.
