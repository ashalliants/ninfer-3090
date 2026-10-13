# Task A: context cache (prefix/KV reuse) for Qwen3.8-Flash-Next

Read `brief-common.md` (same folder) first.

## Why
`run.bat qwen38-flash-next` serves one lane with **no prefix or KV reuse**: every request prefills
its whole prompt (~2,350 tok/s, ~13 s for a 30K agent prompt) even when it extends the previous
turn. The Qwen3.8-27B and 35B models already have the full context cache (capture frontiers,
shared-prefix checkpoints, host tier, single-flight, restore overlap) through the shared
`EngineCore`. The Flash-Next Program (`src/models/qwen4_exp/program/program.h`) stubs every cache
hook: `capture_frontiers() -> {}`, `valid_checkpoint -> false`, `start_capture -> false`,
`plan_reclaim -> {}`, etc., and `qwen4_exp_instance.cpp` refuses `--context-store`. Infernix's own
Flash-Next Program implements a hybrid prefix cache (see its `hybrid-prefix-cache-spec.md`); our
port deliberately dropped it to ship the preview. Your job is to bring reuse back.

## Goal
Repeated and extended prompts on Flash-Next resume from a cached checkpoint instead of
re-prefilling: an agent loop (system prompt + growing conversation, tool results appended) should
prefill only the new suffix. Same user-visible behaviour otherwise.

## Required behaviour
1. **Capture** at the frontiers the Engine asks for (`RequestBasePlan::capture_frontiers`,
   `shared_prefix_frontiers`, end of prompt, end of request via `FinishResult::checkpoint`), the
   way the Qwen3.5 Program does it. A checkpoint holds, at one token frontier: GDN conv +
   recurrent state for the 36 GDN blocks, the PLE conv history, the QSA tails (`ForwardState` in
   `execution/forward.h`), the INT8 KV pages and pooled index keys for [0, frontier) of the 12 QSA
   blocks (`ForwardKV`), and the MTP drafter's KV/state if speculation is on (so a resumed
   request can draft immediately, or define precisely what is rebuilt: `mtp_catch_up` exists).
2. **Restore** at admission: `inspect_source` / `start_binding` / `poll_context` bind the deepest
   usable checkpoint; prefill then runs only from the frontier. Prefix identity must be exact
   (token digests, plus media digests so images never alias; reuse the Qwen3.5 prefix-identity
   code, `src/models/qwen3_5/program/prefix_identity.*`).
3. **Memory**: Flash-Next is RAM- and VRAM-starved (33 GiB of pinned experts; the expert cache
   gets all VRAM left over; elastic KV takes expert frames while a request runs). Checkpoints
   must live in host memory, inside a budget that is part of the startup RAM ledger
   (`plan_host_memory` in `qwen4_exp/memory_plan.*`), default conservative and clearly reported at
   startup, refusing/shrinking cleanly rather than risking the 2 GiB Windows reserve. Never take
   VRAM from the expert cache for resident checkpoints without measuring what it costs. Implement
   the Engine's reclaim/demote/release planning honestly (plan_reclaim, plan_releases,
   checkpoint_footprint, physical_usage) so the ResourceManager's accounting is true.
4. **Exactness**: a resumed request must produce the same greedy token ids as the same request run
   cold, with and without MTP, and with an image in the cached prefix. Where chunk-boundary
   numerics make this impossible (note: prefill calls are 4,096 tokens, and Flash-Next uses a
   canonical A8 cast for placement invariance), measure and state the divergence rather than
   hide it; prefer frontiers aligned so resume == cold.
5. **Speed**: measure TTFT for (a) exact repeat of a 30K prompt, (b) prompt + 1K new tokens,
   (c) an agent-rotation style workload (several conversations interleaved; see
   `tools/bench/run_agent_rotation.py`, `bench/agent-replay`), against the cold baseline on the same
   build in the same sitting (ABBA, GPU lock). Report restore time (expect tens of ms for ~0.5 GB
   over PCIe) and what capture costs on the request that makes the checkpoint (it must not
   noticeably slow decode).
6. **Defaults**: if all gates pass, turn the cache on in the `qwen38-flash-next` launcher profile
   by default (an opt-out env var), remove the corresponding refusals in
   `qwen4_exp_instance.cpp`, update `docs/serving.md`, the README Flash-Next section and
   `docs/maintainer/flash-next-plan.md`. `--context-store` (persistence across restarts) is a
   second step: do it only if the in-memory cache is done and green; otherwise say what remains.
7. **Concurrency stays 1** in this task (another agent is evaluating multi-lane separately, and
   multi-lane will be built on top of your work). Do not paint the design into a one-lane corner:
   keys, handles and ownership should not assume lane 0 where the contract passes a lane.

## Tests / gates
- CPU-only unit tests for the checkpoint registry/accounting/eviction (add to
  `tests/models/qwen4_exp/`), wired into `tests.cmake` the way the existing ones are.
- Real-model tests under the GPU lock (`ninfer_qwen4_exp_engine_real_test`, vision real test):
  resume == cold ids, MTP on/off, image prefix, eviction under a tiny budget, cancel/abort during
  capture, request error after capture, repeated start/stop (expert-state warm start must keep
  working).
- A serving smoke through `run.bat qwen38-flash-next`: chat completions streaming, two
  conversations alternating, tool call, an image, graceful shutdown.
- Existing Flash-Next tests and `ninfer_tests` filtered to the Qwen4Exp / context-cache tests must
  still pass. The full 159-test suite takes ~13 minutes; run it once at the end under the lock.
- No regression: decode tok/s and cold prefill tok/s within noise of master on the same sitting.

## Deliverables
Branch `feat/qwen4exp-context-cache` (one PR, or two if it splits at a natural stand-alone seam:
list them first per PR_POLICY), PR opened against master with the measurements and disclosures.
Plan stages in your report (what is done, what is left).
