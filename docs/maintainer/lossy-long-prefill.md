# Lossy long-prompt prefill: design

Status: design, nothing built. A temporary plan in the sense of `AGENTS.md`: fold its stable facts into
`docs/serving.md`, `docs/maintainer/engine-architecture.md` and
`docs/maintainer/resource-scheduling-and-context-cache.md` when the work lands, then delete it.
Numbers marked **(predicted)** come from the fitted prefill cost model (`tools/bench/prefill_time_model.py`,
`docs/performance.md`) and have not been measured.

## Why

The fleet serves Qwen3.5/3.8-27B (dense, 64 layers: 48 GDN + 16 full attention) from several RTX 3090
boxes behind a router with sticky cache affinity. An uncached prompt of 100K+ tokens costs about 55 s of
prefill and about 3 GB of int8 KV for that whole minute. The scheduling work (#340-#352: layer slices,
yield, shared anchors, backlog gauge) keeps neighbouring streams moving, so the remaining damage is:

- the box is effectively busy for a minute, the router overflows sticky traffic to a second box, and a
  second cold prompt there does the same again;
- the prompt holds a large share of the KV pool while it prefills, leaving the router less free capacity.

Production runs `--kv-dtype rk4v4` (about 18 KB/token, so ~1.8 GB at 100K); the int8 figures elsewhere in
this page are the benchmark configuration. Everything here must work, and be measured, with `rk4v4`.

Lossless speedups on one card are nearly exhausted: the prefill GEMMs run at about two thirds of the
3090's int8 tensor peak and prompt attention is already an int8 FlashAttention-2 kernel
(`prompt_i8_fa2`). At 100K, GEMMs are about 60% of the time and attention about 40%
(`long-prefill-handoff.md`, measured). A large gain therefore needs **less work**, not faster work.

## What success looks like

- A 100K-token uncached prompt on the 27B finishes prefill in **about 15-20 s instead of ~55 s**
  **(predicted)**, and holds **about a fifth of the KV** afterwards.
- Answers on long prompts stay close to full-prefill quality on the gates below.
- The follow-up turns of the same conversation hit the cache like any other conversation does.

## Goals

1. **Token selection ("scout and select").** For a request whose *uncached* prompt suffix is longer than a
   configurable threshold (the user's production value: 30,000 tokens), the 27B itself runs its first
   `L` layers over that suffix ("scout pass"), scores each token by the attention the prompt's last
   tokens pay it at layer `L`, keeps the best fraction, and then prefills only the kept tokens through the
   whole model ("main pass"). The kept tokens keep their **original** RoPE positions; generation continues
   at position = original prompt length. This is GemFilter's two-pass scheme
   ([arXiv 2409.17422](https://arxiv.org/pdf/2409.17422)), chosen over FastKV's continue-from-layer-L
   ([arXiv 2502.01068](https://arxiv.org/abs/2502.01068v4)) because the main pass is then an ordinary
   prefill of a shorter sequence, the cache holds the same token set in every layer, and any error in the
   scout can only change *which* tokens are kept.
2. **Int4 scout arithmetic.** The scout pass's GEMMs run with int4 activations on Ampere int4 tensor cores
   (`mma.sync.m16n8k64.s4`), roughly 2x int8's peak. Because the scout only produces a ranking, its
   rounding never reaches the model's output; the main pass stays on today's exact int8 routes.
3. **Conversation continuity.** The next turn (original prompt + reply + new message) reuses the
   selected state instead of scouting the 100K again.
4. **Observability.** Counters, request-log fields and the backlog gauge reflect the lossy path, so the
   router and the operator can see it.

Order of work: goal 1 with int8 scout (and goal 3, 4), then the quality sweep and a go/no-go, then goal 2.

## Non-goals (v1)

- Multimodal prompts (Vision items): the lossy path is skipped, the prompt prefills normally.
- The 35B-A3B MoE and the Qwen4-exp/Flash-Next Program. The option is accepted only by the Qwen3.5 dense
  Program; others refuse it at startup like other Qwen3.5-only flags (`qwen4_exp_instance.cpp:200`).
- A per-request switch in the HTTP API. The behaviour is server policy; A/B is done by running a server with
  and without the flag. (No external protocol contract changes in v1.)
- Persisting the selection alias (goal 3) through the context store or to peer boxes. On another box the
  conversation is scouted again.
- Pipeline-split (multi-GPU) models.
- Training anything (gist tokens, SwiftKV-style distillation).

## Anti-goals (a failure even if it technically works)

- **A lossy state served as an exact prefix.** A request whose prompt was never selected must never get a
  selected state through ordinary prefix matching, and the reverse. Exactness is part of the cache's
  identity.
- **Quietly on by default for everyone.** Repo rule: risky features are opt-in. The engine flag defaults to
  off; production turns it on through its launcher profile.
- **Slower cold prompts in the bad case.** If the selection keeps most tokens (or the prompt is barely over
  the threshold), the scout's cost must not make the request slower than plain prefill by more than the
  scout pass itself (~25% at `L`≈16); see the "skip" rule below.
- **Breaking agents' exact edits.** Coding agents reproduce exact text from files they read long ago
  (`Edit` with an `old_string`). A selection that drops the middle of a file the agent then edits gives a
  failed tool call. This is measured explicitly (agent replay gate) and is the most likely reason to
  reject a setting.

## Constraints from the code

These were checked by reading the code (file:line at this tree's master, 0312a1d7).

- **Ledger = KV rows.** Decode readiness requires `ledger.size() == ledger_frontier ==
  execution_frontier + 1`, and `prefix_identity` / `prefix_digests` have one entry per KV row
  (`program/decode.cpp:281-289, 975-980`, `commit.cpp:173`, `prefill.cpp:328-338, 802`). Checkpoint
  frontiers are both KV row counts and ledger lengths (`program_impl.h:263-273`,
  `planning/source.cpp:10-22`). **Decision:** the selected sequence's ledger *is* the kept token list, so
  all of these invariants hold unchanged. The original prompt is not the sequence's identity; the alias in
  goal 3 connects the two.
- **Positions.** KV slot (`cache_positions`, contiguous, drives the causal mask) and RoPE position
  (`rope_positions`) are already separate tensors in attention (`text.cpp:880-885`). Text prefill today
  derives RoPE as slot + `rope_delta`; multimodal prefill passes explicit per-token 3-axis positions
  (`text.cpp:1530-1546`). `ResidentPrefixIdentity` already stores a 3-axis position per token
  (`prefix_identity.h:16-62`). Decode uses `execution_frontier + rope_delta`
  (`decode.cpp:313-316`, MTP `:475`, DFlash `:771`). **Decision:** the main pass gives kept prompt tokens
  explicit RoPE positions (their original indices, same value on all three axes); after the prompt the
  sequence's `rope_delta = original_prompt_length - kept_length`, so generated tokens get the right
  positions through the existing decode path.
- **GDN layers** take no positions; the chunked scan and the short convolution depend only on token order
  (`ops/gated_delta_net.h:59-70`, `chunked/prepare.cu:24`). A shorter, gapped token sequence is a valid
  input. Whether the model's GDN layers still behave well on it is the main unknown (no published work
  prunes tokens in a GDN hybrid) and is what the quality sweep measures.
- **Layer ranges.** `TextContext::run_stage_layers(stage, x, ph, tap, first, last)` already runs a range of
  layers (built for `--prefill-layer-slices`, `text.cpp:1131`).
- **No int4-activation MMA exists** (`ops/common/mma.cuh` has s8 k32/k16 and sm120 e2m1 only). The
  27B's large matrices are `Q4_G64_FP16` (gate_up, the Q4 halves of the input pairs) and Q4/Q5_G64 (down,
  attention/GDN out) (`execution/parameters.cpp:139-202`). A 64-wide K group matches one `m16n8k64`
  instruction.

## How it works

### Trigger

At admission, after prefix reuse and hydration are resolved: let `C` be the reused (exact) prefix and `U`
the uncached suffix. The lossy path runs when all hold:

- `--lossy-prefill-min-tokens N` is set (`0`, the default, means off) and `|U| > N`;
- the prompt has no Vision items, the model is the Qwen3.5 dense Program, and it is not pipeline-split;
- no selection gap map (goal 3) already covers the prompt;
- **the conversation itself is not cached**: the reuse source is nothing or only a shared (product-prompt)
  checkpoint, never a private conversation record. A large new turn inside a warm conversation (a 40K
  file read, say) always prefills in full. (User's decision, 2026-10-25: such a turn counts as cached, and
  is rare in real use; the case this is for is a cold start or a router overflow to a box without the
  conversation.)

Otherwise the prompt prefills exactly as today.

**The product prompt is never selected (my call).** On a cold box the shared product prompt (system
messages, tool definitions; 1.6-9K tokens per `cache-provider-baseline.md`) lies inside `U`. Before the
scout, the engine prefills exactly up to the last shared marker (automatic or explicit,
`resource-scheduling-and-context-cache.md` §4.3) as an ordinary prefill, publishing its shared checkpoint
as today; that span becomes part of `C`, and only the rest is `U` (the threshold is checked on that rest).
This keeps tool schemas intact and keeps the box's shared-prefix reuse working for other conversations.

**Capacity and reporting (my call).** Limits tied to positions (`max_context`, the 262,144 RoPE capacity)
apply to the original prompt length. KV reservation for the main pass and decode is by kept length; the
scout's scratch is reserved separately. Usage keeps its current meaning: `prompt_tokens` is the original
prompt length the client sent, `cached_tokens` the exact reused prefix `C` (plus, on later turns, the
rewritten reused span counted in original tokens). The kept count is reported only in the request log and
counters, so the gateway's view of usage does not change.

### Scout pass

Runs layers `0 .. L-1` over `U` in prefill chunks, attending to `C`'s exact KV for the attention layers
below `L` and starting from `C`'s GDN state. At the filter layer `L` (a full-attention layer; the 27B has
them at 3, 7, 11, 15, ...) it computes only K for every token of `U` and Q for the last `q` tokens of the
prompt, and accumulates per-key attention mass: softmax over the keys of `U` per query row and head,
summed over the `q` rows and all heads. (`C` is always kept, so its keys are left out of the scoring; this
also means the scoring never reads the cache's KV dtype. The layer-`L` keys of `U` go to a BF16 scratch
buffer, 2 KB/token, ~200 MB at 100K. My call, made while planning.) Layers above `L`, the MLP of layer `L`, and the LM head do not run.

- The scout must leave **no trace** in the sequence: `C`'s GDN state and KV are what the main pass starts
  from. Its KV rows for layers `< L` and its GDN states go to scratch that is released when the pass ends.
- **Peak KV during the scout** should be about the scouted layers' share (4 of 16 attention layers at
  `L`=15, ~25% of a full prefill's KV), not a full prefill's. If the builder finds per-layer scratch
  impractical, the fallback of reserving full pages for `U` during the scout is acceptable for the first
  version only if it is recorded as a known gap.
  **v1 takes the fallback (my call, made while planning):** a page group holds every Main layer
  (`paged-kv-cache.md`, page-group IDs), so per-layer scratch needs a multi-plane KV view that does not
  exist. The scout writes its attention layers' KV into the sequence's own pages; pages above the kept
  frontier are released when the scout ends. Peak KV is today's, held for the ~15 s scout instead of
  ~55 s, then ~20%. Only the GDN states of layers `< L` are copied to scratch (~40 MB) and restored.
- Scout chunks are ordinary prefill steps for the scheduler (stall budget, yield, layer slices apply as
  they do for any text prefill), and the backlog gauge predicts scout plus main pass.

### Selection

From the per-token scores over `U`:

1. Smooth with a 1-D pooling window, then score fixed blocks of `B` consecutive tokens (keeping whole spans
   matters more for code than keeping isolated tokens; SpecPrefill also keeps chunks,
   [arXiv 2502.02789](https://arxiv.org/pdf/2502.02789)).
2. Always keep: the last `R` tokens of the prompt (the question and recent turns), the first `S` tokens of
   `U`, and every special/template token (role markers, tool-call delimiters), so the chat structure the
   model was trained on survives.
3. Fill up to `keep_ratio * |U|` with the highest-scoring blocks; output kept tokens in original order.
4. **Skip rule:** if the kept count is above `skip_ratio * |U|` (e.g. the forced keeps already cover most
   of a prompt barely over the threshold), keep everything: the main pass is then an ordinary full prefill
   and the request pays only the scout.

All of `L, q, B, R, S, keep_ratio`, the pooling window and `skip_ratio` are server options with defaults
chosen by the quality sweep. Starting points for the sweep: `L` ∈ {7, 11, 15, 23, 31}, `keep_ratio` ∈
{0.1, 0.2, 0.3, 0.5}, `B` ∈ {32, 64, 128}, `q` = 64, `R` = 4,096, `S` = 256.

### Main pass

An ordinary text prefill of the kept token list after `C`, with explicit RoPE positions (original prompt
indices) instead of slot + `rope_delta`. It must stay eligible for everything a text prefill gets today:
layer slices, stall budget, yield, the DFlash2/MTP prefill taps, progress anchors on the *kept* ledger.
(The multimodal positions path is the existing precedent for explicit positions, but routing text through
it would lose layer slices, which refuse multimodal chunks; a text prefill with an optional explicit
position list is the intended shape. Builder's call on the exact form.)

After it, the sequence's ledger is `C + kept(U)`, its identity carries the original positions, and
`rope_delta` is set as above. Everything downstream (decode, speculative decoding, checkpoints, pause
replay) works on that ledger. **Pause replay must re-prefill with the stored positions**, not slot +
`rope_delta` as it does today (`transactions/replay.cpp:17-43`).

### Selection alias (goal 3)

Follow-up turns do not reuse at the end of the previous prompt: they reuse at the typed rewrite points
(`ResponseReplay`, `TurnClosure`) and `recovery_frontier` (`resource-scheduling-and-context-cache.md` §4.1),
and the private record retires shallower points as the conversation goes deeper. So an alias pinned to
one checkpoint would be gone within a turn or two. **Decision (my call):** the mapping is a property of the
lineage, not of one checkpoint.

- A lossy prefill produces a **gap map**: the original-prompt span `U` it covered (start, length, digest of
  the original tokens up to the end of `U`) and the kept token positions inside it. It is small (a few
  hundred KB at 100K tokens at most; run-length encoded over kept blocks it is far smaller).
- Every checkpoint descended from the selected sequence (captures at typed rewrite points, recovery
  frontiers, anchors, generation endpoints, snapshots) carries a reference to its lineage's gap maps.
  The gap map lives as long as any checkpoint referencing it.
- At admission, a prompt whose original tokens up to the end of a gap map's `U` match that map's digest is
  rewritten: that span is replaced by the kept tokens with their original positions, and the remainder
  follows at slot + `rope_delta`. Ordinary prefix matching (including typed rewrite points) then runs on
  the rewritten prompt, so later turns reuse as any conversation does.
- A conversation may accumulate more than one gap map (a second large uncached turn later on, if the
  trigger allows it, see open questions); rewriting applies them in order.
- A prompt that only *partly* matches (diverges inside `U`) gets no rewrite; it is a cold prompt again and
  may be scouted itself.
- Selected checkpoints written to the context store are stored under their real (kept) identity; without
  the alias nothing will match them, which is correct, only wasteful. Persisting aliases is out of scope
  for v1.

### Int4 scout GEMMs (goal 2, after the go/no-go)

A W4A4 route for the scout pass only, sm86: activations quantized per token per 64-wide K group to signed
int4 with an FP16 scale, weights taken from the existing `Q4_G64_FP16` objects (no new artefact),
`mma.sync.m16n8k64.s4.s4.s32` per group, then group scales applied in FP32 on CUDA cores. Matrices stored
as Q5 keep their existing int8 route in the scout (or a builder-chosen alternative). No rotation
(QuaRot-style Hadamard) in the first version: it only matters if outliers spoil the *ranking*, which the
gate below measures. Prior art for the 3090: QuaRot (CUTLASS int4, ~2x over FP16 at block level,
[arXiv 2404.00456](https://arxiv.org/pdf/2404.00456)), APEX4 (W4A4 g128, 2.0-2.5x kernel speedup on the
3090, [arXiv 2606.08761](https://arxiv.org/pdf/2606.08761)). Against our int8 route the expected gain on
the scout's GEMMs is about 1.6x **(predicted)**.

The Ops are `quantize_a4_g64` and `q4_a4_linear` (`include/ninfer/ops/q4_a4_linear.h`, qualified by
`ninfer_q4_a4_linear_test`; the host oracle is checked by `ninfer_a4_g64_reference_test`). The activation
codes are symmetric in [-7,7] and packed in the Q4 row-split nibble order, so both operands enter the MMA
as stored, and Q4 codes are already two's complement, so there is no zero-point correction. Status
**compiled, not run** on a GPU (2026-10-25). A host simulation of the test fixtures predicts the A4 cast
costs about 7% relative L2 on outlier-free activations (cosine about 0.997 against the A8 route) but about
20% (cosine about 0.98) when every 431st channel is 24x, because such a channel zeroes the other 63 of its
group; whether that spoils the ranking is the Jaccard gate's question.

As built (routing, 2026-10-25, **compiled, not run**): `--lossy-prefill-scout-a4` gives the scout its own
copies of layers `[0, L]` (`execution::scout_a4_layers`, held by the Program's scout buffers) in which every
Q4 matrix already on the integer-A8 prefill route carries `LinearPolicy::ScoutA4Int`; the model's own
parameters never hold it (asserted when they are built, and `run_stage_layers` refuses the copies outside
the scout's prefill pass). Under that policy, at exactly the widths the A8 route serves (full 128-token tiles):

- `linear_swiglu` (gate_up, Q4): `quantize_a4_g64`, one `q4_a4_linear` over all 34816 gate/up rows into a
  staged plane, then `silu_mul` over its halves (the A8 route fuses SwiGLU; this one cannot).
- `linear_add` (down and the mixer outputs, where Q4): `quantize_a4_g64`, `q4_a4_linear`, `residual_add`.
  Q5 matrices are not promoted and keep their own route (A8, or cuBLAS under `--prefill-cublas`).
- split `attn_input_proj`: q and k from row slices of the Q4 parent through `q4_a4_linear` straight into
  their tensors; gate and v from the Q5 parent through the A8 route (its own s8 activations), also
  under `--prefill-cublas`, whose route takes the pair whole.
- split `gdn_input_proj`: query/key rows through `q4_a4_linear` into a staged `[4096,T]` plane copied
  into the head of each `qkv` column (`cudaMemcpy2DAsync`); value and z through the A8 route.

Any other Op rejects the policy (`valid_linear_policy` excludes it). Unaligned final chunks take whatever
the int8 scout takes there (A16). The scout's workspace plan adds the int4 planes only when the option is on,
and startup refuses the option when no Q4 matrix is on the integer route (no `--prefill-a8`). Gate
evidence: `ninfer_qwen3_5_scout_a4_kept_overlap_real_test` (32K and 100K haystacks: kept-block Jaccard
over the score-chosen blocks >= 0.9, the forced head and recent blocks left out, which both scouts keep
identically; the all-blocks Jaccard is printed for reference; pass key kept by both), the `PERF ... scout` lines of `ninfer_q4_a4_linear_test` (the Op under
`ScoutA4Int` against the same Op under `AllowA8Int`, so gate_up's separate SwiGLU is counted), and the
sweep's `--workloads jaccard`.

### Alternative strategy: deterministic conversation summary (`--lossy-prefill-strategy summary`)

Added 2026-10-25 at the user's request as a flagged option for comparison, not the production choice.
It ports the rule-based projector of [hermes-vcc-engine](https://github.com/danbiocchi/hermes-vcc-engine)
(Apache-2.0, commit `31ee4fe7`, after the VCC paper, arXiv 2603.29678) to C++. Instead of thinning tokens,
it rewrites the *messages*: it keeps the head (system/developer messages, tool definitions and the first
3 messages) and the tail (the last 20 messages, at least one real user message and everything after it, never
splitting a tool call from its result), and replaces the middle with one User message holding the
projector's sections (goal, errors and fixes, error names, relevant files, commands run with exit codes,
and a session log with `[#N]` references to message indices). The output is deterministic and makes no model call.

- **Trigger:** the same as the token strategy (enabled, cold, and more than `min_tokens` after the product
  prompt). The rendered original is prepared first to count tokens; a thread-safe reuse probe decides
  coldness; if it fires, the summarized `PromptInput` is prepared and admitted as an ordinary exact prompt
  (regular positions, no scout, no gap map).
- **Continuity:** a bounded in-memory registry maps the exact original message prefix that was summarized
  (the head plus the middle, compared exactly, with a digest only as a shortlist) to the summary text. A later
  request whose messages start with that prefix gets the same summary substituted, whatever its length, so
  it renders the same tokens and hits the cache as an ordinary conversation.
- **Budget:** the summary is at most `min(0.06 × max_context, 8000)` tokens and at least 1,500
  (`--lossy-prefill-summary-max-tokens`), as in the original.
- **Usage and log:** `prompt_tokens` is the original rendered length; the request log records
  `lossy_prefill.strategy = "summary"` and the summarized message range.
- **Correctness reference:** the C++ projector (`src/runtime/engine/compaction/vcc_projector.cpp`) must
  reproduce the Python projector's output byte for byte on recorded transcripts: `tools/bench/vcc_golden.py`
  writes the golden files in `tests/fixtures/vcc` from the pinned commit, `ninfer_vcc_projector_test`
  checks them, and `vcc_golden.py --fuzz N --out DIR` with `NINFER_VCC_FIXTURE_DIR=DIR` runs a random
  differential set.
- **Tool names:** the projector knows Hermes-style tool names only (`terminal`/`shell`/`bash`/`run_command`
  for commands; `read_file`, `patch`, `write_file`, `edit_file`, ... for file verbs), case-sensitively.
  Claude-Code-style `Bash`/`Read`/`Edit` calls still appear in the log and file list (by their `file_path`),
  but not under "Commands Run", and their files read with the tool's own name as the verb.
- **Known weakness:** it drops file contents from the middle; Hermes relies on a `session_search` tool to recover
  them, which our clients do not have. The agent-replay `Edit` metric is where the two strategies are compared.

## Expected numbers (predicted, 27B, 100K uncached, chunk 4096)

Baseline split: GEMM ≈ 33 s, attention ≈ 22 s (55 s measured total, 40% attention measured).
Scout at `L` = 16 costs ≈ a quarter of a full prefill; main pass on 20% costs ≈ 0.2 × GEMM + 0.04 × attention.

| setting | prefill | KV after (int8 / rk4v4) |
|---|---|---|
| today | ~55 s | ~3 GB / ~1.8 GB |
| int8 scout `L`=16, keep 20% | ~19-21 s | ~0.6 GB / ~0.36 GB |
| int4 scout `L`=16, keep 20% | ~16-18 s | ~0.6 GB / ~0.36 GB |
| int4 scout `L`=8, keep 20% (if the sweep allows a shallow filter) | ~11-13 s | ~0.6 GB / ~0.36 GB |

## Verification

### Correctness (must pass before any quality number means anything)

- **Identity at keep-all:** with `keep_ratio = 1` (and forced keeps covering everything) the lossy path's
  output, KV, GDN state and greedy tokens are bit-identical to an ordinary prefill. This tests that the
  scout leaves no trace and that explicit positions equal the default ones.
- **Gapped positions:** a GPU test that prefills a kept list with explicit positions and checks the KV and
  logits against a reference computed by the existing per-token positions path (multimodal-style 3-axis
  positions with all axes equal).
- **Alias:** fake-engine tests (`tests/runtime/fake_engine.h`) for: the follow-up turn rewrites and hits
  the cache; a diverging prompt does not; an exact (never-selected) request never receives a selected
  checkpoint; alias removal with its checkpoint; cancel during scout and during main pass leaves nothing
  behind; yield/pause during either pass resumes correctly (replay with stored positions).
- Speculative decoding (DFlash2 one-lane default, MTP two-lane) after a lossy prefill produces the same
  greedy tokens as non-speculative decoding after the same lossy prefill.
- Usual suites per `CLAUDE.md`: `-CpuOnly`, then the `qwen3_5_*` and engine-scheduling GPU tests.

### Quality sweep and gate (decides defaults and go/no-go)

Workloads, each run with the flag off (baseline) and over the sweep grid:

1. **Needle in a haystack** (`eval/configs/*needle_haystack*`, evalscope) at 32K, 64K, 100K and 5 depths.
2. **Agent replay** (`tools/bench/agent_replay`, 30K-120K contexts): tool-call validity, and for `Edit`
   calls whether `old_string` occurs in the file (exact-copy fidelity), plus output divergence from
   baseline.
3. **Teacher-forced divergence:** per-token NLL and KL against the full-prefill model on the real next
   tokens of long prompts, which is cheap, sensitive and needs no judge.
4. A long-context QA set available through evalscope (LongBench-style) if it can be run locally.

Acceptance targets (agreed with the user 2026-10-25; fixed before the sweep, not changed after seeing
results; the final go/no-go is the user's, on the PR's measured numbers): needle accuracy ≥ 95% of baseline at every length; agent-replay `Edit` exact-match rate within
2 points of baseline; prefill time on a 100K prompt ≤ 40% of baseline.

Int4 scout gate: kept-block overlap with the int8 scout (Jaccard, over the score-chosen blocks: the forced
head and recent blocks are left out, 2026-10-25) ≥ 0.9 on the sweep prompts, the quality
gate above unchanged within run-to-run noise, scout GEMMs ≥ 1.4x faster than int8.

### Speed

`tools/bench/run_prefill_stall.py` and the depth sweep with the flag on: prompt total time, scout and main
pass times separately, neighbour worst gap with layer slices on, peak KV pages during scout and after.

## Observability

- Counters: lossy prefills started/skipped/completed, tokens scouted, tokens kept, alias hits.
- Request log (`--request-log-jsonl`): `lossy_prefill: {uncached_tokens, kept_tokens, filter_layer,
  scout_ms, main_ms}` when used. `--request-log-lossy-detail` adds `kept_runs`, the kept offsets of U as
  `[begin, end)` pairs (all of U when skipped, none when declined), which the sweep's Jaccard reads.
- `ninfer_prefill_backlog_seconds` / `/v1/load` predict the lossy cost for requests that will take the path.

As built (engine): `ninfer_lossy_prefill_total{outcome="completed|skipped|aborted"}`, where
`skipped` also counts a request that declined because another lane held the one scout (the
request log tells them apart: `lossy_prefill.outcome` is `completed`, `skipped`, `declined` or
`aborted`); `ninfer_lossy_prefill_scouted_tokens_total`, `ninfer_lossy_prefill_kept_tokens_total`
(not for declined requests, which scout nothing) and `ninfer_lossy_prefill_alias_hits_total`
(admitted prompts the Program rewrote onto a lossy lineage). Prompt progress and the cached-token
count are reported in the client's tokens (`RequestBasePlan::reported_frontier`). The backlog
prices a lossy request as the exact part to M, the scout over U at `Program::lossy_scout_share()`
of a full prefill, and the main pass over `keep_fraction` of U; a queued request is priced that way
when it is long enough and has no media.

Pausing during the scout (as built): `start_pause` refuses while the scout runs, so the Engine
never asks. A pause that only serves the queue (`--yield-long-prefill-ms`) skips a scouting lane
and looks again on the step that ends the scout. A pause for memory ends the scout first
(`Program::abort_lossy_scout`: the lane goes back to M exactly as the scout found it, and the
request continues as an ordinary full prefill of U), then pauses as usual; it counts as `aborted`.
Cancelling during the scout aborts the lane, which drops the scout and its pages.

## Options (names are proposals; builder may adjust)

| option | default | meaning |
|---|---|---|
| `--lossy-prefill-min-tokens N` | `0` (off) | uncached suffix length above which the path runs; production profile sets 30000 |
| `--lossy-prefill-keep F` | from sweep | fraction of the uncached suffix kept |
| `--lossy-prefill-filter-layer L` | from sweep | full-attention layer used to score |
| `--lossy-prefill-block B`, `--lossy-prefill-recent R`, `--lossy-prefill-head S` | from sweep | selection shape |
| `--lossy-prefill-scout-a4` | off until its gate passes | int4 scout GEMMs |

Startup refuses the option for non-Qwen3.5-dense Programs and pipeline-split models. The production
launcher profile is updated in the same change that sets defaults.

## Left to the builder

- The exact admission/transaction shape: whether the scout is a separate transaction or a phase of the
  prefill transaction, and how `resolve_prefill_progress` and `BeginSummary` account for a prompt whose
  length changes after the scout.
- How scout scratch (GDN states for layers `< L`, KV rows for the scouted attention layers) is allocated.
- The attention-mass kernel (a two-pass softmax over all keys for `q` query rows, reduced over rows and
  heads), and whether it can reuse `prompt_i8_fa2` pieces.
- Whether pooled block scores are computed on device or host (100K floats is small either way).
- The int4 kernel's tiling, and how Q5 matrices are handled in the scout.

## Risks

- **GDN on gapped input** (the main unknown). Mitigations to try in the sweep: larger blocks, more recent
  tokens kept, deeper filter layer.
- **Exact-copy fidelity** for agents (anti-goal above).
- **KV dtype.** The scoring at layer `L` uses the scout's own K for `U` (before KV quantization) and reads
  `C`'s layer-`L` keys from the cache in its dtype (`rk4v4` in production). Main-pass and decode attention
  already handle `rk4v4` (`prompt_i8_fa2`); the correctness and quality runs use `rk4v4`.
- **Scout scratch memory** reintroduces the KV pressure this is meant to remove if done with full pages.
- **Replay and anchors** assume slot + `rope_delta` positions today; every path that rebuilds a sequence
  from its ledger must carry explicit positions.
