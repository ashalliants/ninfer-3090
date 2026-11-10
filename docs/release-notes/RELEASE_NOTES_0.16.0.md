# NInfer-3090 v0.16.0

**Long prompts are read faster, agent turns that copy text back decode up to 87% faster, cached conversations are reused far more often, and Qwen3.8-Flash-Next can now be served on a 24 GB card. Upgrading needs action if you start `ninfer-serve` with your own flags or call `/slots`: about 15 serving flags, the `/slots` endpoint, several launcher settings and the old `/metrics` names are gone.**
About 75 merged pull requests since v0.15.1 (#212 to #307).

## Breaking changes: read before upgrading

The context cache was replaced by a new engine (#214). It sizes itself, so the knobs that used to
size the old one no longer exist. **`ninfer-serve` stops at startup with `unknown argument: <flag>`
if you pass a removed flag**, so a stale launch script fails loudly instead of running differently.

| Removed | What to do instead |
|---|---|
| `--host-state-slots`, `--max-shared-prefixes`, `--max-private-continuations`, `--auto-prefix-grid`, `--host-kv-mib`, `--max-long-anchors-per-continuation`, `--auto-long-anchors`, `--progress-anchor-tokens`, `--max-cache-markers-per-request` | Delete them. One pinned Host budget now holds every cached conversation: `--host-context-mib N` (default 8,192 MiB plus 8 StateImages), or `--auto-host-cache` to size it from the RAM still free. |
| `--max-prefill-lanes`, `--prefill-max-skip`, `--decode-rounds-per-prefill` | Delete them. The scheduler decides how prompts and decoding share the card. (v0.15.1's default of reading two long prompts at once no longer applies.) |
| `--output-reservation-tokens` | Delete it. A request now takes KV pages as it decodes. |
| `--lookup-ngram` | `--ngram-draft-tokens` and `--ngram-min-match` (see below). |
| `--slot-save-path` and the `GET` / `POST /slots` endpoints | `--context-store DIR` keeps cached conversations across restarts (off by default, see below). |

Other changes that can break a script, dashboard or client:

- **`/metrics` series are renamed.** `ninfer:*` became `ninfer_*` (Prometheus reserves colons), for
  example `ninfer:requests_total` is now `ninfer_requests_total{outcome="completed"}` and
  `ninfer:prefix_cache_hit_tokens_total` is `ninfer_prompt_tokens_cached_total`. The old names are not
  emitted and there is no alias period. The `llamacpp:*` series are unchanged. The full old-to-new
  table is in the description of #214. Counters now count from when the server attached its engine
  after warmup, not from process start. Update dashboards and alerts.
- **Launcher environment variables.** Removed from `run.sh` / `run.bat`: `NINFER_HOST_STATE_SLOTS`,
  `NINFER_FALLBACK_BASE_CHUNK`, `NINFER_FALLBACK_BASE_CONTEXT`, `NINFER_FALLBACK_BASE_SLOTS` and
  `NINFER_FALLBACK_RUNG`. The launcher now starts at the largest context that fits instead of failing
  and reloading, and `NINFER_FALLBACK=off` still turns that off. The host budget is set with
  `NINFER_HOST_CONTEXT_MIB`; `NINFER_AUTO_HOST_CACHE=on` and the `NINFER_HOST_CACHE_*` limits still work.
- **Tool calls are grammar-constrained by default** (`tool_constraints: "basic"`). A client that sends
  `stop` strings or `ignore_eos` without setting `tool_constraints` falls back to free tool calls
  rather than getting an HTTP 400.
- **API changes from the spec audit (#289, #306).** A string message `name` is now accepted and
  ignored on every role instead of returning 400. `safety_identifier` may be up to 128 characters
  (was 64). A Responses request with `async: true` on a function tool is rejected with
  `async_tools_not_supported`; Anthropic `thinking.type: "between_tools"` returns
  `thinking_type_not_supported`.
- **Request-log schema** is now v29. Anything that parses the JSON request log should accept the new
  fields.
- **Builds from source:** `ninfer_core` and `ninfer_ops` are built without relocatable device code by
  default (`NINFER_CUDA_RDC=OFF`, #307). The release binaries are built this way. Measured on a
  27B benchmark the speed is the same within about 1%, but the compiler produced different register
  allocation for roughly a third of the kernels, so this is a codegen change.

## What this means for you

- **You run `run.sh` / `run.bat` with the stock profiles.** Upgrade and do nothing. The launchers were
  moved to the new Host budget and n-gram copy drafting is on by default in the DFlash2 profiles. If
  you override launcher environment variables, read the breaking-changes list first.
- **You start `ninfer-serve` with your own flags, or scrape `/metrics`, or call `/slots`.** Go through
  the breaking-changes list above. Each removed item has a replacement or can simply be deleted.
- **You drive coding agents on a 27B and read long files and codebases.** Long prompts prefill faster
  and agent turns that write text back from the prompt decode faster, with no flag to set.
- **You run several agents or tenants on a shared box.** The new engine keeps far more of each
  conversation cached (hit rate 32.7% to 78.6%, first word 8.8 s to 2.9 s mean in the test below), and
  you can now add a disk copy that survives restarts.
- **You have no spare VRAM or RAM.** Qwen3.8-Flash-Next and the NVMe checkpoint tier let a conversation
  resume from local disk.

## Faster on the Qwen3.8-27B

RTX 3090, launcher settings, interleaved runs in one sitting, v0.15.1 against master (#280):

| | v0.15.1 | this release | change |
|---|---:|---:|---:|
| reading a 512-token prompt (tok/s) | 1,985 | 2,011 | +1.4% |
| reading a 4,096-token prompt | 3,000 | 3,044 | +1.5% |
| reading a 16,384-token prompt | 2,651 | 2,815 | +6.2% |
| reading a 65,536-token prompt | 1,823 | 2,172 | +19.1% |
| writing 128 tokens that copy the prompt, 2K context | 242.9 | 454.1 | +87% |
| writing 128 tokens that copy the prompt, 32K context | 253.6 | 403.2 | +59% |
| writing 512 tokens from a one-token seed | 62.95 | 60.10 | **-4.5%** |

- **Copy drafting (n-gram).** When the model is about to write text that already appeared in the
  request (an edited file, a tool result, `cat -n` output), it proposes up to 15 copied tokens at once
  and checks them. A wrong copy costs speed, never output. It is on in the one- and two-lane DFlash2
  launcher profiles; the flags are `--ngram-draft-tokens` and `--ngram-min-match`. On a replay of four
  whole agent sessions (up to 117K tokens) decode was 42% faster at one lane, with turns that explain
  rather than copy unchanged to 5% slower.
- **The cost.** Plain generation from a one-token seed is 4.5% slower. It is not the round cost: the
  stream accepts slightly fewer tokens per round after the attention kernels were retuned (#240). On 60
  real chat prompts of 59-76 tokens the change in total rounds was -0.5%, within noise, so we treat
  it as neutral for chat, but it was measured on one card with greedy thinking-mode prompts only.
- Prompts of 129-511 tokens now use cuBLAS (#228) and long prompts use a new INT8 attention kernel
  (#226, #237).

## A cache that keeps conversations (new engine)

Measured on the 27B with four agents on two lanes and histories of about 40K tokens (#214):

| | cache hit rate | first word, mean | first word, worst |
|---|---:|---:|---:|
| before | 32.7% | 8.79 s | 24.7 s |
| after | 78.6% | 2.86 s | 5.13 s |

- The gain needs both the new engine and the removal of the old Windows cap on pinned host memory.
  The new engine with the cap still in place measured 20.6% hits, worse than before.
- Under memory pressure the scheduler drops cached checkpoints first, then pauses a younger request and
  resumes it later from a RAM snapshot or by replaying its tokens.
- Speed is unchanged on the usual rows (within 1% on prefill, plain decode, DFlash2 and MTP) and
  perplexity is bit-identical.
- Pinned host memory is no longer limited to (free VRAM - 1 GiB) / 2. The launcher profiles now pin
  about 8 GiB, and pinning costs about 0.5 GiB of free VRAM.

### Keeping the cache across restarts (`--context-store DIR`, off by default)

- Conversations are written to disk after 30 s idle and at shutdown, and a restart restores them.
  An optional S3-compatible bucket can hold a second copy.
- **Disk-aware eviction (#295).** Conversations pushed out of RAM are written first, so they can come
  back from disk. With a small RAM budget (2 GiB) this lifted the hit rate by 0.14 to 0.42, halved
  the 95th-percentile first word or better, and cut the run time by 8-50% in the test mix. With 8 GiB
  of RAM it made no measurable difference, while still writing 340-460 GB an hour, so on a large RAM
  budget turn it off unless you need restarts.
- Disk write volume is the cost to watch on SSDs. A filter that skips re-writing shared prefixes cut
  those writes from 77 to 3 per run.
- Hydration from disk is priced by a learned read speed (#287).

## Qwen3.8-Flash-Next on a 24 GB card (Windows preview)

The 512-expert Qwen3.8-Flash-Next at 2 bits per weight, with the experts pinned in host RAM (about 33
GiB). Measured on the same RTX 3090 (README):

- **Plain decode is 1.15-1.18x Strata's** on the same GGUF (81 / 80 / 78 tok/s after a 1K / 8K / 32K
  prompt against 71 / 70 / 66). **With MTP 109-113 tok/s**, identical tokens to plain decoding.
- **Prefill 2,345 tok/s at 4K**, 1.23x Strata's, with the n-gram volume on NVMe (`NINFER_NGRAM_VOLUME`).
- **Image input** works, and a pinned downloader fetches the model (`download-model`).
- **It reuses prompts now (#285).** A repeated 24K prompt answers its first word in about 0.12 s
  instead of 10.6 s, and cold prefill and decode are unchanged against the build before.
- **Disk tiers (#296).** `--context-nvme-mib N` keeps checkpoints in a scratch file on local disk
  (removed at exit), so a machine with little spare RAM still reuses prompts: about 0.2 s per repeated
  24K prompt from disk-only against 0.12 s from RAM. With `--context-store`, a restarted server answered
  the 24K request in 0.21 s (RAM) or 0.45 s (disk only), against 10.8 s cold. These restart numbers are
  one run per cell.
- Limits: one request at a time, 32K context, Windows only. It is a preview.

## Agent tool-calling and API

- Custom tools work on Chat Completions (#229) and Responses (#219). `--tolerant-tool-calls`
  (off by default) repairs broken tool-call markup instead of returning it as text, and never keeps a
  call whose argument was cut.
- Several recoveries for tool calls stranded in unclosed thinking, a quoted `</think>`, and calls
  opened inside Markdown code.
- **`--reasoning-loop off|stop|conclude`** (off by default): when thinking keeps repeating whole
  passages, `stop` ends the reply and `conclude` closes the thinking so the model answers. Cost on a 27B
  at 6,000 thinking tokens: under 1% in both single-stream and four-stream runs.
- **Constrained decoding:** `--grammar-file`, `--json-object`, `--json-schema-file`, `--regex` and
  `--choice` on the CLI (from upstream's engine).
- **`--fit-context MIN`** and the launcher: the server starts at the largest context that fits the free
  GPU memory instead of failing and reloading.
- **New Host-allocator metrics (#302)** count failures by cause and report free runs, to show whether
  pinned memory failed from shortage or from fragmentation.

## Also fixed

- Two requests that shared one long prefix now wait for one prefill rather than both reading it (#255).
- A rewritten dialogue keeps its live branch over the abandoned one (#254).
- A crash with `Qwen3.5 resource subtraction underflow` that existed in v0.14.1 to v0.15.1 cannot
  occur on the new engine.
- The default profile starts with more free VRAM (1.80 GiB against 1.08 GiB on the 188K profile).
- Builds on CUDA 13 compile again (#283).
- Cold build time on the 3090 box fell from 14m38s to 6m18s for the attention kernels (#301) and
  another minute with #307. This only matters if you build from source.

## Not verified in this release

- Pipeline stages on two physical GPUs (every multi-stage run used one card twice).
- The context store with a prompt graft installed, and overlay vision plus stages under heavy KV pressure.
- The Qwen3.6-35B tuned profile on the new engine.
- The context store on Flash-Next after the final review rounds of #295; its speed numbers predate them.
- sm_80 and sm_89 builds.
- The full test suite and all real-model tests on this exact commit: the release builds the three
  products only. The pull requests above ran the tests for their own areas.

## Verification

Checked on 2026-10-14 against the release commit, before publishing:

- **Real-card smoke test (Windows, RTX 3090):** the extracted zip started `run.bat qwen38-27b` with only
  Windows on `PATH`, was ready in 8.7 s, answered a chat completion, and `/health` reported
  `0.16.0-rtx3090`. The context was lowered to 150,528 tokens (188,416 asked for) because the desktop
  was using GPU memory, which is the launcher working as designed.
- **Windows zip:** inner and outer checksums match; `ninfer-serve`, `ninfer` and `ninfer_bench` start
  with only Windows on `PATH`.
- **Linux archive:** inner and outer checksums match; a fresh unpack runs `./ninfer-serve --version`,
  `./ninfer-serve --help` and `./run.sh --help`; `scripts/check-linux-scripts.sh` passes. The Linux
  build was not run against a model.

## Install and upgrade

Windows: unzip `ninfer-rtx3090-windows-x64-0.16.0-rtx3090.zip` next to your `models` folder and run
`run.bat <profile>`. Linux: see `docs/release-archive-linux.md` for the glibc, CUDA 12.8, cuBLAS,
FFmpeg 6 and libcurl requirements. Check the SHA256SUMS file for each archive. If you keep your own
launch script, go through the breaking-changes list at the top first.
