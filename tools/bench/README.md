# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## What a long uncached prompt does to a stream next to it

`run_prefill_stall.py` starts a streaming chat completion against a running `ninfer-serve`, lets it
settle, then submits one long uncached prompt and records the stream's event timestamps around it. It
reports the stream's inter-event gaps (p50/p90/p99/max) and events per second in three windows, before,
during and after the long prompt, and the long prompt's own time and prefill rate. The during window starts at
the moment the long request's body has been sent and ends when it finishes; the baseline runs from the stream's
settle point to that send, and the after window starts when the long request ends. A worker that is still running when its bounded wait ends, or a
cancel that lands after the prompt finished, fails the run rather than producing a report.

```bash
# server: two lanes, so the stream and the long prompt run at once
ninfer-serve models/qwen3_8_27b.ninfer --max-concurrency 2 --max-context 131072 \
    --prefill-cublas --prefill-chunk 4096

python tools/bench/run_prefill_stall.py --model qwen3.8-27b --long-tokens 100000 --output stall.json
# a short request 10 s into the long prompt: its time to first event is the head-of-line wait
python tools/bench/run_prefill_stall.py --model qwen3.8-27b --short-delay-seconds 10
# cancel the long prompt at 60 s and resubmit it: the retry's time shows what --progress-anchor-tokens saved
python tools/bench/run_prefill_stall.py --model qwen3.8-27b --cancel-long-after-seconds 60
```

The probe measures whatever the server does, so the same commands compare scheduling options: restart the
server with `--progress-anchor-tokens N` (on master) and, in builds that carry them, the prefill scheduling
options (`--prefill-stall-budget-ms`, `--prefill-decode-share`, `--yield-long-prefill-ms`,
`--prefill-layer-slices`, `--progress-anchors-shared`; each is documented with its own PR).

Start each setting from a fresh server so the prompt cache is cold, and take the baseline (no flags) the
same way. Every run draws a new prompt seed (printed on stderr) so one run cannot warm the next; pass
`--seed` to repeat a prompt on purpose. `--words-per-token` is a guess at the tokenizer: adjust it until the
reported `prompt_tokens` matches `--long-tokens`. `--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'`
adds fields to every request. With `--max-concurrency 1` the long prompt queues behind the stream instead,
which is itself the head-of-line case. `python -I tools/bench/test_prefill_stall.py` checks the tool against a
fake server that stalls its streams on a long request.

## Interleaved A/B between two builds

`run_interleaved_ab.py` compares two executables by alternating them **inside** each repetition.
Use it for any before/after claim on this hardware. The card drifts 3-5% between processes as it
heats and the power cap clamps, which is larger than most effects worth measuring here, so
"all of arm A, then all of arm B" cannot separate a real change from the ramp -- two comparisons
in TODO's history came out with opposite-signed drift on code paths that had not changed.

```bash
python tools/bench/run_interleaved_ab.py \
    --arm-a build-ninja/apps/ninfer.exe \
    --arm-b build-ninja/apps/ninfer-newtile.exe \
    --reps 6 \
    --config 'k6:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 --greedy --spec dflash2 --draft-tokens 6' \
    --control 'k4:{exe} models/qwen3_8_27b_dflash2.ninfer --prompt "..." --max-new 256 --greedy --spec dflash2 --draft-tokens 4'
```

It reports the **paired median** -- the median of per-repetition ratios -- and how many pairs came
out positive. That is deliberately not the ratio of the two medians: on this box the two statistics
disagree by more than the effects being measured.

`--control` names a configuration whose code path is **identical in both arms**, so any ratio it
shows is pure drift; the tool divides it out per repetition and prints a `normalised` column. In the
GDN-tile comparison that control is what made the result readable, because draft counts 4 and 5
stay on an unchanged route.

Two things it refuses rather than lets you get wrong:

- **both arms must sit in the same directory**, in practice `build-ninja/apps/` under different
  names. An executable copied elsewhere fails DLL resolution on Windows, exits 127 and writes an
  empty log, which reads exactly like a model-loading failure.
- a `--config` with no `{exe}` placeholder, which would run the same binary twice.

`--metric-regex` selects what to compare; it defaults to the CLI's `decode speed ... tok/s` summary
line and needs changing for `ninfer_bench`, which prints CSV. A run where any sample produced no
number exits 1, because a comparison with a hole in it is not a comparison.

**One build, an option off and on.** Leave out `--arm-b` and give `--a-args` and `--b-args`
(they must differ): the one executable runs both arms, each command with its arm's arguments
appended. The interleaving and the statistics are the same. `--log-dir DIR` keeps every run's
output, and `--json FILE` writes the result for a script:

```bash
python tools/bench/run_interleaved_ab.py --arm-a build-ninja/tests/ninfer_tests.exe --reps 12 \
    --b-args="--moe-fused-route" \
    --config 'decode:{exe} ninfer_qwen4_exp_engine_real_test --bench --generate 256 --mtp 3 @prompt.ids' \
    --metric-regex 'decode [0-9.]+ s \(([0-9.]+) tok/s\)' --log-dir ab-logs --json ab.json
```

Its self-test, `ninfer_interleaved_ab_test`, uses a fake executable and needs no GPU.

## Greedy identity between two builds (`greedy_identity.py`)

Do two builds produce the same tokens? The tool runs each `ninfer` CLI on every prompt of a JSONL file
(default `bench/fixtures/chat_prompts.jsonl`, 24 prompts) with `--greedy --print-token-ids --max-new N`,
reads the generated ids the CLI prints on stderr, and prints `GREEDY IDENTICAL`, or `GREEDY DIFFERS`
with the first differing prompt and token index (exit 1) and the ids around it. Exit 2 means a build
crashed or printed no ids line, so nothing was compared.

```bash
python3 tools/bench/greedy_identity.py \
    --a build-ninja/apps/ninfer.exe --b build-ninja/apps/ninfer-new.exe \
    --model models/qwen3_8_27b.ninfer --tokens 128 --limit 6 \
    --extra-args="--kv-dtype rk4v4 --max-context 8192"
```

`--extra-args` is one string given to **both** builds; use it for the option under test and to pin
what would otherwise differ. Write it with `=` so a leading `--` is not read as another option.
`--limit N` runs only the first N prompts. Both builds must sit in the same directory under different
names (the same preflight as `run_interleaved_ab.py`). Equal ids say the builds take the same
numerical path; unequal ids are not necessarily a bug (a different reduction order flips a near-tie), but
the index says how far in they agree.

`--a-args` and `--b-args` add arguments to one side only. Leave out `--b` and the one build runs as
both sides, which is how to check that an option designed to keep the output does (the per-side
arguments must differ):

```bash
python3 tools/bench/greedy_identity.py --a build-ninja/apps/ninfer.exe \
    --model models/qwen3_8_27b.ninfer --b-args="--no-cuda-graph"
```

It runs the builds one after the other on the card, a process per prompt and build, so **hold the GPU
lock around the whole run** (CLAUDE.md, "GPU lock"):

```powershell
. tools/gpu-lock.ps1
Invoke-WithGpuLock { python tools/bench/greedy_identity.py --a ... --b ... --model ... }
```

Its self-test, `ninfer_greedy_identity_test`, uses fake CLIs and needs no GPU.

## Verifying the Basalt options on a GPU

The Basalt-learnings branch (issue #383) adds about 25 options, each compiled and CPU-tested but
never run on a GPU. `basalt_gpu_verify.ps1` (Windows) and `basalt_gpu_verify.sh` (Linux) run every
GPU check they need in one go, from `basalt_options.json`: each option's kind (identity, quality,
speed or serving), its GPU-test regex, its identity harness runs, its KLD-gate flag and its A/B
workload. The status of each is in
[docs/maintainer/flash-next-plan.md](../../docs/maintainer/flash-next-plan.md#verification-status).

1. Build the branch (`build-ninja`, Release, sm_86, tests and apps; `-DNINFER_BUILD_BENCHMARKS=ON`
   too for the community preset). For the "every option off equals master" check, build
   origin/master's `ninfer` CLI and copy it into the same `apps` directory under another name, such
   as `ninfer-master.exe`.
2. Gather the inputs. Each step that lacks one is skipped and says which:
   - `--flash-next`: the Flash-Next artifact with the MTP drafter (`qwen4exp-iq2xs-mtp.ninfer`), its
     n-gram volume beside it; `--qwen27b`: the 27B DFlash2 bundle; `--qwen35b`: the 35B-A3B
     (optional);
   - `--kld-base-flash-next` and `--kld-base-27b`: llama.cpp KL-divergence bases
     ([flash-next-plan.md](../../docs/maintainer/flash-next-plan.md#decode-path-kld-gate),
     [perplexity.md](../../docs/perplexity.md)); `--master-cli`: the master CLI from step 1;
   - `--iq3-pack` (repeatable): the converted IQ3_XXS and IQ3_S packs; `--gguf`: the Flash-Next
     GGUF for the converter's real-file test; torch, pytest and numpy for the converter suites;
     `nsys` on the PATH for the `--ggml-decode-mma` policy check.
3. Look at the plan. The dry run prints every step with its commands and checks that every test
   regex matches a name `ctest -N` lists; it exits 1 if one does not:

   ```powershell
   pwsh tools/bench/basalt_gpu_verify.ps1 --build-dir build-ninja --dry-run --flash-next <...> --qwen27b <...>
   ```

4. Run it, in the background (it takes many hours, mostly model loads: every A/B is 24 processes):

   ```powershell
   pwsh tools/bench/basalt_gpu_verify.ps1 --build-dir build-ninja --flash-next <...> --qwen27b <...> `
       --kld-base-flash-next <...> --master-cli build-ninja\apps\ninfer-master.exe
   ```

   `--only ab:,kld:` or `--skip launchers:,full-suite` picks steps by id prefix; `--reps N` changes
   the A/B pairs (12 by default); a step can be rerun alone with
   `python tools/bench/basalt_gpu_verify.py <the same arguments> --action run --step <id>`.

Order: the GPU tests of every touched area (`tools/run-tests.ps1 -GpuOnly -Regex`, grouped by the
artifact their real tests read), the identity harness runs, the KLD gate (one option-off run at 20
and at 100 chunks serves every option; the window check compares window 1 with window 3), the
`--ggml-decode-mma` per-shape timing and nsys policy check, one ABBA A/B per option and variant
(n = 12, the Engine bench through `run_interleaved_ab.py` in its same-build mode, the 27B options
through `run_chat_decode.py`), two combined rows with every winning option on, the ledger's checks
(every option off against master with `greedy_identity.py`, `--relaxed-drafting` starts, a
`--partial-ram-residency` start, the default expert state, a real-server smoke and soak, the
community preset, the converter suites, the IQ3 packs, the launchers from a path with spaces and
`(x86)`, with `package-release.ps1` on Windows), and last the rest of the GPU suite without the slow
tests (`softmax_attention|kv_materialize`, AGENTS.md).

**The GPU lock.** On Windows the script holds the shared lock (`tools/gpu-lock.ps1`) for each step
marked `hold` (one ABBA set, one KLD run, one server start and its probes), releasing it between
steps; test steps go through `tools/run-tests.ps1`, which takes it per test. The Linux script takes
no lock: no Linux script here does, because the Linux box is a rented machine used by one session at
a time. Do not run it on a shared Linux machine without one.

**Results** go to `profiles/bench/basalt-<date>/` (`--out` to change): `results/<step>.json`,
`logs/<step>/`, the KLD rows, the generated inputs, and `summary.md`, a table of every option (its
GPU tests, identity run, KLD numbers and A/B) and every check, ready to paste into the pull request.
A speed variant is marked a win at a paired median of at least +1% with three quarters of the pairs
positive; that only decides what the combined rows turn on, not what ships. Two things stay manual:
the dashboard (open `/dashboard` in a browser at desktop and phone width) and `--mtp-constrained`'s
speed (it needs constrained requests through ninfer-serve).

Its self-test, `ninfer_basalt_gpu_verify_test`, checks `basalt_options.json` against the flags
ninfer-serve parses and the plan against a fake `ctest -N` listing; it needs no GPU.

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Agent rotation (context-cache retention)

`run_agent_rotation.py` drives an already-running `ninfer-serve` with N independent agent
conversations visited round-robin, each growing by one long user turn per visit until it reaches
`--target-tokens`. With more agents than `--max-concurrency`, most visits must restore their
conversation from the Host tier or re-prefill it, and every `--edit-every`th visit rewrites the user
turn two back, which only a long anchor below the edit can serve. The client reports TTFT from the
stream and the weighted cache hit rate from the usage `cached_tokens`, split by first, append and
edit visits; run the server with `--request-log-jsonl` for the reuse path and transfers.

```bash
python3 tools/bench/run_agent_rotation.py --agents 4 --target-tokens 200000 \
    --out profiles/bench/agent-rotation/run.jsonl
```

## Hosted multi-tenant workload (context cache on a provider)

`run_hosted_workload.py` drives an already-running `ninfer-serve` with what a provider serving many
people sees: N tenants, each with one conversation at a time, all conversations opening with one of a
few shared "product" prompts (system text plus a tool catalogue), then a private opening document,
then growing turn by turn with appended tool output. One request is in flight per conversation (a
client waits for its reply); a `429 server_overloaded` answer is retried every second, as a gateway
would hold it, and counted. `hosted_analysis.py` turns one or more runs into the hit split, the
tokens an ideal cache would have saved and why they were lost, TTFT and queue percentiles, lane
utilisation and the store counters. The baseline that uses them is
[`docs/maintainer/cache-provider-baseline.md`](../../docs/maintainer/cache-provider-baseline.md).

```powershell
# server: the tuned 27B MTP flag set plus --request-log-jsonl (and --context-store DIR if wanted)
python tools/bench/run_hosted_workload.py --preset hosted-32 --duration-s 600 `
    --request-log server-requests.jsonl --out runs/c8-h8
python tools/bench/hosted_analysis.py runs/c4-h8 runs/c8-h8 --warmup-s 300 --markdown
```

Presets (token sizes are measured with the server's own `count_tokens`; filler is about 1.35
tokens per word):

| preset | tenants | products (shared prompt tokens) | opening | per turn | think time | other |
|---|---:|---|---|---|---:|---|
| `smoke` | 4 | about 1.2K, 2.3K | 1-4K | 0.3-1.5K | 2 s | 10% edit, 10% branch, ends at 16K |
| `hosted-16` | 16 | 1,608 / 4,563 / 9,090 (measured) | 2-40K | 0.3-4K | 20 s | 5% edit, 5% branch |
| `hosted-32` | 32 | same | 2-40K | 0.3-4K | 40 s | same |
| `hosted-64` | 64 | same | 2-40K | 0.3-4K | 80 s | same |
| `bursty-32` | 32 | same | 2-40K | 0.3-4K | groups | groups of 8 every 45 s; 15% fan-out of 3 over a new 4-16K document |

- Sizes in ranges are log-uniform. Tenant `i` uses product `i mod products`. A conversation ends
  past 100K prompt tokens (16K for `smoke`) and its tenant opens a new one.
- **Turn kinds.** `first` opens a conversation; `append` adds tool output after the last reply;
  `edit` replaces the latest tool output (the prompt continues from the previous reply, a point
  the newer turn superseded); `branch` drops the last 2-6 exchanges and continues from there;
  `fanout` sends 3 sub-agent requests at once, each the history plus one new shared document plus
  its own task (siblings share a prefix that ends at no chat marker); `resume` is one append per
  saved conversation after a restart (`--save-state`, then `--resume-state FILE --resume-once`).
- **Arrivals.** `poisson`: each tenant waits an exponential think time (mean above) after each
  reply, so the offered rate is about tenants / think time while the server keeps up. `bursty`:
  tenants wait until their random group of 8 is released, every 45 s.
- **Replies.** Temperature 0. Thinking off: up to 160 tokens, the reply is replayed verbatim.
  `--thinking on`: up to 640 tokens with `thinking_budget` 384; the client sends back only the
  answer, as most clients do, so the template drops the closed reasoning and the reusable point is
  the prompt before the reply rather than its end.
- `--tenants` and `--think-s` override a preset; `--duration-s` stops issuing turns (in-flight ones
  finish).

Each run directory holds `requests.jsonl` (one record per request: prompt, cached, output tokens,
client TTFT, retries, and the ideal reuse estimates), `server_records.json` (the joined
`request_done` records), `load.jsonl` (`/v1/load` once a second) and `summary.json` (with
`/metrics` before and after). `hosted_analysis.py` documents its classification in its docstring;
its eviction split replays the trace through an LRU of the run's Device + Host bytes and is a
ceiling model, not the engine.

## Raw disk speed (`disk_seq_io.py`)

Sequential write and read with the OS file cache bypassed (`FILE_FLAG_NO_BUFFERING` on Windows,
`O_DIRECT` on Linux), one block at a time, over a scratch file it always deletes, with an optional
buffered read for comparison. It is the disk rate to use in a disk-tier cost model, as opposed to
the context store's end-to-end read path.

```powershell
python tools/bench/disk_seq_io.py --dir $env:TEMP --size-gib 4 --block-mib 4 8 16 32 --buffered
```

## Copy-heavy agent replay

`agent_replay/` is the copy-drafting workload: deterministic multi-turn coding-agent sessions whose
tool results are real files of this repository, so a large share of the model's output reproduces
text already in its prompt. `run_agent_rotation.py` above never copies (its turns are random words)
and cannot show copy-drafting gains. The replay is stdlib-only and drives an already-running
`ninfer-serve` through streaming `/v1/chat/completions` with five tools (`Read`, `Write`, `Edit`,
`Grep`, `Bash`), greedy and thinking off.

Each session starts from a seeded file of the pinned `--commit` (default `HEAD`, recorded as a
hash) and grows turn by turn until its prompt reaches a seeded target in
`[--min-context, --max-context]`. Every turn stages a scripted tool call and its result, then the
model generates the assistant turn. Turns come in shuffled blocks of eight, half of them copies:

| category | observation | the model is asked to |
|---|---|---|
| `explore` | `cat -n` Read slice, plain `sed -n` slice, Grep rows, or a `git show` diff | continue (no instruction) |
| `explain` | `cat -n` Read slice | explain an identifier in prose, no code |
| `rewrite_file` | whole file as a `cat -n` Read | Write the whole file with one identifier renamed |
| `apply_change` | whole file as plain `cat` | return the whole file with one comment added |
| `edit_function` | `cat -n` Read around a 20-60 line block | Edit: old_string and new_string both reproduce the block |
| `diff_revert` | `git show <sha> -- <path>` | write each hunk's pre-change lines |

Tool results, file choices and instructions depend only on the seed and the corpus commit; the
assistant turns are free-running, so the number of turns before a session reaches its target can
differ between builds. Compare decode tok/s over committed tokens and wall time, matched by
`(session, turn)`, not output hashes. Model tool calls are answered with a short acknowledgement
before the next scripted observation.

```bash
# start the server with a request log, then
python -m tools.bench.agent_replay run --port 18120 --sessions 4 --concurrency 1 \
    --min-context 30000 --max-context 120000 \
    --request-log profiles/bench/agent-replay/request_log.jsonl --out profiles/bench/agent-replay/run1
```

`--request-log` names the server's `--request-log-jsonl` file. The runner reads the records written
during its run and joins them to its turns by response id, which adds the request's `speculative`
counters, the server's TTFT/total seconds and the committed output token IDs. Without it the
runner takes decode time from the usage chunk's `timings` (`predicted_ms`; `prompt_ms` is kept as
`prompt_s`, it is not TTFT) and TTFT from client-side stream timing.

Outputs in `--out`:

- `turns.jsonl`: one sample per turn: category, prompt/cached/output tokens, finish reason, TTFT,
  decode seconds and `decode_tps` (output tokens / (total - TTFT)), the timing source, the copy
  overlap, and `counters`, every numeric leaf of the request log's `speculative` object and of
  the usage `timings` under dotted names. New server counters, such as `ngram_*`, are picked up
  without changes here.
- `summary.json`: per category, `copy`, `non_copy` and `all`: token-weighted decode tok/s, its
  median and minimum per turn, TTFT median/mean/max, token sums, length-capped turns,
  `short_turns` (under 128 output tokens: a copy turn this short called a tool instead), summed
  counters, DFlash/MTP acceptance and tokens per round, and the share of output from any
  `*ngram*accepted_tokens` counter.
- `transcripts.jsonl`: the full message list of each session and the message count at each turn.
- `output_token_ids.jsonl` (with `--request-log`): committed output token IDs per turn.

**Copy overlap** is the fraction of the turn's output (content plus tool-call argument values) in
8-unit windows that also occur in the session's prompt so far. Units are identifiers, numbers or
single punctuation characters, whitespace is ignored, and `cat -n` prefixes are removed from both
sides. It shows that copy turns actually copy and bounds what a copy proposer can harvest; it is a
text proxy, not the proposer's token-level match.

`export` writes the prompt and output token sequence of every turn for an offline proposer
simulation (the copy-length distribution of the n-gram copy-drafting plan). The server logs output
IDs but not prompt IDs and has no tokenize endpoint, so each prompt is re-rendered from
`transcripts.jsonl` with the served artifact's own `chat_template.jinja` and `tokenizer.json`
resources, with tool definitions shaped as the server passes them to the template and message text
that spells a special token (a source file quoting `<|video_pad|>`) encoded as plain text, as the
server does. Each prompt's length is checked against the server's `prompt_tokens`. This one step
needs `jinja2` and `tokenizers`:

```bash
python -m tools.bench.agent_replay export --run profiles/bench/agent-replay/run1 \
    --artifact models/qwen3_8_27b.ninfer
```

It writes `token_streams.jsonl` (`prompt_ids`, `output_ids`, `output_source`) and prints how many
rendered prompts match the server's length exactly. Output IDs come from the request log; without
one they are re-tokenized from the recorded text and marked `retokenized_text`.
`--tokenizer-dir` takes a directory holding `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja` instead of an artifact.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v20. The flattened summary and schema-v4 matrix
manifest carry native facts from the report: architecture, public name, actual formats, prefill signature, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks, and the GPU power limit at the start and end of each run. The
matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact LABEL=PATH` entries. Labels identify report groups;
the selected artifact supplies the architecture, public name and weight bindings.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset. Use `dflash7` for
Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B DFlash2 K=7, with companion weights
in the selected artifact. `--sampling greedy` selects exact argmax; the default is stochastic.
Run commands with a selected Python 3.11 interpreter, as in the model-page reproduction entries.
Both serving runners accept `--kv-dtype bf16|int8|fp8|nvfp4|k8v4` (default: `int8`).
Specify it explicitly when recording a new campaign; the Qwen3.8 performance profile uses
`--kv-dtype fp8`. The runner verifies the loaded KV representation and records the selection in
its results. A serial run cannot resume records collected with a different KV dtype.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v8 result and flattened summaries retain the KV dtype, actual `prefill_signature`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v25 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Chat decode A/B (`run_chat_decode.py`)

Decode rate on a chat prompt set through `ninfer-serve`'s OpenAI Chat Completions endpoint,
reported as `concurrency x 1000 / mean TPOT` from the server's own request log -- the metric
`vllm bench serve` prints, so results can be read against stacks that publish it. `output
throughput` (generated tokens / wall clock) is printed alongside and includes prefill and the tail.

Arms are interleaved, in order on even passes and reversed on odd, because between-process spread
on one RTX 3090 is 3-5%; always A/B two binaries inside one sitting rather than across sessions.
Each arm may drop or add server flags, so one binary can be compared against itself under different
options, and may name its own artifact (`;model=PATH`), so two artifacts can be compared on one
binary with the same prompts. `--spec` selects `mtp` (the default), `dflash2` or `none` and
`--draft-tokens` the draft count; the `RESULT` line carries the decode rate, the draft acceptance
and the tokens emitted per round, so one run answers both "faster?" and "does it still draft as
well?". `--model-id` is the id the server publishes, which is the artifact's public name
(`qwen3.8-27b` by default, `qwen3.6-35b-a3b` for the 35B-A3B).

```bash
# two binaries, one artifact
python3 tools/bench/run_chat_decode.py \
  --model models/qwen3_8_27b.ninfer --prompts prompts_real.jsonl \
  --concurrency 1 --reps 2 --out profiles/bench/chat-decode \
  --arm base=/path/to/baseline/ninfer-serve --arm new=./build/apps/ninfer-serve

# one binary, two artifacts, DFlash2, drafting acceptance included
python3 tools/bench/run_chat_decode.py --prompts bench/fixtures/chat_prompts.jsonl \
  --spec dflash2 --draft-tokens 7 --kv-dtype rk4v4 --max-tokens 512 --out profiles/bench/chat-decode \
  --arm old=./build/apps/ninfer-serve\;model=old.ninfer --arm new=./build/apps/ninfer-serve\;model=new.ninfer
```

The prompt file is one JSON object per line with a `prompt` string. The comparisons in
[docs/performance.md](../../docs/performance.md#small-t-tensor-core-kernels-for-verify-and-cohort-decode)
use the eight thinking-off prompts of syv-ai/qwen38-27b-rtx3090's `bench/prompts_real.jsonl`, which
is that project's file and is not vendored here. `bench/fixtures/chat_prompts.jsonl` is this
repository's own set of 24 mixed prompts (code in eight languages, prose, maths, translation and
Chinese); the artifact comparisons of 2026-10 use it with greedy decoding and thinking off.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions.
Both serving runners disable prefix reuse and set extra Device state and Host context capacity to zero.

Schema-v4 outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. C=1 corpus runs also write complete responses in `corpus/<point>/results.jsonl` and
per-request phase summaries in that directory; older campaigns may have only point reports and
server logs. Historical model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --kv-dtype int8 \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --kv-dtype int8 \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.

## Host arena allocation replay (`replay_host_arena.py`)

When the Host context tier is full, an allocation can fail because too few bytes are free or because
enough bytes are free in total but no single run is large enough. `ninfer_host_context_geometry_failures_total`
and `ninfer_host_context_shortage_failures_total` count the two on a live server. This tool asks a
different question offline: **given the exact allocate / release / split sequence of a real run, how many
refusals would another placement policy have had?** It needs no GPU and no new server run.

It reads a trace printed by a *measurement build*. The shipped server does not print one. To make one, add
these lines to `HostContextArena` (`src/core/host_context_arena.cpp`) and keep them out of the commit; kind is
`S` for a state image allocation and `K` for a KV run (the arena does not know the kind, so pass it from every
call site that reaches `HostContextArena::allocate`: `HostStatePool::allocate` and
`ProgramImpl::reserve_capture_destination` (`capture.cpp`, a direct state image allocation) for `S`, and
`HostKVArena::allocate` for `K`. Re-check with `rg "arena.*allocate\("` that no other caller exists, because a
missing one leaves real `A`/`F` events out and the replay still runs on the incomplete sequence):

```cpp
// allocate(), on success:  std::fprintf(stderr, "[trace] A t=%lld off=%zu bytes=%zu kind=%c\n", now_ms(), offset, rounded, kind);
// allocate(), on failure:  std::fprintf(stderr, "[trace] F t=%lld bytes=%zu kind=%c free=%zu largest=%zu extents=%zu %s\n",
//                                        now_ms(), rounded, kind, free_bytes(), largest, free_extents_.size(),
//                                        free_bytes() >= rounded ? "geometry" : "shortage");
// release():               std::fprintf(stderr, "[trace] R t=%lld off=%zu bytes=%zu\n", now_ms(), offset, bytes);
// split():                 std::fprintf(stderr, "[trace] P t=%lld off=%zu left=%zu right=%zu\n", now_ms(), left_offset, left_bytes, right_bytes);
```

```powershell
# a hosted run with the trace build (see "Concurrent serving benchmark" and the hosted workload), then:
python tools/bench/replay_host_arena.py runs/c8-h8/server.err --capacity-mib 8192
python tools/bench/replay_host_arena.py runs/c8-h8/server.err --capacity-mib 8192 --policies legacy,bestfit,twoended_best
```

The first thing it prints is a validation: replaying the trace under `legacy` (first-fit by address, what
the arena does today) must reproduce the real run exactly. If it does not, the script exits without
reporting any other policy; `--force` overrides that for a trace taken under another placement.

Read the result as a hypothesis, not a measurement. An F event is replayed as a probe (would the policy
have placed it right now?) and the live set stays the real run's, so the engine's reaction to a successful
placement, which would change every later event, is not modelled. A live A/B of plain two-ended placement did
not show what the replay predicted for it. `python -I tools/bench/test_replay_host_arena.py` checks the tool
itself and runs as `ninfer_host_arena_replay_test`.

## Randomised serving soak (`serve_soak.py`)

`serve_soak.py` sends a seeded random mix of requests to a running `ninfer-serve` for a fixed time and
reports what went wrong. It is the long-run check the single-purpose probes above are not: leaks, a lane
that stops answering, and error paths that only break after hours. It is adapted in shape from Basalt's
`tools/serve_soak.py` and built on `tools/streaming_http`, so stalls are timed from SSE event arrivals.

| Kind | What it sends | A pass is |
|---|---|---|
| `chat` | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` streams, 32 to 1200 tokens; half carry a unique prefix, so the prompt cache sees both hits and misses | 200 and the stream's terminator (`[DONE]`, `message_stop`) |
| `cancel` | a stream dropped after 1 to 40 events, or 50 ms to 1.5 s after the send (often mid-prefill) | no error status; nothing else is asserted |
| `near` | a prompt of context minus the reply and 256 tokens, sized with `/v1/messages/count_tokens` | 200 |
| `over` | a prompt 2000 tokens beyond the context | 400 `context_length_exceeded` (Anthropic: 400 `invalid_request_error`) |
| `oversize` | a body one MiB above `--max-request-mib` | 413; a reset while uploading is also accepted and counted as `ok_reset`, since the server answers before it reads the body. `ok_reset` can mask a server that dies on the upload: watch for it in `by_kind` and in the sampler's `server_lost` finding |
| `image` | base64 data-URL images from `--images DIR` in OpenAI `image_url` / Anthropic `image` parts | 200, or 400 `vision_disabled` from a server without `--vision`, which skips images for the rest of the run |

The first requests are one of every enabled kind, then random picks with 1 to `--max-concurrency` in
flight; the default is the server's lane count (`len` of `/slots`), so nothing waits in the server's queue
unless you raise it. `near`, `over` and `oversize` are rationed (a few per ten minutes) because each costs a context-sized
prefill or a body of hundreds of MiB (the `oversize` body is streamed from one reused 1 MiB chunk, so the
client holds about 1 MiB however large the limit). `--kinds chat,cancel` narrows the mix. The context is read from `/slots` (`n_ctx`); pass
`--context` to override it.

Every `--sample-seconds` (2) it reads `/v1/load` (queue depth), `/metrics` (the unlabelled `ninfer_*`
series), `/slots` (busy lanes, context in use) and the server's resident memory, and keeps them in the
report's `samples`. It sends a normal `Host`, no `Origin` or `Sec-Fetch-*`, and `--api-key` as
`Authorization: Bearer`, so it passes the server's Host and Origin checks.

**Run it against a live server under the GPU lock.** The server holds the card for the whole soak, so
take the lock for the server's lifetime (`CLAUDE.md`, "GPU lock"), then run the soak from a second shell:

```powershell
. tools/gpu-lock.ps1
Invoke-WithGpuLock {
    $server = Start-Process ninfer-serve -ArgumentList 'model.ninfer','--max-context','32768','--vision' -PassThru
    Start-Sleep 60      # or poll /health
    python tools/bench/serve_soak.py --url http://127.0.0.1:8080 --minutes 30 --pid $server.Id `
        --images C:\soak-images --out soak.json
    $code = $LASTEXITCODE
    Stop-Process $server
}
exit $code
```

On Linux the memory is read from `/proc/<pid>/status` (`VmRSS`). On Windows it needs `psutil`; when that
is not installed the run prints a note and leaves memory out (nothing is installed for it). Without
`--pid` memory is not sampled either. Pass `--seed` to repeat a mix.

**What it flags, and the exit code.** 0 is a clean run; 1 is any of the findings below; 2 is a run that
could not start (unreachable server, `/v1/models` refused, `--images` with no usable file). The report's
`findings`, `failures`, `stalls` and `rss` say which.

- **Unexpected result:** a status or error code other than the table's, a stream that ends without its
  terminator, an error event inside a stream, or a transport error.
- **Stall:** no SSE event for `--stall-seconds` (60), counted from the end of the upload until the first event
  and after that from the previous one (a keep-alive comment does not count). Time the server reports as
  queued does not count: each sampler tick, of the requests still waiting for a first event, the newest
  `requests.queue_total` from `/v1/load` (the server queues in arrival order, so the older ones hold lanes)
  have their clock reset. A request that holds a lane and sends nothing for the limit is flagged with its
  phase (before the first event or mid-stream) and cancelled so the run can end. A request still running
  when the `--drain-seconds` grace period (stall plus 30 s) after the last launch ends is reported the same
  way. With `--max-concurrency` above the lane count the extra requests queue; the queue credit covers them
  only while `/v1/load` reports the queue.
- **Idle lane with a queue:** `queue_total > 0` while some lane in `/slots` has `is_processing` false, for
  `--stall-seconds`, or the queue head's `waited_s` is past that on two samples in a row with an idle lane
  (one sample can fall between a lane finishing and the next admission). Reported once per episode as a
  `stalls` entry with `kind: "queue"`.
- **Memory growth:** the median of the last fifth of the second half of the run minus the median of its
  first fifth is above `--rss-growth-mib` (512). Fewer than 4 samples gives no verdict (the report says so).
- **Server gone:** `/v1/load`, `/metrics` and `/slots` all fail 3 sampling rounds in a row.
- **Harness error:** an exception in a sampler step is recorded in `harness_errors` and fails the run instead of
  silently ending the sampling.

Ctrl-C cancels what is in flight and writes a partial report (`"interrupted": true`, exit 2 unless it already
holds findings). An unexpected exception in the harness exits 2, never 1. `--out` is checked for
writability before the run starts.

`python -I tools/bench/test_serve_soak.py` checks the tool against a fake server (a clean run, a stall, an
unexpected 400 and 500, growing and flat memory, a vanished server) and runs as `ninfer_serve_soak_test`
without a GPU. It says nothing about the real server: the first run against one is the check that the
expected statuses above match what that build returns.


The report (`--out`, `schema_version` 1) holds:

| Field | Content |
|---|---|
| `url`, `model`, `context`, `seed`, `minutes`, `started_unix`, `duration_seconds` | what was run, and for how long |
| `settings` | `kinds`, `max_concurrency`, `lanes`, `stall_seconds`, `rss_growth_mib`, `sample_seconds`, `max_request_mib`, `images` |
| `requests` | `total`, and `by_kind`: for each kind a count per outcome (`ok`, `ok_reset`, `cancelled`, `skipped_no_vision`, `failed`) |
| `failures` | one entry per failed request: `id`, `kind`, `protocol`, `status`, `code`, `events`, `expected`, `at_seconds`, `reason` |
| `stalls` | stalled streams (`id`, `kind`, `protocol`, `phase`, `events`, `quiet_seconds`) and `kind: "queue"` entries (`queue_total`, `lanes`, `busy`, `head_waited_s`) |
| `rss` | `sampled`, `samples`, `first_mib`, `last_mib`, `max_mib`, `growth_last_half_mib`, `limit_mib`, `flagged`, and a `note` when memory was not sampled |
| `samples` | every tick: `t`, `queue`, `queue_total`, `queue_head_waited_s`, `slots`, `slots_busy`, `ctx_used`, `metrics` (unlabelled `ninfer_*`), `rss_mib`, `in_flight`, `requests`, `failed` |
| `interrupted`, `harness_errors` | Ctrl-C and sampler bugs |
| `findings`, `exit_code` | the reasons the run failed (empty when clean), and the process exit code |
