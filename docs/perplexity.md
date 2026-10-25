# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype int8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `nvfp4`, and `k8v4`.

All seven have been measured on this corpus; the results, alongside each format's size and decode
speed, are in [`docs/config-calculator.html`](config-calculator.html).

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Held-out corpus

`ninfer-ppl-1m-v1` is built from public datasets that the model has almost certainly trained on,
which can understate the cost of a weight or KV representation change. The repository also
includes `ninfer-ppl-heldout-2026-09-v1`: 12 streams written after Qwen3.8's release, covering new
English and Chinese Wikipedia articles, arXiv abstracts, new GitHub code in four languages, this
fork's recent code, and chat-template conversations. Use it to judge representation quality, and
keep `ninfer-ppl-1m-v1` for comparisons with the numbers already published against it. Its
selection rules and limitations are in its
[README](../eval/corpora/perplexity-heldout-2026-09/README.md).

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-heldout-2026-09/manifest.json \
  --quick \
  --kv-dtype int8
```

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.

## Decode-path KL-divergence gate

`--corpus` and `--text` score every target from one prefill pass, so they cannot see changes to
decode kernels, the speculative verify forward, or the KV writes made while decoding.
`--kld-base` scores through the decode path instead, against a llama.cpp KL-divergence base. Use it
to judge a numerics change to any of those paths. The method follows Basalt's quality gate.

Build the base once with llama.cpp from a reference model (Basalt uses a Q8_0 GGUF) and a text
file. llama.cpp tokenizes the text, so the base fixes the exact tokens every run scores:

```bash
llama-perplexity -m qwen3_8_27b-q8_0.gguf -f wiki.test.raw -c 512 \
  --kl-divergence-base wiki-512.kld
```

Then score the same tokens with an artifact:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --kld-base wiki-512.kld --chunks 20 --decode-window 3 --kv-dtype int8
```

Each chunk of `n_ctx` tokens is one scoring window, with `n_ctx` taken from the base:

- Positions `[0, n_ctx/2)` are prefilled.
- The rest is teacher-forced through the decode path, `T` tokens per forward
  (`--decode-window T`, default 3, at most 16). Every target in `[n_ctx/2 + 1, n_ctx)` is
  predicted by a decode forward. These are exactly the rows the base holds.
- `T=1` is the ordinary decode step. A wider window is the speculative verify forward, with every
  column committed through the ReplaySSM fold, so `T=3` runs the verify forward of an MTP round
  with two drafts. The last window of a chunk may be partial: it runs at width `T` with the
  remaining columns live, as a speculative round with fewer drafts does.
- Verify-phase routes run in these windows once `T` is wide enough for them. `--mlp-a8-decode`,
  for example, is admitted only at 16 to 32 columns, so a batch-one window reaches it only at
  `T=16`. With `--corpus` and `--text` these routes are inert.

`--chunks N` scores the first `N` chunks (default all). `--context` and `--stride` are refused:
the base fixes the context. Decode scoring loads only the Text backbone, plans no MTP or drafter
weights, and runs on one device. It runs eagerly, without CUDA graphs, but with the attention
envelopes served graph decode captures (the ordinary step's for `T=1`, an MTP round's target verify
with `T-1` drafts otherwise) for an Engine at the same context. The attention route is chosen from
the envelope, so scoring selects the route served decode replays.

Each row compares the model's logits over its public token domain with the base row. llama.cpp may
pad the vocabulary beyond the public token count (to 248,320 rows, for example); those rows are
left out. A row contributes its NLL, the base's NLL, `KL(base || model)` with base log-probabilities at
or below -16 skipped, whether the top-1 tokens agree, and the log-probability gap at the base's
top-1 token. stdout shows PPL with its standard error and the base's PPL, the KLD mean, median, p99
and max, and the top-1 agreement. The output directory (`--output`, or one under
`profiles/perplexity/`) holds `rows.tsv`, one line per row, and `report.json` with the summary.

### Reading the result

Basalt found about 0.01-0.02 KLD of noise in the reference itself, so the absolute KLD against the
base is not a quality verdict on its own. Compare builds with each other: run the same base,
chunks, window and KV format with the change off and on, and compare the two summaries (or
`rows.tsv` row by row). Keep `--decode-window` fixed across a comparison, because different
windows run different kernels.
