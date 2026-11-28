# NInfer CLI

`build/apps/ninfer` runs one request against one v3 `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use the Qwen3.8-27B artifact that `download-model qwen38-27b` writes to
`models/qwen3_8_27b.ninfer`, with INT8 KV storage. [Common options](#common-options) lists
every flag and its default; `./build/apps/ninfer --help` is the exact option contract.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, GPU memory, and speculative-decoding statistics are
written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype int8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

## Constrained output

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-new 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set.
`--regex PATTERN` constrains the complete answer to a regular expression:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''`
permits only empty content. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice)
for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON
Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-new 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with
ordinary decoding, MTP, DFlash and DFlash2. Thinking may precede the constrained answer; an output
limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages:
the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools.
Constraints reject custom stops and `--raw-output`. JSON may use any layout (compact or indented, with
whitespace runs capped at 32 characters) and follows declared property order. Describe the desired content in the prompt; the schema is not added to it automatically. See the
[supported schema subset](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同).

## Thinking and reasoning

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking` or
`--reasoning-effort none`, which both disable thinking, but it can be combined with any other
`--reasoning-effort`.

`--reasoning-loop off|stop|conclude` (default `off`) ends the reply (`stop`) or closes the thinking
with the same control span and lets the model answer (`conclude`) when the open thinking keeps
repeating whole passages: every 512 thinking tokens, 25 % of the last 2,000 words inside a 12-word
passage seen three times. `conclude` stops instead when the output left cannot hold the control
span and one more token. The run summary prints `reasoning loop` when it fires. It cannot be
combined with `--no-thinking`. The check and its limits are described in
[serving](serving.md#openai-chat-completions), under `--reasoning-loop`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
The inserted suffix is never truncated, and a request is never rejected for lacking room for it.
When the output capacity left after the budget cannot hold the complete tokenizer-derived control
suffix plus one post-close model token, Engine lowers the effective budget so both fit. When the
whole capacity is no larger than that, the budget cannot be enforced, so thinking runs to the
`--max-new` limit and may end inside the reasoning. The run summary prints `effective thinking
budget` when it differs from the requested one, and nothing extra when the budget is unenforced; the
exact rule is in [Chat Completions](serving.md#openai-chat-completions).

Normal output sends the inserted guidance to stderr as reasoning. `--print-token-ids` includes the
inserted IDs, while `--raw-output` preserves the raw control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths. Image
and video input needs `--vision`:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype int8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Common options

The table lists executable defaults. The examples above select INT8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024`; Inkling-2: the widest of 2048, 1024, 512 and 256 that fits beside `--max-context` (prefix reuse resumes at a multiple of the chunk, so a wider explicit chunk such as 4096 or 8192 prefills a long prompt faster but resumes less of a chat) |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see the [README](../README.md#several-gpus-pipeline-stages---devices-ab)); overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|nvfp4\|k8v4\|fp16\|fp32` | KV-cache storage; see [Context and memory](#context-and-memory). `rk8v4` is opt-in RotorQuant and `rk4v4` opt-in Lloyd-Max 4-bit keys; all seven are accepted on this fork's sm_86/sm_89 targets. A per-layer spec such as `rk4v4x12,int8x4` (opt-in, lossy; Qwen3.5-family models) gives each full-attention layer its own `int8`, `rk8v4` or `rk4v4` format, runs of `FORMATxCOUNT` in model order; see [serving.md](serving.md). Inkling-2 keeps its own KV rings and takes only `fp16` (the default: **lossy**, at the noise floor of an FP32 reordering, half the bytes of FP32 so 256K context fits on a 32 GB card at a prefill chunk of 1024 without `--spec dspark` and `--vision`; with both it fits at a chunk of 256, the drafter windowed to 16384 positions and reading the target's head, see [the Inkling-2 notes](maintainer/inkling2-model.md#context-with-dspark-and-vision)) or `fp32` (the exact FP32 program, for reproduction; about half the context: it also selects `--prefill-math fp32` unless `--prefill-math tc` is given, which is not exact); see [the Inkling-2 notes](maintainer/inkling2-model.md#fp16-kv-rings). Inkling-2 also takes `int8`, `rk8v4` and `rk4v4` as **lossy, opt-in** low-bit global rings (the sliding rings stay FP16; one request at a time): they free 3968, 4928 and 5952 MiB at 262,144, enough for DSpark's Q8 head and whole-context window beside `--vision`, and decode 11% faster at 100K and 17-18% at 200K, at a KLD from FP16 of 1.06x (`int8`), 2.0x and 2.8x (`rk4v4`) the FP32 noise floor; prefill past 100K is 5-8% slower. See [the Inkling-2 notes](maintainer/inkling2-model.md#low-bit-global-kv-rings-opt-in). `fp16` and `fp32` are Inkling-2 only | `bf16` (Inkling-2: `fp16`) |
| `--dense-q8 head\|all` | Inkling-2 only. **Lossy**, opt-in: the output can differ from the default's, token for token. Re-encodes the BF16 LM head (`head`) or the head and every attention projection (`all`) to 8 bits at load (the artifact is unchanged). The default keeps the BF16 weights, exact; see [the Inkling-2 notes](maintainer/inkling2-model.md#dense-8-bit-re-encoding) for the measured KLD and speed | off (BF16) |
| `--mxfp8-experts auto\|on\|off` | Inkling-2 only. **Exact**: the output is the same, bit for bit. The artifact keeps layers 20-23's experts in BF16 (7.6 GiB); they are the checkpoint's MXFP8 values, so `on` re-encodes them at load to MXFP8 (3.9 GiB), read from the file and never uploaded as BF16. `auto` does so when the BF16 weights would leave less than 4 GiB of the GPU free: on an RTX 3090 that is what lets DSpark, vision and long context fit. A Tesla V100 (`sm_70`) always re-encodes under `auto`: its MXFP8 kernels are faster (12 chats +1.5% plain, +1.8% DSpark at 262,144) and free 3.6 GiB. Startup refuses if a value is not exactly MXFP8. See [the Inkling-2 notes](maintainer/inkling2-model.md#mxfp8-experts) | `auto` |
| `--no-split-attention` | Inkling-2 only. Decode and DSpark verification split each attention column's keys into blocks of 256 by default (FP32, but the softmax sums run in another order, so the last bits differ past 256 visible keys; 1.4x decode at 1K context, 2.4x at 16K). This flag keeps the single-block kernel and reproduces earlier builds bit for bit; see [the Inkling-2 notes](maintainer/inkling2-model.md#split-decode-attention) | off (split) |
| `--prefill-math tc\|fp32` | Inkling-2 only. Prefill forwards of 64 or more columns run on the FP16 tensor cores with FP32 accumulation by default (`tc`: about five times the FP32 prefill's speed; **lossy**, at the noise floor of an FP32 reordering of the sums). `fp32` keeps the exact FP32 prefill. With `--kv-dtype fp32` the default is `fp32`; an explicit `tc` there runs the tensor cores over FP32 rings (not exact). Decode is FP32 either way; see [the Inkling-2 notes](maintainer/inkling2-model.md#tensor-core-prefill) | `tc` (with `--kv-dtype fp32`: `fp32`) |
| `--spec mtp\|dflash\|dflash2\|dspark` | speculative backend; see [Speculative decoding](#speculative-decoding) (`dspark`: Inkling-2, greedy and sampled requests, `--draft-tokens` default 7 on V100 (`sm_70`) builds and 3 elsewhere, with `--draft-min-p` and `--dspark-adaptive`) | off |
| `--draft-min-p P` | Inkling-2 `--spec dspark`: a round's drafts stop before the first whose drafter probability is under P, in [0,1); `0` keeps every draft. Only speed changes, never the output | `0.5` |
| `--dspark-adaptive on\|off` | Inkling-2 `--spec dspark`: each round drafts 1 to `--draft-tokens`, the count recent acceptance and measured round costs favour; `off` drafts `--draft-tokens` every round. Only speed changes | `on` |
| `--dspark-window auto\|N` | Inkling-2 `--spec dspark` only (refused otherwise, `auto` included): the drafter's global layer attends over the last N context positions (`0`: the whole context; else at least the drafter's sliding window). Only the drafts change, never the output. `auto` keeps the whole context when it fits beside `--max-context` and falls back to 32768, then 16384 | `auto` |
| `--draft-model DIR` | Inkling-2 `--spec dflash2` only (refused otherwise): drafts with Inco's DFlash2 drafter read from its Hugging Face release directory (`config.json` and `model.safetensors` of `incoai/Inkling-2-Mini-DFlash2`) in DSpark's place. It takes DSpark's round options and defaults, `--draft-tokens` 1 to 7; only the speed changes, never the output. See [Speculative decoding](#speculative-decoding) | unset |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 (Inkling-2 `--spec dflash2 --draft-model`: `1..7`) | unset |
| `--lm-head-draft` | optimized proposal head; implied by `--spec`, accepted for compatibility | on with `--spec` |
| `--ngram-draft-tokens 0\|15` | n-gram copy drafting beside `--spec dflash2`: when the text being written already appeared in the request (the prompt, a tool result -- including `cat -n` numbered ones -- or the output so far), a round verifies up to 15 copied tokens instead of the draft model's proposal; verification licenses every token, so a wrong copy costs speed, never output. Copies run at one or two lanes in a round; see [Speculative decoding](#n-gram-copy-drafting) | `0` (off) |
| `--adaptive-draft-tokens 0\|3` | adaptive draft window beside `--spec dflash2` with `--draft-tokens` above 3: a second, narrower neural round (3 drafts) is captured next to the configured one and a round runs it when the requests' recent acceptance is low, since a captured graph costs the same however few drafts are live and only its captured width matters; it costs one more set of CUDA Graphs (startup refuses when they do not fit; it is not available at the concurrency where DFlash2 itself does not fit) and the target still verifies every token. Experimental: measured on an RTX 3090 at +2.9% on a low-acceptance prompt and no worse than -0.5% elsewhere, short of the +5% aimed for, and it can change greedy output; see [Adaptive draft window](#adaptive-draft-window) | `0` (off) |
| `--ngram-min-match N` | tokens the end of the text must match before a copy is proposed, `4..64` | `12` |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)). Inkling-2 (with `--prefill-math tc`): the loop controllers and router run as FP32 cuBLAS SGEMMs and each chunk's busiest experts (512 or more routed positions) as FP16 cuBLAS GEMMs on exactly decoded weights, and layers 20-23's other BF16-bank experts on the tensor-core kernels instead of per-expert cuBLAS calls; the result stays at the FP32-reordering noise floor, and `--prefill-math fp32` is unaffected (see [the Inkling-2 notes](maintainer/inkling2-model.md#cublas-prefill---prefill-cublas)) | off; **on** in an `sm_70` (V100) build, where the integer kernels are emulated ([tesla-v100.md](tesla-v100.md)) |
| `--no-prefill-cublas` | turn the route off, for an `sm_70` build whose default is on | |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--no-prefill-a8` | return full prefill tiles to their A16 routes, which is how the integer routes are measured on a whole request | integer routes on |
| `--lm-head-q4\|--lm-head-q6` | narrower language-model-head matrix: a speed- or memory-for-quality trade (DFlash and DFlash2 refuse it) | off |
| `--embedding-q4\|--embedding-q6` | narrower token-embedding matrix, freeing weight memory for context | off |
| `--mtp-experts-q4` | transcode the 35B-A3B's MTP draft-layer experts to Q4/Q6 at load; rejected on the dense 27B | off |
| `--gdn-state-fp16` | FP16 recurrent GDN state, halving each state image | off |
| `--mlp-a8-decode` | integer-activation MLP at decode | off |
| `--verify-a8`, `--no-verify-a8` | `sm_70` (V100) only, Qwen3.5-family models: the Q4/Q5/Q6 linears at 1-8 tokens (plain decode, the MTP and DFlash verify forwards, the LM head) round their activations to int8 per 32 values and run `dp4a`, so a verify round costs about the weight stream at any width. On Qwen3.6-35B-A3B the Q8 mixer projections (12,288 x 2,048, 9,216 x 2,048, 2,048 x 4,096) do the same at 2-8 tokens (a verify; one-token steps stay A16): MTP3 5-14% faster, decode-path KLD at four tokens 3.37e-3 against 2.71e-3; `NINFER_SM70_Q8_VERIFY_A8=0` keeps only those on A16. Lossy at the noise floor: on Qwen3.8-27B, KLD 7.2e-4 against the FP32 decode path and 98.9-99.2% same top-1. Greedy text differs from `--no-verify-a8` at near-ties, and speculative output can still part from plain decoding's, as it does without the flag. `--no-verify-a8` returns those calls to the FP32 kernels; elsewhere the flag is inert. See [tesla-v100.md](tesla-v100.md#qwen38-27b) | **on** in an `sm_70` build; off (inert) elsewhere |
| `--numeric-tripwire [--numeric-tripwire-limit F\|inf]` | debugging aid: a few small probes in every decode round look for NaN, infinity and values beyond the limit (default `65504`, the FP16 maximum; `inf` reports only NaN and infinity) and log the first sighting as one warning line, naming where it was seen, without stopping generation. Single device, Qwen3.5 family; see [numeric tripwire](maintainer/quality-trade-experiments.md#numeric-tripwire----numeric-tripwire) | off, and then free |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay` | `overlay` keeps the Vision tower host-pinned and borrows device memory per image from free KV pages, or the evictable text weight tail when those fall short (no resident Vision cost; needs CUDA VMM). Inkling-2: the tower stays pinned and is copied into the text workspace for each image (about 0.2 s an image; it saves memory only at a prefill chunk of 512 or more), see [the Inkling-2 notes](maintainer/inkling2-model.md#context-with-dspark-and-vision) | `resident` |
| `--vision-max-merged N` | merged-token budget of one media item; larger media downscales at preprocessing | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-loop off\|stop\|conclude` | end the reply or close the thinking when it keeps repeating whole passages | `off` |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding, independent of the thinking options | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |
| `--stop-token-id N`, `--stop TEXT`, `--reasoning-stop TEXT` | add a stop token ID, a stop string matched in the answer content, or a stop string matched in the reasoning; each may be repeated | none |
| `--grammar-file FILE`, `--json-object`, `--json-schema-file FILE`, `--choice TEXT`, `--regex PATTERN` | constrain the answer; mutually exclusive, see [Constrained output](#constrained-output) | none |
| `--raw-output` | expose the frontend's raw output stream | off |
| `--print-token-ids` | include generated token IDs in diagnostics | off |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | diagnostic verbosity on stderr | `info` |

The `--lm-head-*`, `--embedding-*`, `--mtp-experts-q4`, `--gdn-state-fp16`, `--mlp-a8-decode` and `--verify-a8`
flags are speed- or memory-for-quality trades. The measured cost of each is in
[quality trades](maintainer/quality-trade-experiments.md).

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP, the 35B-A3B DFlash or the Qwen3.8-27B
DFlash2 backend with one to fifteen draft positions. Only one backend can be enabled per Engine, and
Every backend uses the optimized proposal head: `--spec` implies it, and `--lm-head-draft` is still accepted but
requires a selected backend. Both
masked-draft backends may be combined with `--vision`:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype int8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --kv-dtype int8 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--vision`; this is what
the launchers pass. DFlash2 accepts every draft count from 1 through 15. Both `groupwise-int` and
`nvfp4` artifacts use the same Engine route, including CUDA Graph, concurrent requests, sampling
penalties, and prefix reuse. An artifact without the companion weights reports a missing DFlash2
component when selected. Vision, MTP and DFlash follow the same rule: their weights are required
only when that component is enabled at startup.

`--spec dspark` runs Inkling-2's DSpark block drafter (Inkling-2 artifacts only; the drafter ships in the
artifact). `--draft-tokens` is optional, 1 through 7, default 7 on V100 (`sm_70`) builds and 3 elsewhere: the most drafts a
round verifies.
Greedy and sampled requests speculate. The output is the same as without it, token for token, including
stop strings, the thinking budget's forced close and the context limit; a sampled request draws the
tokens plain sampling draws under the same seed (its drafts are the drafter's draws under the uniforms
the target draws next, so they only save forwards). `--dspark-adaptive on` (the default) picks each
round's draft count from 1 to `--draft-tokens` from the request's recent acceptance and the measured
round costs, and `--draft-min-p 0.5` (the default; `0` turns it off) drops the drafts from the first
one the drafter gives under that probability on. A round runs only while its measured cost beats the
plain steps it replaces; otherwise the request decodes plainly and probes again after 8 to 64 steps.
Relaxed drafting, `--draft-context-rows`, `--ngram-draft-tokens` and `--adaptive-draft-tokens` are
refused. With `--max-concurrency` above 1 requests decode without DSpark.

`--spec dflash2 --draft-model DIR` drafts with Inco's DFlash2 drafter for Inkling-2-Mini instead
(`incoai/Inkling-2-Mini-DFlash2`; `DIR` is the downloaded release, its `config.json` and `model.safetensors`, read
as-is). Rounds, verification and every option above work as with DSpark, so the output is again the plain decode's.
It accepts more drafts a round than DSpark (3.4 against 2.4 of 7 over 12 chat requests in a reference run) and is
the faster drafter on chat and prose; see [the Inkling-2 notes](maintainer/inkling2-model.md#dflash2-drafter).
`ninfer-serve --speed fast` (lossy, opt-in,
server only) trades exactness for more speed on top of these defaults; see [serving.md](serving.md). Measured on a Tesla V100 (master
`dad1623b`, 2026-11-22): up to 120.7 tok/s on a maths prompt; over 12 chat requests of 300 tokens (ABBA), 93.4-94.4
tok/s greedy against 78.5-78.8 plain, and 83.4-83.5 against 74.9 plain sampled at temperature 0.8 (the same tokens).
The defaults beat a fixed 3 drafts by 8.4% (88.8 against 81.9 tok/s, 2026-11-21); see
[the Inkling-2 notes](maintainer/inkling2-model.md#draft-floor-and-adaptive-width).

### Choosing a draft count

The draft count is a trade on what the output looks like. Each round verifies K+1 columns and runs
K draft-head steps whether or not the drafts survive, so a larger K pays only where the head keeps
guessing right.

- **MTP:** three is the default. It is best or within 2% on prose, and larger counts lose up to 38%
  there. When the output mostly reproduces the input -- refactoring, renaming, applying an edit and
  returning the whole file -- 11 to 15 is up to 1.85x faster than three, and for a coding assistant
  that mostly writes new code seven is about 10% faster. The former MTP-only context lookup
  (`--lookup-ngram`) added nothing on top of MTP there, because the head already copies; it has
  been removed in favour of n-gram copy drafting beside DFlash2 (below).
- **DFlash2:** seven is the checkpoint recommendation and the best mean on this card. The best K
  still depends on the workload, so a deployment serving one kind of work should sweep its own.
  The optimized proposal head, which `--spec` now always enables, measured within noise of the full head for
  DFlash2 at every count.
- **DFlash:** seven forms the measured block length eight; fifteen uses the maximum supported block
  length sixteen.

The tables and sweeps are in [performance](performance.md#choosing-the-draft-count-rtx-3090-qwen38-27b).
The published [performance results](performance.md) use MTP with three draft tokens and DFlash with
seven, both with the optimized proposal head. Draft counts of eight and above add a second CUDA Graph
topology class on the 27B, which reserves about 64 MiB more per lane.

### N-gram copy drafting

`--ngram-draft-tokens 15` (with `--spec dflash2`) adds copy rounds. Before each round the request's
own text is searched for the last 12 or more tokens written (`--ngram-min-match`): the prompt, each
tool result, a de-numbered copy of each `cat -n` / Read-style numbered tool result, and the output
so far. When they occurred before, the round verifies the 15 tokens that followed them there in one
16-column pass and skips the draft model; otherwise it is an ordinary 8-column DFlash2 round. It
pays when the answer repeats its input -- returning a file, applying an edit, quoting a tool result.
As with any draft, every token is licensed by the target's own verification, and a copy that is
wrong, or that a grammar forbids, is rejected; a request whose copies keep missing asks for a longer
match before it copies again. With two lanes a round copies only when a copy offers at least 12
tokens and the other lane can afford it: that lane's row is verified at the copy width too and
decodes about a third slower in that round, so each request keeps a balance -- a small allowance per
round it decodes, minus each widening it took, plus what its own copies saved it -- and a lane
without a copy is widened only while its balance covers the cost. A request that never copies is
therefore slowed by at most about 3% of its decode-round time; with three or more lanes in a round
nothing is copied. The two-lane costs were measured at `--draft-tokens 7` only, so at any other
draft count a two-lane round does not copy either (one lane still does). The counters are in
the request log and `/metrics` (see [serving](serving.md)). On an agent replay at one lane it
decoded 42% faster overall and up to 76% faster on turns that return a file, at about 0.7% on
output that never copies. At two lanes the replay's batch throughput rose 15%, and turns that copy
little stayed within 1% of their speed without copies; see
[performance](performance.md#n-gram-copy-drafting-rtx-3090-qwen38-27b).

### Adaptive draft window

A DFlash2 round costs the same whatever number of its drafts are live (the captured graph runs every
column), and a wide window wastes time when most drafts are rejected: on a low-acceptance workload
K = 3, 7 and 12 decode at 58.9, 53.2 and 46.1 tokens/s. `--adaptive-draft-tokens 3` (with
`--spec dflash2` and `--draft-tokens` above 3) captures a second 3-draft round next to the
configured one. Before each round the engine reads every request's recent acceptance (the drafts
accepted against the drafts offered, weighted toward the last ten rounds) and runs the window that
commits more tokens per second of round time. It leaves a window only for one at least 5% better and
only after eight rounds, so acceptance sitting at the crossover does not flap; requests in the same
round share one graph, so a batch runs the window that is best for the batch as a whole. It starts
every request on the configured window. The target verifies every token at either width, so output
stays what the target would write; the one difference from the fixed window is the different
column count of the target's verify pass, as between any two `--draft-tokens` values. The second
set of graphs is part of the startup memory plan; where it does not fit, startup refuses and the
flag has to be dropped. `request_done.speculative` reports `adaptive_draft_tokens` and
`adaptive_low_rounds`, the rounds the request ran at the narrow window. Off by default and in no launcher profile.

Measured on an RTX 3090 (27B, DFlash2 at K = 7 against `--adaptive-draft-tokens 3`, greedy, INT8 KV,
one to four lanes, three prompts of low, middle and high acceptance): the best cell is +2.94%
(one lane, low acceptance, where the narrow window ran for 885 rounds) and the worst is -0.50%
(one lane, middle acceptance), so no cell fell below the 0.98x bound. It missed the 1.05x aimed for
on the low-acceptance prompt. At two and four lanes the narrow window never ran: lanes sharing a
round share one graph, so a mixed batch rarely favours the narrow one, and the results sit within
0.2% of the fixed window. It also changes greedy output when it switches (the verify pass runs at a
different width, as between a fixed K = 3 and K = 7, which differ on all three prompts), so do not
use it where a byte-stable answer matters. Details in TODO.md.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical allocation
on one RTX 3090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations; `--kv-dtype`
independently selects runtime KV storage.

### KV storage

All seven formats — `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `k8v4`, `nvfp4` — are accepted on SM86;
measured size, decode speed and perplexity for each are in
[`docs/config-calculator.html`](config-calculator.html). The Blackwell-only
`mma.sync...kind::f8f6f4` restriction applies to FP8/NVFP4 *weights and activations*, not to KV
storage.

- `bf16` is the executable default.
- `int8` is the quality reference: use it when quality matters more than context.
- `rk8v4` is the best all-round choice: rotated INT8 keys with a packed signed int4 value plane,
  about 23% smaller than INT8 for about 0.082% perplexity, and the flattest decode curve of any
  format measured.
- `rk4v4` keeps `rk8v4`'s values and stores keys as 4-bit Lloyd-Max indices: 31% smaller than
  `rk8v4`, within 3% of `nvfp4`'s size, better perplexity than `nvfp4` (+0.21% against INT8) and
  `rk8v4`'s decode speed, so it is the choice when context is the limit. The launchers use it.
- `nvfp4` buys the most context, 45% smaller than INT8, at about 13% of decode speed at a 32K
  cache depth.
- `fp8` and `k8v4` are each beaten by `rk8v4` on size, speed and quality together, so neither has a
  niche.

### Capacity

The prepared prompt must fit `--max-context`; generation stops at the remaining context capacity
when necessary. `--kv-capacity N` controls the shared physical Main Text KV pool independently and
is rounded up to the 64-token page size. `--kv-capacity auto` loads the selected weights, measures
the remaining GPU memory, and directly chooses the largest legal page capacity for the complete
enabled runtime layout. This includes the selected speculative backend, fixed sequence state,
unified workspace, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows `--max-context`; the distinction matters
primarily to a concurrent Engine or server.

### Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only the selected
  speculative backend;
- the optimized proposal head is loaded whenever a speculative backend is selected;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input;
- the one-request CLI disables cross-request history and Host context backing, so it does not
  reserve an extra Device checkpoint StateImage or retain continuations.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and do
not accelerate Vision encode. The default speculative and Vision settings produce the smallest
resident profile.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
