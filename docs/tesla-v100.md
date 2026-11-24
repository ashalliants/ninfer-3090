# NInfer on a Tesla V100

NInfer runs on a 32 GB Tesla V100 (Volta, `sm_70`), one of the project's two first-class cards
alongside the RTX 3090. Three models are supported and tuned there: **Qwen3.8-27B**,
**Qwen3.8-Flash-Next**, the 512-expert MoE with the full 262,144-token context, and
**Inkling-2-Mini**, the looped 128-expert MoE with images and DSpark speculative decoding
([spec and measurements](maintainer/inkling2-model.md)). This page covers what you need, how to build and run
the two Qwen models, their measured performance, and the limits; Inkling-2-Mini's run command and measurements
are in its [model card](../model-cards/Inkling-2-Mini-NInfer-V100/README.md) and its spec.
How the port works, and every measurement behind it, is in [docs/maintainer/volta-port.md](maintainer/volta-port.md).

Each release ships a prebuilt V100 archive for Linux, `ninfer-v100-linux-x64-<version>.tar.gz` (the 3090 archives are `sm_86` and do not start on a V100): unpack it, install the CUDA 12.8 runtime libraries and FFmpeg 6 it lists, and use its `run-v100.sh`. Or build from source on Linux, below.

## At a glance

All figures were measured on one V100-PCIE-32GB, connected over an OCuLink PCIe Gen3 x4 eGPU link
(3.3 GB/s host-to-device), on a Linux host with 59 GiB of RAM, with a 250 W power limit
(2026-11-16 to 2026-11-18), except Flash-Next's MTP decode and the 27B's speculative decode, re-measured
at 200 W on 2026-11-24.
Decode is greedy with MTP speculation, as `run-v100.sh` serves it (three drafts; seven for Flash-Next).

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---:|---:|
| **Context** | 262,144 tokens | **262,144 tokens** (`run-v100.sh` default) |
| **Decode**, one stream, MTP | **110 tok/s** on code, 104 thinking, 77-81 story, Chinese and French (MTP3, 2026-11-24, 200 W) | **124.7 tok/s** on code, 101 thinking, 80-81 story and Chinese, 72 French (MTP7 + draft vocabulary, warm cache, 2026-11-24, 200 W) |
| **Decode**, plain | 45 tok/s (200 W) | 53 tok/s (250 W) |
| **Prefill**, ~4K-token prompt | **~1,020 tok/s** | **~990 tok/s** (3,798 tokens in 3.8 s) |
| **Prefill**, long prompts | | 38K tokens in 38.4 s (~1,000 tok/s) |

Flash-Next's decode does not slow down with context. From 10K to 221K tokens of context it stays at
27-29 ms per MTP verification round (69-79 tok/s on those prompts). What moves tok/s is how many
drafted tokens are accepted, which depends on the text, not on the context length.

**Inkling-2-Mini** on the same card (master `dad1623b`, 2026-11-22, 250 W limit, through `ninfer-serve`, except the
rows marked 200 W, re-measured 2026-11-23 (prefill 2026-11-24); the
[model card](../model-cards/Inkling-2-Mini-NInfer-V100/README.md) has the conditions and ranges):

| | Inkling-2-Mini |
|---|---:|
| **Context** | 262,144 tokens with DSpark and vision |
| **Decode**, one stream, DSpark | **up to 120.7 tok/s** (maths prompt); 93.4-94.4 average over 12 chats |
| **Decode**, plain | 78.5-78.8 tok/s; 48 tok/s at 100K context |
| **Decode at 262,144** (200 W) | **94.3 tok/s** DSpark, **78.0** plain, average over 12 chats (`--max-context 262144 --vision --kv-dtype int8`, master `dc867300`) |
| **Decode**, `--max-concurrency 8` | **198.2 tok/s** aggregate (plain) |
| **Prefill** (200 W) | **2,727 tok/s** at 8K (131,072 context, `--prefill-chunk 8192`); 2,227 at 34K, 1,577 at 100K. At the default chunk 2048: 2,170 at 8K, 1,962 at 1K, 1,339 at 100K. At 262,144 with DSpark and vision (chunk 2048): 2,125 at 8K, 1,930 at 1K, 1,324 at 100K |
| **Low-bit KV** (`--kv-dtype rk4v4`, opt-in, lossy) | plain decode **41.1 tok/s at 200K** (FP16 34.8), 53.2 at 100K (47.8); frees 5,952 MiB at 262,144; KLD 2.8x the FP32 floor (`int8`: 1.06x, 40.7 tok/s) |
| **Load** | 9.2-14.2 s |

## What you need

- **GPU:** a Tesla V100 with 32 GB. A 16 GB V100 has not been tried.
- **Host link:** any PCIe link works. The figures above come from a slow eGPU link (Gen3 x4, 3.3
  GB/s), and Flash-Next's prefill is partly bound by it. A full x16 slot (about 12 GB/s) should
  prefill faster; this has not been measured.
- **Toolkit:** CUDA 12.8 or 12.9. CUDA 13 dropped `sm_70`.
- **Qwen3.8-27B:** the 27B artifact and the card. Host RAM needs are modest.
- **Qwen3.8-Flash-Next:** about 64 GB of host RAM (the test host has 59 GiB). The experts (33 GiB) are pinned in RAM and
  cached in VRAM as they are used, so set the memory-lock limit high enough (`run-v100.sh` says how).
  The model takes about 39 GB on disk plus a 28 GB n-gram table, which is read from disk during
  prefill, so put it on an NVMe drive.
- **Cooling:** a V100 is passively cooled and needs real airflow. Sustained prefill holds it at its
  power limit. On the test box it peaked at 68 C over a 221K-token prompt. Watch `nvidia-smi -l 1`
  the first time on a new setup.

## Build

```bash
scripts/v100_build.sh ninfer ninfer-serve
```

`scripts/v100_build.sh` builds in `build-v100/` inside the checkout (set `BUILD_DIR` to move it).
Each worktree gets its own tree, and ccache shares objects between them. It configures with
`-DCMAKE_CUDA_ARCHITECTURES=70`, uses ccache, and limits
parallel compile jobs to fit the host's RAM. One `nvcc` job can peak at 1.5-2 GiB. Set
`NINFER_BUILD_TESTS=1` to configure the tests as well.

## Run

```bash
scripts/run-v100.sh qwen38-flash-next     # or: scripts/run-v100.sh qwen38-27b
```

`run-v100.sh` finds the `sm_70` server in `build-v100/` (or uses `NINFER_SERVER`), checks that the
card is a V100 with nothing else running on it, and starts `run.sh` with the V100's measured
defaults. Every `NINFER_*` variable you set still wins.

The server listens on every interface (`0.0.0.0`), so other machines on the network reach it at
`http://<this machine's IP or host name>:8080`. Set `NINFER_API_KEY` as well: without a key,
anyone who can reach the port can use the model, and the launcher and server both warn about it.
`NINFER_HOST=127.0.0.1` keeps it local to this machine.

These are the model defaults it sets:

| Setting | Model | Effect on the V100 |
|---|---|---|
| `NINFER_CONTEXT=262144` | Flash-Next | Serves the full context. The KV cache starts at 496 MiB and takes VRAM from the expert cache only as a conversation grows, up to 3.6 GiB at 256K tokens |
| `NINFER_SM70_TENSOR_CORES=1` | Flash-Next | Prefill linears and prompt attention run on the FP16 tensor cores: a 3.9K-token prompt goes from 6.9 to 6.0 s. Output is close to the plain server's but not bit for bit |
| `NINFER_PREFILL_CPU_ALL_COLUMNS=1024`, `NINFER_PREFILL_CPU_MAX_COLUMNS=64`, `NINFER_PREFILL_CPU_SHARE=0.5` | Flash-Next | The CPU computes prefill experts that are not cached on the GPU and used by only a few tokens, instead of copying them over the link. Same output |
| `NINFER_NGRAM_RAM_MIB=auto` | Flash-Next | Keeps about half the n-gram table (13 GiB on a 59 GiB host) in RAM after the experts and context cache are sized, which halves a prompt's disk reads |
| `NINFER_MTP_STAGED_DRAFTS=on` | Flash-Next | Drafts one token at a time and stops when no further draft could be accepted: up to +3.6% decode (story, Chinese, French; code unchanged), same output |
| `NINFER_CPU_VERIFY_MISSES=on` | Flash-Next | During decode, experts that miss the cache are computed on CPU threads beside the GPU. A cold cache decodes at 68-80 instead of 48-63 tok/s |
| `NINFER_PARTIAL_RAM_RESIDENCY=on` | Flash-Next UD-IQ3_XXS | Pins only the experts the VRAM cache does not hold (`--partial-ram-residency`), so the 42 GiB bank fits; no images, context 32,768. `off` pins every expert (about 50 GiB) and restores images and the 262,144-token context |
| `NINFER_HOT_EMBEDDING_ROWS=8192` | Flash-Next | Only the common embedding rows stay in VRAM, freeing room for 216 more cached experts |
| `NINFER_SPEC=mtp`, `NINFER_KV_DTYPE=int8` | 27B | MTP3 with INT8 KV: on this card it beats DFlash2 (`--draft-tokens 7`) on story, Chinese, French and sampled prose (77-81 tok/s); DFlash2 wins on code (136 against 110 tok/s), thinking (125 against 104) and long structured output (177 against 123) |
| `NINFER_DRAFT_A8=0` | 27B, DFlash2 | Runs the DFlash2 drafter's linears on its A16 kernels instead of the default int8-activation `dp4a` route: about 9 ms slower per K = 7 round, the same acceptance on most measured prompts |

One default comes from the build rather than the launcher: in an `sm_70` build `--prefill-cublas` is
on, so the 27B's wide prefill GEMMs run on the tensor cores through cuBLAS instead of emulated integer
kernels. For Inkling-2-Mini it moves the tensor-core prefill's loop controllers and router to cuBLAS
SGEMM (FP32) and each chunk's busiest experts (512 or more routed positions) to FP16 cuBLAS GEMMs on
exactly decoded weights: through the server at 8K and the default chunk, 2,069 tok/s against 1,893
(+9%, 200 W); on the program at chunk 8192, +21% (2,514 against 2,082). Layers 20-23's BF16-bank experts
with fewer than 512 positions then moved from per-expert cuBLAS calls onto the tensor-core kernels: at the
default chunk 2,170 tok/s at 8K against 2,071, 1,962 against 1,807 at 1K and 1,339 against 1,300 at 100K
(131,072 context, 200 W, 2026-11-24), with the same KLD. `--no-prefill-cublas` (`NINFER_NO_PREFILL_CUBLAS=1`) turns it off.

Inkling-2-Mini's prefill chunk stays 2048 by default, because prefix reuse resumes a chat at a multiple
of the chunk: a second turn at 8K answers in 1.03 s at chunk 2048, 1.77 s at 4096 and 3.04 s at 8192.
For long single-shot prompts, `--prefill-chunk 4096` or `8192` is the throughput setting (2,227 tok/s
at 34K at chunk 8192; chunk 4096 gave 2,004 before the BF16-bank change and was not re-measured). Chunk
8192 needs 1,746 MiB of workspace (chunk 4096 needed 872 before that change): at 262,144 with DSpark and
vision they fit with `--kv-dtype int8`, not with the default FP16 KV.

### Refusal steering (Flash-Next)

The community refusal-direction vector for Qwen3.8-Flash-Next ships in the Strata repository as
[`data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf`](https://github.com/Niko1221/Strata/tree/main/data/experimental-speed-projection),
480 KB. Strata presents it as an off-by-default "experimental speed projection", but its README
says what it is: a refusal-direction projection that makes the model decline far fewer requests,
removing a safety behaviour. The Qwen Community License of the model applies to it.

The file holds directions for layers 1-47, but its authors apply only layers 4-44 (`layer_range`
in the JSON beside it; their scores are for that range). `--graft` applies every direction in the
file, so make a copy with only those layers, then load it:

```bash
python3 tools/control_vector_layers.py Qwen3.8-Flash-Next-experimental-speed-projection.gguf \
    flash-next-refusal-L4-44.gguf --first 4 --last 44
NINFER_GRAFT="uncensored=$PWD/flash-next-refusal-L4-44.gguf" scripts/run-v100.sh qwen38-flash-next
```

A request then opts in with `"graft": "uncensored"`. `NINFER_DEFAULT_GRAFT=uncensored` applies it to
every request that names no graft, and a request opts out with `"graft": ""`. On the V100,
`ninfer_qwen4_exp_steering_real_test` passes with this copy, and a steered request gives a
different answer from the same unsteered one (2026-11-18). Loaded, the vector costs about 50 us
per token even for requests that do not select it; with no vector loaded, nothing changes.

With the vector on, the model can reason for a long time about a prompt it would otherwise refuse,
and at temperature 0 it can loop in its reasoning. On the V100 (2026-11-19), two such prompts at
temperature 0 reasoned to a 12,000-token limit repeating the same few lines; sampled at
temperature 0.6 they answered after 3,200-7,500 tokens (44-99 s). Narrowing the vector's layer
range did not stop the loops. Three settings did, each request answering without a refusal:

- **A presence penalty** of 1.0 (`NINFER_PRESENCE_PENALTY=1.0`) broke the loop on its own: the same
  temperature-0 request then answered after 2,742 thinking tokens.
- **A thinking cap** (`NINFER_DEFAULT_THINKING_BUDGET=1024`) bounds the time: those prompts then
  answered in 16-25 s, at temperature 0 and 0.6. It caps every request; a request can raise it with
  `"thinking_budget"`.
- **The reasoning-loop guard** (`NINFER_REASONING_LOOP=conclude`) is a backstop. It did not need
  to fire once the presence penalty was set.

```bash
NINFER_GRAFT="uncensored=$PWD/flash-next-refusal-L4-44.gguf" NINFER_DEFAULT_GRAFT=uncensored \
NINFER_PRESENCE_PENALTY=1.0 NINFER_DEFAULT_THINKING_BUDGET=1024 NINFER_REASONING_LOOP=conclude \
NINFER_TEMPERATURE=0.6 NINFER_TOP_P=0.95 NINFER_TOP_K=20 scripts/run-v100.sh qwen38-flash-next
```

### Refusal steering (Qwen3.8-27B)

The 27B loads the GLP control vector
[`BlivionIaG/Qwen3.8-27B-refusal-cvec-GLP-49`](https://huggingface.co/BlivionIaG/Qwen3.8-27B-refusal-cvec-GLP-49)
(`Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1.gguf`, 1 MB: project mode, layers 10-58, alpha 1) as a
`--graft` ([serving.md](serving.md#control-vectors-on-the-dense-models)); it combines with the pi
LoRA (`"graft": "uncensored", "lora": "pi"`). Measured on the V100 (2026-11-20, `--spec mtp`,
INT8 KV, greedy unless stated):

- **Refusals.** Four prompts the base model refuses outright (hotwiring a 1992 Civic, defeating
  supermarket security tags, cheating on an exam, a news paywall): with the vector none opens with
  a refusal. The hotwiring prompt gets step-by-step instructions; the paywall prompt mostly legal
  routes (with pi, also a cookie-clearing tip); the tag and exam prompts are redirected to the
  legitimate process or to preparation rather than refused. The pi adapter alone changes none of the
  refusals, and vector + pi behaves like the vector. A benign prompt (how a fridge works) stays
  coherent in every combination.
- **Thinking.** With thinking on, the steered hotwiring prompt reasoned to a 6,000-token limit
  without answering at temperature 0 and 0.6 (no verbatim loop; with or without pi). With the
  router's 27B sampling (presence penalty 0.5, min-p 0.03) and `thinking_budget` 2048 it answered
  with steps in 47-55 s at both temperatures; the tag prompt answered in 30-38 s.
- **Speed** (300-token decode, 3,067-token prefill, three cooled repetitions): base 65.2 tok/s
  decode, 1,150 tok/s prefill on a server without the vector; base 64.9 / 1,150 with the vector
  loaded (the same tokens and MTP acceptance); steered 62.1 / 1,145 (MTP acceptance 0.51 against
  the base's 0.55 on its different text); pi 58.8 / 1,035; vector + pi 57.3-60.4 / 1,030.

```bash
NINFER_GRAFT="uncensored=$HOME/models/vectors/Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1.gguf" \
NINFER_DEFAULT_THINKING_BUDGET=2048 scripts/run-v100.sh qwen38-27b
```

## Performance

### Qwen3.8-27B

Decode, one stream, through `ninfer-serve` with the router's flags (`--gdn-state-fp16
--prefill-cublas`, INT8 KV, vision overlay), greedy unless noted, 2026-11-24, **200 W** power limit,
no thermal throttling. DFlash2 runs `--draft-tokens 7 --ngram-draft-tokens 15`, plus
`--adaptive-draft-tokens 3` where marked. The current columns are ABBA against the same build with
`NINFER_DRAFT_A8=0` (the drafter on master's kernels); repeats agreed within 0.2 tok/s:

| Prompt | MTP3 | DFlash2 K7 | DFlash2 K7 adaptive | DFlash2 K7 before | MTP3 before `--verify-a8` |
|---|---:|---:|---:|---:|---:|
| code | 110.0-110.1 | **135.5-135.7** | 135.4-135.6 | 110.5-111.3 | 81.2-82.1 |
| think | 104.0 | **125.0-125.2** | 125.1-125.2 | 102.3-103.0 | 77.2-77.9 |
| story | **76.9** | 60.6-60.7 | 60.7 | 52.5-52.8 | 57.0-57.7 |
| Chinese | **81.3-81.4** | 39.7 | 51.2 | 32.4-32.6 | 57.2-57.8 |
| French | **80.6** | 68.3 | 72.9-73.0 | 55.6-56.0 | 57.2-57.6 |
| conversation (T 0.6) | **91.4-91.5** | 85.7 | 81.3-81.4 | 75.0-75.3 | 65.9-66.1 |
| structured JSON | 122.5 | **176.9-177.0** | 176.6 | 146.3-146.6 | 92.3-92.5 |
| prose (T 0.6) | **80.4** | 69.5 | 77.3-77.4 | 56.5 | 58.0-58.1 |
| conversation, greedy | 89.4 | 90.4 | **91.3** | 71.7-71.8 | 68.5-68.6 |
| plain decode (no speculation) | 45.0 | | | | 42.7-43.0 |
| two requests at once (story + code, 256 tokens each), total | **122.0-122.8** | | | | 74.0-74.2 |

So DFlash2 K7 is the fastest setting on code, thinking and structured output, and MTP3 on story,
Chinese, French and sampled prose and conversation. The prompts are the five 300-token ones (code,
think, story, Chinese, French) and Lapilli's basalt-bench fixtures (seed 7). The plain, two-request
and `MTP3 before` figures are from the `--verify-a8` change (ABBA against master `2a1bf0353`).
`ninfer_bench -pg 2048,256` (bench corpus, INT8 KV, 93-95% acceptance): MTP3 **118.9** (88.0 before
`--verify-a8`), DFlash2 K7 141.9 (93.2), plain 43.8 (41.9-42.0). Deeper MTP windows pay only on code
and structured output: MTP5 117.4 code and 148.0 structured, MTP7 158.1 structured, but every other
prompt is slower, so MTP3 stays the default.

DFlash2's gain is its drafter: in an `sm_70` build the block drafter's own linears (the Q8 attention
input, SwiGLU, output and down projections at eight columns, and the Q4 proposal head) round their
activations to int8 per 32 values and run `dp4a` instead of emulated `mma.sync` kernels, about 9 ms
less per K = 7 round. The target still verifies every draft: acceptance on most prompts above is
unchanged at K = 7 (code 246 of 371 drafts with and without it), and greedy text is identical on
code, thinking, Chinese, French and structured output. Story and a long reply part at a near-tie, as
they also do on master between DFlash2 window settings, because a verify round's arithmetic depends
on its width. `NINFER_DRAFT_A8=0` turns it off. Details in
[volta-port.md](maintainer/volta-port.md#decode).

The target's own speed-up behind the MTP3 and plain columns is `--verify-a8`, on by default in an
`sm_70` build: the Q4/Q5/Q6 linears at 1-8 tokens
round their activations to int8 per 32 values and run `dp4a`, so a verify round costs about the same
as a plain step's GEMVs at any width (`--no-verify-a8` turns it off). It is lossy at the noise floor:
on a decode-path KLD gate against the FP32 kernels (17 chunks of 512 tokens from four held-out
domains), KLD 7.2e-4 to 7.8e-4 at one, four and eight tokens, and 98.9-99.2% the same top-1 token;
master's own verify reordering measures 3.8e-4 and 99.4%. Greedy text therefore differs from master
at near-ties (MTP3's code and structured outputs above are byte-identical to master's). Details in
[volta-port.md](maintainer/volta-port.md#decode).

The weights alone take 17.4 ms per token to read at the card's measured 879 GB/s, so plain decode
can reach at most about 57 tok/s; speculation gets past that ceiling. Prefill runs its GEMMs on the
tensor cores through cuBLAS, and prompt attention through FP16 flash attention. 1,150 / 2,186 /
4,392-token prompts prefill at 868 / 959 / 1,020 tok/s.

Decode against context length, MTP3, the same server flags, 256 greedy tokens (`ignore_eos`) on
exact-hit repeats of a checkpointed prompt (Project Gutenberg text from the PG-19 corpus plus a
question), one request at a time (2026-11-24, 200 W, ABBA against master, three repeats each, range
under 0.3%):

| Context | Decode | master | Per MTP round | master |
|---:|---:|---:|---:|---:|
| 4,215 | **75.7 tok/s** | 57.3-57.4 | 33.0 ms | 43.5 ms |
| 42,663 | **52.8 tok/s** | 46.5-46.6 | 51.4 ms | 61.1 ms |
| 172,487 | 20.9 tok/s | 21.6 | 127.1 ms | 136.3 ms |

Attention over the 16 full-attention layers' KV is what slows long-context decode, so the verify
GEMVs' saving is a smaller share there. At 172K a round is 7% cheaper, but this text's continuation
accepted fewer drafts than master's (158 of 289 against 168 of 261), so tok/s came out 3% lower;
acceptance depends on the text, not the context. (The previous figures, 57.9 / 47.9 / 25.0 tok/s at
4,170 / 40,259 / 162,152 tokens on 2026-11-23, used a different prompt.)

**Per-layer KV mixes** (`--kv-dtype rk4v4x12,int8x4` and the like, opt-in, lossy). Same server
flags, only `--kv-dtype` varied, 200 W, interleaved legs (2026-11-23; the paired legs agreed within
0.1%). These ran before `--verify-a8`, so their decode columns are master's. Perplexity is the held-out corpus at a 65,536-token context; ΔNLL is the paired difference
from `int8` in millinats (se over 6 domains):

| KV | Bytes/token | ΔNLL vs `int8` | Decode 4K / 40K / 162K | Prefill 40K / 162K |
|---|---:|---:|---|---:|
| `int8` | 33,792 | — | 57.9 / 47.9 / 25.7 tok/s | 47.0 s / — |
| `rk4v4x4,int8x12` | 29,824 | −0.13 (se 0.36) | 60.0 / 48.4 / 24.4 tok/s | 61.1 s / 535 s |
| `rk4v4x12,int8x4` | 21,888 | +1.69 (se 0.47) | 63.6 / 43.5 / 20.9 tok/s | 89.5 s / 975 s |
| `rk4v4` | 17,920 | +3.56 (se 0.58) | 58.4 / 42.2 / 20.3 tok/s | 103.6 s / 1,194 s |

The `int8` legs' 162K prefills overlapped a thermal pause and are not given. The last four
full-attention layers are the sensitive ones, so a little `int8` there buys back most of `rk4v4`'s
loss. On this card `int8` stays the right choice (it fits 262K at two lanes with 5.2 GiB free and
prefills fastest); the mixes are for when KV bytes are short. See
[the README](../README.md#per-layer-mixes---kv-dtype-rk4v4x12int8x4-opt-in).

### Qwen3.8-Flash-Next

**2026-11-24: the hyper-connection mixer's collapse kernel is cheaper on the V100**, with
the same output bit for bit. It runs 110 times in each MTP round, so every round is about 0.3 ms
shorter. Measured in A B B A order against master on the same box (200 W, no thermal throttling,
300 tokens at temperature 0, warm second request; tok/s, means of the two runs):

| Flags | Build | Story | Code | Chinese | French | Thinking | Thinking, budget 150 |
|---|---|---:|---:|---:|---:|---:|---:|
| The router's (graft, `--presence-penalty 1.0`) | before (master) | 80.7 | 123.2 | 78.9 | 71.7 | 100.0 | 105.6 |
| | **after** | **81.3** | **124.7** | **79.6** | 72.1 | 101.0 | 104.9 |
| Without the graft and presence penalty (Lapilli's settings) | before (master) | 84.8 | 118.7 | 82.4 | 76.6 | 109.9 | 115.7 |
| | **after** | **85.8** | **119.7** | **83.0** | 77.5 | 109.3 | 117.0 |

Both builds produced the same text in every run (md5 of each answer). Story, code and Chinese
moved up 0.7-1.2% in both configurations; French and the two thinking requests vary by up to 4 tok/s
between runs of the same binary, more than the change. Master here is `63762a40`, whose Flash-Next
code is the same as the change's base. How the kernel changed and what else was tried is in
[volta-port.md](maintainer/volta-port.md).

Decode with a warm expert cache (2026-11-23, **200 W** power limit, no thermal throttling sampled
during any run). `run-v100.sh` now drafts up to seven tokens a round with the reduced MTP draft head
(below); the three-draft, full-head figures of master `61b3e032c` are beside them:

| Prompt | MTP7 + draft vocabulary (default) | Tokens per round | MTP3, full head (before) | Tokens per round |
|---|---:|---:|---:|---:|
| Code (Python CSV parser) | **123.3 tok/s** (123.1-123.3) | 3.93 | 119.1 (117.9-119.1) | 3.31 |
| Thinking (a maths word problem) | **100.3 tok/s** (99.7-100.3) | 3.05 | 97.1 (92.0-97.1) | 2.62 |
| Story | **80.5 tok/s** (80.3-80.5) | 1.92 | 80.1 (77.7-80.1) | 1.94 |
| Chinese essay | **78.8 tok/s** (78.6-78.8) | 1.88 | 78.6 (76.7-78.6) | 1.90 |
| French essay | **71.8 tok/s** (71.4-71.8) | 1.78 | 88.7 (83.1-88.7) | 2.18 |

**French is slower with the draft vocabulary: 71.4-71.8 tok/s against 83.1-88.7, 14-19% down.** The
list (below) holds few of the tokens French text uses, so the drafter proposes the wrong ones more
often (1.78 tokens a round against 2.18); the text itself is unchanged. If you mostly generate French,
or any language whose decode drops, turn the head off with `--mtp-draft-vocab off` (or
`NINFER_DRAFT_TOKENS=3` as well for the previous configuration).

Every combination was run in one interleaved sequence on the same server binary, two runs each
(order A B C D E E D C B A; tok/s, means of the two):

| Drafts, head | Code | Thinking | Story | Chinese | French |
|---|---:|---:|---:|---:|---:|
| 3, full (before) | 119.1 | 94.4 | 80.1 | 78.6 | 85.1 |
| **7, draft vocabulary** | **123.2** | **100.0** | **80.5** | **78.7** | 71.6 |
| 5, draft vocabulary | 119.9 | 99.4 | 80.5 | 78.6 | 71.4 |
| 5, full | 121.9 | 98.3 | 79.9 | 77.9 | 84.6 |
| 7, full | 124.2 | 99.1 | 79.9 | 77.8 | 84.4 |

Seven drafts with the full head is 0.8% faster on code but 1.0% slower on Chinese and 0.2% on story;
the draft vocabulary makes each draft step cheaper, which brings Chinese and story back level while
code and thinking keep most of the gain.

**The MTP draft vocabulary.** Each draft step ends in the 248,077-row output head. With
`--mtp-draft-vocab` the drafter scores only the tokens of a list instead: 106,056 of them, whose head
rows (137 MiB, IQ4_XS) are copied once into VRAM at load. Drafts are still checked by the full model,
so the output is the same token for token. The list is the one jesdga's
[Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) pack ships
(`convert/mtp/rt/draft_vocab.bin`, Qwen Community License 1.0), whose use on a V100 we learned from
chris83-dev's Lapilli (MIT). It is **not part of this repository**: `scripts/download-model.sh
qwen38-flash-next` downloads it from that repository at a pinned revision, checks its sha256 and saves it as
`<model>.draft-vocab` beside the model, under jesdga's licence. `ninfer-serve` reads that file by
default on a V100 only (`--mtp-draft-vocab auto`); with no file, or one that does not load, it uses the full head, and
`--mtp-draft-vocab PATH` or `off` choose explicitly. The RTX 3090 keeps the full head: it was not
measured there.

Conditions: `ninfer-serve` with the exact flags the V100 box's router gives Flash-Next (the `run-v100.sh`
profile: INT8 KV, 262,144 context, `--draft-tokens 3`, `--mtp-joint-acceptance`, `--mtp-staged-drafts`,
`--cpu-verify-misses`, plus the refusal-vector graft as the default, `--presence-penalty 1.0` and
`NINFER_SM70_TENSOR_CORES=1`; the new default adds `--draft-tokens 7` and the draft vocabulary); 300 tokens at temperature 0, thinking off except on the thinking prompt;
each figure is the second of two identical requests, so the expert cache and context store are warm.
Every run produced byte-identical text. These replace the 2026-11-17 figures (78 / 96 / 72 / 81 on
story / code / Chinese / French at 250 W, without the graft and presence penalty), which predate the
later decode work on master.

Prefill, by prompt length (2026-11-18):

| Prompt | Time | tok/s |
|---:|---:|---:|
| 939 tokens | 1.86 s | 505 |
| 2,573 tokens | 3.23 s | 797 |
| 3,798 tokens | 3.84 s | 989 |
| 8,009 tokens | 8.24 s | 972 |
| 12,958 tokens | 12.7 s | 1,020 |
| 38,237 tokens | 38.4 s | 996 |
| 221,567 tokens | 245 s | 903 |

The 221,567-token prompt was measured before the 8,192-token calls landed; the others are with
current defaults.

Decode against context length, one request each, 256 tokens:

| Context | Decode | ms per round | Expert-cache hit rate |
|---:|---:|---:|---:|
| 10,670 | 68.8 tok/s | 29.4 | 98.5% |
| 38,240 | 72.2 tok/s | 27.3 | 97.9% |
| 79,228 | 76.8 tok/s | 27.0 | 99.1% |
| 148,129 | 77.9 tok/s | 27.3 | 99.3% |
| 221,567 | 78.8 tok/s | 28.2 | 98.4% |

### Qwen3.8-Flash-Next, UD-IQ3_XXS

`scripts/run-v100.sh qwen38-flash-next-iq3` serves Unsloth's 3-bit build (README, "Which Flash-Next
build?"). It was built and tuned on an RTX 3090; first V100 measurements (2026-11-23, master
`61b3e032c`, 200 W, no thermal throttling sampled):

- **It fits with every expert pinned and no flag changes** beside dropping `--vision`. With the IQ2_XS
  router flags (INT8 KV, 262,144 context, MTP3, `--ngram-ram-mib auto`, the context store) and
  `--ngram-volume` pointed at the shared IQ2_XS volume, the RAM ledger gives the experts all
  47.96 GiB they need ("every expert fits in RAM") by leaving the n-gram table on disk (0.10 GiB of RAM
  cache), and VRAM holds 9,830 experts (40% of 24,576; IQ2_XS holds 73.6%).
- **Decode is 19-48% slower than IQ2_XS.** Same prompts and conditions as above, best of two runs:

  | Prompt | UD-IQ3_XXS | IQ2_XS | Tokens per round, IQ3 / IQ2 |
  |---|---:|---:|---:|
  | Code | 61.9 tok/s | 119.1 | 3.22 / 3.31 |
  | Thinking | 58.3 tok/s | 97.1 | 2.69 / 2.62 |
  | French | 58.7 tok/s | 88.7 | 2.13 / 2.18 |
  | Story | 64.9 tok/s | 80.1 | 1.95 / 1.94 |
  | Chinese | 56.7 tok/s | 78.6 | 2.08 / 1.90 |

  Acceptance is about the same; the smaller expert cache is the cost.
- **Perplexity** on the six fixture texts (5,114 positions, scored by `ninfer_qwen4_exp_forward_real_test
  --dump-logits` on the V100): 5.872 against IQ2_XS's 6.280 (-6.5%), the same ranking as on the 3090
  (5.859 / 6.315; the small differences are the sm_70 kernels' summation order).

What follows was written before the build ran here and still holds for the parts not measured above:

- **Formats.** The build's dense weights and embedding are Q6_K and Q8_0. The sm_70 linear routes
  are generic over the GGML block formats and already cover both: the dp4a decode GEMV (`ggml_mmvq`,
  launch-bounded to three CTAs an SM), the wide `ggml_mmq` and the FP16 tensor-core route
  (`ggml_tensor_core_sm70.cu`, which names Q6_K and Q8_0). Cross-compiled for sm_70 with CUDA 12.8 the
  new code builds cleanly and stays inside the envelope of the existing kernels (`cuobjdump
  --dump-resource-usage`): the Q8_0 decode kernels use at most 128 registers and a 40-byte stack
  (IQ4_XS: 128 and 56), the Q8_0 pair kernels 80 and 40 (the same as IQ4_XS's), none spills to local
  memory, and the Q6_K embedding gathers, split ones included, use 32 registers and no stack.
- **`--hot-embedding-rows`** (`NINFER_HOT_EMBEDDING_ROWS=8192`, on by default in `run-v100.sh`) works
  with the Q6_K embedding: the table is 497 MiB, so keeping 8,192 rows on the card frees 480 MiB for
  the expert cache (311 MiB for the IQ4_XS table). `ninfer_embedding_test` checks the split gather
  bit for bit against the unsplit one.
- **Memory.** The bank is 42.3 GiB. By default the build runs with `--partial-ram-residency`, which
  pins only the experts the VRAM cache does not hold, takes no images and no elastic KV pool, so the
  context defaults to 32,768 tokens; the memory-lock limit it asks for is 42 GiB. With
  `NINFER_PARTIAL_RAM_RESIDENCY=off` every expert is pinned and the full 262,144-token context comes
  back; that is the mode measured above (the 59 GiB host fit it).
- **Not done: Volta tuning of Q6_K.** The IQ4_XS decode GEMV has an sm_70 form that reads a
  sub-block's header in one load (+2-8% per call, about +0.5% in the model). Q6_K, the build's main
  dense format, has none; it would need the same standalone bench on a V100 (`ggml_mmvq` at Flash-Next's
  five dense shapes, T = 1-4), and its sm_70 kernels already carry a larger stack than IQ4_XS's (112
  bytes against 56), a hint that it spends more instructions per sub-block.
- **To check it on a V100:** `scripts/v100_validate.sh --tests` runs the new op tests with the rest, then
  `MODEL_FLASH=/path/qwen4exp-ud-iq3xxs-mtp-vision.ninfer scripts/v100_validate.sh --models` generates
  text from it, and `scripts/run-v100.sh qwen38-flash-next-iq3` serves it.

### What limits Flash-Next here

About a quarter of the 24,576 routed experts do not fit in VRAM. Those live in pinned host RAM and
come over the PCIe link when needed.

- **Decode** barely touches the link. The expert cache hits 97-99% after a prompt and 99.6-100% in
  steady decode, and the CPU computes the misses.
- **Prefill** routes almost every expert of every layer, so each 8,192-token call streams every
  expert it does not have cached. With 8,192-token calls the expert compute dominates: in a
  16K-token prompt the GPU ran kernels for 12.7 of 15.9 s, and the link was busy 5.9 s alongside.
  Short prompts (under ~1K tokens) are dominated by fixed costs:
  the n-gram reads, the first layers, and the link.
- **After a long prompt**, the VRAM the prompt borrowed comes back empty and decode fills it with
  the experts it actually uses. Refilling it in one burst stalled the first seconds of decode on
  this link.

## Limits

- **Linux only, prebuilt or from source.** The release archive is `ninfer-v100-linux-x64-<version>.tar.gz`; `scripts/v100_build.sh` builds the same thing from a checkout.
- **Linux only on the V100.** It was tested on Ubuntu 24.04 with CUDA 12.9 (a 16-thread host).
- **Not bit-identical with tensor cores on.** With `NINFER_SM70_TENSOR_CORES=1` (the Flash-Next
  default), prefill output is close to the plain server's but not bit for bit. Set it to 0 to get
  identical output at slower prefill.
- **Ampere instructions are emulated.** `cp.async`, `ldmatrix` and BF16/INT8 `mma` are emulated on
  Volta. Paths that have not been moved to cuBLAS, to the V100's FP16 tensor cores, or to a
  CUDA-core kernel run correctly but slowly; these are mainly prompt tails under 129 tokens for the
  27B.
- **One request at a time** has been measured for Flash-Next. Two lanes (`NINFER_CONCURRENCY=2`)
  are allowed but have not been tuned on the V100. Inkling-2-Mini batches up to 8 requests
  (`--max-concurrency 8`), without DSpark or adapters at N > 1.
- **Known `sm_70` test failures:**
  - `capped_prefill_steps` and `explicit_positions_equal_rope_delta_path`: workspace `bad_alloc`
    with the cuBLAS prefill route.
  - `gdn_input_proj_conv_record` (Q8, B=1, T=16). Flash-Next does not use it.
