# NInfer on a Tesla V100

NInfer runs on a 32 GB Tesla V100 (Volta, `sm_70`), one of the project's two first-class cards
alongside the RTX 3090. Three models are supported and tuned there: **Qwen3.8-27B**,
**Qwen3.8-Flash-Next**, the 512-expert MoE with the full 262,144-token context, and
**Inkling-2-Mini**, the looped 128-expert MoE with images and DSpark speculative decoding
([spec and measurements](maintainer/inkling2-model.md)). This page covers what you need, how to build and run
the two Qwen models, their measured performance, and the limits; Inkling-2-Mini's run command and measurements
are in its [model card](../model-cards/Inkling-2-Mini-NInfer-V100/README.md) and its spec.
How the port works, and every measurement behind it, is in [docs/maintainer/volta-port.md](maintainer/volta-port.md).

There are no prebuilt V100 archives: the release binaries are `sm_86`. You build from source on
Linux.

## At a glance

All figures were measured on one V100-PCIE-32GB with a 250 W power limit, connected over an
OCuLink PCIe Gen3 x4 eGPU link (3.3 GB/s host-to-device), on a Linux host with 59 GiB of RAM.
Decode is greedy with MTP3 speculation, as `run-v100.sh` serves it (2026-11-16 to 2026-11-18).

| | Qwen3.8-27B | Qwen3.8-Flash-Next |
|---|---:|---:|
| **Context** | 262,144 tokens | **262,144 tokens** (`run-v100.sh` default) |
| **Decode**, one stream, MTP3 | **76-86 tok/s** | **72-96 tok/s** with a warm cache, depending on the text |
| **Decode**, plain | 42 tok/s | 53 tok/s |
| **Prefill**, ~4K-token prompt | **~1,020 tok/s** | **~990 tok/s** (3,798 tokens in 3.8 s) |
| **Prefill**, long prompts | | 38K tokens in 38.4 s (~1,000 tok/s) |

Flash-Next's decode does not slow down with context. From 10K to 221K tokens of context it stays at
27-29 ms per MTP verification round (69-79 tok/s on those prompts). What moves tok/s is how many
drafted tokens are accepted, which depends on the text, not on the context length.

**Inkling-2-Mini** on the same card (master `dad1623b`, 2026-11-22, 250 W limit, through `ninfer-serve`; the
[model card](../model-cards/Inkling-2-Mini-NInfer-V100/README.md) has the conditions and ranges):

| | Inkling-2-Mini |
|---|---:|
| **Context** | 262,144 tokens with DSpark and vision |
| **Decode**, one stream, DSpark | **up to 120.7 tok/s** (maths prompt); 93.4-94.4 average over 12 chats |
| **Decode**, plain | 78.5-78.8 tok/s; 48 tok/s at 100K context |
| **Decode**, `--max-concurrency 8` | **198.2 tok/s** aggregate (plain) |
| **Prefill** | **1,930 tok/s** at 8K (131,072 context, chunk 2048); 1,629 at 1K, 1,244 at 100K. At 262,144 with DSpark and vision (chunk 2048 by default): 1,868 at 8K, 1,671 at 1K, 1,219 at 100K |
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
BUILD_DIR=$HOME/build-v100 scripts/v100_build.sh ninfer ninfer-serve
```

`scripts/v100_build.sh` configures with `-DCMAKE_CUDA_ARCHITECTURES=70`, uses ccache, and limits
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
| `NINFER_SPEC=mtp`, `NINFER_KV_DTYPE=int8` | 27B | MTP3 with INT8 KV, which is faster than DFlash2 on this card (76 against 66 tok/s on code) |

One default comes from the build rather than the launcher: in an `sm_70` build `--prefill-cublas` is
on, so the 27B's wide prefill GEMMs run on the tensor cores through cuBLAS instead of emulated integer
kernels. `--no-prefill-cublas` (`NINFER_NO_PREFILL_CUBLAS=1`) turns it off.

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

Decode, one stream, greedy, INT8 KV:

| | tok/s |
|---|---:|
| Plain | 42.2 |
| MTP3, server, story prompt | 76.1 |
| MTP3, `ninfer_bench -pg 2048,256` | 86.3 |
| DFlash2 (`--draft-tokens 7`) | 67.6 |

The weights alone take 17.4 ms per token to read at the card's measured 879 GB/s, so plain decode
can reach at most about 57 tok/s; speculation gets past that ceiling. Prefill runs its GEMMs on the
tensor cores through cuBLAS, and prompt attention through FP16 flash attention. 1,150 / 2,186 /
4,392-token prompts prefill at 868 / 959 / 1,020 tok/s.

Decode against context length, MTP3, INT8 KV, `ninfer-serve` with the `run-v100.sh` flags, 256 greedy
tokens on an exact-hit repeat of a checkpointed prompt, one request at a time (2026-11-23, **200 W**
power limit, three repeats each, range under 0.5%):

| Context | Decode |
|---:|---:|
| 4,170 | 57.9 tok/s |
| 40,259 | 47.9 tok/s |
| 162,152 | 25.0 tok/s |

Attention over the 16 full-attention layers' KV is what slows long-context decode. Staging the next K/V
tile in registers (sm_70 only) raised the 40K figure from 46.8 and the 162K figure from 23.5 tok/s, with
byte-identical output.

**Per-layer KV mixes** (`--kv-dtype rk4v4x12,int8x4` and the like, opt-in, lossy). Same server
flags, only `--kv-dtype` varied, 200 W, interleaved legs (2026-11-23; the paired legs agreed within
0.1%). Perplexity is the held-out corpus at a 65,536-token context; ΔNLL is the paired difference
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

Decode with a warm expert cache, MTP3: 78 / 96 / 72 / 81 tok/s on story / code / Chinese / French
prompts.

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

### Qwen3.8-Flash-Next, UD-IQ3_XXS (not yet run on a V100)

`scripts/run-v100.sh qwen38-flash-next-iq3` serves Unsloth's 3-bit build (README, "Which Flash-Next
build?"). **It has not been run on a V100**: it was built and tested on an RTX 3090, and what is known
for this card is what compiles and what the op tests check.

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
  `NINFER_PARTIAL_RAM_RESIDENCY=off` every expert is pinned (about 50 GiB of available RAM: the 59 GiB
  test host should fit it, and nothing else should be running), and the full 262,144-token context and
  images come back. Neither mode has been measured on this card.
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

- **No prebuilt binaries.** Build with `scripts/v100_build.sh`.
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
