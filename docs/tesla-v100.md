# NInfer on a Tesla V100

NInfer runs on a 32 GB Tesla V100 (Volta, `sm_70`). Two models are supported and tuned there:
**Qwen3.8-27B**, and **Qwen3.8-Flash-Next**, the 512-expert MoE with the full 262,144-token
context. This page covers what you need, how to build and run it, the measured performance, and
the limits. How the port works, and every measurement behind it, is in
[docs/maintainer/volta-port.md](maintainer/volta-port.md).

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
| `NINFER_HOT_EMBEDDING_ROWS=8192` | Flash-Next | Only the common embedding rows stay in VRAM, freeing room for 216 more cached experts |
| `NINFER_SPEC=mtp`, `NINFER_KV_DTYPE=int8` | 27B | MTP3 with INT8 KV, which is faster than DFlash2 on this card (76 against 66 tok/s on code) |

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
  are allowed but have not been tuned on the V100.
- **Known `sm_70` test failures:**
  - `capped_prefill_steps` and `explicit_positions_equal_rope_delta_path`: workspace `bad_alloc`
    with the cuBLAS prefill route.
  - `gdn_input_proj_conv_record` (Q8, B=1, T=16). Flash-Next does not use it.
