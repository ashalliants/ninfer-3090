# NInfer for the Tesla V100 — release archive

This is the README that ships **inside** the Tesla V100 (`sm_70`) Linux release archive, where every file
sits in one directory. It is the same server as the RTX 3090 archive, compiled for Volta and started with
the V100's own launcher. If you are reading it in a checkout instead, the launchers and downloaders it
names live under `scripts/`, and
[docs/tesla-v100.md](https://github.com/ashalliants/ninfer-3090/blob/master/docs/tesla-v100.md) is the
full guide (measurements, limits, how the port works). Links here are absolute on purpose: the archive
ships no `docs/` directory.

Check `VERSION` for the release this archive was cut from, and `RELEASE_NOTES_*.md` in the archive root for
what changed.

**This archive is for a Tesla V100 only.** The binaries hold `sm_70` machine code and nothing else, so they
will not start on an RTX 3090 or any other card. An RTX 3090 uses the `ninfer-rtx3090-linux-x64-*` archive
from the same release.

## Requirements

- x86-64 Linux with glibc 2.38 or newer (the binaries are built on Ubuntu 24.04)
- A 32 GB Tesla V100 (the project's tuned and measured V100 is the PCIe 32 GB card) with a recent NVIDIA
  driver. A 16 GB V100 is not a target: the Qwen3.8 models and Inkling-2-Mini need the 32 GB
- The CUDA 12.8 runtime libraries, **including cuBLAS** (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`). From NVIDIA's CUDA repository: `sudo apt install cuda-libraries-12-8`. CUDA 12.8 is the
  newest toolkit that still targets Volta, which is why this archive is built with it
- FFmpeg 6 shared libraries, libcurl and libzstd (`sudo apt install ffmpeg libcurl4t64 libzstd1` on Ubuntu 24.04)
- `curl` for the downloaders
- For Qwen3.8-Flash-Next, about 59 GiB of RAM (it keeps its experts in pinned host memory) and a fast SSD

Model artifacts are **not** included. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands:

```bash
./download-model.sh qwen38-27b         # ~19 GB, resumable, verifies size and SHA256
./run-v100.sh qwen38-27b               # serves on http://0.0.0.0:8080/v1
```

`run-v100.sh` is `run.sh` with the V100's measured defaults (MTP3 with INT8 KV for the 27B). It checks that the
card is a V100 with nothing else running on it, and warns about a power limit below the card's 250 W default.

The server listens on **every network interface**, so other machines reach it at
`http://<this machine>:8080/v1`. Set `NINFER_API_KEY=<a secret>` before you start it: without a key anyone who can
reach the port can use the model, and the launcher says so at startup. `NINFER_HOST=127.0.0.1` keeps it local.

The endpoint is OpenAI- and Anthropic-compatible, so anything that speaks `/v1/chat/completions` or
`/v1/messages` works.

### The other V100 models

```bash
./download-model.sh qwen38-flash-next   # the 512-expert MoE, full 262,144-token context
./run-v100.sh qwen38-flash-next         # also: qwen38-flash-next-iq3 (Unsloth's 3-bit build)

./download-model.sh qwen36-35b-a3b      # 21.2 GiB, the 256-expert MoE with vision
./run-v100.sh qwen36-35b-a3b            # MTP3, rk4v4, the Vision tower resident in VRAM

./download-model.sh inkling2-mini       # 22.3 GiB, exact conversion of Unsloth's UD-Q8_K_XL
./ninfer-serve models/inkling2-mini-q8.ninfer --vision --spec dspark --host 0.0.0.0 --api-key <a secret>
```

Inkling-2-Mini has no launcher profile yet: you start `ninfer-serve` on its artifact directly, as above.
`--max-concurrency 8` serves up to eight requests at once (DSpark then applies to a single request).
[Its model card](https://github.com/ashalliants/ninfer-3090/blob/master/model-cards/Inkling-2-Mini-NInfer-V100/README.md)
has the measured figures and options.

The downloader writes into `models/` beside these files, which is where the launcher looks. Set
`NINFER_MODEL_DIR` to keep artifacts elsewhere, or `NINFER_MODEL` to point the launcher at one file.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server (`sm_70`): OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `run-v100.sh` | the V100 launcher: `run-v100.sh <model> [profile]` |
| `run.sh` | the general launcher `run-v100.sh` starts; every `NINFER_*` variable applies to both |
| `download-model.sh` | pinned, resumable artifact downloads with verification |
| `profiles/` | each profile's fixed flags, as `ninfer-serve --config` files |
| `LICENSE` | NInfer's licence (Apache-2.0) |
| `THIRD_PARTY_NOTICES.txt` | the notice for the ported conversation projector (hermes-vcc-engine, Apache-2.0) |
| `SHA256SUMS.txt` | checksums for every file in this directory and in `profiles/` |

## Things worth knowing on a V100

- **Cooling.** A passive V100 needs real airflow. Sustained prefill holds it at its power cap; watch
  `nvidia-smi -l 1` the first time on a new setup.
- **Memory lock.** Flash-Next pins 33 GiB of experts. When `ulimit -l` is below that, the launcher continues with
  unlocked host memory; to lift the limit, as root:
  `printf '%s - memlock unlimited\n' "$USER" > /etc/security/limits.d/90-ninfer-memlock.conf`, then log in again.
- **Overrides.** `NINFER_SERVER`, `NINFER_MODEL_DIR`, `NINFER_MODEL`, `NINFER_HOST`, `NINFER_PORT`, `NINFER_API_KEY`,
  `NINFER_CONTEXT`, `NINFER_SPEC` and the rest of `run.sh`'s variables win over what `run-v100.sh` picks. The header
  of `run-v100.sh` lists every default it sets and why, with the measurements behind each.
- **Limits.** One request at a time has been measured for Flash-Next; Inkling-2-Mini batches up to eight. With
  `NINFER_SM70_TENSOR_CORES=1` (the Flash-Next default) prefill output is close to, not bit for bit, the plain
  server's; set it to `0` for identical output at slower prefill.

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
