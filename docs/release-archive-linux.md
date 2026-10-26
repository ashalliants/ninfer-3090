# NInfer-3090 for Linux — release archive

This is the README that ships **inside** the Linux release archive, where every file sits in one
directory. If you are reading it in a checkout instead, the launchers and downloaders it names live
under `scripts/`, and
[docs/rtx-3090-linux.md](https://github.com/ashalliants/ninfer-3090/blob/master/docs/rtx-3090-linux.md)
is the guide to building from source. Links here are absolute on purpose: the archive ships no
`docs/` directory.

Check `VERSION` for the release this archive was cut from, and `RELEASE_NOTES_*.md` for what
changed.

## Requirements

- x86-64 Linux with glibc 2.38 or newer (the binaries are built on Ubuntu 24.04)
- GeForce RTX 3090 (or 3090 Ti) with a recent NVIDIA driver
- The CUDA 12.8 runtime libraries, **including cuBLAS** (`libcudart.so.12`, `libcublas.so.12`,
  `libcublasLt.so.12`). The cuBLAS prefill route links it, so the binaries need it even when you do
  not turn that route on. From NVIDIA's CUDA repository: `sudo apt install cuda-libraries-12-8`
- FFmpeg 6 shared libraries, libcurl and libzstd (`sudo apt install ffmpeg libcurl4t64 libzstd1` on Ubuntu 24.04)
- `curl` for the downloaders

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands:

```bash
./download-model.sh qwen38-27b         # ~19 GB, resumable, verifies size and SHA256; the DFlash2 bundle, it also carries the MTP weights
./run.sh qwen38-27b                    # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models/` beside these files, which is where the launcher looks. Set
`NINFER_MODEL_DIR` to keep artifacts elsewhere, or `NINFER_MODEL` to point the launcher at a single
file.

That is the recommended profile: Qwen3.8-27B, one lane with up to 262,144 tokens, DFlash2 speculation
plus the draft head (fastest), `rk4v4` KV, cuBLAS prefill, and vision in overlay residency. On a card
that also drives a desktop the server starts with as much of that context as fits and says so
(see "If the GPU is busy").

For the longest context shared by two lanes instead, `NINFER_SPEC=mtp ./run.sh qwen38-27b` runs the
MTP profile at 262,144 tokens with a smaller prefill chunk: slower decode, more
context.

`./run.sh qwen38-27b int8` and `./run.sh qwen38-27b c8` are the older INT8 profiles — one user at
65,536 tokens, and eight concurrent users at 8,192 each.

For the Qwen3.6-35B-A3B MoE instead, which serves more lanes:

```bash
./download-model.sh qwen36-35b-a3b     # ~21 GB, resumable, verifies size and SHA256
./run.sh qwen36-35b-a3b                # serves on http://127.0.0.1:8080/v1
```

That profile runs three lanes, `rk4v4` KV, MTP3 speculation plus the draft head, and vision in
overlay residency, with the lanes sharing one 262,144-token pool (any one request can use all of
it). Three lanes start even beside a desktop, so a headless card has room to spare
(`NINFER_CONCURRENCY=4` should fit there).

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `download-model.sh` | pinned, resumable artifact downloads with verification |
| `run.sh` | serving profiles: `run.sh <model> [profile]` |
| `profiles/` | each profile's fixed flags, as `ninfer-serve --config` files that `run.sh` passes |
| `SHA256SUMS.txt` | checksums for every file in this directory and in `profiles/` |

## Overrides

Nothing here needs editing. `NINFER_SERVER`, `NINFER_MODEL_DIR`, `NINFER_MODEL`, `NINFER_HOST`,
`NINFER_PORT`, `NINFER_CHAT_TEMPLATE` (a local Jinja file, passed to `--chat-template`,
overriding the artifact's built-in template) and `NINFER_ALLOWED_HOST` (more names the server
answers to, such as `gpubox.lan`, comma- or space-separated, each passed as `--allowed-host`) work
for every profile.

| profile | also reads |
|---|---|
| `run.sh <model>` (`tuned`) | `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY`, `NINFER_KV_DTYPE`, `NINFER_SPEC` (27B: `dflash2`, `mtp`, `none`; 35B: `mtp`, `none`), `NINFER_VISION`, `NINFER_VISION_RESIDENCY`, `NINFER_DRAFT_TOKENS`, `NINFER_PREFILL_CHUNK`, `NINFER_HOST_CONTEXT_MIB` (pinned host RAM for the context cache, default 8192), `NINFER_FALLBACK` (`off` makes the profile's context exact instead of fitting it to free GPU memory) |
| `run.sh qwen38-27b int8`, `run.sh qwen38-27b c8` | nothing further; every serving flag is fixed |
| `download-model.sh <model>` | `NINFER_MODEL_DIR`, `NINFER_SKIP_SHA256`; `NINFER_NGRAM_DIR` (qwen38-flash-next: the folder for the n-gram volume) |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If the GPU is busy

**The default profile handles this for you.** Its context is a ceiling: when a desktop or another
job holds enough VRAM that the full context does not fit, the server starts at the largest context
that does (a multiple of 1,024 tokens, keeping 256 MiB spare) and prints one `context lowered to ...
(asked for ...)` line. If not even 32,768 tokens fit, it stops and says how much memory is missing.
A `NINFER_CONTEXT` or `NINFER_KV_CAPACITY` you set is honoured exactly and fails loudly if it does
not fit; `NINFER_FALLBACK=off` does the same for the profile's own context. If pinning the Host
context budget fails (a machine short of RAM), the launcher retries with half of it, down to none,
unless you set `NINFER_HOST_CONTEXT_MIB` yourself.

With an exact context the refusal names the numbers; pick a smaller `NINFER_CONTEXT`. Speculation is the next lever
(`NINFER_SPEC=none`), worth about 992 MiB at the cost of decode speed. Drop vision last: in overlay
residency it costs almost nothing resident, and an `evictable pool window exceeds the evictable
tail` message means the reservation is already tight rather than that the context is too large.

The pinned Host context budget (`--host-context-mib`; the default profiles pass 8192, and without
the flag the server uses 8 GiB plus eight model state images) is host RAM, pinned in full at startup,
and holds retained conversation state, KV pages and pause snapshots. Lower it with
`NINFER_HOST_CONTEXT_MIB` on a machine short of RAM.

On a machine that only serves this process, pass `--auto-host-cache` to ninfer-serve instead of
`--host-context-mib`: it sizes the budget from the RAM that is free after the model loads (or from
the container's memory limit), leaving `--host-cache-reserve-mib` (default 3072) unpinned. The launchers do not use it by default. See
the [serving guide](https://github.com/ashalliants/ninfer-3090/blob/master/docs/serving.md#automatic-host-cache).

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
