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

- x86-64 Linux with a recent NVIDIA driver (CUDA 12.8 runtime or newer)
- GeForce RTX 3090 (or 3090 Ti)
- `curl` for the downloaders

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands:

```bash
./download-qwen36-35b-a3b.sh           # ~21 GB, resumable, verifies size and SHA256
./run-qwen36-35b-a3b-c1-maxctx.sh      # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models/` beside these files, which is where every launcher looks. Set
`NINFER_MODEL_DIR` to keep artifacts elsewhere, or `NINFER_MODEL` to point one launcher at a single
file.

That launcher is the headless profile: two lanes, `rk8v4` KV, MTP3 speculation plus the draft head,
and vision in overlay residency, with `NINFER_CONTEXT` defaulting to 262,144.

Read its header before assuming that context holds. Speculation is not free context — the MTP head
is 856 MiB and the draft head another 136 MiB, roughly 130,000 rk8v4 tokens of KV — so the launcher
documents `NINFER_SPEC=none` as what actually reaches the native 262,144 maximum, at about
183 tok/s instead of 240–280. Drop a rung (196608 / 131072 / 114688 / 98304 / 81920) if startup
refuses. For the dense 27B instead:

```bash
./download-qwen38-27b.sh               # ~17 GB
./run-qwen38-c1-maxctx.sh              # two lanes, 212,992 tokens each
```

`run-qwen38-c1.sh` and `run-qwen38-c8.sh` are the older INT8 profiles — one user at 65,536 tokens,
and eight concurrent users at 8,192 each.

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench` | throughput benchmark against the public Engine route |
| `download-qwen*.sh` | pinned, resumable artifact downloads with verification |
| `run-qwen*.sh` | serving profiles; see the table above |
| `SHA256SUMS.txt` | checksums for every file in this directory |

## Overrides

Nothing here needs editing, but the two families differ and it is worth knowing which you are
running. `NINFER_SERVER` and `NINFER_MODEL_DIR` work everywhere.

| launcher | overrides it reads |
|---|---|
| `run-qwen38-c1-maxctx.sh` | `NINFER_MODEL`, `NINFER_HOST`, `NINFER_PORT`, `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_CAPACITY`, `NINFER_KV_DTYPE`, `NINFER_SPEC`, `NINFER_VISION`, `NINFER_DRAFT_TOKENS` |
| `run-qwen36-35b-a3b-c1-maxctx.sh` | the same, except `NINFER_KV_DTYPE` — that one is fixed at `rk8v4` |
| `run-qwen38-c1.sh`, `run-qwen38-c8.sh`, the two vision launchers | `NINFER_SERVER` and `NINFER_MODEL_DIR` only, plus the artifact path as `$1`. Host, port and every serving flag are baked in at `127.0.0.1:8080` — unlike their Windows `.bat` counterparts, which do read `NINFER_HOST`/`NINFER_PORT`/`NINFER_MODEL`. Use a `-maxctx` launcher if you need to move the port. |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If startup refuses

The message names the numbers. Drop a context rung first —
`NINFER_CONTEXT=196608`, then 131072, 114688, 98304, 81920. Speculation is the next lever
(`NINFER_SPEC=none`), worth about 992 MiB at the cost of decode speed. Drop vision last: in overlay
residency it costs almost nothing resident, and an `evictable pool window exceeds the evictable
tail` message means the reservation is already tight rather than that the context is too large.

`--host-kv-mib 8192` behaves differently here than on Windows: on Linux it really does pin 8 GiB of
host RAM. On Windows/WDDM a pinned host allocation is charged against the card, so the runtime
clamps it hard. Same flag, different platform behaviour, by design.

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
