# NInfer-3090 for Windows — release archive

This is the README that ships **inside** the Windows release archive, where every file sits in one
directory. If you are reading it in a checkout instead, the launchers and downloaders it names live
under `scripts/`, and
[docs/rtx-3090-windows.md](https://github.com/ashalliants/ninfer-3090/blob/master/docs/rtx-3090-windows.md)
is the fuller guide. Links here are absolute on purpose: the archive ships no `docs/` directory.

Check `VERSION` for the release this archive was cut from, and `RELEASE_NOTES_*.md` for what
changed.

## Requirements

- Windows 11 x64
- GeForce RTX 3090 (or 3090 Ti) with a recent NVIDIA driver
- Microsoft Visual C++ 2022 runtime

Model artifacts are **not** included — they are 17–21 GB each. The downloaders below fetch them.

## Quick start

From the directory you unpacked, two commands:

```powershell
.\download-qwen36-35b-a3b.bat          # ~21 GB, resumable, verifies size and SHA256
.\run-qwen36-35b-a3b-c1-maxctx.bat     # serves on http://127.0.0.1:8080/v1
```

The downloader writes into `models\` beside these files, which is where every launcher looks.
`NINFER_MODEL_DIR` moves where the **downloaders** put artifacts; to point a **launcher** at a file
somewhere else, give it `NINFER_MODEL` (the launchers do not read `NINFER_MODEL_DIR`). The plain
`run-qwen38-c1.bat` and `run-qwen38-c8.bat` also take the artifact path as their first argument.

That launcher is the recommended Windows profile: Qwen3.6-35B-A3B with `rk8v4` KV, MTP3 speculation
plus the draft head, and vision in overlay residency. For the dense 27B instead:

```powershell
.\download-qwen38-27b.bat              # ~17 GB
.\run-qwen38-c1-maxctx.bat             # one user, 131,072 tokens
```

`run-qwen38-c1.bat` and `run-qwen38-c8.bat` are the older INT8 profiles — one user at 65,536 tokens,
and eight concurrent users at 8,192 each.

The endpoint is OpenAI-compatible, so anything that speaks `/v1/chat/completions` works. Leave the
API key blank.

## What is in the archive

| file | what it is |
|---|---|
| `ninfer-serve.exe` | the server: OpenAI- and Anthropic-compatible HTTP APIs |
| `ninfer.exe` | one-shot CLI generation, for smoke tests and scripting |
| `ninfer_bench.exe` | throughput benchmark against the public Engine route |
| `download-qwen*.bat` | pinned, resumable artifact downloads with verification |
| `run-qwen*.bat` | serving profiles; see the table above |
| `*.dll` | the FFmpeg and libcurl runtime dependencies |
| `SHA256SUMS.txt` | checksums for every file in this directory |

## Overrides

Nothing here needs editing. **Every** launcher reads `NINFER_MODEL`, `NINFER_SERVER`, `NINFER_HOST`
and `NINFER_PORT`. Beyond those four the set differs per launcher, so read the header of the one you
use:

| launcher | also reads |
|---|---|
| `run-qwen38-c1-maxctx.bat` | `NINFER_CONTEXT`, `NINFER_CONCURRENCY`, `NINFER_KV_DTYPE` |
| `run-qwen36-35b-a3b-c1-maxctx.bat` | `NINFER_CONTEXT`, `NINFER_KV_DTYPE` |
| `run-qwen38-c1.bat`, `run-qwen38-c8.bat` | nothing further; artifact path as `%1` |
| `download-qwen*.bat` | `NINFER_MODEL_DIR` |

> `NINFER_HOST=0.0.0.0` exposes the server to your network **unauthenticated**. The launchers bind
> `127.0.0.1` for that reason. If you set it, put something in front of it.

## If startup refuses

The message names the numbers. A 24 GB card running a desktop has roughly 1.5 GiB less to work
with than a headless one, so the largest profiles do not fit alongside a desktop:

```
requested Engine runtime reservation requires 2864526592 bytes,
but only 2375691264 bytes are available for runtime capacity
```

Drop a context rung first — `NINFER_CONTEXT=98304`, then 81920, then 65536. Speculation is the next
lever (`NINFER_SPEC=none`), worth about 992 MiB at the cost of decode speed. Drop vision last: in
overlay residency it costs almost nothing resident.

To size a profile before running it, open
[docs/config-calculator.html](https://github.com/ashalliants/ninfer-3090/blob/master/docs/config-calculator.html)
from the repository — one self-contained file, no network needed, every constant in it measured on
an RTX 3090. It is not in this archive.

## Full documentation

<https://github.com/ashalliants/ninfer-3090>
