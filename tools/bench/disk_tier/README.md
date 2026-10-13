# Context-store (disk tier) measurements

Scripts behind the context-store numbers in `docs/performance.md` ("Context store ..." sections).
Windows / PowerShell, one RTX 3090, Qwen3.8-27B. Every server script takes the GPU lock
(`%TEMP%\ninfer-gpu.lock`, see `CLAUDE.md`) for its whole run and releases it at the end. Results
go to `%NINFER_DISK_TIER_RUNS%` (default `%TEMP%\ninfer-disk-tier-runs\<tag>`); stores go under
`%TEMP%\ninfer-disk-runs` and are deleted after each arm. `NINFER_MODEL` overrides the artifact.

| Script | What |
|---|---|
| `hosted_rotation.py` | Client: T tenants on S shared system prompts, each visit appends a 800-2400-word tool result, conversations restart at `--max-prompt-tokens`; `--workers` clients take tenants in a fixed rotation. Writes per-request samples and a summary (hit rate, re-prefilled tokens, TTFT p50/p95/p99, store counter deltas from `/metrics`). |
| `abba.ps1` | Starts `ninfer-serve` from `-BinA/-BinB/-BinC` (folders holding `ninfer-serve.exe` and its DLLs) in `-Order` (e.g. `ABBA`, `ABCCBA`), runs `hosted_rotation.py` against each. `-Spec dflash2` (launcher `tuned` default: 188,416 context, draft 7, chunk 4096) or `mtp` (262,144, draft 3, chunk 2048); always `--kv-dtype rk4v4 --lm-head-draft --gdn-state-fp16 --prefill-cublas --fit-context 32768 --min-p 0.03 --presence-penalty 0.5`. `-HostMib`, `-Concurrency`, `-Requests`, `-NoStore`. |
| `restart_abba.ps1` + `restart_probe.py` | Seeds six one-turn conversations (1.5K-16K tokens) into a store, then for each arm restarts on an unbuffered copy (`xcopy /J`) of it with a small Host tier and continues each conversation. |
| `restart_shared.ps1` | Warm start: each arm seeds its own store with the rotation, is killed (no flush), restarts on that store and serves new tenants on the same system prompts. |
| `summarize.py <run dir> [prefix]`, `rates.py <run dir>`, `sum_restart.py <run dir>` | Tables from the results and the server logs (per-request prefill/decode medians). |
| `nvme_seq.cpp` | Unbuffered sequential write/read of one file (`cl /O2 /EHsc nvme_seq.cpp`; `nvme_seq <path> <GiB>`). |

Example (C4, 2 GiB Host tier, three builds):

```powershell
.\tools\bench\disk_tier\abba.ps1 -Tag c4-2g -BinA C:\bins\base -BinB C:\bins\new -Order ABBA -Concurrency 4 -HostMib 2048 -Requests 120
python tools\bench\disk_tier\summarize.py $env:TEMP\ninfer-disk-tier-runs\c4-2g result-
```
