# Raw results (2026-10-13, RTX 3090, Qwen3.8-27B)

`RESULTS.txt` collects every run: the lock log and, per arm, the summary line from `summarize.py`, the per-request prefill/decode medians from `rates.py` and, for restart probes, `sum_restart.py`. Each run directory holds the per-arm `summary-*.json` files and the lock log.

The arm letters map to server builds (folders of `ninfer-serve.exe`):

| Run | A | B | C |
|---|---|---|---|
| `c4-2g` | `master` 04901788 | #287 first commit (no import retry) | + write-on-evict (no retry) |
| `c4-2g-v2` | #287 (with retry) | + write-on-evict | + shared prefixes |
| `pr1-restart`, `pr1-restart2` | `master` | #287 first commit | |
| `pr3-warm`, `pr3-warm2` | write-on-evict only | + shared prefixes (`pr3-warm`: no retry; `pr3-warm2`: with retry) | |
| `nostore-c4` | `master` | whole series (no retry), store off | |
| `diag1` | | write-on-evict with diagnostic logging | |
| `c4-8g`, `c8-2g-mtp`, `c8-8g-mtp` | `master` | this branch before the deepest-point and shared-key fixes | |
| `f2-*` | `master` | this branch with both fixes | |
| `tests-*` | ctest logs | | |

`c8-2g` failed to start: the DFlash2 profile does not fit at 8 lanes; the C8 runs use `-Spec mtp`.
