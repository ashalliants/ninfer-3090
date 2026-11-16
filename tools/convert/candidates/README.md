# Candidate recipe overrides

Allocations of Q4 or Q5 for the 128 attention-output, GDN-output and MLP-down projections of
Qwen3.8-27B, measured on the RTX 3090 on 2026-11-16 (pull request #445). The results are in
[`docs/maintainer/quality-trade-experiments.md`](../../../docs/maintainer/quality-trade-experiments.md)
(the `alloc-activation` and `alloc-absolute` rows and the bullets under that table).

- **`absolute` (adopted).** Chosen from the imatrix-weighted rounding error alone
  (`tools.convert.allocation --metric absolute`). It is now the shipped `qwen3_8_27b` rule
  (`QWEN3_8_27B_Q5_FROM_LAYER` in `tools/convert/official_recipes.py`: Q5 from layer 23 for attention
  output, 24 for GDN output and 30 for MLP down, Q4 before). Mean KLD 0.0288 against 0.0318 for the rule
  it replaced, with the same text bytes; plain decode, prefill and MTP3 speed unchanged.
- **`activation` (measured, not adopted).** Chosen from noise measured on real activations
  (`--metric activation`). Same mean KLD as `absolute`, but 7.6% worse on `synthetic_chat` on one file,
  about 1.5-2% slower with MTP3.

Each file is an `--override` for `python -m tools.convert --recipe qwen3_8_27b`, with a matching
`--plan` text for the KLD harness (`tools/eval/gguf_eval.py export --plan`).

| File | What it is |
|---|---|
| `qwen3_8_27b_activation.py` | The `activation` candidate as an `--override`. Same bytes as the legacy rule. |
| `qwen3_8_27b_activation.gguf-plan.txt` | The same allocation as a harness `--plan` string (plus Q4 embedding, Q6 head). |
| `qwen3_8_27b_official.gguf-plan.txt` | The shipped `qwen3_8_27b` layout (the adopted `absolute` allocation) as a `--plan` string: the baseline for the next comparison. |
| `qwen3_8_27b_legacy.gguf-plan.txt` | The layout before PR #445 (Q5 in layers 0-35, Q4 from 36), kept for the record and as the byte budget of `tools.convert.allocation`. |

The plan files are generated, and `tests/convert/test_allocation.py` fails if they drift from the
rules or the override file. To regenerate them after changing a rule:

```python
from pathlib import Path
from tools.convert.allocation import gguf_plan, qwen3_8_27b_flexible_names
from tools.convert.official_recipes import qwen3_8_27b_legacy_text_format, qwen3_8_27b_text_format
import runpy

folder = Path("tools/convert/candidates")
names = qwen3_8_27b_flexible_names()
shipped = {n: qwen3_8_27b_text_format(n) for n in names}
legacy = {n: qwen3_8_27b_legacy_text_format(n) for n in names}
(folder / "qwen3_8_27b_official.gguf-plan.txt").write_text(gguf_plan(shipped) + "\n")
(folder / "qwen3_8_27b_legacy.gguf-plan.txt").write_text(gguf_plan(legacy) + "\n")
candidate = runpy.run_path(str(folder / "qwen3_8_27b_activation.py"))["ASSIGNMENTS"]
(folder / "qwen3_8_27b_activation.gguf-plan.txt").write_text(gguf_plan(candidate) + "\n")
```
