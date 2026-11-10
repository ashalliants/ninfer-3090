# Candidate recipe overrides

Allocations that are **not** the shipped recipe. Each is an `--override` file for
`python -m tools.convert --recipe qwen3_8_27b`, with a matching `--plan` text for the KLD harness
(`tools/eval/gguf_eval.py export --plan`). See pull request #445 for what they are, how they were
made and how to validate one.

| File | What it is |
|---|---|
| `qwen3_8_27b_activation.py` | Q4/Q5 for the 128 attention-output, GDN-output and MLP-down projections, chosen by `tools.convert.allocation --metric activation`. Same bytes as the official rule. |
| `qwen3_8_27b_activation.gguf-plan.txt` | The same allocation as a harness `--plan` string (plus Q4 embedding, Q6 head). |
| `qwen3_8_27b_absolute.py` / `.gguf-plan.txt` | The second candidate, chosen from the imatrix-weighted weight error alone (`--metric absolute`). Agrees with the activation one on 112 of 128 tensors. |
| `qwen3_8_27b_official.gguf-plan.txt` | The shipped `qwen3_8_27b` layout as a harness `--plan` string: the baseline to compare against. |

The plan files are generated, and `tests/convert/test_allocation.py` fails if they drift from the
override file or from the official rule. To regenerate them after changing either:

```python
from pathlib import Path
from tools.convert.allocation import gguf_plan, qwen3_8_27b_flexible_names
from tools.convert.official_recipes import qwen3_8_27b_text_format
import runpy

folder = Path("tools/convert/candidates")
official = {n: qwen3_8_27b_text_format(n) for n in qwen3_8_27b_flexible_names()}
(folder / "qwen3_8_27b_official.gguf-plan.txt").write_text(gguf_plan(official) + "\n")
candidate = runpy.run_path(str(folder / "qwen3_8_27b_activation.py"))["ASSIGNMENTS"]
(folder / "qwen3_8_27b_activation.gguf-plan.txt").write_text(gguf_plan(candidate) + "\n")
```
