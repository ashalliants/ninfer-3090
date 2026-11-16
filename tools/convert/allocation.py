"""Sensitivity-driven Q4/Q5 allocation for the Qwen3.8-27B recipe.

``qwen3_8_27b`` used to pick 4 or 5 bits for attention/GDN outputs and MLP down projections by
layer index (Q5 below layer 36, Q4 from 36 on: ``qwen3_8_27b_legacy_text_format``). This tool replaces
that rule with a measurement, and the shipped rule is now the result of its ``absolute`` metric
(``QWEN3_8_27B_Q5_FROM_LAYER``). For every such projection it runs the real ``grouped_search`` encoder at Q4 and at Q5 and records the
importance-weighted squared rounding error, ``sum_k w_k (W - W')^2`` summed over rows, where
``w_k`` is the imatrix's mean squared activation of input channel ``k``. That is the expected
squared change of the projection's output under independent channels. It then chooses formats
under the same byte budget as that legacy rule, and writes an ``--override`` file for
``python -m tools.convert``.

Only roles whose Q4 and Q5 kernels are both registered are allocated (``FLEXIBLE_ROLES``).
The error is a proxy: it ignores how much each layer's output matters downstream, so a proposed
allocation must be confirmed with a KLD, speed and perplexity run before it replaces the shipped rule
(docs/maintainer/quality-trade-experiments.md records how the current one was).

    python -m tools.convert.allocation --model /path/to/Qwen3.8-27B \
        --imatrix qwen3_8_27b.imatrix.safetensors --device cpu \
        --out allocation.json --emit-override allocation_override.py

Scoring runs the search encoder once per format per tensor: about 10 s for a 5120x17408 MLP
down projection on an Apple-silicon CPU, so the 128 projections take a quarter of an hour or so.
``--out`` is written (atomically) after every tensor and a repeated run resumes from it, but only if the
imatrix content and the checkpoint are the ones the report was made from; ``--activations`` scores are
merged under the same check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from tools.artifact.formats import get_format
from tools.artifact.layouts import row_split_geometry

from . import imatrix as imatrix_file
from .official_recipes import (
    Q4,
    Q5,
    QWEN3_8_27B_LATE_LAYER,
    imatrix_search,
    qwen3_8_27b_legacy_text_format,
)
from .quantization.groupwise import search_quantize_matrix

FORMATS = (Q4, Q5)
# Roles with both a Q4 and a Q5 registered kernel for the 27B shapes, in the official recipe.
FLEXIBLE_ROLES = ("/attention/output", "/gdn/output", "/mlp/down")
METRICS = ("relative", "absolute", "activation")
REPORT_VERSION = 2


def qwen3_8_27b_flexible_names(layers: int = 64, full_attention_interval: int = 4) -> list[str]:
    """`flexible_parameters` of the real Qwen3.8-27B without opening a checkpoint."""

    names = []
    for layer in range(layers):
        mixer = "attention/output" if layer % full_attention_interval == full_attention_interval - 1 else "gdn/output"
        names += [f"text/layers/{layer}/{mixer}", f"text/layers/{layer}/mlp/down"]
    return names


def gguf_plan(assignments: Mapping[str, str], *, embedding: str = "q4", head: str = "q6") -> str:
    """The ``--plan`` text of ``tools/eval/gguf_eval.py`` for these assignments.

    The harness's role defaults already match the official recipe for every role this tool does
    not allocate (it only differs in the embedding and head, set here, and in these layers).
    """

    short = {Q4: "q4", Q5: "q5"}
    entries = [f"text/token_embedding={embedding}", f"text/output_head={head}"]
    for name, format in sorted(assignments.items(), key=lambda item: (int(item[0].split("/")[2]), item[0])):
        _, _, layer, role = name.split("/", 3)
        entries.append(f"{layer}:{role}={short[format]}")
    return ",".join(entries)


def flexible_parameters(model, layers: range | None = None) -> list[str]:
    """Names of the dense text projections this tool allocates, in model order."""

    if "num_experts" in model.config:
        raise ValueError("allocation applies to Qwen3.5 Dense text layers")
    names = []
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if not name.endswith(FLEXIBLE_ROLES):
            continue
        if layers is None or int(name.split("/")[2]) in layers:
            names.append(name)
    return names


def payload_bytes(format: str, shape: Sequence[int]) -> int:
    return row_split_geometry(format, shape).payload_bytes


def dequantize(
    weight: torch.Tensor, format: str, importance: torch.Tensor | None, *, device: str
) -> torch.Tensor:
    """The weights the artifact would store for ``weight``: the search encoder's round trip, FP32."""

    encoded = search_quantize_matrix(
        weight, get_format(format), importance=importance, negative_scales=True, device=device
    )
    decoded = encoded.scales.to(torch.float32).unsqueeze(-1) * encoded.codes.to(torch.float32)
    return decoded.reshape(weight.shape[0], -1)[:, : weight.shape[1]]


def weighted_error(
    weight: torch.Tensor, format: str, importance: torch.Tensor | None, *, device: str
) -> tuple[float, float]:
    """(importance-weighted squared rounding error, importance-weighted energy) of one chunk."""

    decoded = dequantize(weight, format, importance, device=device)
    n, k = weight.shape
    values = weight.to(device=decoded.device, dtype=torch.float32)
    w = (
        torch.ones(k, dtype=torch.float32, device=decoded.device)
        if importance is None
        else importance.to(device=decoded.device, dtype=torch.float32)
    )
    error = (w * (values - decoded) ** 2).sum(dtype=torch.float64)
    energy = (w * values**2).sum(dtype=torch.float64)
    return float(error), float(energy)


def score_parameter(
    parameter,
    importance: torch.Tensor | None,
    *,
    formats: Sequence[str] = FORMATS,
    device: str = "cpu",
    rows_per_chunk: int = 512,
) -> dict:
    """Weighted rounding error of one projection at each format, plus its weighted energy."""

    n, k = parameter.shape
    if importance is not None and importance.shape != (k,):
        raise ValueError(f"{parameter.name}: imatrix vector length {tuple(importance.shape)} != K={k}")
    errors = {format: 0.0 for format in formats}
    energy = 0.0
    for begin in range(0, n, rows_per_chunk):
        end = min(n, begin + rows_per_chunk)
        weight = parameter.source.values(begin * k, end * k).reshape(end - begin, k)
        if not weight.dtype.is_floating_point:
            raise TypeError(f"{parameter.name}: source must provide floating-point values")
        chunk_energy = 0.0
        for format in formats:
            error, chunk_energy = weighted_error(weight, format, importance, device=device)
            errors[format] += error
        energy += chunk_energy
    return {
        "shape": [n, k],
        "weighted": importance is not None,
        "energy": energy,
        "error": errors,
        "bytes": {format: payload_bytes(format, (n, k)) for format in formats},
    }


@dataclass(frozen=True, slots=True)
class Option:
    format: str
    bytes: int
    cost: float


def _hull(options: Sequence[Option]) -> list[Option]:
    """Lower convex hull of (bytes, cost): the options a budgeted optimum can use."""

    best: list[Option] = []
    for option in sorted(options, key=lambda o: (o.bytes, o.cost)):
        if not best or option.cost < best[-1].cost:
            best.append(option)
    hull: list[Option] = []
    for option in best:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Drop b when it lies on or above the segment a-option.
            if (b.cost - a.cost) * (option.bytes - a.bytes) >= (option.cost - a.cost) * (b.bytes - a.bytes):
                hull.pop()
            else:
                break
        hull.append(option)
    return hull


def allocate(options: Mapping[str, Sequence[Option]], budget_bytes: int) -> dict[str, str]:
    """Pick one format per unit to minimise total cost within ``budget_bytes``.

    Greedy on the convex hull of each unit's options: start every unit at its cheapest
    option and buy the upgrades with the largest cost reduction per byte while they fit.
    Exact when the budget lands on a step boundary; otherwise within one upgrade of optimal.
    """

    hulls = {name: _hull(items) for name, items in options.items()}
    spent = sum(hull[0].bytes for hull in hulls.values())
    if spent > budget_bytes:
        raise ValueError(f"budget {budget_bytes} is below the cheapest allocation {spent}")
    level = {name: 0 for name in hulls}
    steps = []
    for name, hull in hulls.items():
        for i in range(1, len(hull)):
            extra = hull[i].bytes - hull[i - 1].bytes
            gain = hull[i - 1].cost - hull[i].cost
            steps.append((-gain / extra, name, i, extra))
    for _, name, i, extra in sorted(steps):
        if level[name] == i - 1 and spent + extra <= budget_bytes:
            level[name] = i
            spent += extra
    return {name: hulls[name][level[name]].format for name in hulls}


def cost(entry: Mapping, format: str, metric: str) -> float:
    if metric == "activation":
        # Measured on real activations by tools.convert.allocation_activations, merged in as
        # entry["activation"]: output change relative to the layer's residual-stream energy.
        if "activation" not in entry:
            raise ValueError("metric 'activation' needs scores from --activations")
        item = entry["activation"]
        return item["output_error"][format] / item["residual_energy"]
    error = entry["error"][format]
    if metric == "absolute":
        return error
    return error / entry["energy"] if entry["energy"] > 0 else 0.0


def plan(
    scores: Mapping[str, Mapping],
    *,
    metric: str = "relative",
    late_layer: int = QWEN3_8_27B_LATE_LAYER,
    budget_bytes: int | None = None,
) -> dict:
    """Baseline (the legacy layer rule) and proposed formats for the scored projections.

    The default budget is exactly the bytes of the legacy rule over the same projections.
    """

    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    baseline = {name: qwen3_8_27b_legacy_text_format(name, late_layer) for name in scores}
    options = {
        name: [Option(f, entry["bytes"][f], cost(entry, f, metric)) for f in entry["error"]]
        for name, entry in scores.items()
    }
    baseline_bytes = sum(scores[n]["bytes"][f] for n, f in baseline.items())
    budget = baseline_bytes if budget_bytes is None else budget_bytes
    proposed = allocate(options, budget)

    def total(choice):
        return {
            "bytes": sum(scores[n]["bytes"][f] for n, f in choice.items()),
            "cost": sum(cost(scores[n], f, metric) for n, f in choice.items()),
            "q5": sum(f == Q5 for f in choice.values()),
        }

    return {
        "metric": metric,
        "budget_bytes": budget,
        "baseline": baseline,
        "proposed": proposed,
        "totals": {"baseline": total(baseline), "proposed": total(proposed)},
    }


def format_override(proposed: Mapping[str, str], source: str) -> str:
    """Text of an ``--override`` file that assigns the proposed formats."""

    lines = [
        f'"""Generated by tools.convert.allocation from {source}. Pass with --override."""',
        "",
        "from tools.convert.allocation import apply_assignments",
        "",
        "ASSIGNMENTS = {",
        *(f'    "{name}": "{format}",' for name, format in sorted(proposed.items())),
        "}",
        "",
        "",
        "def configure(model, recipe, sources):",
        "    apply_assignments(recipe, sources, ASSIGNMENTS)",
        "",
    ]
    return "\n".join(lines)


def apply_assignments(recipe, sources, assignments: Mapping[str, str]) -> None:
    """Assign formats with the same imatrix search encoder `qwen3_8_27b` uses."""

    search = imatrix_search(sources)
    for name, format in assignments.items():
        recipe.assign(name, format=format, **search)


def file_sha256(path: Path) -> str:
    """Content hash of a (small) input file, so a regenerated file of the same name is not mistaken
    for the one a report was made from."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(model_dir: Path) -> str:
    """Fingerprint of a safetensors checkpoint without reading its weights: the config, the tensor
    index, and every shard's name and size. A different checkpoint, or a re-exported one, changes it."""

    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model_dir / name  # a single-file checkpoint has no index
        digest.update(name.encode() + b"\0" + (path.read_bytes() if path.exists() else b"<absent>"))
    for shard in sorted(model_dir.glob("*.safetensors")):
        digest.update(f"{shard.name}\0{shard.stat().st_size}\0".encode())
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` so that an interruption leaves either the old file or the new one, never half."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def require_same_inputs(found: Mapping, expected: Mapping, keys: Sequence[str], where: Path) -> None:
    """Refuse a report whose recorded inputs differ from this run's (stale scores are never merged)."""

    differing = [key for key in keys if found.get(key) != expected.get(key)]
    if differing:
        raise ValueError(
            f"{where} was made from different inputs ({', '.join(differing)}); "
            "regenerate it, or use another --out"
        )


def _layer_range(text: str | None) -> range | None:
    if text is None:
        return None
    low, _, high = text.partition("-")
    return range(int(low), int(high or low) + 1)


def _settings(args) -> dict:
    return {
        "imatrix_sha256": file_sha256(args.imatrix),
        "checkpoint": checkpoint_identity(args.model),
        "formats": list(FORMATS),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="Qwen3.8-27B checkpoint directory")
    parser.add_argument("--imatrix", type=Path, required=True, help="file from tools.convert.imatrix")
    parser.add_argument("--out", type=Path, required=True, help="JSON report; reused to resume")
    parser.add_argument("--emit-override", type=Path, help="write the proposed --override file here")
    parser.add_argument("--layers", help="score only these layers, e.g. 36-63 (default all)")
    parser.add_argument("--metric", choices=METRICS, default="relative")
    parser.add_argument(
        "--activations", type=Path,
        help="JSON from tools.convert.allocation_activations (required for --metric activation)",
    )
    parser.add_argument("--budget-bytes", type=int, help="default: bytes of the legacy layer rule")
    parser.add_argument("--late-layer", type=int, default=QWEN3_8_27B_LATE_LAYER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    args = parser.parse_args(argv)

    from .qwen3_5 import build_model
    from .sources.safetensors import SafetensorsSource

    settings = _settings(args)
    report = {"version": REPORT_VERSION, "settings": settings, "scores": {}}
    if args.out.exists():
        report = json.loads(args.out.read_text())
        if report.get("version") != REPORT_VERSION:
            raise ValueError(f"{args.out} is from an older version of this tool; use another --out")
        require_same_inputs(report.get("settings", {}), settings, list(settings), args.out)
    vectors = imatrix_file.load(str(args.imatrix))
    with SafetensorsSource(args.model) as base:
        model = build_model(base, components=("text",))
        names = flexible_parameters(model, _layer_range(args.layers))
        for index, name in enumerate(names):
            if name not in report["scores"]:
                report["scores"][name] = score_parameter(
                    model.parameters[name],
                    vectors.get(name),
                    device=args.device,
                    rows_per_chunk=args.rows_per_chunk,
                )
                atomic_write_text(args.out, json.dumps(report))
            print(f"[{index + 1}/{len(names)}] {name}", flush=True)
    scores = {name: dict(report["scores"][name]) for name in names}
    if args.activations:
        activation_report = json.loads(args.activations.read_text())
        require_same_inputs(
            activation_report.get("settings", {}), settings, ("imatrix_sha256", "checkpoint"), args.activations
        )
        measured = activation_report["scores"]
        for name in names:
            if name in measured:
                scores[name]["activation"] = measured[name]
    if args.metric == "activation":
        missing = [name for name in names if "activation" not in scores[name]]
        if missing:
            raise ValueError(f"--activations has no scores for {len(missing)} projections, e.g. {missing[0]}")
    if not all(entry["weighted"] for entry in scores.values()):
        print("warning: some projections have no imatrix vector and were scored unweighted")
    report["plan"] = plan(
        scores, metric=args.metric, late_layer=args.late_layer, budget_bytes=args.budget_bytes
    )
    atomic_write_text(args.out, json.dumps(report, indent=1))
    totals = report["plan"]["totals"]
    for label in ("baseline", "proposed"):
        item = totals[label]
        print(f"{label:9s} {item['bytes'] / 1e9:8.3f} GB  Q5 count {item['q5']:3d}  {args.metric} cost {item['cost']:.6g}")
    if args.emit_override:
        args.emit_override.write_text(format_override(report["plan"]["proposed"], args.out.name))
        print(f"wrote {args.emit_override}")


if __name__ == "__main__":
    main()
