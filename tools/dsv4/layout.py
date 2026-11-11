#!/usr/bin/env python3
"""The exact byte layout of a DeepSeek V4-family checkpoint, from its safetensors headers.

`fetch_headers.py` pulls the headers (no weights). This module groups the tensors into the pieces
an offloading engine cares about and prices each one:

  * routed experts (the bulk; one `ExpertSize` per expert, identical across layers),
  * shared experts and the dense attention path (always resident),
  * Engram tables (huge, read a few rows per token),
  * embeddings/head, MTP and vision (optional, sized separately).

    python tools/dsv4/layout.py SCRATCH/v41                      # report from fetched headers
    python tools/dsv4/layout.py SCRATCH/v41 --tiers 17 46 --bpw 2.31   # + tier split at a re-quant

Everything here is arithmetic on header data, so it runs on any machine.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

from gguf_sizes import BPW

GIB = 1 << 30

# Bits per weight (data + scales) of the formats an expert could be stored in. `fp4_native` is the
# checkpoint's own e2m1 data plus one e8m0 scale byte per 32 weights; the rest are GGML types (the
# one table is `gguf_sizes.BPW`).
BITS_PER_WEIGHT = {"fp4_native": 4.25} | {
    name.lower(): BPW[name] for name in ("IQ4_XS", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S")
}


@dataclasses.dataclass(frozen=True)
class ExpertSize:
    """One routed expert: w1 (gate) and w3 (up) are inter x dim, w2 (down) is dim x inter."""

    dim: int
    inter: int
    block: int = 32  # weights per e8m0 scale along the input axis

    @property
    def params(self) -> int:
        return 3 * self.dim * self.inter

    @property
    def fp4_data_bytes(self) -> int:
        return self.params // 2

    @property
    def fp4_scale_bytes(self) -> int:
        return self.params // self.block

    @property
    def fp4_bytes(self) -> int:
        return self.fp4_data_bytes + self.fp4_scale_bytes

    def bytes_at(self, bits_per_weight: float) -> int:
        """Stored size at an arbitrary rate, rounded up to a whole byte."""
        return -(-int(self.params * bits_per_weight) // 8)


def text_config(config: dict) -> dict:
    return config.get("text_config", config)


def expert_size(config: dict) -> ExpertSize:
    c = text_config(config)
    return ExpertSize(dim=c["hidden_size"], inter=c["moe_intermediate_size"])


def load_headers(directory: Path) -> dict[str, dict]:
    """Every tensor's {dtype, shape, data_offsets} from `directory/headers/*.json`, by name."""
    tensors: dict[str, dict] = {}
    for path in sorted((directory / "headers").glob("*.json")):
        for name, entry in json.loads(path.read_text()).items():
            tensors[name] = {**entry, "shard": path.name.removesuffix(".json")}
    if not tensors:
        raise FileNotFoundError(f"no headers in {directory}/headers (run fetch_headers.py)")
    return tensors


def nbytes(entry: dict) -> int:
    start, end = entry["data_offsets"]
    return end - start


# Order matters: the first pattern that matches names the group.
GROUPS = [
    ("routed_experts", re.compile(r"^layers\.\d+\.ffn\.experts\.")),
    ("mtp_routed_experts", re.compile(r"^mtp\.\d+\.ffn\.experts\.")),
    ("engram_tables", re.compile(r"^layers\.\d+\.engram\.embed\.")),
    ("mtp_dense", re.compile(r"^mtp\.")),
    ("vision", re.compile(r"^(vision|aligner)\.")),
    ("embed_and_head", re.compile(r"^(embed|head)\.")),
    ("shared_experts", re.compile(r"^layers\.\d+\.ffn\.shared_experts\.")),
    ("router", re.compile(r"^layers\.\d+\.ffn\.gate\.")),
    ("engram_projection", re.compile(r"^layers\.\d+\.engram\.")),
    ("attention", re.compile(r"^layers\.\d+\.attn\.")),
]


def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if pattern.match(name):
            return group
    return "other_dense"  # norms, hyper-connection mixers


def group_sizes(tensors: dict[str, dict]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, entry in tensors.items():
        group = group_of(name)
        sizes[group] = sizes.get(group, 0) + nbytes(entry)
    return sizes


def measured_expert_bytes(tensors: dict[str, dict], layer: int, expert: int) -> int:
    prefix = f"layers.{layer}.ffn.experts.{expert}."
    return sum(nbytes(e) for n, e in tensors.items() if n.startswith(prefix))


@dataclasses.dataclass(frozen=True)
class TierSplit:
    """How many routed experts each tier holds, and the fraction of all experts that is."""

    vram: int
    ram: int
    ssd: int
    total: int

    def fractions(self) -> tuple[float, float, float]:
        return self.vram / self.total, self.ram / self.total, self.ssd / self.total


def split_tiers(total_experts: int, expert_bytes: int, vram_bytes: int, ram_bytes: int) -> TierSplit:
    """Fill VRAM, then RAM, with whole experts; the rest goes to the SSD tier."""
    if expert_bytes <= 0 or total_experts < 0 or vram_bytes < 0 or ram_bytes < 0:
        raise ValueError("sizes must be positive")
    vram = min(total_experts, vram_bytes // expert_bytes)
    ram = min(total_experts - vram, ram_bytes // expert_bytes)
    return TierSplit(vram, ram, total_experts - vram - ram, total_experts)


def report(directory: Path, tiers: tuple[float, float] | None, bpw: float | None) -> str:
    config = json.loads((directory / "config.json").read_text())
    c = text_config(config)
    tensors = load_headers(directory)
    sizes = group_sizes(tensors)
    es = expert_size(config)
    n_layers, n_routed, top_k = c["num_hidden_layers"], c["n_routed_experts"], c["num_experts_per_tok"]
    total_experts = n_layers * n_routed

    measured = measured_expert_bytes(tensors, 3, 0)
    if measured != es.fp4_bytes:
        raise AssertionError(f"config says an expert is {es.fp4_bytes} B, headers say {measured} B")

    lines = [f"checkpoint {sum(sizes.values()) / 1e9:,.1f} GB over {len(tensors)} tensors", ""]
    for group, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {group:20s} {size / 1e9:9.3f} GB")
    lines += [
        "",
        f"routed experts: {n_layers} layers x {n_routed} = {total_experts:,}, top-{top_k}"
        f" ({n_layers * top_k} reads per token)",
        f"  one expert: {es.params / 1e6:.1f} M weights, {es.fp4_bytes / 1e6:.2f} MB native"
        f" ({es.fp4_data_bytes / 1e6:.2f} data + {es.fp4_scale_bytes / 1e6:.2f} scales)",
        f"  all routed: {es.params * total_experts / 1e9:.0f} B weights",
        "",
        "all routed experts at each format (and bytes touched per decode token):",
    ]
    for name, bits in BITS_PER_WEIGHT.items():
        per_expert = es.bytes_at(bits)
        lines.append(
            f"  {name:11s} {bits:6.4f} bpw  expert {per_expert / 1e6:6.2f} MB"
            f"  all {per_expert * total_experts / GIB:7.1f} GiB  token {per_expert * n_layers * top_k / GIB:5.2f} GiB"
        )
    if tiers:
        vram_gib, ram_gib = tiers
        use = bpw or BITS_PER_WEIGHT["iq2_xs"]
        per_expert = es.bytes_at(use)
        split = split_tiers(total_experts, per_expert, int(vram_gib * GIB), int(ram_gib * GIB))
        f_vram, f_ram, f_ssd = split.fractions()
        lines += [
            "",
            f"tiers at {use} bpw ({per_expert / 1e6:.2f} MB/expert), {vram_gib} GiB VRAM cache + {ram_gib} GiB RAM:",
            f"  VRAM {split.vram:6,d} experts ({f_vram:5.1%})   RAM {split.ram:6,d} ({f_ram:5.1%})"
            f"   SSD {split.ssd:6,d} ({f_ssd:5.1%})",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="a fetch_headers.py output directory")
    ap.add_argument("--tiers", type=float, nargs=2, metavar=("VRAM_GIB", "RAM_GIB"))
    ap.add_argument("--bpw", type=float, help="bits per weight for the tier split (default iq2_xs)")
    args = ap.parse_args()
    print(report(args.directory, tuple(args.tiers) if args.tiers else None, args.bpw))


if __name__ == "__main__":
    main()
