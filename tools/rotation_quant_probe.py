"""CPU/Apple-silicon probe: does a Hadamard rotation lower Qwen3.8-27B's quantization error?

Answers, before any CUDA work, whether rotating projection inputs (as the Bonsai checkpoints do)
would help the groupwise-int weights of the regular 27B, and INT8 activations (AllowA8Int).
For each probed matrix W [N, K] and real calibration activations X [K, T] it reports the
output error ||W X - W' X'||^2 / ||W X||^2 of each arm:

  primal      W and X as stored; the current recipe.
  online      W Q^T and Q X, Q the 1024-block signed Walsh-Hadamard of hadamard_transform.
              The Bonsai layout: the runtime rotates the input of this projection.
  fold        W diag(g) Q^T and Q diag(1/g) X for matrices that read an RMSNorm output with
              weight g = 1 + gamma. The QuaRot residual-stream layout: the norm weight and the
              rotation are folded into the weights offline, so the runtime does no transform.

Weights are quantized with the converter's own ``search_quantize_matrix`` (q4/q5_g64_fp16,
negative scales, importance = mean squared input channel of that arm, as an imatrix gives).
Activations are quantized like AllowA8Int: s8 with one absmax scale per token and per 64-channel
group, and also with one scale per token.

Steps (each writes into --work):

  fetch    range-read only the probed layers' BF16 tensors from Qwen/Qwen3.8-27B (~5 GB).
  capture  run mlx-community/Qwen3.8-27B-4bit over held-out text and save every probed
           projection's input. Needs mlx and mlx-lm; the 4-bit model's activations stand in
           for BF16's, which is fine for outlier statistics.
  capture-torch  the same without MLX: stream the BF16 model through transformers on the CPU, one
           decoder layer at a time (about 54 GB of shards on disk, a few GB of RAM).
  analyze  quantize and score every arm; prints a table and writes results.json.

    python tools/rotation_quant_probe.py fetch   --work /tmp/hq
    python tools/rotation_quant_probe.py capture --work /tmp/hq
    python tools/rotation_quant_probe.py analyze --work /tmp/hq
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SOURCE = "Qwen/Qwen3.8-27B"
ACTIVATIONS = "mlx-community/Qwen3.8-27B-4bit"
PREFIX = "model.language_model.layers"
# GDN layers 0, 32 and 62 and full-attention layers 3, 31 and 63: early, middle and late.
LAYERS = (0, 3, 31, 32, 62, 63)
BLOCK = 1024
CORPUS = REPO / "eval/corpora/perplexity-heldout-2026-09/data"
DOMAINS = ("arxiv", "github_code", "synthetic_chat", "wikipedia_en", "wikipedia_zh")


def _attention(layer: int) -> bool:
    return layer % 4 == 3


# (weight suffix, captured input, norm whose output the input is or None).
def matrices(layer: int) -> list[tuple[str, str, str | None]]:
    if _attention(layer):
        mixer = [
            ("self_attn.q_proj", "mixer_in", "input_layernorm"),
            ("self_attn.k_proj", "mixer_in", "input_layernorm"),
            ("self_attn.v_proj", "mixer_in", "input_layernorm"),
            ("self_attn.o_proj", "mixer_out", None),
        ]
    else:
        mixer = [
            ("linear_attn.in_proj_qkv", "mixer_in", "input_layernorm"),
            ("linear_attn.in_proj_z", "mixer_in", "input_layernorm"),
            ("linear_attn.out_proj", "mixer_out", None),
        ]
    return mixer + [
        ("mlp.gate_proj", "mlp_in", "post_attention_layernorm"),
        ("mlp.up_proj", "mlp_in", "post_attention_layernorm"),
        ("mlp.down_proj", "down_in", None),
    ]


# ---------------------------------------------------------------------------------------------
# fetch


def _range(url: str, begin: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end - 1}"})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    if len(data) != end - begin:
        raise IOError(f"{url}: asked for {end - begin} bytes, got {len(data)}")
    return data


def fetch(work: Path) -> None:
    from safetensors.numpy import save_file

    base = f"https://huggingface.co/{SOURCE}/resolve/main/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as response:
        weight_map = json.load(response)["weight_map"]
    wanted = {}
    for layer in LAYERS:
        names = [f"{PREFIX}.{layer}.{suffix}.weight" for suffix, _, _ in matrices(layer)]
        names += [f"{PREFIX}.{layer}.{n}.weight" for n in ("input_layernorm", "post_attention_layernorm")]
        for name in names:
            wanted.setdefault(weight_map[name], []).append(name)
    out = work / "weights"
    out.mkdir(parents=True, exist_ok=True)
    for shard, names in sorted(wanted.items()):
        url = base + shard
        length = struct.unpack("<Q", _range(url, 0, 8))[0]
        header = json.loads(_range(url, 8, 8 + length))
        for name in names:
            path = out / f"{name}.safetensors"
            if path.exists():
                continue
            entry = header[name]
            if entry["dtype"] != "BF16":
                raise ValueError(f"{name}: expected BF16, found {entry['dtype']}")
            begin, end = entry["data_offsets"]
            raw = np.frombuffer(_range(url, 8 + length + begin, 8 + length + end), dtype=np.uint16)
            # BF16 -> FP32 is a 16-bit shift; keep FP32 so numpy can store it.
            values = (raw.astype(np.uint32) << 16).view(np.float32).reshape(entry["shape"])
            save_file({"w": values}, str(path))
            print(f"{name} {tuple(entry['shape'])}", flush=True)


# ---------------------------------------------------------------------------------------------
# capture


def _texts(tokens_per_domain: int, tokenizer) -> list[list[int]]:
    sequences = []
    for domain in DOMAINS:
        files = sorted((CORPUS / domain).glob("*.txt"))
        text = files[0].read_text(encoding="utf-8")
        ids = tokenizer.encode(text)
        if len(ids) < tokens_per_domain:
            raise ValueError(f"{domain}: only {len(ids)} tokens")
        sequences.append(ids[:tokens_per_domain])
    return sequences


def capture_torch(work: Path, tokens_per_domain: int) -> None:
    """The MLX-free capture: stream the BF16 model through transformers' own decoder layers.

    The full model (about 54 GB) does not have to fit in RAM: the five held-out sequences run as
    one batch, one decoder layer at a time, each built from its own tensors in FP32 and freed
    before the next. Only the shards are downloaded in full."""

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
    )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(SOURCE).text_config
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(SOURCE)
    ids = torch.tensor(_texts(tokens_per_domain, tokenizer))
    weight_map = json.loads(Path(hf_hub_download(SOURCE, "model.safetensors.index.json")).read_text())["weight_map"]

    def tensors(prefix: str) -> dict[str, torch.Tensor]:
        found = {}
        for name, shard in weight_map.items():
            if name.startswith(prefix):
                with safe_open(hf_hub_download(SOURCE, shard), "pt") as f:
                    found[name[len(prefix):]] = f.get_tensor(name).float()
        return found

    embed = tensors("model.language_model.embed_tokens.")["weight"]
    hidden = embed[ids]
    del embed
    position_ids = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, ids.shape[0], -1)
    rotary = Qwen3_5TextRotaryEmbedding(config=config)
    position_embeddings = rotary(hidden, position_ids)
    store: dict[tuple[int, str], np.ndarray] = {}

    for layer in range(config.num_hidden_layers):
        with torch.device("meta"):
            block = Qwen3_5DecoderLayer(config, layer)
        block.load_state_dict(tensors(f"{PREFIX}.{layer}."), assign=True)
        probed = layer in LAYERS
        handles = []
        if probed:
            mixer = block.self_attn if _attention(layer) else block.linear_attn
            first = mixer.q_proj if _attention(layer) else mixer.in_proj_qkv
            for key, module in (
                ("mixer_in", first),
                ("mixer_out", mixer.o_proj if _attention(layer) else mixer.out_proj),
                ("mlp_in", block.mlp.gate_proj),
                ("down_in", block.mlp.down_proj),
            ):
                def hook(_module, args, key=key):
                    store[(layer, key)] = args[0].reshape(-1, args[0].shape[-1]).half().numpy()

                handles.append(module.register_forward_pre_hook(hook))
        hidden = block(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids[0],
        )
        for handle in handles:
            handle.remove()
        del block
        print(f"layer {layer} done{' (probed)' if probed else ''}", flush=True)
    out = work / "activations"
    out.mkdir(parents=True, exist_ok=True)
    for (layer, name), values in store.items():
        np.save(out / f"{layer}.{name}.npy", values)


def capture(work: Path, tokens_per_domain: int) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    model, tokenizer = load(ACTIVATIONS)
    layers = model.language_model.model.layers
    store: dict[tuple[int, str], list[np.ndarray]] = {}

    def record(module: nn.Module, key: tuple[int, str]) -> None:
        call = module.__call__

        class Hooked(type(module)):
            def __call__(self, x, *args, **kwargs):
                flat = np.array(x.astype(mx.float32)).reshape(-1, x.shape[-1])
                store.setdefault(key, []).append(flat.astype(np.float16))
                return call(x, *args, **kwargs)

        module.__class__ = Hooked

    for layer in LAYERS:
        block = layers[layer]
        if _attention(layer):
            record(block.self_attn.q_proj, (layer, "mixer_in"))
            record(block.self_attn.o_proj, (layer, "mixer_out"))
        else:
            record(block.linear_attn.in_proj_qkv, (layer, "mixer_in"))
            record(block.linear_attn.out_proj, (layer, "mixer_out"))
        record(block.mlp.gate_proj, (layer, "mlp_in"))
        record(block.mlp.down_proj, (layer, "down_in"))

    for ids in _texts(tokens_per_domain, tokenizer):
        logits = model(mx.array([ids]))
        mx.eval(logits)
        print(f"captured {len(ids)} tokens", flush=True)
    out = work / "activations"
    out.mkdir(parents=True, exist_ok=True)
    for (layer, name), parts in store.items():
        np.save(out / f"{layer}.{name}.npy", np.concatenate(parts, axis=0))


# ---------------------------------------------------------------------------------------------
# analyze


def _signs(k: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(k)
    return torch.randint(0, 2, (k,), generator=generator).float() * 2 - 1


def rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q x along dim 0 for x [K, T]: hadamard_transform's forward form, 2^-5 H (s * x) per block."""

    k, t = x.shape
    y = (x * signs[:, None]).reshape(k // BLOCK, BLOCK, t).clone()
    h = 1
    while h < BLOCK:
        y = y.reshape(k // BLOCK, BLOCK // (2 * h), 2, h, t)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return y.reshape(k, t) / BLOCK**0.5


def quantize_weight(w, fmt: str, importance, device):
    from tools.convert.quantization.groupwise import search_quantize_matrix

    encoded = search_quantize_matrix(
        w, fmt, importance=importance, negative_scales=True, device=device
    )
    n, k = w.shape
    deq = encoded.scales.float().unsqueeze(-1) * encoded.codes.float()
    return deq.reshape(n, -1)[:, :k].to(w.device)


def quantize_activation(x, group: int | None):
    """s8 symmetric fake-quantization of x [K, T]: one absmax scale per token and group."""

    k, t = x.shape
    g = k if group is None else group
    grouped = x.reshape(k // g, g, t)
    scale = grouped.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    return (grouped / scale).round().clamp(-127, 127).mul(scale).reshape(k, t)


def crest(x) -> float:
    """Mean over tokens of max|x| / rms(x): how far one channel sticks out of each token."""

    return float((x.abs().amax(dim=0) / x.pow(2).mean(dim=0).sqrt().clamp_min(1e-12)).mean())


def value_plane(work: Path, layer: int, x: torch.Tensor) -> dict[str, float]:
    """rk8v4/rk4v4 value plane (signed int4, absmax/7 per 32 channels) on real V rows, stored
    as is and stored as R v per 256-wide head with R the normalized Hadamard H256/16, the
    rotation the NVFP4 and K8V4 value planes already use. Attention output is linear in V,
    so the relative V error carries over to it."""

    from safetensors.numpy import load_file

    w = torch.from_numpy(
        load_file(str(work / "weights" / f"{PREFIX}.{layer}.self_attn.v_proj.weight.safetensors"))["w"]
    )
    v = (w @ x).reshape(-1, 256, x.shape[1])  # [kv heads, 256, T]
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / 16

    def int4(values):
        grouped = values.reshape(values.shape[0], -1, 32, values.shape[-1])
        scale = grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7.0
        return (grouped / scale).round().clamp(-7, 7).mul(scale).reshape(values.shape)

    energy = float(v.pow(2).sum())
    rotated = torch.einsum("ij,hjt->hit", h, v)
    return {
        "v_int4": float((int4(v) - v).pow(2).sum()) / energy,
        "v_int4_rotated": float((torch.einsum("ji,hjt->hit", h, int4(rotated)) - v).pow(2).sum()) / energy,
        "v_crest": crest(v.reshape(-1, v.shape[-1])),
    }


def analyze(work: Path, device: str, tokens: int | None) -> None:
    from safetensors.numpy import load_file

    sys.path.insert(0, str(REPO))
    torch.set_grad_enabled(False)
    results = []
    for layer in LAYERS:
        for suffix, source, norm in matrices(layer):
            w = torch.from_numpy(
                load_file(str(work / "weights" / f"{PREFIX}.{layer}.{suffix}.weight.safetensors"))["w"]
            )
            x = torch.from_numpy(np.load(work / "activations" / f"{layer}.{source}.npy")).float().T
            if tokens:
                x = x[:, :tokens]
            x = x.contiguous()
            signs = _signs(w.shape[1])
            arms = {"primal": (w, x), "online": (rotate(w.T.contiguous(), signs).T, rotate(x, signs))}
            if norm:
                g = 1 + torch.from_numpy(
                    load_file(str(work / "weights" / f"{PREFIX}.{layer}.{norm}.weight.safetensors"))["w"]
                )
                arms["fold"] = (
                    rotate((w * g).T.contiguous(), signs).T,
                    rotate(x / g[:, None], signs),
                )
            reference = w @ x
            energy = float(reference.pow(2).sum())

            def err(product) -> float:
                return float((reference - product).pow(2).sum()) / energy

            row = {"layer": layer, "matrix": suffix, "shape": list(w.shape)}
            for arm, (wa, xa) in arms.items():
                wa = wa.contiguous()
                importance = xa.pow(2).mean(dim=1)
                cell = {"crest": crest(xa)}
                xg = quantize_activation(xa, 64)
                cell["a8_g64"] = err(wa @ xg)
                cell["a8_tok"] = err(wa @ quantize_activation(xa, None))
                for fmt in ("q4_g64_fp16", "q5_g64_fp16"):
                    wq = quantize_weight(wa, fmt, importance, device)
                    cell[fmt[:2]] = err(wq @ xa)
                    if fmt.startswith("q4"):
                        cell["q4a8"] = err(wq @ xg)
                row[arm] = cell
            if suffix == "self_attn.v_proj":
                row["value_plane"] = value_plane(work, layer, x)
            results.append(row)
            print(_line(row), flush=True)
    (work / "results.json").write_text(json.dumps(results, indent=1))


def _line(row) -> str:
    parts = [f"L{row['layer']:<2} {row['matrix']:<24}"]
    for arm in ("primal", "online", "fold"):
        if arm in row:
            c = row[arm]
            parts.append(
                f"{arm}: q4 {c['q4']:.2e} q5 {c['q5']:.2e} a8 {c['a8_g64']:.2e}"
                f" a8tok {c['a8_tok']:.2e} q4a8 {c['q4a8']:.2e} crest {c['crest']:5.1f}"
            )
    if "value_plane" in row:
        v = row["value_plane"]
        parts.append(f"V int4 {v['v_int4']:.2e} rotated {v['v_int4_rotated']:.2e}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("fetch", "capture", "capture-torch", "analyze"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tokens-per-domain", type=int, default=512)
    parser.add_argument("--tokens", type=int, help="analyze only the first N captured tokens")
    parser.add_argument("--device", default="cpu", help="torch device for the weight search")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.step == "fetch":
        fetch(args.work)
    elif args.step == "capture":
        capture(args.work, args.tokens_per_domain)
    elif args.step == "capture-torch":
        capture_torch(args.work, args.tokens_per_domain)
    else:
        analyze(args.work, args.device, args.tokens)


if __name__ == "__main__":
    main()
