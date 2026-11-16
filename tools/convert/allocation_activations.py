"""Measure Q4/Q5 sensitivity of the allocatable projections on real activations, on CPU.

``tools.convert.allocation`` scores rounding error from the imatrix alone. This tool runs the
checkpoint itself, one decoder layer at a time in FP32 (a layer is under 2 GB, so a 54 GB model
fits a 32 GB machine), on held-out text. For each attention-output, GDN-output and MLP-down
projection it captures the real input ``x`` and records, for Q4 and for Q5,

    output_error = mean over tokens of || x (W - W')^T ||^2     (W' = the encoder's round trip)

together with ``residual_energy``, the mean squared norm of that layer's output hidden state.
``output_error / residual_energy`` is the noise a format adds to the residual stream relative to
the stream's own size. That uses the true activation covariance instead of the imatrix's diagonal
and removes the depth growth of activation scale that makes the absolute metric favour late layers
by construction. It still ignores how later layers amplify or absorb the noise, so it remains a
proxy for KLD.

    python -m tools.convert.allocation_activations --model /path/to/Qwen3.8-27B \
        --imatrix qwen3_8_27b.imatrix.safetensors --corpus eval/corpora/perplexity-heldout-2026-09 \
        --out activations.json --work-dir activations-work

Then pass ``--activations activations.json --metric activation`` to ``tools.convert.allocation``.

The reference forward is checked end to end: after the last layer the tool reports the
cross-entropy of the held-out tokens (a sound pass lands near the model's perplexity, 4.3). The
hidden state is saved after every layer in ``--work-dir`` so an interrupted run resumes.
Requires ``transformers`` (not otherwise a dependency of the converter).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import torch

from . import imatrix as imatrix_file
from .allocation import FORMATS, dequantize

STREAMS = ("wikipedia_en", "wikipedia_zh", "github_code", "own_code", "synthetic_chat", "arxiv")


def layer_projections(layer, kind: str) -> dict[str, torch.nn.Linear]:
    """The allocatable projections of one HF decoder layer, keyed by logical role."""

    mixer = {"role": "gdn/output", "module": getattr(layer, "linear_attn", None), "name": "out_proj"}
    if kind == "full_attention":
        mixer = {"role": "attention/output", "module": layer.self_attn, "name": "o_proj"}
    return {
        mixer["role"]: getattr(mixer["module"], mixer["name"]),
        "mlp/down": layer.mlp.down_proj,
    }


def measure_layer(
    layer,
    kind: str,
    hidden: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    importance: Mapping[str, torch.Tensor | None],
    *,
    formats=FORMATS,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """Run one layer on ``hidden`` and score each projection's rounding noise on its real input.

    ``forward`` maps the layer input to its output (it supplies masks and rotary embeddings);
    ``importance`` maps a role to the imatrix vector (or None) used by the encoder.
    """

    projections = layer_projections(layer, kind)
    captured: dict[str, torch.Tensor] = {}
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args, role=role: captured.__setitem__(
                role, args[0].detach().reshape(-1, args[0].shape[-1])
            )
        )
        for role, module in projections.items()
    ]
    try:
        with torch.no_grad():
            output = forward(hidden)
    finally:
        for handle in handles:
            handle.remove()
    residual_energy = float(output.pow(2).sum(-1).mean())
    result = {}
    for role, module in projections.items():
        weight = module.weight.detach().to(torch.float32)
        x = captured[role]
        errors = {}
        for format in formats:
            decoded = dequantize(weight, format, importance.get(role), device=device)
            delta = (weight - decoded.to(weight.device)).T
            total = 0.0
            for begin in range(0, x.shape[0], 1024):
                total += float((x[begin : begin + 1024] @ delta).pow(2).sum())
            errors[format] = total / x.shape[0]
        result[role] = {
            "output_error": errors,
            "residual_energy": residual_energy,
            "output_energy": float((x @ weight.T).pow(2).sum(-1).mean()),
        }
    return output, result


def read_tokens(corpus: Path, tokenizer, seq_len: int, streams=STREAMS) -> torch.Tensor:
    """One ``seq_len`` window of held-out text per stream, as a [streams, seq_len] int tensor."""

    windows = []
    for stream in streams:
        text = (corpus / "data" / stream / "00.txt").read_text()
        ids = tokenizer.encode(text).ids
        if len(ids) < seq_len:
            raise ValueError(f"{stream}: only {len(ids)} tokens for a {seq_len}-token window")
        windows.append(torch.tensor(ids[:seq_len], dtype=torch.long))
    return torch.stack(windows)


def cross_entropy(hidden, tokens, norm: Callable[[torch.Tensor], torch.Tensor], head_rows: Callable[[int, int], torch.Tensor]):
    """Mean next-token cross-entropy of ``tokens`` given the final hidden states, vocabulary-chunked.

    ``norm`` is the model's final RMSNorm; ``head_rows(a, b)`` returns output-head rows a..b.
    """

    with torch.no_grad():
        h = norm(hidden.to(torch.float32))
    h = h[:, :-1].reshape(-1, h.shape[-1])
    target = tokens[:, 1:].reshape(-1)
    lse = None
    picked = torch.zeros(h.shape[0])
    vocab = 0
    while True:
        rows = head_rows(vocab, vocab + 8192)
        if rows.shape[0] == 0:
            break
        logits = h @ rows.to(torch.float32).T
        chunk = torch.logsumexp(logits, dim=-1)
        lse = chunk if lse is None else torch.logaddexp(lse, chunk)
        inside = (target >= vocab) & (target < vocab + rows.shape[0])
        picked[inside] = logits[inside, target[inside] - vocab]
        vocab += rows.shape[0]
    return float((lse - picked).mean())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="held-out corpus directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="per-layer hidden state, for resuming")
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--layers", type=int, help="stop after this many layers (smoke test)")
    parser.add_argument("--device", default="cpu", help="device for the quantizer search only")
    args = parser.parse_args(argv)

    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
    )

    config_json = json.loads((args.model / "config.json").read_text())
    config = Qwen3_5TextConfig(**config_json.get("text_config", config_json))
    config._attn_implementation = "sdpa"
    weight_map = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model."
    handles: dict[str, object] = {}

    def tensor(name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(args.model / shard), framework="pt")
        if rows is None:
            return handles[shard].get_tensor(name)
        return handles[shard].get_slice(name)[rows[0] : rows[1]]

    vectors = imatrix_file.load(str(args.imatrix))
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokens = read_tokens(args.corpus, tokenizer, args.seq_len)
    batch, length = tokens.shape
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Embedding rows for the tokens actually used, not the 2.5 GB table.
    embed = torch.zeros(batch, length, config.hidden_size)
    unique = torch.unique(tokens).tolist()
    rows = {token: tensor(prefix + "embed_tokens.weight", (token, token + 1)).to(torch.float32) for token in unique}
    for b in range(batch):
        for t in range(length):
            embed[b, t] = rows[int(tokens[b, t])][0]

    position_ids = torch.arange(length).view(1, 1, -1).expand(4, batch, -1)
    text_position_ids, mrope_positions = position_ids[0], position_ids[1:]
    rotary = Qwen3_5TextRotaryEmbedding(config)
    position_embeddings = rotary(embed, mrope_positions)

    report = {"settings": {"seq_len": length, "streams": list(STREAMS)}, "scores": {}}
    start, hidden = 0, embed
    if args.out.exists():
        report = json.loads(args.out.read_text())
    done = sorted(int(path.stem.split("_")[1]) for path in args.work_dir.glob("hidden_*.pt"))
    if done and (args.work_dir / f"hidden_{done[-1]}.pt").exists():
        start = done[-1] + 1
        hidden = torch.load(args.work_dir / f"hidden_{done[-1]}.pt")
        print(f"resuming at layer {start}", flush=True)

    layers = config.num_hidden_layers if args.layers is None else min(args.layers, config.num_hidden_layers)
    for index in range(start, layers):
        kind = config.layer_types[index]
        layer = Qwen3_5DecoderLayer(config, index)
        layer_prefix = f"{prefix}layers.{index}."
        state = {
            name[len(layer_prefix):]: tensor(name).to(torch.float32)
            for name in weight_map
            if name.startswith(layer_prefix)
        }
        layer.load_state_dict(state, strict=True)
        layer.eval()
        role_names = {role: f"text/layers/{index}/{role}" for role in layer_projections(layer, kind)}
        importance = {role: vectors.get(name) for role, name in role_names.items()}

        def forward(h):
            return layer(h, position_embeddings=position_embeddings, attention_mask=None,
                         position_ids=text_position_ids)

        hidden, result = measure_layer(layer, kind, hidden, forward, importance, device=args.device)
        for role, entry in result.items():
            report["scores"][role_names[role]] = entry
        torch.save(hidden, args.work_dir / f"hidden_{index}.pt")
        if index > start:
            (args.work_dir / f"hidden_{index - 1}.pt").unlink(missing_ok=True)
        args.out.write_text(json.dumps(report))
        worst = max(entry["output_error"][FORMATS[0]] / entry["residual_energy"] for entry in result.values())
        print(f"[{index + 1}/{config.num_hidden_layers}] {kind}: finite={bool(torch.isfinite(hidden).all())} "
              f"|h|^2={float(hidden.pow(2).sum(-1).mean()):.4g} worst Q4 noise/residual {worst:.3g}", flush=True)

    if layers == config.num_hidden_layers:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": tensor(prefix + "norm.weight").to(torch.float32)})
        loss = cross_entropy(
            hidden, tokens, norm,
            lambda a, b: tensor("lm_head.weight", (a, b)) if a < config.vocab_size else torch.zeros(0, 1),
        )
        report["reference_check"] = {"cross_entropy": loss, "perplexity": math.exp(loss), "tokens": int(batch * (length - 1))}
        args.out.write_text(json.dumps(report))
        print(f"reference check: cross-entropy {loss:.4f} (perplexity {math.exp(loss):.3f})", flush=True)


if __name__ == "__main__":
    main()
