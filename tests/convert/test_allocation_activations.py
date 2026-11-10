from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)

from tools.convert.allocation import plan
from tools.convert.allocation_activations import (
    cross_entropy,
    layer_projections,
    measure_layer,
)
from tools.convert.official_recipes import Q4, Q5

from .test_qwen3_5 import _config


def _text_config():
    config = Qwen3_5TextConfig(**_config()["text_config"])
    config._attn_implementation = "sdpa"
    return config


def _run(kind_index):
    torch.manual_seed(0)
    config = _text_config()
    layer = Qwen3_5DecoderLayer(config, kind_index).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
    hidden = torch.randn(2, 12, config.hidden_size)
    position_ids = torch.arange(12).view(1, 1, -1).expand(4, 2, -1)
    embeddings = Qwen3_5TextRotaryEmbedding(config)(hidden, position_ids[1:])

    def forward(h):
        return layer(h, position_embeddings=embeddings, attention_mask=None, position_ids=position_ids[0])

    return config, layer, hidden, forward


@pytest.mark.parametrize("index,role", [(0, "gdn/output"), (1, "attention/output")])
def test_measure_layer_scores_each_projection_on_its_real_input(index, role):
    config, layer, hidden, forward = _run(index)
    kind = config.layer_types[index]
    assert set(layer_projections(layer, kind)) == {role, "mlp/down"}

    output, result = measure_layer(layer, kind, hidden, forward, {})
    with torch.no_grad():
        assert torch.allclose(output, forward(hidden))
    # No hook is left behind to capture or slow later calls.
    for module in layer_projections(layer, kind).values():
        assert not module._forward_pre_hooks

    assert set(result) == {role, "mlp/down"}
    for entry in result.values():
        assert entry["residual_energy"] == pytest.approx(float(output.pow(2).sum(-1).mean()))
        assert entry["output_energy"] > 0
        assert 0 < entry["output_error"][Q5] < entry["output_error"][Q4]


def test_measure_layer_error_matches_a_direct_computation():
    from tools.convert.allocation import dequantize

    config, layer, hidden, forward = _run(0)
    captured = {}
    handle = layer.mlp.down_proj.register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("x", args[0].detach().reshape(-1, args[0].shape[-1]))
    )
    with torch.no_grad():
        forward(hidden)
    handle.remove()
    weight = layer.mlp.down_proj.weight.detach()
    decoded = dequantize(weight, Q4, None, device="cpu")
    expected = ((captured["x"] @ (weight - decoded).T) ** 2).sum(-1).mean().item()
    _, result = measure_layer(layer, config.layer_types[0], hidden, forward, {})
    assert result["mlp/down"]["output_error"][Q4] == pytest.approx(expected, rel=1e-5)


def test_cross_entropy_matches_torch_with_a_vocabulary_that_is_not_a_chunk_multiple():
    torch.manual_seed(1)
    batch, length, width, vocab = 2, 7, 5, 20000
    hidden = torch.randn(batch, length, width)
    tokens = torch.randint(0, vocab, (batch, length))
    head = torch.randn(vocab, width)
    value = cross_entropy(hidden, tokens, lambda h: h, lambda a, b: head[a:b])
    logits = hidden[:, :-1].reshape(-1, width) @ head.T
    expected = torch.nn.functional.cross_entropy(logits, tokens[:, 1:].reshape(-1)).item()
    assert value == pytest.approx(expected, rel=1e-5)


def test_activation_metric_uses_residual_relative_error_and_requires_scores():
    def entry(q4, q5, residual):
        return {
            "bytes": {Q4: 100, Q5: 125},
            "error": {Q4: 1.0, Q5: 0.25},
            "energy": 1.0,
            "activation": {"output_error": {Q4: q4, Q5: q5}, "residual_energy": residual},
        }

    # Layer 1 has the same absolute noise but a ten times larger residual stream, so it
    # gains less from extra bits; with budget for one upgrade layer 0 gets it.
    scores = {"text/layers/0/mlp/down": entry(4.0, 1.0, 10.0), "text/layers/1/mlp/down": entry(4.0, 1.0, 100.0)}
    result = plan(scores, metric="activation", late_layer=0, budget_bytes=225)
    assert result["proposed"] == {"text/layers/0/mlp/down": Q5, "text/layers/1/mlp/down": Q4}
    del scores["text/layers/0/mlp/down"]["activation"]
    with pytest.raises(ValueError, match="--activations"):
        plan(scores, metric="activation")
