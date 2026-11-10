from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import pytest
import torch

from tools.convert import allocation
from tools.convert.allocation import (
    Option,
    allocate,
    apply_assignments,
    flexible_parameters,
    format_override,
    plan,
    score_parameter,
    weighted_error,
)
from tools.convert.methods import grouped_search
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    Q4,
    Q5,
    qwen3_8_27b,
    qwen3_8_27b_text_format,
)
from tools.convert.quantization.groupwise import search_quantize_matrix
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

from .test_qwen3_5 import _checkpoint, _config


def _parameter(name, value):
    return Parameter(name, tuple(value.shape), array_source(value, name), inputs=("input",))


def _model(values):
    model = Model({"text": {"config": {}}})
    for name, value in values.items():
        model.add(_parameter(name, value))
    return model


def _options(costs):
    # costs: per unit, a list of (bytes, cost) in the order of formats a, b, c.
    return {
        name: [Option("abc"[i], b, c) for i, (b, c) in enumerate(items)]
        for name, items in costs.items()
    }


def test_allocate_matches_brute_force_when_the_budget_lands_on_a_step():
    generator = torch.Generator().manual_seed(11)
    for _ in range(25):
        costs = {}
        for unit in range(5):
            sizes = sorted(torch.randint(1, 20, (3,), generator=generator).tolist())
            errors = sorted(torch.rand(3, generator=generator).tolist(), reverse=True)
            costs[unit] = list(zip(sizes, errors))
        options = _options(costs)
        budget = sum(items[0][0] for items in costs.values()) + 17
        chosen = allocate(options, budget)
        spent = sum(costs[u]["abc".index(f)][0] for u, f in chosen.items())
        total = sum(costs[u]["abc".index(f)][1] for u, f in chosen.items())
        assert spent <= budget
        best = min(
            sum(costs[u][i][1] for u, i in enumerate(picks))
            for picks in product(range(3), repeat=5)
            if sum(costs[u][i][0] for u, i in enumerate(picks)) <= budget
        )
        # Greedy on the convex hull is within one upgrade of the exact optimum.
        assert total >= best - 1e-12
        worst_step = max(c0 - c1 for items in costs.values() for (_, c0), (_, c1) in zip(items, items[1:]))
        assert total <= best + worst_step + 1e-12


def test_allocate_spends_the_budget_on_the_steepest_upgrade_and_ignores_dominated_options():
    options = {
        "flat": [Option("q4", 10, 1.0), Option("q5", 12, 0.9)],
        "steep": [Option("q4", 10, 1.0), Option("q5", 12, 0.2)],
        "bad": [Option("q4", 10, 1.0), Option("q5", 14, 1.5)],  # dearer and worse
    }
    assert allocate(options, 32) == {"flat": "q4", "steep": "q5", "bad": "q4"}
    assert allocate(options, 30) == {"flat": "q4", "steep": "q4", "bad": "q4"}
    with pytest.raises(ValueError, match="below the cheapest"):
        allocate(options, 29)


def test_weighted_error_matches_the_search_encoder_and_energy_uses_the_weights():
    generator = torch.Generator().manual_seed(5)
    weight = (torch.randn((6, 100), generator=generator) * 0.05).to(torch.bfloat16)
    importance = torch.rand(100, generator=generator) * 3
    error, energy = weighted_error(weight, Q4, importance, device="cpu")

    encoded = search_quantize_matrix(weight, Q4, importance=importance, negative_scales=True, device="cpu")
    decoded = (encoded.scales.float().unsqueeze(-1) * encoded.codes.float()).reshape(6, -1)[:, :100]
    expected = (importance * (weight.float() - decoded) ** 2).sum().item()
    assert error == pytest.approx(expected, rel=1e-6)
    assert energy == pytest.approx((importance * weight.float() ** 2).sum().item(), rel=1e-6)

    q5_error, _ = weighted_error(weight, Q5, importance, device="cpu")
    assert 0 < q5_error < error


def test_score_parameter_chunks_agree_with_one_pass_and_report_bytes():
    generator = torch.Generator().manual_seed(8)
    value = (torch.randn((10, 128), generator=generator) * 0.05).to(torch.bfloat16)
    importance = torch.rand(128, generator=generator)
    parameter = _parameter("text/layers/0/mlp/down", value)
    whole = score_parameter(parameter, importance, rows_per_chunk=10)
    chunked = score_parameter(parameter, importance, rows_per_chunk=3)
    for format in (Q4, Q5):
        assert chunked["error"][format] == pytest.approx(whole["error"][format], rel=1e-5)
    assert chunked["energy"] == pytest.approx(whole["energy"], rel=1e-5)
    assert whole["bytes"][Q5] > whole["bytes"][Q4] > 0 and whole["weighted"]
    with pytest.raises(ValueError, match="imatrix vector length"):
        score_parameter(parameter, torch.ones(64))


def test_plan_moves_bits_to_the_projection_that_gains_most_at_the_official_byte_count():
    generator = torch.Generator().manual_seed(2)
    smooth = (torch.rand((16, 128), generator=generator) * 0.1 - 0.05).to(torch.bfloat16)
    heavy = torch.randn((16, 128), generator=generator).to(torch.bfloat16) ** 3 * 0.05
    scores = {
        "text/layers/0/mlp/down": score_parameter(_parameter("text/layers/0/mlp/down", smooth), None),
        "text/layers/1/mlp/down": score_parameter(_parameter("text/layers/1/mlp/down", heavy), None),
    }
    # With the late layer at 1, the official rule gives layer 0 Q5 and layer 1 Q4.
    result = plan(scores, late_layer=1)
    assert result["baseline"] == {"text/layers/0/mlp/down": Q5, "text/layers/1/mlp/down": Q4}
    assert result["totals"]["proposed"]["bytes"] <= result["totals"]["baseline"]["bytes"]
    assert result["totals"]["proposed"]["cost"] <= result["totals"]["baseline"]["cost"]
    for metric in ("relative", "absolute"):
        assert plan(scores, late_layer=1, metric=metric)["proposed"]["text/layers/1/mlp/down"] in (Q4, Q5)
    with pytest.raises(ValueError, match="metric"):
        plan(scores, metric="kld")


def test_official_rule_helper_matches_the_recipe(tmp_path):
    names = [
        f"text/layers/{layer}/{role}"
        for layer in (0, 35, 36, 63)
        for role in (
            "attention/query", "attention/key", "attention/gate", "attention/value",
            "attention/output", "gdn/query", "gdn/key", "gdn/value", "gdn/z", "gdn/output",
            "mlp/gate", "mlp/up", "mlp/down",
        )
    ]
    values = {n: torch.zeros((2, 64), dtype=torch.bfloat16) for n in names}
    values["text/token_embedding"] = torch.zeros((2, 64), dtype=torch.bfloat16)
    values["text/output_head"] = torch.zeros((2, 64), dtype=torch.bfloat16)
    recipe = Recipe(_model(values))
    qwen3_8_27b(recipe.model, recipe, {"imatrix": SimpleNamespace(path="x")})
    for name in names:
        assert recipe.selections[name][0].format == qwen3_8_27b_text_format(name)
        assert recipe.selections[name][0].method is grouped_search


def test_flexible_parameters_select_the_three_roles_of_a_dense_model(tmp_path):
    with _checkpoint(tmp_path / "source", _config(), {}) as source:
        from tools.convert.qwen3_5 import build_model

        model = build_model(source, components=("text",))
        assert flexible_parameters(model) == [
            "text/layers/0/gdn/output",
            "text/layers/0/mlp/down",
            "text/layers/1/attention/output",
            "text/layers/1/mlp/down",
        ]
        assert flexible_parameters(model, range(1, 2)) == [
            "text/layers/1/attention/output",
            "text/layers/1/mlp/down",
        ]
    with _checkpoint(tmp_path / "moe", _config(moe=True), {}) as source:
        from tools.convert.qwen3_5 import build_model

        with pytest.raises(ValueError, match="Dense"):
            flexible_parameters(build_model(source, components=("text",)))


def test_emitted_override_assigns_the_proposed_formats_with_the_imatrix_search(tmp_path):
    values = {
        "text/layers/0/mlp/down": torch.zeros((2, 64), dtype=torch.bfloat16),
        "text/layers/1/mlp/down": torch.zeros((2, 64), dtype=torch.bfloat16),
    }
    proposed = {"text/layers/0/mlp/down": Q4, "text/layers/1/mlp/down": Q5}
    path = tmp_path / "override.py"
    path.write_text(format_override(proposed, "report.json"))

    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert namespace["ASSIGNMENTS"] == proposed
    recipe = Recipe(_model(values))
    namespace["configure"](recipe.model, recipe, {"imatrix": SimpleNamespace(path="m.safetensors")})
    for name, format in proposed.items():
        selection = recipe.selections[name][0]
        assert selection.format == format
        assert selection.method is grouped_search
        assert selection.parameters == {"imatrix": "m.safetensors", "negative_scales": True}
    assert apply_assignments is allocation.apply_assignments
