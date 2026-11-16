from __future__ import annotations

from itertools import product
import sys
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
    qwen3_8_27b_legacy_text_format,
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


CANDIDATES = ("activation",)  # measured, not adopted; the measured "absolute" allocation is the shipped rule


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_committed_candidate_covers_the_27b_projections_and_keeps_the_legacy_byte_budget(candidate):
    from pathlib import Path

    from tools.convert.allocation import payload_bytes, qwen3_8_27b_flexible_names

    # Qwen3.8-27B: 64 layers, full attention at every fourth layer, hidden 5120, MLP 17408,
    # attention output K = 24 heads x 256, GDN output K = 48 value heads x 128.
    path = Path(allocation.__file__).parent / "candidates" / f"qwen3_8_27b_{candidate}.py"
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assignments = namespace["ASSIGNMENTS"]
    names = qwen3_8_27b_flexible_names()
    assert set(assignments) == set(names)
    assert set(assignments.values()) <= {Q4, Q5}
    shape = lambda name: (5120, 17408) if name.endswith("/mlp/down") else (5120, 6144)  # noqa: E731
    chosen = sum(payload_bytes(f, shape(n)) for n, f in assignments.items())
    legacy = sum(payload_bytes(qwen3_8_27b_legacy_text_format(n), shape(n)) for n in names)
    assert chosen <= legacy


def test_committed_gguf_plans_match_their_assignments_and_the_harness_baseline():
    import importlib.util
    from pathlib import Path

    from tools.convert.allocation import gguf_plan, qwen3_8_27b_flexible_names

    folder = Path(allocation.__file__).parent / "candidates"
    spec = importlib.util.spec_from_file_location(
        "gguf_eval", Path(allocation.__file__).parents[2] / "tools" / "eval" / "gguf_eval.py"
    )
    harness = importlib.util.module_from_spec(spec)
    sys.modules["gguf_eval"] = harness  # its dataclasses look their module up by name
    try:
        spec.loader.exec_module(harness)
    finally:
        del sys.modules["gguf_eval"]

    names = qwen3_8_27b_flexible_names()
    cases = [
        ("qwen3_8_27b_official.gguf-plan.txt", {n: qwen3_8_27b_text_format(n) for n in names}),
        ("qwen3_8_27b_legacy.gguf-plan.txt", {n: qwen3_8_27b_legacy_text_format(n) for n in names}),
    ]
    for candidate in CANDIDATES:
        namespace = {}
        path = folder / f"qwen3_8_27b_{candidate}.py"
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        cases.append((f"qwen3_8_27b_{candidate}.gguf-plan.txt", namespace["ASSIGNMENTS"]))
    for file, assignments in cases:
        text = (folder / file).read_text().strip()
        assert text == gguf_plan(assignments), f"regenerate {file} (see candidates/README.md)"
        # The harness resolves every flexible tensor to exactly the assigned format.
        plan = harness.parse_plan(text)
        for name, format in assignments.items():
            _, _, layer, role = name.split("/", 3)
            assert harness.role_format(plan, int(layer), role) == format
        assert harness.role_format(plan, None, "text/token_embedding") == Q4
        assert harness.role_format(plan, None, "text/output_head") == "q6_g64_fp16"

    # Without a plan entry the harness uses its own role defaults: they must already be the
    # official recipe's formats for every role this tool does not allocate.
    allocated = {"attention/output", "gdn/output", "mlp/down", "text/token_embedding", "text/output_head"}
    for role, format in harness.RECIPE.items():
        if role not in allocated:
            assert format == qwen3_8_27b_text_format(f"text/layers/0/{role}"), role


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


def test_shipped_rule_is_the_measured_allocation_at_the_legacy_byte_budget():
    from tools.convert.allocation import payload_bytes, qwen3_8_27b_flexible_names

    names = qwen3_8_27b_flexible_names()
    q5_layers = {"attention/output": [], "gdn/output": [], "mlp/down": []}
    for name in names:
        _, _, layer, role = name.split("/", 3)
        if qwen3_8_27b_text_format(name) == Q5:
            q5_layers[role].append(int(layer))
    # The allocation measured on the RTX 3090 (docs/maintainer/quality-trade-experiments.md,
    # "alloc-absolute"): Q5 from layer 23 / 24 / 30 on, Q4 before it.
    assert q5_layers["attention/output"] == [23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    assert q5_layers["gdn/output"] == [
        layer for layer in range(24, 63) if layer % 4 != 3
    ]
    assert q5_layers["mlp/down"] == list(range(30, 64))
    assert sum(map(len, q5_layers.values())) == 75
    shape = lambda name: (5120, 17408) if name.endswith("/mlp/down") else (5120, 6144)  # noqa: E731
    shipped = sum(payload_bytes(qwen3_8_27b_text_format(n), shape(n)) for n in names)
    legacy = sum(payload_bytes(qwen3_8_27b_legacy_text_format(n), shape(n)) for n in names)
    assert shipped <= legacy
    # Roles outside the three keep the formats they always had.
    for role, expected in (("attention/query", Q4), ("mlp/gate", Q4), ("gdn/z", Q5), ("attention/value", Q5)):
        assert qwen3_8_27b_text_format(f"text/layers/40/{role}") == expected


def test_input_identity_changes_with_content_and_never_with_the_file_name(tmp_path):
    from tools.convert.allocation import checkpoint_identity, file_sha256, require_same_inputs

    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"imatrix v1")
    second.write_bytes(b"imatrix v1")
    assert file_sha256(first) == file_sha256(second)
    second.write_bytes(b"imatrix v2")
    assert file_sha256(first) != file_sha256(second)

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (checkpoint / "model-00001.safetensors").write_bytes(b"x" * 10)
    before = checkpoint_identity(checkpoint)
    assert checkpoint_identity(checkpoint) == before
    (checkpoint / "model-00001.safetensors").write_bytes(b"x" * 11)        # a re-exported shard
    assert checkpoint_identity(checkpoint) != before
    (checkpoint / "model-00001.safetensors").write_bytes(b"x" * 10)
    (checkpoint / "config.json").write_text('{"hidden_size": 1}')            # another config
    assert checkpoint_identity(checkpoint) != before

    require_same_inputs({"a": 1, "b": 2}, {"a": 1, "b": 2}, ("a", "b"), tmp_path / "r.json")
    with pytest.raises(ValueError, match=r"different inputs \(b\)"):
        require_same_inputs({"a": 1, "b": 2}, {"a": 1, "b": 3}, ("a", "b"), tmp_path / "r.json")
    with pytest.raises(ValueError, match="different inputs"):
        require_same_inputs({}, {"a": 1}, ("a",), tmp_path / "r.json")        # an old report with no record


def test_atomic_write_keeps_the_old_file_when_the_replace_fails(tmp_path, monkeypatch):
    from tools.convert.allocation import atomic_write_text

    path = tmp_path / "report.json"
    atomic_write_text(path, "old")
    assert path.read_text() == "old" and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        atomic_write_text(path, "new")
    assert path.read_text() == "old"


def test_activation_resume_only_trusts_layers_the_report_has_committed(tmp_path, monkeypatch):
    import json

    from tools.convert import allocation_activations as activations

    settings = {"seq_len": 4, "checkpoint": "c1", "imatrix_sha256": "i1", "tokens_sha256": "t1"}
    report_path, work = tmp_path / "report.json", tmp_path / "work"
    work.mkdir()

    report, start, saved = activations.resume_state(report_path, settings, work)
    assert (start, saved, report["scores"]) == (0, None, {})

    for index in range(3):
        report["scores"][f"layer{index}"] = {"v": index}
        activations.commit_layer(report, index, torch.full((2,), float(index)), report_path, work)
    assert sorted(path.name for path in work.iterdir()) == ["hidden_2.pt"]       # earlier ones are gone
    report, start, saved = activations.resume_state(report_path, settings, work)
    assert start == 3 and torch.equal(torch.load(saved), torch.full((2,), 2.0))
    assert sorted(report["scores"]) == ["layer0", "layer1", "layer2"]

    # Interrupted after the hidden state was saved but before the report named it: layer 3 is not
    # done, whatever file is lying around.
    torch.save(torch.zeros(2), work / "hidden_3.pt")
    _, start, saved = activations.resume_state(report_path, settings, work)
    assert start == 3 and saved.name == "hidden_2.pt"

    # Interrupted while the report is written: the old report and the hidden state it names survive.
    report["scores"]["layer3"] = {"v": 3}
    monkeypatch.setattr(activations, "atomic_write_text", lambda *a: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        activations.commit_layer(report, 3, torch.ones(2), report_path, work)
    monkeypatch.undo()
    assert (work / "hidden_2.pt").exists()
    reloaded, start, _ = activations.resume_state(report_path, settings, work)
    assert start == 3 and "layer3" not in reloaded["scores"]

    # A report from other inputs, or whose hidden state has gone, is refused rather than merged.
    for key in settings:
        with pytest.raises(ValueError, match="different inputs"):
            activations.resume_state(report_path, {**settings, key: "other"}, work)
    (work / "hidden_2.pt").unlink()
    with pytest.raises(ValueError, match="is missing"):
        activations.resume_state(report_path, settings, work)
    assert json.loads(report_path.read_text())["completed_layers"] == 3


def test_main_refuses_a_report_or_activation_scores_made_from_other_inputs(tmp_path, monkeypatch):
    import json

    from safetensors.torch import save_file

    from tools.convert.allocation import checkpoint_identity, file_sha256

    entry = {"shape": [2, 64], "weighted": False, "energy": 1.0,
             "error": {Q4: 0.1, Q5: 0.01}, "bytes": {Q4: 1, Q5: 2}}
    scored = []
    monkeypatch.setattr(allocation, "score_parameter", lambda *a, **k: scored.append(1) or dict(entry))
    imatrix = tmp_path / "imatrix.safetensors"
    save_file({"text/layers/0/mlp/down": torch.ones(64)}, str(imatrix))
    out = tmp_path / "report.json"
    with _checkpoint(tmp_path / "source", _config(), {}):
        model_dir = tmp_path / "source"
        base = ["--model", str(model_dir), "--imatrix", str(imatrix), "--out", str(out)]
        allocation.main(base)
        assert len(scored) == 4 and json.loads(out.read_text())["settings"]["checkpoint"] == checkpoint_identity(model_dir)

        scored.clear()
        allocation.main(base)                       # same inputs: resumes, scores nothing new
        assert scored == []

        # A regenerated imatrix with the same name is a different input; the old scores are not kept.
        save_file({"text/layers/0/mlp/down": torch.full((64,), 2.0)}, str(imatrix))
        with pytest.raises(ValueError, match=r"different inputs \(imatrix_sha256\)"):
            allocation.main(base)
        assert scored == []
        save_file({"text/layers/0/mlp/down": torch.ones(64)}, str(imatrix))

        # Activation scores from another checkpoint or imatrix are not merged.
        activations = tmp_path / "activations.json"
        good = {"imatrix_sha256": file_sha256(imatrix), "checkpoint": checkpoint_identity(model_dir)}
        for key in good:
            activations.write_text(json.dumps({"settings": {**good, key: "other"}, "scores": {}}))
            with pytest.raises(ValueError, match=f"different inputs \\({key}\\)"):
                allocation.main([*base[:-2], "--out", str(tmp_path / f"{key}.json"), "--activations", str(activations)])
        activations.write_text(json.dumps({"settings": good, "scores": {}}))
        allocation.main([*base[:-2], "--out", str(tmp_path / "ok.json"), "--activations", str(activations)])
        assert json.loads((tmp_path / "ok.json").read_text())["plan"]["metric"] == "relative"
