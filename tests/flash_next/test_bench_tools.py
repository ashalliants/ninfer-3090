"""compare_dumps and bench_summary: the arithmetic and the refusals (no model or artifact needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.flash_next import bench_summary, compare_dumps

LOG_LINE = (
    "[qwen3.8-flash-next] request: {p} prompt + {g} generated tokens; prefill {pf} s, decode {d} s "
    "({r} tok/s); expert cache 10 frames, 50.0% of 100 routed experts hit"
)


def request(prompt, rate, prefill=1.0, generated=512):
    return LOG_LINE.format(p=prompt, g=generated, pf=prefill, d=generated / rate, r=rate)


def one_hot_logits(vocab, tops, strength=20.0):
    rows = np.zeros((len(tops), vocab), dtype=np.float32)
    rows[np.arange(len(tops)), tops] = strength
    return rows


def test_identical_builds_have_zero_difference_and_zero_kl():
    ids = [1, 2, 3, 2, 1]
    rows = one_hot_logits(5, ids[1:])
    result = compare_dumps.score(ids, rows, rows.copy())
    summary = compare_dumps.summarize(result)
    assert summary["dnll"] == 0.0
    assert summary["kl_a_to_b"] == pytest.approx(0.0, abs=1e-7)
    assert summary["top1_agree"] == 1.0
    assert summary["top1_a"] == summary["top1_b"] == 1.0
    assert summary["ppl_a"] == pytest.approx(summary["ppl_b"])


def test_a_build_that_predicts_worse_has_positive_dnll_and_lower_top1():
    ids = [0, 1, 2, 3, 4, 5]
    good = one_hot_logits(8, ids[1:])
    bad = one_hot_logits(8, [(t + 1) % 8 for t in ids[1:]])  # always the wrong token
    summary = compare_dumps.summarize(compare_dumps.score(ids, good, bad))
    assert summary["dnll"] > 0
    assert summary["ppl_b"] > summary["ppl_a"]
    assert summary["top1_a"] == 1.0 and summary["top1_b"] == 0.0
    assert summary["top1_agree"] == 0.0
    assert summary["kl_a_to_b"] > 0


def test_uniform_logits_give_perplexity_equal_to_the_vocabulary_size():
    ids = [0, 1, 2, 3]
    flat = np.zeros((3, 16), dtype=np.float32)
    summary = compare_dumps.summarize(compare_dumps.score(ids, flat, flat))
    assert summary["ppl_a"] == pytest.approx(16.0)
    assert summary["dnll"] == 0.0


def test_combine_pools_positions_across_texts():
    ids = [0, 1, 2, 3]
    rows = one_hot_logits(6, ids[1:])
    one = compare_dumps.score(ids, rows, rows)
    pooled = compare_dumps.combine([one, one])
    assert len(pooled["nll_a"]) == 6
    assert pooled["top_a"] == 6


def test_extra_rows_beyond_the_predictions_are_ignored():
    ids = [0, 1, 2, 3]
    rows = one_hot_logits(6, [1, 2, 3, 4])  # one more row than there are predictions
    assert len(compare_dumps.score(ids, rows, rows)["nll_a"]) == 3


@pytest.mark.parametrize(
    ("ids", "rows_a", "rows_b", "match"),
    [
        ([1, 2], np.zeros((1, 4), np.float32), np.zeros((1, 4), np.float32), "at least three tokens"),
        ([1, 2, 3, 4], np.zeros((2, 4), np.float32), np.zeros((3, 4), np.float32), "rows for 3 predictions"),
        ([1, 2, 3, 4], np.zeros((3, 4), np.float32), np.zeros((3, 5), np.float32), "different vocabularies"),
        ([1, 2, 3, 4], np.zeros((3, 4), np.float32), np.zeros((3, 4), np.float32), "id 4 is outside"),
        ([1, -1, 2, 3], np.zeros((3, 4), np.float32), np.zeros((3, 4), np.float32), "id -1 is outside"),
    ],
)
def test_score_refuses_inputs_that_do_not_line_up(ids, rows_a, rows_b, match):
    with pytest.raises(ValueError, match=match):
        compare_dumps.score(ids, rows_a, rows_b)


def test_read_ids_ignores_whitespace_and_a_trailing_comma(tmp_path):
    path = tmp_path / "x.ids"
    path.write_text("1, 2,3,\n")
    assert compare_dumps.read_ids(path) == [1, 2, 3]


def write_dump(path, rows):
    header = np.array([rows.shape[1], rows.shape[0]], dtype="<i4").tobytes()
    path.write_bytes(header + rows.astype("<f4").tobytes())


def test_main_reports_every_set_and_writes_json(tmp_path, capsys):
    ids_dir, a_dir, b_dir = (tmp_path / n for n in ("ids", "a", "b"))
    for d in (ids_dir, a_dir, b_dir):
        d.mkdir()
    ids = [0, 1, 2, 3, 4]
    for name in ("code", "doc"):
        (ids_dir / f"s-{name}.ids").write_text(",".join(map(str, ids)))
        write_dump(a_dir / f"s-{name}.bin", one_hot_logits(8, ids[1:]))
        write_dump(b_dir / f"s-{name}.bin", one_hot_logits(8, ids[1:], strength=10.0))
    out = tmp_path / "out.json"
    code = compare_dumps.main(["--ids", str(ids_dir), "--a", str(a_dir), "--b", str(b_dir),
                               "--sets", "s", "--names", "code,doc", "--json", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "s code" in text and "s ALL" in text and "EVERYTHING" in text
    import json

    report = json.loads(out.read_text())
    assert set(report) == {"s-code", "s-doc", "s ALL", "EVERYTHING"}
    assert report["EVERYTHING"]["positions"] == 8
    assert report["EVERYTHING"]["dnll"] > 0  # B is the less confident build


def test_parse_requests_reads_the_server_log_and_ignores_other_lines():
    text = "\n".join(["starting up", request(940, 84.2, prefill=0.84), "noise", request(8000, 80.0)])
    found = bench_summary.parse_requests(text)
    assert [r[0] for r in found] == [940, 8000]
    assert found[0][2] == 0.84 and found[0][4] == 84.2


def run_log(rates_by_length, warmups=12, reps=5):
    lines = [request(10, 50.0)] + [request(10, 50.0)] * warmups
    for prompt, rates in zip((1000, 8000, 30000), rates_by_length):
        lines += [request(prompt, r, prefill=prompt / 1000.0) for r in rates]
    return "\n".join(lines)


def test_summarize_takes_medians_per_length_after_the_warmups():
    log = run_log([[80, 82, 84, 86, 100], [70] * 5, [60, 61, 62, 63, 64]])
    rows = bench_summary.summarize(bench_summary.parse_requests(log))
    assert rows["1K"]["decode"] == (84.0, 80.0, 100.0)
    assert rows["8K"]["decode"][0] == 70.0
    assert rows["30K"]["decode"][0] == 62.0
    # prefill is prompt tokens over prefill seconds: 1000 / 1.0, 8000 / 8.0, 30000 / 30.0
    assert all(math.isclose(rows[k]["prefill"][0], 1000.0) for k in rows)
    assert rows["30K"]["prompt_tokens"] == 30000


def test_summarize_refuses_a_log_that_is_not_the_whole_run():
    log = run_log([[80] * 5, [70] * 5, [60] * 5])
    truncated = "\n".join(log.splitlines()[:-3])
    with pytest.raises(ValueError, match="expected 15 measured requests"):
        bench_summary.summarize(bench_summary.parse_requests(truncated))


def test_summarize_refuses_groups_whose_prompt_lengths_are_wrong():
    swapped = run_log([[80] * 5, [70] * 5, [60] * 5]).replace("1000 prompt", "30000 prompt")
    with pytest.raises(ValueError, match="1K group should hold"):
        bench_summary.summarize(bench_summary.parse_requests(swapped))


def test_main_prints_a_row_per_length(tmp_path, capsys):
    (tmp_path / "run.err.log").write_text(run_log([[80] * 5, [70] * 5, [60] * 5]))
    assert bench_summary.main(["--dir", str(tmp_path), "run"]) == 0
    text = capsys.readouterr().out
    assert "== run: 28 requests logged" in text
    for length in ("1K", "8K", "30K"):
        assert length in text
