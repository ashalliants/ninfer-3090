from __future__ import annotations

from pathlib import Path

from tools.bench.agent_replay.metrics import SourceIndex, summarize_group
from tools.bench.agent_replay.workload import (
    COPY_CATEGORIES,
    Corpus,
    SessionBuilder,
    numbered,
    pre_image_lines,
    strip_line_numbers,
)

REPO = Path(__file__).resolve().parents[3]


def _plans(seed: int, session: int, turns: int):
    corpus = Corpus(REPO, "HEAD")
    builder = SessionBuilder(corpus, seed, session, obs_tokens=(800, 1600), max_copy_tokens=3000)
    task, opening = builder.opening()
    plans = [opening] + [builder.next_turn(turn) for turn in range(1, turns)]
    return task, plans


def test_sessions_are_deterministic_and_seed_dependent():
    task_a, plans_a = _plans(7, 0, 9)
    task_b, plans_b = _plans(7, 0, 9)
    assert task_a == task_b
    assert [(p.category, p.observation.text, p.instruction) for p in plans_a] == \
        [(p.category, p.observation.text, p.instruction) for p in plans_b]
    _, other = _plans(7, 1, 9)
    assert [p.observation.text for p in other] != [p.observation.text for p in plans_a]


def test_every_block_mixes_copy_and_non_copy_turns():
    _, plans = _plans(11, 0, 9)
    categories = [p.category for p in plans[1:]]
    assert set(COPY_CATEGORIES) <= set(categories)
    assert "explain" in categories and "explore" in categories
    for plan in plans:
        if plan.copy:
            assert plan.instruction and plan.max_tokens <= 3000


def test_numbered_reads_strip_to_file_text():
    lines = ["int main() {", "    return 0;", "}"]
    assert strip_line_numbers(numbered(lines, 0, 3)) == "\n".join(lines)


def test_pre_image_counts_context_and_removed_lines_only():
    created = "diff --git a/x b/x\nnew file mode 100644\n--- /dev/null\n+++ b/x\n@@ -0,0 +1,2 @@\n+a\n+b"
    assert pre_image_lines(created) == 0
    edited = "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n keep\n-old\n+new\n keep"
    assert pre_image_lines(edited) == 3


def test_overlap_separates_copies_from_prose():
    source = "\n".join(f"int value_{i} = compute(input_{i}, scale) + bias_{i};" for i in range(40))
    index = SourceIndex()
    index.add(numbered(source.split("\n"), 0, 40))
    covered, total = index.overlap(source.replace("value_3 ", "value_3_checked "))
    assert total > 0 and covered / total > 0.9
    covered, total = index.overlap(
        "The function computes a scaled value per input and adds a bias term to each one."
    )
    assert covered == 0


def test_summary_counts_generic_counters_and_acceptance():
    samples = [
        {"category": "rewrite_file", "error": None, "prompt_tokens": 1000, "cached_tokens": 900,
         "output_tokens": 100, "decode_s": 1.0, "decode_tps": 100.0, "ttft_s": 0.2,
         "overlap_covered": 90, "overlap_units": 100, "finish_reason": "stop",
         "counters": {"speculative.rounds": 20, "speculative.drafted_tokens": 140,
                      "speculative.accepted_tokens": 70, "speculative.ngram_accepted_tokens": 30,
                      "speculative.accepted_per_position": [10, 5]}},
        {"category": "rewrite_file", "error": None, "prompt_tokens": 2000, "cached_tokens": 1900,
         "output_tokens": 300, "decode_s": 2.0, "decode_tps": 150.0, "ttft_s": 0.4,
         "overlap_covered": 10, "overlap_units": 100, "finish_reason": "length",
         "counters": {"speculative.rounds": 30, "speculative.drafted_tokens": 210,
                      "speculative.accepted_tokens": 70,
                      "speculative.accepted_per_position": [1, 2, 3]}},
    ]
    group = summarize_group(samples)
    assert group["decode_tps"] == 400 / 3.0
    assert group["copy_overlap"] == 0.5
    assert group["spec_acceptance"] == 140 / 350
    assert group["spec_tokens_per_round"] == (140 + 50) / 50
    assert group["counters"]["speculative.accepted_per_position"] == [11, 7, 3]
    assert group["speculative.ngram_accepted_tokens.share_of_output"] == 30 / 400
    assert group["length_capped_turns"] == 1


def _timing_sample(**timings):
    return {"server_timings": timings, "client_ttft_s": 0.9, "client_total_s": 2.0,
            "output_tokens": 100}


def test_usage_prompt_ms_is_not_reported_as_ttft():
    from tools.bench.agent_replay.run import apply_timing

    sample = _timing_sample(prompt_ms=300.0, predicted_ms=1000.0)
    apply_timing(sample, None)
    assert sample["ttft_s"] == 0.9
    assert sample["prompt_s"] == 0.3
    assert sample["decode_s"] == 1.0
    apply_timing(sample, {"timings_seconds": {"ttft": 0.5, "total": 1.5}})
    assert sample["ttft_s"] == 0.5 and sample["timing_source"] == "request_log"


def test_run_rejects_settings_that_measure_nothing(tmp_path):
    from tools.bench.agent_replay.run import RunConfig, run

    base = dict(base_url="http://127.0.0.1:1", api_key=None, repo=REPO, commit="HEAD", seed=0,
                sessions=1, concurrency=1, min_context=1, max_context=2, obs_tokens=(1, 2),
                max_copy_tokens=1, max_turns=1, timeout_seconds=1.0, request_log=None,
                out=tmp_path)
    for bad in ({"sessions": 0}, {"max_turns": 0}):
        assert run(RunConfig(**{**base, **bad})) == 2
