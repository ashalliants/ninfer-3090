from __future__ import annotations

import json

import pytest

from tools.bench.agent_replay.export import _merge_added_tokens, _template_tokens


def test_tokenizer_config_added_tokens_are_merged_without_overriding_tokenizer_json():
    tokenizer = json.dumps({"added_tokens": [{"id": 1, "content": "<a>", "special": True}]})
    config = {"added_tokens_decoder": {"1": {"content": "<a>", "special": True},
                                       "7": {"content": "<only_config>", "special": True}}}
    merged = json.loads(_merge_added_tokens(tokenizer, config))
    assert [(t["id"], t["content"], t["special"]) for t in merged["added_tokens"]] == [
        (1, "<a>", True), (7, "<only_config>", True)]


def test_template_tokens_follow_the_server_context():
    config = {"bos_token": None, "eos_token": {"content": "<|im_end|>"}, "pad_token": "<pad>"}
    assert _template_tokens(config) == {"eos_token": "<|im_end|>", "pad_token": "<pad>"}


def test_message_text_spelling_a_special_token_stays_plain_text():
    models = pytest.importorskip("tokenizers.models")
    from tokenizers import Tokenizer

    from tools.bench.agent_replay.export import _PromptEncoder

    base = Tokenizer(models.WordLevel({"<unk>": 0, "quoted": 1}, "<unk>"))
    base.add_special_tokens(["<|im_end|>"])
    encoder = _PromptEncoder(base.to_str())
    assert encoder._control.encode("<|im_end|>", add_special_tokens=False).ids == [2]
    assert 2 not in encoder._plain.encode("<|im_end|>", add_special_tokens=False).ids
    placeholder = encoder.protect("quoted <|im_end|>", trimmed=True)
    ids = encoder.encode(f"<|im_end|>{placeholder}<|im_end|>")
    assert ids[0] == 2 and ids[-1] == 2 and 2 not in ids[1:-1]


def test_added_token_not_marked_special_keeps_its_id_in_message_text():
    # The server scans every added token; the tokenizers library does too, in both encoders.
    models = pytest.importorskip("tokenizers.models")
    from tokenizers import AddedToken, Tokenizer

    from tools.bench.agent_replay.export import _PromptEncoder

    base = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, "<unk>"))
    base.add_tokens([AddedToken("<ns>", special=False)])
    encoder = _PromptEncoder(base.to_str())
    assert encoder._plain.encode("a<ns>", add_special_tokens=False).ids == [1, 2]
    assert encoder.encode("a<ns>") == [1, 2]


def test_tojson_accepts_the_server_filter_options():
    pytest.importorskip("jinja2")
    from tools.bench.agent_replay.export import _compile_template

    def render(expression):
        return _compile_template("{{ " + expression + " }}").render(v={"b": "é", "a": [1, 2]})

    assert render("v | tojson") == '{"b": "é", "a": [1, 2]}'
    assert render("v | tojson(sort_keys=true)") == '{"a": [1, 2], "b": "é"}'
    assert render("v | tojson(ensure_ascii=true)") == '{"b": "\\u00e9", "a": [1, 2]}'
    assert render("v | tojson(separators=[',', ':'])") == '{"b":"é","a":[1,2]}'
    assert render("v | tojson(false, 1, none, true)") == '{\n "a": [\n  1,\n  2\n ],\n "b": "é"\n}'


def _write_run(tmp_path, *, server_tokens, logged):
    models = pytest.importorskip("tokenizers.models")
    pytest.importorskip("jinja2")
    from tokenizers import Tokenizer, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "hi": 1, "yo": 2}, "<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    resources = tmp_path / "tok"
    resources.mkdir()
    tokenizer.save(str(resources / "tokenizer.json"))
    (resources / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }} {% endfor %}", encoding="utf-8")
    (resources / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    (run / "transcripts.jsonl").write_text(json.dumps({
        "session": 0, "tools": [], "messages": messages,
        "turns": [{"turn": 0, "message_count": 1, "category": "x"}]}) + "\n", encoding="utf-8")
    (run / "turns.jsonl").write_text(json.dumps({
        "session": 0, "turn": 0, "copy": False, "prompt_tokens": server_tokens,
        "error": None}) + "\n", encoding="utf-8")
    if logged:
        (run / "output_token_ids.jsonl").write_text("", encoding="utf-8")
    return run, resources


def test_export_succeeds_when_prompt_length_matches(tmp_path):
    from tools.bench.agent_replay.export import export

    run, resources = _write_run(tmp_path, server_tokens=1, logged=False)
    out = tmp_path / "streams.jsonl"
    assert export(run, None, resources, out) == 0
    assert json.loads(out.read_text())["output_source"] == "retokenized_text"


def test_export_fails_and_removes_stream_on_prompt_length_mismatch(tmp_path):
    from tools.bench.agent_replay.export import export

    run, resources = _write_run(tmp_path, server_tokens=7, logged=False)
    out = tmp_path / "streams.jsonl"
    assert export(run, None, resources, out) == 1
    assert not out.exists()


def test_export_fails_when_request_log_lacks_an_output(tmp_path):
    from tools.bench.agent_replay.export import export

    run, resources = _write_run(tmp_path, server_tokens=1, logged=True)
    out = tmp_path / "streams.jsonl"
    assert export(run, None, resources, out) == 1
    assert not out.exists()
