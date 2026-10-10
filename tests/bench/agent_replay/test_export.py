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
