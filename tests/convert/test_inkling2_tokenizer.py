"""Inkling-2 tokenizer contract: the real tokenizer.json pipeline, and token-by-token identity with the GGUF on a
complete synthetic tokenizer that carries the real added tokens (tests/fixtures/inkling2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.convert import inkling2_tokenizer as tok
from tools.convert.inkling2 import Inkling2Error

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "inkling2"
BASE = 100  # synthetic base vocabulary; the real one has 199,998


def _pipeline() -> dict:
    return json.loads((FIXTURES / "tokenizer_pipeline.json").read_text())


def complete(base: int = BASE) -> tuple[dict, dict, int]:
    """(tokenizer.json, GGUF metadata, unpadded vocabulary) that agree; added tokens renumbered after ``base``."""
    f = _pipeline()
    added = [dict(item, id=base + i) for i, item in enumerate(f["added_tokens"])]
    vocab = {f"t{i}": i for i in range(base)}
    tokenizer = {"normalizer": None, "pre_tokenizer": f["pre_tokenizer"],
                 "model": f["model"] | {"vocab": vocab, "merges": [["t", "0"], "t 1"]}, "added_tokens": added}
    tokens = list(vocab) + [item["content"] for item in added]
    by_text = {text: i for i, text in enumerate(tokens)}
    metadata = {
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "inkling2",
        "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": [1] * base + [3] * len(added),
        "tokenizer.ggml.merges": ["t 0", "t 1"],
        "tokenizer.ggml.eos_token_id": by_text[tok.EOS], "tokenizer.ggml.padding_token_id": by_text[tok.PAD],
        "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False,
    }
    return tokenizer, metadata, len(tokens)


def test_the_checkpoint_pipeline_passes():
    f = _pipeline()
    tok.check_pipeline({"normalizer": f["normalizer"], "pre_tokenizer": f["pre_tokenizer"], "model": f["model"],
                        "added_tokens": f["added_tokens"]})
    # The real ids: 199,998 base tokens, 60 added ones, EOS 200,006 and padding 199,999.
    ids = {item["content"]: item["id"] for item in f["added_tokens"]}
    assert f["vocab_size"] == 199_998 and len(ids) == 60 and ids[tok.EOS] == 200_006 and ids[tok.PAD] == 199_999


def test_the_real_gguf_metadata_agrees_with_the_checkpoint():
    metadata = json.loads((FIXTURES / "ud_q8_k_xl_header.json").read_text())["metadata"]
    pipeline = _pipeline()
    assert metadata["tokenizer.ggml.tokens"]["length"] == pipeline["vocab_size"] + len(pipeline["added_tokens"])
    assert metadata["tokenizer.ggml.merges"]["length"] == pipeline["merges"]
    assert metadata["tokenizer.ggml.merges"]["head"] == pipeline["merges_head"]
    assert metadata["tokenizer.chat_template"]["sha256"] == pipeline["chat_template_sha256"]
    with pytest.raises(Inkling2Error, match="complete"):  # a summary is never enough to convert
        tok.check_tokenizer(complete()[0], metadata, 200_058)


def test_a_matching_tokenizer_passes():
    tok.check_tokenizer(*complete())


def _swap_base(t, m):
    t["model"]["vocab"]["t3"], t["model"]["vocab"]["t4"] = 4, 3


def _rename_added(t, m):
    t["added_tokens"][5]["content"] = "<|other|>"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (_swap_base, "in tokenizer.json but"),  # same size, different id-to-string mapping
        (_rename_added, "added token"),
        (lambda t, m: t["model"].update(merges=["t 1", "t 0"]), "merges differ"),
        (lambda t, m: t["model"].update(ignore_merges=False), "ignore_merges"),
        (lambda t, m: t["pre_tokenizer"]["pretokenizers"][0]["pattern"].update(Regex="\\s+"), "pre-tokenizer"),
        (lambda t, m: t.update(normalizer={"type": "NFC"}), "normalizer"),
        (lambda t, m: t["added_tokens"][3].update(special=False), "flags"),
        (lambda t, m: t["added_tokens"][3].update(lstrip=True), "flags"),
        (lambda t, m: t["added_tokens"].pop(), "added tokens must be"),
        (lambda t, m: m["tokenizer.ggml.token_type"].__setitem__(7, 3), "type 3, expected 1"),
        (lambda t, m: m["tokenizer.ggml.token_type"].__setitem__(BASE + 2, 1), "type 1, expected 3"),
        (lambda t, m: m.update({"tokenizer.ggml.pre": "o200k"}), "expected 'gpt2'"),
        (lambda t, m: m.update({"tokenizer.ggml.eos_token_id": 0}), "eos_token_id"),
        (lambda t, m: m.update({"tokenizer.ggml.add_bos_token": True}), "add_bos_token"),
        (lambda t, m: m.update({"tokenizer.ggml.add_eos_token": None}), "add_eos_token"),
    ],
)
def test_a_different_tokenizer_is_refused(mutate, match):
    t, m, n = complete()
    t, m = copy.deepcopy(t), copy.deepcopy(m)
    mutate(t, m)
    with pytest.raises(Inkling2Error, match=match):
        tok.check_tokenizer(t, m, n)


def test_the_gguf_must_list_the_unpadded_vocabulary():
    t, m, n = complete()
    with pytest.raises(Inkling2Error, match="unpadded"):
        tok.check_tokenizer(t, m, n + 1)
