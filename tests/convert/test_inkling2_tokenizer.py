"""Inkling-2 tokenizer contract against the real tokenizer.json pipeline and GGUF metadata (tests/fixtures/inkling2)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.convert import inkling2_tokenizer as tok
from tools.convert.inkling2 import Inkling2Error

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "inkling2"


@pytest.fixture()
def tokenizer() -> dict:
    f = json.loads((FIXTURES / "tokenizer_pipeline.json").read_text())
    return {"normalizer": f["normalizer"], "pre_tokenizer": f["pre_tokenizer"],
            "model": f["model"] | {"vocab": f["vocab_size"]}, "added_tokens": f["added_tokens"]}


@pytest.fixture()
def metadata() -> dict:
    return json.loads((FIXTURES / "ud_q4_k_xl_header.json").read_text())["metadata"]


def test_the_checkpoint_tokenizer_passes(tokenizer, metadata):
    tok.check_tokenizer_json(tokenizer)
    tok.check_gguf(metadata, 200058)


def test_gguf_template_is_the_checkpoints(metadata):
    pipeline = json.loads((FIXTURES / "tokenizer_pipeline.json").read_text())
    assert metadata["tokenizer.chat_template"]["sha256"] == pipeline["chat_template_sha256"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t: t["model"].update(ignore_merges=False), "ignore_merges"),
        (lambda t: t["pre_tokenizer"]["pretokenizers"][0]["pattern"].update(Regex="\\s+"), "pre-tokenizer"),
        (lambda t: t.update(normalizer={"type": "NFC"}), "normalizer"),
        (lambda t: t["added_tokens"][3].update(special=False), "not special"),
        (lambda t: t["added_tokens"].pop(), "added tokens must be"),
        (lambda t: t["model"].update(vocab=199997), "base tokens"),
    ],
)
def test_a_different_tokenizer_is_refused(tokenizer, mutate, match):
    t = copy.deepcopy(tokenizer)
    mutate(t)
    with pytest.raises(Inkling2Error, match=match):
        tok.check_tokenizer_json(t)


@pytest.mark.parametrize(
    "key, value",
    [("tokenizer.ggml.pre", "o200k"), ("tokenizer.ggml.eos_token_id", 199999), ("tokenizer.ggml.add_bos_token", True)],
)
def test_different_gguf_metadata_is_refused(metadata, key, value):
    with pytest.raises(Inkling2Error):
        tok.check_gguf(metadata | {key: value}, 200058)


def test_the_gguf_must_list_the_unpadded_vocabulary(metadata):
    with pytest.raises(Inkling2Error, match="tokens"):
        tok.check_gguf(metadata, 201024)
