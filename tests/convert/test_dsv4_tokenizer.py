"""DeepSeek-V4.1-Flash tokenizer resources: the checkpoint's tokenizer.json checked against the GGUF, the derived
tokenizer_config.json and generation_config.json, the token domain, and (when the real files are present) the real build."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from tools.convert import dsv4, dsv4_model, dsv4_tokenizer as tok
from tools.convert.resources import token_domain
from tools.convert.sources.gguf import GgufModel

from . import dsv4_fixture as fixture
from .test_dsv4_model import tokenizer_file


@pytest.fixture
def metadata():
    return {k: v for k, _, v in fixture._plain(fixture.metadata())}


@pytest.fixture
def official():
    return fixture.official_tokenizer()


@pytest.fixture
def config():
    return fixture.config()


def test_the_pipeline_constant_is_the_one_the_fixture_spells_out():
    assert tok.PIPELINE == fixture.PRE_TOKENIZER


def test_resources_from_a_matching_checkpoint(metadata, official, config):
    out = tok.resources(metadata, config, json.dumps(official).encode())
    assert set(out) == set(tok.ROLES)
    assert json.loads(out["tokenizer.json"]) == official  # as given, byte for byte
    cfg = json.loads(out["tokenizer_config.json"])
    assert (cfg["bos_token"], cfg["eos_token"], cfg["pad_token"]) == ("<bos>", "<eos>", "<pad>")
    assert cfg["add_bos_token"] is False and cfg["model_max_length"] == config["max_position_embeddings"]
    # The runtime reads added_tokens_decoder: it mirrors tokenizer.json, flags and all (<tool> is not special).
    decoder = cfg["added_tokens_decoder"]
    assert decoder["12"] == {"content": "<tool>", "lstrip": False, "normalized": True, "rstrip": False,
                             "single_word": False, "special": False}
    assert decoder["13"]["special"] is True and len(decoder) == 7
    assert json.loads(out["generation_config.json"])["eos_token_id"] == [1]
    assert out["chat_template.jinja"] == b"{{ messages }}"
    count, special = token_domain(json.loads(out["tokenizer.json"]), cfg, len(fixture.TOKENS))
    assert count == 16 and special == (0, 1, 2, 13, 14, 15)  # every added token but <tool>


def test_the_checkpoints_chat_template_overrides_the_ggufs(metadata, official, config):
    out = tok.resources(metadata, config, json.dumps(official).encode(), b"{{ checkpoint }}")
    assert out["chat_template.jinja"] == b"{{ checkpoint }}"
    del metadata["tokenizer.chat_template"]
    with pytest.raises(dsv4.Dsv4Error, match="chat template is required"):
        tok.resources(metadata, config, json.dumps(official).encode())


def _refused(metadata, official, match):
    with pytest.raises(dsv4.Dsv4Error, match=match):
        tok.check_tokenizer(official, metadata)


def test_each_mismatch_is_refused(metadata, official):
    assert tok.check_tokenizer(official, metadata) == fixture.BASE_VOCAB
    other_pre = {**metadata, "tokenizer.ggml.pre": "qwen35"}
    _refused(other_pre, official, "joyai-llm")
    _refused({**metadata, "tokenizer.ggml.model": "llama"}, official, "gpt2")

    wrong_pipeline = copy.deepcopy(official)
    wrong_pipeline["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\p{N}{1,2}"
    _refused(metadata, wrong_pipeline, "pre_tokenizer")
    _refused(metadata, {**official, "pre_tokenizer": None}, "pre_tokenizer")

    _refused(metadata, {**official, "normalizer": {"type": "NFC"}}, "has a normalizer")

    other_vocab = copy.deepcopy(official)
    other_vocab["model"]["vocab"]["a"], other_vocab["model"]["vocab"]["b"] = 4, 3
    _refused(metadata, other_vocab, "token [34] is '[ab]' in tokenizer.json")
    gap = copy.deepcopy(official)
    del gap["model"]["vocab"]["h"]
    _refused(metadata, gap, "not contiguous")
    big = copy.deepcopy(official)
    big["model"]["vocab"].update({f"z{i}": 12 + i for i in range(10)})
    _refused(metadata, big, "larger than the GGUF's")

    _refused(metadata, {**official, "model": {**official["model"], "merges": ["a c"]}}, "merges differ")
    # tokenizers also writes merges as pairs: that is the same list.
    assert tok.check_tokenizer({**official, "model": {**official["model"], "merges": [["a", "b"]]}}, metadata) == fixture.BASE_VOCAB

    wrong_added = copy.deepcopy(official)
    wrong_added["added_tokens"][3]["content"] = "<other>"
    _refused(metadata, wrong_added, "not the GGUF's token 12")
    stripped = copy.deepcopy(official)
    stripped["added_tokens"][4]["lstrip"] = True
    _refused(metadata, stripped, "strips or matches whole words")
    short = copy.deepcopy(official)
    short["added_tokens"] = short["added_tokens"][:-1]
    _refused(metadata, short, "control or user-defined token in the GGUF, so it must be one of tokenizer.json's added tokens")

    _refused({k: v for k, v in metadata.items() if k != "tokenizer.ggml.eos_token_id"}, official, "eos_token_id")
    _refused({**metadata, "tokenizer.ggml.add_bos_token": True}, official, "adds no BOS")
    _refused({k: v for k, v in metadata.items() if k != "tokenizer.ggml.merges"}, official, "tokens and merges are required")


def test_membership_follows_the_ggufs_token_types(metadata, official):
    # Moving the non-special <tool> marker from the added tokens into the vocabulary keeps every id contiguous and
    # every string equal, but the runtime would then BPE-encode the marker instead of matching it whole: refused.
    moved = copy.deepcopy(official)
    moved["model"]["vocab"]["<tool>"] = 12
    moved["added_tokens"] = [a for a in moved["added_tokens"] if a["id"] != 12]
    _refused(metadata, moved, "token 12 .*control or user-defined token in the GGUF")
    # (A control token listed in both the vocabulary and the added tokens is fine: the baseline does it for ids 0-2, as the real file does.)
    # A normal token must be in the vocabulary and not added.
    added_normal = copy.deepcopy(official)
    added_normal["added_tokens"].append({"id": 3, "content": "a", "single_word": False, "lstrip": False,
                                         "rstrip": False, "normalized": False, "special": False})
    _refused(metadata, added_normal, "normal token in the GGUF")
    _refused({**metadata, "tokenizer.ggml.token_type": [*fixture.TOKEN_TYPES[:-1], 5]}, official, "token type 5")
    _refused({**metadata, "tokenizer.ggml.token_type": fixture.TOKEN_TYPES[:-1]}, official, "one type per token")


def test_every_field_the_runtime_loader_reads_is_validated(metadata, official):
    """A file that passes check_tokenizer must load: these are the loader's refusals (tokenizer.cpp)."""
    def broken(mutate):
        value = copy.deepcopy(official)
        mutate(value)
        return value

    cases = {
        "dropout": broken(lambda t: t["model"].__setitem__("dropout", 0.1)),
        "continuing_subword_prefix": broken(lambda t: t["model"].__setitem__("continuing_subword_prefix", "##")),
        "end_of_word_suffix": broken(lambda t: t["model"].__setitem__("end_of_word_suffix", "</w>")),
        "ignore_merges": broken(lambda t: t["model"].__setitem__("ignore_merges", True)),
        "model must be a BPE": broken(lambda t: t["model"].__setitem__("type", "WordPiece")),
        "model.merges must be a list": broken(lambda t: t["model"].pop("merges")),
        "added_tokens must be a list": broken(lambda t: t.pop("added_tokens")),
        "decoder must be ByteLevel": broken(lambda t: t.__setitem__("decoder", {"type": "WordPiece"})),
        "post_processor must be ByteLevel": broken(lambda t: t.__setitem__("post_processor", {"type": "TemplateProcessing"})),
        "must be a boolean": broken(lambda t: t["added_tokens"][0].pop("special")),
        "integer id and non-empty content": broken(lambda t: t["added_tokens"][0].__setitem__("content", "")),
        "listed twice": broken(lambda t: t["added_tokens"].append(dict(t["added_tokens"][0]))),
        "integer ids": broken(lambda t: t["model"]["vocab"].__setitem__("a", "3")),
        "decoder must be ByteLevel (null)": broken(lambda t: t.__setitem__("decoder", None)),
        "ignore_merges (null)": broken(lambda t: t["model"].__setitem__("ignore_merges", None)),
        "ignore_merges (0)": broken(lambda t: t["model"].__setitem__("ignore_merges", 0)),
        # (these two also need the GGUF to hold the duplicated string, or the id/content check fires first)
        "also a vocabulary token": broken(lambda t: t["added_tokens"].__setitem__(3, {**t["added_tokens"][3], "content": "a"})),
        "same content": broken(lambda t: t["added_tokens"].__setitem__(4, {**t["added_tokens"][4], "content": "<tool>"})),
    }
    matches = {"decoder must be ByteLevel (null)": "decoder must be ByteLevel", "ignore_merges (null)": "ignore_merges must be false",
               "ignore_merges (0)": "ignore_merges must be false", "model must be a BPE": "BPE model",
               "model.merges must be a list": "model.merges", "dropout": "dropout", "integer ids": "integer ids"}
    duplicated = {"also a vocabulary token": (12, "a"), "same content": (13, "<tool>")}
    for name, value in cases.items():
        meta = metadata
        if name in duplicated:
            index, text = duplicated[name]
            meta = {**metadata, "tokenizer.ggml.tokens": [*fixture.TOKENS[:index], text, *fixture.TOKENS[index + 1:]]}
        _refused(meta, value, matches.get(name, name))
    # What the loader accepts is accepted: null dropout/prefix/suffix, a null post_processor, a missing decoder.
    accepted = broken(lambda t: (t.update(post_processor=None), t.pop("decoder")))
    accepted["model"].update(dropout=None, continuing_subword_prefix=None, end_of_word_suffix=None)
    assert tok.check_tokenizer(accepted, metadata) == fixture.BASE_VOCAB


def test_build_model_carries_the_resources_into_the_artifact(tmp_path):
    path = fixture.write(tmp_path / "tiny.gguf")
    with GgufModel(path) as gguf:
        model = dsv4_model.build_model(gguf, resource_overrides={"tokenizer.json": tokenizer_file(tmp_path)})
        none = dsv4_model.build_model(gguf)
    refs = model.components["text"]["resources"]
    assert set(refs) == set(tok.ROLES) and set(model.resources) == set(refs.values())
    assert model.token_count == 16 and model.special_token_ids == (0, 1, 2, 13, 14, 15)
    assert none.components["text"]["resources"] == {} and not none.resources


def test_build_model_refuses_what_it_cannot_use(tmp_path):
    path = fixture.write(tmp_path / "tiny.gguf")
    with GgufModel(path) as gguf:
        with pytest.raises(ValueError, match="no consumer"):
            dsv4_model.build_model(gguf, resource_overrides={"preprocessor_config.json": tmp_path / "x"})
        template = tmp_path / "t.jinja"
        template.write_text("x")
        with pytest.raises(ValueError, match="needs --resource tokenizer.json"):
            dsv4_model.build_model(gguf, resource_overrides={"chat_template.jinja": template})
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(fixture.official_tokenizer(normalizer={"type": "NFC"})))
        with pytest.raises(dsv4.Dsv4Error, match="has a normalizer"):
            dsv4_model.build_model(gguf, resource_overrides={"tokenizer.json": bad})


# The real build: the ISTA GGUF's first shard (tokenizer metadata, 5.8 MB) and the checkpoint's tokenizer.json (6 MB).
REAL_TOKENIZER = os.environ.get("NINFER_DSV4_TOKENIZER_JSON")
REAL_SHARD = os.environ.get("NINFER_DSV4_GGUF_SHARD1")
needs_real = pytest.mark.skipif(not (REAL_TOKENIZER and REAL_SHARD), reason="set NINFER_DSV4_TOKENIZER_JSON and NINFER_DSV4_GGUF_SHARD1")


@needs_real
def test_the_real_checkpoint_tokenizer_matches_the_real_gguf():
    from tools.convert.sources.gguf import GgufFile

    metadata = GgufFile(REAL_SHARD).metadata
    raw = Path(REAL_TOKENIZER).read_bytes()
    assert tok.check_tokenizer(json.loads(raw), metadata) == 128000
    out = tok.resources(metadata, {"max_position_embeddings": 1048576}, raw)
    cfg = json.loads(out["tokenizer_config.json"])
    assert len(cfg["added_tokens_decoder"]) == 1283
    count, special = token_domain(json.loads(out["tokenizer.json"]), cfg, 129280)
    # 1,278 GGUF control tokens, of which the checkpoint marks 49 not special: the tool-call and role markers.
    assert count == 129280 and len(special) == 1229
