# Written for NInfer-3090 from the V4.1-Flash checkpoint's tokenizer.json and the ISTA-DASLab GGUF's
# tokenizer metadata (tests/fixtures/dsv4/tokenizer_pipeline.json).
"""DeepSeek-V4.1-Flash tokenizer resources from a GGUF and the checkpoint's own ``tokenizer.json``.

The GGUF carries the vocabulary (129,280 tokens), the BPE merges and a token type per token, but not what
tokenizer.json also says: which added tokens are ``special`` (49 of the GGUF's control tokens are not: the
tool-call and role markers a client must see in the output) and the pre-tokenizer. So, unlike the Qwen
conversion, the resources are not synthesized from the GGUF alone: the checkpoint's ``tokenizer.json`` is
required, and every fact the GGUF also holds is checked against it:

- ``tokenizer.ggml.model`` is ``gpt2`` and ``tokenizer.ggml.pre`` is ``joyai-llm``;
- the pre-tokenizer is exactly the pipeline the runtime implements (:data:`PIPELINE`) and there is no
  normalizer;
- the base vocabulary and the merges equal the GGUF's, and every added token is the GGUF's token at its id
  (the first control tokens are in both lists, which is how the real file is);
- BOS, EOS and padding are tokens the file lists.

``tokenizer_config.json`` and ``generation_config.json`` are derived (the checkpoint ships neither with
what the runtime reads: ``added_tokens_decoder``, the stop ids). The chat template is the GGUF's unless the
caller supplies the checkpoint's own.
"""

from __future__ import annotations

import json
from typing import Mapping

from .dsv4 import Dsv4Error

# What tokenizer.json's pre_tokenizer must be, exactly: three Splits (runs of up to three digits, runs of CJK
# ideographs and kana, the word regex) and byte level. src/models/qwen3_5/frontend/tokenizer.cpp implements it.
_SPLIT = {"behavior": "Isolated", "invert": False}
PIPELINE = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, **_SPLIT},
        {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, **_SPLIT},
        {
            "type": "Split",
            "pattern": {
                "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
                "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
            },
            **_SPLIT,
        },
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
GGUF_PRE = "joyai-llm"
ROLES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _merge_strings(merges) -> list[str]:
    """Merges as "a b" strings; tokenizers also writes them as [a, b] pairs."""
    return [m if isinstance(m, str) else " ".join(m) for m in merges]


def _check_runtime_fields(official: Mapping) -> None:
    """Every field the runtime's tokenizer loader reads, with the types and values it accepts, so a file that
    passes here also loads (tokenizer.cpp: validate_pipeline, load_vocab, load_added_tokens)."""
    def need(ok: bool, what: str) -> None:
        if not ok:
            raise Dsv4Error(f"tokenizer.json: {what}")

    if official.get("pre_tokenizer") != PIPELINE:
        raise Dsv4Error("tokenizer.json's pre_tokenizer is not the DeepSeek-V4 pipeline the runtime implements")
    need(official.get("normalizer") in (None, {"type": "Sequence", "normalizers": []}),
         "has a normalizer; the DeepSeek-V4 pipeline has none")
    # The loader tests presence, not value: a decoder key that is present must be ByteLevel (null is refused),
    # a post_processor key may be null.
    if "decoder" in official:
        need(isinstance(official["decoder"], dict) and official["decoder"].get("type") == "ByteLevel", "decoder must be ByteLevel")
    post = official.get("post_processor")
    need(post is None or (isinstance(post, dict) and post.get("type") == "ByteLevel"), "post_processor must be ByteLevel or null")
    model = official.get("model")
    need(isinstance(model, dict) and model.get("type") == "BPE", "model must be a BPE model")
    need(model.get("dropout") in (None, 0, 0.0) and not isinstance(model.get("dropout"), bool), "model.dropout must be null or 0")
    for field in ("continuing_subword_prefix", "end_of_word_suffix"):
        need(model.get(field) in (None, ""), f"model.{field} must be empty")
    # ignore_merges, if present, must be exactly the boolean false (null and 0 are refused by the loader).
    need(model.get("ignore_merges", False) is False, "model.ignore_merges must be false")
    need(not model.get("byte_fallback", False), "model must not use byte_fallback")
    vocab = model.get("vocab")
    need(isinstance(vocab, dict) and all(type(i) is int for i in vocab.values()), "model.vocab must map tokens to integer ids")
    need(sorted(vocab.values()) == list(range(len(vocab))), "the vocabulary ids are not contiguous from 0")
    need(isinstance(model.get("merges"), list), "model.merges must be a list")
    added = official.get("added_tokens")
    need(isinstance(added, list), "added_tokens must be a list")
    seen = set()
    for item in added:
        need(isinstance(item, dict) and type(item.get("id")) is int and isinstance(item.get("content"), str)
             and item["content"], "each added token needs an integer id and non-empty content")
        for flag in ("single_word", "lstrip", "rstrip", "normalized", "special"):
            need(type(item.get(flag)) is bool, f"added token {item['content']!r}: {flag} must be a boolean")
        need(not (item["single_word"] or item["lstrip"] or item["rstrip"]),
             f"added token {item['content']!r} strips or matches whole words, which the runtime does not do")
        need(item["id"] not in seen, f"added token id {item['id']} is listed twice")
        seen.add(item["id"])


# GGUF token types (llama.cpp): the ones this tokenizer uses.
_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4


def _check_membership(metadata: Mapping, tokens: list, vocab_size: int, added: Mapping) -> None:
    """Which tokens are ordinary vocabulary and which are added, from the GGUF's own token types rather than from
    where tokenizer.json happens to list them: a normal token is in the vocabulary and not added; a control or
    user-defined token is added (it may also sit in the vocabulary: the real file lists its first three control
    tokens, BOS, EOS and a pad marker, in both, and the loader treats the same id and content as one token). A tokenizer.json that moves a marker such as the tool-call tokens into the vocabulary
    would otherwise load but BPE-encode the marker instead of matching it whole."""
    types = metadata.get("tokenizer.ggml.token_type")
    if not isinstance(types, list) or len(types) != len(tokens):
        raise Dsv4Error("tokenizer.ggml.token_type must give one type per token")
    for index, kind in enumerate(types):
        in_vocab, is_added = index < vocab_size, index in added  # in_vocab: ids 0..vocab_size-1
        if kind == _NORMAL:
            if not in_vocab or is_added:
                raise Dsv4Error(f"token {index} ({tokens[index]!r}) is a normal token in the GGUF, so it must be in "
                                "tokenizer.json's vocabulary and not an added token")
        elif kind in (_CONTROL, _USER_DEFINED):
            if not is_added:
                raise Dsv4Error(f"token {index} ({tokens[index]!r}) is a control or user-defined token in the GGUF, so it "
                                "must be one of tokenizer.json's added tokens")
        else:
            raise Dsv4Error(f"token {index} has GGUF token type {kind}, which this tokenizer does not use")


def check_tokenizer(official: Mapping, metadata: Mapping) -> int:
    """Refuse a tokenizer.json that is not this GGUF's tokenizer. Returns the base vocabulary size."""
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != GGUF_PRE:
        raise Dsv4Error(f"only the gpt2 BPE tokenizer with the {GGUF_PRE} pre-tokenizer is supported")
    tokens, merges = metadata.get("tokenizer.ggml.tokens"), metadata.get("tokenizer.ggml.merges")
    if not isinstance(tokens, list) or not isinstance(merges, list):
        raise Dsv4Error("tokenizer.ggml tokens and merges are required")
    _check_runtime_fields(official)
    model = official["model"]
    vocab = model["vocab"]
    if len(vocab) > len(tokens):
        raise Dsv4Error("tokenizer.json's vocabulary is larger than the GGUF's")
    for token, index in vocab.items():
        if tokens[index] != token:
            raise Dsv4Error(f"token {index} is {token!r} in tokenizer.json but {tokens[index]!r} in the GGUF")
    if _merge_strings(model["merges"]) != merges:
        raise Dsv4Error("tokenizer.json's merges differ from the GGUF's")
    added = {}
    for item in official["added_tokens"]:
        index = item["id"]
        if not 0 <= index < len(tokens) or tokens[index] != item["content"]:
            raise Dsv4Error(f"added token {item.get('content')!r} is not the GGUF's token {index}")
        # load_added_tokens: a content may map to one id only, across the added tokens and the vocabulary (the same
        # id and content listed in both is the one allowed overlap).
        if item["content"] in vocab and vocab[item["content"]] != index:
            raise Dsv4Error(f"added token {item['content']!r} is also a vocabulary token at id {vocab[item['content']]}, not {index}")
        added[index] = item
    contents = [item["content"] for item in official["added_tokens"]]
    if len(set(contents)) != len(contents):
        raise Dsv4Error("two added tokens have the same content, which the runtime rejects as ambiguous")
    _check_membership(metadata, tokens, len(vocab), added)
    for key in ("bos", "eos", "padding"):
        index = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if type(index) is not int or not 0 <= index < len(tokens):
            raise Dsv4Error(f"tokenizer.ggml.{key}_token_id does not name a token")
    for flag in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if metadata.get(flag, False) is not False:
            raise Dsv4Error(f"{flag} is true; this tokenizer adds no BOS or EOS")
    return len(vocab)


def resources(
    metadata: Mapping, config: Mapping, tokenizer_json: bytes, chat_template: bytes | None = None
) -> dict[str, bytes]:
    """The four text resources: tokenizer.json as given (after :func:`check_tokenizer`), a tokenizer_config.json
    whose ``added_tokens_decoder`` mirrors it, the chat template and a generation_config.json."""
    official = json.loads(tokenizer_json)
    check_tokenizer(official, metadata)
    tokens = metadata["tokenizer.ggml.tokens"]
    template = chat_template if chat_template is not None else metadata.get("tokenizer.chat_template")
    if isinstance(template, str):
        template = template.encode("utf-8")
    if not template:
        raise Dsv4Error("a chat template is required: the GGUF has none, so supply the checkpoint's")
    decoder = {
        str(item["id"]): {
            "content": item["content"], "lstrip": False, "normalized": bool(item.get("normalized", False)),
            "rstrip": False, "single_word": False, "special": bool(item["special"]),
        }
        for item in official.get("added_tokens", [])
    }
    bos, eos, pad = (tokens[metadata[f"tokenizer.ggml.{k}_token_id"]] for k in ("bos", "eos", "padding"))
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "added_tokens_decoder": decoder,
        "bos_token": bos,
        "clean_up_tokenization_spaces": False,
        "eos_token": eos,
        "model_max_length": config["max_position_embeddings"],
        "pad_token": pad,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": None,
    }
    generation = {
        "bos_token_id": metadata["tokenizer.ggml.bos_token_id"],
        "do_sample": True,
        "eos_token_id": [metadata["tokenizer.ggml.eos_token_id"]],
        "pad_token_id": metadata["tokenizer.ggml.padding_token_id"],
        "temperature": 1.0,
        "top_p": 1.0,
    }
    return {
        "tokenizer.json": tokenizer_json,
        "tokenizer_config.json": _json_bytes(tokenizer_config),
        "chat_template.jinja": template,
        "generation_config.json": _json_bytes(generation),
    }
