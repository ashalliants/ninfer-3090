# Written for NInfer-3090 from Inkling-2-Mini's tokenizer.json (thinkingmachines/Inkling-2-Mini@6e7c8248) and the
# Unsloth GGUF's tokenizer metadata (tests/fixtures/inkling2/tokenizer_pipeline.json, *_header.json).
"""Inkling-2-Mini tokenizer: the facts the runtime must implement, checked against tokenizer.json and the GGUF.

Inkling-2 is byte-level BPE with an o200k-style word regex (one Split, then ByteLevel without its own regex), no
normalizer, and ``ignore_merges``: a pre-token that is itself a vocabulary entry becomes that one token without
applying merges. The runtime tokenizer (src/models/qwen3_5/frontend/tokenizer.cpp) has neither this regex nor
``ignore_merges`` yet; phase 3 of docs/maintainer/inkling2-model.md adds both. Until then this module states the
contract and refuses a tokenizer that differs from it, so the converter cannot ship resources the runtime would
tokenize differently.

:func:`check_pipeline` checks what tokenizer.json alone says: the pipeline (:data:`PIPELINE`), no normalizer, BPE
with ``ignore_merges`` and no byte fallback, and added tokens that are all special, matched whole (no stripping)
and unnormalized. :func:`check_tokenizer` then ties it to the GGUF, whose rows the embedding and head follow:

- the GGUF lists exactly ``unpadded_vocab_size`` tokens (the embedding has more rows: padding), with
  ``tokenizer.ggml.model`` ``gpt2`` and ``tokenizer.ggml.pre`` ``inkling2``;
- tokenizer.json's vocabulary is ids 0..B-1 with every token equal to the GGUF's token at that id, and its merges
  equal the GGUF's; the GGUF marks exactly those B tokens normal;
- the added tokens are ids B..N-1 in order, each the GGUF's token at its id, which the GGUF marks control;
- EOS is ``<|content_model_end_sampling|>`` and padding ``<|endoftext|>`` by id; no BOS or EOS is added.

For Inkling-2-Mini, B is 199,998 and N 200,058.
"""

from __future__ import annotations

from typing import Mapping

from .inkling2 import Inkling2Error

WORD_REGEX = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)
PIPELINE = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": WORD_REGEX}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
GGUF_PRE = "inkling2"
EOS = "<|content_model_end_sampling|>"
PAD = "<|endoftext|>"
_NORMAL, _CONTROL = 1, 3  # GGUF token types


def check_pipeline(tokenizer: Mapping) -> None:
    """Refuse a tokenizer.json whose pipeline, model flags or added-token flags are not Inkling-2's."""
    if tokenizer.get("normalizer") is not None:
        raise Inkling2Error("tokenizer.json: Inkling-2 has no normalizer")
    if tokenizer.get("pre_tokenizer") != PIPELINE:
        raise Inkling2Error("tokenizer.json: the pre-tokenizer is not the Inkling-2 pipeline the runtime implements")
    model = tokenizer.get("model")
    if not isinstance(model, Mapping):
        raise Inkling2Error("tokenizer.json: model is missing")
    expected = {"type": "BPE", "ignore_merges": True, "byte_fallback": False, "dropout": None,
                "continuing_subword_prefix": None, "end_of_word_suffix": None}
    for key, value in expected.items():
        if model.get(key) != value:
            raise Inkling2Error(f"tokenizer.json: model.{key} is {model.get(key)!r}, expected {value!r}")
    added = tokenizer.get("added_tokens")
    if not isinstance(added, list) or not added:
        raise Inkling2Error("tokenizer.json: added_tokens must be a non-empty list")
    for item in added:
        if not isinstance(item, Mapping) or type(item.get("id")) is not int or not isinstance(item.get("content"), str):
            raise Inkling2Error(f"tokenizer.json: malformed added token {item!r}")
        flags = {flag: item.get(flag) for flag in ("special", "single_word", "lstrip", "rstrip", "normalized")}
        if flags != {"special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False}:
            raise Inkling2Error(f"tokenizer.json: added token {item['content']!r} has flags {flags}; "
                                "Inkling-2's are special, matched whole and unnormalized")


def _merge_strings(merges) -> list[str]:
    out = []
    for index, item in enumerate(merges):
        if isinstance(item, list) and len(item) == 2 and all(isinstance(part, str) for part in item):
            out.append(f"{item[0]} {item[1]}")
        elif isinstance(item, str):
            out.append(item)
        else:
            raise Inkling2Error(f"tokenizer.json: merge {index} must be a pair of symbols, got {item!r}")
    return out


def check_tokenizer(tokenizer: Mapping, metadata: Mapping, unpadded_vocab: int) -> None:
    """Refuse a tokenizer.json that is not this GGUF's tokenizer, token by token (see the module docstring)."""
    check_pipeline(tokenizer)
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != GGUF_PRE:
        raise Inkling2Error(
            f"GGUF tokenizer is {metadata.get('tokenizer.ggml.model')!r}/{metadata.get('tokenizer.ggml.pre')!r}, "
            f"expected 'gpt2'/{GGUF_PRE!r}"
        )
    tokens, types, merges = (metadata.get(f"tokenizer.ggml.{k}") for k in ("tokens", "token_type", "merges"))
    if not isinstance(tokens, list) or not isinstance(types, list) or not isinstance(merges, list):
        raise Inkling2Error("the GGUF's complete tokenizer.ggml tokens, token types and merges are required")
    if len(tokens) != unpadded_vocab or len(types) != len(tokens):
        raise Inkling2Error(f"GGUF lists {len(tokens)} tokens and {len(types)} types; expected {unpadded_vocab} "
                            "(the unpadded vocabulary)")
    vocab = tokenizer["model"].get("vocab")
    if not isinstance(vocab, Mapping) or not all(type(i) is int for i in vocab.values()):
        raise Inkling2Error("tokenizer.json: model.vocab must map tokens to integer ids")
    base = len(vocab)
    if sorted(vocab.values()) != list(range(base)):
        raise Inkling2Error("tokenizer.json: the vocabulary ids are not contiguous from 0")
    for token, index in vocab.items():
        if tokens[index] != token:
            raise Inkling2Error(f"token {index} is {token!r} in tokenizer.json but {tokens[index]!r} in the GGUF")
    if _merge_strings(tokenizer["model"].get("merges", [])) != merges:
        raise Inkling2Error("tokenizer.json's merges differ from the GGUF's")
    added = tokenizer["added_tokens"]
    if [item["id"] for item in added] != list(range(base, len(tokens))):
        raise Inkling2Error(f"tokenizer.json: added tokens must be ids {base}..{len(tokens) - 1} in order")
    for item in added:
        if tokens[item["id"]] != item["content"]:
            raise Inkling2Error(f"added token {item['content']!r} is {tokens[item['id']]!r} in the GGUF")
    for index, kind in enumerate(types):
        want = _NORMAL if index < base else _CONTROL
        if kind != want:
            raise Inkling2Error(f"GGUF token {index} ({tokens[index]!r}) has type {kind}, expected {want}")
    for key, text in (("eos", EOS), ("padding", PAD)):
        index = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if type(index) is not int or not 0 <= index < len(tokens) or tokens[index] != text:
            raise Inkling2Error(f"tokenizer.ggml.{key}_token_id must name {text!r}")
    for flag in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        if metadata.get(flag) is not False:
            raise Inkling2Error(f"{flag} must be false: Inkling-2 adds no BOS or EOS")
