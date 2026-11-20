# Written for NInfer-3090 from Inkling-2-Mini's tokenizer.json (thinkingmachines/Inkling-2-Mini@6e7c8248) and the
# Unsloth GGUF's tokenizer metadata (tests/fixtures/inkling2/tokenizer_pipeline.json, *_header.json).
"""Inkling-2-Mini tokenizer: the facts the runtime must implement, checked against tokenizer.json and the GGUF.

Inkling-2 is byte-level BPE with an o200k-style word regex (one Split, then ByteLevel without its own regex), no
normalizer, and ``ignore_merges``: a pre-token that is itself a vocabulary entry becomes that one token without
applying merges. The runtime tokenizer (src/models/qwen3_5/frontend/tokenizer.cpp) has neither this regex nor
``ignore_merges`` yet; phase 3 of docs/maintainer/inkling2-model.md adds both. Until then this module states the
contract and refuses a tokenizer that differs from it, so the converter cannot ship resources the runtime would
tokenize differently.

Checked: the pipeline (:data:`PIPELINE`), no normalizer, BPE with ``ignore_merges`` and no byte fallback; 199,998
base tokens and 60 added tokens at ids 199,998..200,057, all special; the GGUF lists exactly those 200,058 tokens
(the embedding has 201,024 rows: the rest is padding) with ``tokenizer.ggml.pre`` ``inkling2``; EOS 200,006
(``<|content_model_end_sampling|>``) and padding 199,999 (``<|endoftext|>``); no BOS.
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
BASE_VOCAB = 199_998
ADDED = 60
EOS = (200_006, "<|content_model_end_sampling|>")
PAD = (199_999, "<|endoftext|>")


def check_tokenizer_json(tokenizer: Mapping) -> None:
    """Refuse a tokenizer.json whose pipeline, model flags or added tokens are not Inkling-2's."""
    if tokenizer.get("normalizer") is not None:
        raise Inkling2Error("tokenizer.json: Inkling-2 has no normalizer")
    if tokenizer.get("pre_tokenizer") != PIPELINE:
        raise Inkling2Error("tokenizer.json: the pre-tokenizer is not the Inkling-2 pipeline the runtime implements")
    model = tokenizer.get("model", {})
    expected = {"type": "BPE", "ignore_merges": True, "byte_fallback": False, "dropout": None,
                "continuing_subword_prefix": None, "end_of_word_suffix": None}
    for key, value in expected.items():
        if model.get(key) != value:
            raise Inkling2Error(f"tokenizer.json: model.{key} is {model.get(key)!r}, expected {value!r}")
    vocab = model.get("vocab")
    count = vocab if isinstance(vocab, int) else len(vocab or ())
    if count != BASE_VOCAB:
        raise Inkling2Error(f"tokenizer.json: {count} base tokens, expected {BASE_VOCAB}")
    added = tokenizer.get("added_tokens", [])
    if [t["id"] for t in added] != list(range(BASE_VOCAB, BASE_VOCAB + ADDED)):
        raise Inkling2Error(f"tokenizer.json: added tokens must be ids {BASE_VOCAB}..{BASE_VOCAB + ADDED - 1} in order")
    plain = [t["content"] for t in added if not t.get("special")]
    if plain:
        raise Inkling2Error(f"tokenizer.json: added tokens {plain[:3]} are not special")
    by_id = {t["id"]: t["content"] for t in added}
    for token_id, text in (EOS, PAD):
        if by_id.get(token_id) != text:
            raise Inkling2Error(f"tokenizer.json: token {token_id} is {by_id.get(token_id)!r}, expected {text!r}")


def check_gguf(metadata: Mapping, unpadded_vocab: int) -> None:
    """Refuse GGUF tokenizer metadata that is not Inkling-2's. ``tokens`` may be a list or ``{length, ...}``."""
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != GGUF_PRE:
        raise Inkling2Error(
            f"GGUF tokenizer is {metadata.get('tokenizer.ggml.model')!r}/{metadata.get('tokenizer.ggml.pre')!r}, "
            f"expected 'gpt2'/{GGUF_PRE!r}"
        )
    tokens = metadata.get("tokenizer.ggml.tokens")
    length = tokens["length"] if isinstance(tokens, Mapping) else len(tokens or ())
    if length != BASE_VOCAB + ADDED or length != unpadded_vocab:
        raise Inkling2Error(f"GGUF lists {length} tokens; expected {BASE_VOCAB + ADDED} (the unpadded vocabulary)")
    if metadata.get("tokenizer.ggml.eos_token_id") != EOS[0] or metadata.get("tokenizer.ggml.padding_token_id") != PAD[0]:
        raise Inkling2Error("GGUF EOS/padding ids are not Inkling-2's")
    if metadata.get("tokenizer.ggml.add_bos_token") is not False:
        raise Inkling2Error("Inkling-2 adds no BOS token")
