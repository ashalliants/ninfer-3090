#!/usr/bin/env python3
# Written for NInfer-3090 (Inkling-2-Mini bring-up).
"""Goldens for the C++ Inkling-2 tokenizer, from the Hugging Face tokenizers library and the checkpoint's tokenizer.json.

    python -m tools.inkling2.tokenizer_golden TOKENIZER_JSON \\
        --pretokenizer tests/fixtures/inkling2/pretokenizer_golden.json \\
        --encode tests/fixtures/inkling2/tokenizer_encode_golden.json

TOKENIZER_JSON is the checkpoint's own file (26 MB; its pipeline is the one tools/convert/inkling2_tokenizer.py
checks). Needs `tokenizers`. The cases are DeepSeek-V4's (tools/dsv4/tokenizer_golden.py: every script, digits,
marks, emoji, exotic spaces) plus what this pipeline's word regex singles out: case runs (``HTTPServer``,
``camelCase``, ``ÀÉÎõ``), the contractions in either case after a word, ``\\p{N}{1,3}`` runs, punctuation runs
followed by ``/`` and newlines, and Inkling-2's chat markers. ``tests/models/qwen3_5/test_inkling2_tokenizer.cpp``
compares the C++ pre-tokenizer with the first, and, when NINFER_INKLING2_TOKENIZER_JSON points at the same file, the
whole encoder (BPE with ``ignore_merges``) and decode with the second.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tools.dsv4 import tokenizer_golden as dsv4

EXTRA = [
    "HTTPServer", "camelCase", "XMLHttpRequest", "iPhone", "McDonald's", "I'M", "WE'LL", "they're", "You'VE", "it'S",
    "don't", "x'd", "A'm", "'s", "''s", "ÀÉÎõ", "ÉtéÉté", "ǅungla", "ǈ", "ʰello", "aʰb", "日本Abc", "Ωmega", "ΣΑΣ σας",
    "1234567", "a1b22c333d4444", "١٢٣٤", "ⅫⅬ", "x/y", "a//b\n\nc", "!!/\n/", " .../\r\n", "--/", "end.\n", "-> /", "<|end|>",
    "foo\n\n\n  bar", "  \n\t x", "trailing \n", "tab\tsep", "é", "́abc", "áB́c", "x‍y",
    "<|message_user|>Hi<|end_message|><|message_model|><|content_thinking|>", "<|content_model_end_sampling|>",
    "The capital of France is", "Le français, c'est très bien.",
    "it'\u017f", "IT'\u017f", "it'\u212a", "x'\u0131", "I'\u017fx",
]


def pretokenizer_cases(tokenizer_json: Path) -> tuple[list[str], list]:
    from tokenizers import Regex, pre_tokenizers

    stages, cases, _ = dsv4.pretokenizer_cases(tokenizer_json)
    (pattern,) = stages
    pre = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
    random.seed(11)
    cased = "aBcDÀéÎõǅʰ日Ω'sSTtRrEeVvMmLlDd 1/\n."
    extra = EXTRA + ["".join(random.choice(cased) for _ in range(random.randint(1, 24))) for _ in range(200)]
    for text in extra:
        pieces = [piece for piece, _ in pre.pre_tokenize_str(text)]
        assert "".join(pieces) == text, (text, pieces)
        cases.append([text, pieces])
    return stages, cases


def encode_cases(tokenizer_json: Path, cases: list) -> list:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    random.seed(9)
    texts = [c[0] for c in cases[:70]] + [c[0] for c in random.sample(cases[70:], 150)] + EXTRA
    texts += [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "今天天气很好，我们去公园散步吧。2024年11月19日，共计1,234,567人参加。",
        "<|message_system|>You are helpful.<|end_message|><|message_user|>What is 2+2?<|end_message|><|message_model|>",
        "https://huggingface.co/some/model/resolve/main/tokenizer.json?download=true#frag",
        "Привет, мир! مرحبا بالعالم שלום עולם 안녕하세요 สวัสดี",
        "😀🎉👍🏽 emoji and zero-width​joiner and nbsp here",
        # the llama.cpp golden's prompts (docs/maintainer/inkling2-model.md "Validation"); llama.cpp gives the same ids
        'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
        "<|message_user|>Explain in one sentence why the sky is blue.<|end_message|><|message_model|>",
        "Il était une fois, dans un petit village au bord de la mer, une vieille femme qui",
        "x" * 300, "1234567890" * 20, "word " * 120, "a\n" * 80, "Antidisestablishmentarianism" * 3,
    ]
    return [[text, tok.encode(text, add_special_tokens=False).ids] for text in texts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("--pretokenizer", type=Path, required=True)
    parser.add_argument("--encode", type=Path, required=True)
    args = parser.parse_args()
    stages, cases = pretokenizer_cases(args.tokenizer_json)
    args.pretokenizer.write_text(json.dumps({"stages": stages, "cases": cases}, ensure_ascii=False, separators=(",", ":")))
    rows = encode_cases(args.tokenizer_json, cases)
    args.encode.write_text(json.dumps({"count": len(rows), "cases": rows}, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(cases)} pre-tokenizer cases, {len(rows)} encode cases")


if __name__ == "__main__":
    main()
