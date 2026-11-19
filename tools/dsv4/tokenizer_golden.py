#!/usr/bin/env python3
"""Goldens for the C++ DeepSeek-V4 tokenizer, from the Hugging Face tokenizers library and the checkpoint's tokenizer.json.

    python tools/dsv4/tokenizer_golden.py TOKENIZER_JSON --pretokenizer tests/fixtures/dsv4/pretokenizer_golden.json \\
        --encode tests/fixtures/dsv4/tokenizer_encode_golden.json

TOKENIZER_JSON is the checkpoint's own file (deepseek-ai/DeepSeek-V4.1-Flash-1017, 6 MB). Needs `tokenizers`.

- the pre-tokenizer golden: ~1,200 strings (ASCII punctuation and whitespace mixes, digits of every script, CJK and
  kana, accents and combining marks, emoji, exotic spaces such as U+0085 and U+2028) and the pieces the three Split
  stages cut each into;
- the encode golden: ~200 strings (some of the first, natural text, code, tool-call markup with the added tokens)
  and the token ids the whole tokenizer gives them.

`tests/models/qwen3_5/test_dsv4_tokenizer.cpp` compares the C++ pre-tokenizer with the first, and, when
NINFER_DSV4_TOKENIZER_JSON points at the same file, the whole encoder (and decode) with the second.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def pretokenizer_cases(tokenizer_json: Path) -> tuple[list[str], list, list[str]]:
    from tokenizers import Regex, pre_tokenizers

    official = json.loads(tokenizer_json.read_text())["pre_tokenizer"]
    stages = [p for p in official["pretokenizers"] if p["type"] == "Split"]
    pre = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(p["pattern"]["Regex"]), behavior="isolated", invert=False) for p in stages]
    )
    random.seed(4)
    alphabets = {
        "ascii": "abcXYZ019 \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
        "mixed": "aB3 ,.!\n\n  \t",
        "cjk": "你好世界中文日本語ひらがなカタカナ゛゜・ー々〆〇",
        "latin": "éàüñçßøåÅÉ ʼ’“”«»—–…",
        "marks": "éä̀ ñ กิ้",
        "other": "Привет мир العربية עברית हिन्दी ไทย 한국어",
        "num": "0123456789٠١٢٣¹²³½ⅣⅤ１２３",
        "sym": "©®™€£¥§¶•°±×÷∑∫√∞≠≤≥←→↑↓★☆♠♣♥♦☀☁😀🎉👍🏽",
        "space": " \t\n\r\u00a0\u0085\u2028\u2029\u200b\u3000\u2003",
        "fullwidth": "，。！？：；（）【】、ＡＢＣａｂｃ",
    }
    fixed = [
        "", " ", "a", "Hello, world!", "  leading", "trailing  ", "a  b", "a \n b", "\n\n\n", " \n ", "\r\n", "x\r\ny",
        "12345678", "abc123def", "1,234.56", "$100", "3.14159", "２０２４年", "你好,世界!", "こんにちは世界", "カタカナとひらがな",
        "foo_bar", "foo-bar", "!abc", "!!abc", "(hello)", "a.b.c", "http://example.com/path?q=1#x", "x = y + 1;",
        "def f(x):\n    return x**2\n", "\tTabbed", "end.\n\n\nNext", "naïve café", "é", "😀😀", "don't stop", "I'M HERE",
        "日本語123テスト", "中文，English，混合", "a\u3000b", "a\u00a0b", "a\u0085b", "ab\u2028cd", "......", "?!", "~~~", "a~b", "_x", "__init__",
        "<｜begin▁of▁sentence｜>hi", "<|EOT|>", "  \n  x", "x  \n\n  y", "1 2 3", "1\n2", " 12", "12 ", "٣٤٥٦", "①②③",
    ]
    cases = list(fixed)
    for name, alpha in alphabets.items():
        for _ in range(60):
            cases.append("".join(random.choice(alpha) for _ in range(random.randint(1, 24))))
    allchars = "".join(alphabets.values())
    for _ in range(500):
        cases.append("".join(random.choice(allchars) for _ in range(random.randint(1, 40))))
    out = []
    for text in cases:
        pieces = [piece for piece, _ in pre.pre_tokenize_str(text)]
        assert "".join(pieces) == text, (text, pieces)
        out.append([text, pieces])
    return [p["pattern"]["Regex"] for p in stages], out, cases


def encode_cases(tokenizer_json: Path, pretokenizer_cases: list) -> list:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    random.seed(9)
    pre = pretokenizer_cases
    texts = [c[0] for c in pre[:70]]
    texts += [c[0] for c in random.sample(pre[70:], 120)]
    texts += [
        "The quick brown fox jumps over the lazy dog.",
        "DeepSeek-V4.1-Flash runs on a single RTX 3090 with an SSD-tiered expert cache.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "今天天气很好，我们去公园散步吧。2024年11月19日，共计1,234,567人参加。",
        "こんにちは、世界！ひらがなとカタカナと漢字。",
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>Hi there!<｜end▁of▁sentence｜>",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "<|EOT|> text <｜fim▁begin｜>a<｜fim▁hole｜>b<｜fim▁end｜>",
        "  indented\n\n\n    more   spaces\t\ttabs\r\nwindows line",
        "E=mc^2, π≈3.14159, ∑_{i=0}^{n} x_i ≤ 100% — “quoted” ‘text’… ©2024",
        "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/tokenizer.json?download=true#frag",
        "Привет, мир! مرحبا بالعالم שלום עולם 안녕하세요 สวัสดี",
        "😀🎉👍🏽 emoji and zero-width\u200bjoiner and nbsp\u00a0here",
        "é ä composed vs decomposed: é ä",
        "x" * 300,
        "1234567890" * 20,
        "word " * 120,
        "a\n" * 80,
    ]
    rows = []
    for text in texts:
        rows.append([text, tok.encode(text, add_special_tokens=False).ids])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("--pretokenizer", type=Path, required=True)
    parser.add_argument("--encode", type=Path, required=True)
    args = parser.parse_args()
    stages, cases, _ = pretokenizer_cases(args.tokenizer_json)
    args.pretokenizer.write_text(json.dumps({"stages": stages, "cases": cases}, ensure_ascii=False, separators=(",", ":")))
    rows = encode_cases(args.tokenizer_json, cases)
    args.encode.write_text(json.dumps({"count": len(rows), "cases": rows}, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(cases)} pre-tokenizer cases, {len(rows)} encode cases")


if __name__ == "__main__":
    main()
