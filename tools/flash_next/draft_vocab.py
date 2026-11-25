#!/usr/bin/env python3
"""Write Flash-Next's tokenizer-derived MTP draft vocabulary as a file of int32 token ids.

  python tools/flash_next/draft_vocab.py TOKENIZER_JSON OUT.bin [TOKENIZER_CONFIG_JSON]

ninfer-serve builds the same list itself from the artifact's tokenizer (--mtp-draft-vocab auto on a
Tesla V100, or tokenizer); this script writes it out for inspection or for another engine. The rule
(models::qwen4_exp::tokenizer_draft_vocab): every id below 65,536, the ids in [65,536, 120,000)
whose bytes are CJK text, and every added token; sorted, distinct, padded to an even count by
repeating the last id. The added tokens are tokenizer.json's added_tokens plus, when
TOKENIZER_CONFIG_JSON is given, the ids of its added_tokens_decoder, which the runtime's tokenizer
merges in as well (pass it whenever the artifact was converted with one, or a token only it adds is
missing from the list).
"""
import json
import struct
import sys

COMMON_IDS = 65536
CJK_END = 120000


def byte_decoder():
    """The GPT-2 byte-level BPE alphabet, character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def is_cjk(data: bytes) -> bool:
    """draft_vocab_cjk_bytes: a lead byte E3-E9, a whole U+FF00-U+FFEF or U+AC00-U+D7AF character,
    or only continuation bytes."""
    if not data:
        return False
    continuation_only = True
    i = 0
    while i < len(data):
        b = data[i]
        if 0xE3 <= b <= 0xE9:
            return True
        if b & 0xC0 != 0x80:
            continuation_only = False
        if 0xE0 <= b <= 0xEF and i + 2 < len(data):
            b1, b2 = data[i + 1], data[i + 2]
            if b1 & 0xC0 == 0x80 and b2 & 0xC0 == 0x80:
                cp = ((b & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
                if 0xFF00 <= cp <= 0xFFEF or 0xAC00 <= cp <= 0xD7AF:
                    return True
                i += 3
                continue
        i += 1
    return continuation_only


def draft_vocab(tokenizer: dict, tokenizer_config: dict | None = None) -> list:
    vocab = tokenizer["model"]["vocab"]
    added = [t["id"] for t in tokenizer.get("added_tokens", [])]
    if tokenizer_config:
        added += [int(k) for k in tokenizer_config.get("added_tokens_decoder", {})]
    public = max(list(vocab.values()) + added) + 1
    dec = byte_decoder()
    by_id = {i: s for s, i in vocab.items()}
    ids = set(range(min(public, COMMON_IDS)))
    for i in range(COMMON_IDS, min(public, CJK_END)):
        s = by_id.get(i)
        if s is not None and is_cjk(bytes(dec[ch] for ch in s)):
            ids.add(i)
    ids.update(i for i in added if 0 <= i < public)
    out = sorted(ids)
    if len(out) % 2:
        out.append(out[-1])
    return out


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        tokenizer = json.load(f)
    config = None
    if len(sys.argv) == 4:
        with open(sys.argv[3], encoding="utf-8") as f:
            config = json.load(f)
    ids = draft_vocab(tokenizer, config)
    with open(sys.argv[2], "wb") as f:
        f.write(struct.pack("<%di" % len(ids), *ids))
    print(f"{sys.argv[2]}: {len(ids)} ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
