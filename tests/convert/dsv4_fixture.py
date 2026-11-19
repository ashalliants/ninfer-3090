"""A tiny synthetic deepseek41 GGUF with every tensor family, for converter tests.

The dimensions are scaled down (hidden 256 so the experts can use 256-value block types) but the
structure is the real model's: layers that differ in what they add (a compressor with and without a
gate, the indexer's two sides, Engram), a mix of expert record pairs across layers, and every
tensor name and shape derived from the same config the converter reads. Shapes come from
``dsv4.expected_shapes``, so a change to the architecture's rules changes this fixture with it.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

from tools.convert import dsv4

from .gguf_writer import tensor_bytes, write_gguf

H, LAYERS, EXPERTS, INTER, HC = 256, 5, 4, 256, 2
VOCAB = 16
F32, Q8_0, IQ4_NL, BF16 = 0, 8, 20, 30
IQ2_XXS, IQ2_XS, IQ1_M = 16, 17, 29
# (gate/up, down) types of each layer's experts: five of the pairs of the real IQ2_XXS build.
EXPERT_TYPES = (
    (IQ1_M, IQ1_M),
    (IQ2_XXS, IQ1_M),
    (IQ2_XXS, IQ2_XS),
    (IQ2_XS, IQ2_XXS),
    (IQ2_XS, IQ2_XS),
)
ENGRAM_LAYERS, KV_LAYERS, INDEXER_LAYERS = [1], [2, 4], [2, 3, 4]
COMPRESS_RATIOS = [0, 0, 2, 0, 1]  # layer 2 has a gated compressor, layer 4 an ungated one
NGRAM, ENGRAM_HEADS, ENGRAM_ROW = 3, 2, 32
HEAD_VOCAB = [101, 103, 107, 109]  # one Engram layer x 2 heads x (3 - 1) orders
ENGRAM_ROWS = sum(HEAD_VOCAB)

# A tiny tokenizer shaped like the real one: the first control tokens (BOS, EOS, pad) are in the vocabulary AND the
# added tokens; further added tokens follow the 12-token base vocabulary. Tokens are byte-level strings.
TOKENS = ["<bos>", "<eos>", "<pad>", "a", "b", "c", "d", "e", "f", "g", "h", "ab", "<tool>", "<end>", "<think>", "<x>"]
TOKEN_TYPES = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 3, 3]
BASE_VOCAB = 12
MERGES = ["a b"]
CHAT_TEMPLATE = "{{ messages }}"
assert len(TOKENS) == VOCAB == len(TOKEN_TYPES)

# What tokenizer.json's pre_tokenizer must be (tools.convert.dsv4_tokenizer.PIPELINE), spelled out here so the
# fixture does not import the code it tests.
_REGEX = (
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
)
PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": False},
        {"type": "Split", "pattern": {"Regex": "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"}, "behavior": "Isolated", "invert": False},
        {"type": "Split", "pattern": {"Regex": _REGEX}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}


def official_tokenizer(**changes) -> dict:
    """The checkpoint's tokenizer.json for the tiny tokenizer: BOS/EOS/pad in the vocabulary and the added tokens,
    <tool> an ordinary (not special) added token flagged normalized as 47 of the real file's are."""
    def added(index, **flags):
        return {"id": index, "content": TOKENS[index], "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": True, **flags}

    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [added(0), added(1), added(2), added(12, special=False, normalized=True), added(13), added(14), added(15)],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": PRE_TOKENIZER,
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": {TOKENS[i]: i for i in range(BASE_VOCAB)}, "merges": list(MERGES)},
    }
    tokenizer.update(changes)
    return tokenizer


def _value(value):
    if isinstance(value, bool):
        return 7, value
    if isinstance(value, int):
        return 4, value
    if isinstance(value, float):
        return 6, value
    if isinstance(value, str):
        return 8, value
    if value and all(isinstance(v, str) for v in value):
        return 9, (8, value)
    return 9, ((6, value) if any(isinstance(v, float) for v in value) else (4, value))


def metadata(**changes) -> list[tuple[str, int, object]]:
    a = "deepseek41."
    items = {
        "general.architecture": "deepseek41",
        a + "block_count": LAYERS,
        a + "context_length": 4096,
        a + "embedding_length": H,
        a + "attention.head_count": 4,
        a + "attention.head_count_kv": 1,
        a + "rope.scaling.type": "yarn",
        a + "rope.scaling.factor": 16.0,
        a + "rope.scaling.original_context_length": 256,
        a + "rope.scaling.yarn_beta_fast": 32.0,
        a + "rope.scaling.yarn_beta_slow": 1.0,
        a + "rope.freq_base": 10000.0,
        a + "attention.layer_norm_rms_epsilon": 1e-6,
        a + "expert_count": EXPERTS,
        a + "expert_used_count": 2,
        a + "expert_gating_func": 4,
        a + "attention.key_length": 32,
        a + "attention.value_length": 32,
        a + "rope.dimension_count": 16,
        a + "attention.q_lora_rank": 32,
        a + "attention.sliding_window": 8,
        a + "expert_feed_forward_length": INTER,
        a + "expert_shared_count": 1,
        a + "expert_weights_scale": 1.5,
        a + "expert_weights_norm": True,
        a + "swiglu_clamp_exp": [10.0] * LAYERS,
        a + "swiglu_clamp_shexp": [10.0] * LAYERS,
        a + "attention.indexer.head_count": 2,
        a + "attention.indexer.key_length": 32,
        a + "attention.indexer.top_k": 4,
        a + "attention.output_group_count": 2,
        a + "attention.output_lora_rank": 16,
        a + "attention.compress_ratios": COMPRESS_RATIOS + [0, 0, 0],  # the MTP layers' entries
        a + "attention.compress_rope_freq_base": 160000.0,
        a + "hyper_connection.count": HC,
        a + "hyper_connection.sinkhorn_iterations": 20,
        a + "hyper_connection.epsilon": 1e-6,
        a + "hash_layer_count": 0,
        a + "attention.kv_source_layers": KV_LAYERS,
        a + "attention.indexer.source_layers": INDEXER_LAYERS,
        a + "ple.layers": ENGRAM_LAYERS,
        a + "ple.ngram_size": NGRAM,
        a + "ple.heads_per_ngram": ENGRAM_HEADS,
        a + "embedding_length_per_layer_input": ENGRAM_ROW,
        a + "ple.head_vocab_sizes": HEAD_VOCAB,
        a + "engram.pad_token_id": 2,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "joyai-llm",
        "tokenizer.ggml.tokens": TOKENS,
        "tokenizer.ggml.token_type": TOKEN_TYPES,
        "tokenizer.ggml.merges": MERGES,
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.chat_template": CHAT_TEMPLATE,
    }
    items.update({a + k.replace("__", "."): v for k, v in changes.items()})
    return [(key, *_value(value)) for key, value in items.items() if value is not None]


def _type_of(suffix: str) -> int:
    if suffix == "ffn_gate_inp.weight":
        return BF16
    if suffix.endswith(("norm.weight", "sinks.weight", ".bias", "_fn.weight", "_base.weight", "_scale.weight")) or suffix in (
        "engram_gate.weight",
    ):
        return F32
    return Q8_0


def config() -> dict:
    return dsv4.text_config({k: v for k, _, v in _plain(metadata())}, VOCAB)


def _plain(items):
    for key, kind, value in items:
        yield key, kind, (value[1] if kind == 9 else value)


def tensors(drop=(), extra=(), experts=EXPERT_TYPES) -> list[tuple[str, int, tuple[int, ...], bytes]]:
    cfg = config()
    shapes = dsv4.expected_shapes(cfg)
    specs = [
        ("token_embd.weight", Q8_0, (VOCAB, H)),
        ("output.weight", Q8_0, (VOCAB, H)),
        ("output_norm.weight", F32, (H,)),
        (dsv4.ENGRAM_TABLE, IQ4_NL, (ENGRAM_ROWS, ENGRAM_ROW)),
    ]
    for layer in range(LAYERS):
        for suffix in dsv4.layer_suffixes(cfg, layer):
            specs.append((f"blk.{layer}.{suffix}", _type_of(suffix), shapes[suffix]))
        gate_up, down = experts[layer]
        specs += [
            (f"blk.{layer}.ffn_gate_exps.weight", gate_up, (EXPERTS, INTER, H)),
            (f"blk.{layer}.ffn_up_exps.weight", gate_up, (EXPERTS, INTER, H)),
            (f"blk.{layer}.ffn_down_exps.weight", down, (EXPERTS, H, INTER)),
        ]
    specs += list(extra)
    out = []
    for index, (name, kind, shape) in enumerate(s for s in specs if s[0] not in drop):
        ne = tuple(reversed(shape))
        out.append((name, kind, ne, tensor_bytes(kind, ne, seed=index)))
    return out


def write(path: Path, *, metadata_changes=None, drop=(), extra=(), experts=EXPERT_TYPES) -> Path:
    return write_gguf(path, metadata(**(metadata_changes or {})), tensors(drop, extra, experts))
