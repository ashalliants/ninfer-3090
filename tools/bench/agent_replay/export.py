"""Export prompt and output token streams of a replay run for offline proposer simulation.

ninfer-serve logs the committed output token IDs of every request (`--request-log-jsonl`,
`request_done.result.generated_token_ids`) but not the prompt IDs, and it has no tokenize
endpoint. The prompt of each turn is therefore re-rendered here from the recorded transcript
with the artifact's own chat template and tokenizer (read from its `resource/text/*` objects),
and its length is checked against the server's `prompt_tokens`. Unlike the replay itself this
needs `jinja2` and `tokenizers` (tools/bench/requirements.txt).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

TOKENIZER_RESOURCE = "resource/text/tokenizer.json"
TEMPLATE_RESOURCE = "resource/text/chat_template.jinja"
CONFIG_RESOURCE = "resource/text/tokenizer_config.json"
# Template globals the server derives from tokenizer_config.json (frontend.cpp).
_TEMPLATE_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token",
                        "cls_token", "mask_token")


def _load_resources(artifact: Path | None,
                    tokenizer_dir: Path | None) -> tuple[str, str, dict[str, Any]]:
    """tokenizer.json (with tokenizer_config.json's added tokens merged), template, template tokens."""
    if artifact is not None:
        from tools.artifact.reader import Artifact

        with Artifact(artifact) as model:
            tokenizer, template, config = (model.read_object(name).decode("utf-8") for name in
                                           (TOKENIZER_RESOURCE, TEMPLATE_RESOURCE, CONFIG_RESOURCE))
    else:
        assert tokenizer_dir is not None
        tokenizer, template, config = ((tokenizer_dir / name).read_text(encoding="utf-8") for name in
                                       ("tokenizer.json", "chat_template.jinja",
                                        "tokenizer_config.json"))
    config = json.loads(config)
    return _merge_added_tokens(tokenizer, config), template, _template_tokens(config)


def _merge_added_tokens(tokenizer_json: str, config: dict[str, Any]) -> str:
    """Add `added_tokens_decoder` entries missing from tokenizer.json, as server startup does."""
    root = json.loads(tokenizer_json)
    added = root.setdefault("added_tokens", [])
    known = {token["id"] for token in added}
    for key, entry in (config.get("added_tokens_decoder") or {}).items():
        if int(key) not in known:
            added.append({"id": int(key), "content": entry["content"],
                          "single_word": entry.get("single_word", False),
                          "lstrip": entry.get("lstrip", False),
                          "rstrip": entry.get("rstrip", False),
                          "normalized": entry.get("normalized", False),
                          "special": entry.get("special", False)})
    return json.dumps(root, ensure_ascii=False)


def _template_tokens(config: dict[str, Any]) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    for key in _TEMPLATE_TOKEN_KEYS:
        value = config.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            tokens[key] = value
    if "additional_special_tokens" in config:
        tokens["additional_special_tokens"] = config["additional_special_tokens"]
    return tokens


def _compile_template(source: str) -> Any:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
               separators: Any = None, sort_keys: bool = False) -> str:
        """The server's filter (third_party/llama-jinja value.cpp): positional order is
        ensure_ascii, indent, separators, sort_keys; UTF-8 and ", "/": " unless told otherwise."""
        if separators is not None:
            separators = tuple(separators)
            if len(separators) == 1:
                separators += (": ",)
        return json.dumps(value, ensure_ascii=bool(ensure_ascii), indent=indent,
                          separators=separators, sort_keys=bool(sort_keys))

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(source)


def _server_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """The tool definition as ninfer-serve hands it to the template (src/serve/translate.cpp)."""
    function = tool["function"]
    rendered = {"name": function["name"], "parameters": function.get("parameters", {}),
                "strict": bool(function.get("strict", False))}
    if function.get("description"):
        rendered["description"] = function["description"]
    return {"type": "function", "function": rendered}


class _PromptEncoder:
    """Encode a rendered prompt the way the server does.

    Special tokens written by the template are control tokens, but message text that merely
    spells one (a source file quoting `<|im_end|>`) is ordinary text to the server. Message strings
    are therefore protected with private-use placeholders before rendering, and those spans are
    encoded with special-token matching off.
    """

    _PLACEHOLDER = re.compile("(\\d+)")

    def __init__(self, tokenizer_json: str) -> None:
        from tokenizers import Tokenizer

        self._control = Tokenizer.from_str(tokenizer_json)
        self._plain = Tokenizer.from_str(tokenizer_json)
        self._plain.encode_special_tokens = True  # i.e. special-token strings are plain text
        specials = sorted((t["content"] for t in json.loads(tokenizer_json).get("added_tokens", [])
                           if t.get("special")), key=len, reverse=True)
        self._special = re.compile("|".join(map(re.escape, specials))) if specials else None
        self._literals: list[str] = []

    def protect(self, text: str, *, trimmed: bool) -> str:
        """Replace a string that spells a special token by a placeholder for the whole string.

        `trimmed` applies the template's `|trim` up front, since the placeholder hides the
        string's own leading and trailing whitespace from the template.
        """
        if self._special is None or not self._special.search(text):
            return text
        self._literals.append(text.strip() if trimmed else text)
        return f"{len(self._literals) - 1}"

    def encode(self, rendered: str) -> list[int]:
        ids: list[int] = []
        position = 0
        for match in self._PLACEHOLDER.finditer(rendered):
            ids += self._control.encode(rendered[position:match.start()],
                                        add_special_tokens=False).ids
            ids += self._plain.encode(self._literals[int(match.group(1))],
                                      add_special_tokens=False).ids
            position = match.end()
        ids += self._control.encode(rendered[position:], add_special_tokens=False).ids
        return ids

    def encode_text(self, text: str) -> list[int]:
        return self._plain.encode(text, add_special_tokens=False).ids


def _template_messages(messages: list[dict[str, Any]],
                       protect: Callable[..., str]) -> list[dict[str, Any]]:
    """OpenAI wire messages carry tool-call arguments as JSON text; templates want mappings."""
    out = []
    for message in messages:
        message = dict(message)
        for key in ("content", "reasoning_content"):
            if isinstance(message.get(key), str):
                message[key] = protect(message[key], trimmed=True)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                try:
                    arguments = json.loads(function["arguments"])
                except (json.JSONDecodeError, TypeError):
                    arguments = function["arguments"]
                if isinstance(arguments, dict):
                    arguments = {k: protect(v, trimmed=False) if isinstance(v, str) else v
                                 for k, v in arguments.items()}
                function["arguments"] = arguments
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        out.append(message)
    return out


def _output_text(message: dict[str, Any]) -> str:
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        parts.append(call["function"]["arguments"])
    return "\n".join(p for p in parts if p)


def export(run_dir: Path, artifact: Path | None, tokenizer_dir: Path | None,
           out_path: Path) -> int:
    tokenizer_json, template_source, template_tokens = _load_resources(artifact, tokenizer_dir)
    encoder = _PromptEncoder(tokenizer_json)
    template = _compile_template(template_source)
    samples = {(s["session"], s["turn"]): s for s in
               map(json.loads, (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines())}
    ids_path = run_dir / "output_token_ids.jsonl"
    output_ids: dict[tuple[int, int], list[int]] = {}
    if ids_path.exists():
        for line in ids_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            output_ids[(record["session"], record["turn"])] = record["generated_token_ids"]

    turns = matched = logged = 0
    worst = 0
    unverified: list[str] = []  # prompt length missing or different from the server's
    unlogged: list[str] = []  # outputs retokenized although a request log was supplied
    with out_path.open("w", encoding="utf-8") as out:
        for line in (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines():
            transcript = json.loads(line)
            messages = _template_messages(transcript["messages"], encoder.protect)
            tools = [_server_tool(tool) for tool in transcript["tools"]]
            for entry in transcript["turns"]:
                key = (transcript["session"], entry["turn"])
                sample = samples.get(key) or {}
                count = entry["message_count"]
                prompt = template.render(messages=messages[:count], tools=tools, **template_tokens,
                                         add_generation_prompt=True, enable_thinking=False)
                prompt_ids = encoder.encode(prompt)
                failed = sample.get("error") is not None
                if key in output_ids:
                    ids, source = output_ids[key], "request_log"
                    logged += 1
                elif count < len(transcript["messages"]):
                    if ids_path.exists():
                        unlogged.append(f"session {key[0]} turn {key[1]}")
                    text = _output_text(transcript["messages"][count])
                    ids, source = encoder.encode_text(text), "retokenized_text"
                else:
                    ids, source = [], "missing"
                server = sample.get("prompt_tokens")
                turns += 1
                if server == len(prompt_ids):
                    matched += 1
                elif not failed:
                    unverified.append(f"session {key[0]} turn {key[1]}: server "
                                      f"{server if server else 'missing'} != rendered "
                                      f"{len(prompt_ids)}")
                if server:
                    worst = max(worst, abs(server - len(prompt_ids)))
                out.write(json.dumps({
                    "session": key[0], "turn": key[1], "category": entry["category"],
                    "copy": sample.get("copy"), "prompt_tokens_server": server,
                    "prompt_ids": prompt_ids, "output_ids": ids, "output_source": source,
                }) + "\n")
    report = {"turns": turns, "prompt_length_matches_server": matched,
              "max_prompt_length_difference": worst, "output_ids_from_request_log": logged,
              "out": str(out_path)}
    print(json.dumps(report, indent=2))
    problems = [f"prompt length not reproduced ({len(unverified)}): " + "; ".join(unverified[:5])
                if unverified else "",
                f"output not in the request log ({len(unlogged)}): " + "; ".join(unlogged[:5])
                if unlogged else ""]
    if any(problems):
        out_path.unlink(missing_ok=True)  # an inexact stream must not pass for an exact one
        for problem in filter(None, problems):
            print(f"error: {problem}; stream file removed", file=sys.stderr)
        return 1
    return 0 if turns else 1
