#!/usr/bin/env python3
"""Do two ninfer CLI builds, or one build with two sets of options, produce the same greedy tokens?

For every prompt in a prompts file, run each build as

    EXE MODEL --prompt TEXT --max-new N --greedy --print-token-ids [EXTRA ARGS]

and compare the generated token ids the CLI prints on stderr (the `tokens  generated ids ...` line).
Prints `GREEDY IDENTICAL` (exit 0), or `GREEDY DIFFERS` with the first differing prompt and token
index (exit 1), stopping at that prompt. Exit 2 means a build failed, printed no ids line, or the
inputs were wrong, so nothing was compared.

Use it for a change that must not move the output: an option that is off by default, a refactor, a
kernel swapped for an equivalent one. Equal ids over 24 prompts is evidence the two builds take the
same numerical path; unequal ids are not necessarily a bug (a different reduction order flips a
near-tie), but the index shows how deep into the answer the paths agree.

    python tools/bench/greedy_identity.py --a build-ninja/apps/ninfer.exe \\
        --b build-ninja/apps/ninfer-new.exe --model models/qwen3_8_27b.ninfer \\
        --tokens 128 --extra-args="--spec mtp --draft-tokens 3"

  --extra-args   one string, split like a shell command line, appended to BOTH builds. Write it with
                 `=` as above, so a value that starts with `--` is not read as another option. Use it
                 for the option under test (the same flag on both builds) and to pin anything that
                 would otherwise differ (`--kv-dtype`, `--max-context`).
  --a-args, --b-args
                 one string each, appended to that side only, after --extra-args. Same-build mode:
                 leave out --b and the one build runs as both sides, A with --a-args and B with
                 --b-args, which must differ; that is how to check that an option meant to keep the
                 output, such as --no-cuda-graph, does:

    python tools/bench/greedy_identity.py --a build-ninja/apps/ninfer.exe \\
        --model models/qwen3_8_27b.ninfer --b-args="--no-cuda-graph"

  --limit N      only the first N prompts (a quick check before the full set).

The prompts file is JSONL with a "prompt" string per line, as for run_chat_decode.py; the default is
bench/fixtures/chat_prompts.jsonl (24 prompts).

Both builds must sit in the same directory, under different names: the preflight is the one
run_interleaved_ab.py uses (an executable copied away from its build fails DLL resolution, exits 127
and reads like a model-loading failure).

GPU: this tool runs the two builds one after the other, on the card, for as long as the prompts
take. Take the GPU lock around the whole invocation (CLAUDE.md, "GPU lock"):

    . tools/gpu-lock.ps1
    Invoke-WithGpuLock { python tools/bench/greedy_identity.py --a ... --b ... --model ... }
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_interleaved_ab as ab  # noqa: E402  (preflight, split_command)

DEFAULT_PROMPTS = HERE.parent.parent / "bench" / "fixtures" / "chat_prompts.jsonl"

# apps/cli/main.cpp prints `std::left << setw(12) "tokens" << setw(26) "generated ids"` and then the ids
# separated by spaces. With no ids the line ends after the padding.
IDS_LINE = re.compile(r"^tokens\s+generated ids\b[ \t]*(.*?)\s*$", re.MULTILINE)


class Failure(Exception):
    """A build failed or the inputs were unusable: nothing was compared."""


def load_prompts(path: Path, limit: int | None) -> list[str]:
    if not path.is_file():
        raise Failure(f"missing prompts file: {path}")
    prompts: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            prompt = json.loads(line)["prompt"]
        except (ValueError, KeyError, TypeError) as error:
            raise Failure(f"{path}:{number}: expected a JSON object with a \"prompt\" string ({error!r})")
        if not isinstance(prompt, str) or not prompt:
            raise Failure(f"{path}:{number}: \"prompt\" must be a non-empty string")
        prompts.append(prompt)
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise Failure(f"no prompts in {path}")
    return prompts


def parse_ids(stderr: str) -> list[int] | None:
    """The generated ids from the CLI's stderr, or None when it printed no such line."""
    match = IDS_LINE.search(stderr)
    if match is None:
        return None
    try:
        return [int(token) for token in match.group(1).split()]
    except ValueError:
        return None


def generate(label: str, exe: Path, model: str, prompt: str, tokens: int, extra: list[str],
             timeout: float) -> list[int]:
    argv = [str(exe), model, "--prompt", prompt, "--max-new", str(tokens), "--greedy",
            "--print-token-ids", *extra]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Failure(f"build {label} ({exe}) did not finish within {timeout:g} s")
    except OSError as error:
        raise Failure(f"build {label} ({exe}) could not start: {error}")
    if done.returncode != 0:
        tail = "\n    ".join(done.stderr.strip().splitlines()[-5:])
        raise Failure(f"build {label} ({exe}) exited {done.returncode}\n    {tail}")
    ids = parse_ids(done.stderr)
    if ids is None:
        raise Failure(f"build {label} ({exe}) printed no 'tokens  generated ids' line; is it a ninfer CLI "
                      "that supports --print-token-ids?")
    return ids


def first_difference(a: list[int], b: list[int]) -> int | None:
    """Index of the first token where the two lists differ (the shorter length when one is a prefix)."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def window(ids: list[int], index: int) -> str:
    shown = ids[index:index + 5]
    if not shown:
        return "[ended]"
    return "[" + " ".join(map(str, shown)) + (" ...]" if index + 5 < len(ids) else "]")


def compare_builds(args: argparse.Namespace, extra_a: list[str], extra_b: list[str],
                   prompts: list[str]) -> int:
    total = 0
    exe_b = args.b if args.b is not None else args.a
    for number, prompt in enumerate(prompts, 1):
        ids_a = generate("A", args.a, args.model, prompt, args.tokens, extra_a, args.timeout)
        ids_b = generate("B", exe_b, args.model, prompt, args.tokens, extra_b, args.timeout)
        total += len(ids_a)
        index = first_difference(ids_a, ids_b)
        label = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        if index is not None:
            print(f"GREEDY DIFFERS: prompt {number}/{len(prompts)} ({label!r}), token index {index} "
                  f"(0-based; A made {len(ids_a)} tokens, B made {len(ids_b)})")
            print(f"  A {window(ids_a, index)}")
            print(f"  B {window(ids_b, index)}")
            return 1
        print(f"  prompt {number}/{len(prompts)}: {len(ids_a)} tokens identical", flush=True)
    print(f"GREEDY IDENTICAL ({len(prompts)} prompts, {total} tokens)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="first ninfer CLI executable")
    parser.add_argument("--b", type=Path, default=None,
                        help="second ninfer CLI executable; omitted, --a runs as both sides (same-build mode)")
    parser.add_argument("--model", required=True, help="model artifact, the first argument of the CLI")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help="JSONL file with a \"prompt\" per line (default: bench/fixtures/chat_prompts.jsonl)")
    parser.add_argument("--tokens", type=int, default=256, help="--max-new for both builds (default 256)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    parser.add_argument("--extra-args", default="",
                        help="extra CLI arguments for both builds, as one string (write --extra-args=\"...\")")
    parser.add_argument("--a-args", default="", help="extra CLI arguments for A only (write --a-args=\"...\")")
    parser.add_argument("--b-args", default="", help="extra CLI arguments for B only (write --b-args=\"...\")")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds allowed per run (default 1800)")
    args = parser.parse_args(argv)

    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    try:
        split = lambda text: ab.split_command(text) if text.strip() else []  # noqa: E731
        extra = split(args.extra_args)
        extra_a, extra_b = extra + split(args.a_args), extra + split(args.b_args)
        if args.b is None:
            if not args.a.is_file():
                raise Failure(f"missing executable: {args.a}")
            if extra_a == extra_b:
                raise Failure("same-build mode (no --b) needs --a-args and --b-args that differ; "
                              "nothing to compare")
        else:
            try:
                ab.preflight(args.a, args.b)
            except SystemExit as stop:
                raise Failure(str(stop.code))
        prompts = load_prompts(args.prompts, args.limit)
        return compare_builds(args, extra_a, extra_b, prompts)
    except (Failure, ValueError) as failure:
        print(f"greedy_identity: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
