"""Copy-heavy agent replay against a running ninfer-serve.

    python -m tools.bench.agent_replay run --out DIR [--request-log FILE] ...
    python -m tools.bench.agent_replay export --run DIR --tokenizer-dir DIR

See tools/bench/README.md ("Copy-heavy agent replay").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.agent_replay.run import RunConfig, run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _pair(text: str) -> tuple[int, int]:
    low, _, high = text.partition(",")
    values = (int(low), int(high or low))
    if values[0] <= 0 or values[1] < values[0]:
        raise argparse.ArgumentTypeError("expected LOW,HIGH with 0 < LOW <= HIGH")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.bench.agent_replay",
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("run", help="replay sessions against a running server")
    replay.add_argument("--base-url", default="http://127.0.0.1:8080")
    replay.add_argument("--port", type=int, help="shorthand for --base-url http://127.0.0.1:PORT")
    replay.add_argument("--api-key")
    replay.add_argument("--repo", type=Path, default=REPO_ROOT,
                        help="git repository the observations are read from")
    replay.add_argument("--commit", default="HEAD",
                        help="corpus commit; resolved to a hash and recorded (default HEAD)")
    replay.add_argument("--seed", type=int, default=20261010)
    replay.add_argument("--sessions", type=int, default=4)
    replay.add_argument("--concurrency", type=int, choices=(1, 2), default=1,
                        help="sessions in flight at once")
    replay.add_argument("--min-context", type=int, default=30_000,
                        help="lower bound of each session's seeded target prompt size")
    replay.add_argument("--max-context", type=int, default=120_000,
                        help="upper bound; keep it below the server's --max-context")
    replay.add_argument("--obs-tokens", type=_pair, default=(1500, 6000),
                        help="LOW,HIGH approximate tokens per exploratory observation")
    replay.add_argument("--max-copy-tokens", type=int, default=6000,
                        help="output budget cap of one copy turn; bounds copied file size")
    replay.add_argument("--max-turns", type=int, default=200, help="per session")
    replay.add_argument("--timeout-seconds", type=float, default=1800.0)
    replay.add_argument("--request-log", type=Path,
                        help="the server's --request-log-jsonl file: merges speculative counters, "
                             "server timings and output token IDs by response id")
    replay.add_argument("--out", type=Path, required=True, help="output directory")

    exporter = commands.add_parser("export", help="dump prompt+output token streams of a run")
    exporter.add_argument("--run", type=Path, required=True, help="a `run --out` directory")
    source = exporter.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path,
                        help="the served .ninfer; its tokenizer and chat template are used")
    source.add_argument("--tokenizer-dir", type=Path,
                        help="directory with tokenizer.json and chat_template.jinja instead")
    exporter.add_argument("--out", type=Path, help="default: RUN/token_streams.jsonl")

    args = parser.parse_args()
    if args.command == "export":
        from tools.bench.agent_replay.export import export

        return export(args.run, args.artifact, args.tokenizer_dir,
                      args.out or args.run / "token_streams.jsonl")

    if args.min_context > args.max_context:
        parser.error("--min-context must not exceed --max-context")
    base_url = f"http://127.0.0.1:{args.port}" if args.port else args.base_url
    return run(RunConfig(
        base_url=base_url, api_key=args.api_key, repo=args.repo, commit=args.commit,
        seed=args.seed, sessions=args.sessions, concurrency=args.concurrency,
        min_context=args.min_context, max_context=args.max_context, obs_tokens=args.obs_tokens,
        max_copy_tokens=args.max_copy_tokens, max_turns=args.max_turns,
        timeout_seconds=args.timeout_seconds, request_log=args.request_log, out=args.out,
    ))


if __name__ == "__main__":
    sys.exit(main())
