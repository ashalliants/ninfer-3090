#!/usr/bin/env python3
"""Ratchet on direct stderr writes under src/: a file's count may only go down.

Operational output goes through the product logger (docs/maintainer/logging.md). A few low-level
places still write straight to stderr on purpose (the emergency diagnostics of section 1: a CUDA
cleanup failure, a logging-sink failure) and a few more name stderr without writing a log line (a
terminal check). tools/stderr_logging.allow records how many such uses each file has today. This
tool fails when

  * a file under src/ that is not in the allowlist names stderr, or
  * a file's count is higher than the allowlist says,

so a new direct write cannot arrive unnoticed. A file whose count went down passes, with a note;
`--update` lowers the allowlist to match (and drops files that reached zero). The ratchet never
rises by itself: a deliberate new emergency diagnostic is added by editing the allowlist by hand
(`--counts` prints the current numbers), which puts the exception in front of the reviewer.

What counts. After comments (`//`, `/* */`) and string and character literals (including raw
strings) are blanked out, each match of

    \\bstd::cerr\\b | \\bstd::clog\\b | \\bcerr\\b | \\bclog\\b | \\bstderr\\b | \\bperror\\s*\\(

is one use. That covers fprintf/fputs/fputc/fwrite/vfprintf/fflush(stderr), `std::cerr <<`,
`std::clog <<` and perror(). Whole words only: `STDERR_FILENO` and spdlog's
`stderr_color_sink_st` are different identifiers and do not count. A `stderr` in a comment or in a
string (a log message that says "stderr") does not count. Known blind spot: a raw `write(2, ...)`,
which nothing in src/ uses.

Files: the C and C++ sources and headers (.c .cc .cpp .cxx .cu .cuh .h .hpp .hxx .inl) that git
tracks under src/ (os.walk of src/ when the root is not a git checkout). Paths are repo-relative with
forward slashes, sorted, so the result is the same on every platform. apps/, tests/ and bench/ print
their results to stderr by design and are not checked.

usage:
  python tools/check_stderr_logging.py              check against tools/stderr_logging.allow
  python tools/check_stderr_logging.py --update     lower the allowlist to the current counts
  python tools/check_stderr_logging.py --counts     print the current counts in allowlist format
Exit 0 when the ratchet holds, 1 when it is broken (or --update found a rise), 2 on bad input.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = "src"
SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hxx", ".inl")
SKIPPED_DIRS = {".git", "build", "build-ninja", "__pycache__"}

USE = re.compile(
    r"\bstd::cerr\b|\bstd::clog\b|\bcerr\b|\bclog\b|\bstderr\b|\bperror\s*\("
)


def blank_comments_and_literals(text: str) -> str:
    """Return `text` with comments and string/char literals replaced by spaces (newlines kept).

    A small C/C++ scanner rather than a regex, because a comment marker inside a string and a quote
    inside a comment both occur in real code. It does not run the preprocessor, so a #if 0 block
    still counts: the ratchet is deliberately conservative about what it sees.
    """
    out: list[str] = []
    i, n = 0, len(text)

    def blank(chunk: str) -> str:
        return "".join(c if c == "\n" else " " for c in chunk)

    while i < n:
        c = text[i]
        two = text[i : i + 2]
        if two == "//":
            j = i
            while j < n and text[j] != "\n":
                # A backslash-newline continues a line comment onto the next line.
                if text[j] == "\\" and j + 1 < n and text[j + 1] == "\n":
                    j += 2
                    continue
                if text[j] == "\\" and text[j + 1 : j + 3] == "\r\n":
                    j += 3
                    continue
                j += 1
            out.append(blank(text[i:j]))
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(blank(text[i:j]))
            i = j
        elif c == '"':
            # Raw string: R"delim( ... )delim", with an optional encoding prefix before the R.
            j = i
            if i > 0 and text[i - 1] == "R":
                k = i - 1
                while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
                    k -= 1
                if text[k:i] in ("R", "LR", "uR", "UR", "u8R"):
                    open_paren = text.find("(", i + 1)
                    delimiter = text[i + 1 : open_paren] if open_paren >= 0 else ""
                    if open_paren >= 0 and len(delimiter) <= 16 and not any(
                        ch in delimiter for ch in ' ()\\\t\n'
                    ):
                        close = text.find(")" + delimiter + '"', open_paren)
                        j = n if close < 0 else close + len(delimiter) + 2
                        out.append(blank(text[i:j]))
                        i = j
                        continue
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(blank(text[i:j]))
            i = j
        elif c == "'":
            # A quote after a number is a digit separator (1'000, 0xFF'FF), not a character literal.
            k = i
            while k > 0 and (text[k - 1].isalnum() or text[k - 1] in "_'"):
                k -= 1
            if k < i and text[k].isdigit():
                out.append(c)
                i += 1
                continue
            j = i + 1
            while j < n and text[j] != "'" and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append(blank(text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def count_uses(text: str) -> int:
    return len(USE.findall(blank_comments_and_literals(text)))


def list_sources(root: Path) -> list[str]:
    """Repo-relative, forward-slash, sorted paths of the source files under src/."""
    paths: list[str] = []
    if (root / ".git").exists():
        try:
            listed = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--", SOURCE_DIR],
                check=True, capture_output=True,
            ).stdout.decode("utf-8")
        except (OSError, subprocess.CalledProcessError) as error:
            sys.exit(f"check_stderr_logging: git ls-files failed: {error}")
        paths = [p for p in listed.split("\0") if p]
    else:
        base = root / SOURCE_DIR
        for directory, names, files in os.walk(base):
            names[:] = [d for d in names if d not in SKIPPED_DIRS]
            for name in files:
                paths.append(Path(directory, name).relative_to(root).as_posix())
    return sorted(p for p in paths if p.endswith(SUFFIXES) and (root / p).is_file())


def count_tree(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in list_sources(root):
        text = (root / path).read_text(encoding="utf-8", errors="replace")
        count = count_uses(text)
        if count:
            counts[path] = count
    return counts


def read_allowlist(path: Path) -> tuple[list[str], dict[str, int]]:
    """The leading comment lines (kept on --update) and the {path: count} entries."""
    header: list[str] = []
    entries: dict[str, int] = {}
    if not path.is_file():
        sys.exit(f"check_stderr_logging: allowlist {path} is missing")
    seen_entry = False
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if not seen_entry:
                header.append(raw)
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) < 1:
            sys.exit(f"{path}:{number}: expected '<count> <path>' with a positive count, got {raw!r}")
        if parts[1] in entries:
            sys.exit(f"{path}:{number}: {parts[1]} is listed twice")
        seen_entry = True
        entries[parts[1]] = int(parts[0])
    return header, entries


def format_entries(counts: dict[str, int]) -> str:
    return "".join(f"{counts[path]} {path}\n" for path in sorted(counts))


def compare(counts: dict[str, int], allowed: dict[str, int]) -> tuple[list[str], list[str]]:
    """(failures, notes): a rise or a new file fails; a drop is only worth a note."""
    failures: list[str] = []
    notes: list[str] = []
    for path in sorted(counts):
        have, count = allowed.get(path), counts[path]
        if have is None:
            failures.append(f"{path}: names stderr {count} time(s) and is not in the allowlist")
        elif count > have:
            failures.append(f"{path}: names stderr {count} times, allowed {have}")
        elif count < have:
            notes.append(f"{path}: down to {count} (allowlist says {have})")
    for path in sorted(set(allowed) - set(counts)):
        notes.append(f"{path}: no longer names stderr (allowlist says {allowed[path]})")
    return failures, notes


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=here.parent, help="repository root (default: this script's)")
    parser.add_argument("--allow", type=Path, default=None,
                        help="allowlist file (default: tools/stderr_logging.allow under --root)")
    parser.add_argument("--update", action="store_true", help="lower the allowlist to the current counts")
    parser.add_argument("--counts", action="store_true", help="print the current counts in allowlist format")
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / SOURCE_DIR).is_dir():
        print(f"check_stderr_logging: {root / SOURCE_DIR} is not a directory", file=sys.stderr)
        return 2
    allow_path = args.allow if args.allow else root / "tools" / "stderr_logging.allow"

    counts = count_tree(root)
    if args.counts:
        sys.stdout.write(format_entries(counts))
        return 0

    header, allowed = read_allowlist(allow_path)
    failures, notes = compare(counts, allowed)
    for failure in failures:
        print(f"stderr logging: {failure}", file=sys.stderr)
    if failures:
        print("  Write log lines through the product logger (docs/maintainer/logging.md). A deliberate new\n"
              "  emergency diagnostic is added to the allowlist by hand; --update only lowers counts.",
              file=sys.stderr)
        return 1

    if args.update:
        lowered = {path: counts[path] for path in allowed if path in counts}
        text = "".join(line + "\n" for line in header) + format_entries(lowered)
        allow_path.write_text(text, encoding="utf-8", newline="\n")
        for note in notes:
            print(f"stderr logging: lowered {note}")
        print(f"stderr logging: allowlist updated ({len(lowered)} files, {sum(lowered.values())} uses)")
        return 0

    for note in notes:
        print(f"stderr logging: {note}; run `python tools/check_stderr_logging.py --update` to lower it")
    print(f"stderr logging: ok ({len(counts)} files, {sum(counts.values())} uses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
