"""Deterministic copy-heavy agent observations built from real repository files.

Every observation is a pure function of (corpus commit, seed, session index, turn index): files,
line ranges, diffs and instructions are chosen by seeded `random.Random` instances and read with
`git show <commit>:<path>`, so two runs at the same commit see byte-identical tool results. Only
the assistant turns are free-running model output.

| category        | copy | observation                                   | instruction                      |
|-----------------|------|-----------------------------------------------|----------------------------------|
| `explore`       | no   | numbered/plain Read slice, Grep, or a diff    | none (the model continues)       |
| `explain`       | no   | numbered Read slice                           | prose explanation, no code       |
| `rewrite_file`  | yes  | whole file, `cat -n` numbered Read            | Write the file with a rename     |
| `apply_change`  | yes  | whole file, plain `cat`                       | return the file with one comment |
| `edit_function` | yes  | numbered Read around a 20-60 line block       | Edit with old/new block          |
| `diff_revert`   | yes  | `git show <sha> -- <path>` diff               | write the pre-change hunks       |
"""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COPY_CATEGORIES = ("rewrite_file", "apply_change", "edit_function", "diff_revert")
NON_COPY_CATEGORIES = ("explore", "explain")
CATEGORIES = NON_COPY_CATEGORIES + COPY_CATEGORIES

# One block of turns, shuffled per block: half the turns ask for copies, the rest grow the context
# or ask for prose. Copy turns dominate output tokens because they reproduce whole files.
TURN_BLOCK = (
    "explore", "explore", "explore", "explain",
    "rewrite_file", "apply_change", "edit_function", "diff_revert",
)

SOURCE_PREFIXES = ("src/", "include/", "apps/", "tools/", "tests/")
SOURCE_SUFFIXES = (".cpp", ".h", ".cu", ".cuh", ".py")
REPO_ROOT_NAME = "/repo"

# Rough code density for budgeting observations and output limits; the session loop uses the
# server's reported prompt_tokens for every context decision.
CHARS_PER_TOKEN = 3.3

EXPLORE_MAX_TOKENS = 320
EXPLAIN_MAX_TOKENS = 700

KEYWORDS = frozenset(
    "include return static_cast reinterpret_cast const_cast dynamic_cast noexcept nullptr "
    "template typename namespace constexpr unsigned default private public protected virtual "
    "override explicit continue sizeof struct switch static inline extern import lambda "
    "assert raise except finally yield global nonlocal isinstance".split()
)

SYSTEM_PROMPT = (
    "You are a coding agent working in the NInfer repository checked out at /repo (C++/CUDA "
    "inference engine plus Python tooling). Use the tools to inspect and change files. When you "
    "write or edit a file, pass the complete exact text: keep every unchanged line byte-identical, "
    "including indentation, and never abbreviate with comments such as '... rest unchanged'."
)


def _function(name: str, description: str, properties: dict[str, Any],
              required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOLS: list[dict[str, Any]] = [
    _function("Read", "Read a file. Returns lines in `cat -n` format (6-wide line number, tab, "
              "line text).",
              {"file_path": {"type": "string"}, "offset": {"type": "integer"},
               "limit": {"type": "integer"}}, ["file_path"]),
    _function("Write", "Write the complete contents of a file, replacing it.",
              {"file_path": {"type": "string"}, "content": {"type": "string"}},
              ["file_path", "content"]),
    _function("Edit", "Replace one exact occurrence of old_string with new_string in a file.",
              {"file_path": {"type": "string"}, "old_string": {"type": "string"},
               "new_string": {"type": "string"}}, ["file_path", "old_string", "new_string"]),
    _function("Grep", "Search file contents for a pattern; returns path:line:text rows.",
              {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    _function("Bash", "Run a shell command in /repo and return its output.",
              {"command": {"type": "string"}}, ["command"]),
]

NUMBERED_LINE = re.compile(r"^ *\d+\t", re.MULTILINE)
IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")


def strip_line_numbers(text: str) -> str:
    """Remove `cat -n` prefixes so numbered Read results expose the file text itself."""
    return NUMBERED_LINE.sub("", text)


def pre_image_lines(diff: str) -> int:
    """Context and removed lines of a unified diff: what reverting it reproduces."""
    if "\n@@" not in diff:
        return 0
    body = diff.split("\n@@", 1)[1].split("\n")[1:]  # drop the rest of the first hunk header
    return sum(1 for line in body
               if line.startswith(" ") or (line.startswith("-") and not line.startswith("---")))


def est_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN) + 1


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16)


def numbered(lines: list[str], start: int, stop: int) -> str:
    return "\n".join("%6d\t%s" % (i + 1, lines[i]) for i in range(start, stop))


class Corpus:
    """Source files and history of one pinned commit, read through git."""

    def __init__(self, repo: Path, commit: str) -> None:
        self.repo = repo
        self.commit = self._git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()
        listing = self._git("ls-tree", "-r", "--name-only", self.commit).splitlines()
        self.files = sorted(
            p for p in listing if p.startswith(SOURCE_PREFIXES) and p.endswith(SOURCE_SUFFIXES)
        )
        if not self.files:
            raise RuntimeError(f"no source files at {self.commit}")
        self._text: dict[str, str] = {}
        self._history: dict[str, list[str]] = {}

    def _git(self, *args: str) -> str:
        result = subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                                capture_output=True)
        return result.stdout.decode("utf-8", errors="replace")

    def text(self, path: str) -> str:
        if path not in self._text:
            self._text[path] = self._git("show", f"{self.commit}:{path}").replace("\r\n", "\n")
        return self._text[path]

    def lines(self, path: str) -> list[str]:
        return self.text(path).rstrip("\n").split("\n")

    def pick(self, rng: random.Random, min_lines: int, max_lines: int,
             max_chars: int | None = None) -> str:
        for _ in range(400):
            path = rng.choice(self.files)
            count = len(self.lines(path))
            if min_lines <= count <= max_lines and (
                max_chars is None or len(self.text(path)) <= max_chars
            ):
                return path
        raise RuntimeError(f"no file with {min_lines}-{max_lines} lines found")

    def history(self, path: str) -> list[str]:
        """Non-merge commits that touched `path` up to the pinned commit, newest first."""
        if path not in self._history:
            out = self._git("log", "--no-merges", "-n", "20", "--format=%H", self.commit,
                            "--", path)
            self._history[path] = out.split()
        return self._history[path]

    def diff(self, sha: str, path: str) -> str:
        return self._git("show", "--no-color", "--format=commit %H%n%n    %s%n", sha, "--",
                         path).replace("\r\n", "\n")


@dataclass
class Observation:
    """One scripted tool result and the call that produced it."""

    tool: str
    arguments: dict[str, Any]
    text: str


@dataclass
class TurnPlan:
    session: int
    turn: int
    category: str
    observation: Observation
    instruction: str | None
    max_tokens: int
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def copy(self) -> bool:
        return self.category in COPY_CATEGORIES


def _identifiers(text: str) -> list[str]:
    counts: dict[str, int] = {}
    for word in IDENTIFIER.findall(text):
        if word in KEYWORDS or word.isupper():
            continue
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts, key=lambda w: (-counts[w], w))


class SessionBuilder:
    """Plans the scripted side of one session from a seed, one turn at a time."""

    def __init__(self, corpus: Corpus, seed: int, session: int, *,
                 obs_tokens: tuple[int, int], max_copy_tokens: int) -> None:
        self.corpus = corpus
        self.session = session
        self.rng = random.Random(derived_seed("session", seed, session, corpus.commit))
        self.obs_tokens = obs_tokens
        self.max_copy_tokens = max_copy_tokens
        self._block: list[str] = []
        self._focus = corpus.pick(self.rng, 120, 900)

    def opening(self) -> tuple[str, TurnPlan]:
        task = (
            f"We are doing an ownership and naming cleanup around {REPO_ROOT_NAME}/{self._focus}. "
            "Read the relevant code, explain what you find when asked, and make the edits I "
            "request exactly. Start by reading that file."
        )
        obs = self._read_numbered(self._focus, whole=False)
        return task, TurnPlan(self.session, 0, "explore", obs, None, EXPLORE_MAX_TOKENS,
                              {"path": self._focus})

    def next_turn(self, turn: int) -> TurnPlan:
        if not self._block:
            self._block = list(TURN_BLOCK)
            self.rng.shuffle(self._block)
        category = self._block.pop()
        return getattr(self, f"_plan_{category}")(turn)

    # ----- observation primitives ---------------------------------------------------------

    def _budget_chars(self) -> int:
        low, high = self.obs_tokens
        return int(self.rng.randint(low, high) * CHARS_PER_TOKEN)

    def _slice(self, lines: list[str], overhead: int) -> tuple[int, int]:
        budget = self._budget_chars()
        start = self.rng.randrange(0, max(1, len(lines) // 2))
        used, stop = 0, start
        while stop < len(lines) and used < budget:
            used += len(lines[stop]) + overhead
            stop += 1
        return start, stop

    def _read_numbered(self, path: str, whole: bool) -> Observation:
        lines = self.corpus.lines(path)
        start, stop = (0, len(lines)) if whole else self._slice(lines, 8)
        args: dict[str, Any] = {"file_path": f"{REPO_ROOT_NAME}/{path}"}
        if not whole:
            args.update(offset=start + 1, limit=stop - start)
        return Observation("Read", args, numbered(lines, start, stop))

    def _read_plain(self, path: str, whole: bool) -> Observation:
        lines = self.corpus.lines(path)
        if whole:
            return Observation("Bash", {"command": f"cat {path}"}, "\n".join(lines))
        start, stop = self._slice(lines, 1)
        return Observation("Bash", {"command": f"sed -n '{start + 1},{stop}p' {path}"},
                           "\n".join(lines[start:stop]))

    def _grep(self) -> Observation:
        budget = self._budget_chars()
        anchor = self.corpus.pick(self.rng, 40, 4000)
        idents = _identifiers(self.corpus.text(anchor))[:40] or ["Program"]
        pattern = self.rng.choice(idents)
        directory = anchor.split("/", 1)[0] + "/"
        candidates = [p for p in self.corpus.files if p.startswith(directory)]
        # Scan a seeded sample rather than the whole tree.
        sample = sorted(self.rng.sample(candidates, min(len(candidates), 120)))
        rows: list[str] = []
        used = 0
        for path in sample:
            for number, line in enumerate(self.corpus.lines(path), 1):
                if pattern in line:
                    row = f"{path}:{number}:{line.rstrip()[:200]}"
                    rows.append(row)
                    used += len(row) + 1
                    if used >= budget:
                        break
            if used >= budget:
                break
        return Observation("Grep", {"pattern": pattern, "path": f"{REPO_ROOT_NAME}/{directory}"},
                           "\n".join(rows) if rows else "No matches found")

    def _diff(self, max_chars: int, min_pre_image_lines: int = 0
              ) -> tuple[Observation, str, str] | None:
        for _ in range(60):
            path = self.corpus.pick(self.rng, 40, 4000)
            shas = self.corpus.history(path)
            if not shas:
                continue
            sha = self.rng.choice(shas[:8])
            text = self.corpus.diff(sha, path)
            if 600 <= len(text) <= max_chars and "\n@@" in text and \
                    pre_image_lines(text) >= min_pre_image_lines:
                obs = Observation("Bash", {"command": f"git show {sha[:12]} -- {path}"}, text)
                return obs, sha, path
        return None

    def _explore_observation(self) -> Observation:
        kind = self.rng.choice(("read", "read", "read_plain", "grep", "diff"))
        if kind == "grep":
            return self._grep()
        if kind == "diff":
            found = self._diff(self._budget_chars())
            if found is not None:
                return found[0]
        path = self.corpus.pick(self.rng, 60, 4000)
        if kind == "read_plain":
            return self._read_plain(path, whole=False)
        return self._read_numbered(path, whole=False)

    def _copy_budget(self, chars: int, factor: float = 1.0) -> int:
        return min(self.max_copy_tokens, int(est_tokens(chars) * factor * 1.25) + 256)

    def _copy_max_chars(self) -> int:
        return int((self.max_copy_tokens - 256) / 1.25 * CHARS_PER_TOKEN)

    def _whole_file(self) -> str:
        return self.corpus.pick(self.rng, 60, 320, max_chars=self._copy_max_chars())

    # ----- per-category plans -------------------------------------------------------------

    def _plan_explore(self, turn: int) -> TurnPlan:
        return TurnPlan(self.session, turn, "explore", self._explore_observation(), None,
                        EXPLORE_MAX_TOKENS)

    def _plan_explain(self, turn: int) -> TurnPlan:
        path = self.corpus.pick(self.rng, 80, 4000)
        obs = self._read_numbered(path, whole=False)
        idents = _identifiers(strip_line_numbers(obs.text))[:12] or ["this code"]
        subject = self.rng.choice(idents)
        instruction = (
            f"Explain in plain prose how `{subject}` in {REPO_ROOT_NAME}/{path} is used in the "
            "code above: what it owns, which invariants callers rely on, and what would break if "
            "it changed. Do not quote or rewrite any code and do not call tools; four short "
            "paragraphs."
        )
        return TurnPlan(self.session, turn, "explain", obs, instruction, EXPLAIN_MAX_TOKENS,
                        {"path": path, "subject": subject})

    def _plan_rewrite_file(self, turn: int) -> TurnPlan:
        path = self._whole_file()
        text = self.corpus.text(path)
        ranked = _identifiers(text)
        idents = [w for w in ranked if 2 <= text.count(w) <= 12] or ranked or ["value"]
        old = self.rng.choice(idents[:20])
        new = old + ("_checked" if old.islower() or "_" in old else "Checked")
        instruction = (
            f"Rename `{old}` to `{new}` throughout {REPO_ROOT_NAME}/{path}. Call Write with the "
            "COMPLETE updated file contents (without the line-number prefixes). Every other line "
            "must stay byte-identical."
        )
        return TurnPlan(self.session, turn, "rewrite_file", self._read_numbered(path, whole=True),
                        instruction, self._copy_budget(len(text)),
                        {"path": path, "rename": [old, new], "source_chars": len(text)})

    def _plan_apply_change(self, turn: int) -> TurnPlan:
        path = self._whole_file()
        text = self.corpus.text(path)
        comment = "#" if path.endswith(".py") else "//"
        instruction = (
            f"Answer directly, without calling any tool: return {REPO_ROOT_NAME}/{path} with this "
            f"change applied: add one `{comment}` comment line directly above the first function "
            "definition that summarises what the function does. Output the complete resulting "
            "file in a single fenced code block and nothing else; all other lines unchanged."
        )
        return TurnPlan(self.session, turn, "apply_change", self._read_plain(path, whole=True),
                        instruction, self._copy_budget(len(text)),
                        {"path": path, "source_chars": len(text)})

    def _plan_edit_function(self, turn: int) -> TurnPlan:
        path = self.corpus.pick(self.rng, 80, 4000)
        lines = self.corpus.lines(path)
        size = self.rng.randint(20, 60)
        start = self.rng.randrange(0, max(1, len(lines) - size))
        stop = min(len(lines), start + size)
        first, last = max(0, start - 20), min(len(lines), stop + 20)
        obs = Observation("Read", {"file_path": f"{REPO_ROOT_NAME}/{path}", "offset": first + 1,
                                   "limit": last - first}, numbered(lines, first, last))
        block_chars = sum(len(line) + 1 for line in lines[start:stop])
        comment = "#" if path.endswith(".py") else "//"
        instruction = (
            f"Use Edit on {REPO_ROOT_NAME}/{path}: old_string is exactly lines {start + 1}-{stop} "
            "as shown above (without the line-number prefixes); new_string is the same block with "
            f"a `{comment} reviewed` comment line inserted above every line that contains "
            "`return`. Keep everything else identical."
        )
        # old_string and new_string both reproduce the block.
        return TurnPlan(self.session, turn, "edit_function", obs, instruction,
                        self._copy_budget(block_chars, 2.2),
                        {"path": path, "lines": [start + 1, stop], "source_chars": block_chars})

    def _plan_diff_revert(self, turn: int) -> TurnPlan:
        # A diff that creates its file has no pre-image to reproduce.
        found = self._diff(min(self._copy_max_chars(), 14000), min_pre_image_lines=15)
        if found is None:
            return self._plan_apply_change(turn)
        obs, sha, path = found
        instruction = (
            f"Answer directly, without calling any tool: write out the code as it was BEFORE "
            f"commit {sha[:12]} using only the diff above. For each hunk, reproduce its context "
            "and removed lines (without the leading ' ' or '-' marker) in order, one fenced code "
            "block per hunk, and nothing else."
        )
        return TurnPlan(self.session, turn, "diff_revert", obs, instruction,
                        self._copy_budget(len(obs.text)),
                        {"path": path, "commit": sha, "source_chars": len(obs.text)})
