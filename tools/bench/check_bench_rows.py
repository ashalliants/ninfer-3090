#!/usr/bin/env python3
# Adapted from Basalt 8d73a39 tools/check_benchmarks.py (MIT, third_party/basalt/LICENSE).
"""Validate a community rows file: the rows `ninfer_bench --preset community-1` prints.

usage:
  python3 tools/bench/check_bench_rows.py [--fix] FILE    # normally bench/community/rows.json
  python3 tools/bench/check_bench_rows.py --self-test

Exits 0 when FILE is valid and in the house format, else 1 with one line per problem. The house
format is the one text a file may have: keys in the fixed order below, two-space indent, UTF-8
without escapes, LF line ends and one trailing newline. --fix first rewrites a file that parses as
JSON into that format, then validates it; it never changes a value, so a wrong value still fails.

Only the Python standard library is used.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import re
import sys
import tempfile

PRESET = "community-1"
TOP_ORDER = ("preset", "runs")
ROW_ORDER = ("submitter", "date", "commit", "preset", "gpus", "ram_gib", "cpu", "os", "model",
             "prefill_16k_tps", "prefill_64k_tps", "decode_tps", "notes")
ROW_OPTIONAL = {"cpu", "notes"}
GPU_ORDER = ("name", "vram_gib", "power_limit_w", "pcie")
GPU_OPTIONAL = {"pcie"}
RATES = ("prefill_16k_tps", "prefill_64k_tps", "decode_tps")
SYSTEMS = ("Linux", "Windows", "WSL2")
SUBMITTER_MAX = 40
TEXT_MAX = 80
NOTES_MAX = 200
PLACEHOLDER = "YOUR_NAME"
PCIE = re.compile(r"[1-9]\.0 x(1|2|4|8|16|32)")
COMMIT = re.compile(r"[0-9a-f]{7,40}")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


def fix_hint(path: str) -> str:
    return f"python3 tools/bench/check_bench_rows.py --fix {path}"


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_text(value, limit: int) -> bool:
    return (isinstance(value, str) and value.strip() != "" and len(value) <= limit
            and not CONTROL.search(value))


def check_keys(record: dict, order, optional, where: str, problems: list[str]) -> None:
    for key in order:
        if key not in optional and key not in record:
            problems.append(f"{where}missing required key '{key}'")
    for key in record:
        if key not in order:
            problems.append(f"{where}unknown key '{key}'")


def check_gpu(gpu, where: str, problems: list[str]) -> None:
    if not isinstance(gpu, dict):
        problems.append(f"{where}must be an object")
        return
    check_keys(gpu, GPU_ORDER, GPU_OPTIONAL, where, problems)
    if "name" in gpu and not is_text(gpu["name"], TEXT_MAX):
        problems.append(f"{where}name must be text of at most {TEXT_MAX} characters")
    for key in ("vram_gib", "power_limit_w"):
        if key in gpu and not is_whole(gpu[key]):
            problems.append(f"{where}{key} must be a whole number > 0")
    if "pcie" in gpu and not (isinstance(gpu["pcie"], str) and PCIE.fullmatch(gpu["pcie"])):
        problems.append(f"{where}pcie must look like '4.0 x16'")


def check_row(row, where: str, problems: list[str]) -> None:
    if not isinstance(row, dict):
        problems.append(f"{where}must be an object")
        return
    check_keys(row, ROW_ORDER, ROW_OPTIONAL, where, problems)
    if "submitter" in row:
        submitter = row["submitter"]
        if not is_text(submitter, SUBMITTER_MAX):
            problems.append(f"{where}submitter must be text of at most {SUBMITTER_MAX} characters,"
                            " without control characters")
        elif submitter.strip().upper() == PLACEHOLDER:
            problems.append(f"{where}submitter is still the placeholder {PLACEHOLDER}")
    if "date" in row:
        date = row["date"]
        valid = isinstance(date, str) and DATE.fullmatch(date) is not None
        if valid:
            try:
                datetime.date.fromisoformat(date)
            except ValueError:
                valid = False
        if not valid:
            problems.append(f"{where}date must be a real date as YYYY-MM-DD")
    if "commit" in row and not (isinstance(row["commit"], str) and COMMIT.fullmatch(row["commit"])):
        problems.append(f"{where}commit must be 7-40 lowercase hex digits")
    if "preset" in row and row["preset"] != PRESET:
        problems.append(f"{where}preset must be '{PRESET}'")
    if "gpus" in row:
        gpus = row["gpus"]
        if not isinstance(gpus, list) or not gpus:
            problems.append(f"{where}gpus must be a non-empty list")
        else:
            for index, gpu in enumerate(gpus):
                check_gpu(gpu, f"{where}gpus[{index}]: ", problems)
    if "ram_gib" in row and not is_whole(row["ram_gib"]):
        problems.append(f"{where}ram_gib must be a whole number > 0")
    if "cpu" in row and not is_text(row["cpu"], TEXT_MAX):
        problems.append(f"{where}cpu must be text of at most {TEXT_MAX} characters")
    if "os" in row and row["os"] not in SYSTEMS:
        problems.append(f"{where}os must be one of " + ", ".join(SYSTEMS))
    if "model" in row and not is_text(row["model"], TEXT_MAX):
        problems.append(f"{where}model must be text of at most {TEXT_MAX} characters")
    for key in RATES:
        if key in row:
            value = row[key]
            if not is_number(value) or value <= 0:
                problems.append(f"{where}{key} must be a number > 0")
            elif round(value, 1) != value:
                problems.append(f"{where}{key} must have at most one decimal")
    if "notes" in row and not (isinstance(row["notes"], str) and len(row["notes"]) <= NOTES_MAX
                               and not CONTROL.search(row["notes"])):
        problems.append(f"{where}notes must be text of at most {NOTES_MAX} characters")


def validate(data) -> list[str]:
    problems: list[str] = []
    if not isinstance(data, dict):
        return ["the top level must be an object"]
    for key in data:
        if key not in TOP_ORDER:
            problems.append(f"unknown top-level key '{key}'")
    if data.get("preset") != PRESET:
        problems.append(f"the top-level preset must be '{PRESET}'")
    runs = data.get("runs")
    if not isinstance(runs, list):
        problems.append("the top-level 'runs' must be a list")
        return problems
    for index, row in enumerate(runs):
        who = row.get("submitter") if isinstance(row, dict) else None
        label = f"runs[{index}]" + (f" ({who})" if isinstance(who, str) and who else "") + ": "
        check_row(row, label, problems)
    return problems


def ordered(record, order):
    """Known keys in the fixed order, then any others in file order (validation reports those)."""
    if not isinstance(record, dict):
        return record
    return {**{k: record[k] for k in order if k in record},
            **{k: v for k, v in record.items() if k not in order}}


def house_format(data) -> str:
    data = ordered(data, TOP_ORDER)
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        rows = []
        for row in data["runs"]:
            row = ordered(row, ROW_ORDER)
            if isinstance(row, dict) and isinstance(row.get("gpus"), list):
                row["gpus"] = [ordered(gpu, GPU_ORDER) for gpu in row["gpus"]]
            rows.append(row)
        data["runs"] = rows
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    args = list(argv)
    fix = "--fix" in args
    args = [arg for arg in args if arg != "--fix"]
    if len(args) != 1 or args[0].startswith("-"):
        print("usage: check_bench_rows.py [--fix] FILE | --self-test", file=sys.stderr)
        return 2
    path = args[0]
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        data = json.loads(text)
    except (OSError, ValueError) as error:
        print(f"{path}: {error}")
        return 1
    want = house_format(data)
    problems = []
    if text != want:
        if fix:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(want)
            print(f"{path}: rewritten in the house format")
        else:
            problems.append(f"not in the house format; run: {fix_hint(path)}")
    problems += validate(data)
    for problem in problems:
        print(f"{path}: {problem}")
    if problems:
        return 1
    print(f"{path}: ok, {len(data['runs'])} runs")
    return 0


# ------------------------------------------------------------------------------------- self-test

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The line `ninfer_bench --preset community-1` prints for the row of
# tests/test_ninfer_bench_support.cpp (test_community_row), with a plain submitter.
PRINTED_ROW = ('{"submitter":"Ash","date":"2026-10-26","commit":"61561d88a",'
               '"preset":"community-1","gpus":[{"name":"NVIDIA GeForce RTX 3090","vram_gib":24,'
               '"power_limit_w":350,"pcie":"4.0 x16"}],"ram_gib":64,'
               '"cpu":"AMD Ryzen 9 5950X, 32 threads","os":"WSL2","model":"qwen3.6-27b",'
               '"prefill_16k_tps":2345.7,"prefill_64k_tps":2000.0,"decode_tps":41.0}')


def good_row() -> dict:
    return json.loads(PRINTED_ROW)


def self_test() -> int:
    failures: list[str] = []

    def run(path: str, *flags: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([*flags, path])
        return code, out.getvalue()

    def write(path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def read(path: str) -> str:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "rows.json")
        good = {"preset": PRESET, "runs": [good_row(), {**good_row(), "notes": "fans at 100%"}]}

        write(path, house_format(good))
        code, out = run(path)
        check(code == 0 and "ok, 2 runs" in out, f"a good file must pass: {out}")

        write(path, house_format({"preset": PRESET, "runs": []}))
        code, out = run(path)
        check(code == 0, f"an empty runs list must pass: {out}")

        def gpu(row: dict, **changes) -> dict:
            return {**row, "gpus": [{**row["gpus"][0], **changes}]}

        def without(record: dict, key: str) -> dict:
            return {k: v for k, v in record.items() if k != key}

        malformed = [
            ("a missing field", lambda r: without(r, "decode_tps"), "missing required key 'decode_tps'"),
            ("an unknown field", lambda r: {**r, "quant": "NVFP4"}, "unknown key 'quant'"),
            ("a long submitter", lambda r: {**r, "submitter": "x" * 41}, "at most 40 characters"),
            ("the placeholder", lambda r: {**r, "submitter": "your_name"}, "placeholder"),
            ("a control character", lambda r: {**r, "submitter": "a\tb"}, "control characters"),
            ("an impossible date", lambda r: {**r, "date": "2026-02-30"}, "real date"),
            ("a short commit", lambda r: {**r, "commit": "abc12"}, "commit must be"),
            ("a dirty commit", lambda r: {**r, "commit": "61561d88a-dirty"}, "commit must be"),
            ("another preset", lambda r: {**r, "preset": "community-2"}, "preset must be"),
            ("no GPU", lambda r: {**r, "gpus": []}, "non-empty list"),
            ("an unknown power limit", lambda r: gpu(r, power_limit_w=0), "power_limit_w must be"),
            ("a fractional power limit", lambda r: gpu(r, power_limit_w=287.5), "power_limit_w must be"),
            ("a power limit as text", lambda r: gpu(r, power_limit_w="350"), "power_limit_w must be"),
            ("a GPU without a limit", lambda r: {**r, "gpus": [without(r["gpus"][0], "power_limit_w")]},
             "missing required key 'power_limit_w'"),
            ("a bad PCIe link", lambda r: gpu(r, pcie="PCIe 4 x16"), "pcie must look like"),
            ("fractional RAM", lambda r: {**r, "ram_gib": 63.9}, "ram_gib must be"),
            ("an unknown OS", lambda r: {**r, "os": "macOS"}, "os must be one of"),
            ("a blank model", lambda r: {**r, "model": " "}, "model must be"),
            ("a zero rate", lambda r: {**r, "decode_tps": 0}, "decode_tps must be a number > 0"),
            ("a boolean rate", lambda r: {**r, "decode_tps": True}, "decode_tps must be a number > 0"),
            ("two decimals", lambda r: {**r, "prefill_16k_tps": 2345.67}, "at most one decimal"),
            ("long notes", lambda r: {**r, "notes": "n" * 201}, "notes must be"),
        ]
        for name, mutate, expected in malformed:
            write(path, house_format({"preset": PRESET, "runs": [mutate(good_row())]}))
            code, out = run(path)
            check(code == 1 and expected in out, f"{name} must fail with '{expected}': {out}")
            code, out = run(path, "--fix")
            check(code == 1 and expected in out, f"--fix must not repair {name}: {out}")

        for name, top, expected in [
            ("another top-level preset", {"preset": "basalt", "runs": []}, "top-level preset"),
            ("runs not a list", {"preset": PRESET, "runs": {}}, "'runs' must be a list"),
            ("an extra top-level key", {"preset": PRESET, "runs": [], "x": 1}, "unknown top-level key"),
        ]:
            write(path, house_format(top))
            code, out = run(path)
            check(code == 1 and expected in out, f"{name} must fail with '{expected}': {out}")

        # Layouts other than the house format fail until --fix rewrites them, values unchanged.
        reordered = {"runs": [dict(reversed(list(good_row().items())))], "preset": PRESET}
        pasted = '{"preset": "community-1", "runs": [\n' + PRINTED_ROW + "\n]}"
        for name, text in [
            ("a pasted one-line row", pasted),
            ("keys out of order", json.dumps(reordered, indent=2, ensure_ascii=False) + "\n"),
            ("CRLF line ends", house_format(good).replace("\n", "\r\n")),
            ("no trailing newline", house_format(good).rstrip("\n")),
            ("escaped non-ASCII", json.dumps(good, indent=2).replace("Ash", "\\u00c9lise") + "\n"),
            ("a four-space indent", json.dumps(good, indent=4, ensure_ascii=False) + "\n"),
        ]:
            write(path, text)
            code, out = run(path)
            check(code == 1 and "not in the house format" in out, f"{name} must fail: {out}")
            code, out = run(path, "--fix")
            fixed = read(path)
            check(code == 0 and "rewritten in the house format" in out,
                  f"--fix must rewrite {name}: {out}")
            check(fixed == house_format(json.loads(text)) and json.loads(fixed) == json.loads(text),
                  f"--fix must change only the layout of {name}")
            code, out = run(path)
            check(code == 0, f"{name} must pass after --fix: {out}")

        write(path, "{not json")
        code, out = run(path, "--fix")
        check(code == 1 and read(path) == "{not json", "--fix must leave a file that is not JSON alone")

        check(run(os.path.join(directory, "missing.json"))[0] == 1, "a missing file must fail")

    committed = os.path.join(REPO_ROOT, "bench", "community", "rows.json")
    code, out = run(committed)
    check(code == 0, f"the committed rows file must pass: {out}")

    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print("OK community rows checker")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    sys.exit(main(sys.argv[1:]))
