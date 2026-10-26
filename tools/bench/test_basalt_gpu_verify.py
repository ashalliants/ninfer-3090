"""Self-test of basalt_gpu_verify.py. Run: python -I tools/bench/test_basalt_gpu_verify.py

Checks the committed basalt_options.json (every option names a flag ninfer-serve parses, a known
model and kind, valid regexes and workloads), the validator on broken entries, the plan's regex
check against a fake `ctest -N` listing (a misspelt alternative included), --resume and the pinned
results directory, the Linux wrapper's refusal under WSL, and the logic the steps judge results with:
ctest output, the KLD gate's summary and rows, run_chat_decode.py's RESULT lines and answers, the
engine bench's per-job ids and speculation lines, the A/B verdict, the combined rows' choice and the
partial-residency log scan. Pure Python, no GPU, no build tree.
"""
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, str(HERE))
import basalt_gpu_verify as v  # noqa: E402

failures = []


def check(name, condition, detail=""):
    if not condition:
        failures.append(f"{name}: {detail}")


config = v.load_config()
flags = v.serve_flags()

# The committed options file.
problems = v.validate_config(config, flags)
check("basalt_options.json is valid", not problems, str(problems))
listed = {o["flag"] for o in config["options"]}
for flag in ("--partial-ram-residency", "--ggml-decode-mma", "--relaxed-drafting", "--draft-context-rows"):
    check(f"{flag} is listed", flag in listed)
check("the dropped two-stage head is not listed", "--lm-head-two-stage" not in listed)
check("the scan finds --config (parsed apart from the other flags)", "--config" in flags)

# The validator catches broken entries.
broken = copy.deepcopy(config)
broken["options"][1]["flag"] = "--no-such-flag"
broken["options"][2]["kind"] = "fast"
broken["options"][3]["ab"][0]["workload"] = "nowhere"
broken["options"][4]["ab_reason"] = "both given"
broken["options"].append(copy.deepcopy(broken["options"][5]))
quality = next(o for o in broken["options"] if o["kind"] == "quality" and o.get("kld"))
del quality["kld"]
found = "\n".join(v.validate_config(broken, flags))
for needle in ("--no-such-flag: ninfer-serve does not parse", "kind must be one of", "unknown workload 'nowhere'",
               "exactly one of ab", "listed twice", "needs a kld harness flag"):
    check(f"the validator reports {needle!r}", needle in found, found)

# Regex alternatives.
check("a group's alternation is expanded", v.regex_alternatives("^ninfer_(a|b_c)_test$")
      == ["^ninfer_(?:a)_test$", "^ninfer_(?:b_c)_test$"], str(v.regex_alternatives("^ninfer_(a|b_c)_test$")))
check("top-level alternatives are split", v.regex_alternatives("^x$|^y_(p|q)$")
      == ["^x$", "^y_(?:p)$", "^y_(?:q)$"], str(v.regex_alternatives("^x$|^y_(p|q)$")))
check("a character class is not split", v.regex_alternatives("^a[|]b$") == ["^a[|]b$"])


def concrete(alternative):
    """The ctest name an alternative of the options' simple regexes names."""
    return alternative.replace("(?:", "").replace(")", "").strip("^$")


every = sorted({concrete(alt) for o in config["options"] for alt in v.regex_alternatives(o["tests"])}
               | {o["identity"][0][0] for o in config["options"] if o.get("identity")}
               | {c[0] for o in config["options"] for c in o.get("identity", [])}
               | {concrete(alt) for c in config["checks"] if c.get("tests")
                  for alt in v.regex_alternatives(c["tests"])})


def plan_with(names, gpu, options=None, extra=()):
    def fake(build, label=None):
        return list(gpu) if label == "gpu" else list(names)
    real, v.ctest_names = v.ctest_names, fake
    try:
        with tempfile.TemporaryDirectory() as scratch:
            args = ["--build-dir", scratch, "--out", scratch, "--platform", "linux", "--dry-run", *extra]
            if options is not None:
                path = Path(scratch) / "options.json"
                path.write_text(json.dumps(options), encoding="utf-8")
                args += ["--options", str(path)]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = v.main(args)
        return code, out.getvalue()
    finally:
        v.ctest_names = real


code, text = plan_with(every, every)
check("the plan passes when every alternative matches", code == 0, text[-2000:])
check("the plan lists every option", all(o["flag"] in text for o in config["options"]))
check("the plan says which steps hold the lock", " hold " in text and " per-test " in text)
check("a missing model is a SKIP, not an error", "[SKIP: needs --flash-next" in text)
check("the full suite leaves out the slow tests", v.SLOW_TESTS in text)
code, text = plan_with([n for n in every if n != "ninfer_qwen4_exp_gdn_async_commit_test"], every)
check("the plan fails when a regex matches no ctest name", code == 1 and "gdn_async_commit" in text, text[-800:])
code, text = plan_with([n for n in every if n != "ninfer_qwen4_exp_forward_real_test"], every)
check("the plan fails when an identity harness is not a ctest name", code == 1 and
      "identity harness ninfer_qwen4_exp_forward_real_test" in text, text[-800:])
typo = copy.deepcopy(config)
target = next(o for o in typo["options"] if o["flag"] == "--relaxed-drafting")
target["tests"] = "^ninfer_(sampling|speculative_rund|mtp_round|candidate_selector)_test$"
code, text = plan_with(every, every, options=typo)
check("the plan fails on a misspelt alternative the rest of the regex hides", code == 1
      and "speculative_rund" in text, text[-800:])
code, text = plan_with(every, every, extra=("--flash-next", "/no/such/model.ninfer"))
check("a given path that does not exist is a problem", code == 1 and "/no/such/model.ninfer: no such file" in text,
      text[-600:])


def make_ctx(out, **inputs):
    names = ("flash_next", "qwen27b", "qwen35b", "kld_base_flash_next", "kld_base_27b", "master_cli", "gguf",
             "iq3_pack")
    values = {k: inputs.get(k) for k in names}
    return v.make_ctx(v.argparse.Namespace(platform="windows", build_dir=out, out=out, reps=12, resume=False,
                                           options=str(v.OPTIONS_FILE), **values))


# Step ids, lock modes and the per-prompt greedy steps.
with tempfile.TemporaryDirectory() as scratch:
    steps = v.build_steps(make_ctx(scratch))
ids = [s.id for s in steps]
check("step ids are unique", len(ids) == len(set(ids)), str([i for i in ids if ids.count(i) > 1]))
check("a multi-variant option numbers its A/Bs", "ab:--prefill-cpu-share:3" in ids)
check("Windows runs the packager and run.bat for Flash-Next", "launchers:package" in ids
      and "launchers:qwen38-flash-next-tuned" in ids)
check("Windows GPU tests go through run-tests.ps1", any("run-tests.ps1" in line for s in steps for line in s.show))
check("every step has a known lock mode", all(s.lock in ("none", "per-test", "hold") for s in steps))
prompts = len(v.chat_prompt_lines())
check("one greedy-master step per configuration and prompt", sum(i.startswith("greedy-master:") for i in ids)
      == 3 * prompts + 2 * 8, str(sum(i.startswith("greedy-master:") for i in ids)))
check("the 100-chunk KLD runs wait for their 20-chunk runs", all(
    s.data.get("gate") == s.id.replace("-100", "-20") for s in steps if s.id.startswith("kld:") and "100" in s.id))
check("combined rows for prefill and the 27B", {"combined:prefill", "combined:qwen38-27b"} <= set(ids))

# --resume, --action out and the lock column of a step that will skip.
with tempfile.TemporaryDirectory() as scratch:
    out = Path(scratch) / "run"
    common = ["--build-dir", scratch, "--out", str(out), "--platform", "linux"]
    (out / "results").mkdir(parents=True)
    for step_id, status in (("preflight", "PASS"), ("converter-pytests", "SKIP"), ("ggml-mma:timing", "MEASURED")):
        v.result_path(make_ctx(str(out)), step_id).write_text(json.dumps({"id": step_id, "status": status}))

    def listing(*extra):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = v.main([*common, "--action", "steps", *extra])
        return code, [line.split("\t") for line in buffer.getvalue().splitlines()]
    code, rows = listing("--resume")
    listed_ids = [r[0] for r in rows]
    check("--resume leaves out PASS and MEASURED steps", "preflight" not in listed_ids
          and "ggml-mma:timing" not in listed_ids, str(listed_ids[:5]))
    check("--resume runs a skipped step again", "converter-pytests" in listed_ids)
    code, rows = listing()
    check("without --resume every step is listed", "preflight" in [r[0] for r in rows])
    ab = next(r for r in rows if r[0] == "ab:--moe-fused-route")
    check("a hold step that will skip takes no lock", ab[1] == "none" and "[SKIP: needs --flash-next" in ab[2],
          str(ab))
    check("the steps listing writes the run manifest", (out / "run.json").is_file())
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        v.main([*common, "--action", "out"])
    check("--action out prints the pinned directory", buffer.getvalue().strip() == str(out.resolve()))
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = v.main([*common, "--action", "run", "--step", "preflight", "--resume"])
    check("--resume does not rerun a finished step", code == 0 and "not run again" in buffer.getvalue())
    real_root, v.RESULTS_ROOT = v.RESULTS_ROOT, Path(scratch)
    try:
        newer = Path(scratch) / "basalt-2026-01-02"
        newer.mkdir()
        (newer / "run.json").write_text("{}")
        check("--resume without --out continues the newest run", v.default_out(True) == newer)
        today = f"basalt-{v.dt.date.today():%Y-%m-%d}"
        check("without --resume the default is today's directory", v.default_out(False).name == today)
    finally:
        v.RESULTS_ROOT = real_root
    with contextlib.redirect_stderr(io.StringIO()):
        code, _ = listing("--qwen27b", str(Path(scratch) / "missing.ninfer"))
    check("--action steps refuses a missing input before any lock", code == 2)

# The Linux wrapper refuses to run under WSL (the shared 3090 needs the Windows-side lock).
bash = shutil.which("bash")
if bash and os.name != "nt":
    with tempfile.TemporaryDirectory() as scratch:
        fake = Path(scratch) / "version"
        fake.write_text("Linux version 6.6.87.2-microsoft-standard-WSL2 (root@host)\n")
        done = subprocess.run([bash, str(HERE / "basalt_gpu_verify.sh"), "--build-dir", scratch],
                              capture_output=True, text=True, env={**os.environ, "NINFER_PROC_VERSION": str(fake)})
        check("the Linux wrapper refuses under WSL", done.returncode == 2 and "basalt_gpu_verify.ps1" in done.stderr,
              done.stderr)
        fake.write_text("Linux version 6.8.0-45-generic (buildd@lcy02)\n")
        done = subprocess.run([bash, str(HERE / "basalt_gpu_verify.sh"), "--build-dir", scratch, "--out",
                               str(Path(scratch) / "out"), "--only", "nothing-matches"], capture_output=True, text=True,
                              env={**os.environ, "NINFER_PROC_VERSION": str(fake)})
        check("the Linux wrapper runs on a plain Linux kernel", done.returncode == 0 and "WSL" not in done.stderr
              and (Path(scratch) / "out" / "summary.md").is_file(), done.stdout[-300:] + done.stderr)

# Parsers.
ctest = """
 1/4 Test  #93: ninfer_qwen3_5_score_real_test ...........   Passed   12.01 sec
 2/4 Test  #94: ninfer_qwen4_exp_mtp_real_test ...........***Skipped   0.02 sec
 3/4 Test  #95: ninfer_linear_ggml_test ..................***Failed    3.10 sec
 4/4 Test  #96: ninfer_gdn_replay_fold_test ..............***Exception: SegFault  0.30 sec
"""
outcome = v.parse_ctest(ctest)
check("ctest outcomes are read", outcome == {"ninfer_qwen3_5_score_real_test": "Passed",
                                             "ninfer_qwen4_exp_mtp_real_test": "Skipped",
                                             "ninfer_linear_ggml_test": "Failed",
                                             "ninfer_gdn_replay_fold_test": "Exception"}, str(outcome))
kld = v.parse_kld("KL-divergence gate (decode path, T = 3, 20 chunks, 5100 rows)\nPPL:        7.123456 +/- 0.1"
                  "  (base 7.0)\nKLD mean:   0.012345\nKLD median: 0.004\nKLD p99:    0.2\nKLD max:    1.5\n"
                  "same top-1: 93.2100%\n")
check("the KLD summary is read", kld == {"ppl": 7.123456, "kld_mean": 0.012345, "kld_median": 0.004,
                                         "kld_p99": 0.2, "kld_max": 1.5, "same_top1": 93.21}, str(kld))
with tempfile.TemporaryDirectory() as scratch:
    a, b, c = (Path(scratch) / n for n in ("a.tsv", "b.tsv", "c.tsv"))
    header = "chunk\tposition\ttarget\tnll\tnll_base\tkld\tsame_top1\tdlogp_top1\n"
    a.write_text(header + "0\t257\t11\t1.5\t1.4\t0.01\t1\t0\n0\t258\t12\t2.0\t2.1\t0.02\t1\t0\n", encoding="utf-8")
    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    c.write_text(header + "0\t257\t11\t1.5\t1.4\t0.01\t1\t0\n0\t258\t12\t2.25\t2.1\t0.05\t1\t0\n", encoding="utf-8")
    check("identical rows compare identical", v.compare_rows(a, b) == {"identical": True})
    rows = v.compare_rows(a, c)
    check("differing rows are counted with their distance", not rows["identical"] and rows["differing"] == 1
          and abs(rows["max_abs_dnll"] - 0.25) < 1e-12 and abs(rows["max_abs_dkld"] - 0.03) < 1e-12, str(rows))
chat = "\n".join(f"RESULT rep{r} {arm} C1 decode={d} tok/s output_throughput=1.0 tok/s accept={acc}% "
                 f"tok/round=2.000 gen=10 reqs=8" for r, arm, d, acc in
                 ((0, "off", 50.0, 40.0), (0, "on", 55.0, 50.0), (1, "on", 52.0, 52.0), (1, "off", 50.0, 41.0)))
stats = v.parse_chat_results(chat)
check("run_chat_decode RESULT lines pair per rep", stats["pairs"] == 2 and abs(stats["median"] - 1.07) < 1e-9
      and stats["positive"] == 2 and stats["accept_on"] == 51.0, str(stats))
check("a clear win wins", v.wins({"pairs": 12, "median": 1.03, "positive": 10}))
check("a small median does not", not v.wins({"pairs": 12, "median": 1.005, "positive": 12}))
check("a split result does not", not v.wins({"pairs": 12, "median": 1.05, "positive": 8}))
merged = v.union_args([["--moe-precast-input"], ["--moe-precast-input", "--moe-gate-up-registers"],
                       ["--prefill-cpu-share", "0.2"], ["--prefill-cpu-share", "0.6"], ["--hc-int8"]])
check("the combined row takes each flag once, a valued flag's last value",
      merged == ["--moe-precast-input", "--moe-gate-up-registers", "--prefill-cpu-share", "0.6", "--hc-int8"],
      str(merged))


# The engine bench's per-job identity across runs (hash of every job, and the speculation line).
def bench_log(path, hashes, accepted=(30, 31)):
    lines = []
    for job, (value, acc) in zip(("@C:\\fix\\warm.ids", "@/fix/ppl_tokens-chat.ids"), zip(hashes, accepted)):
        lines.append(f"{job}: 1024 prompt tokens, 256 generated; prefill 1.00 s (1000.0 tok/s), decode 3.10 s "
                     f"(82.00 tok/s), after prefill 3.20 s (80.00 tok/s), wall 4.20 s, ids {value}")
        lines.append(f"{job}: mtp rounds 100, plain 2, drafted 300, accepted {acc} (10.0%), 1.300 tokens per round; "
                     "accepted at position 1.. [0.500 0.300 0.100]")
    path.write_text("engine loaded in 50.0 s\n" + "\n".join(lines) + "\n", encoding="utf-8")


with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    same = [root / f"run-{i}.log" for i in range(4)]
    for log in same:
        bench_log(log, ("1111111111111111", "2222222222222222"))
    r = v.engine_identity(same)
    check("equal ids and speculation in every run", r["same_ids"] and r["same_spec"] and
          r["jobs"] == ["ppl_tokens-chat.ids", "warm.ids"], str(r))
    bench_log(same[2], ("1111111111111111", "3333333333333333"))
    r = v.engine_identity(same)
    check("a later job's different ids are caught (not only the last job's)", not r["same_ids"]
          and r["ids_differ"] == ["ppl_tokens-chat.ids"], str(r))
    bench_log(same[2], ("4444444444444444", "2222222222222222"))
    check("the first (warm-up) job is hashed too", not v.engine_identity(same)["same_ids"])
    bench_log(same[2], ("1111111111111111", "2222222222222222"), accepted=(30, 29))
    r = v.engine_identity(same)
    check("a drafter that diverged with equal ids is caught", r["same_ids"] and r["same_spec"] is False, str(r))
    verdict = v.ab_verdict({"kind": "identity", "a": [], "b": ["--x"], "workload": "w"},
                           {"pairs": 12, "median": 1.02, "min": 0.99, "max": 1.05, "positive": 10}, 0, "",
                           r["same_ids"], r["same_spec"], r)
    check("an identity option whose speculation differs fails", verdict["status"] == "FAIL"
          and "DIFFER" in verdict["detail"], str(verdict))
    verdict = v.ab_verdict({"kind": "speed", "a": [], "b": ["--x"], "workload": "w"},
                           {"pairs": 12, "median": 1.02, "min": 0.99, "max": 1.05, "positive": 10}, 0, "",
                           r["same_ids"], r["same_spec"], r)
    check("a speed option only reports it", verdict["status"] == "MEASURED" and verdict["data"]["wins"],
          str(verdict))

    # run_chat_decode's answers, off against on, per rep.
    runs = root / "chat"
    for rep in (0, 1):
        for arm in ("off", "on"):
            (runs / f"rep{rep}_{arm}").mkdir(parents=True)
            (runs / f"rep{rep}_{arm}" / "responses.json").write_text(json.dumps(
                [{"choices": [{"message": {"content": "the same"}}]}]))
    check("equal greedy answers compare equal", v.chat_outputs_equal(runs))
    (runs / "rep1_on" / "responses.json").write_text(json.dumps([{"choices": [{"message": {"content": "other"}}]}]))
    check("a different answer is caught", not v.chat_outputs_equal(runs))
    check("no answers is not equality", not v.chat_outputs_equal(root / "none"))

    # The combined rows' choice: winners only, best variant, no quality flags in the identity row.
    ctx = make_ctx(str(root / "combined"))
    winners = {"ab:--moe-fused-route": 1.03, "ab:--hc-fused-read": 1.04, "ab:--hc-int8": 1.05,
               "ab:--prefill-cpu-share:1": 1.02, "ab:--prefill-cpu-share:3": 1.06, "ab:--gdn-head-map": 1.001}
    for step_id, median in winners.items():
        path = v.result_path(ctx, step_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "MEASURED", "data": {"wins": median >= 1.01, "median": median}}))
    by_id = {s.id: s for s in v.build_steps(ctx)}
    chosen, b_args = v.combined_choice(ctx, by_id["combined:identity"])
    check("the identity row keeps winners and leaves out quality flags", chosen == ["--moe-fused-route"]
          and b_args == ["--moe-fused-route"], f"{chosen} {b_args}")
    chosen, b_args = v.combined_choice(ctx, by_id["combined:all"])
    check("the all row adds the quality winners", chosen == ["--moe-fused-route", "--hc-int8", "--hc-fused-read"]
          and b_args == ["--moe-fused-route", "--hc-int8", "--hc-fused-read"], f"{chosen} {b_args}")
    chosen, b_args = v.combined_choice(ctx, by_id["combined:prefill"])
    check("the prefill row takes the best prefill share", chosen == ["--prefill-cpu-share"]
          and b_args == ["--prefill-cpu-share", "0.6"], f"{chosen} {b_args}")
    captured = {}
    real_ab, v.run_ab_step = v.run_ab_step, lambda c, s: captured.update(s.data) or v.result("MEASURED", "ok")
    try:
        r = v.run_combined_step(ctx, by_id["combined:identity"])
    finally:
        v.run_ab_step = real_ab
    check("the combined step runs one A/B with the winners on", captured.get("a") == [] and
          captured.get("b") == ["--moe-fused-route"] and "1 option(s) on" in r["detail"], f"{captured} {r}")
    check("no winner, no combined run", v.combined_choice(ctx, by_id["combined:qwen38-27b"]) == ([], []))

# The partial-residency start: the cap warning is not a pin failure.
warning = ("warning: --partial-ram-residency pins 30.1 GiB (...), above the 24.0 GiB (half the RAM less 8 GiB) "
           "that Windows lets one process register with cudaHostRegister; NInfer pins with cudaHostAlloc, which may "
           "still succeed. If startup fails to pin: start with --expert-cache-mib 9000 or more")
check("the cap warning is not a pin failure", not v.pin_failure_line(warning))
check("a real pin failure is caught", v.pin_failure_line("Qwen3.8-Flash-Next cannot start: cudaHostAlloc of 30 GiB "
                                                         "failed: out of memory"))

if failures:
    print(f"{len(failures)} check(s) failed:")
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("basalt_gpu_verify self-test: all checks passed")
