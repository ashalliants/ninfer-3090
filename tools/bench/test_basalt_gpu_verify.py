"""Self-test of basalt_gpu_verify.py. Run: python -I tools/bench/test_basalt_gpu_verify.py

Checks the committed basalt_options.json (every option names a flag ninfer-serve parses, a known
model and kind, valid regexes and workloads), the validator on broken entries, the plan's regex
check against a fake `ctest -N` listing, and the parsers the steps read results with (ctest
output, the KLD gate's summary and rows, run_chat_decode.py's RESULT lines, the A/B win rule and
the combined row's arguments). Pure Python, no GPU, no build tree.
"""
import contextlib
import copy
import io
import os
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
check("basalt_options.json is valid", not problems, problems)
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


# The plan: every regex against a fake ctest listing.
def plan_with(names, gpu):
    def fake(build, label=None):
        return list(gpu) if label == "gpu" else list(names)
    real, v.ctest_names = v.ctest_names, fake
    try:
        with tempfile.TemporaryDirectory() as scratch:
            args = ["--build-dir", scratch, "--out", scratch, "--platform", "linux", "--dry-run"]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = v.main(args)
        return code, out.getvalue()
    finally:
        v.ctest_names = real


every = sorted({"ninfer_qwen4_exp_engine_real_test", "ninfer_qwen4_exp_mtp_real_test",
                "ninfer_qwen4_exp_mtp_constrained_test", "ninfer_argmax_probability_test",
                "ninfer_qwen4_exp_ngram_volume_test", "ninfer_qwen4_exp_expert_cache_test", "ninfer_embedding_test",
                "ninfer_gated_delta_net_test", "ninfer_offloaded_moe_layer_test", "ninfer_qwen4_exp_cpu_verify_misses_test",
                "ninfer_offloaded_moe_wide_test", "ninfer_qwen4_exp_gdn_async_commit_test", "ninfer_hyper_connection_test",
                "ninfer_host_flag_wait_test", "ninfer_linear_ggml_test", "ninfer_sampling_test", "ninfer_linear_rows_test",
                "ninfer_qwen3_5_context_rows_real_test", "ninfer_engine_scheduling_test", "ninfer_http_security_test",
                "ninfer_serve_options_test", "ninfer_qwen4_exp_forward_real_test", "ninfer_qwen4_exp_hot_embedding_test",
                "ninfer_qwen4_exp_moe_precast_test", "ninfer_qwen4_exp_prefill_cpu_share_test",
                "ninfer_qwen4_exp_partial_ram_residency_test"})
code, text = plan_with(every, every)
check("the plan passes when every regex matches", code == 0, text[-2000:])
check("the plan lists every option", all(o["flag"] in text for o in config["options"]))
check("the plan says which steps hold the lock", " hold " in text and " per-test " in text)
check("a missing model is a SKIP, not an error", "[SKIP: needs --flash-next" in text)
check("the full suite leaves out the slow tests", v.SLOW_TESTS in text)
code, text = plan_with([n for n in every if n != "ninfer_qwen4_exp_gdn_async_commit_test"], every)
check("the plan fails when a regex matches no ctest name", code == 1 and "gdn_async_commit" in text, text[-800:])
code, text = plan_with([n for n in every if n != "ninfer_qwen4_exp_forward_real_test"], every)
check("the plan fails when an identity harness is not a ctest name", code == 1 and
      "identity harness ninfer_qwen4_exp_forward_real_test" in text, text[-800:])

# Step ids are unique and the A/B ids number multi-variant options.
with tempfile.TemporaryDirectory() as scratch:
    ctx = v.make_ctx(v.argparse.Namespace(platform="windows", build_dir=scratch, out=scratch, reps=12,
                                          options=str(v.OPTIONS_FILE), **{k: None for k in (
                                              "flash_next", "qwen27b", "qwen35b", "kld_base_flash_next",
                                              "kld_base_27b", "master_cli", "gguf", "iq3_pack")}))
    steps = v.build_steps(ctx)
ids = [s.id for s in steps]
check("step ids are unique", len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1])
check("a multi-variant option numbers its A/Bs", "ab:--prefill-cpu-share:3" in ids)
check("Windows runs the packager and run.bat for Flash-Next", "launchers:package" in ids
      and "launchers:qwen38-flash-next-tuned" in ids)
check("Windows GPU tests go through run-tests.ps1", any("run-tests.ps1" in line for s in steps for line in s.show))
check("every step has a known lock mode", all(s.lock in ("none", "per-test", "hold") for s in steps))

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
                                             "ninfer_gdn_replay_fold_test": "Exception"}, outcome)
kld = v.parse_kld("KL-divergence gate (decode path, T = 3, 20 chunks, 5100 rows)\nPPL:        7.123456 +/- 0.1"
                  "  (base 7.0)\nKLD mean:   0.012345\nKLD median: 0.004\nKLD p99:    0.2\nKLD max:    1.5\n"
                  "same top-1: 93.2100%\n")
check("the KLD summary is read", kld == {"ppl": 7.123456, "kld_mean": 0.012345, "kld_median": 0.004,
                                         "kld_p99": 0.2, "kld_max": 1.5, "same_top1": 93.21}, kld)
with tempfile.TemporaryDirectory() as scratch:
    a, b, c = (Path(scratch) / n for n in ("a.tsv", "b.tsv", "c.tsv"))
    header = "chunk\tposition\ttarget\tnll\tnll_base\tkld\tsame_top1\tdlogp_top1\n"
    a.write_text(header + "0\t257\t11\t1.5\t1.4\t0.01\t1\t0\n0\t258\t12\t2.0\t2.1\t0.02\t1\t0\n", encoding="utf-8")
    b.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")
    c.write_text(header + "0\t257\t11\t1.5\t1.4\t0.01\t1\t0\n0\t258\t12\t2.25\t2.1\t0.05\t1\t0\n", encoding="utf-8")
    check("identical rows compare identical", v.compare_rows(a, b) == {"identical": True})
    rows = v.compare_rows(a, c)
    check("differing rows are counted with their distance", not rows["identical"] and rows["differing"] == 1
          and abs(rows["max_abs_dnll"] - 0.25) < 1e-12 and abs(rows["max_abs_dkld"] - 0.03) < 1e-12, rows)
chat = "\n".join(f"RESULT rep{r} {arm} C1 decode={d} tok/s output_throughput=1.0 tok/s accept={acc}% "
                 f"tok/round=2.000 gen=10 reqs=8" for r, arm, d, acc in
                 ((0, "off", 50.0, 40.0), (0, "on", 55.0, 50.0), (1, "on", 52.0, 52.0), (1, "off", 50.0, 41.0)))
stats = v.parse_chat_results(chat)
check("run_chat_decode RESULT lines pair per rep", stats["pairs"] == 2 and abs(stats["median"] - 1.07) < 1e-9
      and stats["positive"] == 2 and stats["accept_on"] == 51.0, stats)
check("a clear win wins", v.wins({"pairs": 12, "median": 1.03, "positive": 10}))
check("a small median does not", not v.wins({"pairs": 12, "median": 1.005, "positive": 12}))
check("a split result does not", not v.wins({"pairs": 12, "median": 1.05, "positive": 8}))
merged = v.union_args([["--moe-precast-input"], ["--moe-precast-input", "--moe-gate-up-registers"],
                       ["--prefill-cpu-share", "0.2"], ["--prefill-cpu-share", "0.6"], ["--hc-int8"]])
check("the combined row takes each flag once, a valued flag's last value",
      merged == ["--moe-precast-input", "--moe-gate-up-registers", "--prefill-cpu-share", "0.6", "--hc-int8"], merged)

if failures:
    print(f"{len(failures)} check(s) failed:")
    for failure in failures:
        print("  " + failure)
    sys.exit(1)
print("basalt_gpu_verify self-test: all checks passed")
