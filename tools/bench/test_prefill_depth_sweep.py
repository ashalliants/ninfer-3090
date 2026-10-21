"""Self-test of run_prefill_depth_sweep.py with a fake ninfer_bench. Run: python -I tools/bench/test_prefill_depth_sweep.py

The fake writes the JSON a real ninfer_bench would, with times from the model but its attention
deliberately 25% slower than the fit and its per-token cost 5% faster (so the combined step cost ~4% faster); the tool must recover those.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prefill_time_model as m  # noqa: E402
import run_prefill_depth_sweep as sweep  # noqa: E402

LENGTHS = [4096, 16384, 32768, 65536, 98304]
ATTENTION, LINEAR = 1.25, 0.95


def check(condition, message):
    if not condition:
        print("FAIL:", message)
        sys.exit(1)


def fake_times(chunk=4096):
    cost = m.load_cost("27b")
    out = {}
    for prompt in LENGTHS:
        fixed = linear = attention = 0.0
        prefix = 0
        while prefix < prompt:
            tokens = min(chunk, prompt - prefix)
            a, b, c = cost.parts_ns(prefix, tokens)
            fixed, linear, attention = fixed + a, linear + b, attention + c
            prefix += tokens
        out[prompt] = (fixed + LINEAR * linear + ATTENTION * attention) / 1e9
    return out


def bench_document(times):
    return {"tool": "ninfer_bench", "tests": [
        {"label": f"pp{p}", "kind": "pp", "n_prompt": p, "n_gen": 0,
         "prefill_tok_s_mean": p / t, "prefill_seconds_mean": t}
        for p, t in times.items()
    ] + [{"label": "tg128", "kind": "tg", "n_prompt": 0, "n_gen": 128, "prefill_seconds_mean": 0.01}]}


def test_parse_ignores_decode_tests():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "b.json")
        json.dump(bench_document(fake_times()), open(path, "w"))
        parsed = sweep.parse_bench_json(__import__("pathlib").Path(path))
    check(sorted(parsed) == LENGTHS, f"parsed {sorted(parsed)}")


def test_fit_recovers_the_planted_scales():
    result = sweep.analyse(fake_times(), 4096, "27b")
    s = result["scales"]
    check(abs(s["attention"] - ATTENTION) < 0.02, f"attention scale {s['attention']}")
    # The base scale blends the unchanged per-step cost (1.0) and the 5% cheaper per-token cost.
    check(0.94 < s["base"] < 0.99, f"base scale {s['base']}")
    check(result["rows"][-1]["ratio"] > result["rows"][0]["ratio"], "the deep prompt should be the worst fit")


def test_a_faithful_card_fits_to_one():
    cost_times = {p: m.prefill_profile(m.load_cost("27b"), p, 4096)["seconds"] for p in LENGTHS}
    s = sweep.analyse(cost_times, 4096, "27b")["scales"]
    check(all(abs(s[k] - 1.0) < 1e-6 for k in ("base", "attention")), f"scales {s}")


def test_one_length_fits_attention_only():
    one = {p: t for p, t in fake_times().items() if p == 98304}
    s = sweep.analyse(one, 4096, "27b")["scales"]
    check(s["free"] == ["attention"], "with one length only attention should be free")


def test_end_to_end_with_a_fake_executable():
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "ninfer_bench")
        document = json.dumps(bench_document(fake_times()))
        with open(exe, "w") as f:
            f.write("#!%s\nimport sys, json\nargs = sys.argv[1:]\n"
                    "out = args[args.index('--output-file') + 1]\n"
                    "assert '--prefill-chunk' in args and '-p' in args\n"
                    "open(out, 'w').write(%r)\n" % (sys.executable, document))
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
        out_json = os.path.join(d, "analysis.json")
        process = subprocess.run(
            [sys.executable, "-I", os.path.join(HERE, "run_prefill_depth_sweep.py"),
             "--ninfer-bench", exe, "--weights", "x.ninfer", "--output", out_json,
             "--", "--kv-dtype", "rk4v4"], capture_output=True, text=True)
        check(process.returncode == 0, process.stderr)
        check("worth a kernel-level look" in process.stdout, process.stdout)
        check(abs(json.load(open(out_json))["scales"]["attention"] - ATTENTION) < 0.02, "end to end scale")


def test_attention_commands_target_the_chunk_width():
    commands = sweep.attention_commands(4096, [4096, 65536])
    check(all("--tokens 4096" in c for c in commands) and "--context 61440" in commands[0],
          f"{commands}")


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
    print("OK prefill depth sweep")
