// Checks the C++ FP64 oracles (inkling2_reference.h) against the NumPy reference's golden vectors
// (tests/fixtures/inkling2/golden.json, written by tools/inkling2/reference.py --emit-golden).
// Host-only (ninfer_inkling2_reference_test). It also builds alone:
//   g++ -std=c++20 -I third_party tests/ops/inkling2_reference_test.cpp -o t && ./t tests/fixtures/inkling2/golden.json

#include "inkling2_reference.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace ref = ninfer::test::inkling2_ref;
using json = nlohmann::json;

namespace {

int failures = 0;

ref::Vec flat(const json& j) {
    ref::Vec out;
    if (j.is_array()) {
        for (const auto& e : j) {
            const ref::Vec part = flat(e);
            out.insert(out.end(), part.begin(), part.end());
        }
    } else {
        out.push_back(j.get<double>());
    }
    return out;
}

void expect(const char* what, const ref::Vec& got, const ref::Vec& want, double tol = 1e-12) {
    if (got.size() != want.size()) {
        std::printf("FAIL %s: %zu values, expected %zu\n", what, got.size(), want.size());
        ++failures;
        return;
    }
    for (std::size_t i = 0; i < got.size(); ++i)
        if (std::abs(got[i] - want[i]) > tol * std::max(1.0, std::abs(want[i]))) {
            std::printf("FAIL %s[%zu]: %.17g, expected %.17g\n", what, i, got[i], want[i]);
            ++failures;
            return;
        }
    std::printf("ok   %s (%zu values)\n", what, got.size());
}

}  // namespace

int main(int argc, char** argv) {
#ifdef NINFER_SOURCE_DIR
    const std::filesystem::path fallback =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "inkling2" / "golden.json";
#else
    const std::filesystem::path fallback;
#endif
    if (argc > 2 || (argc == 1 && fallback.empty())) {
        std::fprintf(stderr, "usage: %s golden.json\n", argv[0]);
        return 2;
    }
    std::ifstream in(argc == 2 ? std::filesystem::path(argv[1]) : fallback);
    if (!in) {
        std::fprintf(stderr, "cannot open the golden vectors\n");
        return 2;
    }
    const json g = json::parse(in);

    {
        const auto& c = g["l2_clip"];
        const ref::Vec scale = flat(c["scale"]);
        ref::Vec y;
        for (const auto& row : c["x"]) {
            const ref::Vec r = ref::l2_clip(flat(row), c["threshold"].get<double>(), scale.data());
            y.insert(y.end(), r.begin(), r.end());
        }
        expect("l2_clip", y, flat(c["y"]));
    }
    {
        const auto& c = g["log_scale"];
        ref::Vec y;
        for (const auto& p : c["pos"]) y.push_back(ref::log_scale(p.get<std::int64_t>(), 8192, 0.1));
        expect("log_scale", y, flat(c["tau"]));
    }
    {
        const auto& c = g["short_conv"];
        const ref::Vec x = flat(c["x"]), taps = flat(c["taps"]), dyn = flat(c["dynamic"]);
        ref::Vec hist = flat(c["history"]);
        expect("short_conv", ref::short_conv(x, hist, taps, 6, 4), flat(c["y"]));
        expect("short_conv history", hist, flat(c["new_history"]));
        ref::Vec hist2 = flat(c["history"]);
        expect("short_conv dynamic", ref::short_conv(x, hist2, taps, 6, 4, &dyn, c["head_width"].get<std::int64_t>()),
               flat(c["y_dynamic"]));
    }
    {
        const auto& c = g["mix"];
        const std::int64_t tokens = static_cast<std::int64_t>(c["sources"][0].size());
        const ref::Vec queries = flat(c["queries"]);
        const std::int64_t nq = static_cast<std::int64_t>(c["queries"].size());
        ref::Vec y(nq * tokens * 6);  // golden layout [Q, T, H]
        for (std::int64_t t = 0; t < tokens; ++t) {
            std::vector<ref::Vec> sources;
            for (const auto& s : c["sources"]) sources.push_back(flat(s[t]));
            const ref::Vec m = ref::mix(sources, queries, 6);
            for (std::int64_t q = 0; q < nq; ++q)
                for (int h = 0; h < 6; ++h) y[(q * tokens + t) * 6 + h] = m[q * 6 + h];
        }
        expect("mix", y, flat(c["y"]));
    }
    {
        const auto& c = g["control"];
        ref::Vec y;
        for (std::size_t t = 0; t < c["current"].size(); ++t) {
            const ref::Vec r = ref::control(flat(c["current"][t]), flat(c["message"][t]), flat(c["down"]),
                                            flat(c["gain"]), flat(c["forget"]));
            y.insert(y.end(), r.begin(), r.end());
        }
        expect("control", y, flat(c["y"]));
    }
    {
        const auto& c = g["route"];
        ref::Vec ids, w;
        for (const auto& z : c["z"]) {
            const ref::Route r = ref::route(flat(z), flat(c["router"]), flat(c["bias"]), c["top_k"].get<std::int64_t>(),
                                            c["scale"].get<double>());
            for (auto i : r.ids) ids.push_back(static_cast<double>(i));
            w.insert(w.end(), r.weights.begin(), r.weights.end());
        }
        expect("route ids", ids, flat(c["ids"]), 0.0);
        expect("route weights", w, flat(c["weights"]));
    }
    {
        const auto& c = g["situ"];
        const ref::Vec gate = flat(c["gate"]), up = flat(c["up"]);
        ref::Vec y;
        for (std::size_t i = 0; i < gate.size(); ++i) y.push_back(ref::situ(gate[i], up[i], 4.0, 25.0));
        expect("situ", y, flat(c["y"]));
    }
    {
        const auto& c = g["attention"];
        const ref::Vec r = flat(c["r"]), table = flat(c["table"]);
        const std::vector<std::int64_t> dist = c["distance"].get<std::vector<std::int64_t>>();
        const std::int64_t window = c["window"].get<std::int64_t>();
        ref::Vec bias;
        for (int h = 0; h < 4; ++h)
            for (auto d : dist) bias.push_back(ref::relative_bias(r.data() + h * 3, table, 3, d));
        expect("relative_bias", bias, flat(c["bias"]));
        std::vector<bool> visible;
        for (auto d : dist) visible.push_back(d >= 0 && d < window);
        expect("attention", ref::attention(flat(c["q"]), flat(c["k"]), flat(c["v"]), bias, flat(c["sink"]), visible, 4, 2, 8),
               flat(c["y"]));
    }
    {
        const auto& c = g["nvfp4"];
        std::vector<std::uint8_t> blocks;
        for (const auto& b : c["blocks"]) blocks.push_back(b.get<std::uint8_t>());
        expect("decode_nvfp4", ref::decode_nvfp4(blocks), flat(c["values"]), 0.0);
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
