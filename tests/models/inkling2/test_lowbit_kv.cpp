// Opt-in low-bit global KV rings (--kv-dtype int8 | rk8v4 | rk4v4 on Inkling-2): the specs set only the global
// visits' bits, the default limits set none, and the ring bytes at 262,144 positions are what the fit plans with
// (no GPU; skips without NINFER_TEST_INKLING2_ARTIFACT, which supplies the config).
#include "artifact/reader.h"
#include "models/inkling2/config.h"
#include "models/inkling2/program/text_program.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    failures += ok ? 0 : 1;
}

constexpr std::size_t kMiB = std::size_t{1} << 20;

ink::ProgramLimits fp16_limits(std::int32_t context) {
    ink::ProgramLimits limits;
    limits.max_context = context, limits.max_chunk = 2048, limits.kv_fp16 = true;
    return limits;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        const ninfer::artifact::Reader reader(artifact_path);
        const ink::TextConfig c = ink::parse_config(reader.directory());
        std::uint32_t globals = 0;
        for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) { globals += c.local[c.physical(v)] ? 0 : 1; }
        check(globals == 8, "8 global visits");

        const ink::ProgramLimits plain = fp16_limits(262144);
        check(plain.kv_lowbit_k.empty() && plain.kv_lowbit_v.empty() && plain.kv_quant_k.empty(), "the default limits code no ring");
        const std::size_t fp16 = ink::kv_ring_bytes(c, plain);
        ink::ProgramLimits fp32 = plain;
        fp32.kv_fp16 = false;
        // FP32 rings are twice FP16's; the globals are 8 x 2 x kv_heads x head_dim x 262,144 FP16 values (8192 MiB), the
        // sliding rings the rest.
        check(ink::kv_ring_bytes(c, fp32) == 2 * fp16, "FP32 rings take twice FP16's bytes");
        ink::ProgramLimits short_context = fp16_limits(8192);
        check(fp16 - ink::kv_ring_bytes(c, short_context) == (262144 - 8192) * std::size_t{32} * 1024, "an FP16 position costs 32 KiB over the global rings");

        struct Form { const char* spec; int kb, vb; std::size_t freed_mib; };
        for (const Form f : {Form{"k8v8:real", 8, 8, 3968}, Form{"k8v4:real", 8, 4, 4928}, Form{"k4v4:real", 4, 4, 5952}}) {
            ink::ProgramLimits limits = plain;
            ink::parse_kv_quant(c, f.spec, limits);
            bool only_globals = limits.kv_lowbit_k.size() == c.num_layer_visits && limits.kv_lowbit_v.size() == c.num_layer_visits && limits.kv_quant_k.empty();
            for (std::uint32_t v = 0; only_globals && v < c.num_layer_visits; ++v) {
                const bool global = !c.local[c.physical(v)];
                only_globals = limits.kv_lowbit_k[v] == (global ? f.kb : 0) && limits.kv_lowbit_v[v] == (global ? f.vb : 0);
            }
            check(only_globals, std::string(f.spec) + " codes the global visits only");
            const std::size_t bytes = ink::kv_ring_bytes(c, limits);
            const std::size_t freed = fp16 - bytes;
            std::printf("  %s: rings %.1f MiB at 262,144 (FP16 %.1f), frees %.1f MiB\n", f.spec, double(bytes) / kMiB, double(fp16) / kMiB, double(freed) / kMiB);
            check(freed / kMiB == f.freed_mib, std::string(f.spec) + " frees " + std::to_string(f.freed_mib) + " MiB at 262,144");
        }
        // A form per visit: the first global visits at 8 bits, the rest at 4; later forms take the visits they name.
        ink::ProgramLimits mixed = plain;
        ink::parse_kv_quant(c, "k4v4:real+k8v8:g=0,1:real", mixed);
        std::uint32_t at8 = 0, at4 = 0;
        for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) { at8 += mixed.kv_lowbit_k[v] == 8, at4 += mixed.kv_lowbit_k[v] == 4; }
        check(at8 == 2 && at4 == 6, "k4v4:real+k8v8:g=0,1:real: two global visits at 8 bits, six at 4");
        bool refused = false;
        try {
            ink::ProgramLimits bad = plain;
            ink::parse_kv_quant(c, "k2v4:real", bad);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "a 2-bit form is refused");

        // Forms apply left to right: k16v16 sets the visits it names back to FP16, and a real form clears a simulated one.
        ink::ProgramLimits cleared = plain;
        ink::parse_kv_quant(c, "k4v4:real+k16v16:g=0", cleared);
        std::uint32_t coded = 0, first_global = 0;
        while (c.local[c.physical(first_global)]) { ++first_global; }
        for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) { coded += cleared.kv_lowbit_k[v] != 0; }
        check(coded == 7 && cleared.kv_lowbit_k[first_global] == 0 && cleared.kv_lowbit_v[first_global] == 0, "k4v4:real+k16v16:g=0 leaves g0 FP16");
        ink::ProgramLimits none = plain;
        ink::parse_kv_quant(c, "k4v4+k16v16", none);
        check(none.kv_quant_k.empty() && none.kv_lowbit_k.empty(), "k4v4+k16v16 codes nothing");
        ink::ProgramLimits replaced = plain;
        ink::parse_kv_quant(c, "k4v4+k8v8:real", replaced);
        check(replaced.kv_quant_k.empty() && replaced.kv_lowbit_k.size() == c.num_layer_visits, "a real form over every visit replaces a simulated one");
        bool mixed_refused = false;
        try {
            ink::ProgramLimits bad = plain;
            ink::parse_kv_quant(c, "k4v4+k8v8:g=0:real", bad);
        } catch (const std::invalid_argument&) { mixed_refused = true; }
        check(mixed_refused, "simulated and real forms on different visits are refused");
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
