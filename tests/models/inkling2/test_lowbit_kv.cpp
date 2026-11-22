// Opt-in low-bit global KV rings (--kv-dtype int8 | rk8v4 | rk4v4 on Inkling-2): the specs set only the global
// visits' bits, the default limits set none, and the ring bytes at 262,144 positions are what the fit plans with
// (no GPU). Then, on the GPU, the simulator's recent window (NINFER_INKLING2_KVQ ...:w=N) across keep() and
// restore(): the positions written again after a cut are coded again. Skips without NINFER_TEST_INKLING2_ARTIFACT.
#include "artifact/reader.h"
#include "core/device.h"
#include "models/inkling2/config.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/text_program.h"

#include <cuda_runtime.h>

#include <span>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

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

        // The recent window across a rejected draft and a snapshot restore (one slot, FP16 rings, 4 positions FP16).
        ninfer::DeviceContext device(0);
        auto model = ink::load_model(reader, device);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { throw std::runtime_error("cannot create a stream"); }
        ink::ProgramLimits limits{1024, 64};
        limits.kv_fp16 = true, limits.max_rollback = 8, limits.decode_graph = false;
        ink::parse_kv_quant(c, "k4v4:w=4", limits);
        {
            ink::TextProgram program(*model, limits, stream);
            std::vector<std::int32_t> tokens(256);
            std::uint32_t seed = 777;
            for (auto& t : tokens) { seed = seed * 1664525U + 1013904223U, t = static_cast<std::int32_t>(1000 + (seed >> 8) % 120000); }
            const auto at = [&](std::int64_t from, std::int64_t n) { return std::span<const std::int32_t>(tokens).subspan(std::size_t(from), std::size_t(n)); };
            for (std::int64_t p = 0; p < 192; p += 64) { (void)program.forward(at(p, 64)); }
            check(program.kv_quant_aged() == 188, "a prefill to 192 codes the positions below 188");
            ink::TextProgram::Snapshot snapshot{ninfer::PinnedHostBuffer(program.snapshot_bytes())};
            program.save(snapshot);
            (void)program.forward(at(192, 8)); // a draft of 8, aged to 196
            check(program.kv_quant_aged() == 196, "the draft's forward codes up to 196");
            program.keep(1);
            check(program.position() == 193 && program.kv_quant_aged() <= 193, "keep(1) brings the coded watermark back to the position");
            for (std::int64_t p = 193; p < 210; ++p) { (void)program.forward(at(p, 1)); }
            check(program.kv_quant_aged() == 206, "decoding to 210 codes up to 206 again, the rewritten positions included");
            program.restore(snapshot);
            check(program.position() == 192 && program.kv_quant_aged() <= 192, "restore() brings it back to the snapshot's position");
            for (std::int64_t p = 192; p < 200; ++p) { (void)program.forward(at(p, 1)); }
            check(program.kv_quant_aged() == 196, "decoding to 200 after the restore codes up to 196");
            if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("the forwards failed"); }
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
