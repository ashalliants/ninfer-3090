// Maintainer benchmark for the Q4 SwiGLU route boundaries, companion to
// q4_q5_attn_input_schedule_bench.cu and written for the same reason: the public bench times the
// route the plan picks, which cannot tell you whether the boundary between two routes is in the
// right place. This one times each schedule at the same column count.
//
// The immediate question it answers: this fork's route table sends 2..32 to SmallTTiled on
// upstream's authority. The boundary it replaced (24) was measured against
// q4_linear_swiglu_small_t_exact, a kernel upstream deleted, so it had nothing behind it -- but
// neither did 32 on sm_86.
//
// Materialized is not included: it needs a workspace and a different launch signature, and it is
// not adjacent to the boundary under test.

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct Schedule {
    const char* name;
    Launch launch;
    // Largest column count the kernel actually processes; 0 means unbounded. gemv_pair is a
    // decode kernel registered for {1,1}: hand it more columns and it silently does one column's
    // work in a constant 122.9 us, which makes it look like it wins everywhere.
    std::int32_t max_cols;
};

const Schedule kSchedules[] = {
    {"gemv_pair", &ninfer::ops::detail::q4_linear_swiglu_gemv_pair_launch, 1},
    {"small_t_tiled", &ninfer::ops::detail::q4_linear_swiglu_small_t_tiled_launch, 32},
    {"split_half_pair_c40",
     &ninfer::ops::detail::q4_linear_swiglu_mma_split_half_pair_r32_c40_launch, 0},
    {"split_half_pair_c48",
     &ninfer::ops::detail::q4_linear_swiglu_mma_split_half_pair_r32_c48_launch, 0},
    {"split_half_pair_c128",
     &ninfer::ops::detail::q4_linear_swiglu_mma_split_half_pair_r32_c128_launch, 0},
};
constexpr int kScheduleCount = static_cast<int>(sizeof(kSchedules) / sizeof(kSchedules[0]));

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

} // namespace

int main(int argc, char** argv) {
    std::vector<std::int32_t> tokens;
    int repeat = 9;
    int warmup = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--tokens" && i + 1 < argc) {
            std::string value(argv[++i]);
            std::size_t start = 0;
            while (start <= value.size()) {
                const std::size_t comma = value.find(',', start);
                const std::string item  = value.substr(start, comma - start);
                if (!item.empty()) { tokens.push_back(std::atoi(item.c_str())); }
                if (comma == std::string::npos) { break; }
                start = comma + 1;
            }
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N]\n", argv[0]);
            return 2;
        }
    }
    if (tokens.empty()) { tokens = {1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64}; }

    const std::int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q4G64_F16S, kGateUpRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_tokens * 2);
    ninfer::DeviceBuffer flush(kFlushBytes);
    cudaStream_t stream = nullptr;

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("# gpu=%s sm=%d%d  q4 swiglu schedules, cold, median of %d\n", properties.name,
                properties.major, properties.minor, repeat);
    std::printf("%6s", "T");
    for (const Schedule& schedule : kSchedules) { std::printf(" %22s", schedule.name); }
    std::printf("   %-22s\n", "winner");

    for (const std::int32_t token_count : tokens) {
        Tensor x(input.p, DType::BF16, {kHidden, token_count});
        Tensor out(output.p, DType::BF16, {kOutputRows, token_count});
        double best_us        = 0.0;
        const char* best_name = "-";
        std::printf("%6d", token_count);
        for (int s = 0; s < kScheduleCount; ++s) {
            const Schedule& schedule = kSchedules[s];
            if (schedule.max_cols != 0 && token_count > schedule.max_cols) {
                std::printf(" %22s", "out-of-domain");
                continue;
            }
            const auto invoke = [&](cudaStream_t launch_stream) {
                schedule.launch(x, packed.weight, out, launch_stream);
            };
            double us = 0.0;
            try {
                us = ninfer::bench::measure_cold_launch(invoke, flush, stream, warmup, repeat)
                         .median_us;
            } catch (const std::exception&) {
                cudaGetLastError();
                std::printf(" %22s", "n/a");
                continue;
            }
            std::printf(" %22.3f", us);
            if (best_us == 0.0 || us < best_us) {
                best_us   = us;
                best_name = schedule.name;
            }
        }
        std::printf("   %-22s\n", best_name);
        std::fflush(stdout);
    }
    return 0;
}
