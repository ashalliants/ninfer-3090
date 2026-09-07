// Route-boundary sweep for the W8 linear-pair k=5120 table (the 27B KV projection pair).
//
// This is the third and last route table the catch-up merge changed, and it changed the most:
//
//   before   {1,4} TwoSimtR8C4   {5,56} TwoSimtR8C8   {57,kAny} DualMmaR32C128
//   after    {1,85} TwoSimtR8C4  {86,960} DualMmaR32C64   {961,kAny} DualMmaR32C128
//
// The middle kernel of the old table no longer exists -- upstream deleted TwoSimtR8C8 along with
// DualMmaR32C80/C96/C112 -- so the table had to be rewritten and could not simply be kept. But the
// rewrite hands a SIMT kernel everything up to 85 columns where this fork previously gave it four,
// and pushes the c128 tile from 57 out to 961, both on sm_120's authority.
//
// Schedules are driven through w8_pair_execute_schedule, the real dispatch minus its
// plan-matches-problem check, rather than through the kernel launches directly: the tiled routes
// slice the token dimension by their own column tile and pick a full-tile variant based on
// alignment, and a bench that reimplemented that would be timing a replica of the Op.

#include "ops/linear_pair/w8/w8_pair_plan.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
namespace detail = ninfer::ops::detail;

constexpr std::int32_t kRows = 1024;

void sweep_for_k(std::int32_t k, const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight first =
        ninfer::bench::make_row_split_weight(QType::W8G32_F16S, kRows, k, k, {0x31, 0x00, 0x3c00});
    ninfer::bench::PackedQuantizedWeight second =
        ninfer::bench::make_row_split_weight(QType::W8G32_F16S, kRows, k, k, {0x31, 0x00, 0x3c00});
    ninfer::DeviceBuffer input(static_cast<std::size_t>(k) * max_tokens * 2);
    ninfer::DeviceBuffer first_out(static_cast<std::size_t>(kRows) * max_tokens * 2);
    ninfer::DeviceBuffer second_out(static_cast<std::size_t>(kRows) * max_tokens * 2);

    const auto make = [&](detail::W8PairScheduleId schedule) {
        return [&, schedule](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {k, tokens});
            Tensor a(first_out.p, DType::BF16, {kRows, tokens});
            Tensor b(second_out.p, DType::BF16, {kRows, tokens});
            detail::w8_pair_execute_schedule(schedule, x, first.weight, second.weight, a, b,
                                             stream);
        };
    };

    // Only the three the k=5120 table can select. The split-K and concat families are reachable
    // from the k=2048 table, but several of them assume that shape, and a kernel run outside the
    // geometry it was written for does not politely throw -- it reads off the end of a buffer and
    // takes the process with it, which is how an earlier sweep of another Op died.
    std::vector<ninfer::bench::SweepEntry> schedules{
        {"two_simt_r8_c4", make(detail::W8PairScheduleId::TwoSimtR8C4), 0},
        {"dual_mma_r32_c64", make(detail::W8PairScheduleId::DualMmaR32C64), 0},
        {"dual_mma_r32_c128", make(detail::W8PairScheduleId::DualMmaR32C128), 0},
    };

    const std::string title             = "w8 linear_pair k=" + std::to_string(k);
    ninfer::bench::SweepOptions options = base;
    options.title                       = title.c_str();
    std::string routed;
    options.routed_name = [k, &routed](std::int32_t tokens) -> const char* {
        const detail::W8PairProblem problem{kRows, k, k, tokens};
        routed = detail::w8_pair_schedule_name(detail::w8_pair_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor a(first_out.p, DType::BF16, {kRows, tokens});
        Tensor b(second_out.p, DType::BF16, {kRows, tokens});
        detail::w8_pair_dispatch(x, first.weight, second.weight, a, b, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::SweepOptions options;
    options.tokens = {1,  2,   4,   8,   16,  24,  32,  48,  56,  64,  80,  85,  86,
                      96, 112, 128, 160, 192, 256, 384, 512, 768, 960, 961, 1024};
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr, "usage: %s [--tokens T,...] [--repeat N] [--warmup N]\n", argv[0]);
        return 2;
    }
    sweep_for_k(5120, options);
    return 0;
}
