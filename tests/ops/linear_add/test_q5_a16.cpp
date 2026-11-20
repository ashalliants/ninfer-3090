#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q5_a16_conformance() {
    // Starts of the registered positive-T regions. run_shape checks b-1/b/b+1 for every start,
    // plus one interior point for every region, through the public Op. The starts track the sm_86
    // tables in q5_linear_add_plan.cpp; 513 starts upstream's whole-wave + narrow-tail composite,
    // whose interiors cover a short, a 128-column and a 256-column tail (the last one keeping the
    // single wide launch).
    //
    // composite_offset re-probes the same starts one whole wave (512 columns) later, where the
    // tail - not the leading launch - is the one resolving against them. The graph tokens cover a
    // tail on each narrow route, 705 for the tail=193 composite-to-wide fallback, and 1025 for a
    // one-column tail after the second whole wave.
    constexpr std::array<std::int32_t, 16> kInteriors{1,  2,   8,   16,  24,  40,  64,  80,
                                                      96, 110, 129, 256, 640, 768, 1024, 4};
    constexpr std::array<std::int32_t, 5> kK6144RouteStarts{3, 33, 104, 128, 513};
    constexpr std::array<std::int32_t, 7> kK6144GraphTokens{513, 515, 545, 616, 640, 705, 1025};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 6144, 401U, kK6144RouteStarts, kInteriors, kK6144GraphTokens, false, 512});
    constexpr std::array<std::int32_t, 6> kK17408RouteStarts{3, 33, 65, 104, 128, 513};
    constexpr std::array<std::int32_t, 8> kK17408GraphTokens{513, 515, 545, 577,
                                                             616, 640, 705, 1025};
    failures +=
        ninfer::test::linear_add::run_shape("Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
                                            ShapeCase{5120, 17408, 409U, kK17408RouteStarts,
                                                      kInteriors, kK17408GraphTokens, false, 512});
    // Every output column at a few widths, through whatever route each resolves to.
    constexpr std::array<std::int32_t, 0> no_starts{};
    constexpr std::array<std::int32_t, 7> full{8, 13, 32, 40, 80, 112, 192};
    for (const auto k : {6144, 17408}) {
        failures += ninfer::test::linear_add::run_shape(
            "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
            ShapeCase{5120, k, 401U, no_starts, full, {}, true, 0});
    }
    // The verify widths around the sm_70 tensor-core form's start (NINFER_STREAM_HMMA_Q5=1:
    // Q5StreamFormat::kHmmaMinCols = 6 for K <= 6144, eight tokens for longer rows), every
    // column through the public Op: b-1/b/b+1 at both K. Without the opt-in (or off sm_70) the
    // same widths check the SIMT route.
    constexpr std::array<std::int32_t, 5> verify_widths{5, 6, 7, 8, 9};
    for (const auto k : {6144, 17408}) {
        failures += ninfer::test::linear_add::run_shape(
            "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
            ShapeCase{5120, k, 405U, no_starts, verify_widths, {}, true, 0});
    }
    // T = 8 alone, so the activation ends where the last token's row does: the K-chunked
    // streaming GEMV (sm_70 decode/verify, T >= 2) stages K = 17408 in chunks that do not divide
    // it, and its last chunk must stage only the K that is left. Reading a full chunk there runs
    // kilobytes past the allocation, which compute-sanitizer memcheck reports.
    constexpr std::array<std::int32_t, 1> exact{8};
    failures += ninfer::test::linear_add::run_shape(
        "Q5_A16 LinearAdd", WeightFormat::Q5G64F16S,
        ShapeCase{5120, 17408, 403U, no_starts, exact, {}, true, 0});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
