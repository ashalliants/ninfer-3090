#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Public numerical cases straddle each registered Q4 implementation interval. They make
        // no assertion about the private route selected for any T.
        // 8, 16 and 17 straddle the small-T dispatch's exact-width boundary as well as the
        // registered intervals: a width that fills its tile runs an instantiation with no runtime
        // column count, and the widths either side of it do not (q4_linear_swiglu_gemv.cu).
        constexpr std::array<std::int32_t, 23> kTokenCases{
            1,   2,   8,   16,  17,  24,  25,  32,  33,  40,  41, 48,
            49,  128, 129, 256, 257, 384, 385, 512, 513, 640, 641,
        };
        const int failures = run_profile(
            "LinearSwiGLU Q4_A16",
            {QType::Q4G64_F16S, 34816, 5120, 17408, 1401U, ActivationCompute::A16}, kTokenCases, std::array<std::int32_t, 4>{7, 25, 49, 128});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
