// Host-only: the byte-range aliasing checks of the Hadamard wrappers. The success cases test the
// shared helper directly (a real call would launch a kernel); the rejection cases go through the
// public wrappers, which validate before any CUDA call, on plain host memory.
#include "ninfer/ops/hadamard_transform.h"
#include "ops/wrapper/byte_ranges.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what);
        std::exit(1);
    }
}

template <typename F> bool rejects(F&& call, const char* needle) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

alignas(64) std::uint8_t g_memory[16384];
alignas(64) std::uint8_t g_signs[2048];

void test_helper() {
    std::uint8_t* a = g_memory;
    constexpr std::size_t n = 4096;
    auto in_place = [&](const void* x, const void* out, std::size_t xb, std::size_t ob) {
        detail::require_in_place_or_disjoint("t", "x", x, xb, "out", out, ob);
    };
    in_place(a, a, n, n);                 // exactly equal
    in_place(a, a + 2 * n, n, n);         // disjoint
    in_place(a + 2 * n, a, n, n);         // disjoint, out first
    in_place(a, a + n, n, n);             // adjacent, not overlapping
    in_place(a + n, a, n, n);             // adjacent, out first
    in_place(a, a + n, 0, n);             // empty ranges overlap nothing
    in_place(a, a, 0, 0);
    check(rejects([&] { in_place(a, a + 2048, n, n); }, "exactly or not at all"),
          "x before out, partial overlap");
    check(rejects([&] { in_place(a + 2048, a, n, n); }, "exactly or not at all"),
          "out before x, partial overlap");
    check(rejects([&] { in_place(a, a + n - 1, n, n); }, "exactly or not at all"),
          "one shared byte");
    check(rejects([&] { in_place(a, a, n, n / 2); }, "exactly or not at all"),
          "same start, different extent");
    detail::require_disjoint("t", "a", a, n, "b", a + n, n);
    check(rejects([&] { detail::require_disjoint("t", "a", a, n, "b", a, n); }, "must not overlap"),
          "equal ranges are not disjoint");
}

void test_wrappers() {
    // [1024, 2] BF16 columns: 4096 bytes. Offsetting by 2048 overlaps the second half.
    Tensor x(g_memory, DType::BF16, {1024, 2});
    Tensor partial_after(g_memory + 2048, DType::BF16, {1024, 2});
    Tensor signs(g_signs, DType::BF16, {1024});
    check(rejects([&] { hadamard_transform(x, signs, false, partial_after, nullptr); }, "alias"),
          "hadamard_transform: out after x");
    Tensor x_late(g_memory + 2048, DType::BF16, {1024, 2});
    Tensor out_early(g_memory, DType::BF16, {1024, 2});
    check(rejects([&] { hadamard_transform(x_late, signs, true, out_early, nullptr); }, "alias"),
          "hadamard_transform: out before x");
    Tensor signs_in_out(g_memory + 1024, DType::BF16, {1024});
    check(rejects([&] { hadamard_transform(x, signs_in_out, false, x, nullptr); }, "overlap"),
          "hadamard_transform: signs inside out");
    Tensor plane(g_memory, DType::BF16, {2048, 2});
    Tensor silu_out(g_memory + 4096, DType::BF16, {1024, 2});
    check(rejects([&] { silu_mul_hadamard(plane, signs, silu_out, nullptr); }, "overlap"),
          "silu_mul_hadamard: out overlapping the plane");
}

} // namespace

int main() {
    test_helper();
    test_wrappers();
    std::printf("hadamard_overlap: passed\n");
    return 0;
}
