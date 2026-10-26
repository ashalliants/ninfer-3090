// The GGML linears' device sub-block decoders (ops/linear/ggml/ggml_blocks.cuh, Format<F>::decode)
// against ggml b11316's own dequantize_row_* results, for every format that has one.
//
// A kernel decodes every 32-value sub-block of tests/fixtures/ggml/<format>.blocks (the fixtures
// of ninfer_ggml_blocks_decode_test) and writes value i as binary32(int8 v_i * scale[i / 16]), the
// weight the linears' dp4a and int8 MMA products stand for. For each format that product is the
// reference's own last FP32 operation, so every output equals ggml's bit pattern; a NaN output only
// has to be a NaN (the device's NaN payload and sign are its own). The affine K-quants (Q4_K,
// Q5_K) write binary32(binary32(v_i * scale) - offset), ggml's own last two operations. IQ2_XS and
// IQ3_XXS have a decoder but no registered linear problem yet, so this is their only device check.

#include "core/weight.h"
#include "ops/linear/ggml/ggml_blocks.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::cuda_check;
namespace g = ninfer::ops::ggml;

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) { throw std::runtime_error("cannot open fixture " + path.string()); }
    return {std::istreambuf_iterator<char>(file), {}};
}

bool is_nan(std::uint32_t bits) { return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x7FFFFFU); }

template <QType F>
__global__ void decode_sub_blocks(const std::uint8_t* blocks, int sub_blocks, float* out) {
    __shared__ std::uint32_t grid_s[512];
    g::Tables tables;
    g::load_tables<F>(tables, grid_s);
    __syncthreads();
    for (int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); s < sub_blocks;
         s += static_cast<int>(gridDim.x * blockDim.x)) {
        g::SubBlockOf<F> sb;
        g::Format<F>::decode(blocks, s, tables, sb);
        for (int i = 0; i < g::kSubValues; ++i) {
            const auto word = static_cast<std::uint32_t>(sb.v[i / 4]);
            const auto v    = static_cast<std::int8_t>(word >> (8 * (i % 4)));
            float value     = __fmul_rn(static_cast<float>(v), sb.scale[i / 16]);
            if constexpr (g::kAffine<F>) value = __fsub_rn(value, sb.offset);
            out[g::kSubValues * s + i] = value;
        }
    }
}

template <QType F>
int check(const char* name) {
    const std::filesystem::path root =
        std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "ggml";
    const auto blocks     = read_file(root / (std::string(name) + ".blocks"));
    const auto expected   = read_file(root / (std::string(name) + ".f32"));
    const GgmlBlock block = ggml_block(F);
    const auto count      = blocks.size() / block.bytes;
    if (block.values != g::Format<F>::kBlockValues || block.bytes != g::Format<F>::kBlockBytes ||
        blocks.size() != count * block.bytes || expected.size() != count * block.values * 4) {
        std::printf("%s: fixture sizes disagree with the block geometry\n", name);
        return 1;
    }
    const auto values      = expected.size() / 4;
    const int sub_blocks   = static_cast<int>(values / g::kSubValues);
    std::uint8_t* d_blocks = nullptr;
    float* d_out           = nullptr;
    cuda_check(cudaMalloc(&d_blocks, blocks.size()), "cudaMalloc blocks");
    cuda_check(cudaMalloc(&d_out, values * sizeof(float)), "cudaMalloc out");
    cuda_check(cudaMemcpy(d_blocks, blocks.data(), blocks.size(), cudaMemcpyHostToDevice),
               "copy blocks");
    decode_sub_blocks<F><<<(sub_blocks + 255) / 256, 256>>>(d_blocks, sub_blocks, d_out);
    cuda_check(cudaGetLastError(), "decode_sub_blocks launch");
    std::vector<float> got(values);
    cuda_check(cudaMemcpy(got.data(), d_out, values * sizeof(float), cudaMemcpyDeviceToHost),
               "copy out");
    cuda_check(cudaFree(d_blocks), "cudaFree blocks");
    cuda_check(cudaFree(d_out), "cudaFree out");
    std::size_t mismatches = 0, nan = 0;
    for (std::size_t i = 0; i < values; ++i) {
        std::uint32_t want = 0;
        std::memcpy(&want, expected.data() + 4 * i, 4); // little-endian host
        const auto bits = std::bit_cast<std::uint32_t>(got[i]);
        nan += is_nan(want);
        if (is_nan(want) ? is_nan(bits) : bits == want) { continue; }
        if (mismatches++ == 0) {
            std::printf("%s: block %zu value %zu is 0x%08x, ggml has 0x%08x\n", name,
                        i / block.values, i % block.values, bits, want);
        }
    }
    std::printf("%s: %zu device values, %zu differ from ggml (%zu NaN)\n", name, values,
                mismatches, nan);
    return mismatches != 0 || nan == 0;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    try {
        int failures = 0;
        failures += check<QType::GGML_Q8_0>("ggml_q8_0");
        failures += check<QType::GGML_IQ4_NL>("ggml_iq4_nl");
        failures += check<QType::GGML_IQ4_XS>("ggml_iq4_xs");
        failures += check<QType::GGML_Q2_0>("ggml_q2_0");
        failures += check<QType::GGML_IQ3_S>("ggml_iq3_s");
        failures += check<QType::GGML_Q6_K>("ggml_q6_k");
        failures += check<QType::GGML_IQ2_XS>("ggml_iq2_xs");
        failures += check<QType::GGML_IQ3_XXS>("ggml_iq3_xxs");
        failures += check<QType::GGML_Q4_K>("ggml_q4_k");
        failures += check<QType::GGML_Q5_K>("ggml_q5_k");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("%s\n", error.what());
        return 1;
    }
}
