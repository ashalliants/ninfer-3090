// Exhaustive checks of the bit-level FP8 and NVFP4 decodes that replace the vendor software
// conversions below sm_89 (ops/linear/fp8/fp8_a16_codec.cuh, ops/linear/nvfp4/nvfp4_stream_format.cuh):
//
//   * fp8_e4m3x2_to_float2_exact against __nv_fp8x2_e4m3's own conversion for all 256 x 256 code
//     pairs (NaN codes excluded: weights never hold them);
//   * nvfp4_scaled_pair_bf16 and fp8_e4m3x2_to_bf16x2_bits (the A16 tensor-core kernels' BF16
//     widenings) for every code/scale and code pair;
//   * Nvfp4StreamFormat::dp4a_word against 2 x the vendor E2M1 decode, for every code in every
//     nibble position, and its half_scales against the vendor E4M3 decode of every scale pair.
//
// Bit-exact equality, not a tolerance: both decodes are claimed exact.

#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_stream_format.cuh"

#include <cuda_bf16.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace ninfer::ops::detail;

__global__ void fp8_kernel(float* exact, float* vendor) {
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); // 0 .. 65535
    if (pair >= 65536) return;
    const float2 a = fp8_e4m3x2_to_float2_exact(static_cast<std::uint32_t>(pair));
    __nv_fp8x2_e4m3 v;
    v.__x          = static_cast<std::uint16_t>(pair);
    const float2 b = static_cast<float2>(v);
    exact[2 * pair] = a.x, exact[2 * pair + 1] = a.y;
    vendor[2 * pair] = b.x, vendor[2 * pair + 1] = b.y;
}

// One word per thread; across the launch every code reaches every nibble position.
__global__ void nvfp4_kernel(const std::uint32_t* words, int count, int* bad_codes) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    Nvfp4StreamFormat::Slice slice{make_uint4(words[i], 0u, 0u, 0u), 0u, 1.0f};
    std::uint32_t even = 0, odd = 0;
    Nvfp4StreamFormat::dp4a_word<0>(slice, even, odd);
    for (int k = 0; k < 8; ++k) {
        const std::uint32_t operand = (k & 1) ? odd : even;
        const auto got  = static_cast<std::int8_t>((operand >> (8 * (k >> 1))) & 0xffu);
        const std::uint8_t code = static_cast<std::uint8_t>((words[i] >> (4 * k)) & 0xfu);
        __nv_fp4x2_e2m1 v;
        v.__x            = code; // low nibble
        const float want = static_cast<float2>(v).x * 2.0f;
        if (static_cast<float>(got) != want) atomicAdd(bad_codes, 1);
    }
}

__global__ void nvfp4_scale_kernel(int* bad_scales) {
    const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= 65536) return;
    const int lo = pair & 0xff, hi = pair >> 8;
    if ((lo & 0x7f) == 0x7f || (hi & 0x7f) == 0x7f) return; // NaN
    Nvfp4StreamFormat::Slice slice{make_uint4(0u, 0u, 0u, 0u), static_cast<std::uint32_t>(pair),
                                   128.0f};
    const float2 got = Nvfp4StreamFormat::half_scales(slice);
    __nv_fp8x2_e4m3 v;
    v.__x             = static_cast<std::uint16_t>(pair);
    const float2 want = static_cast<float2>(v);
    // half_scales is the E4M3 value times 2^-8 times scale_factor: 128 makes it half the value.
    if (got.x != want.x * 0.5f || got.y != want.y * 0.5f) atomicAdd(bad_scales, 1);
}

// The BF16 widenings the A16 tensor-core kernels use: every NVFP4 code byte with every scale, and
// every FP8 code pair, against the vendor conversions rounded to BF16.
__global__ void bf16_widen_kernel(int* bad) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); // code | scale << 8
    if (i >= 65536) return;
    const std::uint8_t code = i & 0xff, scale = i >> 8;
    if ((scale & 0x7f) != 0x7f) {
        __nv_fp4x2_e2m1 c;
        c.__x                = code;
        const float2 values  = static_cast<float2>(c);
        __nv_fp8x2_e4m3 m;
        m.__x                = static_cast<std::uint16_t>(scale | (scale << 8));
        const float multiplier = static_cast<float2>(m).x;
        const __nv_bfloat162 want = __floats2bfloat162_rn(values.x * multiplier,
                                                          values.y * multiplier);
        const unsigned got  = nvfp4_scaled_pair_bf16(code, scale);
        const unsigned wantb = *reinterpret_cast<const unsigned*>(&want);
        // Signed zeros compare equal.
        if (got != wantb && ((got | wantb) & 0x7fff7fffu) != 0) atomicAdd(bad, 1);
    }
    const int lo = i & 0xff, hi = i >> 8;
    if ((lo & 0x7f) != 0x7f && (hi & 0x7f) != 0x7f) {
        __nv_fp8x2_e4m3 v;
        v.__x                     = static_cast<std::uint16_t>(i);
        const float2 f            = static_cast<float2>(v);
        const __nv_bfloat162 want = __floats2bfloat162_rn(f.x, f.y);
        const unsigned got        = fp8_e4m3x2_to_bf16x2_bits(static_cast<unsigned>(i));
        if (got != *reinterpret_cast<const unsigned*>(&want)) atomicAdd(bad + 1, 1);
    }
}

bool is_nan_code(int code) { return (code & 0x7f) == 0x7f; }

int check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        return 1;
    }
    return 0;
}
} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    int failures = 0;

    float *exact = nullptr, *vendor = nullptr;
    failures += check(cudaMalloc(&exact, 131072 * sizeof(float)), "malloc");
    failures += check(cudaMalloc(&vendor, 131072 * sizeof(float)), "malloc");
    fp8_kernel<<<256, 256>>>(exact, vendor);
    failures += check(cudaDeviceSynchronize(), "fp8 kernel");
    std::vector<float> a(131072), b(131072);
    failures += check(cudaMemcpy(a.data(), exact, a.size() * 4, cudaMemcpyDeviceToHost), "copy");
    failures += check(cudaMemcpy(b.data(), vendor, b.size() * 4, cudaMemcpyDeviceToHost), "copy");
    int fp8_bad = 0;
    for (int pair = 0; pair < 65536; ++pair) {
        for (int half = 0; half < 2; ++half) {
            const int code = half ? pair >> 8 : pair & 0xff;
            if (is_nan_code(code)) continue;
            const float x = a[2 * pair + half], y = b[2 * pair + half];
            if (std::memcmp(&x, &y, sizeof(float)) != 0 && !(x == 0.0f && y == 0.0f)) {
                if (fp8_bad++ < 5)
                    std::fprintf(stderr, "FP8 code 0x%02x: exact %g vendor %g\n", code, x, y);
            }
        }
    }
    failures += fp8_bad != 0;
    cudaFree(exact);
    cudaFree(vendor);

    constexpr int kWords = 1 << 16;
    std::vector<std::uint32_t> words(kWords);
    for (int i = 0; i < kWords; ++i) {
        std::uint32_t w = 0;
        for (int k = 0; k < 8; ++k)
            w |= static_cast<std::uint32_t>(((i >> (4 * (k & 3))) + k) & 0xf) << (4 * k); // all 16 per k
        words[i] = w;
    }
    std::uint32_t* device_words = nullptr;
    int* counters               = nullptr;
    failures += check(cudaMalloc(&device_words, kWords * 4), "malloc");
    failures += check(cudaMalloc(&counters, 2 * sizeof(int)), "malloc");
    failures += check(cudaMemset(counters, 0, 2 * sizeof(int)), "memset");
    failures += check(cudaMemcpy(device_words, words.data(), kWords * 4, cudaMemcpyHostToDevice),
                      "copy");
    nvfp4_kernel<<<kWords / 256, 256>>>(device_words, kWords, counters);
    nvfp4_scale_kernel<<<256, 256>>>(counters + 1);
    failures += check(cudaDeviceSynchronize(), "nvfp4 kernels");
    int bad[2] = {};
    failures += check(cudaMemcpy(bad, counters, sizeof(bad), cudaMemcpyDeviceToHost), "copy");
    cudaFree(device_words);
    cudaFree(counters);
    if (bad[0]) std::fprintf(stderr, "NVFP4 dp4a decode: %d codes differ\n", bad[0]);
    if (bad[1]) std::fprintf(stderr, "NVFP4 scale decode: %d scale pairs differ\n", bad[1]);
    failures += bad[0] != 0;
    failures += bad[1] != 0;

    int* widen_bad = nullptr;
    failures += check(cudaMalloc(&widen_bad, 2 * sizeof(int)), "malloc");
    failures += check(cudaMemset(widen_bad, 0, 2 * sizeof(int)), "memset");
    bf16_widen_kernel<<<256, 256>>>(widen_bad);
    failures += check(cudaDeviceSynchronize(), "widen kernel");
    int widen[2] = {};
    failures += check(cudaMemcpy(widen, widen_bad, sizeof(widen), cudaMemcpyDeviceToHost), "copy");
    cudaFree(widen_bad);
    if (widen[0]) std::fprintf(stderr, "NVFP4 BF16 widening: %d code/scale pairs differ\n", widen[0]);
    if (widen[1]) std::fprintf(stderr, "FP8 BF16 widening: %d code pairs differ\n", widen[1]);
    failures += widen[0] != 0;
    failures += widen[1] != 0;

    std::printf("%s FP8 / NVFP4 bit-level decodes\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
