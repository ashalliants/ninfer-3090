// Checks the sm_70 emulation of ldmatrix / mma.sync against naive references.
// Build: nvcc -arch=sm_70 -std=c++20 -I src tests/volta_emulation_test.cu -o volta_emulation_test
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer::ops;

namespace {

__device__ unsigned pack_bf(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return static_cast<unsigned>(__bfloat16_as_ushort(lo)) |
           (static_cast<unsigned>(__bfloat16_as_ushort(hi)) << 16);
}
__device__ unsigned pack_h(__half lo, __half hi) {
    return static_cast<unsigned>(__half_as_ushort(lo)) |
           (static_cast<unsigned>(__half_as_ushort(hi)) << 16);
}

// A: 16x16 row-major, B: 16x8 stored [k][n], C out: 16x8 row-major. Fragments built per PTX layout.
template <bool Bf16>
__global__ void mma16_kernel(const float* A, const float* B, float* C) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto P = [&](float lo, float hi) -> unsigned {
        if constexpr (Bf16) return pack_bf(__float2bfloat16(lo), __float2bfloat16(hi));
        else return pack_h(__float2half(lo), __float2half(hi));
    };
    unsigned a0 = P(A[g * 16 + 2 * t], A[g * 16 + 2 * t + 1]);
    unsigned a1 = P(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]);
    unsigned a2 = P(A[g * 16 + 2 * t + 8], A[g * 16 + 2 * t + 9]);
    unsigned a3 = P(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]);
    unsigned b0 = P(B[(2 * t) * 8 + g], B[(2 * t + 1) * 8 + g]);
    unsigned b1 = P(B[(2 * t + 8) * 8 + g], B[(2 * t + 9) * 8 + g]);
    float c0 = 0.5f, c1 = 0.25f, c2 = -0.5f, c3 = 1.f; // nonzero accumulator input
    if constexpr (Bf16) mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    else mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

__global__ void mma_s8_kernel(const signed char* A, const signed char* B, int* C) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto P = [&](const signed char* base, int stride_k, int idx, int k0) -> unsigned {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<unsigned>(static_cast<unsigned char>(base[idx + (k0 + i) * stride_k])) << (8 * i);
        return v;
    };
    // A is [m][k] (k stride 1, 32 wide); B is [k][n] (k stride 8)
    unsigned a0 = P(A, 1, g * 32, 4 * t), a1 = P(A, 1, (g + 8) * 32, 4 * t);
    unsigned a2 = P(A, 1, g * 32, 16 + 4 * t), a3 = P(A, 1, (g + 8) * 32, 16 + 4 * t);
    unsigned b0 = P(B, 8, g, 4 * t), b1 = P(B, 8, g, 16 + 4 * t);
    int c0 = 1, c1 = 2, c2 = 3, c3 = 4;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

__global__ void mma_tf32_kernel(const float* A, const float* B, float* C) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    mma_tf32(c0, c1, c2, c3, A[g * 8 + t], A[(g + 8) * 8 + t], A[g * 8 + t + 4],
             A[(g + 8) * 8 + t + 4], B[t * 8 + g], B[(t + 4) * 8 + g]);
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

// Four 8x8 b16 matrices, row-major with a padded row pitch of 16 elements.
__global__ void ldmatrix_kernel(const unsigned short* src, unsigned* out) {
    __shared__ __align__(16) unsigned short smem[4 * 8 * 16];
    for (int i = threadIdx.x; i < 4 * 8 * 8; i += 32) {
        const int m = i / 64, r = (i / 8) % 8, c = i % 8;
        smem[(m * 8 + r) * 16 + c] = src[i];
    }
    __syncwarp();
    const int lane = threadIdx.x;
    const unsigned addr = smem_addr(&smem[((lane >> 3) * 8 + (lane & 7)) * 16]);
    unsigned r[4], rt[4], r2[2], r2t[2];
    ldmatrix_x4(r[0], r[1], r[2], r[3], addr);
    ldmatrix_x4_t(rt[0], rt[1], rt[2], rt[3], addr);
    ldmatrix_x2(r2[0], r2[1], addr);
    ldmatrix_x2_t(r2t[0], r2t[1], addr);
    for (int i = 0; i < 4; ++i) { out[(0 * 32 + lane) * 4 + i] = r[i]; out[(1 * 32 + lane) * 4 + i] = rt[i]; }
    for (int i = 0; i < 2; ++i) { out[(2 * 32 + lane) * 4 + i] = r2[i]; out[(3 * 32 + lane) * 4 + i] = r2t[i]; }
}

float rnd() { return (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.f; }
float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
float hf(float x) { return __half2float(__float2half(x)); }

} // namespace

int main() {
    int fails = 0;
    srand(1);
    float *dA, *dB, *dC;
    cudaMallocManaged(&dA, 16 * 16 * 4); cudaMallocManaged(&dB, 16 * 8 * 4); cudaMallocManaged(&dC, 16 * 8 * 4);
    const float acc[4] = {0.5f, 0.25f, -0.5f, 1.f};
    for (int variant = 0; variant < 2; ++variant) {
        for (int i = 0; i < 256; ++i) dA[i] = rnd();
        for (int i = 0; i < 128; ++i) dB[i] = rnd();
        if (variant == 0) mma16_kernel<true><<<1, 32>>>(dA, dB, dC);
        else mma16_kernel<false><<<1, 32>>>(dA, dB, dC);
        cudaDeviceSynchronize();
        float maxerr = 0;
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) {
            // initial C per (g,t) mapping: c0/c1 for rows<8, c2/c3 for rows>=8
            float ref = (m < 8) ? acc[n & 1] : acc[2 + (n & 1)];
            for (int k = 0; k < 16; ++k)
                ref += (variant == 0 ? bf(dA[m * 16 + k]) * bf(dB[k * 8 + n]) : hf(dA[m * 16 + k]) * hf(dB[k * 8 + n]));
            maxerr = fmaxf(maxerr, fabsf(ref - dC[m * 8 + n]));
        }
        printf("mma_%s m16n8k16: max abs err %.3g %s\n", variant == 0 ? "bf16" : "f16", maxerr, maxerr < 1e-4f ? "OK" : "FAIL");
        fails += maxerr >= 1e-4f;
    }
    { // tf32
        for (int i = 0; i < 128; ++i) dA[i] = rnd();
        for (int i = 0; i < 64; ++i) dB[i] = rnd();
        mma_tf32_kernel<<<1, 32>>>(dA, dB, dC); cudaDeviceSynchronize();
        float maxerr = 0;
        auto tf = [](float x) { unsigned u; memcpy(&u, &x, 4); u &= 0xffffe000u; memcpy(&x, &u, 4); return x; };
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) {
            float ref = 0; for (int k = 0; k < 8; ++k) ref += tf(dA[m * 8 + k]) * tf(dB[k * 8 + n]);
            maxerr = fmaxf(maxerr, fabsf(ref - dC[m * 8 + n]));
        }
        printf("mma_tf32 m16n8k8: max abs err %.3g %s\n", maxerr, maxerr < 1e-4f ? "OK" : "FAIL");
        fails += maxerr >= 1e-4f;
    }
    { // s8
        signed char *sA, *sB; int* sC;
        cudaMallocManaged(&sA, 16 * 32); cudaMallocManaged(&sB, 32 * 8); cudaMallocManaged(&sC, 16 * 8 * 4);
        for (int i = 0; i < 512; ++i) sA[i] = rand() % 255 - 127;
        for (int i = 0; i < 256; ++i) sB[i] = rand() % 255 - 127;
        mma_s8_kernel<<<1, 32>>>(sA, sB, sC); cudaDeviceSynchronize();
        int bad = 0;
        for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) {
            int ref = (m < 8) ? 1 + (n & 1) : 3 + (n & 1);
            for (int k = 0; k < 32; ++k) ref += int(sA[m * 32 + k]) * int(sB[k * 8 + n]);
            bad += ref != sC[m * 8 + n];
        }
        printf("mma_s8 m16n8k32: mismatches %d %s\n", bad, bad ? "FAIL" : "OK");
        fails += bad != 0;
    }
    { // ldmatrix
        unsigned short* src; unsigned* out;
        cudaMallocManaged(&src, 256 * 2); cudaMallocManaged(&out, 4 * 32 * 4 * 4);
        for (int i = 0; i < 256; ++i) src[i] = static_cast<unsigned short>(i + 1);
        ldmatrix_kernel<<<1, 32>>>(src, out); cudaDeviceSynchronize();
        int bad = 0;
        auto M = [&](int m, int r, int c) { return static_cast<unsigned>(src[m * 64 + r * 8 + c]); };
        for (int lane = 0; lane < 32; ++lane) for (int i = 0; i < 4; ++i) {
            const int g = lane >> 2, t = lane & 3;
            unsigned e  = M(i, g, 2 * t) | (M(i, g, 2 * t + 1) << 16);
            unsigned et = M(i, 2 * t, g) | (M(i, 2 * t + 1, g) << 16);
            bad += out[(0 * 32 + lane) * 4 + i] != e;
            bad += out[(1 * 32 + lane) * 4 + i] != et;
            if (i < 2) { bad += out[(2 * 32 + lane) * 4 + i] != e; bad += out[(3 * 32 + lane) * 4 + i] != et; }
        }
        printf("ldmatrix x2/x4 (+trans): mismatches %d %s\n", bad, bad ? "FAIL" : "OK");
        fails += bad != 0;
    }
    printf(fails ? "FAILED\n" : "ALL PASS\n");
    return fails != 0;
}
