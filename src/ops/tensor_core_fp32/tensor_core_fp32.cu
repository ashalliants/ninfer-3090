// Written for NInfer-3090. The contract is include/ninfer/ops/tensor_core_fp32.h.

#include "ninfer/ops/tensor_core_fp32.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;
constexpr std::size_t kCublasWorkspaceBytes = std::size_t{4} << 20;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error(std::string("tensor_core_fp32: ") + what + " failed: " + std::to_string(int(status))); }
}

// One handle per device, made on first use.
cublasHandle_t handle() {
    constexpr int kMaxDevices = 16;
    static cublasHandle_t handles[kMaxDevices] = {};
    static std::mutex mutex;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) { throw std::runtime_error("tensor_core_fp32: no device"); }
    std::lock_guard<std::mutex> lock(mutex);
    if (handles[device] == nullptr) {
        check_blas(cublasCreate(&handles[device]), "cublasCreate");
        check_blas(cublasSetMathMode(handles[device], CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    }
    return handles[device];
}

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) { red[warp] = v; }
    __syncthreads();
    float m = red[0];
    for (int w = 1; w < kThreads / 32; ++w) { m = fmaxf(m, red[w]); }
    return m;
}

// 2^e with absmax 2^e in [2^14, 2^15) (e = 0 for a zero or non-finite absmax).
__device__ int scale_exponent(float absmax) {
    if (!(absmax > 0.0F) || !isfinite(absmax)) { return 0; }
    int x = 0;
    (void)frexpf(absmax, &x); // absmax = f 2^x, f in [0.5, 1)
    return 15 - x;
}

// One block per row of n BF16 values: FP16(v 2^e) and 2^-e; with `inexact`, counts the values not carried exactly.
__global__ void __launch_bounds__(kThreads)
bf16_rows_to_f16(const __nv_bfloat16* __restrict__ src, std::int64_t stride, int n, __half* __restrict__ dst, float* __restrict__ inverse,
                 unsigned long long* inexact) {
    __shared__ float red[kThreads / 32];
    const __nv_bfloat16* row = src + static_cast<std::int64_t>(blockIdx.x) * stride;
    __half* out = dst != nullptr ? dst + static_cast<std::int64_t>(blockIdx.x) * n : nullptr;
    float m = 0.0F;
    for (int i = threadIdx.x; i < n; i += kThreads) { m = fmaxf(m, fabsf(__bfloat162float(row[i]))); }
    const int e = scale_exponent(block_max(m, red));
    unsigned count = 0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float v = __bfloat162float(row[i]);
        const __half h = __float2half_rn(ldexpf(v, e));
        if (out != nullptr) { out[i] = h; }
        count += ldexpf(__half2float(h), -e) != v ? 1U : 0U;
    }
    if (inexact != nullptr && count > 0) { atomicAdd(inexact, static_cast<unsigned long long>(count)); }
    if (inverse != nullptr && threadIdx.x == 0) { inverse[blockIdx.x] = ldexpf(1.0F, -e); }
}

// out[r, t] *= row_inverse[r] col_inverse[t] for rows [0, rows) of each column (column stride ld).
__global__ void __launch_bounds__(kThreads)
unscale(float* __restrict__ out, std::int64_t ld, int rows, const float* __restrict__ row_inverse, const float* __restrict__ col_inverse) {
    const int r = blockIdx.x * kThreads + threadIdx.x, t = blockIdx.y;
    if (r >= rows) { return; }
    float* p = out + static_cast<std::int64_t>(t) * ld + r;
    *p = *p * row_inverse[r] * col_inverse[t];
}

std::size_t aligned(std::size_t n) { return (n + 255) / 256 * 256; }

// ---- Experts.

constexpr int kNvfp4Values = 64, kNvfp4Bytes = 36;

// Twice the E2M1 value of a nibble (0, 1, 2, 3, 4, 6, 8, 12 and their negatives), with no table.
__device__ __forceinline__ float nvfp4_code(unsigned nibble) {
    const int c = static_cast<int>(nibble & 7);
    const int mag = c < 2 ? c : (2 + (c & 1)) << ((c >> 1) - 1);
    return static_cast<float>((nibble & 8) != 0 ? -mag : mag);
}

__device__ float ue4m3_half(unsigned byte) {
    if (byte == 0 || byte == 0x7F) { return 0.0F; }
    const int exponent = (byte >> 3) & 0xF, mantissa = byte & 7;
    const float v = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9) : ldexpf(1.0F + mantissa / 8.0F, exponent - 7);
    return v * 0.5F;
}

// Value i of an NVFP4 row (blocks of 36 bytes: 4 scale bytes, 32 code bytes; sub-block s of 16 values: j < 8 the
// low nibble of code byte 8s + j, j + 8 the high one).
__device__ float nvfp4_value(const std::uint8_t* row, int i) {
    const std::uint8_t* blk = row + (i / kNvfp4Values) * kNvfp4Bytes;
    const int v = i % kNvfp4Values, sub = v / 16, j = v % 16;
    const unsigned q = blk[4 + 8 * sub + (j & 7)];
    return nvfp4_code(j < 8 ? q & 0xF : q >> 4) * ue4m3_half(blk[sub]);
}

// One block per row r of an expert's NVFP4 rows (rows r < split from first, the others from second): FP16(v 2^e)
// and 2^-e; with `inexact`, counts the values not carried exactly.
__global__ void __launch_bounds__(kThreads)
nvfp4_rows_to_f16(const std::uint8_t* __restrict__ first, const std::uint8_t* __restrict__ second, int split, std::uint64_t row_bytes, int n,
                  __half* __restrict__ dst, float* __restrict__ inverse, unsigned long long* inexact) {
    __shared__ float red[kThreads / 32];
    const int r = blockIdx.x;
    const std::uint8_t* row = r < split ? first + static_cast<std::uint64_t>(r) * row_bytes : second + static_cast<std::uint64_t>(r - split) * row_bytes;
    float m = 0.0F;
    for (int i = threadIdx.x; i < n; i += kThreads) { m = fmaxf(m, fabsf(nvfp4_value(row, i))); }
    const int e = scale_exponent(block_max(m, red));
    __half* out = dst != nullptr ? dst + static_cast<std::int64_t>(r) * n : nullptr;
    unsigned count = 0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float v = nvfp4_value(row, i);
        const __half h = __float2half_rn(ldexpf(v, e));
        if (out != nullptr) { out[i] = h; }
        count += ldexpf(__half2float(h), -e) != v ? 1U : 0U;
    }
    if (inexact != nullptr && count > 0) { atomicAdd(inexact, static_cast<unsigned long long>(count)); }
    if (inverse != nullptr && threadIdx.x == 0) { inverse[r] = ldexpf(1.0F, -e); }
}

// One block per column j of n FP32 values (column src + col(j) * stride): hi = FP16(v 2^e), lo = FP16(v 2^e - hi)
// (lo may be null), inverse 2^-e.
template <bool kGather>
__global__ void __launch_bounds__(kThreads)
fp32_cols_to_f16(const float* __restrict__ src, std::int64_t stride, const int* __restrict__ order, int top_k, int n, __half* __restrict__ hi,
                 __half* __restrict__ lo, float* __restrict__ inverse) {
    __shared__ float red[kThreads / 32];
    const int j = blockIdx.x;
    const float* col = src + (kGather ? static_cast<std::int64_t>(order[j] / top_k) : static_cast<std::int64_t>(j)) * stride;
    float m = 0.0F;
    for (int i = threadIdx.x; i < n; i += kThreads) { m = fmaxf(m, fabsf(col[i])); }
    const int e = scale_exponent(block_max(m, red));
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float v = ldexpf(col[i], e);
        const __half h = __float2half_rn(v);
        hi[static_cast<std::int64_t>(j) * n + i] = h;
        if (lo != nullptr) { lo[static_cast<std::int64_t>(j) * n + i] = __float2half_rn(v - __half2float(h)); }
    }
    if (threadIdx.x == 0) { inverse[j] = ldexpf(1.0F, -e); }
}

__device__ __forceinline__ float situ(float g, float u, float gate_cap, float up_cap) {
    return gate_cap * tanhf(g / gate_cap) / (1.0F + expf(-g)) * up_cap * tanhf(u / up_cap);
}

// One block per column j of the expert's gate/up results c [2I, n]: a = situ(g, u) in FP32 into act (FP32 [I, n]).
__global__ void __launch_bounds__(kThreads)
situ_cols(const float* __restrict__ c, int inter, const float* __restrict__ row_inverse, const float* __restrict__ col_inverse, const float* scale_e,
          float gate_cap, float up_cap, float* __restrict__ act) {
    const int j = blockIdx.x;
    const float s = (scale_e != nullptr ? *scale_e : 1.0F) * col_inverse[j];
    const float* cj = c + static_cast<std::int64_t>(j) * 2 * inter;
    for (int r = threadIdx.x; r < inter; r += kThreads) {
        const float g = cj[r] * row_inverse[r] * s, u = cj[inter + r] * row_inverse[inter + r] * s;
        act[static_cast<std::int64_t>(j) * inter + r] = situ(g, u, gate_cap, up_cap);
    }
}

// y[:, order[j]] = c[:, j] row_inverse col_inverse[j] scale.
__global__ void __launch_bounds__(kThreads)
down_cols(const float* __restrict__ c, int hidden, const float* __restrict__ row_inverse, const float* __restrict__ col_inverse, const float* scale_e,
          const int* __restrict__ order, float* __restrict__ y) {
    const int j = blockIdx.x;
    const float s = (scale_e != nullptr ? *scale_e : 1.0F) * col_inverse[j];
    float* yj = y + static_cast<std::int64_t>(order[j]) * hidden;
    for (int r = threadIdx.x; r < hidden; r += kThreads) { yj[r] = c[static_cast<std::int64_t>(j) * hidden + r] * row_inverse[r] * s; }
}

// out[:, t] = sum_i weights[i, t] y[:, t k + i], in order, one FMA each from 0.
__global__ void __launch_bounds__(kThreads)
combine_slots(const float* __restrict__ y, const float* __restrict__ weights, int hidden, int top_k, float* __restrict__ out) {
    const int t = blockIdx.x;
    for (int r = threadIdx.x; r < hidden; r += kThreads) {
        float acc = 0.0F;
        for (int i = 0; i < top_k; ++i) { acc = fmaf(weights[t * top_k + i], y[(static_cast<std::int64_t>(t) * top_k + i) * hidden + r], acc); }
        out[static_cast<std::int64_t>(t) * hidden + r] = acc;
    }
}

// ---- Grouped NVFP4 expert GEMMs on the tensor cores (WMMA 16x16x16, FP16 in, FP32 accumulate).
//
// Every NVFP4 value (an E2M1 code times a UE4M3 scale, halved) is exact in FP16 with no scaling: at most 6
// significant bits, magnitudes in [2^-10, 2688]. A CTA computes 128 rows of one expert by up to 64 of its entries:
// each 64-value step decodes one NVFP4 block per row into shared memory and stages the entries' FP16 inputs beside it.

constexpr int kEM = 128, kEN = 64, kEK = 64, kELd = kEK + 8, kECLd = kEN / 2 + 4, kEThreads = 256;

struct ExpertTile {
    int expert, start, count; // entries [start, start + count) of the grouped order, all of `expert`
};

struct ExpertGemmArgs {
    const std::uint8_t* records;
    std::uint64_t stride, first, second, row_bytes; // gate/up: rows from first (gate) and second (up); down: first
    int k;                                          // the reduced dimension (H for gate/up, I for down)
    bool tiled;                                     // the records' tiled order (nvfp4_tiled_block), in groups of
    int group_rows, run_rows;                       // group_rows rows of runs of run_rows
    const ExpertTile* tiles;
    const __half* x;        // FP16 [entries, k], entry-major, in the grouped order
    const float* x_inverse; // per entry (gate/up), or null: x_scale_inverse for all (down)
    float x_scale_inverse;
    const float* scale_e;   // per expert, or null
    // gate/up: act[entry, I] = FP16(situ(g, u) act_scale)
    int inter;
    float gate_cap, up_cap, act_scale;
    __half* act;
    // down: y[:, order[entry]] (H rows)
    int hidden;
    const int* order;
    float* y;
};

// Two FP16 values a * scale, b * scale (exact: at most 6 significant bits) as one word.
__device__ __forceinline__ unsigned nvfp4_pair(unsigned a, unsigned b, float scale) {
    const __half2 h = __floats2half2_rn(nvfp4_code(a) * scale, nvfp4_code(b) * scale);
    return *reinterpret_cast<const unsigned*>(&h);
}

template <bool GateUp>
__global__ void __launch_bounds__(kEThreads) expert_gemm_kernel(ExpertGemmArgs a) {
    namespace wm = nvcuda::wmma;
    constexpr int kInBytes = (kEM + kEN) * kELd * 2, kOutBytes = kEM * kECLd * 4;
    __shared__ __align__(32) unsigned char smem[kInBytes > kOutBytes ? kInBytes : kOutBytes];
    __half (*A)[kELd] = reinterpret_cast<__half (*)[kELd]>(smem);
    __half (*B)[kELd] = reinterpret_cast<__half (*)[kELd]>(smem + kEM * kELd * 2);
    float (*C)[kECLd] = reinterpret_cast<float (*)[kECLd]>(smem);
    const ExpertTile tile = a.tiles[blockIdx.y];
    // Warps 4 (rows) x 2 (entries), 32 x 64 each.
    const int tid = threadIdx.x, warp = tid >> 5, wm_row = (warp & 3) * 32, wn = warp >> 2, wn_col = wn * (kEN / 2);
    constexpr int kNF = kEN / 2 / 16; // entry fragments per warp
    const std::uint8_t* rec = a.records + static_cast<std::uint64_t>(tile.expert) * a.stride;
    // The row this thread decodes (two threads per row, 32 values each).
    const int lr = tid >> 1, half_id = tid & 1;
    const std::uint8_t* row;
    const std::uint8_t* matrix;
    int r;
    if constexpr (GateUp) {
        r      = blockIdx.x * (kEM / 2) + (lr & (kEM / 2 - 1));
        matrix = rec + (lr < kEM / 2 ? a.first : a.second);
    } else {
        r      = blockIdx.x * kEM + lr;
        matrix = rec + a.first;
    }
    row = matrix + static_cast<std::uint64_t>(r) * a.row_bytes;
    wm::fragment<wm::accumulator, 16, 16, 16, float> acc[2][kNF];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < kNF; ++j) { wm::fill_fragment(acc[i][j], 0.0F); }
    }
    // The next step's operands, loaded while the current one multiplies: the row's NVFP4 block (scales, then the two
    // sub-blocks' codes of this thread) and kBLoads 16-byte chunks of the entries' inputs.
    constexpr int kBLoads = kEN * (kEK / 8) / kEThreads;
    unsigned scales = 0, codes[4];
    uint4 bx[kBLoads];
    const auto load = [&](int k0) {
        const std::uint8_t* blk = a.tiled ? matrix + nvfp4_tiled_block(r, k0 / kEK, a.k / kEK, a.group_rows, a.run_rows) * kNvfp4Bytes
                                          : row + static_cast<std::uint64_t>(k0 / kEK) * kNvfp4Bytes;
        scales = *reinterpret_cast<const unsigned*>(blk);
#pragma unroll
        for (int w = 0; w < 4; ++w) { codes[w] = reinterpret_cast<const unsigned*>(blk + 4)[4 * half_id + w]; }
#pragma unroll
        for (int q = 0; q < kBLoads; ++q) {
            const int i = tid + q * kEThreads, n = i / (kEK / 8), c = (i % (kEK / 8)) * 8;
            bx[q] = n < tile.count ? *reinterpret_cast<const uint4*>(a.x + static_cast<std::int64_t>(tile.start + n) * a.k + k0 + c)
                                   : make_uint4(0, 0, 0, 0);
        }
    };
    load(0);
    for (int k0 = 0; k0 < a.k; k0 += kEK) {
#pragma unroll
        for (int ls = 0; ls < 2; ++ls) { // A: sub-blocks 2 half_id and 2 half_id + 1 of the row's block
            const int sub = 2 * half_id + ls;
            const float sc = ue4m3_half((scales >> (8 * sub)) & 0xFF);
            unsigned* dst = reinterpret_cast<unsigned*>(&A[lr][16 * sub]);
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const unsigned b0 = (codes[2 * ls + j / 4] >> (8 * (j % 4))) & 0xFF, b1 = (codes[2 * ls + j / 4] >> (8 * (j % 4 + 1))) & 0xFF;
                dst[j / 2]     = nvfp4_pair(b0, b1, sc);
                dst[4 + j / 2] = nvfp4_pair(b0 >> 4, b1 >> 4, sc);
            }
        }
#pragma unroll
        for (int q = 0; q < kBLoads; ++q) { // B: [kEN entries][64 values], zero past the tile's entries
            const int i = tid + q * kEThreads, n = i / (kEK / 8), c = (i % (kEK / 8)) * 8;
            *reinterpret_cast<uint4*>(&B[n][c]) = bx[q];
        }
        __syncthreads();
        if (k0 + kEK < a.k) { load(k0 + kEK); }
#pragma unroll
        for (int kk = 0; kk < kEK; kk += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> fa[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) { wm::load_matrix_sync(fa[i], &A[wm_row + 16 * i][kk], kELd); }
#pragma unroll
            for (int j = 0; j < kNF; ++j) {
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> fb;
                wm::load_matrix_sync(fb, &B[wn_col + 16 * j][kk], kELd);
#pragma unroll
                for (int i = 0; i < 2; ++i) { wm::mma_sync(acc[i][j], fa[i], fb, acc[i][j]); }
            }
        }
        __syncthreads();
    }
    const float se = a.scale_e != nullptr ? a.scale_e[tile.expert] : 1.0F;
    // The epilogue in two halves of kEN / 2 entries through C.
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        if (wn == part) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < kNF; ++j) { wm::store_matrix_sync(&C[wm_row + 16 * i][16 * j], acc[i][j], kECLd, wm::mem_row_major); }
            }
        }
        __syncthreads();
        const int n0 = part * (kEN / 2);
        if constexpr (GateUp) {
            for (int i = tid; i < (kEM / 2) * (kEN / 2); i += kEThreads) {
                const int r = i & (kEM / 2 - 1), n = i / (kEM / 2);
                if (n0 + n >= tile.count) { continue; }
                const float s = se * a.x_inverse[tile.start + n0 + n];
                const float g = C[r][n] * s, u = C[kEM / 2 + r][n] * s;
                const float v = situ(g, u, a.gate_cap, a.up_cap) * a.act_scale;
                a.act[static_cast<std::int64_t>(tile.start + n0 + n) * a.inter + blockIdx.x * (kEM / 2) + r] = __float2half_rn(v);
            }
        } else {
            for (int i = tid; i < kEM * (kEN / 2); i += kEThreads) {
                const int r = i & (kEM - 1), n = i / kEM;
                if (n0 + n >= tile.count) { continue; }
                a.y[static_cast<std::int64_t>(a.order[tile.start + n0 + n]) * a.hidden + blockIdx.x * kEM + r] = C[r][n] * a.x_scale_inverse * se;
            }
        }
        __syncthreads();
    }
}

struct BankGeometry {
    bool nvfp4;
    std::uint64_t gate_row_bytes, down_row_bytes, up_offset, down_offset, stride;
};

std::uint64_t align256(std::uint64_t n) { return (n + 255) / 256 * 256; }

BankGeometry geometry(const SituExpertBank& bank) {
    BankGeometry g{};
    g.nvfp4 = bank.format == QType::GGML_REC_NVFP4_NVFP4;
    if (g.nvfp4) {
        g.gate_row_bytes = static_cast<std::uint64_t>(bank.hidden / kNvfp4Values) * kNvfp4Bytes;
        g.down_row_bytes = static_cast<std::uint64_t>(bank.intermediate / kNvfp4Values) * kNvfp4Bytes;
        const std::uint64_t gate_bytes = g.gate_row_bytes * bank.intermediate;
        g.up_offset = align256(gate_bytes);
        g.down_offset = align256(g.up_offset + gate_bytes);
        g.stride = align256(g.down_offset + g.down_row_bytes * bank.hidden);
    }
    return g;
}

} // namespace

std::size_t projection_tc_workspace_bytes(std::int32_t k, std::int32_t max_rows, std::int32_t cols) {
    const std::size_t kk = static_cast<std::size_t>(k), r = static_cast<std::size_t>(max_rows), t = static_cast<std::size_t>(cols);
    return aligned(r * kk * 2) + aligned(r * 4) + aligned(t * kk * 2) + aligned(t * 4) + kCublasWorkspaceBytes + 8 * 256;
}

void projection_tc_fp32(std::span<const Tensor* const> inputs, std::span<const Tensor* const> weights, Tensor& out,
                        DeviceArena& workspace, cudaStream_t stream) {
    const char* op = "projection_tc_fp32";
    require(!weights.empty() && (inputs.size() == 1 || inputs.size() == weights.size()), op, "one input, or one per weight");
    const std::int32_t k = inputs[0]->ne[0], cols = inputs[0]->ne[1];
    require(k % 8 == 0, op, "K must be a multiple of 8");
    std::int32_t total = 0, widest = 0;
    for (const Tensor* w : weights) {
        require(w->dtype == DType::BF16 && w->ne[0] == k && w->is_contiguous(), op, "weights must be contiguous BF16 [K, N]");
        total += w->ne[1];
        widest = std::max(widest, w->ne[1]);
    }
    for (const Tensor* x : inputs) {
        require(x->dtype == DType::BF16 && x->ne[0] == k && x->ne[1] == cols && x->is_contiguous(), op, "inputs must be contiguous BF16 [K, T]");
    }
    require(out.dtype == DType::FP32 && out.ne[0] == total && out.ne[1] == cols && out.is_contiguous(), op, "out must be contiguous FP32 [sum N, T]");
    const auto scope = workspace.scope();
    auto* w16  = static_cast<__half*>(workspace.alloc_bytes(static_cast<std::size_t>(widest) * k * 2).data);
    auto* winv = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(widest) * 4).data);
    auto* x16  = static_cast<__half*>(workspace.alloc_bytes(static_cast<std::size_t>(cols) * k * 2).data);
    auto* xinv = static_cast<float*>(workspace.alloc_bytes(static_cast<std::size_t>(cols) * 4).data);
    void* blas_ws = workspace.alloc_bytes(kCublasWorkspaceBytes).data;
    cublasHandle_t h = handle();
    check_blas(cublasSetStream(h, stream), "cublasSetStream");
    check_blas(cublasSetWorkspace(h, blas_ws, kCublasWorkspaceBytes), "cublasSetWorkspace");
    const float one = 1.0F, zero = 0.0F;
    std::int32_t row = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (i == 0 || inputs.size() > 1) {
            bf16_rows_to_f16<<<cols, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(inputs[inputs.size() > 1 ? i : 0]->data), k, k, x16,
                                                            xinv, nullptr);
            check_launch(op);
        }
        const std::int32_t n = weights[i]->ne[1];
        bf16_rows_to_f16<<<n, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(weights[i]->data), k, k, w16, winv, nullptr);
        check_launch(op);
        float* c = static_cast<float*>(out.data) + row;
        check_blas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, cols, k, &one, w16, CUDA_R_16F, k, x16, CUDA_R_16F, k, &zero, c, CUDA_R_32F,
                                total, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "cublasGemmEx");
        unscale<<<dim3((n + kThreads - 1) / kThreads, cols), kThreads, 0, stream>>>(c, total, n, winv, xinv);
        check_launch(op);
        row += n;
    }
}

std::uint64_t bf16_fp16_inexact(const Tensor& w, cudaStream_t stream) {
    const char* op = "bf16_fp16_inexact";
    require(w.dtype == DType::BF16 && w.is_contiguous(), op, "w must be contiguous BF16 [K, N]");
    unsigned long long* count = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&count), sizeof *count, stream) != cudaSuccess) { throw std::runtime_error("bf16_fp16_inexact: alloc"); }
    cudaMemsetAsync(count, 0, sizeof *count, stream);
    bf16_rows_to_f16<<<w.ne[1], kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(w.data), w.ne[0], w.ne[0], nullptr, nullptr, count);
    check_launch(op);
    unsigned long long host = 0;
    cudaMemcpyAsync(&host, count, sizeof host, cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(count, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("bf16_fp16_inexact: sync"); }
    return host;
}

std::size_t moe_situ_experts_tc_workspace_bytes(const SituExpertBank& bank, std::int64_t entries) {
    const std::size_t h = static_cast<std::size_t>(bank.hidden), i = static_cast<std::size_t>(bank.intermediate), n = static_cast<std::size_t>(entries);
    const std::size_t tiles = n / kEN + static_cast<std::size_t>(bank.experts) + 1;
    // x16, its inverses, the order, the tiles, act16, y; the BF16 banks' route adds the decoded rows and the GEMM results.
    const std::size_t common = aligned(h * n * 2) + aligned(n * 4) + aligned(n * 4) + aligned(tiles * sizeof(ExpertTile)) + aligned(i * n * 2) +
                               aligned(h * n * 4);
    const std::size_t bf16 = bank.format == QType::BF16 ? aligned(2 * i * h * 2) + aligned(2 * i * 4) + aligned(h * i * 2) + aligned(h * 4) +
                                                          aligned(2 * i * n * 4) + aligned(i * n * 4) + aligned(n * 4) + aligned(h * n * 4) +
                                                          kCublasWorkspaceBytes
                                                    : 0;
    return common + bf16 + 20 * 256;
}

void moe_situ_experts_tc(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor* weights, float gate_cap, float up_cap,
                         DeviceArena& workspace, Tensor& out, cudaStream_t stream) {
    const char* op = "moe_situ_experts_tc";
    const int hidden = bank.hidden, inter = bank.intermediate;
    const std::int32_t cols = z.ne[1], top_k = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(top_k) * cols;
    require(z.dtype == DType::FP32 && z.ne[0] == hidden && z.nb[0] == 4, op, "z must be FP32 [H, T]");
    require(ids.dtype == DType::I32 && ids.is_contiguous() && ids.ne[1] == cols, op, "ids must be contiguous I32 [k, T]");
    require(out.dtype == DType::FP32 && out.is_contiguous() && out.ne[0] == hidden && out.ne[1] == (weights != nullptr ? cols : n), op,
            "out must be FP32 [H, T] (or [H, k T] without weights)");
    const BankGeometry g = geometry(bank);
    require(g.nvfp4 ? bank.record_stride == g.stride && hidden % kEM == 0 && inter % kEK == 0 && hidden % kEK == 0
                    : bank.format == QType::BF16 && hidden % 8 == 0 && inter % 8 == 0,
            op, "the bank must be NVFP4 records (H a multiple of 128, I of 64) or BF16 banks");
    require(gate_cap > 0.0F && up_cap > 0.0F, op, "the caps must be positive");
    const std::int64_t zs = cols == 1 ? hidden : z.nb[1] / 4;

    // The entries grouped by expert on the host (stable).
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(n));
    if (cudaMemcpyAsync(host_ids.data(), ids.data, n * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": the ids download failed");
    }
    std::vector<int> counts(static_cast<std::size_t>(bank.experts), 0), offsets(static_cast<std::size_t>(bank.experts) + 1, 0), order(static_cast<std::size_t>(n));
    for (const auto e : host_ids) {
        require(e >= 0 && e < bank.experts, op, "an id is out of range");
        ++counts[static_cast<std::size_t>(e)];
    }
    for (int e = 0; e < bank.experts; ++e) { offsets[e + 1] = offsets[e] + counts[e]; }
    {
        std::vector<int> next(offsets.begin(), offsets.end() - 1);
        for (std::int64_t i = 0; i < n; ++i) { order[static_cast<std::size_t>(next[host_ids[i]]++)] = static_cast<int>(i); }
    }
    std::vector<ExpertTile> tiles;
    for (int e = 0; e < bank.experts; ++e) {
        for (int at = offsets[e]; at < offsets[e + 1]; at += kEN) { tiles.push_back({e, at, std::min(kEN, offsets[e + 1] - at)}); }
    }

    const auto scope = workspace.scope();
    const auto take = [&](std::size_t bytes) { return workspace.alloc_bytes(bytes).data; };
    auto* x16 = static_cast<__half*>(take(static_cast<std::size_t>(hidden) * n * 2));
    auto* x_inv = static_cast<float*>(take(n * 4));
    auto* d_order = static_cast<int*>(take(n * 4));
    auto* d_tiles = static_cast<ExpertTile*>(take(std::max<std::size_t>(tiles.size(), 1) * sizeof(ExpertTile)));
    float* y = weights != nullptr ? static_cast<float*>(take(static_cast<std::size_t>(hidden) * n * 4)) : static_cast<float*>(out.data);
    if (cudaMemcpyAsync(d_order, order.data(), n * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        (!tiles.empty() && cudaMemcpyAsync(d_tiles, tiles.data(), tiles.size() * sizeof(ExpertTile), cudaMemcpyHostToDevice, stream) != cudaSuccess)) {
        throw std::runtime_error(std::string(op) + ": the order upload failed");
    }
    fp32_cols_to_f16<true><<<static_cast<unsigned>(n), kThreads, 0, stream>>>(static_cast<const float*>(z.data), zs, d_order, top_k, hidden, x16, nullptr, x_inv);
    check_launch(op);
    if (g.nvfp4) {
        // situ(g, u) is at most gate_cap up_cap in magnitude: FP16 under one power of two below 2^15.
        int act_exp = 0;
        (void)frexpf(gate_cap * up_cap, &act_exp); // gate_cap up_cap < 2^act_exp
        const float act_scale = ldexpf(1.0F, 15 - act_exp), act_inverse = ldexpf(1.0F, act_exp - 15);
        auto* act = static_cast<__half*>(take(static_cast<std::size_t>(inter) * n * 2));
        ExpertGemmArgs a{};
        a.records = bank.records, a.stride = g.stride, a.tiles = d_tiles;
        a.inter = inter, a.gate_cap = gate_cap, a.up_cap = up_cap, a.act_scale = act_scale, a.act = act;
        a.hidden = hidden, a.order = d_order, a.y = y;
        if (!tiles.empty()) {
            a.tiled = bank.tiled;
            a.first = 0, a.second = g.up_offset, a.row_bytes = g.gate_row_bytes, a.k = hidden;
            a.group_rows = kNvfp4GateGroupRows, a.run_rows = nvfp4_rows_per_run(hidden / kNvfp4Values);
            a.x = x16, a.x_inverse = x_inv, a.scale_e = bank.gate_up_scale;
            expert_gemm_kernel<true><<<dim3(inter / (kEM / 2), static_cast<unsigned>(tiles.size())), kEThreads, 0, stream>>>(a);
            check_launch(op);
            a.first = g.down_offset, a.second = 0, a.row_bytes = g.down_row_bytes, a.k = inter;
            a.group_rows = kNvfp4DownGroupRows, a.run_rows = nvfp4_rows_per_run(inter / kNvfp4Values);
            a.x = act, a.x_inverse = nullptr, a.x_scale_inverse = act_inverse, a.scale_e = bank.down_scale;
            expert_gemm_kernel<false><<<dim3(hidden / kEM, static_cast<unsigned>(tiles.size())), kEThreads, 0, stream>>>(a);
            check_launch(op);
        }
    } else {
        auto* w_gu = static_cast<__half*>(take(static_cast<std::size_t>(2) * inter * hidden * 2));
        auto* w_gu_inv = static_cast<float*>(take(static_cast<std::size_t>(2) * inter * 4));
        auto* w_d = static_cast<__half*>(take(static_cast<std::size_t>(hidden) * inter * 2));
        auto* w_d_inv = static_cast<float*>(take(static_cast<std::size_t>(hidden) * 4));
        auto* c1 = static_cast<float*>(take(static_cast<std::size_t>(2) * inter * n * 4));
        auto* act = static_cast<float*>(take(static_cast<std::size_t>(inter) * n * 4));
        auto* a16 = static_cast<__half*>(take(static_cast<std::size_t>(inter) * n * 2));
        auto* a_inv = static_cast<float*>(take(n * 4));
        auto* c2 = static_cast<float*>(take(static_cast<std::size_t>(hidden) * n * 4));
        void* blas_ws = take(kCublasWorkspaceBytes);
        cublasHandle_t hb = handle();
        check_blas(cublasSetStream(hb, stream), "cublasSetStream");
        check_blas(cublasSetWorkspace(hb, blas_ws, kCublasWorkspaceBytes), "cublasSetWorkspace");
        const float one = 1.0F, zero = 0.0F;
        for (int e = 0; e < bank.experts; ++e) {
            const int ne = counts[e], at = offsets[e];
            if (ne == 0) { continue; }
            const auto* gu = static_cast<const __nv_bfloat16*>(bank.gate_up) + static_cast<std::int64_t>(e) * 2 * inter * hidden;
            const auto* dn = static_cast<const __nv_bfloat16*>(bank.down) + static_cast<std::int64_t>(e) * hidden * inter;
            bf16_rows_to_f16<<<2 * inter, kThreads, 0, stream>>>(gu, hidden, hidden, w_gu, w_gu_inv, nullptr);
            bf16_rows_to_f16<<<hidden, kThreads, 0, stream>>>(dn, inter, inter, w_d, w_d_inv, nullptr);
            check_launch(op);
            const __half* xe = x16 + static_cast<std::int64_t>(at) * hidden;
            check_blas(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, 2 * inter, ne, hidden, &one, w_gu, CUDA_R_16F, hidden, xe, CUDA_R_16F, hidden, &zero, c1,
                                    CUDA_R_32F, 2 * inter, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                       "cublasGemmEx gate_up");
            situ_cols<<<ne, kThreads, 0, stream>>>(c1, inter, w_gu_inv, x_inv + at, bank.gate_up_scale != nullptr ? bank.gate_up_scale + e : nullptr,
                                                   gate_cap, up_cap, act);
            fp32_cols_to_f16<false><<<ne, kThreads, 0, stream>>>(act, inter, nullptr, 1, inter, a16, nullptr, a_inv);
            check_launch(op);
            check_blas(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, hidden, ne, inter, &one, w_d, CUDA_R_16F, inter, a16, CUDA_R_16F, inter, &zero, c2,
                                    CUDA_R_32F, hidden, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                       "cublasGemmEx down");
            down_cols<<<ne, kThreads, 0, stream>>>(c2, hidden, w_d_inv, a_inv, bank.down_scale != nullptr ? bank.down_scale + e : nullptr, d_order + at, y);
            check_launch(op);
        }
    }
    if (weights != nullptr) {
        require(weights->dtype == DType::FP32 && weights->is_contiguous() && weights->numel() == n, op, "weights must be FP32 [k, T]");
        combine_slots<<<cols, kThreads, 0, stream>>>(y, static_cast<const float*>(weights->data), hidden, top_k, static_cast<float*>(out.data));
        check_launch(op);
    }
    // The host vectors are read by the uploads above: wait for them before they go out of scope.
    if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error(std::string(op) + ": the experts failed"); }
}

std::uint64_t nvfp4_bank_fp16_inexact(const SituExpertBank& bank, cudaStream_t stream) {
    const char* op = "nvfp4_bank_fp16_inexact";
    const BankGeometry g = geometry(bank);
    require(g.nvfp4 && bank.record_stride == g.stride, op, "the bank must be NVFP4 records");
    unsigned long long* count = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void**>(&count), sizeof *count, stream) != cudaSuccess) { throw std::runtime_error("nvfp4_bank_fp16_inexact: alloc"); }
    cudaMemsetAsync(count, 0, sizeof *count, stream);
    for (int e = 0; e < bank.experts; ++e) {
        const std::uint8_t* rec = bank.records + static_cast<std::uint64_t>(e) * g.stride;
        nvfp4_rows_to_f16<<<2 * bank.intermediate, kThreads, 0, stream>>>(rec, rec + g.up_offset, bank.intermediate, g.gate_row_bytes, bank.hidden, nullptr,
                                                                          nullptr, count);
        nvfp4_rows_to_f16<<<bank.hidden, kThreads, 0, stream>>>(rec + g.down_offset, nullptr, bank.hidden, g.down_row_bytes, bank.intermediate, nullptr,
                                                                nullptr, count);
    }
    check_launch(op);
    unsigned long long host = 0;
    cudaMemcpyAsync(&host, count, sizeof host, cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(count, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("nvfp4_bank_fp16_inexact: sync"); }
    return host;
}

} // namespace ninfer::ops
