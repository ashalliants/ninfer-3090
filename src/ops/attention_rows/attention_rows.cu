// Row bookkeeping of DeepSeek-V4's attention layer; the contract is
// include/ninfer/ops/attention_rows.h. Written for NInfer-3090 against tools/dsv4/attention.py.

#include "ninfer/ops/attention_rows.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("attention_rows: ") + message); }
}

void launched(const char* name) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(name) + ": " + cudaGetErrorString(error));
    }
}

unsigned grid_for(long long items) {
    return static_cast<unsigned>(std::min<long long>((items + kThreads - 1) / kThreads, 1 << 20));
}

bool i32_vector(const Tensor& t, long long n) {
    return t.data != nullptr && t.dtype == DType::I32 && t.is_contiguous() && t.ne[0] == n &&
           t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
}

bool matrix(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 &&
           t.ne[3] == 1 && reinterpret_cast<std::uintptr_t>(t.data) % 4 == 0;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data),
               b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

__global__ void compressed_lengths_kernel(int start, int ratio, int count, int* __restrict__ out) {
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < count; s += gridDim.x * blockDim.x) {
        out[s] = (start + s + 1) / ratio;
    }
}

__global__ void row_lists_kernel(const int* __restrict__ topk, int topk_width, int start,
                                 int window, int raw_base, int comp_base, int columns,
                                 int* __restrict__ rows) {
    const long long width = window + topk_width;
    const long long total = width * columns;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long s = i / width;
        const int j       = static_cast<int>(i % width);
        int row;
        if (j < window) {
            row = (static_cast<long long>(start) + s - (window - 1) + j >= 0)
                      ? raw_base + static_cast<int>(s) + j
                      : -1;
        } else {
            const int v = topk[s * topk_width + (j - window)];
            row         = v >= 0 ? comp_base + v : -1;
        }
        rows[i] = row;
    }
}

// One warp per (block, column): the maximum over the block's positions.
__global__ void block_max_kernel(const float* __restrict__ scores, const int* __restrict__ valid,
                                 int n, int block_size, int blocks, long long total,
                                 float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    for (long long w = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         w < total; w += static_cast<long long>(gridDim.x) * blockDim.x / 32) {
        const long long s = w / blocks;
        const int b       = static_cast<int>(w % blocks);
        const int reach   = max(0, min(valid[s], n));
        float result;
        if (static_cast<long long>(b) * block_size >= reach) {
            result = -INFINITY;
        } else if (b == (reach - 1) / block_size) {
            result = INFINITY;
        } else {
            result = -INFINITY;
            for (int t = b * block_size + lane; t < (b + 1) * block_size; t += 32) {
                result = fmaxf(result, scores[s * n + t]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                result = fmaxf(result, __shfl_xor_sync(0xFFFFFFFFu, result, off));
            }
        }
        if (lane == 0) { out[s * blocks + b] = result; }
    }
}

__global__ void mask_kernel(const int* __restrict__ selected, int kb, int block_size, int n,
                            long long total, float* __restrict__ scores) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long s = i / n;
        const int block   = static_cast<int>(i % n) / block_size;
        const int* list   = selected + s * kb;
        int lo            = 0,
            hi = kb; // lower bound of `block` among the ascending ids (the -1 padding sorts last)
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            const int v   = list[mid] < 0 ? INT_MAX : list[mid];
            if (v < block) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (!(lo < kb && list[lo] == block)) { scores[i] = -INFINITY; }
    }
}

} // namespace

void compressed_lengths(std::int32_t start, std::int32_t ratio, Tensor& out, cudaStream_t stream) {
    require(out.data != nullptr && out.dtype == DType::I32 && out.is_contiguous() &&
                out.ne[1] == 1 && out.ne[2] == 1 && out.ne[3] == 1 && out.ne[0] >= 1,
            "out must be a contiguous I32 vector [T]");
    require(start >= 0 && ratio >= 1 && static_cast<long long>(start) + out.ne[0] <= INT32_MAX,
            "start or ratio out of range");
    compressed_lengths_kernel<<<grid_for(out.ne[0]), kThreads, 0, stream>>>(
        start, ratio, out.ne[0], static_cast<int*>(out.data));
    launched("compressed_lengths");
}

void attention_row_lists(const Tensor* topk, std::int32_t start, std::int32_t window,
                         std::int32_t raw_base, std::int32_t comp_base, Tensor& rows,
                         cudaStream_t stream) {
    require(matrix(rows, DType::I32), "rows must be a contiguous I32 matrix");
    const int columns = rows.ne[1], topk_width = topk != nullptr ? topk->ne[0] : 0;
    require(window >= 1 && start >= 0 && raw_base >= 0 && comp_base >= 0,
            "window, start or a base out of range");
    require(rows.ne[0] == window + topk_width && columns >= 1, "rows must be [window + Kt, T]");
    if (topk != nullptr) {
        require(matrix(*topk, DType::I32) && topk->ne[1] == columns && topk_width >= 1,
                "topk must be I32 [Kt, T]");
        require(!overlaps(rows, *topk), "rows overlaps topk");
    }
    require(static_cast<long long>(raw_base) + columns + window <= INT32_MAX &&
                static_cast<long long>(start) + columns <= INT32_MAX,
            "a row index does not fit I32");
    const long long total = static_cast<long long>(rows.ne[0]) * columns;
    row_lists_kernel<<<grid_for(total), kThreads, 0, stream>>>(
        topk != nullptr ? static_cast<const int*>(topk->data) : nullptr, topk_width, start, window,
        raw_base, comp_base, columns, static_cast<int*>(rows.data));
    launched("attention_row_lists");
}

void block_max_scores(const Tensor& scores, const Tensor& valid, std::int32_t block_size,
                      Tensor& out, cudaStream_t stream) {
    require(matrix(scores, DType::FP32) && matrix(out, DType::FP32),
            "scores and out must be contiguous FP32 matrices");
    const int n = scores.ne[0], columns = scores.ne[1];
    require(block_size >= 1 && n >= 1 && columns >= 1, "block_size, N or T out of range");
    require(i32_vector(valid, columns), "valid must be I32 [T]");
    const int blocks = (n + block_size - 1) / block_size;
    require(out.ne[0] == blocks && out.ne[1] == columns, "out must be [ceil(N / block_size), T]");
    require(!overlaps(out, scores) && !overlaps(out, valid), "out overlaps an input");
    const long long total = static_cast<long long>(blocks) * columns;
    block_max_kernel<<<grid_for(total * 32), kThreads, 0, stream>>>(
        static_cast<const float*>(scores.data), static_cast<const int*>(valid.data), n, block_size,
        blocks, total, static_cast<float*>(out.data));
    launched("block_max_scores");
}

void mask_scores_outside_blocks(const Tensor& selected, std::int32_t block_size, Tensor& scores,
                                cudaStream_t stream) {
    require(matrix(scores, DType::FP32) && matrix(selected, DType::I32),
            "scores FP32 and selected I32 matrices are required");
    const int n = scores.ne[0], columns = scores.ne[1];
    require(block_size >= 1 && n >= 1 && columns >= 1 && selected.ne[1] == columns &&
                selected.ne[0] >= 1,
            "shapes disagree");
    require(!overlaps(scores, selected), "scores overlaps selected");
    const long long total = static_cast<long long>(n) * columns;
    mask_kernel<<<grid_for(total), kThreads, 0, stream>>>(static_cast<const int*>(selected.data),
                                                          selected.ne[0], block_size, n, total,
                                                          static_cast<float*>(scores.data));
    launched("mask_scores_outside_blocks");
}

} // namespace ninfer::ops
