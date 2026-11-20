// The selection half of DeepSeek-V4's indexer; the contract is include/ninfer/ops/position_selection.h.
// Written for NInfer-3090 against the model repo's `Indexer` (inference/model.py).

#include "ninfer/ops/position_selection.h"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 512;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("top_k_positions: ") + message); }
}

// Order-preserving map of a float to an unsigned key (larger float, larger key); -0 and +0 map alike.
__device__ __forceinline__ std::uint32_t ordered_key(float f) {
    if (f == 0.0F) { f = 0.0F; }
    const std::uint32_t u = __float_as_uint(f);
    return (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
}

// One block per column. A four-pass radix select (8 bits a pass, most significant first) finds the
// key of the m-th largest score, and how many of the positions holding exactly that key are needed;
// then one scan of the row in position order keeps every position above the key plus the first
// `need` positions at it, which is the lower-position tie rule and writes the result already sorted.
__global__ void __launch_bounds__(kThreads)
top_k_kernel(const float* __restrict__ scores, const std::int32_t* __restrict__ valid, int n_total, int k, int offset,
             std::int32_t* __restrict__ out) {
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan_storage;
    __shared__ unsigned histogram[256];
    __shared__ std::uint32_t shared_prefix;
    __shared__ int shared_remaining;

    const int s        = blockIdx.x;
    const int n        = min(max(valid[s], 0), n_total);
    const int m        = min(k, n);
    const float* row   = scores + static_cast<std::size_t>(s) * n_total;
    std::int32_t* dest = out + static_cast<std::size_t>(s) * k;

    std::uint32_t threshold = 0;
    int need_equal          = 0;
    const bool everything   = n <= k; // no selection to make: every reachable position is kept
    if (!everything) {
        std::uint32_t prefix = 0, mask = 0;
        int remaining = m;
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int b = threadIdx.x; b < 256; b += kThreads) { histogram[b] = 0; }
            __syncthreads();
            for (int i = threadIdx.x; i < n; i += kThreads) {
                const std::uint32_t key = ordered_key(row[i]);
                if ((key & mask) == prefix) { atomicAdd(&histogram[(key >> shift) & 255U], 1U); }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int above = 0, bin = 255;
                for (; bin > 0; --bin) {
                    if (above + static_cast<int>(histogram[bin]) >= remaining) { break; }
                    above += static_cast<int>(histogram[bin]);
                }
                shared_prefix    = prefix | (static_cast<std::uint32_t>(bin) << shift);
                shared_remaining = remaining - above;
            }
            __syncthreads();
            prefix    = shared_prefix;
            remaining = shared_remaining;
            mask |= 0xFFU << shift;
        }
        threshold  = prefix;
        need_equal = remaining; // positions holding exactly the threshold key that are still needed
    }

    int kept = 0, equal_seen = 0;
    for (int base = 0; base < n; base += kThreads) {
        const int i = base + threadIdx.x;
        bool is_above = false, is_equal = false;
        if (i < n) {
            if (everything) {
                is_above = true;
            } else {
                const std::uint32_t key = ordered_key(row[i]);
                is_above                = key > threshold;
                is_equal                = key == threshold;
            }
        }
        int equal_rank, equal_total;
        Scan(scan_storage).ExclusiveSum(is_equal ? 1 : 0, equal_rank, equal_total);
        __syncthreads();
        const bool selected = is_above || (is_equal && equal_seen + equal_rank < need_equal);
        int rank, total;
        Scan(scan_storage).ExclusiveSum(selected ? 1 : 0, rank, total);
        __syncthreads();
        if (selected) { dest[kept + rank] = i + offset; }
        kept += total;
        equal_seen += equal_total;
    }
    for (int i = kept + threadIdx.x; i < k; i += kThreads) { dest[i] = -1; }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1 &&
           reinterpret_cast<std::uintptr_t>(t.data) % 4 == 0;
}

} // namespace

void top_k_positions(const Tensor& scores, const Tensor& valid, std::int32_t k, std::int32_t offset, Tensor& out,
                     cudaStream_t stream) {
    require(plain(scores, DType::FP32) && plain(valid, DType::I32) && plain(out, DType::I32),
            "contiguous FP32 scores and I32 valid and out are required");
    const int n = scores.ne[0], columns = scores.ne[1];
    require(k >= 1 && k <= kTopKPositionsMax, "k out of range");
    require(n >= 1 && columns >= 1 && offset >= 0, "extent or offset out of range");
    require(valid.numel() == columns && out.ne[0] == k && out.ne[1] == columns, "shapes disagree");
    require(!overlaps(out, scores) && !overlaps(out, valid), "out overlaps an input");
    top_k_kernel<<<columns, kThreads, 0, stream>>>(static_cast<const float*>(scores.data),
                                                   static_cast<const std::int32_t*>(valid.data), n, k, offset,
                                                   static_cast<std::int32_t*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("top_k_positions: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops
