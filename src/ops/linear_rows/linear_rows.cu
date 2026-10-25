// linear_rows, merge_row_argmax and merge_row_topk (include/ninfer/ops/linear_rows.h): the draft
// head's context rows, scored from the full output head by id and merged into a proposal.

#include "ninfer/ops/linear_rows.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kRowsPerBlock    = 4;  // One warp per slot.
constexpr int kColumnTile      = 8;  // Columns one warp accumulates at once.
constexpr int kMergeThreads    = 256;
constexpr int kTopK            = 16;
constexpr int kMaximumTopKRows = 2048;

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void check_launch(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(error));
    }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto a = reinterpret_cast<std::uintptr_t>(lhs);
    const auto b = reinterpret_cast<std::uintptr_t>(rhs);
    return a < b + rhs_bytes && b < a + lhs_bytes;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

bool matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t columns) {
    return t.data != nullptr && t.dtype == dtype && t.ne[0] == rows && t.ne[1] == columns &&
           t.ne[2] == 1 && t.ne[3] == 1 && t.is_contiguous();
}

bool vector(const Tensor& t, DType dtype, std::int32_t count) {
    return matrix(t, dtype, count, 1);
}

// Eight consecutive logical codes of one group, `chunk` counting eight-code steps.
template <QType Format>
__device__ __forceinline__ void decode8(const std::uint8_t* __restrict__ base,
                                        const std::uint8_t* __restrict__ high, int chunk,
                                        float (&w)[8]) {
    if constexpr (Format == QType::Q8_G32_FP16) {
        const uint2 word = __ldg(reinterpret_cast<const uint2*>(base) + chunk);
        const std::uint32_t lanes[2]{word.x, word.y};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = static_cast<float>(static_cast<std::int8_t>((lanes[i >> 2] >> ((i & 3) * 8)) & 0xffU));
        }
    } else {
        const std::uint32_t low = __ldg(reinterpret_cast<const std::uint32_t*>(base) + chunk);
        std::uint32_t top       = 0;
        if constexpr (Format == QType::Q6_G64_FP16) {
            top = __ldg(reinterpret_cast<const std::uint16_t*>(high) + chunk);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            std::uint32_t u = (low >> (i * 4)) & 0xfU;
            if constexpr (Format == QType::Q6_G64_FP16) {
                u |= ((top >> (i * 2)) & 0x3U) << 4;
                w[i] = static_cast<float>(static_cast<int>(u) - ((u & 0x20U) ? 64 : 0));
            } else {
                w[i] = static_cast<float>(static_cast<int>(u) - ((u & 0x8U) ? 16 : 0));
            }
        }
    }
}

template <QType Format>
struct RowSplitGeometry {
    static constexpr int kGroup     = Format == QType::Q8_G32_FP16 ? 32 : 64;
    static constexpr int kBaseBytes = 32;
    static constexpr int kHighBytes = Format == QType::Q6_G64_FP16 ? 16 : 0;
};

template <QType Format>
__device__ __forceinline__ void linear_rows_tile(const __nv_bfloat16* __restrict__ hidden, int K,
                                                 int first, int columns,
                                                 const std::uint8_t* __restrict__ codes,
                                                 const std::uint8_t* __restrict__ high,
                                                 const __half* __restrict__ scales, int id, int C,
                                                 int lane, float* __restrict__ out, int slot) {
    using G = RowSplitGeometry<Format>;

    const int groups              = K / G::kGroup;
    const std::uint8_t* row_codes = codes + static_cast<std::size_t>(id) * groups * G::kBaseBytes;
    const std::uint8_t* row_high =
        G::kHighBytes == 0 ? nullptr
                           : high + static_cast<std::size_t>(id) * groups * G::kHighBytes;
    const __half* row_scales = scales + static_cast<std::size_t>(id) * groups;

    float acc[kColumnTile];
#pragma unroll
    for (int j = 0; j < kColumnTile; ++j) { acc[j] = 0.0F; }
    for (int group = lane; group < groups; group += 32) {
        const std::uint8_t* base = row_codes + static_cast<std::size_t>(group) * G::kBaseBytes;
        const std::uint8_t* top =
            G::kHighBytes == 0 ? nullptr : row_high + static_cast<std::size_t>(group) * G::kHighBytes;
        float partial[kColumnTile];
#pragma unroll
        for (int j = 0; j < kColumnTile; ++j) { partial[j] = 0.0F; }
#pragma unroll
        for (int chunk = 0; chunk < G::kGroup / 8; ++chunk) {
            float w[8];
            decode8<Format>(base, top, chunk, w);
            const int k = group * G::kGroup + chunk * 8;
#pragma unroll
            for (int j = 0; j < kColumnTile; ++j) {
                if (j >= columns) { break; }
                const uint4 packed = __ldg(reinterpret_cast<const uint4*>(
                    hidden + static_cast<std::size_t>(first + j) * K + k));
                const auto* x = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 v = __bfloat1622float2(x[pair]);
                    partial[j]     = fmaf(w[pair * 2], v.x, partial[j]);
                    partial[j]     = fmaf(w[pair * 2 + 1], v.y, partial[j]);
                }
            }
        }
        const float scale = __half2float(row_scales[group]);
#pragma unroll
        for (int j = 0; j < kColumnTile; ++j) { acc[j] = fmaf(scale, partial[j], acc[j]); }
    }
#pragma unroll
    for (int j = 0; j < kColumnTile; ++j) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[j] += __shfl_xor_sync(0xffffffffU, acc[j], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < kColumnTile; ++j) {
            if (j < columns) { out[static_cast<std::size_t>(first + j) * C + slot] = acc[j]; }
        }
    }
}

template <QType Format>
__global__ void linear_rows_kernel(const __nv_bfloat16* __restrict__ hidden, int K, int T,
                                   const std::uint8_t* __restrict__ codes,
                                   const std::uint8_t* __restrict__ high,
                                   const __half* __restrict__ scales, int N,
                                   const std::int32_t* __restrict__ ids, int C,
                                   float* __restrict__ out) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot =
        static_cast<int>(blockIdx.x) * kRowsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (slot >= C) { return; }
    const int id = __ldg(ids + slot);
    // Column tiles stride over gridDim.y, which holds at most 65,535.
    for (int first = static_cast<int>(blockIdx.y) * kColumnTile; first < T;
         first += static_cast<int>(gridDim.y) * kColumnTile) {
        const int columns = min(kColumnTile, T - first);
        if (id < 0 || id >= N) {
            if (lane < columns) {
                out[static_cast<std::size_t>(first + lane) * C + slot] = -INFINITY;
            }
            continue;
        }
        linear_rows_tile<Format>(hidden, K, first, columns, codes, high, scales, id, C, lane, out,
                                 slot);
    }
}

// Better (score, id) first: higher score, then lower id.
__device__ __forceinline__ bool ranks_before(float score, int id, float other_score, int other_id) {
    return score > other_score || (score == other_score && id < other_id);
}

__global__ void merge_row_argmax_kernel(const __nv_bfloat16* __restrict__ logits, int R,
                                        const std::int32_t* __restrict__ row_to_global,
                                        const float* __restrict__ row_scores,
                                        const std::int32_t* __restrict__ row_ids, int C,
                                        std::int32_t* __restrict__ tokens) {
    __shared__ float best_scores[kMergeThreads / 32];
    __shared__ int best_slots[kMergeThreads / 32];
    const int t   = static_cast<int>(blockIdx.x);
    float score   = -INFINITY;
    int slot      = -1;
    for (int c = static_cast<int>(threadIdx.x); c < C; c += kMergeThreads) {
        const float s = row_scores[static_cast<std::size_t>(t) * C + c];
        // Lowest slot among equal scores: a thread visits its slots in ascending order.
        if (row_ids[c] >= 0 && !isnan(s) && (slot < 0 || s > score)) {
            score = s;
            slot  = c;
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_score = __shfl_xor_sync(0xffffffffU, score, offset);
        const int other_slot    = __shfl_xor_sync(0xffffffffU, slot, offset);
        if (other_slot >= 0 && (slot < 0 || ranks_before(other_score, other_slot, score, slot))) {
            score = other_score;
            slot  = other_slot;
        }
    }
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if ((threadIdx.x & 31) == 0) {
        best_scores[warp] = score;
        best_slots[warp]  = slot;
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }
    for (int w = 1; w < kMergeThreads / 32; ++w) {
        if (best_slots[w] >= 0 &&
            (slot < 0 || ranks_before(best_scores[w], best_slots[w], score, slot))) {
            score = best_scores[w];
            slot  = best_slots[w];
        }
    }
    const int row = tokens[t];
    if (row < 0 || row >= R) {
        // Outside the contract (the proposal is an argmax row); a context row still answers.
        if (slot >= 0) { tokens[t] = row_ids[slot]; }
        return;
    }
    const float proposal = __bfloat162float(logits[static_cast<std::size_t>(t) * R + row]);
    tokens[t] = slot >= 0 && score > proposal ? row_ids[slot]
                : row_to_global != nullptr     ? row_to_global[row]
                                               : row;
}

__global__ void merge_row_topk_kernel(const float* __restrict__ row_scores,
                                      const std::int32_t* __restrict__ row_ids, int C,
                                      std::int32_t* __restrict__ candidate_ids,
                                      float* __restrict__ candidate_scores) {
    __shared__ float scores[kTopK + kMaximumTopKRows];
    __shared__ int ids[kTopK + kMaximumTopKRows];
    __shared__ float warp_scores[kMergeThreads / 32];
    __shared__ int warp_ids[kMergeThreads / 32];
    __shared__ int warp_entries[kMergeThreads / 32];
    const int u     = static_cast<int>(blockIdx.x);
    const int count = kTopK + C;
    for (int i = static_cast<int>(threadIdx.x); i < count; i += kMergeThreads) {
        if (i < kTopK) {
            scores[i] = candidate_scores[static_cast<std::size_t>(u) * kTopK + i];
            ids[i]    = candidate_ids[static_cast<std::size_t>(u) * kTopK + i];
        } else {
            const int c   = i - kTopK;
            const float s = row_scores[static_cast<std::size_t>(u) * C + c];
            const int id  = row_ids[c];
            // An empty slot or a NaN score never takes part.
            ids[i]    = id >= 0 && !isnan(s) ? id : -1;
            scores[i] = s;
        }
    }
    __syncthreads();
    for (int rank = 0; rank < kTopK; ++rank) {
        float score = -INFINITY;
        int id      = -1;
        int entry   = -1;
        for (int i = static_cast<int>(threadIdx.x); i < count; i += kMergeThreads) {
            if (ids[i] >= 0 && (entry < 0 || ranks_before(scores[i], ids[i], score, id))) {
                score = scores[i];
                id    = ids[i];
                entry = i;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_xor_sync(0xffffffffU, score, offset);
            const int other_id      = __shfl_xor_sync(0xffffffffU, id, offset);
            const int other_entry   = __shfl_xor_sync(0xffffffffU, entry, offset);
            if (other_entry >= 0 && (entry < 0 || ranks_before(other_score, other_id, score, id))) {
                score = other_score;
                id    = other_id;
                entry = other_entry;
            }
        }
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if ((threadIdx.x & 31) == 0) {
            warp_scores[warp]  = score;
            warp_ids[warp]     = id;
            warp_entries[warp] = entry;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < kMergeThreads / 32; ++w) {
                if (warp_entries[w] >= 0 &&
                    (entry < 0 || ranks_before(warp_scores[w], warp_ids[w], score, id))) {
                    score = warp_scores[w];
                    id    = warp_ids[w];
                    entry = warp_entries[w];
                }
            }
            // Input candidates are never NaN, so sixteen members always exist.
            if (entry >= 0) {
                candidate_scores[static_cast<std::size_t>(u) * kTopK + rank] = score;
                candidate_ids[static_cast<std::size_t>(u) * kTopK + rank]    = id;
                ids[entry]                                                   = -1;
            }
        }
        __syncthreads();
    }
}

template <QType Format>
void launch_linear_rows(const Tensor& hidden, const Weight& head, const Tensor& ids, Tensor& out,
                        cudaStream_t stream) {
    const int C = ids.ne[0];
    const int T = hidden.ne[1];
    const int tiles = (T + kColumnTile - 1) / kColumnTile;
    const dim3 grid(static_cast<unsigned>((C + kRowsPerBlock - 1) / kRowsPerBlock),
                    static_cast<unsigned>(tiles < 65535 ? tiles : 65535));
    linear_rows_kernel<Format><<<grid, kRowsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data), hidden.ne[0], T,
        static_cast<const std::uint8_t*>(head.qdata), static_cast<const std::uint8_t*>(head.qhigh),
        static_cast<const __half*>(head.scales), head.n,
        static_cast<const std::int32_t*>(ids.data), C, static_cast<float*>(out.data));
}

std::size_t code_plane_bytes(const Weight& head) {
    return static_cast<std::size_t>(head.n) * head.k *
           (head.qtype == QType::Q8_G32_FP16 ? 2U : 1U) / 2U;
}

} // namespace

void linear_rows(const Tensor& hidden, const Weight& head, const Tensor& ids, Tensor& out,
                 cudaStream_t stream) {
    constexpr const char* op = "linear_rows";
    const bool q8            = head.qtype == QType::Q8_G32_FP16;
    const bool q6            = head.qtype == QType::Q6_G64_FP16;
    const bool q4            = head.qtype == QType::Q4_G64_FP16;
    require(q8 || q6 || q4, op, "the head must be Q8_G32_FP16, Q6_G64_FP16 or Q4_G64_FP16");
    const int group = q8 ? 32 : 64;
    require(head.layout == QuantLayout::RowSplit && head.scale_dtype == DType::FP16 &&
                head.group == group && head.group_size == static_cast<std::uint32_t>(group) &&
                head.ndim == 2 && head.n > 0 && head.k > 0 && head.k % 128 == 0 &&
                head.shape[0] == head.n && head.shape[1] == head.k &&
                head.padded_shape[1] == head.k,
            op, "the head must be a row-split [N,K] matrix with K a multiple of 128");
    require(aligned(head.qdata, 16) && aligned(head.scales, 2), op,
            "the head's code and scale planes must be aligned");
    require(q6 ? aligned(head.qhigh, 16) : head.qhigh == nullptr, op,
            "the head's high-bit plane does not match its format");
    require(hidden.ne[1] > 0 && matrix(hidden, DType::BF16, head.k, hidden.ne[1]) &&
                aligned(hidden.data, 16),
            op, "hidden must be contiguous 16-byte-aligned BF16 [K,T]");
    require(ids.ne[0] > 0 && vector(ids, DType::I32, ids.ne[0]), op,
            "ids must be contiguous I32 [C]");
    require(matrix(out, DType::FP32, ids.ne[0], hidden.ne[1]), op,
            "out must be contiguous FP32 [C,T]");
    const std::size_t scale_bytes =
        static_cast<std::size_t>(head.n) * (head.k / group) * sizeof(std::uint16_t);
    const std::size_t high_bytes = q6 ? static_cast<std::size_t>(head.n) * head.k / 4 : 0;
    require(!overlaps(out, hidden) && !overlaps(out, ids) &&
                !overlaps(out.data, out.bytes(), head.qdata, code_plane_bytes(head)) &&
                !overlaps(out.data, out.bytes(), head.scales, scale_bytes) &&
                !overlaps(out.data, out.bytes(), head.qhigh, high_bytes),
            op, "out must not overlap an input");
    if (q8) {
        launch_linear_rows<QType::Q8_G32_FP16>(hidden, head, ids, out, stream);
    } else if (q6) {
        launch_linear_rows<QType::Q6_G64_FP16>(hidden, head, ids, out, stream);
    } else {
        launch_linear_rows<QType::Q4_G64_FP16>(hidden, head, ids, out, stream);
    }
    check_launch(op);
}

void merge_row_argmax(const Tensor& logits, const Tensor* row_to_global, const Tensor& row_scores,
                      const Tensor& row_ids, Tensor& tokens, cudaStream_t stream) {
    constexpr const char* op = "merge_row_argmax";
    const std::int32_t R     = logits.ne[0];
    const std::int32_t T     = logits.ne[1];
    const std::int32_t C     = row_ids.ne[0];
    require(R > 0 && T > 0 && matrix(logits, DType::BF16, R, T), op,
            "logits must be contiguous BF16 [R,T]");
    require(row_to_global == nullptr || vector(*row_to_global, DType::I32, R), op,
            "row_to_global must be contiguous I32 [R]");
    require(C > 0 && vector(row_ids, DType::I32, C), op, "row_ids must be contiguous I32 [C]");
    require(matrix(row_scores, DType::FP32, C, T), op, "row_scores must be contiguous FP32 [C,T]");
    require(vector(tokens, DType::I32, T), op, "tokens must be contiguous I32 [T]");
    require(!overlaps(tokens, logits) && !overlaps(tokens, row_scores) &&
                !overlaps(tokens, row_ids) &&
                (row_to_global == nullptr || !overlaps(tokens, *row_to_global)),
            op, "tokens must not overlap an input");
    merge_row_argmax_kernel<<<static_cast<unsigned>(T), kMergeThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), R,
        row_to_global != nullptr ? static_cast<const std::int32_t*>(row_to_global->data) : nullptr,
        static_cast<const float*>(row_scores.data), static_cast<const std::int32_t*>(row_ids.data),
        C, static_cast<std::int32_t*>(tokens.data));
    check_launch(op);
}

void merge_row_topk(const Tensor& row_scores, const Tensor& row_ids, Tensor& candidate_ids,
                    Tensor& candidate_scores, cudaStream_t stream) {
    constexpr const char* op = "merge_row_topk";
    const std::int32_t C     = row_ids.ne[0];
    const std::int32_t U     = row_scores.ne[1];
    require(C > 0 && C <= kMaximumTopKRows && vector(row_ids, DType::I32, C), op,
            "row_ids must be contiguous I32 [C] with C in [1,2048]");
    require(U > 0 && matrix(row_scores, DType::FP32, C, U), op,
            "row_scores must be contiguous FP32 [C,U]");
    require(matrix(candidate_ids, DType::I32, kTopK, U) &&
                matrix(candidate_scores, DType::FP32, kTopK, U),
            op, "candidates must be contiguous I32 and FP32 [16,U]");
    require(!overlaps(candidate_ids, candidate_scores) && !overlaps(candidate_ids, row_scores) &&
                !overlaps(candidate_ids, row_ids) && !overlaps(candidate_scores, row_scores) &&
                !overlaps(candidate_scores, row_ids),
            op, "the candidates must not overlap the rows or each other");
    merge_row_topk_kernel<<<static_cast<unsigned>(U), kMergeThreads, 0, stream>>>(
        static_cast<const float*>(row_scores.data), static_cast<const std::int32_t*>(row_ids.data),
        C, static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(candidate_scores.data));
    check_launch(op);
}

} // namespace ninfer::ops
