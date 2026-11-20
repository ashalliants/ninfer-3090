// Adapted from Infernix a3edb450 src/ops/projection_fp32/projection_fp32.cu (Apache-2.0).
// Modified for NInfer-3090: the BF16 segment form with its narrow and wide mappings only; the
// q8_g32_fp16/q4_g64_fp16 forms and the vocabulary-head tall and tensor-core mappings are not
// ported. Namespaces and includes are NInfer's.

#include "ops/common/smem_limit.h"
#include "ninfer/ops/projection_fp32.h"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads       = 256;
constexpr int kRowsPerWarp   = 4;
constexpr int kRowsPerCta    = kThreads / 32 * kRowsPerWarp;
constexpr int kColumns       = 8; // columns staged per CTA
constexpr int kMaxSegments   = 4;
constexpr int kMaxK          = 3072; // the narrow kernels' (register arrays and staging are sized by it)
// The longest K projection_fp32 serves: beyond kMaxK only the wide mapping (it walks K in slices and has no
// K-sized state), which gives the same bits as the narrow ones.
constexpr int kMaxLongK      = 8192;
// The narrow kernels' second size: K up to DeepSeek-V4's hidden size (5120, 20 chunks a lane), which
// needs more registers and shared memory than the K <= 3072 instances, so it is a separate instantiation.
constexpr int kMaxMidK       = 5120;

struct Segments {
    const bf16* data[kMaxSegments];
    int begin[kMaxSegments + 1]; // first output row of each segment, then the total
    int count;
    int rows; // the total again, at a fixed offset (see total_rows)
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("projection_fp32: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// The total row count. Segment fields are read at compile-time indices only: a runtime index into
// the kernel parameter would copy it to local memory and put a local load before the weight loads.
// (Selecting begin[count] by compile-time compares was folded back into that indexed load, so the
// host stores the total in its own field.)
__device__ __forceinline__ int total_rows(const Segments& segments) { return segments.rows; }

// The address of chunk `chunk` (8 elements) of output row `row`.
__device__ __forceinline__ const uint4* row_chunk(const Segments& segments, int row, int k,
                                                  int chunk) {
    const bf16* data = segments.data[0];
    int begin        = 0;
#pragma unroll
    for (int s = 1; s < kMaxSegments; ++s) {
        if (s < segments.count && row >= segments.begin[s]) {
            data  = segments.data[s];
            begin = segments.begin[s];
        }
    }
    return reinterpret_cast<const uint4*>(data + static_cast<std::size_t>(row - begin) * k) + chunk;
}

// Issues lane `lane`'s chunk loads of `row` (chunks lane, lane + 32, ...) into `loaded`.
template <int LaneChunks>
__device__ __forceinline__ void load_row(const Segments& segments, int row, int k, int chunks,
                                         int lane, uint4 (&loaded)[LaneChunks]) {
    const uint4* w = row_chunk(segments, row, k, 0);
#pragma unroll
    for (int i = 0; i < LaneChunks; ++i) {
        const int chunk = lane + 32 * i;
        if (chunk < chunks) { loaded[i] = w[chunk]; }
    }
}

// Warp w of a CTA owns rows first + w, first + w + Warps, ...; lane l accumulates the 8-element
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an output's bits
// depend only on K. A lane issues all of a row's chunk loads before its FMAs, and the first row's
// loads before x is staged, so the weights' DRAM trip overlaps the staging instead of following
// it. Decode-width calls give each warp one row (Threads = 128, RowsPerWarp = 1), so a router's
// latency is one DRAM round trip; wider calls keep four rows per warp, which stages x in fewer
// CTAs. Both mappings produce the same bits. Columns is the smallest of 1, 2, 4 and kColumns that
// holds the call's columns (a decode step's router is one column, a speculative verification's two
// to four): the same arithmetic per column in less code, which matters because a router call is
// short enough that fetching cold instructions dominated it on Volta (the 8-column form is 37 KiB of
// code: ~23-27 us a call in the model against ~10 us warm).
// Where a result goes: FP32 [rows, T], or rounded to BF16 (as cast_fp32_to_bf16 rounds) with rows
// [0, first_rows) in `first` and the rest in `second`.
struct Fp32Out {
    float* out;
    __device__ __forceinline__ void store(int row, int column, int rows, float v) const {
        out[static_cast<std::size_t>(column) * rows + row] = v;
    }
};
struct Bf16SplitOut {
    bf16* first;
    bf16* second;
    int first_rows;
    __device__ __forceinline__ void store(int row, int column, int rows, float v) const {
        if (row < first_rows) {
            first[static_cast<std::size_t>(column) * first_rows + row] = __float2bfloat16_rn(v);
        } else {
            const int second_rows = rows - first_rows;
            second[static_cast<std::size_t>(column) * second_rows + row - first_rows] = __float2bfloat16_rn(v);
        }
    }
};

template <int Threads, int RowsPerWarp, int Columns = kColumns, class Out = Fp32Out, int MaxK = kMaxK>
__global__ void __launch_bounds__(Threads)
    projection_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments, Out out) {
    constexpr int Warps          = Threads / 32;
    constexpr int kMaxLaneChunks = MaxK / 8 / 32; // 8-element chunks one lane reads per row
    // 16-byte x chunks of the CTA's columns, as one unrolled batch of at most this many per thread.
    constexpr int kStagePerThread = (Columns * MaxK / 8 + Threads - 1) / Threads;
    extern __shared__ uint4 staged[];
    const int first_column = blockIdx.y * kColumns;
    const int count        = min(Columns, columns - first_column);
    const int chunks       = k / 8;
    const int rows         = total_rows(segments);
    const int lane         = threadIdx.x % 32;
    const int first_row    = blockIdx.x * (Warps * RowsPerWarp) + threadIdx.x / 32;
    uint4 loaded[kMaxLaneChunks];
    if (first_row < rows) { load_row<kMaxLaneChunks>(segments, first_row, k, chunks, lane, loaded); }
    // The CTA's columns of x are contiguous, so chunk i of the staging is chunk i of x from there.
    const auto* xs = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(first_column) * k);
#pragma unroll
    for (int u = 0; u < kStagePerThread; ++u) {
        const int i = static_cast<int>(threadIdx.x) + u * Threads;
        if (i < count * chunks) {
            cp_async_mem<16>(staged + i, xs + i);
        }
    }
    cp_commit_mem();
    cp_wait_mem<0>();
    __syncthreads();
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = first_row + r * Warps;
        if (row >= rows) { return; }
        if (r > 0) { load_row<kMaxLaneChunks>(segments, row, k, chunks, lane, loaded); }
        float sums[Columns] = {};
#pragma unroll
        for (int i = 0; i < kMaxLaneChunks; ++i) {
            const int chunk = lane + 32 * i;
            if (chunk >= chunks) { break; }
            const uint4 packed = loaded[i];
            const bf16* wv     = reinterpret_cast<const bf16*>(&packed);
            float wf[8];
            for (int e = 0; e < 8; ++e) { wf[e] = __bfloat162float(wv[e]); }
            for (int column = 0; column < Columns; ++column) {
                if (column < count) {
                    const uint4 xp = staged[column * chunks + chunk];
                    const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                    for (int e = 0; e < 8; ++e) {
                        sums[column] = fmaf(wf[e], __bfloat162float(xv[e]), sums[column]);
                    }
                }
            }
        }
        for (int column = 0; column < Columns; ++column) {
            float v = sums[column];
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
            }
            if (lane == 0 && column < count) {
                out.store(row, first_column + column, rows, v);
            }
        }
    }
}

// Wide calls (prefill chunks). A CTA owns a kWideRows x kWideColumns output tile and walks K in
// slices of 256 elements, which is exactly one 8-element chunk per lane: in slice s lane l handles
// chunk 32 s + l, the narrow kernels' l-th lane partial in the same order. Each slice's weight
// rows and x columns are staged with cp.async (double-buffered); a warp owns a 4-row x 8-column
// sub-tile. After the last slice each warp reduces its partials by the same butterfly, so the bits
// equal the narrow kernels' (an output's summation still depends only on K).
constexpr int kWideRows        = 32; // output rows per CTA
constexpr int kWideColumns     = 16; // output columns per CTA
constexpr int kWideTileRows    = 4;  // per warp
constexpr int kWideTileColumns = 8;
constexpr int kWideRowWarps    = kWideRows / kWideTileRows;       // 8
constexpr int kWideColumnWarps = kWideColumns / kWideTileColumns; // 2
constexpr int kWideThreads     = kWideRowWarps * kWideColumnWarps * 32;
constexpr int kSliceChunks     = 32; // chunks per K slice: one per lane
constexpr int kStageWords      = (kWideRows + kWideColumns) * kSliceChunks; // 16-byte words/stage
constexpr int kWideSharedBytes = 2 * kStageWords * 16;
constexpr int kWideMinColumns  = 64; // narrower calls keep the narrow mapping (same bits)
static_assert(kWideThreads == 512 && kStageWords % kWideThreads == 0);

__device__ __forceinline__ void widen8(const uint4& packed, float (&out)[8]) {
    const auto* v = reinterpret_cast<const bf16*>(&packed);
#pragma unroll
    for (int i = 0; i < 8; ++i) { out[i] = __bfloat162float(v[i]); }
}

// 16 bytes from `src` to shared `dst`, or zeros when `valid` is false (src is then not read).
__device__ __forceinline__ void stage16(uint4* dst, const void* src, bool valid) {
    cp_async_zfill_mem<16>(dst, src, valid ? 16 : 0);
}

__global__ void __launch_bounds__(kWideThreads, 1)
    projection_wide_kernel(const bf16* __restrict__ x, int k, int columns, Segments segments,
                           float* __restrict__ out) {
    extern __shared__ uint4 stage[]; // [2][rows then columns][kSliceChunks]
    const int chunks       = k / 8;
    const int slices       = (chunks + kSliceChunks - 1) / kSliceChunks;
    const int rows         = total_rows(segments);
    const int first_row    = static_cast<int>(blockIdx.x) * kWideRows;
    const int first_column = static_cast<int>(blockIdx.y) * kWideColumns;
    const auto issue       = [&](int slice, int buffer) {
        uint4* base = stage + buffer * kStageWords;
#pragma unroll
        for (int u = 0; u < kStageWords / kWideThreads; ++u) {
            const int i     = static_cast<int>(threadIdx.x) + u * kWideThreads;
            const int line  = i / kSliceChunks;
            const int chunk = slice * kSliceChunks + i % kSliceChunks;
            if (line < kWideRows) {
                const int row    = first_row + line;
                const bool valid = row < rows && chunk < chunks;
                stage16(base + i,
                        valid ? static_cast<const void*>(row_chunk(segments, row, k, chunk)) : x,
                        valid);
            } else {
                const int column = first_column + line - kWideRows;
                const bool valid = column < columns && chunk < chunks;
                stage16(base + i,
                        valid ? static_cast<const void*>(x + static_cast<std::size_t>(column) * k +
                                                         8 * chunk)
                              : x,
                        valid);
            }
        }
        cp_commit_mem();
    };
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r0   = (warp % kWideRowWarps) * kWideTileRows;
    const int c0   = (warp / kWideRowWarps) * kWideTileColumns;
    float sums[kWideTileRows][kWideTileColumns] = {};
    issue(0, 0);
    for (int s = 0; s < slices; ++s) {
        if (s + 1 < slices) {
            issue(s + 1, (s + 1) & 1);
            cp_wait_mem<1>();
        } else {
            cp_wait_mem<0>();
        }
        __syncthreads();
        if (s * kSliceChunks + lane < chunks) { // the narrow kernels' loop ends at the last chunk
            const uint4* base = stage + (s & 1) * kStageWords;
            float wf[kWideTileRows][8];
#pragma unroll
            for (int r = 0; r < kWideTileRows; ++r) {
                widen8(base[(r0 + r) * kSliceChunks + lane], wf[r]);
            }
#pragma unroll
            for (int c = 0; c < kWideTileColumns; ++c) {
                float xf[8];
                widen8(base[(kWideRows + c0 + c) * kSliceChunks + lane], xf);
#pragma unroll
                for (int r = 0; r < kWideTileRows; ++r) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) { sums[r][c] = fmaf(wf[r][i], xf[i], sums[r][c]); }
                }
            }
        }
        __syncthreads(); // the next iteration's issue overwrites this buffer
    }
#pragma unroll
    for (int r = 0; r < kWideTileRows; ++r) {
        const int row = first_row + r0 + r;
#pragma unroll
        for (int c = 0; c < kWideTileColumns; ++c) {
            const int column = first_column + c0 + c;
            float v          = sums[r][c];
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
            }
            if (lane == 0 && row < rows && column < columns) {
                out[static_cast<std::size_t>(column) * rows + row] = v;
            }
        }
    }
}

// x's and the weights' checks, and the weights bound as segments (rows: their total).
Segments bind_segments(const Tensor& x, std::span<const Tensor* const> weights, int& rows, int max_k) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 &&
                x.ne[3] == 1,
            "x must be contiguous BF16 [K, T]");
    const int k = x.ne[0], columns = x.ne[1];
    require(k > 0 && k % 8 == 0 && k <= max_k && columns > 0,
            max_k == kMaxK ? "K must be a multiple of 8 up to 3072" : "K must be a multiple of 8 up to 8192");
    require(!weights.empty() && weights.size() <= kMaxSegments, "between one and four weights");
    require(aligned16(x.data), "x must be 16-byte aligned");
    Segments segments{};
    segments.count = static_cast<int>(weights.size());
    rows           = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Tensor& w = *weights[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == k &&
                    w.ne[1] > 0 && w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each weight must be contiguous, 16-byte aligned BF16 [K, N]");
        segments.data[i]  = static_cast<const bf16*>(w.data);
        segments.begin[i] = rows;
        rows += w.ne[1];
    }
    segments.begin[segments.count] = rows;
    segments.rows                  = rows;
    return segments;
}

} // namespace

bool projection_bf16_split_supported(std::int32_t columns) noexcept { return columns >= 1 && columns <= kColumns; }

void projection_bf16_split(const Tensor& x, std::span<const Tensor* const> weights, Tensor& first, Tensor& second,
                           cudaStream_t stream) {
    int rows                = 0;
    const Segments segments = bind_segments(x, weights, rows, kMaxK);
    const int k = x.ne[0], columns = x.ne[1];
    require(projection_bf16_split_supported(columns), "the BF16 split form serves 1 to 8 columns");
    for (const Tensor* t : {&first, &second}) {
        require(t->data != nullptr && t->dtype == DType::BF16 && t->is_contiguous() && t->ne[1] == columns &&
                    t->ne[2] == 1 && t->ne[3] == 1,
                "the BF16 outputs must be contiguous BF16 [N, T]");
    }
    require(first.ne[0] > 0 && second.ne[0] > 0 && first.ne[0] + second.ne[0] == rows,
            "the BF16 outputs' rows must split the weights' rows");
    const std::size_t staged = static_cast<std::size_t>(columns) * k * sizeof(bf16);
    constexpr int threads = 128, rows_per_warp = 1, rows_per_cta = threads / 32 * rows_per_warp;
    auto* kernel = columns == 1   ? projection_kernel<threads, rows_per_warp, 1, Bf16SplitOut>
                   : columns <= 2 ? projection_kernel<threads, rows_per_warp, 2, Bf16SplitOut>
                   : columns <= 4 ? projection_kernel<threads, rows_per_warp, 4, Bf16SplitOut>
                                  : projection_kernel<threads, rows_per_warp, kColumns, Bf16SplitOut>;
    kernel<<<dim3((rows + rows_per_cta - 1) / rows_per_cta, 1), threads, staged, stream>>>(
        static_cast<const bf16*>(x.data), k, columns, segments,
        Bf16SplitOut{static_cast<bf16*>(first.data), static_cast<bf16*>(second.data), static_cast<int>(first.ne[0])});
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_bf16_split: ") + cudaGetErrorString(error));
    }
}

void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream) {
    require(out.data != nullptr && out.dtype == DType::FP32 && out.is_contiguous(),
            "out must be contiguous FP32");
    int rows                = 0;
    const Segments segments = bind_segments(x, weights, rows, kMaxLongK);
    const int k = x.ne[0], columns = x.ne[1];
    require(out.ne[0] == rows && out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1,
            "out must be FP32 [sum N, T]");
    const std::size_t staged =
        static_cast<std::size_t>(std::min(kColumns, columns)) * k * sizeof(bf16);
    const int column_groups = (columns + kColumns - 1) / kColumns;
    if (k > kMaxK && k <= kMaxMidK && columns <= kColumns) {
        // DeepSeek-V4's K = 5120 projections at decode widths: the narrow mapping at its larger size.
        constexpr int threads = 128, rows_per_warp = 1, rows_per_cta = threads / 32 * rows_per_warp;
        auto* kernel = columns == 1   ? projection_kernel<threads, rows_per_warp, 1, Fp32Out, kMaxMidK>
                       : columns <= 2 ? projection_kernel<threads, rows_per_warp, 2, Fp32Out, kMaxMidK>
                       : columns <= 4 ? projection_kernel<threads, rows_per_warp, 4, Fp32Out, kMaxMidK>
                                      : projection_kernel<threads, rows_per_warp, kColumns, Fp32Out, kMaxMidK>;
        if (staged > 48 * 1024) {
            static const bool configured = [] {
                return cudaFuncSetAttribute(projection_kernel<threads, rows_per_warp, kColumns, Fp32Out, kMaxMidK>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            kColumns * kMaxMidK * static_cast<int>(sizeof(bf16))) == cudaSuccess;
            }();
            require(configured, "the narrow kernel's shared memory could not be configured");
        }
        kernel<<<dim3((rows + rows_per_cta - 1) / rows_per_cta, column_groups), threads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, Fp32Out{static_cast<float*>(out.data)});
    } else if (columns >= kWideMinColumns || k > kMaxK) {
        static_assert(kWideSharedBytes <= kMaxDynamicSharedBytes);
        static const bool configured = [] {
            return cudaFuncSetAttribute(projection_wide_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kWideSharedBytes) == cudaSuccess;
        }();
        require(configured, "the wide kernel's shared memory could not be configured");
        projection_wide_kernel<<<dim3((rows + kWideRows - 1) / kWideRows,
                                      (columns + kWideColumns - 1) / kWideColumns),
                                 kWideThreads, kWideSharedBytes, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, static_cast<float*>(out.data));
    } else if (columns <= kColumns) {
        constexpr int threads = 128, rows_per_warp = 1, rows_per_cta = threads / 32 * rows_per_warp;
        auto* kernel = columns == 1   ? projection_kernel<threads, rows_per_warp, 1>
                       : columns <= 2 ? projection_kernel<threads, rows_per_warp, 2>
                       : columns <= 4 ? projection_kernel<threads, rows_per_warp, 4>
                                      : projection_kernel<threads, rows_per_warp>;
        kernel<<<dim3((rows + rows_per_cta - 1) / rows_per_cta, column_groups), threads, staged, stream>>>(
            static_cast<const bf16*>(x.data), k, columns, segments, Fp32Out{static_cast<float*>(out.data)});
    } else {
        projection_kernel<kThreads, kRowsPerWarp>
            <<<dim3((rows + kRowsPerCta - 1) / kRowsPerCta, column_groups), kThreads, staged,
                stream>>>(static_cast<const bf16*>(x.data), k, columns, segments,
                          Fp32Out{static_cast<float*>(out.data)});
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("projection_fp32: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
