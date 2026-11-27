// ninfer::ops - embedding launcher: variant grid/block/stream setup.
#include "core/weight.h"
#include "ops/launcher/embed_gather.h"

#include "ops/common/math.h"
#include "ops/kernel/embed_gather.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

// ggml's lookup tables in device memory, static to this translation unit (third_party/ggml); the
// IQ3_S rows read iq3s_grid.
#define NINFER_GGML_TABLE_BEGIN(type, name, size) static __device__ const type name[size] = {
#define NINFER_GGML_TABLE_END() };
#include "ggml-common-tables.h"

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock          = 128;
constexpr int kQ6GroupedBlock = kEmbedGatherQ6Group * kEmbedGatherQ6GroupsPerBlock;
constexpr int kQ4GroupedBlock = kEmbedGatherQ4Group * kEmbedGatherQ4GroupsPerBlock;
constexpr int kQ8GroupedBlock = 32;
constexpr int kQ8RowBlock     = 256;

template <int BlocksPerToken, int Threads>
void launch_fp8(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int grid = ids.ne[0] * BlocksPerToken;
    embed_gather_fp8_kernel<BlocksPerToken, Threads><<<grid, Threads, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales), static_cast<__nv_bfloat16*>(out.data));
}

template <int Blocks, int Threads>
void launch_q8_packed(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool PairStore>() {
        embed_gather_q8_packed_5120_kernel<Blocks, Threads, PairStore>
            <<<ids.ne[0] * Blocks, Threads, 0, stream>>>(
                static_cast<const std::int32_t*>(ids.data),
                static_cast<const std::uint8_t*>(table.qdata),
                static_cast<const std::uint8_t*>(table.scales),
                static_cast<__nv_bfloat16*>(out.data));
    };
    if (reinterpret_cast<std::uintptr_t>(out.data) % 4 == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

int grid_for(std::int64_t n) {
    return static_cast<int>(
        std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
}

int grid_for_q6_grouped(std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    return static_cast<int>(std::max<std::int64_t>(1, static_cast<std::int64_t>(T) *
                                                          static_cast<std::int64_t>(group_blocks)));
}

} // namespace

void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream) {
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (route == Q8EmbedRoute::Auto) { route = T <= 6 ? Q8EmbedRoute::Grouped : Q8EmbedRoute::Row; }
    if (route == Q8EmbedRoute::Grouped) {
        const int grid = T * kEmbedGatherQ8Groups;
        embed_gather_q8_grouped_2048_kernel<<<grid, kQ8GroupedBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    } else {
        embed_gather_q8_row_2048_kernel<<<T, kQ8RowBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_dense_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const __nv_bfloat16*>(table.data),
        static_cast<__nv_bfloat16*>(out.data), d, T);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// The Q4_G64 gather of every row from `table`, or (Split) rows below hot_rows from `hot`.
template <bool Split>
void launch_q4(const Tensor& ids, const Weight& table, const Weight* hot, std::int32_t hot_rows,
               Tensor& out, cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    const auto* hot_codes  = hot != nullptr ? static_cast<const std::uint8_t*>(hot->qdata) : nullptr;
    const auto* hot_scales = hot != nullptr ? static_cast<const std::uint8_t*>(hot->scales) : nullptr;
    if (d == table.padded_shape[1] && d % kEmbedGatherQ4Group == 0) {
        const std::int32_t group_blocks = div_up(d / kEmbedGatherQ4Group, kEmbedGatherQ4GroupsPerBlock);
        embed_gather_q4_grouped_kernel<Split><<<T * group_blocks, kQ4GroupedBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, scales, hot_codes, hot_scales, hot_rows,
            static_cast<__nv_bfloat16*>(out.data), d, T);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    embed_gather_q4_kernel<Split><<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, scales, hot_codes, hot_scales, hot_rows,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void embed_gather_q4_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    launch_q4<false>(ids, table, nullptr, 0, out, stream);
}

void embed_gather_q4_split_launch(const Tensor& ids, const Weight& hot, const Weight& cold,
                                  std::int32_t hot_rows, Tensor& out, cudaStream_t stream) {
    launch_q4<true>(ids, cold, &hot, hot_rows, out, stream);
}

void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* high     = static_cast<const std::uint8_t*>(table.qhigh);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == table.padded_shape[1] && d % kEmbedGatherQ6Group == 0) {
        embed_gather_q6_grouped_kernel<<<grid_for_q6_grouped(d, T), kQ6GroupedBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(ids.data), codes, high, scales,
            static_cast<__nv_bfloat16*>(out.data), d, T);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    embed_gather_q6_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, high, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == kEmbedGatherQ8D && table.padded_shape[1] == kEmbedGatherQ8D) {
        embed_gather_q8_2048_launch(ids, table, out, Q8EmbedRoute::Auto, stream);
        return;
    }

    // Four signed codes share their exact group scale. Keep byte-addressed tables on the
    // scalar reader and preserve two-byte output alignment through the packed kernel's stores.
    if (d == 5120 && table.padded_shape[1] == 5120 &&
        reinterpret_cast<std::uintptr_t>(table.qdata) % 4 == 0) {
        if (T <= 128)
            launch_q8_packed<10, 128>(ids, table, out, stream);
        else
            launch_q8_packed<5, 128>(ids, table, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_q8_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), codes, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// ggml's kvalues_iq4nl (ggml-common.h, llama.cpp b11316).
__constant__ std::int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,  38,  53,  69,  89,  113};

// IQ4_XS: {fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]} per 256 values. One thread per output
// value, in ggml's operation order: dl = d * (ls - 32), then y = dl * kvalue, both FP32.
// Split: rows below hot_rows come from `hot`, the rest from `table` (embedding_split); otherwise
// every row comes from `table`.
constexpr int kIq4XsBlockBytes = 136;
template <bool Split>
__global__ void embed_gather_iq4_xs_kernel(const std::int32_t* __restrict__ ids,
                                           const std::uint8_t* __restrict__ table,
                                           const std::uint8_t* __restrict__ hot, std::int32_t hot_rows,
                                           std::int64_t row_bytes, int d,
                                           __nv_bfloat16* __restrict__ out) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (v >= d) { return; }
    const std::int32_t id      = ids[t];
    const std::uint8_t* plane = table;
    if constexpr (Split) {
        if (id < hot_rows) { plane = hot; }
    }
    const std::uint8_t* block = plane + static_cast<std::int64_t>(id) * row_bytes +
                                static_cast<std::int64_t>(v / 256) * kIq4XsBlockBytes;
    const int ib = (v % 256) / 32, j = v % 32;
    const auto d_bits = static_cast<unsigned short>(block[0] | (static_cast<unsigned>(block[1]) << 8));
    const unsigned scales_h = block[2] | (static_cast<unsigned>(block[3]) << 8);
    const int ls = ((block[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = __half2float(__ushort_as_half(d_bits)) * static_cast<float>(ls - 32);
    const std::uint8_t q = block[8 + 16 * ib + (j & 15)];
    const int code       = j < 16 ? (q & 0xF) : (q >> 4);
    out[static_cast<std::size_t>(t) * d + v] = __float2bfloat16_rn(dl * static_cast<float>(kIq4nlValues[code]));
}

// IQ3_S: {fp16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]} per 256 values. One thread per
// output value, in ggml's operation order: db = d * (1 + 2 nibble), then db * grid byte, both FP32,
// negated (a sign-bit flip, as the reference build applies its sign factor) where the sign bit is
// set. Value 32 g + 8 l + j of a block reads grid word qs[8 g + 2 l + j / 4] with bit 8 from qh[g].
constexpr int kIq3sBlockBytes = 110;
__global__ void embed_gather_iq3_s_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ table,
                                          std::int64_t row_bytes, int d, __nv_bfloat16* __restrict__ out) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (v >= d) { return; }
    const std::uint8_t* block = table + static_cast<std::int64_t>(ids[t]) * row_bytes +
                                static_cast<std::int64_t>(v / 256) * kIq3sBlockBytes;
    const int g = (v % 256) / 32, l = (v % 32) / 8, j = v % 8;
    const auto d_bits = static_cast<unsigned short>(block[0] | (static_cast<unsigned>(block[1]) << 8));
    const int nibble  = (block[106 + g / 2] >> (4 * (g % 2))) & 0xF;
    const float db    = __fmul_rn(__half2float(__ushort_as_half(d_bits)), static_cast<float>(1 + 2 * nibble));
    const int qh      = block[66 + g];
    const int index   = j < 4 ? block[2 + 8 * g + 2 * l] | ((qh << (8 - 2 * l)) & 256)
                              : block[3 + 8 * g + 2 * l] | ((qh << (7 - 2 * l)) & 256);
    const std::uint32_t grid = ggml_tables::iq3s_grid[index];
    float value              = __fmul_rn(db, static_cast<float>((grid >> (8 * (j % 4))) & 0xFFU));
    if ((block[74 + 4 * g + l] >> j) & 1) { value = __int_as_float(__float_as_int(value) ^ static_cast<int>(0x80000000U)); }
    out[static_cast<std::size_t>(t) * d + v] = __float2bfloat16_rn(value);
}

// Q6_K: {u8 ql[128]; u8 qh[64]; int8 scales[16]; fp16 d} per 256 values. One thread per output
// value, in ggml's operation order (dequantize_row_q6_K): (d * scale) * (q - 32), FP32 products.
// Value i of a block (h = i / 128, qd = (i % 128) / 32, l = i % 32) takes the nibble qd >> 1 of
// ql[64 h + 32 (qd & 1) + l], two high bits (2 qd) of qh[32 h + l], and scales[i / 16].
constexpr int kQ6kBlockBytes = 210;
// Split: rows below hot_rows come from `hot`, the rest from `table` (embedding_split); otherwise every
// row comes from `table`.
template <bool Split>
__global__ void embed_gather_q6_k_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ table,
                                         const std::uint8_t* __restrict__ hot, std::int32_t hot_rows,
                                         std::int64_t row_bytes, int d, __nv_bfloat16* __restrict__ out) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (v >= d) { return; }
    const std::int32_t id     = ids[t];
    const std::uint8_t* plane = table;
    if constexpr (Split) {
        if (id < hot_rows) { plane = hot; }
    }
    const std::uint8_t* block = plane + static_cast<std::int64_t>(id) * row_bytes +
                                static_cast<std::int64_t>(v / 256) * kQ6kBlockBytes;
    const int i = v % 256, h = i / 128, qd = (i % 128) / 32, l = i % 32;
    const int ql   = block[64 * h + 32 * (qd & 1) + l];
    const int low  = (qd >> 1) ? (ql >> 4) : (ql & 0xF);
    const int high = (block[128 + 32 * h + l] >> (2 * qd)) & 3;
    const int q    = (low | (high << 4)) - 32;
    const auto d_bits = static_cast<unsigned short>(block[208] | (static_cast<unsigned>(block[209]) << 8));
    const float scale = __fmul_rn(__half2float(__ushort_as_half(d_bits)),
                                  static_cast<float>(static_cast<std::int8_t>(block[192 + i / 16])));
    out[static_cast<std::size_t>(t) * d + v] = __float2bfloat16_rn(__fmul_rn(scale, static_cast<float>(q)));
}

// Q5_K: {fp16 d; fp16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]} per 256 values. One thread per
// output value, in ggml's operation order (dequantize_row_q5_K): sub-block j (32 values) is
// d1 * q - m1 with d1 = d * sc_j and m1 = dmin * m_j, FP32, two roundings. Value v takes the low
// nibble (j even) or high nibble (j odd) of qs[32 (j / 2) + l] and bit j of qh[l] as its fifth bit.
constexpr int kQ5kBlockBytes = 176;
__global__ void embed_gather_q5_k_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ table,
                                         std::int64_t row_bytes, int d, __nv_bfloat16* __restrict__ out) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (v >= d) { return; }
    const std::uint8_t* block = table + static_cast<std::int64_t>(ids[t]) * row_bytes +
                                static_cast<std::int64_t>(v / 256) * kQ5kBlockBytes;
    const int i = v % 256, j = i / 32, l = i % 32;
    const auto packed = [&](int n) { return static_cast<int>(block[4 + n]); };
    int sc, m;
    if (j < 4) {
        sc = packed(j) & 63;
        m  = packed(j + 4) & 63;
    } else {
        sc = (packed(j + 4) & 0xF) | ((packed(j - 4) >> 6) << 4);
        m  = (packed(j + 4) >> 4) | ((packed(j) >> 6) << 4);
    }
    const auto d_bits    = static_cast<unsigned short>(block[0] | (static_cast<unsigned>(block[1]) << 8));
    const auto dmin_bits = static_cast<unsigned short>(block[2] | (static_cast<unsigned>(block[3]) << 8));
    const float d1       = __fmul_rn(__half2float(__ushort_as_half(d_bits)), static_cast<float>(sc));
    const float m1       = __fmul_rn(__half2float(__ushort_as_half(dmin_bits)), static_cast<float>(m));
    const int ql         = block[48 + 32 * (j / 2) + l];
    const int q          = ((j & 1) ? (ql >> 4) : (ql & 0xF)) + (((block[16 + l] >> j) & 1) << 4);
    out[static_cast<std::size_t>(t) * d + v] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(d1, static_cast<float>(q)), m1));
}

} // namespace

void embed_gather_ggml_q5_k_launch(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kQ5kBlockBytes;
    embed_gather_q5_k_kernel<<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata), row_bytes, d,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ggml_q6_k_launch(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kQ6kBlockBytes;
    embed_gather_q6_k_kernel<false><<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata), nullptr, 0,
        row_bytes, d, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ggml_q6_k_split_launch(const Tensor& ids, const Weight& hot, const Weight& cold,
                                         std::int32_t hot_rows, Tensor& out, cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kQ6kBlockBytes;
    embed_gather_q6_k_kernel<true><<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(cold.qdata),
        static_cast<const std::uint8_t*>(hot.qdata), hot_rows, row_bytes, d, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ggml_iq3_s_launch(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kIq3sBlockBytes;
    embed_gather_iq3_s_kernel<<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata), row_bytes, d,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ggml_iq4_xs_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                     cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kIq4XsBlockBytes;
    embed_gather_iq4_xs_kernel<false><<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(table.qdata), nullptr, 0,
        row_bytes, d, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_ggml_iq4_xs_split_launch(const Tensor& ids, const Weight& hot, const Weight& cold,
                                           std::int32_t hot_rows, Tensor& out, cudaStream_t stream) {
    const int d = out.ne[0];
    const std::int64_t row_bytes = static_cast<std::int64_t>(d / 256) * kIq4XsBlockBytes;
    embed_gather_iq4_xs_kernel<true><<<dim3((d + 255) / 256, ids.ne[0]), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::uint8_t*>(cold.qdata),
        static_cast<const std::uint8_t*>(hot.qdata), hot_rows, row_bytes, d, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream) {
    const std::int32_t T = ids.ne[0];
    if (T <= 176)
        launch_fp8<10, 128>(ids, table, out, stream);
    else
        launch_fp8<5, 128>(ids, table, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
