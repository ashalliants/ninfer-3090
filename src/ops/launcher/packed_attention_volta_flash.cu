// ninfer::ops::detail - sm_70 route of the vision tower's packed attention (head size 72,
// sixteen heads, full attention within equal-length segments).
//
// The packed attention kernel (softmax_attention/dense/packed/kernel.cuh) is written in Ampere BF16
// mma.sync and ldmatrix, which Volta runs on the shuffle emulation (ops/common/volta_compat.cuh):
// a 4-megapixel image spent 7.6 s of its encode there. This route drives the vendored llama.cpp
// FP16 flash-attention kernel (third_party/llama_cpp_fattn, native mma.m8n8k4) instead, the
// kernel the sm_70 text prompt route uses (gqa_attention_volta_flash.cu):
//
//   * head size 72 is zero-padded to 80, the nearest size the kernel instantiates; the padding
//     adds exact zeros to every Q.K product and produces output columns 72..79, which are dropped;
//   * Q is staged as FP32, K and V as FP16, all unscaled. BF16 converts to FP16 exactly inside
//     FP16's range, so Q.K is exact products summed in FP32, as in the BF16 kernel; the softmax
//     scale is applied to those sums in FP32 (the kernel's logit_softcap argument without
//     use_logit_softcap, a ninfer patch: its own scale argument rounds Q x scale to FP16, which
//     moved outputs of sharp attention rows by ~3%). A value beyond 65504 cannot convert, so the
//     staging kernel flags it and the packed kernel then reruns the whole call on its BF16 path
//     (it launches unconditionally and returns at once unless the flag is set: no host sync);
//   * P x V accumulates in FP16 on Volta (mma_tile_sizes, Volta branch), so its error grows with
//     the keys one CTA sums: 4.3e-3 relative L2 over a 16K-patch image against 1.8e-3 for the BF16
//     kernel. A segment longer than kSplitKeys is therefore split along the keys across CTAs
//     (llama.cpp's stream-K with a whole number of CTAs per output tile), and the uniform fixup
//     kernel merges the partial outputs in FP32: 1.9e-3 at 1,024 keys a CTA (2.2e-3 at 2,048),
//     for 6% more time than no split at 16K;
//   * segments are the kernel's sequence dimension (ne03) and K/V heads equal Q heads (ncols2 = 1,
//     which needs no mask: the last key tile of a segment is bounds-checked);
//   * heads go through in passes sized so the staging stays within kPassBudgetBytes.

#include "fattn-mma-f16.cuh" // vendored; must precede ninfer headers (defines WARP_SIZE)

#include "core/device.h"
#include "core/tensor.h"
#include "ops/softmax_attention/dense/packed/launch.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim    = 72;
constexpr int kHeads      = 16;
constexpr int kPaddedDim  = 80;
constexpr int kNcols1     = 64; // query rows per tile
constexpr int kNcols2     = 1;
constexpr int kKeyTile    = 64; // the config's nbatch_fa at D = 80, 64 columns (checked at launch)
// Keys one CTA sums at most, as whole key tiles (see the header comment).
constexpr int kSplitKeys  = 1024;
constexpr std::size_t kPassBudgetBytes = std::size_t{64} << 20;

std::size_t aligned(std::size_t n) { return (n + 255) / 256 * 256; }

int key_tiles(std::int32_t keys) { return (keys + kKeyTile - 1) / kKeyTile; }

// CTAs per output tile for a segment of `length` keys.
int splits_for(std::int32_t length) {
    const int tiles = key_tiles(length);
    return std::max(1, std::min(tiles, (tiles * kKeyTile + kSplitKeys - 1) / kSplitKeys));
}

// Output tiles of one head over T tokens in segments of `length` (each segment's last tile may be
// partial).
std::int64_t output_tiles(std::int32_t tokens, std::int32_t length) {
    return static_cast<std::int64_t>(tokens / length) * ((length + kNcols1 - 1) / kNcols1);
}

// The staging of one pass of `heads` heads: FP32 Q and output, FP16 K and V ([80, heads, T]
// each), the stream-K partials (float2 per tile column: max and sum, twice, then the D values)
// when the segment splits, and the overflow flag.
std::size_t pass_bytes(std::int32_t tokens, std::int32_t length, int heads) {
    const std::size_t elements = static_cast<std::size_t>(tokens) * heads * kPaddedDim;
    const int splits           = splits_for(length);
    const std::size_t meta =
        splits > 1 ? static_cast<std::size_t>(splits) * output_tiles(tokens, length) * heads *
                         kNcols1 * (2 + kPaddedDim / 2) * sizeof(float2)
                   : 0;
    return 2 * aligned(elements * sizeof(float)) + 2 * aligned(elements * sizeof(half)) +
           aligned(meta) + aligned(sizeof(int));
}

// Halve the heads per pass until the staging fits kPassBudgetBytes.
int heads_per_pass(std::int32_t tokens, std::int32_t length) {
    int heads = kHeads;
    while (heads > 1 && pass_bytes(tokens, length, heads) > kPassBudgetBytes) heads /= 2;
    return heads;
}

struct FlashConfig {
    int nthreads       = 0;
    int nbatch_fa      = 0;
    std::size_t shared = 0;
};

const FlashConfig& flash_config() {
    static const FlashConfig config = [] {
        FlashConfig c;
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        const int cc       = prop.major * 100 + prop.minor * 10;
        constexpr int D    = kPaddedDim;
        constexpr int ncol = kNcols1 * kNcols2;
        const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, ncol, cc);
        const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, ncol, cc);
        const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, ncol, cc);
        const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, ncol, cc);
        const int nstages        = ggml_cuda_fattn_mma_get_nstages(D, D, kNcols1, kNcols2, cc);
        c.nthreads               = ggml_cuda_fattn_mma_get_nthreads(D, D, ncol, cc);
        c.nbatch_fa              = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, ncol, cc);
        const int nwarps         = c.nthreads / WARP_SIZE;
        const int cols_per_warp  = std::min(ncol, get_cols_per_warp(cc));
        const size_t kv1  = size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t kv2  = size_t(c.nbatch_fa) * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t q    = size_t(ncol) * (D / 2 + 4) * sizeof(half2);
        const size_t mask = size_t(kNcols1) * (c.nbatch_fa / 2 + 4) * sizeof(half2);
        const size_t comb = size_t(nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
        const size_t kv   = nstages <= 1 ? kv1 : kv2;
        c.shared = std::max(comb, Q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
        auto kernel = flash_attn_ext_f16<D, D, kNcols1, kNcols2, false, false>;
        cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(c.shared));
        return c;
    }();
    return config;
}

// One thread per (token, head-in-pass, padded dimension): Q -> FP32, K/V -> FP16, zeros past 72.
// A Q, K or V value outside FP16's finite range sets *overflow.
__global__ void stage_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v, std::int64_t q_token,
                             std::int64_t k_token, std::int64_t v_token, int head_begin, int heads,
                             int tokens, float* __restrict__ q_out,
                             half* __restrict__ k_out, half* __restrict__ v_out,
                             int* __restrict__ overflow) {
    const std::int64_t i     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(tokens) * heads * kPaddedDim;
    if (i >= total) return;
    const int d     = static_cast<int>(i % kPaddedDim);
    const int rest  = static_cast<int>(i / kPaddedDim);
    const int h     = rest % heads;
    const int t     = rest / heads;
    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
    if (d < kHeadDim) {
        const int head = head_begin + h;
        qv = __bfloat162float(q[t * q_token + head * kHeadDim + d]);
        kv = __bfloat162float(k[t * k_token + head * kHeadDim + d]);
        vv = __bfloat162float(v[t * v_token + head * kHeadDim + d]);
        constexpr float kMax = 65504.0f;
        if (!(fabsf(kv) <= kMax && fabsf(vv) <= kMax && fabsf(qv) <= kMax)) {
            *overflow = 1;
        }
    }
    q_out[i] = qv;
    k_out[i] = __float2half_rn(kv);
    v_out[i] = __float2half_rn(vv);
}

// FP32 [80, heads, T] -> BF16 out [72, 16, T] at the pass's heads.
__global__ void unstage_kernel(const float* __restrict__ in, int head_begin, int heads, int tokens,
                               __nv_bfloat16* __restrict__ out) {
    const std::int64_t i     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(tokens) * heads * kHeadDim;
    if (i >= total) return;
    const int d    = static_cast<int>(i % kHeadDim);
    const int rest = static_cast<int>(i / kHeadDim);
    const int h    = rest % heads;
    const int t    = rest / heads;
    out[(static_cast<std::int64_t>(t) * kHeads + head_begin + h) * kHeadDim + d] =
        __float2bfloat16_rn(in[(static_cast<std::int64_t>(t) * heads + h) * kPaddedDim + d]);
}

std::int64_t token_stride(const Tensor& t) {
    return t.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

} // namespace

std::size_t packed_attention_volta_workspace_bytes(std::int32_t max_tokens) {
    // pass_bytes grows with T and with L (more splits, more partials); a call that fits 16 heads
    // takes at most what T = L = max_tokens does, and one that halves its heads stays within the
    // budget unless a single head is past it, which again peaks at T = L = max_tokens.
    const int heads         = heads_per_pass(max_tokens, max_tokens);
    const std::size_t bytes = pass_bytes(max_tokens, max_tokens, heads);
    return heads == kHeads ? bytes : std::max(bytes, kPassBudgetBytes);
}

void packed_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         std::int32_t segment_length, float scale,
                                         void* workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t segments = tokens / segment_length;
    const int heads             = heads_per_pass(tokens, segment_length);
    const int splits            = splits_for(segment_length);
    const std::size_t elements  = static_cast<std::size_t>(tokens) * heads * kPaddedDim;
    auto* base                  = static_cast<unsigned char*>(workspace);
    auto* q_f32                 = reinterpret_cast<float*>(base);
    base += aligned(elements * sizeof(float));
    auto* dst = reinterpret_cast<float*>(base);
    base += aligned(elements * sizeof(float));
    auto* k_f16 = reinterpret_cast<half*>(base);
    base += aligned(elements * sizeof(half));
    auto* v_f16 = reinterpret_cast<half*>(base);
    base += aligned(elements * sizeof(half));
    auto* meta = splits > 1 ? reinterpret_cast<float2*>(base) : nullptr;
    base += aligned(splits > 1 ? static_cast<std::size_t>(splits) *
                                     output_tiles(tokens, segment_length) * heads * kNcols1 *
                                     (2 + kPaddedDim / 2) * sizeof(float2)
                               : 0);
    auto* overflow = reinterpret_cast<int*>(base);
    CUDA_CHECK(cudaMemsetAsync(overflow, 0, sizeof(int), stream));

    const FlashConfig& c = flash_config();
    if (c.nbatch_fa != kKeyTile) {
        throw std::logic_error("packed attention volta flash: unexpected key tile");
    }
    const int tiles_per_segment = (segment_length + kNcols1 - 1) / kNcols1;
    const int ntiles_dst        = tiles_per_segment * heads * segments;
    const int nblocks           = ntiles_dst * splits;
    const std::int32_t nb01 = heads * kPaddedDim * static_cast<std::int32_t>(sizeof(float));
    const std::int32_t nb02 = kPaddedDim * static_cast<std::int32_t>(sizeof(float));
    const std::int32_t nb03 = segment_length * nb01;
    const std::int32_t nb11 = heads * kPaddedDim * static_cast<std::int32_t>(sizeof(half));
    const std::int32_t nb12 = kPaddedDim * static_cast<std::int32_t>(sizeof(half));
    const std::int64_t nb13 = static_cast<std::int64_t>(segment_length) * nb11;
    const uint3 ne01        = init_fastdiv_values(segment_length);

    constexpr int kThreads = 256;
    for (int head_begin = 0; head_begin < kHeads; head_begin += heads) {
        const std::int64_t staged = static_cast<std::int64_t>(tokens) * heads * kPaddedDim;
        stage_kernel<<<static_cast<unsigned>((staged + kThreads - 1) / kThreads), kThreads, 0,
                       stream>>>(static_cast<const __nv_bfloat16*>(q.data),
                                 static_cast<const __nv_bfloat16*>(k.data),
                                 static_cast<const __nv_bfloat16*>(v.data), token_stride(q),
                                 token_stride(k), token_stride(v), head_begin, heads, tokens,
                                 q_f32, k_f16, v_f16, overflow);
        CUDA_CHECK(cudaGetLastError());
        flash_attn_ext_f16<kPaddedDim, kPaddedDim, kNcols1, kNcols2, false, false>
            <<<nblocks, dim3(WARP_SIZE, c.nthreads / WARP_SIZE, 1), c.shared, stream>>>(
                reinterpret_cast<const char*>(q_f32), reinterpret_cast<const char*>(k_f16),
                reinterpret_cast<const char*>(v_f16), nullptr, nullptr, nullptr, dst, meta,
                /*scale=*/1.0f, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u,
                /*logit_softcap: the FP32 KQ scale*/ scale,
                kPaddedDim, ne01, heads, segments, nb01, nb02, nb03,
                kPaddedDim, segment_length, heads, segments, nb11, nb12, nb13,
                nb11, nb12, nb13,
                0, 0, 0,
                0, 0, 0);
        CUDA_CHECK(cudaGetLastError());
        if (splits > 1) {
            const uint3 fd0 = init_fastdiv_values(tiles_per_segment * heads);
            const uint3 fd1 = init_fastdiv_values(tiles_per_segment);
            const uint3 fd2 = init_fastdiv_values(tiles_per_segment);
            flash_attn_stream_k_fixup_uniform<kPaddedDim, kNcols1, kNcols2>
                <<<dim3(static_cast<unsigned>(ntiles_dst), kNcols1, kNcols2), dim3(kPaddedDim, 1, 1),
                   0, stream>>>(dst, meta, segment_length, heads, heads, nblocks, 1, splits, fd0,
                                fd1, fd2);
            CUDA_CHECK(cudaGetLastError());
        }
        const std::int64_t written = static_cast<std::int64_t>(tokens) * heads * kHeadDim;
        unstage_kernel<<<static_cast<unsigned>((written + kThreads - 1) / kThreads), kThreads, 0,
                         stream>>>(dst, head_begin, heads, tokens,
                                   static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
    // FP16 could not hold an input: the BF16 kernel recomputes the whole call (a no-op otherwise).
    packed_attention_uniform_launch(q, k, v, segment_length, out, stream, overflow);
}

} // namespace ninfer::ops::detail
