// T2G128 row-split weights through the shared integer-activation GEMM. Three registered input
// widths (the hidden 5120, the attention/GDN mixer output 6144 and the MLP intermediate 17408) and
// any row count that is a whole number of 64-row blocks; T a multiple of 128.

#include "ops/linear/t2/t2_a8.h"

#include "core/device.h"
#include "ops/common/rowsplit_a8_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

using Rows = a8::ContiguousRows<1>;

template <std::int32_t kCols, int NT, class Epilogue>
void launch(const Tensor& x, const Weight& w, Epilogue epilogue, std::int32_t tokens,
            std::int8_t* codes, __half* scales, cudaStream_t stream) {
    constexpr int BN = a8::kWarpsN * NT * 8;
    a8::quantize_activations<kCols, BN><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens, codes, scales);
    CUDA_CHECK(cudaGetLastError());

    const std::size_t smem = a8::shared_bytes<a8::T2Codec, 1, NT, Rows>(kCols);
    const dim3 grid(w.n / Rows::kRowsPerBlock, tokens / BN);
    auto* kernel = a8::a8_mma_kernel<a8::T2Codec, kCols, 1, NT, Rows, Epilogue>;
    if (smem > 48 * 1024) {
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem));
        });
    }
    kernel<<<grid, a8::kThreads, smem, stream>>>(static_cast<const std::uint8_t*>(w.qdata), nullptr,
                                                 static_cast<const __half*>(w.scales), codes,
                                                 scales, tokens, Rows{0}, epilogue, 0);
    CUDA_CHECK(cudaGetLastError());
}

template <std::int32_t kCols, class Epilogue>
void launch_for_tile(const Tensor& x, const Weight& w, Epilogue epilogue, std::int32_t tokens,
                     std::int8_t* codes, __half* scales, cudaStream_t stream) {
    switch (a8::token_tile(tokens, 512)) {
    case 512:
        launch<kCols, 16>(x, w, epilogue, tokens, codes, scales, stream);
        return;
    case 256:
        launch<kCols, 8>(x, w, epilogue, tokens, codes, scales, stream);
        return;
    default:
        launch<kCols, 4>(x, w, epilogue, tokens, codes, scales, stream);
        return;
    }
}

template <class Epilogue>
void run(const Tensor& x, const Weight& w, Epilogue epilogue, WorkspaceArena& workspace,
         cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!t2_a8_supported(w, tokens)) { throw std::invalid_argument("t2 a8: unsupported profile"); }
    auto scope = workspace.scope();
    const DeviceSpan codes =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * static_cast<std::size_t>(w.k));
    const DeviceSpan scales =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) *
                              (static_cast<std::size_t>(w.k) / a8::kGroup) * sizeof(__half));
    auto* code_data  = reinterpret_cast<std::int8_t*>(codes.data);
    auto* scale_data = reinterpret_cast<__half*>(scales.data);
    switch (w.k) {
    case 5120:
        launch_for_tile<5120>(x, w, epilogue, tokens, code_data, scale_data, stream);
        return;
    case 6144:
        launch_for_tile<6144>(x, w, epilogue, tokens, code_data, scale_data, stream);
        return;
    case 17408:
        launch_for_tile<17408>(x, w, epilogue, tokens, code_data, scale_data, stream);
        return;
    default:
        break;
    }
    throw std::invalid_argument("t2 a8: unregistered input width");
}

} // namespace

bool t2_a8_admits(LinearPolicy policy) {
    return policy == LinearPolicy::AllowA8Int || policy == LinearPolicy::AllowA8IntDecode ||
           policy == LinearPolicy::AllowPrefillCublas;
}

bool t2_a8_shape_supported(std::int32_t output_rows, std::int32_t input_rows) {
    return output_rows > 0 && output_rows % Rows::kRowsPerBlock == 0 &&
           (input_rows == 5120 || input_rows == 6144 || input_rows == 17408);
}

bool t2_a8_supported(const Weight& w, std::int32_t tokens) {
    return w.qtype == QType::T2_G128_FP16 && w.layout == QuantLayout::RowSplit &&
           w.qdata != nullptr && w.scales != nullptr && t2_a8_shape_supported(w.n, w.k) &&
           a8::tokens_supported(tokens);
}

std::size_t t2_a8_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t max_tokens) {
    if (!t2_a8_admits(policy) || !t2_a8_shape_supported(output_rows, input_rows) ||
        max_tokens < 128) {
        return 0;
    }
    return a8::activation_workspace_bytes(input_rows, max_tokens);
}

void t2_a8_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    run(x, w, a8::StoreEpilogue{reinterpret_cast<__nv_bfloat16*>(out.data), w.n, 0}, workspace,
        stream);
}

void t2_a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    run(x, w, a8::ResidualAddEpilogue{reinterpret_cast<__nv_bfloat16*>(residual.data), w.n},
        workspace, stream);
}

} // namespace ninfer::ops::detail
