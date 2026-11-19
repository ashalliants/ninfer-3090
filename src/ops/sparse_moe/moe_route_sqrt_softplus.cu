// DeepSeek-V4's router (sqrt-softplus scores, biased selection, unbiased weights); the contract is
// include/ninfer/ops/moe_route_sqrt_softplus.h. Written for NInfer-3090 against the model repo's
// `Gate` (inference/model.py).

#include "ninfer/ops/moe_route_sqrt_softplus.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kPerLane = kRouteSqrtSoftplusMaxExperts / 32;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_route_sqrt_softplus: ") + message); }
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = x > 20.0F ? x : log1pf(expf(x));
    return sqrtf(softplus);
}

// One warp per column: each lane holds up to 16 experts' keys; k rounds of a warp-wide argmax (the
// larger key wins, a tie goes to the lower expert id), the winner masked out of later rounds.
__global__ void __launch_bounds__(32)
route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int experts, int top_k,
             float route_scale, bool normalise, int* __restrict__ ids, float* __restrict__ weights) {
    const int t    = blockIdx.x;
    const int lane = threadIdx.x;
    const float* column = logits + static_cast<std::size_t>(t) * experts;

    float score[kPerLane], key[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        const int e = lane + 32 * j;
        if (e < experts) {
            score[j] = sqrt_softplus(column[e]);
            key[j]   = score[j] + bias[e];
        } else {
            score[j] = 0.0F;
            key[j]   = -INFINITY;
        }
    }

    float chosen[kRouteSqrtSoftplusMaxTopK];
    int chosen_id[kRouteSqrtSoftplusMaxTopK];
    for (int i = 0; i < top_k; ++i) {
        float best = -INFINITY;
        int best_id = 0x7FFFFFFF, best_slot = -1;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            const int e = lane + 32 * j;
            if (e < experts && (key[j] > best || (key[j] == best && e < best_id))) {
                best = key[j];
                best_id = e;
                best_slot = j;
            }
        }
        float win_key = best;
        int win_id    = best_id;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_key = __shfl_xor_sync(0xFFFFFFFFU, win_key, offset);
            const int other_id    = __shfl_xor_sync(0xFFFFFFFFU, win_id, offset);
            if (other_key > win_key || (other_key == win_key && other_id < win_id)) {
                win_key = other_key;
                win_id  = other_id;
            }
        }
        // The winner's lane reports its unbiased score and retires the expert.
        float win_score = 0.0F;
        if (win_id % 32 == lane && best_id == win_id) {
#pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                if (j == best_slot) {
                    win_score = score[j];
                    key[j]    = -INFINITY;
                }
            }
        }
        win_score = __shfl_sync(0xFFFFFFFFU, win_score, win_id % 32);
        chosen[i]    = win_score;
        chosen_id[i] = win_id;
    }

    if (lane == 0) {
        float sum = 0.0F;
        for (int i = 0; i < top_k; ++i) { sum += chosen[i]; }
        for (int i = 0; i < top_k; ++i) {
            const float w = normalise && top_k > 1 ? chosen[i] / (sum + 1e-20F) : chosen[i];
            ids[static_cast<std::size_t>(t) * top_k + i]     = chosen_id[i];
            weights[static_cast<std::size_t>(t) * top_k + i] = w * route_scale;
        }
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

} // namespace

void moe_route_sqrt_softplus(const Tensor& logits, const Tensor& bias, std::int32_t top_k, float route_scale,
                             bool norm_topk_prob, Tensor& ids, Tensor& weights, cudaStream_t stream) {
    require(contiguous(logits, DType::FP32) && contiguous(bias, DType::FP32) && contiguous(ids, DType::I32) &&
                contiguous(weights, DType::FP32),
            "contiguous FP32 logits, bias and weights and I32 ids are required");
    const int experts = logits.ne[0], columns = logits.ne[1];
    require(experts >= 1 && experts <= kRouteSqrtSoftplusMaxExperts && columns >= 1, "expert or column count out of range");
    require(top_k >= 1 && top_k <= experts && top_k <= kRouteSqrtSoftplusMaxTopK, "top_k out of range");
    require(bias.numel() == experts && ids.ne[0] == top_k && ids.ne[1] == columns && weights.ne[0] == top_k &&
                weights.ne[1] == columns,
            "shapes disagree");
    require(std::isfinite(route_scale), "route_scale must be finite");
    for (const Tensor* out : {&ids, &weights}) {
        for (const Tensor* in : {&logits, &bias}) { require(!overlaps(*out, *in), "an output overlaps an input"); }
    }
    require(!overlaps(ids, weights), "ids and weights overlap");
    route_kernel<<<columns, 32, 0, stream>>>(static_cast<const float*>(logits.data), static_cast<const float*>(bias.data),
                                             experts, top_k, route_scale, norm_topk_prob,
                                             static_cast<int*>(ids.data), static_cast<float*>(weights.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("moe_route_sqrt_softplus: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
