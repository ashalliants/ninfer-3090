#pragma once

// --hc-int8: int8 copies (ops::hc_int8_quantize, Basalt's group rule) of the hyper-connection
// matrices of every mixer a Forward runs, made once from the BF16 weights, which stay loaded. The
// mixers' calls of up to 16 columns then read the copies (ops::hyper_connection_mix's int8 form):
// about 0.53 of the bytes per call, and different numerics (each weight rounded to an int8 code
// times its group's binary16 scale), so the option is judged by the decode-path KLD gate, not by
// bitwise identity. Wider calls (prefill) read the BF16 weights.

#include "core/arena.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/hyper_connection.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// One mixer's copies: its down projection (with the injection rows when it combines) and its up
// projection.
struct HcInt8Mixer {
    ops::HcInt8Weights down, up;
};

class HcInt8Copies {
public:
    // The mixers the copies cover, in their layout order: each layer's attn_hc then mlp_hc, the
    // final mixer, then (`drafter` non-null) the drafter's attn_hc, mlp_hc and final_mixer.
    [[nodiscard]] static std::vector<const HyperConnectionParameters*>
    mixers(const std::vector<BlockParameters>& layers, const HyperConnectionParameters& final_mixer,
           const MtpParameters* drafter);
    // The device bytes of these mixers' copies: per mixer the down then the up matrix, each its
    // codes then its scales on 256-byte boundaries, then one 256-byte status word.
    [[nodiscard]] static std::uint64_t
    bytes(std::span<const HyperConnectionParameters* const> mixers);
    // bytes() of one mixer of `down_rows` [rows, width] down rows and a [width, rank] up matrix,
    // without the status word.
    [[nodiscard]] static std::uint64_t mixer_bytes(std::uint32_t down_rows, std::uint32_t width,
                                                   std::uint32_t rank);
    static constexpr std::uint64_t kStatusBytes = 256;

    // Allocates `planned` bytes (std::logic_error when the mixers' copies need another size),
    // quantizes every mixer's matrices on `stream` and waits for them. Throws std::runtime_error
    // naming --hc-int8 when a matrix holds a value that cannot be copied (infinite, NaN, or too
    // large for a binary16 scale).
    HcInt8Copies(std::vector<const HyperConnectionParameters*> mixers, std::uint64_t planned,
                 cudaStream_t stream);

    // The copies of `mixer`; std::invalid_argument when it is not one of the covered mixers.
    [[nodiscard]] const HcInt8Mixer& at(const HyperConnectionParameters& mixer) const;
    [[nodiscard]] bool covers(const HyperConnectionParameters& mixer) const noexcept;

    [[nodiscard]] std::uint64_t device_bytes() const noexcept { return buffer_.bytes; }

private:
    DeviceBuffer buffer_;
    std::unordered_map<const HyperConnectionParameters*, HcInt8Mixer> copies_;
};

} // namespace ninfer::models::qwen4_exp::execution
