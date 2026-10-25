#include "models/qwen4_exp/execution/hc_int8_copies.h"

#include "core/device.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::execution {
namespace {

std::uint64_t align(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

// One [n, k] matrix's codes then scales, each on a 256-byte boundary.
std::uint64_t matrix_bytes(std::uint64_t n, std::uint64_t k) {
    return align(ops::hc_int8_code_bytes(static_cast<std::int32_t>(n),
                                         static_cast<std::int32_t>(k))) +
           align(ops::hc_int8_scale_bytes(static_cast<std::int32_t>(n),
                                          static_cast<std::int32_t>(k)));
}

} // namespace

std::vector<const HyperConnectionParameters*>
HcInt8Copies::mixers(const std::vector<BlockParameters>& layers,
                     const HyperConnectionParameters& final_mixer, const MtpParameters* drafter) {
    std::vector<const HyperConnectionParameters*> out;
    out.reserve(2 * layers.size() + 4);
    for (const auto& block : layers) {
        out.push_back(&block.attn_hc);
        out.push_back(&block.mlp_hc);
    }
    out.push_back(&final_mixer);
    if (drafter != nullptr) {
        out.push_back(&drafter->attn_hc);
        out.push_back(&drafter->mlp_hc);
        out.push_back(&drafter->final_mixer);
    }
    return out;
}

std::uint64_t HcInt8Copies::bytes(std::span<const HyperConnectionParameters* const> mixers) {
    std::uint64_t total = 0;
    for (const auto* mixer : mixers) {
        total += matrix_bytes(mixer->down.weight.n, mixer->down.weight.k) +
                 matrix_bytes(mixer->up.weight.n, mixer->up.weight.k);
    }
    return total + kStatusBytes;
}

std::uint64_t HcInt8Copies::mixer_bytes(std::uint32_t down_rows, std::uint32_t width,
                                        std::uint32_t rank) {
    return matrix_bytes(down_rows, width) + matrix_bytes(width, rank);
}

HcInt8Copies::HcInt8Copies(std::vector<const HyperConnectionParameters*> mixers,
                           std::uint64_t planned, cudaStream_t stream) {
    const std::uint64_t need = bytes(mixers);
    if (need != planned) {
        throw std::logic_error("--hc-int8: the int8 copies need " + std::to_string(need) +
                               " bytes, the plan holds " + std::to_string(planned));
    }
    buffer_        = DeviceBuffer(need);
    auto* base     = static_cast<std::byte*>(buffer_.p);
    auto* status   = base + need - kStatusBytes;
    std::size_t at = 0;
    CUDA_CHECK(cudaMemsetAsync(status, 0, sizeof(unsigned), stream));
    const auto copy = [&](const Weight& w) {
        const std::size_t code_bytes  = ops::hc_int8_code_bytes(w.n, w.k),
                          scale_bytes = ops::hc_int8_scale_bytes(w.n, w.k);
        std::byte* codes              = base + at;
        std::byte* scales             = codes + align(code_bytes);
        at += align(code_bytes) + align(scale_bytes);
        ops::hc_int8_quantize(w, {codes, code_bytes}, {scales, scale_bytes},
                              {status, sizeof(unsigned)}, stream);
        return ops::HcInt8Weights{reinterpret_cast<const std::int8_t*>(codes),
                                  reinterpret_cast<const std::uint16_t*>(scales), w.n, w.k};
    };
    for (const auto* mixer : mixers) {
        HcInt8Mixer copies;
        copies.down = copy(mixer->down.weight);
        copies.up   = copy(mixer->up.weight);
        copies_.emplace(mixer, copies);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    unsigned refused = 0;
    CUDA_CHECK(cudaMemcpy(&refused, status, sizeof refused, cudaMemcpyDeviceToHost));
    if (refused != 0) {
        throw std::runtime_error(
            "--hc-int8: a hyper-connection matrix holds a value its int8 copy cannot represent "
            "(infinite, NaN, or a group whose largest value is past 127 * 65504); remove "
            "--hc-int8");
    }
}

const HcInt8Mixer& HcInt8Copies::at(const HyperConnectionParameters& mixer) const {
    const auto found = copies_.find(&mixer);
    if (found == copies_.end()) {
        throw std::invalid_argument("--hc-int8: a hyper-connection mixer has no int8 copy");
    }
    return found->second;
}

bool HcInt8Copies::covers(const HyperConnectionParameters& mixer) const noexcept {
    return copies_.find(&mixer) != copies_.end();
}

} // namespace ninfer::models::qwen4_exp::execution
