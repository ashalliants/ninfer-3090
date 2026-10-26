#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::artifact {

struct MaterializationPlan;

// A companion is a device object a startup option derives from a stored device object while the
// weights load. It has a placement of its own in the same weight arena, right after its source,
// so the plan's device capacity counts it and a resume after a model suspend rebuilds it; the
// artifact is unchanged. Each source object has at most one companion.
enum class Companion : std::uint8_t {
    // --lm-head-two-stage: the 2-bit copy of a row-split Q8_G32_FP16 [N,K] output head
    // (N % 16 == 0, K % 128 == 0), src/ops/two_stage_head/proxy_codec.h.
    TwoBitHeadProxy,
};

[[nodiscard]] std::string_view companion_name(Companion kind) noexcept;

// Bytes of the companion of a [rows, columns] source.
[[nodiscard]] std::uint64_t companion_bytes(Companion kind, std::span<const std::uint64_t> shape);

// Builds the companion from the source object's stored bytes (multi-threaded; rows are
// independent, so the result does not depend on the worker count).
void build_companion(Companion kind, std::span<const std::uint64_t> shape,
                     std::span<const std::byte> source, std::span<std::byte> destination);

// The bytes of every companion the plan places on `rank`.
[[nodiscard]] std::uint64_t planned_companion_bytes(const MaterializationPlan& plan,
                                                    std::size_t rank = 0);

// Startup refusal when the weights do not fit the free device memory but would without the
// companions `option` asked for; empty otherwise (a plan that does not fit even without them
// fails as any load does).
[[nodiscard]] std::string companion_fit_refusal(std::string_view option, std::uint64_t weight_bytes,
                                                std::uint64_t companion_bytes,
                                                std::uint64_t free_bytes);

// Advice appended to a startup failure for lack of device memory when companions hold some of it.
[[nodiscard]] std::string companion_memory_advice(std::string_view option,
                                                  std::uint64_t companion_bytes);

} // namespace ninfer::artifact
