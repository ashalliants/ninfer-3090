#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::runtime {

// The id the Engine stores a checkpoint image under: lowercase hex of its endpoint key (both prefix
// digests, then the frontier as 8 digits). An image of the same endpoint replaces it. A shared
// prefix's id ends in an extra '5', so it never replaces a conversation image whose endpoint happens
// to fall at the same position. A request can therefore tell from a listed id alone whether the
// image ends on its own prompt, without reading the manifest.
struct ContextStoreImageId {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier = 0;
    bool shared            = false;
};

[[nodiscard]] inline std::string context_store_image_id(const std::array<std::uint64_t, 2>& digests,
                                                        std::uint32_t frontier, bool shared) {
    char text[42];
    std::snprintf(text, sizeof(text), "%016llx%016llx%08x%s",
                  static_cast<unsigned long long>(digests[0]),
                  static_cast<unsigned long long>(digests[1]), static_cast<unsigned>(frontier),
                  shared ? "5" : "");
    return text;
}

// The endpoint key an id names; nullopt for an id the Engine did not make.
[[nodiscard]] inline std::optional<ContextStoreImageId> parse_context_store_image_id(std::string_view id) {
    constexpr std::size_t kLength = 16 + 16 + 8;
    const bool shared             = id.size() == kLength + 1;
    if (id.size() != kLength && !(shared && id.back() == '5')) { return std::nullopt; }
    ContextStoreImageId out;
    out.shared       = shared;
    const auto field = [&](std::size_t at, std::size_t length, auto& value) {
        for (std::size_t i = at; i < at + length; ++i) {
            const char c = id[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) { return false; }
        }
        const auto [end, error] = std::from_chars(id.data() + at, id.data() + at + length, value, 16);
        return error == std::errc() && end == id.data() + at + length;
    };
    if (!field(0, 16, out.digests[0]) || !field(16, 16, out.digests[1]) ||
        !field(32, 8, out.frontier)) {
        return std::nullopt;
    }
    return out;
}

} // namespace ninfer::runtime
