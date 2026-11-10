#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ninfer::artifact {

inline constexpr std::array<std::byte, 8> kEntryMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
inline constexpr std::array<std::byte, 8> kPartMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'P'},
    std::byte{'R'}, std::byte{'T'}, std::byte{0},   std::byte{3},
};
inline constexpr std::size_t kHeaderBytes        = 32;
inline constexpr std::uint64_t kPayloadAlignment = 4096;
using ArtifactId                                 = std::array<std::byte, 16>;

// The artifact's id as 32 lowercase hex digits. Every conversion writes a fresh one, so it names the
// file's content: every copy of the same artifact has it, wherever and whenever the copy was made.
inline std::string artifact_id_hex(const ArtifactId& id) {
    constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(id.size() * 2);
    for (const std::byte byte : id) {
        const auto value = std::to_integer<unsigned>(byte);
        out += kDigits[value >> 4U];
        out += kDigits[value & 0xFU];
    }
    return out;
}

inline std::uint64_t read_u64_le(const std::byte* bytes) noexcept {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

inline std::uint32_t read_u32_le(const std::byte* bytes) noexcept {
    std::uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
        out |= std::uint32_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
    }
    return out;
}

} // namespace ninfer::artifact
