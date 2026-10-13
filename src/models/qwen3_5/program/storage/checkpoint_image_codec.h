#pragma once

// The byte container shared by the checkpoint images of the Programs that implement the context
// store (the Qwen3.5 Program: checkpoint_image.cpp; Qwen3.8-Flash-Next: qwen4_exp/program/
// checkpoint_image.cpp): the magic, host-order POD fields and vectors, the trailing checksum and the
// exact prefix identity (ledger, token types, three-axis positions, media items, rewrite frontiers
// and the shortlist digests). Each Program writes its own version and configuration section, and
// rejects any other.

#include "models/qwen3_5/program/prefix_identity.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ninfer::models::qwen3_5::detail::image_codec {

inline constexpr char kImageMagic[8] = {'N', 'I', 'N', 'F', 'C', 'K', 'P', 'T'};
inline constexpr std::size_t kMaximumModelBindingBytes = 4096;
inline constexpr std::uint32_t kMaximumImagePoints     = 64;

class ImageWriter {
public:
    explicit ImageWriter(std::vector<std::uint8_t>& out) : out_(out) {}

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out_.insert(out_.end(), begin, begin + count);
    }

    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }

    template <class T>
    void vector(const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        pod<std::uint64_t>(values.size());
        bytes(values.data(), values.size() * sizeof(T));
    }

    // Sizes a payload region the caller fills once the whole image is sized, so no asynchronous
    // copy records a destination the vector later reallocates away.
    std::size_t reserve(std::size_t count) {
        const std::size_t offset = out_.size();
        out_.resize(out_.size() + count);
        return offset;
    }

private:
    std::vector<std::uint8_t>& out_;
};

class ImageReader {
public:
    explicit ImageReader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) { throw std::invalid_argument("checkpoint image is truncated"); }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }

    template <class T>
    [[nodiscard]] T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }

    template <class T>
    [[nodiscard]] std::vector<T> vector(std::size_t maximum, const char* label) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto count = pod<std::uint64_t>();
        if (count > maximum) {
            throw std::invalid_argument(std::string("checkpoint image ") + label + " count is out of range");
        }
        std::vector<T> values(static_cast<std::size_t>(count));
        bytes(values.data(), values.size() * sizeof(T));
        return values;
    }

    [[nodiscard]] std::span<const std::uint8_t> payload(std::size_t count) {
        if (count > data_.size() - cursor_) { throw std::invalid_argument("checkpoint image payload is truncated"); }
        const auto region = data_.subspan(cursor_, count);
        cursor_ += count;
        return region;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

// Word-wise, so a multi-GB image hashes at memory speed. Not cryptographic: the store is trusted
// local state and the threat is corruption, not tampering.
[[nodiscard]] std::uint64_t image_checksum(std::span<const std::uint8_t> bytes);

// The image's head up to its configuration section: magic, `version` and the caller's model binding.
void write_head(ImageWriter& writer, std::uint32_t version, std::string_view binding);
// Reads the head: a reader over the body (the image without its trailing checksum) positioned at the
// configuration section, and the stored checksum. Throws std::invalid_argument when the bytes are not
// an image of `version` for `binding`. The caller compares its configuration section next, then calls
// verify_checksum() before it allocates or copies anything.
struct ImageBody {
    ImageReader reader;
    std::span<const std::uint8_t> body;
    std::uint64_t checksum = 0;
};
[[nodiscard]] ImageBody read_head(std::span<const std::uint8_t> image, std::uint32_t version, std::string_view binding);
void verify_checksum(const ImageBody& image);
// Appends the trailing checksum over everything before `checksum_offset` (a reserved word).
void seal(std::vector<std::uint8_t>& bytes, std::size_t checksum_offset);

void write_session(ImageWriter& writer, const std::optional<PreparedSessionKey>& session);
[[nodiscard]] std::optional<PreparedSessionKey> read_session(ImageReader& reader);

// One exact prefix identity. read_identity bounds the ledger by `capacity` tokens and its ids by
// `vocabulary`, and checks the identity and digests match the ledger.
void write_identity(ImageWriter& writer, const std::vector<TokenId>& ledger, const ResidentPrefixIdentity& identity,
                    const PrefixShortlistDigests& digests);
void read_identity(ImageReader& reader, std::uint32_t capacity, std::int64_t vocabulary, std::vector<TokenId>& ledger,
                   ResidentPrefixIdentity& identity, PrefixShortlistDigests& digests);

} // namespace ninfer::models::qwen3_5::detail::image_codec
