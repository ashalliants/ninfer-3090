#include "models/qwen3_5/program/storage/checkpoint_image_codec.h"

#include <array>

namespace ninfer::models::qwen3_5::detail::image_codec {
namespace {

void write_vision_items(ImageWriter& writer, const std::vector<VisionItem>& items) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(items.size()));
    for (const VisionItem& item : items) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(item.modality));
        writer.pod<std::int32_t>(item.grid.temporal);
        writer.pod<std::int32_t>(item.grid.height);
        writer.pod<std::int32_t>(item.grid.width);
        writer.pod<std::uint64_t>(item.patch_begin);
        writer.pod<std::uint64_t>(item.patch_count);
        writer.bytes(item.content_digest.data(), item.content_digest.size());
        writer.vector(item.timestamps);
        writer.pod<std::uint32_t>(static_cast<std::uint32_t>(item.token_spans.size()));
        for (const TokenSpan& span : item.token_spans) {
            writer.pod<std::uint64_t>(span.begin);
            writer.pod<std::uint64_t>(span.count);
        }
    }
}

std::vector<VisionItem> read_vision_items(ImageReader& reader, std::size_t tokens) {
    const auto count = reader.pod<std::uint32_t>();
    if (count > tokens) { throw std::invalid_argument("checkpoint image vision item count is out of range"); }
    std::vector<VisionItem> items(count);
    for (VisionItem& item : items) {
        const auto modality = reader.pod<std::uint8_t>();
        if (modality != static_cast<std::uint8_t>(PromptModality::Image) &&
            modality != static_cast<std::uint8_t>(PromptModality::Video)) {
            throw std::invalid_argument("checkpoint image vision item modality is invalid");
        }
        item.modality      = static_cast<PromptModality>(modality);
        item.grid.temporal = reader.pod<std::int32_t>();
        item.grid.height   = reader.pod<std::int32_t>();
        item.grid.width    = reader.pod<std::int32_t>();
        item.patch_begin   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        item.patch_count   = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        reader.bytes(item.content_digest.data(), item.content_digest.size());
        item.timestamps  = reader.vector<double>(tokens, "vision timestamp");
        const auto spans = reader.pod<std::uint32_t>();
        if (spans > tokens) { throw std::invalid_argument("checkpoint image vision span count is out of range"); }
        item.token_spans.resize(spans);
        for (TokenSpan& span : item.token_spans) {
            span.begin = static_cast<std::size_t>(reader.pod<std::uint64_t>());
            span.count = static_cast<std::size_t>(reader.pod<std::uint64_t>());
        }
    }
    return items;
}

} // namespace

std::uint64_t image_checksum(std::span<const std::uint8_t> bytes) {
    std::uint64_t hash      = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(bytes.size());
    const std::size_t words = bytes.size() / sizeof(std::uint64_t);
    for (std::size_t index = 0; index < words; ++index) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes.data() + index * sizeof(std::uint64_t), sizeof(word));
        hash = (hash ^ word) * 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32U;
    }
    for (std::size_t index = words * sizeof(std::uint64_t); index < bytes.size(); ++index) {
        hash = (hash ^ bytes[index]) * 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 29U;
    }
    return hash;
}

void write_head(ImageWriter& writer, std::uint32_t version, std::string_view binding) {
    if (binding.size() > kMaximumModelBindingBytes) {
        throw std::invalid_argument("checkpoint image model binding is too long");
    }
    writer.bytes(kImageMagic, sizeof(kImageMagic));
    writer.pod(version);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(binding.size()));
    writer.bytes(binding.data(), binding.size());
}

ImageBody read_head(std::span<const std::uint8_t> image, std::uint32_t version, std::string_view binding) {
    if (image.size() < sizeof(kImageMagic) + sizeof(std::uint64_t)) {
        throw std::invalid_argument("checkpoint image is truncated");
    }
    const auto body = image.first(image.size() - sizeof(std::uint64_t));
    ImageBody out{ImageReader(body), body, 0};
    std::memcpy(&out.checksum, image.data() + body.size(), sizeof(out.checksum));
    char magic[sizeof(kImageMagic)] = {};
    out.reader.bytes(magic, sizeof(magic));
    if (std::memcmp(magic, kImageMagic, sizeof(magic)) != 0) {
        throw std::invalid_argument("bytes are not a checkpoint image");
    }
    if (out.reader.pod<std::uint32_t>() != version) {
        throw std::invalid_argument("checkpoint image version is unsupported");
    }
    const auto binding_bytes = out.reader.pod<std::uint32_t>();
    if (binding_bytes > kMaximumModelBindingBytes) {
        throw std::invalid_argument("checkpoint image model binding is too long");
    }
    std::string stored(binding_bytes, '\0');
    out.reader.bytes(stored.data(), binding_bytes);
    if (stored != binding) { throw std::invalid_argument("checkpoint image was written for a different model"); }
    return out;
}

void verify_checksum(const ImageBody& image) {
    if (image_checksum(image.body) != image.checksum) {
        throw std::invalid_argument("checkpoint image is corrupt: checksum mismatch");
    }
}

void seal(std::vector<std::uint8_t>& bytes, std::size_t checksum_offset) {
    const std::uint64_t checksum = image_checksum(std::span<const std::uint8_t>(bytes.data(), checksum_offset));
    std::memcpy(bytes.data() + checksum_offset, &checksum, sizeof(checksum));
}

void write_session(ImageWriter& writer, const std::optional<PreparedSessionKey>& session) {
    writer.pod<std::uint8_t>(session ? 1 : 0);
    if (session) {
        writer.pod<std::uint16_t>(session->size);
        writer.bytes(session->bytes.data(), session->size);
    }
}

std::optional<PreparedSessionKey> read_session(ImageReader& reader) {
    if (reader.pod<std::uint8_t>() == 0) { return std::nullopt; }
    PreparedSessionKey key;
    key.size = reader.pod<std::uint16_t>();
    if (key.size == 0 || key.size > key.bytes.size()) {
        throw std::invalid_argument("checkpoint image session key is out of range");
    }
    reader.bytes(key.bytes.data(), key.size);
    return key;
}

void write_identity(ImageWriter& writer, const std::vector<TokenId>& ledger, const ResidentPrefixIdentity& identity,
                    const PrefixShortlistDigests& digests) {
    writer.vector(ledger);
    writer.vector(identity.token_types());
    for (std::size_t axis = 0; axis < 3; ++axis) { writer.vector(identity.position_axis(axis)); }
    write_vision_items(writer, identity.vision_items());
    writer.vector(identity.rewrite_execution_frontiers());
    writer.vector(digests.image());
}

void read_identity(ImageReader& reader, std::uint32_t capacity, std::int64_t vocabulary, std::vector<TokenId>& ledger,
                   ResidentPrefixIdentity& identity, PrefixShortlistDigests& digests) {
    ledger = reader.vector<TokenId>(capacity, "ledger");
    for (const TokenId id : ledger) {
        if (id < 0 || id >= vocabulary) { throw std::invalid_argument("checkpoint image ledger token is out of domain"); }
    }
    const std::size_t tokens = ledger.size();
    auto token_types         = reader.vector<std::uint8_t>(tokens, "token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) { axis = reader.vector<std::int32_t>(tokens, "position"); }
    auto vision  = read_vision_items(reader, tokens);
    auto rewrite = reader.vector<std::uint32_t>(tokens, "rewrite frontier");
    if (token_types.size() != tokens) { throw std::invalid_argument("checkpoint image identity does not match its ledger"); }
    identity.restore(std::move(token_types), std::move(positions), std::move(vision), std::move(rewrite));
    auto image = reader.vector<std::array<std::uint64_t, 2>>(static_cast<std::size_t>(capacity) + 1U, "digest");
    if (image.size() != tokens + 1U) { throw std::invalid_argument("checkpoint image digests do not match their ledger"); }
    digests.restore(std::move(image));
}

} // namespace ninfer::models::qwen3_5::detail::image_codec
