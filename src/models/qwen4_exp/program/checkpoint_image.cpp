#include "models/qwen4_exp/program/checkpoint_image.h"

#include "core/paged_kv_cache.h"
#include "models/qwen3_5/program/storage/checkpoint_image_codec.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::models::qwen4_exp {
namespace {

namespace codec = qwen3_5::detail::image_codec;
using codec::ImageReader;
using codec::ImageWriter;

// Not the Qwen3.5 Program's version (2): either Program rejects the other's images at the head even
// before the binding (which already differs) is compared.
constexpr std::uint32_t kImageVersion   = 0x464E0001U; // "FN", 1
constexpr std::uint32_t kMaximumLayers  = 64;
// Slabs moved per tier call while exporting or importing (about 60 MiB of page records).
constexpr std::size_t kSlabsPerTransfer = 64;

std::uint32_t pages_for(std::uint32_t frontier) {
    return (frontier + static_cast<std::uint32_t>(kPagedKVPageSize) - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void write_config(ImageWriter& writer, const CheckpointLayout& layout) {
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(kPagedKVPageSize));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(layout.plane_bytes.size()));
    for (const auto& planes : layout.plane_bytes) {
        for (const std::size_t bytes : planes) { writer.pod<std::uint64_t>(bytes); }
    }
    writer.pod<std::uint64_t>(layout.state_bytes);
    writer.pod<std::uint64_t>(layout.residual_bytes);
    writer.pod<std::uint64_t>(layout.logits_bytes);
    writer.pod<std::uint8_t>(layout.vision ? 1 : 0);
    writer.pod<std::uint32_t>(layout.identity_tag);
}

bool config_matches(ImageReader& reader, const CheckpointLayout& layout) {
    bool same            = reader.pod<std::uint32_t>() == static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto layers    = reader.pod<std::uint32_t>();
    if (layers > kMaximumLayers) { throw std::invalid_argument("checkpoint image layer count is out of range"); }
    same = same && layers == layout.plane_bytes.size();
    for (std::uint32_t l = 0; l < layers; ++l) {
        for (std::size_t p = 0; p < 5; ++p) {
            const auto bytes = reader.pod<std::uint64_t>();
            same = same && l < layout.plane_bytes.size() && bytes == layout.plane_bytes[l][p];
        }
    }
    same = same && reader.pod<std::uint64_t>() == layout.state_bytes;
    same = same && reader.pod<std::uint64_t>() == layout.residual_bytes;
    same = same && reader.pod<std::uint64_t>() == layout.logits_bytes;
    same = same && (reader.pod<std::uint8_t>() != 0) == layout.vision;
    same = same && reader.pod<std::uint32_t>() == layout.identity_tag;
    return same;
}

struct PointRecord {
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    std::uint32_t frontier       = 0;
    std::uint32_t full_pages     = 0;
    std::uint32_t mtp_cells      = 0;
    std::optional<TokenId> next_token;
    bool logits             = false;
    std::int32_t rope_delta = 0;
    std::uint32_t identity  = 0;
    std::uint32_t image     = 0;
    std::vector<std::uint32_t> pages; // indices into the page table
    std::array<std::uint64_t, 2> digests{};
};

struct ParsedImage {
    std::optional<qwen3_5::PreparedSessionKey> session;
    std::vector<std::shared_ptr<CheckpointIdentity>> identities;
    std::vector<PointRecord> points;
    std::uint32_t page_count  = 0;
    std::uint32_t image_count = 0;
    std::span<const std::uint8_t> pages;
    std::span<const std::uint8_t> images;
};

ParsedImage parse_image(const CheckpointLayout& layout, std::span<const std::uint8_t> image, std::string_view binding) {
    auto head            = codec::read_head(image, kImageVersion, binding);
    ImageReader& reader  = head.reader;
    if (!config_matches(reader, layout)) {
        throw std::invalid_argument("checkpoint image page, state or drafter layout differs from this Program");
    }
    // Verified before anything is allocated or copied.
    codec::verify_checksum(head);

    ParsedImage parsed;
    parsed.session            = codec::read_session(reader);
    const auto identity_count = reader.pod<std::uint32_t>();
    parsed.page_count         = reader.pod<std::uint32_t>();
    parsed.image_count        = reader.pod<std::uint32_t>();
    const auto point_count    = reader.pod<std::uint32_t>();
    if (point_count == 0 || point_count > codec::kMaximumImagePoints || identity_count > point_count ||
        parsed.image_count > point_count ||
        static_cast<std::uint64_t>(parsed.page_count) > static_cast<std::uint64_t>(point_count) * pages_for(layout.capacity)) {
        throw std::invalid_argument("checkpoint image table sizes are out of range");
    }
    for (std::uint32_t index = 0; index < identity_count; ++index) {
        auto identity = std::make_shared<CheckpointIdentity>();
        codec::read_identity(reader, layout.capacity, layout.vocabulary, identity->ledger, identity->prefix_identity,
                             identity->digests);
        parsed.identities.push_back(std::move(identity));
    }
    const bool drafter = layout.residual_bytes != 0;
    for (std::uint32_t index = 0; index < point_count; ++index) {
        PointRecord point;
        const auto role = reader.pod<std::uint8_t>();
        if (role != static_cast<std::uint8_t>(runtime::CheckpointRole::Continuation) &&
            role != static_cast<std::uint8_t>(runtime::CheckpointRole::InputReplay) &&
            role != static_cast<std::uint8_t>(runtime::CheckpointRole::LongAnchor) &&
            role != static_cast<std::uint8_t>(runtime::CheckpointRole::SharedPrefix)) {
            throw std::invalid_argument("checkpoint image point role is out of range");
        }
        point.role = static_cast<runtime::CheckpointRole>(role);
        // A shared prefix travels alone, without a session: it belongs to no conversation.
        if (point.role == runtime::CheckpointRole::SharedPrefix && (point_count != 1 || parsed.session)) {
            throw std::invalid_argument("checkpoint image shares a prefix with private points");
        }
        point.frontier   = reader.pod<std::uint32_t>();
        point.full_pages = reader.pod<std::uint32_t>();
        point.mtp_cells  = reader.pod<std::uint32_t>();
        const bool next  = reader.pod<std::uint8_t>() != 0;
        const auto token = reader.pod<std::int32_t>();
        if (next) { point.next_token = token; }
        point.logits     = reader.pod<std::uint8_t>() != 0;
        point.rope_delta = reader.pod<std::int32_t>();
        point.identity   = reader.pod<std::uint32_t>();
        point.image      = reader.pod<std::uint32_t>();
        point.pages      = reader.vector<std::uint32_t>(pages_for(layout.capacity), "page");
        point.digests[0] = reader.pod<std::uint64_t>();
        point.digests[1] = reader.pod<std::uint64_t>();
        const auto tag   = reader.pod<std::uint32_t>();
        if (point.frontier == 0 || point.frontier > layout.capacity || point.identity >= identity_count ||
            point.image >= parsed.image_count || point.pages.size() != pages_for(point.frontier) ||
            point.full_pages > point.frontier / static_cast<std::uint32_t>(kPagedKVPageSize) ||
            point.mtp_cells > point.frontier || tag != layout.identity_tag ||
            (next && (token < 0 || token >= layout.vocabulary))) {
            throw std::invalid_argument("checkpoint image point is out of range");
        }
        // The drafter's cells reach the frontier or the position before it (a checkpoint's settled
        // point); without the drafter there are none.
        if (drafter ? point.mtp_cells + 1U < point.frontier : point.mtp_cells != 0) {
            throw std::invalid_argument("checkpoint image drafter position is inconsistent");
        }
        for (const std::uint32_t page : point.pages) {
            if (page >= parsed.page_count) { throw std::invalid_argument("checkpoint image page is out of range"); }
        }
        const auto& identity = *parsed.identities[point.identity];
        if (point.frontier > identity.ledger.size() || point.frontier > identity.digests.size() ||
            identity.digests.at(point.frontier) != point.digests) {
            throw std::invalid_argument("checkpoint image point does not match its identity");
        }
        parsed.points.push_back(std::move(point));
    }
    parsed.pages  = reader.payload(static_cast<std::size_t>(parsed.page_count) * layout.page_bytes());
    parsed.images = reader.payload(static_cast<std::size_t>(parsed.image_count) * layout.image_bytes());
    if (reader.remaining() != 0) { throw std::invalid_argument("checkpoint image has trailing bytes"); }
    return parsed;
}

// A page record's planes, back to back (compact), from a slab or into one.
void compact_page(const CheckpointLayout& layout, const std::byte* slab, std::uint8_t* out) {
    std::size_t at = 0;
    for (std::size_t l = 0; l < layout.plane_bytes.size(); ++l) {
        for (std::size_t p = 0; p < 5; ++p) {
            std::memcpy(out + at, slab + layout.plane_offsets[l][p], layout.plane_bytes[l][p]);
            at += layout.plane_bytes[l][p];
        }
    }
}

void expand_page(const CheckpointLayout& layout, const std::uint8_t* in, std::byte* slab) {
    std::size_t at = 0;
    for (std::size_t l = 0; l < layout.plane_bytes.size(); ++l) {
        for (std::size_t p = 0; p < 5; ++p) {
            std::memcpy(slab + layout.plane_offsets[l][p], in + at, layout.plane_bytes[l][p]);
            at += layout.plane_bytes[l][p];
        }
    }
}

} // namespace

std::size_t CheckpointLayout::page_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& planes : plane_bytes) {
        for (const std::size_t bytes : planes) { total += bytes; }
    }
    return total;
}

qwen3_5::CheckpointImage export_checkpoint_image(const HostCheckpointStore& store, CheckpointTier& tier,
                                                 const CheckpointLayout& layout,
                                                 std::span<const qwen3_5::CheckpointHandle> handles,
                                                 const std::optional<qwen3_5::PreparedSessionKey>& session,
                                                 std::string_view binding) {
    std::vector<qwen3_5::CheckpointHandle> points;
    for (const auto handle : handles) {
        if (std::find(points.begin(), points.end(), handle) == points.end()) { points.push_back(handle); }
    }
    if (points.empty() || points.size() > codec::kMaximumImagePoints) {
        throw std::invalid_argument("checkpoint export needs between one and 64 points");
    }
    for (const auto handle : points) {
        if (!store.valid(handle)) { throw std::invalid_argument("checkpoint is not an exportable recovery point"); }
        // A shared prefix is exported on its own: an image is one conversation or one prefix.
        if (store.at(handle).role == runtime::CheckpointRole::SharedPrefix && (points.size() != 1 || session)) {
            throw std::invalid_argument("a shared prefix is exported alone, without a session");
        }
    }
    // Deepest first: the first key is the image's endpoint.
    std::stable_sort(points.begin(), points.end(),
                     [&](auto a, auto b) { return store.at(a).frontier > store.at(b).frontier; });

    // The tables, each entry once, in first-use order.
    std::vector<const CheckpointIdentity*> identities;
    std::vector<std::uint32_t> page_slabs;
    std::unordered_map<std::uint32_t, std::uint32_t> page_index;
    std::vector<const HostCheckpoint*> images; // the first record naming each state image
    std::vector<PointRecord> records;
    for (const auto handle : points) {
        const HostCheckpoint& r = store.at(handle);
        if (r.image.size() != layout.image_slabs) { throw std::logic_error("context cache: a checkpoint's image size differs"); }
        PointRecord point;
        point.role       = r.role;
        point.frontier   = r.frontier;
        point.full_pages = r.full_pages;
        point.mtp_cells  = r.mtp_cells;
        point.next_token = r.next_token;
        point.logits     = r.logits;
        point.rope_delta = r.rope_delta;
        point.digests    = r.identity->digests.at(r.frontier);
        const auto id    = std::find(identities.begin(), identities.end(), r.identity.get());
        point.identity   = static_cast<std::uint32_t>(id - identities.begin());
        if (id == identities.end()) { identities.push_back(r.identity.get()); }
        const auto im = std::find_if(images.begin(), images.end(),
                                     [&](const HostCheckpoint* other) { return other->image.front() == r.image.front(); });
        point.image   = static_cast<std::uint32_t>(im - images.begin());
        if (im == images.end()) { images.push_back(&r); }
        for (const std::uint32_t slab : r.pages) {
            const auto [found, added] = page_index.try_emplace(slab, static_cast<std::uint32_t>(page_slabs.size()));
            if (added) { page_slabs.push_back(slab); }
            point.pages.push_back(found->second);
        }
        records.push_back(std::move(point));
    }

    qwen3_5::CheckpointImage image;
    ImageWriter writer(image.bytes);
    codec::write_head(writer, kImageVersion, binding);
    write_config(writer, layout);
    codec::write_session(writer, session);
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(identities.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(page_slabs.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(images.size()));
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(records.size()));
    for (const CheckpointIdentity* identity : identities) {
        codec::write_identity(writer, identity->ledger, identity->prefix_identity, identity->digests);
    }
    for (const PointRecord& point : records) {
        writer.pod<std::uint8_t>(static_cast<std::uint8_t>(point.role));
        writer.pod(point.frontier);
        writer.pod(point.full_pages);
        writer.pod(point.mtp_cells);
        writer.pod<std::uint8_t>(point.next_token ? 1 : 0);
        writer.pod<std::int32_t>(point.next_token.value_or(0));
        writer.pod<std::uint8_t>(point.logits ? 1 : 0);
        writer.pod(point.rope_delta);
        writer.pod(point.identity);
        writer.pod(point.image);
        writer.vector(point.pages);
        writer.pod(point.digests[0]);
        writer.pod(point.digests[1]);
        writer.pod(layout.identity_tag);
    }
    const std::size_t page_bytes   = layout.page_bytes();
    const std::size_t pages_offset = writer.reserve(page_slabs.size() * page_bytes);
    const std::size_t image_offset = writer.reserve(images.size() * layout.image_bytes());
    const std::size_t checksum_at  = writer.reserve(sizeof(std::uint64_t));
    if (!page_slabs.empty()) { image.regions.push_back({pages_offset, page_slabs.size() * page_bytes}); }
    image.regions.push_back({image_offset, images.size() * layout.image_bytes()});

    const std::size_t slab_bytes = layout.slab_bytes;
    std::vector<std::byte> buffer(std::max<std::size_t>(kSlabsPerTransfer, layout.image_slabs) * slab_bytes);
    for (std::size_t at = 0; at < page_slabs.size(); at += kSlabsPerTransfer) {
        const std::size_t n = std::min(kSlabsPerTransfer, page_slabs.size() - at);
        tier.read(std::span<const std::uint32_t>(page_slabs).subspan(at, n), std::span<std::byte>(buffer).first(n * slab_bytes));
        for (std::size_t k = 0; k < n; ++k) {
            compact_page(layout, buffer.data() + k * slab_bytes, image.bytes.data() + pages_offset + (at + k) * page_bytes);
        }
    }
    for (std::size_t i = 0; i < images.size(); ++i) {
        const HostCheckpoint& r = *images[i];
        tier.read(r.image, std::span<std::byte>(buffer).first(r.image.size() * slab_bytes));
        std::uint8_t* out = image.bytes.data() + image_offset + i * layout.image_bytes();
        std::memcpy(out, buffer.data(), layout.state_bytes);
        if (layout.residual_bytes != 0) {
            std::memcpy(out + layout.state_bytes, buffer.data() + layout.residual_offset, layout.residual_bytes);
        }
        // Logits that were never captured are stored as zeros, so the bytes do not depend on stale slabs.
        std::uint8_t* logits = out + layout.state_bytes + layout.residual_bytes;
        if (r.logits) {
            std::memcpy(logits, buffer.data() + layout.logits_offset, layout.logits_bytes);
        } else {
            std::memset(logits, 0, layout.logits_bytes);
        }
    }
    codec::seal(image.bytes, checksum_at);

    for (const PointRecord& point : records) {
        image.keys.push_back(qwen3_5::PrefixShortlistKey{
            .digests = point.digests, .frontier = point.frontier, .identity_tag = layout.identity_tag});
    }
    image.tokens        = records.front().frontier;
    const auto& digests = identities[records.front().identity]->digests.image();
    image.prefix_digests.assign(digests.begin(), digests.begin() + static_cast<std::ptrdiff_t>(image.tokens) + 1);
    return image;
}

std::size_t checkpoint_image_slab_bytes(const CheckpointLayout& layout, std::span<const std::uint8_t> image,
                                        std::string_view binding) {
    const ParsedImage parsed = parse_image(layout, image, binding);
    return (static_cast<std::size_t>(parsed.page_count) + static_cast<std::size_t>(parsed.image_count) * layout.image_slabs) *
           layout.slab_bytes;
}

std::optional<qwen3_5::ImportedCheckpoints> import_checkpoint_image(HostCheckpointStore& store, CheckpointTier& tier,
                                                                    const CheckpointLayout& layout,
                                                                    std::span<const std::uint8_t> image,
                                                                    std::string_view binding) {
    const ParsedImage parsed  = parse_image(layout, image, binding);
    const std::uint32_t needed = parsed.page_count + parsed.image_count * layout.image_slabs;
    auto slabs                = store.allocate(needed);
    if (!slabs) { return std::nullopt; }
    // The import holds one reference on each slab until the records hold theirs.
    struct Hold {
        HostCheckpointStore& store;
        std::vector<std::uint32_t>& slabs;
        ~Hold() { store.release(slabs); }
    } hold{store, *slabs};
    const std::span<const std::uint32_t> pages(slabs->data(), parsed.page_count);
    const auto image_slabs = [&](std::uint32_t index) {
        return std::span<const std::uint32_t>(*slabs).subspan(parsed.page_count + index * layout.image_slabs, layout.image_slabs);
    };
    const std::size_t slab_bytes = layout.slab_bytes;
    const std::size_t page_bytes = layout.page_bytes();
    try {
        std::vector<std::byte> buffer(std::max<std::size_t>(kSlabsPerTransfer, layout.image_slabs) * slab_bytes);
        for (std::size_t at = 0; at < pages.size(); at += kSlabsPerTransfer) {
            const std::size_t n = std::min(kSlabsPerTransfer, pages.size() - at);
            std::fill(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(n * slab_bytes), std::byte{0});
            for (std::size_t k = 0; k < n; ++k) {
                expand_page(layout, parsed.pages.data() + (at + k) * page_bytes, buffer.data() + k * slab_bytes);
            }
            tier.write(pages.subspan(at, n), std::span<const std::byte>(buffer).first(n * slab_bytes));
        }
        for (std::uint32_t i = 0; i < parsed.image_count; ++i) {
            const std::uint8_t* in = parsed.images.data() + static_cast<std::size_t>(i) * layout.image_bytes();
            const std::size_t bytes = layout.image_slabs * slab_bytes;
            std::fill(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(bytes), std::byte{0});
            std::memcpy(buffer.data(), in, layout.state_bytes);
            if (layout.residual_bytes != 0) {
                std::memcpy(buffer.data() + layout.residual_offset, in + layout.state_bytes, layout.residual_bytes);
            }
            std::memcpy(buffer.data() + layout.logits_offset, in + layout.state_bytes + layout.residual_bytes,
                        layout.logits_bytes);
            tier.write(image_slabs(i), std::span<const std::byte>(buffer).first(bytes));
        }
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::exception&) {
        return std::nullopt; // the tier could not store the bytes (a full disk): a miss
    }

    qwen3_5::ImportedCheckpoints out;
    out.session = parsed.session;
    // Reserved up front so publishing a record can never fail on the vector's growth afterwards.
    out.points.reserve(parsed.points.size());
    try {
        for (const PointRecord& point : parsed.points) {
            HostCheckpoint r;
            r.identity     = parsed.identities[point.identity];
            r.identity_tag = layout.identity_tag;
            r.frontier     = point.frontier;
            r.role         = point.role;
            r.full_pages   = point.full_pages;
            r.mtp_cells    = point.mtp_cells;
            r.next_token   = point.next_token;
            r.logits       = point.logits;
            r.rope_delta   = point.rope_delta;
            for (const std::uint32_t page : point.pages) { r.pages.push_back(pages[page]); }
            const auto own = image_slabs(point.image);
            r.image.assign(own.begin(), own.end());
            for (const std::uint32_t slab : r.pages) { store.retain(slab); }
            for (const std::uint32_t slab : r.image) { store.retain(slab); }
            try {
                out.points.push_back(store.insert(std::move(r)));
            } catch (...) {
                for (const std::uint32_t slab : r.pages) { store.release(slab); }
                for (const std::uint32_t slab : r.image) { store.release(slab); }
                throw;
            }
        }
    } catch (const std::logic_error& error) {
        for (const auto handle : out.points) { (void)store.erase(handle); }
        throw std::invalid_argument(std::string("checkpoint image is inconsistent: ") + error.what());
    } catch (...) {
        // An allocation failure while inserting: no earlier point stays resident.
        for (const auto handle : out.points) { (void)store.erase(handle); }
        throw;
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
