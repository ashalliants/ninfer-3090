#pragma once

// The Host tier of the Qwen3.8-Flash-Next context cache (docs/maintainer/flash-next-context-cache.md):
// a pool of fixed-size slabs, the checkpoints held in it and the KV pages they share
// (HostCheckpointStore, bookkeeping only), and the tier that holds the slabs' bytes and moves them
// (CheckpointTier; HostTier is pinned RAM).
//
// A slab holds one KV page record (every attention layer's planes of one 64-token page) or one part
// of a state image. KV pages are reference counted: checkpoints of one history and the lane that
// produced them hold the same full pages, so a page is freed when its last holder lets go. A state
// image belongs to exactly one checkpoint.

#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

// What a checkpoint's tokens are: exact tokens, three-axis positions and media, and the rolling
// shortlist digests the Engine's prefix index walks. Shared by every checkpoint of one prompt.
struct CheckpointIdentity {
    std::vector<TokenId> ledger; // at least as long as the deepest checkpoint using it
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_5::detail::PrefixShortlistDigests digests;
};

// One checkpoint: the lane's execution position after tokens [0, frontier).
struct HostCheckpoint {
    std::shared_ptr<const CheckpointIdentity> identity;
    std::uint32_t identity_tag = 0;
    std::uint32_t frontier     = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    // One slab per KV page [0, ceil(frontier / 64)). The first `full_pages` held every layer's 64
    // final positions when captured and may be shared; the rest are this checkpoint's own.
    std::vector<std::uint32_t> pages;
    std::uint32_t full_pages = 0;
    // The state image (recurrent state, the drafter's saved residual, and the logits of position
    // frontier - 1 when `logits`), in slab order.
    std::vector<std::uint32_t> image;
    // The MTP drafter: cells [0, mtp_cells) are written; next_token is the token at `frontier`
    // that cell frontier - 1 encodes, when that cell is written.
    std::uint32_t mtp_cells = 0;
    std::optional<TokenId> next_token;
    bool logits             = false;
    std::int32_t rope_delta = 0;
};

// One copy between a device region and checkpoint slab bytes: `rows` rows of `width` bytes, row r at
// device + r * device_pitch and at byte offset + r * slab_pitch of `slab` (a pitch may run past the
// slab into the next slab index only when slab_pitch is a whole number of slabs: a run of
// consecutive slabs, one row each).
struct SlabCopy {
    std::uint32_t slab        = 0;
    std::size_t offset        = 0;
    std::size_t slab_pitch    = 0;
    std::byte* device         = nullptr;
    std::size_t device_pitch  = 0;
    std::size_t width         = 0;
    std::size_t rows          = 0;
};

// Where checkpoint bytes live. Capture and restore describe their copies in slab terms only, so the
// placement (pinned RAM here; an NVMe file or a remote object later) is not part of them. A tier
// moves the bytes in stream order; the caller synchronizes before reusing the slabs.
class CheckpointTier {
public:
    virtual ~CheckpointTier() = default;
    [[nodiscard]] virtual std::uint32_t slabs() const noexcept    = 0;
    [[nodiscard]] virtual std::size_t slab_bytes() const noexcept = 0;
    virtual void copy(std::span<const SlabCopy> copies, bool to_tier, cudaStream_t stream) = 0;
};

// The pinned host RAM tier: one identity-mapped pinned allocation, copied by the gather kernel
// (core/host_gather.h) so the single WDDM copy engine stays free for the rounds' inputs. Throws
// when the memory cannot be pinned or is not identity-mapped.
class HostTier final : public CheckpointTier {
public:
    HostTier(std::uint32_t slabs, std::size_t slab_bytes, std::size_t descriptors);
    ~HostTier() override;
    HostTier(const HostTier&)            = delete;
    HostTier& operator=(const HostTier&) = delete;
    [[nodiscard]] std::uint32_t slabs() const noexcept override { return slabs_; }
    [[nodiscard]] std::size_t slab_bytes() const noexcept override { return slab_bytes_; }
    void copy(std::span<const SlabCopy> copies, bool to_tier, cudaStream_t stream) override;

private:
    std::uint32_t slabs_;
    std::size_t slab_bytes_;
    void* memory_ = nullptr;
    void* descriptors_ = nullptr;
    std::size_t descriptor_bytes_ = 0;
};

class HostCheckpointStore {
public:
    using Handle = qwen3_5::CheckpointHandle;

    HostCheckpointStore(std::uint32_t slabs, std::size_t slab_bytes);
    HostCheckpointStore(const HostCheckpointStore&)            = delete;
    HostCheckpointStore& operator=(const HostCheckpointStore&) = delete;

    [[nodiscard]] std::size_t slab_bytes() const noexcept { return slab_bytes_; }
    [[nodiscard]] std::uint32_t capacity_slabs() const noexcept { return static_cast<std::uint32_t>(references_.size()); }
    [[nodiscard]] std::uint32_t used_slabs() const noexcept { return capacity_slabs() - static_cast<std::uint32_t>(free_.size()); }
    [[nodiscard]] std::uint32_t free_slabs() const noexcept { return static_cast<std::uint32_t>(free_.size()); }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return slab_bytes_ * capacity_slabs(); }
    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return slab_bytes_ * used_slabs(); }
    [[nodiscard]] std::size_t peak_bytes() const noexcept { return slab_bytes_ * peak_; }
    [[nodiscard]] std::uint32_t slabs_for(std::size_t bytes) const noexcept {
        return static_cast<std::uint32_t>((bytes + slab_bytes_ - 1) / slab_bytes_);
    }

    // `count` free slabs, each holding one reference, or nullopt (nothing taken) when fewer are free.
    [[nodiscard]] std::optional<std::vector<std::uint32_t>> allocate(std::uint32_t count);
    void retain(std::uint32_t slab);
    // Drops one reference; the slab is free again at zero.
    void release(std::uint32_t slab) noexcept;
    void release(std::span<const std::uint32_t> slabs) noexcept;
    [[nodiscard]] std::uint32_t references(std::uint32_t slab) const { return references_.at(slab); }

    // Takes over one reference to each of the record's page and image slabs.
    [[nodiscard]] Handle insert(HostCheckpoint&& record);
    [[nodiscard]] bool valid(Handle handle) const noexcept;
    [[nodiscard]] const HostCheckpoint& at(Handle handle) const;
    // Releases the record's references. False (nothing changed) for a stale or pinned handle.
    bool erase(Handle handle) noexcept;
    // A reader lease: a pinned checkpoint cannot be erased (an in-flight restore reads it).
    void pin(Handle handle);
    void unpin(Handle handle) noexcept;
    [[nodiscard]] bool pinned(Handle handle) const noexcept;
    [[nodiscard]] std::vector<Handle> handles() const;
    [[nodiscard]] std::size_t size() const noexcept;

    // Host bytes of the union of the checkpoints' slabs (a shared page counts once).
    [[nodiscard]] std::size_t footprint_bytes(std::span<const Handle> handles) const;
    // Host bytes freed by erasing exactly this set: slabs no holder outside it references. Zero when
    // any member is stale or pinned (the set cannot be erased as a whole).
    [[nodiscard]] std::size_t released_bytes(std::span<const Handle> handles) const;
    // The finite release actions over `eligible` (the Qwen3.5 Program's physical owner sets): every
    // eligible checkpoint alone, and for each slab whose only holders are eligible checkpoints, the
    // set of those holders (shared pages are freed only when all of them go). Unpinned, valid and
    // deduplicated; a set's members are in ascending slot order.
    [[nodiscard]] std::vector<std::vector<Handle>> release_sets(std::span<const Handle> eligible) const;
    // Whether `earlier`'s tokens are an exact prefix of `later`'s (tokens, positions and media).
    [[nodiscard]] bool prefix(Handle earlier, Handle later) const;
    // Recovery tokens lost by erasing `removed` while `surviving` stay: per lineage of removed
    // checkpoints, the deepest one's frontier above the deepest surviving checkpoint that is its
    // prefix (the Qwen3.5 Program's arithmetic).
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const Handle> removed,
                                              std::span<const Handle> surviving) const;

private:
    struct Slot {
        std::optional<HostCheckpoint> value;
        std::uint64_t generation = 1;
        std::uint32_t pins       = 0;
    };

    std::size_t slab_bytes_;
    std::vector<std::uint32_t> references_;
    std::vector<std::uint32_t> free_;
    std::uint32_t peak_ = 0;
    std::vector<Slot> slots_;
};

} // namespace ninfer::models::qwen4_exp
