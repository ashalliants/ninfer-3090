#include "models/qwen4_exp/program/context_cache.h"

#include "core/host_gather.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen4_exp {
namespace {
// The gather's per-block piece: long copies are split so its blocks share them.
constexpr std::size_t kGatherPiece = 64 * 1024;
// Host tier and descriptors are read through cudaHostGetDevicePointer by the gather kernel, so they
// are allocated mapped (and portable), like PinnedHostBuffer in core/arena.cu.
constexpr unsigned int kMappedPinned = cudaHostAllocPortable | cudaHostAllocMapped;
} // namespace

HostTier::HostTier(std::uint32_t slabs, std::size_t slab_bytes, std::size_t descriptors)
    : slabs_(slabs), slab_bytes_(slab_bytes) {
    const std::size_t bytes = static_cast<std::size_t>(slabs) * slab_bytes;
    if (cudaHostAlloc(&memory_, bytes, kMappedPinned) != cudaSuccess) {
        (void)cudaGetLastError();
        memory_ = nullptr;
        throw std::runtime_error("cudaHostAlloc of " + std::to_string(bytes >> 20) + " MiB failed");
    }
    void* mapped = nullptr;
    if (cudaHostGetDevicePointer(&mapped, memory_, 0) != cudaSuccess || mapped != memory_) {
        (void)cudaGetLastError();
        cudaFreeHost(memory_);
        memory_ = nullptr;
        throw std::runtime_error("pinned host memory is not identity-mapped on this device");
    }
    descriptor_bytes_ = std::max<std::size_t>(descriptors, 64) * sizeof(GatherCopy);
    if (cudaHostAlloc(&descriptors_, descriptor_bytes_, kMappedPinned) != cudaSuccess) {
        (void)cudaGetLastError();
        cudaFreeHost(memory_);
        memory_      = nullptr;
        descriptors_ = nullptr;
        throw std::runtime_error("cudaHostAlloc of the gather descriptors failed");
    }
}

HostTier::~HostTier() {
    if (descriptors_ != nullptr) { cudaFreeHost(descriptors_); }
    if (memory_ != nullptr) { cudaFreeHost(memory_); }
}

void HostTier::copy(std::span<const SlabCopy> copies, bool to_tier, cudaStream_t stream) {
    std::vector<GatherCopy> gather;
    gather.reserve(copies.size());
    for (const SlabCopy& c : copies) {
        if (c.slab >= slabs_ || c.offset + c.width > slab_bytes_ ||
            c.offset + (c.rows - 1) * c.slab_pitch + c.width > (slabs_ - c.slab) * slab_bytes_) {
            throw std::logic_error("context cache: a copy runs outside the Host tier");
        }
        std::byte* host = static_cast<std::byte*>(memory_) + static_cast<std::size_t>(c.slab) * slab_bytes_ + c.offset;
        GatherCopy g;
        g.width  = c.width;
        g.height = c.rows;
        if (to_tier) {
            g.destination       = host;
            g.destination_pitch = c.slab_pitch;
            g.source            = c.device;
            g.source_pitch      = c.device_pitch;
        } else {
            g.destination       = c.device;
            g.destination_pitch = c.device_pitch;
            g.source            = host;
            g.source_pitch      = c.slab_pitch;
        }
        gather.push_back(g);
    }
    split_gather_copies(gather, kGatherPiece);
    if (gather.empty()) { return; }
    const std::size_t bytes = gather.size() * sizeof(GatherCopy);
    if (bytes > descriptor_bytes_) {
        // The old descriptors may still be read by an earlier gather.
        if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("context cache: stream failed"); }
        void* grown = nullptr;
        if (cudaHostAlloc(&grown, bytes + bytes / 2, kMappedPinned) != cudaSuccess) {
            (void)cudaGetLastError();
            throw std::runtime_error("cudaHostAlloc of the gather descriptors failed");
        }
        cudaFreeHost(descriptors_);
        descriptors_      = grown;
        descriptor_bytes_ = bytes + bytes / 2;
    }
    std::memcpy(descriptors_, gather.data(), bytes);
    gather_copies(std::span<const GatherCopy>(static_cast<const GatherCopy*>(descriptors_), gather.size()), stream);
}

std::vector<std::uint32_t> prefill_splits(std::span<const std::uint32_t> capture_frontiers,
                                          std::span<const std::uint32_t> execution_frontiers,
                                          std::uint32_t prompt_tokens) {
    std::vector<std::uint32_t> out;
    out.reserve(capture_frontiers.size() + execution_frontiers.size());
    for (const auto* list : {&capture_frontiers, &execution_frontiers}) {
        for (const std::uint32_t frontier : *list) {
            if (frontier != 0 && frontier < prompt_tokens) { out.push_back(frontier); }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

runtime::PrefillWork prefill_work_between(std::uint32_t begin, std::uint32_t end, std::uint32_t chunk,
                                          std::span<const MediaCost> media) {
    std::uint64_t items = 0, patches = 0;
    if (end > begin) {
        for (const MediaCost& item : media) {
            if (std::any_of(item.spans.begin(), item.spans.end(), [&](const auto& span) {
                    return static_cast<std::uint64_t>(span.first) + span.second > begin && span.first < end;
                })) {
                ++items;
                patches += item.patches;
            }
        }
    }
    return runtime::make_prefill_work(begin, end > begin ? end - begin : 0, items, patches, chunk);
}

SlabRefs::~SlabRefs() {
    if (store_ != nullptr) { store_->release(slabs_); }
}

SlabRefs& SlabRefs::operator=(SlabRefs&& other) noexcept {
    if (this != &other) {
        if (store_ != nullptr) { store_->release(slabs_); }
        store_ = other.store_;
        slabs_ = std::move(other.slabs_);
        other.slabs_.clear();
    }
    return *this;
}

std::vector<std::uint32_t> SlabRefs::take() noexcept {
    std::vector<std::uint32_t> out = std::move(slabs_);
    slabs_.clear();
    return out;
}

HostCheckpointStore::HostCheckpointStore(std::uint32_t slabs, std::size_t slab_bytes)
    : slab_bytes_(slab_bytes), references_(slabs, 0) {
    if (slab_bytes == 0) { throw std::invalid_argument("context cache slabs must have a size"); }
    free_.reserve(slabs);
    // Lowest index on top, so allocation order is ascending (Host addresses grow with it).
    for (std::uint32_t s = slabs; s-- > 0;) { free_.push_back(s); }
}

std::optional<std::vector<std::uint32_t>> HostCheckpointStore::allocate(std::uint32_t count) {
    if (count > free_.size()) { return std::nullopt; }
    std::vector<std::uint32_t> out(free_.end() - count, free_.end());
    free_.resize(free_.size() - count);
    std::reverse(out.begin(), out.end());
    for (const std::uint32_t slab : out) { references_[slab] = 1; }
    peak_ = std::max(peak_, used_slabs());
    return out;
}

void HostCheckpointStore::retain(std::uint32_t slab) {
    if (references_.at(slab) == 0) { throw std::logic_error("context cache: retaining a free slab"); }
    ++references_[slab];
}

void HostCheckpointStore::release(std::uint32_t slab) noexcept {
    if (slab >= references_.size() || references_[slab] == 0) { return; }
    if (--references_[slab] == 0) { free_.push_back(slab); }
}

void HostCheckpointStore::release(std::span<const std::uint32_t> slabs) noexcept {
    for (const std::uint32_t slab : slabs) { release(slab); }
}

HostCheckpointStore::Handle HostCheckpointStore::insert(HostCheckpoint&& record) {
    if (!record.identity || record.frontier == 0 || record.frontier > record.identity->ledger.size() ||
        record.frontier > record.identity->digests.size() ||
        record.pages.size() != (record.frontier + kPagedKVPageSize - 1U) / kPagedKVPageSize) {
        throw std::logic_error("context cache: a checkpoint does not cover its frontier");
    }
    for (const std::uint32_t slab : record.pages) {
        if (references_.at(slab) == 0) { throw std::logic_error("context cache: a checkpoint page is free"); }
    }
    for (const std::uint32_t slab : record.image) {
        if (references_.at(slab) == 0) { throw std::logic_error("context cache: a checkpoint image slab is free"); }
    }
    if (fail_after_) {
        if (*fail_after_ == 0) {
            fail_after_.reset();
            throw std::bad_alloc();
        }
        --*fail_after_;
    }
    std::uint32_t index = 0;
    while (index < slots_.size() && slots_[index].value) { ++index; }
    if (index == slots_.size()) { slots_.emplace_back(); }
    auto& slot = slots_[index];
    slot.value = std::move(record);
    slot.pins  = 0;
    return Handle{this, index, slot.generation};
}

std::vector<HostCheckpointStore::Handle>
HostCheckpointStore::insert_group(HostCheckpoint&& record, std::span<const runtime::CheckpointRole> roles) {
    // The record's own references, released whatever happens: each published copy takes its own.
    SlabRefs pages(*this, std::move(record.pages));
    SlabRefs image(*this, std::move(record.image));
    record.pages.clear();
    record.image.clear();
    if (roles.empty()) { throw std::logic_error("context cache: a capture group has no role"); }
    std::vector<Handle> out;
    try {
        out.reserve(roles.size());
        for (const runtime::CheckpointRole role : roles) {
            HostCheckpoint copy = record;
            copy.role           = role;
            copy.pages          = pages.slabs();
            copy.image          = image.slabs();
            SlabRefs held(*this);
            held.reserve(copy.pages.size() + copy.image.size());
            for (const auto* list : {&copy.pages, &copy.image}) {
                for (const std::uint32_t slab : *list) {
                    retain(slab);
                    held.adopt(slab);
                }
            }
            out.push_back(insert(std::move(copy))); // reserved: the push cannot throw
            (void)held.take();                       // the inserted record owns them now
        }
    } catch (...) {
        for (const Handle handle : out) { (void)erase(handle); }
        throw;
    }
    return out;
}

HostCheckpointStore::Handle HostCheckpointStore::clone(Handle handle) {
    HostCheckpoint copy = at(handle);
    SlabRefs held(*this);
    held.reserve(copy.pages.size() + copy.image.size());
    for (const auto* list : {&copy.pages, &copy.image}) {
        for (const std::uint32_t slab : *list) {
            retain(slab);
            held.adopt(slab);
        }
    }
    const Handle out = insert(std::move(copy));
    (void)held.take(); // the inserted record owns them now
    return out;
}

bool HostCheckpointStore::valid(Handle handle) const noexcept {
    return handle.owner == this && handle.index < slots_.size() && slots_[handle.index].value &&
           slots_[handle.index].generation == handle.generation;
}

const HostCheckpoint& HostCheckpointStore::at(Handle handle) const {
    if (!valid(handle)) { throw std::logic_error("context cache: stale checkpoint handle"); }
    return *slots_[handle.index].value;
}

bool HostCheckpointStore::erase(Handle handle) noexcept {
    if (!valid(handle) || slots_[handle.index].pins != 0) { return false; }
    auto& slot = slots_[handle.index];
    release(slot.value->pages);
    release(slot.value->image);
    slot.value.reset();
    if (++slot.generation == 0) { slot.generation = 1; }
    return true;
}

void HostCheckpointStore::pin(Handle handle) {
    if (!valid(handle)) { throw std::logic_error("context cache: pinning a stale checkpoint"); }
    ++slots_[handle.index].pins;
}

void HostCheckpointStore::unpin(Handle handle) noexcept {
    if (valid(handle) && slots_[handle.index].pins != 0) { --slots_[handle.index].pins; }
}

bool HostCheckpointStore::pinned(Handle handle) const noexcept {
    return valid(handle) && slots_[handle.index].pins != 0;
}

std::vector<HostCheckpointStore::Handle> HostCheckpointStore::handles() const {
    std::vector<Handle> out;
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].value) { out.push_back(Handle{this, i, slots_[i].generation}); }
    }
    return out;
}

std::size_t HostCheckpointStore::size() const noexcept {
    return static_cast<std::size_t>(std::count_if(slots_.begin(), slots_.end(), [](const Slot& s) { return s.value.has_value(); }));
}

std::size_t HostCheckpointStore::footprint_bytes(std::span<const Handle> handles) const {
    std::unordered_set<std::uint32_t> slabs;
    for (const Handle handle : handles) {
        const auto& record = at(handle);
        slabs.insert(record.pages.begin(), record.pages.end());
        slabs.insert(record.image.begin(), record.image.end());
    }
    return slabs.size() * slab_bytes_;
}

std::size_t HostCheckpointStore::released_bytes(std::span<const Handle> handles) const {
    std::unordered_map<std::uint32_t, std::uint32_t> held;
    std::unordered_set<std::uint32_t> records;
    for (const Handle handle : handles) {
        if (!valid(handle) || slots_[handle.index].pins != 0) { return 0; }
        if (!records.insert(handle.index).second) { continue; }
        const auto& record = *slots_[handle.index].value;
        for (const std::uint32_t slab : record.pages) { ++held[slab]; }
        for (const std::uint32_t slab : record.image) { ++held[slab]; }
    }
    std::size_t slabs = 0;
    for (const auto& [slab, count] : held) { slabs += references_[slab] == count ? 1U : 0U; }
    return slabs * slab_bytes_;
}

std::vector<std::vector<HostCheckpointStore::Handle>>
HostCheckpointStore::release_sets(std::span<const Handle> eligible) const {
    std::vector<bool> allowed(slots_.size(), false);
    for (const Handle handle : eligible) {
        if (valid(handle) && slots_[handle.index].pins == 0) { allowed[handle.index] = true; }
    }
    std::vector<std::vector<std::uint32_t>> sets;
    const auto add = [&](std::vector<std::uint32_t> set) {
        if (std::find(sets.begin(), sets.end(), set) == sets.end()) { sets.push_back(std::move(set)); }
    };
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        if (allowed[index]) { add({index}); }
    }
    // Every holder of each slab, in slot order.
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> holders;
    for (std::uint32_t index = 0; index < slots_.size(); ++index) {
        if (!slots_[index].value) { continue; }
        const auto& record = *slots_[index].value;
        for (const auto* list : {&record.pages, &record.image}) {
            for (const std::uint32_t slab : *list) {
                auto& entry = holders[slab];
                if (entry.empty() || entry.back() != index) { entry.push_back(index); }
            }
        }
    }
    for (auto& [slab, set] : holders) {
        if (set.size() < 2) { continue; }
        // A slab some other holder references (the lane, or one record listing it twice) is not
        // freed by releasing these records; such sets are still quoted through released_bytes.
        if (!std::all_of(set.begin(), set.end(), [&](std::uint32_t index) { return allowed[index]; })) { continue; }
        add(set);
    }
    std::vector<std::vector<Handle>> out;
    out.reserve(sets.size());
    for (const auto& set : sets) {
        std::vector<Handle> handles;
        for (const std::uint32_t index : set) { handles.push_back(Handle{this, index, slots_[index].generation}); }
        out.push_back(std::move(handles));
    }
    return out;
}

bool HostCheckpointStore::prefix(Handle earlier, Handle later) const {
    const auto& a = at(earlier);
    const auto& b = at(later);
    if (a.frontier > b.frontier || a.identity_tag != b.identity_tag ||
        a.identity->digests.at(a.frontier) != b.identity->digests.at(a.frontier)) {
        return false;
    }
    if (a.identity == b.identity) { return true; }
    const auto count = static_cast<std::ptrdiff_t>(a.frontier);
    return std::equal(a.identity->ledger.begin(), a.identity->ledger.begin() + count, b.identity->ledger.begin()) &&
           a.identity->prefix_identity.prefix_equals(b.identity->prefix_identity, a.frontier);
}

std::uint64_t HostCheckpointStore::recovery_loss(std::span<const Handle> removed,
                                                 std::span<const Handle> surviving) const {
    std::vector<Handle> ordered;
    for (const Handle handle : removed) {
        if (valid(handle) && std::find(ordered.begin(), ordered.end(), handle) == ordered.end()) {
            ordered.push_back(handle);
        }
    }
    std::sort(ordered.begin(), ordered.end(), [&](Handle left, Handle right) {
        const auto l = at(left).frontier, r = at(right).frontier;
        return l != r ? l > r : left.index < right.index;
    });
    struct Lineage {
        Handle deepest;
        std::uint32_t tokens;
    };
    std::vector<Lineage> lineages;
    for (const Handle handle : ordered) {
        std::uint32_t fallback = 0;
        for (const Handle retained : surviving) {
            if (!valid(retained) || std::find(ordered.begin(), ordered.end(), retained) != ordered.end()) {
                continue;
            }
            const auto frontier = at(retained).frontier;
            if (frontier > fallback && prefix(retained, handle)) { fallback = frontier; }
        }
        const std::uint32_t lost = at(handle).frontier - fallback;
        const auto lineage = std::find_if(lineages.begin(), lineages.end(),
                                          [&](const Lineage& item) { return prefix(handle, item.deepest); });
        if (lineage == lineages.end()) {
            lineages.push_back({handle, lost});
        } else {
            lineage->tokens = std::max(lineage->tokens, lost);
        }
    }
    std::uint64_t total = 0;
    for (const Lineage& lineage : lineages) { total += lineage.tokens; }
    return total;
}

} // namespace ninfer::models::qwen4_exp
