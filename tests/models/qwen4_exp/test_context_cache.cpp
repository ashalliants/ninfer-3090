// The Qwen3.8-Flash-Next context cache's Host tier (docs/maintainer/flash-next-context-cache.md),
// host only: the slab pool, checkpoints sharing KV pages by reference, the bytes a release frees
// (shared pages count only when every holder goes, and a page the lane still mirrors is never
// counted), pins, the release sets the ResourceManager ranks, the lineage arithmetic of recovery
// loss, and the RAM ledger's sizing of the tier beside the experts. The context store's checkpoint
// images over an in-memory tier: an export imports into another store as the same checkpoints (shared
// pages shared again, the bytes of every plane and state part, the drafter's position) and exports
// back to the same bytes; damaged, foreign or mismatched images are refused before a slab is taken;
// a full store or a tier that cannot write leaves nothing behind.

#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/checkpoint_image.h"
#include "models/qwen4_exp/program/context_cache.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <new>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;
using Handle = HostCheckpointStore::Handle;

namespace {

constexpr std::size_t kSlab   = 4096;
constexpr std::uint64_t kMiB  = 1ULL << 20;
constexpr std::uint64_t kGiB  = 1ULL << 30;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// A text prompt identity over `tokens` (positions index + 0 on every axis).
std::shared_ptr<CheckpointIdentity> identity(const std::vector<TokenId>& tokens) {
    models::qwen3_5::PreparedPromptData prompt;
    prompt.token_ids = tokens;
    prompt.token_types.assign(tokens.size(), 0);
    prompt.positions.resize(3 * tokens.size());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens.size(); ++i) { prompt.positions[axis * tokens.size() + i] = static_cast<std::int32_t>(i); }
    }
    auto out    = std::make_shared<CheckpointIdentity>();
    out->ledger = tokens;
    out->prefix_identity.assign(prompt);
    out->digests.assign(prompt);
    return out;
}

std::vector<TokenId> sequence(std::size_t n, TokenId first = 1) {
    std::vector<TokenId> out(n);
    std::iota(out.begin(), out.end(), first);
    return out;
}

std::uint32_t pages_for(std::uint32_t frontier) { return (frontier + 63) / 64; }

// A checkpoint at `frontier` reusing `shared` leading pages (one more reference each) and new slabs
// for the rest and a two-slab image.
Handle checkpoint(HostCheckpointStore& store, std::shared_ptr<CheckpointIdentity> id, std::uint32_t frontier,
                  const std::vector<std::uint32_t>& shared, runtime::CheckpointRole role = runtime::CheckpointRole::InputReplay) {
    HostCheckpoint record;
    record.identity = std::move(id);
    record.frontier = frontier;
    record.role     = role;
    for (const std::uint32_t slab : shared) {
        store.retain(slab);
        record.pages.push_back(slab);
    }
    const std::uint32_t fresh = pages_for(frontier) - static_cast<std::uint32_t>(shared.size());
    const auto pages = store.allocate(fresh);
    require(pages.has_value(), "pool too small for the test checkpoint");
    record.pages.insert(record.pages.end(), pages->begin(), pages->end());
    const auto image = store.allocate(2);
    require(image.has_value(), "pool too small for the test image");
    record.image = *image;
    return store.insert(std::move(record));
}

void test_slab_pool() {
    HostCheckpointStore store(8, kSlab);
    require(store.capacity_bytes() == 8 * kSlab && store.occupied_bytes() == 0, "empty pool");
    auto a = store.allocate(5);
    require(a && a->size() == 5 && store.used_slabs() == 5, "allocate five");
    require(!store.allocate(4).has_value() && store.used_slabs() == 5, "an allocation that does not fit takes nothing");
    store.retain((*a)[0]);
    store.release(*a);
    require(store.used_slabs() == 1 && store.references((*a)[0]) == 1, "a retained slab survives one release");
    store.release((*a)[0]);
    require(store.used_slabs() == 0 && store.peak_bytes() == 5 * kSlab, "all free again; the peak stays");
    require(store.slabs_for(1) == 1 && store.slabs_for(kSlab) == 1 && store.slabs_for(kSlab + 1) == 2, "slab rounding");
    std::printf("ok: slab pool\n");
}

void test_shared_pages_and_release_quotes() {
    HostCheckpointStore store(64, kSlab);
    const auto tokens = sequence(400);
    const auto id     = identity(tokens);
    // P at 200 (4 pages: 3 full + tail), E at 330 sharing P's 3 full pages (6 pages).
    const Handle p = checkpoint(store, id, 200, {});
    const auto& precord = store.at(p);
    const std::vector<std::uint32_t> full(precord.pages.begin(), precord.pages.begin() + 3);
    const Handle e = checkpoint(store, id, 330, full, runtime::CheckpointRole::Continuation);
    require(store.used_slabs() == 4 + 2 + 3 + 2, "E stores only its own pages and image");
    const Handle both[] = {p, e};
    require(store.footprint_bytes(both) == 11 * kSlab, "footprint counts a shared page once");
    require(store.footprint_bytes({&e, 1}) == 8 * kSlab, "E's footprint includes the shared pages");
    // Releasing E alone frees its 3 own pages and image; P alone frees its tail and image only.
    require(store.released_bytes({&e, 1}) == 5 * kSlab, "E alone frees its own slabs");
    require(store.released_bytes({&p, 1}) == 3 * kSlab, "P alone keeps the shared full pages");
    require(store.released_bytes(both) == 11 * kSlab, "both free everything");
    // The lane mirrors the shared pages: they are not freed by releasing both.
    for (const std::uint32_t slab : full) { store.retain(slab); }
    require(store.released_bytes(both) == 8 * kSlab, "a page the lane holds is not counted");
    // A pinned member makes the set unreleasable as a whole.
    store.pin(p);
    require(store.released_bytes(both) == 0 && !store.erase(p), "a pinned checkpoint cannot be released");
    store.unpin(p);
    require(store.erase(p) && !store.valid(p), "unpinned P erases");
    require(store.valid(e) && store.at(e).frontier == 330, "E survives P");
    require(store.used_slabs() == 3 + 3 + 2, "P's tail and image freed, shared pages kept by E and the lane");
    require(store.erase(e), "E erases");
    require(store.used_slabs() == 3, "only the lane's mirror remains");
    store.release(full);
    require(store.used_slabs() == 0, "the lane lets go: empty");
    // A reused slot gets a new generation: the old handle stays stale.
    const Handle again = checkpoint(store, id, 64, {});
    require(again.index == p.index && again.generation != p.generation && !store.valid(p), "slot reuse is generational");
    std::printf("ok: shared pages, footprint and release quotes\n");
}

void test_release_sets() {
    HostCheckpointStore store(64, kSlab);
    const auto id = identity(sequence(300));
    const Handle a = checkpoint(store, id, 128, {});
    const std::vector<std::uint32_t> shared = store.at(a).pages; // two full pages
    const Handle b = checkpoint(store, id, 256, shared);
    const Handle c = checkpoint(store, id, 200, shared);
    const Handle all[] = {a, b, c};
    auto sets = store.release_sets(all);
    const auto has = [&](std::vector<Handle> set) {
        std::sort(set.begin(), set.end(), [](Handle x, Handle y) { return x.index < y.index; });
        return std::find(sets.begin(), sets.end(), set) != sets.end();
    };
    require(has({a}) && has({b}) && has({c}) && has({a, b, c}), "singletons and the shared-page holder set");
    require(store.released_bytes(std::vector<Handle>{a, b, c}) == store.footprint_bytes(all), "the holder set frees everything");
    // Not eligible: a holder outside the allowed set removes its sets.
    const Handle two[] = {a, b};
    sets = store.release_sets(two);
    require(has({a}) && has({b}) && !has({a, b, c}) && !has({a, b}), "a set needs every holder allowed");
    store.pin(b);
    sets = store.release_sets(all);
    require(!has({b}) && !has({a, b, c}), "pinned checkpoints are never offered");
    std::printf("ok: release sets\n");
}

void test_prefix_and_recovery_loss() {
    HostCheckpointStore store(128, kSlab);
    const auto base   = sequence(1000);
    auto branch       = base;
    branch[500]       = 9999; // diverges at 500
    const auto main   = identity(base);
    const auto other  = identity(branch);
    const Handle r100 = checkpoint(store, main, 100, {});
    const Handle r400 = checkpoint(store, main, 400, {});
    const Handle r900 = checkpoint(store, main, 900, {});
    const Handle b800 = checkpoint(store, other, 800, {}); // shares 500 tokens with main
    require(store.prefix(r100, r900) && store.prefix(r400, r900) && !store.prefix(r900, r400), "same identity prefixes");
    require(store.prefix(r400, b800) && !store.prefix(r900, b800), "cross-identity prefix by tokens");
    const Handle all[] = {r100, r400, r900, b800};
    // Removing r900 falls back to r400: 500 tokens lost.
    require(store.recovery_loss({&r900, 1}, all) == 500, "loss to the deepest surviving prefix");
    // Removing r900 and r400 together: one lineage, the deepest loss 800 (falls back to r100).
    const Handle pair[] = {r900, r400};
    require(store.recovery_loss(pair, all) == 800, "one lineage counts its deepest loss once");
    // r900 and b800 are separate lineages: 500 + 400.
    const Handle forks[] = {r900, b800};
    require(store.recovery_loss(forks, all) == 900, "independent lineages add");
    // Nothing survives: the whole frontier is lost.
    require(store.recovery_loss({&r100, 1}, {}) == 100, "no fallback");
    std::printf("ok: prefixes and recovery loss\n");
}

// The capture paths' ownership: a SlabRefs releases what it still holds; a group insert either
// publishes every role or leaves nothing inserted and nothing held, whichever insert fails; a clone
// that fails holds nothing. Run against the pool's occupancy, which a leak would leave raised.
void test_capture_ownership() {
    HostCheckpointStore store(64, kSlab);
    const auto id = identity(sequence(300));
    // A lane mirror of one page stays held throughout (the capture paths retain it again).
    const auto mirror = store.allocate(1);
    require(mirror.has_value(), "mirror page");
    const std::uint32_t baseline = store.used_slabs();
    {
        SlabRefs refs(store, *store.allocate(3));
        require(store.used_slabs() == baseline + 3, "held");
    }
    require(store.used_slabs() == baseline, "a dropped SlabRefs releases its slabs");
    {
        SlabRefs refs(store, *store.allocate(2));
        auto taken = refs.take();
        require(taken.size() == 2 && refs.size() == 0, "take hands the references over");
        store.release(taken);
        refs = SlabRefs(store, *store.allocate(1));
        refs = SlabRefs(store); // assignment releases the old contents
        require(store.used_slabs() == baseline, "move assignment releases what it replaces");
    }
    // A capture record at 200 tokens: the mirror's page (retained once more), 3 new pages, an image.
    const auto make_record = [&] {
        HostCheckpoint r;
        r.identity = id;
        r.frontier = 200;
        store.retain((*mirror)[0]);
        r.pages.push_back((*mirror)[0]);
        const auto pages = *store.allocate(3);
        r.pages.insert(r.pages.end(), pages.begin(), pages.end());
        r.image = *store.allocate(2);
        return r;
    };
    const runtime::CheckpointRole roles[] = {runtime::CheckpointRole::InputReplay, runtime::CheckpointRole::SharedPrefix,
                                             runtime::CheckpointRole::LongAnchor};
    for (std::size_t fail = 0; fail < 3; ++fail) {
        store.inject_insert_failure(fail);
        bool threw = false;
        try {
            (void)store.insert_group(make_record(), roles);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        require(threw, "the injected insert failure propagates");
        require(store.size() == 0 && store.used_slabs() == baseline && store.references((*mirror)[0]) == 1,
                "a group insert failing at role " + std::to_string(fail) + " leaves nothing inserted or held");
    }
    bool threw = false;
    try {
        (void)store.insert_group(make_record(), {});
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw && store.used_slabs() == baseline, "a group without roles is refused and releases its record");
    // Success: one record per role, each holding every slab once.
    const auto handles = store.insert_group(make_record(), roles);
    require(handles.size() == 3 && store.size() == 3, "one record per role");
    for (std::size_t i = 0; i < 3; ++i) { require(store.at(handles[i]).role == roles[i], "roles in order"); }
    const auto& record = store.at(handles[0]);
    require(store.references(record.pages[1]) == 3 && store.references(record.image[0]) == 3 &&
                store.references((*mirror)[0]) == 4,
            "each record holds one reference (the mirror one more)");
    // A clone that fails holds nothing; one that succeeds shares the slabs.
    store.inject_insert_failure(0);
    threw = false;
    try {
        (void)store.clone(handles[0]);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    require(threw && store.size() == 3 && store.references(record.pages[1]) == 3, "a failed clone holds nothing");
    const Handle copy = store.clone(handles[0]);
    require(store.references(store.at(copy).pages[1]) == 4, "a clone takes its own references");
    require(store.erase(copy), "the clone erases");
    for (const Handle handle : handles) { require(store.erase(handle), "erase"); }
    require(store.used_slabs() == baseline, "everything back but the mirror");
    std::printf("ok: capture ownership (SlabRefs, group insert rollback, clone)\n");
}

// Prefill calls end at the capture groups' frontiers and the typed rewrite's execution frontiers,
// cache on or off: merged, ascending, unique, strictly inside the prompt.
void test_prefill_splits() {
    const std::uint32_t captures[]  = {512, 4096, 5000};
    const std::uint32_t execution[] = {100, 512, 3000, 5000};
    const auto splits               = prefill_splits(captures, execution, 5000);
    require(splits == std::vector<std::uint32_t>({100, 512, 3000, 4096}), "execution frontiers split prefill too");
    require(prefill_splits(captures, {}, 5000) == std::vector<std::uint32_t>({512, 4096}), "capture frontiers alone");
    require(prefill_splits({}, execution, 5000) == std::vector<std::uint32_t>({100, 512, 3000}),
            "execution frontiers split without any capture group (the cache off)");
    std::printf("ok: prefill splits include execution frontiers\n");
}

// A source before an image is priced with the image's encode; one past it is not (the suffix
// intersection rule, the same set plan_lane_vision encodes).
void test_prefill_work_media() {
    MediaCost image{.spans = {{1000, 384}}, .patches = 1536};
    const MediaCost media[] = {image};
    const auto before = prefill_work_between(0, 5000, 4096, media);
    const auto at     = prefill_work_between(1000, 5000, 4096, media);
    const auto inside = prefill_work_between(1383, 5000, 4096, media);
    const auto after  = prefill_work_between(1384, 5000, 4096, media);
    require(before.vision_items == 1 && before.vision_patches == 1536, "a cold prefill encodes the image");
    require(at.vision_items == 1 && inside.vision_items == 1, "a suffix holding any image token encodes it");
    require(after.vision_items == 0 && after.vision_patches == 0, "a suffix past the image does not");
    require(after.tokens == 3616 && after.chunks == 1 &&
                after.attention_pairs == runtime::make_prefill_work(1384, 3616, 0, 0, 4096).attention_pairs,
            "token work is the plain suffix work");
    require(prefill_work_between(5000, 5000, 4096, media).vision_items == 0, "an exact hit encodes nothing");
    std::printf("ok: prefill pricing counts the images a source leaves\n");
}

HostMemoryDemand flash_next() {
    HostMemoryDemand d;
    d.reserve      = 2 * kGiB;
    d.expert_banks = 33'812 * kMiB;
    d.later_pinned = 256 * kMiB;
    d.pageable     = 90 * kMiB;
    d.load_staging = 264 * kMiB;
    return d;
}

HostMemorySnapshot snapshot(std::uint64_t available) {
    HostMemorySnapshot s;
    s.total_physical     = 64 * kGiB;
    s.available_physical = available;
    s.available_commit   = available + 16 * kGiB;
    return s;
}

void test_ledger_fit() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t granule = 1 * kMiB, minimum = 512 * kMiB;
    // Plenty of RAM: the request is granted and the ledger with it still fits.
    auto fit = fit_context_cache(snapshot(56 * kGiB), d, 4 * kGiB, minimum, granule);
    require(fit.bytes == 4 * kGiB && !fit.shrunk, "4 GiB fits beside the experts with 56 GiB free");
    HostMemoryDemand with = d;
    with.context_cache    = fit.bytes;
    require(plan_host_memory(snapshot(56 * kGiB), with).fits, "the ledger with the cache fits");
    // Find the free memory at which the experts alone just fit, then grant only what is above it.
    std::uint64_t edge = 30 * kGiB;
    while (!plan_host_memory(snapshot(edge), d).fits) { edge += 16 * kMiB; }
    fit = fit_context_cache(snapshot(edge + 1 * kGiB), d, 4 * kGiB, minimum, granule);
    require(fit.shrunk && fit.bytes >= 1000 * kMiB && fit.bytes <= 1 * kGiB + 16 * kMiB,
            "shrinks to the room above the experts");
    with.context_cache = fit.bytes;
    require(plan_host_memory(snapshot(edge + 1 * kGiB), with).fits, "the shrunk cache still fits");
    // Below the minimum the cache is off; with the experts short it is off too.
    fit = fit_context_cache(snapshot(edge + 256 * kMiB), d, 4 * kGiB, minimum, granule);
    require(fit.bytes == 0 && fit.shrunk, "below 512 MiB the cache is off");
    fit = fit_context_cache(snapshot(edge - 1 * kGiB), d, 4 * kGiB, minimum, granule);
    require(fit.bytes == 0, "no cache when the experts do not fit");
    // A finite memory-lock limit bounds it as well.
    auto locked       = snapshot(56 * kGiB);
    const auto pinned = plan_host_memory(locked, d).pinned_total;
    locked.lock_limit = pinned + 2 * kGiB;
    fit = fit_context_cache(locked, d, 4 * kGiB, minimum, granule);
    require(fit.bytes == 2 * kGiB && fit.shrunk, "the memory-lock limit bounds the cache");
    std::printf("ok: RAM ledger sizes the Host tier\n");
}

void test_free_callback() {
    HostCheckpointStore store(16, kSlab);
    std::vector<std::uint32_t> freed;
    store.on_free([&](std::uint32_t slab) { freed.push_back(slab); });
    const auto id   = identity(sequence(200));
    const Handle a  = checkpoint(store, id, 128, {});
    const auto full = store.at(a).pages;
    const Handle b  = checkpoint(store, id, 192, full);
    require(store.erase(a) && freed.size() == 2, "erasing A frees only its image (its pages are B's too)");
    require(store.erase(b) && freed.size() == 2 + 3 + 2, "erasing B frees every page and its image");
    std::printf("ok: the store reports freed slabs\n");
}

// An in-memory tier for the image tests: whole-slab reads and writes, no device copies.
class MemoryTier final : public CheckpointTier {
public:
    MemoryTier(std::uint32_t slabs, std::size_t slab_bytes)
        : slabs_(slabs), slab_bytes_(slab_bytes), bytes_(static_cast<std::size_t>(slabs) * slab_bytes) {}
    [[nodiscard]] std::uint32_t slabs() const noexcept override { return slabs_; }
    [[nodiscard]] std::size_t slab_bytes() const noexcept override { return slab_bytes_; }
    void copy(std::span<const SlabCopy>, bool, cudaStream_t) override { throw std::logic_error("no device copies here"); }
    void read(std::span<const std::uint32_t> slabs, std::span<std::byte> out) override {
        for (std::size_t i = 0; i < slabs.size(); ++i) { std::memcpy(out.data() + i * slab_bytes_, slab(slabs[i]), slab_bytes_); }
    }
    void write(std::span<const std::uint32_t> slabs, std::span<const std::byte> in) override {
        if (fail_writes) { throw std::system_error(ENOSPC, std::generic_category(), "test: the disk is full"); }
        for (std::size_t i = 0; i < slabs.size(); ++i) { std::memcpy(slab(slabs[i]), in.data() + i * slab_bytes_, slab_bytes_); }
    }
    std::byte* slab(std::uint32_t s) { return bytes_.data() + static_cast<std::size_t>(s) * slab_bytes_; }
    bool fail_writes = false;

private:
    std::uint32_t slabs_;
    std::size_t slab_bytes_;
    std::vector<std::byte> bytes_;
};

// Two attention layers of five planes with padding between them in the record, and a two-slab state
// image (state block, the drafter's residual, logits) with gaps between its parts.
CheckpointLayout test_layout() {
    CheckpointLayout layout;
    layout.slab_bytes      = kSlab;
    layout.plane_offsets   = {{0, 512, 1024, 1152, 1280}, {1536, 2048, 2560, 2688, 2816}};
    layout.plane_bytes     = {{500, 500, 100, 100, 200}, {500, 500, 100, 100, 200}};
    layout.state_bytes     = 6000;
    layout.residual_offset = 6144;
    layout.residual_bytes  = 512;
    layout.logits_offset   = 6656;
    layout.logits_bytes    = 1000;
    layout.image_slabs     = 2;
    layout.identity_tag    = 0xABCD;
    layout.capacity        = 4096;
    layout.vocabulary      = 100000;
    return layout;
}

void fill(MemoryTier& tier, std::span<const std::uint32_t> slabs, std::uint32_t seed) {
    for (const std::uint32_t s : slabs) {
        for (std::size_t i = 0; i < kSlab; ++i) { tier.slab(s)[i] = static_cast<std::byte>((seed * 131 + s * 7 + i * 13) & 0xff); }
    }
}

// A checkpoint as a capture leaves it: its own pages and image filled, `shared` leading pages held.
Handle point(HostCheckpointStore& store, MemoryTier& tier, const CheckpointLayout& layout,
             std::shared_ptr<CheckpointIdentity> id, std::uint32_t frontier, const std::vector<std::uint32_t>& shared,
             runtime::CheckpointRole role, std::uint32_t mtp_cells, std::optional<TokenId> next, bool logits,
             std::uint32_t seed) {
    HostCheckpoint record;
    record.identity     = std::move(id);
    record.identity_tag = layout.identity_tag;
    record.frontier     = frontier;
    record.role         = role;
    record.full_pages   = frontier / 64;
    record.mtp_cells    = mtp_cells;
    record.next_token   = next;
    record.logits       = logits;
    record.rope_delta   = 7;
    for (const std::uint32_t slab : shared) {
        store.retain(slab);
        record.pages.push_back(slab);
    }
    const auto fresh = store.allocate(pages_for(frontier) - static_cast<std::uint32_t>(shared.size()));
    const auto image = store.allocate(layout.image_slabs);
    require(fresh && image, "pool too small for the test point");
    fill(tier, *fresh, seed);
    fill(tier, *image, seed + 100);
    record.pages.insert(record.pages.end(), fresh->begin(), fresh->end());
    record.image = *image;
    return store.insert(std::move(record));
}

bool same_planes(MemoryTier& a, std::uint32_t sa, MemoryTier& b, std::uint32_t sb, const CheckpointLayout& layout) {
    for (std::size_t l = 0; l < layout.plane_bytes.size(); ++l) {
        for (std::size_t p = 0; p < 5; ++p) {
            if (std::memcmp(a.slab(sa) + layout.plane_offsets[l][p], b.slab(sb) + layout.plane_offsets[l][p],
                            layout.plane_bytes[l][p]) != 0) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::byte> image_part(MemoryTier& tier, const HostCheckpoint& r, std::size_t offset, std::size_t bytes) {
    std::vector<std::byte> all(r.image.size() * kSlab);
    tier.read(r.image, all);
    return {all.begin() + static_cast<std::ptrdiff_t>(offset), all.begin() + static_cast<std::ptrdiff_t>(offset + bytes)};
}

template <class F>
bool refused(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_image_round_trip() {
    const CheckpointLayout layout = test_layout();
    HostCheckpointStore a(64, kSlab);
    MemoryTier ta(64, kSlab);
    const auto tokens = sequence(400);
    const auto prompt = identity(tokens);
    const auto reply  = identity(sequence(330));
    // P: the prompt's checkpoint at 200 (its logits, the drafter at 200 with the token at 200); E:
    // the endpoint at 330 sharing P's three full pages (the drafter one cell behind, no logits).
    const Handle p = point(a, ta, layout, prompt, 200, {}, runtime::CheckpointRole::InputReplay, 200, tokens[200], true, 1);
    const std::vector<std::uint32_t> full(a.at(p).pages.begin(), a.at(p).pages.begin() + 3);
    const Handle e = point(a, ta, layout, reply, 330, full, runtime::CheckpointRole::Continuation, 329, std::nullopt, false, 2);
    models::qwen3_5::PreparedSessionKey session;
    session.size = 5;
    std::memcpy(session.bytes.data(), "chat1", 5);
    const Handle both[] = {p, e};
    const auto image    = export_checkpoint_image(a, ta, layout, both, session, "binding-1");
    require(image.keys.size() == 2 && image.keys[0].frontier == 330 && image.keys[1].frontier == 200 && image.tokens == 330,
            "keys deepest first");
    require(image.keys[0].digests == reply->digests.at(330) && image.keys[0].identity_tag == layout.identity_tag,
            "the endpoint's key");
    require(image.prefix_digests.size() == 331 && image.regions.size() == 2, "prefix digests and payload regions");
    // Pages once: P's four and E's three own; two images of two slabs.
    require(checkpoint_image_slab_bytes(layout, image.bytes, "binding-1") == (7 + 4) * kSlab, "slab bytes of the import");

    HostCheckpointStore b(64, kSlab);
    MemoryTier tb(64, kSlab);
    const auto taken = b.allocate(5); // different slab numbers on the importing side
    const auto imported = import_checkpoint_image(b, tb, layout, image.bytes, "binding-1");
    require(imported && imported->points.size() == 2 && imported->session == session, "imported both points");
    require(b.used_slabs() == 5 + 11, "the import holds exactly the image's slabs");
    const HostCheckpoint& be = b.at(imported->points[0]);
    const HostCheckpoint& bp = b.at(imported->points[1]);
    for (const auto& [x, y] : {std::pair{&a.at(e), &be}, std::pair{&a.at(p), &bp}}) {
        require(x->frontier == y->frontier && x->role == y->role && x->full_pages == y->full_pages &&
                    x->mtp_cells == y->mtp_cells && x->next_token == y->next_token && x->logits == y->logits &&
                    x->rope_delta == y->rope_delta && y->identity_tag == layout.identity_tag,
                "a point's fields survive");
        require(x->identity->ledger == y->identity->ledger && x->identity->digests.image() == y->identity->digests.image() &&
                    x->identity->prefix_identity.equals(y->identity->prefix_identity),
                "a point's identity survives");
        for (std::size_t i = 0; i < x->pages.size(); ++i) {
            require(same_planes(ta, x->pages[i], tb, y->pages[i], layout), "every plane of every page survives");
        }
        require(image_part(ta, *x, 0, layout.state_bytes) == image_part(tb, *y, 0, layout.state_bytes) &&
                    image_part(ta, *x, layout.residual_offset, layout.residual_bytes) ==
                        image_part(tb, *y, layout.residual_offset, layout.residual_bytes),
                "the state block and the drafter's residual survive");
    }
    require(std::equal(be.pages.begin(), be.pages.begin() + 3, bp.pages.begin()), "shared pages are shared again");
    require(image_part(ta, a.at(p), layout.logits_offset, layout.logits_bytes) ==
                image_part(tb, bp, layout.logits_offset, layout.logits_bytes),
            "captured logits survive");
    const auto zero = image_part(tb, be, layout.logits_offset, layout.logits_bytes);
    require(std::all_of(zero.begin(), zero.end(), [](std::byte v) { return v == std::byte{0}; }),
            "logits never captured come back as zeros");
    // The same checkpoints export to the same bytes wherever their slabs are.
    const auto again = export_checkpoint_image(b, tb, layout, imported->points, session, "binding-1");
    require(again.bytes == image.bytes, "a re-export is byte-identical");
    b.release(*taken);
    std::printf("ok: a checkpoint image round-trips\n");
}

void test_image_refusals() {
    const CheckpointLayout layout = test_layout();
    HostCheckpointStore a(64, kSlab);
    MemoryTier ta(64, kSlab);
    const auto tokens = sequence(300);
    const auto id     = identity(tokens);
    const Handle p = point(a, ta, layout, id, 256, {}, runtime::CheckpointRole::InputReplay, 256, tokens[256], true, 3);
    const auto image = export_checkpoint_image(a, ta, layout, {&p, 1}, std::nullopt, "binding-1").bytes;
    HostCheckpointStore b(64, kSlab);
    MemoryTier tb(64, kSlab);
    const auto load = [&](const std::vector<std::uint8_t>& bytes, std::string_view binding, const CheckpointLayout& l) {
        return import_checkpoint_image(b, tb, l, bytes, binding);
    };
    auto truncated = image;
    truncated.pop_back();
    require(refused([&] { (void)load(truncated, "binding-1", layout); }), "a truncated image is refused");
    auto corrupt = image;
    corrupt[corrupt.size() / 2] ^= 0x5A;
    require(refused([&] { (void)load(corrupt, "binding-1", layout); }), "a damaged payload is refused");
    auto header = image;
    header[20] ^= 0x01; // inside the binding
    require(refused([&] { (void)load(header, "binding-1", layout); }), "a damaged header is refused");
    require(refused([&] { (void)load(image, "binding-2", layout); }), "another model's image is refused");
    auto other_tag = layout;
    ++other_tag.identity_tag;
    require(refused([&] { (void)load(image, "binding-1", other_tag); }), "another configuration's image is refused");
    auto other_planes = layout;
    other_planes.plane_bytes[1][4] = 196;
    require(refused([&] { (void)load(image, "binding-1", other_planes); }), "another page layout's image is refused");
    auto no_drafter = layout;
    no_drafter.residual_bytes = 0;
    require(refused([&] { (void)load(image, "binding-1", no_drafter); }), "an image with the drafter needs the drafter");
    require(refused([&] { (void)load(std::vector<std::uint8_t>(64, 7), "binding-1", layout); }), "random bytes are refused");
    require(b.used_slabs() == 0, "nothing was taken by a refused image");
    // No room, or a tier that cannot store the bytes: nothing kept.
    HostCheckpointStore small(5, kSlab);
    require(!import_checkpoint_image(small, tb, layout, image, "binding-1") && small.used_slabs() == 0, "no room: a miss");
    tb.fail_writes = true;
    require(!load(image, "binding-1", layout) && b.used_slabs() == 0, "a failing tier: a miss");
    tb.fail_writes = false;
    require(load(image, "binding-1", layout).has_value(), "the same image imports once the tier can write");
    std::printf("ok: damaged, foreign and mismatched images are refused\n");
}

void test_shared_prefix_images() {
    const CheckpointLayout layout = test_layout();
    HostCheckpointStore a(64, kSlab);
    MemoryTier ta(64, kSlab);
    const auto tokens = sequence(300);
    const auto id     = identity(tokens);
    const Handle s = point(a, ta, layout, id, 128, {}, runtime::CheckpointRole::SharedPrefix, 128, tokens[128], false, 4);
    const Handle p = point(a, ta, layout, id, 256, {}, runtime::CheckpointRole::InputReplay, 256, tokens[256], true, 5);
    models::qwen3_5::PreparedSessionKey session;
    session.size = 1;
    session.bytes[0] = 'x';
    const Handle pair[] = {s, p};
    require(refused([&] { (void)export_checkpoint_image(a, ta, layout, pair, std::nullopt, "b"); }),
            "a shared prefix is not exported with private points");
    require(refused([&] { (void)export_checkpoint_image(a, ta, layout, {&s, 1}, session, "b"); }),
            "a shared prefix is not exported with a session");
    const auto image = export_checkpoint_image(a, ta, layout, {&s, 1}, std::nullopt, "b");
    HostCheckpointStore b(64, kSlab);
    MemoryTier tb(64, kSlab);
    const auto imported = import_checkpoint_image(b, tb, layout, image.bytes, "b");
    require(imported && imported->points.size() == 1 && b.at(imported->points[0]).role == runtime::CheckpointRole::SharedPrefix,
            "a shared prefix round-trips alone");
    std::printf("ok: shared prefix images\n");
}

} // namespace

int main() {
    try {
        test_slab_pool();
        test_shared_pages_and_release_quotes();
        test_release_sets();
        test_prefix_and_recovery_loss();
        test_capture_ownership();
        test_prefill_splits();
        test_prefill_work_media();
        test_ledger_fit();
        test_free_callback();
        test_image_round_trip();
        test_image_refusals();
        test_shared_prefix_images();
        std::printf("qwen4_exp context cache tests passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
