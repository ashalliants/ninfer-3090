// The Qwen3.8-Flash-Next context cache's Host tier (docs/maintainer/flash-next-context-cache.md),
// host only: the slab pool, checkpoints sharing KV pages by reference, the bytes a release frees
// (shared pages count only when every holder goes, and a page the lane still mirrors is never
// counted), pins, the release sets the ResourceManager ranks, the lineage arithmetic of recovery
// loss, and the RAM ledger's sizing of the tier beside the experts.

#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/context_cache.h"

#include <algorithm>
#include <cstdio>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>

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
        std::printf("qwen4_exp context cache tests passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
