#include "runtime/engine/context_store/context_store.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ninfer::runtime::ContextStore;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL " << message << '\n';
}

class TempDirectory {
public:
    TempDirectory() {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path() /
                ("ninfer_context_store_test_" + std::to_string(counter.fetch_add(1)) + "_" +
                 std::to_string(
                     std::chrono::steady_clock::now().time_since_epoch().count() & 0xffffff));
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }
    [[nodiscard]] const fs::path& path() const noexcept { return path_; }

private:
    fs::path path_;
};

std::vector<std::uint8_t> pseudo_random(std::size_t size, std::uint32_t seed) {
    std::vector<std::uint8_t> out(size);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (auto& byte : out) {
        state = state * 1664525U + 1013904223U;
        byte  = static_cast<std::uint8_t>(state >> 24U);
    }
    return out;
}

std::vector<std::uint8_t> concat(std::initializer_list<std::vector<std::uint8_t>> parts) {
    std::vector<std::uint8_t> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

ContextStore::Options options_for(const fs::path& directory) {
    ContextStore::Options options;
    options.directory   = directory;
    options.chunk_bytes = 1024;
    return options;
}

ContextStore::Description describe(const std::string& id, std::uint32_t tokens = 100) {
    ContextStore::Description description;
    description.id      = id;
    description.binding = "model-a";
    description.tokens  = tokens;
    description.checkpoints.push_back(ContextStore::CheckpointKey{
        .frontier = tokens - 1, .digests = {0x1111, 0x2222}, .identity_tag = 7});
    return description;
}

std::vector<fs::path> chunk_files(const fs::path& directory) {
    std::vector<fs::path> files;
    for (const auto& item : fs::recursive_directory_iterator(directory / "chunks")) {
        if (item.is_regular_file()) { files.push_back(item.path()); }
    }
    return files;
}

void test_hash() {
    // XXH64 of the empty input with seed 0 is a published constant.
    check(ContextStore::hash({})[0] == 0xef46db3751d8e999ULL, "xxh64 of empty input");
    const auto a = pseudo_random(5000, 1);
    auto b       = a;
    check(ContextStore::hash(a) == ContextStore::hash(a), "hash is not deterministic");
    b[4999] ^= 1;
    check(ContextStore::hash(a) != ContextStore::hash(b), "a one-bit change did not change the hash");
    const auto short_input = pseudo_random(3, 2);
    check(ContextStore::hash(short_input) != ContextStore::hash({}), "short input hashes alike");
}

void test_round_trip() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto head   = pseudo_random(300, 1);
    const auto state  = pseudo_random(2500, 2);
    const auto kv     = pseudo_random(4000, 3);
    const auto tail   = pseudo_random(8, 4);
    const auto image  = concat({head, state, kv, tail});
    const std::array regions{ContextStore::Region{300, 2500}, ContextStore::Region{2800, 4000}};
    const auto put = store.put(describe("abc123"), image, regions);
    check(put.image_bytes == image.size() && put.chunk_bytes_written == 6500 &&
              put.chunk_bytes_reused == 0,
          "first put did not write every region chunk");
    const auto loaded = store.load("abc123");
    check(loaded && *loaded == image, "round trip changed the image");
    const auto info = store.find("abc123");
    check(info && info->tokens == 100 && info->binding == "model-a" &&
              info->checkpoints.size() == 1 && info->checkpoints[0].digests[1] == 0x2222,
          "stored description was not preserved");
    const auto used_before = store.find("abc123")->last_used_ms;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    (void)store.load("abc123", false);
    check(store.find("abc123")->last_used_ms == used_before,
          "an untouched read refreshed the image's last use");
    (void)store.load("abc123");
    check(store.find("abc123")->last_used_ms > used_before, "a read did not refresh last use");
    check(!store.load("deadbeef").has_value(), "an unknown id loaded");
    check(!store.load("../escape").has_value(), "an unsafe id loaded");
}

void test_incremental_put_writes_only_new_chunks() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto head1 = pseudo_random(200, 1);
    const auto kv1   = pseudo_random(5000, 2);
    const auto image1 = concat({head1, kv1});
    (void)store.put(describe("a1"), image1, std::array{ContextStore::Region{200, 5000}});

    // The conversation grew: a different header, the same older pages, 3000 new bytes of pages.
    const auto head2  = pseudo_random(260, 9);
    const auto kv2    = concat({kv1, pseudo_random(3000, 3)});
    const auto image2 = concat({head2, kv2});
    const auto put    = store.put(describe("a2", 140), image2,
                                  std::array{ContextStore::Region{260, 8000}});
    // 4 whole 1024-byte chunks of the old pages are unchanged; the partial fifth chunk and the new
    // pages are written.
    check(put.chunk_bytes_reused == 4096 && put.chunk_bytes_written == 8000 - 4096,
          "a grown image rewrote chunks it already held");
    const auto loaded = store.load("a2");
    check(loaded && *loaded == image2, "grown image did not round trip");
    check(store.load("a1").value_or(std::vector<std::uint8_t>{}) == image1,
          "the earlier image no longer loads");
}

void test_restart_recovers_images() {
    TempDirectory directory;
    const auto image = concat({pseudo_random(100, 1), pseudo_random(3000, 2), pseudo_random(8, 3)});
    {
        ContextStore store(options_for(directory.path()));
        (void)store.put(describe("f00d"), image, std::array{ContextStore::Region{100, 3000}});
    }
    ContextStore reopened(options_for(directory.path()));
    const auto listed = reopened.list();
    check(listed.size() == 1 && listed[0].id == "f00d" && listed[0].checkpoints.size() == 1,
          "restart did not recover the stored image");
    const auto loaded = reopened.load("f00d");
    check(loaded && *loaded == image, "recovered image differs");
}

void test_corruption_is_a_miss_and_removed() {
    TempDirectory directory;
    const auto image = concat({pseudo_random(64, 1), pseudo_random(3000, 2)});
    {
        ContextStore store(options_for(directory.path()));
        (void)store.put(describe("c0ffee"), image, std::array{ContextStore::Region{64, 3000}});
        const auto files = chunk_files(directory.path());
        check(!files.empty(), "no chunk files were written");
        {
            std::fstream file(files.front(), std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(0);
            char byte = 0;
            file.read(&byte, 1);
            file.seekp(0);
            byte ^= 0x55;
            file.write(&byte, 1);
        }
        check(!store.load("c0ffee").has_value(), "a damaged chunk was served");
        check(!store.find("c0ffee").has_value(), "a damaged image stayed listed");
        check(store.stats().corrupt_removed == 1, "a damaged image was not counted");

        // A truncated manifest is dropped at start-up and its chunks are swept.
        (void)store.put(describe("beef"), image, std::array{ContextStore::Region{64, 3000}});
        const fs::path manifest = directory.path() / "manifests" / "beef.manifest";
        fs::resize_file(manifest, fs::file_size(manifest) / 2);
    } // the first store is closed: a directory has one owner
    ContextStore reopened(options_for(directory.path()));
    check(reopened.list().empty() && chunk_files(directory.path()).empty(),
          "a truncated manifest was kept or left chunks behind");
}

void test_space_limit_evicts_least_recently_used() {
    TempDirectory directory;
    ContextStore::Options options = options_for(directory.path());
    std::int64_t now_ms            = 1'000'000;
    options.clock                  = [&] { return now_ms; };
    const auto base                = pseudo_random(6000, 1);
    {
        // Unbounded first, to learn the size of two overlapping images.
        ContextStore probe(options_for(directory.path() / "probe"));
        (void)probe.put(describe("a1"), concat({pseudo_random(50, 1), base}),
                        std::array{ContextStore::Region{50, 6000}});
        (void)probe.put(describe("a2"), concat({pseudo_random(60, 2), base, pseudo_random(2000, 3)}),
                        std::array{ContextStore::Region{60, 8000}});
        options.max_bytes = probe.stats().used_bytes - 100;
    }
    ContextStore store(options);
    (void)store.put(describe("a1"), concat({pseudo_random(50, 1), base}),
                    std::array{ContextStore::Region{50, 6000}});
    now_ms += 1000;
    const auto put = store.put(describe("a2"), concat({pseudo_random(60, 2), base, pseudo_random(2000, 3)}),
                               std::array{ContextStore::Region{60, 8000}});
    check(put.evicted_images == 1 && !store.find("a1") && store.find("a2"),
          "the least recently used image was not the one evicted");
    check(store.load("a2").has_value(), "chunks the survivor shares with the victim were removed");
    check(store.stats().used_bytes <= options.max_bytes, "the size limit was exceeded");
}

void test_ttl_and_use_extend_life() {
    TempDirectory directory;
    ContextStore::Options options = options_for(directory.path());
    std::int64_t now_ms            = 5'000'000;
    options.clock                  = [&] { return now_ms; };
    options.ttl                    = std::chrono::seconds(100);
    {
        ContextStore store(options);
        const auto image = concat({pseudo_random(40, 1), pseudo_random(2000, 2)});
        (void)store.put(describe("aa"), image, std::array{ContextStore::Region{40, 2000}});
        (void)store.put(describe("bb"), concat({pseudo_random(40, 7), pseudo_random(2000, 8)}),
                        std::array{ContextStore::Region{40, 2000}});
        now_ms += 60'000;
        check(store.load("aa").has_value(), "image should still load inside its lifetime");
        now_ms += 60'000; // bb is 120 s old and unused; aa was used 60 s ago
        store.maintain();
        check(!store.find("bb") && store.find("aa"), "expiry removed the wrong image");
        check(store.stats().expired == 1, "expiry was not counted");
    } // the first store is closed: a directory has one owner

    // The use time survives a restart.
    ContextStore reopened(options);
    check(reopened.find("aa").has_value(), "an image used recently expired after restart");
    now_ms += 120'000;
    reopened.maintain();
    check(reopened.list().empty() && chunk_files(directory.path()).empty(),
          "expired images left chunks behind");
}

void test_supersede_and_replace() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto kv = pseudo_random(3000, 1);
    // Image 01 ends at frontier 99, with the key {0x1111, 0x2222} that describe() gives it.
    (void)store.put(describe("01"), concat({pseudo_random(30, 1), kv}),
                    std::array{ContextStore::Region{30, 3000}});
    // An unrelated image whose prefix digests do not reproduce that key supersedes nothing.
    std::vector<std::array<std::uint64_t, 2>> unrelated(130, {7, 7});
    auto stranger           = describe("03", 120);
    stranger.prefix_digests = unrelated;
    (void)store.put(stranger, concat({pseudo_random(40, 5), pseudo_random(500, 6)}),
                    std::array{ContextStore::Region{40, 500}});
    check(store.find("01") && store.find("03"), "an unrelated image superseded another");

    // The next state of the same conversation reproduces 01's endpoint key at frontier 99.
    std::vector<std::array<std::uint64_t, 2>> digests(130, {5, 5});
    digests[99]             = {0x1111, 0x2222};
    auto next               = describe("02", 120);
    next.prefix_digests     = digests;
    (void)store.put(next, concat({pseudo_random(40, 2), kv, pseudo_random(1000, 3)}),
                    std::array{ContextStore::Region{40, 4000}});
    check(!store.find("01") && store.find("02"), "a superseded image was kept");
    check(store.stats().superseded == 1, "supersession was not counted");
    check(store.load("02").has_value(), "the superseding image lost chunks it shared");
    (void)store.erase("03");

    // Replacing an id keeps its shared chunks and drops the ones only the old image used.
    (void)store.put(describe("02", 130), concat({pseudo_random(40, 9), pseudo_random(500, 5)}),
                    std::array{ContextStore::Region{40, 500}});
    check(store.load("02").has_value() && store.list().size() == 1, "replacing an id broke it");
    check(chunk_files(directory.path()).size() == 1, "a replaced image left unreferenced chunks");
}

// A shared prefix (a system prompt every conversation of a tenant starts with) is kept as its own
// image: the flag survives a restart, and a conversation that reproduces its endpoint key (as every
// conversation starting from it does) does not supersede it.
void test_shared_prefix_is_kept_and_flagged() {
    TempDirectory directory;
    const auto kv = pseudo_random(3000, 1);
    {
        ContextStore store(options_for(directory.path()));
        auto prefix   = describe("0a5");
        prefix.shared = true;
        (void)store.put(prefix, concat({pseudo_random(30, 1), kv}),
                        std::array{ContextStore::Region{30, 3000}});
        std::vector<std::array<std::uint64_t, 2>> digests(130, {5, 5});
        digests[99]               = {0x1111, 0x2222};
        auto conversation         = describe("0b", 120);
        conversation.prefix_digests = digests;
        (void)store.put(conversation, concat({pseudo_random(40, 2), kv, pseudo_random(900, 3)}),
                        std::array{ContextStore::Region{40, 3900}});
        check(store.find("0a5") && store.find("0b") && store.stats().superseded == 0,
              "a conversation superseded the shared prefix it starts with");
        check(store.find("0a5")->shared && !store.find("0b")->shared,
              "the shared flag was not recorded");
    }
    ContextStore reopened(options_for(directory.path()));
    const auto info = reopened.find("0a5");
    check(info && info->shared && reopened.find("0b") && !reopened.find("0b")->shared,
          "the shared flag did not survive a restart");
    check(reopened.load("0a5").has_value(), "the shared prefix did not load after a restart");
}

void test_invalid_input_and_failed_put() {
    TempDirectory directory;
    ContextStore store(options_for(directory.path()));
    const auto image = pseudo_random(2000, 1);
    bool threw       = false;
    try {
        (void)store.put(describe("ab"), image,
                        std::array{ContextStore::Region{100, 500}, ContextStore::Region{400, 500}});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "overlapping regions were accepted");
    threw = false;
    try {
        (void)store.put(describe("ab"), image, std::array{ContextStore::Region{1500, 1000}});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a region past the end of the image was accepted");
    threw = false;
    try {
        (void)store.put(describe("not-hex!"), image, {});
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an unsafe id was accepted");

    // The manifest cannot be published: nothing of the image may remain.
    fs::remove_all(directory.path() / "manifests");
    { std::ofstream blocker(directory.path() / "manifests"); }
    threw = false;
    try {
        (void)store.put(describe("cd"), image, std::array{ContextStore::Region{0, 2000}});
    } catch (const std::runtime_error&) { threw = true; }
    check(threw, "a put that could not publish its manifest reported success");
    check(chunk_files(directory.path()).empty() && store.list().empty() &&
              store.stats().put_failures == 1,
          "a failed put left chunks or an entry behind");
}

// An in-memory ObjectStore with switches for the failures a bucket has.
class MemoryObjectStore final : public ninfer::ObjectStore {
public:
    void put(const std::string& key, std::span<const std::uint8_t> bytes) override {
        gate();
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        objects[key] = {std::vector<std::uint8_t>(bytes.begin(), bytes.end()), ++clock};
        ++puts;
    }
    std::optional<std::vector<std::uint8_t>> get(const std::string& key) override {
        gate();
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        if (key.find("chunks/") != std::string::npos) { ++chunk_gets; }
        const auto found = objects.find(key);
        if (found == objects.end()) { return std::nullopt; }
        return found->second.first;
    }
    bool exists(const std::string& key) override {
        gate();
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        return objects.find(key) != objects.end();
    }
    std::vector<ninfer::ObjectInfo> list(const std::string& prefix) override {
        gate();
        std::scoped_lock lock(mutex);
        if (down) { throw std::runtime_error("the bucket is unreachable"); }
        std::vector<ninfer::ObjectInfo> out;
        for (const auto& [key, value] : objects) {
            if (key.rfind(prefix, 0) == 0) {
                out.push_back({key, value.first.size(), value.second});
            }
        }
        return out;
    }
    bool touch(const std::string& key) override {
        gate();
        std::scoped_lock lock(mutex);
        const auto found = objects.find(key);
        if (found == objects.end()) { return false; }
        found->second.second = ++clock;
        ++touches;
        return true;
    }
    void remove(const std::string& key) override {
        std::scoped_lock lock(mutex);
        objects.erase(key);
    }
    // Ends the waits in progress, which fail. (A real store is not used after interrupt(); this one
    // is shared by the stores a test opens in turn, so later calls proceed.)
    void interrupt() noexcept override {
        {
            std::scoped_lock lock(gate_mutex);
            ++interrupt_epoch;
        }
        gate_cv.notify_all();
    }

    // While `hold` is set every transfer waits, as a slow link would; interrupt() ends the wait.
    void set_hold(bool value) {
        {
            std::scoped_lock lock(gate_mutex);
            hold = value;
        }
        gate_cv.notify_all();
    }

    std::mutex mutex;
    std::map<std::string, std::pair<std::vector<std::uint8_t>, std::int64_t>> objects;
    std::int64_t clock = 1'000'000'000'000;
    bool down          = false;
    int puts           = 0;
    int touches        = 0;
    int chunk_gets     = 0;

private:
    void gate() {
        std::unique_lock lock(gate_mutex);
        const std::uint64_t epoch = interrupt_epoch;
        gate_cv.wait(lock, [&] { return !hold || interrupt_epoch != epoch; });
        if (interrupt_epoch != epoch) { throw std::runtime_error("interrupted"); }
    }
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool hold                     = false;
    std::uint64_t interrupt_epoch = 0;
};

ContextStore::Options remote_options(const fs::path& directory,
                                     const std::shared_ptr<MemoryObjectStore>& remote) {
    ContextStore::Options options = options_for(directory);
    options.remote                = remote;
    options.remote_prefix         = "ninfer/";
    options.remote_refresh        = std::chrono::seconds(3600); // tests refresh explicitly
    options.clock                 = [remote] { return remote->clock; };
    return options;
}

bool drained(ContextStore& store) {
    return store.flush_remote(std::chrono::steady_clock::now() + std::chrono::seconds(10));
}

void test_remote_upload_and_second_engine() {
    TempDirectory first_directory;
    TempDirectory second_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const auto kv    = pseudo_random(5000, 4);
    const auto image = concat({pseudo_random(30, 1), kv});
    {
        ContextStore store(remote_options(first_directory.path(), remote));
        (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 5000}});
        check(drained(store), "the upload queue did not drain");
        const auto stats = store.stats();
        check(stats.remote_uploads == 6 && stats.remote_upload_failures == 0,
              "a put did not upload its chunks and manifest");
        // Writing the next state of the conversation uploads only what is new.
        const int before = remote->puts;
        (void)store.put(describe("0a", 120), concat({pseudo_random(30, 1), kv, pseudo_random(500, 8)}),
                        std::array{ContextStore::Region{30, 5500}});
        check(drained(store), "the second upload did not drain");
        check(remote->puts - before == 3, "a rewritten image uploaded chunks the bucket already had");
    }
    // A second engine with an empty directory sees the image and fetches it when asked.
    ContextStore other(remote_options(second_directory.path(), remote));
    check(other.refresh_remote() == 1 && other.list().size() == 1, "the remote image was not listed");
    check(!other.list()[0].local, "an image not fetched yet reports itself local");
    check(other.stats().remote_images == 1, "the remote-only image was not counted");
    check(!other.list()[0].checkpoints.empty() && other.list()[0].checkpoints[0].digests[0] == 0x1111,
          "the remote image lost its checkpoint keys");
    const auto loaded = other.load("0a");
    check(loaded && *loaded == concat({pseudo_random(30, 1), kv, pseudo_random(500, 8)}),
          "an image fetched from the remote differs from what was stored");
    check(other.list()[0].local && other.stats().remote_images == 0, "a fetched image is still remote");
    check(other.stats().remote_downloads == 6, "the fetch did not count its chunks");
}

void test_remote_survives_local_eviction() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore::Options options = remote_options(directory.path(), remote);
    options.max_bytes             = 8000; // room for one image
    ContextStore store(options);
    const auto a = concat({pseudo_random(30, 1), pseudo_random(5000, 2)});
    const auto b = concat({pseudo_random(30, 3), pseudo_random(5000, 4)});
    (void)store.put(describe("0a"), a, std::array{ContextStore::Region{30, 5000}});
    check(drained(store), "upload a");
    remote->clock += 10;
    (void)store.put(describe("0b"), b, std::array{ContextStore::Region{30, 5000}});
    check(drained(store), "upload b");
    check(!store.list().empty() && store.stats().evicted_for_space >= 1, "the local limit evicted nothing");
    // 0a was evicted locally; the remote still holds it and a refresh brings it back.
    (void)store.refresh_remote();
    bool found = false;
    for (const auto& info : store.list()) {
        if (info.id == "0a") { found = true; }
    }
    check(found, "an image evicted locally was not found again through the remote");
    const auto back = store.load("0a");
    check(back && *back == a, "an image evicted locally could not be fetched back");
    check(store.stats().remote_downloads > 0, "the evicted image was not fetched from the remote");
}

void test_remote_damage_and_outage_are_misses() {
    TempDirectory directory;
    TempDirectory other_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before damage");
    }
    // A chunk in the bucket is damaged: the load is a miss and the image is not offered again.
    for (auto& [key, value] : remote->objects) {
        if (key.find("/chunks/") != std::string::npos) {
            value.first[0] ^= 0xff;
            break;
        }
    }
    ContextStore other(remote_options(other_directory.path(), remote));
    (void)other.refresh_remote();
    check(other.list().size() == 1, "the image was not listed before the damage was found");
    check(!other.load("0a").has_value(), "a damaged remote chunk produced an image");
    check(other.list().empty(), "a damaged remote image is still offered");
    check(other.refresh_remote() == 0, "a damaged remote image was imported again");

    // An unreachable bucket costs the upload, never the local write.
    TempDirectory outage_directory;
    auto down = std::make_shared<MemoryObjectStore>();
    down->down = true;
    ContextStore offline(remote_options(outage_directory.path(), down));
    (void)offline.put(describe("0c"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                      std::array{ContextStore::Region{30, 3000}});
    check(drained(offline), "the upload queue did not drain during an outage");
    check(offline.load("0c").has_value(), "an outage lost a locally stored image");
    check(offline.stats().remote_upload_failures >= 1 && offline.stats().remote_uploads == 0,
          "an outage was not counted");
    // The bucket comes back: the next write of the image uploads it all.
    down->down = false;
    (void)offline.put(describe("0c"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                      std::array{ContextStore::Region{30, 3000}});
    check(drained(offline) && offline.stats().remote_uploads == 4, "the recovered upload was incomplete");
}

// A refresh cut short by its deadline registers nothing it did not reach and, because its listing
// is incomplete, removes nothing from the index.
void test_remote_refresh_respects_its_deadline() {
    TempDirectory directory;
    TempDirectory other_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        (void)store.put(describe("0b"), concat({pseudo_random(30, 3), pseudo_random(3000, 4)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before the deadline test");
    }
    ContextStore other(remote_options(other_directory.path(), remote));
    const auto expired = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    check(other.refresh_remote(expired) == 0 && other.list().empty(),
          "a refresh past its deadline still read manifests");
    check(other.refresh_remote() == 2 && other.list().size() == 2, "the images were not registered");
    {
        std::scoped_lock lock(remote->mutex);
        for (auto it = remote->objects.begin(); it != remote->objects.end();) {
            it = it->first.find("0a.manifest") != std::string::npos ? remote->objects.erase(it)
                                                                    : std::next(it);
        }
    }
    // The bucket lost 0a, but a listing cut short by the deadline cannot say so.
    (void)other.refresh_remote(expired);
    check(other.list().size() == 2, "an incomplete listing removed an image from the index");
    (void)other.refresh_remote();
    check(other.list().size() == 1 && other.list()[0].id == "0b",
          "a complete listing did not drop an image the bucket lost");
}

// Several local chunks damaged at their original size are all discarded and fetched again; the
// image still loads.
void test_remote_repairs_several_damaged_local_chunks() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    const std::vector<std::uint8_t> image = concat({pseudo_random(30, 1), pseudo_random(3000, 2)});
    (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "upload before local damage");
    int damaged = 0;
    for (const auto& shard : fs::directory_iterator(directory.path() / "chunks")) {
        if (!shard.is_directory()) { continue; }
        for (const auto& item : fs::directory_iterator(shard.path())) {
            if (item.path().extension() != ".chunk" || damaged == 2) { continue; }
            std::fstream file(item.path(), std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            file.read(&byte, 1);
            byte = static_cast<char>(byte ^ 0xff);
            file.seekp(0);
            file.write(&byte, 1);
            ++damaged;
        }
    }
    check(damaged == 2, "the test did not find two chunks to damage");
    const auto loaded = store.load("0a");
    check(loaded.has_value() && *loaded == image, "damaged local chunks were not repaired together");
    check(store.stats().remote_downloads >= 2, "the damaged chunks were not fetched again");
}

// A corrupt remote copy is quarantined, but a fresh local write of the same id lifts that: once
// the new copy is evicted the bucket's valid manifest can be offered again.
void test_remote_rewrite_lifts_the_quarantine() {
    TempDirectory directory;
    TempDirectory other_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const std::vector<std::uint8_t> image = concat({pseudo_random(30, 1), pseudo_random(3000, 2)});
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before quarantine");
    }
    for (auto& [key, value] : remote->objects) {
        if (key.find("/chunks/") != std::string::npos) {
            value.first[0] ^= 0xff;
            break;
        }
    }
    ContextStore other(remote_options(other_directory.path(), remote));
    (void)other.refresh_remote();
    check(!other.load("0a").has_value() && other.list().empty(), "the damaged image was not quarantined");
    check(other.refresh_remote() == 0, "a quarantined image was imported again");
    (void)other.put(describe("0a"), image, std::array{ContextStore::Region{30, 3000}});
    check(drained(other), "upload of the rewritten image");
    check(other.erase("0a"), "the rewritten image was not held");
    check(other.refresh_remote() == 1, "a rewrite did not lift the quarantine");
}

void test_remote_use_restarts_the_remote_age() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                    std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "upload");
    check(remote->touches == 0, "a fresh upload was touched");
    remote->clock += 2 * 24 * 3600 * 1000LL;
    check(store.load("0a").has_value(), "load after a day");
    check(drained(store), "touch drained");
    check(remote->touches == 4, "using an image did not restart the age of its remote objects");
}


std::size_t chunk_objects(MemoryObjectStore& remote) {
    std::scoped_lock lock(remote.mutex);
    std::size_t count = 0;
    for (const auto& [key, value] : remote.objects) {
        if (key.find("/chunks/") != std::string::npos) { ++count; }
    }
    return count;
}

// A load that does not count as a use (a probe, or a restore that may be abandoned) leaves the age
// of the bucket's copy alone; only a use restarts it.
void test_remote_age_restarts_only_on_use() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                    std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "upload before the age test");
    remote->clock += 2 * 24 * 3600 * 1000LL;
    check(store.load("0a", false).has_value(), "a probing load failed");
    check(drained(store) && remote->touches == 0, "a load that is not a use restarted the remote age");
    store.touch("0a");
    check(drained(store) && remote->touches == 4, "using the image did not restart the remote age");
}

// An image another engine wrote after this one gave up on the id replaces the quarantined copy: the
// next refresh offers it.
void test_remote_quarantine_ends_when_the_manifest_is_rewritten() {
    TempDirectory directory;
    TempDirectory other_directory;
    TempDirectory writer_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before quarantine");
    }
    for (auto& [key, value] : remote->objects) {
        if (key.find("/chunks/") != std::string::npos) {
            value.first[0] ^= 0xff;
            break;
        }
    }
    ContextStore other(remote_options(other_directory.path(), remote));
    (void)other.refresh_remote();
    check(!other.load("0a").has_value() && other.list().empty(), "the damaged image was not quarantined");
    check(other.refresh_remote() == 0, "a quarantined image was imported again");
    // Another engine replaces the image in the bucket.
    remote->clock += 1000;
    const std::vector<std::uint8_t> replacement = concat({pseudo_random(30, 7), pseudo_random(3000, 8)});
    {
        ContextStore writer(remote_options(writer_directory.path(), remote));
        (void)writer.put(describe("0a"), replacement, std::array{ContextStore::Region{30, 3000}});
        check(drained(writer), "upload of the replacement");
    }
    check(other.refresh_remote() == 1, "a rewritten manifest did not end the quarantine");
    const auto loaded = other.load("0a");
    check(loaded.has_value() && *loaded == replacement, "the replacement was not fetched");
}

// An image held entirely inline in its manifest is local; a complete listing that no longer shows it
// does not drop it.
void test_inline_only_image_is_not_remote_only() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    const std::vector<std::uint8_t> inline_image = pseudo_random(100, 9);
    (void)store.put(describe("0c"), inline_image, std::span<const ContextStore::Region>{});
    check(drained(store), "upload of the inline image");
    {
        std::scoped_lock lock(remote->mutex);
        remote->objects.clear();
    }
    (void)store.refresh_remote();
    const auto loaded = store.load("0c");
    check(loaded.has_value() && *loaded == inline_image,
          "a complete listing dropped an image the manifest holds whole");
}

// Registering an image from the bucket's listing proves only its manifest exists: a chunk the bucket
// lost since is put back by the next write that shares it, not skipped as already confirmed.
void test_remote_listing_does_not_confirm_chunks() {
    TempDirectory directory;
    TempDirectory other_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const std::vector<std::uint8_t> shared = pseudo_random(3000, 2);
    {
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), shared}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload before the listing test");
    }
    const std::size_t complete = chunk_objects(*remote);
    ContextStore other(remote_options(other_directory.path(), remote));
    check(other.refresh_remote() == 1, "the image was not registered from the listing");
    {
        std::scoped_lock lock(remote->mutex);
        for (auto it = remote->objects.begin(); it != remote->objects.end(); ++it) {
            if (it->first.find("/chunks/") != std::string::npos) {
                remote->objects.erase(it); // lifecycle expiry took one chunk, the manifest remains
                break;
            }
        }
    }
    check(chunk_objects(*remote) + 1 == complete, "the test did not remove a chunk");
    (void)other.put(describe("0b"), concat({pseudo_random(30, 5), shared}),
                    std::array{ContextStore::Region{30, 3000}});
    check(drained(other), "upload of the image sharing the chunks");
    check(chunk_objects(*remote) == complete,
          "a write sharing a chunk the bucket lost skipped putting it back");
}

// An image written, then evicted or replaced before its upload ran, still reaches the bucket whole.
void test_upload_survives_eviction_and_replacement() {
    TempDirectory directory;
    TempDirectory reader_directory;
    auto remote                   = std::make_shared<MemoryObjectStore>();
    ContextStore::Options options = remote_options(directory.path(), remote);
    options.max_bytes             = 8000; // room for one image
    const auto a                  = concat({pseudo_random(30, 1), pseudo_random(5000, 2)});
    const auto b                  = concat({pseudo_random(30, 3), pseudo_random(5000, 4)});
    const auto a2                 = concat({pseudo_random(30, 5), pseudo_random(5000, 6)});
    {
        ContextStore store(options);
        remote->set_hold(true); // uploads queue up behind a slow link
        (void)store.put(describe("0a"), a, std::array{ContextStore::Region{30, 5000}});
        remote->clock += 10;
        (void)store.put(describe("0b"), b, std::array{ContextStore::Region{30, 5000}});
        check(store.stats().evicted_for_space >= 1, "the limit evicted nothing");
        // A newer write of an image replaces its queued upload.
        remote->clock += 10;
        (void)store.put(describe("0b", 120), a2, std::array{ContextStore::Region{30, 5000}});
        remote->set_hold(false);
        check(drained(store), "uploads did not drain");
        check(store.stats().remote_upload_failures == 0, "an upload failed");
    }
    ContextStore reader(remote_options(reader_directory.path(), remote));
    (void)reader.refresh_remote();
    const auto got_a  = reader.load("0a");
    const auto got_b  = reader.load("0b");
    check(got_a && *got_a == a, "an image evicted before its upload ran never reached the bucket");
    check(got_b && *got_b == a2, "the replacing write's image is not what the bucket holds");
}

// An image registered from the bucket names chunks the directory lacks; writing another image that
// shares them must write them, not count them as already stored.
void test_absent_chunks_are_not_reused() {
    TempDirectory first;
    TempDirectory second;
    auto remote       = std::make_shared<MemoryObjectStore>();
    const auto shared = pseudo_random(4096, 9);
    {
        ContextStore store(remote_options(first.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), shared}),
                        std::array{ContextStore::Region{30, 4096}});
        check(drained(store), "upload a");
    }
    ContextStore store(remote_options(second.path(), remote));
    check(store.refresh_remote() == 1 && !store.list()[0].local, "the image was not registered");
    const auto result = store.put(describe("0c"), concat({pseudo_random(30, 2), shared}),
                                  std::array{ContextStore::Region{30, 4096}});
    check(result.chunk_bytes_reused == 0 && result.chunk_bytes_written == 4096,
          "chunks the directory lacked were counted as reused");
    check(drained(store), "upload c");
    remote->down = true; // everything below must come from the directory
    check(store.load("0c").has_value(), "an image written beside a registered one is not intact");
}

// A chunk damaged on the local disk (same size) is repaired from the bucket.
void test_local_corruption_is_repaired_from_the_remote() {
    TempDirectory directory;
    auto remote      = std::make_shared<MemoryObjectStore>();
    const auto image = concat({pseudo_random(30, 1), pseudo_random(3000, 2)});
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "upload");
    const auto files = chunk_files(directory.path());
    check(!files.empty(), "no chunk files");
    {
        std::fstream file(files.front(), std::ios::binary | std::ios::in | std::ios::out);
        char byte = 0;
        file.read(&byte, 1);
        file.seekp(0);
        byte ^= 0x5a;
        file.write(&byte, 1);
    }
    const auto loaded = store.load("0a");
    check(loaded && *loaded == image, "local corruption was not repaired from the bucket");
    check(store.stats().remote_downloads >= 1 && store.stats().corrupt_removed == 0,
          "the repair did not come from the bucket");
}

// A background fetch that finds the bucket's copy damaged leaves nothing offered.
void test_corrupt_prefetch_is_dropped() {
    TempDirectory first;
    TempDirectory second;
    auto remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore store(remote_options(first.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload");
    }
    for (auto& [key, value] : remote->objects) {
        if (key.find("/chunks/") != std::string::npos) {
            value.first[0] ^= 0xff;
            break;
        }
    }
    ContextStore store(remote_options(second.path(), remote));
    (void)store.refresh_remote();
    check(store.prefetch("0a"), "prefetch refused");
    check(drained(store), "prefetch did not finish");
    check(store.list().empty(), "a damaged prefetched image is still offered");
}

// The bucket forgot a chunk (its lifecycle rule expired it): the next write of an image that uses it
// puts it back, and so does using the image.
void test_expired_remote_chunks_are_put_back() {
    TempDirectory directory;
    auto remote       = std::make_shared<MemoryObjectStore>();
    const auto shared = pseudo_random(4096, 9);
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), concat({pseudo_random(30, 1), shared}),
                    std::array{ContextStore::Region{30, 4096}});
    check(drained(store), "upload a");
    const std::size_t before = chunk_objects(*remote);
    {
        std::scoped_lock lock(remote->mutex);
        for (auto it = remote->objects.begin(); it != remote->objects.end();) {
            it = it->first.find("/chunks/") != std::string::npos ? remote->objects.erase(it)
                                                                  : std::next(it);
        }
    }
    check(chunk_objects(*remote) == 0, "the chunks were not removed");
    remote->clock += 2 * 24 * 3600 * 1000LL; // past the trust in an earlier confirmation
    (void)store.put(describe("0c"), concat({pseudo_random(30, 2), shared}),
                    std::array{ContextStore::Region{30, 4096}});
    check(drained(store), "upload c");
    check(chunk_objects(*remote) == before, "a chunk the bucket lost was not uploaded again");

    // Using an image repairs the bucket the same way.
    {
        std::scoped_lock lock(remote->mutex);
        for (auto it = remote->objects.begin(); it != remote->objects.end();) {
            it = it->first.find("/chunks/") != std::string::npos ? remote->objects.erase(it)
                                                                  : std::next(it);
        }
    }
    remote->clock += 2 * 24 * 3600 * 1000LL;
    check(store.load("0a").has_value(), "load");
    check(drained(store), "touch drained");
    check(chunk_objects(*remote) == before, "using an image did not put back chunks the bucket lost");
}

// This directory's TTL governs its own files: an image only the bucket holds is not hidden by it.
void test_local_ttl_does_not_hide_remote_images() {
    TempDirectory first;
    TempDirectory second;
    auto remote = std::make_shared<MemoryObjectStore>();
    const auto image = concat({pseudo_random(30, 1), pseudo_random(3000, 2)});
    {
        ContextStore store(remote_options(first.path(), remote));
        (void)store.put(describe("0a"), image, std::array{ContextStore::Region{30, 3000}});
        check(drained(store), "upload");
    }
    remote->clock += 3 * 24 * 3600 * 1000LL;
    ContextStore::Options options = remote_options(second.path(), remote);
    options.ttl                   = std::chrono::hours(1);
    ContextStore store(options);
    check(store.refresh_remote() == 1, "an image older than the local TTL was not registered");
    store.maintain();
    const auto loaded = store.load("0a");
    check(loaded && *loaded == image, "an image older than the local TTL could not be used");
    // The bucket drops it (lifecycle): a refresh drops it from the index too, once nothing local backs it.
    TempDirectory third;
    ContextStore other(remote_options(third.path(), remote));
    check(other.refresh_remote() == 1, "register");
    {
        std::scoped_lock lock(remote->mutex);
        remote->objects.clear();
    }
    (void)other.refresh_remote();
    check(other.list().empty(), "an image the bucket no longer lists is still offered");
}

// Shutdown does not wait out a transfer stuck on the network.
void test_shutdown_interrupts_a_blocked_transfer() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const auto started = std::chrono::steady_clock::now();
    {
        remote->set_hold(true);
        ContextStore store(remote_options(directory.path(), remote));
        (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                        std::array{ContextStore::Region{30, 3000}});
        check(!store.flush_remote(std::chrono::steady_clock::now() + std::chrono::milliseconds(200)),
              "a blocked upload reported a drained queue");
    } // the destructor must interrupt the transfer rather than wait for it
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    check(seconds < 10.0, "closing the store waited on a blocked transfer");
}

// The final flush spends its budget on uploads: a queued touch is dropped, not run ahead of them.
void test_final_flush_drops_best_effort_tasks() {
    TempDirectory directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    ContextStore store(remote_options(directory.path(), remote));
    (void)store.put(describe("0a"), concat({pseudo_random(30, 1), pseudo_random(3000, 2)}),
                    std::array{ContextStore::Region{30, 3000}});
    check(drained(store), "first upload");
    remote->clock += 2 * 24 * 3600 * 1000LL;
    remote->set_hold(true);
    (void)store.put(describe("0b"), concat({pseudo_random(30, 3), pseudo_random(3000, 4)}),
                    std::array{ContextStore::Region{30, 3000}});
    store.touch("0a"); // queued behind the blocked upload
    remote->set_hold(false);
    check(store.flush_remote(std::chrono::steady_clock::now() + std::chrono::seconds(10), true),
          "the final flush did not drain");
    check(remote->touches == 0, "a best-effort touch ran during the final flush");
    bool uploaded = false;
    {
        std::scoped_lock lock(remote->mutex);
        for (const auto& [key, value] : remote->objects) {
            uploaded = uploaded || key.find("0b.manifest") != std::string::npos;
        }
    }
    check(uploaded, "the upload queued before the flush was not completed");
}

// Bytes that compress well: a short random pattern repeated.
std::vector<std::uint8_t> compressible(std::size_t size, std::uint32_t seed) {
    const auto pattern = pseudo_random(48, seed);
    std::vector<std::uint8_t> out(size);
    for (std::size_t i = 0; i < size; ++i) { out[i] = pattern[i % pattern.size()]; }
    return out;
}

bool has_suffix(const fs::path& path, const std::string& suffix) {
    const std::string name = path.filename().string();
    return name.size() >= suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::size_t files_with_suffix(const fs::path& directory, const std::string& suffix) {
    std::size_t count = 0;
    for (const auto& file : chunk_files(directory)) { count += has_suffix(file, suffix) ? 1U : 0U; }
    return count;
}

ContextStore::Options compressed_options(const fs::path& directory, std::uint64_t chunk_bytes) {
    ContextStore::Options options = options_for(directory);
    options.compress_chunks       = true;
    options.chunk_bytes           = chunk_bytes;
    return options;
}

std::uint64_t directory_bytes(const fs::path& directory, const std::string& suffix) {
    std::uint64_t total = 0;
    for (const auto& item : fs::recursive_directory_iterator(directory)) {
        if (item.is_regular_file() && has_suffix(item.path(), suffix)) { total += item.file_size(); }
    }
    return total;
}

// Chunks of 1 KiB (a frame per chunk) and of 1 MiB (several 256 KiB stream pieces per frame).
void test_compressed_chunks_round_trip_and_account_for_disk_use() {
    for (const std::uint64_t chunk : {std::uint64_t{1024}, std::uint64_t{1} << 20U}) {
        TempDirectory directory;
        ContextStore store(compressed_options(directory.path(), chunk));
        const std::size_t packed_size = 3 * static_cast<std::size_t>(chunk) + 700;
        const auto image = concat({pseudo_random(64, 1), compressible(packed_size, 2),
                                   pseudo_random(2 * static_cast<std::size_t>(chunk), 3),
                                   pseudo_random(9, 4)});
        const std::array regions{ContextStore::Region{64, packed_size},
                                 ContextStore::Region{64 + packed_size, 2 * chunk}};
        const auto put = store.put(describe("c0de"), image, regions);
        check(put.chunk_bytes_written == packed_size + 2 * chunk,
              "a compressed put changed the raw byte count it reports");
        check(files_with_suffix(directory.path(), ".chunk.zst") == 4,
              "compressible chunks were not stored compressed");
        check(files_with_suffix(directory.path(), ".chunk") == 2,
              "incompressible chunks were not kept raw");
        const auto stats = store.stats();
        check(stats.bytes_stored < stats.bytes_written, "compression saved nothing in the stats");
        check(stats.used_bytes ==
                  directory_bytes(directory.path() / "chunks", "") + directory_bytes(directory.path() / "manifests", ".manifest"),
              "used bytes do not match the files on disk");
        const auto loaded = store.load("c0de");
        check(loaded && *loaded == image, "a compressed image did not round trip");
    }
}

void test_compressed_chunks_survive_restart_and_either_setting() {
    TempDirectory directory;
    const auto image = concat({pseudo_random(64, 1), compressible(5000, 2), pseudo_random(8, 4)});
    {
        ContextStore store(compressed_options(directory.path(), 1024));
        (void)store.put(describe("f1f1"), image, std::array{ContextStore::Region{64, 5000}});
    }
    const std::uint64_t on_disk = directory_bytes(directory.path() / "chunks", "");
    // Reopened with compression off: the scan keeps the compressed chunks and reads them.
    {
        ContextStore reopened(options_for(directory.path()));
        check(files_with_suffix(directory.path(), ".chunk.zst") > 0,
              "the scan swept the compressed chunks");
        check(reopened.stats().used_bytes ==
                  on_disk + directory_bytes(directory.path() / "manifests", ".manifest"),
              "a restart did not count the compressed chunks by their size on disk");
        const auto loaded = reopened.load("f1f1");
        check(loaded && *loaded == image, "a compressed image did not load with compression off");
    }
    // A compressed chunk no manifest names is an orphan like a raw one.
    const fs::path stray = directory.path() / "chunks" / "00" / "00000000000000000000000000000000.chunk.zst";
    fs::create_directories(stray.parent_path());
    { std::ofstream file(stray, std::ios::binary); file << "x"; }
    ContextStore again(options_for(directory.path()));
    check(!fs::exists(stray), "an orphaned compressed chunk survived the scan");
}

void test_compressed_chunk_damage_is_a_miss() {
    const auto image = concat({pseudo_random(64, 1), compressible(4096, 2)});
    const std::array regions{ContextStore::Region{64, 4096}};
    const auto damage = [&](const char* what, auto&& change) {
        TempDirectory directory;
        ContextStore store(compressed_options(directory.path(), 1024));
        (void)store.put(describe("dd01"), image, regions);
        std::vector<fs::path> packed;
        for (const auto& file : chunk_files(directory.path())) {
            if (has_suffix(file, ".chunk.zst")) { packed.push_back(file); }
        }
        std::sort(packed.begin(), packed.end());
        check(packed.size() >= 2, "no compressed chunks to damage");
        change(packed);
        check(!store.load("dd01").has_value(), what);
        check(!store.find("dd01").has_value(), "a damaged image stayed listed");
    };
    damage("a flipped byte in a compressed chunk was served", [](auto& files) {
        std::fstream file(files.front(), std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(static_cast<std::streamoff>(fs::file_size(files.front()) / 2));
        char byte = 0;
        file.seekg(file.tellp());
        file.read(&byte, 1);
        byte ^= 0x5a;
        file.seekp(static_cast<std::streamoff>(fs::file_size(files.front()) / 2));
        file.write(&byte, 1);
    });
    damage("a truncated compressed chunk was served",
           [](auto& files) { fs::resize_file(files.front(), fs::file_size(files.front()) - 3); });
    damage("a compressed chunk with another chunk's bytes was served", [](auto& files) {
        // Same raw length, valid frame, wrong content: only the hash can tell.
        fs::copy_file(files[1], files[0], fs::copy_options::overwrite_existing);
    });
    damage("trailing bytes after a compressed frame were ignored", [](auto& files) {
        std::ofstream file(files.front(), std::ios::binary | std::ios::app);
        file << "junk";
    });
}

void test_chunks_of_both_forms_share_a_store() {
    TempDirectory directory;
    const auto shared = compressible(4096, 2);
    const auto first  = concat({pseudo_random(64, 1), shared});
    {
        ContextStore store(options_for(directory.path())); // raw
        (void)store.put(describe("a001"), first, std::array{ContextStore::Region{64, 4096}});
    }
    ContextStore store(compressed_options(directory.path(), 1024));
    const auto second = concat({pseudo_random(64, 9), shared, compressible(2048, 5)});
    const auto put = store.put(describe("a002", 140), second, std::array{ContextStore::Region{64, 6144}});
    check(put.chunk_bytes_reused == 4096 && put.chunk_bytes_written == 2048,
          "a compressing store rewrote chunks the raw store already held");
    check(files_with_suffix(directory.path(), ".chunk.zst") == 2,
          "only the new chunks were expected to be compressed");
    check(store.load("a001").value_or(std::vector<std::uint8_t>{}) == first &&
              store.load("a002").value_or(std::vector<std::uint8_t>{}) == second,
          "an image mixing raw and compressed chunks did not load");
    check(store.erase("a001") && store.load("a002").value_or(std::vector<std::uint8_t>{}) == second,
          "erasing the image holding the raw chunks broke the other");
}

void test_remote_holds_compressed_objects_and_other_forms_interoperate() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    TempDirectory c_directory;
    auto remote = std::make_shared<MemoryObjectStore>();
    const auto image = concat({pseudo_random(30, 1), compressible(4096, 2), pseudo_random(1024, 3)});
    const std::array regions{ContextStore::Region{30, 5120}};
    ContextStore::Options a_options = remote_options(a_directory.path(), remote);
    a_options.compress_chunks       = true;
    {
        ContextStore a(a_options);
        (void)a.put(describe("0b"), image, regions);
        check(drained(a), "the compressed upload did not drain");
        check(a.stats().remote_upload_failures == 0, "a compressed upload failed");
    }
    std::size_t packed_objects = 0;
    std::size_t raw_objects    = 0;
    for (const auto& [key, value] : remote->objects) {
        packed_objects += has_suffix(key, ".chunk.zst") ? 1U : 0U;
        raw_objects += has_suffix(key, ".chunk") ? 1U : 0U;
    }
    // Four compressible chunks, of which the first and the last are identical, and one random.
    check(packed_objects == 3 && raw_objects == 1, "the bucket does not hold compressed objects");
    // An engine that does not compress fetches them, and finds them when deciding what to upload.
    {
        ContextStore b(remote_options(b_directory.path(), remote));
        check(b.refresh_remote() == 1, "the compressed image was not listed");
        const auto loaded = b.load("0b");
        check(loaded && *loaded == image, "an image fetched in compressed form differs");
        const int before = remote->puts;
        (void)b.put(describe("0c", 140), image, regions); // the same chunks under another id
        check(drained(b), "the second upload did not drain");
        check(remote->puts - before == 1, "chunks the bucket holds compressed were uploaded again");
    }
    // A compressing engine fetches a raw upload.
    auto raw_remote = std::make_shared<MemoryObjectStore>();
    {
        ContextStore d(remote_options(c_directory.path(), raw_remote));
        (void)d.put(describe("0d"), image, regions);
        check(drained(d), "the raw upload did not drain");
    }
    TempDirectory e_directory;
    ContextStore::Options e_options = remote_options(e_directory.path(), raw_remote);
    e_options.compress_chunks       = true;
    ContextStore e(e_options);
    check(e.refresh_remote() == 1, "the raw image was not listed");
    const auto loaded = e.load("0d");
    check(loaded && *loaded == image, "a compressing engine could not fetch raw objects");
    // A damaged compressed object is a corrupt fetch, not a served image.
    TempDirectory f_directory;
    for (auto& [key, value] : remote->objects) {
        if (has_suffix(key, ".chunk.zst")) {
            value.first[value.first.size() / 2] ^= 0x5a;
            break;
        }
    }
    ContextStore f(remote_options(f_directory.path(), remote));
    (void)f.refresh_remote();
    check(!f.load("0b").has_value(), "a damaged compressed object was served");
}

ContextStore::Options peer_options(const fs::path& directory, const fs::path& peer) {
    ContextStore::Options options = options_for(directory);
    options.peers                 = {peer};
    options.peer_poll             = std::chrono::hours(1); // tests call sync_peers() themselves
    return options;
}

std::vector<std::uint8_t> peer_image(std::uint32_t seed, std::size_t region = 5000) {
    return concat({pseudo_random(30, seed), pseudo_random(region, seed + 1000), pseudo_random(8, seed)});
}

void test_owner_lock_refuses_a_second_store_on_one_directory() {
    TempDirectory directory;
    {
        ContextStore first(options_for(directory.path()));
        bool refused = false;
        try {
            ContextStore second(options_for(directory.path()));
        } catch (const std::runtime_error& failure) {
            refused = std::string(failure.what()).find("in use") != std::string::npos;
        }
        check(refused, "a second store on a directory in use was opened");
        (void)first.put(describe("0a"), peer_image(1), std::array{ContextStore::Region{30, 5000}});
    }
    ContextStore again(options_for(directory.path()));
    check(again.list().size() == 1, "a store did not open after the first owner closed");
    TempDirectory other;
    bool threw = false;
    try {
        ContextStore self(peer_options(other.path(), other.path()));
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a store was accepted as its own peer");
}

void test_peer_import_links_chunks_and_survives_the_owner() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = peer_image(1);
    const std::array regions{ContextStore::Region{30, 5000}};
    ContextStore a(options_for(a_directory.path()));
    (void)a.put(describe("0a"), image, regions);
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    check(b.list().empty(), "a store listed images before it looked at its peer");
    check(b.sync_peers() == 1 && b.sync_peers() == 0, "a peer image was not imported exactly once");
    const auto listed = b.list();
    check(listed.size() == 1 && listed[0].local && !listed[0].checkpoints.empty(),
          "an imported image is not an ordinary local image");
    const auto loaded = b.load("0a");
    check(loaded && *loaded == image, "an imported image differs from what the peer stored");
    const auto stats = b.stats();
    check(stats.peer_images == 1 && stats.peer_bytes_linked == 5000 && stats.peer_skipped == 0,
          "peer import statistics are wrong");
    // The chunk files are one file under two names, not a copy.
    for (const auto& file : chunk_files(a_directory.path())) {
        check(fs::hard_link_count(file) == 2, "an imported chunk is a copy, not a hard link");
    }
    check(chunk_files(a_directory.path()).size() == chunk_files(b_directory.path()).size(),
          "the importing store holds other chunks than the peer");
    // The peer removing its image does not touch the imported one.
    check(a.erase("0a"), "the peer could not erase its image");
    check(b.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
          "an imported image stopped loading when the peer erased it");
    for (const auto& file : chunk_files(b_directory.path())) {
        check(fs::hard_link_count(file) == 1, "an erased peer left links behind");
    }
    // The importer erasing its image does not touch the peer's.
    (void)a.put(describe("0b"), peer_image(2), regions);
    check(b.sync_peers() == 1 && b.erase("0b") && a.load("0b").has_value(),
          "erasing an imported image damaged the peer's copy");
}

void test_peer_import_survives_restarts_of_either_store() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = peer_image(3);
    const std::array regions{ContextStore::Region{30, 5000}};
    {
        ContextStore a(options_for(a_directory.path()));
        (void)a.put(describe("0a"), image, regions);
        {
            ContextStore b(peer_options(b_directory.path(), a_directory.path()));
            check(b.sync_peers() == 1, "the image was not imported");
        }
        // B's start-up scan keeps what it imported (it has its own manifest copy) and A's files stay.
        ContextStore b(peer_options(b_directory.path(), a_directory.path()));
        check(b.list().size() == 1 && b.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
              "a restart lost an imported image");
        check(b.sync_peers() == 0, "a restart imported an image it already held");
        check(a.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
              "the peer's image changed when the importer restarted");
    }
    // The peer restarting and sweeping its directory does not touch the importer's copy.
    ContextStore a(options_for(a_directory.path()));
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    check(a.erase("0a") && b.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
          "an imported image depended on the peer's files");
}

void test_both_stores_write_the_same_image() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = peer_image(4);
    const std::array regions{ContextStore::Region{30, 5000}};
    ContextStore a(peer_options(a_directory.path(), b_directory.path()));
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    (void)a.put(describe("0a"), image, regions);
    (void)b.put(describe("0a"), image, regions);
    check(a.sync_peers() == 0 && b.sync_peers() == 0, "an image both stores hold was imported");
    check(a.load("0a").value_or(std::vector<std::uint8_t>{}) == image &&
              b.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
          "two stores writing one image broke one of them");
}

void test_background_poll_imports_without_being_asked() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    ContextStore a(options_for(a_directory.path()));
    ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
    options.peer_poll             = std::chrono::milliseconds(5);
    ContextStore b(options);
    (void)a.put(describe("0a"), peer_image(5), std::array{ContextStore::Region{30, 5000}});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (b.list().empty() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(b.list().size() == 1, "the background poll did not import the peer's image");
}

// The peer erases an image while the other store imports it, over and over: whatever the importer
// ends up holding must be a whole image that loads as what was written, never a partial one.
void test_peer_erasing_during_import_never_leaves_a_partial_image() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    constexpr int kRounds = 300;
    ContextStore a(options_for(a_directory.path()));
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    std::atomic<bool> done{false};
    std::atomic<int> imported{0};
    std::atomic<int> wrong{0};
    std::thread importer([&] {
        while (!done.load()) {
            (void)b.sync_peers();
            for (const auto& info : b.list()) {
                // Rounds put "r<i>" as the hex id of i; the expected bytes follow from the id.
                const auto bytes = b.load(info.id);
                if (bytes) {
                    const std::uint32_t seed = static_cast<std::uint32_t>(std::stoul(info.id, nullptr, 16));
                    if (*bytes != peer_image(seed, 3000 + seed % 7 * 500)) { ++wrong; }
                    ++imported;
                }
                (void)b.erase(info.id);
            }
        }
    });
    for (int round = 1; round <= kRounds; ++round) {
        char id[16];
        std::snprintf(id, sizeof(id), "%x", static_cast<unsigned>(round));
        const std::size_t region = 3000 + static_cast<std::size_t>(round) % 7 * 500;
        (void)a.put(describe(id), peer_image(static_cast<std::uint32_t>(round), region),
                    std::array{ContextStore::Region{30, region}});
        if (round % 2 == 0) {
            // Even rounds let the importer finish first (so imports certainly happen); odd rounds
            // erase at once, racing the import.
            const int before = imported.load();
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            while (imported.load() == before && std::chrono::steady_clock::now() < give_up) {
                std::this_thread::yield();
            }
        }
        (void)a.erase(id);
    }
    done = true;
    importer.join();
    check(wrong.load() == 0, "an image imported while its peer erased it differs from what was written");
    check(imported.load() >= kRounds / 4 && b.stats().peer_images >= static_cast<std::uint64_t>(imported.load()),
          "the importer imported too few images for the race to mean anything");
    for (const auto& info : b.list()) {
        (void)b.erase(info.id);
    }
    check(chunk_files(b_directory.path()).empty(), "an import left chunks nobody references");
}

void test_damaged_peer_files_are_skipped_or_missed_and_never_modified() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = peer_image(6);
    const std::array regions{ContextStore::Region{30, 5000}};
    ContextStore a(options_for(a_directory.path()));
    (void)a.put(describe("0a"), image, regions);
    const fs::path manifest = a_directory.path() / "manifests" / "0a.manifest";
    const auto manifest_size = fs::file_size(manifest);
    fs::resize_file(manifest, manifest_size / 2);
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    check(b.sync_peers() == 0 && b.stats().peer_skipped == 1, "a damaged peer manifest was imported");
    check(b.sync_peers() == 0 && b.stats().peer_skipped == 1,
          "a damaged peer manifest was looked at again");
    check(fs::file_size(manifest) == manifest_size / 2, "the importer modified the peer's manifest");
    // The peer rewrites the image: the new manifest is tried again.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    (void)a.put(describe("0a"), image, regions);
    check(b.sync_peers() == 1, "a rewritten peer manifest was not tried again");
    check(b.load("0a").value_or(std::vector<std::uint8_t>{}) == image, "the rewritten image differs");

    // A chunk damaged on the peer's disk (the same inode the importer links) is a miss for the
    // importer, and its removal leaves the peer's file alone.
    (void)b.erase("0a");
    (void)a.erase("0a"); // only the image whose chunk is damaged remains
    (void)a.put(describe("0c"), peer_image(7), regions);
    const auto files = chunk_files(a_directory.path());
    fs::path victim;
    for (const auto& file : files) {
        if (fs::file_size(file) == 1024) { victim = file; break; }
    }
    check(!victim.empty(), "no full chunk to damage");
    {
        std::fstream file(victim, std::ios::in | std::ios::out | std::ios::binary);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0x33;
        file.seekp(0);
        file.write(&byte, 1);
    }
    check(b.sync_peers() == 1, "an image with a damaged chunk was not imported (size is intact)");
    check(!b.load("0c").has_value() && !b.find("0c").has_value(),
          "a chunk damaged on the peer's disk was served");
    check(fs::exists(victim), "the importer removed the peer's damaged chunk file");
}

void test_peer_images_over_budget_expired_or_gone_are_skipped_once() {
    const std::array regions{ContextStore::Region{30, 5000}};
    {   // Over the importer's budget.
        TempDirectory a_directory;
        TempDirectory b_directory;
        ContextStore a(options_for(a_directory.path()));
        (void)a.put(describe("0a"), peer_image(8), regions);
        ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
        options.max_bytes             = 3000;
        ContextStore b(options);
        for (int i = 0; i < 4; ++i) { check(b.sync_peers() == 0, "an over-budget image was imported"); }
        check(b.stats().peer_skipped == 1 && chunk_files(b_directory.path()).empty(),
              "an over-budget image was retried or left files behind");
    }
    {   // Older than the importer's TTL.
        TempDirectory a_directory;
        TempDirectory b_directory;
        ContextStore a(options_for(a_directory.path()));
        (void)a.put(describe("0a"), peer_image(9), regions);
        ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
        options.ttl                   = std::chrono::hours(1);
        const std::int64_t created    = a.list()[0].created_ms;
        options.clock                 = [created] { return created + 2 * 3600 * 1000; };
        ContextStore b(options);
        check(b.sync_peers() == 0 && b.sync_peers() == 0 && b.stats().peer_skipped == 1,
              "an expired image was imported or retried");
    }
    {   // A chunk the peer already evicted: the link fails, nothing is left behind.
        TempDirectory a_directory;
        TempDirectory b_directory;
        ContextStore a(options_for(a_directory.path()));
        (void)a.put(describe("0a"), peer_image(10), regions);
        const auto files = chunk_files(a_directory.path());
        check(files.size() >= 2, "no chunks to remove");
        fs::remove(files.front());
        ContextStore b(peer_options(b_directory.path(), a_directory.path()));
        check(b.sync_peers() == 0 && b.stats().peer_link_failures == 1 && b.list().empty() &&
                  chunk_files(b_directory.path()).empty(),
              "an image with a missing peer chunk was imported or left links behind");
    }
    {   // No peer directory yet: not an error.
        TempDirectory b_directory;
        ContextStore b(peer_options(b_directory.path(), b_directory.path().parent_path() / "no-such-store"));
        check(b.sync_peers() == 0, "a missing peer directory imported something");
    }
}

void test_peer_import_links_compressed_chunks() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = concat({pseudo_random(30, 1), compressible(4096, 2), pseudo_random(8, 3)});
    ContextStore a(compressed_options(a_directory.path(), 1024));
    (void)a.put(describe("0a"), image, std::array{ContextStore::Region{30, 4096}});
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    check(b.sync_peers() == 1, "an image of compressed chunks was not imported");
    check(files_with_suffix(b_directory.path(), ".chunk.zst") ==
              files_with_suffix(a_directory.path(), ".chunk.zst") &&
              files_with_suffix(b_directory.path(), ".chunk.zst") > 0,
          "the importer did not link the compressed chunk files");
    check(b.load("0a").value_or(std::vector<std::uint8_t>{}) == image,
          "an image of compressed chunks differs after import");
    check(b.stats().used_bytes == a.stats().used_bytes,
          "the importer counts compressed chunks by another size than the peer");
}

// The TTL counts from the peer's last use, not from creation, and the copy keeps that use time.
void test_peer_import_ttl_counts_from_the_peers_last_use() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const std::int64_t hour = 3600 * 1000;
    std::int64_t a_now      = 10 * hour;
    ContextStore::Options a_options = options_for(a_directory.path());
    a_options.clock                 = [&] { return a_now; };
    ContextStore a(a_options);
    (void)a.put(describe("0a"), peer_image(11), std::array{ContextStore::Region{30, 5000}});
    a_now += 90 * 60 * 1000;
    check(a.load("0a").has_value(), "the peer could not load its own image");
    ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
    options.ttl                   = std::chrono::hours(1);
    options.clock                 = [&] { return 10 * hour + 2 * hour; }; // created 2 h ago, used 30 min ago
    ContextStore b(options);
    check(b.sync_peers() == 1 && b.stats().peer_skipped == 0,
          "an image the peer used recently was skipped as expired");
    check(b.list().size() == 1 && b.list()[0].last_used_ms == 10 * hour + 90 * 60 * 1000,
          "the imported image does not carry the peer's use time");
}

// The fit check counts the compressed files the links will occupy, not the raw chunk lengths.
void test_peer_import_budget_counts_compressed_sizes() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    const auto image = concat({pseudo_random(30, 1), compressible(8192, 2), pseudo_random(8, 3)});
    ContextStore a(compressed_options(a_directory.path(), 1024));
    (void)a.put(describe("0a"), image, std::array{ContextStore::Region{30, 8192}});
    const std::uint64_t packed = a.stats().used_bytes;
    check(packed < image.size(), "the test image did not compress");
    ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
    options.max_bytes             = packed + 16;
    ContextStore b(options);
    check(b.sync_peers() == 1 && b.stats().peer_skipped == 0,
          "an image that fits compressed was skipped for its raw size");
}

// A same-size bit flip passes the import's length check; the load that finds it must leave the
// unchanged peer manifest skipped, not imported again at every poll.
void test_damaged_imported_image_is_not_imported_again() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    ContextStore a(options_for(a_directory.path()));
    ContextStore b(peer_options(b_directory.path(), a_directory.path()));
    (void)a.put(describe("0a"), peer_image(6), std::array{ContextStore::Region{30, 5000}});
    check(b.sync_peers() == 1, "the image was not imported");
    const auto files = chunk_files(b_directory.path());
    check(!files.empty(), "the importer holds no chunk files");
    {
        std::fstream file(files.front(), std::ios::in | std::ios::out | std::ios::binary);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0x5a;
        file.seekp(0);
        file.write(&byte, 1);
    }
    check(!b.load("0a").has_value(), "a damaged imported image was served");
    check(b.sync_peers() == 0 && b.list().empty(), "a damaged image was imported again");
}

// An engine with both a bucket and a peer: the bucket listing registers the image first, by name
// only. The peer's copy must still be linked, so the load reads the disk instead of the network.
struct PeerAndRemote {
    TempDirectory a_directory;
    TempDirectory b_directory;
    std::shared_ptr<MemoryObjectStore> remote = std::make_shared<MemoryObjectStore>();
    std::vector<std::uint8_t> image           = peer_image(12);
    std::unique_ptr<ContextStore> a;

    PeerAndRemote() {
        a = std::make_unique<ContextStore>(remote_options(a_directory.path(), remote));
        (void)a->put(describe("0a"), image, std::array{ContextStore::Region{30, 5000}});
        check(drained(*a), "the peer's upload did not drain");
    }
    [[nodiscard]] std::unique_ptr<ContextStore> importer() const {
        ContextStore::Options options = remote_options(b_directory.path(), remote);
        options.peers                 = {a_directory.path()};
        options.peer_poll             = std::chrono::hours(1);
        auto b = std::make_unique<ContextStore>(options);
        (void)b->refresh_remote(); // the start-up refresh runs in the background
        check(b->list().size() == 1 && !b->list()[0].local,
              "the bucket listing did not register the image by name only");
        return b;
    }
};

void test_peer_fills_an_image_the_remote_registered() {
    PeerAndRemote setup;
    const auto b = setup.importer();
    check(b->sync_peers() == 1, "the peer's copy of a bucket-registered image was not linked");
    check(b->list().size() == 1 && b->list()[0].local, "the linked image does not report itself local");
    check(b->stats().peer_images == 1 && b->stats().peer_bytes_linked == 5000,
          "the fill was not counted as a peer import");
    check(b->stats().used_bytes == setup.a->stats().used_bytes,
          "the filled image is counted by other sizes than the peer's");
    setup.remote->chunk_gets = 0;
    check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image,
          "the filled image differs from what the peer stored");
    check(setup.remote->chunk_gets == 0 && b->stats().remote_downloads == 0,
          "the load downloaded chunks the peer had already supplied");
    for (const fs::path& file : chunk_files(setup.b_directory.path())) {
        check(fs::hard_link_count(file) == 2, "a filled chunk is not a link to the peer's file");
    }
    check(b->sync_peers() == 0 && b->stats().peer_images == 1, "a filled image was linked again");
}

// An image already complete here (fetched from the bucket) is not linked from the peer.
void test_peer_skips_an_image_already_fetched_from_the_remote() {
    PeerAndRemote setup;
    const auto b = setup.importer();
    check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image,
          "the image did not load from the bucket");
    check(b->stats().remote_downloads > 0, "the load did not download");
    check(b->sync_peers() == 0 && b->stats().peer_images == 0 && b->stats().peer_skipped == 0,
          "an image already held whole was imported from the peer");
    for (const fs::path& file : chunk_files(setup.b_directory.path())) {
        check(fs::hard_link_count(file) == 1, "a fetched chunk was replaced by a peer link");
    }
}

// A peer chunk of the wrong size is skipped once (the bucket supplies the image), and a same-size
// bit flip is repaired from the bucket by the load; neither is linked again at the next poll.
void test_damaged_peer_chunk_falls_back_to_the_remote_once() {
    {
        PeerAndRemote setup;
        const auto victim = chunk_files(setup.a_directory.path()).front();
        fs::resize_file(victim, fs::file_size(victim) / 2);
        const auto b = setup.importer();
        check(b->sync_peers() == 0 && b->stats().peer_link_failures == 1 &&
                  b->stats().peer_skipped == 1,
              "a peer chunk of the wrong size was linked or not counted");
        check(b->sync_peers() == 0 && b->stats().peer_link_failures == 1,
              "the damaged peer image was looked at again");
        check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image &&
                  b->stats().remote_downloads > 0,
              "the image did not load from the bucket after the peer copy was skipped");
    }
    {
        PeerAndRemote setup;
        const auto b = setup.importer();
        const auto victim = chunk_files(setup.a_directory.path()).front();
        {
            std::fstream file(victim, std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            file.read(&byte, 1);
            byte ^= 0x5a;
            file.seekp(0);
            file.write(&byte, 1);
        }
        check(b->sync_peers() == 1, "an intact-size peer chunk was not linked");
        check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image,
              "a damaged peer chunk was served or not repaired from the bucket");
        check(b->stats().remote_downloads == 1, "the repair did not download only the damaged chunk");
        check(b->sync_peers() == 0 && b->stats().peer_images == 1,
              "the damaged peer image was linked again");
        check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image,
              "the repaired image did not load again");
    }
}

void flip_first_byte(const fs::path& file) {
    std::fstream stream(file, std::ios::in | std::ios::out | std::ios::binary);
    char byte = 0;
    stream.read(&byte, 1);
    byte ^= 0x5a;
    stream.seekp(0);
    stream.write(&byte, 1);
}

// Removes the bucket's copy of the chunk `file` holds, so a repair of it cannot succeed.
void remove_remote_chunk(MemoryObjectStore& remote, const fs::path& file) {
    const std::string hex = file.filename().string().substr(0, 32);
    std::erase_if(remote.objects, [&](const auto& object) {
        return object.first.find("chunks/") != std::string::npos &&
               object.first.find(hex) != std::string::npos;
    });
}

// The image the bucket registered shares its first two chunks with one the importer wrote itself,
// so the peer fills only the rest. A damaged chunk blames the peer only when the peer linked it.
void test_damaged_chunk_blames_the_peer_only_when_the_peer_supplied_it() {
    for (const bool peer_supplied : {false, true}) {
        PeerAndRemote setup;
        const auto b = setup.importer();
        const std::vector<std::uint8_t> region(setup.image.begin() + 30,
                                               setup.image.begin() + 30 + 2048);
        (void)b->put(describe("0b"), concat({pseudo_random(30, 99), region, pseudo_random(8, 99)}),
                     std::array{ContextStore::Region{30, 2048}});
        check(drained(*b), "the importer's own upload did not drain");
        check(b->sync_peers() == 1 && b->stats().peer_bytes_linked == 5000 - 2048,
              "the peer did not fill only the chunks the importer lacked");
        // Own chunks are single files; the peer's are links to the peer's files.
        fs::path victim;
        for (const fs::path& file : chunk_files(setup.b_directory.path())) {
            if ((fs::hard_link_count(file) == 2) == peer_supplied) {
                victim = file;
                break;
            }
        }
        check(!victim.empty(), "no chunk of the wanted origin to damage");
        // A linked chunk is the peer's file too; an own chunk leaves the peer's copy intact.
        flip_first_byte(victim);
        remove_remote_chunk(*setup.remote, victim);
        check(!b->load("0a").has_value(), "an image with an unrepairable damaged chunk was served");
        if (peer_supplied) {
            check(b->sync_peers() == 0, "a peer whose chunk was damaged was linked from again");
        } else {
            check(b->sync_peers() == 1,
                  "a damaged chunk the peer never supplied stopped the peer being asked again");
            check(b->load("0a").value_or(std::vector<std::uint8_t>{}) == setup.image,
                  "the peer's intact copy did not complete the image");
        }
    }
}

// An image imported whole from a peer is the peer's to upload: using it here must not put it in
// the bucket. An image the bucket registered and a peer filled is still repaired from here.
void test_peer_images_are_not_uploaded_by_the_importer() {
    {
        TempDirectory a_directory;
        TempDirectory b_directory;
        const auto remote = std::make_shared<MemoryObjectStore>();
        const auto image  = peer_image(13);
        ContextStore a(options_for(a_directory.path())); // the peer has not uploaded it (yet)
        (void)a.put(describe("0a"), image, std::array{ContextStore::Region{30, 5000}});
        ContextStore::Options options = remote_options(b_directory.path(), remote);
        options.peers                 = {a_directory.path()};
        options.peer_poll             = std::chrono::hours(1);
        ContextStore b(options);
        check(b.sync_peers() == 1, "the peer image was not imported");
        // As the engine restores: a load without touch, then touch() once it is used.
        check(b.load("0a", false).value_or(std::vector<std::uint8_t>{}) == image,
              "the imported image did not load");
        b.touch("0a");
        check(b.load("0a").has_value(), "the imported image did not load with a touch");
        check(drained(b), "the importer's remote queue did not drain");
        check(remote->puts == 0 && remote->objects.empty() && b.stats().remote_uploads == 0,
              "the importer uploaded an image a peer wrote");
    }
    {
        PeerAndRemote setup;
        const auto b = setup.importer();
        check(b->sync_peers() == 1, "the peer did not fill the bucket-registered image");
        std::erase_if(setup.remote->objects, [](const auto& object) {
            return object.first.find("chunks/") != std::string::npos;
        });
        check(b->load("0a").has_value() && drained(*b), "the filled image did not load");
        check(b->stats().remote_uploads > 0,
              "an image the bucket registered was not repaired there");
    }
}

void test_non_positive_peer_poll_is_refused() {
    TempDirectory a_directory;
    TempDirectory b_directory;
    ContextStore a(options_for(a_directory.path()));
    ContextStore::Options options = peer_options(b_directory.path(), a_directory.path());
    options.peer_poll             = std::chrono::milliseconds(0);
    bool threw = false;
    try {
        ContextStore b(options);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a zero peer poll interval was accepted");
}

} // namespace

int main() {
    test_owner_lock_refuses_a_second_store_on_one_directory();
    test_peer_import_links_chunks_and_survives_the_owner();
    test_peer_import_survives_restarts_of_either_store();
    test_both_stores_write_the_same_image();
    test_background_poll_imports_without_being_asked();
    test_peer_erasing_during_import_never_leaves_a_partial_image();
    test_damaged_peer_files_are_skipped_or_missed_and_never_modified();
    test_peer_images_over_budget_expired_or_gone_are_skipped_once();
    test_peer_import_links_compressed_chunks();
    test_peer_import_ttl_counts_from_the_peers_last_use();
    test_peer_import_budget_counts_compressed_sizes();
    test_damaged_imported_image_is_not_imported_again();
    test_non_positive_peer_poll_is_refused();
    test_peer_fills_an_image_the_remote_registered();
    test_peer_skips_an_image_already_fetched_from_the_remote();
    test_damaged_peer_chunk_falls_back_to_the_remote_once();
    test_damaged_chunk_blames_the_peer_only_when_the_peer_supplied_it();
    test_peer_images_are_not_uploaded_by_the_importer();
    test_compressed_chunks_round_trip_and_account_for_disk_use();
    test_compressed_chunks_survive_restart_and_either_setting();
    test_compressed_chunk_damage_is_a_miss();
    test_chunks_of_both_forms_share_a_store();
    test_remote_holds_compressed_objects_and_other_forms_interoperate();
    test_hash();
    test_round_trip();
    test_incremental_put_writes_only_new_chunks();
    test_restart_recovers_images();
    test_corruption_is_a_miss_and_removed();
    test_space_limit_evicts_least_recently_used();
    test_ttl_and_use_extend_life();
    test_supersede_and_replace();
    test_shared_prefix_is_kept_and_flagged();
    test_invalid_input_and_failed_put();
    test_remote_upload_and_second_engine();
    test_remote_survives_local_eviction();
    test_remote_damage_and_outage_are_misses();
    test_remote_use_restarts_the_remote_age();
    test_remote_refresh_respects_its_deadline();
    test_remote_repairs_several_damaged_local_chunks();
    test_remote_rewrite_lifts_the_quarantine();
    test_remote_listing_does_not_confirm_chunks();
    test_remote_age_restarts_only_on_use();
    test_remote_quarantine_ends_when_the_manifest_is_rewritten();
    test_inline_only_image_is_not_remote_only();
    test_upload_survives_eviction_and_replacement();
    test_absent_chunks_are_not_reused();
    test_local_corruption_is_repaired_from_the_remote();
    test_corrupt_prefetch_is_dropped();
    test_expired_remote_chunks_are_put_back();
    test_local_ttl_does_not_hide_remote_images();
    test_shutdown_interrupts_a_blocked_transfer();
    test_final_flush_drops_best_effort_tasks();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
