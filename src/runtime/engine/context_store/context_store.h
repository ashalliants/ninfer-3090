#pragma once

#include "ninfer/object_store.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ninfer::runtime {

// A durable, bounded, incremental store for the byte images of retained sessions.
//
// A session image is several GB for a deep context, and consecutive images of one conversation share
// almost all of it: the older KV pages and the immutable checkpoint states do not change between
// saves, only the newest pages and the endpoint state do. Writing the whole image every time costs
// minutes on a slow disk, so the store splits the declared regions of an image into fixed-size
// chunks addressed by a hash of their content and writes only the chunks it does not already hold.
// Everything outside the regions (the header with the token ledger and identity, and the integrity
// trailer) is small and travels inside the manifest.
//
// The store knows nothing about the image's meaning. The caller declares the regions and the
// checkpoint keys by which a later request finds the image, and decides whether bytes it gets back
// belong to the running model (the image carries its own binding and checksum, which the restore
// validates).
//
// Everything is safe against a crash at any point: chunks are written under a temporary name and
// renamed, the manifest is renamed last, and a start-up scan removes what no manifest names. A
// failure on any single file turns into a miss, never into wrong data.
class ContextStore {
public:
    using Clock = std::function<std::int64_t()>; // Unix milliseconds

    struct Options {
        std::filesystem::path directory;
        // Bytes of chunks and manifests kept; the least recently used images are evicted beyond it.
        // 0 means no limit. Files queued uploads still need count too: when evicting cannot free
        // enough, the oldest queued uploads are given up so their files can go.
        std::uint64_t max_bytes = 0;
        // The volume is shared (another engine, the OS), so the limit above does not keep it from
        // filling. Before a put writes new chunk files it checks that the volume keeps this much
        // free after them; if not, the least recently used images go first, and a put that still
        // does not fit is refused with an error saying the volume is full.
        std::uint64_t reserve_bytes = std::uint64_t{1} << 30U;
        // The bytes free on the directory's volume; defaults to std::filesystem::space.
        std::function<std::uint64_t()> available_bytes;
        // Images not used for this long are removed. Zero means no expiry.
        std::chrono::seconds ttl{0};
        std::uint64_t chunk_bytes = std::uint64_t{32} << 20U;
        // Write new chunks zstd-compressed when that saves at least 3 % (otherwise raw), locally and
        // in the remote. A chunk keeps its raw-content hash and length either way, so manifests do
        // not change, and chunks of both forms are always read whatever this says.
        bool compress_chunks = false;
        int compression_level = 3;
        Clock clock; // defaults to the system clock

        // An optional remote copy of the whole store (an S3-compatible bucket). The directory then
        // acts as a cache of it: every image written is uploaded in the background, images other
        // engines wrote appear in list() and are fetched when loaded, and an image evicted from
        // the directory stays available remotely. The remote is never trusted for correctness:
        // every chunk it returns is verified, and a failure is a miss. Expiry there is the
        // bucket's lifecycle rule; images that are used have their age restarted.
        std::shared_ptr<ObjectStore> remote;
        std::string remote_prefix; // prepended to every key, e.g. "ninfer/"
        std::chrono::seconds remote_refresh{60};
        std::chrono::seconds remote_touch_interval{24 * 3600};
        // How long a written image waits before it is uploaded. A conversation is written after
        // every turn, and each write supersedes the one before: a write superseded while it waits
        // is never uploaded, so a burst of turns costs one upload. Waits whose image is superseded
        // pass their start on, so a conversation that keeps going is still uploaded this often.
        // flush_remote() uploads without waiting.
        std::chrono::seconds remote_upload_delay{0};

        // Directories of other engines' stores on the same volume, read but never written. Every
        // `peer_poll` the manifests there that this store does not hold are imported by hard-linking
        // their chunks (no bytes are copied), so an image another engine finished is usable here
        // within one poll. An image the remote listing registered but whose chunks are not fetched
        // yet takes the peer's chunks the same way, instead of downloading them. A peer deleting or
        // evicting an image afterwards does not affect the imported copy; an image deleted while
        // it is imported is skipped. Failures are misses.
        std::vector<std::filesystem::path> peers;
        std::chrono::milliseconds peer_poll{250};
    };

    // Where in the image the large, deduplicated bytes are.
    struct Region {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
    };

    // The key a later request computes for a prefix of its own prompt: if it matches, the image
    // holds a checkpoint at exactly that prefix.
    struct CheckpointKey {
        std::uint32_t frontier = 0;
        std::array<std::uint64_t, 2> digests{};
        std::uint32_t identity_tag = 0;

        [[nodiscard]] friend constexpr bool operator==(const CheckpointKey&,
                                                       const CheckpointKey&) noexcept = default;
    };

    struct Description {
        std::string id;      // lowercase hex, 1-64 characters
        std::string binding; // identity of the model and configuration that produced the image
        std::uint32_t tokens = 0;
        // The first entry is the image's endpoint (the deepest state it can resume from).
        std::vector<CheckpointKey> checkpoints;
        // The image's own prefix digest at every token frontier 0..tokens (not stored). An older
        // image of the same model whose endpoint key this reproduces is an earlier state of the
        // same conversation: this image contains everything it does, so once this one is durable
        // the older one is removed instead of being kept as a near-duplicate until it ages out.
        std::span<const std::array<std::uint64_t, 2>> prefix_digests;
        // A prefix many conversations share (a system or tools prompt) rather than one
        // conversation: restored before conversations when not everything fits.
        bool shared = false;
    };

    struct Info {
        std::string id;
        std::string binding;
        std::uint32_t tokens        = 0;
        std::uint64_t image_bytes   = 0;
        std::int64_t created_ms     = 0;
        std::int64_t last_used_ms   = 0;
        std::vector<CheckpointKey> checkpoints;
        // Every chunk of the image is in the directory, so load() does not touch the network.
        bool local  = true;
        // Bytes load() would download first: the chunks the directory lacks (0 when local).
        std::uint64_t fetch_bytes = 0;
        bool shared = false;
    };

    struct PutResult {
        std::uint64_t image_bytes         = 0;
        std::uint64_t chunk_bytes_written = 0; // new chunk files
        std::uint64_t chunk_bytes_reused  = 0; // chunks the store already held
        std::uint64_t evicted_images      = 0;
    };

    struct Stats {
        std::uint64_t used_bytes        = 0;
        std::uint64_t images            = 0;
        std::uint64_t puts              = 0;
        std::uint64_t put_failures      = 0;
        std::uint64_t bytes_written     = 0; // raw bytes of new chunks
        std::uint64_t bytes_stored      = 0; // what they took on disk after compression
        std::uint64_t bytes_reused      = 0;
        std::uint64_t loads             = 0;
        std::uint64_t load_misses       = 0;
        std::uint64_t corrupt_removed   = 0;
        // corrupt_removed by what was found: a chunk missing at start-up (no bucket to fetch it
        // from), a manifest that does not parse or does not describe its image, a chunk whose bytes
        // do not match (no bucket to repair it), or the bucket's copy missing or damaged too.
        std::uint64_t corrupt_startup   = 0;
        std::uint64_t corrupt_manifest  = 0;
        std::uint64_t corrupt_chunk     = 0;
        std::uint64_t corrupt_remote    = 0;
        std::uint64_t evicted_for_space = 0;
        std::uint64_t expired           = 0;
        std::uint64_t superseded        = 0;
        std::uint64_t remote_removed    = 0; // images a remote refresh found gone
        // With a remote: images only the remote holds, and the traffic to it.
        std::uint64_t remote_images          = 0;
        std::uint64_t remote_uploads         = 0; // objects uploaded (chunks and manifests)
        std::uint64_t remote_upload_bytes    = 0;
        std::uint64_t remote_upload_failures = 0;
        std::uint64_t remote_downloads       = 0;
        std::uint64_t remote_download_bytes  = 0;
        std::uint64_t remote_download_failures = 0;
        // remote_download_failures by cause: a transfer that threw, an object the bucket did not
        // have, one whose bytes did not match, a chunk that could not be installed here (a full
        // disk), and a listing, or a manifest read of one, that failed.
        std::uint64_t remote_download_failures_transfer = 0;
        std::uint64_t remote_download_failures_missing  = 0;
        std::uint64_t remote_download_failures_damaged  = 0;
        std::uint64_t remote_download_failures_install  = 0;
        std::uint64_t remote_download_failures_listing  = 0;
        std::uint64_t remote_refreshes       = 0;
        // Manifests a refresh downloaded to register images (not counted in remote_downloads).
        std::uint64_t remote_manifest_downloads      = 0;
        std::uint64_t remote_manifest_download_bytes = 0;
        // Superseded images whose manifests were deleted from the bucket, and queued uploads that
        // a newer write of the same conversation made unnecessary.
        std::uint64_t remote_retired          = 0;
        std::uint64_t remote_uploads_skipped  = 0;
        // Queued uploads given up to keep the files they hold within the limits (the queue's
        // length, or max_bytes); their images are uploaded at their next write.
        std::uint64_t remote_uploads_dropped  = 0;
        // With peers: images imported from them, the chunk bytes linked (not copied), images
        // skipped (damaged, expired, over budget, or gone mid-import) and links that failed.
        std::uint64_t peer_images       = 0;
        std::uint64_t peer_bytes_linked = 0;
        std::uint64_t peer_skipped      = 0;
        std::uint64_t peer_link_failures = 0;
        // Times a peer's manifests were listed: a poll that finds the peer's directory and this
        // store unchanged since the last listing skips it.
        std::uint64_t peer_scans        = 0;
    };

    // Creates the directory layout, reads the manifests found there and removes whatever is
    // unreadable, unreferenced or left over from an interrupted write.
    explicit ContextStore(Options options);
    ~ContextStore();
    ContextStore(const ContextStore&)            = delete;
    ContextStore& operator=(const ContextStore&) = delete;

    // Stores `image` (its `regions` chunked, the rest inline). Idempotent for an id already held:
    // the manifest is replaced. Throws std::runtime_error when the files cannot be written or the
    // volume has no room for them (see reserve_bytes); the store is left consistent and nothing of
    // the failed image remains visible.
    PutResult put(const Description& description, std::span<const std::uint8_t> image,
                  std::span<const Region> regions);

    // Reassembles an image, verifying every chunk. nullopt when it is absent or damaged; a damaged
    // image is removed so it is not offered again (with a remote, a chunk file that is missing
    // rather than damaged, and cannot be fetched again, is only a miss). Counts as a use. The image
    // is not evicted while it is loaded.
    // `touch` false reads without refreshing the image's last use; the caller then calls touch()
    // once it has actually used the image.
    // What a load fetched from the bucket first, and how long the fetch took: the missing chunks
    // downloaded, verified and installed (what a waiting request waits for), not the local read
    // and reassembly that follow. Bytes are the chunks' raw length (as Info::fetch_bytes), not
    // what crossed the network, which is smaller for compressed chunks.
    // `expired`: the fetch stopped at `deadline` (a miss, but the image is not damaged: the chunks
    // fetched so far stay, and a later load fetches the rest).
    struct LoadReport {
        std::uint64_t fetched_bytes = 0;
        std::uint64_t fetch_ns      = 0;
        bool expired                = false;
    };
    // `deadline` bounds the download from the bucket: no chunk is started after it.
    [[nodiscard]] std::optional<std::vector<std::uint8_t>>
    load(const std::string& id, bool touch = true, LoadReport* report = nullptr,
         std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max());

    [[nodiscard]] std::vector<Info> list() const;
    [[nodiscard]] std::optional<Info> find(const std::string& id) const;
    // For lookups that run often against a large index (one per request): calls `visit` for every
    // checkpoint key that `match` accepts, with its image's Info as list() reports it, without
    // copying or sorting the index. `match` sees the Info without `local` and `fetch_bytes`, which
    // are worked out only for the images it accepts. Both run under the index lock, so they must
    // be quick and must not call the store.
    using CheckpointMatch = std::function<bool(const Info&, const CheckpointKey&)>;
    using CheckpointVisit = std::function<void(const Info&, const CheckpointKey&)>;
    void for_each_checkpoint(const CheckpointMatch& match, const CheckpointVisit& visit) const;
    bool erase(const std::string& id);
    void touch(const std::string& id);

    // Removes expired images and applies the size limit. Called after each put and at start-up;
    // exposed so a long-idle server can run it.
    void maintain();

    // With a remote: stops the background transfers after the work already queued, waiting at most
    // until `deadline`. For shutdown, after the last put. Returns whether the queue drained. With
    // `final_flush`, queued refreshes, prefetches and touches are dropped first so only uploads use
    // the budget.
    bool flush_remote(std::chrono::steady_clock::time_point deadline, bool final_flush = false);
    // With a remote: lists the bucket and registers images other engines left there that this
    // store does not hold. Also done in the background every remote_refresh. Returns the number of
    // images added. Stops reading manifests at `deadline`, waiting for another refresh counted; an
    // incomplete listing never removes anything from the index.
    std::size_t refresh_remote(std::chrono::steady_clock::time_point deadline =
                                   std::chrono::steady_clock::time_point::max());
    // For a request that found nothing: registers only the images the bucket lists, as of `asked`
    // or later, whose ids `wanted` accepts, and leaves every other new image to the background
    // refresh, so a lookup costs a listing and the manifests it is after, not every manifest
    // uploaded since the last refresh. The listing is shared: one taken (or started) at or after
    // `asked` serves it, and lookups arriving while an older one runs wait for it and then share
    // the one next listing, so the bucket is listed once at a time however many ask. Gives up at
    // `deadline`. Never removes anything from the index.
    struct LookupResult {
        std::size_t registered = 0; // wanted images this lookup added to the index
        std::size_t held       = 0; // wanted images listed that the index holds now
    };
    LookupResult lookup_remote(std::chrono::steady_clock::time_point asked,
                               std::chrono::steady_clock::time_point deadline,
                               const std::function<bool(const std::string& id)>& wanted);
    // With a remote: fetches the image's missing chunks in the background so a later load() is
    // local. False when there is nothing to do.
    bool prefetch(const std::string& id);
    // With peers: imports the images they hold that this store does not (also done in the
    // background every peer_poll). Returns the number imported.
    std::size_t sync_peers();

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] const Options& options() const noexcept { return options_; }

    // 128-bit content hash used for chunk names.
    [[nodiscard]] static std::array<std::uint64_t, 2> hash(std::span<const std::uint8_t> bytes);

    // The id an Engine stores an image under: lowercase hex of its endpoint key, so an image of the
    // same endpoint replaces it. A shared prefix's id ends in an extra digit, so it never replaces a
    // conversation image whose endpoint happens to fall at the same position.
    [[nodiscard]] static std::string image_id(const std::array<std::uint64_t, 2>& digests,
                                              std::uint32_t frontier, bool shared);

private:
    struct ChunkRef {
        std::array<std::uint64_t, 2> hash{};
        std::uint32_t length = 0;
    };
    struct Segment {
        bool chunked = false;
        std::uint64_t length = 0;
        std::vector<std::uint8_t> inline_bytes; // inline segments only
        std::vector<ChunkRef> chunks;           // chunked segments only
    };
    struct Entry {
        Info info;
        std::uint64_t manifest_bytes = 0;
        std::vector<ChunkRef> chunks; // every chunk the manifest names, in order
        // Which write of this id this is. A transfer that outlives the entry it started from
        // must not act on the entry that replaced it.
        std::uint64_t generation = 0;
        // The bucket's modified time of the manifest this entry was imported from (0: written or
        // loaded here). Compared only with other bucket times, never with the local clock.
        std::int64_t remote_modified_ms = 0;
        // With a remote: evicted for space or age. Its chunk files are gone, except those that
        // images still held here share, but the entry and manifest stay, so the image is still
        // offered and its chunks are fetched from the bucket when it is loaded. Dropping the entry
        // instead would make the next refresh download the manifest again.
        bool evicted = false;
        // With a remote: the bucket is known to hold this generation's manifest (an upload of it
        // succeeded, a touch found it, or it was registered from a listing). Only such an image is
        // evicted to the remote; one the bucket may lack is removed whole, as without a remote,
        // rather than kept as an entry whose chunks exist nowhere.
        bool published = false;
        // When this store last confirmed that (default: before any listing, as for a restart). A
        // complete listing taken after it that omits the manifest proves the bucket lost it.
        std::chrono::steady_clock::time_point published_at{};
        // Set on an image whose manifest was imported from a peer: the peer_skipped_ key and
        // manifest time to restore if the manifest proves damaged. Such an image is the peer's to
        // upload, so a use here never puts it in the bucket.
        std::string peer_memo_key;
        std::int64_t peer_stamp = 0;
        // The chunks each peer linked into this entry (peer_skipped_ key -> manifest time and the
        // chunks' hex keys): a damaged chunk quarantines only the peer that supplied it.
        struct PeerFill {
            std::int64_t stamp = 0;
            std::unordered_set<std::string> chunk_keys;
        };
        std::unordered_map<std::string, PeerFill> peer_fills;
    };
    struct ChunkUse {
        std::uint32_t references = 0;
        // Queued uploads, loads and puts that still need the file: it outlives its last reference
        // until they finish, and eviction leaves it.
        std::uint32_t pins       = 0;
        std::uint32_t length     = 0; // raw length
        std::uint64_t stored     = 0; // bytes the chunk file takes on disk (raw or compressed)
        bool present             = false; // the chunk file is in the directory
    };
    // Which form of a chunk the directory holds.
    struct StoredChunk {
        bool compressed     = false;
        std::uint64_t bytes = 0; // file size
    };
    struct RemoteTask {
        enum class Kind : std::uint8_t { Upload, Refresh, Prefetch, Touch } kind;
        std::string id;
        // Upload: the image as it was when it was written (the live entry may have been replaced
        // or evicted since), with its chunk files pinned until the upload ends.
        std::vector<std::uint8_t> manifest;
        std::vector<ChunkRef> chunks;
        // Upload: images this one superseded, whose manifests are deleted from the bucket once
        // this one's manifest is there.
        std::vector<std::string> retire;
        // Upload: not started before this, unless a flush is waiting.
        std::chrono::steady_clock::time_point ready{};
        // Upload: the entry generation the manifest describes; that entry is marked published when
        // the upload succeeds.
        std::uint64_t generation = 0;
    };

    [[nodiscard]] std::filesystem::path manifest_path(const std::string& id) const;
    [[nodiscard]] std::filesystem::path used_path(const std::string& id) const;
    // Holds the checksum of the manifest last published to the bucket: a restart marks the entry
    // published only when its manifest on disk is that one (not a newer write still to upload).
    [[nodiscard]] std::filesystem::path published_path(const std::string& id) const;
    void write_published_locked(const std::string& id, std::span<const std::uint8_t> manifest) const;
    [[nodiscard]] std::filesystem::path chunk_path(const std::array<std::uint64_t, 2>& hash) const;
    [[nodiscard]] std::filesystem::path
    compressed_chunk_path(const std::array<std::uint64_t, 2>& hash) const;
    // The chunk file in the directory, raw form preferred, with the right raw length; nullopt when
    // there is none (a compressed file is checked by the content size in its frame header).
    [[nodiscard]] std::optional<StoredChunk> locate_chunk(const ChunkRef& chunk) const;
    // Reads the chunk into `out` (exactly chunk.length bytes), decompressing a compressed file
    // as a stream straight into `out`. False when the file is missing, short or not a frame of
    // that length; the caller still verifies the hash.
    [[nodiscard]] bool read_chunk(const ChunkRef& chunk, std::span<std::uint8_t> out) const;
    // Writes the chunk (compressed when options_.compress_chunks and it pays) and returns the
    // path written and its size on disk.
    [[nodiscard]] std::pair<std::filesystem::path, StoredChunk>
    write_chunk(const ChunkRef& chunk, std::span<const std::uint8_t> bytes) const;
    // Writes the bytes of a chunk file exactly as the remote holds it.
    void install_chunk_file(const ChunkRef& chunk, bool compressed,
                            std::span<const std::uint8_t> file_bytes) const;
    // The chunk file's bytes for upload, after checking it decodes to the chunk's hash. Throws
    // when the file is missing or damaged. `compressed` receives which form it is.
    [[nodiscard]] std::vector<std::uint8_t> verified_chunk_file(const ChunkRef& chunk,
                                                                bool& compressed) const;
    [[nodiscard]] std::int64_t now_ms() const;
    [[nodiscard]] std::filesystem::path staging_path() const;
    void remove_chunk_files(const std::array<std::uint64_t, 2>& hash) const noexcept;

    void scan();
    bool read_manifest(const std::filesystem::path& path, Entry& entry,
                       std::vector<Segment>* segments) const;
    bool parse_manifest(std::span<const std::uint8_t> data, Entry& entry,
                        std::vector<Segment>* segments) const;
    void write_atomically(const std::filesystem::path& path,
                          std::span<const std::uint8_t> bytes) const;
    void add_references(const Entry& entry);
    void release_references(const Entry& entry);
    void remove_entry_locked(const std::string& id, bool corrupt);
    // A load found `entry` damaged: the peers whose copies are at fault are not imported from again
    // until they rewrite the image. `damaged` lists the bad chunks (each blames only the peer that
    // linked it, and is no longer counted as that peer's); nullptr means the manifest itself, which
    // blames the peer it came from.
    void quarantine_peers_locked(Entry& entry, const std::vector<ChunkRef>* damaged);
    void write_used(const std::string& id, std::int64_t last_used_ms) const;
    [[nodiscard]] std::uint64_t used_bytes_locked() const;
    void maintain_locked(const std::string* protect);
    bool evict_oldest_locked(const std::string* protect);
    // Nothing is left to evict, and queued uploads hold the files: gives up the oldest upload whose
    // files nothing else keeps (not `protect`'s). False when giving one up would free nothing.
    bool drop_queued_upload_locked(const std::string* protect);
    // Before a put writes `bytes`: makes the volume keep options_.reserve_bytes free after them,
    // evicting as needed, or throws. `keep` are the chunks the put reuses, pinned (in `pinned`)
    // until the put has referenced them.
    void make_room_on_volume_locked(const std::string& id, std::uint64_t bytes,
                                    const std::vector<ChunkRef>& keep, std::vector<ChunkRef>& pinned);
    // Gives up the entry's local chunk files (keeping those another image held here still uses, and
    // those a queued upload pins) and keeps the entry, which is then fetched from the bucket when
    // it is loaded. With `apply` false only counts. `holders` is whole_image_holders_locked().
    struct EvictionYield {
        std::uint64_t freed  = 0; // bytes of the image's own files removed
        std::uint64_t pinned = 0; // bytes of its own files a queued upload still needs
    };
    EvictionYield evict_to_remote_locked(
        Entry& entry, const std::unordered_map<std::string, std::uint32_t>& holders, bool apply);
    // Every chunk of the image is in the directory, and it was not evicted to the remote.
    [[nodiscard]] bool held_whole_locked(const Entry& entry) const;
    // For each chunk, how many images held whole use it.
    [[nodiscard]] std::unordered_map<std::string, std::uint32_t> whole_image_holders_locked() const;
    [[nodiscard]] bool entry_local_locked(const Entry& entry) const;
    // Raw bytes of the entry's chunks the directory lacks (each counted once).
    [[nodiscard]] std::uint64_t fetch_bytes_locked(const Entry& entry) const;
    [[nodiscard]] bool entry_has_local_bytes_locked(const Entry& entry) const;

    // The remote tier (all no-ops without options_.remote).
    [[nodiscard]] std::string remote_manifest_key(const std::string& id) const;
    [[nodiscard]] std::string remote_chunk_key(const std::array<std::uint64_t, 2>& hash,
                                               bool compressed) const;
    // The remote object holding the chunk, whichever form some engine uploaded: the form this
    // store writes is tried first. Nullopt when neither exists.
    [[nodiscard]] std::optional<std::pair<std::string, bool>>
    find_remote_chunk(const std::array<std::uint64_t, 2>& hash);
    void enqueue_remote(RemoteTask task);
    // `superseded`: images this write superseded. Their queued uploads are dropped and their
    // manifests are deleted from the bucket after this upload.
    void enqueue_upload(const std::string& id, std::uint64_t generation,
                        std::vector<std::uint8_t> manifest, std::vector<ChunkRef> chunks,
                        std::vector<std::string> superseded);
    // A replacement that will not be published (its upload failed or was dropped): the images it
    // was to retire may be registered from the bucket again. Called with remote_mutex_ held.
    void release_retirements_locked(const std::vector<std::string>& ids);
    void unpin_chunks_locked(const std::vector<ChunkRef>& chunks);
    void remote_loop();
    void upload_image(RemoteTask& task);
    void touch_remote_image(const std::string& id);
    // Drops confirmations, touches and quarantines older than remote_touch_interval: none is
    // trusted any longer, so keeping them would only grow with every chunk ever uploaded.
    void prune_remote_state();
    // Expired: stopped at the deadline, a miss like Transient.
    enum class FetchStatus : std::uint8_t { Complete, Transient, Corrupt, Expired };
    // `generation` receives the generation of the entry the fetch worked from.
    // `fetched` (optional) accumulates the bytes downloaded. No chunk is started after `deadline`.
    [[nodiscard]] FetchStatus fetch_image_chunks(
        const std::string& id, std::uint64_t& generation, std::uint64_t* fetched = nullptr,
        std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::time_point::max());
    // A fetch of `generation` found the image damaged or gone in the bucket: drop it and do not
    // offer it again, unless the entry has been replaced since.
    void settle_failed_fetch(const std::string& id, std::uint64_t generation);
    // `file_bytes` is the chunk file as the remote holds it (raw, or a zstd frame when `compressed`).
    bool install_chunk(const ChunkRef& chunk, bool compressed,
                       std::span<const std::uint8_t> file_bytes);

    Options options_;
    // Serializes everything that changes files (put, load, erase, touch, maintain), so a chunk is
    // never deleted while another call reads or reuses it. Lookups use only `mutex_` and so are not
    // held up by a long write.
    mutable std::mutex io_mutex_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
    std::unordered_map<std::string, ChunkUse> chunks_; // keyed by hex hash
    std::uint64_t chunk_total_bytes_ = 0;
    std::uint64_t manifest_total_bytes_ = 0;
    std::uint64_t next_generation_      = 0; // guarded by mutex_
    Stats stats_;
    // Images being loaded, and how many loads each: eviction and expiry pass them by, and their
    // chunks are pinned, so nothing removes what a load fetched before it reads it.
    std::unordered_map<std::string, std::uint32_t> loading_;
    // Counts changes that can let a peer image in that was not before (an image or chunk files
    // removed here): a peer poll lists the peer again when it moved. Guarded by mutex_.
    std::uint64_t local_changes_ = 0;

    // Remote worker: one thread runs the queued transfers, one at a time.
    std::mutex remote_mutex_;
    std::condition_variable remote_cv_;
    std::condition_variable remote_idle_cv_;
    std::deque<RemoteTask> remote_queue_;
    bool remote_busy_     = false;
    bool remote_stopping_ = false;
    std::uint32_t remote_flushes_ = 0; // flush_remote() calls waiting: delayed uploads run now
    // Chunk keys last confirmed in the bucket, and when: a lifecycle rule may have expired them
    // since, so the confirmation is trusted for remote_touch_interval only.
    std::unordered_map<std::string, std::int64_t> uploaded_;
    std::unordered_map<std::string, std::int64_t> remote_touched_; // id -> last touch (ms)
    // Ids not worth importing again. `remote_modified_ms` is the bucket's time of the manifest that
    // failed: one listed as written since is tried again. `failed_ms` is the local time, used only
    // to expire the entry with the other remote bookkeeping.
    struct RemoteFailure {
        std::int64_t remote_modified_ms = 0;
        std::int64_t failed_ms          = 0;
    };
    std::unordered_map<std::string, RemoteFailure> remote_failed_;
    // Superseded manifests whose deletion from the bucket failed: retried after the next upload.
    std::vector<std::string> retire_backlog_;
    // Serializes refreshes. Timed, so a caller with a deadline gives up at it rather than behind a
    // slow refresh.
    std::timed_mutex refresh_mutex_;
    std::size_t refresh_remote_locked(std::chrono::steady_clock::time_point deadline);
    // The manifests the bucket listed, and when the listing was asked for (it shows the bucket as
    // it was then, not when it ended).
    struct Listing {
        std::chrono::steady_clock::time_point listed_at{};
        std::vector<ObjectInfo> objects;
    };
    // A listing asked for at or after `asked`: the newest one taken, one in progress that started
    // after `asked` (waited for), or a new one once the one in progress ends (listings never run
    // two at a time). Nullptr when none is ready by `deadline`; throws what the bucket throws.
    [[nodiscard]] std::shared_ptr<const Listing>
    listing_since(std::chrono::steady_clock::time_point asked,
                  std::chrono::steady_clock::time_point deadline);
    std::mutex listing_mutex_;
    std::condition_variable listing_cv_;
    std::shared_ptr<const Listing> latest_listing_;
    // When the listing in progress was asked for (nullopt: none is).
    std::optional<std::chrono::steady_clock::time_point> listing_started_;
    // The image id a listed manifest key names (empty when it names none).
    [[nodiscard]] std::string listed_id(const std::string& key) const;
    // Downloads a manifest that a listing asked for at `listed_at` showed and adds its image to the
    // index, unless the index holds it or the id is given up on. True when it was added.
    bool register_remote_image(const ObjectInfo& object, const std::string& id,
                               std::chrono::steady_clock::time_point listed_at);
    std::int64_t last_prune_ms_ = 0; // remote worker thread only

    // Peer import. `owner_` holds the exclusive lock on options_.directory for the store's life.
    struct OwnerLock;
    [[nodiscard]] bool import_peer_image(const std::filesystem::path& peer, const std::string& id,
                                         std::int64_t stamp);
    void peer_loop();
    std::unique_ptr<OwnerLock> owner_;
    // Serializes puts' free-space check and writes across the stores of one parent directory.
    struct VolumeLock;
    std::filesystem::path volume_lock_path_; // empty: no parent directory to share
    // (peer + id) -> modification time of the peer manifest that was skipped: not tried again until
    // the peer rewrites it. Guarded by mutex_.
    std::unordered_map<std::string, std::int64_t> peer_skipped_;
    // (peer + id) -> an image not imported for a reason that can pass without the peer rewriting
    // the manifest (`stamp`, its time): it did not fit here (tried again once `local_changes` has
    // moved), or the peer holds none of the chunks this store lacks (`absent`: tried again once the
    // peer's `.used` time, `used_stamp`, moves). Guarded by mutex_.
    struct PeerWait {
        std::int64_t stamp           = 0;
        std::int64_t used_stamp      = 0;
        std::uint64_t local_changes  = 0;
        bool absent                  = false;
    };
    std::unordered_map<std::string, PeerWait> peer_waits_;
    [[nodiscard]] static std::optional<std::int64_t> peer_used_stamp(const std::filesystem::path& peer,
                                                                     const std::string& id);
    // Per peer: its manifests directory's time and local_changes_ at the last listing. A listing
    // is trusted to stand for later polls only when the directory's time was well in the past
    // (a change in the same tick of the file system clock would not move it). sync_peers only.
    struct PeerScan {
        std::filesystem::file_time_type manifests_time{};
        std::uint64_t local_changes = 0;
        bool trusted                = false;
    };
    std::unordered_map<std::string, PeerScan> peer_scans_;
    std::mutex peer_scan_mutex_; // one sync_peers() at a time
    std::mutex peer_mutex_;
    std::condition_variable peer_cv_;
    bool peer_stopping_ = false;
    std::thread peer_thread_;
    std::thread remote_thread_;
};

} // namespace ninfer::runtime
