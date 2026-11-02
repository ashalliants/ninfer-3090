#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {

struct ObjectInfo {
    std::string key;
    std::uint64_t size   = 0;
    std::int64_t modified_ms = 0; // Unix milliseconds
};

// Thrown by a call that ran out of the time its caller allowed (see ObjectStore::Deadline): not a
// failure of the store, so a caller that set the deadline treats it as running out of time.
class DeadlineExceeded final : public std::runtime_error {
public:
    DeadlineExceeded() : std::runtime_error("object store call passed its deadline") {}
};

// A flat key/value object store: an S3-compatible bucket in production, a map in tests. Every call
// may block on the network. An absent object is nullopt/false, never an error; transport and server
// failures throw std::runtime_error. Implementations are safe for concurrent calls.
class ObjectStore {
public:
    virtual ~ObjectStore() = default;

    // The time a call must be over by; time_point::max() means no limit.
    using Deadline = std::chrono::steady_clock::time_point;

    virtual void put(const std::string& key, std::span<const std::uint8_t> bytes) = 0;
    [[nodiscard]] virtual std::optional<std::vector<std::uint8_t>> get(const std::string& key) = 0;
    // As get(), but gives up with DeadlineExceeded when the transfer is not over by `deadline`. A
    // store that cannot abandon a transfer may only check the deadline before it starts (this
    // default), so a caller must not rely on it to bound a single call tightly.
    [[nodiscard]] virtual std::optional<std::vector<std::uint8_t>> get_before(const std::string& key,
                                                                              Deadline deadline) {
        if (std::chrono::steady_clock::now() >= deadline) { throw DeadlineExceeded(); }
        return get(key);
    }
    [[nodiscard]] virtual bool exists(const std::string& key) = 0;
    // As exists(), but gives up with DeadlineExceeded when the request is not over by `deadline`
    // (a store that cannot abandon one may only check before it starts, like get_before()).
    [[nodiscard]] virtual bool exists_before(const std::string& key, Deadline deadline) {
        if (std::chrono::steady_clock::now() >= deadline) { throw DeadlineExceeded(); }
        return exists(key);
    }
    // Every object whose key starts with `prefix`, in no particular order.
    [[nodiscard]] virtual std::vector<ObjectInfo> list(const std::string& prefix) = 0;
    // As list(), with `deadline` for the whole listing (every page); never a partial result.
    [[nodiscard]] virtual std::vector<ObjectInfo> list_before(const std::string& prefix,
                                                                      Deadline deadline) {
        if (std::chrono::steady_clock::now() >= deadline) { throw DeadlineExceeded(); }
        return list(prefix);
    }
    // Restarts the object's age for lifecycle expiry without transferring its bytes. False when the
    // object is absent.
    virtual bool touch(const std::string& key) = 0;
    virtual void remove(const std::string& key) = 0;

    // Makes every call in progress, and every later one, fail promptly instead of waiting on the
    // network. For shutdown: the store is not used afterwards. May be called from any thread.
    virtual void interrupt() noexcept {}
};

} // namespace ninfer
