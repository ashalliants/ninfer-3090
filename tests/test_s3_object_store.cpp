// The S3 object store client. The timestamp parser, and connection reuse against a minimal local
// HTTP server (POSIX hosts), are checked always; the round trip, listing across pages, special
// characters, touch and remove run against the S3-compatible server named by
// NINFER_TEST_S3_ENDPOINT and NINFER_TEST_S3_BUCKET (credentials from AWS_ACCESS_KEY_ID and
// AWS_SECRET_ACCESS_KEY, "test" if unset), for example a local moto or MinIO, and are skipped
// without them.
#include "product/object_store/s3_object_store.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using ninfer::product::S3Config;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << '\n';
    }
}

#ifndef _WIN32
// Just enough HTTP/1.1 to count connections: every request is answered 200 with a small body and
// the connection kept open, except the first request for a key containing "flaky", answered 503.
class LocalServer {
public:
    LocalServer() {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t length        = sizeof(address);
        if (listener_ < 0 || ::bind(listener_, reinterpret_cast<sockaddr*>(&address), length) != 0 ||
            ::listen(listener_, 16) != 0 ||
            ::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
            throw std::runtime_error("cannot open a local test server");
        }
        port_     = ntohs(address.sin_port);
        acceptor_ = std::thread([this] { accept_loop(); });
    }
    ~LocalServer() {
        ::shutdown(listener_, SHUT_RDWR);
        ::close(listener_);
        acceptor_.join();
        for (auto& [socket, thread] : connections_) {
            ::shutdown(socket, SHUT_RDWR);
            thread.join();
            ::close(socket);
        }
    }
    [[nodiscard]] std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
    std::atomic<int> accepted{0};
    std::atomic<int> requests{0};

private:
    void accept_loop() {
        for (;;) {
            const int socket = ::accept(listener_, nullptr, nullptr);
            if (socket < 0) { return; }
            ++accepted;
            connections_.emplace_back(socket, std::thread([this, socket] { serve(socket); }));
        }
    }
    void serve(int socket) {
        std::string buffer;
        char data[4096];
        for (;;) {
            std::size_t end = buffer.find("\r\n\r\n");
            while (end == std::string::npos) {
                const ssize_t got = ::recv(socket, data, sizeof(data), 0);
                if (got <= 0) { return; }
                buffer.append(data, static_cast<std::size_t>(got));
                end = buffer.find("\r\n\r\n");
            }
            const std::string head = buffer.substr(0, end);
            std::size_t body       = 0;
            if (const auto at = head.find("Content-Length: "); at != std::string::npos) {
                body = std::strtoull(head.c_str() + at + 16, nullptr, 10);
            }
            while (buffer.size() < end + 4 + body) {
                const ssize_t got = ::recv(socket, data, sizeof(data), 0);
                if (got <= 0) { return; }
                buffer.append(data, static_cast<std::size_t>(got));
            }
            buffer.erase(0, end + 4 + body);
            ++requests;
            const bool flaky = head.find("flaky") != std::string::npos && !flaked_.exchange(true);
            const bool is_head = head.rfind("HEAD ", 0) == 0;
            const std::string reply =
                flaky ? "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n"
                      : std::string("HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n") +
                            (is_head ? "" : "hello");
            if (::send(socket, reply.data(), reply.size(), MSG_NOSIGNAL) < 0) { return; }
        }
    }
    int listener_ = -1;
    int port_     = 0;
    std::atomic<bool> flaked_{false};
    std::thread acceptor_;
    std::vector<std::pair<int, std::thread>> connections_; // acceptor thread only, until join
};

// Calls on one thread reuse one connection; a retry after a 5xx answer starts on a new one, as each
// attempt did before handles were kept.
void test_connections_are_reused() {
    LocalServer server;
    S3Config config;
    config.endpoint   = server.endpoint();
    config.bucket     = "bucket";
    config.access_key = "test";
    config.secret_key = "test";
    auto store        = ninfer::product::make_s3_object_store(config);
    try {
        for (int i = 0; i < 5; ++i) {
            const auto got = store->get("probe/manifests/" + std::to_string(i) + ".manifest");
            check(got && got->size() == 5, "a get through a kept connection lost its body");
        }
        check(store->exists("probe/chunks/ab/x.chunk"), "a head through a kept connection failed");
        store->put("probe/chunks/ab/y.chunk", std::vector<std::uint8_t>(1000, 7));
        check(server.accepted == 1 && server.requests == 7,
              "calls on one thread did not share one connection");
        check(store->get("probe/flaky").has_value(), "the retry after a 503 failed");
        check(server.accepted == 2 && server.requests == 9,
              "the retry after a 503 did not start on a new connection");
        // Threads at once each take a handle of their own.
        std::vector<std::thread> threads;
        std::atomic<int> ok{0};
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&] {
                for (int i = 0; i < 10; ++i) { ok += store->get("probe/manifests/x.manifest") ? 1 : 0; }
            });
        }
        for (auto& thread : threads) { thread.join(); }
        check(ok == 40 && server.accepted <= 2 + 4, "concurrent calls failed or opened a connection each");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "EXCEPTION " << error.what() << '\n';
    }
}
#endif
} // namespace

int main(int, char**) {
    check(ninfer::product::parse_s3_timestamp_ms("2026-10-08T19:01:45.000Z") == 1791486105000LL,
          "S3 timestamp with fraction");
    check(ninfer::product::parse_s3_timestamp_ms("1970-01-01T00:00:01Z") == 1000, "S3 timestamp epoch");
    check(ninfer::product::parse_s3_timestamp_ms("not a time") == -1, "S3 timestamp garbage");
#ifndef _WIN32
    test_connections_are_reused();
#endif
    const char* endpoint = std::getenv("NINFER_TEST_S3_ENDPOINT");
    const char* bucket   = std::getenv("NINFER_TEST_S3_BUCKET");
    if (failures != 0) { return 1; }
    if (endpoint == nullptr || bucket == nullptr) {
        std::cout << "skipping the server round trip: NINFER_TEST_S3_ENDPOINT / BUCKET not set\n";
        return 77;
    }
    S3Config config;
    config.endpoint   = endpoint;
    config.bucket     = bucket;
    config.access_key = std::getenv("AWS_ACCESS_KEY_ID") ? std::getenv("AWS_ACCESS_KEY_ID") : "test";
    config.secret_key = std::getenv("AWS_SECRET_ACCESS_KEY") ? std::getenv("AWS_SECRET_ACCESS_KEY") : "test";
    auto store = ninfer::product::make_s3_object_store(config);

    std::vector<std::uint8_t> big(5 * 1024 * 1024 + 17);
    for (std::size_t i = 0; i < big.size(); ++i) { big[i] = static_cast<std::uint8_t>(i * 31 + 7); }
    const std::string key = "probe/manifests/ab.manifest";
    try {
        check(!store->exists(key), "object exists before put");
        check(!store->get(key).has_value(), "get of an absent object");
        store->put(key, big);
        check(store->exists(key), "object missing after put");
        const auto back = store->get(key);
        check(back && *back == big, "round trip differs");
        store->put("probe/chunks/ab/with space & plus+.chunk", std::vector<std::uint8_t>{1, 2, 3});
        for (int i = 0; i < 1100; ++i) {
            store->put("probe/many/" + std::to_string(i), std::vector<std::uint8_t>{1});
        }
        const auto listed = store->list("probe/");
        check(listed.size() == 1102, "listing across pages lost objects");
        bool found = false;
        for (const auto& item : listed) {
            if (item.key == "probe/chunks/ab/with space & plus+.chunk") { found = item.size == 3; }
            check(item.modified_ms > 1'600'000'000'000LL, "modified time not parsed");
        }
        check(found, "a key with special characters did not round trip through listing");
        check(store->touch(key), "touch of an existing object");
        check(!store->touch("probe/nothing-here"), "touch of an absent object reported success");
        const auto after_touch = store->get(key);
        check(after_touch && *after_touch == big, "touch changed the bytes");
        store->remove(key);
        check(!store->exists(key), "object exists after remove");
        store->remove(key); // absent: not an error
        std::cout << "listed " << listed.size() << " objects\n";
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "EXCEPTION " << error.what() << '\n';
    }
    std::cout << (failures == 0 ? "ok" : "FAILED") << '\n';
    return failures == 0 ? 0 : 1;
}
