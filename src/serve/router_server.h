#pragma once

// The router front-end: one public address in front of several models.
//
// It answers the llama.cpp model API itself (GET /models, POST /models/load, /models/unload and
// /models/sleep, GET /models/sse) and the per-model suspend, resume and residency routes, checks the
// API key, and forwards every other request to the HTTP server of the model it names. The model
// comes from the request body's "model" (an id or an alias), or a `model` query parameter; a request
// that names none goes to the most recently used loaded model, except a Responses follow-up
// (GET/DELETE/cancel/input_items of an id, or previous_response_id), which goes to the model that made
// the response (ResponseOwners). The registry loads or wakes the
// model first when autoload allows, making room as its limits require, and a request holds a lease
// on its model until its response has been delivered, so a model in use is never put to sleep.
//
// Bytes are relayed as they arrive, so streamed responses stream, and a client that disconnects
// closes the upstream connection, which cancels the generation.

#include "serve/model_registry.h"
#include "serve/routed_model.h"
#include "serve/serve_options.h"

#include <httplib.h>
#include <spdlog/logger.h>

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ninfer::serve {

// `body` (a JSON object) with its top-level "model" set to `model`, adding the member when there
// is none. Only that member's text changes: every other byte, numbers included, reaches the model
// as the client wrote it. nullopt when `body` is not a JSON object.
[[nodiscard]] std::optional<std::string> with_model_selector(std::string_view body,
                                                             std::string_view model);

// Whether a Content-Type header value names a JSON media type (the token is case-insensitive).
[[nodiscard]] bool is_json_content_type(std::string_view content_type);

// Which model owns each stored Responses id. Every model keeps a private Responses store and an id
// says nothing about where it was made, so the router remembers it from the create reply and sends
// the follow-ups (GET, DELETE, cancel, input_items, previous_response_id) to that model. Bounded: the
// least recently used id is forgotten first, and a forgotten id is routed as before (the model
// answers not-found if it is not the one that made it). Entries outlive the model's load, as the
// model's store does.
class ResponseOwners {
public:
    explicit ResponseOwners(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    void record(const std::string& id, const std::string& model);
    // The owner of `id`, counting the lookup as use.
    [[nodiscard]] std::optional<std::string> owner(const std::string& id);
    void forget(const std::string& id);
    [[nodiscard]] std::size_t size() const;

private:
    using Entry = std::pair<std::string, std::string>; // id, model

    mutable std::mutex mutex_;
    std::size_t capacity_;
    std::list<Entry> recent_; // most recently used first
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

// Finds the id of the response a Responses create reply makes, in its first bytes: the JSON body's
// "id", or the first SSE event's response.id. Fed the reply as it arrives, in pieces of any size.
class ResponseIdScanner {
public:
    void feed(std::string_view data);
    [[nodiscard]] const std::optional<std::string>& id() const noexcept { return id_; }

private:
    std::string seen_;
    std::size_t resume_ = 0; // where the next search starts
    std::optional<std::string> id_;
};

// The response id a Responses follow-up path names (/v1/responses/{id}, .../input_items,
// .../cancel); nullopt for any other path, including the create-style ones that take no id.
[[nodiscard]] std::optional<std::string> response_id_from_path(std::string_view path);

class RouterServer {
public:
    RouterServer(ServeOptions options, ModelRegistry& registry, std::shared_ptr<RoutedPorts> ports,
                 std::shared_ptr<spdlog::logger> logger);
    ~RouterServer();

    RouterServer(const RouterServer&)            = delete;
    RouterServer& operator=(const RouterServer&) = delete;

    bool bind();
    // Blocks until stop() is called; false when the accept loop failed.
    bool listen();
    void stop();

private:
    void register_routes();
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res);
    void handle_models_load(const httplib::Request& req, httplib::Response& res);
    void handle_models_unload(const httplib::Request& req, httplib::Response& res);
    void handle_models_sleep(const httplib::Request& req, httplib::Response& res);
    void handle_models_sse(const httplib::Request& req, httplib::Response& res);
    void handle_residency(const httplib::Request& req, httplib::Response& res);
    void handle_suspend(const httplib::Request& req, httplib::Response& res);
    void handle_resume(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res) const;
    // Routes a request to the model it names (waking or loading it first) and relays the response.
    void route(const httplib::Request& req, httplib::Response& res);
    // Reads a status endpoint (/props, /metrics, /v1/load) from the most recently used model
    // without waking it.
    void route_status(const httplib::Request& req, httplib::Response& res);

    // Relays `method target` with `body` to the server of the model `lease` holds, taking the lease
    // along until the response has been delivered.
    // `tap`, when given, sees the bytes of a 2xx reply before they are queued for the client.
    using Tap = std::function<void(std::string_view)>;
    void relay(const httplib::Request& req, httplib::Response& res, ModelRegistry::Lease lease,
               const std::string& method, const std::string& target, std::string body,
               Tap tap = {});

    ServeOptions options_;
    ModelRegistry& registry_;
    std::shared_ptr<RoutedPorts> ports_;
    std::shared_ptr<spdlog::logger> logger_;
    ResponseOwners response_owners_;
    httplib::Server server_;
};

} // namespace ninfer::serve
