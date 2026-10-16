#pragma once

// The router front-end: one public address in front of several models.
//
// It answers the llama.cpp model API itself (GET /models, POST /models/load, /models/unload and
// /models/sleep, GET /models/sse) and the per-model suspend, resume and residency routes, checks the
// API key, and forwards every other request to the HTTP server of the model it names. The model
// comes from the request body's "model" (an id or an alias), or a `model` query parameter; a request
// that names none goes to the most recently used loaded model. The registry loads or wakes the
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
#include <memory>
#include <string>

namespace ninfer::serve {

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
    void relay(const httplib::Request& req, httplib::Response& res, ModelRegistry::Lease lease,
               const std::string& method, const std::string& target, std::string body);

    ServeOptions options_;
    ModelRegistry& registry_;
    std::shared_ptr<RoutedPorts> ports_;
    std::shared_ptr<spdlog::logger> logger_;
    httplib::Server server_;
};

} // namespace ninfer::serve
