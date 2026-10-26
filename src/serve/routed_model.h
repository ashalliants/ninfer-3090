#pragma once

// A model served behind a router: one Engine behind a GenerationService, answered by an ordinary
// single-model HttpServer that listens on loopback at a port of its own. The router front-end
// (RouterServer) forwards a request to that server once the model registry has loaded or woken the
// model, so every endpoint, log and metric of single-model mode works per model unchanged.
//
// Sleep is the Engine's suspend (the model must be started with --model-suspend); the server and its
// port stay up while the model sleeps, so the status endpoints still answer.

#include "serve/http_server.h"
#include "serve/model_registry.h"

#include <spdlog/logger.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace ninfer::serve {

// The options a routed model's own server runs with: `model` (the options the model was started
// with) on 127.0.0.1, a port the system picks, `id` as its public id, and none of the public
// address's access settings. The router owns those: it checks the key, the Host and the Origin,
// answers CORS preflights and drops access-control-* headers from what it relays. A model with a
// key or --cors of its own would let a browser page reach it on its loopback port, around the
// router's key.
[[nodiscard]] ServeOptions routed_model_options(ServeOptions model, const std::string& id);

// The loopback port each loaded model's server listens on, by model id.
class RoutedPorts {
public:
    void set(const std::string& model, int port);
    void erase(const std::string& model);
    [[nodiscard]] std::optional<int> get(const std::string& model) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, int> ports_;
};

class RoutedModelService final : public ModelService {
public:
    // Starts the model's server on a free loopback port and attaches `service` to it. The service
    // is loaded and warmed up already.
    RoutedModelService(std::string id, std::unique_ptr<GenerationService> service,
                       std::shared_ptr<spdlog::logger> logger, std::shared_ptr<RoutedPorts> ports,
                       std::shared_ptr<OpenAIResponsesStore> responses_store);
    ~RoutedModelService() override;

    RoutedModelService(const RoutedModelService&)            = delete;
    RoutedModelService& operator=(const RoutedModelService&) = delete;

    [[nodiscard]] GenerationService& service() override { return *service_; }

    [[nodiscard]] bool suspendable() const override {
        return service_->engine_options().suspend.enabled;
    }

    void suspend() override { (void)service_->suspend(true); }
    void resume() override { (void)service_->resume(); }
    // A dead accept loop counts as a failure: the registry unloads the model and the next request
    // loads it afresh, instead of relaying to a port nobody serves.
    [[nodiscard]] bool failed() const override {
        return service_->has_failed() || listener_failed_.load();
    }

    // Forwarded so /models reports the limit the engine settled on, not the configured one.
    [[nodiscard]] std::uint32_t effective_max_context() const override {
        return service_->options().max_context;
    }

    [[nodiscard]] int port() const noexcept { return port_; }

private:
    std::string id_;
    // The server holds a pointer to the service: destroyed first.
    std::unique_ptr<GenerationService> service_;
    std::unique_ptr<HttpServer> server_;
    std::shared_ptr<RoutedPorts> ports_;
    std::thread listener_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> listener_failed_{false};
    int port_ = 0;
};

} // namespace ninfer::serve
