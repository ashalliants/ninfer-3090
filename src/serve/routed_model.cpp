#include "serve/routed_model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::serve {

ServeOptions routed_model_options(ServeOptions model, const std::string& id) {
    model.host = "127.0.0.1";
    model.port = 0;
    model.api_key.clear();
    // Keyless on loopback, so its Origin check is what keeps a browser page out; --cors (passed on
    // to every model with the other options) would switch that check off and allow any origin.
    model.enable_cors = false;
    model.allowed_hosts.clear();
    model.model_id_override = id;
    return model;
}

void RoutedPorts::set(const std::string& model, int port) {
    std::lock_guard lock(mutex_);
    ports_[model] = port;
}

void RoutedPorts::erase(const std::string& model) {
    std::lock_guard lock(mutex_);
    ports_.erase(model);
}

std::optional<int> RoutedPorts::get(const std::string& model) const {
    std::lock_guard lock(mutex_);
    const auto found = ports_.find(model);
    if (found == ports_.end()) { return std::nullopt; }
    return found->second;
}

RoutedModelService::RoutedModelService(std::string id, std::unique_ptr<GenerationService> service,
                                       std::shared_ptr<spdlog::logger> logger,
                                       std::shared_ptr<RoutedPorts> ports,
                                       std::shared_ptr<OpenAIResponsesStore> responses_store)
    : id_(std::move(id)), service_(std::move(service)), ports_(std::move(ports)) {
    server_ = std::make_unique<HttpServer>(routed_model_options(service_->options(), id_),
                                           std::move(logger), std::move(responses_store));
    port_                     = server_->bind_any_port();
    if (port_ == 0) { throw std::runtime_error("cannot bind a loopback port for model " + id_); }
    server_->start_serving_during_startup();
    server_->attach(*service_);
    // listen() runs the model's periodic statistics beside the accept loop that is already live.
    listener_ = std::thread([this] {
        bool listened = false;
        try {
            listened = server_->listen();
        } catch (...) {}
        // listen() returns when stop() is called too; only an unrequested end is a failure.
        if (!listened && !stopping_.load()) { listener_failed_ = true; }
    });
    ports_->set(id_, port_);
}

RoutedModelService::~RoutedModelService() {
    stopping_ = true;
    ports_->erase(id_);
    if (server_) { server_->stop(); }
    if (listener_.joinable()) { listener_.join(); }
    server_.reset();
    service_.reset();
}

} // namespace ninfer::serve
