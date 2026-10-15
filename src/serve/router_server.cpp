#include "serve/router_server.h"

#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

bool anthropic_path(const std::string& path) { return path.rfind("/v1/messages", 0) == 0; }

void write_error(const httplib::Request& req, httplib::Response& res, int status, std::string code,
                 std::string message) {
    ApiError error;
    error.status  = status;
    error.type    = status >= 500 ? "server_error" : "invalid_request_error";
    error.code    = std::move(code);
    error.message = std::move(message);
    if (anthropic_path(req.path)) {
        write_anthropic_error(res, error, res.get_header_value("request-id"));
    } else {
        write_openai_error(res, error);
    }
}

// Maps the registry's failures onto HTTP.
void write_registry_error(const httplib::Request& req, httplib::Response& res,
                          std::exception_ptr failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const ModelUnknown& error) {
        write_error(req, res, 404, "model_not_found", error.what());
    } catch (const ModelBusy& error) {
        write_error(req, res, 409, "model_busy", error.what());
    } catch (const ModelNotLoaded& error) {
        write_error(req, res, 503, "model_not_loaded", error.what());
    } catch (const ModelLoadFailed& error) {
        write_error(req, res, 503, "model_load_failed", error.what());
    } catch (const std::exception& error) {
        write_error(req, res, 500, "router_error", error.what());
    }
}

Json model_entry(const ModelStatusSnapshot& status) {
    Json state{{"value", model_state_name(status.state)}, {"args", status.arguments}};
    if (status.failed) {
        state["failed"] = true;
        if (!status.last_error.empty()) { state["last_error"] = status.last_error; }
    }
    return Json{{"id", status.id},
                {"object", "model"},
                {"created", unix_time_now()},
                {"owned_by", "ninfer"},
                {"aliases", status.aliases},
                {"status", std::move(state)}};
}

Json event_json(const ModelEvent& event) {
    Json out{{"model", event.model}, {"status", model_state_name(event.state)}};
    if (event.failed) {
        out["failed"] = true;
        out["error"]  = event.error;
    }
    return out;
}

// A `model` that is present but empty: an invalid selector, not an omitted one.
struct EmptySelector : std::invalid_argument {
    EmptySelector()
        : std::invalid_argument("\"model\" must not be empty; omit it to use the last model") {}
};

// The model a request names: a JSON body's "model" on a POST, else a `model` query parameter, else
// empty when omitted (the most recently used loaded model). A present but empty one throws.
std::string requested_model(const httplib::Request& req, std::optional<Json>& body) {
    if (req.method == "POST" && !req.body.empty() &&
        req.get_header_value("Content-Type").find("json") != std::string::npos) {
        try {
            Json parsed = Json::parse(req.body);
            if (parsed.is_object()) {
                const auto found = parsed.find("model");
                if (found != parsed.end() && found->is_string()) {
                    std::string name = found->get<std::string>();
                    if (name.empty()) { throw EmptySelector(); }
                    body = std::move(parsed);
                    return name;
                }
                // The selector may be in the query; the body is still the one to rewrite.
                body = std::move(parsed);
            }
        } catch (const EmptySelector&) {
            throw;
        } catch (const std::exception&) {
            // The model answers a malformed body in its own words.
        }
    }
    if (req.has_param("model")) {
        std::string name = req.get_param_value("model");
        if (name.empty()) { throw EmptySelector(); }
        return name;
    }
    return {};
}

// The target without its `model` query parameter: the router consumed it, and the model's own
// endpoints reject query keys they do not know.
std::string without_model_param(const std::string& target) {
    const auto question = target.find('?');
    if (question == std::string::npos) { return target; }
    std::string kept;
    std::size_t start = question + 1;
    while (start <= target.size()) {
        std::size_t end = target.find('&', start);
        if (end == std::string::npos) { end = target.size(); }
        const std::string piece = target.substr(start, end - start);
        const std::string key   = httplib::decode_query_component(piece.substr(0, piece.find('=')));
        if (!piece.empty() && key != "model") { kept += (kept.empty() ? "" : "&") + piece; }
        start = end + 1;
    }
    return target.substr(0, question) + (kept.empty() ? "" : "?" + kept);
}

bool hop_by_hop(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "connection" || lower == "keep-alive" || lower == "transfer-encoding" ||
           lower == "content-length" || lower == "content-type" || lower == "host" ||
           lower == "upgrade" || lower == "te" || lower == "trailer" ||
           lower == "proxy-connection" || lower == "content-encoding" ||
           lower.rfind("access-control-", 0) == 0;
}

// Chunks one relay may hold ahead of its client (each is at most one upstream read, about 16 KiB).
constexpr std::size_t kMaxQueuedChunks = 64;

// One relayed response: an upstream request on a thread of its own feeding the bytes the client's
// connection takes as they come, and the means to close the upstream connection early.
struct Relay {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> chunks;
    bool headers_ready = false;
    bool done          = false;
    bool failed        = false;
    int status         = 502;
    httplib::Headers headers;
    std::string error;
    std::shared_ptr<httplib::Client> client;
    std::thread worker;
};

} // namespace

RouterServer::RouterServer(ServeOptions options, ModelRegistry& registry,
                           std::shared_ptr<RoutedPorts> ports,
                           std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), registry_(registry), ports_(std::move(ports)),
      logger_(std::move(logger)) {
    // cpp-httplib is thread-per-connection, and a relayed request holds a router worker for as long
    // as its model holds the matching worker of its own server.
    constexpr std::size_t kBaseWorkers = 32;
    constexpr std::size_t kMaxWorkers  = 320;
    server_.new_task_queue             = [] {
        return new httplib::ThreadPool(kBaseWorkers, kMaxWorkers, kMaxWorkers);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

RouterServer::~RouterServer() = default;

bool RouterServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

bool RouterServer::listen() { return server_.listen_after_bind(); }

void RouterServer::stop() {
    registry_.stop();
    server_.stop();
}

void RouterServer::register_routes() {
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }
    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            write_error(req, res, 401, "invalid_api_key", "missing or invalid API key");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server_.set_exception_handler(
        [](const httplib::Request& req, httplib::Response& res, std::exception_ptr failure) {
            write_registry_error(req, res, std::move(failure));
        });

    server_.Get("/health", [this](const httplib::Request& req, httplib::Response& res) {
        handle_health(req, res);
    });
    server_.Get("/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Post("/models/load", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models_load(req, res);
    });
    server_.Post("/models/unload", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models_unload(req, res);
    });
    server_.Post("/models/sleep", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models_sleep(req, res);
    });
    server_.Get("/models/sse", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models_sse(req, res);
    });
    server_.Get(R"(/v1/models/([^/]+)/residency)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_residency(req, res);
                });
    server_.Post(R"(/v1/models/([^/]+)/suspend)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_suspend(req, res);
                 });
    server_.Post(R"(/v1/models/([^/]+)/resume)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_resume(req, res);
                 });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    // Status endpoints read the model that last served without waking a sleeping one.
    for (const char* path : {"/props", "/metrics", "/v1/load"}) {
        server_.Get(path, [this](const httplib::Request& req, httplib::Response& res) {
            route_status(req, res);
        });
    }
    server_.Get(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        route(req, res);
    });
    server_.Post(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        route(req, res);
    });
    server_.Delete(R"(.*)", [this](const httplib::Request& req, httplib::Response& res) {
        route(req, res);
    });
}

void RouterServer::handle_health(const httplib::Request&, httplib::Response& res) const {
    std::size_t loaded = 0;
    std::size_t total  = 0;
    for (const ModelStatusSnapshot& status : registry_.status()) {
        ++total;
        if (status.state == ModelState::Loaded) { ++loaded; }
    }
    res.set_header("Cache-Control", "no-store");
    res.set_content(Json{{"status", "ok"}, {"models", total}, {"loaded", loaded}}.dump(),
                    "application/json");
}

void RouterServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    Json data = Json::array();
    for (const ModelStatusSnapshot& status : registry_.status()) {
        data.push_back(model_entry(status));
    }
    res.set_content(Json{{"object", "list"}, {"data", std::move(data)}}.dump(),
                    "application/json");
}

void RouterServer::handle_model(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto status    = registry_.status(id);
    if (!status) {
        write_error(req, res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    res.set_content(model_entry(*status).dump(), "application/json");
}

namespace {

// The "model" of a /models/* request body.
std::string body_model(const httplib::Request& req) {
    try {
        const Json body = req.body.empty() ? Json::object() : Json::parse(req.body);
        if (body.is_object()) {
            const auto found = body.find("model");
            if (found != body.end() && found->is_string()) {
                std::string name = found->get<std::string>();
                if (name.empty()) { throw EmptySelector(); }
                return name;
            }
        }
    } catch (const EmptySelector&) {
        throw;
    } catch (const std::exception&) {}
    throw std::invalid_argument("the request body must be a JSON object with a \"model\" string");
}

void write_ok(httplib::Response& res) {
    res.set_content(Json{{"success", true}}.dump(), "application/json");
}

} // namespace

void RouterServer::handle_models_load(const httplib::Request& req, httplib::Response& res) {
    try {
        registry_.load(body_model(req));
        write_ok(res);
    } catch (const std::invalid_argument& error) {
        if (dynamic_cast<const ModelUnknown*>(&error) != nullptr) {
            write_registry_error(req, res, std::current_exception());
        } else {
            write_error(req, res, 400, "invalid_request", error.what());
        }
    } catch (...) {
        write_registry_error(req, res, std::current_exception());
    }
}

void RouterServer::handle_models_unload(const httplib::Request& req, httplib::Response& res) {
    try {
        registry_.unload(body_model(req));
        write_ok(res);
    } catch (const std::invalid_argument& error) {
        if (dynamic_cast<const ModelUnknown*>(&error) != nullptr) {
            write_registry_error(req, res, std::current_exception());
        } else {
            write_error(req, res, 400, "invalid_request", error.what());
        }
    } catch (...) {
        write_registry_error(req, res, std::current_exception());
    }
}

void RouterServer::handle_models_sleep(const httplib::Request& req, httplib::Response& res) {
    try {
        const std::string model = body_model(req);
        bool hold               = false;
        bool wait               = true;
        const Json body         = Json::parse(req.body);
        if (const auto found = body.find("hold"); found != body.end() && found->is_boolean()) {
            hold = found->get<bool>();
        }
        if (const auto found = body.find("wait"); found != body.end() && found->is_boolean()) {
            wait = found->get<bool>();
        }
        registry_.sleep(model, hold, wait);
        write_ok(res);
    } catch (const std::invalid_argument& error) {
        if (dynamic_cast<const ModelUnknown*>(&error) != nullptr) {
            write_registry_error(req, res, std::current_exception());
        } else {
            write_error(req, res, 400, "invalid_request", error.what());
        }
    } catch (...) {
        write_registry_error(req, res, std::current_exception());
    }
}

void RouterServer::handle_models_sse(const httplib::Request&, httplib::Response& res) {
    // Start after what is already kept: a subscriber wants changes, and GET /models has the state.
    std::vector<ModelEvent> skipped;
    auto cursor = std::make_shared<std::uint64_t>(
        registry_.events(0, std::chrono::milliseconds(0), skipped));
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, cursor](size_t, httplib::DataSink& sink) {
            std::vector<ModelEvent> events;
            *cursor = registry_.events(*cursor, std::chrono::milliseconds(15000), events);
            if (!sink.is_writable()) { return false; }
            std::string text;
            for (const ModelEvent& event : events) {
                text += "data: " + event_json(event).dump() + "\n\n";
            }
            if (text.empty()) { text = ": keep-alive\n\n"; }
            return sink.write(text.data(), text.size());
        });
}

void RouterServer::handle_residency(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_.resolve(id);
    if (!resolved) {
        write_error(req, res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    // A loaded or sleeping model answers for itself without being woken.
    if (auto lease = registry_.observe(*resolved)) {
        relay(req, res, std::move(*lease), "GET", "/v1/models/" + *resolved + "/residency", {});
        return;
    }
    res.set_content(Json{{"object", "model.residency"},
                         {"model", *resolved},
                         {"status", "unloaded"},
                         {"state", "unloaded"}}
                        .dump(),
                    "application/json");
}

void RouterServer::handle_suspend(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_.resolve(id);
    if (!resolved) {
        write_error(req, res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    bool hold = false;
    try {
        const Json body = req.body.empty() ? Json::object() : Json::parse(req.body);
        if (!body.is_object()) { throw std::invalid_argument("body is not an object"); }
        for (const auto& [key, value] : body.items()) {
            if (key != "auto_resume" || !value.is_boolean()) {
                throw std::invalid_argument("unexpected field");
            }
            hold = !value.get<bool>();
        }
    } catch (const std::exception&) {
        write_error(req, res, 400, "invalid_request",
                    "request body must be a JSON object with at most a boolean auto_resume");
        return;
    }
    try {
        registry_.sleep(*resolved, hold, /*wait=*/false);
    } catch (const ModelNotLoaded& error) {
        write_error(req, res, 400, "model_suspend_unavailable", error.what());
        return;
    } catch (...) {
        write_registry_error(req, res, std::current_exception());
        return;
    }
    handle_residency(req, res);
}

void RouterServer::handle_resume(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    const auto resolved  = registry_.resolve(id);
    if (!resolved) {
        write_error(req, res, 404, "model_not_found", "model '" + id + "' not found");
        return;
    }
    try {
        registry_.load(*resolved);
    } catch (...) {
        write_registry_error(req, res, std::current_exception());
        return;
    }
    handle_residency(req, res);
}

void RouterServer::route(const httplib::Request& req, httplib::Response& res) {
    std::optional<Json> body;
    std::string name;
    try {
        name = requested_model(req, body);
    } catch (const EmptySelector& error) {
        write_error(req, res, 400, "invalid_request", error.what());
        return;
    }
    ModelRegistry::Lease lease = registry_.acquire(name);
    std::string payload        = req.body;
    // The model knows itself by its id, not by an alias the client used.
    // A selector given in the query is carried into the body, which the models require.
    if (body) {
        const auto found = body->find("model");
        const bool named = found != body->end() && found->is_string();
        if ((named && found->get<std::string>() != lease.model()) || (!named && !name.empty())) {
            (*body)["model"] = lease.model();
            payload          = body->dump();
        }
    }
    relay(req, res, std::move(lease), req.method, without_model_param(req.target),
          std::move(payload));
}

void RouterServer::route_status(const httplib::Request& req, httplib::Response& res) {
    auto lease = registry_.observe("");
    if (!lease) {
        if (req.path == "/metrics") {
            res.set_content("", "text/plain; version=0.0.4; charset=utf-8");
        } else {
            write_error(req, res, 503, "model_not_loaded", "no model is loaded");
        }
        return;
    }
    relay(req, res, std::move(*lease), req.method, without_model_param(req.target), {});
}

void RouterServer::relay(const httplib::Request& req, httplib::Response& res,
                         ModelRegistry::Lease lease, const std::string& method,
                         const std::string& target, std::string body) {
    const auto port = ports_->get(lease.model());
    if (!port) {
        write_error(req, res, 503, "model_not_loaded",
                    "model '" + lease.model() + "' is not serving");
        return;
    }
    auto relayed    = std::make_shared<Relay>();
    relayed->client = std::make_shared<httplib::Client>("127.0.0.1", *port);
    relayed->client->set_connection_timeout(10, 0);
    // A response may wait on a long prefill; the model's own limits decide when it gives up.
    relayed->client->set_read_timeout(24 * 3600, 0);
    relayed->client->set_write_timeout(600, 0);
    relayed->client->set_keep_alive(false);

    httplib::Request upstream;
    upstream.method = method;
    upstream.path   = target;
    upstream.body   = std::move(body);
    for (const auto& [name, value] : req.headers) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        // The model needs no credentials (it listens on loopback for the router alone), and
        // compression is the router's business with its client, not the model's.
        if (hop_by_hop(name) || lower == "authorization" || lower == "x-api-key" ||
            lower == "accept-encoding") {
            continue;
        }
        upstream.headers.emplace(name, value);
    }
    if (!upstream.body.empty()) {
        upstream.set_header("Content-Type", req.get_header_value("Content-Type", "application/json"));
    }
    upstream.response_handler = [relayed](const httplib::Response& response) {
        std::lock_guard lock(relayed->mutex);
        relayed->status        = response.status;
        relayed->headers       = response.headers;
        relayed->headers_ready = true;
        relayed->changed.notify_all();
        return true;
    };
    upstream.content_receiver = [relayed](const char* data, size_t length, size_t, size_t) {
        std::unique_lock lock(relayed->mutex);
        // Backpressure: a client that reads slowly makes the upstream read wait here, so the
        // router holds a bounded backlog rather than the whole response. Closing the relay (the
        // client went away) sets done and ends the wait.
        relayed->changed.wait(lock, [&] {
            return relayed->done || relayed->chunks.size() < kMaxQueuedChunks;
        });
        if (relayed->done) { return false; }
        relayed->chunks.emplace_back(data, length);
        relayed->changed.notify_all();
        return true;
    };
    relayed->worker = std::thread([relayed, request = std::move(upstream)]() mutable {
        httplib::Response response;
        httplib::Error error = httplib::Error::Success;
        const bool ok        = relayed->client->send(request, response, error);
        std::lock_guard lock(relayed->mutex);
        if (!ok && !relayed->headers_ready) {
            relayed->failed = true;
            relayed->error  = httplib::to_string(error);
        }
        relayed->done = true;
        relayed->changed.notify_all();
    });

    // Closes the upstream connection and waits for its thread: the model sees the disconnect and
    // stops generating for a client that is gone.
    const auto finish = [relayed](bool) {
        {
            std::lock_guard lock(relayed->mutex);
            relayed->done = true;
        }
        relayed->changed.notify_all();
        relayed->client->stop();
        if (relayed->worker.joinable()) { relayed->worker.join(); }
    };

    std::unique_lock lock(relayed->mutex);
    relayed->changed.wait(lock, [&] { return relayed->headers_ready || relayed->done; });
    if (!relayed->headers_ready) {
        lock.unlock();
        finish(false);
        write_error(req, res, 502, "model_unreachable",
                    "model '" + lease.model() + "' did not answer" +
                        (relayed->error.empty() ? std::string() : ": " + relayed->error));
        return;
    }
    res.status = relayed->status;
    std::string content_type = "application/json";
    for (const auto& [name, value] : relayed->headers) {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == "content-type") {
            content_type = value;
        } else if (!hop_by_hop(name)) {
            res.set_header(name, value);
        }
    }
    lock.unlock();

    // The lease goes with the response: the model stays awake until the last byte is delivered.
    auto held = std::make_shared<ModelRegistry::Lease>(std::move(lease));
    res.set_chunked_content_provider(
        content_type,
        [relayed](size_t, httplib::DataSink& sink) {
            std::unique_lock lock(relayed->mutex);
            for (;;) {
                if (!relayed->chunks.empty()) {
                    std::string chunk = std::move(relayed->chunks.front());
                    relayed->chunks.pop_front();
                    lock.unlock();
                    relayed->changed.notify_all();
                    return sink.write(chunk.data(), chunk.size());
                }
                if (relayed->done) {
                    sink.done();
                    return true;
                }
                // Wake now and then to notice a client that has gone while the model is busy.
                relayed->changed.wait_for(lock, std::chrono::milliseconds(500));
                if (!sink.is_writable()) { return false; }
            }
        },
        [finish, held](bool success) { finish(success); });
}

} // namespace ninfer::serve
