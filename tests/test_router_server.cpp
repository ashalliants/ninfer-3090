// RouterServer's HTTP surface, without an Engine: a fake ModelRegistry service per model and a
// loopback httplib server standing in for each model's own HttpServer. Covered: API key handling,
// selector precedence (body over query, aliases, empty and mistyped selectors, the default model),
// the query and body rewrite that reaches the model, the /models lifecycle endpoints, suspend, resume
// and residency, the status routes with nothing loaded, header relay, streamed relay with a slow
// client, a client that disconnects (before and after the headers), upstream failure (unreachable,
// not serving, dropped mid-stream), a model whose listener dies being rebuilt on the next request, and
// Responses follow-ups going to the model that made the response (ResponseOwners).

#include "serve/model_registry.h"
#include "serve/routed_model.h"
#include "serve/router_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::serve;
using Json = nlohmann::json;
using namespace std::chrono_literals;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

// Polls `condition` until it holds or `limit` passes.
bool eventually(const std::function<bool()>& condition, std::chrono::milliseconds limit = 10s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(10ms);
    }
    return condition();
}

// A port nothing listens on: bound to learn it, then released.
int free_port() {
    httplib::Server probe;
    const int port = probe.bind_to_any_port("127.0.0.1");
    if (port <= 0) { throw std::runtime_error("no free port"); }
    probe.stop();
    return port;
}

// What an upstream (a model's own server) saw.
struct Seen {
    std::string method;
    std::string target;
    std::string body;
    httplib::Headers headers;

    [[nodiscard]] std::string header(const std::string& name) const {
        const auto found = headers.find(name);
        return found == headers.end() ? std::string() : found->second;
    }
};

// A loopback server standing in for one model's HttpServer. Routes are added before start(); the
// first one that matches wins, so specific routes go before echo().
class Upstream {
public:
    explicit Upstream(std::string id) : id_(std::move(id)) {}

    // Records a request; a route calls this to be counted among what the model saw.
    void note(const httplib::Request& req) {
        std::lock_guard lock(mutex_);
        seen_.push_back(Seen{req.method, req.target, req.body, req.headers});
    }

    ~Upstream() {
        server.stop();
        if (thread_.joinable()) { thread_.join(); }
    }

    // Answers every request with what it was, as JSON, tagged with this model's id.
    void echo() {
        const auto handler = [this](const httplib::Request& req, httplib::Response& res) {
            note(req);
            res.set_header("X-Upstream", id_);
            res.set_content(Json{{"served_by", id_},
                                 {"method", req.method},
                                 {"target", req.target},
                                 {"body", req.body}}
                                .dump(),
                            "application/json");
        };
        server.Get(R"(.*)", handler);
        server.Post(R"(.*)", handler);
        server.Delete(R"(.*)", handler);
    }

    void start() {
        port_   = server.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server.listen_after_bind(); });
    }

    [[nodiscard]] int port() const { return port_; }

    [[nodiscard]] std::vector<Seen> seen() const {
        std::lock_guard lock(mutex_);
        return seen_;
    }

    [[nodiscard]] Seen last() const {
        const auto all = seen();
        if (all.empty()) { throw std::runtime_error("upstream " + id_ + " saw nothing"); }
        return all.back();
    }

    [[nodiscard]] std::size_t count() const { return seen().size(); }

    httplib::Server server;

private:
    std::string id_;
    int port_ = 0;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<Seen> seen_;
};

// Shared with the services the registry creates: how many were built and torn down, and which are
// marked failed (a dead listener).
struct Journal {
    std::mutex mutex;
    std::vector<std::string> calls;
    std::map<std::string, std::atomic<bool>> failing;

    void note(std::string call) {
        std::lock_guard lock(mutex);
        calls.push_back(std::move(call));
    }

    [[nodiscard]] std::size_t count(const std::string& call) {
        std::lock_guard lock(mutex);
        return static_cast<std::size_t>(std::count(calls.begin(), calls.end(), call));
    }
};

// A model without an Engine. Like RoutedModelService, it publishes the port its server listens on
// while it exists.
class FakeService final : public ModelService {
public:
    FakeService(std::string id, bool suspendable, Journal& journal,
                std::shared_ptr<RoutedPorts> ports, int port)
        : id_(std::move(id)), suspendable_(suspendable), journal_(journal), ports_(std::move(ports)) {
        journal_.note("load " + id_);
        ports_->set(id_, port);
        journal_.failing[id_] = false;
    }

    ~FakeService() override {
        ports_->erase(id_);
        journal_.note("unload " + id_);
    }

    GenerationService& service() override { throw std::logic_error("fake service"); }
    bool suspendable() const override { return suspendable_; }
    void suspend() override { journal_.note("sleep " + id_); }
    void resume() override { journal_.note("wake " + id_); }
    bool failed() const override { return journal_.failing.at(id_).load(); }

private:
    std::string id_;
    bool suspendable_;
    Journal& journal_;
    std::shared_ptr<RoutedPorts> ports_;
};

struct Rig {
    struct ModelSpec {
        std::string id;
        std::vector<std::string> aliases = {};
        bool suspendable                 = true;
        std::vector<std::string> arguments = {};
        // This model's --response-store-max-records; 0 takes the rig's `store_records`.
        std::size_t store_records = 0;
    };

    Journal journal;
    std::shared_ptr<RoutedPorts> ports = std::make_shared<RoutedPorts>();
    std::map<std::string, std::unique_ptr<Upstream>> upstreams;
    std::unique_ptr<ModelRegistry> registry;
    std::unique_ptr<RouterServer> router;
    std::thread listener;
    int port = 0;

    // Upstreams are filled in by `customize` (to add routes) before they start; echo() is added after
    // it, as the fallback.
    Rig(const std::vector<ModelSpec>& specs, std::string api_key = {}, bool cors = false,
        const std::function<void(const std::string&, Upstream&)>& customize = {},
        std::uint32_t models_max = 2, std::size_t store_records = kDefaultResponseStoreRecords) {
        std::vector<ModelDefinition> definitions;
        std::map<std::string, bool> suspendable;
        for (const ModelSpec& spec : specs) {
            ModelDefinition definition;
            definition.id        = spec.id;
            definition.aliases   = spec.aliases;
            definition.arguments = spec.arguments;
            definition.options.response_store_max_records =
                spec.store_records != 0 ? spec.store_records : store_records;
            definitions.push_back(std::move(definition));
            suspendable[spec.id] = spec.suspendable;
            auto upstream        = std::make_unique<Upstream>(spec.id);
            if (customize) { customize(spec.id, *upstream); }
            upstream->echo();
            upstream->start();
            upstreams[spec.id] = std::move(upstream);
        }
        RegistryPolicy policy;
        policy.models_max = models_max;
        registry          = std::make_unique<ModelRegistry>(
            std::move(definitions), policy,
            [this, suspendable](const ModelDefinition& definition) {
                return std::make_unique<FakeService>(definition.id, suspendable.at(definition.id),
                                                     journal, ports,
                                                     upstreams.at(definition.id)->port());
            },
            /*router=*/true);
        ninfer::serve::ServeOptions options;
        options.host        = "127.0.0.1";
        options.port        = free_port();
        options.api_key     = std::move(api_key);
        options.enable_cors = cors;
        port                = options.port;
        router = std::make_unique<RouterServer>(options, *registry, ports, nullptr);
        if (!router->bind()) { throw std::runtime_error("router could not bind"); }
        listener = std::thread([this] { router->listen(); });
        // A throw from a constructor skips ~Rig, and destroying a joinable std::thread
        // terminates, so stop and join the listener here before rethrowing.
        try {
            wait_until_accepting();
        } catch (...) {
            router->stop();
            listener.join();
            throw;
        }
    }

    // bind() only reserves the socket; the accept loop starts on `listener` a moment later, and a
    // request sent before then is refused. Probe /health (answered by the router itself, so it
    // loads nothing) until the loop answers, with any status, so the first request cannot race
    // listener startup.
    void wait_until_accepting() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            auto probe = client();
            // Short timeouts keep a stalled probe from outlasting the deadline.
            probe.set_connection_timeout(0, 200000);
            probe.set_read_timeout(0, 200000);
            if (probe.Get("/health")) { return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        throw std::runtime_error("router did not start accepting connections");
    }

    ~Rig() {
        router->stop();
        if (listener.joinable()) { listener.join(); }
    }

    // A client of the router.
    [[nodiscard]] httplib::Client client() const {
        httplib::Client out("127.0.0.1", port);
        out.set_connection_timeout(10, 0);
        out.set_read_timeout(30, 0);
        return out;
    }

    [[nodiscard]] Upstream& up(const std::string& id) const { return *upstreams.at(id); }

    [[nodiscard]] ModelState state(const std::string& id) const {
        return registry->status(id)->state;
    }

    [[nodiscard]] std::uint32_t leases(const std::string& id) const {
        return registry->status(id)->leases;
    }
};

std::string header_value(const httplib::Headers& headers, const std::string& name) {
    const auto found = headers.find(name);
    return found == headers.end() ? std::string() : found->second;
}

// A parsed reply: the status, the body as JSON when it is, and the raw result.
struct Reply {
    int status = 0;
    Json json;
    httplib::Headers headers;
    std::string body;

    [[nodiscard]] std::string error_code() const {
        return json.contains("error") && json["error"].contains("code")
                   ? json["error"]["code"].get<std::string>()
                   : std::string();
    }
};

Reply to_reply(const httplib::Result& result) {
    Reply out;
    if (!result) { return out; }
    out.status  = result->status;
    out.headers = result->headers;
    out.body    = result->body;
    out.json    = Json::parse(result->body, nullptr, /*allow_exceptions=*/false);
    return out;
}

Reply get(const Rig& rig, const std::string& target, const httplib::Headers& headers = {}) {
    auto client = rig.client();
    return to_reply(client.Get(target, headers));
}

Reply post(const Rig& rig, const std::string& target, const std::string& body,
           const httplib::Headers& headers = {}) {
    auto client = rig.client();
    return to_reply(client.Post(target, headers, body, "application/json"));
}

Reply del(const Rig& rig, const std::string& target) {
    auto client = rig.client();
    return to_reply(client.Delete(target));
}

// What the model's server received as the body of its last request.
Json upstream_body(const Rig& rig, const std::string& id) {
    const Seen last = rig.up(id).last();
    return Json::parse(last.body);
}

const std::vector<Rig::ModelSpec> kTwoModels{{.id = "a", .aliases = {"alpha"}}, {.id = "b"}};

void api_key_handling() {
    Rig rig(kTwoModels, "secret");
    expect(get(rig, "/health").status == 200, "/health needs no key");
    const Reply missing = get(rig, "/models");
    expect(missing.status == 401 && missing.error_code() == "invalid_api_key",
           "no key is refused with invalid_api_key");
    expect(!header_value(missing.headers, "X-NInfer-Version").empty(),
           "a refusal still carries the version header");
    expect(get(rig, "/models", {{"Authorization", "Bearer wrong"}}).status == 401,
           "a wrong bearer key is refused");
    expect(get(rig, "/models", {{"x-api-key", "wrong"}}).status == 401,
           "a wrong x-api-key is refused");
    expect(get(rig, "/models", {{"Authorization", "Bearer secret"}}).status == 200,
           "the bearer key is accepted");
    expect(get(rig, "/models", {{"x-api-key", "secret"}}).status == 200,
           "the x-api-key is accepted");
    expect(post(rig, "/v1/chat/completions", R"({"model":"a"})").status == 401,
           "a relayed route is guarded too");
    expect(rig.up("a").count() == 0, "a refused request never reaches a model");
    expect(rig.state("a") == ModelState::Unloaded, "a refused request does not load the model");

    const Reply ok = post(rig, "/v1/chat/completions", R"({"model":"a"})",
                          {{"Authorization", "Bearer secret"}, {"x-api-key", "secret"},
                           {"X-Custom", "kept"}});
    expect(ok.status == 200, "an authorised request is relayed");
    const Seen seen = rig.up("a").last();
    expect(seen.header("Authorization").empty() && seen.header("x-api-key").empty(),
           "credentials are not forwarded to the model");
    expect(seen.header("X-Custom") == "kept", "ordinary headers are forwarded");
    expect(seen.header("Content-Type") == "application/json", "the content type is forwarded");

    const Reply anthropic = post(rig, "/v1/messages", R"({"model":"a"})");
    expect(anthropic.status == 401 && anthropic.json.value("type", "") == "error",
           "a refusal on an Anthropic path uses the Anthropic envelope");
}

void cors_headers() {
    Rig rig(kTwoModels, "secret", /*cors=*/true);
    auto client = rig.client();
    const auto preflight = client.Options("/v1/chat/completions");
    expect(preflight && preflight->status == 204, "a preflight is answered without a key");
    expect(preflight && preflight->get_header_value("Access-Control-Allow-Origin") == "*",
           "CORS headers are on when enabled");
}

void selector_precedence() {
    Rig rig(kTwoModels);
    const std::string path = "/v1/chat/completions";

    // An alias in the body names the model; the model is told its own id.
    Reply reply = post(rig, path, R"({"model":"alpha","temperature":0.70,"n":1})");
    expect(reply.status == 200 && reply.headers.find("X-Upstream") != reply.headers.end() &&
               header_value(reply.headers, "X-Upstream") == "a",
           "an alias in the body routes to its model");
    Json sent = upstream_body(rig, "a");
    expect(sent["model"] == "a", "the alias is rewritten to the model id");
    expect(rig.up("a").last().body == R"({"model":"a","temperature":0.70,"n":1})",
           "only the selector's text changes in the body");

    // A selector given only in the query is carried into the body, and leaves the target.
    reply = post(rig, path + "?model=b&keep=1", R"({"stream":false})");
    expect(reply.status == 200 && header_value(reply.headers, "X-Upstream") == "b",
           "a query selector routes");
    expect(rig.up("b").last().target == path + "?keep=1",
           "the router's own query parameter is stripped, others are kept");
    expect(upstream_body(rig, "b")["model"] == "b" && upstream_body(rig, "b")["stream"] == false,
           "the query selector is written into the body");

    // The body wins over the query.
    reply = post(rig, path + "?model=b", R"({"model":"a"})");
    expect(header_value(reply.headers, "X-Upstream") == "a", "the body selector beats the query");
    expect(rig.up("a").last().target == path, "the unused query selector is still stripped");

    // A query-only selector on a request without a body.
    reply = get(rig, "/v1/responses/resp_1?model=b");
    expect(reply.status == 200 && header_value(reply.headers, "X-Upstream") == "b",
           "a bodyless GET follows its query selector");
    expect(rig.up("b").last().target == "/v1/responses/resp_1", "and its target is stripped");

    // No selector goes to the model that served last, and the body names it.
    reply = post(rig, path, R"({"messages":[]})");
    expect(header_value(reply.headers, "X-Upstream") == "b",
           "no selector goes to the most recently served model");
    expect(upstream_body(rig, "b")["model"] == "b", "and the body is given that model");
    reply = get(rig, "/v1/responses/resp_2");
    expect(header_value(reply.headers, "X-Upstream") == "b", "so does a bodyless request");

    // Empty and mistyped selectors are client errors, not omitted ones.
    const std::size_t before = rig.up("a").count() + rig.up("b").count();
    reply = post(rig, path, R"({"model":""})");
    expect(reply.status == 400 && reply.error_code() == "invalid_request",
           "an empty body selector is refused");
    reply = post(rig, path, R"({"model":5})");
    expect(reply.status == 400 && reply.error_code() == "invalid_request",
           "a non-string body selector is refused");
    reply = post(rig, path + "?model=", R"({"messages":[]})");
    expect(reply.status == 400 && reply.error_code() == "invalid_request",
           "an empty query selector is refused");
    expect(rig.up("a").count() + rig.up("b").count() == before,
           "a refused selector reaches no model");

    // A malformed body is the model's to answer; the router relays it as is.
    reply = post(rig, path + "?model=a", "not json");
    expect(reply.status == 200 && rig.up("a").last().body == "not json",
           "a body that is not JSON is relayed untouched");

    // Unknown models, on both envelopes.
    reply = post(rig, path, R"({"model":"nope"})");
    expect(reply.status == 404 && reply.error_code() == "model_not_found", "an unknown model is 404");
    reply = post(rig, "/v1/messages", R"({"model":"nope"})");
    expect(reply.status == 404 && reply.json.value("type", "") == "error",
           "an unknown model on an Anthropic path uses its envelope");
}

void models_listing() {
    Rig rig({{.id = "a", .aliases = {"alpha"}, .arguments = {"--api-key", "hunter2", "--ctx", "8"}},
             {.id = "b", .arguments = {"--api-key=hunter3"}}});
    const Reply list = get(rig, "/models");
    expect(list.status == 200 && list.json["data"].size() == 2, "GET /models lists every model");
    const Json& first = list.json["data"][0];
    expect(first["id"] == "a" && first["status"]["value"] == "unloaded" &&
               first["aliases"] == Json::array({"alpha"}),
           "an entry carries its id, state and aliases");
    expect(list.body.find("hunter") == std::string::npos, "credentials in arguments are redacted");
    expect(first["status"]["args"] == Json::array({"--api-key", "<redacted>", "--ctx", "8"}),
           "the other arguments are kept");
    expect(get(rig, "/v1/models").json["data"].size() == 2, "/v1/models is the same listing");
    expect(get(rig, "/v1/models/a").json.value("id", "") == "a", "one model is reachable by id");
    const Reply missing = get(rig, "/v1/models/zzz");
    expect(missing.status == 404 && missing.error_code() == "model_not_found",
           "an unknown model id is 404");
    const Reply health = get(rig, "/health");
    expect(health.json["models"] == 2 && health.json["loaded"] == 0, "health counts models");
    post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(get(rig, "/health").json["loaded"] == 1, "and the loaded ones");
}

void lifecycle_endpoints() {
    Rig rig({{.id = "a", .aliases = {"alpha"}}, {.id = "b"}, {.id = "c", .suspendable = false}});

    Reply reply = post(rig, "/models/load", R"({"model":"alpha"})");
    expect(reply.status == 200 && reply.json["success"] == true, "load answers success");
    expect(rig.state("a") == ModelState::Loaded, "load loads the model");
    expect(rig.up("a").count() == 0, "load does not call the model's server");

    expect(post(rig, "/models/load", "").status == 400, "load without a body is 400");
    expect(post(rig, "/models/load", "[]").status == 400, "load with a non-object body is 400");
    expect(post(rig, "/models/load", R"({"model":""})").status == 400, "an empty model is 400");
    reply = post(rig, "/models/load", R"({"model":"nope"})");
    expect(reply.status == 404 && reply.error_code() == "model_not_found",
           "load of an unknown model is 404");

    reply = post(rig, "/models/sleep", R"({"model":"a"})");
    expect(reply.status == 200 && rig.state("a") == ModelState::Sleeping, "sleep suspends");
    expect(!rig.registry->status("a")->held, "without hold, requests may wake it");
    reply = post(rig, "/models/sleep", R"({"model":"a","hold":"yes"})");
    expect(reply.status == 400 && reply.error_code() == "invalid_request",
           "a non-boolean hold is 400");
    reply = post(rig, "/models/sleep", R"({"model":"a","wait":1})");
    expect(reply.status == 400, "a non-boolean wait is 400");
    expect(post(rig, "/models/sleep", R"({"model":"a","hold":true})").status == 200 &&
               rig.registry->status("a")->held,
           "hold keeps requests from waking it");
    reply = post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(reply.status == 503 && reply.error_code() == "model_not_loaded",
           "a held model refuses requests");
    expect(post(rig, "/models/load", R"({"model":"a"})").status == 200 &&
               rig.state("a") == ModelState::Loaded,
           "load wakes a held model");
    expect(post(rig, "/v1/chat/completions", R"({"model":"a"})").status == 200,
           "and it serves again");

    expect(post(rig, "/models/load", R"({"model":"c"})").status == 200, "load c");
    reply = post(rig, "/models/sleep", R"({"model":"c"})");
    expect(reply.status == 503 && reply.error_code() == "model_not_loaded" &&
               rig.state("c") == ModelState::Loaded,
           "a model that cannot sleep refuses, and stays loaded");

    reply = post(rig, "/models/unload", R"({"model":"a"})");
    expect(reply.status == 200 && rig.state("a") == ModelState::Unloaded, "unload unloads");
    expect(post(rig, "/models/unload", R"({"model":"nope"})").status == 404,
           "unload of an unknown model is 404");
    expect(post(rig, "/models/unload", "{}").status == 400, "unload without a model is 400");

}

void residency_suspend_resume() {
    Rig rig(kTwoModels, {}, false, [](const std::string& id, Upstream& upstream) {
        upstream.server.Get("/v1/models/" + id + "/residency",
                            [id, &upstream](const httplib::Request& req, httplib::Response& res) {
                                upstream.note(req);
                                res.set_content(Json{{"object", "model.residency"},
                                                     {"model", id},
                                                     {"state", "from-model"}}
                                                    .dump(),
                                                "application/json");
                            });
    });

    // Nothing loaded: the router answers for the model, without loading it.
    Reply reply = get(rig, "/v1/models/alpha/residency");
    expect(reply.status == 200 && reply.json["state"] == "unloaded" &&
               reply.json["model"] == "a" && reply.json["object"] == "model.residency",
           "an unloaded model reports unloaded from the router");
    expect(rig.state("a") == ModelState::Unloaded && rig.up("a").count() == 0,
           "asking about residency does not load it");
    expect(get(rig, "/v1/models/nope/residency").status == 404, "residency of an unknown model is 404");
    expect(post(rig, "/v1/models/nope/suspend", "").status == 404, "suspend of an unknown is 404");
    expect(post(rig, "/v1/models/nope/resume", "").status == 404, "resume of an unknown is 404");

    // A loaded model answers for itself.
    expect(post(rig, "/models/load", R"({"model":"a"})").status == 200, "load a");
    reply = get(rig, "/v1/models/a/residency");
    expect(reply.status == 200 && reply.json["state"] == "from-model",
           "a loaded model answers its own residency");
    expect(rig.up("a").last().target == "/v1/models/a/residency", "through its own server");

    // Body validation.
    for (const char* body : {"[]", R"({"other":true})", R"({"auto_resume":"no"})",
                             R"({"auto_resume":1})", "not json"}) {
        reply = post(rig, "/v1/models/a/suspend", body);
        expect(reply.status == 400 && reply.error_code() == "invalid_request",
               std::string("a bad suspend body is 400: ") + body);
    }
    expect(rig.state("a") == ModelState::Loaded, "a bad body does not suspend");

    // Suspend: sleeping, and the model's own residency is what comes back.
    reply = post(rig, "/v1/models/a/suspend", "");
    expect(reply.status == 200 && reply.json["state"] == "from-model" &&
               rig.state("a") == ModelState::Sleeping && !rig.registry->status("a")->held,
           "suspend with no body sleeps and may auto-resume");
    reply = post(rig, "/v1/models/a/resume", "");
    expect(reply.status == 200 && rig.state("a") == ModelState::Loaded, "resume wakes");
    reply = post(rig, "/v1/models/a/suspend", R"({"auto_resume":false})");
    expect(reply.status == 200 && rig.registry->status("a")->held &&
               rig.state("a") == ModelState::Sleeping,
           "auto_resume false holds it");
    reply = post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(reply.status == 503 && reply.error_code() == "model_not_loaded",
           "a held suspended model refuses requests");
    reply = post(rig, "/v1/models/a/resume", "");
    expect(reply.status == 200 && !rig.registry->status("a")->held, "resume releases the hold");

    // Suspend of a model that is not loaded just holds it; residency stays the router's.
    reply = post(rig, "/v1/models/b/suspend", R"({"auto_resume":false})");
    expect(reply.status == 200 && reply.json["state"] == "unloaded",
           "suspending an unloaded model reports unloaded");
    reply = post(rig, "/v1/models/b/resume", "");
    expect(reply.status == 200 && rig.state("b") == ModelState::Loaded, "resume loads it");
}

void suspend_refusals() {
    Rig rig({{.id = "a"}, {.id = "c", .suspendable = false}}, {}, false,
            [](const std::string&, Upstream& upstream) {
                upstream.server.Get("/hold", [](const httplib::Request& req, httplib::Response& res) {
                    for (int i = 0; i < 1000 && !req.is_connection_closed(); ++i) {
                        std::this_thread::sleep_for(10ms);
                    }
                    res.set_content("{}", "application/json");
                });
            });
    expect(post(rig, "/models/load", R"({"model":"c"})").status == 200, "load c");
    Reply reply = post(rig, "/v1/models/c/suspend", "");
    expect(reply.status == 400 && reply.error_code() == "model_suspend_unavailable",
           "a model without --model-suspend refuses to suspend");
    expect(rig.state("c") == ModelState::Loaded, "and stays loaded");

    // A request in flight keeps the model awake: suspend is refused, not waited for.
    std::thread inflight([&rig] { (void)get(rig, "/hold?model=a"); });
    expect(eventually([&] { return rig.leases("a") == 1; }), "the request holds a lease");
    reply = post(rig, "/v1/models/a/suspend", "");
    expect(reply.status == 409 && reply.error_code() == "model_busy",
           "suspend while a request runs is 409 model_busy");
    expect(rig.state("a") == ModelState::Loaded, "and the model stays loaded");
    inflight.join();
    expect(eventually([&] { return rig.leases("a") == 0; }), "the lease ends with the response");
    expect(post(rig, "/v1/models/a/suspend", "").status == 200, "idle again, it suspends");
}

void status_routes() {
    Rig rig(kTwoModels);
    for (const char* path : {"/props", "/metrics", "/v1/load"}) {
        const Reply reply = get(rig, path);
        expect(reply.status == 503 && reply.error_code() == "model_not_loaded",
               std::string(path) + " with nothing loaded is 503");
    }
    expect(rig.state("a") == ModelState::Unloaded, "a status read does not load a model");

    expect(post(rig, "/v1/chat/completions", R"({"model":"b"})").status == 200, "serve b");
    Reply reply = get(rig, "/props?model=a");
    expect(reply.status == 200 && reply.json["served_by"] == "b",
           "status routes read the model that served last, not the query's");
    expect(rig.up("b").last().target == "/props", "with the router's query parameter removed");
    expect(get(rig, "/metrics").json["served_by"] == "b", "/metrics too");
    expect(rig.state("a") == ModelState::Unloaded, "and still nothing is woken");
}

void headers_and_methods() {
    Rig rig(kTwoModels, {}, false, [](const std::string& id, Upstream& upstream) {
        upstream.server.Get("/custom", [id](const httplib::Request&, httplib::Response& res) {
            res.status = 418;
            res.set_header("X-Upstream-Only", "1");
            res.set_header("X-NInfer-Version", "from-model");
            res.set_header("Access-Control-Allow-Origin", "model");
            res.set_content("teapot " + id, "text/x-teapot");
        });
    });
    auto client = rig.client();
    const auto result = client.Get("/custom?model=a");
    expect(result && result->status == 418 && result->body == "teapot a",
           "the model's status and body pass through");
    expect(result && result->get_header_value("Content-Type") == "text/x-teapot",
           "and its content type");
    expect(result && result->get_header_value("X-Upstream-Only") == "1", "and its other headers");
    expect(result && result->get_header_value_count("X-NInfer-Version") == 1 &&
               result->get_header_value("X-NInfer-Version") != "from-model",
           "the router's version header replaces the model's");
    expect(result && result->get_header_value("Access-Control-Allow-Origin").empty(),
           "the model's CORS headers are dropped");

    const Reply deleted = del(rig, "/v1/responses/resp_9?model=b");
    expect(deleted.status == 200 && deleted.json["method"] == "DELETE" &&
               deleted.json["target"] == "/v1/responses/resp_9",
           "DELETE is relayed with its method");
    const Reply posted = post(rig, "/v1/responses?model=a", R"({"input":"hi"})");
    expect(posted.json["method"] == "POST" && posted.json["served_by"] == "a",
           "POST is relayed with its method");
}

// A body of `chunks` chunks of kChunk bytes, each byte a function of its position in the body.
constexpr std::size_t kChunk = 16 * 1024;

char pattern_byte(std::uint64_t position) { return static_cast<char>('a' + position % 23); }

void streamed_relay() {
    constexpr std::size_t kChunks = 4096; // 64 MiB: far more than any buffer on the way
    std::atomic<std::size_t> written{0};
    std::atomic<bool> upstream_finished{false};
    Rig rig(kTwoModels, {}, false, [&](const std::string&, Upstream& upstream) {
        upstream.server.Get("/sse", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider(
                "text/event-stream", [step = 0](size_t, httplib::DataSink& sink) mutable {
                    if (step == 3) {
                        sink.done();
                        return true;
                    }
                    const std::string event = "data: " + std::to_string(step++) + "\n\n";
                    std::this_thread::sleep_for(30ms);
                    return sink.write(event.data(), event.size());
                });
        });
        upstream.server.Get("/big", [&](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider(
                "application/octet-stream",
                [&, index = std::size_t{0}](size_t, httplib::DataSink& sink) mutable {
                    if (index == kChunks) {
                        sink.done();
                        return true;
                    }
                    std::string chunk(kChunk, 'x');
                    for (std::size_t i = 0; i < kChunk; ++i) {
                        chunk[i] = pattern_byte(index * kChunk + i);
                    }
                    if (!sink.write(chunk.data(), chunk.size())) { return false; }
                    ++index;
                    written = index;
                    return true;
                },
                [&](bool) { upstream_finished = true; });
        });
    });

    // Events arrive as the model emits them, in order, with the model's content type.
    {
        auto client = rig.client();
        std::vector<std::chrono::steady_clock::time_point> arrivals;
        std::string received;
        const auto result = client.Get(
            "/sse?model=a", httplib::Headers{},
            [&](const httplib::Response& response) {
                expect(response.get_header_value("Content-Type").rfind("text/event-stream", 0) == 0,
                       "a stream keeps its content type");
                return true;
            },
            [&](const char* data, size_t length) {
                arrivals.push_back(std::chrono::steady_clock::now());
                received.append(data, length);
                return true;
            });
        expect(result && result->status == 200, "the stream completes");
        expect(received == "data: 0\n\ndata: 1\n\ndata: 2\n\n", "its events arrive whole and in order");
        expect(arrivals.size() >= 2 && arrivals.back() - arrivals.front() > 40ms,
               "and as they are produced, not all at the end");
    }
    expect(eventually([&] { return rig.leases("a") == 0; }), "the lease ends with the stream");

    // A client that reads slowly holds the model back instead of making the router buffer it all.
    std::atomic<bool> release{false};
    std::atomic<bool> paused{false};
    std::uint64_t total = 0;
    bool intact         = true;
    std::thread reader([&] {
        auto client = rig.client();
        client.set_read_timeout(120, 0);
        const auto result = client.Get(
            "/big?model=b", httplib::Headers{}, [](const httplib::Response&) { return true; },
            [&](const char* data, size_t length) {
                if (!paused.exchange(true)) {
                    while (!release) { std::this_thread::sleep_for(10ms); }
                }
                for (size_t i = 0; i < length; ++i) {
                    if (data[i] != pattern_byte(total + i)) { intact = false; }
                }
                total += length;
                return true;
            });
        expect(result && result->status == 200, "the big body completes once the client reads");
    });
    expect(eventually([&] { return paused.load(); }), "the client reached its first chunk");
    // Wait for the model's output to stop growing: it is held back, not finished.
    std::size_t stable = 0;
    expect(eventually(
               [&] {
                   const std::size_t now = written.load();
                   std::this_thread::sleep_for(300ms);
                   const bool same = now == written.load();
                   stable          = now;
                   return same;
               },
               30s),
           "the model's output stalls behind a slow client");
    expect(stable > 0 && stable < kChunks / 2,
           "so the router did not buffer the response (" + std::to_string(stable) + " of " +
               std::to_string(kChunks) + " chunks written)");
    expect(!upstream_finished.load(), "and the model has not finished");
    expect(rig.leases("b") == 1, "the model stays leased while its response is in flight");
    release = true;
    reader.join();
    expect(total == kChunks * kChunk && intact, "every byte arrives, uncorrupted");
    expect(eventually([&] { return upstream_finished.load(); }), "then the model finishes");
    expect(eventually([&] { return rig.leases("b") == 0; }), "and the lease ends");
}

void client_disconnects() {
    std::atomic<bool> endless_ended{false};
    std::atomic<bool> endless_started{false};
    std::atomic<bool> endless_broke{false};
    std::atomic<bool> slow_saw_close{false};
    Rig rig(kTwoModels, {}, false, [&](const std::string&, Upstream& upstream) {
        upstream.server.Get("/endless", [&](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [&](size_t, httplib::DataSink& sink) {
                    endless_started = true;
                    std::this_thread::sleep_for(5ms);
                    const std::string event = "data: tick\n\n";
                    if (!sink.write(event.data(), event.size())) {
                        endless_broke = true;
                        return false;
                    }
                    return true;
                },
                [&](bool) { endless_ended = true; });
        });
        // Never sends headers: a prefill that takes as long as the client will wait.
        upstream.server.Get("/slow-headers", [&](const httplib::Request& req, httplib::Response& res) {
            for (int i = 0; i < 1500 && !req.is_connection_closed(); ++i) {
                std::this_thread::sleep_for(10ms);
            }
            if (req.is_connection_closed()) { slow_saw_close = true; }
            res.set_content("{}", "application/json");
        });
    });

    // The client leaves mid-stream: the model's request is cancelled and the lease released.
    {
        auto client = rig.client();
        std::size_t bytes = 0;
        const auto result = client.Get(
            "/endless?model=a", httplib::Headers{}, [](const httplib::Response&) { return true; },
            [&](const char*, size_t length) {
                bytes += length;
                return false; // hang up after the first piece
            });
        expect(bytes > 0, "the client received the first piece");
        (void)result;
    }
    expect(eventually([&] { return endless_ended.load(); }), "the model's request ends");
    expect(endless_broke.load() || endless_ended.load(), "the model sees the client gone");
    expect(eventually([&] { return rig.leases("a") == 0; }),
           "and the lease is released, so the model may sleep");

    // The client leaves before any headers: the router notices without waiting for the model.
    {
        auto client = rig.client();
        client.set_read_timeout(1, 0);
        const auto result = client.Get("/slow-headers?model=b");
        expect(!result, "the client gives up waiting");
    }
    expect(eventually([&] { return slow_saw_close.load(); }, 15s),
           "the router closes the upstream request of a client that left");
    expect(eventually([&] { return rig.leases("b") == 0; }), "and releases its lease");
}

void upstream_failures() {
    Rig rig(kTwoModels, {}, false, [](const std::string&, Upstream& upstream) {
        upstream.server.Get("/cut", [](const httplib::Request&, httplib::Response& res) {
            res.set_chunked_content_provider(
                "application/octet-stream", [sent = false](size_t, httplib::DataSink& sink) mutable {
                    if (sent) { return false; } // the model dies: no terminating chunk
                    sent = true;
                    sink.write("partial", 7);
                    return true;
                });
        });
    });

    // The model's port is dead: a 502 in the request's own envelope.
    expect(post(rig, "/models/load", R"({"model":"a"})").status == 200, "load a");
    rig.ports->set("a", free_port());
    Reply reply = post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(reply.status == 502 && reply.error_code() == "model_unreachable",
           "an unreachable model is 502 model_unreachable");
    reply = post(rig, "/v1/messages", R"({"model":"a"})");
    expect(reply.status == 502 && reply.json.value("type", "") == "error",
           "and on an Anthropic path, in its envelope");
    expect(eventually([&] { return rig.leases("a") == 0; }), "a failed relay releases its lease");

    // No port published at all: not serving.
    rig.ports->erase("a");
    reply = post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(reply.status == 503 && reply.error_code() == "model_not_loaded",
           "a model without a port is 503 model_not_loaded");

    // Headers out, then the model dies: the connection is dropped, not ended as if it finished.
    auto client = rig.client();
    std::string received;
    const auto result = client.Get(
        "/cut?model=b", httplib::Headers{}, [](const httplib::Response&) { return true; },
        [&](const char* data, size_t length) {
            received.append(data, length);
            return true;
        });
    expect(!result, "a stream cut short is an error for the client, not a complete body");
    expect(eventually([&] { return rig.leases("b") == 0; }), "and its lease is released");
}

void dead_listener_is_rebuilt() {
    Rig rig(kTwoModels);
    expect(post(rig, "/v1/chat/completions", R"({"model":"a"})").status == 200, "first request");
    expect(rig.journal.count("load a") == 1, "the model loaded once");
    rig.journal.failing.at("a") = true; // the model's accept loop died
    const Reply reply = post(rig, "/v1/chat/completions", R"({"model":"a"})");
    expect(reply.status == 200, "the next request is served");
    expect(rig.journal.count("unload a") == 1 && rig.journal.count("load a") == 2,
           "by a model rebuilt for it, not the dead one");
    expect(rig.state("a") == ModelState::Loaded && !rig.registry->status("a")->failed,
           "which is loaded and no longer marked failed");
}


// The Responses stores of the fake models: each id belongs to the model that made it, and only that
// model finds it again.
struct FakeResponses {
    std::mutex mutex;
    std::map<std::string, std::string> owner; // id -> model
    std::atomic<int> next{0};
    std::string last; // the id of the latest create, stored or not

    void note_created(const std::string& id) {
        std::lock_guard lock(mutex);
        last = id;
    }

    [[nodiscard]] std::string last_created() {
        std::lock_guard lock(mutex);
        return last;
    }

    void add(const std::string& id, const std::string& model) {
        std::lock_guard lock(mutex);
        owner[id] = model;
    }

    [[nodiscard]] bool owned(const std::string& id, const std::string& model) {
        std::lock_guard lock(mutex);
        const auto found = owner.find(id);
        return found != owner.end() && found->second == model;
    }

    void erase(const std::string& id) {
        std::lock_guard lock(mutex);
        owner.erase(id);
    }
};

void respond_not_found(httplib::Response& res) {
    res.status = 404;
    res.set_content(R"({"error":{"code":"not_found"}})", "application/json");
}

void install_responses(const std::string& model, Upstream& upstream, FakeResponses& store) {
    upstream.server.Post("/v1/responses", [&upstream, &store, model](const httplib::Request& req,
                                                                     httplib::Response& res) {
        upstream.note(req);
        const Json body = Json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_object() && body.contains("previous_response_id") &&
            !store.owned(body["previous_response_id"].get<std::string>(), model)) {
            respond_not_found(res);
            return;
        }
        if (body.is_object() && body.value("fail", false)) {
            // An error that still mentions an id: it must not be taken for a stored response.
            res.status = 400;
            res.set_content(Json{{"id", "resp_failed"}, {"error", "rejected"}}.dump(),
                            "application/json");
            return;
        }
        const std::string id = "resp_" + model + std::to_string(store.next++);
        // A create with store:false, or a stream that fails or is cut off, keeps nothing.
        store.note_created(id);
        const bool streamed = body.is_object() && body.value("stream", false);
        // abort_stream: cut off while generating, nothing stored. abort_committed: the model had
        // committed the response (it does so just before the last event) when the connection broke.
        const bool broken   = body.is_object() && (body.value("fail_stream", false) ||
                                                 body.value("abort_stream", false));
        const bool committed_abort = body.is_object() && body.value("abort_committed", false);
        if (!(body.is_object() && body.value("store", true) == false) && !broken) {
            store.add(id, model);
        }
        if (streamed && (broken || committed_abort)) {
            const std::string created = "event: response.created\ndata: {\"response\":{\"id\":\"" +
                                        id + "\"},\"type\":\"response.created\"}\n\n";
            const bool abort = !body.value("fail_stream", false);
            res.set_chunked_content_provider(
                "text/event-stream", [created, abort, step = 0](size_t, httplib::DataSink& sink) mutable {
                    if (step++ == 0) { return sink.write(created.data(), created.size()); }
                    if (abort) { return false; } // the connection drops mid-stream
                    const std::string failed = "event: response.failed\ndata: {}\n\n";
                    if (step == 2) { return sink.write(failed.data(), failed.size()); }
                    sink.done();
                    return true;
                });
            return;
        }
        if (!streamed && body.is_object() && body.value("abort_body", false)) {
            // Stored, then the reply is cut off after its first bytes: the real server commits the
            // response before it writes anything.
            const std::string head = "{\"id\":\"" + id + "\",\"object\":\"resp";
            res.set_chunked_content_provider(
                "application/json", [head, step = 0](size_t, httplib::DataSink& sink) mutable {
                    if (step++ == 0) { return sink.write(head.data(), head.size()); }
                    return false; // the connection drops mid-body
                });
            return;
        }
        if (streamed) {
            // The id is split across two writes.
            const std::string first = "event: response.created\ndata: {\"response\":{\"id\":\"" +
                                      id.substr(0, 7);
            const std::string second = id.substr(7) + "\",\"object\":\"response\"},\"type\":" +
                                       "\"response.created\"}\n\n";
            res.set_chunked_content_provider(
                "text/event-stream", [first, second, step = 0](size_t, httplib::DataSink& sink) mutable {
                    if (step == 0) {
                        ++step;
                        return sink.write(first.data(), first.size());
                    }
                    if (step == 1) {
                        ++step;
                        std::this_thread::sleep_for(50ms);
                        return sink.write(second.data(), second.size());
                    }
                    sink.done();
                    return true;
                });
            return;
        }
        res.set_content(Json{{"id", id}, {"object", "response"}, {"served_by", model}}.dump(),
                        "application/json");
    });
    upstream.server.Post("/v1/responses/input_tokens", [&upstream, &store, model](
                                                           const httplib::Request& req,
                                                           httplib::Response& res) {
        upstream.note(req);
        const Json body = Json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_object() && body.contains("previous_response_id") &&
            !store.owned(body["previous_response_id"].get<std::string>(), model)) {
            respond_not_found(res);
            return;
        }
        res.set_content(Json{{"object", "response.input_tokens"}, {"served_by", model}}.dump(),
                        "application/json");
    });
    const auto lookup = [&upstream, &store, model](const httplib::Request& req,
                                                   httplib::Response& res) {
        upstream.note(req);
        const std::string id = req.matches[1].str();
        if (!store.owned(id, model)) {
            respond_not_found(res);
            return;
        }
        if (req.method == "DELETE") { store.erase(id); }
        res.set_content(Json{{"id", id}, {"served_by", model}, {"method", req.method},
                             {"target", req.target}}
                            .dump(),
                        "application/json");
    };
    upstream.server.Get(R"(/v1/responses/([^/]+)/input_items)", lookup);
    upstream.server.Post(R"(/v1/responses/([^/]+)/cancel)", lookup);
    upstream.server.Get(R"(/v1/responses/([^/]+))", lookup);
    upstream.server.Delete(R"(/v1/responses/([^/]+))", lookup);
}

void response_owner_bookkeeping() {
    // "a" keeps 2 ids, "b" keeps 1.
    ResponseOwners owners([](const std::string& model) { return model == "a" ? 2u : 1u; });
    owners.record("r1", "a");
    owners.record("r2", "b");
    owners.record("r3", "a");
    expect(owners.owner("r2") == "b" && owners.owner("r1") == "a", "an owner is looked up by id");
    expect(!owners.owner("nope"), "an unknown id has none");
    // r1 was used after r3 was made: r3 is now a's least recently used.
    owners.record("r4", "a");
    expect(owners.size() == 3 && !owners.owner("r3") && owners.owner("r1") == "a" &&
               owners.owner("r4") == "a" && owners.owner("r2") == "b",
           "a model's least recently used id is forgotten at its own limit");
    owners.record("r5", "b");
    expect(owners.size() == 3 && !owners.owner("r2") && owners.owner("r5") == "b" &&
               owners.owner("r1") && owners.owner("r4"),
           "a model at its limit forgets only its own ids");
    owners.record("r1", "b");
    expect(owners.owner("r1") == "b" && !owners.owner("r5") && owners.size() == 2,
           "recording an id again moves it to its new owner, within that owner's limit");
    owners.forget("r1");
    owners.forget("r1");
    expect(!owners.owner("r1") && owners.size() == 1, "forgetting is idempotent");
    ResponseOwners tiny([](const std::string&) { return std::size_t{0}; });
    tiny.record("x", "a");
    tiny.record("y", "a");
    expect(tiny.size() == 1 && tiny.owner("y"), "a limit of zero still keeps one");

    const std::string body = R"({"access_programs":null,"id":"resp_0123abcd","object":"response"})";
    ResponseIdScanner whole;
    whole.feed(body);
    expect(whole.id() == "resp_0123abcd", "a JSON reply's id is found");
    for (std::size_t piece = 1; piece < body.size(); ++piece) {
        ResponseIdScanner scanner;
        for (std::size_t at = 0; at < body.size(); at += piece) {
            scanner.feed(std::string_view(body).substr(at, piece));
        }
        expect(scanner.id() == "resp_0123abcd",
               "the id is found across pieces of " + std::to_string(piece));
    }
    const std::string stream =
        "event: response.created\ndata: {\"response\":{\"id\":\"resp_1\"}}\n\nevent: response.failed\n";
    for (std::size_t piece = 1; piece < stream.size(); ++piece) {
        ResponseIdScanner scanner;
        for (std::size_t at = 0; at < stream.size(); at += piece) {
            scanner.feed(std::string_view(stream).substr(at, piece));
        }
        expect(scanner.failed() && scanner.id() == "resp_1",
               "a failure event is found across pieces of " + std::to_string(piece));
    }
    ResponseIdScanner quoted;
    quoted.feed(R"({"id":"resp_2","output":"event: response.failed\nevent: response.failed"})");
    expect(!quoted.failed(), "text quoted in a JSON body is not a failure event");
    ResponseIdScanner completed;
    completed.feed("event: response.created\ndata: {}\n\nevent: response.completed\n");
    expect(!completed.failed(), "a completed stream has not failed");
    ResponseIdScanner sse;
    sse.feed("event: response.created\ndata: {\"response\":{\"id\":\"resp_ff\",\"object\":\"re");
    expect(sse.id() == "resp_ff", "an SSE event's response id is found");
    sse.feed(R"({"id":"resp_other"})");
    expect(sse.id() == "resp_ff", "and the first one stays");
    ResponseIdScanner none;
    none.feed(R"({"previous_response_id":"resp_x","id":"msg_1","items":[{"id":"rs_1"}]})");
    expect(!none.id(), "only a response id counts, not a previous_response_id or an item id");
    ResponseIdScanner late;
    late.feed(std::string(70 * 1024, ' '));
    late.feed(R"({"id":"resp_late"})");
    expect(!late.id(), "an id beyond the scanned prefix is left unrecorded");

    expect(response_id_from_path("/v1/responses/resp_1") == "resp_1", "GET/DELETE path");
    expect(response_id_from_path("/v1/responses/resp_1/input_items") == "resp_1", "input_items path");
    expect(response_id_from_path("/v1/responses/resp_1/cancel") == "resp_1", "cancel path");
    for (const char* path : {"/v1/responses", "/v1/responses/", "/v1/responses/input_tokens",
                             "/v1/responses/compact", "/v1/responses/resp_1/other",
                             "/v1/chat/completions", "/v1/responsesx/resp_1"}) {
        expect(!response_id_from_path(path), std::string("no id in ") + path);
    }
}

std::string sse_response_id(const std::string& body) {
    ResponseIdScanner scanner;
    scanner.feed(body);
    return scanner.id().value_or(std::string());
}

void responses_follow_up_the_owning_model() {
    FakeResponses store;
    Rig rig(kTwoModels, {}, false, [&](const std::string& id, Upstream& upstream) {
        install_responses(id, upstream, store);
    });
    const auto served_by = [](const Reply& reply) { return reply.json.value("served_by", ""); };

    const std::string id_a = post(rig, "/v1/responses", R"({"model":"a","input":"hi"})")
                                 .json.value("id", "");
    const std::string id_b = post(rig, "/v1/responses", R"({"model":"b","input":"hi"})")
                                 .json.value("id", "");
    expect(!id_a.empty() && !id_b.empty() && id_a != id_b, "both models made a response");

    // b served last: without the owner map every one of these went to b and missed.
    Reply reply = get(rig, "/v1/responses/" + id_a);
    expect(reply.status == 200 && served_by(reply) == "a", "GET goes to the model that made it");
    reply = get(rig, "/v1/responses/" + id_b);
    expect(reply.status == 200 && served_by(reply) == "b", "and the other one's goes to the other");
    reply = get(rig, "/v1/responses/" + id_a + "/input_items?limit=5");
    expect(reply.status == 200 && served_by(reply) == "a" &&
               reply.json["target"] == "/v1/responses/" + id_a + "/input_items?limit=5",
           "input_items too, with its query kept");
    get(rig, "/v1/responses/" + id_b);
    reply = post(rig, "/v1/responses/" + id_a + "/cancel", "");
    expect(reply.status == 200 && served_by(reply) == "a", "so does cancel");

    // A model the client names is the client's choice, even a wrong one.
    const std::size_t a_before = rig.up("a").count();
    reply                      = get(rig, "/v1/responses/" + id_a + "?model=b");
    expect(reply.status == 404 && rig.up("a").count() == a_before,
           "an explicit ?model= is honoured");

    // previous_response_id chains in the model that holds the response.
    reply = post(rig, "/v1/responses", R"({"input":"more","previous_response_id":")" + id_a + R"("})");
    expect(reply.status == 200 && served_by(reply) == "a", "a follow-up create with no model");
    expect(upstream_body(rig, "a")["model"] == "a" &&
               upstream_body(rig, "a")["previous_response_id"] == id_a,
           "is given that model's name and keeps its body");
    reply = post(rig, "/v1/responses",
                 R"({"model":"b","input":"more","previous_response_id":")" + id_a + R"("})");
    expect(reply.status == 404 && rig.up("b").last().target == "/v1/responses",
           "one that names another model goes there, and is refused by it");
    const std::string id_chain =
        post(rig, "/v1/responses", R"({"input":"x","previous_response_id":")" + id_a + R"("})")
            .json.value("id", "");
    get(rig, "/v1/responses/" + id_b);
    expect(served_by(get(rig, "/v1/responses/" + id_chain)) == "a",
           "and a response made by a follow-up is owned too");

    // A streamed create is learnt from its first event, though the id arrives in two pieces.
    auto client      = rig.client();
    const auto event = client.Post("/v1/responses?model=b", R"({"stream":true,"input":"hi"})",
                                   "application/json");
    const std::string id_stream = event ? sse_response_id(event->body) : std::string();
    expect(!id_stream.empty(), "a streamed create returns its id");
    post(rig, "/v1/responses", R"({"model":"a","input":"hi"})");
    reply = get(rig, "/v1/responses/" + id_stream);
    expect(reply.status == 200 && served_by(reply) == "b", "which routes its follow-ups too");

    // Counting the tokens of a follow-up reads the same stored response.
    get(rig, "/v1/responses/" + id_b); // b served last
    reply = post(rig, "/v1/responses/input_tokens",
                 R"({"input":"more","previous_response_id":")" + id_a + R"("})");
    expect(reply.status == 200 && served_by(reply) == "a" &&
               upstream_body(rig, "a")["model"] == "a",
           "input_tokens with previous_response_id goes to the owner");
    get(rig, "/v1/responses/" + id_b); // b served last
    reply = post(rig, "/v1/responses/input_tokens", R"({"input":"more"})");
    expect(reply.status == 200 && served_by(reply) == "b", "and without it, by default");

    // A create that is not stored, or whose stream ends in response.failed, has no owner. The model
    // is learnt from the id in the first event, so these are forgotten once the stream has ended.
    // A stream that is only cut off keeps its owner: the router cannot see whether the model had
    // committed it (it does so just before the last event), and keeping is never worse than master.
    const auto owner_forgotten = [&](const std::string& id) {
        return eventually([&] {
            get(rig, "/v1/responses/" + id_b); // b served last
            const std::size_t a_calls = rig.up("a").count();
            return get(rig, "/v1/responses/" + id + "?x=1").status == 404 &&
                   rig.up("a").count() == a_calls;
        });
    };
    const auto create_id = [&](const std::string& body) {
        auto stream_client = rig.client();
        stream_client.Post("/v1/responses?model=a", body, "application/json"); // may be cut off
        return store.last_created();
    };
    const auto json_id = [&](const std::string& body) {
        return post(rig, "/v1/responses", body).json.value("id", "");
    };
    std::string id = json_id(R"({"model":"a","store":false,"input":"x"})");
    expect(!id.empty() && owner_forgotten(id), "a create with store:false is not remembered");
    id = create_id(R"({"stream":true,"store":false,"input":"x"})");
    expect(!id.empty() && owner_forgotten(id), "nor is a streamed one");
    id = create_id(R"({"stream":true,"fail_stream":true,"input":"x"})");
    expect(!id.empty() && owner_forgotten(id), "a stream that ends in response.failed is forgotten");
    const auto owner_kept = [&](const std::string& id) {
        get(rig, "/v1/responses/" + id_b); // b served last
        const std::size_t a_calls = rig.up("a").count();
        get(rig, "/v1/responses/" + id);
        return rig.up("a").count() == a_calls + 1;
    };
    id = create_id(R"({"stream":true,"abort_stream":true,"input":"x"})");
    expect(!id.empty() && eventually([&] { return rig.leases("a") == 0; }) && owner_kept(id),
           "a stream cut off before its end keeps its owner (nothing says it was not stored)");
    id = create_id(R"({"stream":true,"abort_committed":true,"input":"x"})");
    expect(!id.empty() && eventually([&] { return rig.leases("a") == 0; }) &&
               served_by(get(rig, "/v1/responses/" + id)) == "a",
           "and one cut off after the model stored it is still found at its owner");

    // The model reads the body whatever the Content-Type says, so a store:false create without a
    // JSON header is just as unstored, streamed or not.
    const auto create_as = [&](const std::string& content_type, const std::string& body) {
        auto plain_client = rig.client();
        plain_client.Post("/v1/responses?model=a", body, content_type);
        return store.last_created();
    };
    id = create_as("text/plain", R"({"store":false,"input":"x"})");
    expect(!id.empty() && owner_forgotten(id), "store:false is honoured without a JSON Content-Type");
    id = create_as("text/plain", R"({"stream":true,"store":false,"input":"x"})");
    expect(!id.empty() && owner_forgotten(id), "also when streamed");
    {
        auto bare = rig.client();
        const auto sent = bare.Post("/v1/responses?model=a", httplib::Headers{}, R"({"store":false})",
                                    "");
        id = store.last_created();
        expect(sent && !id.empty() && owner_forgotten(id), "or with no Content-Type at all");
    }
    // ... while a stored one without the header is still remembered.
    id = create_as("text/plain", R"({"input":"x"})");
    get(rig, "/v1/responses/" + id_b);
    expect(!id.empty() && served_by(get(rig, "/v1/responses/" + id)) == "a",
           "and a stored create without the header is remembered");

    // Routing reads the body the same way: a body with no JSON Content-Type names its model, has it
    // rewritten to the model's id, and a previous_response_id in it finds the owner.
    {
        auto plain = rig.client();
        get(rig, "/v1/responses/" + id_b); // b served last
        const std::size_t b_before = rig.up("b").count();
        plain.Post("/v1/responses", R"({"model":"a","input":"x"})", "text/plain");
        expect(rig.up("b").count() == b_before && upstream_body(rig, "a")["model"] == "a",
               "a header-less body's \"model\" selects the model");
        plain.Post("/v1/responses", R"({"model":"alpha","input":"x"})", "text/plain");
        expect(upstream_body(rig, "a")["model"] == "a", "and its alias is rewritten to the id");
        get(rig, "/v1/responses/" + id_b); // b served last
        plain.Post("/v1/responses", R"({"input":"x","previous_response_id":")" + id_a + R"("})",
                   "text/plain");
        expect(upstream_body(rig, "a")["previous_response_id"] == id_a &&
                   upstream_body(rig, "a")["model"] == "a",
               "a header-less previous_response_id goes to the owner, with the selector filled in");
        reply = to_reply(plain.Post("/v1/responses", R"({"model":"","input":"x"})", "text/plain"));
        expect(reply.status == 400, "an empty selector is refused whatever the Content-Type");
    }

    // A reply that is not streamed is written after the model stored the response, so one cut off
    // after its id was seen keeps its owner (as does a cut-off stream: see above).
    id = create_id(R"({"abort_body":true,"input":"x"})");
    // The relay has ended (its lease is back) before the owner is looked up, so a late forget
    // cannot be missed.
    expect(eventually([&] { return rig.leases("a") == 0; }), "the cut-off relay ends");
    get(rig, "/v1/responses/" + id_b); // b served last
    expect(!id.empty() && served_by(get(rig, "/v1/responses/" + id)) == "a",
           "a non-streamed reply cut off after its id keeps its owner");

    // A failed create names no response.
    post(rig, "/v1/responses", R"({"model":"a","fail":true})");
    get(rig, "/v1/responses/" + id_b); // b served last
    const std::size_t a_calls = rig.up("a").count();
    reply                     = get(rig, "/v1/responses/resp_failed");
    expect(reply.status == 404 && rig.up("a").count() == a_calls,
           "an error reply's id is not recorded");

    // Delete: the response is gone, and no longer anyone's.
    reply = del(rig, "/v1/responses/" + id_a);
    expect(reply.status == 200 && served_by(reply) == "a", "DELETE goes to the owner");
    get(rig, "/v1/responses/" + id_b);
    const std::size_t a_after_delete = rig.up("a").count();
    reply                            = get(rig, "/v1/responses/" + id_a);
    expect(reply.status == 404 && rig.up("a").count() == a_after_delete,
           "after it, the id routes by default again");
    // A delete the model refuses leaves the owner in place.
    reply = del(rig, "/v1/responses/" + id_stream + "?model=a");
    expect(reply.status == 404, "a delete sent to the wrong model is refused");
    expect(served_by(get(rig, "/v1/responses/" + id_stream)) == "b",
           "and the owner is remembered");

    // The model unloads and comes back for the follow-up (its store outlives it).
    post(rig, "/models/unload", R"({"model":"b"})");
    expect(rig.state("b") == ModelState::Unloaded, "b is unloaded");
    reply = get(rig, "/v1/responses/" + id_stream);
    expect(reply.status == 200 && served_by(reply) == "b" && rig.state("b") == ModelState::Loaded,
           "a follow-up to an unloaded owner loads it");
}

void owner_map_is_bounded() {
    FakeResponses store;
    // Two models of one record each: the map holds two ids, the oldest goes first.
    Rig rig(kTwoModels, {}, false,
            [&](const std::string& id, Upstream& upstream) { install_responses(id, upstream, store); },
            2, /*store_records=*/1);
    const auto make = [&](const std::string& model) {
        return post(rig, "/v1/responses", R"({"model":")" + model + R"(","input":"x"})")
            .json.value("id", "");
    };
    const std::string first  = make("a");
    const std::string second = make("a");
    const std::string third  = make("a");
    // b last, so a forgotten id would be tried there.
    post(rig, "/v1/chat/completions", R"({"model":"b"})");
    const std::size_t a_calls = rig.up("a").count();
    expect(get(rig, "/v1/responses/" + third).status == 200 && rig.up("a").count() == a_calls + 1,
           "a recent id still routes to its owner");
    post(rig, "/v1/chat/completions", R"({"model":"b"})");
    const std::size_t b_calls = rig.up("b").count();
    expect(get(rig, "/v1/responses/" + first).status == 404 && rig.up("b").count() == b_calls + 1,
           "an id forgotten at capacity routes by default");
    (void)second;
}

void owner_map_keeps_each_models_quota() {
    // Each model has its own bound, so a busy model cannot push out another model's ids: a keeps 2,
    // b keeps 3, and b creating many leaves a's alone.
    FakeResponses store;
    Rig rig({{.id = "a", .store_records = 2}, {.id = "b", .store_records = 3}}, {}, false,
            [&](const std::string& id, Upstream& upstream) { install_responses(id, upstream, store); });
    expect(rig.registry->response_store_records("a") == 2u &&
               rig.registry->response_store_records("b") == 3u &&
               !rig.registry->response_store_records("nope"),
           "each model has its own bound");
    const auto make = [&](const std::string& model) {
        return post(rig, "/v1/responses", R"({"model":")" + model + R"(","input":"x"})")
            .json.value("id", "");
    };
    std::vector<std::string> a_ids;
    for (int i = 0; i < 2; ++i) { a_ids.push_back(make("a")); }
    std::vector<std::string> b_ids;
    for (int i = 0; i < 8; ++i) { b_ids.push_back(make("b")); }
    // "a" served last, so a forgotten b id would be tried there, and a forgotten a id on b.
    const auto owned_by = [&](const std::string& id, const std::string& expected,
                              const std::string& other) {
        post(rig, "/v1/chat/completions", R"({"model":")" + other + R"("})");
        const std::size_t calls = rig.up(expected).count();
        return !id.empty() && get(rig, "/v1/responses/" + id).status == 200 &&
               rig.up(expected).count() == calls + 1;
    };
    for (const std::string& id : a_ids) {
        expect(owned_by(id, "a", "b"), "another model's creates leave an id routable");
    }
    for (std::size_t i = b_ids.size() - 3; i < b_ids.size(); ++i) {
        expect(owned_by(b_ids[i], "b", "a"), "a model keeps its latest ids up to its own limit");
    }
    for (std::size_t i = 0; i + 3 < b_ids.size(); ++i) {
        post(rig, "/v1/chat/completions", R"({"model":"a"})");
        const std::size_t b_calls = rig.up("b").count();
        expect(get(rig, "/v1/responses/" + b_ids[i]).status == 404 && rig.up("b").count() == b_calls,
               "a model forgets its older ids beyond its limit (routed by default to a)");
    }
}

} // namespace

int main() {
    try {
        api_key_handling();
        cors_headers();
        selector_precedence();
        models_listing();
        lifecycle_endpoints();
        residency_suspend_resume();
        suspend_refusals();
        status_routes();
        headers_and_methods();
        streamed_relay();
        client_disconnects();
        upstream_failures();
        dead_listener_is_rebuilt();
        response_owner_bookkeeping();
        responses_follow_up_the_owning_model();
        owner_map_is_bounded();
        owner_map_keeps_each_models_quota();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "router server: ok\n";
    return 0;
}
