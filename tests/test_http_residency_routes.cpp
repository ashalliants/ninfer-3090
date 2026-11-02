// The model suspend / residency HTTP API, driven through a real HttpServer over loopback with a
// GenerationService that has no Engine. The fake mirrors the rules GenerationService and the Engine
// enforce (refuse while a request is admitted, refuse when suspend is off, a failed resume keeps the
// model suspended and records last_error), so what is checked here is the server's side: routing,
// model id matching, body validation, status mapping and JSON shape. The service-level gate itself
// needs an Engine and is covered by the GPU tests; the --sleep-idle-seconds sleeper loop is run for
// real against the fake.

#include "serve/generation_service.h"
#include "serve/http_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

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

bool eventually(const std::function<bool()>& condition, std::chrono::milliseconds limit = 10s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) { return true; }
        std::this_thread::sleep_for(10ms);
    }
    return condition();
}

constexpr const char* kModel = "fake-model";

// A GenerationService without an Engine. Residency follows the Engine's rules; everything else the
// server calls answers with empty defaults.
class FakeService final : public GenerationService {
public:
    explicit FakeService(ServeOptions options, bool suspend_enabled = true)
        : GenerationService(std::move(options), NoEngine{}) {
        engine_options_.suspend.enabled = suspend_enabled;
        status_.enabled                 = suspend_enabled;
        load_.model_name                = kModel;
        start_idle_sleeper();
    }

    ~FakeService() override { stop_idle_sleeper(); }

    // What the test turns: requests in flight, and the failures the next call raises.
    std::atomic<std::size_t> admitted{0};
    std::atomic<bool> resume_fails{false};
    // The configured --model-suspend auto-resume default (EngineOptions::suspend.auto_resume).
    bool default_auto_resume = true;
    std::atomic<bool> suspend_unavailable{false};
    std::atomic<bool> suspend_crashes{false};
    std::atomic<int> suspend_refusals{0}; // the next N suspends refuse as busy

    // What the server asked of it.
    std::atomic<int> suspend_calls{0};
    std::atomic<int> resume_calls{0};
    mutable std::atomic<int> prepare_calls{0};

    const ninfer::EngineOptions& engine_options() const override { return engine_options_; }
    ninfer::LoadSummary load_summary() const override { return load_; }
    ninfer::MemorySummary memory_summary() const override { return {}; }
    ninfer::RuntimeStats runtime_stats() const override { return {}; }
    bool is_available() const override { return true; }
    ninfer::ModelSamplingDefaults sampling_defaults() const override { return {}; }
    std::size_t admitted_requests() const override { return admitted.load(); }

    ninfer::ResidencyStatus residency() const override {
        std::lock_guard lock(mutex_);
        return status_;
    }

    ninfer::ResidencyStatus suspend(std::optional<bool> auto_resume) override {
        ++suspend_calls;
        std::lock_guard lock(mutex_);
        if (!status_.enabled) {
            throw std::invalid_argument("model suspend is off (--model-suspend)");
        }
        if (admitted.load() != 0 || suspend_refusals.load() > 0) {
            if (suspend_refusals.load() > 0) { --suspend_refusals; }
            throw ninfer::RequestError(ninfer::RequestErrorKind::Overloaded,
                                       "model suspend requires an idle server");
        }
        if (suspend_unavailable) {
            throw ninfer::RequestError(ninfer::RequestErrorKind::Unavailable, "engine restarting");
        }
        if (suspend_crashes) { throw std::runtime_error("device copy failed"); }
        if (status_.state == ninfer::ModelResidency::Resident) {
            status_.state = ninfer::ModelResidency::Suspended;
            ++status_.suspend_count;
            status_.last_suspend = ninfer::ResidencyTransition{.seconds = 0.5, .state_bytes = 1024};
        }
        // As Engine::suspend: every successful call, including one on an already suspended model,
        // sets auto_resume to the request's value or, when none is given, to the configured default.
        status_.auto_resume = auto_resume.value_or(default_auto_resume);
        return status_;
    }

    ninfer::ResidencyStatus resume() override {
        ++resume_calls;
        std::lock_guard lock(mutex_);
        if (!status_.enabled) {
            throw std::invalid_argument("model suspend is off (--model-suspend)");
        }
        if (resume_fails) {
            status_.last_error = "could not restore device state";
            throw std::runtime_error(status_.last_error);
        }
        status_.last_error.clear();
        if (status_.state == ninfer::ModelResidency::Suspended) {
            status_.state = ninfer::ModelResidency::Resident;
            ++status_.resume_count;
            status_.last_resume = ninfer::ResidencyTransition{.seconds = 0.25, .weight_bytes = 2048};
        }
        // As Engine::resume: auto_resume is left as the last suspend set it.
        return status_;
    }

    // What the Engine does for a request that arrives while the model is held suspended.
    PreparedRequest prepare(const GenerationRequest&, GenerationConsumerMode,
                            ninfer::GenerationObservationOptions, std::function<bool()>,
                            ninfer::ContextCacheHints) const override {
        ++prepare_calls;
        std::lock_guard lock(mutex_);
        if (status_.state == ninfer::ModelResidency::Suspended && !status_.auto_resume) {
            // As GenerationService::prepare reports an Engine request error.
            throw ApiException(request_error_to_api_error(ninfer::RequestError(
                ninfer::RequestErrorKind::Unavailable,
                "the model is suspended and auto_resume is off")));
        }
        throw std::runtime_error("the fake service does not generate");
    }

private:
    ninfer::EngineOptions engine_options_;
    ninfer::LoadSummary load_;
    mutable std::mutex mutex_;
    ninfer::ResidencyStatus status_;
};

ServeOptions make_options(std::uint32_t sleep_idle_seconds = 0) {
    ServeOptions options;
    options.host              = "127.0.0.1";
    options.model_id_override = kModel;
    options.sleep_idle_seconds = sleep_idle_seconds;
    return options;
}

// An HttpServer on a free loopback port, attached to a fake.
struct Rig {
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>(
        "test", std::make_shared<spdlog::sinks::null_sink_mt>());
    FakeService service;
    HttpServer server;
    std::thread listener;
    int port = 0;

    explicit Rig(bool suspend_enabled = true)
        : service(make_options(), suspend_enabled), server(make_options(), logger) {
        port = server.bind_any_port();
        if (port <= 0) { throw std::runtime_error("HTTP server could not bind"); }
        server.attach(service);
        listener = std::thread([this] { server.listen(); });
        // A throw from a constructor skips ~Rig, and destroying a joinable std::thread
        // terminates, so stop and join the listener here before rethrowing.
        try {
            wait_until_accepting();
        } catch (...) {
            server.stop();
            listener.join();
            throw;
        }
    }

    // bind_any_port() only reserves the socket; the accept loop starts on `listener` a moment
    // later, and a request sent before then is refused. Probe until the loop answers (any status
    // will do) so the first route request cannot race listener startup.
    void wait_until_accepting() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            auto probe = client();
            // Short timeouts keep a stalled probe from outlasting the deadline.
            probe.set_connection_timeout(0, 200000);
            probe.set_read_timeout(0, 200000);
            if (probe.Get("/__ready_probe")) { return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        throw std::runtime_error("HTTP server did not start accepting connections");
    }

    ~Rig() {
        server.stop();
        if (listener.joinable()) { listener.join(); }
    }

    [[nodiscard]] httplib::Client client() const {
        httplib::Client out("127.0.0.1", port);
        out.set_connection_timeout(10, 0);
        out.set_read_timeout(30, 0);
        return out;
    }
};

struct Reply {
    int status = 0;
    Json json;

    [[nodiscard]] std::string error_code() const {
        return json.contains("error") ? json["error"].value("code", std::string()) : std::string();
    }
};

Reply to_reply(const httplib::Result& result) {
    Reply out;
    if (!result) { return out; }
    out.status = result->status;
    out.json   = Json::parse(result->body, nullptr, /*allow_exceptions=*/false);
    return out;
}

Reply get(const Rig& rig, const std::string& target) {
    auto client = rig.client();
    return to_reply(client.Get(target));
}

Reply post(const Rig& rig, const std::string& target, const std::string& body = {}) {
    auto client = rig.client();
    return to_reply(client.Post(target, body, "application/json"));
}

const std::string kResidency = std::string("/v1/models/") + kModel + "/residency";
const std::string kSuspend   = std::string("/v1/models/") + kModel + "/suspend";
const std::string kResume    = std::string("/v1/models/") + kModel + "/resume";

void residency_shape() {
    Rig rig;
    const Reply reply = get(rig, kResidency);
    expect(reply.status == 200, "residency answers 200");
    expect(reply.json.value("object", "") == "model.residency" && reply.json["model"] == kModel,
           "it names the object and the model");
    expect(reply.json["enabled"] == true && reply.json["state"] == "resident" &&
               reply.json["auto_resume"] == true,
           "a fresh model is resident with auto-resume");
    expect(reply.json["suspend_count"] == 0 && reply.json["resume_count"] == 0,
           "no transitions yet");
    expect(reply.json["last_suspend"].is_null() && reply.json["last_resume"].is_null() &&
               reply.json["last_error"].is_null(),
           "and nothing reported for them");
    for (const char* key : {"releasable_device_bytes", "host_snapshot_bytes", "host_reserved_bytes"}) {
        expect(reply.json.contains(key) && reply.json[key].is_number(),
               std::string(key) + " is a number");
    }
}

void unknown_model_ids() {
    Rig rig;
    for (const char* verb : {"residency", "suspend", "resume"}) {
        const std::string target = std::string("/v1/models/other-model/") + verb;
        const Reply reply        = std::string(verb) == "residency" ? get(rig, target)
                                                                    : post(rig, target);
        expect(reply.status == 404 && reply.error_code() == "model_not_found",
               std::string("an unknown id on ") + verb + " is 404 model_not_found");
    }
    expect(rig.service.suspend_calls == 0 && rig.service.resume_calls == 0,
           "an unknown id never reaches the service");
}

void suspend_body_validation() {
    Rig rig;
    for (const char* body : {"[]", "5", R"("x")", "not json", R"({"unknown":true})",
                             R"({"auto_resume":"yes"})", R"({"auto_resume":1})",
                             R"({"auto_resume":null})", R"({"auto_resume":true,"extra":1})"}) {
        const Reply reply = post(rig, kSuspend, body);
        expect(reply.status == 400 && reply.error_code() == "invalid_request",
               std::string("a bad suspend body is 400 invalid_request: ") + body);
    }
    expect(rig.service.suspend_calls == 0, "a bad body never reaches the service");
    expect(get(rig, kResidency).json["state"] == "resident", "and suspends nothing");
}

void suspend_and_resume() {
    Rig rig;
    Reply reply = post(rig, kSuspend);
    expect(reply.status == 200 && reply.json["state"] == "suspended" &&
               reply.json["auto_resume"] == true && reply.json["suspend_count"] == 1,
           "an empty body suspends, keeping auto-resume");
    expect(reply.json["last_suspend"].is_object() &&
               reply.json["last_suspend"]["state_bytes"] == 1024 &&
               reply.json["last_suspend"]["ms"] == 500.0,
           "and reports how the suspend went");
    expect(get(rig, kResidency).json["state"] == "suspended", "residency agrees");

    reply = post(rig, kResume);
    expect(reply.status == 200 && reply.json["state"] == "resident" &&
               reply.json["resume_count"] == 1 && reply.json["last_resume"]["weight_bytes"] == 2048,
           "resume brings it back and reports the transition");

    reply = post(rig, kSuspend, "{}");
    expect(reply.status == 200 && reply.json["state"] == "suspended", "{} suspends too");
    post(rig, kResume, "{}");
    reply = post(rig, kSuspend, R"({"auto_resume":false})");
    expect(reply.status == 200 && reply.json["auto_resume"] == false &&
               reply.json["state"] == "suspended",
           "auto_resume false is passed through");
    reply = post(rig, kSuspend, R"({"auto_resume":true})");
    expect(reply.status == 200 && reply.json["auto_resume"] == true && reply.json["suspend_count"] == 3,
           "suspending a suspended model only updates auto_resume");
    reply = post(rig, kSuspend);
    expect(reply.status == 200 && reply.json["auto_resume"] == true &&
               reply.json["suspend_count"] == 3,
           "a repeat suspend without a value resets auto_resume to the default");
    rig.service.default_auto_resume = false;
    reply                           = post(rig, kSuspend);
    expect(reply.status == 200 && reply.json["auto_resume"] == false,
           "and the default is the configured one, not always true");
    reply = post(rig, kResume);
    expect(reply.status == 200 && reply.json["state"] == "resident" &&
               reply.json["auto_resume"] == false,
           "resume leaves auto_resume as the last suspend set it");
}

void resume_body_validation() {
    Rig rig;
    post(rig, kSuspend);
    for (const char* body : {"[]", "5", "not json", R"({"auto_resume":true})", R"({"x":1})"}) {
        const Reply reply = post(rig, kResume, body);
        expect(reply.status == 400 && reply.error_code() == "invalid_request",
               std::string("a non-empty resume body is 400 invalid_request: ") + body);
    }
    expect(rig.service.resume_calls == 0, "a bad body never reaches the service");
    expect(get(rig, kResidency).json["state"] == "suspended", "and resumes nothing");
    expect(post(rig, kResume, "{}").status == 200, "{} is an empty body");
}

void busy_server_refuses_suspend() {
    Rig rig;
    rig.service.admitted = 1;
    Reply reply          = post(rig, kSuspend);
    expect(reply.status == 409 && reply.error_code() == "model_busy",
           "suspend while a request is admitted is 409 model_busy");
    expect(get(rig, kResidency).json["state"] == "resident", "and the model stays resident");
    reply = post(rig, kSuspend, R"({"auto_resume":false})");
    expect(reply.status == 409, "with a body too");
    rig.service.admitted = 0;
    expect(post(rig, kSuspend).status == 200, "idle again, it suspends");
    rig.service.admitted = 1;
    reply                = post(rig, kSuspend, R"({"auto_resume":false})");
    expect(reply.status == 409 && get(rig, kResidency).json["auto_resume"] == true,
           "a suspended model refuses a busy repeat, leaving auto_resume as it was");
}

void suspend_disabled() {
    Rig rig(/*suspend_enabled=*/false);
    expect(get(rig, kResidency).json["enabled"] == false, "residency reports suspend off");
    Reply reply = post(rig, kSuspend);
    expect(reply.status == 400 && reply.error_code() == "model_suspend_disabled",
           "suspend without --model-suspend is 400 model_suspend_disabled");
    reply = post(rig, kResume);
    expect(reply.status == 400 && reply.error_code() == "model_suspend_disabled",
           "and so is resume");
}

void service_failures_map_to_statuses() {
    Rig rig;
    rig.service.suspend_unavailable = true;
    Reply reply                     = post(rig, kSuspend);
    expect(reply.status == 503 && reply.error_code() == "model_unavailable",
           "an unavailable Engine is 503 model_unavailable");
    rig.service.suspend_unavailable = false;
    rig.service.suspend_crashes     = true;
    reply                           = post(rig, kSuspend);
    expect(reply.status == 500 && reply.error_code() == "model_residency_error",
           "any other suspend failure is 500 model_residency_error");
    expect(reply.json["error"]["message"].get<std::string>().find("device copy failed") !=
               std::string::npos,
           "carrying the reason");
    rig.service.suspend_crashes = false;
    expect(get(rig, kResidency).json["state"] == "resident", "and the model is still resident");
}

void failed_resume_reports_last_error() {
    Rig rig;
    post(rig, kSuspend);
    rig.service.resume_fails = true;
    Reply reply              = post(rig, kResume);
    expect(reply.status == 500 && reply.error_code() == "model_residency_error",
           "a failed resume is 500 model_residency_error");
    reply = get(rig, kResidency);
    expect(reply.json["state"] == "suspended" &&
               reply.json["last_error"] == "could not restore device state",
           "the model stays suspended and residency says why");
    rig.service.resume_fails = false;
    reply                    = post(rig, kResume);
    expect(reply.status == 200 && reply.json["state"] == "resident" &&
               reply.json["last_error"].is_null(),
           "the next resume succeeds and clears last_error");
}

void held_model_refuses_requests() {
    Rig rig;
    const std::string chat =
        std::string(R"({"model":")") + kModel + R"(","messages":[{"role":"user","content":"hi"}]})";
    post(rig, kSuspend, R"({"auto_resume":false})");
    const Reply reply = post(rig, "/v1/chat/completions", chat);
    expect(rig.service.prepare_calls == 1, "the request reached the service");
    expect(reply.status == 503 && reply.error_code() == "service_unavailable",
           "a request to a model held suspended is 503 service_unavailable");
    post(rig, kResume);
    expect(post(rig, "/v1/chat/completions", chat).status != 503,
           "once resumed, the request is no longer refused as unavailable");
}

void props_report_sleeping() {
    Rig rig;
    expect(get(rig, "/props").json["is_sleeping"] == false, "/props: awake when resident");
    post(rig, kSuspend);
    expect(get(rig, "/props").json["is_sleeping"] == true, "/props: sleeping when suspended");
    post(rig, kResume);
    expect(get(rig, "/props").json["is_sleeping"] == false, "/props: awake again after resume");
    expect(get(rig, std::string("/v1/models/") + kModel).status == 200,
           "the model route still answers");
}

void idle_sleeper() {
    // --sleep-idle-seconds 1: the sleeper polls every 250 ms and suspends after a quiet second.
    FakeService service(make_options(/*sleep_idle_seconds=*/1));
    service.admitted = 1;
    std::this_thread::sleep_for(2500ms);
    expect(service.suspend_calls == 0, "the sleeper leaves a model alone while a request is admitted");

    service.admitted = 0;
    std::this_thread::sleep_for(500ms);
    expect(service.suspend_calls == 0, "and waits out a full quiet period after the last request");
    expect(eventually([&] { return service.suspend_calls >= 1; }, 5s),
           "then suspends the idle model");
    expect(service.residency().state == ninfer::ModelResidency::Suspended, "which is suspended");
    std::this_thread::sleep_for(2500ms);
    expect(service.suspend_calls == 1, "a model already suspended is not suspended again");
}

void idle_sleeper_retries_after_a_refusal() {
    FakeService service(make_options(/*sleep_idle_seconds=*/1));
    service.suspend_refusals = 1; // a request slipped in between the sample and the suspend
    expect(eventually([&] { return service.suspend_calls >= 1; }, 5s), "the sleeper tries");
    expect(service.residency().state == ninfer::ModelResidency::Resident,
           "a refused suspend leaves the model resident");
    expect(eventually([&] { return service.suspend_calls >= 2; }, 5s),
           "and it tries again after the next quiet period");
    expect(eventually([&] {
        return service.residency().state == ninfer::ModelResidency::Suspended;
    }),
           "which suspends it");

    FakeService crashing(make_options(/*sleep_idle_seconds=*/1));
    crashing.suspend_crashes = true;
    expect(eventually([&] { return crashing.suspend_calls >= 2; }, 8s),
           "a suspend that throws does not stop the sleeper");
}

void sleeper_is_off_by_default() {
    FakeService service(make_options(/*sleep_idle_seconds=*/0));
    std::this_thread::sleep_for(1500ms);
    expect(service.suspend_calls == 0, "without --sleep-idle-seconds nothing suspends the model");
}

} // namespace

int main() {
    try {
        residency_shape();
        unknown_model_ids();
        suspend_body_validation();
        suspend_and_resume();
        resume_body_validation();
        busy_server_refuses_suspend();
        suspend_disabled();
        service_failures_map_to_statuses();
        failed_resume_reports_last_error();
        held_model_refuses_requests();
        props_report_sleeping();
        idle_sleeper();
        idle_sleeper_retries_after_a_refusal();
        sleeper_is_off_by_default();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "http residency routes: ok\n";
    return 0;
}
