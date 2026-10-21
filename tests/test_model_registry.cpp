// The router's model lifecycle and catalog, without an Engine: a fake service records what the
// registry asks of it. Covered: resolving ids and aliases, autoload, making room under models_max
// by putting the least recently used idle model to sleep (or unloading it when it cannot sleep),
// waking a sleeping model, leases holding a model in place, idle sleep, explicit load/unload/sleep,
// the event stream, and the preset/models-directory catalog with its option precedence.

#include "serve/model_catalog.h"
#include "serve/model_registry.h"
#include "serve/router_server.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer::serve;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

struct Journal {
    std::vector<std::string> calls;
    std::map<std::string, int> alive;
    // Read by the registry's idle thread: entries exist before a service is loaded.
    std::map<std::string, std::atomic<bool>> failing;
};

class FakeService final : public ModelService {
public:
    FakeService(std::string id, bool suspendable, Journal& journal)
        : id_(std::move(id)), suspendable_(suspendable), journal_(journal) {
        journal_.calls.push_back("load " + id_);
        ++journal_.alive[id_];
        // A new Engine carries no failure latch; a woken one keeps it.
        if (const auto found = journal_.failing.find(id_); found != journal_.failing.end()) {
            found->second = false;
        }
    }

    ~FakeService() override {
        journal_.calls.push_back("unload " + id_);
        --journal_.alive[id_];
    }

    GenerationService& service() override { throw std::logic_error("fake service"); }

    bool suspendable() const override { return suspendable_; }

    void suspend() override { journal_.calls.push_back("sleep " + id_); }

    void resume() override { journal_.calls.push_back("wake " + id_); }

    bool failed() const override {
        const auto found = journal_.failing.find(id_);
        return found != journal_.failing.end() && found->second.load();
    }

private:
    std::string id_;
    bool suspendable_;
    Journal& journal_;
};

ModelDefinition model(std::string id, std::vector<std::string> aliases = {}) {
    ModelDefinition out;
    out.id      = std::move(id);
    out.aliases = std::move(aliases);
    return out;
}

ModelServiceFactory factory(Journal& journal, std::vector<std::string> non_suspendable = {}) {
    return [&journal, non_suspendable](const ModelDefinition& definition) {
        const bool can_sleep = std::find(non_suspendable.begin(), non_suspendable.end(),
                                         definition.id) == non_suspendable.end();
        return std::make_unique<FakeService>(definition.id, can_sleep, journal);
    };
}

ModelState state_of(const ModelRegistry& registry, const std::string& id) {
    return registry.status(id)->state;
}

void lifecycle_under_one_slot() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 1;
    ModelRegistry registry({model("a", {"alpha"}), model("b"), model("c")}, policy,
                           factory(journal, {"c"}), true);
    expect(state_of(registry, "a") == ModelState::Unloaded, "nothing loads before it is asked");
    {
        auto lease = registry.acquire("alpha");
        expect(lease.model() == "a", "an alias resolves to its model");
        expect(state_of(registry, "a") == ModelState::Loaded, "acquire loads");
    }
    {
        auto lease = registry.acquire("b");
        expect(state_of(registry, "a") == ModelState::Sleeping, "the idle model sleeps for room");
        expect(state_of(registry, "b") == ModelState::Loaded, "the new model loads");
    }
    {
        auto lease = registry.acquire("a");
        expect(state_of(registry, "a") == ModelState::Loaded, "a sleeping model wakes");
        expect(state_of(registry, "b") == ModelState::Sleeping, "the other one sleeps");
    }
    {
        auto lease = registry.acquire("c");
        expect(state_of(registry, "a") == ModelState::Sleeping, "room again by sleep");
    }
    {
        auto lease = registry.acquire("b");
        expect(state_of(registry, "c") == ModelState::Unloaded,
               "a model that cannot sleep is unloaded for room");
        expect(journal.alive["c"] == 0, "its service is gone");
    }
    const std::vector<std::string> expected{"load a",  "sleep a", "load b",   "sleep b", "wake a",
                                            "sleep a", "load c",  "unload c", "wake b"};
    expect(journal.calls == expected, "the registry asked for exactly these transitions");
    bool unknown = false;
    try {
        (void)registry.acquire("nope");
    } catch (const ModelUnknown&) { unknown = true; }
    expect(unknown, "an unknown model is refused");
    bool not_loaded = false;
    try {
        (void)registry.acquire("c", false);
    } catch (const ModelNotLoaded&) { not_loaded = true; }
    expect(not_loaded, "autoload off refuses a model that is not loaded");
}

void leases_hold_models() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max   = 1;
    policy.room_timeout = std::chrono::seconds(5);
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    auto held = std::make_unique<ModelRegistry::Lease>(registry.acquire("a"));
    std::atomic<bool> loaded_b{false};
    std::thread other([&] {
        auto lease = registry.acquire("b");
        loaded_b   = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    expect(!loaded_b, "a leased model is not displaced");
    held.reset();
    other.join();
    expect(loaded_b && state_of(registry, "a") == ModelState::Sleeping,
           "the waiting request proceeds once the lease ends");
}

void idle_sleep_and_events() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 2;
    policy.sleep_idle = std::chrono::seconds(1);
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    std::vector<ModelEvent> events;
    std::uint64_t cursor = registry.events(0, std::chrono::milliseconds(0), events);
    { auto lease = registry.acquire("a"); }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Sleeping &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    expect(state_of(registry, "a") == ModelState::Sleeping, "an idle model goes to sleep");
    cursor = registry.events(cursor, std::chrono::milliseconds(0), events);
    std::vector<std::string> states;
    for (const ModelEvent& event : events) { states.push_back(model_state_name(event.state)); }
    const std::vector<std::string> expected{"loading", "loaded", "unloading", "sleeping"};
    expect(states == expected, "the event stream reports every transition in order");
    registry.unload("a");
    expect(state_of(registry, "a") == ModelState::Unloaded && journal.alive["a"] == 0,
           "unload destroys a sleeping model");
    registry.load("b");
    registry.sleep("b");
    expect(state_of(registry, "b") == ModelState::Sleeping, "an explicit sleep");
    registry.load("b");
    expect(state_of(registry, "b") == ModelState::Loaded, "an explicit load wakes it");
}

void observing_is_not_use() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max = 1;
    policy.sleep_idle = std::chrono::seconds(1);
    ModelRegistry registry({model("a")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    // A monitor polling the model (GET /props, /metrics) must not keep it awake.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Sleeping &&
           std::chrono::steady_clock::now() < deadline) {
        { auto observed = registry.observe("a"); }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    expect(state_of(registry, "a") == ModelState::Sleeping,
           "an observed but otherwise idle model goes to sleep");
}

void failed_model_reloads() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max    = 1;
    journal.failing["a"] = false;
    ModelRegistry registry({model("a")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    journal.failing["a"] = true;
    const auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state_of(registry, "a") != ModelState::Unloaded &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const auto status = registry.status("a");
    expect(status->state == ModelState::Unloaded && status->failed && journal.alive["a"] == 0,
           "a router unloads a model whose Engine failed and reports the failure");
    journal.failing["a"] = false;
    { auto lease = registry.acquire("a"); }
    expect(state_of(registry, "a") == ModelState::Loaded && !registry.status("a")->failed,
           "the next request loads the failed model again");
}

void acquire_does_not_route_to_a_failed_engine() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max    = 1;
    journal.failing["a"] = false;
    ModelRegistry registry({model("a")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    journal.failing["a"] = true;
    // At once, without waiting for the idle thread's pass.
    { auto lease = registry.acquire("a"); }
    const auto loads = std::count(journal.calls.begin(), journal.calls.end(), "load a");
    expect(loads >= 2,"a request that finds a failed Engine gets a fresh one");
    journal.failing["a"] = false;
}

void a_hold_outlives_the_first_load() {
    Journal journal;
    ModelRegistry registry({model("a")}, RegistryPolicy{}, factory(journal), true);
    registry.sleep("a", true, false);
    bool refused = false;
    try {
        (void)registry.acquire("a");
    } catch (const ModelNotLoaded&) { refused = true; }
    expect(refused && journal.alive["a"] == 0, "a held model that never loaded does not autoload");
    registry.load("a");
    { auto lease = registry.acquire("a"); }
    expect(state_of(registry, "a") == ModelState::Loaded, "an explicit load clears the hold");
}

void observing_with_nothing_loaded() {
    Journal journal;
    ModelRegistry registry({model("a"), model("b")}, RegistryPolicy{}, factory(journal), true);
    expect(!registry.observe("").has_value(), "no selector and nothing loaded observes nothing");
    bool unknown = false;
    try {
        (void)registry.observe("nope");
    } catch (const ModelUnknown&) { unknown = true; }
    expect(unknown, "an unknown explicit id is still refused");
}

void store_names_never_collide() {
    const std::vector<std::string> ids{"a/b", "a_b", "a%2Fb", "a.b", "a b", "A", "a", "..", "%",
                                       "%25", "-", "_", "a:b", "a\\b", "\xC3\xA9", "model-1.5"};
    for (const bool capitals : {true, false}) {
        std::set<std::string> seen;
        for (const std::string& id : ids) {
            const std::string name = store_directory_name(id, capitals);
            expect(name.find_first_of("/\\.: ") == std::string::npos, "a store name is one safe segment");
            expect(seen.insert(name).second, "distinct ids give distinct store names: " + id);
        }
    }
    // Case variants stay distinct on the local form (volumes may ignore case) and, being plain, on
    // the S3 form.
    expect(store_directory_name("A", true) != store_directory_name("a", true), "local case variants");
    expect(store_directory_name("A", false) != store_directory_name("a", false), "S3 case variants");
    expect(store_directory_name("Qwen", true) == "%51wen", "capitals are escaped locally");
    expect(store_directory_name("Qwen", false) == "Qwen", "and kept as they are for S3");
}

ModelDefinition with_artifact(std::string id, std::string artifact) {
    ModelDefinition out = model(std::move(id));
    out.options.artifact_path = std::move(artifact);
    return out;
}

void shared_state_files_are_refused() {
    const auto refused = [](const std::vector<ModelDefinition>& models) {
        try {
            check_model_state_paths(models);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    expect(!refused({with_artifact("a", "/m/a.ninfer"), with_artifact("b", "/m/b.ninfer")}),
           "distinct artifacts have distinct default state files");
    expect(refused({with_artifact("a", "/m/a.ninfer"), with_artifact("b", "/m/a.ninfer")}),
           "two models on one artifact would share its default state files");
    auto a = with_artifact("a", "/m/a.ninfer");
    auto b = with_artifact("b", "/m/b.ninfer");
    b.options.flash_next.expert_state_path = "/m/a.ninfer.expert-state";
    expect(refused({a, b}), "an explicit path equal to another model's default is refused");
    b.options.flash_next.expert_state_path.clear();
    b.options.flash_next.ngram_volume_path = "/m/shared.ngram";
    a.options.flash_next.ngram_volume_path = "/m/shared.ngram";
    expect(refused({a, b}), "one explicit n-gram volume for two models is refused");
    a.options.flash_next.ngram_volume_path.clear();
    b.options.flash_next.ngram_volume_path.clear();
    a.options.request_log_jsonl = "/m/b.ninfer";
    expect(refused({a, b}), "the request log may not land on a model's artifact");
    a.options.request_log_jsonl = "/m/requests.jsonl";
    expect(!refused({a, b}), "a shared request log of its own is fine");
    b.options.request_log_jsonl = "/m/a.ninfer";
    expect(refused({a, b}), "a later model's own request log may not land on another model's artifact");
    b.options.request_log_jsonl = "/m/b-requests.jsonl";
    expect(!refused({a, b}), "each model may have a request log of its own");
    // Same artifact basename in two folders, one shared NVMe directory: one scratch file.
    auto c = with_artifact("c", "/m/one/model.ninfer");
    auto d = with_artifact("d", "/m/two/model.ninfer");
    expect(!refused({c, d}), "same basenames in different folders are fine without an NVMe tier");
    for (auto* m : {&c, &d}) {
        m->options.context_cache.enabled          = true;
        m->options.flash_next.context_nvme_bytes  = 1ULL << 30;
        m->options.flash_next.context_nvme_dir    = "/scratch";
    }
    expect(refused({c, d}), "a shared NVMe directory would give same-named artifacts one scratch file");
    d.options.flash_next.context_nvme_dir = "/SCRATCH";
#if defined(_WIN32) || defined(__APPLE__)
    expect(refused({c, d}), "directories differing only in case are one on a case-insensitive volume");
#else
    expect(!refused({c, d}), "directories differing only in case are two on a case-sensitive volume");
#endif
    d.options.flash_next.context_nvme_dir = "/scratch2";
    expect(!refused({c, d}), "separate NVMe directories keep the scratch files apart");
}

void empty_selector_survives_unloading() {
    Journal journal;
    ModelRegistry registry({model("a"), model("b")}, RegistryPolicy{}, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    { auto lease = registry.acquire("b"); }
    registry.unload("a");
    registry.unload("b");
    auto lease = registry.acquire("");
    expect(lease.model() == "b", "an empty selector reaches the model that last served, though unloaded");
}

void empty_selector_follows_requests_not_loads() {
    Journal journal;
    ModelRegistry registry({model("a"), model("b")}, RegistryPolicy{}, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    registry.load("b"); // an administrative load is not a request
    auto lease = registry.acquire("");
    expect(lease.model() == "a", "an empty selector goes to the model that last served a request");
}

void a_resumed_failed_service_is_rebuilt() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max    = 1;
    journal.failing["a"] = false;
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    { auto lease = registry.acquire("a"); }
    journal.failing["a"] = true; // the latch survives suspend and resume
    { auto lease = registry.acquire("b"); } // a is idle and failed: it leaves for room
    auto lease = registry.acquire("a");
    expect(lease.model() == "a" && !journal.failing["a"].load(),
           "a request is leased a fresh service, not the failed one that was woken");
    expect(std::find(journal.calls.begin(), journal.calls.end(), "unload a") != journal.calls.end(),
           "the failed service was unloaded and built afresh first");
}

void a_waiting_request_can_be_cancelled() {
    Journal journal;
    RegistryPolicy policy;
    policy.models_max   = 1;
    policy.room_timeout = std::chrono::seconds(30);
    ModelRegistry registry({model("a"), model("b")}, policy, factory(journal), true);
    auto busy = registry.acquire("a"); // its lease leaves no room for b
    std::atomic<bool> gone{false};
    std::thread hang_up([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        gone = true;
    });
    const auto started = std::chrono::steady_clock::now();
    bool cancelled     = false;
    try {
        (void)registry.acquire("b", std::nullopt, [&] { return gone.load(); });
    } catch (const RequestCancelled&) { cancelled = true; }
    hang_up.join();
    expect(cancelled, "a request whose client left stops waiting for room");
    expect(std::chrono::steady_clock::now() - started < std::chrono::seconds(5),
           "it does not wait out the room timeout");
    expect(state_of(registry, "b") == ModelState::Unloaded, "the cancelled request loaded nothing");
    expect(journal.alive["b"] == 0, "and left no service behind");
}

void selector_rewrite_changes_only_the_selector() {
    const auto rewrite = [](const std::string& body) {
        return with_model_selector(body, "big").value_or("<none>");
    };
    expect(rewrite(R"({"model":"alias","n":1})") == R"({"model":"big","n":1})",
           "a named selector is replaced in place");
    expect(rewrite(R"({ "a" : 1.10 , "model" : "x" , "b" : 1e400 })") ==
               R"({ "a" : 1.10 , "model" : "big" , "b" : 1e400 })",
           "numbers keep the text the client wrote");
    expect(rewrite(R"({"a":0.1000000000000000055511151231257827})") ==
               R"({"model":"big","a":0.1000000000000000055511151231257827})",
           "a missing selector is added without re-serialising the rest");
    expect(rewrite("{}") == R"({"model":"big"})", "an empty object gets just the selector");
    expect(rewrite(R"({"model":null})") == R"({"model":"big"})", "a non-string selector is replaced");
    expect(rewrite(R"({"x":{"model":"inner"},"model":"m"})") == R"({"x":{"model":"inner"},"model":"big"})",
           "a nested model member is left alone");
    expect(rewrite(R"({"model":"a","k":[1,{"model":2}]})") == R"({"model":"big","k":[1,{"model":2}]})",
           "an escaped key is matched by its decoded name");
    expect(rewrite(R"({"model":"a","model":"b"})") == R"({"model":"a","model":"big"})",
           "the last duplicate is the one a parser would use");
    expect(rewrite("[1]") == "<none>" && rewrite(R"({"model":"a")") == "<none>",
           "a body that is not an object is not rewritten");
    expect(is_json_content_type("Application/JSON; charset=utf-8") &&
               is_json_content_type("application/vnd.api+json") && !is_json_content_type("text/plain"),
           "the media type is matched without regard to case");
}

void single_model_mode() {
    Journal journal;
    ModelRegistry registry({model("only")}, RegistryPolicy{}, factory(journal), false);
    registry.load_startup_models();
    expect(state_of(registry, "only") == ModelState::Loaded, "single mode loads its model");
    auto lease = registry.acquire("");
    expect(lease.model() == "only", "a request naming no model gets the one model");
}

void catalog() {
    const auto preset = parse_preset_file(R"(
version = 1
; shared by every model
[*]
max-context = 8192
model-suspend = true

[small]
model = /models/small.ninfer
alias = s, tiny
kv-capacity = 4096
load-on-startup = true

[big]
artifact = "/models/big.ninfer"
model-suspend = false
)");
    const auto models = build_catalog(std::nullopt, preset, {"--kv-capacity", "16384"});
    expect(models.size() == 2, "two models from the preset");
    const CatalogModel& small = models[0].id == "small" ? models[0] : models[1];
    const CatalogModel& big   = models[0].id == "small" ? models[1] : models[0];
    expect(small.artifact == "/models/small.ninfer" && small.load_on_startup,
           "path and startup flag");
    expect(small.aliases == std::vector<std::string>({"s", "tiny"}), "aliases");
    const std::vector<std::string> small_arguments{"--max-context", "8192", "--model-suspend",
                                                   "--kv-capacity", "4096", "--kv-capacity",
                                                   "16384"};
    expect(small.arguments == small_arguments,
           "global, then model, then command line, so the command line wins");
    expect(big.artifact == "/models/big.ninfer", "quoted path");
    const std::vector<std::string> big_arguments{"--max-context", "8192", "--kv-capacity",
                                                 "16384"};
    expect(big.arguments == big_arguments, "a model's false boolean undoes the global true");
    bool router_key = false;
    try {
        (void)build_catalog(std::nullopt,
                            parse_preset_file("[a]\nmodel=/x.ninfer\nunload-idle-seconds=5\n"), {});
    } catch (const std::invalid_argument&) { router_key = true; }
    expect(router_key, "a preset may not set a router option per model");
    bool empty_name = false;
    try {
        (void)parse_preset_file("[ ]\nmodel=/x.ninfer\n");
    } catch (const std::invalid_argument&) { empty_name = true; }
    expect(empty_name, "a preset section needs a name");
    bool duplicate = false;
    try {
        (void)build_catalog(
            std::nullopt,
            parse_preset_file("[a]\nmodel=/x.ninfer\nalias=b\n[b]\nmodel=/y.ninfer\n"), {});
    } catch (const std::invalid_argument&) { duplicate = true; }
    expect(duplicate, "an alias may not repeat another model's id");

    const auto directory = std::filesystem::temp_directory_path() / "ninfer_catalog_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory / "nested");
    std::ofstream(directory / "flat.ninfer") << "x";
    std::ofstream(directory / "nested" / "weights.ninfer") << "x";
    std::ofstream(directory / "readme.txt") << "x";
    const auto found = discover_models(directory);
    expect(found.size() == 2 && found[0].id == "flat" && found[1].id == "nested",
           "a models directory yields its artifacts and single-artifact subdirectories");
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    try {
        lifecycle_under_one_slot();
        leases_hold_models();
        idle_sleep_and_events();
        observing_is_not_use();
        failed_model_reloads();
        acquire_does_not_route_to_a_failed_engine();
        a_hold_outlives_the_first_load();
        observing_with_nothing_loaded();
        store_names_never_collide();
        shared_state_files_are_refused();
        a_waiting_request_can_be_cancelled();
        empty_selector_survives_unloading();
        empty_selector_follows_requests_not_loads();
        a_resumed_failed_service_is_rebuilt();
        selector_rewrite_changes_only_the_selector();
        single_model_mode();
        catalog();
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "model registry: ok\n";
    return 0;
}
