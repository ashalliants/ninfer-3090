#include "serve/model_registry.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

using Clock = std::chrono::steady_clock;

// Events kept for GET /models/sse subscribers that fall behind.
constexpr std::size_t kKeptEvents = 256;

} // namespace

const char* model_state_name(ModelState state) noexcept {
    switch (state) {
    case ModelState::Unloaded:
        return "unloaded";
    case ModelState::Loading:
        return "loading";
    case ModelState::Loaded:
        return "loaded";
    case ModelState::Sleeping:
        return "sleeping";
    case ModelState::Unloading:
        return "unloading";
    }
    return "unloaded";
}

void check_model_state_paths(const std::vector<ModelDefinition>& models) {
    struct Entry {
        std::string what;
        std::string owner; // "" for the shared request log
        std::filesystem::path path;
    };
    const auto normal = [](const std::filesystem::path& path) {
        std::error_code error;
        auto out = std::filesystem::weakly_canonical(path, error);
        return error ? std::filesystem::absolute(path, error).lexically_normal() : out;
    };
    // Windows and macOS volumes usually ignore case, so names differing only in case are one file
    // there; on Linux they are two.
    const auto key = [](const std::filesystem::path& path) {
        std::string text = path.string();
#if defined(_WIN32) || defined(__APPLE__)
        for (char& letter : text) {
            if (letter >= 'A' && letter <= 'Z') { letter = static_cast<char>(letter - 'A' + 'a'); }
        }
#endif
        return text;
    };
    std::vector<Entry> entries;
    for (const ModelDefinition& model : models) {
        const std::filesystem::path artifact = normal(model.options.artifact_path);
        const auto& next                     = model.options.flash_next;
        entries.push_back({"its artifact", model.id, artifact});
        entries.push_back({"its n-gram volume", model.id,
                           next.ngram_volume_path.empty()
                               ? std::filesystem::path(artifact.string() + ".ngram")
                               : normal(next.ngram_volume_path)});
        if (next.save_expert_state) {
            entries.push_back({"its expert state", model.id,
                               next.expert_state_path.empty()
                                   ? std::filesystem::path(artifact.string() + ".expert-state")
                                   : normal(next.expert_state_path)});
        }
        if (model.options.context_cache.enabled && next.context_nvme_bytes != 0) {
            // The runtime names the NVMe scratch file for the artifact's basename (and the
            // process) inside --context-nvme-dir, so same-named artifacts in different folders
            // would share one file there. This is the name's stem; the pid suffix is the same
            // for every model of a router.
            const std::filesystem::path dir =
                next.context_nvme_dir.empty() ? artifact.parent_path()
                                              : normal(next.context_nvme_dir);
            entries.push_back({"its NVMe scratch file", model.id,
                               dir / (artifact.filename().string() + ".context")});
        }
    }
    for (const ModelDefinition& model : models) {
        if (!model.options.request_log_jsonl.empty()) {
            // Models may share one log (same path, owner ""), but a preset section can also give
            // a model a log of its own, and that file must not be another model's state either.
            entries.push_back({"the request log", "", normal(model.options.request_log_jsonl)});
        }
    }
    for (std::size_t first = 0; first < entries.size(); ++first) {
        for (std::size_t second = first + 1; second < entries.size(); ++second) {
            const Entry& a = entries[first];
            const Entry& b = entries[second];
            // Only the shared request log may repeat (owner ""): every other file, the artifact
            // included, is one model's own, and a model's state path may not alias its artifact
            // or another of its own state files either.
            if ((a.owner.empty() && b.owner.empty()) || key(a.path) != key(b.path)) { continue; }
            const auto who = [](const Entry& e) {
                return e.what + (e.owner.empty() ? std::string() : " (model '" + e.owner + "')");
            };
            throw std::invalid_argument(who(a) + " and " + who(b) + " are the same file: " +
                                        a.path.string() +
                                        "; give each model its own (an alias shares one model)");
        }
    }
}

ModelRegistry::ModelRegistry(std::vector<ModelDefinition> models, RegistryPolicy policy,
                             ModelServiceFactory factory, bool router)
    : policy_(policy), factory_(std::move(factory)), router_(router) {
    if (models.empty()) { throw std::invalid_argument("a server needs at least one model"); }
    if (!router && models.size() != 1) {
        throw std::invalid_argument("single-model mode serves exactly one model");
    }
    for (ModelDefinition& definition : models) {
        models_.push_back(Slot{.definition = std::move(definition)});
    }
    // A router also watches for models whose Engine failed, which it unloads.
    if (router_ || policy_.sleep_idle.count() > 0 || policy_.unload_idle.count() > 0) {
        idle_thread_ = std::thread([this] { idle_loop(); });
    }
}

ModelRegistry::~ModelRegistry() {
    stop();
    if (idle_thread_.joinable()) { idle_thread_.join(); }
    std::unique_lock lock(mutex_);
    for (Slot& slot : models_) {
        std::unique_ptr<ModelService> service = std::move(slot.service);
        lock.unlock();
        service.reset();
        lock.lock();
    }
}

void ModelRegistry::stop() noexcept {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    changed_.notify_all();
}

std::optional<std::size_t> ModelRegistry::find_locked(std::string_view model) const {
    if (model.empty()) {
        if (models_.size() == 1) { return 0; }
        // The model that last served a request answers one that names none, whether or not it is
        // resident now (a request for it loads it again); `last_used` also moves on administrative
        // loads and request completions, so it cannot say who served last.
        if (last_served_) { return last_served_; }
        std::optional<std::size_t> best;
        for (std::size_t index = 0; index < models_.size(); ++index) {
            const ModelState state = models_[index].state;
            if (state != ModelState::Loaded && state != ModelState::Sleeping) { continue; }
            if (!best || models_[index].last_used > models_[*best].last_used) { best = index; }
        }
        return best;
    }
    for (std::size_t index = 0; index < models_.size(); ++index) {
        const ModelDefinition& definition = models_[index].definition;
        if (definition.id == model ||
            std::find(definition.aliases.begin(), definition.aliases.end(), model) !=
                definition.aliases.end()) {
            return index;
        }
    }
    return std::nullopt;
}

std::size_t ModelRegistry::require_locked(std::string_view model) const {
    const auto index = find_locked(model);
    if (!index) {
        if (model.empty()) {
            throw ModelUnknown("the request names no model and none is loaded; set \"model\"");
        }
        throw ModelUnknown("model '" + std::string(model) + "' not found");
    }
    return *index;
}

std::optional<std::string> ModelRegistry::resolve(std::string_view model) const {
    std::lock_guard lock(mutex_);
    const auto index = find_locked(model);
    if (!index) { return std::nullopt; }
    return models_[*index].definition.id;
}

std::size_t ModelRegistry::loaded_count_locked() const {
    return static_cast<std::size_t>(
        std::count_if(models_.begin(), models_.end(), [](const Slot& s) {
            // Unloading still owns its device memory until the suspend or teardown finishes.
            return s.state == ModelState::Loaded || s.state == ModelState::Loading ||
                   s.state == ModelState::Unloading;
        }));
}

void ModelRegistry::publish_locked(std::size_t index) {
    const Slot& slot = models_[index];
    events_.push_back(ModelEvent{.model  = slot.definition.id,
                                 .state  = slot.state,
                                 .failed = slot.failed,
                                 .error  = slot.last_error});
    while (events_.size() > kKeptEvents) {
        events_.pop_front();
        ++first_event_;
    }
    changed_.notify_all();
}

void ModelRegistry::put_to_sleep_locked(std::unique_lock<std::mutex>& lock, std::size_t index) {
    Slot& slot                  = models_[index];
    ModelService* const service = slot.service.get();
    slot.state                  = ModelState::Unloading;
    publish_locked(index);
    lock.unlock();
    std::exception_ptr failure;
    try {
        service->suspend();
    } catch (...) { failure = std::current_exception(); }
    lock.lock();
    if (failure) {
        // Still resident: a request may have arrived in between. It stays loaded.
        slot.state = ModelState::Loaded;
        publish_locked(index);
        std::rethrow_exception(failure);
    }
    slot.state = ModelState::Sleeping;
    publish_locked(index);
}

void ModelRegistry::unload_locked(std::unique_lock<std::mutex>& lock, std::size_t index) {
    Slot& slot = models_[index];
    slot.state = ModelState::Unloading;
    publish_locked(index);
    std::unique_ptr<ModelService> service = std::move(slot.service);
    lock.unlock();
    // Destroying the service stops its Engine, which saves what it persists, and frees the model's
    // device and host memory.
    service.reset();
    lock.lock();
    slot.state = ModelState::Unloaded;
    publish_locked(index);
}

bool ModelRegistry::wait_unless_cancelled_locked(std::unique_lock<std::mutex>& lock,
                                                 Clock::time_point deadline,
                                                 const Cancelled& cancelled) {
    if (!cancelled) { return changed_.wait_until(lock, deadline) != std::cv_status::timeout; }
    // Nothing signals a client that has gone, so wake now and then to ask.
    const auto slice = std::min(deadline, Clock::now() + std::chrono::milliseconds(250));
    const bool woken = changed_.wait_until(lock, slice) != std::cv_status::timeout;
    if (cancelled()) { throw RequestCancelled("the request went away while waiting for a model"); }
    return woken || Clock::now() < deadline;
}

bool ModelRegistry::make_room_locked(std::unique_lock<std::mutex>& lock, std::size_t keep,
                                     Clock::time_point deadline, const Cancelled& cancelled) {
    for (;;) {
        if (stopping_) { return false; }
        // Also between victims: a request whose client left must not start another transition.
        if (cancelled && cancelled()) {
            throw RequestCancelled("the request went away while waiting for a model");
        }
        if (policy_.models_max == 0 || loaded_count_locked() < policy_.models_max) { return true; }
        std::optional<std::size_t> victim;
        for (std::size_t index = 0; index < models_.size(); ++index) {
            const Slot& slot = models_[index];
            if (index == keep || slot.state != ModelState::Loaded || slot.leases != 0) { continue; }
            if (!victim || slot.last_used < models_[*victim].last_used) { victim = index; }
        }
        if (victim) {
            if (models_[*victim].service->suspendable()) {
                try {
                    put_to_sleep_locked(lock, *victim);
                } catch (...) {
                    // Its Engine is still settling the request that just let go of it; look again
                    // shortly rather than spin.
                    const auto retry =
                        std::min(deadline, Clock::now() + std::chrono::milliseconds(50));
                    if (!wait_unless_cancelled_locked(lock, retry, cancelled) &&
                        Clock::now() >= deadline) {
                        return false;
                    }
                }
                continue;
            }
            unload_locked(lock, *victim);
            continue;
        }
        if (!wait_unless_cancelled_locked(lock, deadline, cancelled)) { return false; }
    }
}

void ModelRegistry::make_loaded_locked(std::unique_lock<std::mutex>& lock, std::size_t index,
                                       bool honor_hold, const Cancelled& cancelled) {
    const Clock::time_point deadline = Clock::now() + policy_.room_timeout;
    for (;;) {
        Slot& slot = models_[index];
        if (stopping_) { throw ModelLoadFailed("the server is stopping"); }
        if (slot.state == ModelState::Loaded) { return; }
        if (slot.state == ModelState::Loading || slot.state == ModelState::Unloading) {
            if (!wait_unless_cancelled_locked(lock, deadline, cancelled)) {
                throw ModelLoadFailed("model '" + slot.definition.id + "' did not become ready");
            }
            continue;
        }
        if (!make_room_locked(lock, index, deadline, cancelled)) {
            throw ModelLoadFailed("no room to load model '" + slot.definition.id +
                                  "': every loaded model is busy");
        }
        if (slot.state != ModelState::Unloaded && slot.state != ModelState::Sleeping) { continue; }
        // Room-making released the mutex: a hold set meanwhile still applies to a request.
        if (honor_hold && slot.held) {
            throw ModelNotLoaded("model '" + slot.definition.id + "' is suspended");
        }
        const bool waking = slot.state == ModelState::Sleeping;
        slot.state = ModelState::Loading;
        publish_locked(index);
        ModelService* const sleeping     = waking ? slot.service.get() : nullptr;
        const ModelDefinition definition = slot.definition;
        lock.unlock();
        std::unique_ptr<ModelService> created;
        std::exception_ptr failure;
        try {
            if (sleeping != nullptr) {
                sleeping->resume();
            } else {
                created = factory_(definition);
            }
        } catch (...) { failure = std::current_exception(); }
        lock.lock();
        Slot& done = models_[index];
        if (failure) {
            done.state  = waking ? ModelState::Sleeping : ModelState::Unloaded;
            done.failed = true;
            try {
                std::rethrow_exception(failure);
            } catch (const std::exception& error) { done.last_error = error.what(); } catch (...) {
                done.last_error = "unknown error";
            }
            publish_locked(index);
            // The cause (paths, devices) stays in the server log and in last_error, which only an
            // authenticated GET /models shows; the failing request gets just the fact.
            throw ModelLoadFailed("model '" + definition.id +
                                  "' failed to load (see the server log or GET /models)");
        }
        if (created) { done.service = std::move(created); }
        if (done.service) {
            if (const auto effective = done.service->effective_max_context(); effective != 0) {
                done.effective_context = effective;
            }
        }
        done.state  = ModelState::Loaded;
        done.failed = false;
        done.last_error.clear();
        done.last_used = Clock::now();
        publish_locked(index);
        return;
    }
}

ModelRegistry::Lease ModelRegistry::acquire(std::string_view model, std::optional<bool> autoload,
                                            const Cancelled& cancelled) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    Slot& slot              = models_[index];
    if ((slot.state == ModelState::Sleeping || slot.state == ModelState::Unloaded) && slot.held) {
        throw ModelNotLoaded("model '" + slot.definition.id + "' is suspended");
    }
    // A request must not reach an Engine that has latched a failure: unload it once its other
    // requests are gone, then load it afresh. The latch survives suspend and resume, so a service
    // that was woken (or born failed) is checked again before the lease is issued; the second pass
    // is a fresh build, and one that is still failed refuses the request.
    const Clock::time_point deadline = Clock::now() + policy_.room_timeout;
    for (int pass = 0;; ++pass) {
        while (router_ && slot.state == ModelState::Loaded && slot.service->failed()) {
            if (slot.leases == 0) {
                slot.failed     = true;
                slot.last_error = "the model's Engine failed; it loads again on the next request";
                unload_locked(lock, index);
                break;
            }
            if (!wait_unless_cancelled_locked(lock, deadline, cancelled)) {
                throw ModelLoadFailed("model '" + slot.definition.id + "' is still failing");
            }
        }
        if (slot.state != ModelState::Loaded) {
            if (!autoload.value_or(policy_.autoload)) {
                throw ModelNotLoaded("model '" + slot.definition.id + "' is " +
                                     model_state_name(slot.state) + " and autoload is off");
            }
            make_loaded_locked(lock, index, /*honor_hold=*/true, cancelled);
        }
        if (!(router_ && slot.state == ModelState::Loaded && slot.service->failed())) { break; }
        if (pass > 0) {
            throw ModelLoadFailed("model '" + slot.definition.id + "' failed right after loading");
        }
    }
    Slot& ready = models_[index];
    ++ready.leases;
    ready.last_used = Clock::now();
    last_served_    = index;
    return Lease(*this, index, true);
}

void ModelRegistry::release(std::size_t index, bool use) noexcept {
    {
        std::lock_guard lock(mutex_);
        Slot& slot = models_[index];
        if (slot.leases != 0) { --slot.leases; }
        if (use) { slot.last_used = Clock::now(); }
    }
    changed_.notify_all();
}

std::optional<ModelRegistry::Lease> ModelRegistry::observe(std::string_view model) {
    std::lock_guard lock(mutex_);
    // No selector and nothing loaded to default to: nothing to observe (an unknown id still throws).
    if (model.empty() && !find_locked(model)) { return std::nullopt; }
    const std::size_t index = require_locked(model);
    Slot& slot              = models_[index];
    if (slot.state != ModelState::Loaded && slot.state != ModelState::Sleeping) {
        return std::nullopt;
    }
    ++slot.leases;
    return Lease(*this, index, false);
}

void ModelRegistry::load(std::string_view model) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    Slot& slot              = models_[index];
    slot.held               = false;
    // An Engine that latched a failure is rebuilt, as a request would rebuild it, rather than
    // reported as loaded. The latch survives suspend and resume, so a woken service is checked
    // again; the second pass is a fresh build, and one that is still failed is refused.
    for (int pass = 0;; ++pass) {
        if (router_ && slot.state == ModelState::Loaded && slot.service->failed()) {
            if (slot.leases != 0) {
                throw ModelLoadFailed("model '" + slot.definition.id +
                                      "' has failed and still has requests in flight");
            }
            slot.failed     = true;
            slot.last_error = "the model's Engine failed; it loads again on the next request";
            unload_locked(lock, index);
        }
        make_loaded_locked(lock, index, /*honor_hold=*/false);
        if (!(router_ && slot.state == ModelState::Loaded && slot.service->failed())) { break; }
        if (pass > 0) {
            throw ModelLoadFailed("model '" + slot.definition.id + "' failed right after loading");
        }
    }
}

void ModelRegistry::unload(std::string_view model) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    for (;;) {
        Slot& slot = models_[index];
        if (slot.state == ModelState::Unloaded) { return; }
        if (slot.state == ModelState::Loading || slot.state == ModelState::Unloading ||
            slot.leases != 0) {
            changed_.wait(lock);
            continue;
        }
        unload_locked(lock, index);
        return;
    }
}

void ModelRegistry::sleep(std::string_view model, bool hold, bool wait) {
    std::unique_lock lock(mutex_);
    const std::size_t index = require_locked(model);
    for (;;) {
        Slot& slot = models_[index];
        if (slot.state == ModelState::Sleeping || slot.state == ModelState::Unloaded) {
            slot.held = hold;
            publish_locked(index);
            return;
        }
        if (slot.state != ModelState::Loaded || slot.leases != 0) {
            if (!wait) { throw ModelBusy("model '" + slot.definition.id + "' is in use"); }
            changed_.wait(lock);
            continue;
        }
        if (!slot.service->suspendable()) {
            throw ModelNotLoaded("model '" + slot.definition.id +
                                 "' was not started with --model-suspend");
        }
        put_to_sleep_locked(lock, index);
        models_[index].held = hold;
        return;
    }
}

ModelStatusSnapshot ModelRegistry::snapshot_locked(const Slot& slot) const {
    return ModelStatusSnapshot{.id         = slot.definition.id,
                               .aliases    = slot.definition.aliases,
                               .state      = slot.state,
                               .arguments  = slot.definition.arguments,
                               .max_context = slot.effective_context != 0 ? slot.effective_context
                                                         : slot.definition.options.max_context,
                               .vision      = slot.definition.options.enable_vision,
                               .failed     = slot.failed,
                               .last_error = slot.last_error,
                               .leases     = slot.leases,
                               .last_used  = slot.last_used,
                               .held       = slot.held};
}

std::vector<ModelStatusSnapshot> ModelRegistry::status() const {
    std::lock_guard lock(mutex_);
    std::vector<ModelStatusSnapshot> out;
    out.reserve(models_.size());
    for (const Slot& slot : models_) { out.push_back(snapshot_locked(slot)); }
    return out;
}

std::optional<ModelStatusSnapshot> ModelRegistry::status(std::string_view model) const {
    std::lock_guard lock(mutex_);
    const auto index = find_locked(model);
    if (!index) { return std::nullopt; }
    return snapshot_locked(models_[*index]);
}

std::uint64_t ModelRegistry::events(std::uint64_t cursor, std::chrono::milliseconds wait,
                                    std::vector<ModelEvent>& out) {
    std::unique_lock lock(mutex_);
    const auto next = [&] { return first_event_ + events_.size(); };
    if (cursor == 0) { cursor = next(); }
    if (cursor >= next() && !stopping_) {
        changed_.wait_for(lock, wait, [&] { return stopping_ || cursor < next(); });
    }
    if (cursor < first_event_) { cursor = first_event_; }
    for (std::uint64_t position = cursor; position < next(); ++position) {
        out.push_back(events_[static_cast<std::size_t>(position - first_event_)]);
    }
    return next();
}

void ModelRegistry::for_each_service(
    const std::function<void(const std::string&, GenerationService&)>& visit) {
    std::lock_guard lock(mutex_);
    for (Slot& slot : models_) {
        if (slot.service) { visit(slot.definition.id, slot.service->service()); }
    }
}

void ModelRegistry::load_startup_models() {
    std::vector<std::string> startup;
    {
        std::lock_guard lock(mutex_);
        for (const Slot& slot : models_) {
            if (!router_ || slot.definition.load_on_startup) {
                startup.push_back(slot.definition.id);
            }
        }
    }
    for (const std::string& id : startup) { load(id); }
}

void ModelRegistry::idle_loop() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        changed_.wait_for(lock, std::chrono::seconds(1));
        if (stopping_) { break; }
        const Clock::time_point now = Clock::now();
        for (std::size_t index = 0; index < models_.size(); ++index) {
            Slot& slot = models_[index];
            if (slot.leases != 0) { continue; }
            const auto idle = now - slot.last_used;
            try {
                if (router_ && slot.state == ModelState::Loaded && slot.service->failed()) {
                    slot.failed = true;
                    slot.last_error =
                        "the model's Engine failed; it loads again on the next request";
                    unload_locked(lock, index);
                } else if (slot.state == ModelState::Loaded && policy_.sleep_idle.count() > 0 &&
                           idle >= policy_.sleep_idle && slot.service->suspendable()) {
                    put_to_sleep_locked(lock, index);
                } else if ((slot.state == ModelState::Loaded ||
                            slot.state == ModelState::Sleeping) &&
                           !slot.held && policy_.unload_idle.count() > 0 &&
                           idle >= policy_.unload_idle) {
                    // A held sleep lasts until an explicit load(), so it is never auto-unloaded.
                    unload_locked(lock, index);
                }
            } catch (...) {
                // A busy model stays as it is; the next pass looks again.
            }
        }
    }
}

ModelRegistry::Lease::Lease(ModelRegistry& owner, std::size_t index, bool use) noexcept
    : owner_(&owner), index_(index), use_(use) {}

ModelRegistry::Lease::~Lease() {
    if (owner_ != nullptr) { owner_->release(index_, use_); }
}

ModelRegistry::Lease::Lease(Lease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_), use_(other.use_) {}

ModelRegistry::Lease& ModelRegistry::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) {
        if (owner_ != nullptr) { owner_->release(index_, use_); }
        owner_ = std::exchange(other.owner_, nullptr);
        index_ = other.index_;
        use_   = other.use_;
    }
    return *this;
}

GenerationService& ModelRegistry::Lease::service() const {
    if (owner_ == nullptr) { throw std::logic_error("empty model lease"); }
    // The lease keeps the slot loaded, so its service lives as long as the lease does.
    return owner_->models_[index_].service->service();
}

const std::string& ModelRegistry::Lease::model() const {
    if (owner_ == nullptr) { throw std::logic_error("empty model lease"); }
    return owner_->models_[index_].definition.id;
}

} // namespace ninfer::serve
