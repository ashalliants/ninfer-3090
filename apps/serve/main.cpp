#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "product/version/version.h"
#include "serve/generation_service.h"
#include "serve/http_security.h"
#include "serve/http_server.h"
#include "serve/model_catalog.h"
#include "serve/model_registry.h"
#include "serve/routed_model.h"
#include "serve/router_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};
std::atomic<ninfer::serve::RouterServer*> g_router{nullptr};
std::atomic<bool> g_router_stop_requested{false};

// RouterServer::stop() takes the registry's mutex, which a signal handler must not do (the signal
// can land on a thread that holds it). The handler only sets a flag; a watcher thread stops it.
void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
    g_router_stop_requested.store(true);
}

// Time for in-flight error responses to flush once the Engine has latched, before the exit.
constexpr std::chrono::seconds kEngineFailureExitGrace{5};
constexpr int kEngineFailureExitStatus = 3;

// Called on the Engine worker thread after every host-side worker failure. A recovered failure is
// only logged. A latch leaves the process alive holding VRAM and answering 503 forever, so unless
// disabled it is logged FATAL and the process exits non-zero for a supervisor to restart; /health
// is already 503 during the grace period.
void handle_engine_fault(const ninfer::serve::OperationalLog& log,
                         const std::shared_ptr<spdlog::logger>& logger, bool exit_on_failure,
                         const ninfer::EngineFaultEvent& event) {
    // The exit is scheduled before anything that can throw: rendering or writing the event must
    // not be able to keep a latched process alive.
    static std::atomic<bool> exit_scheduled{false};
    const bool first_latch = event.latched && !exit_scheduled.exchange(true);
    if (first_latch && exit_on_failure) {
        try {
            std::thread([logger] {
                std::this_thread::sleep_for(kEngineFailureExitGrace);
                try {
                    logger->flush();
                } catch (...) {}
                std::_Exit(kEngineFailureExitStatus);
            }).detach();
        } catch (...) {
            // The thread could not be started. The exit is mandatory, so take it now without the
            // grace period rather than staying up latched.
            std::_Exit(kEngineFailureExitStatus);
        }
    }
    try {
        log.engine_fault(event);
        if (first_latch) {
            log.engine_fatal_exit(std::chrono::duration<double>(kEngineFailureExitGrace).count(),
                                  exit_on_failure);
        }
        logger->flush();
    } catch (...) {}
}


// Router mode (--models-dir / --models-preset): several models behind one address. Each model is an
// Engine with an HTTP server of its own on loopback; the router front-end forwards to it once the
// registry has loaded or woken the model. Adapted from NInfer-all 0dcb1da9 (Apache-2.0).
int run_router(const char* argv0, const ninfer::serve::ServeOptions& options,
               const std::shared_ptr<spdlog::logger>& logger,
               ninfer::product::StartupLogRenderer& startup_log,
               const ninfer::serve::OperationalLog& operational_log) {
    try {
        // One model's service: its Engine loaded and warmed up. A fault is logged, never fatal to
        // the process: the registry unloads a model whose Engine failed and the next request loads
        // it again.
        // The startup renderer keeps one set of phase and progress state, and the registry builds
        // models outside its own lock: loads take turns.
        std::mutex load_mutex;
        const auto make_service = [&](const ninfer::serve::ServeOptions& model_options) {
            std::lock_guard load_turn(load_mutex);
            auto service = std::make_unique<ninfer::serve::GenerationService>(
                model_options, startup_log.observer(),
                [operational_log](const ninfer::ContextStoreWriteEvent& event) {
                    operational_log.context_store_write(event);
                },
                [operational_log, logger](const ninfer::EngineFaultEvent& event) {
                    try {
                        operational_log.engine_fault(event);
                        logger->flush();
                    } catch (...) {}
                },
                [operational_log](const ninfer::ContextStoreHydrationEvent& event) {
                    operational_log.context_store_hydration_failure(event);
                });
            startup_log.engine_ready(service->load_summary());
            operational_log.engine_capacity(*service);
            using Clock                            = std::chrono::steady_clock;
            const Clock::time_point warmup_started = Clock::now();
            operational_log.warmup_started();
            try {
                service->warmup();
            } catch (const std::exception& exception) {
                operational_log.warmup_failure(
                    std::chrono::duration<double>(Clock::now() - warmup_started).count(),
                    exception.what());
                throw;
            }
            operational_log.warmup_complete(
                std::chrono::duration<double>(Clock::now() - warmup_started).count());
            return service;
        };

        std::optional<ninfer::serve::PresetFile> preset;
        if (options.models_preset) {
            std::ifstream file(*options.models_preset);
            if (!file) {
                operational_log.server_failure(false, "cannot read --models-preset " +
                                                          options.models_preset->string());
                return 1;
            }
            std::stringstream text;
            text << file.rdbuf();
            preset = ninfer::serve::parse_preset_file(text.str());
        }
        std::vector<ninfer::serve::ModelDefinition> definitions;
        for (ninfer::serve::CatalogModel& model : ninfer::serve::build_catalog(
                 options.models_dir, preset, options.model_arguments)) {
            std::vector<std::string> words{argv0, model.artifact.string()};
            words.insert(words.end(), model.arguments.begin(), model.arguments.end());
            std::vector<char*> model_argv;
            for (std::string& word : words) { model_argv.push_back(word.data()); }
            ninfer::serve::ServeOptions model_options = ninfer::serve::parse_serve_options(
                static_cast<int>(model_argv.size()), model_argv.data());
            // The multi-device pipeline is parked: a router model runs on one GPU, as a single server
            // does, and a preset's --devices / --stage-layers must not slip past that at startup.
            if (model_options.devices.size() > 1 || !model_options.stage_layers.empty()) {
                throw std::invalid_argument("model '" + model.id +
                                            "': router mode runs each model on one GPU "
                                            "(--devices takes one id, --stage-layers is unsupported)");
            }
            // The router owns the address and the key; a model answers to the router alone.
            model_options.host              = options.host;
            model_options.port              = options.port;
            model_options.api_key           = options.api_key;
            model_options.model_id_override = model.id;
            // Every model's Engine keeps its own store: a shared one would be scanned, evicted and
            // rewritten by several Engines that do not know of each other.
            if (!model_options.context_store_path.empty()) {
                // Local volumes may ignore case, so capitals are escaped there too.
                model_options.context_store_path /= ninfer::serve::store_directory_name(model.id, true);
            }
            if (!model_options.context_store_s3_bucket.empty()) {
                model_options.context_store_s3_prefix +=
                    ninfer::serve::store_directory_name(model.id, false) + "/";
            }
            definitions.push_back(
                ninfer::serve::ModelDefinition{.id              = model.id,
                                               .aliases         = model.aliases,
                                               .options         = std::move(model_options),
                                               .arguments       = model.arguments,
                                               .load_on_startup = model.load_on_startup});
        }
        // Files and stores the models would share by accident are refused before any Engine starts.
        ninfer::serve::check_model_state_paths(definitions);
        const auto ports = std::make_shared<ninfer::serve::RoutedPorts>();
        // Stored Responses outlive an unload: the store holds no Engine state, so a reloaded model
        // still answers previous_response_id and response lookups.
        std::map<std::string, std::shared_ptr<ninfer::serve::OpenAIResponsesStore>> response_stores;
        for (const auto& definition : definitions) {
            response_stores[definition.id] = std::make_shared<ninfer::serve::OpenAIResponsesStore>(
                definition.options.response_store_max_records,
                definition.options.response_store_max_bytes);
        }
        ninfer::serve::ModelRegistry registry(
            std::move(definitions),
            ninfer::serve::RegistryPolicy{
                .models_max  = options.models_max,
                .autoload    = options.models_autoload,
                .sleep_idle  = std::chrono::seconds(options.sleep_idle_seconds),
                .unload_idle = std::chrono::seconds(options.unload_idle_seconds)},
            [&](const ninfer::serve::ModelDefinition& definition)
                -> std::unique_ptr<ninfer::serve::ModelService> {
                try {
                    return std::make_unique<ninfer::serve::RoutedModelService>(
                        definition.id, make_service(definition.options), logger, ports,
                        response_stores.at(definition.id));
                } catch (const std::exception& exception) {
                    logger->error("model '{}' failed to load: {}", definition.id,
                                  exception.what());
                    throw;
                }
            },
            /*router=*/true);
        ninfer::serve::RouterServer router(options, registry, ports, logger);
        if (!router.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }
        g_router.store(&router);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef SIGBREAK
        std::signal(SIGBREAK, handle_signal);
#endif
        // The router answers from here on; a request for a model that is still loading waits for it.
        bool listened = true;
        std::thread listener([&] { listened = router.listen(); });
        std::atomic<bool> router_done{false};
        std::thread signal_watcher([&] {
            while (!router_done.load() && !g_router_stop_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (g_router_stop_requested.load()) { router.stop(); }
        });
        const auto join_watcher = [&] {
            router_done.store(true);
            signal_watcher.join();
        };
        try {
            registry.load_startup_models();
        } catch (const std::exception& exception) {
            operational_log.server_failure(false, exception.what());
            router.stop();
            listener.join();
            join_watcher();
            g_router.store(nullptr);
            return 1;
        }
        operational_log.server_ready(options.host, options.port, "router",
                                     !options.api_key.empty());
        if (ninfer::serve::exposed_without_key(options.host, !options.api_key.empty())) {
            operational_log.exposure_warning(options.host, options.port);
        }
        listener.join();
        join_watcher();
        g_router.store(nullptr);
        if (!listened) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_router.store(nullptr);
        operational_log.server_failure(false, exception.what());
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "ninfer-serve " << ninfer::product::build_version() << '\n';
        return 0;
    }
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    operational_log.setting_sources(options.setting_sources);
    bool serving = false;
    if (options.router()) {
        return run_router(argv[0], options, logger, startup_log, operational_log);
    }

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        // Answer 503 from here on rather than leaving the accepted connection silent. The socket
        // has been listenable since bind() either way; the difference is whether a caller arriving
        // during the ten seconds of weight loading gets a documented "still loading" or a hang.
        server.start_serving_during_startup();

        ninfer::serve::GenerationService service(
            options, startup_log.observer(),
            [&operational_log](const ninfer::ContextStoreWriteEvent& e) {
                operational_log.context_store_write(e);
            },
            [operational_log, logger, exit_on_failure = options.exit_on_engine_failure](
                const ninfer::EngineFaultEvent& event) {
                handle_engine_fault(operational_log, logger, exit_on_failure, event);
            },
            [&operational_log](const ninfer::ContextStoreHydrationEvent& e) {
                operational_log.context_store_hydration_failure(e);
            });
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#ifdef SIGBREAK
        // Windows has no way to deliver SIGTERM to another process: TerminateProcess kills it
        // outright and the shutdown path -- which flushes the final partial throughput interval --
        // never runs. GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT) is the one graceful stop a parent
        // can request, and the CRT raises it as SIGBREAK.
        std::signal(SIGBREAK, handle_signal);
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());
        if (ninfer::serve::exposed_without_key(options.host, !options.api_key.empty())) {
            operational_log.exposure_warning(options.host, options.port);
        }

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
