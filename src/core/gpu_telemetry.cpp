#include "core/gpu_telemetry.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer {

struct GpuTelemetry::Impl {
    struct Utilization { // nvmlUtilization_t
        unsigned gpu, memory;
    };
    struct Memory { // nvmlMemory_t (v1)
        unsigned long long total, free, used;
    };
    static constexpr unsigned kTemperatureGpuSensor = 0; // NVML_TEMPERATURE_GPU

    explicit Impl(const std::string& pci_bus_id) {
#ifdef _WIN32
        library = ::LoadLibraryA("nvml.dll");
        if (library == nullptr) { return; }
        bind(pci_bus_id, [&](const char* name) { return reinterpret_cast<void*>(::GetProcAddress(library, name)); });
#else
        library = ::dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (library == nullptr) { return; }
        bind(pci_bus_id, [&](const char* name) { return ::dlsym(library, name); });
#endif
    }

    Impl(const std::string& pci_bus_id, const GpuSymbolLookup& lookup) { bind(pci_bus_id, lookup); }

    // Resolves the NVML entry points through `symbol` and opens the device; on any failure it
    // leaves the object unavailable.
    void bind(const std::string& pci_bus_id, const GpuSymbolLookup& symbol) {
        init        = reinterpret_cast<int (*)()>(symbol("nvmlInit_v2"));
        shutdown    = reinterpret_cast<int (*)()>(symbol("nvmlShutdown"));
        by_pci      = reinterpret_cast<int (*)(const char*, void**)>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        utilization = reinterpret_cast<int (*)(void*, Utilization*)>(symbol("nvmlDeviceGetUtilizationRates"));
        memory      = reinterpret_cast<int (*)(void*, Memory*)>(symbol("nvmlDeviceGetMemoryInfo"));
        temperature = reinterpret_cast<int (*)(void*, unsigned, unsigned*)>(symbol("nvmlDeviceGetTemperature"));
        power       = reinterpret_cast<int (*)(void*, unsigned*)>(symbol("nvmlDeviceGetPowerUsage"));
        energy      = reinterpret_cast<int (*)(void*, unsigned long long*)>(symbol("nvmlDeviceGetTotalEnergyConsumption"));
        if (init == nullptr || shutdown == nullptr || by_pci == nullptr || init() != 0) {
            release();
            return;
        }
        initialized = true;
        if (by_pci(pci_bus_id.c_str(), &device) != 0) { release(); }
    }

    ~Impl() { release(); }

    void release() noexcept {
        if (initialized) { (void)shutdown(); }
        initialized = false;
        device      = nullptr;
        if (library == nullptr) { return; }
#ifdef _WIN32
        ::FreeLibrary(library);
#else
        ::dlclose(library);
#endif
        library = nullptr;
    }

#ifdef _WIN32
    HMODULE library = nullptr;
#else
    void* library = nullptr;
#endif
    int (*init)()                                    = nullptr;
    int (*shutdown)()                                = nullptr;
    int (*by_pci)(const char*, void**)               = nullptr;
    int (*utilization)(void*, Utilization*)          = nullptr;
    int (*memory)(void*, Memory*)                    = nullptr;
    int (*temperature)(void*, unsigned, unsigned*)   = nullptr;
    int (*power)(void*, unsigned*)                   = nullptr;
    int (*energy)(void*, unsigned long long*)        = nullptr; // millijoules
    void* device                                     = nullptr;
    bool initialized                                 = false;
};

GpuTelemetry::GpuTelemetry(const std::string& pci_bus_id) : impl_(std::make_unique<Impl>(pci_bus_id)) {}

GpuTelemetry::GpuTelemetry(const std::string& pci_bus_id, const GpuSymbolLookup& lookup)
    : impl_(std::make_unique<Impl>(pci_bus_id, lookup)) {}

GpuTelemetry::~GpuTelemetry() = default;

bool GpuTelemetry::available() const noexcept { return impl_->device != nullptr; }

GpuSample GpuTelemetry::sample() const {
    GpuSample sample;
    if (!available()) { return sample; }
    void* const device = impl_->device;
    if (impl_->utilization != nullptr) {
        Impl::Utilization rates{};
        if (impl_->utilization(device, &rates) == 0) {
            sample.utilization_percent        = rates.gpu;
            sample.memory_utilization_percent = rates.memory;
        }
    }
    if (impl_->memory != nullptr) {
        Impl::Memory memory{};
        if (impl_->memory(device, &memory) == 0) {
            sample.memory_used_bytes  = memory.used;
            sample.memory_total_bytes = memory.total;
        }
    }
    if (impl_->temperature != nullptr) {
        unsigned celsius = 0;
        if (impl_->temperature(device, Impl::kTemperatureGpuSensor, &celsius) == 0) {
            sample.temperature_celsius = celsius;
        }
    }
    if (impl_->power != nullptr) {
        unsigned milliwatts = 0;
        if (impl_->power(device, &milliwatts) == 0) { sample.power_watts = milliwatts / 1000.0; }
    }
    if (impl_->energy != nullptr) {
        unsigned long long millijoules = 0;
        if (impl_->energy(device, &millijoules) == 0) { sample.energy_joules = static_cast<double>(millijoules) / 1000.0; }
    }
    return sample;
}

} // namespace ninfer
