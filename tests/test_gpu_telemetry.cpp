#include "core/gpu_telemetry.h"

#include <iostream>
#include <map>
#include <string>

using namespace ninfer;

namespace {

// A stand-in NVML. The structs mirror nvmlUtilization_t and nvmlMemory_t (v1).
struct Utilization {
    unsigned gpu, memory;
};
struct Memory {
    unsigned long long total, free, used;
};

int g_shutdown_calls     = 0;
int g_init_result        = 0;
int g_handle_result      = 0;
int g_temperature_result = 0;
std::string g_bus_id;
int g_device_storage = 0;

int fake_init() { return g_init_result; }
int fake_shutdown() {
    ++g_shutdown_calls;
    return 0;
}
int fake_by_pci(const char* bus_id, void** device) {
    g_bus_id = bus_id;
    *device  = &g_device_storage;
    return g_handle_result;
}
int fake_utilization(void* device, Utilization* out) {
    if (device != &g_device_storage) { return 1; }
    *out = {87, 41};
    return 0;
}
int fake_memory(void*, Memory* out) {
    *out = {25769803776ULL, 4294967296ULL, 21474836480ULL};
    return 0;
}
int fake_temperature(void*, unsigned sensor, unsigned* out) {
    if (g_temperature_result != 0) { return g_temperature_result; }
    if (sensor != 0) { return 1; } // NVML_TEMPERATURE_GPU
    *out = 71;
    return 0;
}
int fake_power(void*, unsigned* out) {
    *out = 312500; // milliwatts
    return 0;
}

std::map<std::string, void*> full_table() {
    return {{"nvmlInit_v2", reinterpret_cast<void*>(&fake_init)},
            {"nvmlShutdown", reinterpret_cast<void*>(&fake_shutdown)},
            {"nvmlDeviceGetHandleByPciBusId_v2", reinterpret_cast<void*>(&fake_by_pci)},
            {"nvmlDeviceGetUtilizationRates", reinterpret_cast<void*>(&fake_utilization)},
            {"nvmlDeviceGetMemoryInfo", reinterpret_cast<void*>(&fake_memory)},
            {"nvmlDeviceGetTemperature", reinterpret_cast<void*>(&fake_temperature)},
            {"nvmlDeviceGetPowerUsage", reinterpret_cast<void*>(&fake_power)}};
}

GpuSymbolLookup lookup_in(std::map<std::string, void*> table) {
    return [table = std::move(table)](const char* name) -> void* {
        const auto it = table.find(name);
        return it == table.end() ? nullptr : it->second;
    };
}

void reset() {
    g_shutdown_calls = g_init_result = g_handle_result = g_temperature_result = 0;
    g_bus_id.clear();
}

} // namespace

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };

    reset();
    { // every symbol resolves: fields decode, and the bus id reaches the driver
        GpuTelemetry telemetry("00000000:01:00.0", lookup_in(full_table()));
        check(telemetry.available(), "a complete driver must open the device");
        check(g_bus_id == "00000000:01:00.0",
              "the PCI bus id must reach nvmlDeviceGetHandleByPciBusId_v2");
        const GpuSample sample = telemetry.sample();
        check(sample.utilization_percent == 87u && sample.memory_utilization_percent == 41u,
              "utilization decoded wrongly");
        check(sample.memory_used_bytes == 21474836480ULL &&
                  sample.memory_total_bytes == 25769803776ULL,
              "memory decoded wrongly (total/free/used order)");
        check(sample.temperature_celsius == 71u, "temperature decoded wrongly");
        check(sample.power_watts && *sample.power_watts == 312.5,
              "power must convert milliwatts to watts");
    }
    check(g_shutdown_calls == 1, "destroying the sampler must call nvmlShutdown once");

    reset();
    { // one query failing drops that field only
        g_temperature_result = 3;
        GpuTelemetry telemetry("x", lookup_in(full_table()));
        const GpuSample sample = telemetry.sample();
        check(!sample.temperature_celsius && sample.utilization_percent && sample.power_watts,
              "a failing query must leave only its own field unknown");
    }

    reset();
    { // optional queries missing from an older driver leave their fields unknown
        auto table = full_table();
        table.erase("nvmlDeviceGetPowerUsage");
        table.erase("nvmlDeviceGetMemoryInfo");
        GpuTelemetry telemetry("x", lookup_in(table));
        const GpuSample sample = telemetry.sample();
        check(telemetry.available() && !sample.power_watts && !sample.memory_used_bytes &&
                  sample.utilization_percent,
              "missing optional symbols must only drop their fields");
    }

    for (const char* required :
         {"nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetHandleByPciBusId_v2"}) {
        reset();
        auto table = full_table();
        table.erase(required);
        GpuTelemetry telemetry("x", lookup_in(table));
        check(!telemetry.available() && !telemetry.sample().utilization_percent,
              "a missing required symbol must leave the sampler unavailable");
    }

    reset();
    { // init fails: unavailable, and nothing is shut down that never started
        g_init_result = 1;
        GpuTelemetry telemetry("x", lookup_in(full_table()));
        check(!telemetry.available(), "a failed nvmlInit must leave the sampler unavailable");
    }
    check(g_shutdown_calls == 0, "a failed nvmlInit must not be followed by nvmlShutdown");

    reset();
    { // unknown bus id: NVML was initialised, so it is shut down again
        g_handle_result = 2;
        GpuTelemetry telemetry("x", lookup_in(full_table()));
        check(!telemetry.available(), "an unknown device must leave the sampler unavailable");
        check(g_shutdown_calls == 1, "an unknown device must release the initialised driver");
    }
    return failures == 0 ? 0 : 1;
}
