#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace ninfer {

// One reading of a device's live state. A field is nullopt when NVML cannot report it.
struct GpuSample {
    std::optional<unsigned> utilization_percent;        // SMs busy over NVML's last sample period
    std::optional<unsigned> memory_utilization_percent; // memory controller busy, same period
    std::optional<std::uint64_t> memory_used_bytes;     // every process on the card, not just ours
    std::optional<std::uint64_t> memory_total_bytes;
    std::optional<unsigned> temperature_celsius;
    std::optional<double> power_watts;
    // Cumulative since the driver loaded, so only differences are meaningful.
    std::optional<double> energy_joules;
};

using GpuSymbolLookup = std::function<void*(const char*)>;

// Reads GpuSample from NVML, loaded at runtime from the driver's library so that its absence (no
// driver, a container without it) only leaves every field unknown and never stops startup.
class GpuTelemetry {
public:
    // `pci_bus_id` as cudaDeviceGetPCIBusId prints it ("00000000:01:00.0"): CUDA and NVML number
    // devices differently, so the bus id is the join key.
    explicit GpuTelemetry(const std::string& pci_bus_id);
    // Same, with the NVML entry points taken from `lookup` (name -> function, nullptr when absent)
    // instead of the driver's library: the seam that lets a test stand in a fake driver.
    GpuTelemetry(const std::string& pci_bus_id, const GpuSymbolLookup& lookup);
    ~GpuTelemetry();

    GpuTelemetry(const GpuTelemetry&)            = delete;
    GpuTelemetry& operator=(const GpuTelemetry&) = delete;

    // False when NVML or the device could not be opened; sample() then returns all-unknown.
    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] GpuSample sample() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
