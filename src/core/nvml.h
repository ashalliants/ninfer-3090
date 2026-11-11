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
    // The limit the board is held to now, after every limiter, `nvidia-smi -pl` included
    // (nvmlDeviceGetEnforcedPowerLimit). A capped card is slower, so a throughput figure means
    // little without it.
    std::optional<double> power_limit_watts;
    // Cumulative since the driver loaded, so only differences are meaningful.
    std::optional<double> energy_joules;
};

// The device's PCIe link: the highest generation it can run (the current one drops at idle) and the
// current width (the slot's lanes, which a slot shared between two cards halves).
struct PcieLink {
    unsigned max_generation = 0;
    unsigned width          = 0;
};

using GpuSymbolLookup = std::function<void*(const char*)>;

// One device through NVML, loaded at runtime from the driver's library so that its absence (no
// driver, a container without it) only leaves every answer unknown and never stops startup. The
// single place NVML is loaded: the VRAM budget and the metrics endpoint both read it from here.
class Nvml {
public:
    // `pci_bus_id` as cudaDeviceGetPCIBusId prints it ("00000000:01:00.0"): CUDA and NVML number
    // devices differently, so the bus id is the join key.
    explicit Nvml(const std::string& pci_bus_id);
    // Same, with the NVML entry points taken from `lookup` (name -> function, nullptr when absent)
    // instead of the driver's library: the seam that lets a test stand in a fake driver.
    Nvml(const std::string& pci_bus_id, const GpuSymbolLookup& lookup);
    ~Nvml();

    Nvml(const Nvml&)            = delete;
    Nvml& operator=(const Nvml&) = delete;

    // False when NVML or the device could not be opened; every query then reports unknown.
    [[nodiscard]] bool available() const noexcept;

    // Physical device memory less the driver's reserve (nvmlDeviceGetMemoryInfo_v2); 0 unknown.
    [[nodiscard]] std::uint64_t usable_bytes() const;
    // 1 active, 0 inactive, -1 unknown.
    [[nodiscard]] int display_active() const;
    [[nodiscard]] GpuSample sample() const;
    // nullopt when either half cannot be read.
    [[nodiscard]] std::optional<PcieLink> pcie_link() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
