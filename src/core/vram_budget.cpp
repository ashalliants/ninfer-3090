// Adapted from Infernix a3edb450 src/core/vram_budget.cpp (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace.

#include "core/vram_budget.h"

#include "core/nvml.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#endif

namespace ninfer {
namespace {

std::mutex& fake_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::function<VramSnapshot()>& fake_source() {
    static std::function<VramSnapshot()> fake;
    return fake;
}

class DeviceSource final : public VramBudgetSource {
public:
    DeviceSource(int device, bool past_budget) : device_(device) {
        if (cudaGetDeviceProperties(&props_, device) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cannot read the device's properties");
        }
        char bus[32];
        std::snprintf(bus, sizeof(bus), "%08x:%02x:%02x.0", props_.pciDomainID, props_.pciBusID,
                      props_.pciDeviceID);
        nvml_ = std::make_unique<Nvml>(bus);
#ifdef _WIN32
        budget_event_ = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
        open_adapter();
        if (past_budget) {
            const VramSnapshot now     = query();
            const std::uint64_t usable = nvml_->usable_bytes();
            if (now.has_budget && usable > now.local_budget + kPastBudgetMargin) {
                allowance_ = usable - now.local_budget - kPastBudgetMargin;
            }
        }
#else
        (void)past_budget;
#endif
    }

    ~DeviceSource() override {
#ifdef _WIN32
        release_adapter();
        if (budget_event_ != nullptr) { ::CloseHandle(budget_event_); }
#endif
    }

    void* change_event() override {
#ifdef _WIN32
        return budget_event_;
#else
        return nullptr;
#endif
    }

    VramSnapshot query() override {
        const std::lock_guard<std::mutex> lock(mutex_);
        VramSnapshot out;
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            throw std::runtime_error("VRAM budget: cudaMemGetInfo failed");
        }
        out.device_free  = free_bytes;
        out.device_total = total_bytes;
#ifdef _WIN32
        // An adapter's output list is a snapshot of its factory: recreate both after a hot-plug.
        if (factory_ != nullptr && !factory_->IsCurrent()) {
            release_adapter();
            open_adapter();
        }
        if (adapter_ != nullptr) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
                out.has_budget   = true;
                out.local_budget = info.Budget;
                out.local_usage  = info.CurrentUsage;
                if (allowance_ != 0) { // free memory as cudaMemGetInfo derives it, from the wider budget
                    out.budget_allowance = allowance_;
                    out.local_budget += allowance_;
                    out.device_free = out.local_budget > out.local_usage
                                          ? static_cast<std::size_t>(out.local_budget - out.local_usage)
                                          : 0;
                }
            }
            out.outputs = outputs_;
        }
#endif
        const int active = nvml_->display_active();
        if (out.outputs > 0 || active == 1) {
            out.display = DisplayState::Attached;
        } else if (out.outputs == 0 && active != 1) {
            out.display = DisplayState::Headless;
        } else if (out.outputs < 0 && active == 0) {
            out.display = DisplayState::Headless;
        } else {
            out.display = DisplayState::Unknown;
        }
        return out;
    }

private:
#ifdef _WIN32
    void open_adapter() {
        IDXGIFactory4* factory = nullptr;
        if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) { return; }
        LUID luid{};
        std::memcpy(&luid, props_.luid, sizeof(luid));
        IDXGIAdapter3* adapter = nullptr;
        if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter)))) {
            factory->Release();
            return;
        }
        factory_ = factory;
        adapter_ = adapter;
        if (budget_event_ != nullptr &&
            FAILED(adapter_->RegisterVideoMemoryBudgetChangeNotificationEvent(budget_event_, &budget_cookie_))) {
            budget_cookie_ = 0;
        }
        outputs_ = 0;
        for (UINT i = 0;; ++i) {
            IDXGIOutput* output = nullptr;
            if (adapter_->EnumOutputs(i, &output) == DXGI_ERROR_NOT_FOUND) { break; }
            if (output != nullptr) {
                output->Release();
                ++outputs_;
            }
        }
    }

    void release_adapter() noexcept {
        if (adapter_ != nullptr && budget_cookie_ != 0) {
            adapter_->UnregisterVideoMemoryBudgetChangeNotification(budget_cookie_);
            budget_cookie_ = 0;
        }
        if (adapter_ != nullptr) { adapter_->Release(); }
        if (factory_ != nullptr) { factory_->Release(); }
        adapter_ = nullptr;
        factory_ = nullptr;
        outputs_ = -1;
    }

    IDXGIFactory4* factory_ = nullptr;
    IDXGIAdapter3* adapter_ = nullptr;
    int outputs_            = -1;
    HANDLE budget_event_    = nullptr;
    DWORD budget_cookie_    = 0;
#endif
    int device_ = 0;
    std::uint64_t allowance_ = 0; // past the budget (measured at open); 0: the OS budget as reported
    cudaDeviceProp props_{};
    std::unique_ptr<Nvml> nvml_;
    std::mutex mutex_;
};

class FakeSource final : public VramBudgetSource {
public:
    explicit FakeSource(std::function<VramSnapshot()> fake) : fake_(std::move(fake)) {}
    VramSnapshot query() override { return fake_(); }

private:
    std::function<VramSnapshot()> fake_;
};

} // namespace

std::uint64_t spill_shortfall(std::uint64_t free_before, std::uint64_t free_after, std::uint64_t bytes,
                              std::uint64_t tolerance) noexcept {
    const std::uint64_t dropped = free_before > free_after ? free_before - free_after : 0;
    const std::uint64_t missing = bytes > dropped ? bytes - dropped : 0;
    return missing > tolerance ? missing : 0;
}

std::uint64_t SpillGuard::end(std::uint64_t bytes) {
    if (spill_shortfall(before_, source_.query().device_free, bytes) == 0) { return 0; }
    return spill_shortfall(before_, source_.query().device_free, bytes);
}

std::unique_ptr<VramBudgetSource> open_vram_budget_source(int device, bool past_budget) {
    {
        const std::lock_guard<std::mutex> lock(fake_mutex());
        if (fake_source()) { return std::make_unique<FakeSource>(fake_source()); }
    }
    return std::make_unique<DeviceSource>(device, past_budget);
}

namespace testing {
void set_vram_budget_source(std::function<VramSnapshot()> fake) {
    const std::lock_guard<std::mutex> lock(fake_mutex());
    fake_source() = std::move(fake);
}
} // namespace testing

} // namespace ninfer
