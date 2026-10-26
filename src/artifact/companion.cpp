#include "artifact/companion.h"

#include "artifact/materializer.h"
#include "artifact/schema.h"
#include "core/weight_view.h"
#include "ops/two_stage_head/proxy_codec.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <thread>
#include <vector>

namespace ninfer::artifact {
namespace {

namespace proxy = ops::proxy_head;

std::string mib_text(std::uint64_t bytes) {
    return std::to_string((bytes + (1U << 20) - 1) >> 20) + " MiB";
}

std::string gib_text(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f GiB",
                  static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

void require_proxy_shape(std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || !proxy::proxy_shape_supported(static_cast<std::int64_t>(shape[0]),
                                                           static_cast<std::int64_t>(shape[1]))) {
        throw ArtifactError("a 2-bit head copy needs an [N,K] matrix with N a multiple of 16 and K "
                            "a multiple of 128");
    }
}

} // namespace

std::string_view companion_name(Companion kind) noexcept {
    switch (kind) {
    case Companion::TwoBitHeadProxy:
        return "2-bit head copy";
    }
    return "companion";
}

std::uint64_t companion_bytes(Companion kind, std::span<const std::uint64_t> shape) {
    switch (kind) {
    case Companion::TwoBitHeadProxy:
        require_proxy_shape(shape);
        return proxy::proxy_bytes(static_cast<std::int64_t>(shape[0]),
                                  static_cast<std::int64_t>(shape[1]));
    }
    throw ArtifactError("unknown companion kind");
}

void build_companion(Companion kind, std::span<const std::uint64_t> shape,
                     std::span<const std::byte> source, std::span<std::byte> destination) {
    if (kind != Companion::TwoBitHeadProxy) { throw ArtifactError("unknown companion kind"); }
    require_proxy_shape(shape);
    const WeightGeometry geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
    const auto rows               = static_cast<std::int64_t>(shape[0]);
    const auto columns            = static_cast<std::int64_t>(shape[1]);
    if (source.size() != geometry.bytes || geometry.padded_columns != shape[1] ||
        destination.size() != companion_bytes(kind, shape)) {
        throw ArtifactError("2-bit head copy: payload sizes do not match the tensor shape");
    }
    const auto codes  = source.subspan(0, static_cast<std::size_t>(rows * columns));
    const auto scales = source.subspan(static_cast<std::size_t>(geometry.scale_offset),
                                       static_cast<std::size_t>(rows * (columns / 32) * 2));
    std::fill(destination.begin(), destination.end(), std::byte{0});
    // Whole 16-row tiles per worker: a tile's rows share code bytes.
    const std::int64_t tiles   = rows / proxy::kTileRows;
    const std::int64_t workers = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::thread::hardware_concurrency()), 1, tiles);
    const auto run = [&](std::int64_t worker) {
        proxy::pack_proxy_rows(codes, scales, rows, columns, destination,
                               tiles * worker / workers * proxy::kTileRows,
                               tiles * (worker + 1) / workers * proxy::kTileRows);
    };
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(workers));
    for (std::int64_t worker = 1; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            try {
                run(worker);
            } catch (...) { failures[static_cast<std::size_t>(worker)] = std::current_exception(); }
        });
    }
    try {
        run(0);
    } catch (...) { failures[0] = std::current_exception(); }
    for (std::thread& thread : pool) { thread.join(); }
    for (const std::exception_ptr& failure : failures) {
        if (failure) { std::rethrow_exception(failure); }
    }
}

std::uint64_t planned_companion_bytes(const MaterializationPlan& plan, std::size_t rank) {
    std::uint64_t total = 0;
    for (const auto& placement : plan.device_objects) {
        if (placement.companion && placement.rank == rank) { total += placement.bytes; }
    }
    return total;
}

std::string companion_fit_refusal(std::string_view option, std::uint64_t weight_bytes,
                                  std::uint64_t companion_bytes, std::uint64_t free_bytes) {
    if (companion_bytes == 0 || weight_bytes <= free_bytes ||
        weight_bytes - companion_bytes > free_bytes) {
        return {};
    }
    return std::string(option) + " needs " + mib_text(companion_bytes) +
           " of GPU memory for its copy of the output head, and the model weights with it (" +
           gib_text(weight_bytes) + ") do not fit the " + gib_text(free_bytes) +
           " that is free. Start without " + std::string(option) +
           ", or close other programs that use the GPU";
}

std::string companion_memory_advice(std::string_view option, std::uint64_t companion_bytes) {
    return std::string(option) + " holds " + mib_text(companion_bytes) +
           " of that GPU memory for its copy of the output head; start without it to give that "
           "memory back";
}

} // namespace ninfer::artifact
