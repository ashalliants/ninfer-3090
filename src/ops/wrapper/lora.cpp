// ninfer::ops - lora_apply wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/maintainer/op-development.md §2.
#include "ninfer/ops/lora.h"

#include "ops/launcher/lora.h" // detail::lora_apply_launch, detail::lora_splits

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void reject(const std::string& message) {
    throw std::invalid_argument("lora_apply: " + message);
}

bool aligned16(const void* pointer) {
    return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

// Byte extent of a [rows, columns] BF16 view with column stride nb[1].
struct Extent {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
};

Extent extent(const Tensor& t) {
    const auto begin = reinterpret_cast<std::uintptr_t>(t.data);
    const auto last  = static_cast<std::uintptr_t>(t.ne[1] - 1) * static_cast<std::uintptr_t>(t.nb[1]) +
                      static_cast<std::uintptr_t>(t.ne[0]) * sizeof(std::uint16_t);
    return {begin, begin + last};
}

bool overlaps(Extent a, Extent b) { return a.begin < b.end && b.begin < a.end; }

// A [rows, T] BF16 matrix view whose columns are contiguous runs of `rows` values.
void require_matrix(const Tensor& t, std::int32_t columns, const char* what) {
    if (t.dtype != DType::BF16 || t.data == nullptr) {
        reject(std::string(what) + " must be a non-null BF16 tensor");
    }
    if (t.ne[0] <= 0 || t.ne[1] != columns || t.ne[2] != 1 || t.ne[3] != 1) {
        reject(std::string(what) + " must be [rows, T] with T the column map's columns");
    }
    if (t.nb[0] != static_cast<std::int64_t>(sizeof(std::uint16_t)) ||
        (columns > 1 && t.nb[1] < t.nb[0] * t.ne[0])) {
        reject(std::string(what) + " must hold each column's rows contiguously");
    }
}

} // namespace

std::size_t lora_workspace_bytes(std::int32_t rank, std::int32_t k, std::int32_t columns) {
    if (rank <= 0 || k <= 0 || columns <= 0) { return 0; }
    const auto values = static_cast<std::size_t>(detail::lora_splits(k)) *
                        static_cast<std::size_t>(rank) * static_cast<std::size_t>(columns);
    return values * sizeof(float);
}

void lora_apply(const Tensor& x, const LoraDown& down, std::span<const LoraUp> up,
                const LoraColumns& columns, WorkspaceArena& workspace, cudaStream_t stream) {
    if (columns.rows < 1 || columns.rows > kLoraMaxRows || columns.width < 1) {
        reject("the column map needs 1 to 8 rows of at least one column");
    }
    if (static_cast<std::int64_t>(columns.rows) * columns.width > std::numeric_limits<std::int32_t>::max() / 16) {
        reject("too many columns");
    }
    const std::int32_t total = columns.rows * columns.width;
    for (std::int32_t row = 0; row < columns.rows; ++row) {
        const std::int32_t slot = columns.slot[static_cast<std::size_t>(row)];
        if (slot < -1 || slot >= kLoraMaxAdapters) { reject("a column slot is out of range"); }
    }
    require_matrix(x, total, "x");
    if (x.ne[0] % 8 != 0) { reject("K must be a multiple of 8"); }
    if (!aligned16(x.data) || x.nb[1] % 16 != 0) {
        reject("x and its column stride must be 16-byte aligned");
    }
    if (down.rank < 8 || down.rank > kLoraMaxRank || down.rank % 8 != 0) {
        reject("the rank must be a multiple of 8 in [8, 1024]");
    }
    for (const void* a : down.a) {
        if (a != nullptr && !aligned16(a)) { reject("A matrices must be 16-byte aligned"); }
    }
    if (up.empty() || up.size() > static_cast<std::size_t>(kLoraMaxOutputs)) {
        reject("a call updates 1 to 4 outputs");
    }
    for (std::size_t o = 0; o < up.size(); ++o) {
        const LoraUp& output = up[o];
        require_matrix(output.output, total, "an output");
        if (output.rank_count < 8 || output.rank_count > kLoraMaxOutputRank ||
            output.rank_count % 8 != 0 || output.rank_offset < 0 || output.rank_offset % 8 != 0 ||
            output.rank_offset + output.rank_count > down.rank) {
            reject("an output's rank range must be 8-aligned, at most 256 wide and inside the rank");
        }
        for (const void* b : output.b) {
            if (b != nullptr && !aligned16(b)) { reject("B matrices must be 16-byte aligned"); }
        }
        if (overlaps(extent(output.output), extent(x))) { reject("an output overlaps x"); }
        for (std::size_t other = 0; other < o; ++other) {
            if (overlaps(extent(output.output), extent(up[other].output))) {
                reject("two outputs overlap");
            }
        }
    }
    if (!columns.any()) { return; }

    auto scope   = workspace.scope();
    auto partial = workspace.alloc_bytes(lora_workspace_bytes(down.rank, x.ne[0], total));
    detail::lora_apply_launch(x, down, up, columns, static_cast<float*>(partial.data), stream);
}

} // namespace ninfer::ops
