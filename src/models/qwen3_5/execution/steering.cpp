#include "models/qwen3_5/execution/steering.h"

#include "ninfer/ops/direction_project.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

Tensor SteeringParameters::selector(std::int32_t set) const {
    if (set < 0) { return {}; }
    if (set >= sets) {
        throw std::logic_error("a request selected a control vector the Engine has not loaded");
    }
    return Tensor(const_cast<std::int32_t*>(selectors + set), DType::I32, {1});
}

std::shared_ptr<const SteeringParameters>
make_steering_parameters(const std::vector<ControlVector>& vectors) {
    if (vectors.empty()) { return nullptr; }
    if (vectors.size() > kMaxControlVectors) {
        throw std::invalid_argument("at most " + std::to_string(kMaxControlVectors) +
                                    " control vectors may be loaded");
    }
    auto out    = std::make_shared<SteeringParameters>();
    out->sets   = static_cast<std::int32_t>(vectors.size());
    out->layers = vectors.front().layers;
    out->hidden = vectors.front().hidden;
    out->active.assign(static_cast<std::size_t>(out->layers), 0);
    const auto sets   = static_cast<std::size_t>(out->sets);
    const auto layers = static_cast<std::size_t>(out->layers);
    const auto hidden = static_cast<std::size_t>(out->hidden);
    std::vector<float> directions(sets * layers * hidden, 0.0f);
    std::vector<float> coefficients(sets * layers * 2, 0.0f);
    std::vector<std::int32_t> selectors(sets);
    for (std::size_t k = 0; k < sets; ++k) {
        const ControlVector& vector = vectors[k];
        if (vector.layers != out->layers || vector.hidden != out->hidden ||
            vector.directions.size() != layers * hidden || vector.active.size() != layers) {
            throw std::invalid_argument("control vector '" + vector.name +
                                        "' does not match the model's layers and hidden size");
        }
        for (std::size_t j = 0; j < k; ++j) {
            if (vectors[j].name == vector.name) {
                throw std::invalid_argument("two control vectors are named '" + vector.name + "'");
            }
        }
        std::copy(vector.directions.begin(), vector.directions.end(), directions.begin() + k * layers * hidden);
        for (std::size_t l = 0; l < layers; ++l) {
            if (vector.active[l] == 0) { continue; }
            const auto layer                              = static_cast<std::int32_t>(l);
            coefficients[(k * layers + l) * 2]     = vector.project_coefficient(layer);
            coefficients[(k * layers + l) * 2 + 1] = vector.add_coefficient(layer);
            out->active[l]                                = 1;
        }
        selectors[k] = static_cast<std::int32_t>(k);
        out->vectors.push_back({vector.name, vector.digest, vector.describe()});
    }
    const std::size_t direction_bytes   = directions.size() * sizeof(float);
    const std::size_t coefficient_bytes = coefficients.size() * sizeof(float);
    const std::size_t selector_bytes    = selectors.size() * sizeof(std::int32_t);
    out->storage = DeviceBuffer(direction_bytes + coefficient_bytes + selector_bytes);
    out->storage.copy_from_host(directions.data(), direction_bytes, 0);
    out->storage.copy_from_host(coefficients.data(), coefficient_bytes, direction_bytes);
    out->storage.copy_from_host(selectors.data(), selector_bytes, direction_bytes + coefficient_bytes);
    auto* base         = static_cast<std::uint8_t*>(out->storage.p);
    out->directions    = reinterpret_cast<const float*>(base);
    out->coefficients  = reinterpret_cast<const float*>(base + direction_bytes);
    out->selectors     = reinterpret_cast<const std::int32_t*>(base + direction_bytes + coefficient_bytes);
    return out;
}

void steer_residual(const SteeringParameters& steering, std::uint32_t layer, Tensor& residual,
                    const Tensor& rows, cudaStream_t stream) {
    if (rows.data == nullptr || layer >= steering.active.size() || steering.active[layer] == 0) {
        return;
    }
    if (rows.dtype != DType::I32 || rows.numel() <= 0 || residual.ne[0] != steering.hidden ||
        residual.ne[1] % rows.numel() != 0) {
        throw std::logic_error("the control-vector selection does not match the pass's columns");
    }
    const auto row_columns = static_cast<std::int32_t>(residual.ne[1] / rows.numel());
    const std::int64_t l   = layer;
    ops::direction_project(residual, 1, steering.directions + l * steering.hidden,
                           static_cast<std::int64_t>(steering.layers) * steering.hidden,
                           steering.coefficients + l * 2, static_cast<std::int64_t>(steering.layers) * 2,
                           static_cast<const std::int32_t*>(rows.data), row_columns, stream);
}

} // namespace ninfer::models::qwen3_5::execution
