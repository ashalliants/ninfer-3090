// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/parameters.cpp (Apache-2.0).
// Modified for NInfer-3090: see parameters.h.

#include "models/qwen4_exp/execution/parameters.h"

#include "core/weight_view.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    // One projection over consecutive parameters of one parent that share an input (the converter
    // packs such groups, rows in this order).
    LinearParameters linear(std::initializer_list<WeightId> ids) const {
        const auto& first = model_.weight(*ids.begin());
        return with_context(first.name, [&] {
            WeightView view;
            std::uint64_t rows = 0;
            for (const auto id : ids) {
                const auto& bound = model_.weight(id);
                if (bound.view.shape.size() != 2 || bound.view.shape[1] != first.view.shape[1] ||
                    bound.policy != first.policy) {
                    throw std::invalid_argument("joined projections need one K and one activation policy");
                }
                rows += bound.view.shape[0];
                for (const auto& part : bound.view.parts) { view.parts.push_back(part); }
            }
            view.shape = {rows, first.view.shape[1]};
            return LinearParameters{native_weight(view), first.policy};
        });
    }

    // A direct parameter as a Tensor of its reversed logical shape (the fastest axis first).
    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) { throw std::invalid_argument("direct parameter exceeds Tensor rank"); }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    HyperConnectionParameters hc(const HyperConnectionWeights& w) const {
        HyperConnectionParameters out;
        out.norm    = tensor(w.norm);
        out.combine = w.inject.valid();
        out.down    = out.combine ? linear({w.down, w.inject}) : linear({w.down});
        out.up      = linear({w.up});
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(source);
    const auto& w   = source.weights();
    token_embedding = prepare.linear({w.token_embedding}).weight;
    output_head     = prepare.linear({w.output_head}).weight;
    final_mixer     = prepare.hc(w.final_mixer);
    const auto banks = source.expert_banks();
    if (banks.size() != w.layers.size()) { throw std::logic_error("Qwen4Exp: one expert bank per layer"); }
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& layer = w.layers[i];
        BlockParameters out;
        out.attn_hc = prepare.hc(layer.attn_hc);
        out.mlp_hc  = prepare.hc(layer.mlp_hc);
        if (const auto* a = std::get_if<AttentionWeights>(&layer.mixer)) {
            out.mixer = AttentionParameters{prepare.linear({a->query_gate}),
                                            prepare.linear({a->key}),
                                            prepare.linear({a->value}),
                                            prepare.linear({a->output}),
                                            prepare.tensor(a->query_norm),
                                            prepare.tensor(a->key_norm),
                                            prepare.tensor(a->index_query),
                                            prepare.tensor(a->index_key),
                                            prepare.tensor(a->index_query_norm),
                                            prepare.tensor(a->index_key_norm)};
        } else {
            const auto& g = std::get<GdnWeights>(layer.mixer);
            out.mixer     = GdnParameters{prepare.linear({g.qkv}),
                                          prepare.linear({g.z}),
                                          prepare.linear({g.a_projection, g.b_projection}),
                                          prepare.linear({g.output}),
                                          prepare.tensor(g.decay),
                                          prepare.tensor(g.dt_bias),
                                          prepare.tensor(g.convolution),
                                          prepare.tensor(g.norm)};
        }
        out.moe.router       = prepare.tensor(layer.moe.router);
        out.moe.shared_score = prepare.tensor(layer.moe.shared_score);
        out.moe.shared_gate  = prepare.linear({layer.moe.shared_gate});
        out.moe.shared_up    = prepare.linear({layer.moe.shared_up});
        out.moe.shared_down  = prepare.linear({layer.moe.shared_down});
        out.moe.bank         = &banks[i];
        if (layer.ple) {
            const auto& p = *layer.ple;
            out.ple       = PleParameters{prepare.linear({p.key_projection, p.value_projection}),
                                          prepare.tensor(p.key_norm), prepare.tensor(p.query_norm),
                                          prepare.tensor(p.conv_norm), prepare.tensor(p.convolution)};
        }
        layers.push_back(std::move(out));
    }
}

} // namespace ninfer::models::qwen4_exp::execution
