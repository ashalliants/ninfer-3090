// Written for NInfer-3090.
//
// DeepSeek-V4.1-Flash: the text config the converter writes (tools/convert/dsv4.py) parses into the
// per-layer plan the forward pass follows, and refusals name what is wrong. With
// NINFER_TEST_DSV4_ARTIFACT set (a converted artifact) the whole parameter set binds against the
// real directory, resident and left in the file, reading nothing but the resources.

#include "artifact/reader.h"
#include "models/dsv4/config.h"
#include "models/dsv4/model.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using ninfer::artifact::Json;
namespace dsv4 = ninfer::models::dsv4;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The V4.1-Flash GGUF's own values (the conversion record of the IQ2_XXS build).
Json flash_config() {
    Json ratios = Json::array(), clamp = Json::array();
    for (int i = 0; i < 40; ++i) {
        ratios.push_back(i < 2 ? 0 : i < 20 ? 2 : 1);
        clamp.push_back(10.0);
    }
    Json heads = Json::array();
    for (int i = 0; i < 48; ++i) { heads.push_back(16000057 + 2 * i); }
    return Json{{"architecture", "deepseek41"},
                {"model_type", "deepseek41_text"},
                {"hidden_size", 5120},
                {"num_hidden_layers", 40},
                {"vocab_size", 129280},
                {"max_position_embeddings", 1048576},
                {"rms_norm_eps", 1e-20},
                {"num_attention_heads", 64},
                {"num_key_value_heads", 1},
                {"head_dim", 512},
                {"value_head_dim", 512},
                {"rope_dim", 64},
                {"q_lora_rank", 1280},
                {"o_groups", 8},
                {"o_lora_rank", 1024},
                {"sliding_window", 128},
                {"rope_theta", 10000.0},
                {"compress_rope_theta", 160000.0},
                {"rope_scaling", {{"type", "yarn"}, {"factor", 16.0}, {"original_max_position_embeddings", 65536},
                                  {"beta_fast", 32.0}, {"beta_slow", 1.0}}},
                {"compress_ratios", ratios},
                {"kv_source_layers", {2, 8, 14, 20}},
                {"indexer_source_layers", {2, 8, 14, 20, 24, 28, 32, 36}},
                {"indexer_heads", 32},
                {"indexer_head_dim", 128},
                {"indexer_top_k", 512},
                {"num_experts", 384},
                {"num_experts_per_tok", 6},
                {"moe_intermediate_size", 2304},
                {"num_shared_experts", 1},
                {"expert_gating_func", 4},
                {"routed_scaling_factor", 1.5},
                {"norm_topk_prob", true},
                {"swiglu_clamp_experts", clamp},
                {"swiglu_clamp_shared", clamp},
                {"hc_count", 4},
                {"hc_sinkhorn_iterations", 20},
                {"hc_epsilon", 1e-6},
                {"engram_layers", {1, 14}},
                {"engram_ngram_size", 4},
                {"engram_heads_per_ngram", 8},
                {"engram_row_width", 256},
                {"engram_head_vocab_sizes", heads},
                {"engram_pad_token_id", 2}};
}

void check_config() {
    const auto c = dsv4::parse_config(flash_config());
    check(c.hidden_size == 5120 && c.num_hidden_layers == 40 && c.heads == 64 && c.head_dim == 512, "dimensions");
    check(c.residual_width() == 4 * 5120 && c.query_width() == 64 * 512, "derived widths");
    check(c.layers.size() == 40 && c.layers[0].compress_ratio == 0 && c.layers[2].compress_ratio == 2 &&
              c.layers[20].compress_ratio == 1 && c.layers[39].compress_ratio == 1,
          "compress ratios");
    check(c.layers[2].kv_source && c.layers[8].kv_source && c.layers[20].kv_source && !c.layers[3].kv_source, "kv sources");
    check(c.layers[24].index_source && !c.layers[21].index_source && c.layers[2].index_source, "index sources");
    check(c.layers[1].engram && c.layers[14].engram && !c.layers[0].engram, "Engram layers");
    check(c.kv_owner[2] == 2 && c.kv_owner[7] == 2 && c.kv_owner[8] == 8 && c.kv_owner[19] == 14 && c.kv_owner[20] == 20 &&
              c.kv_owner[39] == 20,
          "a layer reads the nearest preceding kv source's compressed KV");
    check(c.candidate_layer == 20 && c.candidate_blocks == 2048 && c.candidate_block_size == 8, "candidate selection");
    check(c.engram.head_vocab_sizes.size() == 2 * 8 * 3, "Engram heads");
    check(c.rms_norm_eps > 0 && c.rms_norm_eps < 1e-19f, "the tiny norm epsilon survives");
}

void check_refusals() {
    const auto refuses = [](Json config, const char* what) {
        try {
            (void)dsv4::parse_config(config);
        } catch (const std::exception&) {
            return;
        }
        std::fprintf(stderr, "FAIL: accepted %s\n", what);
        ++failures;
    };
    Json base = flash_config();
    auto two_kv = base;
    two_kv["num_key_value_heads"] = 2;
    refuses(two_kv, "two KV heads");
    auto no_owner = base;
    no_owner["kv_source_layers"] = {8, 14, 20};
    refuses(no_owner, "a compressing layer before every kv source");
    auto wrong_type = base;
    wrong_type["model_type"] = "qwen4_exp_text";
    refuses(wrong_type, "another model type");
    auto short_ratios = base;
    short_ratios["compress_ratios"].erase(short_ratios["compress_ratios"].end() - 1);
    refuses(short_ratios, "a short compress_ratios list");
    auto no_ratio = base;
    no_ratio["indexer_source_layers"] = {0};
    refuses(no_ratio, "an index source without a compress ratio");
    auto hash_heads = base;
    hash_heads["engram_head_vocab_sizes"].erase(hash_heads["engram_head_vocab_sizes"].end() - 1);
    refuses(hash_heads, "missing Engram heads");
}

void check_artifact(const char* path) {
    ninfer::artifact::Reader reader(path);
    check(dsv4::is_dsv4(reader), "the artifact is recognised as DeepSeek-V4");
    for (const bool in_file : {false, true}) {
        auto plan = dsv4::plan_load(reader, in_file);
        const auto& c = plan.config();
        check(plan.expert_record_strides().size() == c.num_hidden_layers, "a record stride per layer");
        for (const auto stride : plan.expert_record_strides()) { check(stride > 7'000'000 && stride < 11'000'000, "record stride"); }
        check(in_file ? plan.file_banks().size() == c.num_hidden_layers : plan.file_banks().empty(), "file banks");
        if (in_file) {
            check(plan.pinned_bytes() < (1ULL << 30), "banks left in the file are not pinned");
        } else {
            check(plan.pinned_bytes() > (100ULL << 30), "resident banks are pinned");
        }
        check(plan.device_bytes() > (4ULL << 30) && plan.device_bytes() < (8ULL << 30), "the dense weights are about 5.5 GB");
        std::printf("  in_file=%d device %.2f GB pinned %.2f GB\n", in_file ? 1 : 0, double(plan.device_bytes()) / 1e9,
                    double(plan.pinned_bytes()) / 1e9);
    }
}

} // namespace

int main() {
    check_config();
    check_refusals();
    if (const char* path = std::getenv("NINFER_TEST_DSV4_ARTIFACT")) {
        check_artifact(path);
    } else {
        std::printf("SKIP the artifact part: NINFER_TEST_DSV4_ARTIFACT is not set\n");
    }
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
