# DeepSeek-V4's MoE mixing: the clamped SwiGLU of the shared expert and the weighted slot sum.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/moe_mix.cu")
