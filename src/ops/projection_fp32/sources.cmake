# FP32 projections of BF16 activations (router, shared-expert gate and LM-head logits). The BF16
# form is adapted from Infernix a3edb450; the GGML head form reuses the GGML linears' decode route.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/projection_fp32.cu"
  "${CMAKE_CURRENT_LIST_DIR}/projection_fp32_ggml.cu")
