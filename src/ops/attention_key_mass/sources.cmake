# Per-key softmax mass over query rows and heads (lossy-prefill token scoring).
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/attention_key_mass.cu")
